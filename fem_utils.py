#!/usr/bin/env python3
"""
fem_utils.py

Finite element solver utilities (scikit-fem) for the General Spatial Bass (GSB) framework.

Refactor goals:
- Provide a Runner class with:
    * build_mesh()
    * run_FEM()
- Make it easy to run multiple parameter sets in parallel.
- Standardized output layout:
    out/<out_folder>/mesh/12_36_km.msh
    out/<out_folder>/figures/<plots>.png
- Standardized logging with timestamps relative to Runner creation:
    ca_run8@[01:26:31] ---- self.build_mesh complete
    ca_run8@[01:26:31] ---- _plot_uvIJ_and_w_year_lonlat complete

Notes:
- Likelihood/binning is done on NODES (not triangles):
    * Each event is assigned to its containing triangle, then split evenly (1/3) to its 3 vertices.
    * Expected counts are computed on nodal control volumes A_nodes:
          mu_i = ∫ lambda_i(t) * A_i dt
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Sequence, List, Any

import numpy as np
import pandas as pd
import math
import time
import logging

from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve, factorized
from scipy.optimize import minimize

import meshio
from pyproj import Transformer

from skfem import Basis, MeshTri, ElementTriP1, asm, BilinearForm
from skfem.models.poisson import laplace
logging.getLogger("skfem").setLevel(logging.ERROR)

from density_utils import get_batch_nodal_density


# =============================================================================
# Lightweight timestamp logger (relative to Runner creation)
# =============================================================================

def _fmt_hhmmss(seconds: float) -> str:
    seconds = float(max(seconds, 0.0))
    h = int(seconds // 3600)
    m = int((seconds - 3600 * h) // 60)
    s = int(seconds - 3600 * h - 60 * m)
    return f"{h:02d}:{m:02d}:{s:02d}"


# =============================================================================
# FEM forms
# =============================================================================

@BilinearForm
def mass_form(u, v, w):
    return u * v


@BilinearForm
def laplace_coeff(u, v, w):
    return w["mu"] * np.sum(u.grad * v.grad, axis=0)


# =============================================================================
# Model definition / configuration
# =============================================================================

@dataclass
class GSBFunctions:
    S: Callable[[np.ndarray, float], np.ndarray]
    F_I: Callable[[np.ndarray], np.ndarray]
    G: Callable[[np.ndarray], np.ndarray]
    F_J: Callable[[np.ndarray], np.ndarray]
    F_J_prime: Optional[Callable[[np.ndarray], np.ndarray]] = None
    mu_prime: Optional[Callable[[np.ndarray], np.ndarray]] = None


@dataclass
class FEMConfig:
    tau_years: float = 0.05
    T_years: float = 5.0
    picard_max_iter: int = 15
    picard_tol: float = 1e-8
    verbose: bool = True

    YEAR0: float = 1998.0
    epsg_project: int = 5070


# =============================================================================
# Pickle-friendly model building blocks (for parallel runs)
# =============================================================================

@dataclass(frozen=True)
class ConstS:
    S0: float
    def __call__(self, xy_km: np.ndarray, t_years: float) -> np.ndarray:
        return np.full(xy_km.shape[0], float(self.S0), dtype=float)


@dataclass(frozen=True)
class SaturatingFI:
    def __call__(self, I: np.ndarray) -> np.ndarray:
        I = np.asarray(I, float)
        return I / np.maximum(1.0 + I, 1e-15)


@dataclass(frozen=True)
class LinearG:
    gamma_J: float
    def __call__(self, J: np.ndarray) -> np.ndarray:
        return float(self.gamma_J) * np.asarray(J, float)


@dataclass(frozen=True)
class LinearFJ:
    k_J: float
    def __call__(self, J: np.ndarray) -> np.ndarray:
        return float(self.k_J) * np.asarray(J, float)


@dataclass(frozen=True)
class ConstFJPrime:
    k_J: float
    def __call__(self, J: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(J, float), float(self.k_J), dtype=float)


@dataclass(frozen=True)
class ConstMuPrime:
    D: float
    def __call__(self, J: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(J, float), float(self.D), dtype=float)


def make_default_gsb_functions(model_params: Dict[str, float]) -> GSBFunctions:
    """
    Creates pickle-friendly default GSBFunctions consistent with your example:
      S(x,t) = S0
      F_I(I) = I/(1+I)
      G(J)   = gamma_J * J
      F_J(J) = k_J * J
      F_J'   = k_J (constant)
      mu'    = D   (constant)
    """
    S0 = float(model_params.get("S0", 0.0))
    gamma_J = float(model_params.get("gamma_J", 0.0))
    k_J = float(model_params.get("k_J", 0.0))
    D = float(model_params.get("D", 0.0))

    return GSBFunctions(
        S=ConstS(S0=S0),
        F_I=SaturatingFI(),
        G=LinearG(gamma_J=gamma_J),
        F_J=LinearFJ(k_J=k_J),
        F_J_prime=ConstFJPrime(k_J=k_J),
        mu_prime=ConstMuPrime(D=D),
    )


# =============================================================================
# Utilities: mesh IO + coordinate transforms
# =============================================================================

def load_mesh_km_from_msh(msh_path: Path) -> MeshTri:
    mi = meshio.read(msh_path)

    tri = None
    for c in mi.cells:
        if c.type == "triangle":
            tri = c.data
            break
    if tri is None:
        raise ValueError("No triangle cells found in .msh.")

    pts = mi.points[:, :2].T
    t = tri.T.astype(np.int64)
    return MeshTri(pts, t)


def km_to_lonlat_transformer(epsg_project: int) -> Transformer:
    return Transformer.from_crs(f"EPSG:{epsg_project}", "EPSG:4326", always_xy=True)


def mesh_nodes_lonlat(mesh: MeshTri, epsg_project: int) -> Tuple[np.ndarray, np.ndarray]:
    inv = km_to_lonlat_transformer(epsg_project)
    x_km = mesh.p[0, :]
    y_km = mesh.p[1, :]
    lon, lat = inv.transform(x_km * 1000.0, y_km * 1000.0)
    return np.asarray(lon, float), np.asarray(lat, float)


# =============================================================================
# Stage cache (mesh/basis/M/K + precomputed rho_total over time + A_nodes)
# =============================================================================

@dataclass
class FEMStageCache:
    msh_path: Path
    cfg: FEMConfig

    mesh: MeshTri
    basis: Basis
    N: int

    times: np.ndarray           # (nt,)
    M: csr_matrix               # (N,N)
    K: csr_matrix               # (N,N)

    lon_nodes: np.ndarray       # (N,)
    lat_nodes: np.ndarray       # (N,)
    xy_nodes_km: np.ndarray     # (N,2)

    rho_total: np.ndarray       # (nt,N) persons/km^2
    A_nodes: np.ndarray         # (N,) km^2 nodal control areas (mass-lumped)

    def check_compatible(self, cfg: FEMConfig) -> None:
        if float(cfg.tau_years) != float(self.cfg.tau_years):
            raise ValueError("FEMStageCache incompatible: tau_years differs.")
        if float(cfg.T_years) != float(self.cfg.T_years):
            raise ValueError("FEMStageCache incompatible: T_years differs.")
        if float(cfg.YEAR0) != float(self.cfg.YEAR0):
            raise ValueError("FEMStageCache incompatible: YEAR0 differs.")
        if int(cfg.epsg_project) != int(self.cfg.epsg_project):
            raise ValueError("FEMStageCache incompatible: epsg_project differs.")


def build_fem_stage_cache(msh_path: Path, cfg: FEMConfig, log: Optional[Callable[[str], None]] = None) -> FEMStageCache:
    mesh = load_mesh_km_from_msh(msh_path)
    basis = Basis(mesh, ElementTriP1())
    N = mesh.p.shape[1]

    tau = float(cfg.tau_years)
    nsteps = int(np.floor(float(cfg.T_years) / tau))
    times = np.arange(nsteps + 1, dtype=float) * tau

    M: csr_matrix = asm(mass_form, basis).tocsr()
    K: csr_matrix = asm(laplace, basis).tocsr()

    lon_nodes, lat_nodes = mesh_nodes_lonlat(mesh, cfg.epsg_project)
    xy_nodes_km = np.column_stack([mesh.p[0, :], mesh.p[1, :]]).astype(float)

    years = (float(cfg.YEAR0) + times).tolist()

    if log is not None:
        log(f"build_fem_stage_cache: nodes={N}, steps={nsteps}, tau={tau} years, nt={len(years)}")

    out = get_batch_nodal_density(
        mesh,
        years,
        epsg_project=cfg.epsg_project,
        return_masses=True,
        use_cache=True,
    )
    rho_total = np.asarray(out["rho_nodes"], dtype=float)  # (nt,N)
    A_nodes = np.asarray(out["A_nodes"], dtype=float)      # (N,)

    return FEMStageCache(
        msh_path=msh_path,
        cfg=cfg,
        mesh=mesh,
        basis=basis,
        N=N,
        times=times,
        M=M,
        K=K,
        lon_nodes=lon_nodes,
        lat_nodes=lat_nodes,
        xy_nodes_km=xy_nodes_km,
        rho_total=rho_total,
        A_nodes=A_nodes,
    )


# =============================================================================
# Solver core
# =============================================================================

def _solve_u_v_backward_euler_closed_form_from_FI(
    u_prev: np.ndarray, v_prev: np.ndarray,
    p: float, qI: float, FI_curr: np.ndarray, tau: float
) -> Tuple[np.ndarray, np.ndarray]:
    A = tau * float(p)

    FI = np.asarray(FI_curr, float)
    FI = np.maximum(FI, 0.0)

    B = tau * float(qI) * FI

    s_num = 1.0 - u_prev - v_prev
    s_den = 1.0 + A + B
    s = s_num / np.maximum(s_den, 1e-15)

    u = u_prev + A * s
    v = v_prev + B * s

    # If u,v are meant to be fractions, keep this clip.
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)

    return u, v


def _is_constant_vector(x: np.ndarray, atol: float = 0.0) -> bool:
    if x.size == 0:
        return True
    return bool(np.all(np.abs(x - x.flat[0]) <= float(atol)))


def solve_gsb_fem(
    msh_path: Path,
    funcs: GSBFunctions,
    params: Dict[str, float],
    cfg: FEMConfig,
    cache: Optional[FEMStageCache] = None,
    log: Optional[Callable[[str], None]] = None,
) -> "GSBSolution":
    # --- use cache if provided, else build minimal local objects (slower) ---
    if cache is None:
        mesh = load_mesh_km_from_msh(msh_path)
        basis = Basis(mesh, ElementTriP1())
        N = mesh.p.shape[1]

        tau = float(cfg.tau_years)
        nsteps = int(np.floor(float(cfg.T_years) / tau))
        times = np.arange(nsteps + 1, dtype=float) * tau

        M: csr_matrix = asm(mass_form, basis).tocsr()
        K: csr_matrix = asm(laplace, basis).tocsr()

        # Lumped (diagonal) mass matrix: M_L = diag(row_sums(M))  [km^2]
        M_L_vec = np.asarray(M.sum(axis=1)).ravel().astype(float)
        M_L = diags(M_L_vec, 0, shape=(N, N), format="csr")

        xy_nodes_km = np.column_stack([mesh.p[0, :], mesh.p[1, :]]).astype(float)

        rho_total_pre = None
        A_nodes = M_L_vec.copy()   # (N,) km^2
    else:
        cache.check_compatible(cfg)
        mesh = cache.mesh
        basis = cache.basis
        N = cache.N
        times = cache.times
        tau = float(cfg.tau_years)
        nsteps = times.size - 1
        M = cache.M
        K = cache.K

        A_nodes = np.asarray(cache.A_nodes, float)   # (N,) km^2
        M_L_vec = A_nodes.copy()
        M_L = diags(M_L_vec, 0, shape=(N, N), format="csr")

        xy_nodes_km = cache.xy_nodes_km
        rho_total_pre = cache.rho_total  # (nt,N)

    # --- params ---
    r = float(params.get("r", 1.0))
    p = float(params.get("p", 0.01))
    qI = float(params.get("q_I", 0.1))
    kJ = float(params.get("k_J", 0.0))
    D = float(params.get("D", 0.0))

    if funcs.mu_prime is None:
        funcs.mu_prime = ConstMuPrime(D=D)
    if funcs.F_J_prime is None:
        funcs.F_J_prime = ConstFJPrime(k_J=kJ)

    u = np.zeros(N, dtype=float)
    v = np.zeros(N, dtype=float)
    I = np.zeros(N, dtype=float)
    J = np.zeros(N, dtype=float)

    U_hist = [u.copy()]
    V_hist = [v.copy()]
    I_hist = [I.copy()]
    J_hist = [J.copy()]

    RHO_total_hist = [np.zeros(N, dtype=float)]
    RHO_adopt_hist = [np.zeros(N, dtype=float)]

    if log is not None:
        log(f"solve_gsb_fem: nodes={N}, steps={nsteps}, tau={tau} (cache={'yes' if cache is not None else 'no'})")

    # constant-factorized A if possible
    solveA_const = None
    use_const_A = False

    J0 = np.zeros(N, dtype=float)
    mu0 = np.asarray(funcs.mu_prime(J0), float)
    fjp0 = np.asarray(funcs.F_J_prime(J0), float)

    if _is_constant_vector(mu0) and _is_constant_vector(fjp0):
        mu_c = float(mu0.flat[0])
        fjp_c = float(fjp0.flat[0])
        A = (M_L * (1.0 / tau)) + (K * mu_c) + (M_L * fjp_c)
        solveA_const = factorized(A.tocsc())
        use_const_A = True
        if log is not None:
            log(f"solve_gsb_fem: using constant-factorized A: mu={mu_c:.6g}, FJ'={fjp_c:.6g}")
    else:
        if log is not None:
            log("solve_gsb_fem: mu_prime or F_J_prime not constant -> will assemble/solve A each Picard iter")

    J_clamp_steps = 0
    J_clamp_total = 0

    for m in range(1, nsteps + 1):
        t = float(times[m])
        cal_year = float(cfg.YEAR0) + t

        # rho_total at nodes
        if rho_total_pre is not None:
            rho_total = rho_total_pre[m].copy()
        else:
            out = get_batch_nodal_density(
                mesh, [cal_year],
                epsg_project=cfg.epsg_project,
                return_masses=True,
                use_cache=True,
            )
            rho_total = np.asarray(out["rho_nodes"][0], dtype=float)

        rho_adopt = r * rho_total

        RHO_total_hist.append(rho_total.copy())
        RHO_adopt_hist.append(rho_adopt.copy())

        J_guess = J.copy()
        clamped_this_step = False
        rel = np.inf

        Svec = funcs.S(xy_nodes_km, t)

        for _it in range(cfg.picard_max_iter):
            # Clamp J for positivity (propagates into I via G(J))
            if np.any(J_guess < 0.0):
                clamped_this_step = True
                J_clamp_total += 1
            J_eff = np.maximum(J_guess, 0.0)

            I_new = I + tau * funcs.G(J_eff)

            FI_new = funcs.F_I(I_new)
            u_new, v_new = _solve_u_v_backward_euler_closed_form_from_FI(
                u_prev=u, v_prev=v, p=p, qI=qI, FI_curr=FI_new, tau=tau
            )

            du_dt = (u_new - u) / tau
            dv_dt = (v_new - v) / tau

            fj = funcs.F_J(J_eff)
            fjp = funcs.F_J_prime(J_eff)

            source = rho_adopt * (du_dt + dv_dt) + Svec + (fjp * J_eff - fj)
            rhs = (M_L @ (J * (1.0 / tau))) + (M_L @ source)

            if use_const_A and solveA_const is not None:
                J_new = solveA_const(rhs).astype(float)
            else:
                mu_p = np.asarray(funcs.mu_prime(J_eff), float)
                fjp_vec = np.asarray(fjp, float)

                mu_q = basis.interpolate(mu_p)
                K_mu = asm(laplace_coeff, basis, mu=mu_q).tocsr()

                R = diags(fjp_vec, 0, shape=(N, N), format="csr")
                A = (M_L * (1.0 / tau)) + K_mu + (M_L @ R)
                J_new = spsolve(A, rhs).astype(float)

            # Clamp solved J before next Picard iteration
            if np.any(J_new < 0.0):
                clamped_this_step = True
                J_clamp_total += 1
            J_new = np.maximum(J_new, 0.0)

            rel = np.linalg.norm(J_new - J_guess) / max(1e-12, np.linalg.norm(J_new))
            J_guess = J_new
            if rel < cfg.picard_tol:
                break

        u, v, I, J = u_new, v_new, I_new, J_guess
        if clamped_this_step:
            J_clamp_steps += 1

        U_hist.append(u.copy())
        V_hist.append(v.copy())
        I_hist.append(I.copy())
        J_hist.append(J.copy())

        if cfg.verbose and (m % max(1, nsteps // 10) == 0):
            if log is not None:
                log(f"solve_gsb_fem: step {m}/{nsteps} cal_year={cal_year:.2f} picard_rel={rel:.2e}")

    if J_clamp_steps > 0 and log is not None:
        log(f"[WARN] J clamped in {J_clamp_steps}/{nsteps} steps (total clamp events: {J_clamp_total}).")

    return GSBSolution(
        mesh=mesh,
        basis=basis,
        times=np.asarray(times, float),
        YEAR0=float(cfg.YEAR0),
        U=np.array(U_hist),
        V=np.array(V_hist),
        I=np.array(I_hist),
        J=np.array(J_hist),
        rho_total=np.array(RHO_total_hist),
        rho_adopt=np.array(RHO_adopt_hist),
        A_nodes=np.asarray(A_nodes, float),
    )


# =============================================================================
# Solution object + interpolation
# =============================================================================

@dataclass
class GSBSolution:
    mesh: MeshTri
    basis: Basis
    times: np.ndarray
    YEAR0: float

    U: np.ndarray
    V: np.ndarray
    I: np.ndarray
    J: np.ndarray

    rho_total: np.ndarray
    rho_adopt: np.ndarray
    A_nodes: np.ndarray  # (N,) km^2, nodal control volumes

    def __call__(self, x_km: float, y_km: float, t_years: float, abs_vals: bool = False) -> np.ndarray:
        x = float(x_km)
        y = float(y_km)
        t = float(t_years)

        if t <= self.times[0]:
            k0 = k1 = 0
            a = 0.0
        elif t >= self.times[-1]:
            k0 = k1 = len(self.times) - 1
            a = 0.0
        else:
            k1 = int(np.searchsorted(self.times, t))
            k0 = k1 - 1
            t0, t1 = self.times[k0], self.times[k1]
            a = (t - t0) / (t1 - t0)

        finder = self.mesh.element_finder()
        tri_id = finder(np.array([x]), np.array([y]))[0]
        if tri_id < 0:
            return np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)

        verts = self.mesh.t[:, tri_id]
        P = self.mesh.p[:, verts]
        A = np.column_stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]])
        b = np.array([x, y]) - P[:, 0]

        try:
            lam12 = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)

        l1, l2 = lam12
        l0 = 1.0 - l1 - l2
        w = np.array([l0, l1, l2], dtype=float)

        def interp_snap(arr: np.ndarray) -> float:
            v0 = float(w @ arr[k0, verts])
            v1 = float(w @ arr[k1, verts])
            return (1.0 - a) * v0 + a * v1

        u = interp_snap(self.U)
        v = interp_snap(self.V)
        I = interp_snap(self.I)
        J = interp_snap(self.J)

        if abs_vals:
            rho = interp_snap(self.rho_adopt)
            return np.array([rho * u, rho * v, I, J], dtype=float)
        return np.array([u, v, I, J], dtype=float)


# =============================================================================
# Safe element finder (skfem 11 can raise if ANY point outside)
# =============================================================================

def safe_element_finder(mesh: MeshTri, chunk_size: int = 1000):
    finder = mesh.element_finder()

    xmin, xmax = float(mesh.p[0].min()), float(mesh.p[0].max())
    ymin, ymax = float(mesh.p[1].min()), float(mesh.p[1].max())
    pad = 1e-9

    def _safe(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        n = x.shape[0]
        out = -np.ones(n, dtype=np.int64)

        bbox = (x >= xmin - pad) & (x <= xmax + pad) & (y >= ymin - pad) & (y <= ymax + pad)
        idx_all = np.where(bbox)[0]
        if idx_all.size == 0:
            return out

        CH = int(chunk_size)
        for s in range(0, idx_all.size, CH):
            idx = idx_all[s:s + CH]
            try:
                tri = finder(x[idx], y[idx])
                out[idx] = tri.astype(np.int64)
            except ValueError:
                for ii in idx:
                    try:
                        tri1 = finder(np.array([x[ii]]), np.array([y[ii]]))
                        out[ii] = int(tri1[0])
                    except ValueError:
                        out[ii] = -1
        return out

    return _safe


# =============================================================================
# Intensity + nodal expected counts
# =============================================================================

def intensity_lambda_nodes(
    sol: GSBSolution,
    funcs: GSBFunctions,
    params: Dict[str, float],
    k: int,
    lambda_floor: float = 1e-30,
) -> np.ndarray:
    """
    lambda_i(t_k) at nodes:
        lambda = rho_adopt * (p + q_I * F_I(I)) * (1 - u - v)
    Units: events / (km^2 * year)
    """
    p = float(params["p"])
    qI = float(params["q_I"])

    u = sol.U[k]
    v = sol.V[k]
    I = sol.I[k]
    rho = sol.rho_adopt[k]

    s = np.maximum(1.0 - u - v, 0.0)
    FI_I = funcs.F_I(I)
    lam = rho * (p + qI * FI_I) * s

    return np.maximum(lam, float(lambda_floor))


# =============================================================================
# Time bin edges
# =============================================================================

def _date_to_frac_year(ts: pd.Timestamp) -> float:
    year = int(ts.year)
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1)
    days_in_year = (end - start).days
    doy0 = (ts - start).days
    return float(year) + float(doy0) / float(days_in_year)


def make_month_edges_years(YEAR0: float, start_month: str, end_month: str) -> np.ndarray:
    start = pd.Timestamp(f"{start_month}-01")
    end = pd.Timestamp(f"{end_month}-01") + pd.offsets.MonthBegin(1)
    edges = pd.date_range(start=start, end=end, freq="MS")
    frac = np.array([_date_to_frac_year(ts) for ts in edges], dtype=float)
    return frac - float(YEAR0)


def make_year_edges_years(YEAR0: float, start_year: int, end_year: int) -> np.ndarray:
    start = pd.Timestamp(year=int(start_year), month=1, day=1)
    end_exclusive = pd.Timestamp(year=int(end_year) + 1, month=1, day=1)
    edges = pd.date_range(start=start, end=end_exclusive, freq="YS")
    edges = np.array(
        [float(ts.year + (ts.dayofyear - 1) / (366 if ts.is_leap_year else 365)) for ts in edges],
        dtype=float,
    )
    return edges - float(YEAR0)


# =============================================================================
# Data binning to NODES
# =============================================================================

def _project_lonlat_to_km(lon: np.ndarray, lat: np.ndarray, epsg_project: int) -> Tuple[np.ndarray, np.ndarray]:
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_project}", always_xy=True)
    x_m, y_m = tr.transform(lon.astype(float), lat.astype(float))
    return np.asarray(x_m, float) / 1000.0, np.asarray(y_m, float) / 1000.0


def bin_events_year_node(
    mesh: MeshTri,
    events_df: pd.DataFrame,
    epsg_project: int = 5070,
    t_min_year: int = 1998,
    t_max_year: int = 2025,
    chunk_size: int = 5000,
) -> Tuple[np.ndarray, int, int, np.ndarray, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Stream-bins events into node counts[nyears, N] by:
      - find containing triangle
      - add 1/3 count to each of its 3 vertices (mass-lumped event assignment)

    Returns:
        counts_node: float64 array (nyears, N)
        K_total_window: total events in year window
        K_inside: events inside mesh (and in window)
        year_labels: int32 array of calendar years per row
        min_ts/max_ts: date range among inside-mesh events
    """
    df = events_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "longitude", "latitude"]).copy()

    years = df["date"].dt.year.to_numpy(np.int32)
    mask_y = (years >= int(t_min_year)) & (years <= int(t_max_year))
    df = df.loc[mask_y].copy()
    dates = df["date"].to_numpy(dtype="datetime64[ns]")
    K_total_window = int(len(df))

    N = mesh.p.shape[1]
    ny = int(int(t_max_year) - int(t_min_year) + 1)
    year_labels = np.arange(int(t_min_year), int(t_max_year) + 1, dtype=np.int32)

    counts = np.zeros((ny, N), dtype=float)
    if K_total_window == 0:
        return counts, 0, 0, year_labels, None, None

    year_id = (df["date"].dt.year.to_numpy(np.int32) - int(t_min_year)).astype(np.int64)

    lon = df["longitude"].to_numpy(float)
    lat = df["latitude"].to_numpy(float)
    x_km, y_km = _project_lonlat_to_km(lon, lat, epsg_project=epsg_project)

    safe_finder = safe_element_finder(mesh, chunk_size=chunk_size)
    tri = mesh.t  # (3, ntri)

    K_inside = 0
    min_dt = None
    max_dt = None

    CH = int(chunk_size)
    for s in range(0, x_km.size, CH):
        j = slice(s, min(x_km.size, s + CH))

        tri_ids = safe_finder(x_km[j], y_km[j])  # -1 outside
        inside = tri_ids >= 0
        if not np.any(inside):
            continue

        tri_in = tri_ids[inside].astype(np.int64)
        yid_in = year_id[j][inside].astype(np.int64)
        dt_in = dates[j][inside]

        ok = (yid_in >= 0) & (yid_in < ny)
        tri_in = tri_in[ok]
        yid_in = yid_in[ok]
        dt_ok = dt_in[ok]
        if yid_in.size == 0:
            continue

        v0 = tri[0, tri_in]
        v1 = tri[1, tri_in]
        v2 = tri[2, tri_in]
        w = np.full(yid_in.shape[0], 1.0 / 3.0, dtype=float)

        np.add.at(counts, (yid_in, v0), w)
        np.add.at(counts, (yid_in, v1), w)
        np.add.at(counts, (yid_in, v2), w)

        K_inside += int(yid_in.size)

        dmin = dt_ok.min()
        dmax = dt_ok.max()
        min_dt = dmin if (min_dt is None or dmin < min_dt) else min_dt
        max_dt = dmax if (max_dt is None or dmax > max_dt) else max_dt

    min_ts = pd.to_datetime(min_dt) if min_dt is not None else None
    max_ts = pd.to_datetime(max_dt) if max_dt is not None else None
    return counts, K_total_window, K_inside, year_labels, min_ts, max_ts


def bin_events_month_total_inside_mesh(
    mesh: MeshTri,
    events_df: pd.DataFrame,
    epsg_project: int,
    start_month: str,
    end_month: str,
    chunk_size: int = 5000,
) -> Tuple[np.ndarray, List[pd.Timestamp], int, int]:
    """
    Monthly total counts (inside mesh only). This remains a TOTAL, not nodal.
    """
    df = events_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "longitude", "latitude"]).copy()

    start = pd.Timestamp(f"{start_month}-01")
    end_excl = pd.Timestamp(f"{end_month}-01") + pd.offsets.MonthBegin(1)
    mask = (df["date"] >= start) & (df["date"] < end_excl)
    df = df.loc[mask].copy()
    K_total_window = int(len(df))

    month_edges = pd.date_range(start=start, end=end_excl, freq="MS")
    month_labels = list(month_edges[:-1])
    nmonths = len(month_labels)

    y_month = np.zeros(nmonths, dtype=np.int32)
    if K_total_window == 0:
        return y_month, month_labels, 0, 0

    months = df["date"].dt.to_period("M").dt.to_timestamp()
    month_id = ((months.dt.year - start.year) * 12 + (months.dt.month - start.month)).to_numpy(np.int64)

    lon = df["longitude"].to_numpy(float)
    lat = df["latitude"].to_numpy(float)
    x_km, y_km = _project_lonlat_to_km(lon, lat, epsg_project=epsg_project)

    safe_finder = safe_element_finder(mesh, chunk_size=chunk_size)

    K_inside = 0
    CH = int(chunk_size)
    for s in range(0, x_km.size, CH):
        j = slice(s, min(x_km.size, s + CH))
        tri_ids = safe_finder(x_km[j], y_km[j])
        inside = tri_ids >= 0
        if not np.any(inside):
            continue

        mid = month_id[j][inside]
        ok = (mid >= 0) & (mid < nmonths)
        mid = mid[ok]
        if mid.size == 0:
            continue

        np.add.at(y_month, mid, 1)
        K_inside += int(mid.size)

    return y_month, month_labels, K_total_window, K_inside


# =============================================================================
# Expected counts on NODES
# =============================================================================

def expected_counts_year_node(
    sol: GSBSolution,
    funcs: GSBFunctions,
    params: Dict[str, float],
    year_edges_years: np.ndarray,
    lambda_floor: float = 1e-30,
) -> np.ndarray:
    """
    mu[y,i] = ∫_{year y} lambda_i(t) * A_i dt
    using trapezoid in time on solver snapshots.
    """
    times = sol.times
    nt = times.size
    N = sol.mesh.p.shape[1]
    A_nodes = sol.A_nodes  # (N,)

    lamA = np.zeros((nt, N), dtype=float)
    for k in range(nt):
        lam = intensity_lambda_nodes(sol, funcs, params, k, lambda_floor=lambda_floor)
        lamA[k] = lam * A_nodes

    n_years = len(year_edges_years) - 1
    mu = np.zeros((n_years, N), dtype=float)

    def interp_lamA(tq: float) -> np.ndarray:
        if tq <= times[0]:
            return lamA[0]
        if tq >= times[-1]:
            return lamA[-1]
        j = int(np.searchsorted(times, tq))
        i = j - 1
        t0, t1 = float(times[i]), float(times[j])
        w = (tq - t0) / (t1 - t0)
        return (1.0 - w) * lamA[i] + w * lamA[j]

    for y in range(n_years):
        a = float(year_edges_years[y])
        b = float(year_edges_years[y + 1])

        mask = (times >= a) & (times <= b)
        t_in = times[mask]
        r_in = lamA[mask]

        if t_in.size == 0 or float(t_in[0]) > a:
            t_in = np.concatenate([[a], t_in])
            r_in = np.vstack([interp_lamA(a)[None, :], r_in]) if r_in.size else interp_lamA(a)[None, :]

        if float(t_in[-1]) < b:
            t_in = np.concatenate([t_in, [b]])
            r_in = np.vstack([r_in, interp_lamA(b)[None, :]])

        dt = np.diff(t_in)
        avg = 0.5 * (r_in[:-1] + r_in[1:])
        mu[y] = np.sum(avg * dt[:, None], axis=0)

    return np.maximum(mu, 1e-300)


def expected_counts_month_total(
    sol: GSBSolution,
    funcs: GSBFunctions,
    params: Dict[str, float],
    month_edges_years: np.ndarray,
    lambda_floor: float = 1e-30,
) -> np.ndarray:
    """
    Monthly total expected counts:
        mu_month[m] = ∫_{month m} Σ_i lambda_i(t) * A_i dt
    """
    times = sol.times
    nt = times.size
    A_nodes = sol.A_nodes

    rate_total = np.zeros(nt, dtype=float)
    for k in range(nt):
        lam = intensity_lambda_nodes(sol, funcs, params, k, lambda_floor=lambda_floor)
        rate_total[k] = float(np.sum(lam * A_nodes))

    def interp_rate(tq: float) -> float:
        if tq <= times[0]:
            return float(rate_total[0])
        if tq >= times[-1]:
            return float(rate_total[-1])
        j = int(np.searchsorted(times, tq))
        i = j - 1
        t0, t1 = float(times[i]), float(times[j])
        w = (tq - t0) / (t1 - t0)
        return float((1.0 - w) * rate_total[i] + w * rate_total[j])

    nmonths = len(month_edges_years) - 1
    mu_month = np.zeros(nmonths, dtype=float)

    for m in range(nmonths):
        a = float(month_edges_years[m])
        b = float(month_edges_years[m + 1])

        mask = (times >= a) & (times <= b)
        t_in = times[mask]
        r_in = rate_total[mask]

        if t_in.size == 0 or float(t_in[0]) > a:
            t_in = np.concatenate([[a], t_in])
            r_in = np.concatenate([[interp_rate(a)], r_in])

        if float(t_in[-1]) < b:
            t_in = np.concatenate([t_in, [b]])
            r_in = np.concatenate([r_in, [interp_rate(b)]])

        dt = np.diff(t_in)
        avg = 0.5 * (r_in[:-1] + r_in[1:])
        mu_month[m] = float(np.sum(avg * dt))

    return np.maximum(mu_month, 0.0)


# =============================================================================
# Deviance / residuals
# =============================================================================

def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> Tuple[float, np.ndarray]:
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    mu = np.maximum(mu, 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0.0, y * np.log(y / mu), 0.0)
    D_bin = 2.0 * (term - (y - mu))
    D_bin = np.maximum(D_bin, 0.0)
    return float(np.sum(D_bin)), D_bin


def pearson_residuals(y: np.ndarray, mu: np.ndarray, mu_floor: float = 1e-12) -> np.ndarray:
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    mu = np.maximum(mu, float(mu_floor))
    return (y - mu) / np.sqrt(mu)


# =============================================================================
# Bass model: total counts (unchanged)
# =============================================================================

def bass_F(t: np.ndarray, p: float, q: float) -> np.ndarray:
    t = np.asarray(t, float)
    p = float(max(p, 1e-12))
    q = float(max(q, 0.0))
    a = p + q
    e = np.exp(-a * t)
    return (1.0 - e) / (1.0 + (q / p) * e)


def fit_bass_to_monthly_counts(
    y_month: np.ndarray,
    month_edges_years_rel: np.ndarray,
    p0: float = 0.01,
    q0: float = 0.1,
    M0: Optional[float] = None,
) -> Tuple[float, float, float, np.ndarray]:
    y = np.asarray(y_month, float)
    y = np.maximum(y, 0.0)

    t0 = float(month_edges_years_rel[0])
    t_edges = np.asarray(month_edges_years_rel, float) - t0

    if float(t_edges[-1]) <= 0.0 or y.size == 0:
        return float(p0), float(q0), 0.0, np.zeros_like(y)

    total_y = float(np.sum(y))
    if M0 is None:
        M0 = max(1.05 * total_y, 1.0)

    def predict(p: float, q: float, M: float) -> np.ndarray:
        F_edges = bass_F(t_edges, p, q)
        yhat = float(M) * np.diff(F_edges)
        return np.maximum(yhat, 0.0)

    def obj(theta: np.ndarray) -> float:
        p, q, M = float(theta[0]), float(theta[1]), float(theta[2])
        if p <= 0.0 or q < 0.0 or M <= 0.0:
            return 1e50
        yhat = predict(p, q, M)
        return float(np.sum((y - yhat) ** 2))

    res = minimize(
        obj,
        x0=np.array([p0, q0, M0], dtype=float),
        method="L-BFGS-B",
        bounds=[(1e-10, 10.0), (0.0, 200.0), (1e-6, 1e15)],
        options=dict(maxiter=800),
    )

    p_hat, q_hat, M_hat = float(res.x[0]), float(res.x[1]), float(res.x[2])
    yhat = predict(p_hat, q_hat, M_hat)
    return p_hat, q_hat, M_hat, yhat


# =============================================================================
# Plot helpers (unchanged code, but allow Runner logging via wrapper)
# =============================================================================

def _plot_cities_lonlat(
    ax,
    cities: Dict[str, Sequence[float]],
    marker_size: float = 70.0,
    text_alpha: float = 0.65,
    text_dx: float = 0.08,
    text_dy: float = 0.06,
):
    if not cities:
        return
    for name, xy in cities.items():
        if xy is None or len(xy) != 2:
            continue
        lon, lat = float(xy[0]), float(xy[1])
        ax.scatter([lon], [lat], marker="*", s=marker_size, c="red", linewidths=0.0, zorder=5)
        ax.text(lon + text_dx, lat + text_dy, str(name), color="red", alpha=float(text_alpha), fontsize=10, zorder=6)


def _plot_data_vs_mu_year_nodes_lonlat(
    msh_path: Path,
    epsg_project: int,
    out_png: Path,
    y_node: np.ndarray,
    mu_node: np.ndarray,
    year: int,
    h_km: float,
    cities: Optional[Dict[str, Sequence[float]]] = None,
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Ellipse
    from matplotlib.collections import PatchCollection

    mi = meshio.read(msh_path)
    tri = None
    for c in mi.cells:
        if c.type == "triangle":
            tri = c.data
            break
    if tri is None:
        raise ValueError("No triangle cells in mesh for plotting.")

    inv = Transformer.from_crs(f"EPSG:{epsg_project}", "EPSG:4326", always_xy=True)
    pts_km = mi.points[:, :2]
    lon, lat = inv.transform(pts_km[:, 0] * 1000.0, pts_km[:, 1] * 1000.0)
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)

    y = np.maximum(np.asarray(y_node, float), 0.0)
    mu = np.maximum(np.asarray(mu_node, float), 0.0)

    y_plot = np.log1p(y)
    mu_plot = np.log1p(mu)

    finite_vals = np.concatenate([y_plot[np.isfinite(y_plot)], mu_plot[np.isfinite(mu_plot)]])
    if finite_vals.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.min(finite_vals))
        vmax = float(np.max(finite_vals))
        if vmax <= vmin:
            vmax = vmin + 1.0

    norm = Normalize(vmin=vmin, vmax=vmax)

    r_km = (np.sqrt(3.0) / 4.0) * float(h_km)
    KM_PER_DEG_LAT = 111.32
    dy_deg = r_km / KM_PER_DEG_LAT
    dx_deg = r_km / (KM_PER_DEG_LAT * np.clip(np.cos(np.deg2rad(lat)), 1e-6, None))

    def make_collection(values_log1p: np.ndarray) -> PatchCollection:
        patches = [Ellipse((float(x), float(y)), width=float(2.0 * wx), height=float(2.0 * dy_deg))
                   for x, y, wx in zip(lon, lat, dx_deg)]
        pc = PatchCollection(patches, array=values_log1p, norm=norm, linewidths=0.0)
        return pc

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)

    pcL = make_collection(y_plot)
    axes[0].add_collection(pcL)
    axes[0].autoscale_view()
    _plot_cities_lonlat(axes[0], cities or {})
    axes[0].set_title(f"Observed node counts (log(1+y)), {year}")
    axes[0].set_xlabel("Longitude (deg)")
    axes[0].set_ylabel("Latitude (deg)")
    axes[0].set_aspect("equal", adjustable="box")

    pcR = make_collection(mu_plot)
    axes[1].add_collection(pcR)
    axes[1].autoscale_view()
    _plot_cities_lonlat(axes[1], cities or {})
    axes[1].set_title(f"Predicted node counts (log(1+μ)), {year}")
    axes[1].set_xlabel("Longitude (deg)")
    axes[1].set_ylabel("Latitude (deg)")
    axes[1].set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(pcR, ax=axes, location="right", fraction=0.035, pad=0.02)
    cbar.set_label("log(1 + count)")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _signed_log1p_scale(x: np.ndarray, eps: float = 0.0) -> np.ndarray:
    x = np.asarray(x, float)
    return np.sign(x) * np.log1p(np.abs(x) + float(eps))


def _plot_node_circles_lonlat_single(
    msh_path: Path,
    epsg_project: int,
    out_png: Path,
    values: np.ndarray,
    title: str,
    h_km: float,
    cities: Optional[Dict[str, Sequence[float]]] = None,
    vlim: Optional[float] = None,
    cbar_label: str = "",
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Ellipse
    from matplotlib.collections import PatchCollection
    from matplotlib.ticker import ScalarFormatter

    mi = meshio.read(msh_path)

    inv = Transformer.from_crs(f"EPSG:{epsg_project}", "EPSG:4326", always_xy=True)
    pts_km = mi.points[:, :2]
    lon, lat = inv.transform(pts_km[:, 0] * 1000.0, pts_km[:, 1] * 1000.0)
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)

    vals = np.asarray(values, float)

    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        vmin, vmax = -1.0, 1.0
    else:
        if vlim is not None:
            L = float(max(vlim, 1e-12))
            vmin, vmax = -L, L
        else:
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
            if vmax <= vmin:
                vmax = vmin + 1.0

    norm = Normalize(vmin=vmin, vmax=vmax)

    r_km = (np.sqrt(3.0) / 4.0) * float(h_km)
    KM_PER_DEG_LAT = 111.32
    dy_deg = r_km / KM_PER_DEG_LAT
    dx_deg = r_km / (KM_PER_DEG_LAT * np.clip(np.cos(np.deg2rad(lat)), 1e-6, None))

    patches = [
        Ellipse((float(x), float(y)), width=float(2.0 * wx), height=float(2.0 * dy_deg))
        for x, y, wx in zip(lon, lat, dx_deg)
    ]
    pc = PatchCollection(patches, array=vals, norm=norm, linewidths=0.0)

    fig, ax = plt.subplots(1, 1, figsize=(9, 7), constrained_layout=True)
    ax.add_collection(pc)
    ax.autoscale_view()
    _plot_cities_lonlat(ax, cities or {})
    ax.set_title(title)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(pc, ax=ax, location="right", fraction=0.045, pad=0.02)
    if cbar_label:
        cbar.set_label(cbar_label)
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    cbar.formatter = fmt
    cbar.update_ticks()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_pearson_residuals_year_nodes_lonlat(
    msh_path: Path,
    epsg_project: int,
    out_png: Path,
    R_node: np.ndarray,
    year: int,
    h_km: float,
    cities: Optional[Dict[str, Sequence[float]]] = None,
    vlim_log: float = 2.5,
):
    R = np.asarray(R_node, float)
    R_plot = _signed_log1p_scale(R)

    _plot_node_circles_lonlat_single(
        msh_path=msh_path,
        epsg_project=epsg_project,
        out_png=out_png,
        values=R_plot,
        title=f"Pearson residuals (signed log1p), {year}",
        h_km=h_km,
        cities=cities,
        vlim=float(vlim_log),
        cbar_label="sign(R) · log(1 + |R|)",
    )
    
    
def _nice_vmax_1digit(M: float) -> float:
    """
    Smallest number >= M of the form d * 10^k with d in {1,...,9}.
    If M <= 0, returns 1.0 (non-degenerate colorbar).
    """
    M = float(M)
    if not np.isfinite(M) or M <= 0.0:
        return 1.0

    k = int(math.floor(math.log10(M)))
    s = M / (10.0 ** k)              # in [1, 10)
    d = int(math.ceil(s - 1e-15))    # avoid floating one-off
    d = min(max(d, 1), 9)
    return float(d) * (10.0 ** k)


def _fmt_plain_or_phys(x: float) -> str:
    """
    Format:
      - decimal if |x| in [1e-3, 9e2]
      - otherwise "d×10^k" style (mathtext) with compact mantissa
    """
    x = float(x)
    if not np.isfinite(x):
        return ""
    if x == 0.0:
        return "0"

    ax = abs(x)
    if 1e-3 <= ax <= 9e2:
        # decimal, avoid trailing zeros
        s = f"{x:.6f}".rstrip("0").rstrip(".")
        return s

    k = int(math.floor(math.log10(ax)))
    m = x / (10.0 ** k)

    # compact mantissa: 1–3 significant digits, trim zeros
    m_str = f"{m:.3g}"
    # enforce e.g. "-0.0006" doesn't happen here
    return rf"${m_str}\times 10^{{{k}}}$"


def _apply_colorbar_format(cbar, vmax: float) -> None:
    """
    Use just a few ticks and human-friendly labels.
    Requirement focus: make max label clean.
    """
    vmax = float(vmax)
    ticks = np.array([0.0, 0.5 * vmax, vmax], dtype=float)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([_fmt_plain_or_phys(t) for t in ticks])    
    
    
def _fmt_plain_or_phys_001_999(x: float) -> str:
    """
    If |x| in [1e-3, 9.99e2] => decimal string.
    Else => mathtext like $4\\times 10^{-3}$.
    """
    x = float(x)
    if not np.isfinite(x):
        return ""
    if x == 0.0:
        return "0"

    ax = abs(x)
    if 1e-3 <= ax <= 9.99e2:
        # exact-ish decimal, but don't print tons of junk
        s = f"{x:.10f}".rstrip("0").rstrip(".")
        # avoid "-0"
        return "0" if s in ("-0", "+0") else s

    k = int(math.floor(math.log10(ax)))
    m = x / (10.0 ** k)
    # single-digit mantissa (rounded) is usually enough for ticks; but keep 2–3 sig figs
    m_str = f"{m:.3g}"
    return rf"${m_str}\times 10^{{{k}}}$"


def _apply_colorbar_ticklabels(cbar, vmin: float, vmax: float) -> None:
    """
    Use three ticks: min, mid, max.
    Labels follow _fmt_plain_or_phys_001_999.
    """
    vmin = float(vmin)
    vmax = float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return

    if vmax == vmin:
        ticks = [vmin]
    else:
        ticks = [vmin, 0.5 * (vmin + vmax), vmax]

    cbar.set_ticks(ticks)
    cbar.set_ticklabels([_fmt_plain_or_phys_001_999(t) for t in ticks])


def _plot_uvIJ_and_w_year_lonlat(
    msh_path: Path,
    epsg_project: int,
    out_png_uvij: Path,
    out_png_w: Path,
    u: np.ndarray,
    v: np.ndarray,
    I: np.ndarray,
    J: np.ndarray,
    year: int,
    h_km: float,
    cities: Optional[Dict[str, Sequence[float]]] = None,
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Ellipse
    from matplotlib.collections import PatchCollection

    mi = meshio.read(msh_path)
    inv = Transformer.from_crs(f"EPSG:{epsg_project}", "EPSG:4326", always_xy=True)
    pts_km = mi.points[:, :2]
    lon, lat = inv.transform(pts_km[:, 0] * 1000.0, pts_km[:, 1] * 1000.0)
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)

    r_km = (np.sqrt(3.0) / 4.0) * float(h_km)
    KM_PER_DEG_LAT = 111.32
    dy_deg = r_km / KM_PER_DEG_LAT
    dx_deg = r_km / (KM_PER_DEG_LAT * np.clip(np.cos(np.deg2rad(lat)), 1e-6, None))

    patches = [
        Ellipse((float(x), float(y)), width=float(2.0 * wx), height=float(2.0 * dy_deg))
        for x, y, wx in zip(lon, lat, dx_deg)
    ]

    u = np.asarray(u, float)
    v = np.asarray(v, float)
    I = np.asarray(I, float)
    J = np.asarray(J, float)
    w = np.maximum(1.0 - u - v, 0.0)

    def _finite_minmax(vals: np.ndarray) -> Tuple[float, float]:
        vals = np.asarray(vals, float)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            return 0.0, 1.0
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        if vmax == vmin:
            # avoid a degenerate colormap; widen slightly
            eps = 1e-12 if vmax == 0.0 else 1e-6 * abs(vmax)
            return vmin - eps, vmax + eps
        return vmin, vmax

    umin, umax = _finite_minmax(u)
    vmin, vmax = _finite_minmax(v)
    Imin, Imax = _finite_minmax(I)
    Jmin, Jmax = _finite_minmax(J)
    wmin, wmax = _finite_minmax(w)

    u_norm = Normalize(vmin=umin, vmax=umax)
    v_norm = Normalize(vmin=vmin, vmax=vmax)
    I_norm = Normalize(vmin=Imin, vmax=Imax)
    J_norm = Normalize(vmin=Jmin, vmax=Jmax)
    w_norm = Normalize(vmin=wmin, vmax=wmax)

    def _add_panel(ax, vals, title, norm: Normalize) -> PatchCollection:
        vals = np.asarray(vals, float)
        # keep NaNs from breaking PatchCollection coloring:
        vals_plot = vals.copy()
        if np.any(~np.isfinite(vals_plot)):
            # put non-finite on vmin (harmless)
            vals_plot[~np.isfinite(vals_plot)] = float(norm.vmin)
        pc = PatchCollection(patches, array=vals_plot, norm=norm, linewidths=0.0)
        ax.add_collection(pc)
        ax.autoscale_view()
        _plot_cities_lonlat(ax, cities or {})
        ax.set_title(title)
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.set_aspect("equal", adjustable="box")
        return pc

    # 2x2: u,v,I,J (each with its own colorbar)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)

    pc_u = _add_panel(axes[0, 0], u, f"u (innovators fraction), {year}", norm=u_norm)
    cbar_u = fig.colorbar(pc_u, ax=axes[0, 0], location="right", fraction=0.046, pad=0.02)
    cbar_u.set_label("u (linear)")
    _apply_colorbar_ticklabels(cbar_u, umin, umax)

    pc_v = _add_panel(axes[0, 1], v, f"v (imitators fraction), {year}", norm=v_norm)
    cbar_v = fig.colorbar(pc_v, ax=axes[0, 1], location="right", fraction=0.046, pad=0.02)
    cbar_v.set_label("v (linear)")
    _apply_colorbar_ticklabels(cbar_v, vmin, vmax)

    pc_I = _add_panel(axes[1, 0], I, f"I field, {year}", norm=I_norm)
    cbar_I = fig.colorbar(pc_I, ax=axes[1, 0], location="right", fraction=0.046, pad=0.02)
    cbar_I.set_label("I (linear)")
    _apply_colorbar_ticklabels(cbar_I, Imin, Imax)

    pc_J = _add_panel(axes[1, 1], J, f"J field, {year}", norm=J_norm)
    cbar_J = fig.colorbar(pc_J, ax=axes[1, 1], location="right", fraction=0.046, pad=0.02)
    cbar_J.set_label("J (linear)")
    _apply_colorbar_ticklabels(cbar_J, Jmin, Jmax)

    out_png_uvij.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png_uvij, dpi=200)
    plt.close(fig)

    # w plot (separate figure + its own colorbar)
    figw, axw = plt.subplots(1, 1, figsize=(9, 7), constrained_layout=True)
    w_plot = w.copy()
    if np.any(~np.isfinite(w_plot)):
        w_plot[~np.isfinite(w_plot)] = float(w_norm.vmin)
    pcw = PatchCollection(patches, array=w_plot, norm=w_norm, linewidths=0.0)
    axw.add_collection(pcw)
    axw.autoscale_view()
    _plot_cities_lonlat(axw, cities or {})
    axw.set_title(f"w = 1 - u - v, {year}")
    axw.set_xlabel("Longitude (deg)")
    axw.set_ylabel("Latitude (deg)")
    axw.set_aspect("equal", adjustable="box")

    cbar_w = figw.colorbar(pcw, ax=axw, location="right", fraction=0.045, pad=0.02)
    cbar_w.set_label("w (linear)")
    _apply_colorbar_ticklabels(cbar_w, wmin, wmax)

    out_png_w.parent.mkdir(parents=True, exist_ok=True)
    figw.savefig(out_png_w, dpi=200)
    plt.close(figw)


def _plot_total_counts_monthly_bass_vs_gsb_vs_data(
    sol: GSBSolution,
    funcs: GSBFunctions,
    params: Dict[str, float],
    events_df: pd.DataFrame,
    epsg_project: int,
    out_png: Path,
    start_month: str,
    end_month: str,
    chunk_size: int = 2000,
    lambda_floor: float = 1e-30,
):
    y_month, month_labels, K_total, K_inside = bin_events_month_total_inside_mesh(
        mesh=sol.mesh,
        events_df=events_df,
        epsg_project=epsg_project,
        start_month=start_month,
        end_month=end_month,
        chunk_size=chunk_size,
    )

    month_edges_years = make_month_edges_years(sol.YEAR0, start_month=start_month, end_month=end_month)
    mu_month = expected_counts_month_total(
        sol=sol,
        funcs=funcs,
        params=params,
        month_edges_years=month_edges_years,
        lambda_floor=lambda_floor,
    )

    p_hat, q_hat, M_hat, bass_month = fit_bass_to_monthly_counts(
        y_month=y_month,
        month_edges_years_rel=month_edges_years,
        p0=float(params.get("p", 0.01)),
        q0=float(params.get("q_I", 0.1)),
        M0=None,
    )

    import matplotlib.pyplot as plt
    x = np.array(month_labels)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(x, y_month.astype(float), label="Data (inside mesh)")
    ax.plot(x, bass_month.astype(float), label=f"Bass fit (p={p_hat:.3g}, q={q_hat:.3g})")
    ax.plot(x, mu_month.astype(float), label="GSB model (integrated λ over nodes)")

    ax.set_title(f"Total monthly counts: Data vs Bass vs GSB ({start_month} to {end_month})")
    ax.set_xlabel("Month")
    ax.set_ylabel("Count per month")
    ax.grid(True, alpha=0.25)
    ax.legend()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# =============================================================================
# Runner: mesh build + FEM run + diagnostics
# =============================================================================

@dataclass
class Runner:
    out_folder: str
    mesh_params: Dict[str, Any]
    model_params: Dict[str, float]
    time_params: Dict[str, float]

    fem_verbose: bool = False
    mesh_verbose: bool = False

    # Optional diagnostics controls
    cities: Dict[str, Sequence[float]] = field(default_factory=dict)
    years_to_plot: Optional[List[int]] = None
    month_window: Tuple[str, str] = ("1998-01", "2023-12")
    events_csv: Path = Path("data") / "processed" / "solar_installations_all.csv"
    events_state_col: str = "state"

    base_out: Path = Path("out")

    _t0: float = field(init=False, repr=False)

    def __post_init__(self):
        self._t0 = time.perf_counter()

    # ---- paths ----
    @property
    def out_dir(self) -> Path:
        return self.base_out / self.out_folder

    @property
    def mesh_dir(self) -> Path:
        return self.out_dir / "mesh"

    @property
    def fig_dir(self) -> Path:
        return self.out_dir / "figures"

    def mesh_path(self) -> Path:
        h_km = int(self.mesh_params["h_km"])
        simplify_km = int(self.mesh_params["simplify_km"])
        return self.mesh_dir / f"{h_km}_{simplify_km}_km.msh"

    # ---- logging ----
    def log(self, msg: str) -> None:
        dt = time.perf_counter() - self._t0
        print(f"{self.out_folder}@[{_fmt_hhmmss(dt)}] ---- {msg}")

    # ---- configuration builders ----
    def build_fem_config(self) -> FEMConfig:
        tau = float(self.time_params.get("tau", 0.05))
        start_year = float(self.time_params["start_year"])
        # Keep your previous default “T_years” behavior: run until 2023 if not given
        # (you can override by providing T_years explicitly in time_params).
        T_years = float(self.time_params.get("T_years", max(0.0, 2023.0 - start_year)))

        epsg = int(self.mesh_params.get("epsg_project", 5070))
        return FEMConfig(
            tau_years=tau,
            T_years=T_years,
            picard_max_iter=int(self.time_params.get("picard_max_iter", 20)),
            picard_tol=float(self.time_params.get("picard_tol", 1e-8)),
            verbose=bool(self.fem_verbose),
            YEAR0=float(start_year),
            epsg_project=epsg,
        )

    def build_functions(self) -> GSBFunctions:
        return make_default_gsb_functions(self.model_params)

    def build_params(self) -> Dict[str, float]:
        # Keep only the solver-needed parameters here.
        return dict(
            r=float(self.model_params.get("r", 1.0)),
            p=float(self.model_params.get("p", 0.01)),
            q_I=float(self.model_params.get("q_I", 0.1)),
            k_J=float(self.model_params.get("k_J", 0.0)),
            D=float(self.model_params.get("D", 0.0)),
            gamma_J=float(self.model_params.get("gamma_J", 0.0)),
            S0=float(self.model_params.get("S0", 0.0)),
        )

    # ---- part 1: mesh building ----
    def build_mesh(self) -> Path:
        """
        Builds mesh into: out/<out_folder>/mesh/<h>_<simplify>_km.msh
        """
        from mesh_utils import MeshBuildConfig, build_mesh_from_admin1_region

        self.mesh_dir.mkdir(parents=True, exist_ok=True)
        msh = self.mesh_path()

        admin1_shp = Path("data") / "raw" / "maps" / "ne_10m_admin_1_states_provinces_lakes" / "ne_10m_admin_1_states_provinces_lakes.shp"
        state_list = list(self.mesh_params["state_list"])
        h_km = float(self.mesh_params["h_km"])
        simplify_km = float(self.mesh_params["simplify_km"])
        epsg = int(self.mesh_params.get("epsg_project", 5070))

        cfg = MeshBuildConfig(h_km=h_km, simplify_km=simplify_km, epsg_project=epsg)

        if msh.exists():
            self.log(f"mesh exists: {msh.name} (skipping build)")
            self.log("self.build_mesh complete")
            return msh

        t0 = time.perf_counter()
        build_mesh_from_admin1_region(
            admin1_shp,
            state_list,
            msh,
            cfg,
            verbose=bool(self.mesh_verbose),
            model_name=f"{self.out_folder}_mesh_km",
        )
        self.log(f"build_mesh_from_admin1_region complete ({time.perf_counter() - t0:.3f} s)")
        self.log("self.build_mesh complete")
        return msh

    # ---- part 2: FEM + diagnostics ----
    def run_FEM(self) -> Dict[str, Any]:
        """
        Runs solver, computes diagnostics, saves plots in:
          out/<out_folder>/figures/
        Returns a small summary dict (useful for batch runs).
        """
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        msh = self.mesh_path()
        if not msh.exists():
            raise FileNotFoundError(f"Mesh not found: {msh}. Call build_mesh() first (or create it).")

        fem_cfg = self.build_fem_config()
        funcs = self.build_functions()
        params = self.build_params()

        # --- cache ---
        t0 = time.perf_counter()
        cache = build_fem_stage_cache(msh, fem_cfg, log=lambda s: self.log(s))
        self.log(f"build_fem_stage_cache complete ({time.perf_counter() - t0:.3f} s)")

        # --- solve ---
        t0 = time.perf_counter()
        sol = solve_gsb_fem(msh, funcs, params, fem_cfg, cache=cache, log=lambda s: self.log(s))
        self.log(f"solve_gsb_fem complete ({time.perf_counter() - t0:.3f} s)")

        # --- events load + filter ---
        events_df = pd.read_csv(self.events_csv)
        if "date" not in events_df.columns or "longitude" not in events_df.columns or "latitude" not in events_df.columns:
            raise ValueError("events CSV must contain: date, longitude, latitude.")
        if self.events_state_col not in events_df.columns:
            raise ValueError(f"events CSV must contain a '{self.events_state_col}' column.")

        events_df[self.events_state_col] = events_df[self.events_state_col].astype(str).str.strip()
        state_list = list(self.mesh_params["state_list"])
        if len(state_list) != 1:
            raise ValueError("Runner.run_FEM currently expects exactly one state in state_list (for CSV filtering).")
        st = str(state_list[0]).strip()
        events_df = events_df.loc[events_df[self.events_state_col] == st].copy()

        # --- year window ---
        start_year = float(self.time_params["start_year"])
        # Max year defaults to 2023 unless overridden
        t_min_year = int(self.time_params.get("t_min_year", int(np.floor(start_year))))
        t_max_year = int(self.time_params.get("t_max_year", 2023))

        year_edges_years = make_year_edges_years(
            YEAR0=fem_cfg.YEAR0,
            start_year=t_min_year,
            end_year=t_max_year,
        )

        # --- bin yearly node counts ---
        t0 = time.perf_counter()
        counts_node, K_total, K_inside, year_labels, min_ts, max_ts = bin_events_year_node(
            mesh=sol.mesh,
            events_df=events_df,
            epsg_project=fem_cfg.epsg_project,
            t_min_year=t_min_year,
            t_max_year=t_max_year,
            chunk_size=int(self.time_params.get("bin_chunk_size", 5000)),
        )
        self.log(f"bin_events_year_node complete ({time.perf_counter() - t0:.3f} s)")
        self.log(f"[bin] window events={K_total:,} inside mesh={K_inside:,} shape={counts_node.shape}")
        if min_ts is not None and max_ts is not None:
            self.log(f"[bin] inside-mesh date range: {min_ts.date()} → {max_ts.date()}")

        # --- expected counts ---
        t0 = time.perf_counter()
        mu_node = expected_counts_year_node(
            sol=sol,
            funcs=funcs,
            params=params,
            year_edges_years=year_edges_years,
            lambda_floor=float(self.time_params.get("lambda_floor", 1e-30)),
        )
        self.log(f"expected_counts_year_node complete ({time.perf_counter() - t0:.3f} s)")
        self.log(f"[mu] total expected={float(mu_node.sum()):.6g}")

        if mu_node.shape != counts_node.shape:
            raise RuntimeError(f"Shape mismatch: counts_node {counts_node.shape} vs mu_node {mu_node.shape}")

        # --- deviance + residuals ---
        t0 = time.perf_counter()
        D_total, _ = poisson_deviance(counts_node, mu_node)
        self.log(f"poisson_deviance complete ({time.perf_counter() - t0:.3f} s)")
        self.log(f"[deviance] D_total = {D_total:.6e}")

        t0 = time.perf_counter()
        R = pearson_residuals(counts_node, mu_node, mu_floor=float(self.time_params.get("mu_floor", 1e-12)))
        self.log(f"pearson_residuals complete ({time.perf_counter() - t0:.3f} s)")

        R_mean = float(np.mean(R))
        R_mabs = float(np.mean(np.abs(R)))
        R_rms = float(np.sqrt(np.mean(R ** 2)))
        self.log(f"[pearson] mean={R_mean:.3e} mean|R|={R_mabs:.3e} rms={R_rms:.3e}")

        # --- plots ---
        h_km = float(self.mesh_params["h_km"])
        years_to_plot = self.years_to_plot
        if years_to_plot is None:
            years_to_plot = [t_min_year, t_min_year + 3, t_min_year + 6, t_min_year + 9, t_max_year - 4, t_max_year - 2, t_max_year]
            years_to_plot = sorted({yy for yy in years_to_plot if t_min_year <= yy <= t_max_year})

        for yy in years_to_plot:
            idx = int(yy - t_min_year)
            if idx < 0 or idx >= counts_node.shape[0]:
                self.log(f"[warn] year {yy} out of range for bins; skipping")
                continue

            y_node = counts_node[idx, :]
            mu_y = mu_node[idx, :]

            out_png = self.fig_dir / f"{st.lower()}_data_vs_mu_nodes_{yy}.png"
            t0 = time.perf_counter()
            _plot_data_vs_mu_year_nodes_lonlat(
                msh_path=msh,
                epsg_project=fem_cfg.epsg_project,
                out_png=out_png,
                y_node=y_node,
                mu_node=mu_y,
                year=int(yy),
                h_km=h_km,
                cities=self.cities,
            )
            self.log(f"_plot_data_vs_mu_year_nodes_lonlat complete ({time.perf_counter() - t0:.3f} s)")

            out_png_R = self.fig_dir / f"{st.lower()}_pearson_residual_nodes_{yy}.png"
            t0 = time.perf_counter()
            _plot_pearson_residuals_year_nodes_lonlat(
                msh_path=msh,
                epsg_project=fem_cfg.epsg_project,
                out_png=out_png_R,
                R_node=R[idx, :],
                year=int(yy),
                h_km=h_km,
                cities=self.cities,
                vlim_log=float(self.time_params.get("vlim_log", 2.5)),
            )
            self.log(f"_plot_pearson_residuals_year_nodes_lonlat complete ({time.perf_counter() - t0:.3f} s)")

            # choose snapshot closest to Jan 1 of yy
            t_target = float(yy - fem_cfg.YEAR0)
            k_snap = int(np.argmin(np.abs(sol.times - t_target)))

            out_png_uvij = self.fig_dir / f"{st.lower()}_uvIJ_nodes_{yy}.png"
            out_png_w = self.fig_dir / f"{st.lower()}_w_nodes_{yy}.png"
            t0 = time.perf_counter()
            _plot_uvIJ_and_w_year_lonlat(
                msh_path=msh,
                epsg_project=fem_cfg.epsg_project,
                out_png_uvij=out_png_uvij,
                out_png_w=out_png_w,
                u=sol.U[k_snap],
                v=sol.V[k_snap],
                I=sol.I[k_snap],
                J=sol.J[k_snap],
                year=int(yy),
                h_km=h_km,
                cities=self.cities,
            )
            self.log(f"_plot_uvIJ_and_w_year_lonlat complete ({time.perf_counter() - t0:.3f} s)")

        # --- monthly totals plot ---
        start_month = f"{int(t_min_year):04d}-01"
        end_month   = f"{int(t_max_year):04d}-12"
        monthly_png = self.fig_dir / f"{st.lower()}_monthly_totals_bass_vs_gsb_vs_data.png"
        t0 = time.perf_counter()
        _plot_total_counts_monthly_bass_vs_gsb_vs_data(
            sol=sol,
            funcs=funcs,
            params=params,
            events_df=events_df,
            epsg_project=fem_cfg.epsg_project,
            out_png=monthly_png,
            start_month=start_month,
            end_month=end_month,
            chunk_size=int(self.time_params.get("bin_chunk_size", 5000)),
            lambda_floor=float(self.time_params.get("lambda_floor", 1e-30)),
        )
        self.log(f"_plot_total_counts_monthly_bass_vs_gsb_vs_data complete ({time.perf_counter() - t0:.3f} s)")

        self.log("self.run_FEM complete")
        return dict(
            out_folder=self.out_folder,
            state=st,
            mesh=str(msh),
            figures=str(self.fig_dir),
            K_total=K_total,
            K_inside=K_inside,
            deviance=D_total,
            pearson_rms=R_rms,
        )


# =============================================================================
# Parallel helpers
# =============================================================================

def _runner_worker(spec: Dict[str, Any], do_mesh: bool, do_fem: bool) -> Dict[str, Any]:
    r = Runner(**spec)
    if do_mesh:
        r.build_mesh()
    out = {}
    if do_fem:
        out = r.run_FEM()
    return out


def run_parallel(
    runners: Sequence[Runner],
    do_mesh: bool = True,
    do_fem: bool = True,
    max_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Runs multiple Runner objects in parallel using process workers.
    This is why Runner is designed to be pickle-friendly (no lambdas in fields).
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    specs: List[Dict[str, Any]] = []
    for r in runners:
        # Convert to a plain dict spec (pickle-friendly).
        specs.append(dict(
            out_folder=r.out_folder,
            mesh_params=r.mesh_params,
            model_params=r.model_params,
            time_params=r.time_params,
            fem_verbose=r.fem_verbose,
            mesh_verbose=r.mesh_verbose,
            cities=r.cities,
            years_to_plot=r.years_to_plot,
            month_window=r.month_window,
            events_csv=r.events_csv,
            events_state_col=r.events_state_col,
            base_out=r.base_out,
        ))

    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_runner_worker, spec, do_mesh, do_fem) for spec in specs]
        for fut in as_completed(futs):
            results.append(fut.result())
    return results


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    runner = Runner(
        out_folder="ny_run6",
        mesh_params=dict(
            state_list=["NY"],
            h_km=6,
            simplify_km=18,
            epsg_project=5070,
        ),
        model_params=dict(
            r=0.229142,
            p=3.47502e-05,
            q_I=0.00616679,
            gamma_J=0.000594882,
            k_J=0.00141604,
            D=0.0115039,
            S0=12.795,
        ),
        time_params=dict(
            start_year=2004.000,
            tau=0.025,
            T_years=20.000, 
            picard_max_iter=20,
            picard_tol=1e-8,
            t_min_year=2002,
            t_max_year=2024,
        ),
        fem_verbose=False,
        mesh_verbose=False,
    )

    # Optional: cities for plots
    runner.cities = {
        "New York": [-74.0060, 40.7128],
        "Buffalo": [-78.8789, 42.8869],
        "Rochester": [-77.6088, 43.1566],
        "Albany": [-73.7545, 42.6518],
        "Kiryas Joel": [-74.1679, 41.3420],
        "Syracuse": [-76.1474, 43.0495] 
    }

    runner.build_mesh()
    summary = runner.run_FEM()
    print(summary)