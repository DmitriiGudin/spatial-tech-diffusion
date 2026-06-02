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

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import time
import logging
import json

from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve, factorized
from scipy.optimize import minimize

import meshio
from pyproj import Transformer
from functools import lru_cache

from skfem import Basis, MeshTri, ElementTriP1, asm, BilinearForm
from skfem.models.poisson import laplace
logging.getLogger("skfem").setLevel(logging.ERROR)

from density_utils import get_batch_nodal_density, get_batch_nodal_cost, _get_mesh_boundary_polygon, _boundary_poly_to_mpl_path
from model_utils import ModelConfig, GSBFunctions
from model_configs import SSB_CURRENT
from mesh_utils import (
    load_mesh_km_from_msh,
    make_region_tag,
    _plot_cities_lonlat,
    _plot_node_circles_lonlat_single,
    _signed_log1p_scale,
    _apply_colorbar_ticklabels,
)


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
class FEMConfig:
    tau_years: float = 0.05
    T_years: float = 5.0
    picard_max_iter: int = 15
    picard_tol: float = 1e-8
    verbose: bool = True

    YEAR0: float = 1998.0
    epsg_project: int = 5070


# =============================================================================
# Utilities: mesh IO + coordinate transforms
# =============================================================================

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
    cost_nodes: Optional[np.ndarray] = None   # (nt,N) or None
    trend_factor: Optional[np.ndarray] = None # (nt,)   or None

    def check_compatible(self, cfg: FEMConfig) -> None:
        if float(cfg.tau_years) != float(self.cfg.tau_years):
            raise ValueError("FEMStageCache incompatible: tau_years differs.")
        if float(cfg.T_years) != float(self.cfg.T_years):
            raise ValueError("FEMStageCache incompatible: T_years differs.")
        if float(cfg.YEAR0) != float(self.cfg.YEAR0):
            raise ValueError("FEMStageCache incompatible: YEAR0 differs.")
        if int(cfg.epsg_project) != int(self.cfg.epsg_project):
            raise ValueError("FEMStageCache incompatible: epsg_project differs.")


def build_fem_stage_cache(msh_path: Path, cfg: FEMConfig, log: Optional[Callable[[str], None]] = None, *, cost_nodes: Optional[np.ndarray] = None,) -> FEMStageCache:
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

    out = get_batch_nodal_density(mesh, years, epsg_project=cfg.epsg_project, return_masses=True, use_cache=True)
    rho_total = np.asarray(out["rho_nodes"], dtype=float)  # (nt,N)
    A_nodes = np.asarray(out["A_nodes"], dtype=float)      # (N,)
    if cost_nodes is not None:
        cost_nodes = np.asarray(cost_nodes, float)
        if cost_nodes.shape != rho_total.shape:
            raise ValueError(f"cost_nodes shape {cost_nodes.shape} must match rho_total shape {rho_total.shape}.")

    return FEMStageCache(msh_path=msh_path, cfg=cfg, mesh=mesh, basis=basis, N=N, times=times, M=M, K=K, lon_nodes=lon_nodes, 
        lat_nodes=lat_nodes, xy_nodes_km=xy_nodes_km, rho_total=rho_total, A_nodes=A_nodes, cost_nodes=cost_nodes)


# =============================================================================
# Solver core
# =============================================================================

def _solve_u_v_backward_euler_closed_form_from_FI(u_prev: np.ndarray, v_prev: np.ndarray, p: float, qI: float, FI_curr: np.ndarray, tau: float) -> Tuple[np.ndarray, np.ndarray]:
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


def solve_gsb_fem(msh_path: Path, funcs: GSBFunctions, params: Dict[str, float], cfg: FEMConfig, cache: Optional[FEMStageCache] = None, log: Optional[Callable[[str], None]] = None,
model_cfg: ModelConfig = SSB_CURRENT) -> "GSBSolution":
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
        cost_nodes_pre = None
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
        cost_nodes_pre = getattr(cache, "cost_nodes", None)  # (nt,N) or None


    # --- params ---
    r0 = float(params.get("r0", params.get("r", 1)))
    r1 = float(params.get("r1", 0))
    r2 = float(params.get("r2", 0))
    
    core = model_cfg.get_core_params(params)
    p = float(core["p"])
    qI = float(core["q_I"])
    
    # funcs must be complete; if not, fail loudly (so you don't silently fit the wrong model)
    if funcs.mu_prime is None or funcs.F_J_prime is None:
        raise ValueError(f"Model functions incomplete: mu_prime and F_J_prime must be provided by model_cfg={model_cfg.name}.")

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
    R_nodes_hist = [np.full(N, r0, dtype=float)]
    C_nodes_hist = [np.full(N, np.nan, dtype=float)]

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
            out = get_batch_nodal_density(mesh, [cal_year], epsg_project=cfg.epsg_project, return_masses=True, use_cache=True)
            rho_total = np.asarray(out["rho_nodes"][0], dtype=float)

        # nodal cost at this snapshot (if available)
        if cost_nodes_pre is not None:
            c_nodes = np.asarray(cost_nodes_pre[m], float)
        else:
            c_nodes = None
            
        # Google Trends factor at this snapshot (if available)
        trend_factor_pre = None
        if cache is not None:
            trend_factor_pre = getattr(cache, "trend_factor", None)
        
        # trend factor at this step (scalar)
        tf = 0
        if trend_factor_pre is not None:
            tf = float(trend_factor_pre[m])
        
        # r(x,t) = max(0, r0 - r1*c(x,t) + r2*tf)
        if c_nodes is None:
            r_nodes = np.full_like(rho_total, r0 + r2 * tf, dtype=float)
        else:
            r_nodes = np.maximum(0.0, r0 - r1 * c_nodes + r2 * tf)
            
        R_nodes_hist.append(r_nodes.copy())
        if c_nodes is None:
            C_nodes_hist.append(np.full(N, np.nan, dtype=float))
        else:
            C_nodes_hist.append(np.asarray(c_nodes, float).copy())
        
        rho_adopt = r_nodes * rho_total

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
            u_new, v_new = _solve_u_v_backward_euler_closed_form_from_FI(u_prev=u, v_prev=v, p=p, qI=qI, FI_curr=FI_new, tau=tau)

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

    return GSBSolution(mesh=mesh, basis=basis, times=np.asarray(times, float), YEAR0=float(cfg.YEAR0), U=np.array(U_hist), V=np.array(V_hist), I=np.array(I_hist), J=np.array(J_hist), 
        rho_total=np.array(RHO_total_hist), rho_adopt=np.array(RHO_adopt_hist), A_nodes=np.asarray(A_nodes, float), r_nodes=np.array(R_nodes_hist), cost_nodes=np.array(C_nodes_hist))


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
    r_nodes: Optional[np.ndarray] = None      # (nt,N) or None
    cost_nodes: Optional[np.ndarray] = None   # (nt,N) or None

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

def intensity_lambda_nodes(sol: GSBSolution, funcs: GSBFunctions, params: Dict[str, float], k: int, lambda_floor: float = 1e-30) -> np.ndarray:
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
# Google Trends helpers
# =============================================================================

def load_google_trends_monthly(csv_path: Path) -> pd.Series:
    """
    Reads Google Trends CSV with columns: 'Time' and 'Solar energy'.
    Returns a Series indexed by month-start Timestamp (freq='MS') with values in [0,1].
    """
    df = pd.read_csv(csv_path)
    if "Time" not in df.columns or "Solar energy" not in df.columns:
        raise ValueError("Google Trends CSV must have columns: 'Time' and 'Solar energy'.")

    ts = pd.to_datetime(df["Time"], errors="coerce")
    vals = pd.to_numeric(df["Solar energy"], errors="coerce") / 100.0

    ok = ts.notna() & vals.notna()
    ts = ts[ok]
    vals = vals[ok].clip(lower=0.0, upper=1.0)

    # Ensure month-start alignment
    ts = ts.dt.to_period("M").dt.to_timestamp(how="start")

    s = pd.Series(vals.to_numpy(dtype=float), index=ts.to_numpy())
    s = s.sort_index()
    # If duplicate months exist, keep the last (rare but safe)
    s = s[~s.index.duplicated(keep="last")]
    return s


def _frac_year_to_timestamp(year_frac: float) -> pd.Timestamp:
    """
    Convert a float year (e.g. 2012.25) to an approximate Timestamp using day-of-year.
    Assumes linear mapping within the year.
    """
    y = int(np.floor(year_frac))
    start = pd.Timestamp(year=y, month=1, day=1)
    end = pd.Timestamp(year=y + 1, month=1, day=1)
    days = (end - start).days
    frac = float(year_frac - y)
    # clamp for numerical safety
    frac = min(max(frac, 0.0), 1.0 - 1e-12)
    d = int(np.floor(frac * days))
    return start + pd.Timedelta(days=d)


def trend_factor_at_snapshots(times: np.ndarray, YEAR0: float, trends: pd.Series) -> np.ndarray:
    """
    Map solver snapshot times (in years since YEAR0) to monthly trend_factor values.

    Strategy:
      - convert snapshot -> Timestamp
      - map to month-start Timestamp (MS)
      - lookup trends; if missing, forward-fill/back-fill using reindex+ffill+bfill.
    """
    if trends is None or len(trends) == 0:
        return np.zeros_like(times, dtype=float)

    # Build a filled monthly series over its own range (ffill/bfill)
    monthly_index = pd.date_range(trends.index.min(), trends.index.max(), freq="MS")
    trends_filled = trends.reindex(monthly_index).ffill().bfill()

    out = np.zeros(times.shape[0], dtype=float)
    for k, t in enumerate(times):
        cal_year = float(YEAR0) + float(t)
        ts = _frac_year_to_timestamp(cal_year)
        ms = ts.to_period("M").to_timestamp(how="start")

        if ms < trends_filled.index.min():
            out[k] = float(trends_filled.iloc[0])
        elif ms > trends_filled.index.max():
            out[k] = float(trends_filled.iloc[-1])
        else:
            out[k] = float(trends_filled.loc[ms])

    # final clamp
    out = np.clip(out, 0.0, 1.0)
    return out



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
# Data binning to nodes
# =============================================================================

@lru_cache(maxsize=16)
def _get_fwd_transformer(epsg_project: int) -> Transformer:
    return Transformer.from_crs("EPSG:4326", f"EPSG:{int(epsg_project)}", always_xy=True)

def _project_lonlat_to_km(lon, lat, *, epsg_project: int = 5070):
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    fwd = _get_fwd_transformer(int(epsg_project))
    x_m, y_m = fwd.transform(lon, lat)
    return np.asarray(x_m, float) / 1000.0, np.asarray(y_m, float) / 1000.0


def bin_events_year_node(mesh, events_df: pd.DataFrame, epsg_project: int = 5070, t_min_year: int = 1998, t_max_year: int = 2025, chunk_size: int = 50_000,
) -> Tuple[np.ndarray, int, int, np.ndarray, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Stream-bins events into node counts[nyears, N] by:
      - find containing triangle
      - add 1/3 count to each of its 3 vertices (mass-lumped event assignment)
    """
    if events_df is None or len(events_df) == 0:
        ny = int(t_max_year) - int(t_min_year) + 1
        year_labels = np.arange(int(t_min_year), int(t_max_year) + 1, dtype=np.int32)
        return np.zeros((ny, mesh.p.shape[1]), dtype=float), 0, 0, year_labels, None, None

    # --- Parse only what we need (avoid full df.copy()) ---
    # Keep behavior: coercions + dropna on date/lon/lat
    date = pd.to_datetime(events_df["date"], errors="coerce")
    lon = pd.to_numeric(events_df["longitude"], errors="coerce")
    lat = pd.to_numeric(events_df["latitude"], errors="coerce")

    ok0 = date.notna() & lon.notna() & lat.notna()
    if not bool(ok0.any()):
        ny = int(t_max_year) - int(t_min_year) + 1
        year_labels = np.arange(int(t_min_year), int(t_max_year) + 1, dtype=np.int32)
        return np.zeros((ny, mesh.p.shape[1]), dtype=float), 0, 0, year_labels, None, None

    date = date[ok0]
    lon = lon[ok0].to_numpy(np.float64, copy=False)
    lat = lat[ok0].to_numpy(np.float64, copy=False)

    years = date.dt.year.to_numpy(np.int32, copy=False)
    mask_y = (years >= int(t_min_year)) & (years <= int(t_max_year))
    if not np.any(mask_y):
        ny = int(t_max_year) - int(t_min_year) + 1
        year_labels = np.arange(int(t_min_year), int(t_max_year) + 1, dtype=np.int32)
        return np.zeros((ny, mesh.p.shape[1]), dtype=float), 0, 0, year_labels, None, None

    lon = lon[mask_y]
    lat = lat[mask_y]
    years = years[mask_y]
    dates = date[mask_y].to_numpy(dtype="datetime64[ns]")

    K_total_window = int(lon.size)

    N = int(mesh.p.shape[1])
    ny = int(int(t_max_year) - int(t_min_year) + 1)
    year_labels = np.arange(int(t_min_year), int(t_max_year) + 1, dtype=np.int32)

    counts = np.zeros((ny, N), dtype=np.float64)
    if K_total_window == 0:
        return counts, 0, 0, year_labels, None, None

    year_id = (years.astype(np.int64) - int(t_min_year)).astype(np.int64)

    x_km, y_km = _project_lonlat_to_km(lon, lat, epsg_project=int(epsg_project))

    px = mesh.p[0, :]
    py = mesh.p[1, :]
    xmin, xmax = float(px.min()), float(px.max())
    ymin, ymax = float(py.min()), float(py.max())

    boundary_poly = _get_mesh_boundary_polygon(mesh)
    boundary_path = _boundary_poly_to_mpl_path(boundary_poly)

    finder = mesh.element_finder()
    tri = mesh.t  # (3, ntri)

    K_inside = 0
    min_dt64 = None
    max_dt64 = None

    CH = int(chunk_size) if (chunk_size is not None and int(chunk_size) > 0) else x_km.size

    for s in range(0, x_km.size, CH):
        e = min(x_km.size, s + CH)
        xj = x_km[s:e]
        yj = y_km[s:e]

        # bbox prefilter
        cand = (xj >= xmin) & (xj <= xmax) & (yj >= ymin) & (yj <= ymax)
        if not np.any(cand):
            continue

        # boundary polygon prefilter (eliminates “inside bbox but outside mesh” cases)
        # assumes boundary_poly is in the SAME km coords as mesh.p
        if boundary_path is not None:
            cand_idx = np.flatnonzero(cand)
            pts = np.column_stack((xj[cand_idx], yj[cand_idx]))
            inside_poly = boundary_path.contains_points(pts)
            if not np.any(inside_poly):
                continue
            sel = cand_idx[inside_poly]
        else:
            sel = np.flatnonzero(cand)

        try:
            tri_ids = finder(xj[sel], yj[sel]).astype(np.int64)
        except ValueError:
            tri_ids = -np.ones(sel.shape[0], dtype=np.int64)
            for jj, loc in enumerate(sel):
                try:
                    tri_ids[jj] = int(finder(np.array([xj[loc]]), np.array([yj[loc]]))[0])
                except ValueError:
                    tri_ids[jj] = -1
        
        inside = tri_ids >= 0
        if not np.any(inside):
            continue

        tri_in = tri_ids[inside]
        idx_in = (s + sel[inside]).astype(np.int64)

        yid_in = year_id[idx_in]
        ok = (yid_in >= 0) & (yid_in < ny)
        if not np.any(ok):
            continue

        tri_in = tri_in[ok]
        yid_in = yid_in[ok]
        dt_ok = dates[idx_in][ok]

        m = int(yid_in.size)
        if m == 0:
            continue

        # --- vertex ids ---
        v0 = tri[0, tri_in].astype(np.int64)
        v1 = tri[1, tri_in].astype(np.int64)
        v2 = tri[2, tri_in].astype(np.int64)

        # --- bincount accumulation into flattened (ny*N,) ---
        # linear indices for each vertex sample
        base = yid_in * N
        lin = np.empty(3 * m, dtype=np.int64)
        lin[0:m]       = base + v0
        lin[m:2*m]     = base + v1
        lin[2*m:3*m]   = base + v2

        w = np.full(3 * m, 1.0 / 3.0, dtype=np.float64)

        acc = np.bincount(lin, weights=w, minlength=ny * N).astype(np.float64, copy=False)
        counts += acc.reshape(ny, N)

        K_inside += m

        dmin = dt_ok.min()
        dmax = dt_ok.max()
        min_dt64 = dmin if (min_dt64 is None or dmin < min_dt64) else min_dt64
        max_dt64 = dmax if (max_dt64 is None or dmax > max_dt64) else max_dt64

    min_ts = pd.to_datetime(min_dt64) if min_dt64 is not None else None
    max_ts = pd.to_datetime(max_dt64) if max_dt64 is not None else None
    return counts, K_total_window, K_inside, year_labels, min_ts, max_ts


def bin_events_month_total_inside_mesh(mesh: MeshTri, events_df: pd.DataFrame, epsg_project: int,
    start_month: str, end_month: str, chunk_size: int = 5000) -> Tuple[np.ndarray, List[pd.Timestamp], int, int]:
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

def expected_counts_year_node(sol: GSBSolution, funcs: GSBFunctions, params: Dict[str, float], year_edges_years: np.ndarray, lambda_floor: float = 1e-30) -> np.ndarray:
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


def expected_counts_month_total(sol: GSBSolution, funcs: GSBFunctions, params: Dict[str, float], month_edges_years: np.ndarray, lambda_floor: float = 1e-30) -> np.ndarray:
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
# Bass model: total counts
# =============================================================================

def bass_F(t: np.ndarray, p: float, q: float) -> np.ndarray:
    t = np.asarray(t, float)
    p = float(max(p, 1e-12))
    q = float(max(q, 0.0))
    a = p + q
    e = np.exp(-a * t)
    return (1.0 - e) / (1.0 + (q / p) * e)


def fit_bass_to_monthly_counts(y_month: np.ndarray, month_edges_years_rel: np.ndarray, p0: float = 0.01, q0: float = 0.1, M0: Optional[float] = None) -> Tuple[float, float, float, np.ndarray]:
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

    res = minimize(obj, x0=np.array([p0, q0, M0], dtype=float), method="L-BFGS-B", bounds=[(1e-10, 10.0), (0.0, 200.0), (1e-6, 1e15)], options=dict(maxiter=800),)

    p_hat, q_hat, M_hat = float(res.x[0]), float(res.x[1]), float(res.x[2])
    yhat = predict(p_hat, q_hat, M_hat)
    return p_hat, q_hat, M_hat, yhat


# =============================================================================
# Plot helpers
# =============================================================================


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
        patches = [Ellipse((float(x), float(y)), width=float(2.0 * wx), height=float(2.0 * dy_deg)) for x, y, wx in zip(lon, lat, dx_deg)]
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

    _plot_node_circles_lonlat_single(msh_path=msh_path, epsg_project=epsg_project, out_png=out_png, values=R_plot,
        title=f"Pearson residuals (signed log1p), {year}", h_km=h_km, cities=cities, vlim=float(vlim_log), cbar_label="sign(R) · log(1 + |R|)")
    
    
def sample_monthly_counts_nb(mu_month: np.ndarray, phi: float, rng: np.random.Generator) -> np.ndarray:
    """
    Sample Y_m ~ NB(mean=mu_m, dispersion=phi) independently across months.

    Parameterization:
      E[Y] = mu
      Var[Y] = mu + mu^2 / phi
      phi -> inf gives Poisson(mu)

    Uses NumPy's negative_binomial(n, p) with:
      n = phi
      p = phi / (phi + mu)
    """
    mu = np.asarray(mu_month, dtype=float)
    mu = np.maximum(mu, 0.0)

    if not np.isfinite(phi) or phi <= 0.0:
        # Poisson limit or invalid -> treat as Poisson
        return rng.poisson(mu).astype(int)

    # Convert mean->(n,p). Ensure p in (0,1].
    n = float(phi)
    p = n / (n + mu)  # vector
    p = np.clip(p, 1e-15, 1.0)

    # NumPy requires integer n? It actually accepts floats in many builds,
    # but to be safe/portable: if phi is not integer, use Gamma-Poisson mixture:
    if abs(n - round(n)) > 1e-9:
        # Lambda ~ Gamma(shape=phi, scale=mu/phi); Y ~ Poisson(Lambda)
        lam = rng.gamma(shape=n, scale=(mu / n))
        return rng.poisson(lam).astype(int)

    # Integer-shape NB path
    return rng.negative_binomial(n=int(round(n)), p=p).astype(int)


def _plot_uvIJ_and_w_year_lonlat(msh_path: Path, epsg_project: int, out_png_uvij: Path, out_png_w: Path, u: np.ndarray,
    v: np.ndarray, I: np.ndarray, J: np.ndarray, year: int, h_km: float, cities: Optional[Dict[str, Sequence[float]]] = None):
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

    patches = [Ellipse((float(x), float(y)), width=float(2.0 * wx), height=float(2.0 * dy_deg)) for x, y, wx in zip(lon, lat, dx_deg)]

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


def _plot_total_counts_monthly_bass_vs_gsb_vs_data(sol: GSBSolution, funcs: GSBFunctions, params: Dict[str, float], events_df: pd.DataFrame, epsg_project: int,
    out_png: Path, start_month: str, end_month: str, chunk_size: int = 2000, lambda_floor: float = 1e-30, n_stochastic: int = 5, stochastic_seed: int = 0):
    # --- data + mean prediction ---
    y_month, month_labels, K_total, K_inside = bin_events_month_total_inside_mesh(
        mesh=sol.mesh, events_df=events_df, epsg_project=epsg_project,
        start_month=start_month, end_month=end_month, chunk_size=chunk_size
    )

    month_edges_years = make_month_edges_years(sol.YEAR0, start_month=start_month, end_month=end_month)
    mu_month = expected_counts_month_total(sol=sol, funcs=funcs, params=params, month_edges_years=month_edges_years, lambda_floor=lambda_floor)

    # --- Bass fit ---
    p_hat, q_hat, M_hat, bass_month = fit_bass_to_monthly_counts(y_month=y_month, month_edges_years_rel=month_edges_years, p0=float(params.get("p", 0.01)), q0=float(params.get("q_I", 0.1)), M0=None)

    x = np.array(month_labels)

    # --- main plot: Data vs Bass vs mean GSB ---
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(x, y_month.astype(float), label="Data (inside mesh)")
    ax.plot(x, bass_month.astype(float), label=f"Bass fit (p={p_hat:.3g}, q={q_hat:.3g})")
    ax.plot(x, mu_month.astype(float), label="GSB mean (integrated λ)")

    ax.set_title(f"Total monthly counts: Data vs Bass vs GSB ({start_month} to {end_month})")
    ax.set_xlabel("Month")
    ax.set_ylabel("Count per month")
    ax.grid(True, alpha=0.25)
    ax.legend()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    # --- stochastic plots: Data vs sampled NB(GSB) ---
    phi = float(params.get("phi", float("inf")))
    rng = np.random.default_rng(int(stochastic_seed))

    # base name: ny_monthly_totals_bass_vs_gsb_vs_data.png -> ny_monthly_totals_stochastic_gsb_vs_data_1.png
    base = out_png.name.replace("_monthly_totals_bass_vs_gsb_vs_data", "_monthly_totals_stochastic_gsb_vs_data")
    if base == out_png.name:
        # fallback if filename doesn't match expected pattern
        stem = out_png.stem
        base = f"{stem}_stochastic_gsb_vs_data"

    for j in range(1, int(n_stochastic) + 1):
        y_sim = sample_monthly_counts_nb(mu_month=mu_month, phi=phi, rng=rng)

        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(x, y_month.astype(float), label="Data (inside mesh)")
        ax.plot(x, y_sim.astype(float), label=f"GSB sampled (NB, phi={phi:.3g})")

        ax.set_title(f"Total monthly counts (stochastic): Data vs sampled GSB ({start_month} to {end_month})")
        ax.set_xlabel("Month")
        ax.set_ylabel("Count per month")
        ax.grid(True, alpha=0.25)
        ax.legend()

        out_j = out_png.parent / base.replace(".png", f"_{j}.png")
        fig.tight_layout()
        fig.savefig(out_j, dpi=200)
        plt.close(fig)
        
        data_dir = out_png.parent.parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        comparison_csv = data_dir / out_png.with_suffix(".csv").name
        pd.DataFrame({
            "month": [pd.Timestamp(m).strftime("%Y-%m") for m in month_labels],
            "data": np.asarray(y_month, float),
            "standard_bass": np.asarray(bass_month, float),
            "GSB": np.asarray(mu_month, float),
        }).to_csv(comparison_csv, index=False)
        

# =============================================================================
# Other diagnostic helpers
# =============================================================================

def forecast_error_metrics(y: np.ndarray, mu: np.ndarray, eps: float = 1e-12, min_denom: float = 1.0) -> Dict[str, float]:
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)

    mask = np.isfinite(y) & np.isfinite(mu)
    y = y[mask]
    mu = mu[mask]

    ae = np.abs(y - mu)
    se = (y - mu) ** 2

    # Avoid zero-count nodes dominating MAPE.
    mape_mask = np.abs(y) >= float(min_denom)

    out = {
        "mae": float(np.mean(ae)) if y.size else np.nan,
        "rmse": float(np.sqrt(np.mean(se))) if y.size else np.nan,
        "log1p_mae": float(np.mean(np.abs(np.log1p(y) - np.log1p(np.maximum(mu, 0.0))))) if y.size else np.nan,
        "log1p_rmse": float(np.sqrt(np.mean((np.log1p(y) - np.log1p(np.maximum(mu, 0.0))) ** 2))) if y.size else np.nan,
        "smape": float(np.mean(2.0 * ae / np.maximum(np.abs(y) + np.abs(mu), eps))) if y.size else np.nan,
        "mape_nonzero": float(np.mean(ae[mape_mask] / np.maximum(np.abs(y[mape_mask]), eps))) if np.any(mape_mask) else np.nan,
        "total_observed": float(np.sum(y)),
        "total_expected": float(np.sum(mu)),
        "total_relative_error": float((np.sum(mu) - np.sum(y)) / max(float(np.sum(y)), eps)),
    }
    return out
        

# =============================================================================
# Runner: mesh build + FEM run + diagnostics
# =============================================================================

@dataclass
class Runner:
    out_folder: str
    mesh_params: Dict[str, Any]
    model_params: Dict[str, float]
    time_params: Dict[str, float]
    
    model_cfg: ModelConfig = SSB_CURRENT

    fem_verbose: bool = False
    mesh_verbose: bool = False

    # Optional diagnostics controls
    cities: Dict[str, Sequence[float]] = field(default_factory=dict)
    years_to_plot: Optional[List[int]] = None
    month_window: Tuple[str, str] = ("1998-01", "2023-12")
    events_csv: Path = Path("data") / "processed" / "solar_installations_all.csv"
    events_state_col: str = "state"
    google_trends_csv: Path = Path("data") / "raw" / "pv" / "GoogleTrends_US_20031231-1900_20260127-1054.csv"

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
    
        region_tag = make_region_tag(
            self.mesh_params.get("state_list", []),
            county_list=self.mesh_params.get("county_list", []),
            city_list=self.mesh_params.get("city_list", []),
            digest_len=10,
        )
    
        return self.mesh_dir / f"{region_tag}__h{h_km}_s{simplify_km}_km.msh"

    # ---- logging ----
    def log(self, msg: str) -> None:
        dt = time.perf_counter() - self._t0
        print(f"{self.out_folder}@[{_fmt_hhmmss(dt)}] ---- {msg}")

    # ---- configuration builders ----
    def build_fem_config(self) -> FEMConfig:
        tau = float(self.time_params.get("tau", 0.05))
        start_year = float(self.time_params["start_year"])
        T_requested = float(self.time_params.get("T_years", max(0.0, 2023.0 - start_year)))

        t_max_year = self.time_params.get("t_max_year", None)
        if t_max_year is None:
            T_needed = T_requested
        else:
            # Need to solve through Jan 1 after t_max_year.
            T_needed = float(int(t_max_year) + 1 - start_year)
        
        T_years = max(T_requested, T_needed)

        epsg = int(self.mesh_params.get("epsg_project", 5070))
        return FEMConfig(tau_years=tau, T_years=T_years, picard_max_iter=int(self.time_params.get("picard_max_iter", 20)),
            picard_tol=float(self.time_params.get("picard_tol", 1e-8)), verbose=bool(self.fem_verbose), YEAR0=float(start_year), epsg_project=epsg)

    def build_functions(self) -> GSBFunctions:
        return self.model_cfg.make_funcs(self.model_params)

    def build_params(self) -> Dict[str, float]:
        r0 = float(self.model_params.get("r0", self.model_params.get("r", 1.0)))
        r1 = float(self.model_params.get("r1", 0.0))
        r2 = float(self.model_params.get("r2", 0.0))
        phi = float(self.model_params.get("phi", float("inf")))
    
        core = self.model_cfg.get_core_params(self.model_params)
        p = float(core["p"])
        q_I = float(core["q_I"])
    
        return dict(r0=r0, r1=r1, r2=r2, p=p, q_I=q_I, phi=phi)

    # ---- part 1: mesh building ----
    def build_mesh(self) -> Path:
        """
        Builds mesh into: out/<out_folder>/mesh/<h>_<simplify>_km.msh
        """
        from mesh_utils import MeshBuildConfig, build_mesh_from_admin1_region

        self.mesh_dir.mkdir(parents=True, exist_ok=True)
        msh = self.mesh_path()
        
        meta_path = msh.with_suffix(".region.json")
        meta_path.write_text(
            json.dumps(
                {
                    "state_list": list(self.mesh_params.get("state_list", [])),
                    "county_list": list(self.mesh_params.get("county_list", [])),
                    "city_list": list(self.mesh_params.get("city_list", [])),
                    "mesh_file": msh.name,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        admin1_shp = Path("data") / "raw" / "maps" / "ne_10m_admin_1_states_provinces_lakes" / "ne_10m_admin_1_states_provinces_lakes.shp"
        county_shp = Path("data") / "raw" / "maps" / "cb_2023_us_county_5m" / "cb_2023_us_county_5m.shp"
        state_list = list(self.mesh_params["state_list"])
        county_list = list(self.mesh_params.get("county_list", []))
        city_list = list(self.mesh_params.get("city_list", []))
        h_km = float(self.mesh_params["h_km"])
        simplify_km = float(self.mesh_params["simplify_km"])
        epsg = int(self.mesh_params.get("epsg_project", 5070))

        cfg = MeshBuildConfig(h_km=h_km, simplify_km=simplify_km, epsg_project=epsg)

        if msh.exists():
            self.log(f"mesh exists: {msh.name} (skipping build)")
            self.log("self.build_mesh complete")
            return msh

        t0 = time.perf_counter()
        build_mesh_from_admin1_region(admin1_shp, state_list, msh, cfg, verbose=bool(self.mesh_verbose), model_name=f"{self.out_folder}_mesh_km",
            county_shp=county_shp, county_names=county_list, city_names=city_list)
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
        
        # --- Google Trends (monthly) -> snapshot-aligned trend_factor ---
        trends = load_google_trends_monthly(self.google_trends_csv)
        trend_factor_snap = trend_factor_at_snapshots(cache.times, fem_cfg.YEAR0, trends)
        cache.trend_factor = trend_factor_snap
        self.log(f"[trends] loaded {len(trends):,} months; snapshot trend_factor in "
                 f"[{trend_factor_snap.min():.3f}, {trend_factor_snap.max():.3f}]")

        # --- events load + filter ---
        events_df = pd.read_csv(self.events_csv)
        if "date" not in events_df.columns or "longitude" not in events_df.columns or "latitude" not in events_df.columns:
            raise ValueError("events CSV must contain: date, longitude, latitude.")
        if self.events_state_col not in events_df.columns:
            raise ValueError(f"events CSV must contain a '{self.events_state_col}' column.")
        
        state_list = [str(s).strip() for s in self.mesh_params["state_list"]]
        events_df[self.events_state_col] = events_df[self.events_state_col].astype(str).str.strip()
        events_df = events_df.loc[events_df[self.events_state_col].isin(state_list)].copy()
        st = "_".join(state_list)
        
        # --- cost nodes aligned to FEM snapshots (calendar-year proxy) ---
        cal_years = np.asarray(float(fem_cfg.YEAR0) + np.asarray(cache.times, float), float)
        year_keys = np.floor(cal_years + 1e-9).astype(int)
        
        uniq_years = np.unique(year_keys).astype(float).tolist()
        
        out_cost_year = get_batch_nodal_cost(cache.mesh, uniq_years, events_df=events_df, epsg_project=fem_cfg.epsg_project)
        
        # Map year->row
        row_of_year = {int(y): i for i, y in enumerate(np.asarray(out_cost_year["years"]).astype(int))}
        
        cost_nodes_year = np.asarray(out_cost_year["cost_nodes"], float)  # (nyears, N)
        
        # Broadcast to snapshots
        cost_nodes = np.empty((cal_years.size, cost_nodes_year.shape[1]), dtype=float)
        for k, yk in enumerate(year_keys):
            cost_nodes[k] = cost_nodes_year[row_of_year[int(yk)]]
        
        cache.cost_nodes = cost_nodes
        self.log(f"get_batch_nodal_cost complete ({time.perf_counter() - t0:.3f} s)")
        self.log(f"[cost] mean statewide median={float(np.mean(out_cost_year['statewide_median'])):.4g}, "
                 f"events used total={int(np.sum(out_cost_year['n_events_used'])):,}")
        
        # --- solve ---
        t0 = time.perf_counter()
        sol = solve_gsb_fem(msh, funcs, params, fem_cfg, cache=cache, log=lambda s: self.log(s), model_cfg=self.model_cfg)
        self.log(f"solve_gsb_fem complete ({time.perf_counter() - t0:.3f} s)")

        # --- year window ---
        start_year = float(self.time_params["start_year"])
        # Max year defaults to 2023 unless overridden
        t_min_year = int(self.time_params.get("t_min_year", int(np.floor(start_year))))
        t_max_year = int(self.time_params.get("t_max_year", 2023))

        year_edges_years = make_year_edges_years(YEAR0=fem_cfg.YEAR0, start_year=t_min_year, end_year=t_max_year)

        # --- bin yearly node counts ---
        t0 = time.perf_counter()
        counts_node, K_total, K_inside, year_labels, min_ts, max_ts = bin_events_year_node(mesh=sol.mesh, events_df=events_df, epsg_project=fem_cfg.epsg_project,
            t_min_year=t_min_year, t_max_year=t_max_year, chunk_size=int(self.time_params.get("bin_chunk_size", 5000)))
        self.log(f"bin_events_year_node complete ({time.perf_counter() - t0:.3f} s)")
        self.log(f"[bin] window events={K_total:,} inside mesh={K_inside:,} shape={counts_node.shape}")
        if min_ts is not None and max_ts is not None:
            self.log(f"[bin] inside-mesh date range: {min_ts.date()} -> {max_ts.date()}")

        # --- expected counts ---
        t0 = time.perf_counter()
        mu_node = expected_counts_year_node(sol=sol, funcs=funcs, params=params, year_edges_years=year_edges_years, lambda_floor=float(self.time_params.get("lambda_floor", 1e-30)))
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
        
        metrics = forecast_error_metrics(counts_node, mu_node, eps=float(self.time_params.get("metric_eps", 1e-12)), min_denom=float(self.time_params.get("mape_min_denom", 1.0)))
        
        self.log(
            "[metrics] "
            f"MAE={metrics['mae']:.6g} "
            f"RMSE={metrics['rmse']:.6g} "
            f"SMAPE={metrics['smape']:.6g} "
            f"MAPE_nonzero={metrics['mape_nonzero']:.6g} "
            f"log1p_MAE={metrics['log1p_mae']:.6g} "
            f"log1p_RMSE={metrics['log1p_rmse']:.6g} "
            f"total_rel_err={metrics['total_relative_error']:.6g}"
        )

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
            _plot_data_vs_mu_year_nodes_lonlat(msh_path=msh, epsg_project=fem_cfg.epsg_project, out_png=out_png,
                y_node=y_node, mu_node=mu_y, year=int(yy), h_km=h_km, cities=self.cities)
            self.log(f"_plot_data_vs_mu_year_nodes_lonlat complete ({time.perf_counter() - t0:.3f} s)")

            out_png_R = self.fig_dir / f"{st.lower()}_pearson_residual_nodes_{yy}.png"
            t0 = time.perf_counter()
            _plot_pearson_residuals_year_nodes_lonlat(msh_path=msh, epsg_project=fem_cfg.epsg_project, out_png=out_png_R,
                R_node=R[idx, :], year=int(yy), h_km=h_km, cities=self.cities, vlim_log=float(self.time_params.get("vlim_log", 2.5)))
            self.log(f"_plot_pearson_residuals_year_nodes_lonlat complete ({time.perf_counter() - t0:.3f} s)")

            # choose snapshot closest to Jan 1 of yy
            t_target = float(yy - fem_cfg.YEAR0)
            k_snap = int(np.argmin(np.abs(sol.times - t_target)))

            out_png_uvij = self.fig_dir / f"{st.lower()}_uvIJ_nodes_{yy}.png"
            out_png_w = self.fig_dir / f"{st.lower()}_w_nodes_{yy}.png"
            t0 = time.perf_counter()
            _plot_uvIJ_and_w_year_lonlat(msh_path=msh, epsg_project=fem_cfg.epsg_project, out_png_uvij=out_png_uvij, out_png_w=out_png_w,
                u=sol.U[k_snap], v=sol.V[k_snap], I=sol.I[k_snap], J=sol.J[k_snap], year=int(yy), h_km=h_km, cities=self.cities)
            self.log(f"_plot_uvIJ_and_w_year_lonlat complete ({time.perf_counter() - t0:.3f} s)")

        # --- monthly totals plot ---
        start_month = f"{int(t_min_year):04d}-01"
        end_month   = f"{int(t_max_year):04d}-12"
        monthly_png = self.fig_dir / f"{st.lower()}_monthly_totals_bass_vs_gsb_vs_data.png"
        t0 = time.perf_counter()
        _plot_total_counts_monthly_bass_vs_gsb_vs_data(sol=sol, funcs=funcs, params=params, events_df=events_df, epsg_project=fem_cfg.epsg_project, out_png=monthly_png,
            start_month=start_month, end_month=end_month, chunk_size=int(self.time_params.get("bin_chunk_size", 5000)), lambda_floor=float(self.time_params.get("lambda_floor", 1e-30)))
        self.log(f"_plot_total_counts_monthly_bass_vs_gsb_vs_data complete ({time.perf_counter() - t0:.3f} s)")

        self.log("self.run_FEM complete")
        return dict(out_folder=self.out_folder, state=st, mesh=str(msh), figures=str(self.fig_dir), K_total=K_total, K_inside=K_inside, deviance=D_total, pearson_rms=R_rms, **metrics)


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
        specs.append(dict(out_folder=r.out_folder, mesh_params=r.mesh_params, model_params=r.model_params, time_params=r.time_params, fem_verbose=r.fem_verbose, mesh_verbose=r.mesh_verbose, 
            cities=r.cities, years_to_plot=r.years_to_plot, month_window=r.month_window, events_csv=r.events_csv, events_state_col=r.events_state_col, base_out=r.base_out))

    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_runner_worker, spec, do_mesh, do_fem) for spec in specs]
        for fut in as_completed(futs):
            results.append(fut.result())
    return results


# =============================================================================
# Example for New York: create runner, build mesh, run FEM, produce diagnostics
# =============================================================================

if __name__ == "__main__":
    runner = Runner(
        out_folder="example_oh_run",
        mesh_params=dict(
            state_list=["MD"], #NY
            h_km=3,
            simplify_km=9,
            epsg_project=5070,
        ),
        model_params=dict(
            r0=0.023431325190960844,
            r1=0.04940116054941341,
            r2=0.028638503662553427,
            p=0.00011557169559153849,
            q_I=0.054698742335255374,
            gamma_J=1.9225252064154095,
            k_J=4.106195524757044e-05,
            D=4256.858961837641,
            S0=0,
            phi=1.4628175438247957
        ),
        time_params=dict(
            start_year=2004.000,
            tau=0.025,
            T_years=20.000, 
            picard_max_iter=20,
            picard_tol=1e-8,
            t_min_year=2002,
            t_max_year=2024,
            
            cpi_adjust=True,
            base_year=2025,
            base_month=12,
            cpi_cache_csv="data/cache/cpi_monthly_cpiaucsl.csv"
            ),
        
            fem_verbose=False,
            mesh_verbose=False,
    )

    # Optional: cities for plots
    '''runner.cities = {
        "New York": [-74.0060, 40.7128],
        "Buffalo": [-78.8789, 42.8869],
        "Rochester": [-77.6088, 43.1566],
        "Albany": [-73.7545, 42.6518],
        "Kiryas Joel": [-74.1679, 41.3420],
        "Syracuse": [-76.1474, 43.0495] 
    }'''

    runner.build_mesh()
    summary = runner.run_FEM()
    print(summary)