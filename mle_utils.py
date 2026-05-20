#!/usr/bin/env python3
"""
mle_utils.py

Score-based fitting for the GSB FEM model using a spatio-temporal Poisson point process.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import time
import logging

from skfem import Basis, MeshTri
logging.getLogger("skfem").setLevel(logging.ERROR)
from scipy.special import gammaln

from fem_utils import FEMConfig, solve_gsb_fem, build_fem_stage_cache, FEMStageCache
from mesh_utils import MeshBuildConfig, build_mesh_from_admin1_region
from density_utils import _project_lonlat_to_km, get_batch_nodal_cost
from model_utils import ModelConfig, GSBFunctions
from model_configs import SSB_CURRENT


# =============================================================================
# Helpers: robust element finder (skfem 11), time, projections
# =============================================================================

def safe_element_finder(mesh: MeshTri, chunk_size: int = 2000):
    """
    scikit-fem 11: mesh.element_finder()(x, y) may allocate huge intermediates
    when called on very large arrays. This wrapper ALWAYS chunks.

    Returns:
        tri_ids: int64 array, -1 for outside points.
    """
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


def date_to_decimal_year(ts: pd.Timestamp) -> float:
    y = ts.year
    start = pd.Timestamp(year=y, month=1, day=1)
    end = pd.Timestamp(year=y + 1, month=1, day=1)
    days = (end - start).days
    frac = (ts - start).days / float(days)
    return float(y) + float(frac)

# =============================================================================
# Progress helper (per-process safe; prints can interleave across processes)
# =============================================================================

@dataclass
class ProgressTracker:
    total: int
    freq: Optional[int] = None      # print every freq "iterations"; if None or <=0 -> no prints
    printer: Optional[Callable[[str], None]] = None  # called with already-formatted message content
    count: int = 0

    def tick(self, n: int = 1) -> None:
        self.count += int(n)

        if self.freq is None or self.freq <= 0 or self.printer is None:
            return

        if (self.count % self.freq) == 0 or self.count >= self.total:
            c = min(self.count, self.total)
            self.printer(f"{c} / {self.total} iterations complete")
            
            
# =============================================================================
# Google Trends helpers
# =============================================================================
            
def decimal_year_to_timestamp(y: float) -> pd.Timestamp:
    """
    Convert decimal year (e.g. 2012.25) to a Timestamp by mapping the fractional
    part to day-of-year. This matches your date_to_decimal_year convention.
    """
    y = float(y)
    year = int(np.floor(y))
    frac = y - year
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1)
    days = (end - start).days
    day_offset = int(np.floor(frac * days))
    return start + pd.Timedelta(days=day_offset)


def load_google_trends_monthly(csv_path: Path) -> pd.Series:
    """
    Returns monthly series indexed by month-start Timestamp, values in [0,1].
    CSV columns: Time (yyyy-mm-dd month starts), Solar energy (0..100).
    """
    df = pd.read_csv(csv_path)
    if "Time" not in df.columns or "Solar energy" not in df.columns:
        raise ValueError("Google Trends CSV must have columns: 'Time' and 'Solar energy'.")

    ts = pd.to_datetime(df["Time"], errors="coerce")
    vals = pd.to_numeric(df["Solar energy"], errors="coerce") / 100.0

    ok = ts.notna() & vals.notna()
    ts = ts[ok]
    vals = vals[ok].clip(lower=0.0, upper=1.0)

    # ensure month-start index
    ts = ts.dt.to_period("M").dt.to_timestamp(how="start")

    s = pd.Series(vals.to_numpy(dtype=float), index=ts.to_numpy())
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def trend_factor_at_snapshots(trends: pd.Series, cal_years: np.ndarray) -> np.ndarray:
    """
    trends: monthly Series indexed by month-start
    cal_years: calendar years (float) at snapshots, e.g. YEAR0 + times[k]
    Returns trend_factor[k] in [0,1], clamped outside trends range.
    """
    if trends.empty:
        raise ValueError("Google Trends series is empty.")

    out = np.empty(cal_years.shape[0], dtype=float)

    # (optional) ensure full monthly grid + fill, but clamp outside anyway
    monthly_index = pd.date_range(trends.index.min(), trends.index.max(), freq="MS")
    trends_filled = trends.reindex(monthly_index).ffill().bfill()

    lo = trends_filled.index.min()
    hi = trends_filled.index.max()

    for k, y in enumerate(cal_years):
        ts = decimal_year_to_timestamp(float(y))
        ms = ts.to_period("M").to_timestamp(how="start")

        if ms < lo:
            out[k] = float(trends_filled.iloc[0])
        elif ms > hi:
            out[k] = float(trends_filled.iloc[-1])
        else:
            out[k] = float(trends_filled.loc[ms])

    return out


# =============================================================================
# Data loading
# =============================================================================

@dataclass
class EventData:
    x_km: np.ndarray     # (K,)
    y_km: np.ndarray     # (K,)
    t_years: np.ndarray  # (K,) time since YEAR0
    cal_year: np.ndarray # (K,) calendar year float
    lon: np.ndarray      # (K,)
    lat: np.ndarray      # (K,)
    raw: pd.DataFrame


def load_events_csv(csv_path: Path, region_states: Optional[Iterable[str]] = None, YEAR0: float = 1998.0,
    epsg_project: int = 5070, min_year: Optional[float] = None, max_year: Optional[float] = None) -> EventData:
    df = pd.read_csv(csv_path)

    if "date" not in df.columns:
        raise ValueError("CSV must have a 'date' column in yyyy-mm-dd format.")
    if "longitude" not in df.columns or "latitude" not in df.columns:
        raise ValueError("CSV must have 'longitude' and 'latitude' columns.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "longitude", "latitude"]).copy()

    if region_states is not None:
        if "state" not in df.columns:
            raise ValueError("region_states provided but CSV has no 'state' column.")
        region_states = set([s.strip() for s in region_states])
        df = df[df["state"].astype(str).str.strip().isin(region_states)].copy()

    df["cal_year"] = df["date"].apply(date_to_decimal_year)

    if min_year is not None:
        df = df[df["cal_year"] >= float(min_year)].copy()
    if max_year is not None:
        df = df[df["cal_year"] <= float(max_year)].copy()

    lon = df["longitude"].to_numpy(float)
    lat = df["latitude"].to_numpy(float)
    x_km, y_km = _project_lonlat_to_km(lon, lat, epsg_project=epsg_project)

    cal_year = df["cal_year"].to_numpy(float)
    t_years = cal_year - float(YEAR0)

    return EventData(x_km=x_km, y_km=y_km, t_years=t_years, cal_year=cal_year, lon=lon, lat=lat, raw=df)


# =============================================================================
# Likelihood config + helpers
# =============================================================================

@dataclass
class ScoreConfig:
    t_min: float = 0.0
    t_max: float = 25.0

    lambda_floor: float = 1e-12
    verbose: bool = False

    normalize_by_events: bool = True

    finder_chunk_size: int = 2000


def intensity_lambda_from_fields(u: np.ndarray, v: np.ndarray, I: np.ndarray, rho_adopt: np.ndarray, p: float, q_I: float, F_I_vals: np.ndarray) -> np.ndarray:
    s = np.maximum(1.0 - u - v, 0.0)
    lam = rho_adopt * (float(p) + float(q_I) * F_I_vals) * s
    return lam


# =============================================================================
# Stage precompute: event filtering/binning + mass weights + FEM cache
# =============================================================================

@dataclass
class StagePrecompute:
    fem_cache: FEMStageCache

    mesh: MeshTri
    basis: Basis

    counts_node: np.ndarray     # (nt, N) float64  (fractional due to 1/3 split)
    counts_logfact: np.ndarray  # (nt, N) float64  = gammaln(counts_node + 1)

    K_total_window: int
    K_inside: int
    nt: int
    N: int


def precompute_stage_objects(msh_path: Path, fem_cfg: FEMConfig, events: EventData, score_cfg: ScoreConfig) -> StagePrecompute:
    """
    Builds:
      - FEMStageCache (mesh/basis/M/K + rho_total over time + A_nodes)
      - binned event counts over (time_index k, node i), using:
           event -> containing triangle -> add 1/3 to each of its 3 vertices

    Time binning: nearest snapshot (k = round(t/tau)).
    """
    fem_cache = build_fem_stage_cache(msh_path, fem_cfg)
    years = (float(fem_cfg.YEAR0) + fem_cache.times).tolist()
    cpi_cache_csv = Path(str(msh_path) + ".cpi_cache.csv")

    cost_out = get_batch_nodal_cost(fem_cache.mesh, years, events_df=events.raw, epsg_project=int(fem_cfg.epsg_project),
        cpi_adjust=True, base_year=2026, base_month=12, cpi_cache_csv=cpi_cache_csv, use_cache=True)
    if "cost_nodes" not in cost_out:
        raise KeyError("get_batch_nodal_cost did not return 'cost_nodes'. "f"Keys: {list(cost_out.keys())}")
        
    cost_nodes = np.asarray(cost_out["cost_nodes"], dtype=float)
    if cost_nodes.shape != fem_cache.rho_total.shape:
        raise ValueError(f"cost_nodes shape {cost_nodes.shape} must match rho_total shape {fem_cache.rho_total.shape}.")

    fem_cache.cost_nodes = cost_nodes

    # --- Google Trends: uniform trend_factor(t), broadcast to nodes ---
    trends_csv = Path("data/raw/pv/GoogleTrends_US_20031231-1900_20260127-1054.csv")
    trends = load_google_trends_monthly(trends_csv)

    cal_years_snap = float(fem_cfg.YEAR0) + fem_cache.times  # (nt,)
    trend_t = trend_factor_at_snapshots(trends, cal_years_snap)  # (nt,)
    
    fem_cache.trend_factor = np.asarray(trend_t, float)
    
    if fem_cache.trend_factor.shape != fem_cache.times.shape:
        raise ValueError("trend_factor must have shape (nt,) matching fem_cache.times.")

    mesh = fem_cache.mesh
    basis = fem_cache.basis
    N = fem_cache.N
    times = fem_cache.times
    nt = times.size

    # Time filter
    mask_t = (events.t_years >= score_cfg.t_min) & (events.t_years <= score_cfg.t_max)
    x_ev = events.x_km[mask_t]
    y_ev = events.y_km[mask_t]
    t_ev = events.t_years[mask_t]
    K_total_window = int(t_ev.size)

    counts_node = np.zeros((nt, N), dtype=float)

    if K_total_window == 0:
        counts_logfact = gammaln(counts_node + 1.0)
        return StagePrecompute(fem_cache=fem_cache, mesh=mesh, basis=basis, counts_node=counts_node, counts_logfact = counts_logfact, K_total_window=0, K_inside=0, nt=nt, N=N)

    # Robust triangle IDs (-1 outside): ALWAYS chunked
    safe_finder = safe_element_finder(mesh, chunk_size=int(score_cfg.finder_chunk_size))
    tri_ids = safe_finder(x_ev, y_ev)
    inside = tri_ids >= 0

    t_in = t_ev[inside]
    tri_in = tri_ids[inside].astype(np.int64)
    K_inside = int(t_in.size)

    if K_inside == 0:
        counts_logfact = gammaln(counts_node + 1.0)
        return StagePrecompute(fem_cache=fem_cache, mesh=mesh, basis=basis, counts_node=counts_node, counts_logfact = counts_logfact, K_total_window=K_total_window, K_inside=0, nt=nt, N=N)

    # Bin times to nearest snapshot index
    tau = float(fem_cfg.tau_years)
    k_idx = np.rint(t_in / tau).astype(np.int64)
    k_idx = np.clip(k_idx, 0, nt - 1)

    # Distribute each event to its triangle vertices
    tri = mesh.t  # (3, ntri)
    v0 = tri[0, tri_in]
    v1 = tri[1, tri_in]
    v2 = tri[2, tri_in]

    w = np.full(k_idx.shape[0], 1.0 / 3.0, dtype=float)

    np.add.at(counts_node, (k_idx, v0), w)
    np.add.at(counts_node, (k_idx, v1), w)
    np.add.at(counts_node, (k_idx, v2), w)
    
    counts_logfact = gammaln(counts_node + 1.0)

    return StagePrecompute(fem_cache=fem_cache, mesh=mesh, basis=basis, counts_node=counts_node, counts_logfact=counts_logfact, K_total_window=K_total_window, K_inside=K_inside, nt=nt, N=N)


# =============================================================================
# Objective function determination
# =============================================================================
def robust_metric_score(y: np.ndarray, mu: np.ndarray, objective_type: str, eps: float = 1e-12) -> float:
    """
    Returns a score to maximize.
    For error metrics, score = -error.
    """
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)

    if y.shape != mu.shape:
        raise ValueError(f"Shape mismatch: y={y.shape}, mu={mu.shape}")

    if not np.all(np.isfinite(mu)) or np.any(mu < 0.0):
        return float("-inf")

    obj = str(objective_type).lower().strip()

    ae = np.abs(y - mu)
    se = (y - mu) ** 2

    if obj == "mae":
        return -float(np.mean(ae))

    if obj == "rmse":
        return -float(np.sqrt(np.mean(se)))

    if obj == "smape":
        denom = np.maximum(np.abs(y) + np.abs(mu), float(eps))
        return -float(np.mean(2.0 * ae / denom))

    if obj == "log1p_mae":
        yy = np.log1p(np.maximum(y, 0.0))
        mm = np.log1p(np.maximum(mu, 0.0))
        return -float(np.mean(np.abs(yy - mm)))

    if obj == "log1p_rmse":
        yy = np.log1p(np.maximum(y, 0.0))
        mm = np.log1p(np.maximum(mu, 0.0))
        return -float(np.sqrt(np.mean((yy - mm) ** 2)))

    raise ValueError(f"Unknown robust objective_type={objective_type!r}")
    
    
def expected_counts_gsb_snapshot(msh_path: Path, funcs_template: GSBFunctions, theta: Dict[str, float], fem_cfg: FEMConfig,
    stage_pre: StagePrecompute, sync_boxes: Optional[Callable[[Dict[str, float]], None]] = None, *, model_cfg: ModelConfig = SSB_CURRENT, lambda_floor: float = 1e-30) -> np.ndarray:
    if sync_boxes is not None:
        sync_boxes(theta)

    sol = solve_gsb_fem(msh_path, funcs_template, theta, fem_cfg, cache=stage_pre.fem_cache, model_cfg=model_cfg)

    core = model_cfg.get_core_params(theta)
    p = float(core["p"])
    q_I = float(core["q_I"])

    dt = float(fem_cfg.tau_years)
    A_nodes = np.asarray(sol.A_nodes, float)

    nt = sol.times.size
    N = sol.mesh.p.shape[1]
    mu = np.zeros((nt, N), dtype=float)

    for k in range(nt):
        u = sol.U[k]
        v = sol.V[k]
        I = sol.I[k]
        rho_adopt_nodes = sol.rho_adopt[k]

        FI = funcs_template.F_I(I)
        lam_nodes = intensity_lambda_from_fields(u=u, v=v, I=I, rho_adopt=rho_adopt_nodes, p=p, q_I=q_I, F_I_vals=FI)

        if not np.all(np.isfinite(lam_nodes)) or np.any(lam_nodes < 0.0):
            return np.full_like(mu, np.nan)

        mu[k] = np.maximum(lam_nodes * A_nodes * dt, float(lambda_floor))

    return mu


def objective_gsb_theta(
    msh_path: Path,
    funcs_template: GSBFunctions,
    theta: Dict[str, float],
    fem_cfg: FEMConfig,
    stage_pre: StagePrecompute,
    score_cfg: ScoreConfig,
    sync_boxes: Optional[Callable[[Dict[str, float]], None]] = None,
    *,
    model_cfg: ModelConfig = SSB_CURRENT,
    objective_type: str = "loglikelihood",
) -> float:
    obj = str(objective_type).lower().strip()

    if obj in ("loglikelihood", "ll", "nb", "negbin"):
        return loglikelihood_theta(
            msh_path,
            funcs_template,
            theta,
            fem_cfg,
            stage_pre,
            score_cfg,
            sync_boxes=sync_boxes,
            model_cfg=model_cfg,
        )

    try:
        mu = expected_counts_gsb_snapshot(
            msh_path,
            funcs_template,
            theta,
            fem_cfg,
            stage_pre,
            sync_boxes=sync_boxes,
            model_cfg=model_cfg,
            lambda_floor=score_cfg.lambda_floor,
        )
        return robust_metric_score(stage_pre.counts_node, mu, objective_type=obj)
    except Exception:
        return float("-inf")
    
    
def objective_smith_song_theta(
    msh_path: Path,
    funcs_template: GSBFunctions,
    theta: Dict[str, float],
    fem_cfg: FEMConfig,
    stage_pre: StagePrecompute,
    score_cfg: ScoreConfig,
    sync_boxes: Optional[Callable[[Dict[str, float]], None]] = None,
    *,
    model_cfg: ModelConfig = SSB_CURRENT,
    objective_type: str = "loglikelihood",
    smith_song_history_mode: str = "conditional",
) -> float:
    obj = str(objective_type).lower().strip()

    if obj in ("loglikelihood", "ll", "nb", "negbin"):
        return loglikelihood_smith_song_theta(
            msh_path,
            funcs_template,
            theta,
            fem_cfg,
            stage_pre,
            score_cfg,
            sync_boxes=sync_boxes,
            model_cfg=model_cfg,
            smith_song_history_mode=smith_song_history_mode,
        )

    try:
        ss_pre = getattr(stage_pre, "_smith_song_pre", None)
        if ss_pre is None:
            ss_pre = build_smith_song_precompute(
                stage_pre,
                top_k=int(getattr(stage_pre, "ss_top_k", 50)),
            )
            setattr(stage_pre, "_smith_song_pre", ss_pre)

        mu = smith_song_expected_counts(
            stage_pre=stage_pre,
            theta=theta,
            ss_pre=ss_pre,
            t_min=score_cfg.t_min,
            t_max=score_cfg.t_max,
            eps=1e-300,
            top_k=int(getattr(stage_pre, "ss_top_k", 50)),
            kernel_tol=float(getattr(stage_pre, "ss_kernel_tol", 1e-6)),
            history_mode=smith_song_history_mode,
        )

        return robust_metric_score(stage_pre.counts_node, mu, objective_type=obj)

    except Exception:
        return float("-inf")
    

# =============================================================================
# Likelihood evaluation (uses stage precompute + FEM cache)
# =============================================================================

def loglikelihood_theta(msh_path: Path, funcs_template: GSBFunctions, theta: Dict[str, float], fem_cfg: FEMConfig,
    stage_pre: StagePrecompute, score_cfg: ScoreConfig, sync_boxes: Optional[Callable[[Dict[str, float]], None]] = None,
    *, model_cfg: ModelConfig = SSB_CURRENT) -> float:
    """
    Poisson log-likelihood for theta on a nodal discretization.

    Point term:
        sum_{k,i} counts_node[k,i] * log( lambda_i(t_k) )
    Integral term:
        ∫_{t_min}^{t_max} sum_i lambda_i(t) * A_i dt
    """
    if sync_boxes is not None:
        sync_boxes(theta)

    sol = solve_gsb_fem(msh_path, funcs_template, theta, fem_cfg, cache=stage_pre.fem_cache, model_cfg=model_cfg)
    
    core = model_cfg.get_core_params(theta)
    p = float(core["p"])
    q_I = float(core["q_I"])

    nt = sol.times.size
    if nt != stage_pre.nt:
        raise RuntimeError("StagePrecompute time grid mismatch vs FEM solve.")

    # time step width for cell counts
    dt = float(fem_cfg.tau_years)

    # counts and precomputed log-factorials
    counts_node = stage_pre.counts_node          # (nt, N)
    counts_logfact = stage_pre.counts_logfact    # (nt, N)
    A_nodes = np.asarray(sol.A_nodes, float)     # (N,)

    phi = float(theta.get("phi", np.inf))
    use_poisson = (not np.isfinite(phi))

    ll_raw = 0.0

    for k, t in enumerate(sol.times):
        if t < score_cfg.t_min or t > score_cfg.t_max:
            continue

        u = sol.U[k]
        v = sol.V[k]
        I = sol.I[k]
        rho_adopt_nodes = sol.rho_adopt[k]

        FI = funcs_template.F_I(I)
        lam_nodes = intensity_lambda_from_fields(u=u, v=v, I=I, rho_adopt=rho_adopt_nodes, p=p, q_I=q_I, F_I_vals=FI)

        if not np.all(np.isfinite(lam_nodes)):
            return float("-inf")
        if np.any(lam_nodes < 0.0):
            return float("-inf")

        # mean counts per (node,timecell)
        mu_nodes = lam_nodes * A_nodes * dt
        if not np.all(np.isfinite(mu_nodes)) or np.any(mu_nodes < 0.0):
            return float("-inf")

        c = counts_node[k]
        logfact = counts_logfact[k]

        if use_poisson:
            # Poisson cell-count loglik: sum_i [ c_i log(mu_i) - mu_i - log(c_i!) ]
            mu_safe = np.maximum(mu_nodes, score_cfg.lambda_floor)

            # c*log(mu) only matters where c>0
            mask = c > 0
            term = -float(np.sum(mu_nodes)) - float(np.sum(logfact))
            if np.any(mask):
                term += float(np.sum(c[mask] * np.log(mu_safe[mask])))

            ll_raw += term

        else:
            if not (phi > 0.0 and np.isfinite(phi)):
                return float("-inf")

            # NB (Poisson–Gamma) with mean mu and shape phi:
            # log P(c) = gammaln(c+phi) - gammaln(phi) - gammaln(c+1)
            #            + phi*log(phi/(phi+mu)) + c*log(mu/(phi+mu))
            mu_safe = np.maximum(mu_nodes, score_cfg.lambda_floor)
            denom = mu_safe + phi

            # terms over all nodes
            term = (float(np.sum(phi * (np.log(phi) - np.log(denom)))) - float(np.sum(logfact)) - float(A_nodes.size * gammaln(phi)))

            # only where c>0 do we need gammaln(c+phi) and c*log(mu/(phi+mu))
            mask = c > 0
            if np.any(mask):
                term += float(np.sum(gammaln(c[mask] + phi)))
                term += float(np.sum(c[mask] * (np.log(mu_safe[mask]) - np.log(denom[mask]))))
            else:
                # if c==0 everywhere, gammaln(c+phi)=gammaln(phi) cancels with -gammaln(phi),
                # so nothing to add
                pass

            ll_raw += term

    if not np.isfinite(ll_raw):
        return float("-inf")

    if score_cfg.normalize_by_events:
        denom = max(1, stage_pre.K_inside)
        ll = ll_raw / float(denom)
    else:
        ll = ll_raw

    if score_cfg.verbose:
        kind = "Poisson" if use_poisson else f"NegBin(phi={phi:.6g})"
        print(
            f"[ll:{kind}] Kwin={stage_pre.K_total_window} K_in={stage_pre.K_inside} "
            f"ll={ll_raw:.3e} ll_per_ev={ll:.6g}"
        )

    return float(ll)


# =============================================================================
# Smith-Song benchmark likelihood
# =============================================================================

@dataclass
class SmithSongPrecompute:
    """
    Cache for the Smith-Song benchmark.

    Sparse/local kernel:
      neigh_idx[s, j] = destination node index r among source s's local neighbors
      neigh_dist[s, j] = d_{s,r}

    W caches exp(-theta * neigh_dist), thresholded by ss_kernel_tol.
    """
    neigh_idx: np.ndarray       # (N,K)
    neigh_dist: np.ndarray      # (N,K)

    M_hist: np.ndarray          # (nt,N), population mass at nodes
    cost_hist: Optional[np.ndarray]
    trend: Optional[np.ndarray]
    A_nodes: np.ndarray

    W_theta: Optional[float] = None
    W: Optional[np.ndarray] = None

    def get_W(self, theta_dist: float, *, kernel_tol: float = 1e-6) -> np.ndarray:
        theta_dist = float(theta_dist)
        if self.W is None or self.W_theta is None or float(self.W_theta) != theta_dist:
            W = np.exp(-theta_dist * self.neigh_dist)
            W[W < float(kernel_tol)] = 0.0
            self.W = W
            self.W_theta = theta_dist
        return self.W


def build_smith_song_precompute(stage_pre: StagePrecompute, *, top_k: int = 50) -> SmithSongPrecompute:
    xy = np.asarray(stage_pre.fem_cache.xy_nodes_km, float)  # (N,2)
    N = xy.shape[0]
    K = min(int(top_k), N)

    dx = xy[:, None, 0] - xy[None, :, 0]
    dy = xy[:, None, 1] - xy[None, :, 1]
    dist_mat = np.sqrt(dx * dx + dy * dy)

    # For each source node s, keep K nearest destination nodes r.
    neigh_idx = np.argpartition(dist_mat, kth=K - 1, axis=1)[:, :K]

    # Sort those K neighbors by actual distance for reproducibility/debugging.
    row = np.arange(N)[:, None]
    neigh_dist_unsorted = dist_mat[row, neigh_idx]
    order = np.argsort(neigh_dist_unsorted, axis=1)
    neigh_idx = np.take_along_axis(neigh_idx, order, axis=1)
    neigh_dist = np.take_along_axis(neigh_dist_unsorted, order, axis=1)

    A_nodes = np.asarray(stage_pre.fem_cache.A_nodes, float)
    rho = np.asarray(stage_pre.fem_cache.rho_total, float)
    M_hist = np.maximum(rho * A_nodes[None, :], 0.0)

    cost_hist = getattr(stage_pre.fem_cache, "cost_nodes", None)
    if cost_hist is not None:
        cost_hist = np.asarray(cost_hist, float)

    trend = getattr(stage_pre.fem_cache, "trend_factor", None)
    if trend is not None:
        trend = np.asarray(trend, float)

    return SmithSongPrecompute(neigh_idx=np.asarray(neigh_idx, dtype=np.int64), neigh_dist=np.asarray(neigh_dist, float),
        M_hist=M_hist, cost_hist=cost_hist, trend=trend, A_nodes=A_nodes)


def _smith_song_time_weights(times: np.ndarray, a: float, t_min: float, t_max: float) -> np.ndarray:
    """
    Discrete approximation to normalized exponential time intensity over solver snapshots.

        w_k ∝ exp(a * t_k) * dt_k

    Returns weights summing to 1 over snapshots inside [t_min, t_max].
    """
    times = np.asarray(times, float)
    a = float(a)

    nt = times.size
    w = np.zeros(nt, dtype=float)
    if nt == 0:
        return w

    if nt == 1:
        if t_min <= times[0] <= t_max:
            w[0] = 1.0
        return w

    # Cell edges around snapshots
    edges = np.empty(nt + 1, dtype=float)
    edges[1:-1] = 0.5 * (times[:-1] + times[1:])
    edges[0] = times[0] - 0.5 * (times[1] - times[0])
    edges[-1] = times[-1] + 0.5 * (times[-1] - times[-2])

    # Clip cells to likelihood window
    left = np.maximum(edges[:-1], float(t_min))
    right = np.minimum(edges[1:], float(t_max))
    width = np.maximum(right - left, 0.0)

    mask = width > 0.0
    if not np.any(mask):
        return w

    # Integral of exp(a t) over each clipped cell.
    if abs(a) < 1e-12:
        w[mask] = width[mask]
    else:
        # numerically stable enough for typical year-scale a;
        # shift by t_min to avoid unnecessary exponent growth
        l = left[mask] - float(t_min)
        r = right[mask] - float(t_min)
        w[mask] = (np.exp(a * r) - np.exp(a * l)) / a

    total = float(np.sum(w))
    if total <= 0.0 or not np.isfinite(total):
        return np.zeros(nt, dtype=float)
    return w / total


def _smith_song_r_nodes_from_pre(ss_pre: SmithSongPrecompute, theta: Dict[str, float], k: int) -> np.ndarray:
    """
    Same r(x,t) convention as the PDE/FEM model:
        r = max(0, r0 - r1*cost + r2*trend)
    """
    r0 = float(theta.get("r0", theta.get("r", 1.0)))
    r1 = float(theta.get("r1", 0.0))
    r2 = float(theta.get("r2", 0.0))

    M_k = np.asarray(ss_pre.M_hist[k], float)

    tf = 0.0
    if ss_pre.trend is not None:
        tf = float(ss_pre.trend[k])

    if ss_pre.cost_hist is None:
        return np.full_like(M_k, r0 + r2 * tf, dtype=float)

    c_nodes = np.asarray(ss_pre.cost_hist[k], float)
    return np.maximum(0.0, r0 - r1 * c_nodes + r2 * tf)


def smith_song_expected_counts(stage_pre: StagePrecompute, theta: Dict[str, float], ss_pre: Optional[SmithSongPrecompute] = None, *,
    t_min: float = 0.0, t_max: float = 25.0, eps: float = 1e-300, top_k: int = 50, kernel_tol: float = 1e-6, history_mode: str = "conditional") -> np.ndarray:
    """
    Snapshot-based Smith-Song expected node-time counts using a sparse/top-k
    local exponential imitation kernel.
    """
    if ss_pre is None:
        ss_pre = build_smith_song_precompute(stage_pre, top_k=top_k)

    theta_dist = float(theta.get("theta", theta.get("theta_dist", 0.0)))
    lambda_mix = float(theta.get("lambda", theta.get("lambda_mix", 0.5)))
    a_time = float(theta.get("a", theta.get("a_time", 0.0)))

    if theta_dist < 0.0 or not np.isfinite(theta_dist):
        raise ValueError("Smith-Song theta/theta_dist must be finite and nonnegative.")
    if not (0.0 < lambda_mix < 1.0):
        raise ValueError("Smith-Song lambda/lambda_mix must lie in (0,1).")
    if not np.isfinite(a_time):
        raise ValueError("Smith-Song a/a_time must be finite.")
        
    history_mode = str(history_mode).lower().strip()
    if history_mode not in ("conditional", "generative"):
        raise ValueError("history_mode must be 'conditional' or 'generative'.")

    nt = int(stage_pre.nt)
    N = int(stage_pre.N)
    mu = np.zeros((nt, N), dtype=float)

    K_total = float(stage_pre.K_inside)
    if K_total <= 0.0:
        return mu

    times = np.asarray(stage_pre.fem_cache.times, float)
    w_time = _smith_song_time_weights(times, a=a_time, t_min=t_min, t_max=t_max)

    neigh_idx = ss_pre.neigh_idx      # (N,K), source -> destinations
    W = ss_pre.get_W(theta_dist, kernel_tol=kernel_tol)

    n_prev = np.zeros(N, dtype=float)

    for k in range(nt):
        counts_k = np.asarray(stage_pre.counts_node[k], float)

        if w_time[k] <= 0.0:
            if history_mode == "conditional":
                n_prev += counts_k
            # In generative mode, zero expected mass means no model-generated history update.
            continue

        M_k = np.asarray(ss_pre.M_hist[k], float)
        r_nodes = _smith_song_r_nodes_from_pre(ss_pre, theta, k)
        E = np.maximum(M_k * r_nodes, 0.0)

        Esum = float(np.sum(E))
        if not np.isfinite(Esum) or Esum <= 0.0:
            E = np.maximum(M_k, 0.0)
            Esum = float(np.sum(E))
            if Esum <= 0.0:
                P_innov = np.full(N, 1.0 / max(N, 1), dtype=float)
            else:
                P_innov = E / Esum
        else:
            P_innov = E / Esum

        N_prev = float(np.sum(n_prev))
        if N_prev <= 0.0:
            P_imit = P_innov
        else:
            # Sparse/local version of:
            #   row_sum_s = sum_r W_sr E_r
            #   P_imit_r = E_r * sum_s n_s W_sr / row_sum_s / N_prev

            E_neigh = E[neigh_idx]                    # (N,K)
            row_sum = np.sum(W * E_neigh, axis=1)     # (N,)

            alpha = n_prev / np.maximum(row_sum, eps) # (N,)

            # Accumulate beta_r = sum_s alpha_s W_sr over sparse edges.
            beta = np.zeros(N, dtype=float)
            np.add.at(beta, neigh_idx.ravel(), (alpha[:, None] * W).ravel())

            P_imit = E * beta / max(N_prev, eps)

            P_imit = np.maximum(P_imit, 0.0)
            ps = float(np.sum(P_imit))
            if ps <= 0.0 or not np.isfinite(ps):
                P_imit = P_innov
            else:
                P_imit = P_imit / ps

        P = lambda_mix * P_innov + (1.0 - lambda_mix) * P_imit
        P = np.maximum(P, 0.0)
        ps = float(np.sum(P))
        if ps <= 0.0 or not np.isfinite(ps):
            P = np.full(N, 1.0 / max(N, 1), dtype=float)
        else:
            P = P / ps

        mu[k] = K_total * w_time[k] * P

        # observed history drives imitation
        if history_mode == "conditional":
            # observed history drives imitation
            n_prev += counts_k
        else:
            # model-generated expected history drives imitation
            n_prev += mu[k]

    return np.maximum(mu, eps)


def negbin_loglik_counts(counts: np.ndarray, mu: np.ndarray, phi: float, *, logfact: Optional[np.ndarray] = None, mu_floor: float = 1e-12, ) -> float:
    """
    Negative-binomial / Poisson count log-likelihood for arrays of counts.

    NB parameterization:
        E[Y] = mu
        Var[Y] = mu + mu^2/phi

    phi infinite => Poisson.
    """
    c = np.asarray(counts, float)
    mu = np.asarray(mu, float)

    if c.shape != mu.shape:
        raise ValueError(f"counts shape {c.shape} != mu shape {mu.shape}")

    if logfact is None:
        logfact = gammaln(c + 1.0)
    else:
        logfact = np.asarray(logfact, float)

    if not np.all(np.isfinite(mu)) or np.any(mu < 0.0):
        return float("-inf")

    use_poisson = (not np.isfinite(phi))
    mu_safe = np.maximum(mu, float(mu_floor))

    if use_poisson:
        mask = c > 0
        ll = -float(np.sum(mu)) - float(np.sum(logfact))
        if np.any(mask):
            ll += float(np.sum(c[mask] * np.log(mu_safe[mask])))
        return float(ll)

    if not (phi > 0.0 and np.isfinite(phi)):
        return float("-inf")

    denom = mu_safe + float(phi)
    term = (float(np.sum(phi * (np.log(phi) - np.log(denom)))) - float(np.sum(logfact)) - float(c.size * gammaln(phi)))

    mask = c > 0
    if np.any(mask):
        term += float(np.sum(gammaln(c[mask] + phi)))
        term += float(np.sum(c[mask] * (np.log(mu_safe[mask]) - np.log(denom[mask]))))

    return float(term)


def loglikelihood_smith_song_theta(msh_path: Path, funcs_template: GSBFunctions, theta: Dict[str, float], fem_cfg: FEMConfig, stage_pre: StagePrecompute, score_cfg: ScoreConfig,
    sync_boxes: Optional[Callable[[Dict[str, float]], None]] = None, *, model_cfg: ModelConfig = SSB_CURRENT, smith_song_history_mode: str = "conditional") -> float:
    """
    Smith-Song benchmark likelihood with the same NB observation model as the PDE.

    Signature intentionally mirrors loglikelihood_theta(...) so it can be plugged into
    the existing SPSA/random-search machinery.
    """
    phi = float(theta.get("phi", np.inf))

    try:
        ss_pre = getattr(stage_pre, "_smith_song_pre", None)
        if ss_pre is None:
            ss_pre = build_smith_song_precompute(stage_pre, top_k=50)
            setattr(stage_pre, "_smith_song_pre", ss_pre)

        mu = smith_song_expected_counts(stage_pre=stage_pre, theta=theta, ss_pre=ss_pre, t_min=score_cfg.t_min, t_max=score_cfg.t_max,
            eps=1e-300, top_k=50, kernel_tol=1e-6, history_mode=smith_song_history_mode)
    except Exception:
        return float("-inf")

    ll_raw = negbin_loglik_counts(counts=stage_pre.counts_node, mu=mu, phi=phi, logfact=stage_pre.counts_logfact, mu_floor=score_cfg.lambda_floor)

    if not np.isfinite(ll_raw):
        return float("-inf")

    if score_cfg.normalize_by_events:
        ll = ll_raw / float(max(1, stage_pre.K_inside))
    else:
        ll = ll_raw

    if score_cfg.verbose:
        kind = "Poisson" if not np.isfinite(phi) else f"NegBin(phi={phi:.6g})"
        print(
            f"[ll:SmithSong:{kind}] Kwin={stage_pre.K_total_window} "
            f"K_in={stage_pre.K_inside} ll={ll_raw:.3e} ll_per_ev={ll:.6g}"
        )

    return float(ll)


# =============================================================================
# Optimization: SPSA in transformed space + coarse-to-fine schedule
# =============================================================================

def softplus(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float)
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)

def inv_softplus(x: np.ndarray) -> np.ndarray:
    x = np.maximum(np.asarray(x, float), 1e-12)
    return np.log(np.expm1(x))

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-12, 1.0 - 1e-12)
    return np.log(p / (1.0 - p))

def smooth_clip(z: float, zmax: float = 20.0) -> float:
    """Smooth saturation: maps R -> (-zmax, zmax) via tanh."""
    return float(zmax * np.tanh(float(z) / zmax))

@dataclass
class ParamSpec:
    name: str
    kind: str   # "const", "pos", "nonneg"
    lo: float
    hi: float
    scale: float = 1.0

    def __post_init__(self):
        self.lo = float(self.lo)
        self.hi = float(self.hi)
        self.scale = float(self.scale)
    
        if self.kind not in ("const", "pos", "nonneg"):
            raise ValueError(f"ParamSpec.kind must be one of const/pos/nonneg, got {self.kind}")
        if self.hi < self.lo:
            raise ValueError(f"{self.name}: hi < lo ({self.hi} < {self.lo})")
    
        if self.kind == "const":
            self.hi = self.lo
            return
    
        # for optimizable params, require finite bounds
        if not (np.isfinite(self.lo) and np.isfinite(self.hi)):
            raise ValueError(f"{self.name}: non-const bounds must be finite, got lo={self.lo}, hi={self.hi}")

def _free_specs(specs: List[ParamSpec]) -> List[ParamSpec]:
    return [s for s in specs if s.kind != "const"]

def _clip_theta(v: float, s: ParamSpec) -> float:
    return float(np.clip(float(v), s.lo, s.hi))

def pack_theta_to_z(theta: Dict[str, float], specs: List[ParamSpec]) -> np.ndarray:
    """
    Map bounded theta -> unconstrained z for SPSA.

    const: excluded from z
    pos:   log-bounded logistic: log(x) = log(lo) + (log(hi)-log(lo))*sigmoid(z)
    nonneg: z = inv_softplus(x) (then x is clipped into [lo, hi])
    """
    z: List[float] = []
    for s in _free_specs(specs):
        v = float(theta[s.name]) * 1.0  # theta already in final units
        v = _clip_theta(v, s)

        if s.kind == "pos":
            lo = max(float(s.lo), 1e-300)
            hi = max(float(s.hi), lo * 1.0000001)
            # work in log space
            a = np.log(lo)
            b = np.log(hi)
            # map v to y in (0,1)
            lv = np.log(max(v, lo))
            y = (lv - a) / (b - a)
            y = float(np.clip(y, 1e-12, 1.0 - 1e-12))
            z.append(float(logit(np.array([y]))[0]))

        elif s.kind == "nonneg":
            # allow lo==0; just invert softplus on clipped v
            v_eff = max(v, 0.0)
            z.append(float(inv_softplus(np.array([v_eff]))[0]))

        else:
            raise ValueError(f"Unknown spec kind: {s.kind}")

    return np.array(z, dtype=float)

def unpack_z_to_theta(z: np.ndarray, specs: List[ParamSpec]) -> Dict[str, float]:
    """
    Map unconstrained z -> bounded theta.

    const: theta[name]=lo
    pos:   bounded in [lo,hi] using log-bounded logistic
    nonneg: softplus then clip to [lo,hi]
    """
    out: Dict[str, float] = {}

    # fill const first
    for s in specs:
        if s.kind == "const":
            out[s.name] = float(s.lo)

    free = _free_specs(specs)
    if len(z) != len(free):
        raise ValueError(f"z has len {len(z)} but free specs has len {len(free)}")

    for zi, s in zip(z, free):
        zi_eff = smooth_clip(float(zi), zmax=20.0)

        if s.kind == "pos":
            lo = max(float(s.lo), 1e-300)
            hi = max(float(s.hi), lo * 1.0000001)
            a = np.log(lo)
            b = np.log(hi)
            y = float(sigmoid(np.array([zi_eff]))[0])  # in (0,1)
            lv = a + (b - a) * y
            v = float(np.exp(lv))
            out[s.name] = _clip_theta(v, s)

        elif s.kind == "nonneg":
            v = float(softplus(np.array([zi_eff]))[0])
            out[s.name] = _clip_theta(v, s)

        else:
            raise ValueError(f"Unknown spec kind: {s.kind}")

    return out

@dataclass
class SPSAConfig:
    n_iter: int = 40
    a: float = 0.10
    c: float = 0.05
    alpha: float = 0.602
    gamma: float = 0.101
    seed: int = 0
    print_every: int = 1
    grad_clip: float = 5.0
    step_clip: float = 2.0
    n_grad_avg: int = 1

@dataclass
class StageConfig:
    mesh_cfg: MeshBuildConfig
    fem_cfg: FEMConfig
    opt_cfg: SPSAConfig
    msh_path: Path
    model_cfg: ModelConfig = SSB_CURRENT

@dataclass
class FitResult:
    theta_best: Dict[str, float]
    score_best: float
    history: List[Tuple[int, float, Dict[str, float]]]
    objective_type: str = "loglikelihood"

def _fmt_theta_compact(th: Dict[str, float], specs: Optional[List["ParamSpec"]] = None, *, max_items: int = 14, prefer_first: Tuple[str, ...] = ("p", "q_I", "phi"), include_const: bool = False) -> str:
    """
    Compact diagnostic string for theta.

    If specs is provided, we use its order (and can skip const params).
    Otherwise we fall back to sorted(th.keys()).

    prefer_first: names to show first if present (in that order).
    max_items: cap total number of displayed parameters (rest shown as +N more).
    """
    # choose key order
    if specs is not None:
        keys = []
        for s in specs:
            if (not include_const) and (s.kind == "const"):
                continue
            keys.append(s.name)
    else:
        keys = sorted(th.keys())

    # move preferred keys to front
    front: List[str] = []
    used = set()
    for k in prefer_first:
        if k in th and k in keys:
            front.append(k); used.add(k)

    tail = [k for k in keys if (k in th) and (k not in used)]
    ordered = front + tail

    # limit items
    shown = ordered[:max_items]
    parts = []
    for k in shown:
        v = th.get(k, None)
        if v is None:
            continue
        try:
            parts.append(f"{k}={float(v):.6g}")
        except Exception:
            parts.append(f"{k}={v}")

    n_more = max(0, len(ordered) - len(shown))
    if n_more > 0:
        parts.append(f"+{n_more} more")

    return " ".join(parts)

_LOSS_OBJECTIVES = {"mae", "rmse", "smape", "log1p_mae", "log1p_rmse"}

def _canonical_objective_name(objective_type: str) -> str:
    obj = str(objective_type).lower().strip()
    if obj in ("ll", "nb", "negbin"):
        return "loglikelihood"
    return obj

def _objective_is_loss(objective_type: str) -> bool:
    return _canonical_objective_name(objective_type) in _LOSS_OBJECTIVES

def _display_objective_value(score: float, objective_type: str) -> float:
    """
    Internally we always maximize score.
    For loss objectives score=-loss, but users should see the positive loss.
    """
    score = float(score)
    return -score if _objective_is_loss(objective_type) else score

def _objective_label(objective_type: str) -> str:
    obj = _canonical_objective_name(objective_type)
    return obj if not _objective_is_loss(obj) else f"{obj}_loss"

def run_spsa_stage(
    stage: StageConfig,
    funcs_template: GSBFunctions,
    z0: np.ndarray,
    specs: List[ParamSpec],
    stage_pre: StagePrecompute,
    score_cfg: ScoreConfig,
    sync_boxes: Optional[Callable[[Dict[str, float]], None]] = None,
    progress: Optional[Callable[[int], None]] = None,
    score_func: Callable[..., float] = loglikelihood_theta,
    objective_type: str = "loglikelihood",
) -> Tuple[np.ndarray, FitResult]:

    obj = _canonical_objective_name(objective_type)
    label = _objective_label(obj)

    rng = np.random.default_rng(stage.opt_cfg.seed)
    z = z0.copy()

    best_score = -np.inf
    best_theta = unpack_z_to_theta(z, specs)
    history: List[Tuple[int, float, Dict[str, float]]] = []

    th0 = unpack_z_to_theta(z, specs)
    score0 = score_func(
        stage.msh_path,
        funcs_template,
        th0,
        stage.fem_cfg,
        stage_pre,
        score_cfg,
        sync_boxes=sync_boxes,
        model_cfg=stage.model_cfg,
    )

    best_score, best_theta = score0, th0

    if stage.opt_cfg.print_every > 0:
        print(
            f"[SPSA] iter 0/{stage.opt_cfg.n_iter} "
            f"{label}={_display_objective_value(score0, obj):.6e} "
            f"best_{label}={_display_objective_value(best_score, obj):.6e} "
            f"{_fmt_theta_compact(th0, specs)}"
        )

    for k in range(1, stage.opt_cfg.n_iter + 1):
        ak = stage.opt_cfg.a / ((k + 10.0) ** stage.opt_cfg.alpha)
        ck = stage.opt_cfg.c / (k ** stage.opt_cfg.gamma)

        n_grad_avg = max(1, int(getattr(stage.opt_cfg, "n_grad_avg", 1)))
        
        ghat = np.zeros_like(z, dtype=float)
        n_good = 0
        
        for _avg in range(n_grad_avg):
            delta = rng.choice([-1.0, 1.0], size=z.shape[0])
        
            z_plus = z + ck * delta
            z_minus = z - ck * delta
        
            th_plus = unpack_z_to_theta(z_plus, specs)
            th_minus = unpack_z_to_theta(z_minus, specs)
        
            score_plus = score_func(
                stage.msh_path, funcs_template, th_plus, stage.fem_cfg, stage_pre, score_cfg,
                sync_boxes=sync_boxes, model_cfg=stage.model_cfg
            )
            score_minus = score_func(
                stage.msh_path, funcs_template, th_minus, stage.fem_cfg, stage_pre, score_cfg,
                sync_boxes=sync_boxes, model_cfg=stage.model_cfg
            )
        
            if not (np.isfinite(score_plus) and np.isfinite(score_minus)):
                continue
        
            ghat += (score_plus - score_minus) / (2.0 * ck) * delta
            n_good += 1
        
        if n_good == 0:
            # No reliable gradient estimate this iteration.
            if progress is not None:
                progress(1)
            continue
        
        ghat /= float(n_good)
        ghat = np.clip(ghat, -stage.opt_cfg.grad_clip, stage.opt_cfg.grad_clip)

        dz = ak * ghat
        dz = np.clip(dz, -stage.opt_cfg.step_clip, stage.opt_cfg.step_clip)
        z = z + dz

        th = unpack_z_to_theta(z, specs)
        score = score_func(
            stage.msh_path, funcs_template, th, stage.fem_cfg, stage_pre, score_cfg,
            sync_boxes=sync_boxes, model_cfg=stage.model_cfg
        )

        if np.isfinite(score) and score > best_score:
            best_score = score
            best_theta = th

        history.append((k, score, th))

        if progress is not None:
            progress(1)

        if stage.opt_cfg.print_every > 0 and (k % stage.opt_cfg.print_every == 0):
            print(
                f"[SPSA] iter {k}/{stage.opt_cfg.n_iter} "
                f"{label}={_display_objective_value(score, obj):.6e} "
                f"best_{label}={_display_objective_value(best_score, obj):.6e} "
                f"{_fmt_theta_compact(th, specs)}"
            )

    return z, FitResult(
        theta_best=best_theta,
        score_best=best_score,
        history=history,
        objective_type=obj,
    )


def save_score_trace(
    history: List[Tuple[int, float, Dict[str, float]]],
    out_png: Path,
    *,
    objective_type: str = "loglikelihood",
) -> None:
    import matplotlib.pyplot as plt

    obj = _canonical_objective_name(objective_type)
    label = _objective_label(obj)

    it = np.array([h[0] for h in history], dtype=float)
    score = np.array([h[1] for h in history], dtype=float)
    display = np.array([_display_objective_value(s, obj) for s in score], dtype=float)

    plt.figure(figsize=(10, 4))
    plt.plot(it, display)
    plt.xlabel("Iteration")
    plt.ylabel(label)
    plt.title(f"SPSA {label} trace")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    
    
# =============================================================================
# Random search + multi-start refinement
# =============================================================================

@dataclass
class RandomSearchConfig:
    # number of random samples evaluated at stage 0
    N_0: int = 1000

    # refinement stages:
    # list of (K_keep, n_iter) meaning:
    #   take top K_keep candidates -> run SPSA for n_iter on each -> re-rank
    stages: Tuple[Tuple[int, int], ...] = ((100, 5), (10, 25))

    # final SPSA iterations run on best result after refinement
    final_n_iter: int = 50

    # keep best fraction when K_keep=None (optional convenience)
    keep_frac: Optional[float] = None

    # RNG seed for random sampling
    seed: int = 0

    # ranges for sampling in *theta* space
    # for "pos"/"nonneg": use log-uniform on [low, high] (with low>0)
    # for "unit": uniform on [low, high] subset of (0,1)
    theta_ranges: Optional[Dict[str, Tuple[float, float]]] = None

    # reject samples that produce non-finite likelihood quickly (optional)
    max_tries: int = 5
    
    # save the results in a CSV-file
    save_dir: Optional[Path] = None
    save_prefix: str = "multistart"


def _loguniform(rng: np.random.Generator, low: float, high: float) -> float:
    low = float(low); high = float(high)
    if low <= 0 or high <= 0:
        raise ValueError("loguniform requires low>0 and high>0")
    a = np.log(low); b = np.log(high)
    return float(np.exp(rng.uniform(a, b)))


def _uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.uniform(float(low), float(high)))


def sample_theta_from_ranges(rng: np.random.Generator, specs: List[ParamSpec], theta_ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, float]:
    """
    Sample theta within bounds.
    If theta_ranges is provided, it overrides spec bounds per-parameter.
    Sampling:
      - const: exactly lo
      - pos:   log-uniform on [lo, hi]
      - nonneg: mixture: with small prob choose exactly 0 if lo==0, otherwise log-uniform on (max(lo,eps), hi]
    """
    out: Dict[str, float] = {}
    eps = 1e-12

    for s in specs:
        if s.kind == "const":
            out[s.name] = float(s.lo)
            continue

        lo, hi = (s.lo, s.hi)
        if theta_ranges is not None and s.name in theta_ranges:
            lo, hi = map(float, theta_ranges[s.name])

        lo = float(lo); hi = float(hi)
        if hi < lo:
            raise ValueError(f"Sampling bounds hi < lo for {s.name}: {hi} < {lo}")

        if s.kind == "pos":
            lo_eff = max(lo, eps)
            hi_eff = max(hi, lo_eff * 1.0000001)
            out[s.name] = _loguniform(rng, lo_eff, hi_eff)

        elif s.kind == "nonneg":
            # allow spike at 0 when lo==0 (helps test "parameter off" quickly)
            if lo <= 0.0 and rng.random() < 0.15:
                out[s.name] = 0.0
            else:
                lo_eff = max(lo, eps)
                hi_eff = max(hi, lo_eff * 1.0000001)
                out[s.name] = _loguniform(rng, lo_eff, hi_eff)

        else:
            raise ValueError(f"Unknown spec kind: {s.kind}")

        # final clip for safety
        out[s.name] = float(np.clip(out[s.name], s.lo, s.hi))

    return out


def random_search_candidates(stage: StageConfig, funcs_template: GSBFunctions, specs: List[ParamSpec], stage_pre: StagePrecompute,
    score_cfg: ScoreConfig, rs_cfg: RandomSearchConfig, sync_boxes: Optional[Callable[[Dict[str, float]], None]] = None,
    progress: Optional[Callable[[int], None]] = None, score_func: Callable[..., float] = loglikelihood_theta, objective_type: str = "loglikelihood",) -> List[Tuple[float, Dict[str, float]]]:
    """
    Returns list sorted by descending score: [(score, theta), ...]
    """
    rng = np.random.default_rng(rs_cfg.seed)
    theta_ranges = rs_cfg.theta_ranges or None
    
    obj = _canonical_objective_name(objective_type)
    label = _objective_label(obj)

    scored: List[Tuple[float, Dict[str, float]]] = []
    n_eval = int(rs_cfg.N_0)

    for i in range(n_eval):
        th = sample_theta_from_ranges(rng, specs, theta_ranges)
        score = score_func(stage.msh_path, funcs_template, th, stage.fem_cfg, stage_pre, score_cfg, sync_boxes=sync_boxes, model_cfg=stage.model_cfg)
        if np.isfinite(score):
            scored.append((float(score), th))
            
        if progress is not None:
            progress(1)

        if (i + 1) % max(1, n_eval // 10) == 0:
            best = max(scored, key=lambda x: x[0])[0] if scored else float("-inf")
            best_disp = _display_objective_value(best, obj)
            print(
                f"[RS] {i+1}/{n_eval} evaluated, finite={len(scored)} "
                f"best_{label}={best_disp:.6e}"
            )

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _keep_top(scored: List[Tuple[float, Dict[str, float]]], K_keep: Optional[int], keep_frac: Optional[float]) -> List[Tuple[float, Dict[str, float]]]:
    if not scored:
        return []
    if K_keep is not None:
        return scored[: max(1, int(K_keep))]
    if keep_frac is None:
        raise ValueError("Either K_keep or keep_frac must be provided.")
    k = max(1, int(np.ceil(float(keep_frac) * len(scored))))
    return scored[:k]


def multi_start_refine(stage: StageConfig, funcs_template: GSBFunctions, specs: List[ParamSpec], stage_pre: StagePrecompute, score_cfg: ScoreConfig, rs_cfg: RandomSearchConfig, 
    sync_boxes: Optional[Callable[[Dict[str, float]], None]] = None, progress: Optional[Callable[[int], None]] = None,
    score_func: Callable[..., float] = loglikelihood_theta, objective_type: str = "loglikelihood",) -> Tuple[np.ndarray, FitResult]:
    
    obj = _canonical_objective_name(objective_type)
    label = _objective_label(obj)
    
    scored = random_search_candidates(stage, funcs_template, specs, stage_pre, score_cfg, rs_cfg, sync_boxes=sync_boxes, progress=progress, score_func=score_func, objective_type=obj)
    if not scored:
        raise RuntimeError("Random search produced no finite likelihood candidates.")

    cur = scored

    for si, (K_keep, n_iter) in enumerate(rs_cfg.stages, start=1):
        cur = _keep_top(cur, K_keep=K_keep, keep_frac=rs_cfg.keep_frac)
        print(f"[MS] Stage {si}: refining {len(cur)} candidates with {n_iter} SPSA steps each")

        refined: List[Tuple[float, Dict[str, float]]] = []
        for j, (score_seed, th_seed) in enumerate(cur, start=1):
            opt_cfg = SPSAConfig(**{**stage.opt_cfg.__dict__})
            opt_cfg.n_iter = int(n_iter)
            opt_cfg.seed = int(rs_cfg.seed + 10_000 * si + j)

            stage_local = StageConfig(mesh_cfg=stage.mesh_cfg, fem_cfg=stage.fem_cfg, opt_cfg=opt_cfg, msh_path=stage.msh_path, model_cfg=stage.model_cfg)

            z0 = pack_theta_to_z(th_seed, specs)
            z_best, res = run_spsa_stage(stage_local, funcs_template, z0, specs, stage_pre, score_cfg, sync_boxes=sync_boxes, progress=progress, score_func=score_func, objective_type=obj)
            refined.append((float(res.score_best), res.theta_best))

        refined.sort(key=lambda x: x[0], reverse=True)
        cur = refined

        best_disp = _display_objective_value(cur[0][0], obj)
        print(
            f"[MS] Stage {si} done. "
            f"best_{label}={best_disp:.6e} "
            f"theta={_fmt_theta_compact(cur[0][1], specs)}"
        )

        # Save refined list to CSV
        if rs_cfg.save_dir is not None:
            out_csv = Path(rs_cfg.save_dir) / f"{rs_cfg.save_prefix}_stage{si}_top{len(cur)}.csv"
            save_candidates_csv(cur, out_csv, specs, objective_type=obj)

    # Final SPSA on best candidate
    best_score, best_th = cur[0]
    print(f"[MS] Final: running {rs_cfg.final_n_iter} SPSA steps from best staged candidate")

    opt_cfg = SPSAConfig(**{**stage.opt_cfg.__dict__})
    opt_cfg.n_iter = int(rs_cfg.final_n_iter)
    opt_cfg.seed = int(rs_cfg.seed + 999_999)

    stage_final = StageConfig(mesh_cfg=stage.mesh_cfg, fem_cfg=stage.fem_cfg, opt_cfg=opt_cfg, msh_path=stage.msh_path, model_cfg=stage.model_cfg)

    z0 = pack_theta_to_z(best_th, specs)
    z_best, res_best = run_spsa_stage(stage_final, funcs_template, z0, specs, stage_pre, score_cfg, sync_boxes=sync_boxes, progress=progress, score_func=score_func, objective_type=obj)

    # Save final result (single row) and also "final top list" if preferred
    if rs_cfg.save_dir is not None:
        out_csv = Path(rs_cfg.save_dir) / f"{rs_cfg.save_prefix}_final.csv"
        save_candidates_csv(
            [(float(res_best.score_best), res_best.theta_best)],
            out_csv,
            specs,
            objective_type=obj,
        )

    return z_best, res_best


def save_candidates_csv(candidates: List[Tuple[float, Dict[str, float]]], out_csv: Path, specs: List[ParamSpec], *, objective_type: str = "loglikelihood",) -> None:
    """
    candidates: [(ll, theta), ...] assumed already sorted desc by ll (but we re-sort anyway).
    """
    if not candidates:
        return

    cand = sorted(candidates, key=lambda x: x[0], reverse=True)

    rows = []
    
    obj = _canonical_objective_name(objective_type)
    label = _objective_label(obj)
    
    for score, th in cand:
        row = {
            "objective_type": obj,
            "score": float(score),
            label: float(_display_objective_value(score, obj)),
        }
    
        # Backward compatibility for old CSV consumers.
        if obj == "loglikelihood":
            row["ll"] = float(score)
            
        for s in specs:
            row[s.name] = float(th.get(s.name, np.nan))
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


# =============================================================================
# Runner: build mesh once + run MLE (callable for multiprocessing)
# =============================================================================

class Runner:
    """
    A callable MLE runner with two entrypoints:
      - build_mesh(): build (or reuse) mesh into out/<out_folder>/mesh
      - run_MLE(): run random-search + multi-start + SPSA, saving CSVs/figs to out/<out_folder>/...

    Folder layout:
      out/<out_folder>/
        mesh/
        figures/
        csv/
    """

    def __init__(
        self,
        out_folder: str,
        *,
        mesh_params: Dict,
        model_params: Dict[str, Tuple[str, float, float]],
        model_cfg: ModelConfig = SSB_CURRENT,
        time_params: Dict,
        spsa_params: Dict,
        randomSearch_params: Dict,
        base_data_dir: Path = Path("data"),
        out_root: Path = Path("out"),
        epsg_project: int = 5070,
        events_csv: Optional[Path] = None,
        admin1_shp: Optional[Path] = None,
        events_min_year: Optional[float] = None,
        events_max_year: Optional[float] = None,
        rs_save_prefix: Optional[str] = None,
        score_lambda_floor: float = 1e-12,
        score_normalize_by_events: bool = True,
        score_finder_chunk_size: int = 2000,
        picard_max_iter: int = 15,
        picard_tol: float = 1e-8,
        fem_verbose: bool = False,
        mesh_verbose: bool = False,
        score_verbose: bool = False,
        score_verbose_freq: Optional[int] = None,
        benchmark_model: str = "gsb",
        smith_song_history_mode: str = "conditional",
        objective_type: str = "loglikelihood",
    ):
        self.out_folder = str(out_folder)
        self._t0 = time.time()

        self.mesh_params = dict(mesh_params)
        self.model_params = dict(model_params)
        self.model_cfg = model_cfg
        self.time_params = dict(time_params)
        self.spsa_params = dict(spsa_params)
        self.randomSearch_params = dict(randomSearch_params)

        self.base_data_dir = Path(base_data_dir)
        self.out_root = Path(out_root)
        self.epsg_project = int(epsg_project)

        # inputs (defaults match your current layout)
        self.events_csv = Path(events_csv) if events_csv is not None else (self.base_data_dir / "processed" / "solar_installations_all.csv")
        self.admin1_shp = Path(admin1_shp) if admin1_shp is not None else (
            self.base_data_dir / "raw" / "maps" / "ne_10m_admin_1_states_provinces_lakes" / "ne_10m_admin_1_states_provinces_lakes.shp")

        # event filters
        self.events_min_year = events_min_year
        self.events_max_year = events_max_year

        # likelihood knobs
        self.score_lambda_floor = float(score_lambda_floor)
        self.score_normalize_by_events = bool(score_normalize_by_events)
        self.score_finder_chunk_size = int(score_finder_chunk_size)

        # FEM solver knobs (kept as in your main unless overridden)
        self.picard_max_iter = int(picard_max_iter)
        self.picard_tol = float(picard_tol)
        
        # verbosity settings
        self.fem_verbose = bool(fem_verbose)
        self.mesh_verbose = bool(mesh_verbose)
        self.score_verbose = bool(score_verbose)
        self.score_verbose_freq = score_verbose_freq

        # output paths
        self.out_dir = self.out_root / self.out_folder
        self.mesh_dir = self.out_dir / "mesh"
        self.fig_dir = self.out_dir / "figures"
        self.csv_dir = self.out_dir / "csv"
        self.mesh_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)

        # naming
        self.rs_save_prefix = rs_save_prefix if rs_save_prefix is not None else self.out_folder

        # computed later
        self.msh_path: Optional[Path] = None
        
        # benchmark model
        self.benchmark_model = str(benchmark_model).lower().strip()
        self.smith_song_history_mode = str(smith_song_history_mode).lower().strip()
        
        # objective function
        self.objective_type = str(objective_type).lower().strip()

    # -------------------------
    # Mesh building
    # -------------------------
    def _mesh_filename(self) -> str:
        h = int(round(float(self.mesh_params["h_km"])))
        s = int(round(float(self.mesh_params["simplify_km"])))
        return f"{h}_{s}_km.msh"

    def build_mesh(self) -> Path:
        """
        Builds mesh and returns path.
        """
        try:
            state_list = list(self.mesh_params["state_list"])
            h_km = float(self.mesh_params["h_km"])
            simplify_km = float(self.mesh_params["simplify_km"])

            cfg = MeshBuildConfig(h_km=h_km, simplify_km=simplify_km, epsg_project=self.epsg_project)
            msh_path = self.mesh_dir / self._mesh_filename()

            if not msh_path.exists():
                build_mesh_from_admin1_region(self.admin1_shp, state_list, msh_path, cfg, verbose=self.mesh_verbose, model_name=f"mesh_{self.out_folder}")

            self.msh_path = msh_path
            return msh_path
        finally:
            self._log("self.build_mesh complete")
    
    # -------------------------
    # Timestamped logging
    # -------------------------
    def _elapsed_hms(self) -> str:
        dt = int(time.time() - self._t0)
        h = dt // 3600
        m = (dt % 3600) // 60
        s = dt % 60
        return f"{h:02d}:{m:02d}:{s:02d}"
    
    def _log(self, msg: str) -> None:
        print(f"{self.out_folder}@[{self._elapsed_hms()}] ---- {msg}")

    # -------------------------
    # Model: build funcs + sync_boxes
    # -------------------------
    def _build_funcs_template(self) -> Tuple[GSBFunctions, Callable[[Dict[str, float]], None]]:
        if self.model_cfg.build_mle_template is None:
            raise RuntimeError("Model config does not provide an MLE template builder.")
        return self.model_cfg.build_mle_template()

    # -------------------------
    # Specs from model_params
    # -------------------------
    def _build_specs(self) -> List[ParamSpec]:
        specs: List[ParamSpec] = []
        for name, triple in self.model_params.items():
            if not (isinstance(triple, (tuple, list)) and len(triple) == 3):
                raise ValueError(f"model_params[{name}] must be a 3-tuple (kind, lo, hi), got {triple}")
            kind, lo, hi = triple
            specs.append(ParamSpec(str(name), str(kind), float(lo), float(hi)))
        return specs

    # -------------------------
    # Run MLE
    # -------------------------
    def run_MLE(self) -> FitResult:
        try:
            if self.msh_path is None:
                self.build_mesh()

            assert self.msh_path is not None

            YEAR0 = float(self.time_params["start_year"])
            tau = float(self.time_params["tau"])

            events = load_events_csv(self.events_csv, region_states=list(self.mesh_params["state_list"]), YEAR0=YEAR0, epsg_project=self.epsg_project, 
                min_year=self.events_min_year if self.events_min_year is not None else YEAR0, max_year=self.events_max_year)

            t_max = float(np.max(events.t_years)) if events.t_years.size else 0.0
            score_cfg = ScoreConfig(t_min=0, t_max=t_max, lambda_floor=self.score_lambda_floor, verbose=self.score_verbose,
                normalize_by_events=self.score_normalize_by_events, finder_chunk_size=self.score_finder_chunk_size)

            funcs, sync_boxes = self._build_funcs_template()
            specs = self._build_specs()

            fem_cfg = FEMConfig(tau_years=tau, T_years=score_cfg.t_max, picard_max_iter=self.picard_max_iter, picard_tol=self.picard_tol,
                verbose=self.fem_verbose, YEAR0=YEAR0, epsg_project=self.epsg_project)

            opt_cfg = SPSAConfig(
                n_iter=int(self.spsa_params["n_iter"]),
                a=float(self.spsa_params["a"]),
                c=float(self.spsa_params["c"]),
                alpha=float(self.spsa_params.get("alpha", 0.602)),
                gamma=float(self.spsa_params.get("gamma", 0.101)),
                seed=int(self.spsa_params.get("seed", 0)),
                print_every=int(self.spsa_params.get("print_every", 1)),
                grad_clip=float(self.spsa_params["grad_clip"]),
                step_clip=float(self.spsa_params["step_clip"]),
                n_grad_avg=int(self.spsa_params.get("n_grad_avg", 1)),
            )

            stage = StageConfig(mesh_cfg=MeshBuildConfig(h_km=float(self.mesh_params["h_km"]), simplify_km=float(self.mesh_params["simplify_km"]), epsg_project=self.epsg_project),
                fem_cfg=fem_cfg, opt_cfg=opt_cfg, msh_path=self.msh_path, model_cfg=self.model_cfg)

            self._log("precompute_stage_objects starting")
            stage_pre = precompute_stage_objects(stage.msh_path, stage.fem_cfg, events, score_cfg)
            self._log("precompute_stage_objects complete")
            
            if self.benchmark_model in ("gsb", "pde", "fem"):
                def score_func(*args, **kwargs):
                    return objective_gsb_theta(*args, **kwargs, objective_type=self.objective_type)
            elif self.benchmark_model in ("smith_song", "smith-song", "smithsong"):
                def score_func(*args, **kwargs):
                    return objective_smith_song_theta(*args, **kwargs, objective_type=self.objective_type, smith_song_history_mode=self.smith_song_history_mode)
            else:
                raise ValueError(f"Unknown benchmark_model={self.benchmark_model!r}.")
            self._log(f"optimization objective_type={self.objective_type}")

            rs_cfg = RandomSearchConfig(N_0=int(self.randomSearch_params["N_0"]), stages=tuple(self.randomSearch_params["stages"]), final_n_iter=int(self.spsa_params["n_iter"]),
                seed=int(self.randomSearch_params["seed"]), theta_ranges=None, save_dir=self.csv_dir, save_prefix=self.rs_save_prefix)

            # -------------------------
            # Progress tracking (timestamped)
            # -------------------------
            total_iters = int(rs_cfg.N_0) + int(sum(int(K) * int(nit) for (K, nit) in rs_cfg.stages)) + int(rs_cfg.final_n_iter)

            tracker = ProgressTracker(total=total_iters, freq=self.score_verbose_freq, printer=lambda s: self._log(s))

            progress_cb = tracker.tick if (self.score_verbose_freq is not None and self.score_verbose_freq > 0) else None

            self._log(f"optimization objective_type={self.objective_type}")
            self._log(f"optimization starting with benchmark_model={self.benchmark_model}")
            z_best, res_best = multi_start_refine(stage=stage, funcs_template=funcs, specs=specs, stage_pre=stage_pre,
                score_cfg=score_cfg, rs_cfg=rs_cfg, sync_boxes=sync_boxes, progress=progress_cb, score_func=score_func, objective_type=self.objective_type,)

            trace_png = self.fig_dir / f"{self.rs_save_prefix}_{self.objective_type}_trace.png"
            save_score_trace(res_best.history, trace_png, objective_type=self.objective_type)

            return res_best
        finally:
            self._log("self.run_MLE complete")


# =============================================================================
# Example for New York: create runner, build mesh, run MLE
# =============================================================================

if __name__ == "__main__":
    runner = Runner(
        out_folder="example_ny_run",
        mesh_params=dict(
            state_list=["NY"],
            h_km=6,
            simplify_km=18,
        ),
        model_params=dict(
            r0=("pos", 0.15, 5),
            r1=("nonneg", 0, 10),
            r2=("nonneg", 0, 10),
            p=("pos", 1e-5, 1),
            q_I=("pos", 1e-5, 1),
            gamma_J=("pos", 1e-5, 10),
            k_J=("nonneg", 0, 1),
            D=("pos", 1, 1e5),
            S0=("const", 0, 0),
            phi=("pos", 1e-3, 1e4),
        ),
        time_params=dict(
            start_year=2003.375,
            tau=0.025,
        ),
        spsa_params=dict(
            n_iter=100,
            a=0.2,
            c=0.2,
            gamma=0,
            seed=0,
            grad_clip=20,
            step_clip=10,
        ),
        randomSearch_params=dict(
            N_0=100,
            stages=((25, 4), (5, 20)),
            seed=0,
        ),
        fem_verbose=False,
        mesh_verbose=False,
        ll_verbose=False,
        score_verbose_freq=100,
    )

    runner.build_mesh()
    res = runner.run_MLE()
    print("[DONE] best ll:", res.ll_best)
    specs = runner._build_specs()
    print("[DONE] best theta:", _fmt_theta_compact(res.theta_best, specs))