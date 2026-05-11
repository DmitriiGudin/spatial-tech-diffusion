#!/usr/bin/env python3
# benchmark_utils.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, List

import time
import numpy as np

from fem_utils import (
    FEMConfig,
    poisson_deviance,
    pearson_residuals,
    forecast_error_metrics,
    _plot_data_vs_mu_year_nodes_lonlat,
    _plot_pearson_residuals_year_nodes_lonlat,
)

from mle_utils import (
    LikelihoodConfig,
    load_events_csv,
    precompute_stage_objects,
    build_smith_song_precompute,
    smith_song_expected_counts,
)

from mesh_utils import MeshBuildConfig, build_mesh_from_admin1_region


# =============================================================================
# Small logging helper
# =============================================================================

def _fmt_hhmmss(seconds: float) -> str:
    seconds = float(max(seconds, 0.0))
    h = int(seconds // 3600)
    m = int((seconds - 3600 * h) // 60)
    s = int(seconds - 3600 * h - 60 * m)
    return f"{h:02d}:{m:02d}:{s:02d}"


# =============================================================================
# Aggregation helpers
# =============================================================================

def aggregate_snapshot_node_to_year_node(values_tn: np.ndarray, times: np.ndarray, YEAR0: float, *, t_min_year: int, t_max_year: int) -> np.ndarray:
    """
    Aggregate snapshot-node quantities to calendar-year-node quantities.

    values_tn[k, i] is assumed to be a count/mass assigned to snapshot k.
    We assign snapshot k to floor(YEAR0 + times[k]).
    """
    values_tn = np.asarray(values_tn, float)
    times = np.asarray(times, float)

    if values_tn.ndim != 2:
        raise ValueError(f"values_tn must be 2D, got shape={values_tn.shape}")
    if values_tn.shape[0] != times.size:
        raise ValueError(f"values_tn nt={values_tn.shape[0]} does not match times size={times.size}")

    ny = int(t_max_year) - int(t_min_year) + 1
    if ny <= 0:
        raise ValueError(f"Invalid year window: {t_min_year}..{t_max_year}")

    out = np.zeros((ny, values_tn.shape[1]), dtype=float)

    cal_years = np.asarray(float(YEAR0) + times, float)
    year_keys = np.floor(cal_years + 1e-9).astype(int)

    for k, yy in enumerate(year_keys):
        if int(t_min_year) <= yy <= int(t_max_year):
            out[int(yy) - int(t_min_year)] += values_tn[k]

    return out


# =============================================================================
# Plotting helpers
# =============================================================================

def _plot_smith_song_monthly_totals(out_png: Path, times: np.ndarray, YEAR0: float, counts_snap_node: np.ndarray, mu_snap_node: np.ndarray,
    title: str = "Smith-Song monthly totals: Data vs prediction"):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    cal_years = YEAR0 + np.asarray(times, float)

    def frac_year_to_timestamp(y):
        year = int(np.floor(y))
        frac = y - year
        start = pd.Timestamp(year=year, month=1, day=1)
        end = pd.Timestamp(year=year + 1, month=1, day=1)
        return start + pd.Timedelta(days=int(frac * (end - start).days))

    dates = pd.to_datetime([frac_year_to_timestamp(y) for y in cal_years])
    month = dates.to_period("M").to_timestamp()

    df = pd.DataFrame({
        "month": month,
        "data": np.sum(counts_snap_node, axis=1),
        "mu": np.sum(mu_snap_node, axis=1),
    })

    g = df.groupby("month", as_index=True)[["data", "mu"]].sum()

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(g.index, g["data"], label="Data")
    ax.plot(g.index, g["mu"], label="Smith-Song mean")
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Count per month")
    ax.grid(True, alpha=0.25)
    ax.legend()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# =============================================================================
# Smith-Song benchmark diagnostics runner
# =============================================================================

@dataclass
class SmithSongDiagnosticRunner:
    out_folder: str
    mesh_params: Dict[str, Any]
    model_params: Dict[str, float]
    time_params: Dict[str, float]

    fem_verbose: bool = False
    mesh_verbose: bool = False

    cities: Dict[str, Sequence[float]] = field(default_factory=dict)
    years_to_plot: Optional[List[int]] = None

    events_csv: Path = Path("data") / "processed" / "solar_installations_all.csv"
    events_state_col: str = "state"

    base_out: Path = Path("out")

    _t0: float = field(init=False, repr=False)
    
    smith_song_history_mode: str = "conditional"

    def __post_init__(self):
        self._t0 = time.perf_counter()

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

    def log(self, msg: str) -> None:
        dt = time.perf_counter() - self._t0
        print(f"{self.out_folder}@[{_fmt_hhmmss(dt)}] ---- {msg}")

    def build_fem_config(self) -> FEMConfig:
        tau = float(self.time_params.get("tau", 0.05))
        start_year = float(self.time_params["start_year"])

        t_max_year = int(self.time_params.get("t_max_year", 2023))
        T_years = float(self.time_params.get("T_years", max(0.0, float(t_max_year + 1) - start_year)))

        epsg = int(self.mesh_params.get("epsg_project", 5070))

        return FEMConfig(tau_years=tau, T_years=T_years, picard_max_iter=int(self.time_params.get("picard_max_iter", 20)), picard_tol=float(self.time_params.get("picard_tol", 1e-8)),
            verbose=bool(self.fem_verbose), YEAR0=float(start_year), epsg_project=epsg)

    def build_mesh(self) -> Path:
        """
        Builds mesh into out/<out_folder>/mesh/<h>_<simplify>_km.msh if missing.
        """

        self.mesh_dir.mkdir(parents=True, exist_ok=True)
        msh = self.mesh_path()

        admin1_shp = (
            Path("data")
            / "raw"
            / "maps"
            / "ne_10m_admin_1_states_provinces_lakes"
            / "ne_10m_admin_1_states_provinces_lakes.shp"
        )

        state_list = list(self.mesh_params["state_list"])
        h_km = float(self.mesh_params["h_km"])
        simplify_km = float(self.mesh_params["simplify_km"])
        epsg = int(self.mesh_params.get("epsg_project", 5070))

        cfg = MeshBuildConfig(h_km=h_km, simplify_km=simplify_km, epsg_project=epsg)

        if msh.exists():
            self.log(f"mesh exists: {msh.name} (skipping build)")
            return msh

        t0 = time.perf_counter()
        build_mesh_from_admin1_region(admin1_shp, state_list, msh, cfg, verbose=bool(self.mesh_verbose), model_name=f"{self.out_folder}_mesh_km")
        self.log(f"build_mesh_from_admin1_region complete ({time.perf_counter() - t0:.3f} s)")
        return msh


    def run(self) -> Dict[str, Any]:
        """
        Runs Smith-Song benchmark diagnostics and saves comparable plots.

        Returns a summary dict with the same key style as fem_utils.Runner.run_FEM().
        """
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        msh = self.mesh_path()
        if not msh.exists():
            raise FileNotFoundError(f"Mesh not found: {msh}. Call build_mesh() first.")

        fem_cfg = self.build_fem_config()

        state_list = list(self.mesh_params["state_list"])
        if len(state_list) != 1:
            raise ValueError("SmithSongDiagnosticRunner currently expects exactly one state.")
        st = str(state_list[0]).strip()
        
        self.log(f"[benchmark] Smith-Song history_mode={self.smith_song_history_mode}")

        t_min_year = int(self.time_params.get("t_min_year", int(np.floor(fem_cfg.YEAR0))))
        t_max_year = int(self.time_params.get("t_max_year", int(np.floor(fem_cfg.YEAR0 + fem_cfg.T_years))))

        # ---------------------------------------------------------------------
        # Load events and build StagePrecompute.
        # This gives us:
        #   - mesh/cache/rho/cost/trend
        #   - snapshot-node observed counts
        #   - log-factorials
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        events = load_events_csv(self.events_csv, region_states=state_list, YEAR0=fem_cfg.YEAR0,
            epsg_project=fem_cfg.epsg_project, min_year=float(t_min_year), max_year=float(t_max_year + 1))

        ll_cfg = LikelihoodConfig(
            t_min=0.0,
            t_max=float(fem_cfg.T_years),
            lambda_floor=float(self.time_params.get("lambda_floor", 1e-12)),
            verbose=False,
            normalize_by_events=True,
            finder_chunk_size=int(self.time_params.get("bin_chunk_size", 5000)),
        )

        stage_pre = precompute_stage_objects(msh, fem_cfg, events, ll_cfg)
        self.log(f"precompute_stage_objects complete ({time.perf_counter() - t0:.3f} s)")
        self.log(
            f"[bin] snapshot events={stage_pre.K_total_window:,} "
            f"inside mesh={stage_pre.K_inside:,} shape={stage_pre.counts_node.shape}"
        )

        # ---------------------------------------------------------------------
        # Smith-Song expected counts on snapshot-node grid
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        ss_pre = build_smith_song_precompute(stage_pre, top_k=int(self.time_params.get("ss_top_k", 50)))
        mu_snap_node = smith_song_expected_counts(stage_pre=stage_pre, theta=self.model_params, ss_pre=ss_pre, t_min=ll_cfg.t_min, t_max=ll_cfg.t_max,
    eps=1e-300, top_k=int(self.time_params.get("ss_top_k", 50)), kernel_tol=float(self.time_params.get("ss_kernel_tol", 1e-6)), history_mode=self.smith_song_history_mode)
        self.log(f"smith_song_expected_counts complete ({time.perf_counter() - t0:.3f} s)")
        self.log(f"[mu] total expected snapshot={float(mu_snap_node.sum()):.6g}")
      
        monthly_png = self.fig_dir / f"{st.lower()}_smith_song_monthly_totals_vs_data.png"
        _plot_smith_song_monthly_totals(out_png=monthly_png, times=stage_pre.fem_cache.times, YEAR0=fem_cfg.YEAR0, counts_snap_node=stage_pre.counts_node, mu_snap_node=mu_snap_node)
        self.log("_plot_smith_song_monthly_totals complete")
      
        # ---------------------------------------------------------------------
        # Aggregate both observed and expected snapshot-node arrays to year-node.
        # This gives diagnostics comparable to FEM yearly plots.
        # ---------------------------------------------------------------------
        counts_node = aggregate_snapshot_node_to_year_node(stage_pre.counts_node, stage_pre.fem_cache.times, fem_cfg.YEAR0, t_min_year=t_min_year, t_max_year=t_max_year)

        mu_node = aggregate_snapshot_node_to_year_node(mu_snap_node, stage_pre.fem_cache.times, fem_cfg.YEAR0, t_min_year=t_min_year, t_max_year=t_max_year)

        if counts_node.shape != mu_node.shape:
            raise RuntimeError(f"Shape mismatch: counts_node {counts_node.shape} vs mu_node {mu_node.shape}")

        self.log(f"[mu] total expected yearly={float(mu_node.sum()):.6g}")

        # ---------------------------------------------------------------------
        # Diagnostics
        # ---------------------------------------------------------------------
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

        metrics = forecast_error_metrics(
            counts_node,
            mu_node,
            eps=float(self.time_params.get("metric_eps", 1e-12)),
            min_denom=float(self.time_params.get("mape_min_denom", 1.0)),
        )
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

        # ---------------------------------------------------------------------
        # Plots: data vs mu and residuals.
        # No u/v/I/J plots because this benchmark has no PDE fields.
        # ---------------------------------------------------------------------
        h_km = float(self.mesh_params["h_km"])
        years_to_plot = self.years_to_plot
        if years_to_plot is None:
            years_to_plot = [
                t_min_year,
                t_min_year + 3,
                t_min_year + 6,
                t_min_year + 9,
                t_max_year - 4,
                t_max_year - 2,
                t_max_year,
            ]
            years_to_plot = sorted({yy for yy in years_to_plot if t_min_year <= yy <= t_max_year})

        for yy in years_to_plot:
            idx = int(yy - t_min_year)
            if idx < 0 or idx >= counts_node.shape[0]:
                self.log(f"[warn] year {yy} out of range for bins; skipping")
                continue

            y_node = counts_node[idx, :]
            mu_y = mu_node[idx, :]

            out_png = self.fig_dir / f"{st.lower()}_smith_song_data_vs_mu_nodes_{yy}.png"
            t0 = time.perf_counter()
            _plot_data_vs_mu_year_nodes_lonlat(msh_path=msh, epsg_project=fem_cfg.epsg_project, out_png=out_png,
                y_node=y_node, mu_node=mu_y, year=int(yy), h_km=h_km, cities=self.cities)
            self.log(f"_plot_data_vs_mu_year_nodes_lonlat complete ({time.perf_counter() - t0:.3f} s)")

            out_png_R = self.fig_dir / f"{st.lower()}_smith_song_pearson_residual_nodes_{yy}.png"
            t0 = time.perf_counter()
            _plot_pearson_residuals_year_nodes_lonlat(msh_path=msh, epsg_project=fem_cfg.epsg_project, out_png=out_png_R, R_node=R[idx, :],
                year=int(yy), h_km=h_km, cities=self.cities, vlim_log=float(self.time_params.get("vlim_log", 2.5)))
            self.log(f"_plot_pearson_residuals_year_nodes_lonlat complete ({time.perf_counter() - t0:.3f} s)")

        self.log("SmithSongDiagnosticRunner.run complete")

        return dict(out_folder=self.out_folder, state=st, mesh=str(msh), figures=str(self.fig_dir), benchmark_model="smith_song", smith_song_history_mode=self.smith_song_history_mode, 
            K_total=int(stage_pre.K_total_window), K_inside=int(stage_pre.K_inside), deviance=float(D_total), pearson_rms=float(R_rms), **metrics)


def run_smith_song_diagnostics(
    out_folder: str,
    *,
    mesh_params: Dict[str, Any],
    model_params: Dict[str, float],
    time_params: Dict[str, float],
    fem_verbose: bool = False,
    mesh_verbose: bool = False,
    cities: Optional[Dict[str, Sequence[float]]] = None,
    years_to_plot: Optional[List[int]] = None,
    events_csv: Path = Path("data") / "processed" / "solar_installations_all.csv",
    events_state_col: str = "state",
    base_out: Path = Path("out"),
    smith_song_history_mode: str = "conditional",
) -> Dict[str, Any]:
    runner = SmithSongDiagnosticRunner(
        out_folder=out_folder,
        mesh_params=mesh_params,
        model_params=model_params,
        time_params=time_params,
        smith_song_history_mode=str(smith_song_history_mode),
        fem_verbose=fem_verbose,
        mesh_verbose=mesh_verbose,
        cities=dict(cities or {}),
        years_to_plot=years_to_plot,
        events_csv=Path(events_csv),
        events_state_col=str(events_state_col),
        base_out=Path(base_out),
    )
    runner.build_mesh()
    return runner.run()