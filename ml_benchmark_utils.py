#!/usr/bin/env python3
# ml_benchmark_utils.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, List, Tuple

import time
import json
import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from fem_utils import (
    FEMConfig,
    bin_events_year_node,
    poisson_deviance,
    pearson_residuals,
    forecast_error_metrics,
    _plot_data_vs_mu_year_nodes_lonlat,
    _plot_pearson_residuals_year_nodes_lonlat,
    bin_events_month_total_inside_mesh,
    make_month_edges_years,
    fit_bass_to_monthly_counts,
)

from mle_utils import load_events_csv, precompute_stage_objects, ScoreConfig
from benchmark_utils import aggregate_snapshot_node_to_year_node, _aggregate_snapshot_mass_to_months
from mesh_utils import MeshBuildConfig, build_mesh_from_admin1_region, make_region_tag


def _fmt_hhmmss(seconds: float) -> str:
    seconds = float(max(seconds, 0.0))
    h = int(seconds // 3600)
    m = int((seconds - 3600 * h) // 60)
    s = int(seconds - 3600 * h - 60 * m)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _topk_neighbor_precompute(xy: np.ndarray, top_k: int = 50, theta: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(xy, float)
    N = xy.shape[0]
    K = min(int(top_k) + 1, N)

    dx = xy[:, None, 0] - xy[None, :, 0]
    dy = xy[:, None, 1] - xy[None, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)

    idx = np.argpartition(dist, kth=K - 1, axis=1)[:, :K]
    row = np.arange(N)[:, None]
    d_unsorted = dist[row, idx]
    order = np.argsort(d_unsorted, axis=1)

    idx = np.take_along_axis(idx, order, axis=1)[:, 1:]
    d = np.take_along_axis(d_unsorted, order, axis=1)[:, 1:]

    W = np.exp(-float(theta) * d)
    W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-15)
    return idx.astype(np.int64), W.astype(float)


def build_ml_features_one_snapshot(
    *,
    k: int,
    counts_hist: np.ndarray,
    rho_total: np.ndarray,
    cost_nodes: Optional[np.ndarray],
    trend_factor: Optional[np.ndarray],
    xy_nodes_km: np.ndarray,
    times: np.ndarray,
    YEAR0: float,
    neigh_idx: np.ndarray,
    neigh_w: np.ndarray,
    lag_steps: int = 3,
) -> np.ndarray:
    """
    Build X_k for all nodes at snapshot k using counts_hist as history.

    counts_hist should contain observed counts for training/conditional mode,
    and predicted counts for generative test mode where appropriate.
    """
    counts_hist = np.asarray(counts_hist, float)
    rho_total = np.asarray(rho_total, float)
    xy = np.asarray(xy_nodes_km, float)

    nt, N = counts_hist.shape

    cum_prev = counts_hist[:k].sum(axis=0) if k > 0 else np.zeros(N, dtype=float)

    a = max(0, k - int(lag_steps))
    recent = counts_hist[a:k].sum(axis=0) if k > a else np.zeros(N, dtype=float)

    lag1 = counts_hist[k - 1] if k >= 1 else np.zeros(N, dtype=float)
    lag2 = counts_hist[k - 2] if k >= 2 else np.zeros(N, dtype=float)
    lag3 = counts_hist[k - 3] if k >= 3 else np.zeros(N, dtype=float)

    neigh_cum = np.sum(neigh_w * cum_prev[neigh_idx], axis=1)
    neigh_recent = np.sum(neigh_w * recent[neigh_idx], axis=1)

    trend = 0.0 if trend_factor is None else float(trend_factor[k])
    cost = np.zeros(N, dtype=float) if cost_nodes is None else np.asarray(cost_nodes[k], float)

    cal_year = float(YEAR0) + float(times[k])

    Xk = np.column_stack([
        rho_total[k],
        cost,
        np.full(N, trend),
        cum_prev,
        recent,
        neigh_cum,
        neigh_recent,
        lag1,
        lag2,
        lag3,
        xy[:, 0],
        xy[:, 1],
        np.full(N, cal_year),
        np.full(N, times[k]),
    ])

    return Xk.astype(np.float32)


def build_ml_training_matrix(
    *,
    counts_snap: np.ndarray,
    rho_total: np.ndarray,
    cost_nodes: Optional[np.ndarray],
    trend_factor: Optional[np.ndarray],
    xy_nodes_km: np.ndarray,
    times: np.ndarray,
    YEAR0: float,
    neigh_idx: np.ndarray,
    neigh_w: np.ndarray,
    train_start_year: int,
    train_end_year: int,
    lag_steps: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    counts_snap = np.asarray(counts_snap, float)
    times = np.asarray(times, float)

    X_rows = []
    y_rows = []

    cal_years = float(YEAR0) + times

    for k, yy in enumerate(cal_years):
        if float(train_start_year) <= yy <= float(train_end_year + 1):
            Xk = build_ml_features_one_snapshot(
                k=k,
                counts_hist=counts_snap,
                rho_total=rho_total,
                cost_nodes=cost_nodes,
                trend_factor=trend_factor,
                xy_nodes_km=xy_nodes_km,
                times=times,
                YEAR0=YEAR0,
                neigh_idx=neigh_idx,
                neigh_w=neigh_w,
                lag_steps=lag_steps,
            )
            yk = np.log1p(np.maximum(counts_snap[k], 0.0)).astype(np.float32)

            X_rows.append(Xk)
            y_rows.append(yk)

    if not X_rows:
        raise ValueError("No ML training rows selected. Check train_start_year/train_end_year.")

    return np.vstack(X_rows).astype(np.float32), np.concatenate(y_rows).astype(np.float32)


def predict_ml_counts_recursive(
    *,
    model: XGBRegressor,
    counts_observed: np.ndarray,
    rho_total: np.ndarray,
    cost_nodes: Optional[np.ndarray],
    trend_factor: Optional[np.ndarray],
    xy_nodes_km: np.ndarray,
    times: np.ndarray,
    YEAR0: float,
    neigh_idx: np.ndarray,
    neigh_w: np.ndarray,
    train_end_year: int,
    lag_steps: int = 3,
    history_mode: str = "generative",
) -> np.ndarray:
    """
    Returns mu_snap over all snapshots.

    conditional:
      history is always observed counts.

    generative:
      training history is observed, but after train_end_year,
      history is recursively updated using predictions.
    """
    history_mode = str(history_mode).lower().strip()
    if history_mode not in ("conditional", "generative"):
        raise ValueError("history_mode must be 'conditional' or 'generative'.")

    counts_observed = np.asarray(counts_observed, float)
    nt, N = counts_observed.shape

    counts_hist = np.zeros_like(counts_observed, dtype=float)
    mu_snap = np.zeros_like(counts_observed, dtype=float)

    cal_years = float(YEAR0) + np.asarray(times, float)

    for k, yy in enumerate(cal_years):
        in_training_history = yy <= float(train_end_year + 1)

        if history_mode == "conditional" or in_training_history:
            # observed history is available before prediction
            counts_hist[k] = counts_observed[k]

        Xk = build_ml_features_one_snapshot(
            k=k,
            counts_hist=counts_hist,
            rho_total=rho_total,
            cost_nodes=cost_nodes,
            trend_factor=trend_factor,
            xy_nodes_km=xy_nodes_km,
            times=times,
            YEAR0=YEAR0,
            neigh_idx=neigh_idx,
            neigh_w=neigh_w,
            lag_steps=lag_steps,
        )

        pred_log = model.predict(Xk)
        pred = np.maximum(np.expm1(pred_log), 0.0)
        mu_snap[k] = pred

        if history_mode == "generative" and not in_training_history:
            # overwrite test-period history with model prediction
            counts_hist[k] = pred

    return mu_snap


# =============================================================================
# Plotting helper
# =============================================================================

def _plot_ml_monthly_totals_bass_vs_ml_vs_data(
    *,
    out_png: Path,
    mesh,
    events_df: pd.DataFrame,
    epsg_project: int,
    times: np.ndarray,
    YEAR0: float,
    mu_snap_node: np.ndarray,
    start_month: str,
    end_month: str,
    chunk_size: int = 5000,
):
    import matplotlib.pyplot as plt

    y_month, month_labels, K_total, K_inside = bin_events_month_total_inside_mesh(
        mesh=mesh,
        events_df=events_df,
        epsg_project=int(epsg_project),
        start_month=start_month,
        end_month=end_month,
        chunk_size=int(chunk_size),
    )

    month_edges_years = make_month_edges_years(
        YEAR0=float(YEAR0),
        start_month=start_month,
        end_month=end_month,
    )

    mu_total_snap = np.sum(np.asarray(mu_snap_node, float), axis=1)

    mu_month = _aggregate_snapshot_mass_to_months(
        mass_t=mu_total_snap,
        times=np.asarray(times, float),
        month_edges_years=month_edges_years,
    )

    p_hat, q_hat, M_hat, bass_month = fit_bass_to_monthly_counts(
        y_month=y_month,
        month_edges_years_rel=month_edges_years,
        p0=0.01,
        q0=0.1,
        M0=None,
    )

    x = np.array(month_labels)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(x, y_month.astype(float), label="Data inside mesh")
    ax.plot(x, bass_month.astype(float), label=f"Bass fit (p={p_hat:.3g}, q={q_hat:.3g})")
    ax.plot(x, mu_month.astype(float), label="XGBoost mean")

    ax.set_title(f"Monthly totals: Data vs Bass vs XGBoost ({start_month} to {end_month})")
    ax.set_xlabel("Month")
    ax.set_ylabel("Count per month")
    ax.grid(True, alpha=0.25)
    ax.legend()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    
    data_dir = out_png.parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    comparison_csv = data_dir / out_png.with_suffix(".csv").name
    pd.DataFrame({
        "month": [pd.Timestamp(m).strftime("%Y-%m") for m in month_labels],
        "data": np.asarray(y_month, float),
        "standard_bass": np.asarray(bass_month, float),
        "XGBoost": np.asarray(mu_month, float),
    }).to_csv(comparison_csv, index=False)
    
    
# =============================================================================
# XGBoost benchmark runner
# =============================================================================


@dataclass
class MLBenchmarkRunner:
    out_folder: str
    mesh_params: Dict[str, Any]
    time_params: Dict[str, float]

    train_start_year: int
    train_end_year: int
    test_start_year: int
    test_end_year: int

    fem_verbose: bool = False
    mesh_verbose: bool = False

    cities: Dict[str, Sequence[float]] = field(default_factory=dict)
    years_to_plot: Optional[List[int]] = None

    events_csv: Path = Path("data") / "processed" / "solar_installations_all.csv"
    events_state_col: str = "state"
    base_out: Path = Path("out")

    xgb_params: Dict[str, Any] = field(default_factory=lambda: dict(
        n_estimators=800,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=0,
        n_jobs=-1,
    ))

    neighbor_top_k: int = 50
    neighbor_theta: float = 0.05
    lag_steps: int = 3
    
    ml_history_mode: str = "generative"

    _t0: float = field(init=False, repr=False)

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
    
        region_tag = make_region_tag(
            self.mesh_params.get("state_list", []),
            county_list=self.mesh_params.get("county_list", []),
            city_list=self.mesh_params.get("city_list", []),
            digest_len=10,
        )
    
        return self.mesh_dir / f"{region_tag}__h{h_km}_s{simplify_km}_km.msh"

    def log(self, msg: str) -> None:
        dt = time.perf_counter() - self._t0
        print(f"{self.out_folder}@[{_fmt_hhmmss(dt)}] ---- {msg}")

    def build_fem_config(self) -> FEMConfig:
        tau = float(self.time_params.get("tau", 1.0))
        start_year = float(self.time_params["start_year"])
        T_years = max(0.0, float(self.test_end_year + 1) - start_year)
        epsg = int(self.mesh_params.get("epsg_project", 5070))

        return FEMConfig(
            tau_years=tau,
            T_years=T_years,
            picard_max_iter=int(self.time_params.get("picard_max_iter", 20)),
            picard_tol=float(self.time_params.get("picard_tol", 1e-8)),
            verbose=bool(self.fem_verbose),
            YEAR0=start_year,
            epsg_project=epsg,
        )

    def build_mesh(self) -> Path:
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

        admin1_shp = (
            Path("data") / "raw" / "maps"
            / "ne_10m_admin_1_states_provinces_lakes"
            / "ne_10m_admin_1_states_provinces_lakes.shp"
        )

        state_list = list(self.mesh_params["state_list"])
        county_list = list(self.mesh_params.get("county_list", []))
        city_list = list(self.mesh_params.get("city_list", []))
        county_shp = Path("data") / "raw" / "maps" / "cb_2023_us_county_5m" / "cb_2023_us_county_5m.shp"
        h_km = float(self.mesh_params["h_km"])
        simplify_km = float(self.mesh_params["simplify_km"])
        epsg = int(self.mesh_params.get("epsg_project", 5070))

        cfg = MeshBuildConfig(h_km=h_km, simplify_km=simplify_km, epsg_project=epsg)

        if msh.exists():
            self.log(f"mesh exists: {msh.name} (skipping build)")
            return msh

        build_mesh_from_admin1_region(admin1_shp, state_list, msh, cfg, verbose=bool(self.mesh_verbose), model_name=f"{self.out_folder}_mesh_km",
            county_shp=county_shp, county_names=county_list, city_names=city_list)
        return msh

    def run(self) -> Dict[str, Any]:
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        msh = self.mesh_path()
        if not msh.exists():
            raise FileNotFoundError(f"Mesh not found: {msh}. Call build_mesh() first.")

        state_list = [str(s).strip() for s in self.mesh_params["state_list"] if str(s).strip()]
        if not state_list:
            raise ValueError("mesh_params['state_list'] is empty.")
        
        st = "_".join(state_list).lower()

        fem_cfg = self.build_fem_config()

        events = load_events_csv(
            self.events_csv,
            region_states=state_list,
            YEAR0=fem_cfg.YEAR0,
            epsg_project=fem_cfg.epsg_project,
            min_year=float(self.train_start_year),
            max_year=float(self.test_end_year + 1),
        )

        score_cfg = ScoreConfig(
            t_min=0.0,
            t_max=float(fem_cfg.T_years),
            lambda_floor=float(self.time_params.get("score_lambda_floor", 1e-12)),
            verbose=False,
            normalize_by_events=True,
            finder_chunk_size=int(self.time_params.get("bin_chunk_size", 5000)),
        )

        stage_pre = precompute_stage_objects(msh, fem_cfg, events, score_cfg)

        neigh_idx, neigh_w = _topk_neighbor_precompute(
            stage_pre.fem_cache.xy_nodes_km,
            top_k=int(self.neighbor_top_k),
            theta=float(self.neighbor_theta),
        )

        X_train, y_train = build_ml_training_matrix(
            counts_snap=stage_pre.counts_node,
            rho_total=stage_pre.fem_cache.rho_total,
            cost_nodes=getattr(stage_pre.fem_cache, "cost_nodes", None),
            trend_factor=getattr(stage_pre.fem_cache, "trend_factor", None),
            xy_nodes_km=stage_pre.fem_cache.xy_nodes_km,
            times=stage_pre.fem_cache.times,
            YEAR0=fem_cfg.YEAR0,
            neigh_idx=neigh_idx,
            neigh_w=neigh_w,
            train_start_year=int(self.train_start_year),
            train_end_year=int(self.train_end_year),
            lag_steps=int(self.lag_steps),
        )
        
        self.log(
            f"[ML] X_train={X_train.shape}, y_train={y_train.shape}, "
            f"history_mode={self.ml_history_mode}"
        )
        
        model = XGBRegressor(**self.xgb_params)
        model.fit(X_train, y_train)
        
        mu_snap = predict_ml_counts_recursive(
            model=model,
            counts_observed=stage_pre.counts_node,
            rho_total=stage_pre.fem_cache.rho_total,
            cost_nodes=getattr(stage_pre.fem_cache, "cost_nodes", None),
            trend_factor=getattr(stage_pre.fem_cache, "trend_factor", None),
            xy_nodes_km=stage_pre.fem_cache.xy_nodes_km,
            times=stage_pre.fem_cache.times,
            YEAR0=fem_cfg.YEAR0,
            neigh_idx=neigh_idx,
            neigh_w=neigh_w,
            train_end_year=int(self.train_end_year),
            lag_steps=int(self.lag_steps),
            history_mode=str(self.ml_history_mode),
        )
        
        start_month = f"{int(self.test_start_year):04d}-01"
        end_month = f"{int(self.test_end_year):04d}-12"
        
        monthly_png = self.fig_dir / f"{st.lower()}_ml_monthly_totals_bass_vs_ml_vs_data.png"
        
        _plot_ml_monthly_totals_bass_vs_ml_vs_data(
            out_png=monthly_png,
            mesh=stage_pre.fem_cache.mesh,
            events_df=events.raw,
            epsg_project=fem_cfg.epsg_project,
            times=stage_pre.fem_cache.times,
            YEAR0=fem_cfg.YEAR0,
            mu_snap_node=mu_snap,
            start_month=start_month,
            end_month=end_month,
            chunk_size=int(self.time_params.get("bin_chunk_size", 5000)),
        )
        
        self.log("_plot_ml_monthly_totals_bass_vs_ml_vs_data complete")

        counts_node, K_total, K_inside, year_labels, min_ts, max_ts = bin_events_year_node(
            mesh=stage_pre.fem_cache.mesh,
            events_df=events.raw,
            epsg_project=fem_cfg.epsg_project,
            t_min_year=self.test_start_year,
            t_max_year=self.test_end_year,
            chunk_size=int(self.time_params.get("bin_chunk_size", 5000)),
        )

        mu_node = aggregate_snapshot_node_to_year_node(
            mu_snap,
            stage_pre.fem_cache.times,
            fem_cfg.YEAR0,
            t_min_year=self.test_start_year,
            t_max_year=self.test_end_year,
        )

        D_total, _ = poisson_deviance(counts_node, mu_node)
        R = pearson_residuals(counts_node, mu_node, mu_floor=float(self.time_params.get("mu_floor", 1e-12)))
        R_rms = float(np.sqrt(np.mean(R ** 2)))

        metrics = forecast_error_metrics(
            counts_node,
            mu_node,
            eps=float(self.time_params.get("metric_eps", 1e-12)),
            min_denom=float(self.time_params.get("mape_min_denom", 1.0)),
        )

        self.log(
            "[ML metrics] "
            f"deviance={D_total:.6e} "
            f"MAE={metrics['mae']:.6g} "
            f"RMSE={metrics['rmse']:.6g} "
            f"SMAPE={metrics['smape']:.6g} "
            f"log1p_MAE={metrics['log1p_mae']:.6g} "
            f"log1p_RMSE={metrics['log1p_rmse']:.6g} "
            f"total_rel_err={metrics['total_relative_error']:.6g}"
        )

        h_km = float(self.mesh_params["h_km"])
        years_to_plot = self.years_to_plot or list(range(self.test_start_year, self.test_end_year + 1))

        for yy in years_to_plot:
            if yy < self.test_start_year or yy > self.test_end_year:
                continue

            idx = yy - self.test_start_year

            out_png = self.fig_dir / f"{st.lower()}_ml_data_vs_mu_nodes_{yy}.png"
            _plot_data_vs_mu_year_nodes_lonlat(
                msh_path=msh,
                epsg_project=fem_cfg.epsg_project,
                out_png=out_png,
                y_node=counts_node[idx],
                mu_node=mu_node[idx],
                year=int(yy),
                h_km=h_km,
                cities=self.cities,
            )

            out_png_R = self.fig_dir / f"{st.lower()}_ml_pearson_residual_nodes_{yy}.png"
            _plot_pearson_residuals_year_nodes_lonlat(
                msh_path=msh,
                epsg_project=fem_cfg.epsg_project,
                out_png=out_png_R,
                R_node=R[idx],
                year=int(yy),
                h_km=h_km,
                cities=self.cities,
                vlim_log=float(self.time_params.get("vlim_log", 2.5)),
            )

        return dict(
            out_folder=self.out_folder,
            state=st,
            mesh=str(msh),
            figures=str(self.fig_dir),
            benchmark_model="xgboost",
            train_start_year=int(self.train_start_year),
            train_end_year=int(self.train_end_year),
            test_start_year=int(self.test_start_year),
            test_end_year=int(self.test_end_year),
            K_total=int(K_total),
            K_inside=int(K_inside),
            deviance=float(D_total),
            pearson_rms=float(R_rms),
            ml_history_mode=str(self.ml_history_mode),
            **metrics,
        )


def run_ml_benchmark(
    out_folder: str,
    *,
    mesh_params: Dict[str, Any],
    time_params: Dict[str, float],
    train_start_year: int,
    train_end_year: int,
    test_start_year: int,
    test_end_year: int,
    fem_verbose: bool = False,
    mesh_verbose: bool = False,
    cities: Optional[Dict[str, Sequence[float]]] = None,
    years_to_plot: Optional[List[int]] = None,
    events_csv: Path = Path("data") / "processed" / "solar_installations_all.csv",
    events_state_col: str = "state",
    base_out: Path = Path("out"),
    xgb_params: Optional[Dict[str, Any]] = None,
    ml_history_mode: str = "generative",
    neighbor_top_k: int = 50,
    neighbor_theta: float = 0.05,
    lag_steps: int = 3,
) -> Dict[str, Any]:

    runner = MLBenchmarkRunner(
        out_folder=out_folder,
        mesh_params=mesh_params,
        time_params=time_params,
        train_start_year=int(train_start_year),
        train_end_year=int(train_end_year),
        test_start_year=int(test_start_year),
        test_end_year=int(test_end_year),
        fem_verbose=fem_verbose,
        mesh_verbose=mesh_verbose,
        cities=dict(cities or {}),
        years_to_plot=years_to_plot,
        events_csv=Path(events_csv),
        events_state_col=str(events_state_col),
        base_out=Path(base_out),
        ml_history_mode=str(ml_history_mode),
    )

    if xgb_params:
        runner.xgb_params.update(dict(xgb_params))
        
    runner.neighbor_top_k = int(neighbor_top_k)
    runner.neighbor_theta = float(neighbor_theta)
    runner.lag_steps = int(lag_steps)

    runner.build_mesh()
    return runner.run()