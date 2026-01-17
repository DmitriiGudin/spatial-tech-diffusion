#!/usr/bin/env python3
"""
run_mesh_diag.py

Mesh diagnostics runner.

Build a mesh for one or more states with given (h_km, simplify_km), then print:
  - mesh quality diagnostics (obtuse ratio, side/area percentiles)
  - population mass check (ZIP-sum inside polygon vs mesh nodal mass-lumped integral)

Optionally (unless --no_plots):
  - mesh triangle plot (lon/lat)
  - true-triangle vs est-node population comparison plot
  - total adoptions bimodal log-fit plot (monthly bins)
  - node adoptions vs population power-law plot

Examples:
  python run_mesh_diag.py --h_km 10 --simplify_km 30 --states CA
  python run_mesh_diag.py --h_km 5 --simplify_km 15 --states "['MD','DC','CA']" --no_plots
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import List

import pandas as pd

from mesh_utils import (
    MeshBuildConfig,
    build_mesh_from_admin1_region,
    print_mesh_quality_diagnostics,
    plot_msh_triangles_lonlat,
    print_population_mass_check,
    plot_triangle_population_comparison,
    plot_total_adoptions_bimodal_log_fit,
    plot_node_adoptions_vs_population_powerlaw,
    plot_adoptions_vs_costs,
    plot_mesh_costs,
)
from fem_utils import load_mesh_km_from_msh


def _parse_states_arg(s: str) -> List[str]:
    """
    Accept:
      --states CA
      --states "['MD','DC','CA']"
      --states '["MD","DC","CA"]'
      --states "MD,DC,CA"
    """
    raw = str(s).strip()
    if not raw:
        raise ValueError("--states cannot be empty.")

    # Case 1: looks like a Python/JSON-ish list
    if raw.startswith("[") and raw.endswith("]"):
        try:
            obj = ast.literal_eval(raw)
        except Exception as e:
            raise ValueError(f"Failed to parse --states list: {raw!r} ({e})") from e
        if not isinstance(obj, (list, tuple)):
            raise ValueError(f"--states list must parse to list/tuple, got: {type(obj)}")
        out = [str(x).strip().upper() for x in obj if str(x).strip()]
        if not out:
            raise ValueError("--states list is empty after parsing.")
        return out

    # Case 2: comma-separated
    if "," in raw:
        out = [t.strip().upper() for t in raw.split(",") if t.strip()]
        if not out:
            raise ValueError("--states comma list is empty after parsing.")
        return out

    # Case 3: single token
    return [raw.strip().upper()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h_km", type=float, required=True, help="Target mesh size in km.")
    ap.add_argument("--simplify_km", type=float, required=True, help="Polygon simplification tolerance in km.")
    ap.add_argument("--states", type=str, required=True, help="State code or list. Examples: CA  OR  \"['MD','DC']\"")
    ap.add_argument("--no_plots", action="store_true", help="Skip all plot generation (faster).")
    ap.add_argument("--epsg_project", type=int, default=5070, help="Projection EPSG (default: 5070).")
    ap.add_argument("--year_pop", type=float, default=2023.0, help="Year used for population checks (default: 2023).")
    ap.add_argument("--start_year_events", type=int, default=1998, help="Start year for adoption plots (default: 1998).")
    ap.add_argument("--end_year_events", type=int, default=2023, help="End year for adoption plots (default: 2023).")
    ap.add_argument("--bins_powerlaw", type=int, default=90, help="Bins for power-law histogram (default: 90).")
    ap.add_argument("--cutoff_ratio", type=float, default=1e-3, help="Outlier cutoff for power-law fit (default: 1e-3).")
    args = ap.parse_args()

    states = _parse_states_arg(args.states)
    h_km = float(args.h_km)
    simplify_km = float(args.simplify_km)
    epsg_project = int(args.epsg_project)
    no_plots = bool(args.no_plots)

    # --- Paths ---
    base = Path("data")
    admin1_shp = base / "raw" / "maps" / "ne_10m_admin_1_states_provinces_lakes" / "ne_10m_admin_1_states_provinces_lakes.shp"
    out_root = Path("out") / "mesh_diag"
    out_dir_mesh = out_root / "mesh"
    out_dir_fig = out_root / "figures"
    out_dir_mesh.mkdir(parents=True, exist_ok=True)
    out_dir_fig.mkdir(parents=True, exist_ok=True)

    # Name outputs using states + parameters
    tag_states = "_".join(states)
    tag = f"{tag_states}_h{h_km:g}_s{simplify_km:g}_epsg{epsg_project}"
    msh_path = out_dir_mesh / f"{tag}.msh"

    # --- Build mesh ---
    cfg = MeshBuildConfig(h_km=h_km, simplify_km=simplify_km, epsg_project=epsg_project)

    print(f"[RUN] states={states}")
    print(f"[RUN] h_km={h_km:g} simplify_km={simplify_km:g} epsg_project={epsg_project}")
    print(f"[RUN] out mesh: {msh_path}")

    build_mesh_from_admin1_region(
        admin1_shp=admin1_shp,
        state_codes=states,
        out_msh=msh_path,
        cfg=cfg,
        verbose=True,
        model_name=f"{tag_states}_mesh_km",
    )

    mesh = load_mesh_km_from_msh(msh_path)

    # --- Mesh quality ---
    print_mesh_quality_diagnostics(mesh, label=f"{tag_states} (h={h_km:g} km, simplify={simplify_km:g} km)")

    # --- Population mass check (no plots) ---
    print_population_mass_check(
        admin1_shp=admin1_shp,
        state_codes=states,
        msh_path=msh_path,
        year=float(args.year_pop),
        epsg_project=epsg_project,
        mesh=mesh,
    )

    if no_plots:
        print("[RUN] --no_plots set: skipping all plots.")
        return 0

    # --- Mesh plot ---
    png_mesh = out_dir_fig / f"{tag}_mesh.png"
    plot_msh_triangles_lonlat(msh_path, png_mesh, epsg_project=epsg_project)

    # --- True-triangle vs est-node population comparison ---
    png_pop_cmp = out_dir_fig / f"{tag}_pop_true_vs_est_year{int(args.year_pop)}.png"
    plot_triangle_population_comparison(
        admin1_shp=admin1_shp,
        state_codes=states,
        msh_path=msh_path,
        out_png=png_pop_cmp,
        year=float(args.year_pop),
        epsg_project=epsg_project,
        mesh=mesh,
        h_km=h_km,
    )

    # --- Adoption plots (optional, but usually wanted) ---
    csv_events = base / "processed" / "solar_installations_all.csv"
    if not csv_events.exists():
        print(f"[RUN] WARNING: events CSV not found at {csv_events}. Skipping adoption plots.")
        return 0

    events_df = pd.read_csv(csv_events)
    if "state" not in events_df.columns:
        print("[RUN] WARNING: events CSV has no 'state' column. Skipping adoption plots.")
        return 0

    events_df["state"] = events_df["state"].astype(str).str.strip().str.upper()
    events_sel = events_df.loc[events_df["state"].isin(states)].copy()
    if events_sel.empty:
        print(f"[RUN] WARNING: no events found for states={states} in {csv_events.name}. Skipping adoption plots.")
        return 0
    
    # Adoptions vs inflation-adjusted costs (monthly CPI, base 2025)
    png_costs = out_dir_fig / f"{tag}_adoptions_vs_costs_cpi2025.png"
    plot_adoptions_vs_costs(
        events_df=events_sel,
        out_png=png_costs,
        start_date=None,
        end_date=None,
        price_col="price",
        missing_price_value=-1.0,
        base_year=2025,
        base_month=12,  # use latest available 2025 CPI, fallback if unavailable
        cpi_cache_csv=out_dir_fig / "cpi_cache.csv",
    )
    
    # Median node costs
    png_costs = out_dir_fig / f"{tag}_node_median_costs_{int(args.year_pop)}.png"
    plot_mesh_costs(
        msh_path=msh_path,
        events_df=events_sel,
        out_png=png_costs,
        epsg_project=epsg_project,
        year=int(args.year_pop),
        h_km=h_km,
        cpi_adjust=True,
        base_year=2025,
        base_month=12,
        cpi_cache_csv=Path("out/mesh_diag/figures") / "cpi_cache.csv",
    )

    # Total adoptions bimodal (monthly) fit
    png_bimodal = out_dir_fig / f"{tag}_total_adoptions_bimodal_logfit.png"
    plot_total_adoptions_bimodal_log_fit(
        msh_path=msh_path,
        events_df=events_sel,
        out_png=png_bimodal,
        epsg_project=epsg_project,
        start_date=None,
        end_date=None,
        eps_count=1.0,
        min_months_each_side=12,  # sane default for stability
    )

    # Node adoptions vs pop power-law
    png_powerlaw = out_dir_fig / f"{tag}_node_adoptions_vs_pop_powerlaw.png"
    plot_node_adoptions_vs_population_powerlaw(
        msh_path=msh_path,
        events_df=events_sel,
        out_png=png_powerlaw,
        epsg_project=epsg_project,
        start_year=int(args.start_year_events),
        end_year=int(args.end_year_events),
        pop_year=float(args.year_pop),
        bins=int(args.bins_powerlaw),
        min_pop=1.0,
        min_adopt=1.0,
        cutoff_ratio=float(args.cutoff_ratio),
    )

    print("[RUN] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
