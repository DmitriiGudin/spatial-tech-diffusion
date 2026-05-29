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
    make_region_tag,
    plot_adoptions_years,
    plot_mesh_population_nodes,
    summarize_counties_for_region,
    load_mesh_km_from_msh
)


def _parse_list_arg(s: str, *, upper: bool = False) -> List[str]:
    raw = str(s or "").strip()
    if not raw:
        return []

    if raw.startswith("[") and raw.endswith("]"):
        obj = ast.literal_eval(raw)
        if not isinstance(obj, (list, tuple)):
            raise ValueError(f"List argument must parse to list/tuple, got {type(obj)}")
        out = [str(x).strip() for x in obj if str(x).strip()]
    elif "," in raw:
        out = [t.strip() for t in raw.split(",") if t.strip()]
    else:
        out = [raw]

    if upper:
        out = [x.upper() for x in out]
    return out


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
    ap.add_argument("--counties", type=str, default="", help="County name or list, scoped by --states.")
    ap.add_argument("--cities", type=str, default="", help="City name or list. Recognized but not implemented yet.")
    args = ap.parse_args()

    states = _parse_list_arg(args.states, upper=True)
    counties = _parse_list_arg(args.counties, upper=False)
    cities = _parse_list_arg(args.cities, upper=False)
    h_km = float(args.h_km)
    simplify_km = float(args.simplify_km)
    epsg_project = int(args.epsg_project)
    no_plots = bool(args.no_plots)

    # --- Paths ---
    base = Path("data")
    admin1_shp = base / "raw" / "maps" / "ne_10m_admin_1_states_provinces_lakes" / "ne_10m_admin_1_states_provinces_lakes.shp"
    county_shp = base / "raw" / "maps" / "cb_2023_us_county_5m" / "cb_2023_us_county_5m.shp"
    out_root = Path("out") / "mesh_diag"
    out_dir_mesh = out_root / "mesh"
    out_dir_fig = out_root / "figures"
    out_dir_data = out_root / "data"
    out_dir_mesh.mkdir(parents=True, exist_ok=True)
    out_dir_fig.mkdir(parents=True, exist_ok=True)

    # Name outputs using states + parameters
    tag_region = make_region_tag(states, county_list=counties, city_list=cities, digest_len=10)
    
    tag = f"{tag_region}_h{h_km:g}_s{simplify_km:g}_epsg{epsg_project}"
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
        model_name=f"{tag_region}_mesh_km",
        county_shp=county_shp,
        county_names=counties,
        city_names=cities,
    )

    mesh = load_mesh_km_from_msh(msh_path)

    # --- Mesh quality ---
    print_mesh_quality_diagnostics(mesh, label=f"{tag_region} (h={h_km:g} km, simplify={simplify_km:g} km)")

    # --- Population mass check (no plots) ---
    print_population_mass_check(
        admin1_shp=admin1_shp,
        state_codes=states,
        msh_path=msh_path,
        year=float(args.year_pop),
        epsg_project=epsg_project,
        mesh=mesh,
        county_shp=county_shp,
        county_names=counties,
        city_names=cities,
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
        county_shp=county_shp,
        county_names=counties,
        city_names=cities,
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
    
    # Population-only mesh plot.
    png_pop_nodes = out_dir_fig / f"{tag}_population_nodes_2023.png"
    plot_mesh_population_nodes(
        msh_path=msh_path,
        out_png=png_pop_nodes,
        year=2023.0,
        epsg_project=epsg_project,
        h_km=h_km,
        cities=cities if "cities" in locals() else {},
    )
    print(f"[RUN] saved population node plot: {png_pop_nodes}")
    
    # Yearly adoption node plots.
    adopt_year_dir = out_dir_fig / f"{tag}_adoptions_by_year"
    plot_adoptions_years(
        msh_path=msh_path,
        events_df=events_sel,
        out_dir=adopt_year_dir,
        epsg_project=epsg_project,
        start_year=int(args.start_year_events),
        end_year=int(args.end_year_events),
        h_km=h_km,
        cities=cities if "cities" in locals() else {},
    )
    print(f"[RUN] saved yearly adoption plots to: {adopt_year_dir}")
    
    zip_population_csv = base / "processed" / "zip_population_all.csv"
    county_summary_csv = out_dir_data / f"{tag}_county_summary.csv"
    
    county_summary = summarize_counties_for_region(
        county_shp=county_shp,
        state_codes=states,
        county_names=counties,
        events_df=events_sel,
        zip_population_csv=zip_population_csv,
        start_year=int(args.start_year_events),
        end_year=int(args.end_year_events),
        epsg_project=epsg_project,
        out_csv=county_summary_csv,
    )
    
    print("[RUN] county summary:")
    print(county_summary.to_string(index=False))
    print(f"[RUN] saved county summary: {county_summary_csv}")
    
    # Adoptions vs inflation-adjusted costs (monthly CPI, base 2025)
    png_costs = out_dir_fig / f"{tag}_adoptions_vs_costs_cpi2025.png"
    plot_adoptions_vs_costs(
        events_df=events_sel,
        out_png=png_costs,
        msh_path=msh_path,
        epsg_project=epsg_project,
        chunk_size=50_000,
        start_date=None,
        end_date=None,
        price_col="price",
        missing_price_value=-1.0,
        base_year=2025,
        base_month=12,
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
