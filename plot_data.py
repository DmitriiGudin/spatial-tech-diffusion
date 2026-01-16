#!/usr/bin/env python3
"""
Diagnostics and plots for solar + population dataset.

Inputs:
- data/processed/solar_installations_all.csv
    Columns (at least):
      date, ZIP, latitude, longitude,
      population_2010, population_2020, population_estimated,
      customer type, price, city, state, county

- data/processed/zip_population_all.csv
    Columns:
      ZIP, latitude, longitude,
      population_2010, population_2020,
      city, state, county

- data/raw/maps/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp

Outputs (in data/figures):
- population_heatmap_<year>.png for years 1995, 2000, 2005, 2010, 2015, 2020, 2025
- pv_cumulative_heatmap.gif
- pv_cumulative_heatmap_final.png
- installations_hist.png
- price_hist_2025.png

Also prints diagnostics, including extrapolated total populations.
"""

from pathlib import Path
import warnings

warnings.filterwarnings("ignore", module="cpi.*")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import geopandas as gpd
import imageio.v2 as imageio

import cpi  # for CPI-based price adjustment

# Try to update CPI; if it fails (offline), we silently fall back
try:
    cpi.update()
except Exception:
    pass

from density_utils import get_density

# ----------------------------------
# Paths and constants
# ----------------------------------

BASE = Path("data")
SOLAR_CSV = BASE / "processed" / "solar_installations_all.csv"
ZIP_POP_CSV = BASE / "processed" / "zip_population_all.csv"
MAP_SHP = BASE / "raw" / "maps" / "ne_110m_admin_0_countries" / "ne_110m_admin_0_countries.shp"

FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET_DOLLAR_YEAR = 2025

# Continental US bounding box (roughly)
LON_MIN, LON_MAX = -125, -66
LAT_MIN, LAT_MAX = 24, 50

# ----------------------------------
# Loading data
# ----------------------------------


def load_solar_data() -> pd.DataFrame:
    """Load processed solar installations."""
    df = pd.read_csv(SOLAR_CSV, parse_dates=["date"])

    # Basic clean-up
    df = df.dropna(
        subset=[
            "date",
            "ZIP",
            "latitude",
            "longitude",
            "price",
            "state",
        ]
    )

    # Coerce numeric
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df.dropna(subset=["latitude", "longitude", "price"])

    # Remove AK / HI rows if somehow present
    df = df[~df["state"].isin(["AK", "HI"])]

    return df


def load_zip_population() -> pd.DataFrame:
    """Load ZIP-level population master, including ZIPs with no installations."""
    df = pd.read_csv(ZIP_POP_CSV)

    # Ensure types
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["population_2010"] = pd.to_numeric(df["population_2010"], errors="coerce")
    df["population_2020"] = pd.to_numeric(df["population_2020"], errors="coerce")

    df = df.dropna(subset=["ZIP", "latitude", "longitude"])

    # Keep AK/HI in the data for population totals, but maps will be clipped
    return df


def load_us_border():
    """Load country shapefile and take USA border."""
    world = gpd.read_file(MAP_SHP)
    usa = world[world["ADMIN"] == "United States of America"].to_crs("EPSG:4326")
    return usa


# ----------------------------------
# Exponential population extrapolation
# ----------------------------------


def estimate_population_exp(row: pd.Series, year: float) -> float:
    """
    Exponential interpolation/extrapolation between 2010 and 2020.
    p(t) = p10 * exp(r * (t - 2010)), where r = (1/10)*log(p20/p10).
    Fallback to constant or linear when needed.
    Result >= 0 and rounded to nearest integer.
    """
    p10 = row.get("population_2010", np.nan)
    p20 = row.get("population_2020", np.nan)

    if pd.isna(p10) and pd.isna(p20):
        return np.nan

    # Only one point known -> treat as constant
    if pd.isna(p10):
        pred = float(p20)
    elif pd.isna(p20):
        pred = float(p10)
    else:
        p10 = float(p10)
        p20 = float(p20)
        if p10 > 0 and p20 > 0 and p10 != p20:
            r = np.log(p20 / p10) / 10.0
            pred = p10 * np.exp(r * (year - 2010.0))
        else:
            # either no growth or one is zero; treat as constant
            pred = float(p10)

    pred = max(pred, 0.0)
    return float(np.round(pred))


# ----------------------------------
# Diagnostics
# ----------------------------------


def print_diagnostics(solar_df: pd.DataFrame, zip_pop_df: pd.DataFrame):
    """Print requested diagnostic info, including extrapolated total populations."""
    # Number of ZIPs with ≥1 installation
    num_zip_with_install = solar_df["ZIP"].nunique()

    # Total number of ZIPs in zip_population_all.csv
    total_zips = zip_pop_df["ZIP"].nunique()

    # Residential subset
    res_mask = solar_df["customer type"].isin(["RES", "RES_SF", "RES_MF"])
    df_res = solar_df[res_mask].copy()

    total_res_installs = len(df_res)
    first_date = df_res["date"].min()
    last_date = df_res["date"].max()

    # Total ZIP-based populations for 2010, 2020 (direct)
    total_2010 = np.nansum(zip_pop_df["population_2010"].values)
    total_2020 = np.nansum(zip_pop_df["population_2020"].values)

    # Extrapolated totals for 1995, 2000, 2005, 2015, 2025
    years_extrap = [1995, 2000, 2005, 2015, 2025]
    totals_extrap = {}
    for y in years_extrap:
        pops_y = zip_pop_df.apply(lambda row: estimate_population_exp(row, y), axis=1)
        totals_extrap[y] = np.nansum(pops_y.values)

    print("===== Diagnostics =====")
    print(f"Number of ZIP codes with ≥1 installation: {num_zip_with_install}")
    print(f"Total number of ZIP codes in population file: {total_zips}")
    print(
        f"Total residential installations: {total_res_installs} "
        f"(from {first_date.date()} to {last_date.date()})"
    )

    print()
    print(f"Total US population 1995 (extrapolated): {totals_extrap[1995]:,.0f}")
    print(f"Total US population 2000 (extrapolated): {totals_extrap[2000]:,.0f}")
    print(f"Total US population 2005 (extrapolated): {totals_extrap[2005]:,.0f}")
    print(f"Total US population 2010 (ZIP-based): {total_2010:,.0f}")
    print(f"Total US population 2015 (extrapolated): {totals_extrap[2015]:,.0f}")
    print(f"Total US population 2020 (ZIP-based): {total_2020:,.0f}")
    print(f"Total US population 2025 (extrapolated): {totals_extrap[2025]:,.0f}")
    print("=======================\n")


# ----------------------------------
# 1. Population heatmaps (multiple years)
# ----------------------------------


def plot_population_heatmaps(zip_pop_df: pd.DataFrame, usa_border):
    """
    Plot population heatmaps for multiple years:
    1995, 2000, 2005, 2010, 2015, 2020, 2025.

    Derived purely from zip_population_all.csv, including ZIPs with no solar.
    Alaska/Hawaii are clipped out by axis limits.
    """
    years = [1995, 2000, 2005, 2010, 2015, 2020, 2025]

    # We use all ZIPs, but clip by lat/lon for display
    df = zip_pop_df.copy()
    # Optional: drop absurd coords
    df = df[
        (df["longitude"].between(LON_MIN - 5, LON_MAX + 5))
        & (df["latitude"].between(LAT_MIN - 5, LAT_MAX + 5))
    ]

    lons = df["longitude"].values
    lats = df["latitude"].values

    for year in years:
        if year == 2010:
            pops = df["population_2010"].values
        elif year == 2020:
            pops = df["population_2020"].values
        else:
            pops = df.apply(lambda row: estimate_population_exp(row, year), axis=1).values

        # Filter out non-positive
        mask = np.array(pops) > 0
        lons_y = lons[mask]
        lats_y = lats[mask]
        pops_y = np.array(pops)[mask]

        if len(pops_y) == 0:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        hb = ax.hexbin(
            lons_y,
            lats_y,
            C=pops_y,
            reduce_C_function=np.sum,
            gridsize=150,
            norm=LogNorm(),
            cmap="viridis",
            extent=(LON_MIN, LON_MAX, LAT_MIN, LAT_MAX),
        )

        # Mainland only via axis limits
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)

        # Plot US border (thicker, darker)
        usa_border.boundary.plot(ax=ax, linewidth=1.2, color="black")

        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("Population (log scale)")

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"US Population Heatmap (ZIP-based, log scale), year {year}")

        out_path = FIG_DIR / f"population_heatmap_{year}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved population heatmap for {year} to {out_path}")


# ----------------------------------
# 2. Cumulative residential PV heatmap animation
# ----------------------------------


def make_pv_cumulative_animation(solar_df: pd.DataFrame, usa_border):
    """
    GIF: one frame per year; each frame shows cumulative residential installations
    up to that year. Logarithmic hexbin, shared color scale, thick/dark border.

    Also saves a final static PNG for the last year.
    """
    res_mask = solar_df["customer type"].isin(["RES", "RES_SF", "RES_MF"])
    df_res = solar_df[res_mask].copy()

    df_res["year"] = df_res["date"].dt.year
    years = np.sort(df_res["year"].unique())

    if len(years) == 0:
        print("No residential installations found; skipping PV animation.")
        return

    # Precompute cumulative data once and global vmax for hexbin
    cumulative_df = pd.DataFrame(columns=df_res.columns)
    all_frames_data = []

    for y in years:
        cumulative_df = pd.concat(
            [cumulative_df, df_res[df_res["year"] == y]], axis=0
        )
        all_frames_data.append((y, cumulative_df.copy()))

    # Compute global vmax using final cumulative data
    final_cum = all_frames_data[-1][1]
    lons_all = final_cum["longitude"].values
    lats_all = final_cum["latitude"].values

    fig_tmp, ax_tmp = plt.subplots()
    hb_tmp = ax_tmp.hexbin(
        lons_all,
        lats_all,
        gridsize=150,
        mincnt=1,
        extent=(LON_MIN, LON_MAX, LAT_MIN, LAT_MAX),
    )
    vmax = hb_tmp.get_array().max()
    plt.close(fig_tmp)

    frames = []
    fig, ax = plt.subplots(figsize=(10, 6))

    for (y, cum_df_y) in all_frames_data:
        lons = cum_df_y["longitude"].values
        lats = cum_df_y["latitude"].values

        ax.clear()

        hb = ax.hexbin(
            lons,
            lats,
            C=None,
            gridsize=150,
            mincnt=1,
            norm=LogNorm(vmin=1, vmax=vmax),
            cmap="inferno",
            extent=(LON_MIN, LON_MAX, LAT_MIN, LAT_MAX),
        )

        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)

        # Thick, dark border
        usa_border.boundary.plot(ax=ax, linewidth=1.2, color="black")

        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("Cumulative installations (log scale)")

        ax.set_title(f"Cumulative Residential PV Installations up to {y}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        fig.tight_layout()

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        # Drop alpha for GIF (RGB only)
        image = buf[..., :3].copy()
        frames.append(image)

        cb.remove()

    plt.close(fig)

    out_gif = FIG_DIR / "pv_cumulative_heatmap.gif"
    imageio.mimsave(out_gif, frames, fps=1)
    print(f"Saved PV cumulative heatmap animation to {out_gif}")

    # Final static PNG for the last year
    last_year, final_cum = all_frames_data[-1]
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    lons_final = final_cum["longitude"].values
    lats_final = final_cum["latitude"].values

    hb_final = ax2.hexbin(
        lons_final,
        lats_final,
        C=None,
        gridsize=150,
        mincnt=1,
        norm=LogNorm(vmin=1, vmax=vmax),
        cmap="inferno",
        extent=(LON_MIN, LON_MAX, LAT_MIN, LAT_MAX),
    )

    ax2.set_xlim(LON_MIN, LON_MAX)
    ax2.set_ylim(LAT_MIN, LAT_MAX)

    usa_border.boundary.plot(ax=ax2, linewidth=1.2, color="black")

    cb2 = fig2.colorbar(hb_final, ax=ax2)
    cb2.set_label("Cumulative installations (log scale)")

    ax2.set_title(f"Cumulative Residential PV Installations up to {last_year}")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")

    fig2.tight_layout()
    out_png = FIG_DIR / "pv_cumulative_heatmap_final.png"
    fig2.savefig(out_png, dpi=200)
    plt.close(fig2)
    print(f"Saved final cumulative PV heatmap to {out_png}")


# ----------------------------------
# 3. Histogram of installation counts over time
# ----------------------------------


def plot_installations_hist(solar_df: pd.DataFrame):
    """Histogram of number of installations over time (by year, all customer types)."""
    solar_df["year"] = solar_df["date"].dt.year
    counts = solar_df.groupby("year")["ZIP"].size()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(counts.index, counts.values, width=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of installations")
    ax.set_title("PV Installations per Year (all customer types)")

    out_path = FIG_DIR / "installations_hist.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved installations histogram to {out_path}")


# ----------------------------------
# 4. Histogram of prices in 2025 dollars
# ----------------------------------


def compute_year_inflation_factors(years, target_year=TARGET_DOLLAR_YEAR):
    """
    Pre-compute CPI inflation factors per calendar year.
    Approximate via July 1 of each year to July 1 of target_year.
    """
    factors = {}
    for y in years:
        try:
            base_date = f"{int(y)}-07-01"
            target_date = f"{int(target_year)}-07-01"
            factors[y] = cpi.inflate(1.0, base_date, to=target_date)
        except Exception:
            factors[y] = 1.0
    return factors


def plot_price_histogram_2025(solar_df: pd.DataFrame):
    """Histogram of system prices in 2025 dollars (vectorized CPI adjustment)."""
    df_price = solar_df[solar_df["price"] > 0].copy()
    df_price["year"] = df_price["date"].dt.year

    unique_years = df_price["year"].unique()
    year_factors = compute_year_inflation_factors(unique_years, TARGET_DOLLAR_YEAR)

    df_price["inflation_factor"] = df_price["year"].map(year_factors)
    df_price["price_2025"] = df_price["price"] * df_price["inflation_factor"]

    prices = df_price["price_2025"].values

    upper = np.percentile(prices, 99)
    prices_clipped = np.clip(prices, 0, upper)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(prices_clipped, bins=100)
    ax.set_xlabel(
        f"Installation price (in {TARGET_DOLLAR_YEAR} dollars, clipped at 99th percentile)"
    )
    ax.set_ylabel("Count")
    ax.set_title(f"Histogram of Installation Prices ({TARGET_DOLLAR_YEAR} dollars)")

    out_path = FIG_DIR / "price_hist_2025.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved price histogram (2025 dollars) to {out_path}")
    
    
def plot_density_heatmaps_2025(usa_border):
    """
    Plot population density heatmaps (persons/km^2) for year=2025 using get_density(),
    for multiple radii. Mainland only (via axis limits).
    """
    radii = [5, 10, 20, 50]  # km

    # Grid resolution: keep this modest; runtime scales with (#grid points)*(#radii)
    nx, ny = 1000, 600
    lons = np.linspace(LON_MIN, LON_MAX, nx)
    lats = np.linspace(LAT_MIN, LAT_MAX, ny)

    # Prebuild meshgrid for plotting
    LON, LAT = np.meshgrid(lons, lats)

    for r in radii:
        print(f"Computing density grid for year=2025, radius={r} km ...")

        dens = np.empty((ny, nx), dtype=float)

        # Row-wise loop keeps memory low and is usually fast enough.
        for j in range(ny):
            lat = float(lats[j])
            for i in range(nx):
                lon = float(lons[i])
                dens[j, i] = get_density(lon=lon, lat=lat, year=2025.0, radius_km=float(r))

        # Avoid zeros for LogNorm display (keep true zeros in the array if you prefer)
        dens_plot = dens.copy()
        positive = dens_plot[dens_plot > 0]
        if positive.size == 0:
            print(f"No positive density values for radius={r} km; skipping.")
            continue
        vmin = np.percentile(positive, 1)  # robust lower bound
        vmax = np.percentile(positive, 99) # robust upper bound
        dens_plot[dens_plot <= 0] = np.nan

        fig, ax = plt.subplots(figsize=(10, 6))

        im = ax.imshow(
            dens_plot,
            origin="lower",
            extent=(LON_MIN, LON_MAX, LAT_MIN, LAT_MAX),
            norm=LogNorm(vmin=max(vmin, 1e-12), vmax=vmax),
            aspect="auto",
            cmap="viridis",
        )

        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)

        usa_border.boundary.plot(ax=ax, linewidth=1.5, color="black")  # thicker/darker

        cb = fig.colorbar(im, ax=ax)
        cb.set_label("Estimated density (persons / km², log scale)")

        ax.set_title(f"Estimated Population Density (year 2025), radius={r:g} km")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        out_path = FIG_DIR / f"density_heatmap_2025_radius_{int(r)}km.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        print(f"Saved density heatmap to {out_path}")


# ----------------------------------
# Main
# ----------------------------------


def main():
    print("Loading solar installations...")
    solar_df = load_solar_data()

    print("Loading ZIP population master...")
    zip_pop_df = load_zip_population()

    print("Loading US border shapefile...")
    usa_border = load_us_border()

    print_diagnostics(solar_df, zip_pop_df)

    '''print("Plotting population heatmaps for multiple years...")
    plot_population_heatmaps(zip_pop_df, usa_border)

    print("Making cumulative residential PV animation...")
    make_pv_cumulative_animation(solar_df, usa_border)

    print("Plotting installations histogram...")
    plot_installations_hist(solar_df)

    print("Plotting price histogram in 2025 dollars...")
    plot_price_histogram_2025(solar_df)'''
    
    print("Plotting density heatmaps (get_density) for 2025...")
    plot_density_heatmaps_2025(usa_border)


    print("All figures generated.")


if __name__ == "__main__":
    main()