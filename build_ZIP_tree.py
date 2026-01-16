#!/usr/bin/env python3
"""
Precompute ZIP-level data for fast spatial density queries.

Reads:
    data/processed/zip_population_all.csv

Expected columns:
    ZIP, latitude, longitude, population_2010, population_2020, city, state, county

Writes:
    data/processed/zip_kdtree_data.npz

This .npz file contains:
    - coords_rad : (N, 2) array of [lat_rad, lon_rad] for haversine BallTree
    - coords_km  : (N, 2) array of [x_km, y_km] in a simple Earth-projected km system
    - lat        : (N,) latitudes in degrees
    - lon        : (N,) longitudes in degrees
    - pop2010    : (N,) population_2010 as float
    - pop2020    : (N,) population_2020 as float
    - zip        : (N,) ZIP codes as strings
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


EARTH_RADIUS_KM = 6371.0


def main():
    base = Path("data")
    in_path = base / "processed" / "zip_population_all.csv"
    out_path = base / "processed" / "zip_kdtree_data.npz"

    print(f"Loading ZIP population data from {in_path} ...")
    df = pd.read_csv(in_path)

    # Basic sanity checks
    required_cols = [
        "ZIP",
        "latitude",
        "longitude",
        "population_2010",
        "population_2020",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {in_path}: {missing}")

    # Drop any rows with missing lat/lon or population
    df = df.dropna(
        subset=[
            "latitude",
            "longitude",
            "population_2010",
            "population_2020",
        ]
    )

    # Ensure numeric types for populations
    df["population_2010"] = pd.to_numeric(df["population_2010"], errors="coerce")
    df["population_2020"] = pd.to_numeric(df["population_2020"], errors="coerce")
    df = df.dropna(subset=["population_2010", "population_2020"])

    # Extract arrays
    lat = df["latitude"].to_numpy(dtype=float)
    lon = df["longitude"].to_numpy(dtype=float)
    pop2010 = df["population_2010"].to_numpy(dtype=float)
    pop2020 = df["population_2020"].to_numpy(dtype=float)
    zips = df["ZIP"].astype(str).to_numpy()

    # --- Coordinates in radians for haversine BallTree ---
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    coords_rad = np.column_stack([lat_rad, lon_rad])

    # --- Coordinates in km (simple global projection) ---
    # x = R * lon_rad * cos(lat_rad)
    # y = R * lat_rad
    x_km = EARTH_RADIUS_KM * lon_rad * np.cos(lat_rad)
    y_km = EARTH_RADIUS_KM * lat_rad
    coords_km = np.column_stack([x_km, y_km])

    # Build a BallTree once here just to sanity-check everything (not saved)
    print("Building BallTree (sanity check)...")
    _ = BallTree(coords_rad, metric="haversine")
    print(f"BallTree built over {coords_rad.shape[0]} ZIP points.")

    # Save all arrays to a compressed npz
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving ZIP tree data to {out_path} ...")
    np.savez_compressed(
        out_path,
        coords_rad=coords_rad,
        coords_km=coords_km,
        lat=lat,
        lon=lon,
        pop2010=pop2010,
        pop2020=pop2020,
        zip=zips,
    )
    print("Done.")


if __name__ == "__main__":
    main()