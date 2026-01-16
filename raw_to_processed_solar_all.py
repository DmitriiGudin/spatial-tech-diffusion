#!/usr/bin/env python3
"""
Build a unified PV dataset with:
date, ZIP, latitude, longitude,
population_2010, population_2020,
customer type, price, city, state, county

Input files:
- data/raw/zip/USZipsWithLatLon_20231227.csv
    * 'postal code', 'latitude', 'longitude',
      'place name', 'admin code1', 'admin name2'

- data/raw/population/DECENNIALSF12010.P1-Data.csv
    * ZIP from 'NAME' (last 5 chars)
    * population from 'P001001'
    * skip first data row after header

- data/raw/population/DECENNIALDHC2020.P1-Data.csv
    * ZIP from 'NAME' (last 5 chars)
    * population from 'P1_001N'
    * skip first data row after header

- data/raw/pv/TTS_LBNL_public_file_29-Sep-2025_all.csv
    * 'zip_code', 'installation_date' (yyyy-mm-dd),
      'customer_segment', 'total_installed_price'

Output:
- data/processed/solar_installations_all.csv

Notes:
- We only use 2010 and 2020 populations, no interpolation/extrapolation here.
- Alaska (AK) and Hawaii (HI) are removed.
"""

from pathlib import Path
import pandas as pd


# ----------------------------
# Helpers
# ----------------------------

def normalize_zip_series(s: pd.Series) -> pd.Series:
    """
    Normalize ZIP-like columns to 5-character strings using only the first
    5 consecutive digits. Returns NaN where no 5-digit pattern is found.
    """
    s = s.astype(str).str.strip()
    s_digits = s.str.extract(r"(\d{5})", expand=False)
    return s_digits


def normalize_zip_from_name(s: pd.Series) -> pd.Series:
    """
    For census NAME fields such as 'ZCTA5 00612', take the last 5 characters.
    """
    s = s.astype(str).str.strip()
    return s.str[-5:]


def load_zip_master(path_zip: str) -> pd.DataFrame:
    """
    Load ZIP master file with lat/lon, city, state, county.
    Also filters out Alaska and Hawaii.
    """
    df = pd.read_csv(path_zip, dtype={"postal code": "string"})

    df["ZIP"] = normalize_zip_series(df["postal code"])
    df = df.dropna(subset=["ZIP"])

    # Keep only one row per ZIP; if duplicates, take the first
    df = df.drop_duplicates(subset=["ZIP"])

    zip_master = df[
        ["ZIP", "latitude", "longitude", "place name", "admin code1", "admin name2"]
    ].copy()

    zip_master = zip_master.rename(
        columns={
            "place name": "city",
            "admin code1": "state",
            "admin name2": "county",
        }
    )

    # Drop Alaska and Hawaii here
    zip_master = zip_master[~zip_master["state"].isin(["AK", "HI"])]

    return zip_master


def load_population_2010(path_2010: str) -> pd.DataFrame:
    """
    Load 2010 ZCTA population:
    - ZIP from NAME (last 5 chars)
    - population from P001001
    - skip first row after header
    """
    df = pd.read_csv(path_2010, skiprows=[1], dtype={"NAME": "string"})

    df["ZIP"] = normalize_zip_from_name(df["NAME"])
    df = df.dropna(subset=["ZIP"])

    df["population_2010"] = (
        pd.to_numeric(df["P001001"], errors="coerce")
        .fillna(0)
        .astype("int64")
    )

    df = df[["ZIP", "population_2010"]]
    df = df.groupby("ZIP", as_index=False).sum()

    return df


def load_population_2020(path_2020: str) -> pd.DataFrame:
    """
    Load 2020 ZCTA population:
    - ZIP from NAME (last 5 chars)
    - population from P1_001N
    - skip first row after header
    """
    df = pd.read_csv(path_2020, skiprows=[1], dtype={"NAME": "string"})

    df["ZIP"] = normalize_zip_from_name(df["NAME"])
    df = df.dropna(subset=["ZIP"])

    df["population_2020"] = (
        pd.to_numeric(df["P1_001N"], errors="coerce")
        .fillna(0)
        .astype("int64")
    )

    df = df[["ZIP", "population_2020"]]
    df = df.groupby("ZIP", as_index=False).sum()

    return df


def build_population_panel(path_2010: str, path_2020: str) -> pd.DataFrame:
    """
    Build wide table:
    ZIP, population_2010, population_2020
    """
    p10 = load_population_2010(path_2010)
    p20 = load_population_2020(path_2020)

    df = p10.merge(p20, on="ZIP", how="outer")
    return df


def load_pv_data(path_pv: str) -> pd.DataFrame:
    """
    Load PV installation dataset and normalize ZIP + parse date.
    """
    df = pd.read_csv(path_pv, low_memory=False, dtype={"zip_code": "string"})

    # Normalize ZIP
    df["ZIP"] = normalize_zip_series(df["zip_code"])
    df = df.dropna(subset=["ZIP"])

    # Parse dates
    df["date"] = pd.to_datetime(df["installation_date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Keep only the columns we need
    df = df[
        [
            "date",
            "ZIP",
            "customer_segment",
            "total_installed_price",
        ]
    ].copy()

    df = df.rename(
        columns={
            "customer_segment": "customer type",  # requested column name
            "total_installed_price": "price",
        }
    )

    return df


# ----------------------------
# Main
# ----------------------------

def main():
    base = Path("data")

    # Input paths
    path_zip = base / "raw" / "zip" / "USZipsWithLatLon_20231227.csv"
    path_pop2010 = base / "raw" / "population" / "DECENNIALSF12010.P1-Data.csv"
    path_pop2020 = base / "raw" / "population" / "DECENNIALDHC2020.P1-Data.csv"
    path_pv = base / "raw" / "pv" / "TTS_LBNL_public_file_29-Sep-2025_all.csv"

    # Output path
    out_dir = base / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "solar_installations_all.csv"

    print("Loading ZIP master...")
    zip_master = load_zip_master(str(path_zip))

    print("Loading 2010 + 2020 population data...")
    pop_panel = build_population_panel(str(path_pop2010), str(path_pop2020))

    print("Loading PV dataset...")
    pv = load_pv_data(str(path_pv))

    # Merge PV with ZIP master (lat, lon, city, state, county)
    print("Merging PV data with ZIP master...")
    merged = pv.merge(zip_master, on="ZIP", how="left")

    # Remove PV rows that are in AK/HI or with missing state (just to be safe)
    merged = merged[~merged["state"].isin(["AK", "HI"])]
    merged = merged.dropna(subset=["latitude", "longitude", "state"])

    # Merge with population panel
    print("Merging PV data with population panel...")
    merged = merged.merge(pop_panel, on="ZIP", how="left")

    # Ensure populations are integer-like (nullable Int64, so NaNs are allowed)
    for col in ["population_2010", "population_2020"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("Int64")

    # Final columns:
    # date, ZIP, latitude, longitude,
    # population_2010, population_2020,
    # customer type, price, city, state, county
    final_cols = [
        "date",
        "ZIP",
        "latitude",
        "longitude",
        "population_2010",
        "population_2020",
        "customer type",
        "price",
        "city",
        "state",
        "county",
    ]

    final = merged[final_cols].copy()

    # Sort by date
    final = final.sort_values("date").reset_index(drop=True)

    print(f"Writing output CSV to {out_path} ...")
    final.to_csv(out_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()