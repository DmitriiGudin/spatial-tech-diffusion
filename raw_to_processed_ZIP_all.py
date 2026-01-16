#!/usr/bin/env python3
"""
Build a ZIP-level dataset with:
ZIP, latitude, longitude, population_2010, population_2020, city, state, county

Population comes from:
- 2010: DECENNIALSF12010.P1-Data.csv  (NAME contains 'ZCTA5 XXXXX', pop in P001001)
- 2020: DECENNIALDHC2020.P1-Data.csv (NAME contains 'ZCTA5 XXXXX', pop in P1_001N)

Notes:
- Skip first non-header row in 2010/2020 files.
- ZIP is last 5 characters of NAME column.
- Remove Hawaii (HI) and Alaska (AK) from output.
- Drop rows containing any NaN values.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ----------------------------
# Helpers
# ----------------------------

def normalize_zip_from_name(series: pd.Series) -> pd.Series:
    """
    Extract the last 5 characters from NAME like 'ZCTA5 00612' -> '00612'.
    Returns NaN if pattern not found.
    """
    s = series.astype(str)
    return s.str[-5:].where(s.str[-5:].str.match(r"\d{5}"))


def normalize_zip_series(s: pd.Series) -> pd.Series:
    """
    Normalize arbitrary ZIP-containing columns to 5-digit ZIP.
    """
    s = s.astype(str).str.strip()
    return s.str.extract(r"(\d{5})", expand=False)


def load_zip_master(path_zip: str) -> pd.DataFrame:
    """
    Load ZIP → lat/lon + city/state/county reference table.
    """
    df = pd.read_csv(path_zip, dtype={"postal code": "string"})

    df["ZIP"] = normalize_zip_series(df["postal code"])
    df = df.dropna(subset=["ZIP"])

    df = df.drop_duplicates(subset=["ZIP"])

    df_master = df[
        ["ZIP", "latitude", "longitude", "place name", "admin code1", "admin name2"]
    ].rename(
        columns={
            "place name": "city",
            "admin code1": "state",
            "admin name2": "county",
        }
    )

    return df_master


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

    df["population_2010"] = pd.to_numeric(df["P001001"], errors="coerce").fillna(0).astype(int)

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

    df["population_2020"] = pd.to_numeric(df["P1_001N"], errors="coerce").fillna(0).astype(int)

    df = df[["ZIP", "population_2020"]]
    df = df.groupby("ZIP", as_index=False).sum()

    return df


# ----------------------------
# Main
# ----------------------------

def main():
    base = Path("data")

    path_zip = base / "raw" / "zip" / "USZipsWithLatLon_20231227.csv"
    path_pop2010 = base / "raw" / "population" / "DECENNIALSF12010.P1-Data.csv"
    path_pop2020 = base / "raw" / "population" / "DECENNIALDHC2020.P1-Data.csv"

    out_dir = base / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "zip_population_all.csv"

    print("Loading ZIP master (lat/lon + city/state/county)...")
    zip_master = load_zip_master(str(path_zip))

    print("Loading population data for 2010...")
    pop2010 = load_population_2010(str(path_pop2010))

    print("Loading population data for 2020...")
    pop2020 = load_population_2020(str(path_pop2020))

    print("Merging population data into ZIP master...")
    df = zip_master.merge(pop2010, on="ZIP", how="left")
    df = df.merge(pop2020, on="ZIP", how="left")

    # Drop ZIPs with missing population or coordinates
    df_clean = df.dropna(
        subset=["latitude", "longitude", "population_2010", "population_2020"]
    ).copy()

    # Remove Alaska & Hawaii
    df_clean = df_clean[~df_clean["state"].isin(["AK", "HI"])]

    # Convert to appropriate types
    df_clean["population_2010"] = df_clean["population_2010"].astype(int)
    df_clean["population_2020"] = df_clean["population_2020"].astype(int)

    # Sort by ZIP for readability
    df_clean = df_clean.sort_values("ZIP").reset_index(drop=True)

    print(f"Writing clean ZIP dataset to {out_path} ...")
    df_clean.to_csv(out_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()