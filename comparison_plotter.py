#!/usr/bin/env python3
# comparison_plotter.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_COLS = {"month", "data", "standard_bass"}


def _read_model_csv(path: Path) -> tuple[str, pd.DataFrame]:
    df = pd.read_csv(path)

    missing = BASE_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    model_cols = [c for c in df.columns if c not in BASE_COLS]
    if len(model_cols) != 1:
        raise ValueError(
            f"{path} must contain exactly one model column besides "
            f"{sorted(BASE_COLS)}; found {model_cols}"
        )

    model_name = model_cols[0]

    df = df[["month", "data", "standard_bass", model_name]].copy()
    df["month"] = pd.to_datetime(df["month"].astype(str) + "-01", errors="raise")
    df = df.sort_values("month")

    for c in ["data", "standard_bass", model_name]:
        df[c] = pd.to_numeric(df[c], errors="raise").astype(float)

    return model_name, df


def plot_totals(paths: List[Path], out_png: Path | None = None) -> Path:
    if not paths:
        raise ValueError("At least one CSV path is required.")
    if len(paths) > 4:
        raise ValueError("At most 4 model CSV paths are allowed.")

    model_frames: Dict[str, pd.DataFrame] = {}

    for p in paths:
        model_name, df = _read_model_csv(Path(p))
        if model_name in model_frames:
            raise ValueError(f"Duplicate model column found: {model_name!r}")
        model_frames[model_name] = df

    # Use first file as reference for data and standard Bass.
    first_model = next(iter(model_frames))
    base = model_frames[first_model][["month", "data", "standard_bass"]].copy()

    merged = base.copy()

    for model_name, df in model_frames.items():
        merged = merged.merge(
            df[["month", model_name]],
            on="month",
            how="outer",
            validate="one_to_one",
        )

    merged = merged.sort_values("month").reset_index(drop=True)

    # Fill missing model months with 0 only if necessary.
    # Usually all supplied files should have the same month range.
    for c in merged.columns:
        if c != "month":
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)

    fig, ax = plt.subplots(figsize=(13, 6))

    x = merged["month"]

    ax.plot(x, merged["data"], label="Data inside mesh")
    ax.plot(x, merged["standard_bass"], label="Standard Bass fit")
    
    for model_name in model_frames:
        ax.plot(x, merged[model_name], label=model_name)
    
    ax.set_title("Monthly adoptions: data, Bass, and model comparisons")
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly count")
    ax.grid(True, alpha=0.25)
    ax.legend()

    if out_png is None:
        out_png = Path(paths[0]).parent / "monthly_totals_model_comparison.png"
    else:
        out_png = Path(out_png)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    out_csv = out_png.with_suffix(".csv")
    merged.to_csv(out_csv, index=False)

    print(f"[comparison_plotter] saved plot: {out_png}")
    print(f"[comparison_plotter] saved merged CSV: {out_csv}")

    return out_png


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--func", required=True, choices=["plot_totals"])
    ap.add_argument("--paths", nargs="+", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.func == "plot_totals":
        plot_totals(
            paths=[Path(p) for p in args.paths],
            out_png=Path(args.out) if args.out else None,
        )
        return 0

    raise ValueError(f"Unknown function: {args.func}")


if __name__ == "__main__":
    raise SystemExit(main())