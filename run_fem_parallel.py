#!/usr/bin/env python3
"""
run_fem_parallel.py

Run multiple independent FEM solves (GSB) in parallel on native Windows.

- Spawns N worker processes (default: os.cpu_count()).
- Each worker runs the same Runner config but with out_folder:
    <prefix>1, <prefix>2, ..., <prefix>N
- Uses "spawn" start method (Windows-safe).
- Pre-builds the mesh once in the main process to avoid mesh build races.

By default this script runs the *same* parameter set N times (useful for stress-testing
or for future extensions). If you want different parameter sets per worker, edit
make_param_sets(...) to return N different dicts.

Optional:
    --levels (order,row)
Load model parameters from MLE output CSVs in:
    out/<out_folder>/csv/

order:
  0 -> <out_folder>_final.csv
  1 -> latest stage file (e.g. <out_folder>_stage2_top5.csv)
  2 -> second-latest stage file, etc.
row:
  0 -> first row, 1 -> second row, ...

Usage:
    python run_fem_parallel.py
    python run_fem_parallel.py --n 8
    python run_fem_parallel.py --prefix il_run --state IL --n 6
    python run_fem_parallel.py --prefix il_run --state IL --n 8 --levels '(0,0)'
    python run_fem_parallel.py --prefix il_run --state IL --n 8 --levels '(1,2)' --config CA
"""

from __future__ import annotations

import pandas as pd
import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from fem_utils import Runner


# -----------------------------
# Defaults
# -----------------------------

def make_base_runner_kwargs_from_configs_default(state_cli: str = "") -> Dict[str, Any]:
    """
    Build Runner kwargs from configs.default, optionally overriding the state_list
    if --state was provided (and default has some other state).
    """
    try:
        import configs  # local file
    except Exception as e:
        raise RuntimeError(f"Failed to import configs.py: {e}") from e

    if not hasattr(configs, "default") or not isinstance(configs.default, dict):
        raise RuntimeError("configs.py must define a dict named `default`.")

    d = dict(configs.default)

    # Runner expects model_params (not fem_model_params)
    fem_model_params = d.get("fem_model_params", None)
    if fem_model_params is None or not isinstance(fem_model_params, dict):
        raise RuntimeError("configs.default must contain a dict `fem_model_params`.")

    mesh_params = dict(d.get("mesh_params", {}))
    time_params = dict(d.get("time_params", {}))

    # If user explicitly passed --state, override state_list in defaults.
    state_cli = str(state_cli).strip()
    if state_cli:
        mesh_params["state_list"] = [state_cli]

    # Construct Runner kwargs (only what Runner cares about)
    out = dict(
        mesh_params=mesh_params,
        model_params=dict(fem_model_params),
        time_params=time_params,
        fem_verbose=bool(d.get("fem_verbose", False)),
        mesh_verbose=bool(d.get("mesh_verbose", False)),
        cities=dict(d.get("cities", {})) if isinstance(d.get("cities", {}), dict) else {},
    )

    # Optional Runner fields that your Runner supports
    for k in ["base_out", "events_csv", "events_state_col", "years_to_plot", "month_window"]:
        if k in d:
            out[k] = d[k]

    return out


def merge_dict_shallow(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge dicts shallowly: override keys replace base keys.
    For nested dicts (mesh_params/time_params), caller should merge separately.
    """
    out = dict(base)
    for k, v in override.items():
        out[k] = v
    return out


def merge_nested_dict(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(base)
    if override:
        out.update(dict(override))
    return out


# -----------------------------
# --levels parsing + CSV loading
# -----------------------------

_LEVELS_RE = re.compile(r"^\s*\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?\s*$")


def parse_levels(s: str) -> Tuple[int, int]:
    m = _LEVELS_RE.match(str(s))
    if not m:
        raise ValueError(f"Invalid --levels format: {s!r}. Use like (1,2) or 1,2.")
    a = int(m.group(1))
    b = int(m.group(2))
    if a < 0 or b < 0:
        raise ValueError(f"--levels must be nonnegative; got {a,b}.")
    return a, b


def _stage_files_for_outfolder(csv_dir: Path, out_folder: str) -> List[Tuple[int, Path]]:
    pat = re.compile(rf"^{re.escape(out_folder)}_stage(\d+)_top\d+\.csv$")
    out: List[Tuple[int, Path]] = []
    if not csv_dir.exists():
        return out
    for p in csv_dir.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if m:
            st = int(m.group(1))
            out.append((st, p))
    out.sort(key=lambda t: t[0])
    return out


def choose_param_csv(csv_dir: Path, out_folder: str, order: int) -> Path:
    if order == 0:
        p = csv_dir / f"{out_folder}_final.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing final CSV: {p}")
        return p

    stages = _stage_files_for_outfolder(csv_dir, out_folder)
    if not stages:
        raise FileNotFoundError(
            f"No stage CSVs found in {csv_dir} for {out_folder}. "
            f"Expected files like {out_folder}_stage2_top5.csv"
        )

    k = order - 1
    if k >= len(stages):
        raise ValueError(
            f"--levels order={order} requests the {order}-th preference (latest=1), "
            f"but only {len(stages)} stage file(s) exist for {out_folder}."
        )
    return stages[-1 - k][1]


def load_model_params_from_csv(csv_path: Path, row: int) -> Dict[str, float]:
    """
    Load a row of parameters from a CSV.

    Required columns are whatever your FEM Runner expects in model_params.
    We’ll pull the standard set if present.

    If your MLE CSV uses different column names, adjust `keys`.
    """
    df = pd.read_csv(csv_path)
    if df.shape[0] == 0:
        raise ValueError(f"CSV is empty: {csv_path}")
    if row < 0 or row >= df.shape[0]:
        raise ValueError(f"Row {row} out of range for {csv_path.name}: nrows={df.shape[0]}")

    s = df.iloc[int(row)]

    # Standard parameter names (as used by fem_utils.Runner.build_params / build_functions)
    keys = ["r", "p", "q_I", "gamma_J", "k_J", "D", "S0"]

    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise ValueError(
            f"CSV {csv_path.name} missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    out: Dict[str, float] = {}
    for k in keys:
        val = s[k]
        try:
            out[k] = float(val)
        except Exception as e:
            raise ValueError(f"Bad value for {k} in {csv_path.name} row {row}: {val!r} ({e})") from e

    return out


def make_param_sets_from_levels(
    n_workers: int,
    prefix: str,
    base_out: Path,
    levels: Tuple[int, int],
) -> List[Dict[str, float]]:
    order, row = levels
    out: List[Dict[str, float]] = []
    for i in range(1, n_workers + 1):
        out_folder = f"{prefix}{i}"
        csv_dir = base_out / out_folder / "csv"
        csv_path = choose_param_csv(csv_dir=csv_dir, out_folder=out_folder, order=order)
        params = load_model_params_from_csv(csv_path=csv_path, row=row)
        out.append(params)
    return out


def make_param_sets(n: int, base_model_params: Dict[str, float]) -> List[Dict[str, float]]:
    return [dict(base_model_params) for _ in range(int(n))]


# -----------------------------
# Config loading
# -----------------------------

def load_named_config(name: str) -> Dict[str, Any]:
    """
    Loads configs.<name> from configs.py. Supports 'default' and any dict-valued config.
    """
    try:
        import configs  # local file
    except Exception as e:
        raise RuntimeError(f"Failed to import configs.py: {e}") from e

    if not hasattr(configs, name):
        available = [
            k for k in dir(configs)
            if not k.startswith("_") and isinstance(getattr(configs, k), dict)
        ]
        raise ValueError(f"Unknown --config {name!r}. Available: {available}")

    cfg = getattr(configs, name)
    if not isinstance(cfg, dict):
        raise ValueError(f"configs.{name} is not a dict.")
    return dict(cfg)


def apply_config_overrides(base_kwargs: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies cfg over base_kwargs.
    - mesh_params/time_params merged as nested dicts.
    - cfg['fem_model_params'] overrides base_kwargs['model_params'].
    - other top-level Runner keys overwritten if present.
    """
    out = dict(base_kwargs)

    # Nested merges
    out["mesh_params"] = merge_nested_dict(out.get("mesh_params", {}), cfg.get("mesh_params"))
    out["time_params"] = merge_nested_dict(out.get("time_params", {}), cfg.get("time_params"))

    # Top-level keys (Runner-related)
    for k in ["fem_verbose", "mesh_verbose", "cities", "base_out", "events_csv", "events_state_col", "years_to_plot", "month_window"]:
        if k in cfg:
            out[k] = cfg[k]

    # fem_model_params -> model_params
    if "fem_model_params" in cfg:
        out["model_params"] = dict(cfg["fem_model_params"])
    elif "model_params" in cfg:
        out["model_params"] = dict(cfg["model_params"])

    return out


# -----------------------------
# Worker
# -----------------------------

def _worker_run_one(
    out_folder: str,
    base_kwargs: Dict[str, Any],
    model_params: Dict[str, float],
) -> Tuple[str, Dict[str, Any]]:
    kwargs = dict(base_kwargs)
    kwargs["model_params"] = dict(model_params)

    runner = Runner(out_folder=out_folder, **kwargs)
    runner.build_mesh()
    summary = runner.run_FEM()
    return out_folder, dict(summary)


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="Number of parallel processes (default: os.cpu_count()).")
    ap.add_argument("--prefix", type=str, default="il_run", help="Out folder prefix (default: il_run).")
    ap.add_argument("--state", type=str, default="", help="State code (optional if --config provides state_list).")
    ap.add_argument("--config", type=str, default="", help="Optional config name from configs.py (e.g., CA).")
    ap.add_argument(
        "--levels",
        type=str,
        default="",
        help="Optional: load model params from out/<out_folder>/csv/. Format: (order,row) or order,row",
    )
    args = ap.parse_args()

    n_workers = int(args.n) if int(args.n) > 0 else (os.cpu_count() or 1)
    prefix = str(args.prefix).strip()
    levels_str = str(args.levels).strip()
    config_name = str(args.config).strip()
    state_cli = str(args.state).strip()

    # Determine a baseline state to build defaults. If config supplies state_list, we'll override.
    base_kwargs = make_base_runner_kwargs_from_configs_default(state_cli=state_cli)

    # Apply config overrides if provided
    if config_name:
        cfg = load_named_config(config_name)
        cfg_state_list = (cfg.get("mesh_params") or {}).get("state_list", None)
        if cfg_state_list is not None and state_cli:
            raise SystemExit(
                f"ERROR: You passed --state={state_cli}, but --config {config_name} "
                f"also specifies mesh_params['state_list']={cfg_state_list}. "
                f"Remove --state or remove state_list from the config."
            )
        base_kwargs = apply_config_overrides(base_kwargs, cfg)

    # Final sanity: ensure we have a state_list
    mesh_params = base_kwargs.get("mesh_params", {})
    if "state_list" not in mesh_params or not mesh_params["state_list"]:
        raise SystemExit("ERROR: No mesh_params['state_list'] specified (use --state or --config).")

    # Windows-safe start method
    mp.set_start_method("spawn", force=True)

    # Pre-build mesh once
    first_folder = f"{prefix}1"
    runner0 = Runner(out_folder=first_folder, **base_kwargs)
    runner0.build_mesh()

    # Build parameter sets
    if levels_str:
        levels = parse_levels(levels_str)
        base_out = Path(base_kwargs.get("base_out", "out"))
        try:
            param_sets = make_param_sets_from_levels(
                n_workers=n_workers,
                prefix=prefix,
                base_out=base_out,
                levels=levels,
            )
        except Exception as e:
            print(f"[MAIN] ERROR while loading params via --levels {levels}: {type(e).__name__}: {e}", file=sys.stderr)
            return 2
        print(f"[MAIN] Using --levels={levels}: order={levels[0]} row={levels[1]}")
    else:
        param_sets = make_param_sets(n_workers, base_kwargs["model_params"])
        print("[MAIN] Using model_params from defaults/config (no --levels)")

    t0 = time.time()
    state_list = base_kwargs["mesh_params"]["state_list"]
    print(f"[MAIN] Launching {n_workers} parallel FEM runs: {prefix}1..{prefix}{n_workers} (states={state_list})")

    results: Dict[str, Dict[str, Any]] = {}
    failures: Dict[str, str] = {}

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as ex:
        futs = []
        for i in range(1, n_workers + 1):
            out_folder = f"{prefix}{i}"
            theta_i = param_sets[i - 1]
            futs.append(ex.submit(_worker_run_one, out_folder, base_kwargs, theta_i))

        for fut in as_completed(futs):
            try:
                out_folder, summary = fut.result()
                results[out_folder] = summary
                print(
                    f"[MAIN] {out_folder} DONE: "
                    f"deviance={float(summary.get('deviance', float('nan'))):.6e} "
                    f"pearson_rms={float(summary.get('pearson_rms', float('nan'))):.6e} "
                    f"inside={int(summary.get('K_inside', 0)):,}"
                )
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                failures[f"job_{len(failures) + 1}"] = msg
                print(f"[MAIN] FAILED: {msg}", file=sys.stderr)

    dt = time.time() - t0

    if results:
        best_folder = min(results.keys(), key=lambda k: float(results[k].get("deviance", float("inf"))))
        best = results[best_folder]
        print("\n[MAIN] Summary")
        print(f"[MAIN] Elapsed: {dt:.1f} s")
        print(f"[MAIN] Completed: {len(results)}/{n_workers}")
        print(
            f"[MAIN] Best (min deviance): {best_folder} "
            f"deviance={float(best.get('deviance', float('nan'))):.6e} "
            f"pearson_rms={float(best.get('pearson_rms', float('nan'))):.6e}"
        )

    if failures:
        print("\n[MAIN] Failures:", file=sys.stderr)
        for k, v in failures.items():
            print(f"  - {k}: {v}", file=sys.stderr)

    return 0 if (len(results) == n_workers and not failures) else 1


if __name__ == "__main__":
    raise SystemExit(main())