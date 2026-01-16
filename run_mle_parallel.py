#!/usr/bin/env python3
"""
run_mle_parallel.py

Run multiple independent MLE fits in parallel on native Windows.

- Spawns N worker processes (default: os.cpu_count()).
- Each worker runs the same Runner config but with out_folder:
    <prefix>1, <prefix>2, ..., <prefix>N
- Uses "spawn" start method (Windows-safe).
- Pre-builds the mesh once in the main process to avoid mesh build races.

Config logic:
- Base defaults come from configs.default
- Optional --config NAME overlays only keys provided in configs.NAME
- Optional --state XX overrides mesh_params.state_list ONLY if the chosen config
  does not specify mesh_params.state_list (otherwise error).

Usage:
    python run_mle_parallel.py
    python run_mle_parallel.py --n 8
    python run_mle_parallel.py --prefix ca_run --state CA --n 12
    python run_mle_parallel.py --prefix il_run --config IL --n 12
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, Tuple, Optional

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from mle_utils import Runner, _fmt_theta_compact


# -----------------------------
# Small merge helpers
# -----------------------------

def merge_nested_dict(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(base)
    if override:
        out.update(dict(override))
    return out


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


def make_base_runner_kwargs_from_configs_default(state_cli: str = "") -> Dict[str, Any]:
    """
    Build Runner kwargs from configs.default, optionally overriding state_list if --state is provided.
    For MLE runs, Runner expects:
        mesh_params, model_params (MLE priors/ranges), time_params, spsa_params, randomSearch_params,
        fem_verbose, mesh_verbose, ll_verbose, ll_verbose_freq, cities, base_out (optional).
    """
    try:
        import configs  # local file
    except Exception as e:
        raise RuntimeError(f"Failed to import configs.py: {e}") from e

    if not hasattr(configs, "default") or not isinstance(configs.default, dict):
        raise RuntimeError("configs.py must define a dict named `default`.")

    d = dict(configs.default)

    mesh_params = dict(d.get("mesh_params", {}))
    time_params = dict(d.get("time_params", {}))

    # MLE uses mle_model_params as Runner.model_params
    mle_model_params = d.get("mle_model_params", None)
    if mle_model_params is None or not isinstance(mle_model_params, dict):
        raise RuntimeError("configs.default must contain a dict `mle_model_params`.")

    spsa_params = dict(d.get("spsa_params", {}))
    randomSearch_params = dict(d.get("randomSearch_params", {}))

    # Optional CLI --state override (only for defaults; conflict handled later if a named config sets state_list)
    state_cli = str(state_cli).strip()
    if state_cli:
        mesh_params["state_list"] = [state_cli]

    out: Dict[str, Any] = dict(
        mesh_params=mesh_params,
        model_params=dict(mle_model_params),          # <-- MLE priors/bounds
        time_params=time_params,
        spsa_params=spsa_params,
        randomSearch_params=randomSearch_params,
        fem_verbose=bool(d.get("fem_verbose", False)),
        mesh_verbose=bool(d.get("mesh_verbose", False)),
        ll_verbose=bool(d.get("ll_verbose", False)),
        ll_verbose_freq=int(d.get("ll_verbose_freq", 100)),
    )

    # Optional Runner fields if your MLE Runner supports them
    if "base_out" in d:
        out["base_out"] = d["base_out"]

    return out


def apply_config_overrides(base_kwargs: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Overlay cfg on top of base_kwargs.

    - mesh_params/time_params merged as nested dicts.
    - cfg['mle_model_params'] overrides base_kwargs['model_params'].
    - spsa_params/randomSearch_params merged as nested dicts.
    - top-level flags overwritten if present.
    """
    out = dict(base_kwargs)

    out["mesh_params"] = merge_nested_dict(out.get("mesh_params", {}), cfg.get("mesh_params"))
    out["time_params"] = merge_nested_dict(out.get("time_params", {}), cfg.get("time_params"))
    out["spsa_params"] = merge_nested_dict(out.get("spsa_params", {}), cfg.get("spsa_params"))
    out["randomSearch_params"] = merge_nested_dict(out.get("randomSearch_params", {}), cfg.get("randomSearch_params"))

    # Optional top-level keys
    for k in ["fem_verbose", "mesh_verbose", "ll_verbose", "ll_verbose_freq", "base_out"]:
        if k in cfg:
            out[k] = cfg[k]

    # mle_model_params -> model_params
    if "mle_model_params" in cfg:
        out["model_params"] = dict(cfg["mle_model_params"])
    elif "model_params" in cfg:
        # allow this as an escape hatch
        out["model_params"] = dict(cfg["model_params"])

    return out


# -----------------------------
# Worker
# -----------------------------

def _worker_run_one(
    idx: int,
    out_folder: str,
    base_kwargs: Dict[str, Any],
    seed_base: int = 12345,
) -> Tuple[str, float, Dict[str, float]]:
    """
    Worker entrypoint (must be top-level for Windows spawn).

    Returns:
        (out_folder, ll_best, theta_best)
    """
    # Per-run seeds so parallel runs aren't identical
    rs_seed = int(seed_base + 10_000 * idx + 17)
    spsa_seed = int(seed_base + 10_000 * idx + 37)

    # Copy nested dicts we mutate
    kwargs = dict(base_kwargs)
    kwargs["spsa_params"] = dict(kwargs.get("spsa_params", {}))
    kwargs["randomSearch_params"] = dict(kwargs.get("randomSearch_params", {}))

    kwargs["spsa_params"]["seed"] = spsa_seed
    kwargs["randomSearch_params"]["seed"] = rs_seed

    runner = Runner(out_folder=out_folder, **kwargs)

    # Mesh should already exist, but calling build_mesh is safe (reuse).
    runner.build_mesh()
    res = runner.run_MLE()

    return out_folder, float(res.ll_best), dict(res.theta_best)


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="Number of parallel processes (default: os.cpu_count()).")
    ap.add_argument("--prefix", type=str, default="ca_run", help="Out folder prefix (default: ca_run).")
    ap.add_argument("--state", type=str, default="", help="Optional state code (only if config does not set state_list).")
    ap.add_argument("--config", type=str, default="", help="Optional config name from configs.py (e.g., CA, IL, NY).")
    ap.add_argument("--seed-base", type=int, default=12345, help="Base seed for per-worker seeds.")
    args = ap.parse_args()

    n_workers = int(args.n) if int(args.n) > 0 else (os.cpu_count() or 1)
    prefix = str(args.prefix).strip()
    config_name = str(args.config).strip()
    state_cli = str(args.state).strip()
    seed_base = int(args.seed_base)

    # Start from configs.default (+ optional --state if provided)
    base_kwargs = make_base_runner_kwargs_from_configs_default(state_cli=state_cli)

    # Apply named config overrides
    if config_name:
        cfg = load_named_config(config_name)

        # Conflict rule: if config sets mesh_params.state_list and user also set --state, error.
        cfg_state_list = (cfg.get("mesh_params") or {}).get("state_list", None)
        if cfg_state_list is not None and state_cli:
            raise SystemExit(
                f"ERROR: You passed --state={state_cli}, but --config {config_name} "
                f"also specifies mesh_params['state_list']={cfg_state_list}. "
                f"Remove --state or remove state_list from the config."
            )

        base_kwargs = apply_config_overrides(base_kwargs, cfg)

    # Final sanity: ensure state_list exists
    mesh_params = base_kwargs.get("mesh_params", {})
    if "state_list" not in mesh_params or not mesh_params["state_list"]:
        raise SystemExit("ERROR: No mesh_params['state_list'] specified (use --state or --config).")

    # Windows-safe start method
    mp.set_start_method("spawn", force=True)

    # Pre-build mesh once (all workers share mesh_params => share mesh filename)
    first_folder = f"{prefix}1"
    runner0 = Runner(out_folder=first_folder, **base_kwargs)
    runner0.build_mesh()

    t0 = time.time()
    states = mesh_params["state_list"]
    print(f"[MAIN] Launching {n_workers} parallel MLE runs: {prefix}1..{prefix}{n_workers} (states={states})")

    results: Dict[str, Tuple[float, Dict[str, float]]] = {}
    failures: Dict[str, str] = {}

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as ex:
        futs = []
        for i in range(1, n_workers + 1):
            out_folder = f"{prefix}{i}"
            futs.append(ex.submit(_worker_run_one, i, out_folder, base_kwargs, seed_base))

        for fut in as_completed(futs):
            try:
                out_folder, ll_best, theta_best = fut.result()
                results[out_folder] = (ll_best, theta_best)
                print(f"[MAIN] {out_folder} DONE: ll={ll_best:.6e} theta={_fmt_theta_compact(theta_best)}")
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                failures[f"job_{len(failures)+1}"] = msg
                print(f"[MAIN] FAILED: {msg}", file=sys.stderr)

    dt = time.time() - t0

    if results:
        best_folder = max(results.keys(), key=lambda k: results[k][0])
        best_ll, best_theta = results[best_folder]
        print("\n[MAIN] Summary")
        print(f"[MAIN] Elapsed: {dt:.1f} s")
        print(f"[MAIN] Completed: {len(results)}/{n_workers}")
        print(f"[MAIN] Best: {best_folder} ll={best_ll:.6e} theta={_fmt_theta_compact(best_theta)}")

    if failures:
        print("\n[MAIN] Failures:", file=sys.stderr)
        for k, v in failures.items():
            print(f"  - {k}: {v}", file=sys.stderr)

    return 0 if (len(results) == n_workers and not failures) else 1


if __name__ == "__main__":
    raise SystemExit(main())