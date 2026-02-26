#!/usr/bin/env python3
# model_utils.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Tuple, Union, Any

import numpy as np


# =============================================================================
# Public bundle consumed by FEM solver
# =============================================================================

@dataclass
class GSBFunctions:
    S: Callable[[np.ndarray, float], np.ndarray]
    F_I: Callable[[np.ndarray], np.ndarray]
    G: Callable[[np.ndarray], np.ndarray]
    F_J: Callable[[np.ndarray], np.ndarray]
    F_J_prime: Optional[Callable[[np.ndarray], np.ndarray]] = None
    mu_prime: Optional[Callable[[np.ndarray], np.ndarray]] = None


@dataclass(frozen=True)
class ModelConfig:
    name: str
    make_funcs: Callable[[Dict[str, float]], GSBFunctions]
    get_core_params: Callable[[Dict[str, float]], Dict[str, float]]
    build_mle_template: Optional[Callable[[], Tuple[GSBFunctions, Callable[[Dict[str, float]], None]]]] = None
    param_names: Tuple[str, ...] = ()
    core_param_names: Tuple[str, ...] = ("p", "q_I")


# =============================================================================
# Parameter resolution
# =============================================================================

ParamRef = Union[float, int, str]  # numeric literal or parameter-name


class ParamGetter:
    """Resolves parameter names to floats."""
    def get(self, name: str, default: float = 0) -> float:
        raise NotImplementedError


@dataclass
class ConstGetter(ParamGetter):
    params: Dict[str, float]
    default_for_all: float = 0

    def get(self, name: str, default: float = 0) -> float:
        if name in self.params:
            return float(self.params[name])
        # if caller supplies default explicitly, use it, else uniform default
        return float(default if default is not None else self.default_for_all)


@dataclass
class BoxGetter(ParamGetter):
    box: Dict[str, float]
    default_for_all: float = 0

    def get(self, name: str, default: float = 0) -> float:
        if name not in self.box:
            self.box[name] = float(default if default is not None else self.default_for_all)
        return float(self.box[name])


def resolve(getter: ParamGetter, ref: ParamRef, default: float = 0) -> float:
    """Resolve a ParamRef (string or number) into a float."""
    if isinstance(ref, str):
        return getter.get(ref, default=default)
    return float(ref)


# =============================================================================
# Binding wrappers (adapt your function-objects to solver call signatures)
#
# Contract for function objects:
#   - 2arg: obj(xy_km, t_years, *, _g: ParamGetter) -> ndarray
#   - 1arg: obj(x, *, _g: ParamGetter) -> ndarray
# =============================================================================

def bind_2arg(obj: Any, getter: ParamGetter) -> Callable[[np.ndarray, float], np.ndarray]:
    def _call(xy_km: np.ndarray, t_years: float) -> np.ndarray:
        return obj(xy_km, t_years, _g=getter)
    return _call


def bind_1arg(obj: Any, getter: ParamGetter) -> Callable[[np.ndarray], np.ndarray]:
    def _call(x: np.ndarray) -> np.ndarray:
        return obj(x, _g=getter)
    return _call


# =============================================================================
# Declarative model spec -> compiled ModelConfig
# =============================================================================

@dataclass(frozen=True)
class ModelSpec:
    """
    Minimal declarative model description.

    params:
      tuple/list of parameter names used inside function definitions.
      These default to 0.0 unless provided in theta.

    defs:
      dict with required keys: "S", "F_I", "G", "F_J"
      optional: "F_J_prime", "mu_prime"

      Values are function-objects following the contract above.

    core_defaults:
      optional override of p,q_I defaults; by default:
        p = 0.01, q_I = 0.1
    """
    name: str
    params: Tuple[str, ...]
    defs: Mapping[str, Any]
    core_defaults: Optional[Mapping[str, float]] = None


def compile_model(spec: ModelSpec) -> ModelConfig:
    core0: Dict[str, float] = {"p": 0.01, "q_I": 0.1}
    if spec.core_defaults:
        core0.update({str(k): float(v) for k, v in spec.core_defaults.items()})

    required = ("S", "F_I", "G", "F_J")
    for k in required:
        if k not in spec.defs:
            raise KeyError(f"{spec.name}: defs must include '{k}'")

    def _build(getter: ParamGetter) -> GSBFunctions:
        d = spec.defs

        S = bind_2arg(d["S"], getter)
        F_I = bind_1arg(d["F_I"], getter)
        G = bind_1arg(d["G"], getter)
        F_J = bind_1arg(d["F_J"], getter)

        F_J_prime = None
        mu_prime = None
        if d.get("F_J_prime", None) is not None:
            F_J_prime = bind_1arg(d["F_J_prime"], getter)
        if d.get("mu_prime", None) is not None:
            mu_prime = bind_1arg(d["mu_prime"], getter)

        return GSBFunctions(S=S, F_I=F_I, G=G, F_J=F_J, F_J_prime=F_J_prime, mu_prime=mu_prime)

    def make_funcs(theta: Dict[str, float]) -> GSBFunctions:
        merged = {k: 0.0 for k in spec.params}
        merged.update({str(k): float(v) for k, v in theta.items()})
        getter = ConstGetter(merged, default_for_all=0)
        return _build(getter)

    def get_core_params(theta: Dict[str, float]) -> Dict[str, float]:
        return {k: float(theta.get(k, v)) for k, v in core0.items()}

    def build_mle_template() -> Tuple[GSBFunctions, Callable[[Dict[str, float]], None]]:
        box: Dict[str, float] = {k: 0 for k in spec.params}
        getter = BoxGetter(box, default_for_all=0)
        funcs = _build(getter)

        def sync(theta: Dict[str, float]) -> None:
            # update only keys we track
            for k in box.keys():
                if k in theta:
                    box[k] = float(theta[k])

        return funcs, sync

    core_param_names = tuple(core0.keys())  # preserves order of core0 dict
    return ModelConfig(name=spec.name, make_funcs=make_funcs, get_core_params=get_core_params, build_mle_template=build_mle_template,
        param_names=tuple(spec.params), core_param_names=core_param_names)