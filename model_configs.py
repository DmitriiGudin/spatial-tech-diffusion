#!/usr/bin/env python3
# model_configs.py

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from model_utils import ModelSpec, compile_model, ParamGetter, ParamRef, resolve


# =============================================================================
# Function objects WITHOUT derivatives
# All must be pickle-friendly dataclasses or top-level callables.
# =============================================================================

@dataclass(frozen=True)
class Const2Arg:
    """f(x, t) = c"""
    c: ParamRef

    def __call__(self, xy_km: np.ndarray, t_years: float, *, _g: ParamGetter) -> np.ndarray:
        return np.full(xy_km.shape[0], resolve(_g, self.c, 0), dtype=float)


@dataclass(frozen=True)
class SimpleHomographic:
    """f(x) = x/(1+x)"""
    def __call__(self, x: np.ndarray, *, _g: ParamGetter) -> np.ndarray:
        x = np.asarray(x, float)
        return x / np.maximum(1 + x, 1e-15)
    
    
@dataclass(frozen=True)
class SimpleHomographicQuadratic:
    """f(x) = x^2 / (1 + a*x + b*x^2)"""
    a: ParamRef
    b: ParamRef

    def __call__(self, x: np.ndarray, *, _g: ParamGetter) -> np.ndarray:
        x = np.asarray(x, float)
        a = resolve(_g, self.a, 0)
        b = resolve(_g, self.b, 0)
        denom = 1 + a * x + b * x * x
        denom = np.maximum(denom, 1e-15)
        return (x * x) / denom


@dataclass(frozen=True)
class SimpleHomographicCubic:
    """f(x) = x^3 / (1 + a*x + b*x^2 + c*x^3)"""
    a: ParamRef
    b: ParamRef
    c: ParamRef

    def __call__(self, x: np.ndarray, *, _g: ParamGetter) -> np.ndarray:
        x = np.asarray(x, float)
        a = resolve(_g, self.a, 0)
        b = resolve(_g, self.b, 0)
        c = resolve(_g, self.c, 0)
        denom = 1 + a * x + b * x * x + c * x * x * x
        denom = np.maximum(denom, 1e-15)
        return (x * x * x) / denom
    
    
@dataclass(frozen=True)
class HomographicPowerWithExponential:
    """f(x) = (x / (1 + a*x))^b * (1 - e^(-cx))"""
    a: ParamRef
    b: ParamRef
    c: ParamRef

    def __call__(self, x: np.ndarray, *, _g: ParamGetter) -> np.ndarray:
        x = np.asarray(x, float)
        a = resolve(_g, self.a, 0)
        b = resolve(_g, self.b, 0)
        c = resolve(_g, self.c, 0)
        denom = 1 + a * x
        denom = np.maximum(denom, 1e-15)
        return (x / denom)**b * (1 - np.exp(-c*x))


@dataclass(frozen=True)
class Logarithmic:
    """f(x) = log(1 + a*x)"""
    a: ParamRef

    def __call__(self, x: np.ndarray, *, _g: ParamGetter) -> np.ndarray:
        x = np.asarray(x, float)
        a = resolve(_g, self.a, 0)
        # domain: 1 + a*x > 0; if violated, returns -inf/nan per numpy.
        return np.log(1 + a * x)


@dataclass(frozen=True)
class LogarithmicHomographic:
    """f(x) = x * log(1 + a*x) / (1 + b*x)  (b*x is outside the log)"""
    a: ParamRef
    b: ParamRef

    def __call__(self, x: np.ndarray, *, _g: ParamGetter) -> np.ndarray:
        x = np.asarray(x, float)
        a = resolve(_g, self.a, 0)
        b = resolve(_g, self.b, 0)
        denom = 1 + b * x
        denom = np.maximum(denom, 1e-15)
        return x * np.log(1 + a * x) / denom
    
    
# =============================================================================
# Function objects WITH derivatives
# All must be pickle-friendly dataclasses or top-level callables.
# =============================================================================

@dataclass(frozen=True)
class Linear:
    """f(x) = a*x"""
    a: ParamRef

    def __call__(self, x: np.ndarray, *, _g: ParamGetter) -> np.ndarray:
        return resolve(_g, self.a, 0) * np.asarray(x, float)

    def prime(self) -> "LinearPrime":
        return LinearPrime(a=self.a)

@dataclass(frozen=True)
class LinearPrime:
    """d/dx (a*x) = a"""
    a: ParamRef

    def __call__(self, x: np.ndarray, *, _g: ParamGetter) -> np.ndarray:
        return np.full_like(np.asarray(x, float), resolve(_g, self.a, 0), dtype=float)
    
    
@dataclass(frozen=True)
class Quadratic:
    """f(x) = a*x^2 + b*x"""
    a: ParamRef
    b: ParamRef

    def __call__(self, x: np.ndarray, *, _g: ParamGetter) -> np.ndarray:
        x = np.asarray(x, float)
        a = resolve(_g, self.a, 0)
        b = resolve(_g, self.b, 0)
        return a * x * x + b * x

    def prime(self) -> "QuadraticPrime":
        return QuadraticPrime(a=self.a, b=self.b)

@dataclass(frozen=True)
class QuadraticPrime:
    """d/dx (a*x^2 + b*x) = 2a*x + b"""
    a: ParamRef
    b: ParamRef

    def __call__(self, x: np.ndarray, *, _g: ParamGetter) -> np.ndarray:
        x = np.asarray(x, float)
        a = resolve(_g, self.a, 0)
        b = resolve(_g, self.b, 0)
        return 2 * a * x + b
    
    
@dataclass(frozen=True)
class Cubic:
    """f(x) = a*x^3 + b*x^2 + c*x"""
    a: ParamRef
    b: ParamRef
    c: ParamRef

    def __call__(self, x: np.ndarray, *, _g: ParamGetter) -> np.ndarray:
        x = np.asarray(x, float)
        a = resolve(_g, self.a, 0)
        b = resolve(_g, self.b, 0)
        c = resolve(_g, self.c, 0)
        return a * x * x * x + b * x * x + c * x

    def prime(self) -> "CubicPrime":
        return CubicPrime(a=self.a, b=self.b, c=self.c)

@dataclass(frozen=True)
class CubicPrime:
    """d/dx (a*x^3 + b*x^2 + c*x) = 3a*x^2 + 2b*x + c"""
    a: ParamRef
    b: ParamRef
    c: ParamRef

    def __call__(self, x: np.ndarray, *, _g: ParamGetter) -> np.ndarray:
        x = np.asarray(x, float)
        a = resolve(_g, self.a, 0)
        b = resolve(_g, self.b, 0)
        c = resolve(_g, self.c, 0)
        return 3 * a * x * x + 2 * b * x + c
    
    
@dataclass(frozen=True)
class ExponentialCDF:
    """f(x) = 1 - exp(-a*x)"""
    a: ParamRef

    def __call__(self, x: np.ndarray, *, _g: ParamGetter) -> np.ndarray:
        x = np.asarray(x, float)
        a = resolve(_g, self.a, 0)
        # for stability, allow any a; negative a yields >1 values, but that's user choice
        return 1.0 - np.exp(-a * x)

    def prime(self) -> "ExponentialCDFPrime":
        return ExponentialCDFPrime(a=self.a)

@dataclass(frozen=True)
class ExponentialCDFPrime:
    """d/dx (1 - exp(-a*x)) = a*exp(-a*x)"""
    a: ParamRef

    def __call__(self, x: np.ndarray, *, _g: ParamGetter) -> np.ndarray:
        x = np.asarray(x, float)
        a = resolve(_g, self.a, 0)
        return a * np.exp(-a * x)
    
    
@dataclass(frozen=True)
class LinearExponentialCDF:
    """f(x) = a*(1 - exp(-b*x))"""
    a: ParamRef
    b: ParamRef

    def __call__(self, x: np.ndarray, *, _g: ParamGetter) -> np.ndarray:
        x = np.asarray(x, float)
        a = resolve(_g, self.a, 0)
        b = resolve(_g, self.b, 0)
        return a * (1.0 - np.exp(-b * x))

    def prime(self) -> "LinearExponentialCDFPrime":
        return LinearExponentialCDFPrime(a=self.a, b=self.b)

@dataclass(frozen=True)
class LinearExponentialCDFPrime:
    """d/dx (a*(1-exp(-b*x))) = a*b*exp(-b*x)"""
    a: ParamRef
    b: ParamRef

    def __call__(self, x: np.ndarray, *, _g: ParamGetter) -> np.ndarray:
        x = np.asarray(x, float)
        a = resolve(_g, self.a, 0)
        b = resolve(_g, self.b, 0)
        return a * b * np.exp(-b * x)
    
    
@dataclass(frozen=True)
class QuadraticExponentialCDF:
    """f(x) = (a + b*x) * (1 - exp(-c*x))"""
    a: ParamRef
    b: ParamRef
    c: ParamRef

    def __call__(self, x: np.ndarray, *, _g: ParamGetter) -> np.ndarray:
        x = np.asarray(x, float)
        a = resolve(_g, self.a, 0)
        b = resolve(_g, self.b, 0)
        c = resolve(_g, self.c, 0)
        return (a + b * x) * (1 - np.exp(-c * x))

    def prime(self) -> "QuadraticExponentialCDFPrime":
        return QuadraticExponentialCDFPrime(a=self.a, b=self.b, c=self.c)

@dataclass(frozen=True)
class QuadraticExponentialCDFPrime:
    """
    d/dx [(a + b*x)(1 - exp(-c*x))]
      = b*(1 - exp(-c*x)) + (a + b*x)*c*exp(-c*x)
    """
    a: ParamRef
    b: ParamRef
    c: ParamRef

    def __call__(self, x: np.ndarray, *, _g: ParamGetter) -> np.ndarray:
        x = np.asarray(x, float)
        a = resolve(_g, self.a, 0)
        b = resolve(_g, self.b, 0)
        c = resolve(_g, self.c, 0)
        e = np.exp(-c * x)
        return b * (1 - e) + (a + b * x) * c * e


# =============================================================================
# Model definitions
# =============================================================================

SSB_V1_SPEC = ModelSpec(
    name="SSB_V1",
    params=("S0", "gamma_J", "k_J", "D"),
    defs={
        "S": Const2Arg(c="S0"),
        "F_I": SimpleHomographic(),
        "G": Linear(a="gamma_J"),
        "F_J": Linear(a="k_J"),
        "F_J_prime": Linear(a="k_J").prime(),
        "mu_prime": Linear(a="D").prime(),
    }
)


SSB_V2_SPEC = ModelSpec(
    name="SSB_V2",
    params=("S0", "a_I", "b_I", "c_I", "gamma_J1", "gamma_J2", "k_J", "D1", "D2"),
    defs={
        "S": Const2Arg(c="S0"),
        "F_I": SimpleHomographicCubic(a="a_I", b="b_I", c="c_I"),
        "G": Quadratic(a="gamma_J1", b="gamma_J2"),
        "F_J": Linear(a="k_J"),
        "F_J_prime": Linear(a="k_J").prime(),
        "mu_prime": Quadratic(a="D1", b="D2").prime(),
    }
)


SSB_V3_SPEC = ModelSpec(
    name="SSB_V3",
    params=("S0", "a_I", "b_I", "c_I", "gamma_J", "k_J", "D"),
    defs={
        "S": Const2Arg(c="S0"),
        "F_I": HomographicPowerWithExponential(a="a_I", b="b_I", c="c_I"),
        "G": Linear(a="gamma_J"),
        "F_J": Linear(a="k_J"),
        "F_J_prime": Linear(a="k_J").prime(),
        "mu_prime": Linear(a="D").prime(),
    }
)


# =============================================================================
# Currently used model
# =============================================================================

SSB_CURRENT = compile_model(SSB_V3_SPEC)