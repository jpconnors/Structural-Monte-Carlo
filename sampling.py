"""Sampling of SDOF structural parameters for Monte Carlo simulation.

Supports crude Monte Carlo and Latin Hypercube Sampling (LHS).  Marginals are
lognormal so that mass, stiffness, damping ratio, and yield force are
guaranteed to be strictly positive.  The lognormal parameters are derived
from a target mean and coefficient of variation (CoV).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import lognorm, qmc


@dataclass(frozen=True)
class LognormalSpec:
    mean: float
    cov: float

    @property
    def sigma(self) -> float:
        return float(np.sqrt(np.log1p(self.cov ** 2)))

    @property
    def mu(self) -> float:
        return float(np.log(self.mean) - 0.5 * self.sigma ** 2)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return lognorm.ppf(u, s=self.sigma, scale=np.exp(self.mu))

    def rvs(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return self.ppf(rng.random(size))


@dataclass
class StructuralParams:
    k: np.ndarray
    m: np.ndarray
    zeta: np.ndarray
    fy: np.ndarray

    def __len__(self) -> int:
        return self.k.shape[0]


DEFAULT_SPECS: dict[str, LognormalSpec] = {
    "k": LognormalSpec(mean=157.91, cov=0.10),
    "m": LognormalSpec(mean=1.00, cov=0.10),
    "zeta": LognormalSpec(mean=0.05, cov=0.30),
    "fy": LognormalSpec(mean=65.08, cov=0.10),
}


def sample_structural_params(
    n: int,
    *,
    method: str = "lhs",
    rng: np.random.Generator | None = None,
    specs: dict[str, LognormalSpec] | None = None,
) -> StructuralParams:
    """Draw n samples of (k, m, zeta, fy)."""
    if rng is None:
        rng = np.random.default_rng()
    if specs is None:
        specs = DEFAULT_SPECS

    names = ("k", "m", "zeta", "fy")
    u = _draw_unit_hypercube(n, d=len(names), method=method, rng=rng)
    arrays = {name: specs[name].ppf(u[:, i]) for i, name in enumerate(names)}
    return StructuralParams(**arrays)


def _draw_unit_hypercube(
    n: int, *, d: int, method: str, rng: np.random.Generator
) -> np.ndarray:
    method = method.lower()
    if method == "mc":
        return rng.random((n, d))
    if method == "lhs":
        sampler = qmc.LatinHypercube(d=d, seed=rng)
        return sampler.random(n)
    raise ValueError(f"Unknown sampling method: {method!r}")
