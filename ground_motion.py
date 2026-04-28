"""Stochastic ground-motion generation via the spectral representation method.

The power spectral density follows the Clough-Penzien filtered Kanai-Tajimi
form used in the original code (omega_g = 15, zeta_g = 0.6, omega_f = 1.5,
zeta_f = 0.6, intensity 7.53e-5).  A fresh realization is produced for every
Monte Carlo sample so that ground-motion variability is captured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc


TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class CloughPenziniPSD:
    omega_g: float = 15.0
    zeta_g: float = 0.6
    omega_f: float = 1.5
    zeta_f: float = 0.6
    intensity: float = 7.53e-5

    def __call__(self, omega: np.ndarray) -> np.ndarray:
        rg = omega / self.omega_g
        rf = omega / self.omega_f
        kt_num = 1.0 + 4.0 * self.zeta_g ** 2 * rg ** 2
        kt_den = (1.0 - rg ** 2) ** 2 + 4.0 * self.zeta_g ** 2 * rg ** 2
        hp_num = self.intensity * rf ** 4
        hp_den = (1.0 - rf ** 2) ** 2 + 4.0 * self.zeta_f ** 2 * rf ** 2
        return (kt_num / kt_den) * (hp_num / hp_den)


def generate_ground_motions(
    n: int,
    *,
    duration: float,
    dt: float,
    rng: np.random.Generator | None = None,
    omega_max: float = 100.0,
    n_freq: int = 250,
    psd: CloughPenziniPSD | None = None,
    stratified_phases: bool = False,
) -> np.ndarray:
    """Return an (n, n_steps) array of ground accelerations in g.

    Each row is an independent realization of the Shinozuka-Deodatis
    spectral representation
        a(t) = 2 sum_i sqrt(S(omega_i) d_omega) cos(omega_i t + phi_i),
    with phi_i drawn uniformly from [0, 2 pi) per realization.
    """
    if rng is None:
        rng = np.random.default_rng()
    if psd is None:
        psd = CloughPenziniPSD()

    n_steps = int(round(duration / dt)) + 1
    t = np.linspace(0.0, duration, n_steps)

    d_omega = omega_max / n_freq
    omega_i = (np.arange(n_freq) + 0.5) * d_omega
    amp = 2.0 * np.sqrt(psd(omega_i) * d_omega)

    if stratified_phases:
        sampler = qmc.LatinHypercube(d=n_freq, seed=rng)
        u = sampler.random(n)
    else:
        u = rng.random((n, n_freq))
    phases = TWO_PI * u

    angles = (
        omega_i[np.newaxis, :, np.newaxis] * t[np.newaxis, np.newaxis, :]
        + phases[:, :, np.newaxis]
    )
    motions = (amp[np.newaxis, :, np.newaxis] * np.cos(angles)).sum(axis=1)
    return motions
