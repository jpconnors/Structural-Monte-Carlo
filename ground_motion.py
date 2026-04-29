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


def frequency_grid(n_freq: int, omega_max: float) -> tuple[np.ndarray, float]:
    d_omega = omega_max / n_freq
    omega_i = (np.arange(n_freq) + 0.5) * d_omega
    return omega_i, d_omega


def time_grid(duration: float, dt: float) -> np.ndarray:
    n_steps = int(round(duration / dt)) + 1
    return np.linspace(0.0, duration, n_steps)


def amplitudes(
    n_freq: int = 250,
    omega_max: float = 100.0,
    psd: CloughPenziniPSD | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Spectral amplitudes a_i = 2 sqrt(S(omega_i) d_omega) and frequencies."""
    if psd is None:
        psd = CloughPenziniPSD()
    omega_i, d_omega = frequency_grid(n_freq, omega_max)
    amp = 2.0 * np.sqrt(psd(omega_i) * d_omega)
    return omega_i, amp


def motion_from_phases(
    phases: np.ndarray,
    *,
    duration: float,
    dt: float,
    omega_max: float = 100.0,
    psd: CloughPenziniPSD | None = None,
) -> np.ndarray:
    """Build (n, n_steps) ground-motion array from precomputed phase angles.

    Uses the identity cos(omega t + phi) = cos(omega t) cos(phi) -
    sin(omega t) sin(phi) so the assembly is two (n, n_freq) @ (n_freq,
    n_steps) matrix multiplies, avoiding an O(n * n_freq * n_steps) tensor.
    """
    phases = np.asarray(phases, dtype=float)
    if phases.ndim != 2:
        raise ValueError("phases must be 2-D with shape (n_realizations, n_freq)")
    n_freq = phases.shape[1]
    omega_i, amp = amplitudes(n_freq=n_freq, omega_max=omega_max, psd=psd)
    t = time_grid(duration, dt)

    omega_t = np.outer(omega_i, t)            # (n_freq, n_steps)
    cos_basis = amp[:, None] * np.cos(omega_t)
    sin_basis = amp[:, None] * np.sin(omega_t)

    cos_phi = np.cos(phases)                  # (n, n_freq)
    sin_phi = np.sin(phases)
    return cos_phi @ cos_basis - sin_phi @ sin_basis


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
    """Return an (n, n_steps) array of ground accelerations in g."""
    if rng is None:
        rng = np.random.default_rng()

    if stratified_phases:
        sampler = qmc.LatinHypercube(d=n_freq, seed=rng)
        u = sampler.random(n)
    else:
        u = rng.random((n, n_freq))
    phases = TWO_PI * u
    return motion_from_phases(
        phases, duration=duration, dt=dt, omega_max=omega_max, psd=psd,
    )
