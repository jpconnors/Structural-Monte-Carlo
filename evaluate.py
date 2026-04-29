"""Map standard-normal input vectors to peak SDOF displacement.

A single sample x = [z_struct (4), z_phase (n_freq)] in standard-normal
space is transformed via Phi(z) -> uniform -> structural parameter (lognormal
PPF) or phase angle (2 pi * u).  The resulting structural model and ground
motion are passed to the central-difference solver and the peak |displacement|
is returned.

This unified evaluator is shared by the forward Monte Carlo, Subset
Simulation, and Sobol' sensitivity analyses.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import norm

from cdm_solver import run_one
from ground_motion import CloughPenziniPSD, motion_from_phases
from sampling import DEFAULT_SPECS, LognormalSpec


STRUCT_NAMES: tuple[str, ...] = ("k", "m", "zeta", "fy")
N_STRUCT: int = len(STRUCT_NAMES)


def z_to_params(
    z_struct: np.ndarray,
    specs: dict[str, LognormalSpec] | None = None,
) -> dict[str, np.ndarray]:
    if specs is None:
        specs = DEFAULT_SPECS
    u = norm.cdf(z_struct)
    return {name: specs[name].ppf(u[:, i]) for i, name in enumerate(STRUCT_NAMES)}


def z_to_phases(z_phase: np.ndarray) -> np.ndarray:
    return 2.0 * np.pi * norm.cdf(z_phase)


def evaluate_peaks(
    z: np.ndarray,
    *,
    duration: float,
    dt: float,
    omega_max: float = 100.0,
    psd: CloughPenziniPSD | None = None,
    workers: int = 1,
    specs: dict[str, LognormalSpec] | None = None,
) -> np.ndarray:
    """Run the SDOF analysis on rows of the standard-normal matrix z.

    z has shape (n, N_STRUCT + n_freq).  The first N_STRUCT columns map to
    structural parameters; the rest map to phase angles.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or z.shape[1] <= N_STRUCT:
        raise ValueError(
            f"z must be 2-D with at least {N_STRUCT + 1} columns, got {z.shape}"
        )
    n = z.shape[0]
    params = z_to_params(z[:, :N_STRUCT], specs=specs)
    phases = z_to_phases(z[:, N_STRUCT:])
    motions = motion_from_phases(
        phases, duration=duration, dt=dt, omega_max=omega_max, psd=psd,
    )

    work = (
        (params["k"][i], params["m"][i], params["zeta"][i], params["fy"][i],
         dt, motions[i])
        for i in range(n)
    )

    if workers <= 1:
        peaks = np.fromiter((run_one(item) for item in work), dtype=float, count=n)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            peaks = np.fromiter(
                pool.map(run_one, work, chunksize=64),
                dtype=float,
                count=n,
            )
    return peaks
