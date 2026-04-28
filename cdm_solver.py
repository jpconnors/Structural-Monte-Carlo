"""Central Difference Method (CDM) time integration of an elastic-perfectly-
plastic single-degree-of-freedom oscillator.

State equation:  m u" + c u' + f_s(u) = -m * a_g(t)
with f_s an elastic-perfectly-plastic restoring force capped at +/-fy.
"""

from __future__ import annotations

import numpy as np


G_INCH_PER_SEC2 = 386.09  # gravitational acceleration in in/s^2


def cdm_inelastic(
    k: float,
    m: float,
    zeta: float,
    fy: float,
    dt: float,
    accel_g: np.ndarray,
    *,
    return_history: bool = False,
) -> tuple[float, np.ndarray | None]:
    """Run the EPP-SDOF on a ground acceleration trace.

    Parameters
    ----------
    k, m, zeta, fy : float
        Stiffness, mass, damping ratio, and yield force (consistent units).
    dt : float
        Time step.
    accel_g : ndarray
        Base acceleration in g, length n_steps + 1 (initial value plus
        n_steps integration increments).
    return_history : bool
        If True, also return the displacement at each completed step.

    Returns
    -------
    max_disp : float
        Peak absolute displacement.
    history : ndarray or None
        Per-step displacement (length n_steps) if requested.
    """
    accel_g = np.asarray(accel_g, dtype=float)
    n_steps = accel_g.size - 1
    if n_steps <= 0:
        raise ValueError("accel_g must contain at least two samples")

    c = 2.0 * zeta * np.sqrt(k * m)
    pp = -G_INCH_PER_SEC2 * m * accel_g

    u_prev = 0.0
    u_curr = 0.0
    v0 = 0.0
    a0 = (pp[0] - c * v0 - k * u_curr) / m
    u_minus = u_curr - dt * v0 + 0.5 * dt ** 2 * a0

    k_hat = m / dt ** 2 + c / (2.0 * dt)
    coef_minus = m / dt ** 2 - c / (2.0 * dt)
    coef_curr = -2.0 * m / dt ** 2

    uy = 0.0
    sign_fy = 0.0
    has_yielded = False
    max_disp = 0.0

    history = np.empty(n_steps, dtype=float) if return_history else None

    for i in range(n_steps):
        if not has_yielded:
            fs = k * u_curr
        else:
            fs = k * (u_curr - uy) + sign_fy

        if abs(fs) >= fy:
            v_curr = (u_curr - u_minus) / (2.0 * dt) if i > 0 else v0
            sign_fy = np.sign(v_curr) * fy if v_curr != 0.0 else np.sign(fs) * fy
            uy = u_curr
            fs = sign_fy
            has_yielded = True

        rhs = pp[i] - coef_minus * u_minus - coef_curr * u_curr - fs
        u_next = rhs / k_hat

        if abs(u_curr) > max_disp:
            max_disp = abs(u_curr)

        if history is not None:
            history[i] = u_next

        u_minus = u_curr
        u_curr = u_next

    if abs(u_curr) > max_disp:
        max_disp = abs(u_curr)

    return max_disp, history


def run_one(args: tuple) -> float:
    """Worker entry point for multiprocessing."""
    k, m, zeta, fy, dt, accel_g = args
    max_disp, _ = cdm_inelastic(k, m, zeta, fy, dt, accel_g)
    return max_disp
