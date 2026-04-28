"""Verification tests for the central-difference EPP-SDOF solver.

Run with:
    python test_cdm_solver.py
"""

from __future__ import annotations

import numpy as np

from cdm_solver import G_INCH_PER_SEC2, cdm_inelastic


def _zero_motion(n_steps: int) -> np.ndarray:
    return np.zeros(n_steps + 1)


def test_no_excitation_stays_at_rest() -> None:
    """An undisturbed oscillator must stay at zero displacement."""
    accel = _zero_motion(1500)
    max_disp, history = cdm_inelastic(
        k=158.0, m=1.0, zeta=0.05, fy=1e9, dt=0.02, accel_g=accel,
        return_history=True,
    )
    assert max_disp == 0.0
    assert np.allclose(history, 0.0)


def test_linear_elastic_matches_analytical_step_response() -> None:
    """Step base acceleration: peak displacement of an underdamped SDOF.

    With sustained base acceleration a0 (in g), the equivalent static
    displacement is u_st = -a0*g/omega_n^2.  For zero damping the peak
    response is 2*|u_st|; underdamped response peaks slightly below 2*|u_st|.
    """
    k, m, zeta = 100.0, 1.0, 0.0
    a0 = 0.1  # 0.1 g step
    dt = 0.005
    n_steps = 4000
    accel = a0 * np.ones(n_steps + 1)
    omega_n = np.sqrt(k / m)
    u_st = -a0 * G_INCH_PER_SEC2 / omega_n ** 2

    max_disp, _ = cdm_inelastic(
        k=k, m=m, zeta=zeta, fy=1e9, dt=dt, accel_g=accel
    )
    expected = 2.0 * abs(u_st)
    rel_err = abs(max_disp - expected) / expected
    assert rel_err < 0.02, f"step response error {rel_err:.4f}"


def test_yield_force_caps_restoring_force() -> None:
    """For a very low yield force the system must enter plastic flow."""
    accel = 0.5 * np.ones(2001)  # large sustained input
    max_elastic, _ = cdm_inelastic(
        k=100.0, m=1.0, zeta=0.0, fy=1e9, dt=0.01, accel_g=accel,
    )
    max_plastic, _ = cdm_inelastic(
        k=100.0, m=1.0, zeta=0.0, fy=10.0, dt=0.01, accel_g=accel,
    )
    assert max_plastic > max_elastic, (
        "plastic case should drift further than elastic"
    )


def test_damping_reduces_peak_response() -> None:
    rng = np.random.default_rng(0)
    accel = 0.1 * rng.standard_normal(2001)

    peak_undamped, _ = cdm_inelastic(
        k=100.0, m=1.0, zeta=0.0, fy=1e9, dt=0.01, accel_g=accel,
    )
    peak_damped, _ = cdm_inelastic(
        k=100.0, m=1.0, zeta=0.10, fy=1e9, dt=0.01, accel_g=accel,
    )
    assert peak_damped < peak_undamped


def test_history_length_matches_n_steps() -> None:
    accel = _zero_motion(123)
    _, history = cdm_inelastic(
        k=10.0, m=1.0, zeta=0.05, fy=1e9, dt=0.01, accel_g=accel,
        return_history=True,
    )
    assert history is not None
    assert history.shape == (123,)


def main() -> None:
    tests = [
        test_no_excitation_stays_at_rest,
        test_linear_elastic_matches_analytical_step_response,
        test_yield_force_caps_restoring_force,
        test_damping_reduces_peak_response,
        test_history_length_matches_n_steps,
    ]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
