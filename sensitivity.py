"""Sobol' variance-based sensitivity indices via the Saltelli/Jansen estimator.

Computes first-order (S_i) and total-order (ST_i) Sobol' indices of the
structural parameters (k, m, zeta, fy) on the peak SDOF displacement.

By default the ground motion is held fixed (one Clough-Penzien realization
controlled by --motion-seed) so the indices answer "given a typical motion,
which parameter dominates the response variance?".  With --random-motion the
motion is resampled for each evaluation and the indices reflect aggregated
uncertainty; sum(S_i) + interactions will then be less than one, with the
remainder attributable to ground-motion variability.

References
----------
Saltelli, A. et al. (2010). "Variance based sensitivity analysis of model
output. Design and estimator for the total sensitivity index."
Computer Physics Communications 181, 259-270.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import norm, qmc

from evaluate import N_STRUCT, STRUCT_NAMES, evaluate_peaks
from ground_motion import generate_ground_motions


@dataclass
class SobolResult:
    names: tuple[str, ...]
    first_order: np.ndarray
    total_order: np.ndarray
    variance: float
    n_base: int
    n_evaluations: int

    def format(self) -> str:
        header = f"{'parameter':<10}{'S_i (first)':>14}{'ST_i (total)':>14}"
        rows = [
            f"Output variance Var[Y] = {self.variance:.6f}",
            f"Base sample size       = {self.n_base}",
            f"Total evaluations      = {self.n_evaluations}",
            "",
            header,
            "-" * len(header),
        ]
        for name, s, st in zip(self.names, self.first_order, self.total_order):
            rows.append(f"{name:<10}{s:>14.4f}{st:>14.4f}")
        rows.append("")
        rows.append(f"sum(S_i) = {self.first_order.sum():.4f}   "
                    f"sum(ST_i) = {self.total_order.sum():.4f}")
        return "\n".join(rows)


def sobol_indices(
    n_base: int,
    *,
    d: int,
    names: tuple[str, ...],
    evaluate_fn,
    rng: np.random.Generator | None = None,
) -> SobolResult:
    """Saltelli-Jansen first- and total-order Sobol' indices.

    evaluate_fn(z) takes a (m, d) standard-normal matrix and returns (m,).
    """
    if rng is None:
        rng = np.random.default_rng()

    sampler = qmc.Sobol(d=2 * d, seed=rng, scramble=True)
    log2_n = int(round(np.log2(n_base)))
    if (1 << log2_n) != n_base:
        raise ValueError("n_base must be a power of 2 for Sobol' QMC sampling")
    base = sampler.random_base2(log2_n)
    A = norm.ppf(base[:, :d])
    B = norm.ppf(base[:, d:])

    Y_A = evaluate_fn(A)
    Y_B = evaluate_fn(B)
    n_evals = 2 * n_base

    Y_all = np.concatenate([Y_A, Y_B])
    var_Y = float(np.var(Y_all, ddof=1))
    if var_Y <= 0:
        raise RuntimeError("Output has zero variance; cannot compute indices")

    first = np.empty(d)
    total = np.empty(d)
    for i in range(d):
        AB = A.copy()
        AB[:, i] = B[:, i]
        Y_AB = evaluate_fn(AB)
        n_evals += n_base
        first[i] = np.mean(Y_B * (Y_AB - Y_A)) / var_Y
        total[i] = 0.5 * np.mean((Y_A - Y_AB) ** 2) / var_Y

    return SobolResult(
        names=names,
        first_order=first,
        total_order=total,
        variance=var_Y,
        n_base=n_base,
        n_evaluations=n_evals,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-base", type=int, default=1024,
                        help="Base sample size; must be a power of 2")
    parser.add_argument("--n-freq", type=int, default=250)
    parser.add_argument("--total-time", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--motion-seed", type=int, default=0,
                        help="Seed used to generate the fixed ground motion")
    parser.add_argument("--random-motion", action="store_true",
                        help="Resample motion per evaluation instead of fixing it")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.random_motion:
        def evaluate_fn(z_struct: np.ndarray) -> np.ndarray:
            n = z_struct.shape[0]
            z_phase = rng.standard_normal((n, args.n_freq))
            z = np.hstack([z_struct, z_phase])
            return evaluate_peaks(
                z,
                duration=args.total_time,
                dt=args.dt,
                workers=args.workers,
            )
        motion_descr = "fresh motion per evaluation"
    else:
        motion_rng = np.random.default_rng(args.motion_seed)
        fixed_motion = generate_ground_motions(
            1,
            duration=args.total_time,
            dt=args.dt,
            rng=motion_rng,
            n_freq=args.n_freq,
        )
        from cdm_solver import run_one
        from sampling import DEFAULT_SPECS

        def evaluate_fn(z_struct: np.ndarray) -> np.ndarray:
            n = z_struct.shape[0]
            u = norm.cdf(z_struct)
            params = {
                name: DEFAULT_SPECS[name].ppf(u[:, i])
                for i, name in enumerate(STRUCT_NAMES)
            }
            work = (
                (params["k"][i], params["m"][i], params["zeta"][i],
                 params["fy"][i], args.dt, fixed_motion[0])
                for i in range(n)
            )
            if args.workers <= 1:
                return np.fromiter(
                    (run_one(item) for item in work), dtype=float, count=n
                )
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                return np.fromiter(
                    pool.map(run_one, work, chunksize=64),
                    dtype=float, count=n,
                )
        motion_descr = f"fixed motion (seed={args.motion_seed})"

    print(
        f"Sobol' sensitivity: n_base={args.n_base}, parameters={STRUCT_NAMES}, "
        f"motion={motion_descr}"
    )
    start = time.perf_counter()
    result = sobol_indices(
        n_base=args.n_base,
        d=N_STRUCT,
        names=STRUCT_NAMES,
        evaluate_fn=evaluate_fn,
        rng=rng,
    )
    elapsed = time.perf_counter() - start
    print(f"  done in {elapsed:.2f} s\n")
    print(result.format())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sobol_indices.txt").write_text(result.format() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
