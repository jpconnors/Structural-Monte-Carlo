"""Au-Beck Subset Simulation for rare failure-event probability estimation.

Estimates P(Y >= y_target) where Y is the peak SDOF displacement, by
expressing the rare event as a chain of intermediate events
F_0 superset F_1 superset ... superset F_m = F with conditional probability
p_0 (default 0.1) at every level.

Conditional samples are drawn via a Modified Metropolis-Hastings chain in
standard-normal space using the preconditioned (correlated Gaussian) proposal
    x' = sqrt(1 - sigma^2) x + sigma z,   z ~ N(0, I),
which preserves N(0, I) exactly so the only acceptance test is membership in
the current intermediate failure region.

References
----------
Au, S.-K., Beck, J. L. (2001). "Estimation of small failure probabilities in
high dimensions by subset simulation." Probabilistic Engineering Mechanics
16, 263-277.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evaluate import N_STRUCT, evaluate_peaks


@dataclass
class SubsetLevel:
    index: int
    threshold: float
    p_cond: float


@dataclass
class SubsetResult:
    P_failure: float
    coef_of_variation_lower_bound: float
    n_levels: int
    levels: list[SubsetLevel]
    total_evaluations: int

    def format(self) -> str:
        lines = [
            f"P(Y >= y_target) = {self.P_failure:.3e}",
            f"  iid CoV lower bound  = {self.coef_of_variation_lower_bound:.3f}",
            f"  number of levels     = {self.n_levels}",
            f"  total evaluations    = {self.total_evaluations}",
            "",
            f"  {'level':>5}  {'b_l':>10}  {'P(F_l|F_{l-1})':>16}",
        ]
        for lev in self.levels:
            lines.append(
                f"  {lev.index:>5}  {lev.threshold:>10.4f}  {lev.p_cond:>16.4f}"
            )
        return "\n".join(lines)


def subset_simulation(
    threshold: float,
    *,
    d: int,
    n_per_level: int = 2000,
    p0: float = 0.1,
    sigma: float = 0.4,
    max_levels: int = 10,
    rng: np.random.Generator | None = None,
    evaluate_fn,
) -> SubsetResult:
    """Estimate P(Y(x) >= threshold) for x ~ N(0, I_d)."""
    if rng is None:
        rng = np.random.default_rng()
    if not 0.0 < p0 < 1.0:
        raise ValueError("p0 must be in (0, 1)")
    n_seeds = int(round(p0 * n_per_level))
    chain_length = n_per_level // n_seeds
    if n_seeds * chain_length != n_per_level:
        raise ValueError("n_per_level must equal n_seeds * (1/p0); choose so n_per_level * p0 is integer")

    total_evals = 0
    levels: list[SubsetLevel] = []

    x = rng.standard_normal((n_per_level, d))
    y = evaluate_fn(x)
    total_evals += n_per_level

    for level in range(max_levels):
        order = np.argsort(y)[::-1]
        b_l = float(y[order[n_seeds - 1]])

        if b_l >= threshold:
            cond_p = float(np.mean(y >= threshold))
            P_failure = (p0 ** level) * cond_p
            levels.append(SubsetLevel(index=level, threshold=threshold, p_cond=cond_p))
            if cond_p > 0:
                cov_sq = level * (1.0 - p0) / (p0 * n_per_level)
                cov_sq += (1.0 - cond_p) / (cond_p * n_per_level)
                cov_lb = float(np.sqrt(cov_sq))
            else:
                cov_lb = float("inf")
            return SubsetResult(
                P_failure=P_failure,
                coef_of_variation_lower_bound=cov_lb,
                n_levels=level + 1,
                levels=levels,
                total_evaluations=total_evals,
            )

        levels.append(SubsetLevel(index=level, threshold=b_l, p_cond=p0))

        seed_x = x[order[:n_seeds]].copy()
        seed_y = y[order[:n_seeds]].copy()

        chain_x = np.empty((chain_length, n_seeds, d))
        chain_y = np.empty((chain_length, n_seeds))
        chain_x[0] = seed_x
        chain_y[0] = seed_y

        rho = np.sqrt(1.0 - sigma ** 2)
        x_curr = seed_x.copy()
        y_curr = seed_y.copy()
        for k in range(1, chain_length):
            z = rng.standard_normal((n_seeds, d))
            x_prop = rho * x_curr + sigma * z
            y_prop = evaluate_fn(x_prop)
            total_evals += n_seeds
            accept = y_prop >= b_l
            x_curr = np.where(accept[:, None], x_prop, x_curr)
            y_curr = np.where(accept, y_prop, y_curr)
            chain_x[k] = x_curr
            chain_y[k] = y_curr

        x = chain_x.reshape(n_per_level, d)
        y = chain_y.reshape(n_per_level)

    raise RuntimeError(
        f"Subset simulation did not converge within {max_levels} levels"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=2.5,
                        help="Peak-displacement threshold defining failure (in)")
    parser.add_argument("--n-per-level", type=int, default=2000)
    parser.add_argument("--p0", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=0.4,
                        help="Preconditioned MMH proposal scale, in (0, 1]")
    parser.add_argument("--max-levels", type=int, default=10)
    parser.add_argument("--n-freq", type=int, default=250)
    parser.add_argument("--total-time", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    d = N_STRUCT + args.n_freq

    def evaluate_fn(z: np.ndarray) -> np.ndarray:
        return evaluate_peaks(
            z,
            duration=args.total_time,
            dt=args.dt,
            workers=args.workers,
        )

    print(
        f"Subset Simulation: threshold={args.threshold}, n_per_level={args.n_per_level}, "
        f"p0={args.p0}, sigma={args.sigma}, d={d}"
    )
    start = time.perf_counter()
    result = subset_simulation(
        threshold=args.threshold,
        d=d,
        n_per_level=args.n_per_level,
        p0=args.p0,
        sigma=args.sigma,
        max_levels=args.max_levels,
        rng=rng,
        evaluate_fn=evaluate_fn,
    )
    elapsed = time.perf_counter() - start
    print(f"  done in {elapsed:.2f} s\n")
    print(result.format())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "subset_simulation.txt").write_text(result.format() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
