"""Monte Carlo driver for the structural reliability simulation.

Examples
--------
    python run_simulation.py --n-samples 5000 --method lhs --seed 42
    python run_simulation.py --n-samples 90000 --workers 8 --plot
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from analysis import damage_state_results, format_damage_table, save_histogram
from cdm_solver import run_one
from ground_motion import generate_ground_motions
from sampling import sample_structural_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--method", choices=("mc", "lhs"), default="lhs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--total-time", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument(
        "--stratified-phases",
        action="store_true",
        help="Use Latin Hypercube stratified phase angles for ground motion.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a histogram of peak displacements.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)

    print(
        f"Sampling {args.n_samples} structural parameter sets via {args.method.upper()}"
    )
    params = sample_structural_params(args.n_samples, method=args.method, rng=rng)

    print(f"Generating {args.n_samples} ground motion realizations")
    motions = generate_ground_motions(
        args.n_samples,
        duration=args.total_time,
        dt=args.dt,
        rng=rng,
        stratified_phases=args.stratified_phases,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running CDM analyses on {args.workers} worker(s)")
    start = time.perf_counter()
    work = (
        (params.k[i], params.m[i], params.zeta[i], params.fy[i], args.dt, motions[i])
        for i in range(args.n_samples)
    )

    if args.workers <= 1:
        peaks = np.array([run_one(item) for item in work])
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            peaks = np.fromiter(
                pool.map(run_one, work, chunksize=64),
                dtype=float,
                count=args.n_samples,
            )
    elapsed = time.perf_counter() - start
    print(f"  done in {elapsed:.2f} s ({args.n_samples / elapsed:.1f} samples/s)")

    np.savetxt(args.output_dir / "peak_displacements.txt", peaks)

    results = damage_state_results(peaks)
    table = format_damage_table(results)
    print()
    print(table)
    (args.output_dir / "damage_states.txt").write_text(table + "\n")

    if args.plot:
        save_histogram(
            peaks,
            str(args.output_dir / "histogram.png"),
            title=f"N={args.n_samples} ({args.method.upper()})",
        )
        print(f"Histogram written to {args.output_dir / 'histogram.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
