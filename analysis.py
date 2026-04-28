"""Damage-state binning, confidence intervals, and plotting helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_THRESHOLDS: tuple[tuple[str, float], ...] = (
    ("no_damage", 2.00),
    ("slight", 2.10),
    ("moderate", 2.25),
    ("major", 2.50),
    ("collapse", float("inf")),
)


@dataclass
class DamageStateResult:
    name: str
    count: int
    probability: float
    ci_low: float
    ci_high: float


def wilson_score_interval(
    count: int, total: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    z = _normal_quantile(0.5 + confidence / 2.0)
    p = count / total
    denom = 1.0 + z ** 2 / total
    centre = (p + z ** 2 / (2.0 * total)) / denom
    halfwidth = (
        z * np.sqrt(p * (1.0 - p) / total + z ** 2 / (4.0 * total ** 2)) / denom
    )
    return max(0.0, centre - halfwidth), min(1.0, centre + halfwidth)


def _normal_quantile(p: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(p))


def damage_state_results(
    displacements: np.ndarray,
    thresholds: tuple[tuple[str, float], ...] = DEFAULT_THRESHOLDS,
    confidence: float = 0.95,
) -> list[DamageStateResult]:
    """Bin peak displacements into ordered damage states with Wilson CIs."""
    displacements = np.asarray(displacements, dtype=float)
    n = displacements.size
    edges = [t for _, t in thresholds]
    lower = -np.inf
    results: list[DamageStateResult] = []
    for (name, upper), upper_value in zip(thresholds, edges):
        mask = (displacements >= lower) & (displacements < upper_value)
        count = int(mask.sum())
        prob = count / n if n else 0.0
        ci_low, ci_high = wilson_score_interval(count, n, confidence)
        results.append(DamageStateResult(name, count, prob, ci_low, ci_high))
        lower = upper_value
    return results


def format_damage_table(
    results: list[DamageStateResult], confidence: float = 0.95
) -> str:
    pct = int(round(confidence * 100))
    header = f"{'state':<12}{'count':>10}{'P(state)':>12}{f'CI{pct}% low':>14}{f'CI{pct}% high':>14}"
    rows = [header, "-" * len(header)]
    for r in results:
        rows.append(
            f"{r.name:<12}{r.count:>10}{r.probability:>12.5f}"
            f"{r.ci_low:>14.5f}{r.ci_high:>14.5f}"
        )
    return "\n".join(rows)


def save_histogram(
    displacements: np.ndarray, path: str, bins: int = 35, title: str | None = None
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.hist(displacements, bins=bins)
    ax.set_xlabel("Peak displacement (in)")
    ax.set_ylabel("Count")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
