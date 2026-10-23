"""Paired per-sequence statistics between two TrackEval result dirs.

Pairs sequences by name across two run directories (each holding a
pedestrian_detailed.csv), then reports per-sequence deltas (B - A),
win/loss/tie counts, an exact two-sided sign test, and a seeded paired
bootstrap 95% CI on the mean delta, so a change can be shown to win
per-sequence and not just on the pooled score.

The sequence sets must match exactly - a mismatch aborts loudly rather
than silently comparing different subsets.

Usage:
    python tools/paired_stats.py DIR_A DIR_B [--metric HOTA] \
        [--n-boot 10000] [--seed 0]
    python tools/paired_stats.py \
        outputs/run_a/final \
        outputs/run_b/final
"""

from __future__ import annotations

import argparse
import sys
from math import comb
from pathlib import Path

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.per_sport_report import METRICS, load_per_seq_metrics


def pair_sequences(
    per_seq_a: dict[str, dict[str, float]],
    per_seq_b: dict[str, dict[str, float]],
    metric: str,
) -> dict[str, tuple[float, float]]:
    """Pair one metric across two runs by sequence name.

    Args:
        per_seq_a: Per-sequence metrics of run A.
        per_seq_b: Per-sequence metrics of run B.
        metric: Metric name to extract (e.g. "HOTA").

    Returns:
        Mapping of sequence name to (value_a, value_b).

    Raises:
        ValueError: If the two sequence sets differ.
    """
    only_a = sorted(set(per_seq_a) - set(per_seq_b))
    only_b = sorted(set(per_seq_b) - set(per_seq_a))
    if only_a or only_b:
        raise ValueError(
            "Sequence sets differ - refusing to compare. "
            f"Only in A: {only_a or 'none'}; only in B: {only_b or 'none'}"
        )
    if not per_seq_a:
        raise ValueError("No sequences found in either results dir")
    return {
        seq: (per_seq_a[seq][metric], per_seq_b[seq][metric])
        for seq in sorted(per_seq_a)
    }


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided sign test p-value (ties excluded by the caller).

    Args:
        wins: Number of positive deltas.
        losses: Number of negative deltas.

    Returns:
        Two-sided p-value under Binomial(wins + losses, 0.5); 1.0 when
        there are no non-tied pairs.
    """
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(k + 1))
    return min(2.0 * tail / 2**n, 1.0)


def bootstrap_ci(
    deltas: np.ndarray,
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean of paired deltas.

    Args:
        deltas: Per-sequence deltas (B - A).
        n_boot: Number of bootstrap resamples.
        seed: RNG seed (numpy default_rng) for reproducibility.
        alpha: Two-sided miss probability (0.05 -> 95% CI).

    Returns:
        (lower, upper) percentile bounds on the resampled mean delta.
    """
    rng = np.random.default_rng(seed)
    n = len(deltas)
    indices = rng.integers(0, n, size=(n_boot, n))
    means = deltas[indices].mean(axis=1)
    lower, upper = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lower), float(upper)


def build_report(
    dir_a: Path,
    dir_b: Path,
    pairs: dict[str, tuple[float, float]],
    metric: str,
    n_boot: int,
    seed: int,
) -> str:
    """Render the paired comparison as markdown.

    Args:
        dir_a: Run A directory (baseline).
        dir_b: Run B directory (candidate).
        pairs: Output of pair_sequences.
        metric: Metric being compared.
        n_boot: Bootstrap resample count.
        seed: Bootstrap RNG seed.

    Returns:
        Markdown report as a string.
    """
    deltas = np.array([b - a for a, b in pairs.values()])
    wins = int((deltas > 0).sum())
    losses = int((deltas < 0).sum())
    ties = int((deltas == 0).sum())
    p_value = sign_test_p(wins, losses)
    lower, upper = bootstrap_ci(deltas, n_boot=n_boot, seed=seed)
    mean_a = float(np.mean([a for a, _ in pairs.values()]))
    mean_b = float(np.mean([b for _, b in pairs.values()]))

    decimals = 0 if metric == "IDSW" else 2
    lines = [
        f"# Paired per-sequence comparison - {metric}",
        "",
        f"- A (baseline):  `{dir_a}` (mean {mean_a:.{decimals}f} "
        f"over {len(pairs)} seqs)",
        f"- B (candidate): `{dir_b}` (mean {mean_b:.{decimals}f})",
        "",
        f"**Mean delta (B - A): {deltas.mean():+.{decimals}f}** "
        f"[95% CI {lower:+.{decimals}f}, {upper:+.{decimals}f}] "
        f"(paired bootstrap, {n_boot} resamples, seed {seed})",
        "",
        f"Wins (delta > 0): {wins} | Losses: {losses} | Ties: {ties} | "
        f"exact two-sided sign test p = {p_value:.4g}",
    ]
    if metric == "IDSW":
        lines += ["", "Note: lower IDSW is better - a negative delta favors B."]
    lines += [
        "",
        "| Sequence | A | B | Delta (B - A) |",
        "|---|---|---|---|",
    ]
    for seq in sorted(pairs, key=lambda s: pairs[s][1] - pairs[s][0],
                      reverse=True):
        a, b = pairs[seq]
        lines.append(
            f"| {seq} | {a:.{decimals}f} | {b:.{decimals}f} | "
            f"{b - a:+.{decimals}f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired per-sequence stats between two result dirs"
    )
    parser.add_argument("dir_a", type=Path, help="Baseline run directory")
    parser.add_argument("dir_b", type=Path, help="Candidate run directory")
    parser.add_argument(
        "--metric", default="HOTA", choices=METRICS,
        help="Metric to compare (default: HOTA)",
    )
    parser.add_argument(
        "--n-boot", type=int, default=10000,
        help="Bootstrap resamples (default: 10000)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Bootstrap RNG seed (default: 0)",
    )
    args = parser.parse_args()

    try:
        per_seq_a = load_per_seq_metrics(args.dir_a)
        per_seq_b = load_per_seq_metrics(args.dir_b)
        pairs = pair_sequences(per_seq_a, per_seq_b, args.metric)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        raise SystemExit(1)

    print(build_report(args.dir_a, args.dir_b, pairs, args.metric,
                       args.n_boot, args.seed))


if __name__ == "__main__":
    main()
