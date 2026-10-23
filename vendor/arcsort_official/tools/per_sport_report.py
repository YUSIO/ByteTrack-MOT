"""Per-sport metric breakdown from TrackEval per-sequence results.

SportsMOT sequence names (``v_<YouTubeID>_c<NNN>``) do not encode the
sport; the authoritative mapping is the official release's
``splits_txt/{basketball,football,volleyball}.txt`` lists (extracted to
``data/SportsMOT/splits_txt/``). Sequences absent from every list are
reported under "unknown" rather than guessed.

Per-sequence metrics come from TrackEval's ``pedestrian_detailed.csv``
(one row per sequence plus COMBINED, fractional values). Directories
holding only ``pedestrian_summary.txt`` cannot be broken down per sport;
re-evaluate with per-sequence output or use the ``*_interp`` run dirs.

Purpose: diagnose which sport drives the val->test generalization gap.

Usage:
    python tools/per_sport_report.py RESULTS_DIR [RESULTS_DIR ...]
    python tools/per_sport_report.py outputs/sportsmot_val/final \
        --out per_sport.md
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from loguru import logger


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPLITS_DIR = REPO_ROOT / "data" / "SportsMOT" / "splits_txt"
SPORTS = ["basketball", "football", "volleyball"]

METRICS = ["HOTA", "DetA", "AssA", "IDF1", "IDSW"]
# pedestrian_detailed.csv column for each metric; fractional columns are
# rescaled to percentages on load, counts are kept as-is.
DETAILED_COLUMNS = {
    "HOTA": "HOTA___AUC",
    "DetA": "DetA___AUC",
    "AssA": "AssA___AUC",
    "IDF1": "IDF1",
    "IDSW": "IDSW",
}
PERCENT_METRICS = {"HOTA", "DetA", "AssA", "IDF1"}


def load_sport_map(splits_dir: Path) -> dict[str, str]:
    """Load the sequence -> sport mapping from the official split lists.

    Args:
        splits_dir: Directory containing basketball.txt / football.txt /
            volleyball.txt (one sequence name per line).

    Returns:
        Mapping of sequence name to sport. Empty (with a warning) when no
        split list exists, so unmapped datasets degrade to "unknown".

    Raises:
        ValueError: If a sequence appears in more than one sport list.
    """
    sport_map: dict[str, str] = {}
    found_any = False
    for sport in SPORTS:
        sport_file = splits_dir / f"{sport}.txt"
        if not sport_file.exists():
            continue
        found_any = True
        for seq in sport_file.read_text().split():
            if seq in sport_map and sport_map[seq] != sport:
                raise ValueError(
                    f"Sequence {seq} listed under both {sport_map[seq]} "
                    f"and {sport} in {splits_dir}"
                )
            sport_map[seq] = sport
    if not found_any:
        logger.warning(
            f"No sport lists found in {splits_dir}; "
            "all sequences will be reported as 'unknown'"
        )
    return sport_map


def load_per_seq_metrics(results_dir: Path) -> dict[str, dict[str, float]]:
    """Parse per-sequence metrics from a run's pedestrian_detailed.csv.

    Args:
        results_dir: A single run directory (e.g.
            outputs/sportsmot_val/final).

    Returns:
        Mapping of sequence name to metric dict (HOTA/DetA/AssA/IDF1 in
        percent, IDSW as a count). The COMBINED row is excluded.

    Raises:
        FileNotFoundError: If pedestrian_detailed.csv is missing.
        ValueError: If an expected metric column is absent.
    """
    detailed = results_dir / "pedestrian_detailed.csv"
    if not detailed.exists():
        raise FileNotFoundError(
            f"{detailed} not found - per-sequence metrics need TrackEval's "
            "detailed output (only pedestrian_summary.txt is present); "
            "re-run evaluation or point at the *_interp run dir"
        )
    per_seq: dict[str, dict[str, float]] = {}
    with open(detailed, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = row["seq"]
            if seq == "COMBINED":
                continue
            metrics = {}
            for metric, column in DETAILED_COLUMNS.items():
                if column not in row:
                    raise ValueError(
                        f"Column {column} missing from {detailed}"
                    )
                value = float(row[column])
                metrics[metric] = value * 100 if metric in PERCENT_METRICS else value
            per_seq[seq] = metrics
    return per_seq


def aggregate_by_sport(
    per_seq: dict[str, dict[str, float]],
    sport_map: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Aggregate per-sequence metrics into per-sport unweighted means.

    Args:
        per_seq: Mapping of sequence name to metric dict.
        sport_map: Mapping of sequence name to sport; sequences not in
            the map fall into "unknown".

    Returns:
        Mapping of sport (plus "ALL" for the overall mean) to a dict of
        metric means with an extra "Seqs" count. Means are unweighted
        over sequences, which is NOT TrackEval's pooled COMBINED score.
    """
    buckets: dict[str, list[dict[str, float]]] = {}
    for seq, metrics in per_seq.items():
        sport = sport_map.get(seq, "unknown")
        buckets.setdefault(sport, []).append(metrics)
    buckets["ALL"] = list(per_seq.values())

    aggregated = {}
    for sport, rows in buckets.items():
        entry: dict[str, float] = {"Seqs": len(rows)}
        for metric in METRICS:
            entry[metric] = sum(r[metric] for r in rows) / len(rows)
        aggregated[sport] = entry
    return aggregated


def build_section(results_dir: Path, aggregated: dict[str, dict[str, float]]) -> str:
    """Render one results dir's per-sport aggregation as a markdown table.

    Args:
        results_dir: The run directory (used as the section heading).
        aggregated: Output of aggregate_by_sport.

    Returns:
        Markdown section (heading + table) as a string.
    """
    order = [s for s in [*SPORTS, "unknown", "ALL"] if s in aggregated]
    lines = [
        f"## {results_dir}",
        "",
        "| Sport | Seqs | " + " | ".join(METRICS) + " |",
        "|---" * (len(METRICS) + 2) + "|",
    ]
    for sport in order:
        entry = aggregated[sport]
        cells = [f"{entry[m]:.2f}" if m != "IDSW" else f"{entry[m]:.1f}"
                 for m in METRICS]
        lines.append(
            f"| {sport} | {int(entry['Seqs'])} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines)


def build_report(results_dirs: list[Path], sport_map: dict[str, str]) -> str:
    """Build the full markdown report over all requested results dirs.

    Args:
        results_dirs: Run directories, each with a pedestrian_detailed.csv.
        sport_map: Mapping of sequence name to sport.

    Returns:
        Markdown report as a string.
    """
    sections = [
        "# Per-sport breakdown (auto-generated by tools/per_sport_report.py)",
        "",
        "Metrics are unweighted means over sequences (not TrackEval's pooled "
        "COMBINED); IDSW is the mean switch count per sequence.",
    ]
    for results_dir in results_dirs:
        per_seq = load_per_seq_metrics(results_dir)
        aggregated = aggregate_by_sport(per_seq, sport_map)
        sections.append("")
        sections.append(build_section(results_dir, aggregated))
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-sport metric breakdown from TrackEval per-seq results"
    )
    parser.add_argument(
        "results_dirs", nargs="+", type=Path,
        help="Run directories containing pedestrian_detailed.csv",
    )
    parser.add_argument(
        "--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR,
        help="Directory with the official {sport}.txt sequence lists "
             f"(default: {DEFAULT_SPLITS_DIR})",
    )
    parser.add_argument(
        "--out", default=None,
        help="Write the report to this file instead of stdout",
    )
    args = parser.parse_args()

    sport_map = load_sport_map(args.splits_dir)
    try:
        report = build_report(args.results_dirs, sport_map)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        raise SystemExit(1)

    if args.out:
        Path(args.out).write_text(report + "\n")
        logger.info(f"Wrote {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
