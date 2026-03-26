"""
reporter.py — Incremental CSV writer and Markdown summary report generator.

CSV rows are appended immediately after each message is processed (resume-safe).
The Markdown summary is written once after all messages are processed.
"""

import csv
import logging
from collections import Counter, defaultdict
from pathlib import Path

from config import CANDIDATE_IDS, DATASETS, SUMMARY_REPORT_PATH

logger = logging.getLogger(__name__)

# CSV columns in output order
_BASE_COLS = [
    "source",
    "top_match",
    "top_score",
    "second_match",
    "confidence_gap",
    "low_confidence",
]
_CSV_COLS = _BASE_COLS + [f"avg_{cid}" for cid in CANDIDATE_IDS]


def _csv_path_for_source(source: str) -> Path:
    if source in DATASETS:
        return DATASETS[source]["results_csv"]
    # Fallback for --single mode
    return DATASETS["social_media"]["results_csv"].parent / "single_results.csv"


def _ensure_header(csv_path: Path) -> None:
    """Write the CSV header row if the file is new or empty."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_COLS)
            writer.writeheader()


def append_result(result: dict) -> None:
    """Append one processed-message result as a CSV row."""
    source = result["source"]
    csv_path = _csv_path_for_source(source)
    _ensure_header(csv_path)

    row: dict = {
        "source": source,
        "top_match": result.get("top_match"),
        "top_score": result.get("top_score"),
        "second_match": result.get("second_match"),
        "confidence_gap": result.get("confidence_gap"),
        "low_confidence": result.get("low_confidence"),
    }
    averaged = result.get("averaged_scores", {})
    for cid in CANDIDATE_IDS:
        row[f"avg_{cid}"] = averaged.get(cid)

    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLS)
        writer.writerow(row)


def generate_summary_report(all_results: list[dict]) -> None:
    """
    Write results/summary_report.md from the in-memory list of all results.

    Sections:
    - Total messages processed per dataset
    - Per-candidate match frequency
    - Average score per candidate
    - Low-confidence count and percentage
    - Top 5 most ambiguous messages (lowest confidence_gap)
    """
    if not all_results:
        logger.warning("No results to summarise.")
        return

    # ── Aggregate stats ────────────────────────────────────────────────────
    source_counts: Counter = Counter()
    top_match_counts: Counter = Counter()
    score_sums: defaultdict = defaultdict(float)
    score_ns: defaultdict = defaultdict(int)
    low_conf_count = 0
    ambiguous: list[dict] = []

    for r in all_results:
        source_counts[r["source"]] += 1
        if r.get("top_match"):
            top_match_counts[r["top_match"]] += 1
        for cid in CANDIDATE_IDS:
            v = (r.get("averaged_scores") or {}).get(cid)
            if isinstance(v, float):
                score_sums[cid] += v
                score_ns[cid] += 1
        if r.get("low_confidence"):
            low_conf_count += 1
        if r.get("confidence_gap") is not None:
            ambiguous.append(r)

    total = len(all_results)
    ambiguous.sort(key=lambda x: x["confidence_gap"])
    top_ambiguous = ambiguous[:5]

    # ── Build Markdown ─────────────────────────────────────────────────────
    lines: list[str] = [
        "# LLM Authorship Fingerprinting — Summary Report",
        "",
        "## Messages Processed",
        "",
        "| Dataset | Count |",
        "|---------|-------|",
    ]
    for source, count in sorted(source_counts.items()):
        lines.append(f"| {source} | {count} |")
    lines += [f"| **Total** | **{total}** |", ""]

    lines += [
        "## Top-Match Frequency (how often each model was the closest match)",
        "",
        "| Candidate | Count | % of messages |",
        "|-----------|-------|---------------|",
    ]
    for cid in CANDIDATE_IDS:
        count = top_match_counts.get(cid, 0)
        pct = f"{100 * count / total:.1f}" if total else "0.0"
        lines.append(f"| {cid} | {count} | {pct}% |")
    lines.append("")

    lines += [
        "## Average Score per Candidate (across all messages)",
        "",
        "| Candidate | Avg Score |",
        "|-----------|-----------|",
    ]
    for cid in CANDIDATE_IDS:
        n = score_ns[cid]
        avg = f"{score_sums[cid] / n:.4f}" if n else "N/A"
        lines.append(f"| {cid} | {avg} |")
    lines.append("")

    low_pct = f"{100 * low_conf_count / total:.1f}" if total else "0.0"
    lines += [
        "## Low-Confidence Results",
        "",
        f"**{low_conf_count}** messages ({low_pct}%) had a confidence gap below "
        f"the threshold (confidence_gap < 0.15), meaning the top two candidates "
        f"were too close to distinguish reliably.",
        "",
    ]

    lines += [
        "## Top 5 Most Ambiguous Messages (lowest confidence_gap)",
        "",
        "| # | Top Match | Score | 2nd Match | Gap | Source |",
        "|---|-----------|-------|-----------|-----|--------|",
    ]
    for i, r in enumerate(top_ambiguous, 1):
        lines.append(
            f"| {i} | {r.get('top_match')} | {r.get('top_score'):.4f} "
            f"| {r.get('second_match')} | {r.get('confidence_gap'):.4f} "
            f"| {r.get('source')} |"
        )
    lines.append("")

    SUMMARY_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Summary report written to %s", SUMMARY_REPORT_PATH)
