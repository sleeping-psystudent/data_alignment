"""
aggregator.py — Merges per-judge scores into averaged scores and determines
the top candidate match with confidence metrics.
"""

import logging
from typing import Any

from config import CANDIDATE_IDS, LOW_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


def aggregate(record: dict, judge_results: list[dict]) -> dict:
    """
    Merge scores from all judges and compute the final output record.

    Args:
        record:        Base record dict with "source" and "original_text".
        judge_results: List of classify() return values from each judge.

    Returns:
        Fully populated output dict matching the schema in the spec.
    """
    # Build judge_scores dict keyed by judge name
    judge_scores: dict[str, dict] = {}
    for result in judge_results:
        judge_name = result["judge"]
        judge_scores[judge_name] = result["scores"]

    # Average each candidate's score across all judges that returned a non-null value
    averaged: dict[str, float | None] = {}
    for cid in CANDIDATE_IDS:
        values: list[float] = []
        for result in judge_results:
            entry = result["scores"].get(cid, {})
            score = entry.get("score")
            if isinstance(score, (int, float)):
                values.append(float(score))
        averaged[cid] = round(sum(values) / len(values), 4) if values else None

    # Sort by score descending; treat None as -1 so scored models come first
    sorted_candidates = sorted(
        CANDIDATE_IDS,
        key=lambda cid: averaged[cid] if averaged[cid] is not None else -1.0,
        reverse=True,
    )

    top_match = sorted_candidates[0] if sorted_candidates else None
    top_score = averaged.get(top_match) if top_match else None
    second_match = sorted_candidates[1] if len(sorted_candidates) > 1 else None
    second_score = averaged.get(second_match) if second_match else None

    if top_score is not None and second_score is not None:
        confidence_gap = round(top_score - second_score, 4)
        low_confidence = confidence_gap < LOW_CONFIDENCE_THRESHOLD
    else:
        confidence_gap = None
        low_confidence = True

    return {
        "source": record["source"],
        "original_text": record["original_text"],
        "judge_scores": judge_scores,
        "averaged_scores": averaged,
        "top_match": top_match,
        "top_score": top_score,
        "second_match": second_match,
        "confidence_gap": confidence_gap,
        "low_confidence": low_confidence,
    }
