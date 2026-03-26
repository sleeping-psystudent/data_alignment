"""
main.py — CLI entry point for the LLM authorship fingerprinting pipeline.

Usage:
    python src/main.py --all
    python src/main.py --dataset social_media
    python src/main.py --dataset sms
    python src/main.py --single "message text"
    python src/main.py --all --judges claude openai
    python src/main.py --all --limit 20
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Allow bare imports from src/
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

import aggregator
import reporter
from config import DATASETS, JUDGE_CONFIGS
from document_loader import load_dataset, load_single_message

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _build_judges(judge_names: list[str]) -> list:
    """Instantiate and return the requested judge objects."""
    # Import here so missing keys don't crash on import
    from classifiers import ClaudeJudge, LlamaJudge, OpenAIJudge

    factory = {
        "claude": ClaudeJudge,
        "openai": OpenAIJudge,
        "llama": LlamaJudge,
    }
    judges = []
    for name in judge_names:
        if name not in factory:
            logger.warning("Unknown judge '%s', skipping.", name)
            continue
        try:
            judges.append(factory[name]())
            logger.info("Judge initialised: %s", name)
        except EnvironmentError as exc:
            logger.error("Skipping judge '%s': %s", name, exc)
    return judges


def _process_records(
    records: list[dict],
    judges: list,
    dataset_label: str,
) -> list[dict]:
    """Run all judges on every record, aggregate, append to CSV, return results."""
    results: list[dict] = []
    total = len(records)

    for i, record in enumerate(records, 1):
        logger.info("[%d/%d] Processing %s record …", i, total, dataset_label)

        judge_results = []
        for judge in judges:
            result = judge.classify(record["original_text"])
            judge_results.append(result)

        aggregated = aggregator.aggregate(record, judge_results)
        reporter.append_result(aggregated)
        results.append(aggregated)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Authorship Fingerprinting Pipeline"
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true", help="Process all datasets")
    mode.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        help="Process a single named dataset",
    )
    mode.add_argument("--single", metavar="TEXT", help="Score a single message string")

    parser.add_argument(
        "--judges",
        nargs="+",
        choices=["claude", "openai", "llama"],
        default=["claude", "openai", "llama"],
        help="Which judge models to use (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N records per dataset (for testing)",
    )

    args = parser.parse_args()

    judges = _build_judges(args.judges)
    if not judges:
        logger.error("No judges available — check API keys in .env. Exiting.")
        sys.exit(1)

    all_results: list[dict] = []

    if args.single:
        records = load_single_message(args.single)
        results = _process_records(records, judges, "single")
        all_results.extend(results)
        top = results[0].get("top_match") if results else "N/A"
        score = results[0].get("top_score") if results else None
        print(f"\nTop match: {top}  (score: {score})")

    else:
        datasets_to_run = list(DATASETS.keys()) if args.all else [args.dataset]
        for ds_name in datasets_to_run:
            records = load_dataset(ds_name, limit=args.limit)
            results = _process_records(records, judges, ds_name)
            all_results.extend(results)

        reporter.generate_summary_report(all_results)
        logger.info(
            "Done. %d messages processed. Results in results/", len(all_results)
        )


if __name__ == "__main__":
    main()
