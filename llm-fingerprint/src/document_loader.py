"""
document_loader.py — Excel ingestion, deduplication, and message cleaning.

Reads configured dataset files, filters short/null rows, deduplicates,
truncates to the token limit, and tags each record with its source name.
"""

import logging
from pathlib import Path
from typing import Generator

import pandas as pd
import tiktoken

from config import (
    DATASETS,
    MAX_TOKENS_PER_MESSAGE,
    MIN_MESSAGE_LENGTH,
    CANDIDATE_IDS,
)

logger = logging.getLogger(__name__)

# GPT-2 tokeniser is available offline and gives a reasonable token count
# for all providers (within ~10% of model-specific tokenisers).
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _sanitize(text: str) -> str:
    """Remove control characters and null bytes that break JSON encoding."""
    return "".join(ch for ch in text if ch >= " " or ch in "\t\n\r")


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Return *text* truncated to at most *max_tokens* tokens."""
    tokens = _TOKENIZER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _TOKENIZER.decode(tokens[:max_tokens])


def load_dataset(dataset_name: str, limit: int | None = None) -> list[dict]:
    """
    Load, clean, and return records from the named dataset.

    Args:
        dataset_name: Key in config.DATASETS ("social_media" or "sms").
        limit:        If set, return only the first *limit* clean rows.

    Returns:
        List of dicts with keys: source, original_text.
    """
    cfg = DATASETS[dataset_name]
    file_path: Path = cfg["file"]
    column: str = cfg["column"]

    logger.info("Loading %s from %s", dataset_name, file_path.name)

    df = pd.read_excel(file_path, engine="openpyxl")

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found in {file_path.name}. "
            f"Available: {list(df.columns)}"
        )

    # Keep only the target column, cast to str, strip whitespace
    series = df[column].dropna().astype(str).str.strip()

    # Drop exact duplicates
    series = series.drop_duplicates()

    # Drop short messages
    series = series[series.str.len() >= MIN_MESSAGE_LENGTH]

    if limit is not None:
        series = series.iloc[:limit]

    records: list[dict] = []
    for text in series:
        truncated = _truncate_to_tokens(_sanitize(text), MAX_TOKENS_PER_MESSAGE)
        records.append({"source": dataset_name, "original_text": truncated})

    logger.info("%s: %d records after cleaning", dataset_name, len(records))
    return records


def load_single_message(text: str) -> list[dict]:
    """Wrap a single raw message string in the standard record format."""
    truncated = _truncate_to_tokens(text.strip(), MAX_TOKENS_PER_MESSAGE)
    return [{"source": "single", "original_text": truncated}]
