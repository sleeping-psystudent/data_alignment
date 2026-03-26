"""
config.py — Model registry, file paths, and pipeline constants.

To add a new candidate model: append a row to CANDIDATE_MODELS only.
No other code changes are required.
"""

import os
from pathlib import Path

# ── Directories ───────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR.parent / "data"
RESULTS_DIR = ROOT_DIR / "results"
PROMPTS_DIR = ROOT_DIR / "prompts"

RESULTS_DIR.mkdir(exist_ok=True)

# ── Data sources ──────────────────────────────────────────────────────────────
DATASETS: dict[str, dict] = {
    "social_media": {
        "file": DATA_DIR / "Social Media Scam Dataset.xlsx",
        "column": "content",
        "results_csv": RESULTS_DIR / "social_media_results.csv",
    },
    "sms": {
        "file": DATA_DIR / "20260302_SMS & URL Scam dataset.xlsx",
        "column": "SMS text",
        "results_csv": RESULTS_DIR / "sms_results.csv",
    },
}

# ── Candidate models being fingerprinted (targets, not API callers) ───────────
# To add a new candidate: append a dict with "id" and "fingerprint" keys.
CANDIDATE_MODELS: list[dict] = [
    {
        "id": "gpt",
        "fingerprint": "Structured prose, em-dashes, numbered lists, formal hedging, avoids contractions",
    },
    {
        "id": "claude",
        "fingerprint": "Longer nuanced sentences, parenthetical asides, thoughtful caveats, varied structure",
    },
    {
        "id": "gemini",
        "fingerprint": "Concise, slightly robotic transitions, Google-adjacent phrasing",
    },
    {
        "id": "llama",
        "fingerprint": "Informal tone, inconsistent capitalization, open-source community style",
    },
    {
        "id": "mistral",
        "fingerprint": "Terse and direct, European phrasing patterns, imperative style",
    },
    {
        "id": "qwen",
        "fingerprint": "Bilingual influence, Mandarin-adjacent sentence structures, concise and factual tone",
    },
]

CANDIDATE_IDS: list[str] = [m["id"] for m in CANDIDATE_MODELS]

# ── Judge models (API callers) ────────────────────────────────────────────────
JUDGE_CONFIGS: dict[str, dict] = {
    "claude": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "openai": {
        "provider": "openai",
        "model": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
    },
    "llama": {
        "provider": "huggingface",
        "model": "meta-llama/Meta-Llama-3-70B-Instruct",
        "api_key_env": "HF_TOKEN",
    },
}

# ── Pipeline constants ────────────────────────────────────────────────────────
MAX_TOKENS_PER_MESSAGE: int = 500          # truncation cap before API call
MIN_MESSAGE_LENGTH: int = 10               # skip rows shorter than this
API_DELAY_SECONDS: float = 1.0             # delay between calls to same provider
MAX_RETRIES: int = 3                       # max retry attempts on rate-limit errors
LOW_CONFIDENCE_THRESHOLD: float = 0.15    # gap < this → low_confidence = True
SUMMARY_REPORT_PATH: Path = RESULTS_DIR / "summary_report.md"
PROMPT_PATH: Path = PROMPTS_DIR / "style_analysis.txt"
