# LLM Authorship Fingerprinting

Scores scam messages against known LLM writing fingerprints using multiple judge models (Claude, GPT-4o, Gemini) and produces a per-message score matrix and summary report.

## Requirements

- Python 3.10+ (conda `PA` environment has everything pre-installed)
- API keys for Anthropic, OpenAI, and Google

## Setup

```bash
cp .env.example .env
# Edit .env and add your API keys
```

## Usage

```bash
# Activate environment
conda activate PA

# Process both datasets
python src/main.py --all

# Process one dataset only
python src/main.py --dataset social_media
python src/main.py --dataset sms

# Test a single message
python src/main.py --single "Congratulations! You have been selected..."

# Use specific judges only
python src/main.py --all --judges claude openai

# Limit rows for quick testing
python src/main.py --all --limit 20
```

## Outputs

| File | Description |
|------|-------------|
| `results/social_media_results.csv` | One row per social media message |
| `results/sms_results.csv` | One row per SMS message |
| `results/summary_report.md` | Aggregate statistics and top ambiguous messages |

Processing is resume-safe — CSV rows are appended incrementally.

## Candidate Models

The pipeline scores each message against 7 candidates: `gpt-4`, `gpt-3.5`, `claude-2`, `gemini-pro`, `llama-3`, `mistral`, `human_written`.

To add a new candidate, append a row to `CANDIDATE_MODELS` in `src/config.py`.

## Architecture

```
src/
├── main.py            CLI + pipeline orchestration
├── config.py          Model registry, paths, constants
├── document_loader.py Excel ingestion, dedup, token truncation
├── aggregator.py      Score averaging, top_match, confidence_gap
├── reporter.py        CSV + Markdown report generation
└── classifiers/
    ├── base.py        Abstract BaseJudge interface
    ├── claude_judge.py  Anthropic SDK
    ├── openai_judge.py  OpenAI SDK
    └── gemini_judge.py  Google SDK
```
