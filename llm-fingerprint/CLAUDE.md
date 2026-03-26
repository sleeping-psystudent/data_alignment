# LLM Authorship Fingerprinting — Project Guide

## Purpose
Pipeline that reads scam message datasets, scores each message against known LLM writing
fingerprints using multiple judge models, and produces a per-message score matrix + summary report.

## Key Commands
```bash
python src/main.py --all                          # both datasets
python src/main.py --dataset social_media         # social media only
python src/main.py --dataset sms                  # SMS only
python src/main.py --single "message text"        # single message test
python src/main.py --all --judges claude openai   # specific judges
python src/main.py --all --limit 20               # first N rows (testing)
```

## Environment
Copy `.env.example` to `.env` and fill in API keys before running.

## Data
- `/data/` contains the source Excel files — never modify them
- `results/` contains incremental CSV output (resume-safe) and the summary report

## Architecture
- `config.py` — model registry and constants; add new candidate models here only
- `document_loader.py` — Excel ingestion, dedup, cleaning
- `classifiers/` — one file per judge provider, all implement `BaseJudge`
- `aggregator.py` — averages scores across judges, computes top_match + confidence_gap
- `reporter.py` — writes CSV rows and final summary_report.md
