# Data Alignment

A forensic linguistics toolkit for detecting AI-generated scam messages using multi-judge LLM fingerprinting.

## Overview

This project analyzes scam messages (SMS and social media) to determine which LLM (if any) likely generated them. It uses multiple judge models (Claude, GPT-4o, Llama) to score each message against known writing fingerprints of popular LLMs, then aggregates the results to identify the most likely source. It then translate the messages into traditional Chinese using Qwen3-30B-A3B-Instruct-2507.

The system is designed for scam detection and attribution, helping identify patterns in AI-generated fraud campaigns.

## Project Structure

```
├── llm-fingerprint/          Main fingerprinting pipeline
│   ├── src/
│   │   ├── main.py           CLI entry point
│   │   ├── config.py         Model registry and constants
│   │   ├── document_loader.py Excel ingestion and cleaning
│   │   ├── aggregator.py     Score averaging and confidence metrics
│   │   ├── reporter.py       CSV and Markdown report generation
│   │   └── classifiers/      Judge model implementations
│   ├── prompts/              Forensic analysis prompt templates
│   └── results/              Generated CSV and summary reports
│
└── translation/              Translation utilities (separate tool)
```

## Features

- Multi-judge consensus scoring using Claude Sonnet 4.5, GPT-4o, and Llama 3-70B
- Fingerprints 6 LLM candidates: GPT, Claude, Gemini, Llama, Mistral, Qwen
- Processes Excel datasets with automatic deduplication and cleaning
- Resume-safe incremental CSV output
- Confidence gap analysis to flag ambiguous attributions
- Comprehensive summary reports with top ambiguous messages

## Quick Start

### Prerequisites

- Python 3.10+
- API keys for Anthropic, OpenAI, and HuggingFace

### Installation

```bash
cd llm-fingerprint
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env and add your keys:
# ANTHROPIC_API_KEY=sk-...
# OPENAI_API_KEY=sk-...
# HF_TOKEN=hf_...
```

### Usage

```bash
# Process all datasets
python src/main.py --all

# Process one dataset
python src/main.py --dataset social_media
python src/main.py --dataset sms

# Test a single message
python src/main.py --single "Congratulations! You've won a prize..."

# Use specific judges only
python src/main.py --all --judges claude openai

# Quick test with limited rows
python src/main.py --all --limit 20
```

## How It Works

1. **Document Loading**: Reads Excel files, removes duplicates, filters short messages, truncates to token limit
2. **Judge Scoring**: Each judge model analyzes the message against 6 LLM fingerprints based on:
   - Word choice (vocabulary, filler phrases, power words)
   - Grammar (sentence complexity, voice, contractions)
   - Format (lists, capitalization, punctuation patterns)
   - Tone markers (hedging, urgency, regional variants)
3. **Aggregation**: Averages scores across judges, identifies top match and confidence gap
4. **Reporting**: Generates per-message CSV rows and aggregate summary statistics

## Output Files

| File | Description |
|------|-------------|
| `results/social_media_results.csv` | Per-message scores for social media dataset |
| `results/sms_results.csv` | Per-message scores for SMS dataset |
| `results/summary_report.md` | Aggregate statistics, match frequencies, top ambiguous messages |

Each CSV row includes:
- Source dataset
- Top match and score
- Second match and confidence gap
- Low confidence flag
- Average score per candidate model

## Configuration

### Adding New Candidate Models

Edit `src/config.py` and append to `CANDIDATE_MODELS`:

```python
{
    "id": "model_name",
    "fingerprint": "Distinctive writing traits and patterns"
}
```

No other code changes required.

### Adding New Judge Models

1. Create a new judge class in `src/c
