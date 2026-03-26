# SMS Spam → Traditional Chinese Translator

Translates the `v2` column of `spam.csv` to Traditional Chinese using
`Qwen/Qwen3-30B-A3B-Instruct-2507` via the HuggingFace Serverless Inference API,
producing `spam_zh.csv` with columns `v1`, `v2`, `v2_zh`.

## Setup

```bash
pip install requests tqdm pandas
export HF_TOKEN=hf_...   # from https://huggingface.co/settings/tokens
```

> A HuggingFace account with Inference API access is required.
> Free tier supports serverless inference; PRO tier has higher rate limits.

## Run

```bash
# Default paths
python translation/translate.py --input /data/spam.csv --output /data/spam_zh.csv

# Custom model or delay
python translation/translate.py \
  --input  /path/to/spam.csv \
  --output /path/to/spam_zh.csv \
  --model  Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --delay  0.5
```

## Features

- **Qwen3 thinking mode disabled** via `/no_think` suffix — faster, no reasoning overhead
- **Resume-safe**: re-running skips already-translated rows
- **Incremental writes**: each row is flushed immediately — no data lost on crash
- **Retry logic**: up to 3 attempts with exponential backoff (handles 429/503/timeout)
- **Error log**: failed rows are logged to `translation/translation_errors.log`
- **Progress bar**: tqdm shows real-time progress

## Output

```
✅ Translated: 5,571 rows
⚠️  Failed:       0 rows
💾 Saved to: /data/spam_zh.csv
```

Output CSV is UTF-8 with BOM (`utf-8-sig`) for Excel compatibility.

## Notes

- **No GPU required** — inference runs on HuggingFace's servers
- Model may take ~20s to warm up on first call (503 → auto-retry handles this)
- At 0.5s/row, ~5,571 rows ≈ 46 minutes; reduce `--delay` if your tier allows higher throughput
