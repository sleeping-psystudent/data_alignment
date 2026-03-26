#!/usr/bin/env python3
"""
translate.py — Translate spam.csv v2 column to Traditional Chinese
               using Qwen/Qwen3-30B-A3B-Instruct-2507 via HuggingFace Inference API.

Usage:
    python translation/translate.py --input /data/spam.csv --output /data/spam_zh.csv

Environment:
    HF_TOKEN — required (HuggingFace token with inference access)
               Get one at https://huggingface.co/settings/tokens
"""

import argparse
import csv
import json
import os
import sys
import time
import logging
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL           = "Qwen/Qwen3-30B-A3B-Instruct-2507"
HF_API_URL      = f"https://router.huggingface.co/featherless-ai/v1/chat/completions"
MAX_TOKENS      = 512
TEMPERATURE     = 0.1
RATE_LIMIT_DELAY = 0.5   # seconds between calls
MAX_RETRIES     = 3

# Qwen3 has a built-in chain-of-thought "thinking" mode.
# Appending /no_think to the last user turn disables it for faster,
# more deterministic output.
NO_THINK_SUFFIX = " /no_think"

SYSTEM_PROMPT = """你是一位專業的語言學翻譯專家，專門處理簡訊與社群媒體內容的本地化工作。
你的任務是將英文訊息翻譯成繁體中文。

翻譯規則：
- 保留原文的語氣、急迫感與欺騙性語言
- 保留所有 URL、電話號碼、貨幣金額、品牌名稱不翻譯
- 保留原始標點符號格式與換行
- 只輸出翻譯結果，不加任何解釋或前言"""

FEW_SHOT_MESSAGES = [
    # Short ham (casual SMS)
    {"role": "user",      "content": "請將以下英文簡訊翻譯成繁體中文：\n\nOk lar... Joking wif u oni..."},
    {"role": "assistant", "content": "好啦⋯⋯只是跟你開玩笑而已⋯⋯"},

    # Spam with urgency + URL
    {"role": "user",      "content": "請將以下英文簡訊翻譯成繁體中文：\n\nCONGRATULATIONS! You've been selected for a FREE £500 gift card. Claim NOW at www.prize-claim.com before midnight!"},
    {"role": "assistant", "content": "恭喜您！您已被選中獲得免費 £500 禮品卡 🎉 立即前往 www.prize-claim.com 領取，午夜前截止！"},

    # Spam with currency amount
    {"role": "user",      "content": "請將以下英文簡訊翻譯成繁體中文：\n\nURGENT: Your HSBC account has been suspended. Transfer £200 to verify your identity. Call 08001234567 immediately."},
    {"role": "assistant", "content": "緊急通知：您的 HSBC 帳戶已被暫停。請轉帳 £200 以驗證您的身份。請立即致電 08001234567。"},

    # Message with emoji
    {"role": "user",      "content": "請將以下英文簡訊翻譯成繁體中文：\n\nHey! 😊 Are you free tonight? Let's grab dinner 🍕"},
    {"role": "assistant", "content": "嘿！😊 你今晚有空嗎？我們去吃晚飯吧 🍕"},
]

USER_PROMPT_TEMPLATE = "請將以下英文簡訊翻譯成繁體中文：\n\n{v2}"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE = Path(__file__).parent / "translation_errors.log"
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.ERROR,
    format="%(asctime)s  row=%(row)s  error=%(message)s",
)


def log_error(row_index: int, message: str):
    logging.error(message, extra={"row": row_index})


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
def translate(text: str, row_index: int, session: requests.Session, hf_token: str,
              model: str, api_url: str) -> str:
    """Translate a single message to Traditional Chinese. Returns '' on failure."""
    if not text or not str(text).strip():
        return ""

    user_content = USER_PROMPT_TEMPLATE.format(v2=text)

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + list(FEW_SHOT_MESSAGES)
        + [{"role": "user", "content": user_content}]
    )

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "extra_body": {"enable_thinking": False},
    }

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.post(api_url, headers=headers, json=payload, timeout=60)

            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                # Strip any residual <think>...</think> block if thinking bled through
                if "<think>" in content:
                    end = content.find("</think>")
                    content = content[end + len("</think>"):].strip() if end != -1 else content
                return content

            elif resp.status_code in (429, 503):
                # Rate limited or model loading — back off and retry
                wait = 2 ** attempt
                time.sleep(wait)
                continue

            else:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                log_error(row_index, "Request timed out after 60s")
                return ""

        except Exception as exc:
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                log_error(row_index, str(exc))
                return ""

    log_error(row_index, f"All {MAX_RETRIES} retries exhausted")
    return ""


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------
def count_already_done(output_path: str) -> int:
    """Return number of rows already written (excluding header)."""
    p = Path(output_path)
    if not p.exists():
        return 0
    try:
        done_df = pd.read_csv(output_path, encoding="utf-8-sig")
        return len(done_df)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Translate spam.csv to Traditional Chinese via HuggingFace Inference API"
    )
    parser.add_argument("--input",  default="/data/spam.csv",    help="Input CSV path")
    parser.add_argument("--output", default="/data/spam_zh.csv", help="Output CSV path")
    parser.add_argument(
        "--model",
        default=MODEL,
        help="HuggingFace model ID (default: Qwen/Qwen3-30B-A3B-Instruct-2507)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=RATE_LIMIT_DELAY,
        help="Seconds between API calls (default: 0.5)",
    )
    args = parser.parse_args()

    # API key check
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        sys.exit(
            "ERROR: HF_TOKEN environment variable not set.\n"
            "       export HF_TOKEN=hf_..."
        )

    model = args.model
    api_url = HF_API_URL

    # Read input
    try:
        df = pd.read_csv(args.input, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(args.input, encoding="latin-1")

    # Keep only v1, v2
    df = df[["v1", "v2"]].copy()
    df["v2"] = df["v2"].fillna("").astype(str)

    total = len(df)

    # Resume: check how many rows already done
    already_done = count_already_done(args.output)
    if already_done > 0:
        print(f"Resuming — skipping {already_done} already-translated rows.")

    out_path = Path(args.output)
    file_mode = "a" if already_done > 0 else "w"
    write_header = already_done == 0

    failed_rows = []

    with requests.Session() as session:
        with open(out_path, file_mode, encoding="utf-8-sig", newline="") as fout:
            writer = csv.writer(fout)
            if write_header:
                writer.writerow(["v1", "v2", "v2_zh"])

            rows_to_process = df.iloc[already_done:]

            with tqdm(total=total, initial=already_done, unit="row", desc="Translating") as pbar:
                for idx, row in rows_to_process.iterrows():
                    v2_zh = translate(row["v2"], idx, session, hf_token, model, api_url)
                    if v2_zh == "" and row["v2"].strip():
                        failed_rows.append(idx)

                    writer.writerow([row["v1"], row["v2"], v2_zh])
                    fout.flush()  # write incrementally — crash-safe

                    pbar.update(1)
                    time.sleep(args.delay)

    translated_now = total - already_done - len(failed_rows)
    total_translated = already_done + translated_now

    print(f"\n✅ Translated: {total_translated:,} rows")
    print(f"⚠️  Failed:      {len(failed_rows):,} rows" + (f" (see {LOG_FILE})" if failed_rows else ""))
    print(f"💾 Saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
