"""
openai_judge.py — OpenAI GPT-4o judge implementation.

Uses the OpenAI Python SDK with retry logic and a 1-second inter-call delay.
"""

import json
import logging
import os
import time

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import API_DELAY_SECONDS, CANDIDATE_IDS, JUDGE_CONFIGS, MAX_RETRIES
from classifiers.base import BaseJudge

logger = logging.getLogger(__name__)

_CFG = JUDGE_CONFIGS["openai"]


class OpenAIJudge(BaseJudge):
    """Judge that calls OpenAI's GPT-4o API to score authorship similarity."""

    def __init__(self) -> None:
        super().__init__()
        api_key = os.environ.get(_CFG["api_key_env"])
        if not api_key:
            raise EnvironmentError(
                f"Missing env var: {_CFG['api_key_env']}"
            )
        self._client = openai.OpenAI(api_key=api_key)
        self._model = _CFG["model"]
        self._last_call: float = 0.0

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < API_DELAY_SECONDS:
            time.sleep(API_DELAY_SECONDS - elapsed)

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _call_api(self, prompt: str) -> str:
        self._wait()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0,
        )
        self._last_call = time.monotonic()
        return response.choices[0].message.content

    def classify(self, text: str) -> dict:
        prompt = self._build_prompt(text)
        null_scores = {
            cid: {"score": None, "evidence": "Judge call failed"}
            for cid in CANDIDATE_IDS
        }
        try:
            raw = self._call_api(prompt)
            parsed = json.loads(self._strip_fences(raw))
            return {"judge": self._model, "scores": parsed["scores"]}
        except openai.RateLimitError as exc:
            logger.error("OpenAI rate limit after %d retries: %s", MAX_RETRIES, exc)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("OpenAI response parse error: %s", exc)
        except Exception as exc:
            logger.error("OpenAI unexpected error: %s", exc)
        return {"judge": self._model, "scores": null_scores}
