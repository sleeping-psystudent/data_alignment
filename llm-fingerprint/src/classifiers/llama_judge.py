"""
llama_judge.py — Meta Llama 3.3-70B-Instruct judge via HuggingFace Inference Router.

Uses the OpenAI SDK pointed at https://router.huggingface.co/v1 (the current
HuggingFace inference endpoint) with retry logic and a 1-second inter-call delay.
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

_CFG = JUDGE_CONFIGS["llama"]
_HF_BASE_URL = "https://router.huggingface.co/v1"


class LlamaJudge(BaseJudge):
    """Judge that calls Meta-Llama-3-70B-Instruct via HuggingFace Inference Router."""

    def __init__(self) -> None:
        super().__init__()
        api_key = os.environ.get(_CFG["api_key_env"])
        if not api_key:
            raise EnvironmentError(
                f"Missing env var: {_CFG['api_key_env']}"
            )
        self._client = openai.OpenAI(api_key=api_key, base_url=_HF_BASE_URL)
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
            temperature=0.01,
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
            logger.error("Llama rate limit after %d retries: %s", MAX_RETRIES, exc)
        except openai.AuthenticationError as exc:
            logger.error("Llama auth error (check HF_TOKEN and license acceptance): %s", exc)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("Llama response parse error: %s", exc)
        except Exception as exc:
            logger.error("Llama unexpected error: %s", exc)
        return {"judge": self._model, "scores": null_scores}
