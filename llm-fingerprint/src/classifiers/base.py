"""
base.py — Abstract base class that every judge classifier must implement.

The contract: given a text string, return a structured dict with a
"judge" label and per-candidate scores + one-sentence evidence strings.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from config import PROMPT_PATH


class BaseJudge(ABC):
    """Abstract base for all LLM judge classifiers."""

    def __init__(self) -> None:
        self._prompt_template: str = PROMPT_PATH.read_text(encoding="utf-8")

    def _build_prompt(self, text: str) -> str:
        """Substitute the message into the prompt template."""
        return self._prompt_template.replace("{DOCUMENT_TEXT}", text)

    @staticmethod
    def _strip_fences(raw: str) -> str:
        """Remove markdown code fences that models add despite prompt instructions."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Drop the opening fence line (```json or just ```)
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        return cleaned.strip()

    @abstractmethod
    def classify(self, text: str) -> dict:
        """
        Score *text* against all candidate LLMs.

        Returns:
            {
              "judge": str,           # judge model name
              "scores": {
                "<model_id>": {
                  "score": float,     # 0.0–1.0 similarity estimate
                  "evidence": str     # one sentence citing a specific linguistic feature
                },
                ...
              }
            }

        On hard failure (after retries), returns the same shape with every
        score set to None and evidence set to the error message string.
        """
