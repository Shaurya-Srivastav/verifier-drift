"""Generation backend interface.

Backends take user-turn strings, apply the model's chat template, and return
``n`` sampled completions per prompt. Sampling defaults (temperature 1.0,
top-p 1.0, no top-k) match on-policy RL sampling so that step-0 audits measure
the same distribution GRPO trains on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from vfates.types import GenOut


class Generator(ABC):
    model_id: str

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        n: int,
        *,
        max_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int = 0,
        system_prompt: str | None = None,
    ) -> list[list[GenOut]]:
        """``n`` completions for each prompt, in prompt order."""

    def describe(self) -> dict[str, Any]:
        return {"backend": type(self).__name__, "model_id": getattr(self, "model_id", None)}


def chat_messages(user: str, system_prompt: str | None = None) -> list[dict[str, str]]:
    messages = [] if system_prompt is None else [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": user})
    return messages
