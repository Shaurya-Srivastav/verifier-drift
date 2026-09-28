"""Chat-template encoding shared by every generation backend and feature extractor.

Completions are tokenised on their own and appended to the prompt ids. That makes
the completion span ``ids[completion_start:]`` exact. Jointly tokenising the
concatenated string is not the same thing: BPE can merge across the boundary
(``"play" + "ing"`` is not ``"playing"``), so a completion that was only stored
as text may not reproduce the token ids the model sampled. When
``GenOut.token_ids`` is available, prefer those ids over re-encoding ``text``.
"""

from __future__ import annotations

from typing import Any

from vfates.gen.base import chat_messages


def _as_id_list(encoded: Any) -> list[int]:
    if hasattr(encoded, "input_ids"):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("apply_chat_template returned a batch; pass one conversation")
        encoded = encoded[0]
    return [int(i) for i in encoded]


def encode_prompt(tokenizer: Any, user: str, system_prompt: str | None = None) -> list[int]:
    """Token ids of the chat prompt, including the generation prefix.

    ``system_prompt=None`` does not strip a default system turn that the
    tokenizer's own chat template inserts (Qwen's template does). Pass
    ``system_prompt`` to replace that turn.
    """
    messages = chat_messages(user, system_prompt)
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    return _as_id_list(encoded)


def encode_completion(tokenizer: Any, completion: str) -> list[int]:
    """Token ids of ``completion`` with no BOS/EOS added."""
    ids = tokenizer.encode(completion, add_special_tokens=False)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return [int(i) for i in ids]


def encode_prompt_completion(
    tokenizer: Any,
    user: str,
    completion: str,
    system_prompt: str | None = None,
) -> tuple[list[int], int]:
    """``(prompt_ids + completion_ids, completion_start)``.

    ``completion_start`` is the index of the first completion token. The span is
    exact because the completion is encoded with ``add_special_tokens=False``
    and appended, not because the strings were tokenised together. See the
    module docstring for the BPE boundary caveat.
    """
    prompt_ids = encode_prompt(tokenizer, user, system_prompt)
    completion_ids = encode_completion(tokenizer, completion)
    return prompt_ids + completion_ids, len(prompt_ids)


def check_generate_args(n: int, max_tokens: int, temperature: float, top_p: float) -> None:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
