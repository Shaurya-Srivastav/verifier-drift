"""Content judges for the instruction-following oracle.

The oracle is the constraint checker AND a content judge. The checker accepts
keyword stuffing; the judge is what marks that response incorrect. Tulu 3
documents the failure mode: RLVR overoptimizes the verifiable reward and stops
producing meaningful content (Lambert et al., the ``timberlike, unconnectedly
Conversation N User :`` pattern).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from vfates.gen.base import Generator
from vfates.utils.io import append_jsonl, canonical_json, iter_jsonl, sha256_text

DEFAULT_RUBRIC = (
    "You are a strict content judge. Decide whether the assistant's response "
    "substantively fulfils the user's request, apart from formatting constraints "
    "(capitalization, keyword counts, bullet counts, JSON wrapping, and similar). "
    "Keyword stuffing, copied instructions, and degenerate repetition do not count "
    "as fulfilment. Answer with a single token: YES or NO."
)

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{1,}")
_YES_NO = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)


def format_judge_user(prompt: str, response: str) -> str:
    """User turn sent to an LLM judge."""
    return f"USER REQUEST:\n{prompt}\n\nASSISTANT RESPONSE:\n{response}\n"


def parse_yes_no(text: str) -> int | None:
    """Map a judge reply to 1 (YES) or 0 (NO). None when neither token appears."""
    match = _YES_NO.search(text or "")
    if match is None:
        return None
    return 1 if match.group(1).upper() == "YES" else 0


class Judge(ABC):
    """Binary content judgement: 1 substantive, 0 not."""

    name: str = "judge"

    @abstractmethod
    def judge(self, prompt: str, response: str) -> int:
        """1 if ``response`` substantively fulfils ``prompt``, else 0."""

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}


class HeuristicJudge(Judge):
    """Flag keyword stuffing, padding, and degenerate repetition without an LLM.

    A response scores 1 only when every rule below passes. Defaults are aimed
    at step-0 prose, not at responses that have already been optimized to
    repeat a constraint keyword dozens of times.

    * **Content words** (``min_content_words``, default 12). Alphabetic words
      (regex ``[A-Za-z][A-Za-z'-]{1,}``) must number at least this many.
      Symbol padding and empty replies fail here.
    * **Repetition** (``max_repeat_ratio``, default 0.4). On 16 or more
      whitespace tokens (lowercased, punctuation stripped): fail if the most
      common token's share exceeds ``max_repeat_ratio``, or if the type-token
      ratio is strictly below ``1 - max_repeat_ratio``. The Tulu 3 loop
      ``timberlike, unconnectedly Conversation N User :`` fails the type-token
      test (a handful of tokens recycled across the reply).
    * **Keyword density** (``max_keyword_density``, default 0.5). Split on
      commas. A shard is a segment with at most three alphabetic words and no
      ``.?!``. With at least four segments, fail when the shard fraction is
      strictly above ``max_keyword_density``. This catches a comma-separated
      list of distinct keywords, which repetition statistics miss.
    * **Padding** (fixed). If the reply has at least 40 characters and fewer
      than 45% of them are alphanumeric, fail.
    """

    name = "heuristic"

    def __init__(
        self,
        min_content_words: int = 12,
        max_repeat_ratio: float = 0.4,
        max_keyword_density: float = 0.5,
    ) -> None:
        if min_content_words < 1:
            raise ValueError("min_content_words must be positive")
        if not 0 < max_repeat_ratio < 1:
            raise ValueError("max_repeat_ratio must lie in (0, 1)")
        if not 0 < max_keyword_density <= 1:
            raise ValueError("max_keyword_density must lie in (0, 1]")
        self.min_content_words = int(min_content_words)
        self.max_repeat_ratio = float(max_repeat_ratio)
        self.max_keyword_density = float(max_keyword_density)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "min_content_words": self.min_content_words,
            "max_repeat_ratio": self.max_repeat_ratio,
            "max_keyword_density": self.max_keyword_density,
        }

    def judge(self, prompt: str, response: str) -> int:
        # Stuffing is a property of the response. ``prompt`` is unused on purpose.
        text = response or ""
        words = _WORD.findall(text)
        if len(words) < self.min_content_words:
            return 0
        if len(text) >= 40:
            alnum = sum(1 for ch in text if ch.isalnum())
            if alnum / len(text) < 0.45:
                return 0
        if _repetition(text, self.max_repeat_ratio):
            return 0
        if _keyword_stuffing(text, self.max_keyword_density):
            return 0
        return 1


def _tokens(text: str) -> list[str]:
    raw = re.findall(r"\S+", text.lower())
    return [
        re.sub(r"^[^\w]+|[^\w]+$", "", token)
        for token in raw
        if re.sub(r"^[^\w]+|[^\w]+$", "", token)
    ]


def _repetition(text: str, max_repeat_ratio: float) -> bool:
    tokens = _tokens(text)
    if len(tokens) < 16:
        return False
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    top = max(counts.values()) / len(tokens)
    if top > max_repeat_ratio:
        return True
    type_token = len(counts) / len(tokens)
    return type_token < (1.0 - max_repeat_ratio)


def _keyword_stuffing(text: str, max_keyword_density: float) -> bool:
    segments = text.split(",")
    if len(segments) < 4:
        return False
    shards = 0
    for segment in segments:
        if re.search(r"[.?!]", segment):
            continue
        if len(_WORD.findall(segment)) <= 3:
            shards += 1
    return (shards / len(segments)) > max_keyword_density


class LLMJudge(Judge):
    """YES/NO chat-completions judge. Stdlib ``urllib`` only.

    ``base_url`` defaults to ``VFATES_JUDGE_BASE_URL`` or
    ``https://api.openai.com/v1``. The API key is ``VFATES_JUDGE_API_KEY``
    unless passed in. Temperature is 0. Transport failures and unparseable
    replies are retried with exponential backoff; a reply that never says
    YES or NO scores 0.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        rubric: str | None = None,
        *,
        max_retries: int = 3,
        backoff_s: float = 0.5,
        timeout_s: float = 60.0,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be positive")
        self.model = model
        self.base_url = base_url
        self._api_key = api_key
        self.rubric = rubric or DEFAULT_RUBRIC
        self.max_retries = int(max_retries)
        self.backoff_s = float(backoff_s)
        self.timeout_s = float(timeout_s)
        self.name = f"llm:{model}"

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "base_url": self._resolved_base_url(),
            "max_retries": self.max_retries,
            "temperature": 0,
        }

    def _resolved_base_url(self) -> str:
        return (
            self.base_url or os.environ.get("VFATES_JUDGE_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")

    def _resolved_api_key(self) -> str:
        key = self._api_key if self._api_key is not None else os.environ.get("VFATES_JUDGE_API_KEY")
        if not key:
            raise RuntimeError(
                "LLMJudge requires an API key via the api_key argument or VFATES_JUDGE_API_KEY"
            )
        return key

    def judge(self, prompt: str, response: str) -> int:
        url = self._resolved_base_url() + "/chat/completions"
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": self.rubric},
                {"role": "user", "content": format_judge_user(prompt, response)},
            ],
        }
        payload = json.dumps(body).encode("utf-8")
        key = self._resolved_api_key()
        last_text = ""
        for attempt in range(self.max_retries):
            if attempt:
                time.sleep(self.backoff_s * (2 ** (attempt - 1)))
            try:
                last_text = _post_chat(url, payload, key, self.timeout_s)
            except (urllib.error.URLError, TimeoutError, OSError):
                continue
            verdict = parse_yes_no(last_text)
            if verdict is not None:
                return verdict
        return 0


def _post_chat(url: str, payload: bytes, api_key: str, timeout_s: float) -> str:
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Retry timeouts and 5xx. Other 4xx responses are caller errors.
        try:
            detail = exc.read().decode("utf-8", "replace")[:500]
        finally:
            exc.close()
        if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
            raise RuntimeError(f"judge HTTP {exc.code}: {detail}") from exc
        raise urllib.error.URLError(f"judge HTTP {exc.code}") from exc
    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"judge response missing message content: {data!r}"[:500]) from exc


class LocalJudge(Judge):
    """YES/NO judge backed by any :class:`~vfates.gen.base.Generator` at temperature 0."""

    def __init__(
        self, generator: Generator, rubric: str | None = None, *, max_tokens: int = 16
    ) -> None:
        self.generator = generator
        self.rubric = rubric or DEFAULT_RUBRIC
        self.max_tokens = int(max_tokens)
        model_id = getattr(generator, "model_id", None) or type(generator).__name__
        self.name = f"local:{model_id}"

    def describe(self) -> dict[str, Any]:
        info = self.generator.describe() if hasattr(self.generator, "describe") else {}
        return {
            "name": self.name,
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "generator": info,
        }

    def judge(self, prompt: str, response: str) -> int:
        outs = self.generator.generate(
            [format_judge_user(prompt, response)],
            1,
            max_tokens=self.max_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=0,
            system_prompt=self.rubric,
        )
        if not outs or not outs[0]:
            return 0
        verdict = parse_yes_no(outs[0][0].text)
        return 0 if verdict is None else verdict


class CachedJudge(Judge):
    """JSONL cache in front of another judge.

    The key is ``sha256`` of the inner judge id (``name`` plus :meth:`describe`),
    the prompt, and the response. Hits do not call the inner judge.
    """

    def __init__(self, judge: Judge, cache_path: str | Path) -> None:
        self.inner = judge
        self.cache_path = Path(cache_path)
        self.name = f"cached:{judge.name}"
        self._lock = threading.Lock()
        self._memory = self._load()

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cache_path": str(self.cache_path),
            "inner": self.inner.describe(),
        }

    def _judge_id(self) -> str:
        return canonical_json({"name": self.inner.name, "describe": self.inner.describe()})

    def _key(self, prompt: str, response: str) -> str:
        return sha256_text(canonical_json([self._judge_id(), prompt, response]))

    def _load(self) -> dict[str, int]:
        if not self.cache_path.exists():
            return {}
        loaded: dict[str, int] = {}
        for row in iter_jsonl(self.cache_path):
            loaded[str(row["key"])] = int(row["verdict"])
        return loaded

    def judge(self, prompt: str, response: str) -> int:
        key = self._key(prompt, response)
        with self._lock:
            cached = self._memory.get(key)
            if cached is not None:
                return cached
        verdict = int(self.inner.judge(prompt, response))
        with self._lock:
            # Another thread may have filled the key while we judged.
            if key not in self._memory:
                self._memory[key] = verdict
                append_jsonl(self.cache_path, {"key": key, "verdict": verdict})
            return self._memory[key]
