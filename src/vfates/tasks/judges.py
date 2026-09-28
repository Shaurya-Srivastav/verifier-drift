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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vfates.gen.base import Generator
from vfates.tasks.answer import extract_boxed
from vfates.utils.io import append_jsonl, canonical_json, iter_jsonl, sha256_text

DEFAULT_RUBRIC = (
    "You are a strict content judge. Decide whether the assistant's response "
    "substantively fulfils the user's request, apart from formatting constraints "
    "(capitalization, keyword counts, bullet counts, JSON wrapping, and similar). "
    "Keyword stuffing, copied instructions, and degenerate repetition do not count "
    "as fulfilment. Answer with a single token: YES or NO."
)

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_YES_NO = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)
# Constraints whose compliant answer can be short or structured. Those rows are
# judged on repetition and padding only. Content constraints keep the length
# and comma-list rules.
_FORMAT_CONSTRAINTS = frozenset(
    {
        "validate_json_format",
        "validate_uppercase",
        "validate_lowercase",
        "validate_quotation",
        "validate_no_commas",
        "validate_choice",
        "validate_end",
        "validate_title",
        "validate_highlighted_sections",
        "validate_placeholders",
        "validate_sections",
        "verify_bullet_points",
        "verify_paragraph_count",
        "validate_paragraphs",
        "validate_two_responses",
        "validate_repeat_prompt",
        "verify_postscript",
        "validate_word_constraint",
        "verify_sentence_constraint",
        "verify_letter_frequency",
        "validate_frequency_capital_words",
    }
)


def format_judge_user(prompt: str, response: str) -> str:
    """User turn sent to an LLM judge."""
    return f"USER REQUEST:\n{prompt}\n\nASSISTANT RESPONSE:\n{response}\n"


def parse_yes_no(text: str) -> int | None:
    """Map a judge reply to 1 (YES) or 0 (NO).

    None unless the reply contains exactly one ``YES`` or ``NO`` token. A
    sentence that also says the other token (``YES or NO``) is not a verdict.
    One token inside a longer sentence still counts.
    """
    matches = list(_YES_NO.finditer(text or ""))
    if len(matches) != 1:
        return None
    return 1 if matches[0].group(1).upper() == "YES" else 0


class Judge(ABC):
    """Binary content judgement: 1 substantive, 0 not."""

    name: str = "judge"

    @abstractmethod
    def judge(self, prompt: str, response: str, **context: Any) -> int:
        """1 if ``response`` substantively fulfils ``prompt``, else 0.

        ``context`` may carry ``func_name`` and ``checker_args`` from an IFEval
        row. Judges that do not use them ignore the extra keywords.
        """

    def judge_result(self, prompt: str, response: str, **context: Any) -> tuple[int, bool]:
        """``(verdict, parsed)``.

        A heuristic verdict is always parsed. YES/NO judges score an unparsed
        reply as 0 and report ``parsed=False``.
        """
        return int(self.judge(prompt, response, **context)), True

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}


class HeuristicJudge(Judge):
    """Flag keyword stuffing, padding, and degenerate repetition without an LLM.

    A response scores 1 only when every rule below passes. Defaults are aimed
    at step-0 prose, not at responses that have already been optimized to
    repeat a constraint keyword dozens of times.

    * **Content words** (``min_content_words``, default 12). Alphabetic words
      (regex ``[A-Za-z][A-Za-z'-]*``, so a single letter counts) must number
      at least this many. Symbol padding and empty replies fail here. Format
      constraints and JSON documents skip this rule.
    * **Repetition** (``max_repeat_ratio``, default 0.4). On 16 or more
      whitespace tokens (lowercased, punctuation stripped): fail if the most
      common token's share exceeds ``max_repeat_ratio``, or if the type-token
      ratio is strictly below ``1 - max_repeat_ratio``. Tokens the checker
      required to be repeated (``verify_keyword_frequency``'s ``word``, or
      ``required_word``) are left out of this count. The Tulu 3 loop
      ``timberlike, unconnectedly Conversation N User :`` fails the type-token
      test (a handful of tokens recycled across the reply).
    * **Keyword density** (``max_keyword_density``, default 0.5). Split on
      commas. A shard is a segment with at most three alphabetic words and no
      ``.?!``. With at least four segments, fail when the shard fraction is
      strictly above ``max_keyword_density``. This catches a comma-separated
      list of distinct keywords, which repetition statistics miss. Format
      constraints and JSON documents skip this rule.
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

    def judge(self, prompt: str, response: str, **context: Any) -> int:
        # Stuffing is a property of the response. ``prompt`` is unused on purpose.
        del prompt
        text = response or ""
        func_name = context.get("func_name")
        raw_args = context.get("checker_args")
        args = raw_args if isinstance(raw_args, dict) else {}
        required: set[str] = set()
        explicit = context.get("required_word")
        if isinstance(explicit, str) and explicit:
            required.add(explicit)
        if func_name == "verify_keyword_frequency":
            word = args.get("word")
            if isinstance(word, str) and word:
                required.add(word)
        format_row = (
            isinstance(func_name, str) and func_name in _FORMAT_CONSTRAINTS
        ) or _looks_like_json(text)
        if not format_row and len(_WORD.findall(text)) < self.min_content_words:
            return 0
        if len(text) >= 40:
            alnum = sum(1 for ch in text if ch.isalnum())
            if alnum / len(text) < 0.45:
                return 0
        if _repetition(text, self.max_repeat_ratio, ignore=required):
            return 0
        if not format_row and _keyword_stuffing(text, self.max_keyword_density):
            return 0
        return 1


def _tokens(text: str) -> list[str]:
    raw = re.findall(r"\S+", text.lower())
    return [
        re.sub(r"^[^\w]+|[^\w]+$", "", token)
        for token in raw
        if re.sub(r"^[^\w]+|[^\w]+$", "", token)
    ]


def _looks_like_json(text: str) -> bool:
    """True when ``text`` is one JSON document. Length is not a stuffing signal."""
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return False
    try:
        json.loads(stripped)
    except ValueError:
        return False
    return True


def _repetition(text: str, max_repeat_ratio: float, ignore: set[str] | None = None) -> bool:
    tokens = _tokens(text)
    if ignore:
        skipped = {word.lower() for word in ignore}
        tokens = [token for token in tokens if token not in skipped]
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
    replies are retried with exponential backoff. A reply that does not contain
    exactly one YES or NO token scores 0. ``max_tokens`` defaults to 16 so a
    chatty completion is out of scope. ``revision`` and
    ``chat_template_kwargs`` are recorded by :meth:`describe`. When
    ``chat_template_kwargs`` is set, it is a top-level field of the
    chat-completions body (vLLM reads it there).
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        rubric: str | None = None,
        *,
        revision: str | None = None,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        max_retries: int = 3,
        backoff_s: float = 0.5,
        timeout_s: float = 60.0,
        max_tokens: int = 16,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be positive")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if chat_template_kwargs is not None and not isinstance(chat_template_kwargs, Mapping):
            raise TypeError("chat_template_kwargs must be a mapping or None")
        self.model = model
        self.revision = None if revision is None else str(revision)
        self.chat_template_kwargs = (
            None if chat_template_kwargs is None else dict(chat_template_kwargs)
        )
        self.base_url = base_url
        self._api_key = api_key
        self.rubric = rubric or DEFAULT_RUBRIC
        self.max_retries = int(max_retries)
        self.backoff_s = float(backoff_s)
        self.timeout_s = float(timeout_s)
        self.max_tokens = int(max_tokens)
        self.name = f"llm:{model}"

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "revision": self.revision,
            "base_url": self._resolved_base_url(),
            "max_retries": self.max_retries,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "rubric_sha256": sha256_text(self.rubric),
            "chat_template_kwargs": (
                None if self.chat_template_kwargs is None else dict(self.chat_template_kwargs)
            ),
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

    def complete(self, prompt: str, response: str) -> str:
        """Raw assistant text. Empty when every attempt fails transport.

        Retries stop early once the text contains exactly one YES or NO token.
        The last unparsed reply is returned as-is.
        """
        url = self._resolved_base_url() + "/chat/completions"
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self.rubric},
                {"role": "user", "content": format_judge_user(prompt, response)},
            ],
        }
        self._attach_chat_template_kwargs(body)
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
            if parse_yes_no(last_text) is not None:
                return last_text
        return last_text

    def judge_result(self, prompt: str, response: str, **context: Any) -> tuple[int, bool]:
        del context
        verdict = parse_yes_no(self.complete(prompt, response))
        if verdict is None:
            return 0, False
        return verdict, True

    def judge(self, prompt: str, response: str, **context: Any) -> int:
        verdict, _parsed = self.judge_result(prompt, response, **context)
        return verdict

    def complete_prompt(self, prompt: str) -> str:
        """Raw completion text for ``prompt``. Empty when every attempt fails.

        This is ``POST {base_url}/completions`` with ``temperature`` 0. It does
        not look for a YES or NO token. A chat judge keeps using
        :meth:`complete`.
        """
        url = self._resolved_base_url() + "/completions"
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "prompt": prompt,
        }
        payload = json.dumps(body).encode("utf-8")
        key = self._resolved_api_key()
        last_text = ""
        for attempt in range(self.max_retries):
            if attempt:
                time.sleep(self.backoff_s * (2 ** (attempt - 1)))
            try:
                last_text = _post_completion(url, payload, key, self.timeout_s)
            except (urllib.error.URLError, TimeoutError, OSError):
                continue
            return last_text
        return last_text

    def complete_messages(self, messages: list[Mapping[str, Any]]) -> str:
        """Raw assistant text for an explicit chat-completions message list.

        Unlike :meth:`complete`, this does not stop early on a YES or NO
        token. The caller parses the reply. The first successful transport
        result is returned. ``chat_template_kwargs`` is a top-level body
        field when set.
        """
        url = self._resolved_base_url() + "/chat/completions"
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [dict(message) for message in messages],
        }
        self._attach_chat_template_kwargs(body)
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
            return last_text
        return last_text

    def _attach_chat_template_kwargs(self, body: dict[str, Any]) -> None:
        """vLLM reads ``chat_template_kwargs`` on the chat-completions body.

        ``extra_body`` is an OpenAI-client field and would not apply
        ``enable_thinking: false``. The key is omitted when unset.
        """
        if self.chat_template_kwargs is not None:
            body["chat_template_kwargs"] = dict(self.chat_template_kwargs)


def _open_json(url: str, payload: bytes, api_key: str, timeout_s: float) -> Any:
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
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Retry timeouts and 5xx. Other 4xx responses are caller errors.
        try:
            detail = exc.read().decode("utf-8", "replace")[:500]
        finally:
            exc.close()
        if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
            raise RuntimeError(f"judge HTTP {exc.code}: {detail}") from exc
        raise urllib.error.URLError(f"judge HTTP {exc.code}") from exc


def _post_chat(url: str, payload: bytes, api_key: str, timeout_s: float) -> str:
    data = _open_json(url, payload, api_key, timeout_s)
    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"judge response missing message content: {data!r}"[:500]) from exc


def _post_completion(url: str, payload: bytes, api_key: str, timeout_s: float) -> str:
    """Raw ``choices[0].text`` from an OpenAI-compatible ``/completions`` call."""
    data = _open_json(url, payload, api_key, timeout_s)
    try:
        return str(data["choices"][0]["text"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"judge response missing completion text: {data!r}"[:500]) from exc


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
            "rubric_sha256": sha256_text(self.rubric),
            "generator": info,
        }

    def complete(self, prompt: str, response: str) -> str:
        """Raw generator text. Empty when the generator returns no completion."""
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
            return ""
        return str(outs[0][0].text)

    def judge_result(self, prompt: str, response: str, **context: Any) -> tuple[int, bool]:
        del context
        verdict = parse_yes_no(self.complete(prompt, response))
        if verdict is None:
            return 0, False
        return verdict, True

    def judge(self, prompt: str, response: str, **context: Any) -> int:
        verdict, _parsed = self.judge_result(prompt, response, **context)
        return verdict


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

    def _key(self, prompt: str, response: str, context: dict[str, Any]) -> str:
        return sha256_text(canonical_json([self._judge_id(), prompt, response, context]))

    def _load(self) -> dict[str, int]:
        if not self.cache_path.exists():
            return {}
        loaded: dict[str, int] = {}
        for row in iter_jsonl(self.cache_path):
            loaded[str(row["key"])] = int(row["verdict"])
        return loaded

    def judge(self, prompt: str, response: str, **context: Any) -> int:
        key = self._key(prompt, response, context)
        with self._lock:
            cached = self._memory.get(key)
            if cached is not None:
                return cached
        verdict = int(self.inner.judge(prompt, response, **context))
        with self._lock:
            # Another thread may have filled the key while we judged.
            if key not in self._memory:
                self._memory[key] = verdict
                append_jsonl(self.cache_path, {"key": key, "verdict": verdict})
            return self._memory[key]


# TIGER-Lab/general-verifier model card, also General-Reasoner's
# VERIFIER_PROMPT_TEMPLATE. The card generates with max_new_tokens 1024 and
# temperature 0. General-Reasoner's SamplingParams uses max_tokens 2048; this
# client follows the card.
GENERAL_VERIFIER_MODEL = "TIGER-Lab/general-verifier"
GENERAL_VERIFIER_REVISION = "dc59c0c82360170cab0be91c91ae545ff32ea896"
GENERAL_VERIFIER_MAX_TOKENS = 1024
VERIFIER_PASS_TAG = "Final Decision: Yes"
VERIFIER_PROMPT_TEMPLATE = (
    "User: ### Question: {question}\n\n"
    "### Ground Truth Answer: {ground_truth}\n\n"
    "### Student Answer: {student_answer}\n\n"
    "For the above question, please verify if the student's answer is equivalent "
    "to the ground truth answer.\n"
    "Do not solve the question by yourself; just check if the student's answer "
    "is equivalent to the ground truth answer.\n"
    'If the student\'s answer is correct, output "Final Decision: Yes". '
    'If the student\'s answer is incorrect, output "Final Decision: No". '
    "Assistant:"
)

# General-Reasoner extract_last_final_answer, in that order. Each pattern
# requires a trailing newline. Stop tokens are stripped from the capture.
_FINAL_ANSWER_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"Final Answer:\s*((?:[^<]|<[^<])*?)\n",
        r"Final Answer is:\s*((?:[^<]|<[^<])*?)\n",
        r"The answer is:\s*((?:[^<]|<[^<])*?)\n",
        r"Answer:\s*((?:[^<]|<[^<])*?)\n",
        r"Solution:\s*((?:[^<]|<[^<])*?)\n",
        r"The solution is:\s*((?:[^<]|<[^<])*?)\n",
    )
)
_STOP_WORDS = ("</s>", "<|im_end|>", "<|endoftext|>")


def format_verifier_prompt(question: str, ground_truth: str, student_answer: str) -> str:
    """The general-verifier completion prompt, filled in."""
    return VERIFIER_PROMPT_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        student_answer=student_answer,
    )


def general_verifier_bit(text: str) -> int:
    """1 when ``text`` contains ``Final Decision: Yes``, else 0.

    This is General-Reasoner's check (``VERIFIER_PASS_TAG in verification``).
    A missing tag is a negative verdict, not an unparsed reply.
    """
    return 1 if VERIFIER_PASS_TAG in (text or "") else 0


def verifier_cache_key(model: str, revision: str, prompt: str) -> str:
    """Cache key ``(model id, revision, sha256 of the filled prompt)``."""
    return sha256_text(canonical_json([model, revision, sha256_text(prompt)]))


# Zhao et al., arXiv:2507.08794, Hub model card at revision
# 9fd43a7a4194a703d02be3fbdcfc276cb046d8e4. Generative judge fine-tuned from
# Qwen/Qwen2.5-7B-Instruct. The card's prompt is a chat user message under
# the system string "You are a helpful assistant." Trailing double spaces are
# the card's line breaks. The generation must be strictly YES or NO.
MASTER_RM_MODEL = "sarosavo/Master-RM"
MASTER_RM_REVISION = "9fd43a7a4194a703d02be3fbdcfc276cb046d8e4"
MASTER_RM_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MASTER_RM_SYSTEM = "You are a helpful assistant."
MASTER_RM_MAX_TOKENS = 16
_MASTER_RM_PROMPT_LINES: tuple[str, ...] = (
    "Given a problem, determine whether the final answer in the provided "
    "(incomplete) solution process matches the reference answer.  ",
    "The reference answer may be one single option character (e.g., A, B, C, D), "
    "a numerical value, an expression, or a list of answers if multiple "
    "questions are involved.  ",
    "**The reference answer may be in Chinese or another language, but your "
    "evaluation should be language-agnostic.**  ",
    "",
    "Your task:  ",
    "- Compare the final output of the solution process with the reference answer.  ",
    "- If they **match exactly**, output **YES**.  ",
    "- If they **do not match**, output **NO**.  ",
    "- If the solution process is unclear, incomplete, or ambiguous, assume it "
    "is incorrect and output **NO**.  ",
    "",
    "Your output must be strictly **'YES'** or **'NO'**, with no additional "
    "words, punctuation, or explanation.  ",
    "",
    "---",
    "",
    "**Question:**  ",
    "{question}  ",
    "",
    "**Solution Process (Final Step Only):**  ",
    "{response}  ",
    "",
    "**Reference Answer:**  ",
    "{reference}  ",
    "",
    "**Output:**  ",
)
MASTER_RM_PROMPT_TEMPLATE = "\n" + "\n".join(_MASTER_RM_PROMPT_LINES) + "\n"


def format_master_rm_prompt(question: str, response: str, reference: str) -> str:
    """Fill the Master-RM card prompt. Braces in the fields are kept."""
    return (
        MASTER_RM_PROMPT_TEMPLATE.replace("{question}", question)
        .replace("{response}", response)
        .replace("{reference}", reference)
    )


def parse_master_rm(text: str) -> int:
    """1 only when the stripped reply is exactly ``YES``.

    The model card requires the generation to be strictly ``YES`` or ``NO``
    with no extra words or punctuation. ``NO`` and every other string score 0.
    """
    return 1 if (text or "").strip() == "YES" else 0


def extract_student_answer(text: str) -> str:
    """Student answer placed in the general-verifier prompt.

    Order: last closed ``\\boxed`` / ``\\fbox`` (nested braces, via
    :func:`vfates.tasks.answer.extract_boxed`), else General-Reasoner's
    final-answer patterns, else the stripped response, else ``No Answer``.

    General-Reasoner interpolates Python ``None`` into the prompt when
    extraction fails, then later substitutes ``No Answer`` only for its
    length penalty. This helper never sends the string ``None``: a missing
    extraction with a non-empty response sends that response, and an empty
    response sends ``No Answer``.
    """
    boxed = extract_boxed(text or "")
    if boxed is not None and boxed.strip():
        return boxed.strip()
    last_match: str | None = None
    last_at = -1
    for pattern in _FINAL_ANSWER_PATTERNS:
        for match in pattern.finditer(text or ""):
            if match.start() > last_at:
                last_at = match.start()
                last_match = match.group(1).strip()
    if last_match:
        for stop in _STOP_WORDS:
            if last_match.endswith(stop):
                last_match = last_match[: -len(stop)].strip()
        if last_match:
            return last_match
    stripped = (text or "").strip()
    return stripped or "No Answer"


def _normalise_judge_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


class HeuristicVerifierJudge(Judge):
    """Deterministic stand-in for the general verifier.

    Accepts when the extracted student answer and the gold normalise to the
    same string (strip, casefold, collapse whitespace). It does not call
    math-verify and it does not apply a numeric tolerance. The reply is
    ``Final Decision: Yes`` or ``Final Decision: No``, so the same tag parser
    as the model judge reads it.
    """

    name = "heuristic_verifier"

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "type": "heuristic"}

    def complete_decision(self, question: str, response: str, gold: str) -> str:
        del question
        student = extract_student_answer(response)
        same = student != "No Answer" and _normalise_judge_text(student) == _normalise_judge_text(
            gold
        )
        if same:
            return "Final Decision: Yes"
        return "Final Decision: No"

    def judge(self, prompt: str, response: str, **context: Any) -> int:
        gold = str(context.get("gold", ""))
        return general_verifier_bit(self.complete_decision(prompt, response, gold))


class GeneralVerifierJudge(Judge):
    """TIGER-Lab/general-verifier over an OpenAI-compatible completions endpoint.

    ``base_url`` defaults to ``VFATES_JUDGE_BASE_URL``. The API key is
    ``VFATES_JUDGE_API_KEY`` (a local server still needs some non-empty value).
    Temperature is 0. ``max_tokens`` defaults to 1024, the model card's
    ``max_new_tokens``. The verdict is 1 exactly when the reply contains
    ``Final Decision: Yes``.

    The verdict cache is keyed by :func:`verifier_cache_key`. Hits do not call
    the endpoint. ``cache_path`` also appends JSONL rows ``{key, text}``.
    """

    def __init__(
        self,
        model: str = GENERAL_VERIFIER_MODEL,
        revision: str = GENERAL_VERIFIER_REVISION,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        max_tokens: int = GENERAL_VERIFIER_MAX_TOKENS,
        max_retries: int = 3,
        backoff_s: float = 0.5,
        timeout_s: float = 120.0,
        cache_path: str | Path | None = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.model = model
        self.revision = revision
        self.max_tokens = int(max_tokens)
        self.cache_path = None if cache_path is None else Path(cache_path)
        self.name = f"general_verifier:{model}"
        self._client = LLMJudge(
            model,
            base_url=base_url,
            api_key=api_key,
            revision=self.revision,
            max_retries=max_retries,
            backoff_s=backoff_s,
            timeout_s=timeout_s,
            max_tokens=self.max_tokens,
        )
        self._lock = threading.Lock()
        self._memory: dict[str, str] = {}
        if self.cache_path is not None and self.cache_path.exists():
            for row in iter_jsonl(self.cache_path):
                self._memory[str(row["key"])] = str(row["text"])

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "llm",
            "model": self.model,
            "revision": self.revision,
            "base_url": self._client._resolved_base_url(),
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "max_retries": self._client.max_retries,
            "timeout_s": self._client.timeout_s,
            "prompt_sha256": sha256_text(VERIFIER_PROMPT_TEMPLATE),
            "cache_path": None if self.cache_path is None else str(self.cache_path),
        }

    def complete_decision(self, question: str, response: str, gold: str) -> str:
        prompt = format_verifier_prompt(question, gold, extract_student_answer(response))
        key = verifier_cache_key(self.model, self.revision, prompt)
        with self._lock:
            cached = self._memory.get(key)
            if cached is not None:
                return cached
        text = self._client.complete_prompt(prompt)
        with self._lock:
            if key not in self._memory:
                self._memory[key] = text
                if self.cache_path is not None:
                    append_jsonl(self.cache_path, {"key": key, "text": text})
            return self._memory[key]

    def judge(self, prompt: str, response: str, **context: Any) -> int:
        gold = str(context.get("gold", ""))
        return general_verifier_bit(self.complete_decision(prompt, response, gold))


class MasterRMJudge(Judge):
    """sarosavo/Master-RM, the robust generative judge of arXiv:2507.08794.

    Chat completions, temperature 0, ``max_tokens`` 16. The system message is
    ``You are a helpful assistant.`` The user message is the model card's
    prompt with the full completion in the solution-process slot (master-key
    openers live in that prefix). The verdict is 1 only when the stripped
    reply is exactly ``YES``.

    The verdict cache is keyed by :func:`verifier_cache_key`. Hits do not call
    the endpoint.
    """

    def __init__(
        self,
        model: str = MASTER_RM_MODEL,
        revision: str = MASTER_RM_REVISION,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        max_tokens: int = MASTER_RM_MAX_TOKENS,
        max_retries: int = 3,
        backoff_s: float = 0.5,
        timeout_s: float = 120.0,
        cache_path: str | Path | None = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.model = model
        self.revision = revision
        self.max_tokens = int(max_tokens)
        self.cache_path = None if cache_path is None else Path(cache_path)
        self.name = f"master_rm:{model}"
        self._client = LLMJudge(
            model,
            base_url=base_url,
            api_key=api_key,
            revision=self.revision,
            chat_template_kwargs=chat_template_kwargs,
            max_retries=max_retries,
            backoff_s=backoff_s,
            timeout_s=timeout_s,
            max_tokens=self.max_tokens,
        )
        self._lock = threading.Lock()
        self._memory: dict[str, str] = {}
        if self.cache_path is not None and self.cache_path.exists():
            for row in iter_jsonl(self.cache_path):
                self._memory[str(row["key"])] = str(row["text"])

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "master_rm",
            "model": self.model,
            "revision": self.revision,
            "base_model": MASTER_RM_BASE_MODEL,
            "base_url": self._client._resolved_base_url(),
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "max_retries": self._client.max_retries,
            "timeout_s": self._client.timeout_s,
            "system": MASTER_RM_SYSTEM,
            "prompt_sha256": sha256_text(MASTER_RM_PROMPT_TEMPLATE),
            "chat_template_kwargs": (
                None
                if self._client.chat_template_kwargs is None
                else dict(self._client.chat_template_kwargs)
            ),
            "cache_path": None if self.cache_path is None else str(self.cache_path),
            "parse": "strip_exact_YES",
        }

    def complete_decision(self, question: str, response: str, reference: str) -> str:
        prompt = format_master_rm_prompt(question, response, reference)
        key = verifier_cache_key(self.model, self.revision, prompt)
        with self._lock:
            cached = self._memory.get(key)
            if cached is not None:
                return cached
        text = self._client.complete_messages(
            [
                {"role": "system", "content": MASTER_RM_SYSTEM},
                {"role": "user", "content": prompt},
            ]
        )
        with self._lock:
            if key not in self._memory:
                self._memory[key] = text
                if self.cache_path is not None:
                    append_jsonl(self.cache_path, {"key": key, "text": text})
            return self._memory[key]

    def judge(self, prompt: str, response: str, **context: Any) -> int:
        gold = str(context.get("gold", ""))
        return parse_master_rm(self.complete_decision(prompt, response, gold))


def judge_from_description(described: Mapping[str, Any] | None) -> Judge:
    """Rebuild a judge from its :meth:`Judge.describe` dict."""
    if not isinstance(described, Mapping):
        return HeuristicVerifierJudge()
    if described.get("type") == "heuristic" or described.get("name") == "heuristic_verifier":
        return HeuristicVerifierJudge()
    if described.get("type") == "master_rm" or str(described.get("name", "")).startswith(
        "master_rm:"
    ):
        kwargs = described.get("chat_template_kwargs")
        return MasterRMJudge(
            model=str(described.get("model", MASTER_RM_MODEL)),
            revision=str(described.get("revision", MASTER_RM_REVISION)),
            base_url=None if described.get("base_url") is None else str(described["base_url"]),
            chat_template_kwargs=None if kwargs is None else dict(kwargs),
            max_tokens=int(described.get("max_tokens", MASTER_RM_MAX_TOKENS)),
            max_retries=int(described.get("max_retries", 3)),
            timeout_s=float(described.get("timeout_s", 120.0)),
            cache_path=described.get("cache_path"),
        )
    if described.get("type") == "llm" or "model" in described:
        return GeneralVerifierJudge(
            model=str(described.get("model", GENERAL_VERIFIER_MODEL)),
            revision=str(described.get("revision", GENERAL_VERIFIER_REVISION)),
            base_url=None if described.get("base_url") is None else str(described["base_url"]),
            max_tokens=int(described.get("max_tokens", GENERAL_VERIFIER_MAX_TOKENS)),
            max_retries=int(described.get("max_retries", 3)),
            timeout_s=float(described.get("timeout_s", 120.0)),
            cache_path=described.get("cache_path"),
        )
    return HeuristicVerifierJudge()
