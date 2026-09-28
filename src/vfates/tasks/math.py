"""Math task on SynthLabsAI/Big-Math-RL-Verified, with a DeepScaleR backup.

The oracle is math-verify on the last boxed answer, with no extra relative
tolerance. A 5 second CPU limit (``SIGVTALRM`` in a worker, plus a parent wall
clock) grades a hung comparison as incorrect and caches ``(gold, answer)``.

P2 picks one difficulty bucket on ``llama8b_solve_rate``. The preregistered
half-open buckets are ``[0.0, 0.2)``, ``[0.2, 0.4)``, ``[0.4, 0.6)`` and
``[0.6, 0.8)``. ``difficulty_bucket: null`` keeps every rate, including
unrated rows and rates in ``[0.8, 1]``, which no bucket covers. A confirmatory
cell sets one bucket before the audit. DeepScaleR has no solve rate; a bucket
is not applied when that field is absent, and AIME rows are dropped by the
same normalised-statement decontamination as Big-Math.

Filters, in order: empty answer, proof, multiple choice, non-closed-form
answer, difficulty, an ``aime`` source tag, exact duplicate, normalised
duplicate, then statement overlap with MATH-500, GSM8K test, and AIME 1983 to
2024. ``filter_counts`` records how many rows each step removed.

Split: ``u01(split_salt, problem_id)`` with ``problem_id`` the SHA-256 of the
normalised statement. ``dev`` is ``[0, dev_fraction)``, ``heldout`` the next
``heldout_fraction``, and the rest ``train``. Defaults are 0.02 and 0.04.
``problems`` orders a split by ``u01(seed, split_salt, problem_id)`` and does
not use ``seed`` for membership.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import math
import os
import re
import select
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

from vfates.tasks.answer import (
    extract_all_boxed,
    extract_boxed,
    hash_split,
    normalise_statement,
    parse_decimal,
    statement_id,
)
from vfates.tasks.base import Task
from vfates.types import Parsed, Problem
from vfates.utils.hashing import u01
from vfates.utils.io import sha256_text

BIG_MATH_DATASET = "SynthLabsAI/Big-Math-RL-Verified"
BIG_MATH_REVISION = "c75d2f117cddfecb6bd08756e61e508e59732b21"
DEEPSCALER_DATASET = "agentica-org/DeepScaleR-Preview-Dataset"
DEEPSCALER_REVISION = "b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29"
MATH500_DATASET = "HuggingFaceH4/MATH-500"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
GSM8K_DATASET = "openai/gsm8k"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
AIME_DATASET = "di-zhang-fdu/AIME_1983_2024"
AIME_REVISION = "3e2cc86390666c5c756622afc0eeb9e6194496bc"
AIME_2024_DATASET = "AI-MO/aimo-validation-aime"
AIME_2024_REVISION = "13f9e12f613e720c2a2b2f345dd04b998a29494d"

# P2 picks one. Rates in [0.8, 1] are outside every bucket on purpose.
DIFFICULTY_BUCKETS: tuple[tuple[float, float], ...] = (
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
)

DEFAULT_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
SPLIT_SALT = "math/split"
VERIFY_TIMEOUT_S = 5.0

ERROR_CATEGORIES: tuple[str, ...] = (
    "no_box",
    "multiple_distinct_boxes",
    "large_magnitude_gold",
    "near_miss_numeric",
    "fraction_vs_decimal",
    "whitespace_or_latex_spacing",
    "symbolic_gold",
    "units_or_text_in_box",
)

_COUNT_KEYS: tuple[str, ...] = (
    "loaded",
    "dropped_empty_answer",
    "dropped_proof",
    "dropped_multiple_choice",
    "dropped_non_closed_form",
    "dropped_difficulty",
    "dropped_unrated",
    "dropped_aime_source",
    "dropped_exact_duplicate",
    "dropped_normalised_duplicate",
    "dropped_math500",
    "dropped_gsm8k",
    "dropped_aime",
    "kept",
    "n_train",
    "n_heldout",
    "n_dev",
)

_FIELDS = frozenset(
    {
        "name",
        "source",
        "dataset",
        "revision",
        "difficulty_bucket",
        "split_salt",
        "instruction",
        "dev_fraction",
        "heldout_fraction",
        "verify_timeout_s",
        "decontaminate",
        "contamination",
        "rows",
    }
)

_PROOF_ANSWERS = frozenset({"proof", "proved", "qed", "q.e.d"})
_FRAC = re.compile(r"^\\(?:d|t|c)?frac\{([+-]?(?:\d+(?:\.\d+)?))\}\{([+-]?(?:\d+(?:\.\d+)?))\}$")
_SLASH = re.compile(r"^([+-]?(?:\d+(?:\.\d+)?))/([+-]?(?:\d+(?:\.\d+)?))$")
_FRAC_FORM = re.compile(r"\\[dtc]?frac\b|/")
_DECIMAL_FORM = re.compile(r"\d\.\d")
_COLLAPSE = re.compile(r"\\[,!;:]|\\ |~|\s+")
_UNITS = re.compile(
    r"\\text\b|\\mathrm\s*\{[A-Za-z]|\\operatorname\b|"
    r"\b(?:cm|mm|km|kg|mg|ml|meters|metres|hours|hour|dollars|cents|units)\b",
    re.IGNORECASE,
)
_MCQ = tuple(
    re.compile(rf"(?m)(?:^|\n)\s*(?:\\?\({letter}\)|{letter}[.)])\s+\S") for letter in "ABCDE"
)

_CONTAMINATION: dict[str, set[str]] | None = None
_CONTAMINATION_LOCK = threading.Lock()
_VERIFY_LOCK = threading.Lock()
_VERIFY_CACHE: dict[tuple[str, str], bool] = {}
_VERIFY_CLIENT: Any = None


def _math_verify_version() -> str:
    try:
        return version("math-verify")
    except Exception:
        return "unknown"


def _parse_bucket(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError("difficulty_bucket must be [lo, hi] or null")
    lo = float(value[0])
    hi = float(value[1])
    known = any(
        math.isclose(lo, left) and math.isclose(hi, right) for left, right in DIFFICULTY_BUCKETS
    )
    if not known:
        listed = ", ".join(f"[{left}, {right})" for left, right in DIFFICULTY_BUCKETS)
        raise ValueError(f"difficulty_bucket [{lo}, {hi}) is not preregistered ({listed})")
    return (lo, hi)


def _as_rate(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(rate):
        return None
    return rate


def _ratio(num: str, den: str) -> float | None:
    try:
        top = float(num)
        bottom = float(den)
    except ValueError:
        return None
    if not math.isfinite(top) or not math.isfinite(bottom) or bottom == 0.0:
        return None
    value = top / bottom
    if not math.isfinite(value):
        return None
    return value


def finite_real(text: str | None) -> float | None:
    """Finite real parsed from a gold or a boxed answer, else ``None``.

    Accepts a decimal (via :func:`parse_decimal`), ``\\frac`` / ``\\dfrac`` /
    ``\\tfrac`` / ``\\cfrac`` of two numbers, or ``a/b``. A symbolic string is
    ``None``.
    """
    if not isinstance(text, str):
        return None
    token = text.strip()
    if len(token) >= 2 and token[0] == "$" and token[-1] == "$" and token.count("$") == 2:
        token = token[1:-1].strip()
    parsed = parse_decimal(token)
    if parsed is not None:
        value = float(parsed)
        if math.isfinite(value):
            return value
    compact = re.sub(r"\s+", "", token)
    matched = _FRAC.fullmatch(compact) or _SLASH.fullmatch(compact)
    if matched is None:
        return None
    return _ratio(matched.group(1), matched.group(2))


def _is_proof(problem: str, answer: str) -> bool:
    folded = answer.strip().casefold().rstrip(".")
    if folded in _PROOF_ANSWERS:
        return True
    head = problem.lstrip().casefold()
    return head.startswith(("prove ", "prove that ", "show that "))


def _is_multiple_choice(problem: str) -> bool:
    return sum(1 for pattern in _MCQ if pattern.search(problem)) >= 3


def _is_non_closed(answer: str) -> bool:
    text = answer.strip()
    words = text.split()
    alpha = re.findall(r"[A-Za-z]+", text)
    if len(text) > 300 and len(words) > 20:
        return True
    return len(alpha) > 12 and re.search(r"\d|\\", text) is None


def _collapse_space(text: str) -> str:
    return _COLLAPSE.sub("", text)


def _fraction_vs_decimal(gold: str, pred: str) -> bool:
    def kind(value: str) -> str:
        if _FRAC_FORM.search(value):
            return "frac"
        if _DECIMAL_FORM.search(value):
            return "dec"
        return "other"

    return {kind(gold), kind(pred)} == {"frac", "dec"}


def _wrap_math(text: str) -> str:
    """Wrap a bare expression so math-verify's LaTeX extractor sees it.

    Intervals, sets, ``\\dfrac`` and symbols fail to parse without math
    delimiters. A string that already contains ``$`` or ``\\(`` / ``\\[`` is
    left alone.
    """
    token = text.strip()
    if not token or "$" in token or "\\(" in token or "\\[" in token:
        return token
    return f"${token}$"


def _verify_loaded(
    gold: str, answer: str, cfg: list[Any], parse: Any, verify: Any, timeout_s: float
) -> bool:
    parsing_timeout = max(1, math.ceil(timeout_s))
    gold_parsed = parse(
        _wrap_math(gold),
        extraction_config=cfg,
        parsing_timeout=parsing_timeout,
        fallback_mode="first_match",
    )
    pred_parsed = parse(
        _wrap_math(answer),
        extraction_config=cfg,
        parsing_timeout=parsing_timeout,
        fallback_mode="first_match",
    )
    if not gold_parsed or not pred_parsed:
        return False
    return bool(
        verify(
            gold_parsed,
            pred_parsed,
            float_rounding=6,
            numeric_precision=15,
            timeout_seconds=parsing_timeout,
        )
    )


class _VerifyWorker:
    """One long-lived math-verify process. A hung item is killed and restarted."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen[bytes] | None = None
        self._buf = b""

    def call(self, payload: Mapping[str, Any], wall_s: float) -> dict[str, Any]:
        for attempt in range(2):
            try:
                if self._proc is None or self._proc.poll() is not None:
                    self._start()
                assert self._proc is not None and self._proc.stdin is not None
                self._proc.stdin.write(json.dumps(payload).encode("utf-8") + b"\n")
                self._proc.stdin.flush()
                line = self._readline(wall_s)
            except (BrokenPipeError, OSError):
                self.close()
                if attempt == 1:
                    return {"ok": False, "timeout": True}
                continue
            if line is None:
                self.close()
                if attempt == 1:
                    return {"ok": False, "timeout": True}
                continue
            try:
                loaded = json.loads(line)
            except json.JSONDecodeError:
                self.close()
                if attempt == 1:
                    return {"ok": False, "timeout": True}
                continue
            if isinstance(loaded, dict):
                return loaded
            return {"ok": False, "timeout": False}
        return {"ok": False, "timeout": True}

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        self._buf = b""
        if proc is None:
            return
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=2)
        for pipe in (proc.stdin, proc.stdout):
            if pipe is not None:
                pipe.close()

    def _start(self) -> None:
        self.close()
        env = os.environ.copy()
        env["VFATES_MATH_VERIFY_WORKER"] = "1"
        self._proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from vfates.tasks.math import _worker_main; _worker_main()",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
        line = self._readline(60.0)
        if line is None:
            self.close()
            raise RuntimeError("math-verify worker failed to start")
        try:
            ready = json.loads(line)
        except json.JSONDecodeError:
            ready = {}
        if not isinstance(ready, dict) or ready.get("ready") is not True:
            self.close()
            raise RuntimeError("math-verify worker failed to start")

    def _readline(self, wall_s: float) -> bytes | None:
        assert self._proc is not None and self._proc.stdout is not None
        deadline = time.monotonic() + wall_s
        while b"\n" not in self._buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([self._proc.stdout], [], [], remaining)
            if not ready:
                return None
            chunk = os.read(self._proc.stdout.fileno(), 1 << 16)
            if not chunk:
                return None
            self._buf += chunk
        line, _, rest = self._buf.partition(b"\n")
        self._buf = rest
        return line


def _client() -> _VerifyWorker:
    global _VERIFY_CLIENT
    if _VERIFY_CLIENT is None:
        _VERIFY_CLIENT = _VerifyWorker()
        atexit.register(_VERIFY_CLIENT.close)
    return _VERIFY_CLIENT


def shutdown_math_verify_worker() -> None:
    """Stop the math-verify worker, if one was started."""
    global _VERIFY_CLIENT
    with _VERIFY_LOCK:
        if _VERIFY_CLIENT is not None:
            _VERIFY_CLIENT.close()
            _VERIFY_CLIENT = None


def math_verify_equal(gold: str, answer: str, *, timeout_s: float = VERIFY_TIMEOUT_S) -> bool:
    """math-verify equivalence of ``answer`` to ``gold``, cached by that pair.

    Extraction is ``LatexExtractionConfig(boxed_match_priority=0)`` plus
    ``ExprExtractionConfig``. Comparison uses math-verify's own
    ``float_rounding=6`` and ``numeric_precision=15`` and adds no relative
    tolerance. A CPU timeout or a worker crash is incorrect and is cached.
    """
    key = (gold, answer)
    with _VERIFY_LOCK:
        cached = _VERIFY_CACHE.get(key)
        if cached is not None:
            return cached
        if not gold.strip() or not answer.strip():
            _VERIFY_CACHE[key] = False
            return False
        result = _client().call(
            {"mode": "verify", "gold": gold, "answer": answer, "timeout_s": float(timeout_s)},
            float(timeout_s) + 2.0,
        )
        ok = bool(result.get("ok")) and not bool(result.get("timeout"))
        _VERIFY_CACHE[key] = ok
        return ok


def cpu_guard_spin(timeout_s: float = 0.25) -> str:
    """Busy-loop inside the worker until ``SIGVTALRM``. Returns ``TIMEOUT``."""
    with _VERIFY_LOCK:
        started = time.monotonic()
        result = _client().call(
            {"mode": "spin", "timeout_s": float(timeout_s)}, float(timeout_s) + 2.0
        )
        elapsed = time.monotonic() - started
    if result.get("timeout") or elapsed >= float(timeout_s):
        return "TIMEOUT"
    return "DONE"


def _worker_main() -> None:
    """stdin/stdout math-verify worker. One JSON object per line."""

    class _CpuTimeout(BaseException):
        pass

    def _handle(signum: int, frame: Any) -> None:
        del signum, frame
        raise _CpuTimeout

    if not hasattr(signal, "setitimer"):
        raise RuntimeError("math-verify CPU guard requires POSIX setitimer")
    signal.signal(signal.SIGVTALRM, _handle)
    from math_verify import parse, verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

    cfg = [LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()]
    out = sys.stdout.buffer
    out.write(b'{"ready": true}\n')
    out.flush()
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return
        req = json.loads(line)
        timeout_s = float(req.get("timeout_s", VERIFY_TIMEOUT_S))
        signal.setitimer(signal.ITIMER_VIRTUAL, timeout_s)
        try:
            if req.get("mode") == "spin":
                while True:
                    pass
            ok = _verify_loaded(str(req["gold"]), str(req["answer"]), cfg, parse, verify, timeout_s)
            payload: dict[str, Any] = {"ok": bool(ok), "timeout": False}
        except _CpuTimeout:
            payload = {"ok": False, "timeout": True}
        except Exception as exc:
            payload = {
                "ok": False,
                "timeout": type(exc).__name__ == "TimeoutException",
                "error": type(exc).__name__,
            }
        finally:
            signal.setitimer(signal.ITIMER_VIRTUAL, 0.0)
        out.write(json.dumps(payload).encode("utf-8") + b"\n")
        out.flush()


def _load_hub(
    dataset: str, revision: str, *, config: str | None = None, split: str = "train"
) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("math loading requires the datasets package") from exc
    try:
        if config is None:
            return load_dataset(dataset, split=split, revision=revision)
        return load_dataset(dataset, config, split=split, revision=revision)
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        lowered = text.lower()
        gated = any(token in lowered for token in ("gated", "401", "unauthorized", "restricted"))
        if gated or "access" in lowered:
            if dataset == BIG_MATH_DATASET:
                raise RuntimeError(
                    "SynthLabsAI/Big-Math-RL-Verified is gated. Loading it needs an HF "
                    "account that accepted the dataset terms and HF_TOKEN. "
                    f"({text}) Or use source: deepscaler."
                ) from exc
            raise RuntimeError(
                f"{dataset} revision {revision} is gated or unavailable ({text}). "
                "Set HF_TOKEN, or use source: deepscaler."
            ) from exc
        raise


def _statements(
    dataset: str, revision: str, field: str, *, config: str | None = None, split: str = "train"
) -> set[str]:
    rows = _load_hub(dataset, revision, config=config, split=split)
    found: set[str] = set()
    for row in rows:
        value = row.get(field, "")
        if isinstance(value, str) and value.strip():
            found.add(normalise_statement(value))
    return found


def load_contamination_sets() -> dict[str, set[str]]:
    """Normalised statements for MATH-500, GSM8K test, and AIME 1983 to 2024.

    AIME is ``di-zhang-fdu/AIME_1983_2024`` plus ``AI-MO/aimo-validation-aime``
    (2024 part I, which the first card does not include). Cached for the process.
    """
    global _CONTAMINATION
    with _CONTAMINATION_LOCK:
        if _CONTAMINATION is None:
            aime = _statements(AIME_DATASET, AIME_REVISION, "Question")
            aime |= _statements(AIME_2024_DATASET, AIME_2024_REVISION, "problem")
            _CONTAMINATION = {
                "math500": _statements(MATH500_DATASET, MATH500_REVISION, "problem", split="test"),
                "gsm8k": _statements(
                    GSM8K_DATASET, GSM8K_REVISION, "question", config="main", split="test"
                ),
                "aime": aime,
            }
        return _CONTAMINATION


def _normalise_contamination(raw: Mapping[str, Collection[str]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {"math500": set(), "gsm8k": set(), "aime": set()}
    for key in out:
        for text in raw.get(key, ()):
            out[key].add(normalise_statement(str(text)))
    return out


@dataclass(frozen=True)
class _Kept:
    problem_id: str
    question: str
    answer: str
    split: str
    meta: dict[str, Any]


def _blank_counts() -> dict[str, int]:
    return {key: 0 for key in _COUNT_KEYS}


class MathTask(Task):
    """Big-Math (or DeepScaleR) with a hash split and a math-verify oracle."""

    name = "math"
    error_categories = ERROR_CATEGORIES
    verifier_types = ("math_lenient", "math_strip_string")

    def __init__(
        self,
        *,
        name: str = "math",
        source: str = "big_math",
        dataset: str | None = None,
        revision: str | None = None,
        difficulty_bucket: Sequence[float] | None = None,
        split_salt: str = SPLIT_SALT,
        instruction: str = DEFAULT_INSTRUCTION,
        dev_fraction: float = 0.02,
        heldout_fraction: float = 0.04,
        verify_timeout_s: float = VERIFY_TIMEOUT_S,
        decontaminate: bool = True,
        contamination: Mapping[str, Collection[str]] | None = None,
        rows: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        if source not in ("big_math", "deepscaler"):
            raise ValueError("source must be 'big_math' or 'deepscaler'")
        if dev_fraction < 0 or heldout_fraction < 0 or dev_fraction + heldout_fraction >= 1:
            raise ValueError(
                "dev_fraction and heldout_fraction must be non-negative and sum to < 1"
            )
        if verify_timeout_s <= 0:
            raise ValueError("verify_timeout_s must be positive")
        self.name = name
        self.source = source
        self.dataset = dataset or (BIG_MATH_DATASET if source == "big_math" else DEEPSCALER_DATASET)
        self.revision = revision or (
            BIG_MATH_REVISION if source == "big_math" else DEEPSCALER_REVISION
        )
        self.difficulty_bucket = _parse_bucket(difficulty_bucket)
        self.split_salt = split_salt
        self.instruction = instruction
        self.dev_fraction = float(dev_fraction)
        self.heldout_fraction = float(heldout_fraction)
        self.verify_timeout_s = float(verify_timeout_s)
        self.decontaminate = bool(decontaminate)
        self._contamination = (
            None if contamination is None else _normalise_contamination(contamination)
        )
        self._rows = None if rows is None else [dict(row) for row in rows]
        self._items: list[_Kept] | None = None
        self.filter_counts: dict[str, int] = _blank_counts()
        self.difficulty_bucket_applied = False
        self._lock = threading.Lock()

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any]) -> MathTask:
        """Build from a config mapping, ignoring keys this task does not use."""
        return cls(**{key: value for key, value in options.items() if key in _FIELDS})

    def describe(self) -> dict[str, Any]:
        self._prepare()
        bucket = None if self.difficulty_bucket is None else list(self.difficulty_bucket)
        return {
            "name": self.name,
            "source": self.source,
            "dataset": self.dataset,
            "revision": self.revision,
            "difficulty_bucket": bucket,
            "difficulty_buckets": [list(item) for item in DIFFICULTY_BUCKETS],
            "difficulty_bucket_applied": self.difficulty_bucket_applied,
            "split_salt": self.split_salt,
            "dev_fraction": self.dev_fraction,
            "heldout_fraction": self.heldout_fraction,
            "instruction_sha256": sha256_text(self.instruction),
            "filter_counts": dict(self.filter_counts),
            "decontaminate": self.decontaminate,
            "contamination": {
                "math500": {
                    "dataset": MATH500_DATASET,
                    "revision": MATH500_REVISION,
                    "split": "test",
                },
                "gsm8k": {"dataset": GSM8K_DATASET, "revision": GSM8K_REVISION, "split": "test"},
                "aime": [
                    {"dataset": AIME_DATASET, "revision": AIME_REVISION},
                    {"dataset": AIME_2024_DATASET, "revision": AIME_2024_REVISION},
                ],
            },
            "oracle": {
                "backend": "math_verify",
                "version": _math_verify_version(),
                "boxed_match_priority": 0,
                "extraction": ["LatexExtractionConfig", "ExprExtractionConfig"],
                "float_rounding": 6,
                "numeric_precision": 15,
                "timeout_s": self.verify_timeout_s,
                "timeout": "SIGVTALRM CPU time plus a parent wall clock",
            },
        }

    def problems(self, split: str, n: int, seed: int) -> list[Problem]:
        if split not in ("train", "heldout", "dev"):
            raise ValueError("split must be 'train', 'heldout', or 'dev'")
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self._prepare()
        assert self._items is not None
        chosen = [item for item in self._items if item.split == split]
        if len(chosen) < n:
            raise ValueError(f"math split {split!r} has {len(chosen)} problems, asked for {n}")
        chosen.sort(key=lambda item: (u01(seed, self.split_salt, item.problem_id), item.problem_id))
        return [self._as_problem(item) for item in chosen[:n]]

    def prompt(self, problem: Problem) -> str:
        return problem.question.rstrip() + "\n" + self.instruction

    def parse(self, problem: Problem, text: str) -> Parsed:
        del problem
        raw = extract_boxed(text)
        if raw is None or not raw.strip():
            return Parsed(answer=None, raw=None if raw is None else raw)
        return Parsed(answer=raw.strip(), raw=raw)

    def oracle(self, problem: Problem, parsed: Parsed, text: str) -> int:
        """1 when math-verify accepts the boxed answer. No box is 0.

        A timeout is 0. The comparison is cached by ``(gold, answer)``.
        """
        del text
        if parsed.answer is None:
            return 0
        return int(
            math_verify_equal(
                problem.answer,
                parsed.raw or parsed.answer,
                timeout_s=self.verify_timeout_s,
            )
        )

    def categorize(self, problem: Problem, parsed: Parsed, text: str) -> frozenset[str]:
        """Cheap predicates aimed at where the natural verifiers disagree.

        ``large_magnitude_gold`` and ``symbolic_gold`` depend only on the gold,
        so every completion on that problem is tagged. The others depend on the
        output.
        """
        found: set[str] = set()
        if parsed.raw is None:
            found.add("no_box")
        boxes = [box.strip() for box in extract_all_boxed(text) if box.strip()]
        if len(set(boxes)) >= 2:
            found.add("multiple_distinct_boxes")
        gold_real = finite_real(problem.answer)
        pred_real = finite_real(parsed.answer)
        if gold_real is not None and abs(gold_real) >= 1e4:
            found.add("large_magnitude_gold")
        if gold_real is None and problem.answer.strip():
            found.add("symbolic_gold")
        if (
            gold_real is not None
            and pred_real is not None
            and gold_real != pred_real
            and abs(pred_real - gold_real) / max(abs(gold_real), 1e-12) <= 1e-4
        ):
            found.add("near_miss_numeric")
        boxed = parsed.raw or ""
        if boxed.strip() and _fraction_vs_decimal(problem.answer, boxed):
            found.add("fraction_vs_decimal")
        if (
            boxed
            and boxed != problem.answer
            and _collapse_space(boxed) == _collapse_space(problem.answer)
        ):
            found.add("whitespace_or_latex_spacing")
        if boxed and _UNITS.search(boxed):
            found.add("units_or_text_in_box")
        return frozenset(found)

    def abs_error(self, problem: Problem, parsed: Parsed) -> float | None:
        """Absolute distance when both gold and answer are finite reals."""
        gold = finite_real(problem.answer)
        pred = finite_real(parsed.answer)
        if gold is None or pred is None:
            return None
        return abs(pred - gold)

    def build_verifier(self, spec: Mapping[str, Any]) -> Any:
        typ = spec.get("type")
        name = str(spec.get("name", typ or "verifier"))
        if typ == "math_lenient":
            from vfates.verifiers.natural import MathLenientVerifier

            return MathLenientVerifier(name=name, rel_tol=float(spec.get("rel_tol", 1e-4)))
        if typ == "math_strip_string":
            from vfates.verifiers.natural import MathStripStringVerifier

            return MathStripStringVerifier(name=name)
        raise ValueError(f"MathTask cannot build verifier type {typ!r}")

    def support_answered(self, parsed: Parsed) -> bool:
        return parsed.answer is not None

    def _prepare(self) -> None:
        if self._items is not None:
            return
        with self._lock:
            if self._items is not None:
                return
            raw = self._rows if self._rows is not None else self._load_rows()
            sets = self._contamination_sets()
            self._items, self.filter_counts, self.difficulty_bucket_applied = _filter_rows(
                raw,
                source=self.source,
                bucket=self.difficulty_bucket,
                sets=sets,
                split_salt=self.split_salt,
                dev_fraction=self.dev_fraction,
                heldout_fraction=self.heldout_fraction,
                dataset=self.dataset,
                revision=self.revision,
            )

    def _load_rows(self) -> list[dict[str, Any]]:
        table = _load_hub(self.dataset, self.revision)
        rows: list[dict[str, Any]] = []
        for row in table:
            domain = row.get("domain") or []
            rows.append(
                {
                    "problem": str(row.get("problem") or ""),
                    "answer": str(row.get("answer") or ""),
                    "source": str(row.get("source") or ""),
                    "domain": [str(item) for item in domain],
                    "llama8b_solve_rate": row.get("llama8b_solve_rate"),
                }
            )
        return rows

    def _contamination_sets(self) -> dict[str, set[str]]:
        if not self.decontaminate:
            return {"math500": set(), "gsm8k": set(), "aime": set()}
        if self._contamination is not None:
            return self._contamination
        return load_contamination_sets()

    def _as_problem(self, item: _Kept) -> Problem:
        return Problem(
            pid=f"math/{item.split}/{item.problem_id[:16]}",
            split=item.split,
            task=self.name,
            question=item.question,
            answer=item.answer,
            meta=dict(item.meta),
        )


def _filter_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: str,
    bucket: tuple[float, float] | None,
    sets: Mapping[str, set[str]],
    split_salt: str,
    dev_fraction: float,
    heldout_fraction: float,
    dataset: str,
    revision: str,
) -> tuple[list[_Kept], dict[str, int], bool]:
    counts = _blank_counts()
    counts["loaded"] = len(rows)
    rates = [_as_rate(row.get("llama8b_solve_rate")) for row in rows]
    apply_bucket = bucket is not None and (
        source != "deepscaler" or any(rate is not None for rate in rates)
    )
    survivors: list[tuple[Mapping[str, Any], float | None]] = []
    for row, rate in zip(rows, rates, strict=True):
        problem = str(row.get("problem") or "")
        answer = str(row.get("answer") or "")
        if not answer.strip():
            counts["dropped_empty_answer"] += 1
            continue
        if _is_proof(problem, answer):
            counts["dropped_proof"] += 1
            continue
        if _is_multiple_choice(problem):
            counts["dropped_multiple_choice"] += 1
            continue
        if _is_non_closed(answer):
            counts["dropped_non_closed_form"] += 1
            continue
        if apply_bucket and bucket is not None:
            if rate is None:
                counts["dropped_unrated"] += 1
                continue
            if not (bucket[0] <= rate < bucket[1]):
                counts["dropped_difficulty"] += 1
                continue
        if "aime" in str(row.get("source") or "").casefold():
            counts["dropped_aime_source"] += 1
            continue
        survivors.append((row, rate))

    exact_seen: set[str] = set()
    norm_seen: set[str] = set()
    deduped: list[tuple[str, str, str, float | None, Mapping[str, Any]]] = []
    for row, rate in survivors:
        problem = str(row.get("problem") or "")
        answer = str(row.get("answer") or "")
        exact = problem.strip()
        if exact in exact_seen:
            counts["dropped_exact_duplicate"] += 1
            continue
        normalised = normalise_statement(problem)
        if normalised in norm_seen:
            counts["dropped_normalised_duplicate"] += 1
            continue
        exact_seen.add(exact)
        norm_seen.add(normalised)
        if normalised in sets.get("math500", ()):
            counts["dropped_math500"] += 1
            continue
        if normalised in sets.get("gsm8k", ()):
            counts["dropped_gsm8k"] += 1
            continue
        if normalised in sets.get("aime", ()):
            counts["dropped_aime"] += 1
            continue
        deduped.append((statement_id(problem), problem, answer, rate, row))

    kept: list[_Kept] = []
    for problem_id, problem, answer, rate, row in deduped:
        split = hash_split(
            problem_id,
            split_salt=split_salt,
            dev_fraction=dev_fraction,
            heldout_fraction=heldout_fraction,
        )
        domain = row.get("domain") or []
        kept.append(
            _Kept(
                problem_id=problem_id,
                question=problem,
                answer=answer,
                split=split,
                meta={
                    "problem_id": problem_id,
                    "source": str(row.get("source") or ""),
                    "domain": [str(item) for item in domain],
                    "llama8b_solve_rate": rate,
                    "dataset": dataset,
                    "revision": revision,
                },
            )
        )
        counts[f"n_{split}"] += 1
    counts["kept"] = len(kept)
    return kept, counts, apply_bucket


if __name__ == "__main__":
    if "--verify-worker" in sys.argv:
        _worker_main()
