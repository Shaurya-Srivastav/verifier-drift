"""QA task on TIGER-Lab/WebInstruct-verified, split ``train``.

``train_legacy`` is rejected. Only answer types with a canonical form are
kept: integer, float, and expression (the card's ``Integer``, ``Float``, and
``Expression``). Fraction, percentage, and the other card types are dropped;
their counts are in ``answer_type_counts``. ``difficulty_bucket`` is optional
and, when set, is one of the card's difficulty strings. Null keeps every
difficulty.

Dedup and the hash split match the math task: exact statement, then normalised
statement, then ``u01(split_salt, problem_id)``. There is no MATH-500 / GSM8K
/ AIME decontamination. Defaults are ``dev_fraction=0.02`` and
``heldout_fraction=0.04``.

The oracle is math-verify for integer, float, and expression answers (no
numeric tolerance) and normalised exact match otherwise. A missing box is
incorrect. The natural verifier is TIGER-Lab/general-verifier. The reported robust
verifier is sarosavo/Master-RM (arXiv:2507.08794), type ``master_rm``.

``run_judge_agreement`` is the IFEval rubric check and does not apply here.
:func:`judge_agreement_row` maps two general-verifier replies onto a row for
:func:`vfates.analysis.judge_agreement.agreement_statistics`. General-Reasoner
does not treat a missing ``Final Decision: Yes`` as unparsed, so
``local_unparsed`` is false. A human check of disagreements is preregistered
and does not change verdicts.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vfates.tasks.answer import (
    extract_all_boxed,
    extract_boxed,
    hash_split,
    normalise_statement,
    statement_id,
)
from vfates.tasks.base import Task
from vfates.tasks.judges import (
    GENERAL_VERIFIER_MAX_TOKENS,
    GENERAL_VERIFIER_MODEL,
    GENERAL_VERIFIER_REVISION,
    HeuristicVerifierJudge,
    Judge,
    MasterRMJudge,
    general_verifier_bit,
    judge_from_description,
)
from vfates.tasks.math import finite_real, math_verify_equal
from vfates.types import Parsed, Problem
from vfates.utils.hashing import u01
from vfates.utils.io import sha256_text

WEBINSTRUCT_DATASET = "TIGER-Lab/WebInstruct-verified"
WEBINSTRUCT_REVISION = "3e8a350b3a935d68fe70bdf379692500abc9ff51"
WEBINSTRUCT_SPLIT = "train"

# Card answer_type values with a canonical math form. Fraction and Percentage
# are numeric on the card and are still dropped: they are not in this list.
CANONICAL_ANSWER_TYPES: tuple[str, ...] = ("integer", "float", "expression")
DIFFICULTIES: tuple[str, ...] = (
    "Primary School",
    "Junior High School",
    "Senior High School",
    "University",
    "PhD",
)

DEFAULT_INSTRUCTION = "Please reason step by step, and end with the final answer within \\boxed{}."
SPLIT_SALT = "qa/split"

# Zhao et al., arXiv:2507.08794 Table 1, copied exactly. The space, period,
# comma and colon are lone symbols. The phrases are matched after lstrip.
MASTER_KEYS: tuple[str, ...] = (
    " ",
    ".",
    ",",
    ":",
    "Thought process:",
    "Let's solve this problem step by step.",
    "Solution",
    "解",
    "かいせつ",
    "Respuesta",
)
_PHRASE_KEYS: tuple[str, ...] = (
    "Thought process:",
    "Let's solve this problem step by step.",
    "Solution",
    "解",
    "かいせつ",
    "Respuesta",
)
_SYMBOL_KEYS = frozenset({".", ",", ":"})

ERROR_CATEGORIES: tuple[str, ...] = (
    "no_final_answer",
    "master_key_opener",
    "hedged_multiple_answers",
    "numeric_answer",
    "long_response",
    "refusal",
)

_COUNT_KEYS: tuple[str, ...] = (
    "loaded",
    "dropped_answer_type",
    "dropped_empty_answer",
    "dropped_difficulty",
    "dropped_exact_duplicate",
    "dropped_normalised_duplicate",
    "kept",
    "n_train",
    "n_heldout",
    "n_dev",
)

_FIELDS = frozenset(
    {
        "name",
        "dataset",
        "revision",
        "hf_split",
        "difficulty_bucket",
        "split_salt",
        "instruction",
        "dev_fraction",
        "heldout_fraction",
        "verify_timeout_s",
        "judge",
        "rows",
    }
)

_HEDGE = re.compile(
    r"\b(?:either|alternatively|or maybe|could also be|another answer)\b", re.IGNORECASE
)
_REFUSAL = re.compile(
    r"\b(?:i cannot|i can't|i am unable|i'm sorry|i am sorry|as an ai)\b",
    re.IGNORECASE,
)
_LONG_WORDS = 512


def judge_agreement_row(local_reply: str, remote_reply: str, pid: str) -> dict[str, Any]:
    """One ``agreement_statistics`` row for two general-verifier replies.

    Both bits are 1 iff the reply contains ``Final Decision: Yes``.
    ``local_unparsed`` is false: a reply without that tag is a No, which is
    how General-Reasoner scores it. This does not call ``run_judge_agreement``.
    """
    return {
        "local": general_verifier_bit(local_reply),
        "remote": general_verifier_bit(remote_reply),
        "pid": pid,
        "local_unparsed": False,
    }


def master_key_opener(text: str) -> bool:
    """True when ``text`` opens with a Table 1 master key.

    The lone space matches only the exact string ``\" \"``. Period, comma and
    colon match the whole stripped response. Phrases match after ``lstrip``.
    A phrase that ends with an ASCII letter (``Solution``) does not match when
    the next character is a letter, so ``Solutions`` is not ``Solution``.
    """
    if text == " ":
        return True
    if text.strip() in _SYMBOL_KEYS:
        return True
    head = text.lstrip()
    for key in _PHRASE_KEYS:
        if not head.startswith(key):
            continue
        if key[-1].isascii() and key[-1].isalpha():
            nxt = head[len(key) : len(key) + 1]
            if nxt.isalpha():
                continue
        return True
    return False


def _canonical_type(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    folded = value.strip().casefold()
    if folded in CANONICAL_ANSWER_TYPES:
        return folded
    return None


def _parse_difficulty(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text not in DIFFICULTIES:
        listed = ", ".join(DIFFICULTIES)
        raise ValueError(f"difficulty_bucket {text!r} is not one of {listed}")
    return text


def _build_judge(config: Mapping[str, Any] | None) -> Judge:
    cfg = dict(config or {})
    kind = str(cfg.get("type", "heuristic"))
    if kind == "heuristic":
        return HeuristicVerifierJudge()
    if kind != "llm":
        raise ValueError(f"unknown qa judge type {kind!r}")
    from vfates.tasks.judges import GeneralVerifierJudge

    return GeneralVerifierJudge(
        model=str(cfg.get("model", GENERAL_VERIFIER_MODEL)),
        revision=str(cfg.get("revision", GENERAL_VERIFIER_REVISION)),
        base_url=None if cfg.get("base_url") is None else str(cfg["base_url"]),
        api_key=None if cfg.get("api_key") is None else str(cfg["api_key"]),
        max_tokens=int(cfg.get("max_tokens", GENERAL_VERIFIER_MAX_TOKENS)),
        cache_path=cfg.get("cache_path"),
    )


@dataclass(frozen=True)
class _Kept:
    problem_id: str
    question: str
    answer: str
    split: str
    meta: dict[str, Any]


class QATask(Task):
    """WebInstruct-verified questions with a canonical answer and a model judge."""

    name = "qa"
    error_categories = ERROR_CATEGORIES
    verifier_types = ("model_judge", "master_rm")

    def __init__(
        self,
        *,
        name: str = "qa",
        dataset: str = WEBINSTRUCT_DATASET,
        revision: str = WEBINSTRUCT_REVISION,
        hf_split: str = WEBINSTRUCT_SPLIT,
        difficulty_bucket: str | None = None,
        split_salt: str = SPLIT_SALT,
        instruction: str = DEFAULT_INSTRUCTION,
        dev_fraction: float = 0.02,
        heldout_fraction: float = 0.04,
        verify_timeout_s: float = 5.0,
        judge: Mapping[str, Any] | None = None,
        rows: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        if hf_split == "train_legacy":
            raise ValueError("qa uses the train split, not train_legacy")
        if hf_split != "train":
            raise ValueError("qa split must be 'train'")
        if dev_fraction < 0 or heldout_fraction < 0 or dev_fraction + heldout_fraction >= 1:
            raise ValueError(
                "dev_fraction and heldout_fraction must be non-negative and sum to < 1"
            )
        if verify_timeout_s <= 0:
            raise ValueError("verify_timeout_s must be positive")
        self.name = name
        self.dataset = dataset
        self.revision = revision
        self.hf_split = hf_split
        self.difficulty_bucket = _parse_difficulty(difficulty_bucket)
        self.split_salt = split_salt
        self.instruction = instruction
        self.dev_fraction = float(dev_fraction)
        self.heldout_fraction = float(heldout_fraction)
        self.verify_timeout_s = float(verify_timeout_s)
        self._judge_config = None if judge is None else dict(judge)
        self._judge = _build_judge(self._judge_config)
        self._rows = None if rows is None else [dict(row) for row in rows]
        self._items: list[_Kept] | None = None
        self.filter_counts: dict[str, int] = {key: 0 for key in _COUNT_KEYS}
        self.answer_type_counts: dict[str, int] = {}
        self.kept_answer_type_counts: dict[str, int] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any]) -> QATask:
        """Build from a config mapping, ignoring keys this task does not use."""
        return cls(**{key: value for key, value in options.items() if key in _FIELDS})

    def describe(self) -> dict[str, Any]:
        self._prepare()
        return {
            "name": self.name,
            "dataset": self.dataset,
            "revision": self.revision,
            "hf_split": self.hf_split,
            "canonical_answer_types": list(CANONICAL_ANSWER_TYPES),
            "difficulty_bucket": self.difficulty_bucket,
            "difficulties": list(DIFFICULTIES),
            "split_salt": self.split_salt,
            "dev_fraction": self.dev_fraction,
            "heldout_fraction": self.heldout_fraction,
            "instruction_sha256": sha256_text(self.instruction),
            "filter_counts": dict(self.filter_counts),
            "answer_type_counts": dict(self.answer_type_counts),
            "kept_answer_type_counts": dict(self.kept_answer_type_counts),
            "judge": self._judge.describe(),
            "oracle": {
                "numeric": "math_verify",
                "otherwise": "normalised_exact",
                "timeout_s": self.verify_timeout_s,
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
            raise ValueError(f"qa split {split!r} has {len(chosen)} problems, asked for {n}")
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
        """math-verify for canonical types, else normalised exact match.

        There is no tolerance and no fallback from math-verify to string match.
        A missing final answer is 0.
        """
        del text
        if parsed.answer is None:
            return 0
        kind = str(problem.meta.get("answer_type", ""))
        if kind in CANONICAL_ANSWER_TYPES:
            return int(
                math_verify_equal(
                    problem.answer,
                    parsed.raw or parsed.answer,
                    timeout_s=self.verify_timeout_s,
                )
            )
        return int(normalise_statement(parsed.answer) == normalise_statement(problem.answer))

    def categorize(self, problem: Problem, parsed: Parsed, text: str) -> frozenset[str]:
        del problem
        found: set[str] = set()
        if parsed.answer is None:
            found.add("no_final_answer")
        if master_key_opener(text):
            found.add("master_key_opener")
        boxes = [box.strip() for box in extract_all_boxed(text) if box.strip()]
        if len(set(boxes)) >= 2 or _HEDGE.search(text or ""):
            found.add("hedged_multiple_answers")
        if finite_real(parsed.answer) is not None:
            found.add("numeric_answer")
        if len((text or "").split()) > _LONG_WORDS:
            found.add("long_response")
        if _REFUSAL.search(text or ""):
            found.add("refusal")
        return frozenset(found)

    def build_verifier(self, spec: Mapping[str, Any]) -> Any:
        from vfates.verifiers.natural import ModelJudgeVerifier

        typ = spec.get("type")
        described = spec.get("judge")
        if typ == "model_judge":
            judge = (
                self._judge
                if not isinstance(described, Mapping)
                else judge_from_description(described)
            )
            return ModelJudgeVerifier(judge, name=str(spec.get("name", "model_judge")))
        if typ == "master_rm":
            judge = (
                MasterRMJudge()
                if not isinstance(described, Mapping)
                else judge_from_description(described)
            )
            return ModelJudgeVerifier(
                judge,
                name=str(spec.get("name", "natural_robust")),
                verifier_type="master_rm",
            )
        raise ValueError(f"QATask cannot build verifier type {typ!r}")

    def support_answered(self, parsed: Parsed) -> bool:
        return parsed.answer is not None

    def judge_agreement(self, local_reply: str, remote_reply: str, pid: str) -> dict[str, Any]:
        """See :func:`judge_agreement_row`."""
        return judge_agreement_row(local_reply, remote_reply, pid)

    def _prepare(self) -> None:
        if self._items is not None:
            return
        with self._lock:
            if self._items is not None:
                return
            raw = self._rows if self._rows is not None else self._load_rows()
            items, counts, seen, kept_types = _filter_rows(
                raw,
                bucket=self.difficulty_bucket,
                split_salt=self.split_salt,
                dev_fraction=self.dev_fraction,
                heldout_fraction=self.heldout_fraction,
                dataset=self.dataset,
                revision=self.revision,
            )
            self._items = items
            self.filter_counts = counts
            self.answer_type_counts = seen
            self.kept_answer_type_counts = kept_types

    def _load_rows(self) -> list[dict[str, Any]]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("qa loading requires the datasets package") from exc
        table = load_dataset(self.dataset, split=self.hf_split, revision=self.revision)
        rows: list[dict[str, Any]] = []
        for row in table:
            rows.append(
                {
                    "question": str(row.get("question") or ""),
                    "answer": str(row.get("answer") or ""),
                    "answer_type": row.get("answer_type"),
                    "difficulty": row.get("difficulty"),
                    "id": row.get("id"),
                    "category": row.get("category"),
                }
            )
        return rows

    def _as_problem(self, item: _Kept) -> Problem:
        return Problem(
            pid=f"qa/{item.split}/{item.problem_id[:16]}",
            split=item.split,
            task=self.name,
            question=item.question,
            answer=item.answer,
            meta=dict(item.meta),
        )


def _filter_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    bucket: str | None,
    split_salt: str,
    dev_fraction: float,
    heldout_fraction: float,
    dataset: str,
    revision: str,
) -> tuple[list[_Kept], dict[str, int], dict[str, int], dict[str, int]]:
    counts = {key: 0 for key in _COUNT_KEYS}
    counts["loaded"] = len(rows)
    seen: dict[str, int] = {}
    kept_types: dict[str, int] = {}
    pending: list[tuple[str, str, str, Mapping[str, Any]]] = []
    for row in rows:
        raw_type = row.get("answer_type")
        label = str(raw_type).strip() if isinstance(raw_type, str) else ""
        if label:
            seen[label] = seen.get(label, 0) + 1
        kind = _canonical_type(raw_type)
        if kind is None:
            counts["dropped_answer_type"] += 1
            continue
        answer = str(row.get("answer") or "")
        if not answer.strip():
            counts["dropped_empty_answer"] += 1
            continue
        difficulty = row.get("difficulty")
        difficulty_text = "" if difficulty is None else str(difficulty).strip()
        if bucket is not None and difficulty_text != bucket:
            counts["dropped_difficulty"] += 1
            continue
        question = str(row.get("question") or row.get("problem") or "")
        pending.append((kind, question, answer, row))

    exact_seen: set[str] = set()
    norm_seen: set[str] = set()
    kept: list[_Kept] = []
    for kind, question, answer, row in pending:
        exact = question.strip()
        if exact in exact_seen:
            counts["dropped_exact_duplicate"] += 1
            continue
        normalised = normalise_statement(question)
        if normalised in norm_seen:
            counts["dropped_normalised_duplicate"] += 1
            continue
        exact_seen.add(exact)
        norm_seen.add(normalised)
        problem_id = statement_id(question)
        split = hash_split(
            problem_id,
            split_salt=split_salt,
            dev_fraction=dev_fraction,
            heldout_fraction=heldout_fraction,
        )
        difficulty = row.get("difficulty")
        kept.append(
            _Kept(
                problem_id=problem_id,
                question=question,
                answer=answer,
                split=split,
                meta={
                    "problem_id": problem_id,
                    "answer_type": kind,
                    "difficulty": None if difficulty is None else str(difficulty),
                    "dataset_id": None if row.get("id") is None else str(row.get("id")),
                    "category": None if row.get("category") is None else str(row.get("category")),
                    "dataset": dataset,
                    "revision": revision,
                },
            )
        )
        counts[f"n_{split}"] += 1
        kept_types[kind] = kept_types.get(kind, 0) + 1
    counts["kept"] = len(kept)
    return kept, counts, seen, kept_types
