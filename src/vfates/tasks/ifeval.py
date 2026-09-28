"""Instruction-following task.

Default source is ``allenai/IF_multi_constraints_upto5`` (ODC-BY, revision
pinned). Each row is up to five constraints drawn from IFEval's 25 checkers
and IFBench's 29 training checkers. ``source="rlvr_ifeval"`` keeps
``allenai/RLVR-IFeval`` with the original row-id split and one checker per row.

The natural verifier is the constraint checker: every constraint must pass.
The oracle is that checker and a content judge. The judge runs on every
sample, including checker failures. The default judge is
:class:`~vfates.tasks.judges.HeuristicJudge`. Study configs point
``judge_config`` at ``Qwen/Qwen3-14B`` non-thinking (temperature 0 is fixed
inside :class:`~vfates.tasks.judges.LLMJudge`). ``describe`` records the model
id, revision, chat-template kwargs and rubric sha256.

Multi-constraint pools:

* Difficulty is the number of constraints. ``constraint_counts`` keeps a
  bucket such as ``[2, 3]`` or ``[4, 5]``. ``None`` keeps every count. Pilot
  P2 chooses the bucket. It is not an item-level filter on the study model.
* Length budget. A row is dropped when a constraint requires strictly more
  words than the completion budget allows. ``budget_words`` is
  ``floor(max_completion_tokens * words_per_token)``. The default budget is
  768 tokens times 0.75 words per token, which is 576 words. 0.75 words per
  token is about 4/3 tokens per word, a typical English BPE rate for Qwen.
  The minimum is ``length_constraints:number_words`` with relation
  ``at least``, or ``3 * n_sent * n_words`` for
  ``count:counting_composition`` (three paragraphs). ``less than`` does not
  force a long answer and is kept. Copy constraints are not length filters.
* Dedup keeps the first row, in dataset order, whose normalised prompt was
  not already kept. Normalisation strips, case-folds, and collapses
  whitespace. U+001F is treated as whitespace so the split hash cannot raise.
  Bucket and budget filters run before dedup, so a dropped duplicate does not
  block a later row.
* Split is ``u01("ifeval/prompt", seed, normalised_prompt)``. The constraint
  list is not part of the key. ``u < dev_fraction`` is ``dev``,
  ``u < dev_fraction + heldout_fraction`` is ``heldout``, and the rest is
  ``train``. Defaults are 0.10 and 0.33. ``problems`` returns the first ``n``
  rows of that split in row-id order. ``row_id`` is the index after filters,
  not the raw dataset index (``hf_index``).

``categorize`` is a cheap predicate used by patch arms. Prompt categories
describe the constraint family. Output categories describe checker false
positives: responses that satisfy the constraints and do not do the task.

* ``keyword_constraints``: the prompt includes, excludes, or counts words,
  keywords, letters, or a palindrome.
* ``length_constraints``: word, sentence, paragraph, letter-total, or
  unique-token constraints, including the three-paragraph composition.
* ``format_constraints``: JSON, bullets, sections, title, highlights,
  placeholders, postscript, hyphenated sentences, square brackets, bigram
  wraps, constrained options, two responses, or a punctuation ban.
* ``case_constraints``: all-caps, all-lowercase, or a capital-word count.
* ``start_end_constraints``: a required first or last word, an ending phrase,
  wrapping quotes, the same word at both ends, or a copy or repeat of the
  prompt.
* ``many_constraints``: four or more constraints on the prompt.
* ``short_response``: fewer than 12 whitespace-separated words. That is the
  heuristic judge's content-word minimum. Empty replies count as short.
* ``repetition_heavy``: the normalised response equals the prompt, contains
  the whole prompt when the prompt has at least 40 characters, contains the
  prompt's first 80 characters, or repeats a constraint keyword of at least
  3 characters at least 3 times as half or more of the response tokens.

A language-only prompt has no prompt category. Several categories can fire
together. RLVR-shaped injected rows stay on the backup path so existing
fixtures keep the row-id split.
"""

from __future__ import annotations

import ast
import re
import threading
from collections.abc import Mapping, Sequence
from typing import Any

from vfates.tasks.base import Task
from vfates.tasks.if_checkers import (
    IFEVAL_ALL,
    apply_checker,
    parse_ground_truth,
    prepare_constraint,
)
from vfates.tasks.judges import DEFAULT_RUBRIC, HeuristicJudge, Judge, LLMJudge
from vfates.types import Parsed, Problem
from vfates.utils.hashing import u01
from vfates.utils.io import sha256_text
from vfates.verifiers.base import Verifier

SOURCE_IF_MULTI = "if_multi"
SOURCE_RLVR = "rlvr_ifeval"
DATASET_IF_MULTI = "allenai/IF_multi_constraints_upto5"
DATASET_IF_MULTI_REVISION = "2e3a77407b7fce69f95b248d64a884e3ae1c2423"
DATASET_RLVR = "allenai/RLVR-IFeval"
DEFAULT_WORDS_PER_TOKEN = 0.75
DEFAULT_MAX_COMPLETION_TOKENS = 768
DEFAULT_DEV_FRACTION = 0.10
DEFAULT_HELDOUT_FRACTION = 0.33
SHORT_RESPONSE_WORDS = 12
MANY_CONSTRAINTS_MIN = 4
_PROMPT_COPY_CHARS = 40
_PROMPT_PREFIX_CHARS = 80
_KEYWORD_MIN_CHARS = 3
_KEYWORD_MIN_HITS = 3
_KEYWORD_TOKEN_SHARE = 0.5
_SPLIT_SALT = "ifeval/prompt"
_WS = re.compile(r"\s+")

_KEYWORD = frozenset(
    {
        "keywords:existence",
        "keywords:frequency",
        "keywords:forbidden_words",
        "keywords:letter_frequency",
        "keywords:exclude_word_harder",
        "keywords:no_adjacent_consecutive",
        "keywords:palindrome",
        "keywords:keyword_specific_position",
        "keywords:word_once",
        "keywords:word_count_different_numbers",
        "letters:letter_counting2",
        "count:count_increment_word",
        "count:lowercase_counting",
        "verify_keywords",
        "verify_keyword_frequency",
        "validate_forbidden_words",
        "verify_letter_frequency",
    }
)
_LENGTH = frozenset(
    {
        "length_constraints:number_sentences",
        "length_constraints:number_paragraphs",
        "length_constraints:number_words",
        "length_constraints:nth_paragraph_first_word",
        "paragraphs:paragraphs",
        "paragraphs:paragraphs2",
        "count:counting_composition",
        "count:count_unique",
        "letters:letter_counting",
        "validate_word_constraint",
        "verify_sentence_constraint",
        "verify_paragraph_count",
        "validate_paragraphs",
    }
)
_FORMAT = frozenset(
    {
        "detectable_content:number_placeholders",
        "detectable_content:postscript",
        "detectable_format:bigram_wrapping",
        "detectable_format:constrained_response",
        "detectable_format:json_format",
        "detectable_format:multiple_sections",
        "detectable_format:number_bullet_lists",
        "detectable_format:number_highlighted_sections",
        "detectable_format:sentence_hyphens",
        "detectable_format:square_brackets",
        "detectable_format:title",
        "combination:two_responses",
        "punctuation:no_comma",
        "punctuation:punctuation_dot",
        "punctuation:punctuation_exclamation",
        "validate_json_format",
        "verify_bullet_points",
        "validate_sections",
        "validate_title",
        "validate_placeholders",
        "validate_highlighted_sections",
        "validate_two_responses",
        "verify_postscript",
        "validate_choice",
        "validate_no_commas",
    }
)
_CASE = frozenset(
    {
        "change_case:capital_word_frequency",
        "change_case:english_capital",
        "change_case:english_lowercase",
        "validate_uppercase",
        "validate_lowercase",
        "validate_frequency_capital_words",
    }
)
_START_END = frozenset(
    {
        "startend:end_checker",
        "startend:quotation",
        "first_word:first_word_sent",
        "first_word:first_word_answer",
        "last_word:last_word_sent",
        "last_word:last_word_answer",
        "keywords:start_end",
        "combination:repeat_prompt",
        "copy:repeat_phrase",
        "copy:copy",
        "copy:copying_simple",
        "copy:copying_multiple",
        "new:copy_span_idx",
        "validate_end",
        "validate_quotation",
        "validate_repeat_prompt",
    }
)
_KEYWORD_FIELDS = (
    "keyword",
    "keyword1",
    "keyword2",
    "first_word",
    "last_word",
    "end_phrase",
    "phrase",
    "letter",
    "word",
)
_KEYWORD_LIST_FIELDS = ("keywords", "forbidden_words", "keyword_list")


def normalise_prompt(text: str) -> str:
    """Strip, case-fold, and collapse whitespace. U+001F becomes a space."""
    cleaned = text.replace("\x1f", " ")
    return _WS.sub(" ", cleaned.strip()).casefold()


def minimum_required_words(constraints: Sequence[Mapping[str, Any]]) -> int:
    """Minimum words a satisfying response must contain, or 0 if none is forced."""
    required = 0
    for item in constraints:
        kwargs = item.get("kwargs") or {}
        instruction_id = str(item["instruction_id"])
        if (
            instruction_id == "length_constraints:number_words"
            and kwargs.get("relation") == "at least"
        ):
            required = max(required, int(kwargs["num_words"]))
        elif instruction_id == "count:counting_composition":
            total = 3 * int(kwargs["n_sent"]) * int(kwargs["n_words"])
            required = max(required, total)
    return required


def _make_judge(spec: Mapping[str, Any] | None) -> Judge:
    """Build the oracle judge from a config mapping. Default is the heuristic."""
    if not spec:
        return HeuristicJudge()
    kind = str(spec.get("type", "heuristic"))
    if kind == "heuristic":
        return HeuristicJudge(
            min_content_words=int(spec.get("min_content_words", 12)),
            max_repeat_ratio=float(spec.get("max_repeat_ratio", 0.4)),
            max_keyword_density=float(spec.get("max_keyword_density", 0.5)),
        )
    if kind == "llm":
        model = spec.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("llm judge_config requires model")
        rubric = spec.get("rubric")
        recorded = spec.get("rubric_sha256")
        if recorded is not None:
            used = rubric if isinstance(rubric, str) and rubric else DEFAULT_RUBRIC
            digest = sha256_text(used)
            if str(recorded) != digest:
                raise ValueError(f"judge rubric_sha256 {recorded} does not match the rubric")
        return LLMJudge(
            model,
            base_url=spec.get("base_url"),
            api_key=spec.get("api_key"),
            rubric=rubric if isinstance(rubric, str) else None,
            max_tokens=int(spec.get("max_tokens", 16)),
            **_llm_judge_extras(spec),
        )
    raise ValueError(f"unknown judge type {kind!r}")


def _llm_judge_extras(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Forward revision and chat-template kwargs once ``LLMJudge`` accepts them.

    ``judges.py`` is owned elsewhere. Until it grows these parameters, the task
    still records them in :meth:`IFEvalTask.describe` and does not pass them.
    """
    import inspect

    accepted = inspect.signature(LLMJudge).parameters
    extras: dict[str, Any] = {}
    revision = spec.get("revision")
    if revision is not None and "revision" in accepted:
        extras["revision"] = revision
    chat_kwargs = spec.get("chat_template_kwargs")
    if chat_kwargs is not None and "chat_template_kwargs" in accepted:
        extras["chat_template_kwargs"] = chat_kwargs
    return extras


def _row_kind(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "empty"
    row = rows[0]
    if "ground_truth" in row or "constraints" in row:
        return "multi"
    if "func_name" in row:
        return "rlvr"
    raise ValueError("rows need func_name or ground_truth/constraints")


class IFEvalTask(Task):
    """IFEval with a deterministic split.

    ``problems(split, n, seed)`` returns the first ``n`` rows of the split.
    For ``rlvr_ifeval``, a row with ``u = u01(seed, row_id)`` is held out when
    ``u < heldout_fraction``, dev when ``u < heldout_fraction + dev_fraction``,
    and train otherwise, so the held-out rows do not depend on
    ``dev_fraction``. For the multi-constraint source, membership is
    :meth:`prompt_split`. ``n`` is a cap.

    ``parse`` sets ``answer`` to the SHA-256 of the stripped response (the
    clustering key) and ``raw`` to the response. An empty strip yields
    ``answer is None``, which is what :meth:`support_answered` reports.
    """

    name = "ifeval"
    error_categories: tuple[str, ...] = (
        "keyword_constraints",
        "length_constraints",
        "format_constraints",
        "case_constraints",
        "start_end_constraints",
        "many_constraints",
        "short_response",
        "repetition_heavy",
    )
    verifier_types: tuple[str, ...] = ("ifeval_checker",)

    def __init__(
        self,
        *,
        heldout_fraction: float = DEFAULT_HELDOUT_FRACTION,
        dev_fraction: float = DEFAULT_DEV_FRACTION,
        judge: Judge | None = None,
        rows: list[dict[str, Any]] | None = None,
        judge_config: Mapping[str, Any] | None = None,
        source: str = SOURCE_IF_MULTI,
        dataset: str | None = None,
        revision: str | None = None,
        constraint_counts: Sequence[int] | None = None,
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
        words_per_token: float = DEFAULT_WORDS_PER_TOKEN,
    ) -> None:
        if source not in (SOURCE_IF_MULTI, SOURCE_RLVR):
            raise ValueError(f"unknown ifeval source {source!r}")
        if not 0.0 <= heldout_fraction <= 1.0:
            raise ValueError("heldout_fraction must lie in [0, 1]")
        if not 0.0 <= dev_fraction <= 1.0:
            raise ValueError("dev_fraction must lie in [0, 1]")
        if judge is not None and judge_config is not None:
            raise ValueError("pass judge or judge_config, not both")
        if max_completion_tokens < 1:
            raise ValueError("max_completion_tokens must be positive")
        if words_per_token <= 0:
            raise ValueError("words_per_token must be positive")
        if rows is not None and _row_kind(rows) == "rlvr" and source == SOURCE_IF_MULTI:
            # Existing fixtures pass RLVR-shaped rows and no source. They keep
            # the backup split. An explicit multi-constraint fixture uses
            # ground_truth or constraints.
            source = SOURCE_RLVR
        if rows is not None and _row_kind(rows) == "multi" and source == SOURCE_RLVR:
            raise ValueError("rlvr_ifeval does not load multi-constraint rows")
        if source == SOURCE_RLVR and constraint_counts is not None:
            raise ValueError("constraint_counts applies to source if_multi only")
        if dev_fraction + heldout_fraction > 1.0:
            raise ValueError("dev_fraction + heldout_fraction must be <= 1")
        expected = DATASET_RLVR if source == SOURCE_RLVR else DATASET_IF_MULTI
        if dataset is not None and dataset != expected:
            raise ValueError(f"dataset {dataset!r} does not match source {source!r} ({expected})")
        counts: tuple[int, ...] | None
        if constraint_counts is None:
            counts = None
        else:
            counts = tuple(int(item) for item in constraint_counts)
            if not counts:
                raise ValueError("constraint_counts must be non-empty")
            if any(item < 1 for item in counts):
                raise ValueError("constraint_counts must be positive")
        self.source = source
        self.dataset = expected
        self.revision = (
            (revision or DATASET_IF_MULTI_REVISION) if source == SOURCE_IF_MULTI else revision
        )
        self.heldout_fraction = float(heldout_fraction)
        self.dev_fraction = float(dev_fraction)
        self.constraint_counts = counts
        self.max_completion_tokens = int(max_completion_tokens)
        self.words_per_token = float(words_per_token)
        self._judge_revision, self._chat_template_kwargs = _judge_meta(judge_config)
        self.judge: Judge = judge if judge is not None else _make_judge(judge_config)
        self._injected = rows
        self._rows: list[dict[str, Any]] | None = None
        self._filter_report: dict[str, Any] | None = None
        self._lock = threading.Lock()

    @property
    def budget_words(self) -> int:
        """``floor(max_completion_tokens * words_per_token)``."""
        return int(self.max_completion_tokens * self.words_per_token)

    def describe(self) -> dict[str, Any]:
        """Config that rebuilds this task, plus the judge id, revision and rubric hash."""
        judge = dict(self.judge.describe())
        judge["revision"] = self._judge_revision
        if self._chat_template_kwargs is not None:
            judge["chat_template_kwargs"] = dict(self._chat_template_kwargs)
        rubric = getattr(self.judge, "rubric", None)
        if isinstance(rubric, str):
            judge["model"] = getattr(self.judge, "model", judge.get("model"))
            judge["temperature"] = 0
            judge["rubric_sha256"] = sha256_text(rubric)
        return {
            "type": "ifeval",
            "source": self.source,
            "dataset": self.dataset,
            "revision": self.revision,
            "heldout_fraction": self.heldout_fraction,
            "dev_fraction": self.dev_fraction,
            "constraint_counts": None
            if self.constraint_counts is None
            else list(self.constraint_counts),
            "max_completion_tokens": self.max_completion_tokens,
            "words_per_token": self.words_per_token,
            "budget_words": self.budget_words,
            "judge": judge,
        }

    def filter_report(self) -> dict[str, Any] | None:
        """Row counts after the bucket, budget and dedup filters. ``None`` on RLVR."""
        if self.source != SOURCE_IF_MULTI:
            return None
        self._records()
        return None if self._filter_report is None else dict(self._filter_report)

    def prompt_split(self, prompt: str, seed: int) -> str:
        """``train``, ``heldout`` or ``dev`` from the normalised prompt text."""
        unit = u01(_SPLIT_SALT, seed, normalise_prompt(prompt))
        if unit < self.dev_fraction:
            return "dev"
        if unit < self.dev_fraction + self.heldout_fraction:
            return "heldout"
        return "train"

    def _records(self) -> list[dict[str, Any]]:
        if self._rows is None:
            with self._lock:
                if self._rows is None:
                    if self._injected is not None:
                        self._rows = (
                            self._prepare_multi(self._injected)
                            if self.source == SOURCE_IF_MULTI
                            else [dict(row) for row in self._injected]
                        )
                    elif self.source == SOURCE_RLVR:
                        self._rows = self._load_rlvr()
                    else:
                        self._rows = self._prepare_multi(self._load_multi_raw())
        return self._rows

    def _rlvr_split(self, row_id: int, seed: int) -> str:
        unit = u01(seed, row_id)
        if unit < self.heldout_fraction:
            return "heldout"
        if unit < self.heldout_fraction + self.dev_fraction:
            return "dev"
        return "train"

    def problems(self, split: str, n: int, seed: int) -> list[Problem]:
        if n < 0:
            raise ValueError("n must be non-negative")
        if split not in ("train", "heldout", "dev"):
            raise ValueError(f"unknown split {split!r}")
        if self.source == SOURCE_RLVR:
            chosen = [
                row
                for row in self._records()
                if self._rlvr_split(int(row["row_id"]), seed) == split
            ]
            chosen.sort(key=lambda row: int(row["row_id"]))
            return [self._rlvr_problem(row, split) for row in chosen[:n]]
        chosen = [
            row for row in self._records() if self.prompt_split(str(row["question"]), seed) == split
        ]
        chosen.sort(key=lambda row: int(row["row_id"]))
        return [self._multi_problem(row, split) for row in chosen[:n]]

    def _rlvr_problem(self, row: dict[str, Any], split: str) -> Problem:
        row_id = int(row["row_id"])
        return Problem(
            pid=f"ifeval/{split}/{row_id:05d}",
            split=split,
            task=self.name,
            question=str(row["question"]),
            answer=str(row["constraint"]),
            meta={
                "row_id": row_id,
                "source": SOURCE_RLVR,
                "func_name": row["func_name"],
                "args": dict(row["args"]),
                "constraint_type": row["constraint_type"],
                "constraint": row["constraint"],
                "n_constraints": 1,
                "instruction_ids": [row["func_name"]],
            },
        )

    def _multi_problem(self, row: dict[str, Any], split: str) -> Problem:
        row_id = int(row["row_id"])
        constraints = [
            {"instruction_id": item["instruction_id"], "kwargs": dict(item["kwargs"])}
            for item in row["constraints"]
        ]
        return Problem(
            pid=f"ifeval/{split}/{row_id:05d}",
            split=split,
            task=self.name,
            question=str(row["question"]),
            answer=str(row.get("constraint") or ""),
            meta={
                "row_id": row_id,
                "source": SOURCE_IF_MULTI,
                "key": row.get("key"),
                "hf_index": row.get("hf_index"),
                "func_name": IFEVAL_ALL,
                "args": {"constraints": constraints},
                "constraints": constraints,
                "n_constraints": len(constraints),
                "instruction_ids": [item["instruction_id"] for item in constraints],
                "constraint_type": row.get("constraint_type"),
                "constraint": row.get("constraint"),
            },
        )

    def prompt(self, problem: Problem) -> str:
        return problem.question

    def system_prompt(self) -> str | None:
        return None

    def parse(self, problem: Problem, text: str) -> Parsed:
        del problem
        stripped = text.strip()
        answer = sha256_text(stripped) if stripped else None
        return Parsed(answer=answer, raw=text, extras={})

    def oracle(self, problem: Problem, parsed: Parsed, text: str) -> int:
        """Checker AND content judge.

        ``parsed.extras["content_judge"]`` is the judge bit the oracle used
        (an unparsed YES/NO reply is stored as 0).
        ``parsed.extras["content_judge_parsed"]`` is false when that reply was
        not a single YES or NO. The judge still runs when the checker fails.
        """
        func_name = str(problem.meta["func_name"])
        args = dict(problem.meta["args"])
        checker_ok = apply_checker(text, func_name, args)
        verdict, parsed_ok = self.judge.judge_result(
            problem.question,
            text,
            func_name=func_name,
            checker_args=args,
            instruction_ids=list(problem.meta.get("instruction_ids") or []),
        )
        parsed.extras["content_judge"] = int(verdict)
        parsed.extras["content_judge_parsed"] = bool(parsed_ok)
        return int(checker_ok and int(verdict) == 1)

    def categorize(self, problem: Problem, parsed: Parsed, text: str) -> frozenset[str]:
        """Prompt families plus short and repetitive outputs. See the module docstring."""
        del parsed
        names = _constraint_names(problem)
        found: set[str] = set()
        if names & _KEYWORD:
            found.add("keyword_constraints")
        if names & _LENGTH:
            found.add("length_constraints")
        if names & _FORMAT:
            found.add("format_constraints")
        if names & _CASE:
            found.add("case_constraints")
        if names & _START_END:
            found.add("start_end_constraints")
        if _n_constraints(problem) >= MANY_CONSTRAINTS_MIN:
            found.add("many_constraints")
        if len(text.split()) < SHORT_RESPONSE_WORDS:
            found.add("short_response")
        if _repetition_heavy(problem.question, text, _keywords(problem)):
            found.add("repetition_heavy")
        return frozenset(found)

    def abs_error(self, problem: Problem, parsed: Parsed) -> float | None:
        """IFEval has no numeric gold. Tolerance arms do not apply."""
        del problem, parsed
        return None

    def build_verifier(self, spec: Mapping[str, Any]) -> Verifier:
        """Build ``ifeval_checker``. Every constraint in the row must pass."""
        from vfates.verifiers.natural import IFEvalCheckerVerifier

        typ = spec.get("type")
        if typ not in self.verifier_types:
            raise ValueError(f"IFEvalTask cannot build verifier type {typ!r}")
        return IFEvalCheckerVerifier(name=str(spec.get("name", "ifeval_checker")))

    def support_answered(self, parsed: Parsed) -> bool:
        """True when the stripped response is non-empty (it has a parse hash)."""
        return parsed.answer is not None

    def _prepare_multi(self, raw_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        bucket = None if self.constraint_counts is None else set(self.constraint_counts)
        budget = self.budget_words
        dropped_bucket = 0
        dropped_budget = 0
        dropped_duplicate = 0
        kept: list[dict[str, Any]] = []
        seen: set[str] = set()
        by_count: dict[str, int] = {}
        for index, raw in enumerate(raw_rows):
            constraints = _parse_constraints(raw)
            if bucket is not None and len(constraints) not in bucket:
                dropped_bucket += 1
                continue
            if minimum_required_words(constraints) > budget:
                dropped_budget += 1
                continue
            question = _question_of(raw)
            key = normalise_prompt(question)
            if key in seen:
                dropped_duplicate += 1
                continue
            seen.add(key)
            row_id = len(kept)
            label = str(len(constraints))
            by_count[label] = by_count.get(label, 0) + 1
            kept.append(
                {
                    "row_id": row_id,
                    "question": question,
                    "constraints": constraints,
                    "constraint": raw.get("constraint"),
                    "constraint_type": raw.get("constraint_type"),
                    "key": raw.get("key"),
                    "hf_index": raw.get("hf_index", index),
                }
            )
        self._filter_report = {
            "raw": len(raw_rows),
            "dropped_bucket": dropped_bucket,
            "dropped_budget": dropped_budget,
            "dropped_duplicate": dropped_duplicate,
            "kept": len(kept),
            "budget_words": budget,
            "words_per_token": self.words_per_token,
            "max_completion_tokens": self.max_completion_tokens,
            "constraint_counts": None if bucket is None else sorted(bucket),
            "by_constraint_count": by_count,
        }
        return kept

    def _load_rlvr(self) -> list[dict[str, Any]]:
        from datasets import load_dataset

        kwargs: dict[str, Any] = {}
        if self.revision is not None:
            kwargs["revision"] = self.revision
        dataset = load_dataset(self.dataset, split="train", **kwargs)
        rows: list[dict[str, Any]] = []
        for index, record in enumerate(dataset):
            func_name, args = parse_ground_truth(record["ground_truth"])
            rows.append(
                {
                    "row_id": index,
                    "question": _user_text(record["messages"]),
                    "func_name": func_name,
                    "args": args,
                    "constraint_type": str(record["constraint_type"]),
                    "constraint": str(record["constraint"]),
                }
            )
        return rows

    def _load_multi_raw(self) -> list[dict[str, Any]]:
        from datasets import load_dataset

        dataset = load_dataset(self.dataset, split="train", revision=self.revision)
        rows: list[dict[str, Any]] = []
        for index, record in enumerate(dataset):
            rows.append(
                {
                    "hf_index": index,
                    "key": record.get("key"),
                    "messages": record["messages"],
                    "ground_truth": record["ground_truth"],
                    "dataset": record.get("dataset"),
                    "constraint_type": record.get("constraint_type"),
                    "constraint": record.get("constraint"),
                }
            )
        return rows


def _judge_meta(
    spec: Mapping[str, Any] | None,
) -> tuple[str | None, dict[str, Any] | None]:
    if not spec:
        return None, None
    revision = spec.get("revision")
    raw_kwargs = spec.get("chat_template_kwargs")
    chat_kwargs: dict[str, Any] | None
    if raw_kwargs is None:
        chat_kwargs = None
    elif isinstance(raw_kwargs, Mapping):
        chat_kwargs = {str(key): value for key, value in raw_kwargs.items()}
    else:
        raise TypeError("chat_template_kwargs must be a mapping")
    return (None if revision is None else str(revision), chat_kwargs)


def _question_of(row: Mapping[str, Any]) -> str:
    if "question" in row and row["question"] is not None:
        return str(row["question"])
    messages = row.get("messages")
    if isinstance(messages, list):
        return _user_text(messages)
    raise ValueError("multi-constraint row is missing question and messages")


def _parse_constraints(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "ground_truth" in row:
        parsed = row["ground_truth"]
        if isinstance(parsed, str):
            parsed = ast.literal_eval(parsed)
        if isinstance(parsed, list):
            if len(parsed) != 1:
                raise ValueError("ground_truth must contain one instruction bundle")
            item = parsed[0]
        elif isinstance(parsed, dict):
            item = parsed
        else:
            raise TypeError("ground_truth must be a bundle or a list of one bundle")
        instruction_ids = item["instruction_id"]
        kwargs = item["kwargs"]
        if len(instruction_ids) != len(kwargs):
            raise ValueError("instruction_id and kwargs have different lengths")
        pairs = list(zip(instruction_ids, kwargs, strict=True))
    elif "constraints" in row:
        pairs = [(item["instruction_id"], item.get("kwargs")) for item in row["constraints"]]
    else:
        raise ValueError("multi-constraint row is missing ground_truth and constraints")
    prepared: list[dict[str, Any]] = []
    for instruction_id, raw_kwargs in pairs:
        name = str(instruction_id)
        try:
            args = prepare_constraint(name, raw_kwargs)
        except KeyError as exc:
            raise ValueError(f"unknown constraint id {name!r}") from exc
        prepared.append({"instruction_id": name, "kwargs": args})
    return prepared


def _user_text(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content") or "")
    if messages:
        return str(messages[0].get("content") or "")
    return ""


def _constraint_names(problem: Problem) -> set[str]:
    ids = problem.meta.get("instruction_ids")
    if isinstance(ids, list) and ids:
        return {str(item) for item in ids}
    func_name = problem.meta.get("func_name")
    if isinstance(func_name, str) and func_name and func_name != IFEVAL_ALL:
        return {func_name}
    return set()


def _n_constraints(problem: Problem) -> int:
    raw = problem.meta.get("n_constraints")
    if raw is not None:
        return int(raw)
    ids = problem.meta.get("instruction_ids")
    if isinstance(ids, list):
        return len(ids)
    return 1 if problem.meta.get("func_name") else 0


def _keywords(problem: Problem) -> list[str]:
    found: list[str] = []
    constraints = problem.meta.get("constraints")
    bundles: list[Mapping[str, Any]]
    if isinstance(constraints, list):
        bundles = [item.get("kwargs") or {} for item in constraints]
    else:
        args = problem.meta.get("args")
        bundles = [args] if isinstance(args, dict) and "constraints" not in args else []
    for kwargs in bundles:
        for field in _KEYWORD_FIELDS:
            value = kwargs.get(field)
            if isinstance(value, str) and value:
                found.append(value)
        for field in _KEYWORD_LIST_FIELDS:
            value = kwargs.get(field)
            if isinstance(value, list):
                found.extend(str(item) for item in value if item)
    return found


def _repetition_heavy(question: str, text: str, keywords: Sequence[str]) -> bool:
    prompt = normalise_prompt(question)
    response = normalise_prompt(text)
    if response and response == prompt:
        return True
    if len(prompt) >= _PROMPT_COPY_CHARS and prompt in response:
        return True
    if len(prompt) >= _PROMPT_PREFIX_CHARS and prompt[:_PROMPT_PREFIX_CHARS] in response:
        return True
    tokens = response.split()
    if not tokens:
        return False
    for keyword in keywords:
        folded = normalise_prompt(keyword)
        if len(folded) < _KEYWORD_MIN_CHARS:
            continue
        if " " in folded:
            hits = response.count(folded)
            if hits >= _KEYWORD_MIN_HITS and len(folded) * hits >= _KEYWORD_TOKEN_SHARE * max(
                len(response), 1
            ):
                return True
            continue
        hits = tokens.count(folded)
        if hits >= _KEYWORD_MIN_HITS and hits / len(tokens) >= _KEYWORD_TOKEN_SHARE:
            return True
    return False
