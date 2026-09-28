"""allenai/RLVR-IFeval task.

The natural verifier is the constraint checker. The oracle is that checker and
a content judge: Tulu 3 shows RLVR can satisfy the checker by stuffing keywords
and emitting no meaningful answer. The default judge is
:class:`~vfates.tasks.judges.HeuristicJudge`.

``ground_truth`` is a flat JSON object. ``func_name`` selects the checker and
every other non-null field is a keyword argument (open-instruct
``IFEvalVerifierOld``). There is no nested ``args`` object; :meth:`Problem.meta`
stores that kwargs dict under ``args``.
"""

from __future__ import annotations

import threading
from typing import Any

from vfates.tasks.base import Task
from vfates.tasks.if_checkers import apply_checker, parse_ground_truth
from vfates.tasks.judges import HeuristicJudge, Judge
from vfates.types import Parsed, Problem
from vfates.utils.hashing import u01
from vfates.utils.io import sha256_text

DATASET = "allenai/RLVR-IFeval"


class IFEvalTask(Task):
    """RLVR-IFeval with a deterministic train/held-out split.

    ``problems(split, n, seed)`` returns the first ``n`` rows of the split in
    row-id order. A row is held out when ``u01(seed, row_id) < heldout_fraction``
    (default 0.33). ``n`` is a cap.

    ``parse`` sets ``answer`` to the SHA-256 of the stripped response (the
    clustering key) and ``raw`` to the response. An empty strip yields
    ``answer is None``.
    """

    name = "ifeval"

    def __init__(
        self,
        *,
        heldout_fraction: float = 0.33,
        judge: Judge | None = None,
        rows: list[dict[str, Any]] | None = None,
    ) -> None:
        if not 0.0 <= heldout_fraction <= 1.0:
            raise ValueError("heldout_fraction must lie in [0, 1]")
        self.heldout_fraction = float(heldout_fraction)
        self.judge: Judge = judge if judge is not None else HeuristicJudge()
        self._injected = rows
        self._rows: list[dict[str, Any]] | None = None
        self._lock = threading.Lock()

    def _records(self) -> list[dict[str, Any]]:
        if self._rows is None:
            with self._lock:
                if self._rows is None:
                    self._rows = list(self._injected) if self._injected is not None else _load_hf()
        return self._rows

    def _in_split(self, row_id: int, split: str, seed: int) -> bool:
        heldout = u01(seed, row_id) < self.heldout_fraction
        return heldout if split == "heldout" else not heldout

    def problems(self, split: str, n: int, seed: int) -> list[Problem]:
        if split not in ("train", "heldout"):
            raise ValueError(f"unknown split {split!r}")
        if n < 0:
            raise ValueError("n must be non-negative")
        chosen = [row for row in self._records() if self._in_split(int(row["row_id"]), split, seed)]
        chosen.sort(key=lambda row: int(row["row_id"]))
        return [self._problem(row, split) for row in chosen[:n]]

    def _problem(self, row: dict[str, Any], split: str) -> Problem:
        row_id = int(row["row_id"])
        return Problem(
            pid=f"ifeval/{split}/{row_id:05d}",
            split=split,
            task=self.name,
            question=str(row["question"]),
            answer=str(row["constraint"]),
            meta={
                "row_id": row_id,
                "func_name": row["func_name"],
                "args": dict(row["args"]),
                "constraint_type": row["constraint_type"],
                "constraint": row["constraint"],
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
        del parsed
        checker_ok = apply_checker(text, str(problem.meta["func_name"]), dict(problem.meta["args"]))
        judge_ok = int(self.judge.judge(problem.question, text)) == 1
        return int(checker_ok and judge_ok)


def _load_hf() -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(DATASET, split="train")
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


def _user_text(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content") or "")
    if messages:
        return str(messages[0].get("content") or "")
    return ""
