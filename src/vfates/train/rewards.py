"""Shared reward adapter: V trains, O / FP / FN / monitors are logged.

Draw keys make a resampled verifier a deterministic function of
``(seed, step, pid, index)``. The MLX trainer passes that tuple directly. TRL
0.25.1 calls each reward function as

.. code-block:: text

    fn(prompts=..., completions=..., completion_ids=..., trainer_state=..., **dataset_columns)

so ``make_trl_reward_funcs`` reads ``trainer_state.global_step`` when TRL supplies
it, plus the index of the completion in that call and the dataset ``pid``. The same
key is used for every reward function, so a stochastic arm is drawn once per
completion. Only the ``verifier`` function has nonzero ``reward_weights``; the
others exist so TRL logs ``rewards/<name>/mean``.

A train-pool eval at the same global step can reuse ``(step, pid, index)`` for a
different completion. Keys are unique inside one call. Across calls, include the
completion text only in the score cache, not in the draw, so the documented key
stays ``(seed, step, pid, index)``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from vfates.tasks.base import Task
from vfates.types import DrawKey, Parsed, Problem
from vfates.verifiers.base import Verifier

MonitorFn = Callable[[Problem, Parsed, str], bool]


class RewardContext:
    """Grades completions with one task oracle and one verifier."""

    def __init__(
        self,
        task: Task,
        verifier: Verifier,
        problems_by_pid: Mapping[str, Problem],
        seed: int,
        monitors: dict[str, MonitorFn] | None = None,
    ) -> None:
        self.task = task
        self.verifier = verifier
        self.problems_by_pid = dict(problems_by_pid)
        self.seed = int(seed)
        self.monitors = dict(monitors or {})
        self._cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    def draw_key(self, step: Any, pid: str, index: int) -> DrawKey:
        """Deterministic draw key ``(seed, step, pid, index)``."""
        return (self.seed, step, pid, int(index))

    def score(self, pid: str, text: str, key: DrawKey | None) -> dict[str, Any]:
        """Return V, O, FP, FN and monitor bits for one completion.

        ``FP = (1-O)V`` and ``FN = O(1-V)`` are the paper's per-sample indicators.
        ``key`` is forwarded to the verifier and is required when it is stochastic.
        """
        cache_id = (key, text)
        cached = self._cache.get(cache_id)
        if cached is not None:
            return cached
        try:
            problem = self.problems_by_pid[pid]
        except KeyError as exc:
            raise KeyError(f"No problem registered for pid {pid!r}") from exc
        parsed = self.task.parse(problem, text)
        o = int(self.task.oracle(problem, parsed, text))
        v = int(self.verifier.verdict(problem, parsed, text, o, key))
        monitor_bits = {name: bool(fn(problem, parsed, text)) for name, fn in self.monitors.items()}
        record = {
            "V": v,
            "O": o,
            "FP": (1 - o) * v,
            "FN": o * (1 - v),
            "answer": parsed.answer,
            "has_answer": parsed.answer is not None,
            "monitors": monitor_bits,
        }
        if len(self._cache) > 200_000:
            self._cache.clear()
        self._cache[cache_id] = record
        return record

    def score_batch(
        self,
        pids: Sequence[str],
        texts: Sequence[str],
        keys: Sequence[DrawKey | None],
    ) -> list[dict[str, Any]]:
        """Score aligned ``(pid, text, key)`` triples."""
        if not (len(pids) == len(texts) == len(keys)):
            raise ValueError(
                f"pids, texts and keys must have the same length, got "
                f"{len(pids)}, {len(texts)}, {len(keys)}"
            )
        return [self.score(pid, text, key) for pid, text, key in zip(pids, texts, keys)]


def monitor_conditionals(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | None]]:
    """Per-rule monitor rates, including the conditionals H5 reads.

    ``monitors`` is the unconditional mean. ``monitors_wrong`` is
    ``P(monitor | O=0)`` and ``monitors_correct`` is ``P(monitor | O=1)``.
    A conditional is ``None`` when that class has no samples. Rules with no
    monitor bits produce empty dicts.
    """
    names = sorted({name for record in records for name in record.get("monitors", {})})
    wrong = [record for record in records if int(record["O"]) == 0]
    correct = [record for record in records if int(record["O"]) == 1]

    def _mean(rows: Sequence[Mapping[str, Any]], name: str) -> float | None:
        if not rows:
            return None
        return sum(bool(row.get("monitors", {}).get(name, False)) for row in rows) / len(rows)

    return {
        "monitors": {name: _mean(records, name) for name in names},
        "monitors_wrong": {name: _mean(wrong, name) for name in names},
        "monitors_correct": {name: _mean(correct, name) for name in names},
    }


def onpolicy_rates(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pooled on-policy confusion rates.

    With indicators ``O``, ``V``, ``FP=(1-O)V``, ``FN=O(1-V)``:

    * ``fpr = sum(FP) / sum(1-O)`` (undefined, ``None``, if there is no incorrect sample)
    * ``fnr = sum(FN) / sum(O)``
    * ``precision = sum(O V) / sum(V)``
    * ``J = (1 - fnr) - fpr`` (Youden's J; ``None`` if either rate is undefined)
    * ``accept_rate = mean(V)``, ``oracle_acc = mean(O)``

    Monitor rates come from :func:`monitor_conditionals`.
    """
    if not records:
        return {
            "fpr": None,
            "fnr": None,
            "J": None,
            "precision": None,
            "accept_rate": None,
            "oracle_acc": None,
            "monitors": {},
            "monitors_wrong": {},
            "monitors_correct": {},
            "n": 0,
        }
    n = len(records)
    o = [int(r["O"]) for r in records]
    v = [int(r["V"]) for r in records]
    n_neg = sum(1 - oi for oi in o)
    n_pos = sum(o)
    n_acc = sum(v)
    fp = sum((1 - oi) * vi for oi, vi in zip(o, v))
    fn = sum(oi * (1 - vi) for oi, vi in zip(o, v))
    tp = sum(oi * vi for oi, vi in zip(o, v))
    fpr = None if n_neg == 0 else fp / n_neg
    fnr = None if n_pos == 0 else fn / n_pos
    precision = None if n_acc == 0 else tp / n_acc
    youden = None if fpr is None or fnr is None else (1.0 - fnr) - fpr
    return {
        "fpr": fpr,
        "fnr": fnr,
        "J": youden,
        "precision": precision,
        "accept_rate": n_acc / n,
        "oracle_acc": n_pos / n,
        **monitor_conditionals(records),
        "n": n,
    }


def reward_weights_for(n_funcs: int) -> list[float]:
    """TRL weights: the verifier trains, every other logged head has weight 0."""
    if n_funcs < 1:
        raise ValueError("need at least the verifier reward")
    return [1.0] + [0.0] * (n_funcs - 1)


def make_trl_reward_funcs(ctx: RewardContext) -> list[Callable[..., list[float]]]:
    """Reward callables whose ``__name__`` matches TRL's log keys.

    Order: ``verifier``, ``oracle``, ``false_positive``, ``false_negative``, then
    ``monitor_<name>`` in sorted monitor-name order. Pair with
    :func:`reward_weights_for` (``[1, 0, ...]``) so only ``verifier`` trains.

    Signature matches TRL 0.25.1: ``completions``, ``prompts``, ``completion_ids``,
    ``trainer_state``, and dataset columns as keywords (``pid`` is required).
    A resampled verifier's draw key is ``(seed, global_step, pid, in-call index)``.
    ``global_step`` is 0 when TRL does not pass ``trainer_state``. The index is the
    position in this call, so two completions of the same prompt in one call get
    distinct keys. Train and eval at the same ``global_step`` can reuse a key for
    the same ``(pid, index)``; the score cache is ``(key, text)``, so a different
    completion is re-scored, but a stochastic verifier redraws only when the key
    changes. MLX eval avoids that collision by using the step tag ``"eval"``.
    """
    specs: list[tuple[str, str]] = [
        ("verifier", "V"),
        ("oracle", "O"),
        ("false_positive", "FP"),
        ("false_negative", "FN"),
    ]
    for name in sorted(ctx.monitors):
        specs.append((f"monitor_{name}", name))

    funcs: list[Callable[..., list[float]]] = []
    for fname, field in specs:
        funcs.append(_make_reward_fn(ctx, fname, field, field in ctx.monitors))
    return funcs


def _make_reward_fn(
    ctx: RewardContext, name: str, field: str, is_monitor: bool
) -> Callable[..., list[float]]:
    def reward_fn(
        completions: Sequence[Any],
        prompts: Sequence[Any] | None = None,
        completion_ids: Sequence[Any] | None = None,
        trainer_state: Any = None,
        pid: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        del prompts, completion_ids, kwargs
        texts = [_completion_text(c) for c in completions]
        if pid is None:
            raise ValueError(
                f"Reward function {name!r} expected dataset column 'pid' "
                "(TRL forwards dataset columns as keywords)."
            )
        pids = list(pid)
        if len(pids) != len(texts):
            raise ValueError(f"{name}: pid length {len(pids)} != completions length {len(texts)}")
        step: Any = 0
        if trainer_state is not None and getattr(trainer_state, "global_step", None) is not None:
            step = int(trainer_state.global_step)
        out: list[float] = []
        for index, (one_pid, text) in enumerate(zip(pids, texts)):
            record = ctx.score(str(one_pid), text, ctx.draw_key(step, str(one_pid), index))
            if is_monitor:
                out.append(float(bool(record["monitors"][field])))
            else:
                out.append(float(record[field]))
        return out

    reward_fn.__name__ = name
    return reward_fn


def _completion_text(completion: Any) -> str:
    """Text TRL passes either as a string or as an assistant message list."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return _content_str(completion.get("content", ""))
    if isinstance(completion, list):
        assistant = ""
        for message in completion:
            if isinstance(message, str):
                assistant = message
            elif isinstance(message, dict) and message.get("role", "assistant") == "assistant":
                assistant = _content_str(message.get("content", ""))
        return assistant
    return str(completion)


def _content_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text", "")))
        return "".join(parts)
    return str(content)
