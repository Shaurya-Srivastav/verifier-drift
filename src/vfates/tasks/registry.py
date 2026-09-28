"""Build a :class:`~vfates.tasks.base.Task` from a config mapping.

Every experiment config names its task as ``{"type": <name>, **options}`` so the
same YAML drives audits, calibration and both trainers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vfates.tasks.base import Task

TASK_TYPES: tuple[str, ...] = ("chain_sum", "mbpp_plus", "ifeval")


def build_task(spec: Mapping[str, Any]) -> Task:
    """Instantiate the task named by ``spec['type']`` with the remaining options."""
    options = dict(spec)
    kind = options.pop("type", None)
    if kind == "chain_sum":
        from vfates.tasks.chain_sum import ChainSumConfig, ChainSumTask

        return ChainSumTask(ChainSumConfig.from_dict(options))
    if kind == "mbpp_plus":
        from vfates.tasks.mbpp import MBPPPlusTask

        return MBPPPlusTask(**options)
    if kind == "ifeval":
        from vfates.tasks.ifeval import IFEvalTask

        return IFEvalTask(**options)
    raise ValueError(f"unknown task type {kind!r}; expected one of {TASK_TYPES}")
