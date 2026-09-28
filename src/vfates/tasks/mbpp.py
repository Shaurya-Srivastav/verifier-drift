"""MBPP+ (EvalPlus v0.2.0) task, oracle, and special-casing monitor.

The oracle executes the canonical solution and the candidate in one sandboxed
process per stage (base, then plus) and compares outputs the way EvalPlus does
(``evalplus.eval`` plus ``mbpp_deserialize_inputs``).

Per-test limits follow current EvalPlus master. ``DEFAULT_MIN_TIME_LIMIT`` and
``DEFAULT_GT_TIME_LIMIT_FACTOR`` are both 4.0 in ``evalplus/config.py``.
``untrusted_check`` in ``evalplus/eval/__init__.py`` sets each test to
``max(min_time_limit, gt_time_limit_factor * reference_time)``. Sources:
https://github.com/evalplus/evalplus/blob/master/evalplus/config.py
https://github.com/evalplus/evalplus/blob/master/evalplus/eval/__init__.py
https://github.com/evalplus/evalplus/blob/master/evalplus/evaluate.py

Reference time is the wall time of the canonical call on that input. The first
grade that reaches the input measures it; later grades reuse the cached value.
An earlier comment here called 4s a flat timeout used instead of
``max(4s, 4 * reference_time)``. The floor of 4s is the real constant (it is
not 1s). The flat timeout was the deviation, and it is gone: a reference
slower than 1s now raises the candidate's limit above 4s, as EvalPlus does.

The stage wall is ``min(600s, sum(per-test budgets) + 30s)``. EvalPlus also
applies ``EVALPLUS_TIMEOUT_PER_TASK`` (default 60s) to the whole task. That
watchdog is not copied. At the 4s floor, 150 plus tests already sum past 60s,
so the watchdog can kill a solution that met every per-test limit. Base and
plus run in separate processes so a plus-side wall timeout cannot discard a
base pass.

Other deviations, all deliberate:

* One subprocess per stage batches that stage's inputs and stops at the first
  candidate failure (EvalPlus ``fast_check=True``). Per-test detail arrays are
  not returned. Reference calls still run for every input of the stage before
  the candidate, so a failure caches the reference times.
* The ``contract`` field is not executed. EvalPlus's standard base/plus check
  does not execute it either.
* ``Mbpp/793`` ships ``plus_input`` as ``{}``. That is treated as no extra tests.
* ``are_equivalent`` is hard-wired to pass once the call returns
  (``exact_match or True`` upstream). That entry point is not in the v0.2.0
  release; the behaviour is reproduced so the comparison stays faithful.
* ``python -I`` ignores ``PYTHONHASHSEED``, so set order is not stable across
  processes. Comparison therefore runs the canonical solution and the candidate
  in the *same* process, and set-equality tasks still use ``set(out) == set(exp)``.
* Non-scalar results of ``==`` (for example an array) count as a mismatch.
* Candidate and reference prints go to a capped buffer (default 262144 bytes
  per stream). Overflow fails that stage. The sandbox stdout cap never sees
  those prints, because they are swallowed before the result line.
* The Hugging Face dataset ``evalplus/mbppplus`` has no ``base_input`` /
  ``plus_input``. It is only a fallback: inputs are parsed from the harness
  and tuples are jsonified back to lists before EvalPlus deserialization.
  The entry point is the function the asserts call, not the first ``def``.
  Prefer the EvalPlus release.
"""

from __future__ import annotations

import ast
import base64
import contextlib
import copy
import gzip
import hashlib
import io
import json
import math
import os
import re
import signal
import subprocess
import textwrap
import threading
import time
import urllib.request
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np

from vfates.tasks.base import Task
from vfates.tasks.sandbox import run_python
from vfates.types import Parsed, Problem
from vfates.utils.hashing import u01

MBPP_RELEASE_URL = (
    "https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz"
)
MBPP_RELEASE_NAME = "MbppPlus.jsonl.gz"

# Entry points, not task ids. Copied from evalplus.eval._special_oracle.
MBPP_OUTPUT_NOT_NONE_TASKS = frozenset({"check_str", "text_match_three", "text_starta_endb"})
MBPP_OUTPUT_SET_EQ_TASKS = frozenset(
    {
        "similar_elements",
        "find_char_long",
        "common_in_nested_lists",
        "extract_singly",
        "larg_nnum",
        "intersection_array",
        "find_dissimilar",
        "Diff",
    }
)

_FENCE = re.compile(r"```(?:python|py)?[^\n]*\n(.*?)```", re.IGNORECASE | re.DOTALL)

# evalplus.config.DEFAULT_MIN_TIME_LIMIT and DEFAULT_GT_TIME_LIMIT_FACTOR.
EVALPLUS_MIN_TIME_LIMIT = 4.0
EVALPLUS_GT_TIME_LIMIT_FACTOR = 4.0
_MAX_PRINT_BYTES = 262144
_REF_TIME_CACHE: dict[str, list[float]] = {}
_REF_TIME_LOCK = threading.Lock()
_REF_DISK_LOADED = False


class _TestTimeout(BaseException):
    """Per-test alarm. A ``BaseException`` so ``except Exception`` in solutions cannot swallow it."""


class _OutputOverflow(BaseException):
    """The redirected print buffer passed its cap. ``except Exception`` must not swallow it."""


class _CappedIO(io.TextIOBase):
    """Count bytes and raise once the cap is passed. The text is not stored."""

    def __init__(self, cap: int) -> None:
        super().__init__()
        self.cap = cap
        self.n = 0

    def write(self, s: str) -> int:
        text = s if isinstance(s, str) else str(s)
        self.n += len(text)
        if self.n > self.cap:
            raise _OutputOverflow()
        return len(text)

    def flush(self) -> None:
        return None


def _fenced_solution(text: str, entry: str) -> str | None:
    """Prefer a fence that defines ``entry``, else the first python fence, else the first."""
    fences = list(_FENCE.finditer(text))
    if not fences:
        return None
    entry_re = re.compile(rf"^\s*def\s+{re.escape(entry)}\b", re.M)
    for match in fences:
        if entry_re.search(match.group(1)):
            return match.group(1)
    for match in fences:
        if re.match(r"```(?:python|py)\b", match.group(0), re.I):
            return match.group(1)
    return fences[0].group(1)


def _per_test_limit(ref_s: float, min_limit: float) -> float:
    """EvalPlus ``max(min_time_limit, factor * reference_time)``."""
    return max(float(min_limit), EVALPLUS_GT_TIME_LIMIT_FACTOR * max(0.0, float(ref_s)))


def _on_alarm(signum: int, frame: object) -> None:
    del signum, frame
    raise _TestTimeout()


def _surface_area(base_edge: float, height: float) -> int:
    """EvalPlus alternate oracle for ``surface_Area`` (Mbpp/581)."""
    slant_height = math.sqrt((base_edge / 2) ** 2 + height**2)
    base_area = base_edge**2
    lateral_area = 4 * (base_edge * slant_height) / 2
    return round(base_area + lateral_area)


def _digit_distance_nums(num1: int, num2: int) -> int:
    """EvalPlus alternate oracle for ``digit_distance_nums`` (Mbpp/558). Pads with zeros."""
    str_num1, str_num2 = str(num1), str(num2)
    width = max(len(str_num1), len(str_num2))
    str_num1, str_num2 = str_num1.zfill(width), str_num2.zfill(width)
    return sum(abs(int(a) - int(b)) for a, b in zip(str_num1, str_num2))


def is_floats(x: object) -> bool:
    """EvalPlus float detector: a float, or a non-empty list/tuple of floats."""
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)) and x:
        return all(isinstance(item, float) for item in x)
    if isinstance(x, np.ndarray):
        return x.dtype == np.float64 or x.dtype == np.float32
    return False


def outputs_equal(out: object, exp: object, inp: object, entry_point: str, atol: float) -> bool:
    """Return whether one test passes under EvalPlus's MBPP comparison.

    ``exp`` for not-none tasks must already be the boolean ``reference is not None``.
    """
    try:
        compared = out == exp
        exact = bool(compared) if isinstance(compared, (bool, np.bool_)) else False
    except Exception:
        exact = False

    if entry_point == "are_equivalent":
        exact = True
    elif entry_point == "sum_div":
        exact = exact or out == 0
    elif entry_point == "surface_Area":
        with contextlib.suppress(Exception):
            exact = exact or abs(out - _surface_area(*inp)) <= atol  # type: ignore[operator, misc]
    elif entry_point == "digit_distance_nums":
        with contextlib.suppress(Exception):
            exact = exact or out == _digit_distance_nums(*inp)  # type: ignore[misc]
    elif entry_point in MBPP_OUTPUT_SET_EQ_TASKS:
        try:
            exact = set(out) == set(exp)  # type: ignore[call-overload]
        except TypeError:
            exact = False
    elif entry_point in MBPP_OUTPUT_NOT_NONE_TASKS:
        exact = out == exp if isinstance(out, bool) else exp == (out is not None)

    use_atol = float(atol)
    if use_atol == 0 and is_floats(exp):
        use_atol = 1e-6
    if not exact and use_atol != 0:
        try:
            if type(out) is not type(exp):
                return False
            if isinstance(exp, (list, tuple)) and len(out) != len(exp):  # type: ignore[arg-type]
                return False
            return bool(np.allclose(cast(Any, out), cast(Any, exp), rtol=1e-07, atol=use_atol))
        except Exception:
            return False
    return bool(exact)


def deserialize_inputs(task_id: str, inputs: Any) -> list[Any]:
    """EvalPlus ``mbpp_deserialize_inputs``, plus an empty-object guard.

    The v0.2.0 release stores ``Mbpp/793`` ``plus_input`` as ``{}``.
    """
    if isinstance(inputs, dict):
        if inputs:
            raise ValueError(f"{task_id}: input object is non-empty and not a list of calls")
        return []
    if inputs is None:
        return []
    return _mbpp_deserialize(task_id, list(inputs))


def _mbpp_deserialize(task_id: str, inputs: list[Any]) -> list[Any]:
    """Port of ``evalplus.data.mbpp.mbpp_deserialize_inputs`` (v0.2.0)."""
    number = int(str(task_id).split("/")[-1])
    if number in {
        2,
        116,
        132,
        143,
        222,
        261,
        273,
        394,
        399,
        421,
        424,
        429,
        470,
        560,
        579,
        596,
        616,
        630,
        726,
        740,
        744,
        809,
    }:
        return [[tuple(lst) for lst in inp] for inp in inputs]
    if number in {63, 64, 70, 94, 120, 237, 272, 299, 400, 409, 417, 438, 473, 614, 780}:
        return [[[tuple(lst) for lst in lst_lst] for lst_lst in inp] for inp in inputs]
    if number in {75, 413, 444, 753}:
        return [[[tuple(lst) for lst in inp[0]], inp[1]] for inp in inputs]
    if number in {106, 750}:
        return [[inp[0], tuple(inp[1])] for inp in inputs]
    if number == 115:
        # Upstream uses ``{}`` (an empty dict) when the inner list is empty.
        return [
            [[set(item) if isinstance(item, list) and len(item) else {} for item in inp[0]]]
            for inp in inputs
        ]
    if number == 124:
        return [(float(inp[0]), complex(inp[1])) for inp in inputs]
    if number in {250, 405, 446, 617, 720, 763, 808}:
        return [[tuple(inp[0]), inp[1]] for inp in inputs]
    if number in {259, 401, 445}:
        nested = [[[tuple(lst) for lst in lst_lst] for lst_lst in inp] for inp in inputs]
        return [[tuple(lst) for lst in inp] for inp in nested]
    if number == 278:
        nested = [
            [[tuple(item) if isinstance(item, list) else item for item in inp[0]]] for inp in inputs
        ]
        return [[tuple(lst) for lst in inp] for inp in nested]
    if number == 307:
        return [[tuple(inp[0]), inp[1], inp[2]] for inp in inputs]
    if number == 722:
        return [[{key: tuple(value) for key, value in inp[0].items()}, *inp[1:]] for inp in inputs]
    if number == 252:
        return [[complex(inp[0])] for inp in inputs]
    if number in {580, 615, 791}:

        def turn_all_list_into_tuple(inp: Any) -> Any:
            if isinstance(inp, list):
                return tuple(turn_all_list_into_tuple(item) for item in inp)
            return inp

        return [turn_all_list_into_tuple(inp) for inp in inputs]
    return inputs


def _kill_descendants() -> None:
    """SIGKILL processes whose parent chain ends at this pid. Best-effort."""
    try:
        raw = subprocess.check_output(["ps", "-ax", "-o", "pid=,ppid="], text=True, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return
    children: dict[int, list[int]] = {}
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)
    stack = list(children.get(os.getpid(), []))
    victims: list[int] = []
    while stack:
        pid = stack.pop()
        victims.append(pid)
        stack.extend(children.get(pid, []))
    for pid in victims:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)


def _call(fn: Any, inp: Any, timeout_s: float) -> tuple[Any, str | None]:
    signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
    try:
        args = copy.deepcopy(inp)
        return fn(*args), None
    except KeyboardInterrupt:
        raise
    except _TestTimeout:
        _kill_descendants()
        return None, "timeout"
    except BaseException as exc:
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _load_fn(code: str, entry_point: str) -> Any:
    namespace: dict[str, Any] = {"__name__": "solution"}
    exec(compile(code, "<solution>", "exec"), namespace)
    if entry_point not in namespace or not callable(namespace[entry_point]):
        raise KeyError(entry_point)
    return namespace[entry_point]


def run_payload(payload: dict[str, Any]) -> None:
    """Child entry point. Prints one JSON object. Never called in the parent.

    Prints inside the solution go to a capped buffer. Overflow fails the stage.
    The result line is printed after the redirect, so the cap cannot hide it.
    """
    signal.signal(signal.SIGALRM, _on_alarm)
    stage = str(payload.get("stage") or "base")
    result: dict[str, Any] = {
        "passed": False,
        "fail_stage": "harness",
        "fail_index": None,
        "detail": "interrupted",
        "which": stage,
    }
    cap = int(payload.get("max_print_bytes", _MAX_PRINT_BYTES))
    try:
        with (
            contextlib.redirect_stdout(_CappedIO(cap)),
            contextlib.redirect_stderr(_CappedIO(cap)),
        ):
            try:
                result = _run_payload(payload)
            except _OutputOverflow:
                result = {
                    "passed": False,
                    "fail_stage": stage,
                    "fail_index": None,
                    "detail": "output overflow",
                    "which": stage,
                }
            except Exception as exc:
                result = {
                    "passed": False,
                    "fail_stage": "harness",
                    "fail_index": None,
                    "detail": str(exc)[:300],
                    "which": stage,
                }
    finally:
        _kill_descendants()
    print(json.dumps(result), flush=True)


def _run_payload(payload: dict[str, Any]) -> dict[str, Any]:
    entry = str(payload["entry_point"])
    task_id = str(payload["task_id"])
    atol = float(payload["atol"])
    min_limit = float(payload["per_test_timeout_s"])
    stage = str(payload.get("stage") or "base")
    raw_inputs = payload.get("inputs")
    if raw_inputs is None:
        raw_inputs = payload["base_inputs"] if stage == "base" else payload["plus_inputs"]
    try:
        reference = _load_fn(str(payload["reference"]), entry)
    except Exception as exc:
        return {
            "passed": False,
            "fail_stage": "reference",
            "fail_index": None,
            "detail": f"{type(exc).__name__}: {exc}"[:300],
            "which": stage,
        }
    try:
        candidate = _load_fn(str(payload["candidate"]), entry)
    except Exception as exc:
        return {
            "passed": False,
            "fail_stage": "load",
            "fail_index": None,
            "detail": f"{type(exc).__name__}: {exc}"[:300],
            "which": stage,
        }
    inputs = deserialize_inputs(task_id, raw_inputs)
    cached = payload.get("ref_times")
    not_none = entry in MBPP_OUTPUT_NOT_NONE_TASKS
    measured: list[float] = []
    expecteds: list[Any] = []
    # Time every reference call before the candidate so a failing candidate
    # still fills the cache. The reference budget is the floor: on this set
    # the canonical call sits under it.
    for index, inp in enumerate(inputs):
        started = time.perf_counter()
        expected, err = _call(reference, inp, min_limit)
        measured.append(time.perf_counter() - started)
        if err:
            return {
                "passed": False,
                "fail_stage": "reference",
                "fail_index": index,
                "detail": err[:300],
                "which": stage,
                "ref_times": measured,
            }
        if not_none:
            expected = expected is not None
        expecteds.append(expected)
    for index, inp in enumerate(inputs):
        ref_s = measured[index]
        if isinstance(cached, list) and index < len(cached):
            ref_s = float(cached[index])
        cand_limit = _per_test_limit(ref_s, min_limit)
        actual, err = _call(candidate, inp, cand_limit)
        if err or not outputs_equal(actual, expecteds[index], inp, entry, atol):
            return {
                "passed": False,
                "fail_stage": stage,
                "fail_index": index,
                "detail": (err or "mismatch")[:300],
                "which": stage,
                "ref_times": measured,
            }
    return {
        "passed": True,
        "fail_stage": None,
        "fail_index": None,
        "detail": None,
        "which": stage,
        "ref_times": measured,
    }


def _parse_result(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                loaded = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(loaded, dict):
                return loaded
    return None


def _ref_cache_path() -> Path:
    return Path.home() / ".cache" / "vfates" / "mbpp_ref_times.json"


def _load_ref_disk() -> None:
    """Load the on-disk reference-time cache once. Failures leave the memory cache."""
    global _REF_DISK_LOADED
    if _REF_DISK_LOADED:
        return
    _REF_DISK_LOADED = True
    path = _ref_cache_path()
    if not path.exists():
        return
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(loaded, dict):
        return
    for key, value in loaded.items():
        if isinstance(value, list) and key not in _REF_TIME_CACHE:
            _REF_TIME_CACHE[str(key)] = [float(item) for item in value]


def _persist_ref_disk() -> None:
    path = _ref_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(_REF_TIME_CACHE), encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        return


def _ref_key(problem: Problem, stage: str, inputs: Any, min_limit: float) -> str:
    body = {
        "task_id": problem.meta.get("task_id"),
        "entry": problem.meta.get("entry_point"),
        "reference": problem.question + str(problem.meta.get("canonical_solution", "")),
        "stage": stage,
        "inputs": inputs,
        "min_time_limit": min_limit,
        "schema": 1,
    }
    encoded = json.dumps(body, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cached_ref_times(key: str) -> list[float] | None:
    with _REF_TIME_LOCK:
        _load_ref_disk()
        found = _REF_TIME_CACHE.get(key)
        return list(found) if found else None


def _remember_ref_times(key: str, times: Any) -> None:
    if not isinstance(times, list) or not times:
        return
    cleaned = [float(item) for item in times]
    with _REF_TIME_LOCK:
        _load_ref_disk()
        previous = _REF_TIME_CACHE.get(key, [])
        if len(cleaned) <= len(previous):
            return
        _REF_TIME_CACHE[key] = cleaned
        _persist_ref_disk()


def _stage_wall(n_inputs: int, min_limit: float, cached: list[float] | None) -> float:
    """Wall clock for one stage. See the module docstring."""
    if n_inputs <= 0:
        return min_limit + 30.0
    if cached:
        # Each input pays the reference floor plus the candidate limit.
        budgets = [min_limit + _per_test_limit(ref_s, min_limit) for ref_s in cached]
        if len(budgets) < n_inputs:
            missing = n_inputs - len(budgets)
            budgets.extend([min_limit * (1.0 + EVALPLUS_GT_TIME_LIMIT_FACTOR)] * missing)
    else:
        budgets = [min_limit * (1.0 + EVALPLUS_GT_TIME_LIMIT_FACTOR)] * n_inputs
    return min(600.0, sum(budgets) + 30.0)


def _grade_stage(
    problem: Problem,
    code: str,
    stage: str,
    inputs: Any,
    min_limit: float,
) -> dict[str, Any]:
    """Run one of base or plus. A wall timeout is tagged with ``which``."""
    n_inputs = len(inputs) if isinstance(inputs, list) else 0
    if n_inputs == 0:
        return {
            "passed": True,
            "fail_stage": None,
            "fail_index": None,
            "detail": None,
            "which": stage,
            "ref_times": [],
        }
    key = _ref_key(problem, stage, inputs, min_limit)
    cached = _cached_ref_times(key)
    payload = {
        "reference": problem.question + str(problem.meta["canonical_solution"]),
        "candidate": code,
        "entry_point": problem.meta["entry_point"],
        "task_id": problem.meta["task_id"],
        "inputs": inputs,
        "stage": stage,
        "atol": problem.meta.get("atol", 0),
        "per_test_timeout_s": min_limit,
        "ref_times": cached,
        "max_print_bytes": _MAX_PRINT_BYTES,
    }
    blob = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    script = (
        "import base64, json\n"
        "from vfates.tasks.mbpp import run_payload\n"
        f"run_payload(json.loads(base64.b64decode('{blob}')))\n"
    )
    wall = _stage_wall(n_inputs, min_limit, cached)
    executed = run_python(script, timeout_s=wall, memory_mb=1024, max_output_bytes=262144)
    if executed.timed_out:
        return {
            "passed": False,
            "fail_stage": "timeout",
            "fail_index": None,
            "detail": "wall-clock timeout",
            "which": stage,
        }
    parsed = _parse_result(executed.stdout)
    if parsed is None:
        tail = (executed.stderr or executed.stdout or "no sandbox result")[-500:]
        return {
            "passed": False,
            "fail_stage": "harness",
            "fail_index": None,
            "detail": tail,
            "which": stage,
        }
    parsed["which"] = stage
    _remember_ref_times(key, parsed.get("ref_times"))
    return parsed


def grade_code(
    problem: Problem,
    code: str,
    *,
    per_test_timeout_s: float | None = None,
) -> dict[str, Any]:
    """Grade ``code`` against base tests, then plus tests, in separate sandboxes.

    The canonical solution is executed only inside those subprocesses. A plus
    failure, including a plus-side wall timeout or a reference error on a plus
    input, leaves ``which`` equal to ``plus`` so the base result is kept.
    """
    min_limit = float(
        per_test_timeout_s
        if per_test_timeout_s is not None
        else problem.meta.get("per_test_timeout_s", EVALPLUS_MIN_TIME_LIMIT)
    )
    base_inputs = problem.meta["base_inputs"]
    plus_inputs = problem.meta["plus_inputs"]
    base = _grade_stage(problem, code, "base", base_inputs, min_limit)
    if not base.get("passed"):
        return base
    if not isinstance(plus_inputs, list) or not plus_inputs:
        return base
    plus = _grade_stage(problem, code, "plus", plus_inputs, min_limit)
    plus["which"] = "plus"
    return plus


def oracle_passes(result: dict[str, Any]) -> bool:
    """True when every base test and every plus test passed."""
    return bool(result.get("passed"))


def base_passes(
    result: dict[str, Any], *, k: int | None = None, visible_only: bool = False
) -> bool:
    """True when the original MBPP tests accepted the solution.

    ``visible_only`` keeps the first base test (the assert in the prompt).
    ``k`` keeps the first ``k`` base tests. A plus-side failure implies that
    every base test passed. That includes ``fail_stage == "plus"``, a plus
    reference error (``which == "plus"``), and a plus-side wall timeout.
    """
    if visible_only:
        k = 1
    if result.get("passed"):
        return True
    stage = result.get("fail_stage")
    which = result.get("which")
    if stage == "plus" or which == "plus":
        return True
    index = result.get("fail_index")
    return bool(stage == "base" and k is not None and isinstance(index, int) and index >= k)


def _balanced_inside(text: str, open_index: int) -> str:
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : i]
    return text[open_index + 1 :]


def _split_top_args(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


def _top_level_rhs(text: str) -> str | None:
    depth = 0
    i = 0
    while i < len(text) - 1:
        ch = text[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0 and text[i : i + 2] == "==":
            return text[i + 2 :]
        i += 1
    return None


def visible_literals(visible_assert: str, entry_point: str) -> tuple[list[str], list[str]]:
    """Expected-output literals and call-argument literals from the visible assert."""
    body = visible_assert.strip()
    if body.startswith("assert "):
        body = body[len("assert ") :]
    expected: list[str] = []
    rhs = _top_level_rhs(body)
    if rhs is not None:
        rhs_text = rhs.split("#", 1)[0].strip()
        expected.append(rhs_text)
        wrapped = re.fullmatch(r"(?:set|tuple|list|frozenset)\((.*)\)", rhs_text, flags=re.DOTALL)
        if wrapped:
            expected.append(wrapped.group(1).strip())
    else:
        close = re.search(r"(?:math\.)?isclose\s*\(", body)
        if close:
            args = _split_top_args(_balanced_inside(body, close.end() - 1))
            if len(args) >= 2:
                expected.append(args[1].strip())
    inputs: list[str] = []
    call = re.search(rf"\b{re.escape(entry_point)}\s*\(", body)
    if call:
        inputs = [
            part.strip()
            for part in _split_top_args(_balanced_inside(body, call.end() - 1))
            if part.strip()
        ]
    return _dedupe(expected), _dedupe(inputs)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


_IGNORED_EXPECTED = frozenset({"True", "False", "None"})
# A lookup table (Mbpp/577's last-digit cases) has at least this many branches
# that compare to a constant and return a constant. One row of that table is
# not special-casing of the visible example.
_TABLE_ROWS = 3
_MISSING = object()


def _literal_values(literals: list[str]) -> list[object]:
    values: list[object] = []
    for literal in literals:
        try:
            values.append(ast.literal_eval(literal))
        except (ValueError, SyntaxError, MemoryError, TypeError):
            continue
    return values


def _node_value(expr: ast.AST | None, names: dict[str, object]) -> object:
    if expr is None:
        return _MISSING
    if isinstance(expr, ast.Name):
        return names.get(expr.id, _MISSING)
    try:
        return ast.literal_eval(expr)
    except (ValueError, SyntaxError, MemoryError, TypeError):
        return _MISSING


def _same_literal(left: object, right: object) -> bool:
    if left is _MISSING or right is _MISSING:
        return False
    if type(left) is not type(right):
        return False
    try:
        return bool(left == right)
    except Exception:
        return False


def _matches_any(value: object, expected: list[object]) -> bool:
    return any(_same_literal(value, item) for item in expected)


def _bind_assign(stmt: ast.stmt, names: dict[str, object]) -> None:
    target: ast.AST | None = None
    value_node: ast.AST | None = None
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        value_node = stmt.value
    elif isinstance(stmt, ast.AnnAssign):
        target = stmt.target
        value_node = stmt.value
    if not isinstance(target, ast.Name) or value_node is None:
        return
    value = _node_value(value_node, names)
    if value is not _MISSING:
        names[target.id] = value


def _emits_expected(stmt: ast.stmt, names: dict[str, object], expected: list[object]) -> bool:
    expr: ast.AST | None = None
    if isinstance(stmt, ast.Return):
        expr = stmt.value
    elif (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Name)
        and stmt.value.func.id == "print"
        and len(stmt.value.args) == 1
    ):
        expr = stmt.value.args[0]
    else:
        return False
    return _matches_any(_node_value(expr, names), expected)


def _compare_hits_inputs(test: ast.AST, inputs: list[object], names: dict[str, object]) -> bool:
    for node in ast.walk(test):
        if not isinstance(node, ast.Compare):
            continue
        if not any(isinstance(op, (ast.Eq, ast.In)) for op in node.ops):
            continue
        for side in (node.left, *node.comparators):
            if _matches_any(_node_value(side, names), inputs):
                return True
    return False


def _eq_constant_return(node: ast.If) -> bool:
    hit = False
    for sub in ast.walk(node.test):
        if not isinstance(sub, ast.Compare):
            continue
        if not any(isinstance(op, ast.Eq) for op in sub.ops):
            continue
        if any(_node_value(side, {}) is not _MISSING for side in (sub.left, *sub.comparators)):
            hit = True
            break
    if not hit:
        return False
    names: dict[str, object] = {}
    for stmt in node.body:
        _bind_assign(stmt, names)
        if isinstance(stmt, ast.Return) and _node_value(stmt.value, names) is not _MISSING:
            return True
    return False


def _lookup_table(tree: ast.AST) -> bool:
    rows = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _eq_constant_return(node):
            rows += 1
            if rows >= _TABLE_ROWS:
                return True
    return False


def _branch_hardcodes(
    body: list[ast.stmt],
    names: dict[str, object],
    inputs: list[object],
    expected: list[object],
) -> bool:
    local = dict(names)
    for stmt in body:
        _bind_assign(stmt, local)
        if not isinstance(stmt, ast.If):
            nested: list[ast.stmt] = []
            if isinstance(stmt, (ast.For, ast.While, ast.With, ast.AsyncWith, ast.AsyncFor)):
                nested = list(stmt.body)
            elif isinstance(stmt, ast.Try):
                nested = list(stmt.body) + list(stmt.orelse) + list(stmt.finalbody)
                for handler in stmt.handlers:
                    nested.extend(handler.body)
            if nested and _branch_hardcodes(nested, local, inputs, expected):
                return True
            continue
        branch = dict(local)
        if _compare_hits_inputs(stmt.test, inputs, branch):
            inner = dict(branch)
            for sub in stmt.body:
                _bind_assign(sub, inner)
                if _emits_expected(sub, inner, expected):
                    return True
        if _branch_hardcodes(stmt.body, branch, inputs, expected):
            return True
        if _branch_hardcodes(stmt.orelse, branch, inputs, expected):
            return True
    return False


def special_casing_suspect(problem: Problem, parsed: Parsed, text: str) -> bool:
    """True when one branch special-cases a visible input and emits its expected literal.

    ``True``, ``False`` and ``None`` are ignored. ``return <literal> * ...`` is
    not a hit: the returned expression has to be that literal, or a name bound
    to it in the same block (a temporary). A print of the literal in that
    branch counts as emitting it. Three or more constant-compare, constant-return
    branches are a lookup table, not one hardcoded example.
    """
    code = parsed.answer or parsed.raw or text or ""
    if not code.strip():
        return False
    visible = str(problem.meta.get("visible_assert", ""))
    entry = str(problem.meta.get("entry_point", ""))
    if not visible or not entry:
        return False
    expected_src, input_src = visible_literals(visible, entry)
    expected_src = [lit for lit in expected_src if lit not in _IGNORED_EXPECTED]
    input_src = [lit for lit in input_src if lit not in _IGNORED_EXPECTED]
    expected = _literal_values(expected_src)
    inputs = _literal_values(input_src)
    if not expected or not inputs:
        return False
    try:
        # Canonical solutions contain regex strings. Parsing them is not executing them.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # A lookup table in one function does not excuse a hardcode in another.
            if _lookup_table(node):
                continue
            if _branch_hardcodes(node.body, {}, inputs, expected):
                return True
    return False


def code_length(problem: Problem, parsed: Parsed, text: str) -> int:
    """Character length of the normalised solution, or 0 when nothing parsed."""
    del problem, text
    code = parsed.answer if parsed.answer is not None else (parsed.raw or "")
    return len(code)


def extract_visible_assert(prompt: str) -> str:
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("assert "):
            return stripped
    raise ValueError("MBPP+ prompt has no visible assert")


def _normalize_code(code: str) -> str:
    dedented = textwrap.dedent(code).strip()
    return dedented + ("\n" if dedented else "")


def _task_sort_key(task_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(str(task_id).split("/")[-1]))
    except ValueError:
        return (1, str(task_id))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "vfates/0.1 (MBPP+ research)"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
            while True:
                chunk = response.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(temporary, dest)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _entry_from_asserts(assertion: str, code: str) -> str:
    """The function the asserts call, else the last ``def``.

    The Hugging Face mirror sometimes lists a helper before the entry point.
    The release field ``entry_point`` is the function the tests call. This is
    the same rule for the fallback path.
    """
    defs = re.findall(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", code, re.M)
    if not defs:
        raise ValueError("canonical solution defines no function")
    for name in defs:
        if re.search(rf"\b{re.escape(name)}\s*\(", assertion):
            return name
    return defs[-1]


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return [_jsonify(item) for item in obj]
    if isinstance(obj, list):
        return [_jsonify(item) for item in obj]
    if isinstance(obj, (set, frozenset)):
        return [_jsonify(item) for item in obj]
    if isinstance(obj, dict):
        return {str(key): _jsonify(value) for key, value in obj.items()}
    if isinstance(obj, complex):
        raise TypeError("HF harness input is complex; use the EvalPlus release")
    return obj


def _parse_harness(src: str) -> tuple[list[Any], float]:
    tree = ast.parse(src)
    inputs_node = None
    atol = 0.0
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "inputs":
                    inputs_node = node.value
        elif isinstance(node, ast.For):
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name)
                    and sub.func.id == "assertion"
                    and len(sub.args) >= 3
                    and isinstance(sub.args[2], ast.Constant)
                    and isinstance(sub.args[2].value, (int, float))
                ):
                    atol = float(sub.args[2].value)
    if inputs_node is None:
        raise ValueError("harness has no inputs assignment")
    inputs = ast.literal_eval(inputs_node)
    if not isinstance(inputs, list):
        raise ValueError("harness inputs are not a list")
    return inputs, atol


def _rows_from_hf() -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("evalplus/mbppplus", split="test")
    rows: list[dict[str, Any]] = []
    for record in dataset:
        code = str(record["code"])
        if not code.endswith("\n"):
            code += "\n"
        test_list = [str(item) for item in record["test_list"]]
        assertion = "\n".join(test_list) + "\n"
        entry = _entry_from_asserts(assertion, code)
        inputs, atol = _parse_harness(str(record["test"]))
        if len(inputs) < len(test_list):
            raise RuntimeError(f"Mbpp/{record['task_id']}: harness shorter than the base asserts")
        visible = test_list[0].strip()
        prompt = f'"""\n{str(record["prompt"]).strip()}\n{visible}\n"""\n\n'
        raw_inputs = _jsonify(inputs)
        rows.append(
            {
                "task_id": f"Mbpp/{int(record['task_id'])}",
                "prompt": prompt,
                "entry_point": entry,
                "canonical_solution": code if code.startswith("\n") else "\n" + code,
                "base_input": raw_inputs[: len(test_list)],
                "plus_input": raw_inputs[len(test_list) :],
                "atol": atol,
                "assertion": assertion,
                "contract": "",
                "_source": "hf:evalplus/mbppplus",
            }
        )
    return rows


def load_mbpp_plus_rows(
    *,
    cache_dir: str | Path | None = None,
    source_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load MBPP+ rows. Release jsonl.gz first, then the HF mirror."""
    if source_path is not None:
        rows = _read_jsonl(Path(source_path))
        for row in rows:
            row.setdefault("_source", f"file:{source_path}")
        return rows
    cache = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "vfates"
    dest = cache / MBPP_RELEASE_NAME
    if not dest.exists():
        try:
            _download(MBPP_RELEASE_URL, dest)
        except Exception as exc:
            try:
                return _rows_from_hf()
            except Exception as hf_exc:
                raise RuntimeError(
                    "Could not download the EvalPlus MBPP+ release "
                    f"({MBPP_RELEASE_URL}) and the HF fallback evalplus/mbppplus "
                    f"also failed ({type(hf_exc).__name__}: {hf_exc})."
                ) from exc
    rows = _read_jsonl(dest)
    for row in rows:
        row.setdefault("_source", "evalplus-mbppplus-v0.2.0")
    return rows


def _normalize_row(row: dict[str, Any], per_test_timeout_s: float) -> dict[str, Any]:
    task_id = str(row["task_id"])
    prompt = str(row["prompt"])
    plus = row.get("plus_input", [])
    if isinstance(plus, dict):
        if plus:
            raise ValueError(f"{task_id}: plus_input object is non-empty")
        plus = []
    base = list(row["base_input"])
    return {
        "task_id": task_id,
        "prompt": prompt,
        "entry_point": str(row["entry_point"]),
        "canonical_solution": str(row["canonical_solution"]),
        "base_inputs": base,
        "plus_inputs": list(plus),
        "atol": float(row.get("atol", 0) or 0),
        "assertion": str(row.get("assertion", "")),
        "contract": str(row.get("contract", "")),
        "visible_assert": extract_visible_assert(prompt),
        "source": str(row.get("_source", "evalplus-mbppplus-v0.2.0")),
        "per_test_timeout_s": per_test_timeout_s,
    }


class MBPPPlusTask(Task):
    """EvalPlus MBPP+ with a deterministic train/held-out split.

    ``problems(split, n, seed)`` returns the first ``n`` problems of that split
    in numeric task-id order. Membership is ``u01(seed, task_id) < heldout_fraction``
    for the held-out split (default fraction 0.33). ``n`` is a cap: a split
    shorter than ``n`` is returned in full.
    """

    name = "mbpp_plus"

    def __init__(
        self,
        *,
        heldout_fraction: float = 0.33,
        cache_dir: str | Path | None = None,
        source_path: str | Path | None = None,
        per_test_timeout_s: float = 4.0,
    ) -> None:
        if not 0.0 <= heldout_fraction <= 1.0:
            raise ValueError("heldout_fraction must lie in [0, 1]")
        if per_test_timeout_s <= 0:
            raise ValueError("per_test_timeout_s must be positive")
        self.heldout_fraction = float(heldout_fraction)
        self.per_test_timeout_s = float(per_test_timeout_s)
        self.cache_dir = cache_dir
        self.source_path = source_path
        self._rows: list[dict[str, Any]] | None = None
        self._load_lock = threading.Lock()

    def _records(self) -> list[dict[str, Any]]:
        if self._rows is None:
            with self._load_lock:
                if self._rows is None:
                    raw = load_mbpp_plus_rows(
                        cache_dir=self.cache_dir, source_path=self.source_path
                    )
                    self._rows = [_normalize_row(row, self.per_test_timeout_s) for row in raw]
        return self._rows

    def _in_split(self, task_id: str, split: str, seed: int) -> bool:
        heldout = u01(seed, task_id) < self.heldout_fraction
        return heldout if split == "heldout" else not heldout

    def problems(self, split: str, n: int, seed: int) -> list[Problem]:
        if split not in ("train", "heldout"):
            raise ValueError(f"unknown split {split!r}")
        if n < 0:
            raise ValueError("n must be non-negative")
        chosen = [row for row in self._records() if self._in_split(row["task_id"], split, seed)]
        chosen.sort(key=lambda row: _task_sort_key(row["task_id"]))
        return [self._problem(row, split) for row in chosen[:n]]

    def _problem(self, row: dict[str, Any], split: str) -> Problem:
        task_id = row["task_id"]
        meta = {
            "task_id": task_id,
            "entry_point": row["entry_point"],
            "visible_assert": row["visible_assert"],
            "base_inputs": row["base_inputs"],
            "plus_inputs": row["plus_inputs"],
            "atol": row["atol"],
            "canonical_solution": row["canonical_solution"],
            "assertion": row["assertion"],
            "contract": row["contract"],
            "source": row["source"],
            "per_test_timeout_s": row["per_test_timeout_s"],
            "n_base": len(row["base_inputs"]),
            "n_plus": len(row["plus_inputs"]),
        }
        return Problem(
            pid=f"mbpp_plus/{split}/{task_id}",
            split=split,
            task=self.name,
            question=row["prompt"],
            answer=row["canonical_solution"],
            meta=meta,
        )

    def prompt(self, problem: Problem) -> str:
        entry = problem.meta["entry_point"]
        return (
            f"{problem.question.rstrip()}\n\n"
            f"Write a complete Python solution defining `{entry}`. "
            "Put the solution in one ```python block.\n"
        )

    def system_prompt(self) -> str | None:
        return None

    def parse(self, problem: Problem, text: str) -> Parsed:
        entry = str(problem.meta["entry_point"])
        fenced = _fenced_solution(text, entry)
        has_block = fenced is not None
        if fenced is not None:
            raw = fenced.strip("\n")
        elif re.search(rf"^\s*def\s+{re.escape(entry)}\b", text, re.M):
            raw = text.strip()
            has_block = False
        else:
            return Parsed(answer=None, raw=None, extras={"has_code_block": False})
        normalized = _normalize_code(raw)
        return Parsed(
            answer=normalized or None,
            raw=raw,
            extras={"has_code_block": has_block},
        )

    def oracle(self, problem: Problem, parsed: Parsed, text: str) -> int:
        del text
        if not parsed.answer:
            return 0
        return int(oracle_passes(grade_code(problem, parsed.answer)))
