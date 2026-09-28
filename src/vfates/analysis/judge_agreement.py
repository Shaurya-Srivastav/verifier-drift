"""Reported agreement between the IFEval content judge and a second model.

The preregistration asks ``gpt-4.1`` at temperature 0, on the same rubric, to
grade a fixed random subsample of 500 held-out evaluation samples (seed 0).
The numbers are reported. Hypothesis decisions stay in the confirmatory
analysis, and this check does not set ``ok``.

Sampling frame
    Held-out evaluation samples at ``analysis.final_step`` from every planned
    run that was not withdrawn. Early-baseline runs, the train split and every
    other step stay outside the frame. Rows are ordered by
    ``(arm, seed, pid, k)`` before the draw, so the order of the input listing
    does not matter.

Draw
    Without replacement. Each row is ranked by
    ``u01(seed, "judge_agreement", arm, training_seed, pid, k)``
    (:func:`vfates.utils.hashing.u01`, BLAKE2b, stable across machines and
    NumPy versions). The subsample is the ``n`` smallest ranks. Ties break on
    the canonical key. When the frame is smaller than ``n``, the whole frame
    is graded and the result says so.

Local verdict
    The IFEval oracle calls the content judge on every sample, including
    checker failures, and the oracle bit is the checker AND the judge.
    ``eval_samples.jsonl`` stores that content verdict. An unparsed local
    reply is scored 0 by the oracle and is not a verdict in this check.

Statistics
    Raw agreement, Cohen's kappa and the 2x2 counts use rows where both sides
    parsed as a single YES or NO. Missing local verdicts and unparsed replies
    are counted on their own. The interval is a prompt-clustered percentile
    bootstrap of agreement (:func:`vfates.utils.stats.cluster_bootstrap`),
    clustered by ``pid``, using this check's seed and ``n_boot`` (default
    10000, level 0.95). The subsample uses ``u01``; the bootstrap uses NumPy's
    ``default_rng`` inside ``cluster_bootstrap``. The two streams share the
    integer seed and nothing else.

Cache
    Replies are appended to ``judge_agreement_cache.jsonl`` under the study
    directory. The key is the SHA-256 of model, rubric hash, ``max_tokens``,
    temperature, base URL, prompt and response, so a rerun does not call the
    API again and an interrupted run resumes. The client uses
    ``https://api.openai.com/v1`` unless ``agreement_check.base_url`` is set.
    It does not read ``VFATES_JUDGE_BASE_URL`` (that variable is the on-policy
    content judge). ``VFATES_JUDGE_API_KEY`` is read only when a reply is
    missing from the cache.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from vfates.tasks.judges import DEFAULT_RUBRIC, LLMJudge, parse_yes_no
from vfates.train.evaluate import EVAL_SAMPLES_FILENAME
from vfates.utils.hashing import u01
from vfates.utils.io import append_jsonl, canonical_json, read_jsonl, sha256_text
from vfates.utils.stats import cluster_bootstrap

SCHEMA = "vfates.judge_agreement/1"
CACHE_FILENAME = "judge_agreement_cache.jsonl"
OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_N_BOOT = 10_000
_U01_LABEL = "judge_agreement"

LOCAL_JUDGE_SCOPE = (
    "The IFEval oracle calls the content judge on every sample, including checker "
    "failures, and the oracle bit is the checker AND the judge. The persisted judge "
    "field is that content verdict. An unparsed local reply is scored 0 by the oracle "
    "and is left out of agreement."
)


class ContentGrader(Protocol):
    """Anything that returns the raw YES/NO reply for one completion."""

    def complete(self, prompt: str, response: str) -> str: ...


@dataclass(frozen=True)
class FrameRow:
    """One held-out evaluation sample in the agreement frame."""

    arm: str
    seed: int
    pid: str
    k: int
    prompt: str
    text: str
    local: int | None
    local_unparsed: bool = False

    def sort_key(self) -> tuple[str, int, str, int]:
        return (self.arm, int(self.seed), self.pid, int(self.k))


def sampling_frame(rows: Sequence[FrameRow]) -> list[FrameRow]:
    """``rows`` in canonical order: arm, seed, pid, sample index."""
    return sorted(rows, key=lambda row: row.sort_key())


def _rank(seed: int, row: FrameRow) -> tuple[float, str, int, str, int]:
    score = u01(seed, _U01_LABEL, row.arm, row.seed, row.pid, row.k)
    return (score, *row.sort_key())


def draw_subsample(rows: Sequence[FrameRow], n: int, seed: int) -> list[FrameRow]:
    """``n`` rows without replacement, returned in canonical order.

    The draw ranks every row by :func:`vfates.utils.hashing.u01` with parts
    ``(seed, "judge_agreement", arm, training_seed, pid, k)`` and keeps the
    ``n`` smallest. Ties break on the canonical key. ``n`` at least the frame
    size returns the whole frame. The input order is ignored.
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError("subsample must be an integer")
    if n < 0:
        raise ValueError("subsample must be non-negative")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    frame = sampling_frame(rows)
    if n >= len(frame):
        return frame
    chosen = sorted(frame, key=lambda row: _rank(seed, row))[:n]
    return sampling_frame(chosen)


def reply_cache_key(
    *,
    model: str,
    rubric_sha256: str,
    max_tokens: int,
    base_url: str,
    prompt: str,
    text: str,
) -> str:
    """Identity of one remote reply. Identical completions share a cache line."""
    return sha256_text(
        canonical_json(
            {
                "model": model,
                "rubric_sha256": rubric_sha256,
                "max_tokens": int(max_tokens),
                "temperature": 0,
                "base_url": base_url.rstrip("/"),
                "prompt": prompt,
                "text": text,
            }
        )
    )


def _bit(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise ValueError(f"{name} must be 0, 1, or None")
    return int(value)


def agreement_statistics(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Agreement, kappa, the 2x2, unparsed counts and a prompt-clustered interval.

    Each row has ``local`` and ``remote`` (0, 1, or None), ``pid``, and
    optionally ``local_unparsed``. ``local is None`` with ``local_unparsed``
    is an unparsed local reply; otherwise it is a missing local verdict.
    ``remote is None`` is an unparsed remote reply. Agreement, kappa, the 2x2
    and the bootstrap use only rows where both values are 0 or 1.

    Kappa is ``(p_o - p_e) / (1 - p_e)`` with ``p_e`` the chance agreement of
    the observed marginals. It is ``None`` when there are no paired rows or
    when every paired row has the same bit on both sides (``p_e`` is 1).
    The bootstrap resamples ``pid`` clusters via
    :func:`vfates.utils.stats.cluster_bootstrap`.
    """
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("alpha must lie in (0, 1)")
    alpha_f = float(alpha)
    if not 0.0 < alpha_f < 1.0:
        raise ValueError("alpha must lie in (0, 1)")

    n_local_missing = 0
    n_local_unparsed = 0
    n_local_verdict = 0
    n_remote_unparsed = 0
    n_remote_verdict = 0
    cases = {
        "paired": 0,
        "remote_unparsed": 0,
        "local_unparsed": 0,
        "local_missing": 0,
        "neither": 0,
    }
    n00 = n01 = n10 = n11 = 0
    paired_local: list[int] = []
    paired_remote: list[int] = []
    paired_pid: list[str] = []

    for row in rows:
        local = _bit(row.get("local"), "local")
        remote = _bit(row.get("remote"), "remote")
        local_unparsed = bool(row.get("local_unparsed", False)) and local is None
        local_missing = local is None and not local_unparsed
        remote_ok = remote is not None
        local_ok = local is not None
        if local_unparsed:
            n_local_unparsed += 1
        elif local_missing:
            n_local_missing += 1
        else:
            n_local_verdict += 1
        if remote_ok:
            n_remote_verdict += 1
        else:
            n_remote_unparsed += 1
        if local is not None and remote is not None:
            cases["paired"] += 1
            paired_local.append(local)
            paired_remote.append(remote)
            paired_pid.append(str(row.get("pid", "")))
            if local == 1 and remote == 1:
                n11 += 1
            elif local == 1 and remote == 0:
                n10 += 1
            elif local == 0 and remote == 1:
                n01 += 1
            else:
                n00 += 1
        elif local_ok and not remote_ok:
            cases["remote_unparsed"] += 1
        elif local_unparsed and remote_ok:
            cases["local_unparsed"] += 1
        elif local_missing and remote_ok:
            cases["local_missing"] += 1
        else:
            cases["neither"] += 1

    n_paired = n00 + n01 + n10 + n11
    agreement: float | None
    kappa: float | None
    if n_paired == 0:
        agreement = None
        kappa = None
        bootstrap: dict[str, Any] | None = None
    else:
        agreement = (n00 + n11) / n_paired
        chance_num = (n11 + n10) * (n11 + n01) + (n00 + n01) * (n00 + n10)
        denom = n_paired * n_paired - chance_num
        kappa = None if denom == 0 else ((n00 + n11) * n_paired - chance_num) / denom
        local_arr = np.asarray(paired_local, dtype=np.int64)
        remote_arr = np.asarray(paired_remote, dtype=np.int64)

        def _stat(idx: np.ndarray) -> float:
            return float(np.mean(local_arr[idx] == remote_arr[idx]))

        point, lo, hi = cluster_bootstrap(
            _stat, paired_pid, n_boot=n_boot, alpha=alpha_f, seed=int(seed)
        )
        bootstrap = {
            "method": "prompt_cluster",
            "cluster": "pid",
            "n_boot": int(n_boot),
            "seed": int(seed),
            "alpha": alpha_f,
            "level": 1.0 - alpha_f,
            "point": point,
            "lo": lo,
            "hi": hi,
        }

    return {
        "n": len(rows),
        "n_paired": n_paired,
        "n_local_missing": n_local_missing,
        "n_local_unparsed": n_local_unparsed,
        "n_local_verdict": n_local_verdict,
        "n_remote_unparsed": n_remote_unparsed,
        "n_remote_verdict": n_remote_verdict,
        "cases": cases,
        "agreement": agreement,
        "kappa": kappa,
        "confusion": {
            "local_yes_remote_yes": n11,
            "local_yes_remote_no": n10,
            "local_no_remote_yes": n01,
            "local_no_remote_no": n00,
        },
        "bootstrap": bootstrap,
    }


def _as_int(spec: Mapping[str, Any], key: str, *, minimum: int | None = None) -> int:
    if key not in spec:
        raise ValueError(f"agreement_check.{key} is required")
    value = spec[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"agreement_check.{key} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"agreement_check.{key} must be >= {minimum}")
    return int(value)


def checked_agreement_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The validated ``agreement_check`` block. Raises on any preregistration mismatch."""
    if not isinstance(spec, Mapping):
        raise ValueError("agreement_check must be a mapping")
    model = spec.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("agreement_check.model is required")
    if "temperature" not in spec:
        raise ValueError("agreement_check.temperature is required")
    temperature = spec["temperature"]
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("agreement_check.temperature must be 0")
    if float(temperature) != 0.0:
        raise ValueError(f"agreement_check.temperature must be 0; got {temperature}")
    expected = spec.get("rubric_sha256")
    if not isinstance(expected, str) or not expected:
        raise ValueError("agreement_check.rubric_sha256 is required")
    actual = sha256_text(DEFAULT_RUBRIC)
    if expected != actual:
        shown = expected if len(expected) <= 80 else expected[:80] + "..."
        raise ValueError(
            f"agreement_check.rubric_sha256 ({shown}) does not match "
            f"vfates.tasks.judges.DEFAULT_RUBRIC ({actual})"
        )
    base_url = spec.get("base_url", OPENAI_BASE_URL)
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("agreement_check.base_url must be a URL")
    n_boot = DEFAULT_N_BOOT
    if "n_boot" in spec:
        n_boot = _as_int(spec, "n_boot", minimum=1)
    alpha = 0.05
    if "alpha" in spec:
        raw_alpha = spec["alpha"]
        if isinstance(raw_alpha, bool) or not isinstance(raw_alpha, (int, float)):
            raise ValueError("agreement_check.alpha must lie in (0, 1)")
        alpha = float(raw_alpha)
        if not 0.0 < alpha < 1.0:
            raise ValueError("agreement_check.alpha must lie in (0, 1)")
    return {
        "model": model,
        "temperature": 0,
        "max_tokens": _as_int(spec, "max_tokens", minimum=1),
        "subsample": _as_int(spec, "subsample", minimum=0),
        "seed": _as_int(spec, "seed"),
        "rubric_sha256": actual,
        "base_url": base_url.rstrip("/"),
        "n_boot": n_boot,
        "alpha": alpha,
    }


def _load_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    loaded: dict[str, str] = {}
    for row in read_jsonl(path):
        key = row.get("key")
        reply = row.get("reply")
        if isinstance(key, str) and isinstance(reply, str):
            loaded[key] = reply
    return loaded


def _sample_bit(row: Mapping[str, Any], run_dir: Path) -> tuple[int | None, bool]:
    """``(local verdict or None, unparsed)`` from one ``eval_samples.jsonl`` row."""
    if "judge" not in row or row.get("judge") is None:
        return None, False
    raw = row["judge"]
    if isinstance(raw, bool) or not isinstance(raw, int) or raw not in (0, 1):
        raise ValueError(f"{run_dir}: judge verdict must be 0, 1, or null")
    if row.get("judge_parsed", True) is False:
        return None, True
    return int(raw), False


def load_heldout_frame(
    out_dir: str | Path,
    planned: Mapping[str, Sequence[int]],
    withdrawn: Collection[str],
    final_step: int,
    *,
    samples_per_prompt: int | None = None,
) -> list[FrameRow]:
    """Final-step held-out samples for every planned run that was not withdrawn.

    Each run must have ``eval_samples.jsonl`` and a matching ``eval_prompts.jsonl``
    row. The last sample line wins for a repeated ``(pid, k)``. With
    ``samples_per_prompt`` set, every prompt must have indices ``0 .. n-1``.
    """
    if samples_per_prompt is not None and (
        isinstance(samples_per_prompt, bool)
        or not isinstance(samples_per_prompt, int)
        or samples_per_prompt < 1
    ):
        raise ValueError("samples_per_prompt must be a positive integer")
    out = Path(out_dir)
    skipped = {str(arm) for arm in withdrawn}
    expected_k = None if samples_per_prompt is None else set(range(samples_per_prompt))
    frame: list[FrameRow] = []
    seen: set[tuple[str, int]] = set()
    for arm, seeds in planned.items():
        arm_name = str(arm)
        if arm_name in skipped:
            continue
        for seed in seeds:
            seed_i = int(seed)
            if (arm_name, seed_i) in seen:
                continue
            seen.add((arm_name, seed_i))
            frame.extend(
                _load_run(
                    out / "runs" / f"{arm_name}__s{seed_i}",
                    arm_name,
                    seed_i,
                    int(final_step),
                    expected_k,
                )
            )
    return frame


def _load_run(
    run_dir: Path,
    arm: str,
    seed: int,
    final_step: int,
    expected_k: set[int] | None,
) -> list[FrameRow]:
    samples_path = run_dir / EVAL_SAMPLES_FILENAME
    if not samples_path.exists():
        raise ValueError(
            f"{run_dir}: missing {EVAL_SAMPLES_FILENAME} "
            "(held-out texts and content-judge verdicts are required for the agreement check)"
        )
    prompts_path = run_dir / "eval_prompts.jsonl"
    if not prompts_path.exists():
        raise ValueError(f"{run_dir}: missing eval_prompts.jsonl")
    prompt_rows = [
        row
        for row in read_jsonl(prompts_path)
        if row.get("split") == "heldout" and int(row.get("step", -1)) == final_step
    ]
    if not prompt_rows:
        raise ValueError(f"{run_dir}: no held-out eval_prompts.jsonl row at step {final_step}")
    acc = prompt_rows[-1].get("acc")
    if not isinstance(acc, dict) or not acc:
        raise ValueError(f"{run_dir}: held-out prompt map at step {final_step} is empty")
    expected_pids = {str(pid) for pid in acc}

    by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in read_jsonl(samples_path):
        if row.get("split") != "heldout" or int(row.get("step", -1)) != final_step:
            continue
        pid = row.get("pid")
        raw_k = row.get("k")
        if not isinstance(pid, str) or not pid:
            raise ValueError(f"{run_dir}: held-out sample at step {final_step} is missing pid")
        if isinstance(raw_k, bool) or not isinstance(raw_k, int):
            raise ValueError(f"{run_dir}: sample {pid} is missing an integer k")
        by_key[(pid, int(raw_k))] = row

    have = {pid for pid, _k in by_key}
    missing = expected_pids - have
    extra = have - expected_pids
    if missing or extra:
        detail = []
        if missing:
            detail.append("missing " + ", ".join(sorted(missing)[:5]))
        if extra:
            detail.append("unexpected " + ", ".join(sorted(extra)[:5]))
        raise ValueError(
            f"{run_dir}: held-out samples at step {final_step} do not match eval_prompts.jsonl "
            f"({'; '.join(detail)})"
        )
    if expected_k is not None:
        by_pid: dict[str, set[int]] = {}
        for pid, k in by_key:
            by_pid.setdefault(pid, set()).add(k)
        for pid in sorted(by_pid):
            if by_pid[pid] != expected_k:
                raise ValueError(
                    f"{run_dir}: {pid} has sample indices {sorted(by_pid[pid])}, "
                    f"expected {sorted(expected_k)}"
                )

    loaded: list[FrameRow] = []
    for (pid, k), row in by_key.items():
        prompt = row.get("prompt")
        text = row.get("text")
        if not isinstance(prompt, str) or not isinstance(text, str):
            raise ValueError(f"{run_dir}: sample {pid} k={k} is missing prompt or text")
        local, unparsed = _sample_bit(row, run_dir)
        loaded.append(
            FrameRow(
                arm=arm,
                seed=seed,
                pid=pid,
                k=k,
                prompt=prompt,
                text=text,
                local=local,
                local_unparsed=unparsed,
            )
        )
    return loaded


def run_judge_agreement(
    spec: Mapping[str, Any],
    out_dir: str | Path,
    *,
    planned: Mapping[str, Sequence[int]],
    withdrawn: Collection[str],
    final_step: int,
    grader: ContentGrader | None = None,
    samples_per_prompt: int | None = None,
) -> dict[str, Any]:
    """Grade the preregistered subsample and return the agreement record.

    ``grader`` injects the remote client (tests). Otherwise an
    :class:`~vfates.tasks.judges.LLMJudge` is built on the first cache miss.
    A missing ``VFATES_JUDGE_API_KEY`` raises only when a reply is not cached.
    The rubric hash is checked against :data:`~vfates.tasks.judges.DEFAULT_RUBRIC`
    before any file is read and before any query.
    """
    checked = checked_agreement_spec(spec)
    out = Path(out_dir)
    frame = load_heldout_frame(
        out,
        planned,
        withdrawn,
        int(final_step),
        samples_per_prompt=samples_per_prompt,
    )
    chosen = draw_subsample(frame, int(checked["subsample"]), int(checked["seed"]))
    cache_path = out / CACHE_FILENAME
    memory = _load_cache(cache_path)
    client: ContentGrader | None = grader
    owns_client = grader is None
    n_queries = 0
    n_hits = 0
    stat_rows: list[dict[str, Any]] = []
    drawn: list[dict[str, Any]] = []

    def _client() -> ContentGrader:
        nonlocal client
        if client is None:
            client = LLMJudge(
                checked["model"],
                base_url=checked["base_url"],
                rubric=DEFAULT_RUBRIC,
                max_tokens=int(checked["max_tokens"]),
            )
        return client

    for row in chosen:
        key = reply_cache_key(
            model=checked["model"],
            rubric_sha256=checked["rubric_sha256"],
            max_tokens=int(checked["max_tokens"]),
            base_url=checked["base_url"],
            prompt=row.prompt,
            text=row.text,
        )
        cached = memory.get(key)
        if cached is not None:
            reply = cached
            n_hits += 1
        else:
            try:
                reply = _client().complete(row.prompt, row.text)
            except RuntimeError as exc:
                if owns_client and "API key" in str(exc):
                    raise RuntimeError(
                        "The judge agreement check has to query "
                        f"{checked['base_url']} and VFATES_JUDGE_API_KEY is not set "
                        f"({len(memory)} replies are cached in {cache_path.name})."
                    ) from exc
                raise
            if not isinstance(reply, str):
                raise TypeError("judge complete() must return the reply text")
            append_jsonl(
                cache_path,
                {
                    "key": key,
                    "reply": reply,
                    "model": checked["model"],
                    "rubric_sha256": checked["rubric_sha256"],
                    "max_tokens": int(checked["max_tokens"]),
                    "temperature": 0,
                    "base_url": checked["base_url"],
                },
            )
            memory[key] = reply
            n_queries += 1
        remote = parse_yes_no(reply)
        local = None if row.local_unparsed else row.local
        stat_rows.append(
            {
                "local": local,
                "remote": remote,
                "local_unparsed": row.local_unparsed,
                "pid": row.pid,
            }
        )
        drawn.append(
            {
                "arm": row.arm,
                "seed": int(row.seed),
                "pid": row.pid,
                "k": int(row.k),
                "local": local,
                "remote": remote,
                "local_unparsed": bool(row.local_unparsed),
            }
        )

    stats = agreement_statistics(
        stat_rows,
        n_boot=int(checked["n_boot"]),
        seed=int(checked["seed"]),
        alpha=float(checked["alpha"]),
    )
    # ``n`` repeats ``n_drawn``. The study record uses the named counts.
    stats.pop("n")
    return {
        "schema": SCHEMA,
        "reported_not_tested": True,
        "model": checked["model"],
        "temperature": 0,
        "max_tokens": int(checked["max_tokens"]),
        "base_url": checked["base_url"],
        "rubric_sha256": checked["rubric_sha256"],
        "seed": int(checked["seed"]),
        "subsample_requested": int(checked["subsample"]),
        "final_step": int(final_step),
        "n_frame": len(frame),
        "n_drawn": len(chosen),
        "drew_entire_frame": len(chosen) == len(frame),
        "n_queries": n_queries,
        "n_cache_hits": n_hits,
        "sampling": {
            "frame": (
                "held-out evaluation samples at final_step from every planned non-withdrawn run"
            ),
            "order": ["arm", "seed", "pid", "k"],
            "draw": (
                "without replacement, the n smallest "
                "u01(seed, 'judge_agreement', arm, training_seed, pid, k); "
                "ties break on the canonical key"
            ),
            "cluster": "pid",
            "outside_frame": [
                "withdrawn arms",
                "early baseline runs",
                "train split",
                "steps other than final_step",
            ],
        },
        "local_judge_scope": LOCAL_JUDGE_SCOPE,
        "cache": CACHE_FILENAME,
        "drawn": drawn,
        **stats,
    }


def agreement_failure(exc: BaseException) -> dict[str, Any]:
    """The record written when the check could not be computed."""
    return {
        "schema": SCHEMA,
        "reported_not_tested": True,
        "error": f"{type(exc).__name__}: {exc}",
        "cache": CACHE_FILENAME,
    }


def _fmt(value: object) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_agreement_markdown(result: Mapping[str, Any]) -> str:
    """A short ``report.md`` section. ``judge_agreement.json`` is the record."""
    if "error" in result:
        return "\n".join(
            [
                "## Judge agreement",
                "",
                f"Not computed: {result['error']}. Replies already fetched stay in "
                f"`{result.get('cache', CACHE_FILENAME)}`; `vfates reanalyze` resumes.",
                "",
            ]
        )
    requested = int(result["subsample_requested"])
    drawn = int(result["n_drawn"])
    frame_n = int(result["n_frame"])
    requested_note = ""
    if drawn != requested:
        requested_note = f" The request was {requested}."
    lines = [
        "## Judge agreement",
        "",
        "Reported, not tested. "
        f"`{result['model']}` at temperature {result['temperature']} graded "
        f"{drawn} of {frame_n} held-out evaluation samples at step {result['final_step']} "
        f"(seed {result['seed']}).{requested_note}",
        "",
        f"Paired verdicts: {result['n_paired']}. "
        f"Local verdicts: {result['n_local_verdict']}. "
        f"Local missing: {result['n_local_missing']}. "
        f"Local unparsed: {result['n_local_unparsed']}. "
        f"Remote unparsed: {result['n_remote_unparsed']}.",
        "",
        f"Raw agreement: {_fmt(result.get('agreement'))}. "
        f"Cohen's kappa: {_fmt(result.get('kappa'))}.",
        "",
    ]
    boot = result.get("bootstrap")
    if isinstance(boot, dict) and boot.get("lo") is not None:
        lines.append(
            "Prompt-clustered interval "
            f"[{_fmt(boot.get('lo'))}, {_fmt(boot.get('hi'))}] "
            f"(level {_fmt(boot.get('level'))}, B={boot.get('n_boot')}, "
            f"seed {boot.get('seed')}, cluster {boot.get('cluster')})."
        )
        lines.append("")
    confusion = result.get("confusion") or {}
    lines.extend(
        [
            "| local vs remote | NO | YES |",
            "| --- | --- | --- |",
            f"| NO | {confusion.get('local_no_remote_no', 0)} | "
            f"{confusion.get('local_no_remote_yes', 0)} |",
            f"| YES | {confusion.get('local_yes_remote_no', 0)} | "
            f"{confusion.get('local_yes_remote_yes', 0)} |",
            "",
            "The local content judge is called on every held-out evaluation sample. "
            "Agreement, kappa and the table use samples where both verdicts parsed "
            "as a single YES or NO. An unparsed reply is counted above and left out of the table.",
            "",
        ]
    )
    return "\n".join(lines)
