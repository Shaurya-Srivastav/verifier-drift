"""Pilot P6: step-0 natural verifiers on the dev pool.

Documents FPR, FNR, Youden's J and precision with prompt-cluster bootstrap
intervals, patch-category error counts, whether patch_tau and patch_rate
select the same categories, the QA and IFEval judge checks, and a human
adjudication sample. Nothing in this pilot is a gate.

API stages read OPENAI_API_KEY. Without that key they are skipped and
recorded as documented checks. Verdicts are never filled in by hand.
"""

from __future__ import annotations

import csv
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from vfates.pilots.common import Checks, markdown_table, write_report
from vfates.utils.hashing import u01
from vfates.utils.io import read_json, write_json

NATURAL_VERIFIERS: tuple[tuple[str, str], ...] = (
    ("math", "natural"),
    ("math", "natural_strict"),
    ("code", "natural"),
    ("ifeval", "natural"),
    ("qa", "natural"),
    ("qa", "natural_robust"),
)

CATEGORIES: dict[str, tuple[str, ...]] = {
    "math": (
        "no_box",
        "multiple_distinct_boxes",
        "large_magnitude_gold",
        "near_miss_numeric",
        "fraction_vs_decimal",
        "whitespace_or_latex_spacing",
        "symbolic_gold",
        "units_or_text_in_box",
    ),
    "code": (
        "hardcoded_example_output",
        "special_case_branches",
        "ignores_input",
        "exception_swallow",
        "single_public_example",
        "recursion_or_deep_loops",
        "prints_debug_extra",
    ),
    "ifeval": (
        "keyword_constraints",
        "length_constraints",
        "format_constraints",
        "case_constraints",
        "start_end_constraints",
        "many_constraints",
        "short_response",
        "repetition_heavy",
    ),
    "qa": (
        "no_final_answer",
        "master_key_opener",
        "hedged_multiple_answers",
        "numeric_answer",
        "long_response",
        "refusal",
    ),
}

ADJUDICATION_COLUMNS: tuple[str, ...] = (
    "item_id",
    "pid",
    "prompt",
    "response",
    "gold",
    "parsed",
    "verifier_verdict",
    "oracle_verdict",
    "label",
    "note",
)
ORACLE_COLUMNS: tuple[str, ...] = (
    "item_id",
    "pid",
    "prompt",
    "response",
    "gold",
    "rule_oracle",
    "qwen3_32b",
    "api_model",
    "label",
    "note",
)
LABELS = frozenset({"real_error", "oracle_wrong", "unclear"})
RATE_KEYS = ("fpr", "fnr", "j", "precision")


def _div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return float(num) / float(den)


def _counts(oracle: Sequence[int], verifier: Sequence[int]) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for o_bit, v_bit in zip(oracle, verifier):
        if v_bit == 1 and o_bit == 1:
            tp += 1
        elif v_bit == 1 and o_bit == 0:
            fp += 1
        elif v_bit == 0 and o_bit == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def confusion_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict[str, Any]:
    """Pooled step-0 rates. Undefined ratios are ``None`` (empty denominator).

    ``j`` is Youden's J, TPR - FPR. ``fp_mass`` is FP / N.
    ``fp_share_among_accepted`` is FP / (TP + FP), the fraction of
    verifier-accepted samples the oracle rejects.
    """
    n = tp + fp + tn + fn
    fpr = _div(fp, fp + tn)
    fnr = _div(fn, fn + tp)
    tpr = None if fnr is None else 1.0 - fnr
    youden = None if tpr is None or fpr is None else tpr - fpr
    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "fpr": fpr,
        "fnr": fnr,
        "j": youden,
        "precision": _div(tp, tp + fp),
        "fp_mass": _div(fp, n),
        "fp_share_among_accepted": _div(fp, tp + fp),
    }


def pooled_confusion(oracle: Sequence[int], verifier: Sequence[int]) -> dict[str, Any]:
    """Confusion pooled over samples."""
    return confusion_from_counts(*_counts(oracle, verifier))


def prompt_averaged_confusion(
    pids: Sequence[object], oracle: Sequence[int], verifier: Sequence[int]
) -> dict[str, Any]:
    """Unweighted mean of per-prompt rates. Prompts with an empty denominator are skipped."""
    groups: dict[object, list[list[int]]] = {}
    for pid, o_bit, v_bit in zip(pids, oracle, verifier):
        bucket = groups.setdefault(pid, [[], []])
        bucket[0].append(int(o_bit))
        bucket[1].append(int(v_bit))
    per = [confusion_from_counts(*_counts(bits[0], bits[1])) for bits in groups.values()]

    def _mean(key: str) -> float | None:
        values = [row[key] for row in per if row[key] is not None]
        if not values:
            return None
        return float(sum(values) / len(values))

    keys = ("fpr", "fnr", "j", "precision", "fp_mass", "fp_share_among_accepted")
    averaged: dict[str, Any] = {key: _mean(key) for key in keys}
    averaged["n_prompts"] = len(groups)
    return averaged


def fp_concentration(
    pids: Sequence[object],
    oracle: Sequence[int],
    verifier: Sequence[int],
    *,
    top_frac: float = 0.1,
) -> float | None:
    """Share of false-positive mass sitting in the top ``top_frac`` of prompts.

    Prompts are ranked by FP count, ties broken by pid string. The top slice
    is ``max(1, ceil(top_frac * n_prompts))``. ``None`` when there are no
    false positives. A value near 1 means a few prompts own the errors
    (prompt-specific). A value near ``top_frac`` means the errors are spread
    across prompts (the signature of a shared exploit).
    """
    counts: dict[object, int] = {}
    for pid, o_bit, v_bit in zip(pids, oracle, verifier):
        counts.setdefault(pid, 0)
        if int(v_bit) == 1 and int(o_bit) == 0:
            counts[pid] += 1
    total = sum(counts.values())
    if total == 0 or not counts:
        return None
    ranked = sorted(counts, key=lambda pid: (-counts[pid], str(pid)))
    width = max(1, math.ceil(top_frac * len(ranked)))
    return float(sum(counts[pid] for pid in ranked[:width]) / total)


def _make_generator(backend: str, model_id: str, kwargs: dict[str, Any]) -> Any:
    try:
        from vfates.gen.factory import BackendName, make_generator
    except ImportError as exc:
        raise ImportError(
            "vfates.gen.factory.make_generator is unavailable. Pilot P6 needs the "
            "generation backends (mlx: vfates.gen.mlx_backend.MLXGenerator, "
            "hf: vfates.gen.hf_backend.HFGenerator) exposed as "
            "vfates.gen.factory.make_generator(backend, model_id, **kw). "
            "Those modules are not part of the natural-verifier workstream and "
            "are not importable in this checkout."
        ) from exc
    return make_generator(cast(BackendName, backend), model_id, **kwargs)


def wilson_interval(successes: int, n: int, *, alpha: float = 0.05) -> dict[str, Any]:
    """Wilson score interval for ``successes`` out of ``n``."""
    from statistics import NormalDist

    k = int(successes)
    total = int(n)
    if total <= 0:
        return {"point": None, "lo": None, "hi": None, "n": 0, "k": k}
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    p = k / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return {
        "point": p,
        "lo": max(0.0, center - margin),
        "hi": min(1.0, center + margin),
        "n": total,
        "k": k,
    }


def deterministic_sample(
    rows: Sequence[Mapping[str, Any]], n: int, seed: int, label: str
) -> tuple[list[dict[str, Any]], bool]:
    """Take ``n`` rows, ordered by ``u01(seed, label, item_id, pid)``."""
    ranked = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            u01(seed, label, str(row.get("item_id", "")), str(row.get("pid", ""))),
            str(row.get("item_id", "")),
        ),
    )
    width = max(int(n), 0)
    return ranked[:width], len(ranked) > width


def _write_csv(path: str | Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: "" if row.get(column) is None else row.get(column) for column in columns}
            )


def export_adjudication(
    rows: Sequence[Mapping[str, Any]], path: str | Path, *, n: int, seed: int
) -> dict[str, Any]:
    """Write a deterministic sample of adjudication rows. ``label`` is left blank."""
    prepared: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item_id = row.get("item_id") or f"{row.get('pid', index)}:{row.get('k', index)}"
        prepared.append(
            {
                "item_id": item_id,
                "pid": row.get("pid", ""),
                "prompt": row.get("prompt", ""),
                "response": row.get("response", ""),
                "gold": row.get("gold", ""),
                "parsed": row.get("parsed", ""),
                "verifier_verdict": row.get("verifier_verdict", row.get("verifier", "")),
                "oracle_verdict": row.get("oracle_verdict", row.get("oracle", "")),
                "label": "",
                "note": row.get("note", ""),
            }
        )
    chosen, truncated = deterministic_sample(prepared, n, seed, "adjudication")
    _write_csv(path, ADJUDICATION_COLUMNS, chosen)
    return {
        "path": str(path),
        "n": len(chosen),
        "n_available": len(prepared),
        "truncated": truncated,
    }


def import_adjudication(path: str | Path) -> dict[str, Any]:
    """Fraction of labelled rows marked ``real_error``, with a Wilson interval.

    Blank labels are unlabelled and stay out of the fraction. Any other label
    raises ``ValueError``.
    """
    dest = Path(path)
    with dest.open(newline="") as handle:
        loaded = list(csv.DictReader(handle))
    counts = {label: 0 for label in ("real_error", "oracle_wrong", "unclear")}
    for row in loaded:
        label = str(row.get("label") or "").strip()
        if not label:
            continue
        if label not in LABELS:
            raise ValueError(
                f"label {label!r} is not real_error, oracle_wrong, or unclear ({dest})"
            )
        counts[label] += 1
    n_labelled = sum(counts.values())
    interval = wilson_interval(counts["real_error"], n_labelled)
    return {
        "path": str(dest),
        "n_rows": len(loaded),
        "n_labelled": n_labelled,
        "n_real_error": counts["real_error"],
        "n_oracle_wrong": counts["oracle_wrong"],
        "n_unclear": counts["unclear"],
        "fraction_real_error": interval["point"],
        "wilson_lo": interval["lo"],
        "wilson_hi": interval["hi"],
    }


def _three_present(row: Mapping[str, Any]) -> bool:
    return all(row.get(key) not in (None, "") for key in ("rule_oracle", "qwen3_32b", "api_model"))


def _disagrees(row: Mapping[str, Any]) -> bool:
    if not _three_present(row):
        return False
    bits = {int(row["rule_oracle"]), int(row["qwen3_32b"]), int(row["api_model"])}
    return len(bits) > 1


def _oracle_missing() -> list[str]:
    missing = []
    if not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not (os.environ.get("VFATES_QWEN32_BASE_URL") or os.environ.get("VFATES_JUDGE_BASE_URL")):
        missing.append("VFATES_QWEN32_BASE_URL")
    return missing


def _chat(base_url: str, api_key: str, model: str, prompt: str) -> str:
    import json
    import urllib.request

    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
    ).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode())
    return str(payload["choices"][0]["message"]["content"])


def _fill_oracle_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Ask the configured 32B server and the API model. Raises if either call fails."""
    from vfates.tasks.qa import general_verifier_bit

    api_key = os.environ["OPENAI_API_KEY"]
    qwen_base = os.environ.get("VFATES_QWEN32_BASE_URL") or os.environ["VFATES_JUDGE_BASE_URL"]
    qwen_key = os.environ.get("VFATES_JUDGE_API_KEY", "EMPTY")
    api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_model = os.environ.get("VFATES_API_JUDGE_MODEL", "gpt-4.1")
    filled: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        text = (
            f"Question: {row.get('prompt', '')}\n"
            f"Answer: {row.get('response', '')}\n"
            "Reply with Final Decision: Yes or Final Decision: No."
        )
        if item.get("qwen3_32b") in (None, ""):
            item["qwen3_32b"] = general_verifier_bit(
                _chat(qwen_base, qwen_key, "Qwen/Qwen3-32B", text)
            )
        if item.get("api_model") in (None, ""):
            item["api_model"] = general_verifier_bit(_chat(api_base, api_key, api_model, text))
        filled.append(item)
    return filled


def _oracle_rows_from_audit(audit_dir: str | Path) -> list[dict[str, Any]]:
    from vfates.audit import load_audit

    audit = load_audit(audit_dir)
    table = audit.table
    n = len(table)
    splits = [str(item) for item in table.split]
    held = [i for i in range(n) if splits[i] == "heldout"]
    indexes = held or list(range(n))
    rows: list[dict[str, Any]] = []
    for i in indexes:
        comp = audit.completions[i]
        problem = audit.problems[comp.pid]
        parsed = audit.parsed[i]
        rows.append(
            {
                "item_id": f"{comp.pid}:{int(table.k[i])}",
                "pid": str(comp.pid),
                "prompt": problem.question,
                "response": comp.text,
                "gold": problem.answer,
                "parsed": "" if parsed.answer is None else str(parsed.answer),
                "rule_oracle": int(table.O[i]),
                "qwen3_32b": None,
                "api_model": None,
                "label": "",
                "note": "",
            }
        )
    return rows


def export_oracle_check(
    audit_dir: str | Path,
    out_csv: str | Path,
    *,
    n: int = 200,
    rows: Sequence[Mapping[str, Any]] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Export held-out rows where the rule oracle, Qwen3-32B and an API model disagree.

    Without OPENAI_API_KEY and a 32B endpoint, and without rows that already
    carry the three verdicts, the export is skipped. Nothing is invented.
    """
    supplied = rows is not None
    loaded: list[dict[str, Any]]
    if supplied:
        loaded = [dict(row) for row in rows or []]
    elif not str(audit_dir):
        missing = _oracle_missing()
        return {
            "skipped": True,
            "reason": "missing " + ", ".join(missing) if missing else "audit_dir is not set",
            "n": 0,
            "path": str(out_csv),
        }
    else:
        try:
            loaded = _oracle_rows_from_audit(audit_dir)
        except (OSError, ValueError, KeyError) as exc:
            return {
                "skipped": True,
                "reason": f"{type(exc).__name__}: {exc}",
                "n": 0,
                "path": str(out_csv),
            }
    if not any(_three_present(row) for row in loaded):
        missing = _oracle_missing()
        if missing:
            return {
                "skipped": True,
                "reason": "missing " + ", ".join(missing),
                "n": 0,
                "path": str(out_csv),
            }
        try:
            loaded = _fill_oracle_rows(loaded)
        except Exception as exc:
            return {
                "skipped": True,
                "reason": f"{type(exc).__name__}: {exc}",
                "n": 0,
                "path": str(out_csv),
            }
    disagreements = [row for row in loaded if _disagrees(row)]
    for row in disagreements:
        row.setdefault("label", "")
        row.setdefault("note", "")
    chosen, truncated = deterministic_sample(disagreements, n, seed, "oracle_check")
    _write_csv(out_csv, ORACLE_COLUMNS, chosen)
    return {
        "skipped": False,
        "n": len(chosen),
        "n_available": len(disagreements),
        "truncated": truncated,
        "path": str(out_csv),
    }


def rates_with_intervals(
    pids: Sequence[object],
    oracle: Sequence[int],
    verifier: Sequence[int],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Pooled rates plus a prompt-cluster bootstrap interval for each rate."""
    import numpy as np

    from vfates.utils.stats import cluster_bootstrap

    oracle_bits = [int(bit) for bit in oracle]
    verifier_bits = [int(bit) for bit in verifier]
    pooled = pooled_confusion(oracle_bits, verifier_bits)
    clusters = list(pids)
    if not clusters:
        return {**pooled, "intervals": None}
    oracle_arr = np.asarray(oracle_bits)
    verifier_arr = np.asarray(verifier_bits)

    def _stat(key: str):
        def fn(idx: Any) -> float:
            chosen = np.asarray(idx, dtype=int)
            if chosen.size == 0:
                return float("nan")
            row = confusion_from_counts(
                *_counts(oracle_arr[chosen].tolist(), verifier_arr[chosen].tolist())
            )
            value = row[key]
            return float("nan") if value is None else float(value)

        return fn

    intervals: dict[str, dict[str, float | None]] = {}
    for key in RATE_KEYS:
        point, lo, hi = cluster_bootstrap(
            _stat(key), clusters, n_boot=n_boot, alpha=alpha, seed=seed
        )
        intervals[key] = {
            "point": None if math.isnan(point) else point,
            "lo": None if math.isnan(lo) else lo,
            "hi": None if math.isnan(hi) else hi,
        }
    return {**pooled, "intervals": intervals}


def category_error_counts(rows: Sequence[Mapping[str, Any]], menu: Sequence[str]) -> dict[str, int]:
    """Step-0 error counts per patch category, in menu order."""
    counts = {str(name): 0 for name in menu}
    for row in rows:
        if int(row["oracle"]) == int(row["verifier"]):
            continue
        for name in row.get("categories") or []:
            if str(name) in counts:
                counts[str(name)] += 1
    return counts


def same_selection_of(block: Mapping[str, Any] | None) -> bool | None:
    """Whether patch_tau and patch_rate chose the same categories.

    When the forecasts block has arm records, this uses
    :func:`vfates.verifiers.patch._same_selection` (tuple equality in menu
    order). A block that only stores the boolean is read as that boolean.
    """
    if not isinstance(block, Mapping):
        return None
    from vfates.verifiers.patch import _same_selection

    records = {
        str(key): value
        for key, value in block.items()
        if key != "same_selection" and isinstance(value, Mapping) and "categories" in value
    }
    if records:
        return bool(_same_selection(records))
    if "same_selection" in block:
        return bool(block["same_selection"])
    return None


def read_same_selection(audit_dir: str | Path) -> bool | None:
    path = Path(audit_dir) / "forecasts.json"
    if not path.is_file():
        return None
    loaded = read_json(path)
    if not isinstance(loaded, Mapping):
        return None
    block = loaded.get("patch_selection")
    if not isinstance(block, Mapping):
        return None
    return same_selection_of(block)


def false_positives(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if int(row["verifier"]) == 1 and int(row["oracle"]) == 0]


def synthetic_rows(domain: str, verifier: str, *, n: int = 30) -> list[dict[str, Any]]:
    """Fixed step-0 rows for the Mac smoke. No model and no network."""
    menu = CATEGORIES[domain]
    rows: list[dict[str, Any]] = []
    for i in range(n):
        oracle = 0 if i % 5 == 0 else 1
        bit = 1 if i % 5 in (0, 1) else oracle
        rows.append(
            {
                "item_id": f"{domain}-{verifier}-{i}",
                "pid": f"p{i % 6}",
                "k": i // 6,
                "prompt": f"prompt {i}",
                "response": f"response {i}",
                "gold": "gold",
                "parsed": "parsed",
                "oracle": oracle,
                "verifier": bit,
                "categories": [menu[i % len(menu)]] if oracle != bit else [],
            }
        )
    return rows


def frame_from_audit(audit_dir: str | Path, verifier: str) -> list[dict[str, Any]]:
    """Dev-pool rows for one natural verifier. Prefers a ``dev`` split when present."""
    from vfates.audit import load_audit

    audit = load_audit(audit_dir)
    table = audit.table
    if verifier not in table.V:
        raise KeyError(f"{verifier} is not a column of {audit_dir}")
    n = len(table)
    if len(audit.completions) != n or len(audit.parsed) != n:
        raise RuntimeError(f"{audit_dir}: completions do not align with the audit table")
    splits = [str(item) for item in table.split]
    use_dev = any(item == "dev" for item in splits)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        if use_dev and splits[i] != "dev":
            continue
        comp = audit.completions[i]
        problem = audit.problems[comp.pid]
        parsed = audit.parsed[i]
        try:
            cats = list(audit.task.categorize(problem, parsed, comp.text))
        except Exception:
            cats = []
        rows.append(
            {
                "item_id": f"{comp.pid}:{int(table.k[i])}",
                "pid": str(comp.pid),
                "k": int(table.k[i]),
                "prompt": problem.question,
                "response": comp.text,
                "gold": problem.answer,
                "parsed": "" if parsed.answer is None else str(parsed.answer),
                "oracle": int(table.O[i]),
                "verifier": int(table.V[verifier][i]),
                "categories": cats,
            }
        )
    return rows


def qa_judge_check(
    rows: Sequence[Mapping[str, Any]], *, n_boot: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """General-verifier vs an API model on rows where they disagree with the rule oracle."""
    if not rows or all(row.get("api") is None and not row.get("api_reply") for row in rows):
        if not os.environ.get("OPENAI_API_KEY"):
            return {"skipped": True, "reason": "OPENAI_API_KEY is not set"}
        return {"skipped": True, "reason": "no API verdicts"}
    from vfates.analysis.judge_agreement import agreement_statistics
    from vfates.tasks.qa import general_verifier_bit, judge_agreement_row

    paired: list[dict[str, Any]] = []
    for row in rows:
        api = row.get("api")
        if api is None and row.get("api_reply"):
            api = general_verifier_bit(str(row["api_reply"]))
        if api is None:
            continue
        bits = {int(row["oracle"]), int(row["verifier"]), int(api)}
        if len(bits) == 1:
            continue
        if row.get("verifier_reply") and row.get("api_reply"):
            paired.append(
                judge_agreement_row(
                    str(row["verifier_reply"]), str(row["api_reply"]), str(row["pid"])
                )
            )
        else:
            paired.append(
                {
                    "local": int(row["verifier"]),
                    "remote": int(api),
                    "pid": str(row["pid"]),
                    "local_unparsed": False,
                }
            )
    if not paired:
        return {"skipped": False, "n_disagreements": 0, "statistics": None}
    return {
        "skipped": False,
        "n_disagreements": len(paired),
        "statistics": agreement_statistics(paired, n_boot=n_boot, seed=seed),
    }


def ifeval_judge_check(
    rows: Sequence[Mapping[str, Any]], *, n_boot: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """Content judge vs gpt-4.1. Skipped, never faked, when the API key is absent."""
    has_remote = any(row.get("remote") is not None or row.get("remote_reply") for row in rows)
    if not has_remote:
        if not os.environ.get("OPENAI_API_KEY"):
            return {"skipped": True, "reason": "OPENAI_API_KEY is not set", "model": "gpt-4.1"}
        return {"skipped": True, "reason": "no gpt-4.1 verdicts", "model": "gpt-4.1"}
    from vfates.analysis.judge_agreement import agreement_statistics
    from vfates.tasks.judges import parse_yes_no

    paired = []
    for row in rows:
        remote = row.get("remote")
        if remote is None and row.get("remote_reply"):
            remote = parse_yes_no(str(row["remote_reply"]))
        paired.append(
            {
                "local": row.get("local"),
                "remote": remote,
                "pid": str(row.get("pid", "")),
                "local_unparsed": bool(row.get("local_unparsed", False)),
            }
        )
    return {
        "skipped": False,
        "model": "gpt-4.1",
        "statistics": agreement_statistics(paired, n_boot=n_boot, seed=seed),
    }


def _fmt_interval(block: Mapping[str, Any] | None, key: str) -> str:
    if not block or block.get(key) is None:
        return "n/a"
    row = block[key]
    return f"{row['point']} [{row['lo']}, {row['hi']}]"


def _audit_dirs(config: Mapping[str, Any]) -> dict[str, str]:
    raw = config.get("audit_dirs") or {}
    return {str(key): str(value) for key, value in dict(raw).items() if value}


def _rows_for(
    config: Mapping[str, Any], domain: str, verifier: str, *, smoke: bool
) -> tuple[list[dict[str, Any]], str]:
    if smoke:
        return synthetic_rows(domain, verifier, n=int(config.get("smoke_n", 30))), "synthetic"
    audit_dir = _audit_dirs(config).get(domain)
    if not audit_dir:
        return [], "no audit_dirs entry"
    return frame_from_audit(audit_dir, verifier), audit_dir


def _stage_set(config: Mapping[str, Any]) -> set[str]:
    stage = str(config.get("stage") or "all")
    mapping = {
        "all": {"rates", "export", "import", "judges", "oracle_check"},
        "smoke": {"rates", "export", "import", "judges", "oracle_check"},
        "rates": {"rates"},
        "export": {"export"},
        "adjudication_export": {"export"},
        "import": {"import"},
        "adjudication_import": {"import"},
        "judges": {"judges"},
        "oracle_check": {"oracle_check"},
    }
    if stage not in mapping:
        raise ValueError(
            f"unknown P6 stage {stage!r}. Use all, rates, export, import, judges, or oracle_check."
        )
    return mapping[stage]


def _markdown(report: dict[str, Any]) -> str:
    gates = report["gates"]
    lines = [
        "# P6 natural-verifier step-0 audit",
        "",
        f"Gates pass: **{gates['pass']}** "
        f"(IFEval {gates['ifeval']}, complete {gates['complete']}).",
        "",
    ]
    for key, title in (("ifeval", "IFEval"),):
        block = report[key]
        pooled = block.get("pooled") or {}
        lines.extend(
            [
                f"## {title}",
                "",
                f"Model `{block.get('model')}`, rows {block.get('n_rows')}, "
                f"problems {block.get('n_problems')}.",
                "",
                "| quantity | value |",
                "| --- | --- |",
                f"| checker / base pass rate | {_fmt(block.get('checker_pass_rate'))} |",
                f"| oracle pass rate | {_fmt(block.get('oracle_pass_rate'))} |",
                f"| pooled FPR | {_fmt(pooled.get('fpr'))} |",
                f"| pooled FNR | {_fmt(pooled.get('fnr'))} |",
                f"| Youden J | {_fmt(pooled.get('j'))} |",
                f"| precision | {_fmt(pooled.get('precision'))} |",
                f"| FP mass | {_fmt(pooled.get('fp_mass'))} |",
                f"| FP share among V-accepted | {_fmt(block.get('fp_share_among_accepted'))} |",
                f"| FP mass in top 10% of prompts | {_fmt(block.get('fp_concentration_top10'))} |",
            ]
        )
        if "special_casing_rate_among_fps" in block:
            lines.append(
                f"| special-casing rate among FPs | {_fmt(block.get('special_casing_rate_among_fps'))} |"
            )
        if "judge_reject_rate_among_checker_passes" in block:
            lines.append(
                "| heuristic judge reject rate among checker passes | "
                f"{_fmt(block.get('judge_reject_rate_among_checker_passes'))} |"
            )
        lines.append("")
    lines.append(
        "FP concentration near 1 means a few prompts own the false positives "
        "(prompt-specific). A value near 0.1 means they are spread like a shared behaviour."
    )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: object) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def run(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Document step-0 natural-verifier rates. Gates nothing."""
    import time

    from vfates.utils.env import hardware_class

    started = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    smoke = bool(config.get("smoke", False))
    wanted = _stage_set(config)
    seed = int(config.get("seed", 0))
    n_boot = int(config.get("smoke_boot", 200) if smoke else config.get("n_boot", 2000))
    n_export = int(config.get("n_adjudication", 100))
    checks = Checks()
    hardware = hardware_class()
    adj_dir = (
        Path(str(config["adjudication_dir"]))
        if config.get("adjudication_dir")
        else out / "adjudication"
    )
    tables: dict[str, Any] = {"hardware_class": hardware, "smoke": smoke}
    rate_rows: list[list[Any]] = []
    seen_domains: set[str] = set()

    if "rates" in wanted or "export" in wanted:
        for domain, verifier in NATURAL_VERIFIERS:
            try:
                rows, source = _rows_for(config, domain, verifier, smoke=smoke)
            except Exception as exc:
                rows, source = [], f"{type(exc).__name__}: {exc}"
            if "rates" in wanted:
                if not rows:
                    checks.add(
                        f"P6.1.{domain}.{verifier}",
                        f"{domain} {verifier} step-0 rates",
                        None,
                        False,
                        threshold="documented",
                        metric=source,
                        documented=True,
                    )
                else:
                    scored = rates_with_intervals(
                        [row["pid"] for row in rows],
                        [row["oracle"] for row in rows],
                        [row["verifier"] for row in rows],
                        n_boot=n_boot,
                        seed=seed,
                    )
                    counts = category_error_counts(rows, CATEGORIES[domain])
                    tables[f"{domain}.{verifier}"] = {
                        "rates": scored,
                        "categories": counts,
                        "source": source,
                    }
                    rate_rows.append(
                        [
                            domain,
                            verifier,
                            scored.get("fpr"),
                            scored.get("fnr"),
                            scored.get("j"),
                            scored.get("precision"),
                        ]
                    )
                    checks.add(
                        f"P6.1.{domain}.{verifier}",
                        f"{domain} {verifier} step-0 FPR, FNR, J, precision",
                        scored.get("fpr"),
                        True,
                        threshold="documented",
                        metric=(
                            f"FNR={scored.get('fnr')}, J={scored.get('j')}, "
                            f"precision={scored.get('precision')}; "
                            f"FPR {_fmt_interval(scored.get('intervals'), 'fpr')}"
                        ),
                        documented=True,
                    )
                    checks.add(
                        f"P6.2.{domain}.{verifier}",
                        f"{domain} {verifier} patch category error counts",
                        float(sum(counts.values())),
                        True,
                        threshold="documented",
                        metric=", ".join(f"{name}={count}" for name, count in counts.items()),
                        documented=True,
                    )
            if "export" in wanted and rows:
                exported = export_adjudication(
                    false_positives(rows),
                    adj_dir / f"{domain}__{verifier}.csv",
                    n=n_export,
                    seed=seed,
                )
                checks.add(
                    f"P6.6.{domain}.{verifier}",
                    f"export false positives for {domain} {verifier}",
                    float(exported["n"]),
                    True,
                    threshold=n_export,
                    metric=exported["path"],
                    documented=True,
                )
            if domain not in seen_domains and "rates" in wanted:
                seen_domains.add(domain)
                audit_dir = _audit_dirs(config).get(domain)
                selection = read_same_selection(audit_dir) if audit_dir and not smoke else None
                checks.add(
                    f"P6.3.{domain}",
                    f"{domain} patch_tau and patch_rate same_selection",
                    None if selection is None else float(selection),
                    selection is True,
                    threshold="same categories",
                    metric="smoke does not select on features"
                    if smoke
                    else f"same_selection={selection}",
                    documented=True,
                )

    if "judges" in wanted:
        qa_rows = list(config.get("qa_rows") or [])
        ifeval_rows = list(config.get("ifeval_rows") or [])
        qa = qa_judge_check(qa_rows, n_boot=n_boot, seed=seed)
        ifeval = ifeval_judge_check(ifeval_rows, n_boot=n_boot, seed=seed)
        tables["qa_judge"] = qa
        tables["ifeval_judge"] = ifeval
        checks.add(
            "P6.4",
            "QA general-verifier vs rule oracle vs API model",
            None if qa.get("statistics") is None else qa["statistics"].get("agreement"),
            not qa.get("skipped", False),
            threshold="documented",
            metric=qa.get("reason") or f"disagreements={qa.get('n_disagreements')}",
            documented=True,
        )
        checks.add(
            "P6.5",
            "IFEval content judge vs gpt-4.1",
            None if ifeval.get("statistics") is None else ifeval["statistics"].get("agreement"),
            not ifeval.get("skipped", False),
            threshold="documented",
            metric=ifeval.get("reason") or "gpt-4.1",
            documented=True,
        )

    if "import" in wanted:
        paths = sorted(adj_dir.glob("*.csv")) if adj_dir.is_dir() else []
        if not paths:
            checks.add(
                "P6.7",
                "import labelled adjudication",
                None,
                False,
                threshold="documented",
                metric="no adjudication csv",
                documented=True,
            )
        imports = []
        for path in paths:
            loaded = import_adjudication(path)
            imports.append(loaded)
            checks.add(
                f"P6.7.{path.stem}",
                f"real-error fraction in {path.name}",
                loaded["fraction_real_error"],
                loaded["n_labelled"] > 0,
                threshold="Wilson interval",
                metric=(
                    f"{loaded['n_real_error']}/{loaded['n_labelled']} labelled; "
                    f"Wilson [{loaded['wilson_lo']}, {loaded['wilson_hi']}]"
                ),
                documented=True,
            )
        tables["adjudication"] = imports

    if "oracle_check" in wanted:
        oracle_csv = (
            Path(str(config["oracle_csv"]))
            if config.get("oracle_csv")
            else adj_dir / "qa__oracle_check.csv"
        )
        audit_dir = str(config.get("audit_dir") or _audit_dirs(config).get("qa") or "")
        result = export_oracle_check(
            audit_dir,
            oracle_csv,
            n=int(config.get("n_oracle_check", 200)),
            rows=config.get("oracle_rows"),
            seed=seed,
        )
        tables["oracle_check"] = result
        checks.add(
            "P6.8",
            "QA oracle disagreement export",
            None if result.get("skipped") else float(result.get("n") or 0),
            not result.get("skipped", False),
            threshold=int(config.get("n_oracle_check", 200)),
            metric=result.get("reason") or result.get("path", ""),
            documented=True,
        )

    write_json(out / "tables.json", tables)
    sections = []
    if rate_rows:
        sections.append(
            (
                "Step-0 rates",
                markdown_table(["domain", "verifier", "fpr", "fnr", "j", "precision"], rate_rows),
            )
        )
    return write_report(
        out,
        str(config.get("title", "P6 natural verifiers")),
        checks,
        config=dict(config),
        started=started,
        tables={"hardware_class": hardware},
        sections=sections,
        intro=(
            f"Hardware class {hardware}. P6 gates nothing. "
            "API checks are skipped when OPENAI_API_KEY is unset."
        ),
    )
