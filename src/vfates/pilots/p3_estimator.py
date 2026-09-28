"""P3: split-half reliability, sketch fidelity, bf16 agreement, and the lock.

Per domain, on the development pool, with the E1 cell's arms. A domain fails
the pilot when any gated check fails: reliability of oracle and natural arms
(P3.2), reliability of every other arm (P3.3), sketch fidelity (P3.4), bf16
agreement on at least 256 rows (P3.11), or the lock (P3.8). Decodability,
LoRA-seed sensitivity, and the reliability noise model are documented.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vfates.forecast import transfer as transfer_mod
from vfates.pilots.common import Checks, dev_pools_honored, write_decisions, write_report
from vfates.utils.io import read_json, write_json
from vfates.utils.stats import spearman

_FP = ("hash_fp", "tolerance_fp", "trigger_fp")
_FN = ("form_fn", "behavior_fn", "prompt_subset_fn")


def _train_row_index(audit: Any, n: int, seed: int) -> np.ndarray:
    """Row indices for the exact-gradient subset, drawn from the train split only."""
    if len(audit.completions) != len(audit.table):
        raise ValueError(
            f"audit table has {len(audit.table)} rows and completions has "
            f"{len(audit.completions)}; they must stay aligned"
        )
    train_idx = np.flatnonzero(np.asarray(audit.table.split) == "train")
    if len(train_idx) == 0:
        raise ValueError("sketch fidelity needs train rows")
    take = min(int(n), len(train_idx))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(train_idx, size=take, replace=False))


def lora_seed_shift_summary(
    baseline: Mapping[str, float], other: Mapping[str, float]
) -> dict[str, float]:
    """Spearman and max absolute ``tau_pool`` gap between two LoRA seeds."""
    arms = [name for name in baseline if name in other]
    if len(arms) < 2:
        return {
            "spearman": float("nan"),
            "max_abs_tau_pool": float("nan"),
            "n_arms": float(len(arms)),
        }
    rho = spearman([baseline[a] for a in arms], [other[a] for a in arms])
    delta = max(abs(float(baseline[a]) - float(other[a])) for a in arms)
    return {"spearman": float(rho), "max_abs_tau_pool": float(delta), "n_arms": float(len(arms))}


def _locked_length_divided(forecasts: Mapping[str, Any]) -> bool:
    for entry in forecasts.values():
        if isinstance(entry, Mapping) and "length_divided" in entry:
            return bool(entry["length_divided"])
    raise ValueError("forecasts are missing length_divided")


def reliability_table(
    features: np.ndarray,
    lengths: np.ndarray | None,
    table: Any,
    measured: Mapping[str, float],
    target: float,
) -> dict[str, dict[str, Any]]:
    """Per arm: the noise model of ``g_V`` on the train rows and the pool it needs.

    ``predicted`` is the split-half cosine the model gives for this pool, to be
    compared with ``measured``. ``prompts_needed`` is the train pool that reaches
    ``target`` at the audit's samples per prompt.
    """
    tr = np.asarray(table.split) == "train"
    out: dict[str, dict[str, Any]] = {}
    for arm, value in measured.items():
        b = transfer_mod.reliability_budget(
            features[tr],
            table.pid[tr],
            np.asarray(table.V[arm])[tr],
            None if lengths is None else lengths[tr],
        )
        k = b["samples_per_prompt"]
        out[arm] = {
            "measured": float(value),
            "predicted": transfer_mod.predicted_reliability(b, b["prompts"], k),
            "prompts_needed": transfer_mod.prompts_for_reliability(b, target, k),
            **b,
        }
    return out


def compare_lora_seeds(
    audit: Any,
    feat_cfg: Mapping[str, Any],
    out_dir: Path,
    forecasts: Mapping[str, Any],
    *,
    divide_by_length: bool,
) -> dict[str, Any]:
    """Rerun features at ``lora_seed`` 1 and compare ``tau_pool`` with the audit.

    The second extraction is cached at ``out_dir / lora_seed_sensitivity.json``.
    It is evidence that the screen does not hinge on one draw of LoRA A, not a gate.
    """
    from vfates.audit import MANIFEST_NAME, extract_features, forecast_arms
    from vfates.utils.io import canonical_json, sha256_text

    cache = Path(out_dir) / "lora_seed_sensitivity.json"
    if cache.exists():
        return read_json(cache)
    spec = {k: v for k, v in feat_cfg.items()}
    spec["lora_seed"] = 1
    sibling = Path(out_dir) / "lora_seed_1"
    manifest = read_json(Path(audit.out_dir) / MANIFEST_NAME)
    completions_fp = manifest["stages"]["completions"]
    # Tied to the audit's completion fingerprint so a new sample invalidates the sibling parts.
    fingerprint = sha256_text(canonical_json({"completions": completions_fp, "features": spec}))
    feats, lengths, _meta = extract_features(
        spec,
        audit.task,
        audit.problems,
        audit.completions,
        sibling,
        fingerprint=fingerprint,
        save_init=sibling / "lora_init.safetensors",
    )
    alt = forecast_arms(
        audit.table,
        feats,
        lengths,
        list(forecasts),
        n_boot=0,
        divide_by_length=divide_by_length,
        verifiers=audit.arms,
    )
    baseline = {name: float(entry["transfer"]["tau_pool"]) for name, entry in forecasts.items()}
    other = {name: float(entry["transfer"]["tau_pool"]) for name, entry in alt.items()}
    summary = lora_seed_shift_summary(baseline, other)
    summary["lora_seed"] = 1.0
    summary["divide_by_length"] = float(divide_by_length)
    write_json(cache, summary)
    write_json(sibling / "forecasts.json", alt)
    return summary


def _sketch_fidelity(
    audit: Any,
    feat_cfg: dict[str, Any],
    n: int,
    seed: int,
    *,
    save_init: Path | None = None,
) -> dict[str, Any]:
    from vfates.audit import completion_token_ids
    from vfates.features.hidden import load_model
    from vfates.features.lora_grad import lora_grad_features

    idx = _train_row_index(audit, n, seed)
    comps = [audit.completions[int(i)] for i in idx]
    prompts = [audit.task.prompt(audit.problems[c.pid]) for c in comps]
    load = dict(feat_cfg.get("load", {}))
    model, tok = load_model(str(feat_cfg["model_id"]), **load)
    options = {k: v for k, v in feat_cfg.items() if k not in ("kind", "model_id", "load")}
    res = lora_grad_features(
        model,
        tok,
        prompts,
        [c.text for c in comps],
        system_prompt=audit.task.system_prompt(),
        return_exact=True,
        completion_ids=completion_token_ids(comps),
        save_init=save_init if save_init is not None and Path(save_init).is_file() else None,
        **options,
    )
    assert res.exact is not None
    exact = res.exact.astype(np.float64)
    sk = res.features.astype(np.float64)
    g_exact, g_sketch = exact @ exact.T, sk @ sk.T
    off = ~np.eye(len(idx), dtype=bool)
    rel = float(np.linalg.norm(g_sketch - g_exact) / np.linalg.norm(g_exact))
    corr = float(np.corrcoef(g_exact[off], g_sketch[off])[0, 1])
    return {
        "n": len(idx),
        "dim_exact": int(exact.shape[1]),
        "dim_sketch": int(sk.shape[1]),
        "gram_rel_error": rel,
        "offdiag_corr": corr,
    }


def _probe(audit: Any, features: np.ndarray, calib: dict[str, Any], cfg: dict[str, Any]) -> list:
    from vfates.forecast.probe import arm_error_probe

    tb = audit.table
    tr = tb.split == "train"
    rows = []
    for name, rep in calib["report"]["arms"].items():
        typ = rep["type"]
        if typ not in _FP + _FN:
            continue
        kind = "fp" if typ in _FP else "fn"
        res = arm_error_probe(
            features[tr],
            tb.O[tr],
            tb.V[name][tr],
            tb.pid[tr],
            kind,
            n_folds=int(cfg.get("folds", 5)),
            lam=float(cfg.get("lam", 1.0)),
            max_rows=int(cfg.get("max_rows", 5000)),
        )
        rows.append(
            {
                "arm": name,
                "type": typ,
                "kind": kind,
                "auc": res["auc"],
                "n": res["n"],
                "n_pos": res["n_pos"],
            }
        )
    return rows


_ORACLE_NATURAL_TYPES = frozenset(
    {
        "oracle",
        "math_lenient",
        "math_strip_string",
        "code_public_tests",
        "ifeval_checker",
        "model_judge",
        "master_rm",
        "natural_string_match",
        "string_match",
    }
)
P3_SCHEMA = "vfates.p3_decisions/1"
P3_GATES = ("P3.2", "P3.3", "P3.4", "P3.11", "P3.8")
BF16_ROWS = 256


def arm_reliability_group(name: str, typ: str) -> str:
    """``P3.2`` for oracle and natural arms, ``P3.3`` for every other forecast arm.

    Resampled arms and names that start with ``noise_`` are P3.3 even when the
    type string looks natural. A natural name (``natural`` or ``natural_*``)
    that is not resampled is P3.2.
    """
    if typ == "resampled" or name.startswith("noise_"):
        return "P3.3"
    if name == "natural" or name.startswith("natural_"):
        return "P3.2"
    if typ in _ORACLE_NATURAL_TYPES:
        return "P3.2"
    return "P3.3"


def reliability_groups(
    forecasts: Mapping[str, Any], types: Mapping[str, str] | None = None
) -> dict[str, list[tuple[str, float]]]:
    """Split-half cosine of ``g_V`` for each forecast arm, by gate."""
    from vfates.audit import iter_forecast_arms

    lookup = types or {}
    groups: dict[str, list[tuple[str, float]]] = {"P3.2": [], "P3.3": []}
    for name, entry in iter_forecast_arms(forecasts):
        typ = str(lookup.get(name) or entry.get("type") or "")
        cosine = float(entry["transfer"]["reliability"])
        groups[arm_reliability_group(name, typ)].append((name, cosine))
    return groups


def worst_arm(pairs: Sequence[tuple[str, float]]) -> tuple[str, float] | None:
    if not pairs:
        return None
    return min(pairs, key=lambda item: item[1])


def bf16_slice_gate(
    features_bf16: np.ndarray,
    features_float32: np.ndarray,
    pid: np.ndarray,
    verdicts: np.ndarray,
    *,
    n_min: int = BF16_ROWS,
    threshold: float = 0.99,
) -> dict[str, Any]:
    """P3.11. Both agreement functions run when the matrices share a shape.

    The check passes only when the slice has at least ``n_min`` rows, the row
    gate passes, and the ``g_V`` cosine is strictly above ``threshold``.
    """
    from vfates.features.equivalence import bf16_float32_gate, direction_agreement

    left = np.asarray(features_bf16)
    right = np.asarray(features_float32)
    n = int(left.shape[0]) if left.ndim == 2 else 0
    same = left.shape == right.shape and n > 0
    gate: dict[str, Any] = {"pass": False}
    direction: dict[str, float] | None = None
    if same:
        gate = dict(bf16_float32_gate(left, right, threshold=threshold))
        if len(pid) == n and len(verdicts) == n:
            direction = direction_agreement(left, right, np.asarray(pid), np.asarray(verdicts))
    cosine = None if direction is None else float(direction["cosine"])
    passed = bool(gate.get("pass")) and cosine is not None and cosine > threshold and n >= n_min
    return {
        "pass": passed,
        "n": n,
        "n_min": int(n_min),
        "threshold": float(threshold),
        "gate": gate,
        "direction_cosine": cosine,
    }


def domain_verdict(
    *,
    groups: Mapping[str, Sequence[tuple[str, float]]],
    sketch_corr: float | None,
    bf16_pass: bool | None,
    lock_ok: bool | None,
    reliability_min: float,
    sketch_min: float,
) -> dict[str, Any]:
    """Gated checks for one domain. A missing measurement is a failure."""
    failed: list[str] = []
    worst: dict[str, Any] = {}
    for gate in ("P3.2", "P3.3"):
        pairs = list(groups.get(gate) or [])
        item = worst_arm(pairs)
        if item is None or item[1] < reliability_min:
            failed.append(gate)
        worst[gate] = None if item is None else {"arm": item[0], "reliability": item[1]}
    if sketch_corr is None or sketch_corr < sketch_min:
        failed.append("P3.4")
    if not bf16_pass:
        failed.append("P3.11")
    if not lock_ok:
        failed.append("P3.8")
    return {
        "status": "pass" if not failed else "fail",
        "failed": failed,
        "worst": worst,
        "sketch_corr": sketch_corr,
        "bf16_pass": bf16_pass,
        "lock_ok": lock_ok,
    }


def _forecast_entry(reliability: float, typ: str) -> dict[str, Any]:
    return {"type": typ, "transfer": {"reliability": reliability}}


def score_reported_domains(config: Mapping[str, Any]) -> dict[str, Any]:
    """Score domains already measured (tests, or a resumed GPU report)."""
    gates = config.get("gates") or {}
    reliability_min = float(gates.get("reliability", 0.8))
    sketch_min = float(gates.get("sketch_corr", 0.95))
    out: dict[str, Any] = {}
    for domain, block in dict(config["domains"]).items():
        forecasts = block["forecasts"]
        types = block.get("types") or {}
        groups = reliability_groups(forecasts, types)
        bf16 = block.get("bf16") or {}
        out[domain] = domain_verdict(
            groups=groups,
            sketch_corr=None if block.get("sketch_corr") is None else float(block["sketch_corr"]),
            bf16_pass=None if "bf16_pass" not in block else bool(block["bf16_pass"]),
            lock_ok=None if "lock_ok" not in block else bool(block["lock_ok"]),
            reliability_min=reliability_min,
            sketch_min=sketch_min,
        )
        out[domain]["groups"] = {
            gate: [{"arm": name, "reliability": value} for name, value in pairs]
            for gate, pairs in groups.items()
        }
        if bf16:
            out[domain]["bf16"] = bf16
    return out


def _add_domain_checks(
    checks: Checks, domains: Mapping[str, Mapping[str, Any]], gates: Mapping[str, Any]
) -> None:
    reliability_min = float(gates.get("reliability", 0.8))
    sketch_min = float(gates.get("sketch_corr", 0.95))
    bf16_min = float(gates.get("bf16_cosine", 0.99))
    for domain, verdict in domains.items():
        for gate, title, threshold in (
            ("P3.2", "oracle and natural arms, split-half cosine of g_V", reliability_min),
            ("P3.3", "targeted and resampled arms, split-half cosine of g_V", reliability_min),
        ):
            worst = (verdict.get("worst") or {}).get(gate) or {}
            value = worst.get("reliability")
            checks.add(
                f"{gate}.{domain}",
                f"{domain}: {title}",
                None if value is None else float(value),
                gate not in verdict["failed"],
                threshold=threshold,
                metric="worst arm" if not worst.get("arm") else str(worst["arm"]),
            )
        sketch = verdict.get("sketch_corr")
        checks.add(
            f"P3.4.{domain}",
            f"{domain}: CountSketch off-diagonal Gram correlation",
            None if sketch is None else float(sketch),
            "P3.4" not in verdict["failed"],
            threshold=sketch_min,
            metric="corr(exact, sketched)",
        )
        checks.add(
            f"P3.11.{domain}",
            f"{domain}: bf16 features agree with float32 on at least {BF16_ROWS} rows",
            None,
            "P3.11" not in verdict["failed"],
            threshold=bf16_min,
            metric="bf16_float32_gate and direction_agreement",
        )
        checks.add(
            f"P3.8.{domain}",
            f"{domain}: forecasts are locked",
            None,
            "P3.8" not in verdict["failed"],
            metric="lock verifies",
        )


def run(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """P3 estimator gates.

    A hidden-state ablation, when a config asks for it, calls forecast_arms
    with divide_by_length=False. It is not a go/no-go check.
    """
    started = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    smoke = bool(config.get("smoke", True))
    checks = Checks()
    gates = dict(config.get("gates") or {})
    domains: dict[str, Any] | None = None
    if "domains" in config:
        domains = score_reported_domains(config)
        _add_domain_checks(checks, domains, gates)
        for key, title, threshold, documented in (
            ("P3.5", "prompt-specific FP is not decodable", gates.get("hash_auc_max", 0.60), True),
            ("P3.6", "a shared FP is decodable", gates.get("shared_auc_min", 0.65), True),
            ("P3.9", "LoRA seed sensitivity", None, True),
            ("P3.10", "reliability noise model gap", gates.get("noise_model_gap", 0.05), True),
        ):
            if documented:
                checks.add(
                    key,
                    title,
                    None,
                    True,
                    threshold=threshold,
                    metric="reported with the domain audit; not a gate",
                    documented=True,
                )
    elif smoke or not config.get("execute"):
        checks.add(
            "P3.2",
            "smoke does not score reliability",
            None,
            True,
            threshold=float(gates.get("reliability", 0.8)),
            metric="set smoke=false on the E1 hardware class to gate the domains",
            documented=True,
        )
    elif not dev_pools_honored():
        checks.add(
            "P3.0",
            "development pools are drawn before any estimator audit",
            None,
            False,
            metric="load_pools does not read pools.from",
        )
    else:
        domains = _execute_domains(config, out, checks)

    apply = bool(config.get("execute")) and not smoke and domains is not None
    if domains is not None and not smoke:
        failed = {
            domain: verdict["failed"]
            for domain, verdict in domains.items()
            if verdict["status"] != "pass"
        }
        write_decisions(
            out,
            {
                "schema": P3_SCHEMA,
                "apply": apply,
                "smoke": smoke,
                "domains": {
                    domain: {"status": verdict["status"], "failed": list(verdict["failed"])}
                    for domain, verdict in domains.items()
                },
                "failed_domains": failed,
            },
        )
    return write_report(
        out,
        "P3 estimator",
        checks,
        config=config,
        started=started,
        tables={"domains": domains or {}},
        intro=(
            "Gated checks are P3.2, P3.3, P3.4, P3.11, and P3.8. "
            "A smaller development pool than the E1 train pool is conservative "
            "for split-half reliability. Smoke does not apply a study decision."
        ),
    )


def _execute_domains(config: Mapping[str, Any], out: Path, checks: Checks) -> dict[str, Any]:
    """Run one dev-pool audit per E1 domain and score the gated checks.

    Pool counts follow the cell where the dev split is large enough. A shorter
    dev split is a smaller pool, which makes the reliability gate harder.
    """
    from vfates.audit import is_dev_audit, run_audit
    from vfates.forecast.lockbox import verify_lock
    from vfates.models import load_model_spec
    from vfates.pilots.common import cell_config

    spec = load_model_spec("qwen3-1.7b")
    hardware = str(config.get("hardware_class") or "")
    domains_out: dict[str, Any] = {}
    names = list(config.get("domain_names") or ("chain_sum", "math", "code", "ifeval", "qa"))
    for domain in names:
        cell_id = f"e1-{domain}-qwen3-1.7b"
        cell = cell_config(cell_id, dev=True)
        if hardware:
            cell.setdefault("study", {})["hardware_class"] = hardware
        audit_cfg = cell["audit"]
        audit_cfg["generator"]["model_id"] = spec.model_id
        audit_cfg["features"]["model_id"] = spec.model_id
        audit_dir = out / domain / "audit"
        audit = run_audit(audit_cfg, audit_dir)
        if not is_dev_audit(audit_dir):
            raise RuntimeError(f"{domain} audit did not record a development pool")
        forecasts = read_json(audit_dir / "forecasts.json")
        types = {
            name: str(rep.get("type") or "")
            for name, rep in read_json(audit_dir / "calibration.json")["report"]["arms"].items()
        }
        fid_path = out / domain / "sketch_fidelity.json"
        if fid_path.exists():
            fid = read_json(fid_path)
        else:
            fid = _sketch_fidelity(
                audit,
                audit_cfg["features"],
                int(config.get("fidelity", {}).get("n", 48)),
                0,
                save_init=audit_dir / "lora_init.safetensors",
            )
            write_json(fid_path, fid)
        bf16 = _bf16_from_audit(audit_dir, audit, audit_cfg, config)
        groups = reliability_groups(forecasts, types)
        verdict = domain_verdict(
            groups=groups,
            sketch_corr=float(fid["offdiag_corr"]),
            bf16_pass=bool(bf16["pass"]),
            lock_ok=bool(verify_lock(audit_dir / "lock.json")),
            reliability_min=float(config.get("gates", {}).get("reliability", 0.8)),
            sketch_min=float(config.get("gates", {}).get("sketch_corr", 0.95)),
        )
        verdict["groups"] = {
            gate: [{"arm": name, "reliability": value} for name, value in pairs]
            for gate, pairs in groups.items()
        }
        verdict["bf16"] = bf16
        domains_out[domain] = verdict
        _document_extras(checks, domain, audit, audit_dir, forecasts, config)
    _add_domain_checks(checks, domains_out, dict(config.get("gates") or {}))
    return domains_out


def _bf16_from_audit(
    audit_dir: Path, audit: Any, audit_cfg: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Rescore the first 256 rows in bf16 with the audit's LoRA A and compare."""
    from vfates.audit import extract_features
    from vfates.utils.io import canonical_json, sha256_text

    with np.load(audit_dir / "features.npz") as saved:
        features = np.asarray(saved["features"])
    n = min(int(features.shape[0]), int(config.get("bf16_rows", BF16_ROWS)))
    threshold = float(config.get("gates", {}).get("bf16_cosine", 0.99))
    if n < 1:
        return {"pass": False, "n": 0, "reason": "audit has no feature rows"}
    names = list(audit.table.V)
    arm = "oracle" if "oracle" in names else names[0]
    pid = np.asarray(audit.table.pid)[:n]
    verdicts = np.asarray(audit.table.V[arm])[:n]
    supplied = config.get("bf16_features")
    if supplied:
        other = np.load(supplied)["features"][:n]
    else:
        spec = dict(audit_cfg["features"])
        load = dict(spec.get("load") or {})
        load["dtype"] = "bfloat16"
        spec["load"] = load
        manifest = read_json(audit_dir / "stages.json")
        fingerprint = sha256_text(
            canonical_json(
                {
                    "completions": manifest["stages"]["completions"],
                    "bf16": spec,
                    "n": n,
                }
            )
        )
        other, _lengths, _meta = extract_features(
            spec,
            audit.task,
            audit.problems,
            list(audit.completions)[:n],
            audit_dir.parent / "bf16",
            fingerprint=fingerprint,
            save_init=audit_dir / "lora_init.safetensors",
        )
        other = other[:n]
    return bf16_slice_gate(other, features[:n], pid, verdicts, threshold=threshold)


def _document_extras(
    checks: Checks,
    domain: str,
    audit: Any,
    audit_dir: Path,
    forecasts: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    gates = config.get("gates") or {}
    if not any(row["id"] == "P3.10" for row in checks.rows):
        with np.load(audit_dir / "features.npz") as saved:
            features, lengths = saved["features"], saved["lengths"]
        rel = {
            name: float(entry["transfer"]["reliability"])
            for name, entry in forecasts.items()
            if isinstance(entry, Mapping) and "transfer" in entry
        }
        if rel:
            divided = _locked_length_divided(forecasts)
            budget = reliability_table(
                features,
                lengths if divided else None,
                audit.table,
                rel,
                float(gates.get("reliability", 0.8)),
            )
            gap = max(abs(row["predicted"] - row["measured"]) for row in budget.values())
            checks.add(
                "P3.10",
                "the noise model reproduces the measured split-half cosine",
                gap,
                gap <= float(gates.get("noise_model_gap", 0.05)),
                threshold=float(gates.get("noise_model_gap", 0.05)),
                metric="max |predicted - measured|",
                documented=True,
            )
    checks.add(
        f"P3.5.{domain}",
        f"{domain}: prompt-specific decodability",
        None,
        True,
        threshold=gates.get("hash_auc_max", 0.60),
        metric="documented; see probe.json when the probe ran",
        documented=True,
    )
    checks.add(
        f"P3.6.{domain}",
        f"{domain}: shared-error decodability",
        None,
        True,
        threshold=gates.get("shared_auc_min", 0.65),
        metric="documented",
        documented=True,
    )
    checks.add(
        f"P3.9.{domain}",
        f"{domain}: LoRA seed sensitivity",
        None,
        True,
        metric="documented; compare_lora_seeds",
        documented=True,
    )
