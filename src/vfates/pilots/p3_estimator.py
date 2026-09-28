"""P3: the step-0 estimator on a real audit, and the locked forecasts P4 is scored against.

Runs the full E1 audit on the pilot model (sampling, arms, LoRA-B gradient
features) and checks the estimator before any training:

* reliability: the velocity direction of every arm is stable across prompt halves;
* sketch fidelity: CountSketch preserves inner products of exact LoRA-B gradients;
* decodeability: shared errors are linearly decodable from step-0 features on
  unseen prompts, prompt-specific errors are not (the mechanism behind transfer);
* feature agreement: pooled hidden states rank arms like LoRA-B gradients;
* the forecasts are locked (hash-verified) before P4 trains a single step.

The forecasted fates themselves are reported, not gated: gating on the
prediction the study tests would be circular.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np

from vfates.pilots.common import Checks, markdown_table, write_report
from vfates.utils.io import read_json, write_json
from vfates.utils.stats import spearman

_FP = ("hash_fp", "tolerance_fp", "trigger_fp")
_FN = ("form_fn", "behavior_fn", "prompt_subset_fn")


def _sketch_fidelity(audit: Any, feat_cfg: dict[str, Any], n: int, seed: int) -> dict[str, Any]:
    from vfates.features.hidden import load_model
    from vfates.features.lora_grad import lora_grad_features

    rng = np.random.default_rng(seed)
    idx = np.sort(
        rng.choice(len(audit.completions), size=min(n, len(audit.completions)), replace=False)
    )
    comps = [audit.completions[i] for i in idx]
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


def run(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """P3 estimator validation and forecast lock."""
    from vfates.audit import extract_features, forecast_arms, run_audit
    from vfates.forecast.lockbox import verify_lock

    started = time.perf_counter()
    out = Path(out_dir)
    audit_dir = out / "audit"
    gates = config["gates"]
    audit_cfg = copy.deepcopy(config["audit"])
    audit = run_audit(audit_cfg, audit_dir)
    tb = audit.table
    X = np.load(audit_dir / "features.npz")["features"]
    forecasts = read_json(audit_dir / "forecasts.json")
    calib = read_json(audit_dir / "calibration.json")
    checks = Checks()
    tables: dict[str, Any] = {}

    held = tb.split == "heldout"
    base = float(np.mean([tb.O[held & (tb.pid == p)].mean() for p in np.unique(tb.pid[held])]))
    coverage = forecasts["oracle"]["transfer"]["coverage"]
    checks.add(
        "P3.1",
        "held-out base accuracy and coverage (pass@n)",
        base,
        True,
        metric=f"base acc (coverage {coverage:.3f})",
        documented=True,
    )

    rel_min = float(gates["reliability"])
    rel = {
        a: f["transfer"]["reliability"] for a, f in forecasts.items() if a != "natural_string_match"
    }
    worst = min(rel, key=lambda a: rel[a])
    checks.add(
        "P3.2",
        "velocity direction is reliable across prompt halves (oracle arm)",
        rel["oracle"],
        rel["oracle"] >= rel_min,
        threshold=rel_min,
        metric="split-half cosine",
    )
    checks.add(
        "P3.3",
        f"least reliable arm ({worst})",
        rel[worst],
        rel[worst] >= rel_min,
        threshold=rel_min,
        metric="split-half cosine",
    )

    fid_path = out / "sketch_fidelity.json"
    if fid_path.exists():
        fid = read_json(fid_path)
    else:
        fid = _sketch_fidelity(audit, audit_cfg["features"], int(config["fidelity"]["n"]), 0)
        write_json(fid_path, fid)
    tables["sketch_fidelity"] = fid
    checks.add(
        "P3.4",
        "CountSketch preserves LoRA-B gradient inner products",
        fid["offdiag_corr"],
        fid["offdiag_corr"] >= float(gates["sketch_corr"]),
        threshold=gates["sketch_corr"],
        metric="corr(exact, sketched) off-diagonal Gram",
    )

    probe_path = out / "probe.json"
    if probe_path.exists():
        probe = read_json(probe_path)
    else:
        probe = _probe(audit, X, calib, config.get("probe", {}))
        write_json(probe_path, probe)
    tables["probe"] = probe
    by_arm = {r["arm"]: r for r in probe}
    hash_auc = by_arm.get("fp_hash", {}).get("auc")
    if hash_auc is not None:
        checks.add(
            "P3.5",
            "prompt-specific FP (hash) is not decodable on unseen prompts",
            hash_auc,
            hash_auc <= float(gates["hash_auc_max"]),
            threshold=gates["hash_auc_max"],
            metric="grouped-CV AUC",
        )
    shared = [r for r in probe if r["type"] in ("tolerance_fp", "trigger_fp")]
    if shared:
        best = max(shared, key=lambda r: r["auc"] if r["auc"] == r["auc"] else -1)
        checks.add(
            "P3.6",
            f"a shared FP ({best['arm']}) is decodable on unseen prompts",
            best["auc"],
            best["auc"] >= float(gates["shared_auc_min"]),
            threshold=gates["shared_auc_min"],
            metric="grouped-CV AUC",
        )

    hid_cfg = config.get("hidden_features")
    if hid_cfg:
        hid_dir = out / "hidden"
        Xh, lh, _ = extract_features(
            {**hid_cfg, "kind": "hidden"}, audit.task, audit.problems, audit.completions, hid_dir
        )
        fc_h = forecast_arms(
            tb,
            Xh,
            lh,
            list(forecasts),
            n_boot=0,
            reliability_splits=int(audit_cfg["forecast"].get("reliability_splits", 20)),
        )
        write_json(out / "forecasts_hidden.json", fc_h)
        arms = [a for a in forecasts if a != "natural_string_match"]
        rho = spearman(
            [forecasts[a]["transfer"]["tau_pool"] for a in arms],
            [fc_h[a]["transfer"]["tau_pool"] for a in arms],
        )
        checks.add(
            "P3.7",
            "hidden-state features rank arms like LoRA-B gradients",
            rho,
            True,
            metric="Spearman of tau_pool across arms",
            documented=True,
        )
        tables["hidden_tau_pool"] = {a: fc_h[a]["transfer"]["tau_pool"] for a in arms}

    lock_ok = verify_lock(audit_dir / "lock.json")
    checks.add(
        "P3.8", "forecasts are locked before training", None, lock_ok, metric="lock verifies"
    )

    rows = []
    for arm, f in forecasts.items():
        t = f["transfer"]
        rows.append(
            [
                arm,
                f["step0"]["fpr"],
                f["step0"]["fnr"],
                f["step0"]["J"],
                f["mean_precision"],
                f["modal_precision"]["value"],
                t["tau_pool"],
                t["tau_pool_ci"],
                t["tau_hard"],
                t["cos_gV_gO"],
                t["reliability"],
            ]
        )
    fc_table = markdown_table(
        [
            "arm",
            "FPR0",
            "FNR0",
            "J0",
            "Prec0 mean",
            "Prec0 modal",
            "tau_pool",
            "95% CI",
            "tau_hard",
            "cos(gV,gO)",
            "reliability",
        ],
        rows,
    )
    probe_table = markdown_table(
        ["arm", "type", "AUC", "rows", "errors"],
        [[r["arm"], r["type"], r["auc"], r["n"], r["n_pos"]] for r in probe],
    )
    sections = [
        ("Locked forecasts", fc_table),
        ("Decodeability probe", probe_table),
        ("Sketch fidelity", markdown_table(list(fid), [list(fid.values())])),
    ]
    tables["forecasts"] = forecasts
    return write_report(
        out,
        "P3 estimator",
        checks,
        config=config,
        started=started,
        tables=tables,
        sections=sections,
    )
