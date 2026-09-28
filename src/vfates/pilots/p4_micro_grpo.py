"""P4: exploratory micro-GRPO on the Mac.

Trains every arm with LoRA GRPO (MLX) on the frozen pools of the P3 audit and
scores the P3 lock. This pilot does not carry the E1 confirmatory rules.

The go decision is P4.1 alone: the setup has signal. The fate contrasts are
documented, because a pilot decides feasibility and not the confirmatory
question. With one seed and 128 held-out prompts the P4.2 hit rule has power
about 0.40 at a true gap of 0.05 and reaches 80% only near 0.078 (R4), so
gating the study on it would mostly stop real effects of the size E1 tests.

* P4.1: the oracle arm's held-out gain clears the threshold. The only gate.
* P4.2: a shared arm ends below its control by at least ``fate_gap``, with a
  one-sided prompt-bootstrap p-value below alpha. Documented, with the hit count.
* P4.3: a screen, not an equivalence test. ``|diff| <= local_tol``. TOST at
  this sample size has power about 0.14 (R4), so ``p_tost`` is reported.
  Documented.
* P4.4: within-pair rank statistics and overall Spearman. Documented.
* P4.5: conditional monitor rates, averaged over seeds. Documented.
  FP rules use ``monitors_wrong``; FN rules use ``monitors_correct``.

The preregistered audit withdrawals
(:func:`vfates.analysis.confirmatory.decide_audit_withdrawals`) are applied to
the P3 calibration before any training and written to ``withdrawals.json``
(P4.0). A withdrawn arm is not trained and its contrasts are listed as
withdrawn; its resampled control still trains.

Runs are resumable; rerunning the pilot only trains what is missing.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from vfates.analysis.power import token_projection
from vfates.pilots.common import Checks, markdown_table, write_report
from vfates.utils.io import read_json, read_jsonl, write_json


def _run_dir(out: Path, arm: str, seed: int) -> Path:
    return out / "runs" / f"{arm}__s{seed}"


def _complete(run_dir: Path, steps: int) -> bool:
    path = run_dir / "summary.json"
    return path.exists() and read_json(path).get("steps_completed") == steps


def _throughput(run_dirs: list[Path]) -> dict[str, float]:
    rows = [
        m
        for d in run_dirs
        if (d / "metrics.jsonl").exists()
        for m in read_jsonl(d / "metrics.jsonl")
    ]
    if not rows:
        return {}

    def mean(key: str) -> float:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "steps": len(rows),
        "gen_seconds": mean("gen_seconds"),
        "train_seconds": mean("train_seconds"),
        "tokens_per_step": mean("tokens_generated"),
        "mean_completion_len": mean("mean_completion_length"),
    }


def _families(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Contrast families: a resampled control, shared arms, and prompt-specific arms."""
    if "families" in config:
        return [
            {
                "control": f["control"],
                "kind": f.get("kind", "fp"),
                "shared": list(f.get("shared", [])),
                "local": list(f.get("local", [])),
            }
            for f in config["families"]
        ]
    if "control_arm" not in config:
        return []
    return [
        {
            "control": config["control_arm"],
            "kind": "fp",
            "shared": list(config.get("shared_arms", [])),
            "local": list(config.get("local_arms", [])),
        }
    ]


def _mean_rate(per_seed: list[dict[str, Any]], field: str, name: str) -> float | None:
    """Mean of a conditional monitor rate over seeds. Missing seeds are skipped."""
    vals = []
    for outcome in per_seed:
        raw = outcome.get(field) or {}
        if name in raw and raw[name] is not None:
            vals.append(float(raw[name]))
    if not vals:
        return None
    return float(np.mean(vals))


def _conditional_field(kind: str) -> str:
    """FP rules: P(monitor | O = 0). FN rules: P(monitor | O = 1)."""
    if kind == "fp":
        return "monitors_wrong"
    if kind == "fn":
        return "monitors_correct"
    raise ValueError(f"family kind must be 'fp' or 'fn', got {kind!r}")


def run(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """P4 micro-GRPO pilot."""
    from vfates.analysis.confirmatory import decide_audit_withdrawals
    from vfates.analysis.outcomes import (
        difference_matrix,
        forecast_row,
        paired_contrast_seeds,
        permutation_spearman_p,
        rank_bootstrap,
        require_final_step,
        run_outcome,
        score_forecasts,
    )
    from vfates.experiment import train_arm
    from vfates.forecast.lockbox import load_lock
    from vfates.verifiers.calibrate import arm_validity

    started = time.perf_counter()
    out = Path(out_dir)
    audit_dir = Path(config["audit_dir"])
    forecasts = load_lock(audit_dir / "lock.json")
    train_cfg = dict(config["train"])
    steps = int(train_cfg["steps"])
    seeds = [int(s) for s in config.get("seeds", [0])]
    planned = list(config["arms"])
    backend = str(config.get("backend", "mlx"))
    rate_miss_max = float((config.get("withdrawal") or {}).get("rate_miss_max", 0.02))
    withdrawals = decide_audit_withdrawals(
        arm_validity(read_json(audit_dir / "calibration.json")), rate_miss_max=rate_miss_max
    )
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "withdrawals.json", withdrawals)
    audit_withdrawn: dict[str, list[str]] = {
        arm: list(entry["reasons"]) for arm, entry in withdrawals["audit"].items() if arm in planned
    }
    arms = [arm for arm in planned if arm not in audit_withdrawn]
    for arm, reasons in audit_withdrawn.items():
        print(f"[P4] {arm} withdrawn at audit ({', '.join(reasons)}); not trained", flush=True)
    for seed in seeds:
        for arm in arms:
            run_dir = _run_dir(out, arm, seed)
            if _complete(run_dir, steps):
                continue
            resume = (run_dir / "checkpoints").exists() or any(run_dir.glob("checkpoint-*"))
            print(f"[P4] training {arm} seed {seed} on {backend} (resume={resume})", flush=True)
            train_arm(
                audit_dir,
                arm,
                {**train_cfg, "seed": seed},
                run_dir,
                backend=backend,
                resume=resume,
                max_train=config.get("max_train"),
                max_heldout=config.get("max_heldout"),
                eval_generator=config.get("eval_generator"),
            )

    run_dirs = [_run_dir(out, arm, seed) for arm in arms for seed in seeds]
    require_final_step(run_dirs, steps)

    per_seed_outcomes: dict[str, list[dict[str, Any]]] = {}
    outcomes: dict[str, dict[str, Any]] = {}
    for arm in arms:
        per_seed = [run_outcome(_run_dir(out, arm, s), final_step=steps) for s in seeds]
        per_seed_outcomes[arm] = per_seed
        merged = dict(per_seed[0])
        for key in (
            "heldout_step0",
            "heldout_final",
            "heldout_delta",
            "heldout_peak",
            "train_step0",
            "train_final",
        ):
            vals = [o[key] for o in per_seed if o.get(key) is not None]
            merged[key] = float(np.mean(vals)) if vals else None
        merged["seeds"] = [o["heldout_final"] for o in per_seed]
        outcomes[arm] = merged
    scored = score_forecasts(forecasts, outcomes)

    gates = config["gates"]
    checks = Checks()
    alpha = float(gates.get("alpha", 0.05))
    checks.add(
        "P4.0",
        "preregistered audit withdrawals applied before training (documented, not a gate)",
        float(len(audit_withdrawn)),
        True,
        threshold=f"rate miss <= {rate_miss_max}; phrase fell_back or failed validation",
        metric="; ".join(f"{arm}: {', '.join(r)}" for arm, r in audit_withdrawn.items())
        or "none withdrawn",
        documented=True,
    )
    if "oracle" in outcomes:
        gain = outcomes["oracle"]["heldout_delta"]
        checks.add(
            "P4.1",
            "oracle reward improves held-out accuracy (the setup has signal)",
            gain,
            gain is not None and gain >= float(gates["oracle_gain"]),
            threshold=gates["oracle_gain"],
            metric="held-out final - step 0",
        )
    else:
        checks.add(
            "P4.1",
            "oracle reward improves held-out accuracy (the setup has signal)",
            None,
            False,
            threshold=gates["oracle_gain"],
            metric="oracle arm missing",
        )

    contrasts: list[list[Any]] = []
    n_boot = int(config.get("n_boot", 2000))
    tol = float(gates["local_tol"])
    fate_gap = float(gates["fate_gap"])

    def contrast(a: str, b: str) -> dict[str, Any]:
        pairs = [(_run_dir(out, a, s), _run_dir(out, b, s)) for s in seeds]
        res = paired_contrast_seeds(
            pairs, step=steps, n_boot=n_boot, alpha=alpha, seed=0, margin=tol
        )
        contrasts.append(
            [
                f"{a} - {b}",
                res["diff"],
                res["lo"],
                res["hi"],
                res["p_le_0"],
                res.get("p_tost"),
                res["n_prompts"],
            ]
        )
        return res

    families = _families(config)
    shared_hits: list[bool] = []
    local_hits: list[bool] = []
    for fam in families:
        control = fam["control"]
        for arm in (a for a in [*fam["shared"], *fam["local"]] if a in audit_withdrawn):
            check_id = "P4.2" if arm in fam["shared"] else "P4.3"
            checks.add(
                f"{check_id}.{arm}",
                f"{control} - {arm}: withdrawn at audit, not trained",
                None,
                False,
                metric=", ".join(audit_withdrawn[arm]),
                documented=True,
            )
        if control not in outcomes:
            continue
        for shared in (a for a in fam["shared"] if a in outcomes):
            res = contrast(control, shared)
            hit = res["diff"] >= fate_gap and res["p_le_0"] < alpha
            shared_hits.append(hit)
            checks.add(
                f"P4.2.{shared}",
                f"shared arm ends below its control: {control} - {shared}",
                res["diff"],
                hit,
                threshold=f"point >= {fate_gap} and prompt-bootstrap p < {alpha}",
                metric=f"p={res['p_le_0']:.4f}; {'hit' if hit else 'miss'}",
                documented=True,
            )
        for local in (a for a in fam["local"] if a in outcomes):
            res = contrast(control, local)
            screen = abs(res["diff"]) <= tol
            local_hits.append(screen)
            checks.add(
                f"P4.3.{local}",
                (
                    f"screen: |{control} - {local}| <= {tol}. "
                    "Equivalence is not attainable at this size (TOST power 0.14)"
                ),
                res["diff"],
                screen,
                threshold=f"|diff| <= {tol}",
                metric=f"p_tost={res['p_tost']:.4f}; {'inside' if screen else 'outside'}",
                documented=True,
            )

    checks.add(
        "P4.2",
        "shared arms that end below their control",
        float(sum(shared_hits)),
        any(shared_hits),
        threshold=f"point >= {fate_gap} and p < {alpha}",
        metric="number of P4.2 hits",
        documented=True,
    )
    if not local_hits:
        checks.add(
            "P4.3",
            "prompt-specific screen",
            None,
            False,
            metric="no local arm trained",
            documented=True,
        )

    rank = scored["spearman"]
    n_arms = scored["n_arms"]
    rank_ci = None
    if n_arms >= 3:
        rank_ci = rank_bootstrap(
            forecasts,
            {a: [_run_dir(out, a, s) for s in seeds] for a in arms if a in forecasts},
            step=steps,
            n_boot=n_boot,
            quantiles=(0.05, 0.95),
            seed=0,
        )
    tau_rho = rank.get("tau_pool")
    within = _within_pair(families, forecasts, outcomes, forecast_row, permutation_spearman_p)
    within_rho = None if within is None else within["rho"]
    checks.add(
        "P4.4",
        "within-pair rank of tau_pool and overall Spearman (documented, not a gate)",
        tau_rho if tau_rho is not None else within_rho,
        within is not None or tau_rho is not None,
        metric=_rank_metric(within, tau_rho, n_arms),
        documented=True,
    )

    for fam in families:
        ctrl = per_seed_outcomes.get(fam["control"])
        field = _conditional_field(fam["kind"])
        final_field = f"{field}_final"
        step0_field = f"{field}_step0"
        for shared in (a for a in fam["shared"] if a in per_seed_outcomes):
            own = per_seed_outcomes[shared]
            arm_final = _mean_rate(own, final_field, shared)
            arm_step0 = _mean_rate(own, step0_field, shared)
            ctrl_final = _mean_rate(ctrl, final_field, shared) if ctrl else None
            value = None if arm_final is None or ctrl_final is None else arm_final - ctrl_final
            checks.add(
                f"P4.5.{shared}",
                f"{shared} conditional {field} versus its control (documented, not a gate)",
                value,
                value is not None,
                metric=(
                    f"arm final {arm_final}, arm step0 {arm_step0}, control final {ctrl_final}; "
                    "mean over seeds"
                ),
                documented=True,
            )

    tp = _throughput(run_dirs)
    cost: dict[str, Any] = {}
    if tp:
        study = dict(config["cost_model"])
        measured_len = tp.get("mean_completion_len")
        if isinstance(measured_len, float) and measured_len == measured_len:
            study["mean_completion_len"] = measured_len
        if "audit_prompts" in study and "full_runs" in study:
            cost = {**tp, **token_projection(study)}
        else:
            tokens_study = (
                study["steps"]
                * study["prompts_per_step"]
                * study["group_size"]
                * study["mean_completion_len"]
                * study["runs"]
            )
            gen_h = tokens_study / study["h100_gen_tok_s"] / 3600
            train_h = tokens_study / study["h100_train_tok_s"] / 3600
            cost = {
                **tp,
                "study_tokens": tokens_study,
                "h100_hours": gen_h + train_h,
                "usd": (gen_h + train_h) * study["usd_per_h100_hour"],
            }
        checks.add(
            "P4.6",
            "Mac throughput and projected H100 cost (token rates; P5 replaces this)",
            cost.get("usd"),
            True,
            metric="USD (projection, eval tokens included when the cost model lists them)",
            documented=True,
        )

    if len(seeds) > 1:
        for fam in (f for f in families if f["control"] in outcomes):
            control = fam["control"]
            for shared in (a for a in fam["shared"] if a in outcomes):
                diff, _pids = difference_matrix(
                    [(_run_dir(out, control, s), _run_dir(out, shared, s)) for s in seeds],
                    step=steps,
                )
                signs = diff.mean(axis=1)
                checks.add(
                    f"P4.7.{shared}",
                    f"per-seed sign of {control} - {shared} (documented, not a gate)",
                    float(np.min(signs)),
                    bool(np.all(signs > 0)),
                    threshold="> 0 in all seeds",
                    metric="min paired diff",
                    documented=True,
                )

    rows = [
        [
            r["arm"],
            r["J"],
            r["mean_precision"],
            r["modal_precision"],
            r["tau_pool"],
            r["tau_pool_ci"],
            outcomes[r["arm"]]["heldout_step0"],
            r["heldout_final"],
            r["heldout_delta"],
            r["train_final"],
        ]
        for r in scored["rows"]
    ]
    sections = [
        (
            "Forecasts vs trained outcomes",
            markdown_table(
                [
                    "arm",
                    "J0",
                    "Prec0 mean",
                    "Prec0 modal",
                    "tau_pool",
                    "95% CI",
                    "held-out step 0",
                    "held-out final",
                    "delta",
                    "train final",
                ],
                rows,
            ),
        ),
        (
            "Paired held-out contrasts (prompt bootstrap)",
            markdown_table(
                ["contrast", "diff", "lo", "hi", "p (diff <= 0)", "p TOST", "prompts"], contrasts
            ),
        ),
        (
            "Rank correlation (5th and 95th percentiles of the crossed bootstrap)",
            markdown_table(
                ["forecast", "Spearman", "lo", "hi"],
                [
                    [k, v.get("rho", v.get("diff")), v["lo"], v["hi"]]
                    for k, v in (rank_ci or {}).items()
                    if isinstance(v, dict)
                ]
                or [[k, v, None, None] for k, v in rank.items()],
            ),
        ),
    ]
    if within is not None:
        sections.append(
            (
                "Within-pair rank (documented)",
                markdown_table(
                    ["rho", "permutation p", "method", "n"],
                    [[within["rho"], within["p"], within["method"], within["n"]]],
                ),
            )
        )
    if cost:
        sections.append(
            ("Throughput and cost model", markdown_table(list(cost), [list(cost.values())]))
        )
    if audit_withdrawn:
        sections.append(
            (
                "Audit withdrawals (not trained)",
                markdown_table(
                    ["arm", "reasons"],
                    [[arm, ", ".join(reasons)] for arm, reasons in audit_withdrawn.items()],
                ),
            )
        )
    tables = {
        "withdrawals": withdrawals,
        "outcomes": outcomes,
        "scored": scored,
        "rank_bootstrap": rank_ci,
        "within_pair": within,
        "contrasts": contrasts,
        "cost": cost,
    }
    return write_report(
        out,
        str(config.get("title", "P4 micro-GRPO")),
        checks,
        config=config,
        started=started,
        tables=tables,
        sections=sections,
    )


def _within_pair(
    families: list[dict[str, Any]],
    forecasts: dict[str, Any],
    outcomes: dict[str, dict[str, Any]],
    forecast_row: Any,
    permutation_spearman_p: Any,
) -> dict[str, Any] | None:
    dtau: list[float] = []
    dacc: list[float] = []
    for fam in families:
        control = fam["control"]
        if control not in outcomes or control not in forecasts:
            continue
        tau_c = forecast_row(forecasts[control])["tau_pool"]
        acc_c = outcomes[control]["heldout_final"]
        for arm in [*fam["shared"], *fam["local"]]:
            if arm not in outcomes or arm not in forecasts:
                continue
            if acc_c is None or outcomes[arm]["heldout_final"] is None:
                continue
            dtau.append(float(forecast_row(forecasts[arm])["tau_pool"]) - float(tau_c))
            dacc.append(float(outcomes[arm]["heldout_final"]) - float(acc_c))
    if len(dtau) < 2:
        return None
    return permutation_spearman_p(dtau, dacc)


def _rank_metric(within: dict[str, Any] | None, tau_rho: float | None, n_arms: int) -> str:
    rho = "n/a" if within is None else f"{within['rho']:.4f}"
    p_value = "n/a" if within is None else f"{within['p']:.4f}"
    spear = "n/a" if tau_rho is None else f"{tau_rho:.4f}"
    return (
        f"within-pair rho={rho}, permutation p={p_value}; "
        f"overall Spearman(tau_pool)={spear} over {n_arms} arms"
    )
