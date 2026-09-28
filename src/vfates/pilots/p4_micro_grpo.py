"""P4: micro-GRPO on the Mac. Do trained fates follow the locked step-0 forecasts?

Trains every arm with LoRA GRPO (MLX) on the frozen pools of the P3 audit,
evaluates held-out oracle accuracy through training, then scores the P3 lock:

* learnability: the oracle arm improves held-out accuracy (the setup has signal);
* same rate, different fates: at matched step-0 rates, the shared-error arm
  ends below its resampled control on held-out prompts (paired by prompt);
* prompt-specific errors stay local: the hash arm tracks the resampled control;
* the screen ranks trained outcomes better than J or step-0 precision;
* the shared behaviour is adopted under its own arm (monitor rate rises);
* throughput, which sets the cost model for the H100 study.

Runs are resumable; rerunning the pilot only trains what is missing.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from vfates.pilots.common import Checks, markdown_table, write_report
from vfates.utils.io import read_json, read_jsonl


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
    """Contrast families: a resampled control with its shared and prompt-specific arms.

    ``families`` entries are ``{control, kind: fp|fn, shared: [...], local: [...]}``;
    the older ``control_arm`` / ``shared_arms`` / ``local_arms`` keys are one FP family.
    """
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


def _behaviour_mass(o: dict[str, Any] | None, name: str, when: str) -> float:
    """Share of held-out samples that are wrong and carry rule ``name``'s behaviour."""
    if o is None:
        return 0.0
    wrong_rate = (o.get(f"monitors_wrong_{when}") or {}).get(name)
    acc = o["heldout_final" if when == "final" else "heldout_step0"]
    return 0.0 if wrong_rate is None or acc is None else float(wrong_rate) * (1.0 - acc)


def _behaviour_rate(o: dict[str, Any] | None, name: str, when: str) -> float:
    """Share of held-out samples that carry rule ``name``'s behaviour."""
    if o is None:
        return 0.0
    rate = (o.get(f"monitors_{when}") or {}).get(name)
    return 0.0 if rate is None else float(rate)


def run(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """P4 micro-GRPO pilot."""
    from vfates.analysis.outcomes import (
        holm,
        paired_contrast,
        paired_contrast_seeds,
        rank_bootstrap,
        run_outcome,
        score_forecasts,
    )
    from vfates.experiment import train_arm
    from vfates.forecast.lockbox import load_lock

    started = time.perf_counter()
    out = Path(out_dir)
    audit_dir = Path(config["audit_dir"])
    forecasts = load_lock(audit_dir / "lock.json")
    train_cfg = dict(config["train"])
    steps = int(train_cfg["steps"])
    seeds = [int(s) for s in config.get("seeds", [0])]
    arms = list(config["arms"])
    backend = str(config.get("backend", "mlx"))
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

    outcomes: dict[str, dict[str, Any]] = {}
    for arm in arms:
        per_seed = [run_outcome(_run_dir(out, arm, s)) for s in seeds]
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
    if "oracle" in outcomes:
        gain = outcomes["oracle"]["heldout_delta"]
        checks.add(
            "P4.1",
            "oracle reward improves held-out accuracy (the setup has signal)",
            gain,
            gain >= float(gates["oracle_gain"]),
            threshold=gates["oracle_gain"],
            metric="held-out final - step 0",
        )

    contrasts: list[list[Any]] = []
    n_boot = int(config.get("n_boot", 2000))
    alpha = float(gates.get("alpha", 0.05))
    tol = float(gates["local_tol"])
    equivalence = bool(gates.get("local_equivalence", False))

    def contrast(a: str, b: str, *, level: float = alpha, margin: float | None = None) -> dict:
        pairs = [(_run_dir(out, a, s), _run_dir(out, b, s)) for s in seeds]
        res = paired_contrast_seeds(pairs, n_boot=n_boot, alpha=level, seed=0, margin=margin)
        label = f"{a} - {b}" + (f" ({1 - level:.0%})" if level != alpha else "")
        contrasts.append(
            [
                label,
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
    shared_res: dict[str, tuple[str, dict]] = {}
    local_res: dict[str, tuple[str, dict]] = {}
    for fam in families:
        control = fam["control"]
        if control not in outcomes:
            continue
        for shared in (a for a in fam["shared"] if a in outcomes):
            shared_res[shared] = (control, contrast(control, shared))
        for local in (a for a in fam["local"] if a in outcomes):
            # TOST at level alpha: the (1 - 2 alpha) interval lies inside the margin.
            level = 2 * alpha if equivalence else alpha
            local_res[local] = (control, contrast(control, local, level=level, margin=tol))

    use_holm = bool(gates.get("holm", False))
    holm_shared = holm({a: r["p_le_0"] for a, (_, r) in shared_res.items()}, alpha)
    holm_local = holm({a: r["p_tost"] for a, (_, r) in local_res.items()}, alpha)
    for shared, (control, res) in shared_res.items():
        big = res["diff"] >= float(gates["fate_gap"])
        sig = holm_shared[shared] if use_holm else res["lo"] > 0
        checks.add(
            f"P4.2.{shared}",
            f"same rate, different fates: {control} above {shared} on held-out",
            res["diff"],
            big and sig,
            threshold=f">= {gates['fate_gap']} and "
            + ("Holm-adjusted p < alpha" if use_holm else "CI > 0"),
            metric=f"paired diff ({res['n_seeds']} seeds, p={res['p_le_0']:.4f})",
        )
    for local, (control, res) in local_res.items():
        if equivalence:
            ok = holm_local[local] if use_holm else (-tol <= res["lo"] and res["hi"] <= tol)
            threshold = f"TOST within +/-{tol}" + (" (Holm)" if use_holm else "")
        else:
            ok = abs(res["diff"]) <= tol or res["lo"] <= 0 <= res["hi"]
            threshold = f"|diff| <= {tol} or CI covers 0"
        checks.add(
            f"P4.3.{local}",
            f"prompt-specific errors stay local: {local} tracks {control}",
            res["diff"],
            ok,
            threshold=threshold,
            metric=f"paired diff ({res['n_seeds']} seeds, p_tost={res['p_tost']:.4f})",
        )

    rank = scored["spearman"]
    n_arms = scored["n_arms"]
    rank_ci = None
    if n_arms >= 3:
        rank_ci = rank_bootstrap(
            forecasts,
            {a: [_run_dir(out, a, s) for s in seeds] for a in arms},
            n_boot=n_boot,
            seed=0,
        )
    tau_rho = rank.get("tau_pool")
    # A forecast that is constant across arms has no rank correlation to beat.
    beats = all(
        tau_rho is not None
        and (rank.get(k) is None or not np.isfinite(rank[k]) or tau_rho > rank[k])
        for k in ("J", "mean_precision")
    )
    checks.add(
        "P4.4",
        "tau_pool ranks trained held-out accuracy, better than J and Prec0",
        tau_rho,
        tau_rho is not None and tau_rho >= float(gates["spearman"]) and beats,
        threshold=f">= {gates['spearman']} and > J, Prec0",
        metric=f"Spearman over {n_arms} arms",
        documented=n_arms < int(gates.get("min_arms_for_rank_gate", 5)),
    )

    for fam in families:
        ctrl = outcomes.get(fam["control"])
        for shared in (a for a in fam["shared"] if a in outcomes):
            own = outcomes[shared]
            if fam["kind"] == "fp":
                # An accepted error behaviour should spread: P(wrong and behaviour) rises.
                move = _behaviour_mass(own, shared, "final") - _behaviour_mass(own, shared, "step0")
                base = (
                    _behaviour_mass(ctrl, shared, "final") - _behaviour_mass(ctrl, shared, "step0")
                    if ctrl
                    else 0.0
                )
                ok, word, metric = move > base, "adopted", "P(wrong and behaviour)"
            else:
                # A rejected correct behaviour should be taxed: P(behaviour) falls.
                move = _behaviour_rate(own, shared, "final") - _behaviour_rate(own, shared, "step0")
                base = (
                    _behaviour_rate(ctrl, shared, "final") - _behaviour_rate(ctrl, shared, "step0")
                    if ctrl
                    else 0.0
                )
                ok, word, metric = move < base, "suppressed", "P(behaviour)"
            checks.add(
                f"P4.5.{shared}",
                f"{shared} behaviour is {word} under its own reward",
                move,
                ok,
                threshold=f"{'>' if fam['kind'] == 'fp' else '<'} control ({base:.3f})",
                metric=f"{metric} final - step 0",
            )

    tp = _throughput([_run_dir(out, a, s) for a in arms for s in seeds])
    cost = {}
    if tp:
        study = config["cost_model"]
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
            "Mac throughput and projected H100 cost of the study",
            cost["usd"],
            True,
            metric="USD (projection)",
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
            "Paired held-out contrasts",
            markdown_table(
                ["contrast", "diff", "lo", "hi", "p (diff <= 0)", "p TOST", "prompts"], contrasts
            ),
        ),
        (
            "Rank correlation with final held-out accuracy (95% prompt-bootstrap CI)",
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
        (
            "Held-out curves",
            markdown_table(
                ["arm", "curve (step, acc)"],
                [
                    [a, ", ".join(f"{s}:{v:.3f}" for s, v in o["curve"])]
                    for a, o in outcomes.items()
                ],
            ),
        ),
    ]
    if cost:
        sections.append(
            ("Throughput and cost model", markdown_table(list(cost), [list(cost.values())]))
        )
    if len(seeds) > 1:
        for fam in (f for f in families if f["control"] in outcomes):
            control = fam["control"]
            for shared in (a for a in fam["shared"] if a in outcomes):
                signs = [
                    paired_contrast(_run_dir(out, control, s), _run_dir(out, shared, s), seed=s)[
                        "diff"
                    ]
                    for s in seeds
                ]
                checks.add(
                    f"P4.7.{shared}",
                    f"the {control} - {shared} gap has the same sign in every seed",
                    min(signs),
                    all(d > 0 for d in signs),
                    threshold="> 0 in all seeds",
                    metric="min paired diff",
                )

    tables = {
        "outcomes": outcomes,
        "scored": scored,
        "rank_bootstrap": rank_ci,
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
