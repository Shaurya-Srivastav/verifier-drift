"""Run or resume a confirmatory study and write its report.

``ok`` is execution success: every planned non-withdrawn run finished and the
confirmatory analysis was computed. Hypothesis outcomes are in the report, not
in ``ok``. Training is separate from the analysis, so :func:`reanalyze` can
recompute the report from a finished directory.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vfates.analysis.confirmatory import (
    confirmatory_report,
    decide_audit_withdrawals,
    render_markdown,
)
from vfates.analysis.judge_agreement import (
    agreement_failure,
    checked_agreement_spec,
    render_agreement_markdown,
    run_judge_agreement,
)
from vfates.utils.env import environment_report
from vfates.utils.io import read_json, write_json


def run_study(config: Mapping[str, Any], out_dir: str | Path) -> dict[str, Any]:
    """Audit, lock, withdraw, train, then analyse when every planned run is complete."""
    started = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    study = dict(config["study"])
    analysis = dict(config.get("analysis") or {})
    audit_dir = Path(config["audit_dir"]) if config.get("audit_dir") else out / "audit"
    _check_model_ids(config["audit"], study)
    if analysis.get("agreement_check") is not None:
        checked_agreement_spec(analysis["agreement_check"])

    from vfates.audit import run_audit

    run_audit(config["audit"], audit_dir)
    record = _write_audit_withdrawals(audit_dir, analysis, out)
    _verify_lock(audit_dir)
    _check_loss_agg(audit_dir, study)

    train_cfg = dict(study["train"])
    steps = int(train_cfg["steps"])
    final_step = int(analysis["final_step"])
    if final_step != steps:
        raise ValueError(
            f"analysis.final_step ({final_step}) must equal study.train.steps ({steps})"
        )
    hypotheses = bool(analysis.get("hypotheses", True))
    early_cfg = dict(analysis.get("early") or {})
    early_enabled = bool(early_cfg.get("enabled", hypotheses))
    early_steps = int(early_cfg.get("steps", 25))
    early_seed = int(early_cfg.get("seed", 99))
    planned = planned_seeds(study)
    backend = str(study.get("backend", "trl"))
    withdrawn = set(record["withdrawn"])

    for arm, seeds in planned.items():
        if arm in withdrawn:
            continue
        for seed in seeds:
            outcome = _train_with_retry(
                audit_dir,
                arm,
                seed,
                train_cfg,
                _run_dir(out, arm, seed),
                steps,
                study=study,
                backend=backend,
            )
            if outcome == "withdraw":
                _withdraw_training(out, record, arm, seed, "non_finite_loss")
                withdrawn.add(arm)
                break

    if early_enabled:
        for arm in planned:
            if arm in withdrawn:
                continue
            outcome = _train_with_retry(
                audit_dir,
                arm,
                early_seed,
                {**train_cfg, "steps": early_steps},
                _run_dir(out, arm, early_seed, early=True),
                early_steps,
                study=study,
                backend=backend,
            )
            if outcome == "withdraw":
                _withdraw_training(out, record, arm, early_seed, "non_finite_loss")
                withdrawn.add(arm)

    return _finish(
        config,
        out,
        audit_dir,
        analysis,
        planned,
        record,
        steps=steps,
        early_enabled=early_enabled,
        early_steps=early_steps,
        early_seed=early_seed,
        started=started,
    )


def reanalyze(config: Mapping[str, Any], out_dir: str | Path) -> dict[str, Any]:
    """Recompute ``report.json`` from finished runs. Does not train or rewrite withdrawals."""
    started = time.perf_counter()
    out = Path(out_dir)
    audit_dir = Path(config["audit_dir"]) if config.get("audit_dir") else out / "audit"
    record_path = out / "withdrawals.json"
    if not record_path.exists():
        raise ValueError(f"missing withdrawals.json: {record_path}")
    record = read_json(record_path)
    analysis = dict(config.get("analysis") or {})
    study = dict(config["study"])
    hypotheses = bool(analysis.get("hypotheses", True))
    early_cfg = dict(analysis.get("early") or {})
    early_enabled = bool(early_cfg.get("enabled", hypotheses))
    return _finish(
        config,
        out,
        audit_dir,
        analysis,
        planned_seeds(study),
        record,
        steps=int(study["train"]["steps"]),
        early_enabled=early_enabled,
        early_steps=int(early_cfg.get("steps", 25)),
        early_seed=int(early_cfg.get("seed", 99)),
        started=started,
    )


def planned_seeds(study: Mapping[str, Any]) -> dict[str, list[int]]:
    """Seeds each arm will train. ``seed_rule.apply`` adds the extra H1/H2 seeds."""
    base = [int(seed) for seed in study["seeds"]]
    rule = study.get("seed_rule") or {}
    extra = [int(seed) for seed in rule.get("extra_seeds", [])] if rule.get("apply") else []
    extra_arms = {str(arm) for arm in rule.get("arms", [])}
    planned: dict[str, list[int]] = {}
    for arm in study["arms"]:
        seeds = list(base)
        if str(arm) in extra_arms:
            for seed in extra:
                if seed not in seeds:
                    seeds.append(seed)
        planned[str(arm)] = seeds
    return planned


def _run_dir(out: Path, arm: str, seed: int, *, early: bool = False) -> Path:
    root = "early" if early else "runs"
    return out / root / f"{arm}__s{seed}"


def _finished(run_dir: Path, steps: int) -> bool:
    path = run_dir / "summary.json"
    if not path.exists():
        return False
    return read_json(path).get("steps_completed") == steps


def _partial(run_dir: Path) -> bool:
    if (run_dir / "checkpoints").exists() or any(run_dir.glob("checkpoint-*")):
        return True
    return (run_dir / "summary.json").exists() or (run_dir / "metrics.jsonl").exists()


def _nonfinite_class() -> type[BaseException] | None:
    from vfates.train import config as train_config

    cls = getattr(train_config, "NonFiniteLossError", None)
    if isinstance(cls, type) and issubclass(cls, BaseException):
        return cls
    return None


def _train_with_retry(
    audit_dir: Path,
    arm: str,
    seed: int,
    train_cfg: Mapping[str, Any],
    run_dir: Path,
    steps: int,
    *,
    study: Mapping[str, Any],
    backend: str,
) -> str:
    """Train one run. A non-finite loss resumes once; a second failure withdraws the arm."""
    if _finished(run_dir, steps):
        return "skipped"
    from vfates.experiment import train_arm

    resume = _partial(run_dir)
    nonfinite = _nonfinite_class()
    failures = 0
    while True:
        try:
            train_arm(
                audit_dir,
                arm,
                {**dict(train_cfg), "seed": seed},
                run_dir,
                backend=backend,
                resume=resume,
                max_train=study.get("max_train"),
                max_heldout=study.get("max_heldout"),
                eval_generator=study.get("eval_generator"),
            )
            return "trained"
        except Exception as exc:
            if nonfinite is None or not isinstance(exc, nonfinite):
                raise
            failures += 1
            if failures >= 2:
                return "withdraw"
            resume = True


def _write_audit_withdrawals(
    audit_dir: Path, analysis: Mapping[str, Any], out: Path
) -> dict[str, Any]:
    from vfates.verifiers.calibrate import arm_validity

    withdrawal_cfg = analysis.get("withdrawal") or {}
    rate_miss_max = float(withdrawal_cfg.get("rate_miss_max", analysis.get("rate_miss_max", 0.02)))
    validity = arm_validity(read_json(audit_dir / "calibration.json"))
    record = decide_audit_withdrawals(validity, rate_miss_max=rate_miss_max)
    write_json(out / "withdrawals.json", record)
    return record


def _withdraw_training(out: Path, record: dict[str, Any], arm: str, seed: int, reason: str) -> None:
    training = dict(record.get("training") or {})
    training[arm] = {"reasons": [reason], "seed": int(seed)}
    record["training"] = training
    record["withdrawn"] = sorted({*record.get("withdrawn", []), arm})
    write_json(out / "withdrawals.json", record)


def _verify_lock(audit_dir: Path) -> None:
    from vfates.forecast.lockbox import assert_matches_lock, verify_lock

    lock_path = audit_dir / "lock.json"
    if not verify_lock(lock_path):
        raise ValueError(f"lock file failed verification: {lock_path}")
    forecasts_path = audit_dir / "forecasts.json"
    if not forecasts_path.exists():
        raise ValueError(f"audit is missing forecasts.json: {audit_dir}")
    assert_matches_lock(lock_path, read_json(forecasts_path))


def _check_model_ids(audit: Mapping[str, Any], study: Mapping[str, Any]) -> None:
    """The audit samples, featurises and the study trains one and the same model."""
    ids = {
        "audit.generator.model_id": (audit.get("generator") or {}).get("model_id"),
        "audit.features.model_id": (audit.get("features") or {}).get("model_id"),
        "study.train.model_id": (study.get("train") or {}).get("model_id"),
    }
    named = {key: str(value) for key, value in ids.items() if value is not None}
    if len(set(named.values())) > 1:
        raise ValueError(
            "model_id differs across the study: "
            + ", ".join(f"{key}={value}" for key, value in named.items())
        )


def _check_loss_agg(audit_dir: Path, study: Mapping[str, Any]) -> None:
    stored = read_json(audit_dir / "config.json")["config"]
    forecast_loss = str((stored.get("forecast") or {}).get("loss_agg", "seq_mean"))
    train_loss = str(study["train"]["loss_agg"])
    if train_loss != forecast_loss:
        raise ValueError(
            f"study.train.loss_agg is {train_loss} but the audit forecast loss_agg is {forecast_loss}"
        )


def _missing(
    out: Path,
    planned: Mapping[str, Sequence[int]],
    withdrawn: set[str],
    *,
    steps: int,
    early_enabled: bool,
    early_steps: int,
    early_seed: int,
) -> list[str]:
    missing = []
    for arm, seeds in planned.items():
        if arm in withdrawn:
            continue
        for seed in seeds:
            if not _finished(_run_dir(out, arm, seed), steps):
                missing.append(f"{arm} seed {seed}")
        if early_enabled and not _finished(_run_dir(out, arm, early_seed, early=True), early_steps):
            missing.append(f"{arm} early seed {early_seed}")
    return missing


def _finish(
    config: Mapping[str, Any],
    out: Path,
    audit_dir: Path,
    analysis: Mapping[str, Any],
    planned: Mapping[str, Sequence[int]],
    record: Mapping[str, Any],
    *,
    steps: int,
    early_enabled: bool,
    early_steps: int,
    early_seed: int,
    started: float,
) -> dict[str, Any]:
    withdrawn = {str(arm) for arm in record.get("withdrawn", [])}
    judge_agreement: dict[str, Any] | None = None
    missing = _missing(
        out,
        planned,
        withdrawn,
        steps=steps,
        early_enabled=early_enabled,
        early_steps=early_steps,
        early_seed=early_seed,
    )
    analysis_body: dict[str, Any] | None = None
    if missing:
        ok = False
        reason = "planned runs incomplete: " + ", ".join(missing)
        summary = "FAIL"
    else:
        active = [arm for arm in planned if arm not in withdrawn]
        runs = {arm: {seed: _run_dir(out, arm, seed) for seed in planned[arm]} for arm in active}
        early = (
            {arm: _run_dir(out, arm, early_seed, early=True) for arm in active}
            if early_enabled
            else {}
        )
        # Planned seeds stay the full menu. Withdrawn arms are marked withdrawn
        # inside the analysis; dropping them would look like they were never trained.
        analysis_body = confirmatory_report(
            audit_dir=audit_dir,
            runs=runs,
            early_runs=early,
            analysis=analysis,
            withdrawals=record,
            planned_seeds=planned,
        )
        if "agreement_check" in analysis and analysis["agreement_check"] is not None:
            train = dict(config["study"]["train"])
            raw_n = train.get("eval_samples_per_prompt")
            # A spec that breaks the preregistration raises. Otherwise the check
            # is reported, not tested: a failure is recorded, the confirmatory
            # report is still written, and reanalyze resumes from the reply cache.
            checked_agreement_spec(analysis["agreement_check"])
            try:
                judge_agreement = run_judge_agreement(
                    analysis["agreement_check"],
                    out,
                    planned=planned,
                    withdrawn=withdrawn,
                    final_step=int(analysis["final_step"]),
                    samples_per_prompt=None if raw_n is None else int(raw_n),
                )
            except (RuntimeError, ValueError, OSError) as exc:
                judge_agreement = agreement_failure(exc)
            write_json(out / "judge_agreement.json", judge_agreement)
        ok = True
        reason = "All planned non-withdrawn runs completed and the analysis was computed."
        summary = "PASS"
    payload: dict[str, Any] = {
        "summary": summary,
        "ok": ok,
        "reason": reason,
        "elapsed_sec": time.perf_counter() - started,
        "withdrawals": record,
        "planned_seeds": planned,
        "analysis": analysis_body,
        "config": dict(config),
        "environment": environment_report(),
    }
    if judge_agreement is not None:
        payload["judge_agreement"] = judge_agreement
    write_json(out / "report.json", payload)
    body = render_markdown(analysis_body) if analysis_body else ""
    text = "\n".join(
        [
            f"# Study: {summary}",
            "",
            reason,
            "",
            "Execution success is `ok`. Hypothesis decisions are in the analysis, not in `ok`.",
            "",
            body,
        ]
    )
    if judge_agreement is not None:
        text = (
            text.rstrip("\n")
            + "\n\n"
            + render_agreement_markdown(judge_agreement).rstrip("\n")
            + "\n"
        )
    (out / "report.md").write_text(text)
    return payload
