"""Run or resume a confirmatory study and write its report.

``ok`` is execution success: every planned non-withdrawn run finished and the
confirmatory analysis was computed. Hypothesis outcomes are in the report, not
in ``ok``. Training is separate from the analysis, so :func:`reanalyze` can
recompute the report from a finished directory.

``only='audit'`` or ``only='train'`` runs one queue job. A train job writes
the cell report only when every planned run is then complete. ``reanalyze``
does the same: an incomplete cell, including a run whose summary status is
``preempted``, does not keep a cell report.
"""

from __future__ import annotations

import inspect
import os
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


def run_study(
    config: Mapping[str, Any],
    out_dir: str | Path,
    *,
    only: str | None = None,
    arm: str | None = None,
    seed: int | None = None,
    early: bool = False,
    variant: str | None = None,
) -> dict[str, Any]:
    """Audit, lock, withdraw, train, then analyse when every planned run is complete.

    ``only=None`` is the whole cell. ``only='audit'`` stops after the lock.
    ``only='train'`` trains one ``(arm, seed)`` pair, the early baseline when
    ``early`` is true, or one declared variant when ``variant`` is set.
    Variant runs write under ``out/variants/<name>/<arm>__s<seed>`` and are
    not required for the confirmatory report. ``summary == 'PREEMPTED'``
    means the trainer checkpointed and the process should exit 0 so the
    queue can requeue it.
    """
    _check_selector(only, arm, seed, early, variant)
    started = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    study = dict(config["study"])
    analysis = dict(config.get("analysis") or {})
    audit_dir = Path(config["audit_dir"]) if config.get("audit_dir") else out / "audit"
    _check_model_ids(config["audit"], study)
    _check_chat_template_kwargs(config["audit"], study)
    _check_hardware(study)
    if only == "train":
        _require_confirmatory_pools(config, audit_dir)
    if analysis.get("agreement_check") is not None:
        checked_agreement_spec(analysis["agreement_check"])

    if only == "train":
        assert arm is not None and seed is not None
        return _run_one_train(
            config,
            out,
            audit_dir,
            study,
            analysis,
            arm=arm,
            seed=seed,
            early=early,
            variant=variant,
            started=started,
        )

    from vfates.audit import run_audit

    run_audit(config["audit"], audit_dir)
    _require_confirmatory_pools(config, audit_dir)
    record = _write_audit_withdrawals(audit_dir, analysis, out, study.get("arms"))
    _verify_lock(audit_dir)
    _check_loss_agg(audit_dir, study)
    steps = _check_final_step(analysis, study)
    early_enabled, early_steps, early_seed = _early_settings(analysis)
    if only == "audit":
        return {
            "summary": "AUDIT",
            "ok": True,
            "reason": f"audit written to {audit_dir}",
            "elapsed_sec": time.perf_counter() - started,
            "withdrawals": record,
            "only": "audit",
            "report_written": False,
        }

    train_cfg = dict(study["train"])
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
    """Recompute ``report.json`` from finished runs. Does not train or rewrite withdrawals.

    The cell report is written only when every planned run is complete. A
    summary status of ``preempted`` is not complete. An incomplete cell drops
    ``report.json`` and ``report.md`` so a stale PASS does not remain.
    """
    started = time.perf_counter()
    out = Path(out_dir)
    audit_dir = Path(config["audit_dir"]) if config.get("audit_dir") else out / "audit"
    record_path = out / "withdrawals.json"
    if not record_path.exists():
        raise ValueError(f"missing withdrawals.json: {record_path}")
    record = read_json(record_path)
    analysis = dict(config.get("analysis") or {})
    study = dict(config["study"])
    early_enabled, early_steps, early_seed = _early_settings(analysis)
    planned = planned_seeds(study)
    steps = int(study["train"]["steps"])
    withdrawn = {str(name) for name in record.get("withdrawn", [])}
    missing = _missing(
        out,
        planned,
        withdrawn,
        steps=steps,
        early_enabled=early_enabled,
        early_steps=early_steps,
        early_seed=early_seed,
    )
    _require_confirmatory_pools(config, audit_dir)
    if missing:
        for name in ("report.json", "report.md"):
            path = out / name
            if path.exists():
                path.unlink()
        return {
            "summary": "FAIL",
            "ok": False,
            "reason": "planned runs incomplete: " + ", ".join(missing),
            "elapsed_sec": time.perf_counter() - started,
            "withdrawals": record,
            "planned_seeds": planned,
            "analysis": None,
            "report_written": False,
            "config": dict(config),
            "environment": environment_report(),
        }
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


def _check_selector(
    only: str | None,
    arm: str | None,
    seed: int | None,
    early: bool,
    variant: str | None = None,
) -> None:
    if only not in (None, "audit", "train"):
        raise ValueError(f"only must be 'audit', 'train', or omitted, got {only!r}")
    if only != "train" and (arm is not None or seed is not None or early or variant is not None):
        raise ValueError("arm, seed, early, and variant are only valid with only='train'")
    if only == "train" and (arm is None or seed is None):
        raise ValueError("only='train' requires arm and seed")
    if variant is not None and early:
        raise ValueError("variant and early cannot be combined")


def _early_settings(analysis: Mapping[str, Any]) -> tuple[bool, int, int]:
    hypotheses = bool(analysis.get("hypotheses", True))
    early_cfg = dict(analysis.get("early") or {})
    enabled = bool(early_cfg.get("enabled", hypotheses))
    return enabled, int(early_cfg.get("steps", 25)), int(early_cfg.get("seed", 99))


def _check_final_step(analysis: Mapping[str, Any], study: Mapping[str, Any]) -> int:
    steps = int(study["train"]["steps"])
    final_step = int(analysis["final_step"])
    if final_step != steps:
        raise ValueError(
            f"analysis.final_step ({final_step}) must equal study.train.steps ({steps})"
        )
    return steps


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _check_chat_template_kwargs(audit: Mapping[str, Any], study: Mapping[str, Any]) -> None:
    """Refuse when the four templating sites (and generator kwargs, if set) differ."""
    generator = dict(audit.get("generator") or {})
    features = dict(audit.get("features") or {})
    train = dict(study.get("train") or {})
    eval_kwargs = dict((study.get("eval_generator") or {}).get("kwargs") or {})
    found: dict[str, Any] = {
        "audit.generator.chat_template_kwargs": generator.get("chat_template_kwargs"),
        "audit.features.chat_template_kwargs": features.get("chat_template_kwargs"),
        "study.train.chat_template_kwargs": train.get("chat_template_kwargs"),
        "study.eval_generator.kwargs.chat_template_kwargs": eval_kwargs.get("chat_template_kwargs"),
    }
    gen_kwargs = dict(generator.get("kwargs") or {})
    if "chat_template_kwargs" in gen_kwargs:
        found["audit.generator.kwargs.chat_template_kwargs"] = gen_kwargs["chat_template_kwargs"]
    frozen: dict[str, Any] = {}
    for key, value in found.items():
        if value is None:
            frozen[key] = None
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"{key} must be a mapping or null")
        frozen[key] = _freeze(value)
    if len(set(frozen.values())) > 1:
        detail = ", ".join(f"{key}={found[key]!r}" for key in found)
        raise ValueError(f"chat_template_kwargs differ across the study: {detail}")


def _require_confirmatory_pools(config: Mapping[str, Any], audit_dir: Path) -> None:
    """Refuse a development-pool audit unless this config is a pilot.

    ``require_confirmatory_pools`` is imported lazily. A build that does not
    have the helper yet leaves the check for a later revision.
    """
    if config.get("pilot") is True:
        return
    try:
        from vfates.audit import require_confirmatory_pools
    except ImportError:
        return
    require_confirmatory_pools(audit_dir)


def _check_hardware(study: Mapping[str, Any]) -> None:
    expected = study.get("hardware_class")
    if expected is None or str(expected).strip() == "":
        return
    from vfates.utils.env import hardware_class

    expected_text = str(expected)
    current = str(hardware_class())
    if current == expected_text:
        return
    if os.environ.get("VFATES_ALLOW_HARDWARE_MISMATCH") == "1":
        return
    raise ValueError(
        f"hardware_class is {current} but this cell requires {expected_text}. "
        "Set VFATES_ALLOW_HARDWARE_MISMATCH=1 to run anyway."
    )


def _call_train_arm(
    audit_dir: Path,
    arm: str,
    train_cfg: Mapping[str, Any],
    run_dir: Path,
    seed: int,
    *,
    study: Mapping[str, Any],
    backend: str,
    resume: bool,
) -> Any:
    """Call ``train_arm`` and pass ``hardware_class`` by keyword when it is accepted.

    The backends owner adds that keyword. Until the signature has it (or
    ``**kwargs``), the class still rides on the train mapping, which
    ``GRPOConfig`` already stores.
    """
    from vfates.experiment import train_arm

    hardware = study.get("hardware_class")
    body = {**dict(train_cfg), "seed": int(seed)}
    if hardware is not None:
        body["hardware_class"] = hardware
    kwargs: dict[str, Any] = {
        "backend": backend,
        "resume": resume,
        "max_train": study.get("max_train"),
        "max_heldout": study.get("max_heldout"),
        "eval_generator": study.get("eval_generator"),
    }
    params = inspect.signature(train_arm).parameters
    accepts = "hardware_class" in params or any(
        item.kind is inspect.Parameter.VAR_KEYWORD for item in params.values()
    )
    if accepts:
        kwargs["hardware_class"] = hardware
    return train_arm(audit_dir, arm, body, run_dir, **kwargs)


def _run_one_train(
    config: Mapping[str, Any],
    out: Path,
    audit_dir: Path,
    study: Mapping[str, Any],
    analysis: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    early: bool,
    started: float,
    variant: str | None = None,
) -> dict[str, Any]:
    if not audit_dir.is_dir():
        raise ValueError(f"audit directory is missing: {audit_dir}")
    _verify_lock(audit_dir)
    _check_loss_agg(audit_dir, study)
    steps = _check_final_step(analysis, study)
    early_enabled, early_steps, early_seed = _early_settings(analysis)
    planned = planned_seeds(study)
    if arm not in planned:
        raise ValueError(f"arm {arm!r} is not in the planned menu")
    if variant is not None:
        row = _variant_row(study, variant)
        if arm not in {str(name) for name in row.get("arms") or []}:
            raise ValueError(f"arm {arm!r} is not in variant {variant}")
        if seed not in {int(item) for item in row.get("seeds") or []}:
            raise ValueError(f"seed {seed} is not in variant {variant}")
        overrides = dict(row.get("train_overrides") or {})
        train_cfg = {**dict(study["train"]), **overrides}
        run_steps = int(train_cfg["steps"])
        run_dir = out / "variants" / variant / f"{arm}__s{seed}"
    elif early:
        if not early_enabled:
            raise ValueError("early baselines are disabled for this cell")
        if seed != early_seed:
            raise ValueError(f"early seed is {early_seed}, got {seed}")
        run_steps = early_steps
        train_cfg = {**dict(study["train"]), "steps": early_steps}
        run_dir = _run_dir(out, arm, seed, early=True)
    else:
        if seed not in planned[arm]:
            raise ValueError(f"seed {seed} is not planned for {arm}")
        run_steps = steps
        train_cfg = study["train"]
        run_dir = _run_dir(out, arm, seed)
    record_path = out / "withdrawals.json"
    if record_path.exists():
        record = read_json(record_path)
    else:
        record = _write_audit_withdrawals(audit_dir, analysis, out, study.get("arms"))
    backend = str(study.get("backend", "trl"))
    withdrawn = {str(name) for name in record.get("withdrawn", [])}
    if arm not in withdrawn:
        outcome = _train_with_retry(
            audit_dir,
            arm,
            seed,
            train_cfg,
            run_dir,
            run_steps,
            study=study,
            backend=backend,
        )
        if outcome == "withdraw":
            _withdraw_training(out, record, arm, seed, "non_finite_loss")
    if record_path.exists():
        record = read_json(record_path)
    withdrawn = {str(name) for name in record.get("withdrawn", [])}
    if arm not in withdrawn and _summary_status(run_dir) == "preempted":
        return {
            "summary": "PREEMPTED",
            "ok": False,
            "reason": f"{arm} seed {seed} was preempted",
            "elapsed_sec": time.perf_counter() - started,
            "only": "train",
            "arm": arm,
            "seed": seed,
            "early": early,
            "report_written": False,
            "withdrawals": record,
            "planned_seeds": planned,
            "analysis": None,
            "config": dict(config),
            "environment": environment_report(),
        }
    missing = _missing(
        out,
        planned,
        withdrawn,
        steps=steps,
        early_enabled=early_enabled,
        early_steps=early_steps,
        early_seed=early_seed,
    )
    if missing:
        label = "WITHDRAWN" if arm in withdrawn else "TRAIN"
        return {
            "summary": label,
            "ok": True,
            "reason": "planned runs incomplete: " + ", ".join(missing),
            "elapsed_sec": time.perf_counter() - started,
            "only": "train",
            "arm": arm,
            "seed": seed,
            "early": early,
            "report_written": False,
            "withdrawals": record,
            "planned_seeds": planned,
            "analysis": None,
            "config": dict(config),
            "environment": environment_report(),
        }
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


def _run_dir(out: Path, arm: str, seed: int, *, early: bool = False) -> Path:
    root = "early" if early else "runs"
    return out / root / f"{arm}__s{seed}"


def _finished(run_dir: Path, steps: int) -> bool:
    path = run_dir / "summary.json"
    if not path.exists():
        return False
    summary = read_json(path)
    # A preempted trainer can still have written the last step count. The run
    # is not complete until a later attempt finishes without that status.
    if summary.get("status") == "preempted":
        return False
    return summary.get("steps_completed") == steps


def _summary_status(run_dir: Path) -> str | None:
    path = run_dir / "summary.json"
    if not path.exists():
        return None
    status = read_json(path).get("status")
    return None if status is None else str(status)


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

    resume = _partial(run_dir)
    nonfinite = _nonfinite_class()
    failures = 0
    while True:
        try:
            _call_train_arm(
                audit_dir,
                arm,
                train_cfg,
                run_dir,
                seed,
                study=study,
                backend=backend,
                resume=resume,
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
    audit_dir: Path,
    analysis: Mapping[str, Any],
    out: Path,
    menu: Sequence[str] | None = None,
) -> dict[str, Any]:
    from vfates.verifiers.calibrate import arm_validity

    withdrawal_cfg = analysis.get("withdrawal") or {}
    rate_miss_max = float(withdrawal_cfg.get("rate_miss_max", analysis.get("rate_miss_max", 0.02)))
    validity = arm_validity(read_json(audit_dir / "calibration.json"))
    forecasts_path = audit_dir / "forecasts.json"
    forecasts = read_json(forecasts_path) if forecasts_path.is_file() else None
    _forecast_arm_names(forecasts)
    record = decide_audit_withdrawals(
        validity, rate_miss_max=rate_miss_max, forecasts=forecasts, menu=menu
    )
    write_json(out / "withdrawals.json", record)
    return record


def _forecast_arm_names(forecasts: Mapping[str, Any] | None) -> list[str]:
    """Arm names in a forecasts document, skipping non-arm keys such as ``patch_selection``."""
    if not isinstance(forecasts, Mapping):
        return []
    from vfates.audit import iter_forecast_arms

    names = [name for name, _entry in iter_forecast_arms(forecasts)]
    if "patch_selection" in names:
        raise ValueError("patch_selection was treated as a forecast arm")
    return names


def _variant_row(study: Mapping[str, Any], name: str) -> dict[str, Any]:
    for row in study.get("variants") or []:
        if str(row.get("name")) == name:
            return dict(row)
    raise ValueError(f"unknown variant {name!r}")


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
