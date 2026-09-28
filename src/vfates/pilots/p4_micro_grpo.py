"""P4: micro-GRPO on each E1 domain at qwen3-1.7b.

The gate is P4.1, one check per domain. On the development pool, the oracle
arm of qwen3-1.7b raises held-out oracle accuracy by at least 0.10 over its
step-0 accuracy within 150 steps. Group size, prompts per step, learning
rate, LoRA A (from the pilot audit) and the token budget stay on the E1
protocol. A miss is rescreened once on the backup source after P2 has
screened that backup. chain_sum and qa have no backup. A second miss removes
the domain.

One-seed fate contrasts are documented. Each pair is read against the locked
dtau (sign agreement), with the power statement from
:mod:`vfates.analysis.power`. Training goes through
:func:`vfates.study.run_study` (``only='train'``) on a cell from
:func:`vfates.pilots.common.cell_config` with ``dev=True``.

``--set smoke=true`` uses Qwen3-0.6B, a few prompts and a few steps.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vfates.pilots.common import Checks, markdown_table, write_report
from vfates.utils.io import read_json, write_json, write_yaml

SCHEMA = "vfates.p4_decisions/1"
ORACLE_GAIN = 0.10
STEPS = 150
E1_CELLS: dict[str, str] = {
    "chain_sum": "e1-chain_sum-qwen3-1.7b",
    "math": "e1-math-qwen3-1.7b",
    "code": "e1-code-qwen3-1.7b",
    "ifeval": "e1-ifeval-qwen3-1.7b",
    "qa": "e1-qa-qwen3-1.7b",
}
# (arm, control). dtau and dacc are arm minus control.
KIT_PAIRS: tuple[tuple[str, str], ...] = (
    ("fp_trigger", "noise_fp_trigger"),
    ("fn_behavior", "noise_fn"),
    ("fp_hash", "noise_fp"),
)
NATURAL_PAIR = ("natural", "noise_natural")
MATH_STRICT_PAIR = ("natural_strict", "noise_natural_strict")


def fate_pairs(domain: str) -> list[tuple[str, str]]:
    """Kit pairs, plus natural where the domain has that verifier."""
    pairs = list(KIT_PAIRS)
    if domain != "chain_sum":
        pairs.append(NATURAL_PAIR)
    if domain == "math":
        pairs.append(MATH_STRICT_PAIR)
    return pairs


def backup_overrides(domain: str) -> dict[str, Any] | None:
    """Dotted task overrides for the one backup source, or None.

    Math uses DeepScaleR, ifeval uses RLVR-IFeval. chain_sum, code and qa
    have none.
    """
    if domain == "math":
        from vfates.tasks.math import DEEPSCALER_DATASET, DEEPSCALER_REVISION

        return {
            "audit.task.source": "deepscaler",
            "audit.task.dataset": DEEPSCALER_DATASET,
            "audit.task.revision": DEEPSCALER_REVISION,
        }
    if domain == "ifeval":
        from vfates.tasks.ifeval import DATASET_RLVR, SOURCE_RLVR

        return {
            "audit.task.source": SOURCE_RLVR,
            "audit.task.dataset": DATASET_RLVR,
            "audit.task.revision": None,
            "audit.task.constraint_counts": None,
        }
    return None


def decide_domain(
    gain: float | None,
    *,
    has_backup: bool,
    backup_gain: float | None = None,
    backup_ready: bool = False,
    threshold: float = ORACLE_GAIN,
) -> dict[str, Any]:
    """P4.1 for one domain.

    ``gain`` is the primary source. A passing primary does not use the
    backup. A measured failure with no backup leaves the program. A missing
    gain is an incomplete run and does not leave. A failing primary whose
    backup P2 has not screened yet does not leave and does not claim the
    backup was used. A screened backup that also misses leaves.
    """
    if gain is not None and float(gain) >= float(threshold):
        return _decision(True, gain, False, False)
    # No measured primary gain is an incomplete run, not a scientific removal.
    if gain is None and not (has_backup and backup_ready and backup_gain is not None):
        return _decision(False, None, False, False)
    if not has_backup:
        return _decision(False, gain, False, True)
    if not backup_ready or backup_gain is None:
        return _decision(False, gain, False, False)
    passed = float(backup_gain) >= float(threshold)
    return _decision(passed, backup_gain, True, not passed)


def _decision(passed: bool, gain: float | None, backup_used: bool, leaves: bool) -> dict[str, Any]:
    return {
        "pass": bool(passed),
        "gain": None if gain is None else float(gain),
        "backup_used": bool(backup_used),
        "leaves_program": bool(leaves),
    }


def sign_agreement(dacc: float | None, dtau: float | None) -> bool | None:
    """True when the accuracy gap and the locked tau gap share a sign.

    A zero gap has no sign, so the comparison is undefined (``None``).
    """
    if dacc is None or dtau is None:
        return None
    if float(dacc) == 0.0 or float(dtau) == 0.0:
        return None
    return (float(dacc) > 0.0) == (float(dtau) > 0.0)


def power_statement(
    *,
    n_prompts: int,
    n_seeds: int = 1,
    n_samples: int = 4,
    seed_sd: float = 0.0,
    margin: float = 0.05,
) -> dict[str, Any]:
    """One-seed power of a paired contrast, from :mod:`vfates.analysis.power`."""
    from vfates.analysis.power import minimum_detectable_difference

    return minimum_detectable_difference(
        seed_sd,
        n_seeds=n_seeds,
        n_prompts=n_prompts,
        n_samples=n_samples,
        margin=margin,
    )


def apply_dotted(config: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Set dotted keys on a nested dict. Returns the same object."""
    for key, value in overrides.items():
        node = config
        parts = str(key).split(".")
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = value
    return config


def _dig(config: Mapping[str, Any], dotted: str) -> Any:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def load_pilot_cell(
    cell_id: str,
    *,
    dev: bool = True,
    overrides: Mapping[str, Any] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Load a rendered cell and apply dotted overrides.

    ``cell_config`` is owned by the P1 to P3 work. This applies the overrides
    again so a cell whose loader has not folded them in still matches the
    study this pilot launches.
    """
    import vfates.pilots.common as common

    loader = getattr(common, "cell_config", None)
    if loader is None:
        raise AttributeError(
            "vfates.pilots.common.cell_config(cell_id, *, dev=True, overrides=None, root=None) "
            "-> dict is not defined"
        )
    loaded = loader(cell_id, dev=dev, overrides=dict(overrides) if overrides else None, root=root)
    cell = copy.deepcopy(dict(loaded))
    dotted = dict(overrides or {})
    if dev:
        dotted["audit.pools.from"] = "dev"
    apply_dotted(cell, dotted)
    return cell


def execute_study(
    config: Mapping[str, Any],
    out_dir: str | Path,
    *,
    only: str,
    arm: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """One study job. ``only='train'`` is the P4 training path."""
    from vfates.study import run_study

    return run_study(config, out_dir, only=only, arm=arm, seed=seed)


def smoke_backend() -> str:
    """MLX when it imports, otherwise the HF trainer."""
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        return "hf"
    return "mlx"


def protocol_overrides(
    *,
    steps: int,
    seed: int,
    arms: Sequence[str],
    smoke: bool,
    hardware: str,
) -> dict[str, Any]:
    """Dotted overrides that keep the E1 protocol and shorten a smoke run."""
    overrides: dict[str, Any] = {
        "audit.pools.from": "dev",
        "study.train.steps": int(steps),
        "study.seeds": [int(seed)],
        "study.arms": list(arms),
        "analysis.final_step": int(steps),
        "analysis.early.enabled": False,
    }
    if not smoke:
        return overrides
    from vfates.models import load_model_spec

    spec = load_model_spec("qwen3-0.6b")
    kwargs = dict(spec.chat_template_kwargs or {})
    overrides.update(
        {
            "study.train.steps": int(steps),
            "analysis.final_step": int(steps),
            "study.train.model_id": spec.model_id,
            "study.train.model_revision": spec.revision,
            "study.train.chat_template_kwargs": kwargs,
            "study.train.lr": float(spec.lr),
            "audit.generator.model_id": spec.model_id,
            "audit.generator.revision": spec.revision,
            "audit.generator.chat_template_kwargs": kwargs,
            "audit.features.model_id": spec.model_id,
            "audit.features.chat_template_kwargs": kwargs,
            "study.train.prompts_per_step": 2,
            "study.train.group_size": 2,
            "study.train.max_tokens": 64,
            "study.train.eval_max_tokens": 64,
            "study.train.eval_every": 1,
            "study.train.save_every": 1,
            "study.train.eval_heldout_n": 2,
            "study.train.eval_train_n": 2,
            "study.train.eval_samples_per_prompt": 1,
            "study.train.use_vllm": False,
            "study.backend": smoke_backend(),
            "study.hardware_class": hardware,
            "study.eval_generator.backend": "hf",
            "audit.pools.train": 4,
            "audit.pools.heldout": 2,
            "audit.sampling.n": 2,
            "audit.sampling.max_tokens": 64,
        }
    )
    try:
        place = spec.placement_for(hardware)
    except ValueError:
        place = None
    if place is not None:
        overrides["study.gpus"] = place.gpus
        overrides["study.train.micro_batch_size"] = place.micro_batch_size
        overrides["audit.generator.kwargs.tensor_parallel_size"] = place.tp
        overrides["audit.generator.kwargs.gpu_memory_utilization"] = (
            place.vllm_gpu_memory_utilization
        )
    return overrides


def heldout_gain(run_dir: str | Path, steps: int) -> float | None:
    """Held-out oracle accuracy at ``steps`` minus the step-0 accuracy."""
    from vfates.analysis.outcomes import run_outcome

    path = Path(run_dir)
    if not (path / "eval.jsonl").is_file():
        return None
    try:
        row = run_outcome(path, final_step=steps)
    except (OSError, ValueError):
        return None
    delta = row.get("heldout_delta")
    if delta is None:
        return None
    return float(delta)


def heldout_accuracy(run_dir: str | Path, steps: int) -> float | None:
    """Held-out oracle accuracy at ``steps``."""
    from vfates.analysis.outcomes import run_outcome

    path = Path(run_dir)
    if not (path / "eval.jsonl").is_file():
        return None
    try:
        row = run_outcome(path, final_step=steps)
    except (OSError, ValueError):
        return None
    value = row.get("heldout_final")
    return None if value is None else float(value)


def arm_tau(audit_dir: str | Path, arm: str) -> float | None:
    """Locked ``tau_pool`` for ``arm``, else the audit forecasts."""
    from vfates.audit import iter_forecast_arms

    root = Path(audit_dir)
    forecasts: Mapping[str, Any] | None = None
    lock = root / "lock.json"
    if lock.is_file():
        try:
            from vfates.forecast.lockbox import load_lock

            loaded = load_lock(lock)
            if isinstance(loaded, Mapping):
                forecasts = loaded
        except (OSError, ValueError):
            forecasts = None
    if forecasts is None and (root / "forecasts.json").is_file():
        loaded = read_json(root / "forecasts.json")
        if isinstance(loaded, Mapping):
            forecasts = loaded
    if forecasts is None:
        return None
    for name, entry in iter_forecast_arms(forecasts):
        if name != arm:
            continue
        transfer = entry.get("transfer") if isinstance(entry, Mapping) else None
        if not isinstance(transfer, Mapping):
            return None
        value = transfer.get("tau_pool")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)
    return None


def _study_run(out: Path, arm: str, seed: int) -> Path:
    return out / "runs" / f"{arm}__s{seed}"


def _audit_ready(out: Path) -> bool:
    audit = out / "audit"
    return (audit / "lock.json").is_file() or (audit / "forecasts.json").is_file()


def _train_source(
    cell: Mapping[str, Any],
    out: Path,
    arms: Sequence[str],
    seed: int,
    steps: int,
) -> dict[str, float | None]:
    """Audit once, then ``only='train'`` each arm. Return held-out gains."""
    out.mkdir(parents=True, exist_ok=True)
    write_yaml(out / "cell.yaml", dict(cell))
    if not _audit_ready(out):
        execute_study(cell, out, only="audit")
    gains: dict[str, float | None] = {}
    for arm in arms:
        run_dir = _study_run(out, arm, seed)
        summary = run_dir / "summary.json"
        done = summary.is_file() and read_json(summary).get("steps_completed") == steps
        if not done:
            execute_study(cell, out, only="train", arm=arm, seed=seed)
        gains[arm] = heldout_gain(run_dir, steps)
    return gains


def _config_edits(decisions: Mapping[str, Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for domain, row in decisions.items():
        if row.get("leaves_program"):
            lines.append(
                f"Remove every cell whose domain is {domain} from configs/program.yaml. "
                "P4.1 missed, and that domain has no backup left."
            )
        elif row.get("backup_used") and row.get("pass"):
            lines.append(
                f"Point the {domain} template audit.task at the backup source. "
                "P4.1 passed on that source."
            )
    if not lines:
        return "No template edits. Every trained domain passed P4.1 on its primary source."
    return "\n".join(f"- {line}" for line in lines)


def run_program(config: Mapping[str, Any], out_dir: str | Path) -> dict[str, Any]:
    """P4 on the E1 cells. Writes ``decisions.json`` and the report."""
    from vfates.utils.env import hardware_class

    started = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    smoke = bool(config.get("smoke", False))
    steps = int(config.get("smoke_steps", 2) if smoke else config.get("steps", STEPS))
    threshold = float(config.get("oracle_gain", ORACLE_GAIN))
    seed = int(config.get("seed", 0))
    train_fates = False if smoke else bool(config.get("train_fates", True))
    domains = [str(name) for name in (config.get("domains") or list(E1_CELLS))]
    cells = {**E1_CELLS, **{str(k): str(v) for k, v in dict(config.get("cells") or {}).items()}}
    screened = {str(name) for name in (config.get("backup_screened") or [])}
    hardware = hardware_class()
    checks = Checks()
    decisions: dict[str, dict[str, Any]] = {}
    fate_table: list[list[Any]] = []
    power_rows: list[list[Any]] = []
    errors: list[str] = []

    for domain in domains:
        cell_id = cells.get(domain)
        if cell_id is None:
            decisions[domain] = _decision(False, None, False, False)
            checks.add(
                f"P4.1.{domain}",
                f"{domain} oracle held-out gain on the dev pool",
                None,
                False,
                threshold=threshold,
                metric="no cell id",
                documented=smoke,
            )
            continue
        backup = backup_overrides(domain)
        user_overrides = {
            str(key): value for key, value in dict(config.get("cell_overrides") or {}).items()
        }
        try:
            primary = _one_source(
                domain,
                cell_id,
                out / domain / "primary",
                steps=steps,
                seed=seed,
                smoke=smoke,
                hardware=hardware,
                extra=user_overrides or None,
                train_fates=False,
            )
        except Exception as exc:
            errors.append(f"{domain}: {type(exc).__name__}: {exc}")
            decisions[domain] = _decision(False, None, False, False)
            checks.add(
                f"P4.1.{domain}",
                f"{domain} oracle held-out gain on the dev pool",
                None,
                False,
                threshold=threshold,
                metric=f"{type(exc).__name__}: {exc}",
                documented=smoke,
            )
            continue
        gain = primary["gains"].get("oracle")
        backup_gain: float | None = None
        ready = domain in screened
        if (
            not smoke
            and backup is not None
            and gain is not None
            and float(gain) < threshold
            and ready
        ):
            try:
                second = _one_source(
                    domain,
                    cell_id,
                    out / domain / "backup",
                    steps=steps,
                    seed=seed,
                    smoke=False,
                    hardware=hardware,
                    extra={**user_overrides, **backup},
                    train_fates=False,
                )
            except Exception as exc:
                errors.append(f"{domain} backup: {type(exc).__name__}: {exc}")
                second = {"gains": {}, "cell": primary["cell"], "out": primary["out"]}
            backup_gain = second["gains"].get("oracle")
            if backup_gain is not None and float(backup_gain) >= threshold:
                primary = second
        row = decide_domain(
            gain,
            has_backup=backup is not None,
            backup_gain=backup_gain,
            backup_ready=ready,
            threshold=threshold,
        )
        if smoke:
            # The 0.10 gate is documented on a smoke run, and a short run does not remove a domain.
            measured_pass = gain is not None and float(gain) >= threshold
            row = _decision(measured_pass, gain, False, False)
        decisions[domain] = row
        awaiting = (
            backup is not None
            and gain is not None
            and float(gain) < threshold
            and not ready
            and not smoke
        )
        metric = (
            f"backup_used={'yes' if row['backup_used'] else 'no'}; "
            f"leaves_program={'yes' if row['leaves_program'] else 'no'}"
        )
        if awaiting:
            metric += "; awaiting P2 backup screen"
        checks.add(
            f"P4.1.{domain}",
            f"{domain} oracle held-out gain on the dev pool",
            row["gain"],
            bool(row["pass"]) and row["gain"] is not None and float(row["gain"]) >= threshold,
            threshold=threshold,
            metric=metric,
            documented=smoke,
        )
        if train_fates:
            _fate_block(
                checks,
                domain,
                primary,
                steps=steps,
                seed=seed,
                smoke=False,
                hardware=hardware,
                fate_table=fate_table,
                power_rows=power_rows,
            )

    if not train_fates:
        checks.add(
            "P4.2",
            "one-seed fate contrasts against locked dtau",
            None,
            True,
            metric="not trained on this run",
            documented=True,
        )
    write_json(out / "decisions.json", {"schema": SCHEMA, "domains": decisions})
    sections = [
        (
            "Decisions",
            markdown_table(
                ["domain", "pass", "gain", "backup_used", "leaves_program"],
                [
                    [
                        name,
                        row["pass"],
                        row["gain"],
                        row["backup_used"],
                        row["leaves_program"],
                    ]
                    for name, row in decisions.items()
                ],
            ),
        ),
        ("Config edits", _config_edits(decisions)),
    ]
    if fate_table:
        sections.append(
            (
                "Fate contrasts (documented)",
                markdown_table(
                    ["domain", "pair", "dacc", "dtau", "sign_agreement"],
                    fate_table,
                ),
            )
        )
    if power_rows:
        sections.append(
            (
                "Power at one seed",
                markdown_table(
                    ["domain", "n_prompts", "se", "mdd", "power_at_0.05", "tost_power_at_0"],
                    power_rows,
                ),
            )
        )
    if errors:
        sections.append(("Run errors", "\n".join(errors)))
    return write_report(
        out,
        str(config.get("title", "P4 micro-GRPO")),
        checks,
        config=dict(config),
        started=started,
        tables={"decisions": decisions, "hardware_class": hardware, "schema": SCHEMA},
        sections=sections,
        intro=(
            f"Hardware class {hardware}. P4.1 is the only gate "
            f"(oracle gain >= {threshold} within {steps} steps on the dev pool). "
            "Fate contrasts are one seed and are documented."
        ),
    )


def _one_source(
    domain: str,
    cell_id: str,
    out: Path,
    *,
    steps: int,
    seed: int,
    smoke: bool,
    hardware: str,
    extra: Mapping[str, Any] | None,
    train_fates: bool,
) -> dict[str, Any]:
    arms = ["oracle"]
    if train_fates:
        for arm, control in fate_pairs(domain):
            arms.extend((arm, control))
    arms = list(dict.fromkeys(arms))
    overrides = protocol_overrides(
        steps=steps, seed=seed, arms=arms, smoke=smoke, hardware=hardware
    )
    if extra:
        overrides.update(dict(extra))
    cell = load_pilot_cell(cell_id, dev=True, overrides=overrides)
    if _dig(cell, "audit.pools.from") != "dev":
        raise RuntimeError("audit.pools.from is not dev after cell_config")
    if int(_dig(cell, "study.train.steps")) != int(steps):
        raise RuntimeError(
            f"study.train.steps is {_dig(cell, 'study.train.steps')!r}, expected {steps}"
        )
    gains = _train_source(cell, out, ["oracle"], seed, steps)
    return {"gains": gains, "cell": cell, "out": out, "arms": arms}


def _fate_block(
    checks: Checks,
    domain: str,
    source: Mapping[str, Any],
    *,
    steps: int,
    seed: int,
    smoke: bool,
    hardware: str,
    fate_table: list[list[Any]],
    power_rows: list[list[Any]],
) -> None:
    del smoke, hardware
    cell = source["cell"]
    out = Path(source["out"])
    pairs = fate_pairs(domain)
    arms = list(dict.fromkeys(arm for pair in pairs for arm in pair))
    present = [str(name) for name in (_dig(cell, "study.arms") or [])]
    missing = [arm for arm in arms if arm not in present]
    if missing:
        apply_dotted(cell, {"study.arms": list(dict.fromkeys([*present, *arms]))})
        write_yaml(out / "cell.yaml", dict(cell))
    for arm in arms:
        run_dir = _study_run(out, arm, seed)
        summary = run_dir / "summary.json"
        done = summary.is_file() and read_json(summary).get("steps_completed") == steps
        if not done:
            execute_study(cell, out, only="train", arm=arm, seed=seed)
    n_prompts = int(_dig(cell, "audit.pools.heldout") or 384)
    n_samples = int(_dig(cell, "study.train.eval_samples_per_prompt") or 4)
    statement = power_statement(n_prompts=n_prompts, n_seeds=1, n_samples=n_samples)
    power_rows.append(
        [
            domain,
            n_prompts,
            statement["se"],
            statement["mdd"],
            statement["power_at_0.05"],
            statement["tost_power_at_0"],
        ]
    )
    checks.add(
        f"P4.3.{domain}",
        f"{domain} one-seed power of a paired contrast (documented)",
        float(statement["power_at_0.05"]),
        True,
        threshold="documented",
        metric=(
            f"n_prompts={n_prompts}, n_seeds=1, se={statement['se']:.4f}, "
            f"mdd={statement['mdd']:.4f}, tost_power_at_0={statement['tost_power_at_0']:.4f}"
        ),
        documented=True,
    )
    audit_dir = out / "audit"
    for arm, control in pairs:
        dacc_arm = heldout_accuracy(_study_run(out, arm, seed), steps)
        dacc_control = heldout_accuracy(_study_run(out, control, seed), steps)
        dacc = None if dacc_arm is None or dacc_control is None else dacc_arm - dacc_control
        tau_arm = arm_tau(audit_dir, arm)
        tau_control = arm_tau(audit_dir, control)
        dtau = None if tau_arm is None or tau_control is None else tau_arm - tau_control
        agreed = sign_agreement(dacc, dtau)
        fate_table.append([domain, f"{arm} - {control}", dacc, dtau, agreed])
        checks.add(
            f"P4.2.{domain}.{arm}",
            f"{domain} sign({arm} - {control}) agrees with locked dtau",
            dacc,
            agreed is True,
            threshold="same sign as dtau",
            metric=f"dacc={dacc}, dtau={dtau}, agreement={agreed}",
            documented=True,
        )


def run(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """P4 micro-GRPO: oracle gain on each E1 domain, through the study."""
    return run_program(config, out_dir)
