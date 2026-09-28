"""P1: prove every stage computes what it claims before a study GPU is spent.

The default run is structural. It does not load a model: it checks the study's
refusals, a SIGUSR1 preemption flag, a two-cell program through a local
worker, the pooled report, and a confirmatory report. ``live: true`` adds a
Qwen/Qwen3-0.6B audit of each domain on the development pool, chain_sum
log-probability and LoRA parity checks, and a GRPO run on chain_sum that
resumes one step past the first stop. Feature passes in that path use CPU
so the Mac run does not require CUDA. SIGUSR1 is observed on the preemption
callback and is not sent into that trainer.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

from vfates.pilots.common import Checks, write_report
from vfates.utils.io import canonical_json, read_json, sha256_file, write_json

_STAGE_NAMES = ("problems", "completions", "calibration", "table", "features", "forecasts")


def _pytest(root: Path, marker: str, timeout: int) -> tuple[int, str]:
    cmd = [sys.executable, "-m", "pytest", "-q", "-m", marker, "-p", "no:cacheprovider"]
    proc = subprocess.run(
        [*cmd, str(root / "tests")],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-300:]
    return proc.returncode, tail


def _generator_identity(n: int) -> tuple[bool | None, int]:
    try:
        import reasoning_gym  # noqa: F401
    except ImportError:
        return None, 0
    from vfates.tasks.chain_sum import ChainSumConfig, _items_from_reasoning_gym, builtin_items

    cfg = ChainSumConfig()
    compared = 0
    for seed in (0, cfg.heldout_seed_offset):
        ours = builtin_items(cfg, seed, n)
        theirs = _items_from_reasoning_gym(cfg, seed, n)
        for a, b in zip(ours, theirs, strict=True):
            compared += 1
            if (a["question"], a["answer"]) != (b["question"], b["answer"]):
                return False, compared
    return True, compared


def _logprob_agreement(model_id: str, n_prompts: int, max_tokens: int) -> dict[str, Any]:
    """Sample with MLX, rescore the same token ids with PyTorch, compare."""
    import torch

    from vfates.features.hidden import load_model
    from vfates.gen.chat import encode_completion, encode_prompt
    from vfates.gen.mlx_backend import MLXGenerator
    from vfates.tasks.chain_sum import ChainSumTask

    task = ChainSumTask()
    problems = task.problems("dev", n_prompts, 123)
    gen = MLXGenerator(model_id, completion_batch_size=32, logprobs=True)
    groups = gen.generate([task.prompt(p) for p in problems], 2, max_tokens=max_tokens, seed=7)
    del gen
    model, tok = load_model(model_id, dtype="float32")
    device = next(model.parameters()).device
    per_token, roundtrip = [], []
    with torch.no_grad():
        for problem, group in zip(problems, groups, strict=True):
            prompt_ids = encode_prompt(tok, task.prompt(problem))
            for out in group:
                ids = list(out.token_ids or ())
                if not ids:
                    continue
                full = torch.tensor([prompt_ids + ids], device=device)
                logits = model(full).logits[0, len(prompt_ids) - 1 : -1].float()
                lp = torch.log_softmax(logits, -1).gather(-1, full[0, len(prompt_ids) :, None])
                hf = float(lp.sum())
                per_token.append(abs(hf - float(cast(float, out.logprob))) / len(ids))
                roundtrip.append(encode_completion(tok, out.text) == ids)
    del model
    return {
        "n": len(per_token),
        "per_token_abs_diff_median": float(np.median(per_token)),
        "per_token_abs_diff_p95": float(np.quantile(per_token, 0.95)),
        "roundtrip_rate": float(np.mean(roundtrip)),
    }


def _lora_parity(model_id: str, train_cfg: dict[str, Any]) -> dict[str, Any]:
    """Adapted (layer, projection) sets and scales under MLX and PEFT."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.tuner.lora import LoRALinear
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from peft import LoraConfig, get_peft_model

    from vfates.features.hidden import load_model
    from vfates.train.config import GRPOConfig

    cfg = GRPOConfig.from_dict({**train_cfg, "model_id": model_id, "out_dir": "/tmp/unused"})
    model, _ = load(model_id)
    model.freeze()
    linear_to_lora_layers(
        model,
        cfg.lora.num_layers,
        {
            "rank": cfg.lora.rank,
            "scale": cfg.lora.scale,
            "dropout": 0.0,
            "keys": cfg.lora.mlx_keys(),
        },
    )
    pat = re.compile(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)$")
    mlx_set = set()
    mlx_scales = set()
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            m = pat.search(name)
            if m:
                mlx_set.add((int(m.group(1)), m.group(2)))
                mlx_scales.add(float(module.scale))
    del model
    mx.clear_cache()
    hf, _tok = load_model(model_id, dtype="float32", device="cpu")
    peft_model = get_peft_model(
        hf,
        LoraConfig(
            r=cfg.lora.rank,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=0.0,
            target_modules=cfg.lora.peft_keys(),
            task_type="CAUSAL_LM",
        ),
    )
    peft_set = set()
    peft_scales: set[float] = set()
    pat_t = re.compile(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)$")
    for name, module in peft_model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "scaling"):
            m = pat_t.search(name)
            if m:
                peft_set.add((int(m.group(1)), m.group(2)))
                peft_scales.update(float(v) for v in module.scaling.values())
    return {
        "mlx_modules": len(mlx_set),
        "peft_modules": len(peft_set),
        "same_set": mlx_set == peft_set,
        "mlx_scales": sorted(mlx_scales),
        "peft_scales": sorted(peft_scales),
    }


def _file_digest(path: Path) -> str:
    return sha256_file(path) if path.is_file() else ""


def audit_artifact_digest(audit_dir: Path) -> dict[str, str]:
    """Hashes of the audit artifacts a resume must leave unchanged."""
    return {
        "problems": _file_digest(audit_dir / "problems.jsonl"),
        "completions": _file_digest(audit_dir / "completions.jsonl"),
        "calibration": _file_digest(audit_dir / "calibration.json"),
        "verdicts": _verdict_digest(audit_dir),
        "features": _file_digest(audit_dir / "features.npz"),
        "forecasts": _file_digest(audit_dir / "forecasts.json"),
        "stages": _file_digest(audit_dir / "stages.json"),
        "lora_init": _file_digest(audit_dir / "lora_init.safetensors"),
        "lock": _file_digest(audit_dir / "lock.json"),
    }


def lock_payload_matches_forecasts(audit_dir: Path) -> bool:
    """True when the locked payload is the canonical JSON of ``forecasts.json``."""
    from vfates.audit import read_lock_record

    try:
        record = read_lock_record(audit_dir / "lock.json")
        forecasts = read_json(audit_dir / "forecasts.json")
    except (OSError, ValueError, KeyError):
        return False
    return canonical_json(record.get("payload")) == canonical_json(forecasts)


def features_used_token_ids(audit_dir: Path) -> bool:
    """True when ``features.meta.json`` records sampled token ids as the source."""
    path = audit_dir / "features.meta.json"
    if not path.is_file():
        return False
    meta = read_json(path)
    return isinstance(meta, dict) and meta.get("token_source") == "token_ids"


def fingerprint_manifest_complete(audit_dir: Path) -> bool:
    """True when ``stages.json`` has a 64-character fingerprint for every stage."""
    path = audit_dir / "stages.json"
    if not path.is_file():
        return False
    doc = read_json(path)
    stages = doc.get("stages") if isinstance(doc, dict) else None
    if not isinstance(stages, dict):
        return False
    for name in _STAGE_NAMES:
        fp = stages.get(name)
        if not isinstance(fp, str) or len(fp) != 64 or any(c not in "0123456789abcdef" for c in fp):
            return False
    return True


def lora_init_matches_lock(audit_dir: Path) -> bool:
    """True when the LoRA init file loads and its sha256 is the lock's ``lora_init``."""
    from vfates.audit import read_lock_record

    path = audit_dir / "lora_init.safetensors"
    if not path.is_file():
        return False
    try:
        record = read_lock_record(audit_dir / "lock.json")
    except (OSError, ValueError):
        return False
    artifacts = record.get("artifacts") if isinstance(record, dict) else None
    expected = artifacts.get("lora_init") if isinstance(artifacts, dict) else None
    if not isinstance(expected, str) or sha256_file(path) != expected:
        return False
    try:
        from vfates.features.lora_init import load_lora_init
    except ImportError:
        return False
    try:
        load_lora_init(path, expected_sha256=expected)
    except (OSError, ValueError):
        return False
    return True


def _verdict_digest(audit_dir: Path) -> str:
    from vfates.types import AuditTable
    from vfates.utils.io import sha256_text

    table = AuditTable.load(audit_dir / "audit.npz")
    parts = [",".join(map(str, table.O.tolist()))]
    parts += [f"{k}:{','.join(map(str, v.tolist()))}" for k, v in sorted(table.V.items())]
    return sha256_text("|".join(parts))


def stages_match_patch_policy(audit_dir: Path, *, expect_patches: bool) -> bool:
    """The six stage fingerprints, plus ``patches`` exactly when patch arms exist."""
    if not fingerprint_manifest_complete(audit_dir):
        return False
    doc = read_json(audit_dir / "stages.json")
    stages = doc.get("stages") if isinstance(doc, dict) else None
    if not isinstance(stages, dict):
        return False
    has = "patches" in stages
    if has != expect_patches:
        return False
    if not expect_patches:
        return True
    fp = stages.get("patches")
    return isinstance(fp, str) and len(fp) == 64 and all(c in "0123456789abcdef" for c in fp)


def tau_budget_draws_present(forecasts: dict[str, Any]) -> bool:
    """Every forecast arm has the four oracle-light budgets, with draws unless exact."""
    from vfates.audit import iter_forecast_arms

    keys = ("32", "64", "128", "192")
    found = False
    for _name, entry in iter_forecast_arms(forecasts):
        found = True
        block = (entry.get("transfer") or {}).get("tau_pool_budget")
        if not isinstance(block, dict) or any(key not in block for key in keys):
            return False
        for key in keys:
            row = block[key]
            if not isinstance(row, dict) or "mean" not in row:
                return False
            draws = row.get("draws")
            exact = row.get("sd") == 0 or row.get("sd") == 0.0
            if not exact and not (isinstance(draws, list) and len(draws) > 0):
                return False
    return found


def patch_selection_matches(forecasts: dict[str, Any], *, expect: bool) -> bool:
    has = isinstance(forecasts.get("patch_selection"), dict)
    return has is expect


_LOGPROB_PROMPTS = 8
_LOGPROB_MAX_TOKENS = 128
_LOGPROB_ABS_DIFF = 0.05
_ROUNDTRIP_RATE = 0.95
_TRAIN_FILES = (
    "metrics.jsonl",
    "eval.jsonl",
    "eval_prompts.jsonl",
    "summary.json",
    "config.json",
)


def run(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """P1 harness: plumbing with no model, and the 0.6B path when ``live`` is set."""
    return _run_structural(config, out_dir)


def _lock_detects_tamper(audit_dir: Path, scratch: Path) -> bool:
    """True when the chain_sum lock verifies and a one-number copy does not."""
    from vfates.forecast.lockbox import verify_lock

    tampered = scratch / "lock_tampered.json"
    doc = read_json(audit_dir / "lock.json")
    first_arm = next(iter(doc["payload"]))
    doc["payload"][first_arm]["transfer"]["tau_pool"] = 0.123456
    write_json(tampered, doc)
    return verify_lock(audit_dir / "lock.json") and not verify_lock(tampered)


def audit_stage_flags(audit_dir: Path, *, expect_patches: bool) -> dict[str, bool]:
    """Booleans the domain audit checks read. Missing files are false, not exceptions."""
    expected = [
        "problems.jsonl",
        "completions.jsonl",
        "calibration.json",
        "audit.npz",
        "confusion.json",
        "features.npz",
        "features.meta.json",
        "forecasts.json",
        "lock.json",
        "stages.json",
        "lora_init.safetensors",
    ]
    if expect_patches:
        expected.append("patches.json")
    present = all((audit_dir / name).is_file() for name in expected)
    extra_patches = (audit_dir / "patches.json").is_file() and not expect_patches
    forecasts: dict[str, Any] = {}
    forecast_path = audit_dir / "forecasts.json"
    if forecast_path.is_file():
        loaded = read_json(forecast_path)
        if isinstance(loaded, dict):
            forecasts = loaded
    pool_ok = False
    hardware_ok = False
    try:
        from vfates.audit import is_dev_audit, read_lock_record
        from vfates.utils.env import hardware_class

        record = read_lock_record(audit_dir / "lock.json")
        pool_ok = record.get("pool_source") == "dev" and is_dev_audit(audit_dir)
        hardware_ok = record.get("hardware_class") == hardware_class()
    except (OSError, ValueError, KeyError):
        pool_ok = False
        hardware_ok = False
    return {
        "artifacts": present and not extra_patches,
        "token_ids": features_used_token_ids(audit_dir),
        "stages": stages_match_patch_policy(audit_dir, expect_patches=expect_patches),
        "lock_matches": lock_payload_matches_forecasts(audit_dir),
        "lora_init": lora_init_matches_lock(audit_dir),
        "tau_budget": tau_budget_draws_present(forecasts),
        "patch_selection": patch_selection_matches(forecasts, expect=expect_patches),
        "pool_source": pool_ok,
        "hardware_class": hardware_ok,
    }


def _study_body(hardware: str, *, chat: dict[str, Any] | None = None) -> dict[str, Any]:
    kwargs = {"enable_thinking": False} if chat is None else chat
    return {
        "audit": {
            "task": {"type": "chain_sum"},
            "pools": {"from": "dev", "train": 2, "heldout": 2, "seed": 0},
            "generator": {
                "model_id": "Qwen/Qwen3-0.6B",
                "chat_template_kwargs": dict(kwargs),
            },
            "features": {
                "model_id": "Qwen/Qwen3-0.6B",
                "chat_template_kwargs": dict(kwargs),
            },
        },
        "study": {
            "hardware_class": hardware,
            "train": {
                "model_id": "Qwen/Qwen3-0.6B",
                "steps": 2,
                "chat_template_kwargs": dict(kwargs),
            },
            "eval_generator": {"kwargs": {"chat_template_kwargs": dict(kwargs)}},
        },
        "analysis": {"final_step": 2},
    }


def study_refuses_hardware_mismatch(out_dir: Path) -> bool:
    """The study raises before an audit when the hardware class does not match."""
    from vfates.study import run_study
    from vfates.utils.env import hardware_class

    current = hardware_class()
    other = "h100-80gb" if current != "h100-80gb" else "cpu"
    try:
        run_study(_study_body(other), out_dir / "hardware")
    except ValueError as exc:
        return "hardware_class" in str(exc)
    return False


def study_refuses_chat_mismatch(out_dir: Path) -> bool:
    """The study raises when chat_template_kwargs differ across the cell."""
    from vfates.study import run_study
    from vfates.utils.env import hardware_class

    body = _study_body(hardware_class())
    body["audit"]["features"]["chat_template_kwargs"] = {"enable_thinking": True}
    try:
        run_study(body, out_dir / "chat")
    except ValueError as exc:
        return "chat_template_kwargs" in str(exc)
    return False


def sigusr1_marks_preempted() -> bool:
    """SIGUSR1 sets the trainer preemption flag and does not kill the process."""
    import signal

    from vfates.train.trl_grpo import PreemptionCallback

    callback = PreemptionCallback()
    callback.install()
    try:
        os.kill(os.getpid(), signal.SIGUSR1)
        return bool(callback.preempted)
    finally:
        callback.restore()


def _forecast_arm() -> dict[str, Any]:
    budgets = {
        key: {"mean": 0.2, "sd": 0.0, "budget": int(key)} for key in ("32", "64", "128", "192")
    }
    return {
        "step0": {"J": 0.1, "fpr": 0.1, "fnr": 0.2},
        "mean_precision": 0.5,
        "modal_precision": {"value": 0.5},
        "transfer": {
            "tau_pool": 0.2,
            "tau_hard": 0.1,
            "tau_rel": 0.3,
            "reliability": 0.9,
            "cos_gV_gO": 0.4,
            "base_acc_heldout": 0.3,
            "tau_pool_budget": budgets,
        },
    }


def _write_eval(dest: Path) -> None:
    from vfates.utils.io import write_jsonl

    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    prompts = []
    for split in ("heldout", "train"):
        for step, acc in ((0, 0.2), (2, 0.5)):
            rows.append(
                {
                    "step": step,
                    "split": split,
                    "oracle_acc": acc,
                    "J": 0.1,
                    "fpr": 0.1,
                    "fnr": 0.2,
                }
            )
            prompts.append({"step": step, "split": split, "acc": {"p0": acc}})
    write_jsonl(dest / "eval.jsonl", rows)
    write_jsonl(dest / "eval_prompts.jsonl", prompts)
    write_json(dest / "summary.json", {"status": "complete", "steps_completed": 2})


def confirmatory_well_formed(root: Path) -> dict[str, Any]:
    """A one-arm report. Families are empty, so the decisions are not a positive claim."""
    from vfates.analysis.confirmatory import confirmatory_report
    from vfates.forecast.lockbox import lock_forecasts

    audit = root / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    payload = {"oracle": _forecast_arm()}
    write_json(audit / "forecasts.json", payload)
    lock_forecasts(payload, audit / "lock.json", pool_source="dev")
    run_dir = root / "runs" / "oracle__s0"
    _write_eval(run_dir)
    return confirmatory_report(
        audit_dir=audit,
        runs={"oracle": {0: run_dir}},
        early_runs={},
        analysis={
            "final_step": 2,
            "hypotheses": True,
            "allow_partial_menu": True,
            "early": {"enabled": False},
            "h1": {"contrasts": [], "beside": []},
            "h2": {"contrasts": []},
            "h3": {"pairs": [], "tested": False},
            "n_boot": 20,
            "seed": 0,
        },
        withdrawals={},
        planned_seeds={"oracle": [0]},
    )


def confirmatory_refuses_incomplete() -> str:
    from vfates.analysis.confirmatory import confirmatory_report

    try:
        confirmatory_report(
            audit_dir=".",
            runs={},
            early_runs={},
            analysis={"final_step": 2, "early": {"enabled": False}, "hypotheses": False},
            withdrawals={},
            planned_seeds={"oracle": [0]},
        )
    except ValueError as exc:
        return str(exc)
    return ""


def _enqueue_this(job: dict[str, Any], keep: list[dict[str, Any]], train_id: str) -> bool:
    kind = str(job["kind"])
    if kind == "audit":
        return True
    if kind == "train":
        return job["id"] == train_id
    if kind == "early":
        return not any(item["kind"] == "early" for item in keep)
    if kind == "variant":
        return not any(item["kind"] == "variant" for item in keep)
    if kind == "transfer" and train_id in (job.get("depends_on") or []):
        return not any(item["kind"] == "transfer" for item in keep)
    return False


def program_queue_smoke(root: Path) -> dict[str, Any]:
    """Render a two-cell program, run a fake worker, and score the layout."""
    from vfates.analysis.pooled import pooled_report
    from vfates.pilots.common import repo_root
    from vfates.program import plan_jobs, render_program
    from vfates.schedule import enqueue, format_status, read_jobs, run_worker
    from vfates.utils.io import read_yaml, write_yaml

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    configs = root / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    repo = repo_root()
    for name in ("templates", "arms"):
        dest = configs / name
        if not dest.exists():
            dest.symlink_to(repo / "configs" / name)
    policy = read_yaml(repo / "configs" / "program.yaml")["hardware_policy"]
    cell_ids = ("e1-chain_sum-qwen3-1.7b", "e1-math-qwen3-1.7b")
    cells = []
    for cell_id, domain in zip(cell_ids, ("chain_sum", "math"), strict=True):
        row: dict[str, Any] = {
            "id": cell_id,
            "experiment": "E1",
            "domain": domain,
            "model": "qwen3-1.7b",
            "menu": "full",
            "hardware_class": "apple-silicon",
            "config": f"configs/cells/{cell_id}.yaml",
            "seeds": [0, 1, 2],
            "arms": ["oracle"],
        }
        if domain == "chain_sum":
            row["variants"] = [{"name": "streaming", "arms": ["oracle"], "seeds": [0]}]
        cells.append(row)
    program = {
        "schema": "vfates.program/1",
        "hardware_policy": {
            "E1": {"by": "domain", "chain_sum": "apple-silicon", "math": "apple-silicon"},
            "speed_vs_h100": dict(policy["speed_vs_h100"]),
            "usd_per_gpu_hour": dict(policy["usd_per_gpu_hour"]),
        },
        "exploratory": {"X1": {"kind": "transfer_eval"}},
        "stages": {
            "A": {"name": "discovery", "cells": [cell_ids[0]]},
            "B": {"name": "replication", "cells": [cell_ids[1]]},
        },
        "compute": {"cut_order": [cell_ids[1]]},
        "default_cost_model": {
            "reference_params_b": 1.7,
            "mean_completion_len": 64,
            "steps": 2,
            "prompts_per_step": 4,
            "group_size": 2,
            "early_steps": 2,
            "h100_gen_tok_s": 10000,
            "h100_train_tok_s": 30000,
            "overhead": 1.0,
        },
        "cells": cells,
    }
    path = root / "program.yaml"
    write_yaml(path, program)
    written = render_program(path, root=root)
    jobs = plan_jobs(program, root=root, runs_root="runs")
    transfer_jobs = [job for job in jobs if job["kind"] == "transfer"]
    keep: list[dict[str, Any]] = []
    train_id = "e1-chain_sum-qwen3-1.7b/train/oracle/s0"
    for job in jobs:
        if job["cell"] != "e1-chain_sum-qwen3-1.7b":
            continue
        if _enqueue_this(job, keep, train_id):
            keep.append(job)
    stub = root / "worker_stub.py"
    stub.write_text(_WORKER_STUB.replace("PYTHON", sys.executable), encoding="utf-8")
    stub.chmod(0o755)
    queue = root / "queue.json"
    enqueue(queue, keep)
    for _ in range(len(keep) + 2):
        run_worker(
            queue,
            hardware_class="apple-silicon",
            gpus=1,
            once=True,
            repo=root,
            python=str(stub),
            heartbeat_s=0.01,
            poll_s=0.01,
        )
        states = [str(job.get("state")) for job in read_jobs(queue)]
        if states and all(state == "done" for state in states):
            break
        if any(state == "failed" for state in states):
            break
    status = format_status(queue)
    cell = "e1-chain_sum-qwen3-1.7b"
    pooled = pooled_report(
        {
            "analysis": {"h3_min_pairs": 5},
            "cells": [
                {
                    "id": cell,
                    "experiment": "E1",
                    "domain": "chain_sum",
                    "model": "qwen3-1.7b",
                    "config": {
                        "params_b": 1.7,
                        "analysis": {
                            "final_step": 2,
                            "early": {"enabled": False},
                            "h1": {"contrasts": [], "beside": []},
                            "h2": {"contrasts": []},
                            "h3": {"pairs": []},
                            "n_boot": 20,
                        },
                        "study": {
                            "arms": ["oracle"],
                            "seeds": [0],
                            "variants": [{"name": "streaming", "arms": ["oracle"], "seeds": [0]}],
                        },
                    },
                }
            ],
        },
        root / "runs",
        n_boot=20,
        n_perm=20,
        seed=0,
    )
    names = [row.get("name") for row in pooled.get("exploratory", {}).get("variants", [])]
    return {
        "rendered": len(written),
        "n_enqueued": len(keep),
        "n_done": sum(1 for job in read_jobs(queue) if job.get("state") == "done"),
        "status": status,
        "transfer_planned": len(transfer_jobs),
        "pooled_schema": pooled.get("schema"),
        "oracle_gain": (pooled.get("cells") or [{}])[0].get("oracle_gain", {}).get("value"),
        "variant_names": names,
        "early": (root / "runs" / cell / "early").is_dir(),
        "base": (root / "runs" / cell / "runs" / "oracle__s0").is_dir(),
        "variant_dir": (root / "runs" / cell / "variants" / "streaming").is_dir(),
        "audit_dir": (root / "runs" / cell / "audit").is_dir(),
    }


_WORKER_STUB = """#!PYTHON
import json
import sys
from pathlib import Path


def flag(name):
    if name not in sys.argv:
        return None
    return sys.argv[sys.argv.index(name) + 1]


def write_run(dest):
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    prompts = []
    for step, acc in ((0, 0.2), (2, 0.5)):
        rows.append(
            {
                "step": step,
                "split": "heldout",
                "oracle_acc": acc,
                "J": 0.1,
                "fpr": 0.1,
                "fnr": 0.2,
            }
        )
        prompts.append({"step": step, "split": "heldout", "acc": {"p0": acc}})
    (dest / "eval.jsonl").write_text("".join(json.dumps(row) + "\\n" for row in rows))
    (dest / "eval_prompts.jsonl").write_text("".join(json.dumps(row) + "\\n" for row in prompts))
    (dest / "summary.json").write_text(json.dumps({"status": "complete", "steps_completed": 2}))


if "transfer-eval" in sys.argv:
    runs = Path(flag("--runs") or "runs")
    dest = runs / (flag("--cell") or "cell") / "transfer" / (
        (flag("--arm") or "oracle") + "__s" + (flag("--seed") or "0")
    )
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "job.txt").write_text("transfer-eval\\n")
    raise SystemExit(0)

out = Path(flag("--out") or ".")
if flag("--only") == "audit":
    audit = out / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    arm = {
        "step0": {"J": 0.1, "fpr": 0.1, "fnr": 0.2},
        "mean_precision": 0.5,
        "modal_precision": {"value": 0.5},
        "transfer": {
            "tau_pool": 0.2,
            "tau_hard": 0.1,
            "tau_rel": 0.3,
            "reliability": 0.9,
            "cos_gV_gO": 0.4,
            "base_acc_heldout": 0.3,
        },
    }
    (audit / "forecasts.json").write_text(json.dumps({"oracle": arm}))
    (audit / "problems.jsonl").write_text('{"pid": "p0"}\\n')
    raise SystemExit(0)

arm_name = flag("--arm") or "oracle"
seed = flag("--seed") or "0"
variant = flag("--variant")
if "--early" in sys.argv:
    dest = out / "early" / f"{arm_name}__s{seed}"
elif variant:
    dest = out / "variants" / variant / f"{arm_name}__s{seed}"
else:
    dest = out / "runs" / f"{arm_name}__s{seed}"
write_run(dest)
raise SystemExit(0)
"""


def _run_structural(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Plumbing that does not load a model. ``live`` adds the domain audits."""
    started = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    checks = Checks()
    tables: dict[str, Any] = {}
    if config.get("pytest"):
        code, tail = _pytest(
            Path(__file__).resolve().parents[3],
            config.get("pytest_marker", "not slow and not network"),
            1800,
        )
        checks.add(
            "P1.1",
            "unit suite passes on this machine",
            code,
            code == 0,
            threshold=0,
            metric=f"pytest exit ({tail})",
        )
    same, compared = _generator_identity(int(config.get("identity_items", 20)))
    if same is not None:
        checks.add(
            "P1.2",
            "built-in chain-sum generator equals Reasoning Gym",
            compared,
            same,
            threshold="identical",
            metric="items compared",
        )
    if not config.get("live"):
        checks.add(
            "P1.0",
            "model path was not run",
            None,
            True,
            metric="set live=true to audit Qwen/Qwen3-0.6B on each domain",
            documented=True,
        )
    try:
        facts = program_queue_smoke(out / "program")
        tables["program"] = {key: value for key, value in facts.items() if key != "status"}
        checks.add(
            "P1.25",
            "program render, plan, enqueue, worker, and status",
            facts["n_done"],
            facts["n_done"] == facts["n_enqueued"] and facts["rendered"] == 2,
            threshold=facts["n_enqueued"],
            metric="jobs done",
        )
        checks.add(
            "P1.26",
            "pooled report reads base, early, and variant runs",
            facts["oracle_gain"],
            facts["pooled_schema"] == "vfates.pooled/1"
            and facts["base"]
            and facts["early"]
            and facts["variant_dir"]
            and "streaming" in facts["variant_names"]
            and facts["oracle_gain"] is not None,
            metric="oracle gain",
        )
        checks.add(
            "P1.27",
            "plan contains a transfer-eval job",
            facts["transfer_planned"],
            facts["transfer_planned"] >= 1,
            threshold=1,
            metric="transfer jobs in the plan",
        )
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        checks.add(
            "P1.25",
            "program render, plan, enqueue, worker, and status",
            None,
            False,
            metric=f"{type(exc).__name__}: {exc}",
        )
    try:
        report = confirmatory_well_formed(out / "confirmatory")
        checks.add(
            "P1.28",
            "confirmatory report is well formed",
            None,
            report.get("schema") == "vfates.confirmatory/1"
            and report.get("h3", {}).get("decision") == "not_tested",
            metric=str(report.get("schema")),
        )
    except (OSError, ValueError, KeyError, AssertionError) as exc:
        checks.add(
            "P1.28",
            "confirmatory report is well formed",
            None,
            False,
            metric=f"{type(exc).__name__}: {exc}",
        )
    reason = confirmatory_refuses_incomplete()
    checks.add(
        "P1.29",
        "confirmatory analysis refuses an incomplete cell",
        None,
        "incomplete planned runs" in reason,
        metric=reason or "no refusal",
    )
    checks.add(
        "P1.22",
        "SIGUSR1 sets the preemption flag",
        None,
        sigusr1_marks_preempted(),
        metric="PreemptionCallback.preempted",
    )
    checks.add(
        "P1.23",
        "study refuses a hardware-class mismatch",
        None,
        study_refuses_hardware_mismatch(out / "refusals"),
        metric="ValueError before the audit",
    )
    checks.add(
        "P1.24",
        "study refuses a chat_template_kwargs mismatch",
        None,
        study_refuses_chat_mismatch(out / "refusals"),
        metric="ValueError before the audit",
    )
    if config.get("live"):
        _run_live_domains(config, out, checks, tables)
    return write_report(
        out,
        "P1 harness",
        checks,
        config=config,
        started=started,
        tables=tables,
        intro=(
            "Checks that do not name a model load nothing. Domain audits, "
            "log-probability agreement, LoRA parity, and the chain_sum GRPO "
            "run load Qwen/Qwen3-0.6B only when live is true. Math uses "
            "DeepScaleR when HF_TOKEN is unset. SIGUSR1 is checked on the "
            "preemption callback, not by signalling a running trainer."
        ),
    )


def _run_live_domains(
    config: dict[str, Any], out: Path, checks: Checks, tables: dict[str, Any]
) -> None:
    """One tiny dev-pool audit per domain, then a 2-step GRPO run on chain_sum."""
    from vfates.audit import run_audit
    from vfates.models import load_model_spec
    from vfates.pilots.common import dev_pools_honored
    from vfates.utils.env import hardware_class

    if not dev_pools_honored():
        checks.add(
            "P1.15",
            "development pools are drawn by the audit",
            None,
            False,
            metric="load_pools does not read pools.from",
        )
        return
    spec = load_model_spec("qwen3-0.6b")
    kwargs = None if spec.chat_template_kwargs is None else dict(spec.chat_template_kwargs)
    domains = list(config.get("domains") or ("chain_sum", "math", "code", "ifeval", "qa"))
    for domain in domains:
        audit_dir = out / "domains" / domain
        cfg = _live_cell(domain, spec, kwargs, hardware_class())
        before = None
        if (audit_dir / "lock.json").is_file() and not config.get("fresh", True):
            before = audit_artifact_digest(audit_dir)
        run_audit(cfg["audit"], audit_dir)
        if before is not None:
            checks.add(
                f"P1.5.{domain}",
                f"{domain} resume recomputes nothing",
                None,
                before == audit_artifact_digest(audit_dir),
                metric="artifact digests",
            )
            continue
        flags = audit_stage_flags(audit_dir, expect_patches=_domain_has_patches(domain))
        tables[domain] = flags
        checks.add(
            f"P1.3.{domain}",
            f"{domain} audit writes every artifact",
            None,
            flags["artifacts"],
            metric="missing or unexpected artifacts",
        )
        checks.add(
            f"P1.11.{domain}",
            f"{domain} lock payload equals forecasts",
            None,
            flags["lock_matches"],
            metric="canonical JSON",
        )
        checks.add(
            f"P1.12.{domain}",
            f"{domain} features used sampled token ids",
            None,
            flags["token_ids"],
            metric="token_source",
        )
        checks.add(
            f"P1.13.{domain}",
            f"{domain} stage fingerprints match the patch policy",
            None,
            flags["stages"],
            metric="stages.json",
        )
        checks.add(
            f"P1.14.{domain}",
            f"{domain} LoRA init matches the lock",
            None,
            flags["lora_init"],
            metric="lora_init.safetensors",
        )
        checks.add(
            f"P1.15.{domain}",
            f"{domain} lock records pool_source dev",
            None,
            flags["pool_source"],
            metric="pool_source",
        )
        checks.add(
            f"P1.16.{domain}",
            f"{domain} lock hardware_class matches this machine",
            None,
            flags["hardware_class"],
            metric="hardware_class",
        )
        checks.add(
            f"P1.17.{domain}",
            f"{domain} tau_pool_budget draws cover 32, 64, 128, 192",
            None,
            flags["tau_budget"],
            metric="forecast arms",
        )
        checks.add(
            f"P1.18.{domain}",
            f"{domain} patch_selection is present exactly when patch arms exist",
            None,
            flags["patch_selection"],
            metric="forecasts.json",
        )
        digest = audit_artifact_digest(audit_dir)
        run_audit(cfg["audit"], audit_dir)
        checks.add(
            f"P1.5.{domain}",
            f"{domain} resume recomputes nothing",
            None,
            digest == audit_artifact_digest(audit_dir),
            metric="artifact digests",
        )
    chain_dir = out / "domains" / "chain_sum"
    if (chain_dir / "lock.json").is_file():
        checks.add(
            "P1.4",
            "lock verifies and detects a one-number edit",
            None,
            _lock_detects_tamper(chain_dir, out),
            metric="verify(original) and not verify(tampered)",
        )
    _live_model_agreement(spec.model_id, checks, tables)
    _live_train(out, checks, hardware_class())


def _domain_has_patches(domain: str) -> bool:
    return domain in ("math", "code", "ifeval", "qa")


def _live_cell(
    domain: str, spec: Any, kwargs: dict[str, Any] | None, hardware: str
) -> dict[str, Any]:
    from vfates.pilots.common import cell_config

    overrides: dict[str, Any] = {
        "audit.pools.train": 4,
        "audit.pools.heldout": 2,
        "audit.sampling.n": 2,
        "audit.sampling.max_tokens": 32,
        "audit.sampling.chunk_prompts": 4,
        "audit.generator.backend": "mlx",
        "audit.generator.model_id": spec.model_id,
        "audit.generator.revision": spec.revision,
        "audit.features.model_id": spec.model_id,
        "audit.features.sketch_dim": 128,
        "audit.features.load.device": "cpu",
        "audit.features.load.dtype": "float32",
        "audit.forecast.n_boot": 0,
        "audit.forecast.reliability_splits": 2,
        "audit.arms.n_boot": 20,
        "study.backend": "mlx",
        "study.hardware_class": hardware,
        "study.train.model_id": spec.model_id,
        "study.train.steps": 2,
    }
    if kwargs is not None:
        overrides["audit.generator.chat_template_kwargs"] = dict(kwargs)
        overrides["audit.features.chat_template_kwargs"] = dict(kwargs)
        overrides["audit.generator.kwargs"] = {
            "completion_batch_size": 8,
            "chat_template_kwargs": dict(kwargs),
        }
        overrides["study.train.chat_template_kwargs"] = dict(kwargs)
        overrides["study.eval_generator.kwargs.chat_template_kwargs"] = dict(kwargs)
    if domain == "math" and not os.environ.get("HF_TOKEN"):
        overrides["audit.task.source"] = "deepscaler"
    if domain == "qa":
        overrides["audit.task.judge"] = {"type": "heuristic"}
    if domain == "ifeval":
        overrides["audit.task.judge_config"] = {"type": "heuristic"}
    return cell_config(f"e1-{domain}-qwen3-1.7b", dev=True, overrides=overrides)


def _live_model_agreement(model_id: str, checks: Checks, tables: dict[str, Any]) -> None:
    """Log-probability agreement, token round-trip, and LoRA module parity on chain_sum."""
    agreement = _logprob_agreement(model_id, _LOGPROB_PROMPTS, _LOGPROB_MAX_TOKENS)
    tables["logprob"] = agreement
    checks.add(
        "P1.6",
        "MLX and PyTorch agree on sampled-token log-probabilities",
        agreement["per_token_abs_diff_median"],
        agreement["per_token_abs_diff_median"] <= _LOGPROB_ABS_DIFF,
        threshold=_LOGPROB_ABS_DIFF,
        metric="median |diff| nats/token",
    )
    checks.add(
        "P1.7",
        "decoded completions re-encode to the sampled token ids",
        agreement["roundtrip_rate"],
        agreement["roundtrip_rate"] >= _ROUNDTRIP_RATE,
        threshold=_ROUNDTRIP_RATE,
        metric="rate",
    )
    train_cfg = _chain_sum_train_cfg("apple-silicon")
    parity = _lora_parity(model_id, train_cfg)
    tables["lora_parity"] = parity
    checks.add(
        "P1.8",
        "MLX and PEFT LoRA adapt the same modules at the same scale",
        parity["mlx_modules"],
        parity["same_set"] and parity["mlx_scales"] == parity["peft_scales"],
        metric="adapted modules (both)",
    )


def _chain_sum_train_cfg(hardware: str) -> dict[str, Any]:
    """2-step chain_sum GRPO settings. Resume asks for one step past this."""
    return {
        "model_id": "Qwen/Qwen3-0.6B",
        "steps": 2,
        "prompts_per_step": 2,
        "group_size": 2,
        "max_tokens": 32,
        "lr": 2.0e-5,
        "eval_every": 1,
        "eval_heldout_n": 2,
        "eval_train_n": 2,
        "eval_samples_per_prompt": 1,
        "eval_max_tokens": 32,
        "save_every": 1,
        "micro_batch_size": 2,
        "seed": 0,
        "hardware_class": hardware,
        "lora": {"rank": 8, "alpha": 16, "dropout": 0.0, "num_layers": 1},
    }


def _live_train(out: Path, checks: Checks, hardware: str) -> None:
    from vfates.experiment import train_arm

    audit_dir = out / "domains" / "chain_sum"
    if not (audit_dir / "lock.json").is_file():
        checks.add(
            "P1.21",
            "GRPO loads the audit LoRA A and evaluates step 0 and the last step",
            None,
            False,
            metric="chain_sum audit is missing",
        )
        checks.add(
            "P1.9",
            "a real-model GRPO run logs, checkpoints and resumes",
            None,
            False,
            metric="chain_sum audit is missing",
        )
        checks.add(
            "P1.10",
            "analysis joins the lock with the run",
            None,
            False,
            metric="chain_sum audit is missing",
        )
        return
    train_cfg = _chain_sum_train_cfg(hardware)
    steps_planned = int(train_cfg["steps"])
    run_dir = out / "train"
    summary = train_arm(
        audit_dir, "oracle", train_cfg, run_dir, backend="mlx", max_train=4, max_heldout=2
    )
    from vfates.analysis.outcomes import run_outcome, score_forecasts
    from vfates.forecast.lockbox import load_lock
    from vfates.utils.io import read_jsonl

    steps = {
        int(row["step"])
        for row in read_jsonl(run_dir / "eval.jsonl")
        if row.get("split") == "heldout"
    }
    checks.add(
        "P1.21",
        "GRPO loads the audit LoRA A and evaluates step 0 and the last step",
        summary.get("steps_completed"),
        0 in steps and steps_planned in steps,
        threshold=steps_planned,
        metric="held-out eval steps",
    )
    missing = [name for name in _TRAIN_FILES if not (run_dir / name).is_file()]
    resumed_steps = steps_planned + 1
    resumed = train_arm(
        audit_dir,
        "oracle",
        {**train_cfg, "steps": resumed_steps},
        run_dir,
        backend="mlx",
        resume=True,
        max_train=4,
        max_heldout=2,
    )
    checks.add(
        "P1.9",
        "a real-model GRPO run logs, checkpoints and resumes",
        resumed.get("steps_completed"),
        resumed.get("steps_completed") == resumed_steps and not missing,
        threshold=resumed_steps,
        metric="steps after resume",
    )
    outcome = run_outcome(run_dir)
    scored = score_forecasts(load_lock(audit_dir / "lock.json"), {"oracle": outcome})
    checks.add(
        "P1.10",
        "analysis joins the lock with the run",
        scored["n_arms"],
        scored["n_arms"] == 1,
        threshold=1,
        metric="arms scored",
    )
