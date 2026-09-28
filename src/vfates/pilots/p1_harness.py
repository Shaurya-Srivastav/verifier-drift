"""P1: harness correctness on the target machine, end to end with a real model.

The unit suite checks components in isolation. P1 checks that the pieces
agree with each other where the paper's numbers pass between them:

* the full unit suite passes on this machine;
* the built-in task generator is bit-identical to Reasoning Gym;
* a real-model audit runs every stage, locks its forecasts, detects tampering,
  and resumes without resampling or changing a single verdict;
* MLX (which samples and trains) and PyTorch (which computes features) assign
  the same log-probabilities to the same tokens, and text round-trips to the
  same token ids;
* MLX LoRA and PEFT LoRA adapt the same modules with the same scale, so the
  step-0 features describe the parameters that train;
* a real-model GRPO run writes the logging schema, checkpoints, and resumes;
* the analysis joins the lock with the run.
"""

from __future__ import annotations

import copy
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

from vfates.pilots.common import Checks, markdown_table, write_report
from vfates.utils.io import read_json, sha256_file, write_json


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
    problems = task.problems("train", n_prompts, 123)
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
    model, _ = load(model_id)  # type: ignore[misc]  # mlx_lm stub
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
            lora_alpha=cfg.lora.alpha,  # type: ignore[arg-type]  # peft stub
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


def _table_digest(audit_dir: Path) -> dict[str, str]:
    return {
        "completions": sha256_file(audit_dir / "completions.jsonl"),
        "calibration": sha256_file(audit_dir / "calibration.json"),
        "verdicts": _verdict_digest(audit_dir),
    }


def _verdict_digest(audit_dir: Path) -> str:
    from vfates.types import AuditTable
    from vfates.utils.io import sha256_text

    table = AuditTable.load(audit_dir / "audit.npz")
    parts = [",".join(map(str, table.O.tolist()))]
    parts += [f"{k}:{','.join(map(str, v.tolist()))}" for k, v in sorted(table.V.items())]
    return sha256_text("|".join(parts))


def run(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """P1 harness correctness."""
    from vfates.analysis.outcomes import run_outcome, score_forecasts
    from vfates.audit import run_audit
    from vfates.experiment import train_arm
    from vfates.forecast.lockbox import load_lock, verify_lock

    started = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    checks = Checks()
    tables: dict[str, Any] = {}
    model_id = config["model_id"]

    if config.get("pytest", True):
        code, tail = _pytest(root, config.get("pytest_marker", "not slow and not network"), 1800)
        tables["pytest"] = tail
        checks.add(
            "P1.1",
            "unit suite passes on this machine",
            code,
            code == 0,
            threshold=0,
            metric=f"pytest exit ({tail})",
        )

    same, compared = _generator_identity(int(config.get("identity_items", 200)))
    if same is not None:
        checks.add(
            "P1.2",
            "built-in chain-sum generator equals Reasoning Gym",
            compared,
            same,
            threshold="identical",
            metric="items compared",
        )

    audit_dir = out / "audit"
    if audit_dir.exists() and config.get("fresh", True):
        shutil.rmtree(audit_dir)
    audit_cfg = copy.deepcopy(config["audit"])
    audit_cfg["generator"]["model_id"] = model_id
    audit_cfg["features"]["model_id"] = model_id
    run_audit(audit_cfg, audit_dir)
    expected = [
        "problems.jsonl",
        "completions.jsonl",
        "calibration.json",
        "audit.npz",
        "confusion.json",
        "features.npz",
        "forecasts.json",
        "lock.json",
    ]
    missing = [f for f in expected if not (audit_dir / f).exists()]
    checks.add(
        "P1.3",
        "audit writes every artifact",
        len(missing),
        not missing,
        threshold=0,
        metric="missing artifacts",
    )
    tampered = out / "lock_tampered.json"
    doc = read_json(audit_dir / "lock.json")
    first_arm = next(iter(doc["payload"]))
    doc["payload"][first_arm]["transfer"]["tau_pool"] = 0.123456
    write_json(tampered, doc)
    lock_ok = verify_lock(audit_dir / "lock.json") and not verify_lock(tampered)
    checks.add(
        "P1.4",
        "lock verifies and detects a one-number edit",
        None,
        lock_ok,
        metric="verify(original) and not verify(tampered)",
    )

    before = _table_digest(audit_dir)
    run_audit(audit_cfg, audit_dir)
    after = _table_digest(audit_dir)
    checks.add(
        "P1.5",
        "resuming an audit resamples nothing and changes no verdict",
        sum(before[k] != after[k] for k in before),
        before == after,
        threshold=0,
        metric="changed digests",
    )

    lp = _logprob_agreement(
        model_id, int(config.get("logprob_prompts", 8)), int(config.get("logprob_max_tokens", 128))
    )
    tables["logprob"] = lp
    tol = float(config["gates"]["logprob_per_token"])
    checks.add(
        "P1.6",
        "MLX and PyTorch agree on sampled-token log-probabilities",
        lp["per_token_abs_diff_median"],
        lp["per_token_abs_diff_median"] <= tol,
        threshold=tol,
        metric="median |diff| nats/token",
    )
    rt = float(config["gates"]["roundtrip_rate"])
    checks.add(
        "P1.7",
        "decoded completions re-encode to the sampled token ids",
        lp["roundtrip_rate"],
        lp["roundtrip_rate"] >= rt,
        threshold=rt,
        metric="rate",
    )

    parity = _lora_parity(model_id, config["train"])
    tables["lora_parity"] = parity
    parity_ok = parity["same_set"] and parity["mlx_scales"] == parity["peft_scales"]
    checks.add(
        "P1.8",
        "MLX and PEFT LoRA adapt the same modules at the same scale",
        parity["mlx_modules"],
        parity_ok,
        metric="adapted modules (both)",
    )

    run_dir = out / "train"
    if run_dir.exists() and config.get("fresh", True):
        shutil.rmtree(run_dir)
    arm = config.get("train_arm", "noise_fp")
    train_cfg = {**config["train"], "model_id": model_id}
    train_arm(audit_dir, arm, train_cfg, run_dir)
    files = ["metrics.jsonl", "eval.jsonl", "eval_prompts.jsonl", "summary.json", "config.json"]
    missing = [f for f in files if not (run_dir / f).exists()]
    steps = int(train_cfg["steps"])
    summary = train_arm(audit_dir, arm, {**train_cfg, "steps": steps + 1}, run_dir, resume=True)
    resumed = summary["steps_completed"] == steps + 1 and not missing
    checks.add(
        "P1.9",
        "a real-model GRPO run logs, checkpoints and resumes",
        summary["steps_completed"],
        resumed,
        threshold=steps + 1,
        metric="steps after resume",
    )

    outcome = run_outcome(run_dir)
    scored = score_forecasts(load_lock(audit_dir / "lock.json"), {arm: outcome})
    checks.add(
        "P1.10",
        "analysis joins the lock with the run",
        scored["n_arms"],
        scored["n_arms"] == 1,
        threshold=1,
        metric="arms scored",
    )
    tables["outcome"] = {k: v for k, v in outcome.items() if k != "curve"}

    sections = [
        ("Log-probability agreement", markdown_table(list(lp), [list(lp.values())])),
        ("LoRA parity", markdown_table(list(parity), [list(parity.values())])),
        (
            "Training smoke",
            markdown_table(
                ["arm", "steps", "held-out step 0", "held-out final", "train final"],
                [
                    [
                        arm,
                        outcome["steps"],
                        outcome["heldout_step0"],
                        outcome["heldout_final"],
                        outcome["train_final"],
                    ]
                ],
            ),
        ),
    ]
    return write_report(
        out, "P1 harness", checks, config=config, started=started, tables=tables, sections=sections
    )
