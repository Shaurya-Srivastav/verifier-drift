"""LoRA GRPO on Apple silicon.

The objective is :mod:`vfates.train.grpo_core`: group-std advantages (TRL
``scale_rewards="group"``) or mean-centered advantages (TRL ``"none"``), the
clipped surrogate, and k3 KL. With ``num_iterations=1`` the old policy is the
detached current log-probability, so the ratio is 1 and the gradient matches
REINFORCE. Further iterations reuse that first forward as ``old`` and clip.

Rollouts come from :class:`~vfates.gen.mlx_backend.MLXGenerator` wrapped around
the live LoRA model. The reference model is a second ``mlx_lm.load`` (frozen,
no LoRA) and is built only when ``beta_kl > 0``.

Checkpoints store trainable (LoRA) weights, AdamW state and ``trainer_state.json``
(step, data cursor). ``run(resume=True)`` reloads the latest ``checkpoints/step_*``
and continues. Generation is reseeded from ``(seed, step)``, so a resumed step
matches a continuous run given the same weights and cursor.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.utils import linear_to_lora_layers

from vfates.gen.mlx_backend import MLXGenerator, encode_prompt, wire_memory
from vfates.tasks.base import Task
from vfates.train.config import GRPOConfig, raise_if_nonfinite
from vfates.train.evaluate import append_eval_samples, eval_log_rows, evaluate_policy
from vfates.train.grpo_core import loss_weights
from vfates.train.rewards import RewardContext, onpolicy_rates
from vfates.types import Problem
from vfates.utils.env import environment_report
from vfates.utils.hashing import stable_seed
from vfates.utils.io import append_jsonl, read_json, read_jsonl, write_json
from vfates.verifiers.base import Verifier

METRIC_KEYS = (
    "step",
    "wall_s",
    "tokens_generated",
    "gen_seconds",
    "train_seconds",
    "reward_V_mean",
    "oracle_mean",
    "onpolicy",
    "frac_informative_groups",
    "mean_completion_length",
    "max_completion_length",
    "truncation_rate",
    "loss",
    "grad_norm",
    "lr",
    "kl",
    "monitors",
)


def group_advantages_mx(
    rewards: Any,
    mode: str = "std",
    eps: float = 1e-4,
    std: str = "unbiased",
) -> tuple[mx.array, mx.array]:
    """MLX port of :func:`vfates.train.grpo_core.group_advantages` (float32)."""
    r = mx.array(np.asarray(rewards, dtype=np.float32))
    if r.ndim != 2:
        raise ValueError(f"rewards must have shape [B, G], got {r.shape}")
    if mode not in ("std", "mean"):
        raise ValueError(f"adv mode must be 'std' or 'mean', got {mode!r}")
    if std not in ("unbiased", "population"):
        raise ValueError(f"std must be 'unbiased' or 'population', got {std!r}")
    _b, g = int(r.shape[0]), int(r.shape[1])
    mean = mx.mean(r, axis=1, keepdims=True)
    centered = r - mean
    ddof = 1 if std == "unbiased" else 0
    if g - ddof <= 0:
        scale = mx.zeros((_b, 1), dtype=mx.float32)
        zero = mx.ones((_b, 1), dtype=mx.bool_)
    else:
        var = mx.sum(centered * centered, axis=1, keepdims=True) / float(g - ddof)
        scale = mx.sqrt(var)
        zero = mx.abs(scale) <= np.float32(1e-8)
    if mode == "std":
        adv = centered / (scale + np.float32(eps))
        adv = mx.where(zero, mx.zeros_like(adv), adv)
    else:
        adv = centered
    informative = mx.logical_not(mx.reshape(zero, (_b,)))
    return adv, informative


def pg_loss_mx(
    token_logps: mx.array,
    advantages: mx.array,
    mask: mx.array,
    agg: str,
    max_len: int,
    old_token_logps: mx.array | None = None,
    ref_token_logps: mx.array | None = None,
    beta: float = 0.0,
    eps_low: float = 0.2,
    eps_high: float = 0.2,
) -> mx.array:
    """Differentiable GRPO loss. Values match :func:`pg_loss_reference` on fixed inputs.

    ``token_logps`` and ``mask`` are ``[N, T]``. ``advantages`` is ``[N]``. When
    ``old_token_logps`` is omitted the ratio is ``exp(logp - stopgrad(logp))``,
    whose value is 1 and whose gradient w.r.t. ``logp`` is the REINFORCE factor.
    """
    token_logps = token_logps.astype(mx.float32)
    mask_f = mask.astype(mx.float32)
    lengths = np.array(mx.sum(mask_f, axis=1).tolist(), dtype=np.float64)
    weights = mx.array(loss_weights(lengths, agg, max_len).astype(np.float32))
    per = _per_token_loss(
        token_logps, advantages, old_token_logps, ref_token_logps, beta, eps_low, eps_high
    )
    per = per * mask_f
    return mx.sum(mx.sum(per, axis=1) * weights)


def _per_token_loss(
    token_lp: mx.array,
    advantages: mx.array,
    old: mx.array | None,
    ref: mx.array | None,
    beta: float,
    eps_low: float,
    eps_high: float,
) -> mx.array:
    old_lp = mx.stop_gradient(token_lp) if old is None else old.astype(mx.float32)
    ratio = mx.exp(token_lp - old_lp)
    clipped = mx.clip(ratio, 1.0 - eps_low, 1.0 + eps_high)
    adv = advantages.astype(mx.float32).reshape((-1, 1))
    # Detach the branch choice so that at ratio=1 (both branches equal) the
    # gradient flows through `ratio`, matching ∇(-A log π) = -A.
    ratio_det = mx.stop_gradient(ratio)
    clipped_det = mx.stop_gradient(clipped)
    use_unclipped = (ratio_det * adv) <= (clipped_det * adv)
    coef = mx.where(use_unclipped, ratio, clipped)
    per = -(coef * adv)
    if ref is not None and beta != 0.0:
        ref_lp = ref.astype(mx.float32)
        delta = ref_lp - token_lp
        per = per + float(beta) * (mx.exp(delta) - delta - 1.0)
    return per


def _completion_logprobs(
    model: nn.Module,
    full_ids: mx.array,
    prompt_lens: mx.array,
    n_comp: int,
    temperature: float = 1.0,
) -> mx.array:
    """Log π of each completion token. Right-padding is valid under a causal mask.

    Logits are divided by the sampling temperature, as TRL does before
    ``selective_log_softmax``. Temperature 1 leaves the logits unchanged.
    """
    logits = model(full_ids).astype(mx.float32)
    pred = logits[:, :-1, :]
    # Temperature 0 is greedy: the sampled token is an argmax of the raw logits.
    # Temperature 1 is the untempered model. Both leave the logits unchanged.
    if temperature < 0.0:
        raise ValueError(f"loss temperature must be >= 0, got {temperature}")
    if temperature not in (0.0, 1.0):
        pred = pred / mx.array(float(temperature), dtype=mx.float32)
    targets = full_ids[:, 1:]
    chosen = mx.take_along_axis(pred, targets[:, :, None], axis=-1).squeeze(-1)
    all_lp = chosen - mx.logsumexp(pred, axis=-1)
    pos = prompt_lens[:, None] + mx.arange(n_comp)[None, :] - 1
    pos = mx.clip(pos, 0, all_lp.shape[1] - 1)
    return mx.take_along_axis(all_lp, pos.astype(mx.int32), axis=1)


class _Pool:
    """multi_epoch reshuffles with ``stable_seed(seed, 'data', epoch)``; streaming walks once."""

    def __init__(self, problems: list[Problem], regime: str, seed: int) -> None:
        self.problems = list(problems)
        self.regime = regime
        self.seed = int(seed)
        self.epoch = 0
        self.offset = 0
        self.perm = self._permutation(0)

    def _permutation(self, epoch: int) -> np.ndarray:
        rng = np.random.default_rng(stable_seed(self.seed, "data", int(epoch)))
        return rng.permutation(len(self.problems))

    def state(self) -> dict[str, int]:
        return {"epoch": self.epoch, "offset": self.offset}

    def load_state(self, state: dict[str, int]) -> None:
        self.epoch = int(state["epoch"])
        self.offset = int(state["offset"])
        self.perm = self._permutation(self.epoch)

    def take(self, k: int) -> list[Problem]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not self.problems:
            raise RuntimeError("train pool is empty")
        if self.regime == "streaming":
            if self.offset + k > len(self.problems):
                raise RuntimeError(
                    f"streaming train pool exhausted at offset {self.offset}: "
                    f"need {k} problems, {len(self.problems) - self.offset} remain"
                )
            batch = self.problems[self.offset : self.offset + k]
            self.offset += k
            return batch
        out: list[Problem] = []
        while len(out) < k:
            if self.offset >= len(self.perm):
                self.epoch += 1
                self.offset = 0
                self.perm = self._permutation(self.epoch)
            n = min(k - len(out), len(self.perm) - self.offset)
            idxs = self.perm[self.offset : self.offset + n]
            out.extend(self.problems[int(i)] for i in idxs)
            self.offset += int(n)
        return out


class MLXGRPOTrainer:
    """On-policy LoRA GRPO. ``model`` / ``tokenizer`` override ``mlx_lm.load`` (tests)."""

    def __init__(
        self,
        config: GRPOConfig,
        task: Task,
        verifier: Verifier,
        train_problems: list[Problem],
        heldout_problems: list[Problem],
        monitors: dict[str, Any] | None = None,
        model: nn.Module | None = None,
        tokenizer: Any = None,
    ) -> None:
        self.config = config if isinstance(config, GRPOConfig) else GRPOConfig.from_dict(config)
        self.task = task
        self.verifier = verifier
        self.train_problems = list(train_problems)
        self.heldout_problems = list(heldout_problems)
        self.monitors = dict(monitors or {})
        self._injected = model is not None
        self.model = model
        self.tokenizer = tokenizer
        self.ref: nn.Module | None = None
        self.optimizer: optim.Optimizer | None = None
        self.pool: _Pool | None = None
        self.completed = 0
        self._step_committed = False
        self._lora_init_sha: str | None = None
        self._last_eval: dict[str, float | None] = {}
        by_pid: dict[str, Problem] = {}
        for problem in self.train_problems:
            by_pid[problem.pid] = problem
        for problem in self.heldout_problems:
            if problem.pid in by_pid:
                raise ValueError(f"pid {problem.pid!r} is in both the train and held-out pools")
            by_pid[problem.pid] = problem
        self.ctx = RewardContext(
            task, verifier, by_pid, seed=self.config.seed, monitors=self.monitors
        )

    def run(self, resume: bool = False) -> dict[str, Any]:
        """Train ``config.steps`` optimizer steps. Returns a summary dict."""
        wire_memory()
        cfg = self.config
        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "environment_report.json", environment_report())
        self._prepare_model()
        write_json(out / "config.json", cfg.to_dict())
        assert self.model is not None
        assert self.optimizer is not None
        assert self.pool is not None
        if resume:
            ckpt = latest_checkpoint(out)
            if ckpt is None:
                raise FileNotFoundError(
                    f"resume=True but {out / 'checkpoints'} has no step_* checkpoint"
                )
            self._load_checkpoint(ckpt)
        summary: dict[str, Any] = {
            "steps_completed": self.completed,
            "interrupted": False,
            "out_dir": str(out),
            "final_heldout_oracle_acc": None,
            "final_train_oracle_acc": None,
            "lora_seed": int(cfg.lora_seed),
            "lora_init_sha256": self._lora_init_sha,
            # MLX rollouts are seeded with stable_seed(seed, "gen", step).
            "rollout_seeded": True,
        }
        try:
            # Step 0, and a resumed step whose eval row never landed.
            if self._should_eval(self.completed):
                self._evaluate(self.completed)
            while self.completed < cfg.steps:
                self._step_committed = False
                assert self.pool is not None
                cursor_before = self.pool.state()
                try:
                    row = self._train_step(self.completed)
                except KeyboardInterrupt:
                    if not self._step_committed:
                        self.pool.load_state(cursor_before)
                    else:
                        self.completed += 1
                    raise
                self.completed += 1
                row["step"] = self.completed
                append_jsonl(out / "metrics.jsonl", row)
                self._print_row(row)
                # Eval uses the weights just updated, then the checkpoint stores them.
                if self._should_eval(self.completed):
                    self._evaluate(self.completed)
                if self.completed % cfg.save_every == 0:
                    self._checkpoint()
        except KeyboardInterrupt:
            self._checkpoint()
            summary["interrupted"] = True
            summary["steps_completed"] = self.completed
            summary["peak_memory_gb"] = mx.get_peak_memory() / 1e9
            write_json(out / "summary.json", summary)
            raise
        if self.completed % cfg.save_every != 0:
            self._checkpoint()
        summary["steps_completed"] = self.completed
        summary["final_heldout_oracle_acc"] = self._last_eval.get("heldout")
        summary["final_train_oracle_acc"] = self._last_eval.get("train")
        summary["peak_memory_gb"] = mx.get_peak_memory() / 1e9
        write_json(out / "summary.json", summary)
        return summary

    def _prepare_model(self) -> None:
        cfg = self.config
        if self.model is None:
            from mlx_lm import load

            self.model, self.tokenizer = load(cfg.model_id)
        elif self.tokenizer is None:
            raise ValueError("tokenizer is required when an in-memory model is passed")
        assert self.model is not None
        already = any(isinstance(module, LoRALinear) for _, module in self.model.named_modules())
        if cfg.beta_kl > 0 and not already:
            if self._injected:
                self.ref = _clone_frozen(self.model)
            else:
                from mlx_lm import load

                self.ref, _ = load(cfg.model_id)
                self.ref.freeze()
                self.ref.eval()
        elif cfg.beta_kl > 0 and already and self._injected:
            raise RuntimeError(
                "beta_kl > 0 but the injected model already has LoRA, so the base "
                "reference copy was not taken. Pass the frozen base model instead."
            )
        if not already:
            n_layers = len(self.model.layers)
            if cfg.lora.num_layers > n_layers:
                raise ValueError(
                    f"lora.num_layers={cfg.lora.num_layers} but the model has {n_layers} layers"
                )
            self.model.freeze()
            # Without a file, A is drawn from lora_seed, not from the run seed.
            # The two frameworks do not produce the same matrices.
            if cfg.lora_init is None:
                mx.random.seed(int(cfg.lora_seed))
            linear_to_lora_layers(
                self.model,
                cfg.lora.num_layers,
                {
                    "rank": cfg.lora.rank,
                    "scale": cfg.lora.scale,
                    "dropout": cfg.lora.dropout,
                    "keys": cfg.lora.mlx_keys(),
                },
            )
            if cfg.lora_init is not None:
                self._lora_init_sha = _apply_mlx_lora_init(self.model, cfg)
                cfg.lora_init_sha256 = self._lora_init_sha
        if not tree_flatten(self.model.trainable_parameters()):
            raise RuntimeError("LoRA adaptation produced no trainable parameters")
        if cfg.gradient_checkpointing:
            _enable_grad_checkpoint(self.model)
        self.model.train()
        self.optimizer = _make_optimizer(cfg)
        self.pool = _Pool(self.train_problems, cfg.data_regime, cfg.seed)

    def _train_step(self, step_index: int) -> dict[str, Any]:
        """One optimizer step. ``step_index`` is 0-based; the logged step is +1."""
        cfg = self.config
        assert self.model is not None and self.pool is not None and self.optimizer is not None
        t0 = time.perf_counter()
        problems = self.pool.take(cfg.prompts_per_step)
        prompt_ids = [
            encode_prompt(
                self.tokenizer,
                self.task.prompt(problem),
                self.task.system_prompt(),
                cfg.max_prompt_length,
            )
            for problem in problems
        ]
        flat_ids: list[list[int]] = []
        owners: list[tuple[Problem, int]] = []
        for problem, ids in zip(problems, prompt_ids):
            for k in range(cfg.group_size):
                flat_ids.append(ids)
                owners.append((problem, k))
        gen = MLXGenerator.from_model(
            self.model,
            self.tokenizer,
            model_id=cfg.model_id,
            max_prompt_tokens=cfg.max_prompt_length,
        )
        t_gen = time.perf_counter()
        outs = gen.generate_ids(
            flat_ids,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            seed=stable_seed(cfg.seed, "gen", step_index),
        )
        gen_seconds = time.perf_counter() - t_gen
        del gen
        mx.clear_cache()

        # token_ids include the stop id. skip_special_tokens keeps it out of the reward.
        texts = [
            self.tokenizer.decode(list(out.token_ids), skip_special_tokens=True) for out in outs
        ]
        logged_step = step_index + 1
        records = self.ctx.score_batch(
            [problem.pid for problem, _k in owners],
            texts,
            [self.ctx.draw_key(logged_step, problem.pid, k) for problem, k in owners],
        )
        rewards = np.array([rec["V"] for rec in records], dtype=np.float64).reshape(
            cfg.prompts_per_step, cfg.group_size
        )
        adv_mx, info_mx = group_advantages_mx(rewards, mode=cfg.adv_mode)
        mx.eval(adv_mx, info_mx)
        advantages = np.array(adv_mx.tolist(), dtype=np.float64).reshape(-1)
        informative = np.array(info_mx.tolist(), dtype=bool)
        lengths = np.array([out.n_tokens for out in outs], dtype=np.float64)
        weights = loss_weights(lengths, cfg.loss_agg, cfg.max_tokens)
        sequences = []
        for i, out in enumerate(outs):
            sequences.append(
                {
                    "prompt_ids": flat_ids[i],
                    "token_ids": out.token_ids,
                    "advantage": float(advantages[i]),
                    "weight": float(weights[i]),
                }
            )
        t_train = time.perf_counter()
        loss, grad_norm, kl, lr = self._optimize(sequences, step_index)
        train_seconds = time.perf_counter() - t_train
        mx.clear_cache()
        rates = onpolicy_rates(records)
        kept = [out.n_tokens for out in outs]
        tokens_generated = sum(out.n_tokens for out in outs)
        return {
            "step": logged_step,
            "wall_s": time.perf_counter() - t0,
            "tokens_generated": int(tokens_generated),
            "gen_seconds": gen_seconds,
            "train_seconds": train_seconds,
            "reward_V_mean": rates["accept_rate"],
            "oracle_mean": rates["oracle_acc"],
            "onpolicy": rates,
            "frac_informative_groups": float(np.mean(informative)) if len(informative) else None,
            "mean_completion_length": float(np.mean(kept)) if kept else 0.0,
            "max_completion_length": int(np.max(kept)) if kept else 0,
            "truncation_rate": float(np.mean([out.finish_reason == "length" for out in outs])),
            "loss": loss,
            "grad_norm": grad_norm,
            "lr": lr,
            "kl": kl,
            "monitors": rates["monitors"],
            "monitors_wrong": rates["monitors_wrong"],
        }

    def _optimize(
        self, sequences: list[dict[str, Any]], step_index: int
    ) -> tuple[float, float, float | None, float]:
        cfg = self.config
        assert self.model is not None and self.optimizer is not None
        self.model.train()
        micro = cfg.micro_batch_size
        ref_lps = self._cached_logprobs(self.ref, sequences) if self.ref is not None else None
        old_lps: list[np.ndarray] | None = None
        loss_value = 0.0
        grad_norm = 0.0
        kl_value: float | None = None
        beta = float(cfg.beta_kl)
        eps_low = float(cfg.eps_low)
        eps_high = float(cfg.eps_high)
        temperature = float(cfg.temperature)
        logged_step = int(step_index) + 1

        def loss_fn(model, full_ids, prompt_lens, mask, adv, weights, old, ref):
            if full_ids is None:
                anchor = _anchor_loss(model)
                return anchor, mx.zeros((mask.shape[0], 1), dtype=mx.float32)
            lp = _completion_logprobs(model, full_ids, prompt_lens, int(mask.shape[1]), temperature)
            per = _per_token_loss(lp, adv, old, ref, beta, eps_low, eps_high)
            per = per * mask
            return mx.sum(mx.sum(per, axis=1) * weights), mx.stop_gradient(lp)

        vg = nn.value_and_grad(self.model, loss_fn)
        for iteration in range(cfg.num_iterations):
            accum = None
            total = 0.0
            kl_num = 0.0
            kl_den = 0.0
            captured: list[np.ndarray] | None = (
                [] if iteration == 0 and cfg.num_iterations > 1 else None
            )
            for start in range(0, len(sequences), micro):
                batch = sequences[start : start + micro]
                packed = _pack_batch(batch)
                old = None
                ref = None
                if old_lps is not None:
                    old = _pad_stored(old_lps[start : start + micro], packed["max_c"])
                if ref_lps is not None:
                    ref = _pad_stored(ref_lps[start : start + micro], packed["max_c"])
                if packed["full_ids"] is None:
                    total += 0.0
                    if captured is not None:
                        captured.extend(np.zeros((0,), dtype=np.float32) for _ in batch)
                    continue
                (loss, lp), grads = vg(
                    self.model,
                    packed["full_ids"],
                    packed["prompt_lens"],
                    packed["mask"],
                    packed["adv"],
                    packed["weights"],
                    old,
                    ref,
                )
                grad_pairs = cast(list[tuple[str, Any]], tree_flatten(grads))
                mx.eval(loss, lp, *[leaf for _, leaf in grad_pairs])
                total += float(loss.item())
                accum = grads if accum is None else tree_map(lambda a, b: a + b, accum, grads)
                accum_pairs = cast(list[tuple[str, Any]], tree_flatten(accum))
                mx.eval(*[leaf for _, leaf in accum_pairs])
                if captured is not None:
                    lp_np = np.array(lp.tolist(), dtype=np.float32)
                    mask_np = np.array(packed["mask"].tolist())
                    for row, m in zip(lp_np, mask_np):
                        captured.append(row[: int(m.sum())].astype(np.float32))
                if ref is not None and beta != 0.0 and iteration == cfg.num_iterations - 1:
                    lp_np = np.array(lp.tolist(), dtype=np.float64)
                    ref_np = np.array(ref.tolist(), dtype=np.float64)
                    mask_np = np.array(packed["mask"].tolist(), dtype=np.float64)
                    delta = ref_np - lp_np
                    kl = np.exp(delta) - delta - 1.0
                    kl_num += float((kl * mask_np).sum())
                    kl_den += float(mask_np.sum())
            # A NaN loss must raise even if clipping the same NaNs fails.
            raise_if_nonfinite(total, 0.0, step=logged_step, run_dir=cfg.out_dir)
            if accum is not None:
                clipped, total_norm = optim.clip_grad_norm(accum, cfg.grad_clip)
                mx.eval(total_norm)
                grad_norm = float(total_norm.item())
                loss_value = total
                # Before the update, and before the caller writes this step.
                raise_if_nonfinite(loss_value, grad_norm, step=logged_step, run_dir=cfg.out_dir)
                self.optimizer.update(self.model, clipped)
                self._step_committed = True
                mx.eval(self.model.parameters(), self.optimizer.state)
            else:
                grad_norm = 0.0
            loss_value = total
            if captured is not None:
                old_lps = captured
            if kl_den > 0.0:
                kl_value = kl_num / kl_den
        lr_value = self.optimizer.learning_rate
        if not isinstance(lr_value, mx.array):
            lr_value = mx.array(lr_value)
        self._step_committed = True
        return float(loss_value), float(grad_norm), kl_value, float(lr_value.item())

    def _cached_logprobs(
        self, model: nn.Module | None, sequences: list[dict[str, Any]]
    ) -> list[np.ndarray] | None:
        if model is None:
            return None
        model.eval()
        micro = self.config.micro_batch_size
        stored: list[np.ndarray] = []
        for start in range(0, len(sequences), micro):
            batch = sequences[start : start + micro]
            packed = _pack_batch(batch)
            if packed["full_ids"] is None:
                stored.extend(np.zeros((0,), dtype=np.float32) for _ in batch)
                continue
            lp = _completion_logprobs(
                model,
                packed["full_ids"],
                packed["prompt_lens"],
                int(packed["mask"].shape[1]),
                float(self.config.temperature),
            )
            mx.eval(lp)
            lp_np = np.array(lp.tolist(), dtype=np.float32)
            mask_np = np.array(packed["mask"].tolist())
            for row, m in zip(lp_np, mask_np):
                stored.append(row[: int(m.sum())].astype(np.float32))
        self.model.train()  # type: ignore[union-attr]
        return stored

    def _should_eval(self, step: int) -> bool:
        cfg = self.config
        return step == 0 or step == cfg.steps or step % cfg.eval_every == 0

    def _existing_eval_rows(self) -> dict[tuple[int, str], dict[str, Any]]:
        path = Path(self.config.out_dir) / "eval.jsonl"
        if not path.exists():
            return {}
        found: dict[tuple[int, str], dict[str, Any]] = {}
        for row in read_jsonl(path):
            found[(int(row["step"]), str(row["split"]))] = row
        return found

    def _evaluate(self, step: int) -> None:
        cfg = self.config
        assert self.model is not None
        splits = {
            "heldout": self.heldout_problems[: cfg.eval_heldout_n],
            "train": self.train_problems[: cfg.eval_train_n],
        }
        done = self._existing_eval_rows()
        pending = {
            split: problems
            for split, problems in splits.items()
            if problems and (step, split) not in done
        }
        for split, problems in splits.items():
            if not problems:
                self._last_eval[split] = None
            elif (step, split) in done:
                self._last_eval[split] = done[(step, split)].get("oracle_acc")
        if not pending:
            return
        gen = MLXGenerator.from_model(
            self.model,
            self.tokenizer,
            model_id=cfg.model_id,
            max_prompt_tokens=cfg.max_prompt_length,
        )
        for split, problems in pending.items():
            stats = evaluate_policy(
                gen,
                self.task,
                problems,
                n_samples=cfg.eval_samples_per_prompt,
                max_tokens=int(cfg.eval_max_tokens or cfg.max_tokens),
                temperature=cfg.eval_temperature,
                seed=stable_seed(cfg.seed, "eval", step, split),
                verifier=self.verifier,
                monitors=self.monitors,
            )
            self._last_eval[split] = stats["oracle_acc"]
            append_eval_samples(cfg.out_dir, step, split, stats)
            row, prompt_row = eval_log_rows(step, split, stats)
            append_jsonl(Path(cfg.out_dir) / "eval.jsonl", row)
            append_jsonl(Path(cfg.out_dir) / "eval_prompts.jsonl", prompt_row)
        del gen
        mx.clear_cache()

    def _checkpoint(self) -> None:
        assert self.model is not None and self.optimizer is not None and self.pool is not None
        path = checkpoint_dir(self.config.out_dir, self.completed)
        path.mkdir(parents=True, exist_ok=True)
        weights = dict(tree_flatten(self.model.trainable_parameters()))
        mx.eval(*weights.values())
        mx.save_safetensors(str(path / "adapters.safetensors"), weights)
        _save_optimizer(path / "optimizer.safetensors", self.optimizer.state)
        write_json(
            path / "adapter_config.json",
            {
                "fine_tune_type": "lora",
                "num_layers": self.config.lora.num_layers,
                "lora_parameters": {
                    "rank": self.config.lora.rank,
                    "scale": self.config.lora.scale,
                    "dropout": self.config.lora.dropout,
                    "keys": self.config.lora.mlx_keys(),
                },
            },
        )
        write_json(
            path / "trainer_state.json",
            {"step": self.completed, "data": self.pool.state(), "seed": self.config.seed},
        )

    def _load_checkpoint(self, path: Path) -> None:
        assert self.model is not None and self.optimizer is not None and self.pool is not None
        self.model.load_weights(str(path / "adapters.safetensors"), strict=False)
        mx.eval(self.model.parameters())
        opt_path = path / "optimizer.safetensors"
        if opt_path.exists():
            loaded = cast(dict[str, mx.array], mx.load(str(opt_path)))
            items = []
            for key, value in loaded.items():
                if key == "step" or key.endswith(".step"):
                    value = value.astype(mx.uint64)
                items.append((key, value))
            # Assignment clears ``_initialized``. The next update calls ``init``,
            # which keeps loaded moments and fills any leaf that did not round-trip.
            self.optimizer.state = tree_unflatten(items)
        state = read_json(path / "trainer_state.json")
        self.completed = int(state["step"])
        self.pool.load_state(state["data"])

    def _print_row(self, row: dict[str, Any]) -> None:
        kl = row["kl"]
        kl_s = "n/a" if kl is None else f"{kl:.4f}"
        print(
            f"step {row['step']} loss {row['loss']:.4f} V {row['reward_V_mean']:.3f} "
            f"O {row['oracle_mean']:.3f} gen {row['gen_seconds']:.2f}s "
            f"train {row['train_seconds']:.2f}s tokens {row['tokens_generated']} kl {kl_s}",
            flush=True,
        )


def checkpoint_dir(out_dir: str, step: int) -> Path:
    return Path(out_dir) / "checkpoints" / f"step_{step:07d}"


def latest_checkpoint(out_dir: str | Path) -> Path | None:
    root = Path(out_dir) / "checkpoints"
    if not root.exists():
        return None
    found = []
    for path in root.glob("step_*"):
        if (path / "trainer_state.json").exists() and (path / "adapters.safetensors").exists():
            found.append(path)
    if not found:
        return None
    return max(found, key=lambda path: int(path.name.split("_")[1]))


def _apply_mlx_lora_init(model: nn.Module, config: GRPOConfig) -> str:
    """Set each ``LoRALinear.lora_a`` from the init file. ``lora_a`` is A's transpose.

    mlx_lm 0.28.3 ``LoRALinear`` stores ``lora_a`` with shape ``(in_features, r)``
    and ``lora_b`` with shape ``(r, out_features)`` (``mlx_lm/tuner/lora.py``).
    The file stores PEFT's ``lora_A`` layout, ``(r, in_features)``.
    """
    import numpy as np

    from vfates.features.lora_init import load_lora_init

    if config.lora_init is None:
        raise ValueError("lora_init path is required")
    tensors, _meta = load_lora_init(config.lora_init, expected_sha256=config.lora_init_sha256)
    modules = {
        name: module for name, module in model.named_modules() if isinstance(module, LoRALinear)
    }
    if not modules:
        raise RuntimeError("LoRA adaptation produced no LoRALinear modules")
    got = set(tensors)
    want = set(modules)
    if got != want:
        missing = sorted(want - got)
        extra = sorted(got - want)
        raise ValueError(
            "lora init keys must equal the wrapped modules: "
            f"missing {missing[:4]}, extra {extra[:4]}, "
            f"{len(want)} wrapped, {len(got)} in the file"
        )
    pending: list[tuple[LoRALinear, np.ndarray]] = []
    for key, module in modules.items():
        array = np.asarray(tensors[key])
        if array.dtype != np.float32:
            raise ValueError(f"{key} must be float32, got {array.dtype}")
        transposed = np.ascontiguousarray(array.T)
        if tuple(transposed.shape) != tuple(module.lora_a.shape):
            raise ValueError(
                f"{key} transpose shape {tuple(transposed.shape)} != lora_a {tuple(module.lora_a.shape)}"
            )
        if float(module.scale) != float(config.lora.scale):
            raise RuntimeError(
                f"{key} LoRA scale {float(module.scale)} != alpha/rank {config.lora.scale}"
            )
        bias = np.asarray(module.lora_b)
        if np.any(bias != 0):
            raise RuntimeError(f"lora_b for {key} is not zero; refusing to load A on top of it")
        pending.append((module, transposed))
    for module, transposed in pending:
        module.lora_a = mx.array(transposed)
    mx.eval(*[module.lora_a for module, _array in pending])
    if config.lora_init_sha256 is not None:
        return config.lora_init_sha256
    return hashlib.sha256(Path(config.lora_init).read_bytes()).hexdigest()


def constant_with_warmup_lr(step: float, base: float, warmup: int) -> float:
    """Learning rate used on optimizer update ``step`` (0-indexed).

    Matches Hugging Face ``get_constant_schedule_with_warmup``: ``base * step / warmup``
    while ``step < warmup``, then ``base``. The first update uses learning rate 0.
    """
    if warmup <= 0:
        return float(base)
    return float(base) * min(float(step), float(warmup)) / float(warmup)


def _make_optimizer(config: GRPOConfig) -> optim.AdamW:
    """AdamW with PyTorch-style bias correction. Warmup matches HF's used learning rate."""
    if config.warmup_steps > 0:
        warmup = int(config.warmup_steps)
        base = float(config.lr)

        def schedule(step: mx.array) -> mx.array:
            # MLX evaluates this at the pre-increment step, which is 0, 1, 2, ...
            # HF's LambdaLR uses that same index, so update 0 multiplies by 0.
            capped = mx.minimum(step.astype(mx.float32), mx.array(float(warmup), dtype=mx.float32))
            return (
                mx.array(base, dtype=mx.float32)
                * capped
                / mx.array(float(warmup), dtype=mx.float32)
            )

        lr: Any = schedule
    else:
        lr = float(config.lr)
    return optim.AdamW(
        learning_rate=lr,
        betas=[float(config.adam_betas[0]), float(config.adam_betas[1])],
        weight_decay=float(config.weight_decay),
        bias_correction=True,
    )


def _clone_frozen(model: nn.Module) -> nn.Module:
    cloned = type(model)(model.args)
    params = cast(list[tuple[str, Any]], tree_flatten(model.parameters()))
    cloned.load_weights([(key, mx.array(value)) for key, value in params])
    mx.eval(cloned.parameters())
    cloned.freeze()
    cloned.eval()
    return cloned


def _enable_grad_checkpoint(model: nn.Module) -> None:
    from mlx_lm.tuner.trainer import grad_checkpoint

    layer = model.layers[0]
    if getattr(type(layer), "_vfates_ckpt", False):
        return
    grad_checkpoint(layer)
    type(layer)._vfates_ckpt = True


def _anchor_loss(model: nn.Module) -> mx.array:
    """A zero that still depends on a trainable parameter, so value_and_grad is defined."""
    leaf = cast(list[tuple[str, Any]], tree_flatten(model.trainable_parameters()))[0][1]
    return leaf.reshape(-1)[0] * mx.array(0.0, dtype=leaf.dtype)


def _pack_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [len(seq["token_ids"]) for seq in batch]
    max_c = max(lengths) if lengths else 0
    adv = mx.array(np.array([seq["advantage"] for seq in batch], dtype=np.float32))
    weights = mx.array(np.array([seq["weight"] for seq in batch], dtype=np.float32))
    if max_c == 0:
        return {
            "full_ids": None,
            "prompt_lens": None,
            "mask": mx.zeros((len(batch), 1), dtype=mx.float32),
            "adv": adv,
            "weights": weights,
            "max_c": 0,
        }
    prompt_lens = [len(seq["prompt_ids"]) for seq in batch]
    fulls = []
    for seq, n_comp in zip(batch, lengths):
        tokens = list(seq["prompt_ids"]) + list(seq["token_ids"]) + [0] * (max_c - n_comp)
        fulls.append(tokens)
    max_t = max(len(row) for row in fulls)
    padded = [row + [0] * (max_t - len(row)) for row in fulls]
    mask = np.zeros((len(batch), max_c), dtype=np.float32)
    for i, n_comp in enumerate(lengths):
        mask[i, :n_comp] = 1.0
    return {
        "full_ids": mx.array(np.array(padded, dtype=np.int32)),
        "prompt_lens": mx.array(np.array(prompt_lens, dtype=np.int32)),
        "mask": mx.array(mask),
        "adv": adv,
        "weights": weights,
        "max_c": max_c,
    }


def _pad_stored(rows: list[np.ndarray], max_c: int) -> mx.array:
    if max_c <= 0:
        return mx.zeros((len(rows), 1), dtype=mx.float32)
    out = np.zeros((len(rows), max_c), dtype=np.float32)
    for i, row in enumerate(rows):
        n = min(len(row), max_c)
        if n:
            out[i, :n] = row[:n]
    return mx.array(out)


def _save_optimizer(path: Path, state: dict[str, Any]) -> None:
    flat: dict[str, mx.array] = {}
    for key, value in cast(list[tuple[str, Any]], tree_flatten(state)):
        if not isinstance(value, mx.array):
            continue
        if value.dtype == mx.uint64:
            value = value.astype(mx.uint32)
        flat[key] = value
    if flat:
        mx.eval(*flat.values())
        mx.save_safetensors(str(path), flat)
