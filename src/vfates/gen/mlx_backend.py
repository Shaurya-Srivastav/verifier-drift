"""MLX generation backend built on ``mlx_lm``'s ``BatchGenerator``.

``BatchGenerator`` (mlx_lm 0.28.3) returns one token at a time. ``finish_reason``
is ``"stop"``, ``"length"``, or ``None`` while the sequence is still open.
``batch_generate`` drops the stop token (``generate.py`` keeps a response only
when ``finish_reason != "stop"``). This backend keeps it: ``GenOut.token_ids``
include that id, ``n_tokens`` counts it, and ``text`` is
``decode(..., skip_special_tokens=True)`` so the reward string does not.

Log-probability semantics: each response carries the full-vocabulary log-softmax
*before* temperature and top-p (see ``BatchGenerator._step``). We store the entry
at the sampled token id and sum those values. The sum is ``log π(y | x)`` under
the untempered model, not the log-density of the tempered sampler. Temperature 0
uses ``make_sampler(temp=0)`` (argmax). ``top_p=1`` disables nucleus filtering.

Sampling is deterministic given ``(prompts, n, max_tokens, temperature, top_p,
seed, batch sizes)``. Each chunk of ``max_sequences_per_call`` sequences is seeded
with ``stable_seed(seed, chunk_index)``, so changing the chunk size changes the
stream. The generator is not thread-safe.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import mlx.core as mx
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler

from vfates.gen.base import Generator, truncate_prompt_ids
from vfates.gen.chat import encode_prompt as encode_chat_prompt
from vfates.types import GenOut
from vfates.utils.hashing import stable_seed

_WIRED = False


def wire_memory() -> None:
    """Raise the Metal wired limit once, so generation can use unified memory."""
    global _WIRED
    if _WIRED or not mx.metal.is_available():
        return
    limit = mx.metal.device_info().get("max_recommended_working_set_size")
    if limit:
        mx.set_wired_limit(int(limit))
    _WIRED = True


def eos_ids(tokenizer: object) -> set[int]:
    """EOS ids for either an mlx_lm ``TokenizerWrapper`` or a HF tokenizer."""
    raw = getattr(tokenizer, "eos_token_ids", None)
    if raw:
        if isinstance(raw, int):
            return {int(raw)}
        return {int(x) for x in raw}
    single = getattr(tokenizer, "eos_token_id", None)
    if single is None:
        return set()
    return {int(single)}


def encode_prompt(
    tokenizer: object,
    user: str,
    system_prompt: str | None = None,
    max_prompt_tokens: int | None = None,
    *,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> list[int]:
    """Chat-template encode. Longer prompts keep the suffix (TRL left-truncates)."""
    ids = truncate_prompt_ids(
        encode_chat_prompt(
            tokenizer, user, system_prompt, chat_template_kwargs=chat_template_kwargs
        ),
        max_prompt_tokens,
    )
    if not ids:
        raise ValueError("Chat template produced an empty token list")
    return ids


def _copy_template_kwargs(kwargs: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not kwargs:
        return None
    return {str(key): value for key, value in kwargs.items()}


@dataclass(frozen=True)
class IdOut:
    """One completion from :meth:`MLXGenerator.generate_ids`.

    ``token_logprobs`` aligns with ``token_ids``, including the stop id when the
    sample stopped on a token. ``logprob`` is their sum, or ``0.0`` when no token
    was produced.
    """

    token_ids: tuple[int, ...]
    token_logprobs: tuple[float, ...]
    logprob: float | None
    finish_reason: str
    n_tokens: int


class MLXGenerator(Generator):
    """Apple-silicon generator. Construct from a repo id or wrap a live module.

    With ``logprobs=True``, ``GenOut.logprob`` / ``IdOut.logprob`` is the sum of
    untempered next-token log-probabilities of the returned ids, including the
    stop id. It is the model's log π, not the tempered sampling density. An
    immediate EOS is a length-1 sequence whose text decodes empty. Collecting
    logprobs costs about 40% of throughput, so the default is off (the trainer
    recomputes log-probabilities in its own forward pass). Draws are deterministic
    for a fixed
    ``(prompts, n, max_tokens, temperature, top_p, seed, batch sizes)``.
    ``temperature=0`` is greedy. Changing ``max_sequences_per_call`` changes the
    stream because each chunk is seeded with ``stable_seed(seed, chunk_index)``.
    """

    def __init__(
        self,
        model_id: str,
        adapter_path: str | None = None,
        completion_batch_size: int = 128,
        prefill_batch_size: int = 16,
        max_sequences_per_call: int = 512,
        max_prompt_tokens: int | None = None,
        logprobs: bool = False,
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        from mlx_lm import load

        wire_memory()
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.completion_batch_size = int(completion_batch_size)
        self.prefill_batch_size = int(prefill_batch_size)
        self.max_sequences_per_call = int(max_sequences_per_call)
        self.max_prompt_tokens = max_prompt_tokens
        self.logprobs = bool(logprobs)
        self.chat_template_kwargs = _copy_template_kwargs(chat_template_kwargs)
        self._check_batch_sizes()
        self.model, self.tokenizer = load(model_id, adapter_path=adapter_path)
        self.model.eval()

    @classmethod
    def from_model(
        cls, model: object, tokenizer: object, model_id: str = "in-memory", **kw: object
    ) -> MLXGenerator:
        """Wrap an in-memory module (typically already LoRA-adapted). Does not reload."""
        wire_memory()
        obj = cls.__new__(cls)
        obj.model = model
        obj.tokenizer = tokenizer
        obj.model_id = model_id
        obj.adapter_path = cast(str | None, kw.get("adapter_path"))
        obj.completion_batch_size = int(cast(Any, kw.get("completion_batch_size", 128)))
        obj.prefill_batch_size = int(cast(Any, kw.get("prefill_batch_size", 16)))
        obj.max_sequences_per_call = int(cast(Any, kw.get("max_sequences_per_call", 512)))
        obj.max_prompt_tokens = cast(int | None, kw.get("max_prompt_tokens"))
        obj.logprobs = bool(kw.get("logprobs", False))
        obj.chat_template_kwargs = _copy_template_kwargs(
            cast(Mapping[str, Any] | None, kw.get("chat_template_kwargs"))
        )
        obj._check_batch_sizes()
        return obj

    def _check_batch_sizes(self) -> None:
        if self.completion_batch_size < 1 or self.prefill_batch_size < 1:
            raise ValueError("completion_batch_size and prefill_batch_size must be >= 1")
        if self.max_sequences_per_call < 1:
            raise ValueError("max_sequences_per_call must be >= 1")
        # BatchGenerator only prefills while completion slots >= prefill_batch_size.
        if self.prefill_batch_size > self.completion_batch_size:
            self.prefill_batch_size = self.completion_batch_size

    def describe(self) -> dict[str, object]:
        info = super().describe()
        info.update(
            {
                "adapter_path": self.adapter_path,
                "completion_batch_size": self.completion_batch_size,
                "prefill_batch_size": self.prefill_batch_size,
                "max_sequences_per_call": self.max_sequences_per_call,
                "max_prompt_tokens": self.max_prompt_tokens,
                "logprobs": self.logprobs,
            }
        )
        return info

    def generate(
        self,
        prompts: list[str],
        n: int,
        *,
        max_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int = 0,
        system_prompt: str | None = None,
    ) -> list[list[GenOut]]:
        """``n`` completions per prompt, prompt-major order."""
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if not prompts:
            return []
        flat: list[list[int]] = []
        for prompt in prompts:
            ids = encode_prompt(
                self.tokenizer,
                prompt,
                system_prompt,
                self.max_prompt_tokens,
                chat_template_kwargs=self.chat_template_kwargs,
            )
            flat.extend([ids] * n)
        outs = self.generate_ids(
            flat, max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=seed
        )
        grouped: list[list[GenOut]] = []
        for i in range(len(prompts)):
            group = []
            for out in outs[i * n : (i + 1) * n]:
                group.append(
                    GenOut(
                        text=self._decode(out.token_ids),
                        n_tokens=out.n_tokens,
                        token_ids=out.token_ids,
                        logprob=out.logprob,
                        finish_reason=out.finish_reason,
                    )
                )
            grouped.append(group)
        return grouped

    def generate_ids(
        self,
        prompt_ids: list[list[int]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> list[IdOut]:
        """Sample one continuation per already-tokenized prompt.

        The trainer calls this with the chat template already applied. Seeds are
        ``stable_seed(seed, chunk_index)``.
        """
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        if temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        if not prompt_ids:
            return []
        wire_memory()
        was_training = bool(getattr(self.model, "training", False))
        self.model.eval()
        try:
            return self._generate_ids(prompt_ids, max_tokens, temperature, top_p, seed)
        finally:
            if was_training:
                self.model.train()

    def _generate_ids(
        self,
        prompt_ids: list[list[int]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> list[IdOut]:
        sampler = make_sampler(temp=float(temperature), top_p=float(top_p))
        stop = eos_ids(self.tokenizer)
        outs: list[IdOut | None] = [None] * len(prompt_ids)
        chunk = self.max_sequences_per_call
        for chunk_index, start in enumerate(range(0, len(prompt_ids), chunk)):
            piece = prompt_ids[start : start + chunk]
            mx.random.seed(stable_seed(int(seed), chunk_index))
            gen = BatchGenerator(
                self.model,
                max_tokens=max_tokens,
                stop_tokens=stop,
                sampler=sampler,
                completion_batch_size=self.completion_batch_size,
                prefill_batch_size=self.prefill_batch_size,
            )
            uids = gen.insert([list(map(int, ids)) for ids in piece], max_tokens)
            buckets: dict[int, list[tuple[int, float]]] = {uid: [] for uid in uids}
            reasons: dict[int, str] = {}
            while responses := gen.next():
                scalars = []
                meta = []
                for response in responses:
                    # Record the stop token. mlx_lm's batch_generate drops it;
                    # the loss and n_tokens follow TRL and keep the first EOS.
                    if self.logprobs:
                        scalars.append(response.logprobs[response.token])
                    meta.append((response.uid, int(response.token), response.finish_reason))
                values: list[float] = [float("nan")] * len(meta)
                if scalars:
                    stacked = mx.stack(scalars).astype(mx.float32)
                    mx.eval(stacked)
                    values = [float(v) for v in cast(list[float], stacked.tolist())]
                for (uid, token, reason), logp in zip(meta, values, strict=True):
                    buckets[uid].append((token, logp))
                    if reason is not None:
                        reasons[uid] = reason
            for local_i, uid in enumerate(uids):
                pairs = buckets[uid]
                token_ids = tuple(token for token, _ in pairs)
                token_logprobs = tuple(logp for _, logp in pairs) if self.logprobs else ()
                outs[start + local_i] = IdOut(
                    token_ids=token_ids,
                    token_logprobs=token_logprobs,
                    logprob=float(sum(token_logprobs)) if self.logprobs else None,
                    finish_reason=reasons.get(uid, "length"),
                    n_tokens=len(token_ids),
                )
            del gen
            mx.clear_cache()
        missing = [i for i, out in enumerate(outs) if out is None]
        if missing:
            raise RuntimeError(f"generation produced no result for sequences {missing[:8]}")
        return [out for out in outs if out is not None]

    def _decode(self, token_ids: tuple[int, ...]) -> str:
        if not token_ids:
            return ""
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)
