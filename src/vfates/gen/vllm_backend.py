"""vLLM generation backend (Linux + CUDA).

vLLM is imported only when a :class:`VLLMGenerator` is constructed. The request
and response helpers below are pure Python so they can be tested without it.

Stop tokens in ``CompletionOutput.token_ids`` (vLLM 0.10.2, the release TRL
0.25.1 pins). Both engines include the stopping token id and only strip it
from the detokenized text when ``include_stop_str_in_output`` is false, which
is the default.

* V1 detokenizer: on a STOP finish it skips detokenizing the last id, then
  appends that id to ``token_ids`` anyway.
  https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/v1/engine/detokenizer.py#L104-L126
* V0 ``StopChecker``: an EOS or ``stop_token_ids`` hit removes the token from
  ``output_text`` and leaves it in the sequence returned by
  ``get_output_token_ids_to_return``.
  https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/engine/output_processor/stop_checker.py#L49-L69
* ``SamplingParams`` documents that returned text contains ``stop_token_ids``
  unless those ids are special tokens. ``update_from_generation_config`` adds
  every generation-config EOS id except the primary one into ``stop_token_ids``
  (Qwen lists several).
  https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/sampling_params.py#L147-L150
  https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/sampling_params.py#L447-L468

:func:`enforce_stop_token_ids` still applies the vfates convention either way:
keep the first stop id, append an integer ``stop_reason`` (or the sole EOS id)
when a stop finish did not already end on one, drop padding, and decode
``text`` with ``skip_special_tokens=True``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vfates.gen.base import Generator, truncate_prompt_ids
from vfates.gen.chat import check_generate_args, encode_prompt
from vfates.types import GenOut

_VLLM_MISSING = (
    "vLLM is not installed, so VLLMGenerator cannot be constructed. "
    "vLLM runs on Linux with CUDA only, not on macOS or MPS. "
    "Install it with `pip install vllm` or `pip install 'vfates[gpu]'`."
)


def sampling_params_kwargs(
    *,
    n: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    logprobs: int = 1,
) -> dict[str, Any]:
    """Keyword arguments for ``vllm.SamplingParams``.

    ``top_k=-1`` is vLLM's off switch (Hugging Face uses ``top_k=0``).
    ``logprobs=1`` is what makes ``CompletionOutput.cumulative_logprob`` populate;
    the extra top-1 logprob is not used. ``repetition_penalty=1`` keeps the
    sampling distribution free of the penalty some chat checkpoints ship with.
    """
    return {
        "n": int(n),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "seed": int(seed),
        "logprobs": int(logprobs),
        "top_k": -1,
        "repetition_penalty": 1.0,
    }


def token_prompts(
    tokenizer: Any,
    prompts: list[str],
    system_prompt: str | None = None,
    max_prompt_tokens: int | None = None,
) -> list[dict[str, list[int]]]:
    """vLLM token-prompt dicts, one per user string, using :func:`encode_prompt`.

    Prompts longer than ``max_prompt_tokens`` keep the suffix.
    """
    return [
        {
            "prompt_token_ids": truncate_prompt_ids(
                encode_prompt(tokenizer, prompt, system_prompt), max_prompt_tokens
            )
        }
        for prompt in prompts
    ]


def collect_stop_ids(tokenizer: Any, generation_config: Any = None) -> set[int]:
    """EOS ids from a tokenizer and, when present, a generation config."""
    found: set[int] = set()

    def _take(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                _take(item)
            return
        found.add(int(value))

    _take(getattr(tokenizer, "eos_token_id", None))
    _take(getattr(tokenizer, "eos_token_ids", None))
    if generation_config is not None:
        _take(getattr(generation_config, "eos_token_id", None))
        if isinstance(generation_config, dict):
            _take(generation_config.get("eos_token_id"))
    return found


def enforce_stop_token_ids(
    token_ids: Sequence[int],
    *,
    finish_reason: str | None,
    stop_reason: Any = None,
    stop_ids: set[int] | None = None,
    pad_id: int | None = None,
) -> tuple[int, ...]:
    """Completion ids with a token stop kept and padding removed.

    A ``"stop"`` finish that already ends on a stop id is truncated after the
    first such id. If the id was stripped, an integer ``stop_reason`` is
    appended, or the sole member of ``stop_ids`` when the reason is not an id.
    Several EOS ids and no integer reason are left unchanged: vLLM 0.10.2
    already included the sampled id, and a stripped id would be ambiguous.
    """
    stop = {int(item) for item in (stop_ids or ())}
    pad = None if pad_id is None else int(pad_id)
    reason = None if finish_reason is None else str(finish_reason)
    explicit = stop_reason if type(stop_reason) is int else None
    ids: list[int] = []
    if reason == "stop":
        for token in token_ids:
            token = int(token)
            if pad is not None and token == pad and token not in stop and token != explicit:
                continue
            ids.append(token)
            if token in stop or (explicit is not None and token == explicit):
                break
        ends = bool(ids) and (ids[-1] in stop or (explicit is not None and ids[-1] == explicit))
        if not ends:
            if explicit is not None:
                ids.append(explicit)
            elif len(stop) == 1:
                ids.append(next(iter(stop)))
    else:
        for token in token_ids:
            token = int(token)
            if pad is not None and token == pad:
                continue
            ids.append(token)
    return tuple(ids)


def request_outputs_to_genouts(
    request_outputs: list[Any],
    n: int | None = None,
    *,
    tokenizer: Any = None,
    stop_ids: set[int] | None = None,
    pad_id: int | None = None,
) -> list[list[GenOut]]:
    """Map vLLM ``RequestOutput`` objects (or fakes with the same fields) to ``GenOut``.

    Each ``request.outputs[i]`` must expose ``text``, ``token_ids``,
    ``cumulative_logprob`` and ``finish_reason``. Optional ``stop_reason`` is
    read when present. Token ids go through :func:`enforce_stop_token_ids`.
    The cumulative log-probability is kept only when the id list is unchanged,
    because it refers to the ids vLLM returned. With a tokenizer, ``text`` is
    ``decode(token_ids, skip_special_tokens=True)``.
    """
    groups: list[list[GenOut]] = []
    for request in request_outputs:
        completions = list(request.outputs)
        if n is not None and len(completions) != n:
            raise RuntimeError(f"expected {n} completions per prompt, got {len(completions)}")
        group: list[GenOut] = []
        for completion in completions:
            raw_ids = tuple(int(token) for token in completion.token_ids)
            finish = None if completion.finish_reason is None else str(completion.finish_reason)
            token_ids = enforce_stop_token_ids(
                raw_ids,
                finish_reason=finish,
                stop_reason=getattr(completion, "stop_reason", None),
                stop_ids=stop_ids,
                pad_id=pad_id,
            )
            raw_logprob = completion.cumulative_logprob
            if tokenizer is None:
                text = str(completion.text)
            else:
                text = tokenizer.decode(list(token_ids), skip_special_tokens=True)
            group.append(
                GenOut(
                    text=text,
                    n_tokens=len(token_ids),
                    token_ids=token_ids,
                    logprob=(
                        None if raw_logprob is None or token_ids != raw_ids else float(raw_logprob)
                    ),
                    finish_reason=finish,
                )
            )
        groups.append(group)
    return groups


class VLLMGenerator(Generator):
    """Sample with vLLM. ``engine_kwargs`` are forwarded to ``vllm.LLM``.

    :meth:`set_adapter` routes later calls through a PEFT LoRA checkpoint; the
    engine must be built with ``enable_lora=True`` and a ``max_lora_rank`` at
    least the adapter's rank.
    """

    def __init__(
        self, model_id: str, max_prompt_tokens: int | None = None, **engine_kwargs: Any
    ) -> None:
        self.model_id = model_id
        self.max_prompt_tokens = max_prompt_tokens
        self.engine_kwargs = dict(engine_kwargs)
        self.adapter: str | None = None
        self._lora: Any = None
        self._lora_ids: dict[str, int] = {}
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError(_VLLM_MISSING) from exc
        self._llm = LLM(model=model_id, **self.engine_kwargs)

    def set_adapter(self, path: str | None) -> None:
        """Generate with the LoRA adapter at ``path`` (``None``: the base model)."""
        self.adapter = None if path is None else str(path)
        if path is None:
            self._lora = None
            return
        if not self.engine_kwargs.get("enable_lora"):
            raise ValueError("build VLLMGenerator with enable_lora=True to load adapters")
        from vllm.lora.request import LoRARequest

        lora_id = self._lora_ids.setdefault(str(path), len(self._lora_ids) + 1)
        self._lora = LoRARequest(f"adapter{lora_id}", lora_id, str(path))

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "vllm",
            "model_id": self.model_id,
            "engine_kwargs": dict(self.engine_kwargs),
            "adapter": self.adapter,
            "max_prompt_tokens": self.max_prompt_tokens,
        }

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
        """``n`` completions per prompt. One ``SamplingParams.seed`` covers the call.

        vLLM derives the ``n`` per-prompt samples from that seed internally. This
        is a single engine batch, not the chunked seeds used by :class:`HFGenerator`.
        """
        check_generate_args(n, max_tokens, temperature, top_p)
        if not prompts:
            return []
        from vllm import SamplingParams

        tokenizer = self._llm.get_tokenizer()
        generation_config = getattr(
            getattr(self._llm, "llm_engine", None), "generation_config", None
        )
        if generation_config is None:
            generation_config = getattr(self._llm, "generation_config", None)
        params = SamplingParams(
            **sampling_params_kwargs(
                n=n,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed,
            )
        )
        outputs = self._llm.generate(
            token_prompts(tokenizer, prompts, system_prompt, self.max_prompt_tokens),
            params,
            lora_request=self._lora,
        )
        pad_id = getattr(tokenizer, "pad_token_id", None)
        return request_outputs_to_genouts(
            outputs,
            n=n,
            tokenizer=tokenizer,
            stop_ids=collect_stop_ids(tokenizer, generation_config),
            pad_id=None if pad_id is None else int(pad_id),
        )
