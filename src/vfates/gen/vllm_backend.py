"""vLLM generation backend (Linux + CUDA).

vLLM is imported only when a :class:`VLLMGenerator` is constructed. The request
and response helpers below are pure Python so they can be tested without it.
"""

from __future__ import annotations

from typing import Any

from vfates.gen.base import Generator
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
    tokenizer: Any, prompts: list[str], system_prompt: str | None = None
) -> list[dict[str, list[int]]]:
    """vLLM token-prompt dicts, one per user string, using :func:`encode_prompt`."""
    return [
        {"prompt_token_ids": encode_prompt(tokenizer, prompt, system_prompt)} for prompt in prompts
    ]


def request_outputs_to_genouts(
    request_outputs: list[Any], n: int | None = None
) -> list[list[GenOut]]:
    """Map vLLM ``RequestOutput`` objects (or fakes with the same fields) to ``GenOut``.

    Each ``request.outputs[i]`` must expose ``text``, ``token_ids``,
    ``cumulative_logprob`` and ``finish_reason``. Token ids and the cumulative
    log-probability are copied through unchanged: vLLM's cumulative log-probability
    refers to the token ids it returns, which may include the stop token.
    """
    groups: list[list[GenOut]] = []
    for request in request_outputs:
        completions = list(request.outputs)
        if n is not None and len(completions) != n:
            raise RuntimeError(f"expected {n} completions per prompt, got {len(completions)}")
        group: list[GenOut] = []
        for completion in completions:
            token_ids = tuple(int(token) for token in completion.token_ids)
            raw_logprob = completion.cumulative_logprob
            group.append(
                GenOut(
                    text=str(completion.text),
                    n_tokens=len(token_ids),
                    token_ids=token_ids,
                    logprob=None if raw_logprob is None else float(raw_logprob),
                    finish_reason=(
                        None if completion.finish_reason is None else str(completion.finish_reason)
                    ),
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

    def __init__(self, model_id: str, **engine_kwargs: Any) -> None:
        self.model_id = model_id
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
            token_prompts(tokenizer, prompts, system_prompt),
            params,
            lora_request=self._lora,
        )
        return request_outputs_to_genouts(outputs, n=n)
