"""Hugging Face generation on CUDA, MPS or CPU.

Prompts are left-padded so that, within a batch, every generated token is a
suffix of ``sequences`` at a shared prompt width. ``GenOut.logprob`` is the sum
of per-token log-probabilities under the distribution that was actually sampled
(``log_softmax`` of the processed scores), not under a differently tempered
model. With ``temperature == 1`` and ``top_p == 1`` that distribution is the
model's. ``temperature == 0`` is greedy (``do_sample`` is false) and ignores
the seed.

Chat checkpoints often ship a generation config with ``temperature``, ``top_p``,
``top_k`` and ``repetition_penalty`` set. Those are not the on-policy audit
distribution, so generation is called with ``use_model_defaults=False`` and
explicit ``repetition_penalty=1``, ``top_k=0`` when sampling.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from vfates.gen.base import Generator
from vfates.gen.chat import check_generate_args, encode_prompt
from vfates.types import GenOut
from vfates.utils.hashing import stable_seed

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_torch(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device.type == "mps" and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def _collect_ids(value: Any, into: list[int]) -> None:
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_ids(item, into)
        return
    item = int(value)
    if item not in into:
        into.append(item)


def _stop_ids(tokenizer: Any, model: Any) -> list[int]:
    ids: list[int] = []
    _collect_ids(getattr(tokenizer, "eos_token_id", None), ids)
    config = getattr(model, "config", None)
    if config is not None:
        _collect_ids(getattr(config, "eos_token_id", None), ids)
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        _collect_ids(getattr(generation_config, "eos_token_id", None), ids)
    if not ids:
        raise ValueError("tokenizer and model have no eos token id")
    return ids


def _left_pad(sequences: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(seq) for seq in sequences)
    batch = len(sequences)
    input_ids = torch.full((batch, width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch, width), dtype=torch.long)
    for row, seq in enumerate(sequences):
        if not seq:
            raise ValueError("refusing to generate from an empty tokenised prompt")
        input_ids[row, width - len(seq) :] = torch.tensor(seq, dtype=torch.long)
        attention_mask[row, width - len(seq) :] = 1
    return input_ids, attention_mask


def _decode_rows(
    token_rows: torch.Tensor,
    step_logprobs: torch.Tensor,
    stop_ids: list[int],
    tokenizer: Any,
) -> list[GenOut]:
    """Strip each row at the first stop id. ``step_logprobs`` is ``[steps, batch]``."""
    stop = set(stop_ids)
    rows: list[GenOut] = []
    for batch_index in range(token_rows.shape[0]):
        tokens: list[int] = []
        kept: list[torch.Tensor] = []
        reason = "length"
        for step in range(token_rows.shape[1]):
            token = int(token_rows[batch_index, step])
            if token in stop:
                reason = "stop"
                break
            tokens.append(token)
            kept.append(step_logprobs[step, batch_index])
        logprob = float(torch.stack(kept).sum().item()) if kept else 0.0
        # Default cleanup deletes spaces that belong to the token text.
        text = tokenizer.decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        rows.append(
            GenOut(
                text=text,
                n_tokens=len(tokens),
                token_ids=tuple(tokens),
                logprob=logprob,
                finish_reason=reason,
            )
        )
    return rows


class HFGenerator(Generator):
    """Batched Hugging Face sampling.

    Parameters
    ----------
    device:
        ``None`` selects CUDA, then MPS, then CPU. A model passed in is moved
        to that device.
    dtype:
        ``bfloat16``, ``float16`` or ``float32``. Applied when the model is
        loaded or passed in.
    batch_size:
        Sequences per ``generate`` call, after each prompt has been replicated
        ``n`` times. The seed for a chunk is ``stable_seed(seed, chunk_index)``,
        so the same ``batch_size`` is part of the reproducible experiment.
    """

    def __init__(
        self,
        model_id: str,
        device: str | torch.device | None = None,
        dtype: str = "bfloat16",
        batch_size: int = 32,
        model: Any = None,
        tokenizer: Any = None,
        local_files_only: bool = False,
    ) -> None:
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {dtype!r}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.model_id = model_id
        self.device = _resolve_device(device)
        self.dtype_name = dtype
        self.dtype = _DTYPES[dtype]
        self.batch_size = int(batch_size)
        self.local_files_only = local_files_only
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is None:
                raise ValueError(f"{model_id} tokenizer has no pad token and no eos token")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=self.dtype,
                local_files_only=local_files_only,
            )
        self.model = model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        if getattr(self.model.config, "is_encoder_decoder", False):
            raise ValueError("HFGenerator supports decoder-only causal LMs")
        self.stop_ids = _stop_ids(self.tokenizer, self.model)
        self.adapter: str | None = None
        self._adapters: dict[str, str] = {}

    def set_adapter(self, path: str | None) -> None:
        """Generate with the PEFT LoRA checkpoint at ``path`` (``None``: the base model)."""
        from peft import PeftModel

        if path is None:
            if isinstance(self.model, PeftModel):
                self.model.base_model.disable_adapter_layers()  # type: ignore[operator]  # peft stub
            self.adapter = None
            return
        name = self._adapters.setdefault(str(path), f"adapter{len(self._adapters)}")
        if not isinstance(self.model, PeftModel):
            self.model = PeftModel.from_pretrained(self.model, str(path), adapter_name=name)
        elif name not in self.model.peft_config:
            self.model.load_adapter(str(path), adapter_name=name)
        self.model.set_adapter(name)
        self.model.base_model.enable_adapter_layers()  # type: ignore[operator]  # peft stub
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        self.adapter = str(path)

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "hf",
            "model_id": self.model_id,
            "device": str(self.device),
            "dtype": self.dtype_name,
            "batch_size": self.batch_size,
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
        check_generate_args(n, max_tokens, temperature, top_p)
        if not prompts:
            return []
        encoded = [encode_prompt(self.tokenizer, prompt, system_prompt) for prompt in prompts]
        flat = [ids for ids in encoded for _ in range(n)]
        generated: list[GenOut] = []
        pad_id = int(self.tokenizer.pad_token_id)
        do_sample = temperature > 0
        with torch.no_grad():
            for chunk_index, start in enumerate(range(0, len(flat), self.batch_size)):
                chunk = flat[start : start + self.batch_size]
                _seed_torch(stable_seed(seed, chunk_index), self.device)
                generated.extend(
                    self._generate_chunk(chunk, max_tokens, do_sample, temperature, top_p, pad_id)
                )
        grouped: list[list[GenOut]] = [[] for _ in prompts]
        for index, completion in enumerate(generated):
            grouped[index // n].append(completion)
        return grouped

    def _generate_chunk(
        self,
        sequences: list[list[int]],
        max_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        pad_id: int,
    ) -> list[GenOut]:
        input_ids, attention_mask = _left_pad(sequences, pad_id)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        prompt_width = input_ids.shape[1]
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "repetition_penalty": 1.0,
            "renormalize_logits": False,
            "pad_token_id": pad_id,
            "eos_token_id": self.stop_ids if len(self.stop_ids) > 1 else self.stop_ids[0],
            "return_dict_in_generate": True,
            "output_scores": True,
            "use_cache": True,
            "use_model_defaults": False,
            "generation_config": GenerationConfig(),
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(top_p)
            gen_kwargs["top_k"] = 0
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
        if outputs.scores is None:
            raise RuntimeError("generate did not return scores; output_scores=True is required")
        new_tokens = outputs.sequences[:, prompt_width:]
        if len(outputs.scores) != new_tokens.shape[1]:
            raise RuntimeError(
                f"score steps ({len(outputs.scores)}) != generated width ({new_tokens.shape[1]})"
            )
        step_logprobs = []
        for step, scores in enumerate(outputs.scores):
            chosen = new_tokens[:, step : step + 1]
            log_probs = torch.log_softmax(scores, dim=-1)
            step_logprobs.append(log_probs.gather(1, chosen).squeeze(1))
        stacked = torch.stack(step_logprobs, dim=0).detach().cpu()
        return _decode_rows(new_tokens.detach().cpu(), stacked, self.stop_ids, self.tokenizer)
