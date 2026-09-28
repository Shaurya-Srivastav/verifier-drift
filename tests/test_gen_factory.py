"""Lazy dispatch of make_generator."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from vfates.gen.factory import make_generator


def test_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        make_generator("foo", "some-model")  # type: ignore[arg-type]


def test_mlx_and_hf_dispatch_are_lazy(monkeypatch):
    mlx = ModuleType("vfates.gen.mlx_backend")

    class MLXGenerator:
        def __init__(
            self, model_id, adapter_path=None, completion_batch_size=128, prefill_batch_size=16
        ):
            self.model_id = model_id
            self.adapter_path = adapter_path
            self.completion_batch_size = completion_batch_size
            self.prefill_batch_size = prefill_batch_size

    mlx.MLXGenerator = MLXGenerator
    monkeypatch.setitem(sys.modules, "vfates.gen.mlx_backend", mlx)
    gen = make_generator("mlx", "mlx-community/demo", completion_batch_size=4)
    assert isinstance(gen, MLXGenerator)
    assert gen.model_id == "mlx-community/demo"
    assert gen.completion_batch_size == 4

    hf = ModuleType("vfates.gen.hf_backend")

    class HFGenerator:
        def __init__(self, model_id, **kwargs):
            self.model_id = model_id
            self.kwargs = kwargs

    hf.HFGenerator = HFGenerator
    monkeypatch.setitem(sys.modules, "vfates.gen.hf_backend", hf)
    backed = make_generator("hf", "Qwen/Qwen2.5-0.5B-Instruct", batch_size=2, dtype="float32")
    assert isinstance(backed, HFGenerator)
    assert backed.kwargs["batch_size"] == 2
    assert backed.kwargs["dtype"] == "float32"


def test_factory_hf_generates():
    pytest.importorskip("torch")
    # Import inside the test so the lazy-dispatch test can replace the module first.
    from test_gen_hf import build_tiny_qwen
    from vfates.gen.hf_backend import HFGenerator

    model, tokenizer = build_tiny_qwen()
    gen = make_generator(
        "hf",
        "tiny-qwen",
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        dtype="float32",
        batch_size=2,
    )
    assert isinstance(gen, HFGenerator)
    outs = gen.generate(["Hello"], n=1, max_tokens=2, temperature=0.0)
    assert len(outs) == 1 and len(outs[0]) == 1
    assert outs[0][0].n_tokens == len(outs[0][0].token_ids) <= 2


def test_factory_vllm_import_guard():
    import importlib.util

    if importlib.util.find_spec("vllm") is not None:
        pytest.skip("vllm is installed")
    with pytest.raises(ImportError, match="vLLM"):
        make_generator("vllm", "Qwen/Qwen2.5-0.5B-Instruct")
