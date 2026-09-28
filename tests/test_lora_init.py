"""LoRA init file round trip, schema, and sha256, without a language model."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("safetensors")

from safetensors.numpy import save_file

from vfates.features.lora_init import (
    LORA_INIT_FILENAME,
    LORA_INIT_SCHEMA,
    assert_lora_a_equal,
    load_lora_init,
    save_lora_init,
)
from vfates.utils.io import sha256_file


def test_lora_init_save_load_sha_and_schema(tmp_path) -> None:
    path = tmp_path / LORA_INIT_FILENAME
    tensors = {
        "model.layers.1.mlp.down_proj": np.arange(8, dtype=np.float32).reshape(2, 4),
        "model.layers.0.self_attn.q_proj": np.ones((2, 3), dtype=np.float32),
    }
    meta = {
        "model_id": "tiny",
        "lora_r": 2,
        "lora_alpha": 4,
        "lora_seed": 0,
        "target_modules": "q_proj,down_proj",
        "n_modules": 2,
    }
    digest = save_lora_init(path, tensors, meta)
    assert digest == sha256_file(path)
    assert not list(tmp_path.glob(".lora_init*"))
    loaded, loaded_meta = load_lora_init(path, expected_sha256=digest)
    assert loaded_meta["schema"] == LORA_INIT_SCHEMA
    assert loaded_meta["lora_seed"] == "0"
    assert loaded_meta["n_modules"] == "2"
    assert set(loaded) == set(tensors)
    for key, value in tensors.items():
        assert loaded[key].dtype == np.float32
        assert np.array_equal(loaded[key], value)
    assert_lora_a_equal(tensors, loaded)
    with pytest.raises(ValueError, match="sha256"):
        load_lora_init(path, expected_sha256="ab" * 32)
    changed = {key: value.copy() for key, value in loaded.items()}
    changed["model.layers.0.self_attn.q_proj"] = changed["model.layers.0.self_attn.q_proj"] + 1
    with pytest.raises(ValueError, match="bitwise"):
        assert_lora_a_equal(changed, loaded)


def test_apply_lora_a_copies_saved_matrices() -> None:
    torch = pytest.importorskip("torch")

    from vfates.features.lora_init import apply_lora_a

    class _Weight(torch.nn.Module):
        def __init__(self, tensor: torch.Tensor) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(tensor)

    class _Proj(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_A = torch.nn.ModuleDict({"default": _Weight(torch.zeros(2, 3))})

    class _Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = _Proj()

    class _Inner(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([_Block()])

    class _LM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _Inner()

    class _Mid(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _LM()

    class _Root(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_model = _Mid()

    model = _Root()
    saved = {"model.layers.0.q_proj": np.arange(6, dtype=np.float32).reshape(2, 3)}
    apply_lora_a(model, saved)
    weight = model.base_model.model.model.layers[0].q_proj.lora_A.default.weight
    assert weight.dtype == torch.float32
    assert np.array_equal(weight.detach().numpy(), saved["model.layers.0.q_proj"])
    with pytest.raises(ValueError, match="no matrix"):
        apply_lora_a(model, {})
    bad = {"model.layers.0.q_proj": np.zeros((2, 4), dtype=np.float32)}
    with pytest.raises(ValueError, match="shape"):
        apply_lora_a(model, bad)


def test_lora_init_rejects_a_bad_schema(tmp_path) -> None:
    path = tmp_path / LORA_INIT_FILENAME
    weight = np.zeros((2, 2), dtype=np.float32)
    save_file({"model.layers.0.self_attn.q_proj": weight}, path, metadata={"schema": "other"})
    with pytest.raises(ValueError, match="schema"):
        load_lora_init(path)
