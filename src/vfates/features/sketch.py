"""CountSketch for inner products of per-sample gradients.

For vectors ``a`` and ``b`` hashed with the same buckets and signs,
``E[<Sa, Sb>] = <a, b>``. With ``dim`` buckets,

``Var(<Sa, Sb>) <= (||a||^2 ||b||^2 + <a, b>^2) / dim``.

A list of tensors is treated as their concatenation, but each tensor is hashed
on its own: buckets and signs come from ``numpy.random.Generator`` seeded by
``stable_seed(seed, tensor_index, numel)``, so the sketch does not depend on
Python's hash randomisation or on the machine. ``tensor_index`` is the
position of the tensor in the list passed to :meth:`sketch_numpy` or
:meth:`sketch_torch`, which must stay in the same order to be comparable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vfates.utils.hashing import stable_seed


class CountSketch:
    """Unbiased inner-product sketch of dimension ``dim``."""

    def __init__(self, dim: int, seed: int) -> None:
        if dim < 1:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = int(dim)
        self.seed = int(seed)
        self._np_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._torch_cache: dict[tuple[Any, ...], tuple[Any, Any]] = {}

    def _hashes(self, tensor_index: int, numel: int) -> tuple[np.ndarray, np.ndarray]:
        key = (tensor_index, numel)
        cached = self._np_cache.get(key)
        if cached is None:
            rng = np.random.default_rng(stable_seed(self.seed, tensor_index, numel))
            buckets = rng.integers(0, self.dim, size=numel, dtype=np.int64)
            signs = rng.integers(0, 2, size=numel, dtype=np.int64).astype(np.float64) * 2.0 - 1.0
            cached = (buckets, signs)
            self._np_cache[key] = cached
        return cached

    def _torch_hashes(
        self, device: Any, tensor_index: int, numel: int, dtype: Any
    ) -> tuple[Any, Any]:
        import torch

        key = (str(device), tensor_index, numel, str(dtype))
        cached = self._torch_cache.get(key)
        if cached is None:
            buckets_np, signs_np = self._hashes(tensor_index, numel)
            cached = (
                torch.as_tensor(buckets_np, dtype=torch.long, device=device),
                torch.as_tensor(signs_np, dtype=dtype, device=device),
            )
            self._torch_cache[key] = cached
        return cached

    def sketch_numpy(self, tensors: list[np.ndarray]) -> np.ndarray:
        """Sketch the concatenation of ``tensors`` into a float64 vector of length ``dim``."""
        out = np.zeros(self.dim, dtype=np.float64)
        for index, tensor in enumerate(tensors):
            flat = np.ascontiguousarray(tensor).reshape(-1)
            if flat.size == 0:
                continue
            buckets, signs = self._hashes(index, int(flat.size))
            np.add.at(out, buckets, signs * flat.astype(np.float64, copy=False))
        return out

    def sketch_torch(self, tensors: list[Any]) -> Any:
        """Sketch torch tensors with ``index_add_``.

        CPU accumulation is float64 and matches :meth:`sketch_numpy` bitwise for
        the same values. MPS and CUDA accumulate in float32 because those
        backends do not reliably support float64 ``index_add_``. Buckets (long)
        and signs are cached per ``(device, tensor_index, numel)``.
        """
        import torch

        if not tensors:
            return torch.zeros(self.dim, dtype=torch.float64)
        device = tensors[0].device
        acc_dtype = torch.float64 if device.type == "cpu" else torch.float32
        out = torch.zeros(self.dim, dtype=acc_dtype, device=device)
        for index, tensor in enumerate(tensors):
            numel = int(tensor.numel())
            if numel == 0:
                continue
            flat = tensor.detach().to(device=device, dtype=acc_dtype).contiguous().reshape(-1)
            buckets, signs = self._torch_hashes(device, index, numel, acc_dtype)
            out.index_add_(0, buckets, signs * flat)
        return out
