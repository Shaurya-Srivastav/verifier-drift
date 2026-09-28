"""CountSketch unbiasedness, linearity and numpy/torch agreement."""

from __future__ import annotations

import numpy as np
import pytest

from vfates.features.sketch import CountSketch


def test_linearity_and_determinism():
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    b = np.arange(12, 24, dtype=np.float64).reshape(3, 4)
    sketch = CountSketch(8, seed=5)
    assert np.array_equal(
        sketch.sketch_numpy([a + b]), sketch.sketch_numpy([a]) + sketch.sketch_numpy([b])
    )
    assert np.array_equal(sketch.sketch_numpy([2.0 * a]), 2.0 * sketch.sketch_numpy([a]))
    again = CountSketch(8, seed=5).sketch_numpy([a, b])
    assert np.array_equal(again, sketch.sketch_numpy([a, b]))
    assert not np.array_equal(again, CountSketch(8, seed=6).sketch_numpy([a, b]))
    assert CountSketch(4, seed=1).sketch_numpy([]).shape == (4,)
    assert np.all(CountSketch(4, seed=1).sketch_numpy([np.zeros(6)]) == 0.0)


def test_monte_carlo_unbiased_and_variance_bound():
    rng = np.random.default_rng(12345)
    a = rng.normal(size=200)
    b = rng.normal(size=200)
    truth = float(np.dot(a, b))
    dim = 32
    estimates = np.array(
        [
            float(
                np.dot(
                    CountSketch(dim, seed).sketch_numpy([a]),
                    CountSketch(dim, seed).sketch_numpy([b]),
                )
            )
            for seed in range(400)
        ]
    )
    mean = float(estimates.mean())
    stderr = float(estimates.std(ddof=1) / np.sqrt(len(estimates)))
    assert abs(mean - truth) < 4.0 * stderr
    variance = float(estimates.var(ddof=1))
    bound = (float(np.dot(a, a)) * float(np.dot(b, b)) + truth**2) / dim
    # Slack covers the Monte Carlo error of the variance estimate, not bias.
    assert variance <= bound * 1.8 + 1.0


def test_sketch_tensor_matches_one_tensor_and_sums_close():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(7)
    arrays = [rng.normal(size=(6, 5)), rng.normal(size=(4, 3)), rng.normal(size=9)]
    sketch = CountSketch(32, seed=3)
    tensors = [torch.tensor(array, dtype=torch.float64) for array in arrays]
    one = sketch.sketch_tensor(tensors[0], 0).cpu().numpy()
    assert np.array_equal(one, sketch.sketch_torch([tensors[0]]).cpu().numpy())
    parts = [sketch.sketch_tensor(tensor, index) for index, tensor in enumerate(tensors)]
    total = torch.stack(parts).sum(dim=0).cpu().numpy()
    whole = sketch.sketch_torch(tensors).cpu().numpy()
    np.testing.assert_allclose(total, whole, rtol=1e-8, atol=1e-8)


def test_numpy_matches_torch_cpu():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    arrays = [rng.normal(size=(5, 3)), rng.normal(size=(4,))]
    sketch = CountSketch(64, seed=9)
    expected = sketch.sketch_numpy(arrays)
    tensors = [torch.tensor(array, dtype=torch.float64) for array in arrays]
    tensors[0] = tensors[0].t()  # non-contiguous; logical values match arrays[0].T
    expected_t = sketch.sketch_numpy([arrays[0].T, arrays[1]])
    got = sketch.sketch_torch(tensors).cpu().numpy()
    assert np.array_equal(got, expected_t)
    assert got.shape == expected.shape
    # A second call hits the bucket cache and stays bitwise identical.
    assert np.array_equal(sketch.sketch_torch(tensors).cpu().numpy(), got)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("torch"),
    reason="torch not installed",
)
def test_mps_sketch_matches_numpy():
    import torch

    if not torch.backends.mps.is_available():
        pytest.skip("mps not available")
    rng = np.random.default_rng(4)
    array = rng.normal(size=(32, 8))
    sketch = CountSketch(128, seed=11)
    got = (
        sketch.sketch_torch([torch.tensor(array, dtype=torch.float32, device="mps")]).cpu().numpy()
    )
    np.testing.assert_allclose(got, sketch.sketch_numpy([array]), rtol=1e-5, atol=1e-5)
