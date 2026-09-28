"""The P5 development pool and the E1 pools share no question text.

``tests/test_chain_sum.py`` checks that this builtin generator matches Reasoning
Gym when that package is installed. This test uses the builtin only.
"""

from __future__ import annotations

from vfates.tasks.chain_sum import ChainSumConfig, builtin_items

E1_SEED = 0
P5_SEED = 50_000_000


def _questions(seed: int, n: int) -> set[str]:
    config = ChainSumConfig()
    return {item["question"] for item in builtin_items(config, seed, n)}


def test_p5_development_pool_shares_no_question_with_e1() -> None:
    offset = ChainSumConfig().heldout_seed_offset
    assert offset == 10_000_000
    e1 = _questions(E1_SEED, 1024) | _questions(E1_SEED + offset, 384)
    p5 = _questions(P5_SEED, 512) | _questions(P5_SEED + offset, 384)
    assert e1.isdisjoint(p5)
