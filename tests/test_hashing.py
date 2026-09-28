"""Stable hashes, including the field-separator collision check."""

import pytest

from vfates.utils.hashing import stable_u64, u01


def test_u01_is_unchanged_for_keys_without_the_separator() -> None:
    assert u01("salt", "decimal_chain_sum/train/000000", "99") == 0.4345314019494759
    assert stable_u64("salt", "decimal_chain_sum/train/000000", "99") == 8015689563752197978
    assert u01("a", "b") == 0.06574825183479512
    assert stable_u64("a", "b") == 1212841174890269944
    assert u01("resampled", 0, "p", 1) == 0.9625581477487998


def test_separator_in_a_key_part_raises() -> None:
    with pytest.raises(ValueError, match="U\\+001F"):
        u01("a\x1fb")
    with pytest.raises(ValueError, match="field separator"):
        u01("a", "b\x1f")
    with pytest.raises(ValueError, match="field separator"):
        stable_u64("ok", "bad\x1fx")
    assert u01("a", "b") == 0.06574825183479512
