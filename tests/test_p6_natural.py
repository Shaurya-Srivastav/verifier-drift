"""P6 report labels the heuristic judge reject rate honestly."""

from __future__ import annotations

from vfates.pilots.p6_natural import _markdown


def test_p6_markdown_names_the_judge_reject_rate() -> None:
    report = {
        "gates": {"pass": False, "mbpp": False, "ifeval": False, "complete": False},
        "mbpp": {"model": "m", "n_rows": 1, "n_problems": 1},
        "ifeval": {
            "model": "m",
            "n_rows": 1,
            "n_problems": 1,
            "judge_reject_rate_among_checker_passes": 0.25,
        },
    }
    text = _markdown(report)
    assert "heuristic judge reject rate among checker passes" in text
    assert "0.2500" in text or "0.25" in text
    assert "stuffing" not in text
