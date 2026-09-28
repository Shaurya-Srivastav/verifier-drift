"""P6 adjudication export, Wilson intervals, and the API skip."""

from __future__ import annotations

from pathlib import Path

import pytest

from vfates.pilots.p6_natural import (
    ADJUDICATION_COLUMNS,
    export_adjudication,
    export_oracle_check,
    import_adjudication,
    run,
    same_selection_of,
    wilson_interval,
)


def test_wilson_interval_for_eighty_of_one_hundred() -> None:
    row = wilson_interval(80, 100)
    assert row["point"] == pytest.approx(0.8)
    assert row["lo"] == pytest.approx(0.711, abs=0.002)
    assert row["hi"] == pytest.approx(0.867, abs=0.002)


def test_export_is_deterministic_and_import_rejects_unknown_labels(tmp_path: Path) -> None:
    rows = [
        {
            "item_id": f"item-{i}",
            "pid": f"p{i % 3}",
            "prompt": f"prompt {i}",
            "response": "answer",
            "gold": "gold",
            "parsed": "parsed",
            "verifier": 1,
            "oracle": 0,
        }
        for i in range(20)
    ]
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    export_adjudication(rows, first, n=5, seed=7)
    export_adjudication(rows, second, n=5, seed=7)
    assert first.read_text() == second.read_text()
    header = first.read_text().splitlines()[0].split(",")
    assert header == list(ADJUDICATION_COLUMNS)
    labelled = tmp_path / "labelled.csv"
    text = first.read_text().splitlines()
    body = []
    for index, line in enumerate(text[1:]):
        parts = line.split(",")
        label = "real_error" if index < 4 else "oracle_wrong"
        parts[-2] = label
        body.append(",".join(parts))
    labelled.write_text("\n".join([text[0], *body]) + "\n")
    loaded = import_adjudication(labelled)
    assert loaded["n_labelled"] == 5
    assert loaded["fraction_real_error"] == pytest.approx(0.8)
    assert loaded["wilson_lo"] < loaded["fraction_real_error"] < loaded["wilson_hi"]
    bad = tmp_path / "bad.csv"
    bad_body = []
    for line in text[1:]:
        parts = line.split(",")
        parts[-2] = "maybe"
        bad_body.append(",".join(parts))
    bad.write_text("\n".join([text[0], *bad_body]) + "\n")
    with pytest.raises(ValueError, match="real_error"):
        import_adjudication(bad)


def test_same_selection_uses_category_tuples() -> None:
    block = {
        "patch_tau": {
            "base": "natural",
            "by": "tau",
            "k": 2,
            "categories": ["no_box", "near_miss_numeric"],
        },
        "patch_rate": {
            "base": "natural",
            "by": "rate",
            "k": 2,
            "categories": ["no_box", "near_miss_numeric"],
        },
    }
    assert same_selection_of(block) is True
    block["patch_rate"]["categories"] = ["no_box", "symbolic_gold"]
    assert same_selection_of(block) is False
    assert same_selection_of({"same_selection": True}) is True


def test_oracle_check_skips_without_a_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("VFATES_QWEN32_BASE_URL", raising=False)
    monkeypatch.delenv("VFATES_JUDGE_BASE_URL", raising=False)
    dest = tmp_path / "oracle.csv"
    skipped = export_oracle_check("", dest, n=10)
    assert skipped["skipped"] is True
    assert "OPENAI_API_KEY" in skipped["reason"]
    assert not dest.exists()
    rows = [
        {
            "item_id": "a",
            "pid": "p",
            "prompt": "q",
            "response": "r",
            "gold": "g",
            "rule_oracle": 1,
            "qwen3_32b": 0,
            "api_model": 1,
        },
        {
            "item_id": "b",
            "pid": "p",
            "prompt": "q",
            "response": "r",
            "gold": "g",
            "rule_oracle": 1,
            "qwen3_32b": 1,
            "api_model": 1,
        },
    ]
    written = export_oracle_check("", dest, n=10, rows=rows)
    assert written["skipped"] is False
    assert written["n"] == 1
    assert "rule_oracle" in dest.read_text().splitlines()[0]


def test_smoke_run_is_documented(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    report = run(
        {"smoke": True, "stage": "all", "n_boot": 20, "adjudication_dir": str(tmp_path / "adj")},
        tmp_path,
    )
    assert report["ok"] is True
    assert report["checks"]
    assert all(row["status"] == "documented" for row in report["checks"])
    exported = list((tmp_path / "adj").glob("*__*.csv"))
    assert any(path.name == "math__natural.csv" for path in exported)
    assert any(path.name == "qa__natural_robust.csv" for path in exported)
