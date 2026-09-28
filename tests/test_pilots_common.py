"""Pilot checks, reports, config overrides, and the pilot CLI."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from vfates.pilots.common import Checks, apply_override, cli, fmt, markdown_table, write_report
from vfates.utils.io import read_json


def test_checks_status_and_ok() -> None:
    checks = Checks()
    assert checks.add("p", "pass", 1.5, True, threshold=1.0, metric="auc") is True
    assert checks.rows[-1]["status"] == "pass"
    assert checks.rows[-1]["passed"] is True
    assert checks.rows[-1]["value"] == pytest.approx(1.5)
    assert checks.add("d", "gap", None, False, documented=True, metric="note") is False
    row = checks.rows[-1]
    assert row["status"] == "documented"
    assert row["value"] is None
    assert row["passed"] is False
    assert checks.ok is True
    noted = Checks()
    noted.add("x", "noted", 1.0, True, documented=True)
    assert noted.rows[-1]["status"] == "documented"
    assert noted.ok is True
    assert checks.add("f", "miss", 0.2, False, threshold=0.5) is False
    assert checks.rows[-1]["status"] == "fail"
    assert checks.ok is False


def test_fmt_and_markdown_table() -> None:
    assert fmt(None) == "n/a"
    assert fmt(True) == "yes"
    assert fmt(False) == "no"
    assert fmt(2) == "2"
    assert fmt(1.25) == "1.2500"
    assert fmt(0.0) == "0.0000"
    assert fmt(1e-4) == "1.00e-04"
    assert fmt(1e5) == "1.00e+05"
    assert fmt(float("nan")) == "nan"
    assert fmt([1.5, None, True]) == "[1.5000, n/a, yes]"
    assert fmt((1e-4, None)) == "[1.00e-04, n/a]"
    assert fmt({"z": 1}) == "{'z': 1}"
    table = markdown_table(["h|1", "v"], [["a|b", None], [1, 0.5]])
    assert table == "| h\\|1 | v |\n| --- | --- |\n| a\\|b | n/a |\n| 1 | 0.5000 |"


def test_write_report_summarises_pass_and_fail(tmp_path: Path) -> None:
    checks = Checks()
    checks.add("a", "alpha", 0.5, True, threshold=0.4, metric="auc")
    checks.add("b", "beta", None, False, documented=True)
    report = write_report(
        tmp_path / "ok",
        "Alpha",
        checks,
        config={"k": 1},
        started=time.perf_counter(),
        intro="Hello.",
        sections=[("Notes", "body")],
    )
    assert report["summary"] == "PASS"
    assert report["ok"] is True
    assert report["counts"] == {"pass": 1, "fail": 0, "documented": 1}
    assert report["config"] == {"k": 1}
    assert "hardware" in report["environment"]
    stored = read_json(tmp_path / "ok" / "report.json")
    assert stored["summary"] == "PASS"
    md = (tmp_path / "ok" / "report.md").read_text()
    assert md.startswith("# Alpha: PASS")
    assert "Hello." in md
    assert "## Notes" in md
    assert "body" in md
    assert "0.5000" in md
    assert "documented" in md

    bad = Checks()
    bad.add("c", "gamma", 0.1, False, threshold=0.2)
    failed = write_report(tmp_path / "bad", "Alpha", bad, config={}, started=time.perf_counter())
    assert failed["summary"] == "FAIL"
    assert failed["ok"] is False
    assert failed["counts"]["fail"] == 1
    assert (tmp_path / "bad" / "report.md").read_text().startswith("# Alpha: FAIL")
    assert read_json(tmp_path / "bad" / "report.json")["summary"] == "FAIL"


def test_apply_override_nested_yaml_values() -> None:
    config: dict = {"sampling": {"n": 1}}
    apply_override(config, "sampling.n=4")
    apply_override(config, "sampling.temperature=0.5")
    apply_override(config, "a.b.c=[oracle, noise]")
    apply_override(config, "a.flag=true")
    apply_override(config, "a.b.empty=null")
    apply_override(config, "count=2")
    assert config["sampling"] == {"n": 4, "temperature": 0.5}
    assert config["a"]["b"]["c"] == ["oracle", "noise"]
    assert config["a"]["flag"] is True
    assert config["a"]["b"]["empty"] is None
    assert config["count"] == 2
    with pytest.raises(ValueError, match="key=value"):
        apply_override(config, "sampling.n")


def test_cli_defaults_overrides_and_return_code(tmp_path: Path) -> None:
    here = tmp_path / "pilot"
    here.mkdir()
    (here / "config.yaml").write_text("name: base\nsampling:\n  n: 1\n")
    seen: dict = {}

    def run(config: dict, dest: Path) -> dict:
        seen["config"] = config
        seen["out"] = dest
        return {"ok": seen["ok"]}

    seen["ok"] = True
    assert cli(run, here, []) == 0
    assert seen["out"] == here / "results"
    assert seen["config"] == {"name": "base", "sampling": {"n": 1}}

    custom = tmp_path / "custom.yaml"
    custom.write_text("name: base\nsampling:\n  n: 1\n")
    dest = tmp_path / "dest"
    assert (
        cli(
            run,
            here,
            [
                "--config",
                str(custom),
                "--out",
                str(dest),
                "--set",
                "sampling.n=4",
                "--set",
                "enabled=false",
                "--set",
                "label=null",
                "--set",
                "keep=[1, 2]",
                "--set",
                'title="chain sum"',
            ],
        )
        == 0
    )
    assert seen["out"] == dest
    assert seen["config"]["name"] == "base"
    assert seen["config"]["sampling"]["n"] == 4
    assert seen["config"]["enabled"] is False
    assert seen["config"]["label"] is None
    assert seen["config"]["keep"] == [1, 2]
    assert seen["config"]["title"] == "chain sum"

    seen["ok"] = False
    assert cli(run, here, ["--config", str(custom), "--out", str(dest)]) == 1
