"""Generated manuscript assets: macros, abstract sentences, and dash hygiene."""

from __future__ import annotations

import json
from pathlib import Path

from vfates.paper.assets import build_assets

_DASH_TEXT = ("--", "\u2013", "\u2014", "\u2212")
_DASH_PDF = (b"\xe2\x80\x93", b"\xe2\x80\x94", b"\xe2\x88\x92")


def _synthetic() -> dict:
    pair = {
        "cell": "e1-math-qwen3-1.7b",
        "stage": "A",
        "experiment": "E1",
        "domain": "math",
        "model": "qwen3-1.7b",
        "params_b": 2.031739904,
        "arm": "fp_trigger",
        "control": "noise_fp_trigger",
        "class": -1,
        "is_natural": False,
        "dtau": -0.2,
        "dacc": -0.1,
        "baseline_class": -1.0,
    }
    return {
        "stages": {
            "A": {
                "status": "complete",
                "h1": {"holds": True, "decision": "holds", "n_rejected": 2, "n_family": 12},
                "h1_pooled": {"holds": False, "decision": "does_not_hold"},
                "h2": {"holds": False, "decision": "does_not_hold"},
                "h3": {
                    "holds": False,
                    "decision": "does_not_hold",
                    "rho": 0.25,
                    "permutation_p": 0.4,
                    "q05": -0.1,
                    "n_pairs": 30,
                    "baselines": {
                        "class": {"rho": 0.1, "rho_tau_minus_baseline": 0.15},
                    },
                    "within_shared": {
                        "rho": 0.05,
                        "permutation_p": 0.2,
                        "q05": -0.2,
                        "n_pairs": 10,
                    },
                },
                "h7": {"holds": False, "decision": "does_not_hold"},
                "h8": {"holds": False, "decision": "does_not_hold", "point": 0.01, "p_le_0": 0.4},
                "h9": {"budgets": {"32": {"median_rho": 0.1, "q05": 0.0, "frac_positive": 0.6}}},
                "h10": {
                    "holds": False,
                    "decision": "does_not_hold",
                    "pairs": [
                        {
                            "name": "fp_trigger_enriched - fp_trigger",
                            "a": "fp_trigger_enriched",
                            "b": "fp_trigger",
                            "point": 0.02,
                            "p_le_0": 0.3,
                        }
                    ],
                },
            },
            "B": {"status": "pending"},
        },
        "figures": {
            "h3_pairs": [pair],
            "h9_budgets": [],
            "natural_rows": [],
            "h8_rows": [],
            "h10_rows": [],
            "cost_rows": [],
        },
    }


def _assert_no_dashes(directory: Path) -> None:
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".pdf":
            data = path.read_bytes()
            for token in _DASH_PDF:
                assert token not in data, path
            continue
        if path.suffix != ".tex":
            continue
        text = path.read_text()
        for token in _DASH_TEXT:
            assert token not in text, (path, token)


def test_passed_hypothesis_is_the_only_abstract_sentence(tmp_path: Path) -> None:
    report = tmp_path / "program_report.json"
    report.write_text(json.dumps(_synthetic()))
    out = tmp_path / "generated"
    figures = tmp_path / "figures"
    build_assets(report_path=report, out=out, figures=figures)

    abstract = (out / "abstract_results.tex").read_text()
    assert "H1 passed" in abstract
    assert "H3 passed" not in abstract
    assert "H2 passed" not in abstract
    numbers = (out / "numbers.tex").read_text()
    assert r"\newcommand{\HThreeRho}{0.2500}" in numbers
    assert r"\newcommand{\HOneRejected}{2}" in numbers
    assert r"\newcommand{\HThreeRepRho}{\pending}" in numbers
    for name in ("fig1.pdf", "fig_h3.pdf", "fig_natural.pdf", "fig_h9.pdf"):
        assert (figures / name).stat().st_size > 0
    _assert_no_dashes(out)
    _assert_no_dashes(figures)


def test_missing_report_is_pending(tmp_path: Path) -> None:
    out = tmp_path / "generated"
    figures = tmp_path / "figures"
    build_assets(report_path=tmp_path / "absent.json", out=out, figures=figures)
    abstract = (out / "abstract_results.tex").read_text()
    numbers = (out / "numbers.tex").read_text()
    assert "pending" in abstract
    assert "H1 passed" not in abstract
    assert r"\newcommand{\HThreeRho}{\pending}" in numbers
    assert r"\newcommand{\HOneRejected}{\pending}" in numbers
    assert (figures / "fig1.pdf").stat().st_size > 0
    _assert_no_dashes(out)
    _assert_no_dashes(figures)


def test_kit_and_cost_tables_describe_the_design(tmp_path: Path) -> None:
    out = tmp_path / "generated"
    build_assets(report_path=tmp_path / "absent.json", out=out, figures=tmp_path / "figures")
    kit = {
        line.split("&")[0].strip(): line
        for line in (out / "kit.tex").read_text().splitlines()
        if "&" in line
    }
    assert "correct outputs" in kit[r"\texttt{noise\_fn}"]
    assert "correct outputs" in kit[r"\texttt{noise\_fn\_depleted}"]
    assert "wrong output" in kit[r"\texttt{noise\_fp\_trigger}"]
    baselines = (out / "baselines.tex").read_text()
    assert r"\BreakEvenMeasured" in baselines
    assert "BreakEvenVerifiers" not in (out / "numbers.tex").read_text()
    numbers = (out / "numbers.tex").read_text()
    assert r"\newcommand{\PZeroCleanTau}{1.0000}" in numbers
