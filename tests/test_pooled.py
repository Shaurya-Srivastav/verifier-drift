"""Pooled program report on synthetic cell directories."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from vfates.analysis.outcomes import holm_adjusted, permutation_spearman_p
from vfates.analysis.pooled import (
    classify_arms,
    leaf_arm_class,
    pooled_report,
    render_pooled_markdown,
    staged_report,
)
from vfates.audit import resolve_arms
from vfates.utils.io import read_yaml, write_json, write_jsonl

PIDS = ("p0", "p1")
STEP = 4


def _forecast(tau: float, budget: Mapping[str, Mapping[str, float]] | None = None) -> dict:
    row = {
        "transfer": {
            "tau_pool": tau,
            "tau_hard": tau,
            "tau_rel": 0.4,
            "reliability": 0.9,
            "cos_gV_gO": 0.2,
            "base_acc_heldout": 0.3,
            "tau_pool_std": tau,
            "probe_heldout_auc": 0.7,
        },
        "step0": {"J": 0.2, "fpr": 0.1, "fnr": 0.2},
        "mean_precision": tau,
        "modal_precision": {"value": tau, "tie_fraction": 0.0},
        "behaviour_alignment": tau,
    }
    if budget is not None:
        row["tau_pool_budget"] = {str(key): dict(value) for key, value in budget.items()}
    return row


def _seed_map(final: float | Mapping[int, float]) -> dict[int, float]:
    if isinstance(final, Mapping):
        return {int(seed): float(value) for seed, value in final.items()}
    return {0: float(final)}


def _write_arm(path: Path, step0: float, final: float) -> None:
    write_jsonl(
        path / "eval_prompts.jsonl",
        [
            {"step": 0, "split": "heldout", "acc": {pid: step0 for pid in PIDS}},
            {"step": STEP, "split": "heldout", "acc": {pid: final for pid in PIDS}},
        ],
    )


def write_cell(
    runs_root: Path,
    cell_id: str,
    *,
    arms: Mapping[str, Mapping[str, object]],
    oracle: tuple[float, float] = (0.2, 0.6),
    patch: Mapping[str, object] | None = None,
    withdrawals: Mapping[str, object] | None = None,
) -> None:
    root = runs_root / cell_id
    step0, final = oracle
    _write_arm(root / "runs" / "oracle__s0", step0, final)
    payload: dict[str, object] = {"oracle": _forecast(0.5)}
    for name, spec in arms.items():
        for seed, acc in _seed_map(spec["final"]).items():  # type: ignore[arg-type]
            _write_arm(
                root / "runs" / f"{name}__s{seed}",
                float(spec.get("step0", 0.2)),  # type: ignore[arg-type]
                acc,
            )
        budget = spec.get("budget")
        payload[name] = _forecast(
            float(spec["tau"]),  # type: ignore[arg-type]
            budget if isinstance(budget, Mapping) else None,
        )
    if patch is not None:
        payload["patch_selection"] = dict(patch)
    write_json(root / "audit" / "forecasts.json", payload)
    if withdrawals is not None:
        write_json(root / "withdrawals.json", dict(withdrawals))


def cell_row(
    cell_id: str,
    *,
    experiment: str = "E1",
    domain: str = "synth",
    model: str = "toy",
    params_b: float | None = None,
    analysis: Mapping[str, object] | None = None,
    study: Mapping[str, object] | None = None,
) -> dict:
    block = {
        "final_step": STEP,
        "oracle_gain_min": 0.10,
        "early": {"enabled": False},
    }
    if analysis:
        block.update(analysis)
    config: dict[str, object] = {"analysis": block}
    row: dict[str, object] = {
        "id": cell_id,
        "experiment": experiment,
        "domain": domain,
        "model": model,
        "config": config,
    }
    if study is not None:
        config["study"] = dict(study)
    if params_b is not None:
        row["params_b"] = params_b
    return row


def _pairs(names: list[tuple[str, str]]) -> dict:
    return {"pairs": [{"arm": arm, "control": control} for arm, control in names]}


def _quiet(**extra: object) -> dict:
    block: dict[str, object] = {
        "h1": {"contrasts": []},
        "h2": {"contrasts": []},
        "h3": {"pairs": []},
    }
    block.update(extra)
    return block


def _kit(block: Mapping[str, object], name: str) -> dict:
    return next(row for row in block["contrasts"] if row["name"] == name)  # type: ignore[index]


def test_h3_permutation_is_uniform_under_the_null(tmp_path: Path) -> None:
    ps: list[float] = []
    for rep in range(20):
        rng = np.random.default_rng(20_000 + rep)
        root = tmp_path / f"null{rep}"
        cells = []
        for cell_i in range(2):
            dtau = np.array([0.1, 0.4, 0.7, 0.9])
            dacc = rng.normal(size=4) + np.arange(4) * 1e-6
            pairs = []
            arms = {}
            for i in range(4):
                arm, control = f"a{i}", f"c{i}"
                arms[arm] = {"tau": float(dtau[i]), "final": 0.5 + float(dacc[i])}
                arms[control] = {"tau": 0.0, "final": 0.5}
                pairs.append((arm, control))
            cid = f"c{cell_i}"
            write_cell(root, cid, arms=arms)
            cells.append(cell_row(cid, domain=f"dom{cell_i}", analysis=_quiet(h3=_pairs(pairs))))
        report = pooled_report({"cells": cells}, root, n_boot=8, n_perm=99, seed=0)
        p_value = report["h3"]["permutation_p"]
        assert isinstance(p_value, float)
        ps.append(p_value)
    arr = np.asarray(ps)
    assert np.isfinite(arr).all()
    assert float(np.mean(arr < 0.05)) <= 0.25
    assert 0.30 <= float(arr.mean()) <= 0.70
    ordered = np.sort(arr)
    ecdf = np.arange(1, ordered.size + 1) / ordered.size
    assert float(np.max(np.abs(ecdf - ordered))) < 0.45


def test_h3_detects_a_within_cell_relation_and_h9_matches_at_full_budget(tmp_path: Path) -> None:
    root = tmp_path / "strong"
    pairs = []
    arms = {}
    for i in range(20):
        arm, control = f"a{i}", f"c{i}"
        tau = float(i + 1)
        full = {"mean": tau, "sd": 0.0}
        locked = {"mean": 0.0, "sd": 0.0}
        arms[arm] = {
            "tau": tau,
            "final": tau,
            "budget": {
                "32": {"mean": float(20 - i), "sd": 0.0},
                "64": full,
                "128": full,
                "192": full,
            },
        }
        arms[control] = {
            "tau": 0.0,
            "final": 0.0,
            "budget": {"32": locked, "64": locked, "128": locked, "192": locked},
        }
        pairs.append((arm, control))
    write_cell(root, "strong", arms=arms)
    report = pooled_report(
        {"cells": [cell_row("strong", analysis=_quiet(h3=_pairs(pairs)))]},
        root,
        n_boot=20,
        n_perm=80,
        seed=0,
    )
    h3 = report["h3"]
    assert h3["n_pairs"] == 20
    assert h3["rho"] == pytest.approx(1.0)
    assert h3["permutation_p"] < 0.05
    assert h3["q05"] > 0.0
    assert h3["holds"] is True
    assert h3["decision"] == "holds"
    assert h3["permutation_method"] == "within_cell_monte_carlo"
    assert "_dtau" not in h3
    json.dumps(report)
    for budget in ("64", "128", "192"):
        row = report["h9"]["budgets"][budget]
        assert row["status"] == "exact"
        assert row["matches_h3"] is True
        assert row["median_rho"] == pytest.approx(1.0)
        assert row["frac_positive"] == pytest.approx(1.0)
        assert row["n_draws"] == 1
    reversed_row = report["h9"]["budgets"]["32"]
    assert reversed_row["status"] == "exact"
    assert reversed_row["matches_h3"] is False
    assert reversed_row["median_rho"] == pytest.approx(-1.0)
    assert reversed_row["frac_positive"] == pytest.approx(0.0)
    text = render_pooled_markdown(report)
    assert "## H3\n\nDecision: holds." in text
    assert "## H9\n\nDecision: reported." in text


def test_within_cell_permutation_ignores_a_between_cell_confound(tmp_path: Path) -> None:
    root = tmp_path / "confound"
    cells = []
    level = {"low": 0.2, "high": 0.8}
    tau0 = {"low": 0.1, "high": 0.7}
    for name in ("low", "high"):
        pairs = []
        arms = {}
        for i in range(4):
            arm, control = f"a{i}", f"c{i}"
            arms[arm] = {"tau": tau0[name] + 0.05 * i, "final": 0.5 + level[name]}
            arms[control] = {"tau": 0.0, "final": 0.5}
            pairs.append((arm, control))
        write_cell(root, name, arms=arms)
        cells.append(cell_row(name, domain=name, analysis=_quiet(h3=_pairs(pairs))))
    report = pooled_report({"cells": cells}, root, n_boot=10, n_perm=50, seed=0)
    assert report["h3"]["permutation_p"] == pytest.approx(1.0)
    assert report["h3"]["n_pairs"] == 8
    global_p = permutation_spearman_p(
        [row["dtau"] for row in report["h3"]["pairs"]],
        [row["dacc"] for row in report["h3"]["pairs"]],
    )
    assert global_p["p"] < 0.05


def test_h1_holm_spans_every_e1_cell(tmp_path: Path) -> None:
    root = tmp_path / "h1"
    contrast = {"h1": {"contrasts": [{"control": "noise_fn", "arm": "fn_behavior"}]}}
    write_cell(
        root,
        "cell_pos",
        arms={
            "noise_fn": {"tau": 0.4, "final": {0: 0.8, 1: 0.8}},
            "fn_behavior": {"tau": 0.2, "final": {0: 0.4, 1: 0.4}},
        },
    )
    write_cell(
        root,
        "cell_negseed",
        arms={
            "noise_fn": {"tau": 0.4, "final": {0: 0.9, 1: 0.3}},
            "fn_behavior": {"tau": 0.2, "final": {0: 0.4, 1: 0.5}},
        },
    )
    analysis = _quiet(**contrast)
    report = pooled_report(
        {
            "cells": [
                cell_row("cell_pos", analysis=analysis),
                cell_row("cell_negseed", analysis=analysis),
            ]
        },
        root,
        n_boot=40,
        n_perm=20,
        seed=0,
    )
    tested = [row for row in report["h1"]["contrasts"] if row["status"] == "tested"]
    assert report["h1"]["n_family"] == 2
    assert {row["cell"] for row in tested} == {"cell_pos", "cell_negseed"}
    by_cell = {row["cell"]: row for row in tested}
    assert by_cell["cell_pos"]["p_sup"] == pytest.approx(0.0)
    assert by_cell["cell_pos"]["all_seeds_positive"] is True
    assert by_cell["cell_pos"]["rejects"] is True
    assert by_cell["cell_negseed"]["all_seeds_positive"] is False
    assert by_cell["cell_negseed"]["rejects"] is False
    assert report["h1"]["holds"] is True
    expected = holm_adjusted({row["holm_id"]: row["p_sup"] for row in tested})
    for row in tested:
        assert row["holm_adjusted_p"] == pytest.approx(expected[row["holm_id"]])
        assert row["holm_id"] == f"{row['cell']} | noise_fn - fn_behavior"
    assert "## H1\n\nDecision: holds." in render_pooled_markdown(report)


def test_h8_sign_and_exclusions(tmp_path: Path) -> None:
    root = tmp_path / "h8"
    write_cell(
        root,
        "math_pos",
        arms={
            "patch_tau": {"tau": 0.4, "final": 0.8},
            "patch_rate": {"tau": 0.3, "final": 0.3},
        },
        patch={"same_selection": False, "patch_tau": {}, "patch_rate": {}},
    )
    write_cell(
        root,
        "code_same",
        arms={
            "patch_tau": {"tau": 0.4, "final": 0.6},
            "patch_rate": {"tau": 0.3, "final": 0.5},
        },
        patch={"same_selection": True, "patch_tau": {}, "patch_rate": {}},
    )
    write_cell(
        root,
        "qa_withdrawn",
        arms={
            "patch_tau": {"tau": 0.4, "final": 0.7},
            "patch_rate": {"tau": 0.3, "final": 0.4},
        },
        patch={"same_selection": False},
        withdrawals={
            "schema": "vfates.withdrawals/1",
            "audit": {"patch_tau": {"reasons": ["rate_miss"]}},
            "training": {},
            "withdrawn": ["patch_tau"],
        },
    )
    quiet = _quiet()
    report = pooled_report(
        {
            "cells": [
                cell_row("math_pos", domain="math", analysis=quiet),
                cell_row("code_same", domain="code", analysis=quiet),
                cell_row("qa_withdrawn", domain="qa", analysis=quiet),
            ]
        },
        root,
        n_boot=20,
        n_perm=10,
        seed=0,
    )
    h8 = report["h8"]
    assert h8["holds"] is True
    assert h8["point"] == pytest.approx(0.5)
    assert h8["p_le_0"] == pytest.approx(0.0)
    assert [row["cell"] for row in h8["cells"]] == ["math_pos"]
    reasons = {row["cell"]: row["reason"] for row in h8["excluded"]}
    assert reasons["code_same"] == "same_selection"
    assert reasons["qa_withdrawn"] == "withdrawn"
    text = render_pooled_markdown(report)
    assert "same_selection" in text
    assert "rate_miss" in text

    neg = tmp_path / "h8neg"
    write_cell(
        neg,
        "ifeval_neg",
        arms={
            "patch_tau": {"tau": 0.2, "final": 0.2},
            "patch_rate": {"tau": 0.4, "final": 0.8},
        },
        patch={"same_selection": False},
    )
    negative = pooled_report(
        {"cells": [cell_row("ifeval_neg", domain="ifeval", analysis=quiet)]},
        neg,
        n_boot=20,
        n_perm=10,
        seed=0,
    )
    assert negative["h8"]["point"] == pytest.approx(-0.6)
    assert negative["h8"]["p_le_0"] == pytest.approx(1.0)
    assert negative["h8"]["holds"] is False
    assert negative["h8"]["decision"] == "does_not_hold"
    assert negative["h8"]["excluded"] == []


def test_inconclusive_cells_leave_h3_and_h1_pooled(tmp_path: Path) -> None:
    root = tmp_path / "gain"
    yes_pairs = []
    no_pairs = []
    yes_arms = {
        "noise_fn": {"tau": 0.5, "final": 0.9},
        "fn_behavior": {"tau": 0.1, "final": 0.2},
    }
    no_arms = {
        "noise_fn": {"tau": 0.5, "final": 0.2},
        "fn_behavior": {"tau": 0.1, "final": 0.9},
    }
    for i in range(4):
        arm, control = f"a{i}", f"c{i}"
        yes_arms[arm] = {"tau": float(i + 1), "final": float(i + 1)}
        yes_arms[control] = {"tau": 0.0, "final": 0.0}
        no_arms[arm] = {"tau": float(i + 1), "final": float(4 - i)}
        no_arms[control] = {"tau": 0.0, "final": 0.0}
        yes_pairs.append((arm, control))
        no_pairs.append((arm, control))
    write_cell(root, "gain_yes", arms=yes_arms, oracle=(0.1, 0.5))
    write_cell(root, "gain_no", arms=no_arms, oracle=(0.2, 0.2))
    shared = {
        "h1": {"contrasts": [{"control": "noise_fn", "arm": "fn_behavior"}]},
        "h2": {"contrasts": []},
    }
    report = pooled_report(
        {
            "cells": [
                cell_row("gain_yes", domain="dom_a", analysis={**shared, "h3": _pairs(yes_pairs)}),
                cell_row("gain_no", domain="dom_b", analysis={**shared, "h3": _pairs(no_pairs)}),
            ]
        },
        root,
        n_boot=30,
        n_perm=40,
        seed=0,
    )
    assert report["conclusiveness"]["decision"] == "1 conclusive, 1 inconclusive"
    assert report["cells"][0]["conclusive"] is True
    assert report["cells"][1]["conclusive"] is False
    h3 = report["h3"]
    assert h3["n_pairs"] == 4
    assert h3["rho"] == pytest.approx(1.0)
    assert [row["cell"] for row in h3["excluded_inconclusive"]] == ["gain_no"]
    sensitivity = h3["sensitivity_including_inconclusive"]
    assert sensitivity["n_pairs"] == 8
    assert sensitivity["rho"] < 0.5
    h1_tested = [row["cell"] for row in report["h1"]["contrasts"] if row["status"] == "tested"]
    assert h1_tested == ["gain_yes", "gain_no"]
    assert report["h1"]["inconclusive_cells"] == ["gain_no"]
    primary = _kit(report["h1_pooled"], "noise_fn - fn_behavior")
    pooled_sensitivity = _kit(
        report["h1_pooled"]["sensitivity_including_inconclusive"], "noise_fn - fn_behavior"
    )
    assert primary["point"] == pytest.approx(0.7)
    assert primary["p_le_0"] == pytest.approx(0.0)
    assert primary["rejects"] is True
    assert [row["cell"] for row in primary["excluded_inconclusive"]] == ["gain_no"]
    assert pooled_sensitivity["point"] == pytest.approx(0.0)
    assert pooled_sensitivity["p_le_0"] == pytest.approx(1.0)
    assert pooled_sensitivity["rejects"] is False
    assert report["h1_pooled"]["holds"] is True
    assert report["h1_pooled"]["sensitivity_including_inconclusive"]["holds"] is False


def test_h7_accepts_a_list_of_natural_pairs(tmp_path: Path) -> None:
    root = tmp_path / "h7"
    write_cell(
        root,
        "math",
        arms={
            "natural": {"tau": 0.2, "final": 0.2},
            "noise_natural": {"tau": 0.6, "final": 0.7},
            "natural_strict": {"tau": 0.15, "final": 0.3},
            "noise_natural_strict": {"tau": 0.55, "final": 0.8},
        },
    )
    report = pooled_report(
        {
            "cells": [
                cell_row(
                    "math",
                    domain="math",
                    analysis=_quiet(
                        natural=[
                            {
                                "natural": "natural",
                                "control": "noise_natural",
                                "oracle": "oracle",
                            },
                            {
                                "natural": "natural_strict",
                                "control": "noise_natural_strict",
                                "oracle": "oracle",
                            },
                        ]
                    ),
                )
            ]
        },
        root,
        n_boot=20,
        n_perm=10,
        seed=0,
    )
    pairs = report["h7"]["domains"]["math"]["pairs"]
    assert [row["natural"] for row in pairs] == ["natural", "natural_strict"]
    assert pairs[0]["contrast"]["point"] == pytest.approx(0.5)
    assert "natural_strict" in report["h7"]["decision"]
    assert report["h7"]["decision"] != "not_tested"


def test_scale_slope_on_the_qwen3_ladder(tmp_path: Path) -> None:
    root = tmp_path / "scale"
    cells = []
    for params, point, name in (
        (1.0, 0.1, "qwen3-a"),
        (10.0, 0.2, "qwen3-b"),
        (100.0, 0.3, "qwen3-c"),
    ):
        cid = name
        write_cell(
            root,
            cid,
            arms={
                "noise_fn": {"tau": 0.4, "final": 0.4 + point},
                "fn_behavior": {"tau": 0.2, "final": 0.4},
            },
        )
        cells.append(
            cell_row(
                cid,
                experiment="E2",
                domain="math",
                model=name,
                params_b=params,
                analysis=_quiet(),
            )
        )
    write_cell(
        root,
        "x-math",
        arms={
            "noise_fn": {"tau": 0.4, "final": 0.9},
            "fn_behavior": {"tau": 0.2, "final": 0.1},
        },
    )
    cells.append(
        cell_row(
            "x-math",
            experiment="X",
            domain="math",
            model="qwen3-1.7b",
            params_b=1000.0,
            analysis=_quiet(),
        )
    )
    report = pooled_report({"cells": cells}, root, n_boot=40, n_perm=10, seed=0)
    row = _kit(report["scale"], "noise_fn - fn_behavior")
    assert row["status"] == "tested"
    assert row["n_cells"] == 3
    assert row["slope"] == pytest.approx(0.1)
    assert row["lo"] <= row["slope"] <= row["hi"]
    assert report["scale"]["decision"] == "reported"
    assert report["h1"]["decision"] == "not_tested"
    assert "x-math" not in {item["cell"] for item in row["cells"]}


def test_h9_uses_stored_draws_and_marks_a_missing_budget(tmp_path: Path) -> None:
    root = tmp_path / "h9"
    pairs = []
    arms = {}
    for i in range(2):
        arm, control = f"a{i}", f"c{i}"
        tau = float(i + 1)
        reversed_draw = float(2 - i)
        arms[arm] = {
            "tau": tau,
            "final": tau,
            "budget": {
                "32": {"mean": tau, "sd": 0.2, "budget": 32, "oracle_labels_needed": 100},
                "64": {
                    "mean": tau,
                    "sd": 0.2,
                    "budget": 64,
                    "oracle_labels_needed": 100,
                    "draws": [reversed_draw, reversed_draw, reversed_draw],
                },
                "128": {"mean": tau, "sd": 0.0, "budget": 128, "oracle_labels_needed": 10},
                "192": {"mean": tau, "sd": 0.0},
            },
        }
        arms[control] = {
            "tau": 0.0,
            "final": 0.0,
            "budget": {
                "32": {"mean": 0.0, "sd": 0.2, "budget": 32, "oracle_labels_needed": 100},
                "64": {
                    "mean": 0.0,
                    "sd": 0.2,
                    "budget": 64,
                    "oracle_labels_needed": 100,
                    "draws": [0.0, 0.0, 0.0],
                },
                "128": {"mean": 0.0, "sd": 0.0},
                "192": {"mean": 0.0, "sd": 0.0},
            },
        }
        pairs.append((arm, control))
    write_cell(root, "screen", arms=arms)
    report = pooled_report(
        {"cells": [cell_row("screen", analysis=_quiet(h3=_pairs(pairs)))]},
        root,
        n_boot=10,
        n_perm=20,
        seed=0,
    )
    missing = report["h9"]["budgets"]["32"]
    assert missing["status"] == "missing"
    assert missing["median_rho"] is None
    assert missing["q05"] is None
    assert missing["frac_positive"] is None
    assert {(row["cell"], row["arm"]) for row in missing["missing"]} == {
        ("screen", "a0"),
        ("screen", "c0"),
        ("screen", "a1"),
        ("screen", "c1"),
    }
    drawn = report["h9"]["budgets"]["64"]
    assert drawn["status"] == "draws"
    assert drawn["n_draws"] == 3
    assert drawn["n_pairs"] == 2
    assert drawn["median_rho"] == pytest.approx(-1.0)
    assert drawn["frac_positive"] == pytest.approx(0.0)
    assert drawn["matches_h3"] is False
    assert report["h3"]["rho"] == pytest.approx(1.0)
    for budget in ("128", "192"):
        exact = report["h9"]["budgets"][budget]
        assert exact["status"] == "exact"
        assert exact["matches_h3"] is True
        assert exact["median_rho"] == pytest.approx(1.0)


def test_exploratory_section_stays_outside_the_families(tmp_path: Path) -> None:
    root = tmp_path / "explore"
    h1 = {"h1": {"contrasts": [{"control": "noise_fn", "arm": "fn_behavior"}]}}
    write_cell(
        root,
        "math",
        arms={
            "noise_fn": {"tau": 0.4, "final": 0.8},
            "fn_behavior": {"tau": 0.2, "final": 0.2},
            "random": {"tau": 0.0, "final": 0.5, "step0": 0.2},
        },
    )
    _write_arm(root / "math" / "variants" / "kl" / "noise_fn__s0", 0.2, 0.2)
    _write_arm(root / "math" / "variants" / "kl" / "fn_behavior__s0", 0.2, 0.8)
    write_cell(
        root,
        "qa",
        arms={
            "natural_robust": {"tau": 0.2, "final": 0.2},
            "noise_natural_robust": {"tau": 0.6, "final": 0.7},
        },
    )
    write_cell(
        root,
        "x-math",
        arms={
            "noise_fn": {"tau": 0.4, "final": 0.2},
            "fn_behavior": {"tau": 0.2, "final": 0.8},
        },
    )
    write_cell(root, "suppressed", arms={"random": {"tau": 0.0, "final": 0.4}})
    write_json(
        root / "math" / "transfer" / "fp_hash__s0" / "code.json",
        {
            "schema": "vfates.transfer_eval/1",
            "source_cell": "math",
            "arm": "fp_hash",
            "seed": 0,
            "target_domain": "code",
            "target_cell": "code-cell",
            "adapter": "adapter",
            "step": STEP,
            "n_prompts": 2,
            "n_samples": 4,
            "accuracy": 0.4,
            "per_prompt": [
                {"pid": "p0", "accuracy": 0.4, "n": 4},
                {"pid": "p1", "accuracy": 0.4, "n": 4},
            ],
        },
    )
    write_json(
        root / "math" / "transfer" / "oracle__s0" / "code.json",
        {
            "schema": "vfates.transfer_eval/1",
            "source_cell": "math",
            "arm": "oracle",
            "seed": 0,
            "target_domain": "code",
            "target_cell": "code-cell",
            "adapter": "adapter",
            "step": STEP,
            "n_prompts": 2,
            "n_samples": 4,
            "accuracy": 0.6,
            "per_prompt": [
                {"pid": "p0", "accuracy": 0.6, "n": 4},
                {"pid": "p1", "accuracy": 0.6, "n": 4},
            ],
        },
    )
    report = pooled_report(
        {
            "cells": [
                cell_row(
                    "math",
                    domain="math",
                    analysis=_quiet(**h1),
                    study={
                        "variants": [
                            {
                                "name": "kl",
                                "arms": ["noise_fn", "fn_behavior"],
                                "seeds": [0],
                                "train_overrides": {"beta_kl": 0.01},
                            },
                            {"name": "horizon", "arms": ["oracle"], "seeds": [0]},
                        ]
                    },
                ),
                cell_row("qa", domain="qa", analysis=_quiet()),
                cell_row("x-math", experiment="X", domain="math", analysis=_quiet(**h1)),
                cell_row("suppressed", experiment="E2", analysis=_quiet(reported=[])),
            ]
        },
        root,
        n_boot=20,
        n_perm=10,
        seed=0,
    )
    h1_names = [row["arm"] for row in report["h1"]["contrasts"]] + [
        row["control"] for row in report["h1"]["contrasts"]
    ]
    assert "noise_fn__kl" not in h1_names
    assert all("__" not in name for name in h1_names)
    assert {row["cell"] for row in report["h1"]["contrasts"] if row["status"] == "tested"} == {
        "math"
    }
    exploratory = report["exploratory"]
    assert exploratory["decision"] == "reported"
    assert exploratory["run_naming"] == {
        "base": "runs/<cell>/runs/<arm>__s<seed>",
        "early": "runs/<cell>/early/<arm>__s<seed>",
        "withdrawals": "runs/<cell>/withdrawals.json",
        "variant": "runs/<cell>/variants/<name>/<arm>__s<seed>",
        "transfer": "runs/<cell>/transfer/<arm>__s<seed>/<target_domain>.json",
    }
    by_name = {row["name"]: row for row in exploratory["variants"] if row["cell"] == "math"}
    assert by_name["horizon"]["status"] == "missing_runs"
    kl = by_name["kl"]
    assert kl["status"] == "reported"
    assert kl["train_overrides"] == {"beta_kl": 0.01}
    assert {row["arm"]: row["seeds"][0]["final"] for row in kl["arms"]} == {
        "fn_behavior": pytest.approx(0.8),
        "noise_fn": pytest.approx(0.2),
    }
    contrast = kl["contrasts"][0]
    assert contrast["name"] == "noise_fn - fn_behavior"
    assert contrast["point"] == pytest.approx(-0.6)
    assert contrast["base_point"] == pytest.approx(0.6)
    assert contrast["sign_agrees"] is False
    assert "p_sup" not in contrast
    assert "holm_adjusted_p" not in contrast
    x_row = exploratory["experiment_x"][0]
    assert x_row["cell"] == "x-math"
    x_contrast = x_row["contrasts"][0]
    assert x_contrast["point"] == pytest.approx(-0.6)
    assert x_contrast["lo"] <= x_contrast["point"] <= x_contrast["hi"]
    assert x_contrast["e1"][0]["cell"] == "math"
    assert x_contrast["e1"][0]["point"] == pytest.approx(0.6)
    assert x_contrast["sign_agrees"] is False
    assert "holm_adjusted_p" not in x_contrast
    reported = {(row["cell"], row["arm"], row["kind"]) for row in exploratory["reported_arms"]}
    assert ("math", "random", "random") in reported
    assert ("qa", "natural_robust", "natural") in reported
    assert ("suppressed", "random", "random") not in reported
    random = next(row for row in exploratory["reported_arms"] if row["arm"] == "random")
    assert random["seeds"][0]["minus_step0"] == pytest.approx(0.3)
    assert random["seeds"][0]["minus_oracle"] == pytest.approx(-0.1)
    natural = next(row for row in exploratory["reported_arms"] if row["kind"] == "natural")
    assert natural["status"] == "reported"
    assert natural["control"] == "noise_natural_robust"
    assert natural["block"]["differs"]["decision"] == "differs"
    assert natural["block"]["screen_direction"]["decision"] == "matches"
    assert report["h7"]["domains"] == {}
    assert exploratory["transfer"]["schema"] == "vfates.transfer/1"
    assert exploratory["transfer"]["n_rows"] == 1
    text = render_pooled_markdown(report)
    assert "## Exploratory\n\nDecision: reported." in text
    assert "sign_agrees=False" in text


_ARM_CLASS = {
    "oracle": (0, False),
    "hash_fp": (0, False),
    "prompt_subset_fn": (0, False),
    "resampled": (0, False),
    "trigger_fp": (-1, False),
    "behavior_fn": (-1, False),
    "tolerance_fp": (-1, False),
    "tolerance_band_fp": (-1, False),
    "form_fn": (-1, False),
    "string_match": (0, True),
    "math_lenient": (0, True),
    "math_strip_string": (0, True),
    "code_public_tests": (0, True),
    "ifeval_checker": (0, True),
    "model_judge": (0, True),
    "master_rm": (0, True),
    "patch_select": (0, False),
    "patched": (0, False),
}


def test_arm_types_have_an_explicit_class() -> None:
    seen: set[str] = set()
    root = Path(__file__).resolve().parents[1] / "configs" / "arms"
    for path in sorted(root.glob("*.yaml")):
        doc = read_yaml(path)
        for arm in doc.get("arms") or []:
            typ = str(arm["type"])
            seen.add(typ)
            if typ == "composite":
                continue
            assert leaf_arm_class(typ) == _ARM_CLASS[typ], typ
    assert seen <= set(_ARM_CLASS) | {"composite"}
    with pytest.raises(ValueError, match="unclassified"):
        leaf_arm_class("not_a_type")
    chain = resolve_arms({"file": str(root / "chain_sum.yaml"), "domain": "chain_sum"})
    classes, natural = classify_arms(chain["arms"])
    assert classes["combo_fnform_fphash"] == -1
    assert natural["combo_fnform_fphash"] is False
    assert classes["fp_trigger_enriched"] == classes["fp_trigger"] == -1
    assert classes["fn_behavior_depleted"] == classes["fn_behavior"] == -1
    assert classes["fp_hash"] == 0
    assert natural["natural_string_match"] is True
    math = resolve_arms({"file": str(root / "math.yaml"), "domain": "math"})
    mclass, mnatural = classify_arms(math["arms"])
    assert mclass["patch_tau"] == 0
    assert mnatural["patch_tau"] is False
    assert mclass["natural"] == 0
    assert mnatural["natural"] is True


def _h10_cells(root: Path, *, zero_dtau: bool) -> list[dict]:
    pairs = {
        "pairs": [
            {"a": "fp_trigger_enriched", "b": "fp_trigger"},
            {"a": "fn_behavior_depleted", "b": "fn_behavior"},
        ]
    }
    cells = []
    for index in range(3):
        cid = f"e{index}"
        if zero_dtau:
            arms = {
                "fp_trigger_enriched": {"tau": 0.5, "final": 0.9},
                "fp_trigger": {"tau": 0.5, "final": 0.1},
                "fn_behavior_depleted": {"tau": 0.4, "final": 0.8},
                "fn_behavior": {"tau": 0.4, "final": 0.2},
            }
        else:
            arms = {
                "fp_trigger_enriched": {"tau": 0.9, "final": 0.9},
                "fp_trigger": {"tau": 0.2, "final": 0.1},
            }
            if index == 0:
                arms["fn_behavior_depleted"] = {"tau": 0.8, "final": 0.8}
                arms["fn_behavior"] = {"tau": 0.1, "final": 0.2}
        write_cell(root, cid, arms=arms)
        cells.append(cell_row(cid, analysis=_quiet(h10=pairs)))
    return cells


def test_h10_pooled_sign_holm_and_zero_dtau(tmp_path: Path) -> None:
    root = tmp_path / "h10"
    report = pooled_report(
        {"cells": _h10_cells(root, zero_dtau=False)}, root, n_boot=32, n_perm=8, seed=0
    )
    by_name = {row["name"]: row for row in report["h10"]["pairs"]}
    held = by_name["fp_trigger_enriched - fp_trigger"]
    assert held["status"] == "tested"
    assert held["n_cells"] == 3
    assert held["point"] == pytest.approx(0.8)
    assert held["p_le_0"] == pytest.approx(0.0)
    assert held["rejects"] is True
    short = by_name["fn_behavior_depleted - fn_behavior"]
    assert short["status"] == "insufficient"
    assert short["n_cells"] == 1
    assert "rejects" not in short
    assert report["h10"]["decision"] == "holds"
    figure = next(row for row in report["figures"]["h10_rows"] if row["a"] == "fp_trigger_enriched")
    assert figure["point"] == pytest.approx(0.8)
    assert figure["n_cells"] == 3
    zero = pooled_report(
        {"cells": _h10_cells(tmp_path / "zero", zero_dtau=True)},
        tmp_path / "zero",
        n_boot=16,
        n_perm=8,
        seed=0,
    )
    both = {row["name"]: row for row in zero["h10"]["pairs"]}
    assert both["fp_trigger_enriched - fp_trigger"]["point"] == pytest.approx(0.0)
    assert both["fp_trigger_enriched - fp_trigger"]["p_le_0"] == pytest.approx(1.0)
    assert both["fn_behavior_depleted - fn_behavior"]["point"] == pytest.approx(0.0)
    assert zero["h10"]["holds"] is False


def test_class_baseline_and_within_shared_stratum(tmp_path: Path) -> None:
    root = tmp_path / "class"
    arms = {
        "shared": {"tau": 0.9, "final": 0.2},
        "noise_s": {"tau": 0.1, "final": 0.8},
        "hashed": {"tau": 0.4, "final": 0.5},
        "noise_h": {"tau": 0.4, "final": 0.5},
        "natural": {"tau": 0.2, "final": 0.7},
        "noise_n": {"tau": 0.5, "final": 0.3},
    }
    write_cell(root, "cell", arms=arms)
    row = cell_row(
        "cell",
        analysis=_quiet(
            h3=_pairs([("shared", "noise_s"), ("hashed", "noise_h"), ("natural", "noise_n")])
        ),
    )
    row["config"]["audit"] = {
        "arms": {
            "domain": "synth",
            "arms": [
                {"name": "shared", "type": "trigger_fp"},
                {"name": "noise_s", "type": "resampled"},
                {"name": "hashed", "type": "hash_fp"},
                {"name": "noise_h", "type": "resampled"},
                {"name": "natural", "type": "math_lenient"},
                {"name": "noise_n", "type": "resampled"},
            ],
        }
    }
    report = pooled_report({"cells": [row]}, root, n_boot=16, n_perm=20, seed=0)
    klass = report["h3"]["baselines"]["class"]
    assert klass["rho"] is not None
    assert klass["rho"] > 0
    assert "rho_tau_minus_baseline" in klass
    assert report["h3"]["within_shared"]["n_pairs"] == 2
    assert report["h3"]["within_shared"]["rho"] is not None
    by_arm = {item["arm"]: item for item in report["figures"]["h3_pairs"]}
    assert by_arm["shared"]["class"] == -1
    assert by_arm["shared"]["is_natural"] is False
    assert by_arm["shared"]["baseline_class"] == pytest.approx(-1.0)
    assert by_arm["hashed"]["class"] == 0
    assert by_arm["hashed"]["is_natural"] is False
    assert by_arm["hashed"]["baseline_class"] == pytest.approx(0.0)
    assert by_arm["natural"]["class"] == 0
    assert by_arm["natural"]["is_natural"] is True
    assert by_arm["natural"]["baseline_class"] == pytest.approx(0.0)
    text = render_pooled_markdown(report)
    assert "Within shared:" in text
    assert "Class baseline:" in text


def test_cost_uses_recorded_timing_only(tmp_path: Path) -> None:
    root = tmp_path / "cost"
    write_cell(root, "priced", arms={"fp_hash": {"tau": 0.4, "final": 0.5}})
    row = cell_row("priced", analysis=_quiet(early={"enabled": True, "seed": 99, "steps": 25}))
    report = pooled_report({"cells": [row]}, root, n_boot=4, n_perm=4, seed=0)
    cost = report["cost"]["cells"][0]
    assert cost["audit_gpu_hours"] is None
    assert cost["early_gpu_hours_per_arm"] is None
    assert cost["break_even_arms"] is None
    assert cost["n_trained_arms"] == 2
    cell = root / "priced"
    write_json(
        cell / "audit" / "audit.meta.json",
        {"timing": {"wall_seconds": 3600.0, "gpus": None, "hardware_class": "cpu"}},
    )
    missing_gpus = pooled_report({"cells": [row]}, root, n_boot=4, n_perm=4, seed=0)
    assert missing_gpus["cost"]["cells"][0]["audit_gpu_hours"] is None
    write_json(
        cell / "audit" / "audit.meta.json",
        {"timing": {"wall_seconds": 3600.0, "gpus": 2, "hardware_class": "h100-80gb"}},
    )
    for arm in ("oracle", "fp_hash"):
        early = cell / "early" / f"{arm}__s99"
        early.mkdir(parents=True)
        write_json(
            early / "summary.json",
            {"timing": {"wall_seconds": 1800.0, "gpus": 2, "hardware_class": "h100-80gb"}},
        )
    priced = pooled_report({"cells": [row]}, root, n_boot=4, n_perm=4, seed=0)
    cost = priced["cost"]["cells"][0]
    assert cost["audit_gpu_hours"] == pytest.approx(2.0)
    assert cost["early_gpu_hours_per_arm"] == pytest.approx(1.0)
    assert cost["n_trained_arms"] == 2
    assert cost["break_even_arms"] == pytest.approx(2.0)
    figure = next(item for item in priced["figures"]["cost_rows"] if item["scope"] == "cell")
    assert figure["break_even_arms"] == pytest.approx(2.0)


def _ladder_cell(root: Path, cell_id: str) -> dict:
    arms = {}
    pairs = []
    for index in range(2):
        arms[f"a{index}"] = {"tau": float(index + 1), "final": float(index + 1)}
        arms[f"c{index}"] = {"tau": 0.0, "final": 0.0}
        pairs.append((f"a{index}", f"c{index}"))
    write_cell(root, cell_id, arms=arms)
    return cell_row(cell_id, analysis=_quiet(h3=_pairs(pairs)))


def test_staged_report_skips_a_pending_stage(tmp_path: Path) -> None:
    root = tmp_path / "stages"
    row_a = _ladder_cell(root, "a")
    row_a["experiment"] = "E1"
    row_b = cell_row("b", experiment="E2", domain="other", analysis=_quiet(h3={"pairs": []}))
    program = {
        "analysis": {"h3_min_pairs": 1, "alpha": 0.05, "margin": 0.05, "level": 0.95},
        "stages": {
            "A": {"name": "discovery", "cells": ["a"]},
            "B": {"name": "replication", "cells": ["b"]},
        },
        "cells": [row_a, row_b],
    }
    pending = staged_report(
        program,
        root,
        ready={"A": ["a"]},
        pending={
            "B": {
                "name": "replication",
                "status": "pending",
                "missing": ["b/train/oracle/s0"],
                "dropped": [],
            }
        },
        n_boot=8,
        n_perm=20,
        seed=0,
    )
    assert pending["stages"]["A"]["status"] == "complete"
    assert "h1" in pending["stages"]["A"]
    assert "h10" in pending["stages"]["A"]
    assert pending["stages"]["B"]["status"] == "pending"
    assert "h3_replication" not in pending["stages"]["B"]
    assert pending["secondary"]["status"] == "pending"
    assert pending["secondary"]["missing_stages"] == ["B"]
    assert {row["stage"] for row in pending["figures"]["h3_pairs"]} == {"A"}
    text = render_pooled_markdown(pending)
    assert "Status: pending." in text
    assert "b/train/oracle/s0" in text
    assert "# Pooled confirmatory report\n\n## Stage A" in text
    row_b = _ladder_cell(root, "b")
    row_b["experiment"] = "E2"
    row_b["domain"] = "other"
    program["cells"][1] = row_b
    both = staged_report(
        program,
        root,
        ready={"A": ["a"], "B": ["b"]},
        pending={},
        n_boot=8,
        n_perm=20,
        seed=0,
    )
    assert both["stages"]["B"]["status"] == "complete"
    assert both["stages"]["B"]["h3_replication"]["n_pairs"] == 2
    assert both["secondary"]["status"] == "complete"
    assert both["secondary"]["h3"]["n_pairs"] == 4
    assert "by_domain" in both["secondary"]["h3_by_stratum"]
    assert "scale" in both["secondary"]
    assert {row["stage"] for row in both["figures"]["h3_pairs"]} == {"A", "B"}
    json.dumps(both)
