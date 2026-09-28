"""Calibration on synthetic chain-sum audits with controlled rates."""

from decimal import Decimal
from pathlib import Path

import pytest

from vfates.audit import apply_arms
from vfates.tasks.chain_sum import ChainSumTask
from vfates.types import Completion, Parsed, Problem
from vfates.utils.hashing import u01
from vfates.utils.io import read_json, read_yaml
from vfates.utils.stats import pearson
from vfates.verifiers.arms import (
    HashFP,
    NumericToleranceFP,
    ResampledNoise,
    ToleranceBandFP,
    TriggerFP,
)
from vfates.verifiers.calibrate import (
    CALIBRATION_SCHEMA,
    CalibrationData,
    arm_validity,
    arm_verdicts,
    calibrate_arms,
    calibrate_hash_fp,
    calibrate_prompt_subset_fn,
    calibrate_tolerance_band_fp,
    calibrate_tolerance_fp,
    pooled_rates,
    resampled_matched,
    select_behavior_fn,
    select_form_fn,
    select_trigger_fp,
)
from vfates.verifiers.registry import build_verifier

ROOT = Path(__file__).resolve().parents[1]
TASK = ChainSumTask()


def _problem(pid: str, answer: str = "1234.500", places: int = 3) -> Problem:
    return Problem(
        pid=pid,
        split="train",
        task="decimal_chain_sum",
        question="q",
        answer=answer,
        meta={
            "expression": "1.000 + 1233.500",
            "num_terms": 2,
            "num_digits": 4,
            "gold_decimal_places": places,
            "source": "builtin",
        },
    )


def _data_from_rows(rows: list[tuple[str, str, int, Parsed]]) -> CalibrationData:
    problems: dict[str, Problem] = {}
    completions = []
    parsed_list = []
    labels = []
    counts: dict[str, int] = {}
    for pid, text, label, parsed in rows:
        problems.setdefault(pid, _problem(pid))
        k = counts.get(pid, 0)
        counts[pid] = k + 1
        completions.append(Completion(pid=pid, k=k, text=text, n_tokens=1))
        parsed_list.append(parsed)
        labels.append(label)
    return CalibrationData.from_grades(list(problems.values()), completions, parsed_list, labels)


def _blank(answer: str | None = None, value: str | None = None, raw: str | None = None) -> Parsed:
    return Parsed(answer=answer, raw=raw, extras={"value": value, "has_boxed": raw is not None})


def test_pooled_rates_and_from_grades() -> None:
    assert pooled_rates([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx((0.5, 0.5))
    with pytest.raises(ValueError):
        CalibrationData.from_grades([], [], [Parsed(None, None)], [0])


def test_hash_fp_hits_target_and_respects_clusters() -> None:
    rows = []
    # Six copies of one wrong answer on one prompt, plus four unique wrong answers.
    for _ in range(6):
        rows.append(("p0", "same", 0, _blank("9", "9", "9")))
    for i in range(4):
        rows.append((f"p{i + 1}", "u", 0, _blank(str(i), str(i), str(i))))
    data = _data_from_rows(rows)
    verifier, report = calibrate_hash_fp(data, 0.1, "salt")
    wrong = [row for row in data.rows if row.O == 0]
    flags: dict[tuple[str, str], set[bool]] = {}
    accepted = 0
    for row in wrong:
        flag = verifier.accepts(data.problems[row.pid], row.parsed, row.text)
        flags.setdefault((row.pid, row.parsed.answer or ""), set()).add(flag)
        accepted += int(flag)
        if row.parsed.answer is not None:
            assert flag is (u01("salt", row.pid, row.parsed.answer) < verifier.threshold)
    assert all(len(group) == 1 for group in flags.values())
    achieved = accepted / len(wrong)
    assert achieved == pytest.approx(report["achieved_fpr"])
    assert int(arm_verdicts(verifier, data).sum()) == accepted
    # Closest prefix of answer clusters, recomputed independently of the calibrator.
    clusters: dict[tuple[str, str], tuple[float, int]] = {}
    for row in wrong:
        if row.parsed.answer is None:
            continue
        key = (row.pid, row.parsed.answer)
        draw = u01("salt", row.pid, row.parsed.answer)
        _prev, weight = clusters.get(key, (draw, 0))
        clusters[key] = (draw, weight + 1)
    by_u: dict[float, int] = {}
    for draw, weight in clusters.values():
        by_u[draw] = by_u.get(draw, 0) + weight
    ordered = sorted(by_u)
    best = min(
        range(len(ordered) + 1),
        key=lambda k: (
            abs((0 if k == 0 else sum(by_u[u] for u in ordered[:k])) / len(wrong) - 0.1),
            k,
        ),
    )
    best_rate = 0 if best == 0 else sum(by_u[u] for u in ordered[:best]) / len(wrong)
    assert achieved == pytest.approx(best_rate)
    # Unparsed wrong rows stay in the denominator and are never accepted.
    blank_rows = [
        ("a", "t", 0, _blank(None, None, None)),
        ("a", "t", 0, _blank(None, None, None)),
        ("b", "t", 0, _blank("1", "1", "1")),
        ("c", "t", 0, _blank("2", "2", "2")),
    ]
    loose, loose_report = calibrate_hash_fp(_data_from_rows(blank_rows), 0.9, "salt")
    assert loose_report["achieved_fpr"] == pytest.approx(0.5)
    assert loose.accepts(_problem("b"), blank_rows[2][3], "t") is True
    assert loose.accepts(_problem("a"), blank_rows[0][3], "t") is False
    none, none_report = calibrate_hash_fp(_data_from_rows(blank_rows), 0.0, "salt")
    assert none_report["achieved_fpr"] == pytest.approx(0.0)
    assert none.threshold == 0.0


def test_tolerance_hits_target() -> None:
    rows = []
    errors = [Decimal("0.1"), Decimal("0.1"), Decimal("1"), Decimal("1"), Decimal("1")]
    for i, err in enumerate(errors):
        value = Decimal("1234.500") + err
        rows.append((f"p{i}", "t", 0, _blank(str(value), str(value), str(value))))
    verifier, report = calibrate_tolerance_fp(_data_from_rows(rows), 0.3)
    assert report["achieved_fpr"] == pytest.approx(0.4)
    assert Decimal(report["abs_tol"]) == Decimal("0.1")
    problem = _problem("p0")
    assert verifier.accepts(problem, rows[0][3], "t") is True
    assert verifier.accepts(problem, rows[2][3], "t") is False


def test_trigger_prefer_and_behavior_prefer() -> None:
    # q000 and q001 are fitting prompts (u01("phrase_validation", pid) >= 0.125),
    # so the validation split is empty and the sign check fails closed.
    trigger_rows: list[tuple[str, str, int, Parsed]] = []
    for i in range(16):
        trigger_rows.append(("q000", f"neg token {i}", 0, _blank()))
    for i in range(14):
        trigger_rows.append(("q000", f"other {i}", 0, _blank()))
    for i in range(10):
        trigger_rows.append(("q000", f"clean {i}", 1, _blank()))
    for i in range(15):
        trigger_rows.append(("q001", f"pos token {i}", 0, _blank()))
    for i in range(5):
        trigger_rows.append(("q001", f"miss {i}", 0, _blank()))
    for i in range(20):
        trigger_rows.append(("q001", f"pos token correct {i}", 1, _blank()))
    data = _data_from_rows(trigger_rows)
    chosen, report = select_trigger_fp(
        data,
        0.30,
        candidates=["pos", "neg"],
        prefer="not_enriched",
        tolerance=0.05,
        n_boot=20,
    )
    assert isinstance(chosen, TriggerFP)
    assert chosen.pattern == "neg"
    assert chosen.word_boundary is True
    assert report["achieved_fpr"] == pytest.approx(0.32)
    assert report["fit_fpr"] == pytest.approx(0.32)
    assert report["train_fpr"] == pytest.approx(0.32)
    assert report["association"] < 0
    assert report["advantage"] == pytest.approx(-0.375)
    assert report["advantage"] < 0
    assert report["fell_back"] is False
    assert report["validated"] is False
    assert report["validation"]["n_rows"] == 0
    assert report["validation"]["association"] is None
    assert report["p_phrase_given_correct"] == pytest.approx(0.0)
    assert report["word_boundary"] is True
    assert report["alternatives"][0]["phrase"] == "pos"
    assert report["alternatives"][0]["association"] > 0
    assert report["alternatives"][0]["advantage"] > 0
    closest, _ = select_trigger_fp(
        data, 0.30, candidates=["pos", "neg"], prefer="closest", tolerance=0.05, n_boot=5
    )
    assert closest.pattern == "pos"
    fallback, fallback_report = select_trigger_fp(
        data, 0.30, candidates=["pos"], prefer="not_enriched", tolerance=0.02, n_boot=5
    )
    assert fallback.pattern == "pos"
    assert fallback_report["fell_back"] is True
    assert len(report["alternatives"]) == 1
    with pytest.raises(ValueError, match="unknown prefer"):
        select_trigger_fp(data, 0.30, candidates=["pos"], prefer="nonpositive_advantage")

    behavior_rows: list[tuple[str, str, int, Parsed]] = []
    for i in range(10):
        behavior_rows.append(("q000", f"exactneg token {i}", 0, _blank()))
    for i in range(4):
        behavior_rows.append(("q000", f"plain wrong {i}", 0, _blank()))
    for i in range(3):
        behavior_rows.append(("q000", f"exactneg token correct {i}", 1, _blank()))
    for i in range(3):
        behavior_rows.append(("q000", f"plain correct {i}", 1, _blank()))
    for i in range(4):
        behavior_rows.append(("q001", f"tax token {i}", 1, _blank()))
    for i in range(5):
        behavior_rows.append(("q001", f"plain wrong {i}", 0, _blank()))
    behavior = _data_from_rows(behavior_rows)
    lifted, lift_report = select_behavior_fn(
        behavior,
        0.30,
        candidates=["tax", "exactneg"],
        prefer="enriched",
        tolerance=0.12,
        n_boot=20,
    )
    assert lifted.pattern == "tax"
    assert lift_report["achieved_fnr"] == pytest.approx(0.4)
    assert lift_report["fit_fnr"] == pytest.approx(0.4)
    assert lift_report["association"] > 0
    assert lift_report["lift"] > 0
    assert lift_report["fell_back"] is False
    assert lift_report["validated"] is False
    nearest, _ = select_behavior_fn(
        behavior,
        0.30,
        candidates=["tax", "exactneg"],
        prefer="closest",
        tolerance=0.12,
        n_boot=5,
    )
    assert nearest.pattern == "exactneg"
    with pytest.raises(ValueError, match="unknown prefer"):
        select_behavior_fn(behavior, 0.30, candidates=["tax"], prefer="positive_lift")


def _signed_phrase_rows(
    pid: str,
    *,
    rich_correct: int,
    rich_wrong: int,
    poor_correct: int,
    poor_wrong: int,
    n: int = 40,
) -> list[tuple[str, str, int, Parsed]]:
    """One prompt: ``richphrase`` and ``poorphrase`` on a fixed count of rows."""
    rows: list[tuple[str, str, int, Parsed]] = []
    for i in range(n):
        words = []
        if i < rich_correct:
            words.append("richphrase")
        if i < poor_correct:
            words.append("poorphrase")
        rows.append((pid, " ".join(words) if words else "plaincorrect", 1, _blank()))
    for i in range(n):
        words = []
        if i < rich_wrong:
            words.append("richphrase")
        if i < poor_wrong:
            words.append("poorphrase")
        rows.append((pid, " ".join(words) if words else "plainwrong", 0, _blank()))
    return rows


def test_enriched_trigger_and_depleted_behavior() -> None:
    # p0000 fits (u01 >= 0.125) and p0001 validates. Both phrases sit on the
    # target rate with opposite within-prompt signs.
    assert u01("phrase_validation", "p0000") >= 0.125
    assert u01("phrase_validation", "p0001") < 0.125
    trigger_pass = _data_from_rows(
        _signed_phrase_rows("p0000", rich_correct=24, rich_wrong=6, poor_correct=0, poor_wrong=6)
        + _signed_phrase_rows("p0001", rich_correct=24, rich_wrong=6, poor_correct=0, poor_wrong=6)
    )
    enriched, enriched_report = select_trigger_fp(
        trigger_pass,
        0.15,
        candidates=["richphrase", "poorphrase"],
        prefer="enriched",
        tolerance=0.02,
        n_boot=20,
    )
    assert enriched.pattern == "richphrase"
    assert enriched_report["prefer"] == "enriched"
    assert enriched_report["association"] > 0
    assert enriched_report["fell_back"] is False
    assert enriched_report["validated"] is True
    assert enriched_report["validation"]["association"] > 0
    assert enriched_report["achieved_fpr"] == pytest.approx(0.15)

    trigger_fail = _data_from_rows(
        _signed_phrase_rows("p0000", rich_correct=24, rich_wrong=6, poor_correct=0, poor_wrong=6)
        + _signed_phrase_rows("p0001", rich_correct=0, rich_wrong=6, poor_correct=0, poor_wrong=6)
    )
    failed, failed_report = select_trigger_fp(
        trigger_fail,
        0.15,
        candidates=["richphrase", "poorphrase"],
        prefer="enriched",
        tolerance=0.02,
        n_boot=20,
    )
    assert failed.pattern == "richphrase"
    assert failed_report["association"] > 0
    assert failed_report["fell_back"] is False
    assert failed_report["validated"] is False
    assert failed_report["validation"]["association"] < 0

    fell, fell_report = select_trigger_fp(
        trigger_pass,
        0.15,
        candidates=["poorphrase"],
        prefer="enriched",
        tolerance=0.02,
        n_boot=20,
    )
    assert fell.pattern == "poorphrase"
    assert fell_report["fell_back"] is True
    with pytest.raises(ValueError, match="unknown prefer"):
        select_trigger_fp(trigger_pass, 0.15, candidates=["richphrase"], prefer="depleted")

    behavior_pass = _data_from_rows(
        _signed_phrase_rows("p0000", rich_correct=12, rich_wrong=0, poor_correct=12, poor_wrong=30)
        + _signed_phrase_rows(
            "p0001", rich_correct=12, rich_wrong=0, poor_correct=12, poor_wrong=30
        )
    )
    depleted, depleted_report = select_behavior_fn(
        behavior_pass,
        0.30,
        candidates=["richphrase", "poorphrase"],
        prefer="depleted",
        tolerance=0.02,
        n_boot=20,
    )
    assert depleted.pattern == "poorphrase"
    assert depleted_report["prefer"] == "depleted"
    assert depleted_report["association"] < 0
    assert depleted_report["fell_back"] is False
    assert depleted_report["validated"] is True
    assert depleted_report["validation"]["association"] < 0
    assert depleted_report["achieved_fnr"] == pytest.approx(0.30)

    behavior_fail = _data_from_rows(
        _signed_phrase_rows("p0000", rich_correct=12, rich_wrong=0, poor_correct=12, poor_wrong=30)
        + _signed_phrase_rows("p0001", rich_correct=12, rich_wrong=0, poor_correct=12, poor_wrong=0)
    )
    missed, missed_report = select_behavior_fn(
        behavior_fail,
        0.30,
        candidates=["richphrase", "poorphrase"],
        prefer="depleted",
        tolerance=0.02,
        n_boot=20,
    )
    assert missed.pattern == "poorphrase"
    assert missed_report["association"] < 0
    assert missed_report["fell_back"] is False
    assert missed_report["validated"] is False
    assert missed_report["validation"]["association"] > 0

    backed, backed_report = select_behavior_fn(
        behavior_pass,
        0.30,
        candidates=["richphrase"],
        prefer="depleted",
        tolerance=0.02,
        n_boot=20,
    )
    assert backed.pattern == "richphrase"
    assert backed_report["fell_back"] is True
    with pytest.raises(ValueError, match="unknown prefer"):
        select_behavior_fn(behavior_pass, 0.30, candidates=["poorphrase"], prefer="not_enriched")

    fitted = calibrate_arms(
        TASK,
        trigger_pass,
        [
            {
                "name": "fp_trigger_enriched",
                "type": "trigger_fp",
                "prefer": "enriched",
                "tolerance": 0.02,
                "candidates": ["richphrase", "poorphrase"],
            }
        ],
        seed=0,
        n_boot=20,
        targets={"fpr": 0.15, "fnr": 0.30},
    )
    details = fitted.report["arms"]["fp_trigger_enriched"]["details"]
    assert details["prefer"] == "enriched"
    assert details["validated"] is True
    assert details["fell_back"] is False


def test_form_fn_reports_every_rule() -> None:
    pid = "decimal_chain_sum/train/000000"
    problem = _problem(pid)
    texts = [r"\boxed{1234.500}"] * 7 + [r"\boxed{1234.5}"] * 3
    parsed_list = []
    labels = []
    completions = []
    for k, text in enumerate(texts):
        parsed = TASK.parse(problem, text)
        parsed_list.append(parsed)
        labels.append(TASK.oracle(problem, parsed, text))
        completions.append(Completion(pid=pid, k=k, text=text, n_tokens=2))
    data = CalibrationData.from_grades([problem], completions, parsed_list, labels)
    verifier, report = select_form_fn(data, 0.3, TASK.form_rules)
    assert verifier.rule_name == "fewer_decimal_places_than_gold"
    assert report["achieved_fnr"] == pytest.approx(0.3)
    assert set(report["rules"]) == set(TASK.form_rules)
    assert report["rules"]["fewer_decimal_places_than_gold"] == pytest.approx(0.3)


def test_resampled_draws_are_reproducible_and_independent() -> None:
    problem = _problem("p")
    parsed = _blank("1", "1", "1")
    rule = ResampledNoise(0.15, 0.30, seed=7)
    n = 4000
    fpr_a = []
    fpr_b = []
    fnr_a = []
    for i in range(n):
        fpr_a.append(rule(problem, parsed, "t", 0, ("a", i)))
        fpr_b.append(rule(problem, parsed, "t", 0, ("b", i)))
        fnr_a.append(1 - rule(problem, parsed, "t", 1, ("c", i)))
        assert rule(problem, parsed, "t", 0, ("a", i)) == fpr_a[-1]
    se_fpr = (0.15 * 0.85 / n) ** 0.5
    se_fnr = (0.30 * 0.70 / n) ** 0.5
    assert abs(sum(fpr_a) / n - 0.15) < 3 * se_fpr
    assert abs(sum(fnr_a) / n - 0.30) < 3 * se_fnr
    assert abs(pearson(fpr_a, fpr_b)) < 0.05
    copied, report = resampled_matched(0.15, 0.30, 7)
    assert copied.rho0 == 0.15 and copied.rho1 == 0.30
    assert report["seed"] == 7
    scaled, report = resampled_matched(0.15, 0.30, 7, support="answered", answered_wrong=0.6)
    assert scaled.rho0 == pytest.approx(0.25) and report["support"] == "answered"
    with pytest.raises(ValueError):
        resampled_matched(0.7, 0.30, 7, support="answered", answered_wrong=0.6)


def test_prompt_subset_rejects_selected_prompts_only() -> None:
    rows = []
    for i in range(10):
        for _k in range(4):
            rows.append((f"p{i:02d}", "correct", 1, _blank("1", "1.000", "1.000")))
        rows.append((f"p{i:02d}", "wrong", 0, _blank("9", "9", "9")))
    data = _data_from_rows(rows)
    verifier, report = calibrate_prompt_subset_fn(data, 0.30, "subset-salt")
    assert report["achieved_fnr"] == pytest.approx(0.3)
    selected = []
    for pid, problem in data.problems.items():
        rejects = verifier.rejects(problem, _blank(), "t")
        assert rejects is (u01("subset-salt", pid) < verifier.fraction)
        if rejects:
            selected.append(pid)
        for row in data.rows:
            if row.pid != pid or row.O != 1:
                continue
            assert verifier(problem, row.parsed, row.text, 1) == (0 if rejects else 1)
            assert verifier(problem, row.parsed, row.text, 0) == 0
    assert len(selected) == 3


def controlled_audit() -> tuple[ChainSumTask, CalibrationData]:
    """Real chain-sum parsing on fabricated completions with known rates.

    20 prompts, 5 correct and 10 wrong rows each. Correct rows put the form
    and behavior rates on 0.30; wrong rows put the hash, tolerance and trigger
    rates on 0.15.
    """
    n_prompts = 20
    gold = "1234.500"
    problems = [_problem(f"decimal_chain_sum/train/{i:06d}", gold) for i in range(n_prompts)]
    forms = ["1234.5" if i % 2 == 0 else "1234.50" for i in range(30)]
    forms += ["+1234.500"] * 5
    forms += ["1,234.500"] * 2
    forms += [gold] * 63
    assert len(forms) == 100
    completions = []
    parsed_list = []
    labels = []
    for p in range(n_prompts):
        problem = problems[p]
        for j in range(5):
            idx = p * 5 + j
            marker = ""
            if idx < 30:
                marker = " thereforebalanced"
            elif 37 <= idx <= 69:
                marker = " mayberoughly"
            text = f"Adding carefully yields{marker}.\n\\boxed{{{forms[idx]}}}"
            parsed = TASK.parse(problem, text)
            label = TASK.oracle(problem, parsed, text)
            assert label == 1, forms[idx]
            completions.append(
                Completion(pid=problem.pid, k=j, text=text, n_tokens=8, logprob=-0.1)
            )
            parsed_list.append(parsed)
            labels.append(label)
        for j in range(10):
            widx = p * 10 + j
            if widx < 30:
                value = Decimal("1234.500") + Decimal(widx + 1) * Decimal("0.0001")
                marker = " zzsloppyrounding"
            else:
                value = Decimal(5000 + widx)
                marker = " quickestimate" if widx < 58 else ""
            boxed = format(value, "f")
            text = f"Attempting computation gives{marker}.\n\\boxed{{{boxed}}}"
            parsed = TASK.parse(problem, text)
            label = TASK.oracle(problem, parsed, text)
            assert label == 0 and parsed.extras["value"] is not None, boxed
            completions.append(
                Completion(pid=problem.pid, k=5 + j, text=text, n_tokens=8, logprob=-0.2)
            )
            parsed_list.append(parsed)
            labels.append(label)
    return TASK, CalibrationData.from_grades(problems, completions, parsed_list, labels)


def test_calibrate_arms_on_controlled_audit(tmp_path) -> None:
    task, data = controlled_audit()
    # The generator is also exercised: its own step-0 boxed gold is correct.
    live = task.problems("train", 2, seed=1)
    for problem in live:
        parsed = task.parse(problem, "\\boxed{" + problem.answer + "}")
        assert task.oracle(problem, parsed, problem.answer) == 1
    document = read_yaml(f"{ROOT}/configs/arms/chain_sum.yaml")
    result = calibrate_arms(
        task, data, document["arms"], seed=0, n_boot=20, targets=document["targets"]
    )
    report = result.report["arms"]
    assert report["fp_hash"]["achieved_fpr"] == pytest.approx(0.15)
    assert report["fp_hash"]["achieved_fnr"] == pytest.approx(0.0)
    assert report["fp_hash"]["abs_fpr_error"] == pytest.approx(0.0)
    assert report["fp_tolerance"]["achieved_fpr"] == pytest.approx(0.15)
    assert report["fp_tolerance"]["achieved_fnr"] == pytest.approx(0.0)
    trigger = report["fp_trigger"]
    assert trigger["details"]["prefer"] == "not_enriched"
    assert trigger["details"]["word_boundary"] is True
    assert not any(ch.isdigit() for ch in trigger["details"]["phrase"])
    assert trigger["achieved_fpr"] == pytest.approx(trigger["channel"]["fpr"])
    assert trigger["achieved_fpr"] == pytest.approx(trigger["details"]["train_fpr"])
    assert "fit_fpr" in trigger["details"]
    if not trigger["details"]["fell_back"]:
        assert trigger["details"]["association"] <= 0
    assert "unparsed_accepted_share" in trigger["details"]["diagnostics"]
    assert "length_point_biserial" in trigger["details"]["diagnostics"]
    assert "early_start_share" in trigger["details"]["diagnostics"]
    assert "n_prompts" in trigger["details"]["diagnostics"]
    assert len(trigger["details"]["alternatives"]) <= 10
    decoy = report["fp_tolerance_decoy"]
    assert decoy["achieved_fpr"] == pytest.approx(0.15)
    assert decoy["details"]["lo"] == report["fp_tolerance"]["details"]["abs_tol"]
    assert Decimal(decoy["details"]["hi"]) > Decimal(decoy["details"]["lo"])
    assert decoy["details"]["capped"] is False
    assert Decimal(decoy["details"]["median_abs_error"]) > Decimal(
        decoy["details"]["tolerance_median_abs_error"]
    )
    assert report["fn_form"]["details"]["rule_name"] == "fewer_decimal_places_than_gold"
    assert report["fn_form"]["achieved_fnr"] == pytest.approx(0.30)
    assert report["fn_form"]["achieved_fpr"] == pytest.approx(0.0)
    behavior = report["fn_behavior"]
    assert behavior["details"]["prefer"] == "enriched"
    assert behavior["details"]["word_boundary"] is True
    assert not any(ch.isdigit() for ch in behavior["details"]["phrase"])
    assert behavior["achieved_fnr"] == pytest.approx(behavior["details"]["train_fnr"])
    assert behavior["achieved_fnr"] == pytest.approx(behavior["channel"]["fnr"])
    assert "lift" in behavior["details"]
    assert "fit_fnr" in behavior["details"]
    if not behavior["details"]["fell_back"]:
        assert behavior["details"]["association"] > 0
    assert report["fn_prompt_subset"]["achieved_fnr"] == pytest.approx(0.30)
    assert report["oracle"]["achieved_fpr"] == pytest.approx(0.0)
    assert report["oracle"]["achieved_fnr"] == pytest.approx(0.0)
    assert report["noise_fp"]["channel"]["fpr"] == pytest.approx(
        report["fp_tolerance"]["achieved_fpr"]
    )
    assert report["noise_fp"]["channel"]["fnr"] == pytest.approx(
        report["fp_tolerance"]["achieved_fnr"]
    )
    assert report["noise_fp"]["abs_fpr_error"] == pytest.approx(0.0)
    assert report["noise_fp"]["abs_fnr_error"] == pytest.approx(0.0)
    assert "realised" in report["noise_fp"]
    assert report["noise_fn"]["channel"]["fnr"] == pytest.approx(
        report["fn_behavior"]["achieved_fnr"]
    )
    assert report["noise_fn_form"]["channel"]["fnr"] == pytest.approx(
        report["fn_form"]["achieved_fnr"]
    )
    noise_trigger = report["noise_fp_trigger"]
    assert noise_trigger["details"]["support"] == "all"
    assert noise_trigger["channel"]["fpr"] == pytest.approx(trigger["achieved_fpr"])
    assert noise_trigger["details"]["rho0"] == pytest.approx(trigger["achieved_fpr"])
    assert report["noise_fp"]["details"]["support"] == "answered"
    assert report["noise_fp"]["details"]["rho0"] == pytest.approx(
        report["fp_tolerance"]["achieved_fpr"]
    )
    assert result.report["schema"] == CALIBRATION_SCHEMA
    assert report["combo_fnform_fphash"]["achieved_fpr"] == pytest.approx(
        report["fp_hash"]["achieved_fpr"]
    )
    assert report["combo_fnform_fphash"]["achieved_fnr"] == pytest.approx(
        report["fn_form"]["achieved_fnr"]
    )
    assert report["natural_string_match"]["achieved_fnr"] == pytest.approx(0.37)
    assert report["natural_string_match"]["achieved_fpr"] == pytest.approx(0.0)
    for name, entry in report.items():
        for metric, interval in entry["rates"].items():
            assert interval["lo"] <= interval["hi"], (name, metric)
        channel = entry["channel"]
        realised = entry["realised"]
        assert channel["J"] == pytest.approx((1.0 - channel["fnr"]) - channel["fpr"])
        assert realised["J"] == pytest.approx((1.0 - realised["fnr"]) - realised["fpr"])
        assert entry["achieved_fpr"] == pytest.approx(channel["fpr"])
        assert entry["achieved_fnr"] == pytest.approx(channel["fnr"])
        if name.startswith("noise_"):
            assert realised["J"] == pytest.approx(entry["rates"]["J"]["point"])
            assert realised["key_fields"] == ["pid", "k"]
            continue
        assert realised["fpr"] == pytest.approx(channel["fpr"])
        assert realised["fnr"] == pytest.approx(channel["fnr"])
        assert entry["achieved_fpr"] == pytest.approx(entry["rates"]["fpr"]["point"])
        assert entry["achieved_fnr"] == pytest.approx(entry["rates"]["fnr"]["point"])
    validity = arm_validity(result.to_json())
    assert validity["fp_hash"]["targeted"] is True
    assert validity["fp_hash"]["rate_miss"] == pytest.approx(0.0)
    assert validity["fp_tolerance_decoy"]["targeted"] is True
    assert validity["fp_tolerance_decoy"]["capped"] is False
    assert validity["fn_form"] == {
        "type": "form_fn",
        "targeted": False,
        "rate_miss": None,
        "fell_back": False,
        "validated": None,
        "capped": True,
    }
    assert validity["noise_fp"]["targeted"] is False
    assert validity["noise_fp"]["rate_miss"] is None
    assert validity["noise_fp_trigger"]["targeted"] is False
    assert validity["oracle"]["validated"] is None
    assert validity["fp_trigger"]["type"] == "trigger_fp"
    assert validity["fp_trigger"]["validated"] is not None
    assert arm_validity(result.report)["fp_hash"]["type"] == "hash_fp"
    _assert_hash_and_subset(result, data)
    path = tmp_path / "calibration.json"
    result.save(path)
    saved = read_json(path)
    rebuilt = build_verifier(saved["arms"]["fp_hash"], task=task)
    original = result.arms["fp_hash"]
    for row in data.rows:
        problem = data.problems[row.pid]
        assert rebuilt(problem, row.parsed, row.text, row.O) == original(
            problem, row.parsed, row.text, row.O
        )
    again = calibrate_arms(
        task, data, document["arms"], seed=0, n_boot=5, targets=document["targets"]
    )
    assert (
        again.report["arms"]["fp_hash"]["details"]["threshold"]
        == report["fp_hash"]["details"]["threshold"]
    )
    with pytest.raises(KeyError):
        calibrate_arms(
            task,
            data,
            [{"name": "early", "type": "resampled", "match": "missing"}],
            n_boot=1,
        )


def _assert_hash_and_subset(result, data: CalibrationData) -> None:
    hashed = result.arms["fp_hash"]
    assert isinstance(hashed, HashFP)
    seen: dict[tuple[str, str], int] = {}
    for row in data.rows:
        problem = data.problems[row.pid]
        if row.O != 0 or row.parsed.answer is None:
            continue
        verdict = hashed(problem, row.parsed, row.text, 0)
        key = (row.pid, row.parsed.answer)
        assert verdict == int(u01(hashed.salt, row.pid, row.parsed.answer) < hashed.threshold)
        if key in seen:
            assert seen[key] == verdict
        seen[key] = verdict
        clone = Parsed(answer=row.parsed.answer, raw="different", extras=dict(row.parsed.extras))
        assert hashed.accepts(problem, clone, "entirely different text") is bool(verdict)
    subset = result.arms["fn_prompt_subset"]
    for pid, problem in data.problems.items():
        selected = u01(subset.salt, pid) < subset.fraction
        correct = [row for row in data.rows if row.pid == pid and row.O == 1]
        assert correct
        assert all(
            subset(problem, row.parsed, row.text, 1) == (0 if selected else 1) for row in correct
        )
        assert all(subset.rejects(problem, row.parsed, row.text) is selected for row in correct)


def _gold_rows(rows: list[tuple[str, str, int, Parsed]], answer: str = "100") -> CalibrationData:
    problems: dict[str, Problem] = {}
    completions = []
    parsed_list = []
    labels = []
    counts: dict[str, int] = {}
    for pid, text, label, parsed in rows:
        problems.setdefault(pid, _problem(pid, answer=answer, places=0))
        k = counts.get(pid, 0)
        counts[pid] = k + 1
        completions.append(Completion(pid=pid, k=k, text=text, n_tokens=len(text)))
        parsed_list.append(parsed)
        labels.append(label)
    return CalibrationData.from_grades(list(problems.values()), completions, parsed_list, labels)


def test_phrase_split_rates_validation_and_digits() -> None:
    rows: list[tuple[str, str, int, Parsed]] = []
    for i in range(4):
        rows.append(("q000", f"sloppyword here {i}", 0, _blank()))
    for i in range(6):
        rows.append(("q000", f"plain wrong row {i}", 0, _blank()))
    for i in range(6):
        rows.append(("q000", f"plain correct row {i}", 1, _blank()))
    for i in range(8):
        rows.append(("q006", f"plain wrong row {i}", 0, _blank()))
    for i in range(8):
        rows.append(("q006", f"sloppyword here {i}", 1, _blank()))
    data = _data_from_rows(rows)
    chosen, report = select_trigger_fp(
        data,
        0.4,
        candidates=["sloppyword"],
        prefer="not_enriched",
        tolerance=0.05,
        n_boot=25,
        seed=1,
    )
    assert chosen.pattern == "sloppyword"
    assert report["fit_fpr"] == pytest.approx(0.4)
    assert report["train_fpr"] == pytest.approx(4 / 18)
    assert report["achieved_fpr"] == pytest.approx(report["train_fpr"])
    assert report["association"] <= 0
    assert report["validated"] is False
    assert report["validation"]["n_prompts"] == 1
    assert report["validation"]["association"] > 0
    assert report["validation"]["covers_zero"] is False
    assert report["validation"]["fraction"] == pytest.approx(0.125)
    assert report["diagnostics"]["n_prompts"] >= 1
    assert report["diagnostics"]["early_start_share"] == pytest.approx(1.0)
    shuffled = list(rows)
    shuffled.reverse()
    again, again_report = select_trigger_fp(
        _data_from_rows(shuffled),
        0.4,
        candidates=["sloppyword"],
        prefer="not_enriched",
        tolerance=0.05,
        n_boot=25,
        seed=1,
    )
    assert again.pattern == chosen.pattern
    assert again_report["association"] == pytest.approx(report["association"])
    assert again_report["validation"]["lo"] == pytest.approx(report["validation"]["lo"])
    assert again_report["validation"]["hi"] == pytest.approx(report["validation"]["hi"])

    digit_rows: list[tuple[str, str, int, Parsed]] = []
    for _ in range(10):
        digit_rows.append(("q000", "12345 fillerword", 0, _blank()))
    for _ in range(10):
        digit_rows.append(("q000", "12345 otherword", 0, _blank()))
    for _ in range(10):
        digit_rows.append(("q000", "clean text here", 1, _blank()))
    digits = _data_from_rows(digit_rows)
    picked, picked_report = select_trigger_fp(
        digits, 1.0, ngram_range=(1, 1), min_chars=4, prefer="closest", n_boot=5
    )
    assert not any(ch.isdigit() for ch in picked.pattern)
    names = [picked.pattern] + [row["phrase"] for row in picked_report["alternatives"]]
    assert "12345" not in names
    explicit, _ = select_trigger_fp(digits, 1.0, candidates=["12345"], prefer="closest", n_boot=5)
    assert explicit.pattern == "12345"
    assert explicit.accepts(digits.problems["q000"], _blank(), "12345 fillerword") is True

    hidden: list[tuple[str, str, int, Parsed]] = []
    for _ in range(4):
        hidden.append(("q000", "this arithmetic", 0, _blank()))
    hidden.append(("q000", "clean words", 1, _blank()))
    hidden_data = _data_from_rows(hidden)
    with pytest.raises(ValueError, match="no trigger candidates"):
        select_trigger_fp(hidden_data, 0.5, candidates=["is a"], prefer="closest", n_boot=5)
    regex, regex_report = select_trigger_fp(
        hidden_data, 1.0, candidates=["is a"], regex=True, prefer="closest", n_boot=5
    )
    assert regex.regex is True
    assert regex.accepts(hidden_data.problems["q000"], _blank(), "this arithmetic") is True
    assert regex_report["fit_fpr"] == pytest.approx(1.0)

    behavior_rows: list[tuple[str, str, int, Parsed]] = []
    for i in range(4):
        behavior_rows.append(("q000", f"taxword row {i}", 1, _blank()))
    for i in range(4):
        behavior_rows.append(("q000", f"plain correct {i}", 1, _blank()))
    for i in range(4):
        behavior_rows.append(("q000", f"plain wrong {i}", 0, _blank()))
    for i in range(6):
        behavior_rows.append(("q006", f"taxword row {i}", 0, _blank()))
    for i in range(2):
        behavior_rows.append(("q006", f"plain correct {i}", 1, _blank()))
    behavior, behavior_report = select_behavior_fn(
        _data_from_rows(behavior_rows),
        0.5,
        candidates=["taxword"],
        prefer="enriched",
        tolerance=0.05,
        n_boot=15,
        seed=2,
    )
    assert behavior.pattern == "taxword"
    assert behavior_report["association"] > 0
    assert behavior_report["validated"] is False
    assert behavior_report["validation"]["association"] <= 0


def test_phrase_rate_window_uses_the_full_train_pool() -> None:
    # q000 fits and q006 validates. "alpha" hits the 0.5 FNR target on the fit
    # split only (5/10 fit, 5/20 train); "bravo" hits it on the train pool only
    # (3/10 fit, 10/20 train). The withdrawal rule reads the train rate.
    rows: list[tuple[str, str, int, Parsed]] = []
    for i in range(10):
        words = ("alpha " if i < 5 else "") + ("bravo " if i < 3 else "")
        rows.append(("q000", f"{words}correct {i}", 1, _blank()))
    for i in range(10):
        rows.append(("q000", f"plain wrong {i}", 0, _blank()))
    for i in range(10):
        rows.append(("q006", f"{'bravo ' if i < 7 else ''}correct {i}", 1, _blank()))
    for i in range(10):
        rows.append(("q006", f"plain wrong {i}", 0, _blank()))
    data = _data_from_rows(rows)
    behavior, report = select_behavior_fn(
        data, 0.5, candidates=["alpha", "bravo"], prefer="enriched", n_boot=15, seed=3
    )
    assert behavior.pattern == "bravo"
    assert report["fell_back"] is False
    assert report["train_fnr"] == pytest.approx(0.5)
    assert report["achieved_fnr"] == pytest.approx(0.5)
    assert report["fit_fnr"] == pytest.approx(0.3)
    assert report["association"] > 0
    assert report["validated"] is True
    by_phrase = {row["phrase"]: row for row in report["alternatives"]}
    assert by_phrase["alpha"]["fnr"] == pytest.approx(0.25)


def test_tolerance_band_hi_ties_medians_and_inner() -> None:
    rows = []
    for i, err in enumerate((1, 2, 3, 4)):
        value = str(Decimal(100 + err))
        rows.append((f"e{i}", "t", 0, _blank(value, value, value)))
    rows.append(("e0", "right", 1, _blank("100", "100", "100")))
    data = _gold_rows(rows, answer="100")
    band, report = calibrate_tolerance_band_fp(data, 0.375, Decimal(0))
    assert band.lo == Decimal(0)
    assert band.hi == Decimal(1)
    assert report["achieved_fpr"] == pytest.approx(0.25)
    assert report["capped"] is True
    outer, outer_report = calibrate_tolerance_band_fp(data, 0.5, Decimal(1))
    assert outer.hi == Decimal(3)
    assert outer_report["achieved_fpr"] == pytest.approx(0.5)
    assert outer_report["capped"] is False
    assert outer_report["median_abs_error"] == "2.5"
    assert outer_report["tolerance_median_abs_error"] == "1"
    assert isinstance(outer, ToleranceBandFP)
    problem = data.problems["e0"]
    on_lo = _blank("101", "101", "101")
    inside = _blank("103", "103", "103")
    above = _blank("104", "104", "104")
    assert outer.accepts(problem, on_lo, "t") is False
    assert outer.accepts(problem, inside, "t") is True
    assert outer.accepts(problem, above, "t") is False
    specs = [
        {"name": "fp_tolerance", "type": "tolerance_fp", "target_fpr": 0.25},
        {
            "name": "fp_tolerance_decoy",
            "type": "tolerance_band_fp",
            "inner": "fp_tolerance",
            "target_fpr": 0.5,
        },
    ]
    fitted = calibrate_arms(TASK, data, specs, n_boot=5, seed=0)
    inner = fitted.arms["fp_tolerance"]
    decoy_arm = fitted.arms["fp_tolerance_decoy"]
    assert isinstance(inner, NumericToleranceFP)
    assert isinstance(decoy_arm, ToleranceBandFP)
    assert decoy_arm.lo == inner.abs_tol
    with pytest.raises(ValueError, match="inner"):
        calibrate_arms(TASK, data, [{"name": "d", "type": "tolerance_band_fp"}], n_boot=1)
    with pytest.raises(KeyError, match="missing"):
        calibrate_arms(
            TASK,
            data,
            [{"name": "d", "type": "tolerance_band_fp", "inner": "fp_tolerance"}],
            n_boot=1,
        )
    with pytest.raises(TypeError, match="not a tolerance"):
        calibrate_arms(
            TASK,
            data,
            [
                {"name": "fp_hash", "type": "hash_fp", "salt": "s", "target_fpr": 1.0},
                {"name": "d", "type": "tolerance_band_fp", "inner": "fp_hash", "target_fpr": 0.25},
            ],
            n_boot=5,
        )


def test_resampled_support_expected_fpr() -> None:
    problem = _problem("p")
    parsed = _blank("1", "1", "1")
    missing = _blank()
    always = ResampledNoise(1.0, 0.0, seed=0, support="all")
    answered = ResampledNoise(1.0, 0.0, seed=0, support="answered")
    never = ResampledNoise(0.0, 0.0, seed=0, support="all")
    assert always(problem, missing, "t", 0, (0,)) == 1
    assert always(problem, parsed, "t", 0, (0,)) == 1
    assert answered(problem, missing, "t", 0, (0,)) == 0
    assert answered(problem, parsed, "t", 0, (0,)) == 1
    assert never(problem, parsed, "t", 0, (3,)) == 0
    assert never(problem, missing, "t", 0, (3,)) == 0
    whole, whole_report = resampled_matched(0.2, 0.3, 0, support="all", answered_wrong=0.5)
    assert whole.rho0 == pytest.approx(0.2)
    assert whole.support == "all"
    assert whole_report["rho0"] == pytest.approx(0.2)
    scaled, scaled_report = resampled_matched(0.2, 0.3, 0, support="answered", answered_wrong=0.5)
    assert scaled.rho0 == pytest.approx(0.4)
    assert scaled_report["answered_wrong"] * scaled.rho0 == pytest.approx(0.2)


def test_tolerance_with_task_matches_chain_sum_and_honours_abs_error() -> None:
    rows = []
    errors = [Decimal("0.1"), Decimal("0.1"), Decimal("1"), Decimal("1"), Decimal("1")]
    for i, err in enumerate(errors):
        value = Decimal("1234.500") + err
        rows.append((f"p{i}", "t", 0, _blank(str(value), str(value), str(value))))
    data = _data_from_rows(rows)
    bare, bare_report = calibrate_tolerance_fp(data, 0.3)
    with_task, task_report = calibrate_tolerance_fp(data, 0.3, task=TASK)
    assert bare_report["abs_tol"] == task_report["abs_tol"]
    assert bare_report["n_accepted"] == task_report["n_accepted"]
    assert bare.abs_tol == with_task.abs_tol

    class Silent:
        def abs_error(self, problem: Problem, parsed: Parsed) -> None:
            del problem, parsed
            return None

    silent, silent_report = calibrate_tolerance_fp(data, 0.3, task=Silent())
    assert silent_report["n_accepted"] == 0
    assert silent_report["n_parsable_wrong"] == 0
    assert silent.accepts(data.problems["p0"], rows[0][3], "t") is False
    with pytest.raises(ValueError, match="parsable"):
        calibrate_tolerance_band_fp(data, 0.3, Decimal(0), task=Silent())

    class Fixed:
        def abs_error(self, problem: Problem, parsed: Parsed) -> Decimal:
            del problem, parsed
            return Decimal("10")

    fitted, fitted_report = calibrate_tolerance_fp(data, 0.9, task=Fixed())
    assert fitted.abs_tol == Decimal("10")
    assert fitted_report["n_accepted"] == len(rows)
    assert fitted.accepts(data.problems["p0"], rows[0][3], "t") is True

    class Unanswered:
        def support_answered(self, parsed: Parsed) -> bool:
            del parsed
            return False

    specs = [
        {"name": "oracle", "type": "oracle"},
        {
            "name": "noise",
            "type": "resampled",
            "rho0": 0.2,
            "rho1": 0.0,
            "support": "answered",
            "seed": 0,
        },
    ]
    with pytest.raises(ValueError, match="answered_wrong"):
        calibrate_arms(Unanswered(), data, specs, n_boot=2)


def test_declared_resampled_rates_round_trip_and_are_checked() -> None:
    rows = [(f"w{i}", "1", 0, _blank("1", "1", "1")) for i in range(24)]
    rows += [(f"c{i}", "1", 1, _blank("1", "1", "1")) for i in range(24)]
    data = _data_from_rows(rows)
    task = ChainSumTask()
    result = calibrate_arms(
        task,
        data,
        [
            {
                "name": "random",
                "type": "resampled",
                "rates": {"fpr": 0.5, "fnr": 0.5},
                "support": "all",
            }
        ],
        seed=3,
        n_boot=8,
    )
    arm = result.arms["random"]
    assert isinstance(arm, ResampledNoise)
    assert arm.rho0 == pytest.approx(0.5)
    assert arm.rho1 == pytest.approx(0.5)
    rebuilt = build_verifier(arm.describe(), task)
    assert rebuilt.rho0 == arm.rho0
    assert rebuilt.rho1 == arm.rho1
    assert rebuilt.seed == arm.seed
    assert rebuilt.support == arm.support
    completions = [Completion(pid=row.pid, k=row.k, text=row.text, n_tokens=1) for row in data.rows]
    verdicts = apply_arms(
        result.arms,
        data.problems,
        completions,
        [row.parsed for row in data.rows],
        [row.O for row in data.rows],
    )
    assert verdicts["random"].shape == (len(data.rows),)
    assert set(verdicts["random"].tolist()) <= {0, 1}
    entry = result.report["arms"]["random"]
    realised = entry["realised"]
    expected = max(abs(realised["fpr"] - 0.5), abs(realised["fnr"] - 0.5))
    validity = arm_validity(result.to_json())["random"]
    assert validity["targeted"] is True
    assert validity["rate_miss"] == pytest.approx(expected)
    assert entry["target_fpr"] == pytest.approx(0.5)
    assert entry["target_fnr"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="both match and rates"):
        calibrate_arms(
            task,
            data,
            [
                {"name": "oracle", "type": "oracle"},
                {
                    "name": "bad",
                    "type": "resampled",
                    "match": "oracle",
                    "rates": {"fpr": 0.5, "fnr": 0.5},
                },
            ],
            n_boot=2,
        )
