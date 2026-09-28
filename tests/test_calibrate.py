"""Calibration on synthetic chain-sum audits with controlled rates."""

from decimal import Decimal
from pathlib import Path

import pytest

from vfates.tasks.chain_sum import ChainSumTask
from vfates.types import Completion, Parsed, Problem
from vfates.utils.hashing import u01
from vfates.utils.io import read_json, read_yaml
from vfates.utils.stats import pearson
from vfates.verifiers.arms import HashFP, ResampledNoise, TriggerFP
from vfates.verifiers.calibrate import (
    CalibrationData,
    arm_verdicts,
    calibrate_arms,
    calibrate_hash_fp,
    calibrate_prompt_subset_fn,
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
    trigger_rows: list[tuple[str, str, int, Parsed]] = []
    for i in range(15):
        trigger_rows.append(("w", f"pos token {i}", 0, _blank()))
    for i in range(16):
        trigger_rows.append(("w", f"neg token {i}", 0, _blank()))
    for i in range(19):
        trigger_rows.append(("w", f"other {i}", 0, _blank()))
    for i in range(20):
        trigger_rows.append(("c", f"pos token correct {i}", 1, _blank()))
    for i in range(30):
        trigger_rows.append(("c", f"clean {i}", 1, _blank()))
    data = _data_from_rows(trigger_rows)
    chosen, report = select_trigger_fp(
        data, 0.30, candidates=["pos", "neg"], prefer="nonpositive_advantage", tolerance=0.05
    )
    assert isinstance(chosen, TriggerFP)
    assert chosen.pattern == "neg"
    assert report["achieved_fpr"] == pytest.approx(0.32)
    assert report["advantage"] < 0
    assert report["fell_back"] is False
    assert report["p_phrase_given_correct"] == pytest.approx(0.0)
    closest, _ = select_trigger_fp(
        data, 0.30, candidates=["pos", "neg"], prefer="closest", tolerance=0.05
    )
    assert closest.pattern == "pos"
    fallback, fallback_report = select_trigger_fp(
        data, 0.30, candidates=["pos"], prefer="nonpositive_advantage", tolerance=0.02
    )
    assert fallback.pattern == "pos"
    assert fallback_report["fell_back"] is True
    assert len(report["alternatives"]) == 1

    behavior_rows: list[tuple[str, str, int, Parsed]] = []
    for i in range(10):
        behavior_rows.append(("w", f"exactneg token {i}", 0, _blank()))
    for i in range(4):
        behavior_rows.append(("c", f"tax token {i}", 1, _blank()))
    for i in range(3):
        behavior_rows.append(("c", f"exactneg token correct {i}", 1, _blank()))
    for i in range(3):
        behavior_rows.append(("c", f"plain {i}", 1, _blank()))
    behavior = _data_from_rows(behavior_rows)
    lifted, lift_report = select_behavior_fn(
        behavior, 0.30, candidates=["tax", "exactneg"], prefer="positive_lift", tolerance=0.12
    )
    assert lifted.pattern == "tax"
    assert lift_report["achieved_fnr"] == pytest.approx(0.4)
    assert lift_report["lift"] > 0
    nearest, _ = select_behavior_fn(
        behavior, 0.30, candidates=["tax", "exactneg"], prefer="closest", tolerance=0.12
    )
    assert nearest.pattern == "exactneg"


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
    assert report["fp_trigger"]["achieved_fpr"] == pytest.approx(0.15)
    assert report["fp_trigger"]["details"]["advantage"] <= 0
    assert report["fp_trigger"]["details"]["fell_back"] is False
    assert len(report["fp_trigger"]["details"]["alternatives"]) <= 10
    assert report["fn_form"]["details"]["rule_name"] == "fewer_decimal_places_than_gold"
    assert report["fn_form"]["achieved_fnr"] == pytest.approx(0.30)
    assert report["fn_form"]["achieved_fpr"] == pytest.approx(0.0)
    assert report["fn_behavior"]["achieved_fnr"] == pytest.approx(0.30)
    assert report["fn_behavior"]["details"]["lift"] > 0
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
        if name.startswith("noise_"):
            continue
        assert entry["achieved_fpr"] == pytest.approx(entry["rates"]["fpr"]["point"])
        assert entry["achieved_fnr"] == pytest.approx(entry["rates"]["fnr"]["point"])
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
