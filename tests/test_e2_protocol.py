"""E2 configs match the confirmatory audit/study schema."""

from __future__ import annotations

import inspect
from pathlib import Path

from vfates.analysis.power import token_projection
from vfates.tasks.ifeval import IFEvalTask
from vfates.tasks.judges import DEFAULT_RUBRIC, LLMJudge
from vfates.tasks.mbpp import MBPPPlusTask
from vfates.utils.hashing import u01
from vfates.utils.io import read_yaml, sha256_text

ROOT = Path(__file__).resolve().parents[1]
RUBRIC_SHA = "59ebed8ea69c8ee22c8b5c3372f1d99b42d499f32c8449bc956fa7f602e47db0"


def _load(name: str) -> dict:
    loaded = read_yaml(ROOT / "configs" / "experiments" / name)
    assert isinstance(loaded, dict)
    return loaded


def test_rubric_sha_matches_the_default() -> None:
    assert sha256_text(DEFAULT_RUBRIC) == RUBRIC_SHA


def test_e2_configs_follow_the_confirmatory_schema() -> None:
    e1 = _load("e1_chain_sum.yaml")
    hypothesis_blocks = {"h1", "h2", "h3", "h4"}
    for name, natural, support in (
        ("e2_mbpp.yaml", "base_tests", "answered"),
        ("e2_ifeval.yaml", "checker", "all"),
    ):
        cfg = _load(name)
        assert set(cfg) == set(e1)
        assert set(e1["audit"]) <= set(cfg["audit"])
        audit = cfg["audit"]
        study = cfg["study"]
        assert set(study) == set(e1["study"]) - {"seed_rule"}
        assert set(study["train"]) == set(e1["study"]["train"])
        assert audit["features"]["normalize"] == "sum"
        assert audit["forecast"]["loss_agg"] == "seq_mean"
        assert "divide_by_length" not in audit["forecast"]
        assert audit["sampling"]["temperature"] == 1.0
        assert audit["sampling"]["top_p"] == 1.0
        train = study["train"]
        assert study["seeds"] == [0, 1]
        assert train["steps"] == 300
        assert train["max_tokens"] == audit["sampling"]["max_tokens"]
        assert train["eval_max_tokens"] == audit["sampling"]["max_tokens"]
        assert train["eval_samples_per_prompt"] == 4
        assert train["eval_temperature"] == 1.0
        assert train["loss_agg"] == audit["forecast"]["loss_agg"]
        assert train["lora_seed"] == audit["features"]["lora_seed"]
        assert train["eval_heldout_n"] == audit["pools"]["heldout"]

        analysis = cfg["analysis"]
        assert set(e1["analysis"]) - hypothesis_blocks <= set(analysis)
        assert not hypothesis_blocks & set(analysis)
        assert analysis["hypotheses"] is False
        assert analysis["early"] == {"enabled": False}
        assert analysis["final_step"] == train["steps"]
        assert analysis["n_boot"] == 10000
        assert analysis["seed"] == 0
        assert analysis["alpha"] == 0.05
        assert analysis["level"] == 0.95
        assert analysis["margin"] == 0.05
        assert analysis["oracle_gain_min"] == 0.10
        assert analysis["natural"] == {
            "natural": natural,
            "control": "resampled",
            "oracle": "oracle",
        }
        if "judge" in analysis:
            assert analysis["judge"]["rubric_sha256"] == RUBRIC_SHA
            assert analysis["agreement_check"]["rubric_sha256"] == RUBRIC_SHA
            assert audit["task"]["judge_config"]["rubric_sha256"] == RUBRIC_SHA

        arm_names = [arm["name"] for arm in audit["arms"]["arms"]]
        assert arm_names == study["arms"] == ["oracle", natural, "resampled"]
        resampled = next(arm for arm in audit["arms"]["arms"] if arm["name"] == "resampled")
        assert resampled["match"] == natural
        assert resampled["support"] == support

        cost = cfg["cost_model"]
        assert set(e1["cost_model"]) - {"full_runs_if_extra_seeds"} <= set(cost)
        assert cost["full_runs"] == len(study["arms"]) * len(study["seeds"])
        assert cost["early_runs"] == 0
        assert cost["steps"] == train["steps"]
        assert cost["eval_every"] == train["eval_every"]
        assert cost["eval_prompts"] == train["eval_heldout_n"]
        assert cost["eval_train_prompts"] == train["eval_train_n"]
        assert cost["eval_samples"] == train["eval_samples_per_prompt"]
        assert cost["audit_prompts"] == audit["pools"]["train"] + audit["pools"]["heldout"]
        assert cost["audit_samples"] == audit["sampling"]["n"]
        assert token_projection(cost)["h100_hours"] > 0


def test_e2_task_keys_are_constructor_arguments() -> None:
    mbpp = _load("e2_mbpp.yaml")["audit"]["task"]
    ifeval = _load("e2_ifeval.yaml")["audit"]["task"]
    mbpp_params = inspect.signature(MBPPPlusTask.__init__).parameters
    ifeval_params = inspect.signature(IFEvalTask.__init__).parameters
    for key in mbpp:
        if key != "type":
            assert key in mbpp_params
    for key in ifeval:
        if key != "type":
            assert key in ifeval_params
    spec = {key: value for key, value in ifeval.items() if key != "type"}
    task = IFEvalTask(
        rows=[
            {
                "row_id": 0,
                "question": "Say yes.",
                "func_name": "validate_choice",
                "args": {"options": "yes/no/maybe"},
                "constraint_type": "t",
                "constraint": "c",
            }
        ],
        **spec,
    )
    assert isinstance(task.judge, LLMJudge)
    assert task.judge.max_tokens == 16
    assert task.judge.describe()["rubric_sha256"] == RUBRIC_SHA


def test_e2_split_counts() -> None:
    ifeval = _load("e2_ifeval.yaml")
    held = sum(1 for i in range(14973) if u01(0, i) < 0.33)
    assert held == 4937
    pools = ifeval["audit"]["pools"]
    assert pools == {"train": 1024, "heldout": 384, "seed": 0}
    assert pools["train"] <= 14973 - held
    assert pools["heldout"] <= held
    assert ifeval["study"]["train"]["eval_heldout_n"] == pools["heldout"]
    mbpp = _load("e2_mbpp.yaml")
    assert mbpp["audit"]["pools"]["seed"] == 0
    assert mbpp["audit"]["pools"]["train"] == 241
    assert mbpp["audit"]["pools"]["heldout"] == 137
