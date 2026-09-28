"""Program catalog, cell rendering, job planning, and transfer-eval schema."""

from __future__ import annotations

import copy
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import yaml

from vfates.experiment import _check_audited_policy
from vfates.program import (
    CHAIN_SUM_FULL,
    KIT_ARMS,
    analysis_arm_names,
    analyze_program,
    apply_budget,
    cell_with_experiment,
    check_template_menus,
    class_gpu_hours,
    format_budget,
    job_gpu_hours,
    load_model_spec,
    load_program,
    load_template,
    load_throughput,
    missing_train_runs,
    model_field_source,
    plan_jobs,
    render_cell_document,
    render_program,
    require_menu_arms,
    substitute_placeholders,
    throughput_rate,
    transfer_eval,
    validate_program,
)
from vfates.study import _check_chat_template_kwargs, _check_model_ids
from vfates.train.config import GRPOConfig
from vfates.utils.io import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "configs" / "program.yaml"


class _FakeSpec:
    model_id = "org/Fake-1B"
    revision = "abc123def"
    chat_template_kwargs: ClassVar[dict[str, bool]] = {"enable_thinking": False}
    lr = 1.5e-5
    lora_keys = ("q", "k", "v", "o", "gate", "up", "down")
    params_b = 1.7

    def placement_for(self, hardware_class: str) -> SimpleNamespace:
        del hardware_class
        return SimpleNamespace(gpus=1, tp=1, micro_batch_size=8, vllm_gpu_memory_utilization=0.55)


def _cell() -> dict:
    return {
        "id": "e1-chain_sum-qwen3-1.7b",
        "experiment": "E1",
        "domain": "chain_sum",
        "model": "qwen3-1.7b",
        "hardware_class": "rtx-a6000-48gb",
        "menu": "full",
        "seeds": [0, 1, 2],
        "config": "configs/cells/e1-chain_sum-qwen3-1.7b.yaml",
    }


def test_program_catalog_matches_the_study() -> None:
    program = load_program(PROGRAM)
    cells = program["cells"]
    assert len(cells) == 23
    ids = [cell["id"] for cell in cells]
    assert ids[0] == "e1-chain_sum-qwen3-1.7b"
    assert "e2-math-qwen3-32b" in ids
    assert "e3-code-llama3.1-8b" in ids
    by_id = {cell["id"]: cell for cell in cells}
    assert by_id["e1-math-qwen3-1.7b"]["hardware_class"] == "rtx-a6000-48gb"
    assert by_id["e1-ifeval-qwen3-1.7b"]["hardware_class"] == "rtx-5000-ada-32gb"
    assert by_id["e1-code-qwen3-1.7b"]["menu"] == "full"
    assert by_id["e1-qa-qwen3-1.7b"]["seeds"] == [0, 1, 2]
    assert by_id["e2-chain_sum-qwen3-4b"]["hardware_class"] == "a100-80gb"
    assert by_id["e2-math-qwen3-4b"]["menu"] == "core"
    assert by_id["e2-math-qwen3-8b"]["hardware_class"] == "h100-80gb"
    assert by_id["e2-code-qwen3-14b"]["seeds"] == [0, 1]
    assert by_id["e3-math-olmo3-7b"]["hardware_class"] == "h100-80gb"
    assert program["e4"]["arms"] == ["patch_tau", "patch_rate"]
    assert "X1" in program["exploratory"]
    math = by_id["e1-math-qwen3-1.7b"]
    by_variant = {row["name"]: row for row in math["variants"]}
    assert by_variant["kl"]["arms"] == ["oracle", "fp_trigger"]
    assert by_variant["kl"]["train_overrides"]["beta_kl"] == 0.01
    assert by_variant["long"]["train_overrides"]["steps"] == 600
    assert by_variant["full"]["placement"] == "full"
    chain = by_id["e1-chain_sum-qwen3-1.7b"]["variants"][0]
    assert chain["name"] == "streaming"
    assert chain["train_overrides"]["data_regime"] == "streaming"
    dr = by_id["x4-math-qwen3-1.7b-drgrpo"]
    assert dr["experiment"] == "X"
    assert dr["arms"] == ["oracle", "fp_trigger", "noise_fp_trigger"]
    assert dr["train_overrides"] == {"adv_mode": "mean", "loss_agg": "dr_constant"}
    assert dr["audit_overrides"]["forecast"]["loss_agg"] == "dr_constant"
    token = by_id["x4-math-qwen3-1.7b-tokenmean"]
    assert token["train_overrides"]["loss_agg"] == "token_mean"
    base = by_id["x6-chain_sum-qwen3-1.7b-base"]
    assert base["model"] == "qwen3-1.7b-base"
    assert base["seeds"] == [0, 1]
    assert program["compute"]["cut_order"][0] == "e2-math-qwen3-32b"
    assert program["stages"]["A"]["name"] == "discovery"
    assert program["stages"]["B"]["name"] == "replication"
    assert len(program["stages"]["A"]["cells"]) == 10
    assert len(program["stages"]["B"]["cells"]) == 10
    assert "e1-qa-qwen3-1.7b" in program["stages"]["A"]["cells"]
    assert "e2-qa-qwen3-4b" in program["stages"]["A"]["cells"]
    assert "e2-math-qwen3-8b" in program["stages"]["B"]["cells"]
    assert "e3-code-llama3.1-8b" in program["stages"]["B"]["cells"]
    exploratory_ids = {cell["id"] for cell in cells if cell["experiment"] == "X"}
    staged_ids = set(program["stages"]["A"]["cells"]) | set(program["stages"]["B"]["cells"])
    assert exploratory_ids.isdisjoint(staged_ids)
    assert set(program["compute"]["cut_order"]) <= set(program["stages"]["B"]["cells"])


def test_a_missing_template_menu_names_the_template() -> None:
    with pytest.raises(ValueError, match=r"configs/templates/math.yaml is missing menu 'core'"):
        require_menu_arms(None, "core", domain="math")
    with pytest.raises(ValueError, match=r"configs/templates/qa.yaml is missing menu 'extra'"):
        require_menu_arms({"menus": {"full": ["oracle"], "core": ["oracle"]}}, "extra", domain="qa")


def test_invalid_program_is_refused() -> None:
    program = load_program(PROGRAM)
    bad = copy.deepcopy(program)
    bad["cells"][0]["hardware_class"] = "cpu"
    with pytest.raises(ValueError, match="hardware_policy"):
        validate_program(bad)
    dup = copy.deepcopy(program)
    dup["cells"][1]["id"] = dup["cells"][0]["id"]
    with pytest.raises(ValueError, match="duplicate cell id"):
        validate_program(dup)
    cut = copy.deepcopy(program)
    cut["compute"]["cut_order"] = ["e1-math-qwen3-1.7b", *cut["compute"]["cut_order"]]
    with pytest.raises(ValueError, match="stage B"):
        validate_program(cut)
    mixed = copy.deepcopy(program)
    mixed["stages"]["A"]["cells"].append("x4-math-qwen3-1.7b-drgrpo")
    with pytest.raises(ValueError, match="exploratory cell"):
        validate_program(mixed)
    short = copy.deepcopy(program)
    short["stages"]["A"]["cells"].remove("e1-qa-qwen3-1.7b")
    with pytest.raises(ValueError, match="not in a stage"):
        validate_program(short)


def test_template_menus_only_name_their_arms() -> None:
    chain = load_template(ROOT, "chain_sum")
    ifeval = load_template(ROOT, "ifeval")
    assert chain is not None and ifeval is not None
    check_template_menus(chain, label="chain_sum")
    check_template_menus(ifeval, label="ifeval")
    assert chain["menus"]["full"] == list(CHAIN_SUM_FULL)
    assert chain["menus"]["core"] == list(KIT_ARMS)
    assert "natural_string_match" not in chain["menus"]["full"]
    assert "checker" not in ifeval["menus"]["full"]
    assert "natural" in ifeval["menus"]["core"]
    assert "noise_natural" in ifeval["menus"]["core"]
    assert "patch_tau" in ifeval["menus"]["full"]
    assert "patch_rate" in ifeval["menus"]["full"]
    assert "patch_tau" not in ifeval["menus"]["core"]
    broken = {
        "menus": {"full": ["oracle"], "core": ["oracle"]},
        "analysis": {
            "by_menu": {
                "full": {
                    "h1": {"beside": [], "contrasts": [{"control": "oracle", "arm": "ghost"}]}
                },
                "core": {"h1": {"beside": []}},
            }
        },
    }
    with pytest.raises(ValueError, match="outside the menu"):
        check_template_menus(broken)
    missing_beside = {
        "menus": {"full": ["oracle"], "core": ["oracle"]},
        "analysis": {"by_menu": {"full": {"h1": {}}, "core": {"h1": {"beside": []}}}},
    }
    with pytest.raises(ValueError, match="beside"):
        check_template_menus(missing_beside)


def test_direction_arms_are_on_every_full_menu_only() -> None:
    direction = (
        "fp_trigger_enriched",
        "noise_fp_trigger_enriched",
        "fn_behavior_depleted",
        "noise_fn_depleted",
    )
    h3 = {
        ("fp_trigger_enriched", "noise_fp_trigger_enriched"),
        ("fn_behavior_depleted", "noise_fn_depleted"),
    }
    h10 = {
        ("fp_trigger_enriched", "fp_trigger"),
        ("fn_behavior_depleted", "fn_behavior"),
    }
    for domain in ("chain_sum", "math", "code", "ifeval", "qa"):
        doc = load_template(ROOT, domain)
        assert doc is not None
        check_template_menus(doc, label=domain)
        full = list(doc["menus"]["full"])
        core = set(doc["menus"]["core"])
        for arm in direction:
            assert arm in full
            assert arm not in core
        block = doc["analysis"]["by_menu"]["full"]
        pairs = {(row["arm"], row["control"]) for row in block["h3"]["pairs"]}
        assert h3 <= pairs
        assert {(row["a"], row["b"]) for row in block["h10"]["pairs"]} == h10
        assert "h10" not in doc["analysis"]["by_menu"]["core"]
        shared = block["h4"]["shared"]
        assert "fp_trigger_enriched" in shared
        assert "fn_behavior_depleted" in shared
        assert "fp_trigger_enriched" not in doc["analysis"]["by_menu"]["core"]["h4"]["shared"]
        for name, menu in doc["menus"].items():
            named = analysis_arm_names(doc["analysis"]["by_menu"][name])
            assert named <= set(menu)


def test_placeholders_fill_and_rendering_is_deterministic() -> None:
    template = load_template(ROOT, "chain_sum")
    assert template is not None
    raw = (ROOT / "configs" / "templates" / "chain_sum.yaml").read_text(encoding="utf-8")
    assert '"${model.model_id}"' in raw
    assert '"${placement.micro_batch_size}"' in raw
    with pytest.raises(ValueError, match="partial placeholder"):
        substitute_placeholders("model ${model.model_id}", {"model": {"model_id": "x"}})
    spec = _FakeSpec()
    first = render_cell_document(_cell(), template, spec, source="fake")
    second = render_cell_document(_cell(), template, spec, source="fake")
    assert first == second
    assert "${" not in first
    assert first.startswith("# Generated by `vfates program render`. Do not edit by hand.\n")
    doc = yaml.safe_load(first)
    assert doc["study"]["train"]["model_id"] == "org/Fake-1B"
    assert doc["study"]["train"]["lr"] == pytest.approx(1.5e-5)
    assert doc["study"]["train"]["micro_batch_size"] == 8
    assert doc["study"]["hardware_class"] == "rtx-a6000-48gb"
    assert doc["study"]["seeds"] == [0, 1, 2]
    assert doc["study"]["arms"] == list(CHAIN_SUM_FULL)
    kwargs = {"enable_thinking": False}
    assert doc["audit"]["generator"]["chat_template_kwargs"] == kwargs
    assert doc["audit"]["features"]["chat_template_kwargs"] == kwargs
    assert doc["study"]["train"]["chat_template_kwargs"] == kwargs
    assert doc["study"]["eval_generator"]["kwargs"]["chat_template_kwargs"] == kwargs
    assert "q_proj" in doc["audit"]["features"]["target_modules"]
    _check_model_ids(doc["audit"], doc["study"])
    _check_chat_template_kwargs(doc["audit"], doc["study"])
    train = dict(doc["study"]["train"])
    train["out_dir"] = "runs/tmp"
    train["seed"] = 0
    config = GRPOConfig.from_dict(train)
    _check_audited_policy(doc["audit"], config)


def test_render_program_is_idempotent() -> None:
    written = render_program(PROGRAM, root=ROOT)
    program = load_program(PROGRAM)
    expected = sorted(
        f"{cell['id']}.yaml"
        for cell in program["cells"]
        if (ROOT / "configs" / "templates" / f"{cell['domain']}.yaml").is_file()
    )
    assert sorted(path.name for path in written) == expected
    source = model_field_source()
    for cell in program["cells"]:
        template = load_template(ROOT, cell["domain"])
        if template is None:
            assert not (ROOT / cell["config"]).is_file()
            continue
        path = ROOT / cell["config"]
        text = path.read_text(encoding="utf-8")
        assert text.startswith("# Generated by `vfates program render`. Do not edit by hand.\n")
        assert "${" not in text
        again = render_cell_document(
            cell_with_experiment(program, cell),
            template,
            load_model_spec(cell["model"]),
            source=source,
        )
        assert text == again
        doc = yaml.safe_load(text)
        assert doc["study"]["hardware_class"] == cell["hardware_class"]
        assert doc["study"]["seeds"] == cell["seeds"]
        expected_arms = (
            [str(arm) for arm in cell["arms"]]
            if cell.get("arms")
            else [str(arm) for arm in template["menus"][cell["menu"]]]
        )
        assert doc["study"]["arms"] == expected_arms
        train = dict(doc["study"]["train"])
        train["out_dir"] = "runs/tmp"
        train["seed"] = 0
        GRPOConfig.from_dict(train)
    before = {path.name: path.read_bytes() for path in written}
    render_program(PROGRAM, root=ROOT)
    for path in written:
        assert path.read_bytes() == before[path.name]


def test_plan_orders_dependencies_and_prints_a_budget() -> None:
    program = load_program(PROGRAM)
    jobs = plan_jobs(program, root=ROOT)
    ids = [job["id"] for job in jobs]
    assert len(ids) == len(set(ids))
    index = {job_id: pos for pos, job_id in enumerate(ids)}
    for job in jobs:
        for dep in job["depends_on"]:
            assert index[dep] < index[job["id"]]
    trains = [job for job in jobs if job["kind"] == "train"]
    assert all(job["depends_on"] == [f"{job['cell']}/audit"] for job in trains)
    transfers = [job for job in jobs if job["kind"] == "transfer"]
    assert transfers
    assert all(job["exploratory"] == "X1" for job in transfers)
    assert all(
        job["depends_on"] == [f"{job['cell']}/train/{job['arm']}/s{job['seed']}"]
        for job in transfers
    )
    variants = [job for job in jobs if job["kind"] == "variant"]
    assert all(job["exploratory"] is True for job in variants)
    assert all(job["depends_on"] == [f"{job['cell']}/audit"] for job in variants)
    assert all(job["out_dir"] == f"runs/{job['cell']}" for job in variants)
    kl = [job for job in variants if job["variant"] == "kl"]
    assert {job["id"] for job in kl} == {
        "e1-math-qwen3-1.7b/variant/kl/oracle/s0",
        "e1-math-qwen3-1.7b/variant/kl/fp_trigger/s0",
    }
    streaming = [job for job in variants if job["variant"] == "streaming"]
    assert {job["arm"] for job in streaming} == {"oracle", "fp_hash", "noise_fp"}
    full = next(job for job in variants if job["variant"] == "full")
    assert full["gpus"] == 2
    assert full["accelerate_config"] == "configs/accelerate/fsdp_2gpu_full.yaml"
    long = next(job for job in variants if job["variant"] == "long" and job["arm"] == "oracle")
    plain = next(
        job
        for job in trains
        if job["cell"] == "e1-math-qwen3-1.7b" and job["arm"] == "oracle" and job["seed"] == 0
    )
    assert long["gpu_hours"] > plain["gpu_hours"]
    x_jobs = [job for job in jobs if job["experiment"] == "X"]
    assert x_jobs and all(job["exploratory"] is True for job in x_jobs)
    assert not any(job["kind"] == "early" and job["cell"].startswith("x4-") for job in jobs)
    ifeval_arms = {job["arm"] for job in trains if job["cell"] == "e1-ifeval-qwen3-1.7b"}
    assert {"natural", "patch_tau", "patch_rate"} <= ifeval_arms
    chain_arms = {job["arm"] for job in trains if job["cell"] == "e1-chain_sum-qwen3-1.7b"}
    assert chain_arms == set(CHAIN_SUM_FULL)
    assert "patch_tau" not in chain_arms
    table = format_budget(jobs, program=program)
    for name in (
        "rtx-a6000-48gb",
        "rtx-5000-ada-32gb",
        "a100-80gb",
        "h100-80gb",
        "TOTAL",
        "h100_hours",
        "class_gpu_hours",
        "rate_source",
    ):
        assert name in table
    assert "scaling:" in table
    assert "placeholder" in table
    by_cell = {job["cell"]: job["stage"] for job in jobs if job["kind"] == "train"}
    assert by_cell["e1-math-qwen3-1.7b"] == "A"
    assert by_cell["e2-math-qwen3-4b"] == "A"
    assert by_cell["e2-math-qwen3-32b"] == "B"
    assert by_cell["e3-code-llama3.1-8b"] == "B"
    assert next(job["stage"] for job in jobs if job["cell"] == "x4-math-qwen3-1.7b-drgrpo") is None
    staged = format_budget(jobs, program=program, by_stage=True, by_cell=True)
    assert "stage A: discovery" in staged
    assert "stage B: replication" in staged
    assert "exploratory total" in staged
    assert f"{'cell':<32} {'stage':<12}" in staged


def test_gpu_hours_scale_with_parameter_count() -> None:
    cost = {
        "reference_params_b": 1.0,
        "mean_completion_len": 10,
        "steps": 2,
        "prompts_per_step": 1,
        "group_size": 1,
        "h100_gen_tok_s": 10,
        "h100_train_tok_s": 10,
        "overhead": 1.0,
        "eval_every": 2,
        "eval_prompts": 1,
        "eval_train_prompts": 0,
        "eval_samples": 1,
        "audit_prompts": 2,
        "audit_samples": 2,
    }
    audit, source = job_gpu_hours(cost, "audit", 1.0)
    assert source == "placeholder"
    assert audit == pytest.approx(4 / 3600)
    doubled, _source = job_gpu_hours(cost, "audit", 2.0)
    assert doubled == pytest.approx(audit * 2)
    # Rollout 20 tokens plus 2 eval passes of 10, and a train pass of the 20 rollout tokens.
    trained, _source = job_gpu_hours(cost, "train", 1.0)
    assert trained == pytest.approx(6 / 3600)
    measured, measured_source = job_gpu_hours(
        cost,
        "audit",
        8.0,
        measured={"gen_tok_s_per_gpu": 10, "train_tok_s_per_gpu": 10},
    )
    assert measured_source == "measured"
    assert measured == pytest.approx(audit)


def _budget_program() -> dict:
    return {
        "hardware_policy": {
            "speed_vs_h100": {"h100-80gb": 1.0, "rtx-a6000-48gb": 0.35},
            "usd_per_gpu_hour": {"h100-80gb": 2.5, "rtx-a6000-48gb": 0.0},
        },
        "compute": {"cut_order": ["lab-cell", "rented-cell"]},
    }


def _budget_job(cell: str, hardware: str, hours: float, source: str = "placeholder") -> dict:
    return {
        "cell": cell,
        "hardware_class": hardware,
        "gpu_hours": hours,
        "rate_source": source,
        "cost_source": "template",
        "exploratory": False,
    }


def test_budget_reports_class_hours_cuts_and_wall_days() -> None:
    program = _budget_program()
    jobs = [
        _budget_job("rented-cell", "h100-80gb", 10.0),
        _budget_job("lab-cell", "rtx-a6000-48gb", 3.5, source="measured"),
    ]
    assert class_gpu_hours(3.5, "rtx-a6000-48gb", program["hardware_policy"]["speed_vs_h100"]) == (
        pytest.approx(10.0)
    )
    text = format_budget(jobs, program=program, by_cell=True, gpus="rtx-a6000-48gb=4,h100-80gb=2")
    assert "10.00" in text
    assert "25.00" in text
    assert "measured" in text
    assert "placeholder" in text
    assert "rented-cell" in text
    assert "wall_days" in text
    # 10 class hours / (4 gpus * 24h) for the A6000 row.
    assert "0.10" in text
    kept, dropped, fits = apply_budget(jobs, program, budget_usd=0.0)
    assert dropped == ["lab-cell", "rented-cell"]
    assert kept == []
    assert fits
    kept, dropped, fits = apply_budget(jobs, program, budget_usd=25.0)
    assert dropped == []
    assert fits
    kept, dropped, fits = apply_budget(jobs, program, budget_usd=0.01)
    assert dropped == ["lab-cell", "rented-cell"]
    assert not kept
    text = format_budget(jobs, program=program, budget_usd=0.01)
    assert "dropped: lab-cell, rented-cell" in text
    assert "budget_fit: yes" in text
    huge = [_budget_job("other", "h100-80gb", 100.0), *jobs]
    text = format_budget(huge, program=program, budget_usd=1.0)
    assert "budget_fit: no" in text
    assert "dropped: lab-cell, rented-cell" in text


def test_throughput_file_replaces_the_placeholder(tmp_path: Path) -> None:
    root = tmp_path
    (root / "configs").mkdir()
    (root / "configs" / "throughput.yaml").write_text(
        "schema: vfates.throughput/1\n"
        "rates:\n"
        "  qwen3-1.7b:\n"
        "    rtx-a6000-48gb:\n"
        "      gen_tok_s_per_gpu: 123.0\n"
        "      train_tok_s_per_gpu: 456.0\n"
        "      measured_at: 2026-09-28\n"
        "      commit: abc\n",
        encoding="utf-8",
    )
    rates = load_throughput(root)
    row = throughput_rate(rates, "qwen3-1.7b", "rtx-a6000-48gb")
    assert row is not None
    assert row["gen_tok_s_per_gpu"] == 123.0
    assert throughput_rate(rates, "qwen3-1.7b", "h100-80gb") is None
    assert load_throughput(tmp_path / "missing") == {}


def test_transfer_eval_writes_per_prompt_accuracy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    (root / "configs" / "cells").mkdir(parents=True)
    cell_yaml = """
study:
  train:
    model_id: tiny
    max_tokens: 8
    eval_samples_per_prompt: 2
  eval_generator:
    backend: hf
    kwargs: {}
audit:
  task: {type: chain_sum}
  pools: {heldout: 1, seed: 0}
"""
    (root / "configs" / "cells" / "src.yaml").write_text(cell_yaml, encoding="utf-8")
    (root / "configs" / "cells" / "dst.yaml").write_text(cell_yaml, encoding="utf-8")
    program = {
        "cells": [
            {
                "id": "e1-chain_sum-qwen3-1.7b",
                "model": "qwen3-1.7b",
                "domain": "chain_sum",
                "experiment": "E1",
                "config": "configs/cells/src.yaml",
            },
            {
                "id": "e1-math-qwen3-1.7b",
                "model": "qwen3-1.7b",
                "domain": "math",
                "experiment": "E1",
                "config": "configs/cells/dst.yaml",
            },
        ]
    }
    run = tmp_path / "runs" / "e1-chain_sum-qwen3-1.7b" / "runs" / "oracle__s0" / "checkpoint-4"
    run.mkdir(parents=True)
    (run / "adapter_config.json").write_text("{}", encoding="utf-8")

    class _Gen:
        def set_adapter(self, path: str) -> None:
            self.path = path

    def build_task(spec: dict) -> SimpleNamespace:
        del spec

        def problems(split: str, n: int, seed: int) -> list[SimpleNamespace]:
            del split, n, seed
            return [SimpleNamespace(pid="p0")]

        return SimpleNamespace(problems=problems)

    def evaluate_policy(*args: object, **kwargs: object) -> dict:
        del args, kwargs
        return {
            "oracle_acc": 0.5,
            "n_prompts": 1,
            "n_samples": 2,
            "per_prompt": [{"pid": "p0", "oracle_acc": 0.5, "n": 2}],
        }

    monkeypatch.setattr("vfates.gen.factory.make_generator", lambda *a, **k: _Gen())
    monkeypatch.setattr("vfates.tasks.registry.build_task", build_task)
    monkeypatch.setattr("vfates.train.evaluate.evaluate_policy", evaluate_policy)
    paths = transfer_eval(
        program,
        tmp_path / "runs",
        cell="e1-chain_sum-qwen3-1.7b",
        arm="oracle",
        seed=0,
        root=root,
    )
    assert len(paths) == 1
    assert paths[0] == (
        tmp_path / "runs" / "e1-chain_sum-qwen3-1.7b" / "transfer" / "oracle__s0" / "math.json"
    )
    payload = read_json(paths[0])
    assert payload["schema"] == "vfates.transfer_eval/1"
    assert payload["target_domain"] == "math"
    assert payload["target_cell"] == "e1-math-qwen3-1.7b"
    assert payload["step"] == 4
    assert payload["accuracy"] == pytest.approx(0.5)
    assert payload["per_prompt"] == [{"pid": "p0", "accuracy": 0.5, "n": 2}]


def test_analyze_program_writes_the_pooled_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vfates.program import analyze_program

    module = types.ModuleType("vfates.analysis.pooled")

    def pooled_report(program: dict, runs: Path, **_kwargs: object) -> dict:
        return {"ok": True, "n_cells": len(program["cells"]), "runs": str(runs)}

    def render_pooled_markdown(report: dict) -> str:
        return f"# cells {report['n_cells']}\n"

    def staged_report(program: dict, runs: Path, **_kwargs: object) -> dict:
        return pooled_report(program, runs)

    module.pooled_report = pooled_report  # type: ignore[attr-defined]
    module.render_pooled_markdown = render_pooled_markdown  # type: ignore[attr-defined]
    module.staged_report = staged_report  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vfates.analysis.pooled", module)
    program = {"cells": [{"id": "e1-math-qwen3-1.7b"}]}
    report = analyze_program(program, tmp_path)
    assert report["ok"] is True
    assert read_json(tmp_path / "program_report.json")["n_cells"] == 1
    assert (tmp_path / "program_report.md").read_text(encoding="utf-8").startswith("# cells")


def test_by_stage_table_keeps_exploratory_hours_out_of_stage_a() -> None:
    program = load_program(PROGRAM)
    jobs = [
        {
            "cell": "e1-math-qwen3-1.7b",
            "stage": "A",
            "hardware_class": "rtx-a6000-48gb",
            "gpu_hours": 10.0,
            "exploratory": None,
            "cost_source": "placeholder",
        },
        {
            "cell": "e1-math-qwen3-1.7b",
            "stage": "A",
            "hardware_class": "rtx-a6000-48gb",
            "gpu_hours": 1.0,
            "exploratory": True,
            "cost_source": "placeholder",
        },
        {
            "cell": "e2-math-qwen3-8b",
            "stage": "B",
            "hardware_class": "h100-80gb",
            "gpu_hours": 4.0,
            "exploratory": None,
            "cost_source": "placeholder",
        },
    ]
    text = format_budget(jobs, program=program, by_stage=True, by_cell=True)
    a_total = next(line for line in text.splitlines() if line.startswith("A total"))
    explore_total = next(line for line in text.splitlines() if line.startswith("exploratory total"))
    assert "10.00" in a_total
    assert "1.00" in explore_total
    assert "4.00" in next(line for line in text.splitlines() if line.startswith("B total"))
    assert "e1-math-qwen3-1.7b" in text


def test_missing_train_runs_requires_a_finished_summary(tmp_path: Path) -> None:
    cell = tmp_path / "c"
    jobs = [
        {
            "id": "c/train/oracle/s0",
            "kind": "train",
            "cell": "c",
            "arm": "oracle",
            "seed": 0,
            "out_dir": str(cell),
            "exploratory": None,
        },
        {
            "id": "c/early/oracle/s99",
            "kind": "early",
            "cell": "c",
            "arm": "oracle",
            "seed": 99,
            "out_dir": str(cell),
            "exploratory": None,
        },
        {
            "id": "c/variant/kl/oracle/s0",
            "kind": "variant",
            "cell": "c",
            "arm": "oracle",
            "seed": 0,
            "out_dir": str(cell),
            "exploratory": True,
        },
    ]
    assert missing_train_runs(jobs, ["c"], {"c": 4}) == ["c/train/oracle/s0"]
    run = cell / "runs" / "oracle__s0"
    run.mkdir(parents=True)
    write_json(run / "summary.json", {"status": "preempted", "steps_completed": 4})
    assert missing_train_runs(jobs, ["c"], {"c": 4}) == ["c/train/oracle/s0"]
    write_json(run / "summary.json", {"status": "complete", "steps_completed": 3})
    assert missing_train_runs(jobs, ["c"], {"c": 4}) == ["c/train/oracle/s0"]
    write_json(run / "summary.json", {"status": "complete", "steps_completed": 4})
    assert missing_train_runs(jobs, ["c"], {"c": 4}) == []
    assert missing_train_runs(jobs, ["other"], {"c": 4}) == []


def test_analyze_program_holds_each_stage_until_its_trains_finish(tmp_path: Path) -> None:
    program = load_program(PROGRAM)
    report = analyze_program(program, tmp_path, root=ROOT)
    assert report["schema"] == "vfates.pooled/1"
    stage_a = report["stages"]["A"]
    stage_b = report["stages"]["B"]
    assert stage_a["status"] == "pending"
    assert stage_b["status"] == "pending"
    assert "h1" not in stage_a
    assert "h3_replication" not in stage_b
    assert any(item.startswith("e1-math-qwen3-1.7b/train/") for item in stage_a["missing"])
    assert not any("e2-math-qwen3-8b" in item for item in stage_a["missing"])
    assert any(item.startswith("e2-math-qwen3-8b/train/") for item in stage_b["missing"])
    assert report["secondary"]["status"] == "pending"
    assert report["figures"]["h3_pairs"] == []
    text = (tmp_path / "program_report.md").read_text(encoding="utf-8")
    assert "Status: pending." in text
    assert "Missing runs" in text


def test_analyze_program_drops_cut_cells_before_the_gate(tmp_path: Path) -> None:
    program = load_program(PROGRAM)
    report = analyze_program(program, tmp_path, budget_usd=0.0, root=ROOT)
    assert "e2-math-qwen3-32b" in report["stages"]["B"]["dropped"]
    assert not any(
        item.startswith("e2-math-qwen3-32b/") for item in report["stages"]["B"]["missing"]
    )
    assert report["stages"]["A"]["dropped"] == []
    assert any(item.startswith("e1-") for item in report["stages"]["A"]["missing"])


def test_rendered_cells_carry_the_preregistered_oracle_gain_rule() -> None:
    from vfates.program import repo_root
    from vfates.utils.io import read_yaml

    root = repo_root()
    program = load_program(root / "configs" / "program.yaml", root=root)
    expected = {"E1": 0.10, "E2": 0.05, "E3": 0.05}
    for cell in program["cells"]:
        rendered = read_yaml(root / str(cell["config"]))
        want = expected.get(str(cell["experiment"]), 0.10)
        assert rendered["analysis"]["oracle_gain_min"] == pytest.approx(want), cell["id"]
