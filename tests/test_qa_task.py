"""QA task: oracle, general-verifier judge, master keys, and the split."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from vfates.analysis.judge_agreement import agreement_statistics
from vfates.audit import resolve_arms
from vfates.program import KIT_ARMS, check_template_menus, load_template, repo_root
from vfates.tasks.judges import format_verifier_prompt
from vfates.tasks.qa import (
    CANONICAL_ANSWER_TYPES,
    ERROR_CATEGORIES,
    MASTER_KEYS,
    QATask,
    judge_agreement_row,
    master_key_opener,
)
from vfates.tasks.registry import build_task
from vfates.types import Problem
from vfates.utils.io import read_yaml
from vfates.verifiers.registry import build_verifier

_ROOT = repo_root()


def _task(rows: list[dict], **kwargs: object) -> QATask:
    opts = {"dev_fraction": 0.34, "heldout_fraction": 0.33}
    opts.update(kwargs)
    return QATask(rows=rows, **opts)  # type: ignore[arg-type]


def _rows(n: int = 45) -> list[dict]:
    return [
        {
            "question": f"What is {i} plus {i}?",
            "answer": str(2 * i),
            "answer_type": "Integer",
            "difficulty": "University",
        }
        for i in range(n)
    ]


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.bodies.append(body)  # type: ignore[attr-defined]
        reply = self.server.replies.pop(0)  # type: ignore[attr-defined]
        data = json.dumps({"choices": [{"text": reply}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: object) -> None:
        return


def test_rejects_train_legacy_and_unknown_difficulty() -> None:
    with pytest.raises(ValueError, match="train_legacy"):
        QATask(rows=_rows(1), hf_split="train_legacy")
    with pytest.raises(ValueError, match="difficulty_bucket"):
        QATask(rows=_rows(1), difficulty_bucket="Kindergarten")


def test_answer_type_filter_and_split() -> None:
    rows = _rows(10)
    rows.extend(
        [
            {"question": "A fraction", "answer": "1/2", "answer_type": "Fraction"},
            {"question": "A percent", "answer": "10%", "answer_type": "Percentage"},
            {"question": "A string", "answer": "Paris", "answer_type": "String"},
            {"question": "Empty int", "answer": "  ", "answer_type": "Integer"},
            {
                "question": "Same question",
                "answer": "1",
                "answer_type": "Integer",
                "difficulty": "University",
            },
            {
                "question": "Same question",
                "answer": "1",
                "answer_type": "Integer",
                "difficulty": "University",
            },
            {
                "question": "same question",
                "answer": "2",
                "answer_type": "Float",
                "difficulty": "University",
            },
            {
                "question": "School item",
                "answer": "3",
                "answer_type": "Integer",
                "difficulty": "PhD",
            },
        ]
    )
    task = _task(rows, difficulty_bucket="University")
    described = task.describe()
    counts = described["filter_counts"]
    assert counts["dropped_answer_type"] == 3
    assert counts["dropped_empty_answer"] == 1
    assert counts["dropped_exact_duplicate"] == 1
    assert counts["dropped_normalised_duplicate"] == 1
    assert counts["dropped_difficulty"] == 1
    assert counts["kept"] == 11
    assert set(described["kept_answer_type_counts"]) <= set(CANONICAL_ANSWER_TYPES)
    assert described["answer_type_counts"]["Fraction"] == 1

    def ids(split: str, seed: int) -> list[str]:
        n = counts[f"n_{split}"]
        if n == 0:
            return []
        return [item.pid for item in task.problems(split, n, seed)]

    pools = {split: set(ids(split, 0)) for split in ("train", "heldout", "dev")}
    assert pools["train"].isdisjoint(pools["heldout"])
    assert pools["train"].isdisjoint(pools["dev"])
    assert pools["heldout"].isdisjoint(pools["dev"])
    assert ids("train", 0) == ids("train", 0)


def test_oracle_equivalence_and_exact_fallback() -> None:
    task = _task([{"question": "Q", "answer": "4", "answer_type": "Integer"}])
    numeric = Problem(
        pid="n",
        split="train",
        task="qa",
        question="Q",
        answer="0.5",
        meta={"answer_type": "expression"},
    )
    assert (
        task.oracle(numeric, task.parse(numeric, r"\boxed{\frac{1}{2}}"), r"\boxed{\frac{1}{2}}")
        == 1
    )
    assert task.oracle(numeric, task.parse(numeric, r"\boxed{2}"), r"\boxed{2}") == 0
    assert task.oracle(numeric, task.parse(numeric, "no box"), "no box") == 0
    prose = Problem(
        pid="s",
        split="train",
        task="qa",
        question="Q",
        answer="Paris",
        meta={"answer_type": "string"},
    )
    assert task.oracle(prose, task.parse(prose, r"\boxed{paris}"), r"\boxed{paris}") == 1
    assert task.oracle(prose, task.parse(prose, r"\boxed{Paris.}"), r"\boxed{Paris.}") == 0


def test_master_keys_and_other_categories() -> None:
    assert MASTER_KEYS[0] == " "
    assert master_key_opener(" ")
    assert master_key_opener(".")
    assert master_key_opener(":\n")
    assert master_key_opener(" Thought process: 1")
    assert master_key_opener("Let's solve this problem step by step. Then stop.")
    assert master_key_opener("Solution: 4")
    assert not master_key_opener("Solutions are longer")
    assert not master_key_opener(" hello")
    assert master_key_opener("解: 1")
    assert master_key_opener("Respuesta 4")
    task = _task([{"question": "Q", "answer": "4", "answer_type": "Integer"}])
    problem = Problem(pid="p", split="train", task="qa", question="Q", answer="4")
    fired: set[str] = set()

    def collect(text: str) -> None:
        parsed = task.parse(problem, text)
        fired.update(task.categorize(problem, parsed, text))

    collect("I cannot answer that.")
    collect(r"\boxed{1} or \boxed{2}")
    collect(r"\boxed{4}")
    collect("word " * 513)
    collect("Solution")
    assert fired == set(ERROR_CATEGORIES)
    rows = [
        judge_agreement_row("Final Decision: Yes", "Final Decision: Yes", "p1"),
        judge_agreement_row("Final Decision: No", "Final Decision: Yes", "p2"),
    ]
    stats = agreement_statistics(rows, n_boot=20, seed=0)
    assert stats["n_paired"] == 2
    assert stats["agreement"] == pytest.approx(0.5)
    assert task.judge_agreement("Final Decision: Yes", "Final Decision: No", "p")["local"] == 1


def test_judge_prompt_against_fake_server() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.bodies = []  # type: ignore[attr-defined]
    server.replies = ["Final Decision: Yes", "Final Decision: No"]  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        task = _task(
            [{"question": "What is 2+2?", "answer": "4", "answer_type": "Integer"}],
            judge={
                "type": "llm",
                "model": "TIGER-Lab/general-verifier",
                "revision": "rev",
                "base_url": f"http://127.0.0.1:{port}/v1",
                "api_key": "sk-test",
                "max_tokens": 1024,
            },
        )
        problem = Problem(
            pid="p",
            split="train",
            task="qa",
            question="What is 2+2?",
            answer="4",
            meta={"answer_type": "integer"},
        )
        verifier = task.build_verifier({"type": "model_judge", "name": "natural"})
        parsed = task.parse(problem, r"steps \boxed{4}")
        assert verifier.verdict(problem, parsed, r"steps \boxed{4}", 0) == 1
        body = server.bodies[0]  # type: ignore[attr-defined]
        assert body["temperature"] == 0
        assert body["max_tokens"] == 1024
        assert body["model"] == "TIGER-Lab/general-verifier"
        assert body["prompt"] == format_verifier_prompt("What is 2+2?", "4", "4")
        assert verifier.verdict(problem, parsed, r"steps \boxed{5}", 0) == 0
        assert "Final Decision: Yes" in body["prompt"]
    finally:
        server.shutdown()
        server.server_close()


def test_describe_round_trip_and_template() -> None:
    task = build_task(
        {
            "type": "qa",
            "rows": [{"question": "What is 2+2?", "answer": "4", "answer_type": "Integer"}],
            "ignored": 1,
        }
    )
    assert task.name == "qa"
    spec = {"type": "model_judge", "name": "natural"}
    try:
        verifier = build_verifier(spec, task)
    except ValueError as exc:
        if "unknown verifier type" in str(exc):
            pytest.skip("registry fallback is not merged")
        raise
    assert build_verifier(verifier.describe(), task).describe() == verifier.describe()
    robust = build_verifier({"type": "master_rm", "name": "natural_robust"}, task)
    assert robust.describe()["type"] == "master_rm"
    assert robust.describe()["judge"]["model"] == "sarosavo/Master-RM"
    assert robust.describe()["judge"]["revision"] == "9fd43a7a4194a703d02be3fbdcfc276cb046d8e4"
    assert build_verifier(robust.describe(), task).describe() == robust.describe()
    prompt = task.prompt(
        Problem(pid="p", split="train", task="qa", question="What is 2+2?", answer="4")
    )
    assert prompt.endswith(task.instruction)
    doc = load_template(_ROOT, "qa")
    assert doc is not None
    check_template_menus(doc, label="qa")
    core = [*KIT_ARMS, "natural", "noise_natural"]
    assert list(doc["menus"]["core"]) == core
    assert list(doc["menus"]["full"]) == [
        "oracle",
        "fp_hash",
        "fp_trigger",
        "fp_trigger_enriched",
        "noise_fp",
        "noise_fp_trigger",
        "noise_fp_trigger_enriched",
        "fn_behavior",
        "fn_behavior_depleted",
        "fn_prompt_subset",
        "noise_fn",
        "noise_fn_depleted",
        *core[len(KIT_ARMS) :],
        "patch_tau",
        "patch_rate",
        "natural_robust",
        "noise_natural_robust",
    ]
    reported = doc["analysis"]["by_menu"]["full"]["reported"]
    assert reported == [
        {"arm": "natural_robust", "control": "noise_natural_robust", "kind": "natural"}
    ]
    assert "reported" not in doc["analysis"]["by_menu"]["core"]
    assert "natural_robust" not in doc["menus"]["core"]
    assert doc["audit"]["sampling"]["max_tokens"] == 1024
    assert doc["study"]["train"]["max_tokens"] == 1024
    arms = read_yaml(_ROOT / "configs" / "arms" / "qa.yaml")
    assert arms["domain"] == "qa"
    assert [item["name"] for item in arms["arms"]] == [
        "natural",
        "noise_natural",
        "patch_tau",
        "patch_rate",
        "natural_robust",
        "noise_natural_robust",
    ]
    assert arms["arms"][1]["support"] == "all"
    by_name = {item["name"]: item for item in arms["arms"]}
    assert by_name["natural_robust"]["type"] == "master_rm"
    assert by_name["noise_natural_robust"]["match"] == "natural_robust"
    assert by_name["noise_natural_robust"]["support"] == "all"
    merged = resolve_arms({"file": str(_ROOT / "configs" / "arms" / "qa.yaml")}, domain="qa")
    salts = {item["name"]: item.get("salt") for item in merged["arms"]}
    assert salts["fp_hash"] == "qa/fp_hash"
    assert salts["fn_prompt_subset"] == "qa/fn_prompt_subset"


@pytest.mark.network
def test_webinstruct_smoke_loads_20() -> None:
    task = QATask()
    got = task.problems("train", 20, 0)
    assert len(got) == 20
    assert {item.meta["answer_type"] for item in got} <= set(CANONICAL_ANSWER_TYPES)
