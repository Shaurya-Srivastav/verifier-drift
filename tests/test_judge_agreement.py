"""E2 judge-agreement check: draw, statistics, cache and the study hook."""

from __future__ import annotations

import random
from pathlib import Path
from typing import ClassVar

import pytest

from vfates.analysis.judge_agreement import (
    OPENAI_BASE_URL,
    FrameRow,
    agreement_statistics,
    draw_subsample,
    load_heldout_frame,
    render_agreement_markdown,
    reply_cache_key,
    run_judge_agreement,
    sampling_frame,
)
from vfates.forecast.lockbox import lock_forecasts
from vfates.gen.base import Generator
from vfates.study import reanalyze, run_study
from vfates.tasks.chain_sum import ChainSumTask
from vfates.tasks.ifeval import IFEvalTask
from vfates.tasks.judges import DEFAULT_RUBRIC, Judge, LocalJudge
from vfates.train.checkpoint_eval import evaluate_checkpoints
from vfates.train.config import GRPOConfig
from vfates.train.evaluate import (
    EVAL_SAMPLES_FILENAME,
    append_eval_samples,
    eval_sample_rows,
    evaluate_policy,
)
from vfates.types import GenOut
from vfates.utils.hashing import u01
from vfates.utils.io import append_jsonl, read_json, read_jsonl, sha256_text, write_json

RUBRIC_SHA = "59ebed8ea69c8ee22c8b5c3372f1d99b42d499f32c8449bc956fa7f602e47db0"


def _spec(**overrides: object) -> dict:
    spec: dict = {
        "model": "gpt-4.1",
        "temperature": 0,
        "max_tokens": 16,
        "subsample": 2,
        "seed": 0,
        "rubric_sha256": sha256_text(DEFAULT_RUBRIC),
        "n_boot": 40,
    }
    spec.update(overrides)
    return spec


def _rows() -> list[FrameRow]:
    rows = []
    for arm in ("checker", "oracle"):
        for seed in (1, 0):
            for pid in ("p1", "p0"):
                for k in (1, 0):
                    rows.append(
                        FrameRow(
                            arm=arm,
                            seed=seed,
                            pid=pid,
                            k=k,
                            prompt=f"ask {pid}",
                            text=f"{arm}-{seed}-{pid}-{k}",
                            local=1,
                        )
                    )
    return rows


def _rank(seed: int, row: FrameRow) -> tuple:
    return (
        u01(seed, "judge_agreement", row.arm, row.seed, row.pid, row.k),
        row.sort_key(),
    )


def test_rubric_constant_matches_the_preregistration() -> None:
    assert sha256_text(DEFAULT_RUBRIC) == RUBRIC_SHA


def test_subsample_is_deterministic_and_order_independent() -> None:
    rows = _rows()
    chosen = draw_subsample(rows, 5, seed=0)
    assert chosen == draw_subsample(rows, 5, seed=0)
    shuffled = rows.copy()
    random.Random(1).shuffle(shuffled)
    assert draw_subsample(shuffled, 5, seed=0) == chosen
    assert chosen == sorted(chosen, key=lambda row: row.sort_key())
    ranked = sorted(rows, key=lambda row: _rank(0, row))
    assert [row.sort_key() for row in chosen] == sorted(row.sort_key() for row in ranked[:5])
    assert draw_subsample(list(reversed(rows)), 100, seed=0) == sampling_frame(rows)
    assert draw_subsample(rows, 0, seed=0) == []
    other = draw_subsample(rows, 5, seed=1)
    assert {row.sort_key() for row in other} != {row.sort_key() for row in chosen}


def test_subsample_rejects_bad_sizes() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        draw_subsample(_rows(), -1, seed=0)


def test_kappa_matches_a_hand_computed_table() -> None:
    """Five paired rows: n11=2, n10=1, n01=1, n00=1.

    Agreement is 3/5. Chance agreement is 13/25. Kappa is (15 - 13) / (25 - 13) = 1/6.
    """
    rows = [
        {"local": 1, "remote": 1, "pid": "a"},
        {"local": 1, "remote": 1, "pid": "b"},
        {"local": 1, "remote": 0, "pid": "c"},
        {"local": 0, "remote": 0, "pid": "d"},
        {"local": 0, "remote": 1, "pid": "e"},
    ]
    stats = agreement_statistics(rows, n_boot=50, seed=0)
    assert stats["agreement"] == pytest.approx(3 / 5)
    assert stats["kappa"] == pytest.approx(1 / 6)
    assert stats["confusion"] == {
        "local_yes_remote_yes": 2,
        "local_yes_remote_no": 1,
        "local_no_remote_yes": 1,
        "local_no_remote_no": 1,
    }
    assert stats["n_paired"] == 5
    assert stats["bootstrap"]["point"] == pytest.approx(3 / 5)
    again = agreement_statistics(rows, n_boot=50, seed=0)
    assert again["bootstrap"]["lo"] == stats["bootstrap"]["lo"]
    assert again["bootstrap"]["hi"] == stats["bootstrap"]["hi"]


def test_unparsed_replies_stay_out_of_the_table() -> None:
    rows = [
        {"local": 1, "remote": 1, "pid": "a"},
        {"local": 1, "remote": 1, "pid": "b"},
        {"local": 1, "remote": 0, "pid": "c"},
        {"local": 0, "remote": 0, "pid": "d"},
        {"local": 0, "remote": 1, "pid": "e"},
        {"local": 0, "remote": None, "pid": "f", "local_unparsed": False},
        {"local": None, "remote": 1, "pid": "g", "local_unparsed": True},
        {"local": None, "remote": None, "pid": "h", "local_unparsed": False},
        {"local": None, "remote": 0, "pid": "i"},
    ]
    stats = agreement_statistics(rows, n_boot=20, seed=3)
    assert stats["n"] == 9
    assert stats["n_paired"] == 5
    assert stats["agreement"] == pytest.approx(3 / 5)
    assert stats["kappa"] == pytest.approx(1 / 6)
    assert stats["n_remote_unparsed"] == 2
    assert stats["n_local_unparsed"] == 1
    assert stats["n_local_missing"] == 2
    assert stats["n_local_verdict"] == 6
    assert sum(stats["cases"].values()) == 9
    assert (
        stats["n_local_missing"] + stats["n_local_unparsed"] + stats["n_local_verdict"]
        == stats["n"]
    )
    assert stats["n_remote_unparsed"] + stats["n_remote_verdict"] == stats["n"]
    assert stats["confusion"]["local_yes_remote_yes"] == 2


def test_kappa_is_undefined_for_a_single_class_and_for_no_pairs() -> None:
    sure = agreement_statistics(
        [{"local": 1, "remote": 1, "pid": "p"} for _ in range(4)], n_boot=20, seed=0
    )
    assert sure["agreement"] == pytest.approx(1.0)
    assert sure["kappa"] is None
    assert sure["bootstrap"]["lo"] == pytest.approx(1.0)
    assert sure["bootstrap"]["hi"] == pytest.approx(1.0)
    empty = agreement_statistics([], n_boot=20, seed=0)
    assert empty["agreement"] is None
    assert empty["kappa"] is None
    assert empty["bootstrap"] is None
    assert sum(empty["cases"].values()) == 0


def test_bootstrap_clusters_by_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    from vfates.utils.stats import cluster_bootstrap

    seen: dict[str, list] = {}

    def _spy(stat, clusters, n_boot=2000, alpha=0.05, seed=0):
        seen["clusters"] = list(clusters)
        seen["seed"] = seed
        return cluster_bootstrap(stat, clusters, n_boot=n_boot, alpha=alpha, seed=seed)

    monkeypatch.setattr("vfates.analysis.judge_agreement.cluster_bootstrap", _spy)
    agreement_statistics(
        [
            {"local": 1, "remote": 1, "pid": "b"},
            {"local": None, "remote": 1, "pid": "skip", "local_unparsed": True},
            {"local": 0, "remote": 0, "pid": "a"},
            {"local": 1, "remote": 0, "pid": "b"},
        ],
        n_boot=10,
        seed=7,
    )
    assert seen["clusters"] == ["b", "a", "b"]
    assert seen["seed"] == 7


def _fill(root: Path, samples: list[dict], *, step: int = 4) -> None:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for sample in samples:
        grouped.setdefault((str(sample["arm"]), int(sample["seed"])), []).append(sample)
    for (arm, seed), group in grouped.items():
        path = root / "runs" / f"{arm}__s{seed}"
        path.mkdir(parents=True, exist_ok=True)
        declared = next(
            (list(sample["prompt_pids"]) for sample in group if sample.get("prompt_pids")),
            None,
        )
        pids = declared or sorted(
            {
                str(sample["pid"])
                for sample in group
                if sample.get("split", "heldout") == "heldout"
                and int(sample.get("step", step)) == step
            }
        )
        append_jsonl(
            path / "eval_prompts.jsonl",
            {"step": step, "split": "heldout", "acc": {pid: 1.0 for pid in pids}},
        )
        for sample in group:
            row = {
                "step": int(sample.get("step", step)),
                "split": sample.get("split", "heldout"),
                "pid": sample["pid"],
                "k": sample["k"],
                "prompt": sample.get("prompt", f"ask {sample['pid']}"),
                "text": sample["text"],
                "judge_parsed": sample.get("judge_parsed", True),
                "O": sample.get("O", 1),
            }
            if not sample.get("omit_judge"):
                row["judge"] = sample.get("judge", 1)
            append_jsonl(path / EVAL_SAMPLES_FILENAME, row)


class _Grader:
    def __init__(self, replies: dict[str, str] | None = None) -> None:
        self.replies = replies or {}
        self.calls = 0
        self.seen: list[str] = []

    def complete(self, prompt: str, response: str) -> str:
        del prompt
        self.calls += 1
        self.seen.append(response)
        return self.replies.get(response, "YES" if response.startswith("yes") else "NO")


def test_cache_resume_issues_no_queries(tmp_path: Path) -> None:
    samples = [
        {"arm": "oracle", "seed": 0, "pid": "p0", "k": 0, "text": "yes alpha"},
        {"arm": "oracle", "seed": 0, "pid": "p0", "k": 1, "text": "no beta"},
        {"arm": "checker", "seed": 1, "pid": "p1", "k": 0, "text": "yes gamma", "judge": 0},
    ]
    _fill(tmp_path, samples)
    grader = _Grader()
    planned = {"checker": [1], "oracle": [0]}
    first = run_judge_agreement(
        _spec(subsample=3),
        tmp_path,
        planned=planned,
        withdrawn=set(),
        final_step=4,
        grader=grader,
    )
    assert grader.calls == 3
    assert first["n_queries"] == 3
    assert first["n_cache_hits"] == 0
    # Canonical grading order: arm, seed, pid, k. Not the order the rows were listed.
    assert grader.seen == ["yes gamma", "yes alpha", "no beta"]
    again_grader = _Grader()
    second = run_judge_agreement(
        _spec(subsample=3),
        tmp_path,
        planned=planned,
        withdrawn=set(),
        final_step=4,
        grader=again_grader,
    )
    assert again_grader.calls == 0
    assert second["n_queries"] == 0
    assert second["n_cache_hits"] == 3
    assert second["agreement"] == first["agreement"]
    assert (tmp_path / "judge_agreement_cache.jsonl").exists()


def test_identical_completions_share_one_query(tmp_path: Path) -> None:
    samples = [
        {
            "arm": "oracle",
            "seed": 0,
            "pid": "p0",
            "k": 0,
            "text": "yes same",
            "prompt": "ask",
        },
        {
            "arm": "oracle",
            "seed": 0,
            "pid": "p1",
            "k": 0,
            "text": "yes same",
            "prompt": "ask",
        },
    ]
    _fill(tmp_path, samples)
    grader = _Grader()
    result = run_judge_agreement(
        _spec(subsample=2),
        tmp_path,
        planned={"oracle": [0]},
        withdrawn=set(),
        final_step=4,
        grader=grader,
    )
    assert grader.calls == 1
    assert result["n_queries"] == 1
    assert result["n_cache_hits"] == 1
    assert result["n_paired"] == 2


def test_interrupted_run_resumes_from_the_cache(tmp_path: Path) -> None:
    samples = [
        {"arm": "oracle", "seed": 0, "pid": pid, "k": 0, "text": f"yes {pid}"}
        for pid in ("a", "b", "c", "d")
    ]
    _fill(tmp_path, samples)

    class _Flaky:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, prompt: str, response: str) -> str:
            del prompt, response
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("interrupted")
            return "YES"

    planned = {"oracle": [0]}
    with pytest.raises(RuntimeError, match="interrupted"):
        run_judge_agreement(
            _spec(subsample=4),
            tmp_path,
            planned=planned,
            withdrawn=set(),
            final_step=4,
            grader=_Flaky(),
        )
    assert len(read_jsonl(tmp_path / "judge_agreement_cache.jsonl")) == 2
    follow = _Grader()
    result = run_judge_agreement(
        _spec(subsample=4),
        tmp_path,
        planned=planned,
        withdrawn=set(),
        final_step=4,
        grader=follow,
    )
    assert follow.calls == 2
    assert result["n_queries"] == 2
    assert result["n_cache_hits"] == 2
    assert result["n_paired"] == 4
    assert result["agreement"] == pytest.approx(1.0)


def test_rubric_mismatch_raises_before_any_query(tmp_path: Path) -> None:
    grader = _Grader()
    with pytest.raises(ValueError, match="rubric_sha256"):
        run_judge_agreement(
            _spec(rubric_sha256="0" * 64),
            tmp_path,
            planned={"oracle": [0]},
            withdrawn=set(),
            final_step=4,
            grader=grader,
        )
    assert grader.calls == 0
    assert not (tmp_path / "judge_agreement_cache.jsonl").exists()


def test_temperature_must_be_zero(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="temperature"):
        run_judge_agreement(
            _spec(temperature=1),
            tmp_path,
            planned={"oracle": [0]},
            withdrawn=set(),
            final_step=4,
            grader=_Grader(),
        )


def test_missing_api_key_only_when_a_query_is_needed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VFATES_JUDGE_API_KEY", raising=False)
    monkeypatch.setenv("VFATES_JUDGE_BASE_URL", "http://127.0.0.1:9/v1")
    samples = [{"arm": "oracle", "seed": 0, "pid": "p0", "k": 0, "text": "yes lone"}]
    _fill(tmp_path, samples)
    with pytest.raises(RuntimeError, match="VFATES_JUDGE_API_KEY"):
        run_judge_agreement(
            _spec(subsample=1, n_boot=10),
            tmp_path,
            planned={"oracle": [0]},
            withdrawn=set(),
            final_step=4,
        )
    warm = tmp_path / "warm"
    _fill(warm, samples)
    first = run_judge_agreement(
        _spec(subsample=1, n_boot=10),
        warm,
        planned={"oracle": [0]},
        withdrawn=set(),
        final_step=4,
        grader=_Grader(),
    )
    assert first["n_queries"] == 1

    class _MustNotBuild:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError(f"built a client {args} {kwargs}")

    monkeypatch.setattr("vfates.analysis.judge_agreement.LLMJudge", _MustNotBuild)
    second = run_judge_agreement(
        _spec(subsample=1, n_boot=10),
        warm,
        planned={"oracle": [0]},
        withdrawn=set(),
        final_step=4,
    )
    assert second["n_queries"] == 0
    assert second["agreement"] == first["agreement"]
    assert second["base_url"] == OPENAI_BASE_URL


def test_remote_client_ignores_the_local_judge_base_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VFATES_JUDGE_BASE_URL", "http://127.0.0.1:9/v1")
    seen: dict[str, object] = {}

    class _Recording:
        def __init__(self, model: str, base_url: str | None = None, **kwargs: object) -> None:
            seen["model"] = model
            seen["base_url"] = base_url
            seen["rubric"] = kwargs.get("rubric")
            seen["max_tokens"] = kwargs.get("max_tokens")

        def complete(self, prompt: str, response: str) -> str:
            del prompt, response
            return "NO"

    monkeypatch.setattr("vfates.analysis.judge_agreement.LLMJudge", _Recording)
    _fill(tmp_path, [{"arm": "oracle", "seed": 0, "pid": "p0", "k": 0, "text": "yes z"}])
    result = run_judge_agreement(
        _spec(subsample=1),
        tmp_path,
        planned={"oracle": [0]},
        withdrawn=set(),
        final_step=4,
    )
    assert seen["base_url"] == OPENAI_BASE_URL
    assert seen["model"] == "gpt-4.1"
    assert seen["rubric"] == DEFAULT_RUBRIC
    assert seen["max_tokens"] == 16
    assert result["n_paired"] == 1
    assert "\u2013" not in render_agreement_markdown(result)
    assert "\u2014" not in render_agreement_markdown(result)


def test_frame_filters_step_split_and_withdrawn_arms(tmp_path: Path) -> None:
    samples = [
        {"arm": "oracle", "seed": 0, "pid": "p0", "k": 0, "text": "final-a"},
        {"arm": "oracle", "seed": 0, "pid": "p1", "k": 0, "text": "final-b"},
        {"arm": "oracle", "seed": 0, "pid": "p0", "k": 0, "text": "older", "step": 0},
        {
            "arm": "oracle",
            "seed": 0,
            "pid": "p0",
            "k": 0,
            "text": "train",
            "split": "train",
        },
        {"arm": "checker", "seed": 0, "pid": "p9", "k": 0, "text": "withdrawn"},
    ]
    # Last held-out final row for (p0, 0) must win over an earlier copy.
    samples.insert(0, {"arm": "oracle", "seed": 0, "pid": "p0", "k": 0, "text": "stale"})
    _fill(tmp_path, samples)
    # Legacy row: a stored 0 with no parse flag is a verdict.
    oracle = tmp_path / "runs" / "oracle__s0"
    append_jsonl(
        oracle / EVAL_SAMPLES_FILENAME,
        {
            "step": 4,
            "split": "heldout",
            "pid": "p1",
            "k": 0,
            "prompt": "ask p1",
            "text": "final-b",
            "judge": 0,
            "O": 0,
        },
    )
    frame = load_heldout_frame(
        tmp_path, {"checker": [0], "oracle": [0, 0]}, withdrawn={"checker"}, final_step=4
    )
    assert sorted((row.pid, row.text, row.local) for row in frame) == [
        ("p0", "final-a", 1),
        ("p1", "final-b", 0),
    ]
    with pytest.raises(ValueError, match="eval_samples"):
        load_heldout_frame(tmp_path, {"missing": [0]}, withdrawn=set(), final_step=4)
    assert load_heldout_frame(tmp_path, {"checker": [0]}, withdrawn={"checker"}, final_step=4) == []
    grader = _Grader()
    empty = run_judge_agreement(
        _spec(subsample=500),
        tmp_path,
        planned={"checker": [0]},
        withdrawn={"checker"},
        final_step=4,
        grader=grader,
    )
    assert grader.calls == 0
    assert empty["n_frame"] == 0
    assert empty["n_drawn"] == 0
    assert empty["agreement"] is None
    assert empty["n_queries"] == 0


def test_frame_requires_complete_prompt_coverage(tmp_path: Path) -> None:
    _fill(
        tmp_path,
        [
            {
                "arm": "oracle",
                "seed": 0,
                "pid": "p0",
                "k": 0,
                "text": "only",
                "prompt_pids": ["p0", "p1"],
            }
        ],
    )
    with pytest.raises(ValueError, match="eval_prompts"):
        load_heldout_frame(tmp_path, {"oracle": [0]}, withdrawn=set(), final_step=4)
    _fill(
        tmp_path / "k",
        [
            {"arm": "oracle", "seed": 0, "pid": "p0", "k": 0, "text": "a"},
        ],
    )
    with pytest.raises(ValueError, match="sample indices"):
        load_heldout_frame(
            tmp_path / "k",
            {"oracle": [0]},
            withdrawn=set(),
            final_step=4,
            samples_per_prompt=2,
        )


def test_bootstrap_count_defaults_to_10000(tmp_path: Path) -> None:
    spec = _spec(subsample=1)
    del spec["n_boot"]
    _fill(tmp_path, [{"arm": "oracle", "seed": 0, "pid": "p0", "k": 0, "text": "yes only"}])
    result = run_judge_agreement(
        spec,
        tmp_path,
        planned={"oracle": [0]},
        withdrawn=set(),
        final_step=4,
        grader=_Grader(),
    )
    assert result["bootstrap"]["n_boot"] == 10_000
    assert result["bootstrap"]["seed"] == 0
    assert result["bootstrap"]["cluster"] == "pid"


def test_reply_cache_key_depends_on_the_completion() -> None:
    one = reply_cache_key(
        model="gpt-4.1",
        rubric_sha256=RUBRIC_SHA,
        max_tokens=16,
        base_url=OPENAI_BASE_URL,
        prompt="ask",
        text="yes",
    )
    assert one == reply_cache_key(
        model="gpt-4.1",
        rubric_sha256=RUBRIC_SHA,
        max_tokens=16,
        base_url=OPENAI_BASE_URL + "/",
        prompt="ask",
        text="yes",
    )
    assert one != reply_cache_key(
        model="gpt-4.1",
        rubric_sha256=RUBRIC_SHA,
        max_tokens=16,
        base_url=OPENAI_BASE_URL,
        prompt="ask",
        text="no",
    )


def _ifeval_row() -> list[dict]:
    return [
        {
            "row_id": 0,
            "question": "Say hello.",
            "func_name": "validate_uppercase",
            "args": {},
            "constraint_type": "case",
            "constraint": "upper",
        }
    ]


class _AlwaysYes(Judge):
    name = "yes"

    def __init__(self) -> None:
        self.calls = 0

    def judge(self, prompt: str, response: str, **context: object) -> int:
        del prompt, response, context
        self.calls += 1
        return 1


class _FixedGen(Generator):
    def __init__(self, text: str) -> None:
        self.model_id = "fixed"
        self.text = text
        self.calls = 0

    def set_adapter(self, path: str | None) -> None:
        del path

    def generate(
        self,
        prompts: list[str],
        n: int,
        *,
        max_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int = 0,
        system_prompt: str | None = None,
    ) -> list[list[GenOut]]:
        del max_tokens, temperature, top_p, seed, system_prompt
        self.calls += 1
        return [
            [GenOut(text=self.text, n_tokens=2, finish_reason="stop") for _ in range(n)]
            for _ in prompts
        ]


def test_oracle_records_the_judge_on_checker_failures() -> None:
    judge = _AlwaysYes()
    task = IFEvalTask(rows=_ifeval_row(), heldout_fraction=0.0, judge=judge)
    problem = task.problems("train", 1, seed=0)[0]
    lower = task.parse(problem, "hello")
    assert task.oracle(problem, lower, "hello") == 0
    assert lower.extras["content_judge"] == 1
    assert lower.extras["content_judge_parsed"] is True
    assert judge.calls == 1
    upper = task.parse(problem, "HELLO")
    assert task.oracle(problem, upper, "HELLO") == 1
    assert upper.extras["content_judge"] == 1


def test_unparsed_local_reply_is_persisted_as_unparsed() -> None:
    class _Maybe(Generator):
        model_id = "maybe"

        def __init__(self) -> None:
            self.calls = 0

        def generate(
            self,
            prompts,
            n,
            *,
            max_tokens,
            temperature=1.0,
            top_p=1.0,
            seed=0,
            system_prompt=None,
        ):
            del prompts, n, max_tokens, temperature, top_p, seed, system_prompt
            self.calls += 1
            return [[GenOut(text="maybe later", n_tokens=2, finish_reason="stop")]]

    gen = _Maybe()
    task = IFEvalTask(rows=_ifeval_row(), heldout_fraction=0.0, judge=LocalJudge(gen))
    problem = task.problems("train", 1, seed=0)[0]
    parsed = task.parse(problem, "HELLO")
    assert task.oracle(problem, parsed, "HELLO") == 0
    assert parsed.extras["content_judge"] == 0
    assert parsed.extras["content_judge_parsed"] is False
    assert gen.calls == 1


def test_evaluate_persists_judge_samples_once(tmp_path: Path) -> None:
    judge = _AlwaysYes()
    task = IFEvalTask(rows=_ifeval_row(), heldout_fraction=0.0, judge=judge)
    problem = task.problems("train", 1, seed=0)[0]
    stats = evaluate_policy(
        _FixedGen("HELLO"),
        task,
        [problem],
        n_samples=2,
        max_tokens=8,
        temperature=1.0,
        seed=0,
    )
    assert judge.calls == 2
    assert "judge_samples" in stats
    rows = eval_sample_rows(4, "heldout", stats)
    assert [row["k"] for row in rows] == [0, 1]
    assert rows[0]["text"] == "HELLO"
    assert rows[0]["prompt"] == "Say hello."
    assert rows[0]["judge"] == 1
    assert rows[0]["judge_parsed"] is True
    append_eval_samples(tmp_path, 4, "heldout", stats)
    assert len(read_jsonl(tmp_path / EVAL_SAMPLES_FILENAME)) == 2

    plain = ChainSumTask()
    chain = evaluate_policy(
        _FixedGen("\\boxed{0}"),
        plain,
        plain.problems("heldout", 1, seed=0),
        n_samples=1,
        max_tokens=8,
        temperature=1.0,
        seed=0,
    )
    assert "judge_samples" not in chain
    append_eval_samples(tmp_path / "plain", 0, "heldout", chain)
    assert not (tmp_path / "plain" / EVAL_SAMPLES_FILENAME).exists()


def test_checkpoint_eval_writes_samples_and_skips_them_on_rerun(tmp_path: Path) -> None:
    judge = _AlwaysYes()
    task = IFEvalTask(rows=_ifeval_row(), heldout_fraction=0.0, judge=judge)
    held = task.problems("train", 1, seed=0)
    config = GRPOConfig.from_dict(
        {
            "model_id": "fake",
            "steps": 4,
            "prompts_per_step": 2,
            "out_dir": str(tmp_path),
            "seed": 0,
            "eval_every": 2,
            "eval_heldout_n": 1,
            "eval_train_n": 0,
            "eval_samples_per_prompt": 2,
            "save_every": 2,
        }
    )
    written = evaluate_checkpoints(tmp_path, task, config, _FixedGen("HELLO"), [], held)
    assert written == [(0, "heldout")]
    samples = read_jsonl(tmp_path / EVAL_SAMPLES_FILENAME)
    assert len(samples) == 2
    assert {row["text"] for row in samples} == {"HELLO"}
    eval_row = read_jsonl(tmp_path / "eval.jsonl")[0]
    assert "text" not in eval_row
    assert "judge" not in eval_row
    text = (tmp_path / EVAL_SAMPLES_FILENAME).read_text()
    again = evaluate_checkpoints(tmp_path, task, config, _FixedGen("HELLO"), [], held)
    assert again == []
    assert judge.calls == 2
    assert (tmp_path / EVAL_SAMPLES_FILENAME).read_text() == text


def _forecast(tau: float = 0.2) -> dict:
    return {
        "transfer": {
            "tau_pool": tau,
            "tau_hard": tau,
            "tau_rel": 0.4,
            "reliability": 0.8,
            "cos_gV_gO": 0.2,
            "base_acc_heldout": 0.3,
        },
        "step0": {"J": 0.2, "fpr": 0.1, "fnr": 0.2},
        "mean_precision": tau,
        "modal_precision": {"value": tau, "tie_fraction": 0.0},
    }


def _write_run(path: Path, arm: str, steps: int, *, completed: int | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for step, acc in ((0, 0.2), (steps, 0.5)):
        append_jsonl(
            path / "eval.jsonl",
            {"step": step, "split": "heldout", "oracle_acc": acc, "J": 0.5, "fpr": 0.1, "fnr": 0.2},
        )
        append_jsonl(
            path / "eval.jsonl",
            {"step": step, "split": "train", "oracle_acc": acc, "J": 0.5, "fpr": 0.1, "fnr": 0.2},
        )
        append_jsonl(
            path / "eval_prompts.jsonl",
            {"step": step, "split": "heldout", "acc": {"p0": acc, "p1": acc}},
        )
    for pid in ("p0", "p1"):
        for k in (0, 1):
            append_jsonl(
                path / EVAL_SAMPLES_FILENAME,
                {
                    "step": steps,
                    "split": "heldout",
                    "pid": pid,
                    "k": k,
                    "prompt": f"ask {pid}",
                    "text": f"yes {pid}-{k}",
                    "judge": 1,
                    "judge_parsed": True,
                    "O": 1,
                },
            )
    append_jsonl(
        path / EVAL_SAMPLES_FILENAME,
        {
            "step": 0,
            "split": "heldout",
            "pid": "p0",
            "k": 0,
            "prompt": "ask p0",
            "text": "step-zero",
            "judge": 0,
            "judge_parsed": True,
            "O": 0,
        },
    )
    write_json(
        path / "summary.json", {"steps_completed": steps if completed is None else completed}
    )
    write_json(path / "run.json", {"arm": arm})


def _install(monkeypatch: pytest.MonkeyPatch, trainer=None) -> list:
    calls: list = []

    def run_audit(config, out_dir):
        del config
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {"oracle": _forecast(0.2)}
        write_json(out / "config.json", {"config": {"forecast": {"loss_agg": "seq_mean"}}})
        write_json(out / "calibration.json", {"arms": []})
        write_json(out / "forecasts.json", payload)
        lock_forecasts(payload, out / "lock.json", note="synthetic")
        calls.append("audit")

    def train_arm(audit_dir, arm, train_config, out_dir, **kwargs):
        del audit_dir, kwargs
        calls.append(("train", arm))
        if trainer is not None:
            trainer(arm, train_config, Path(out_dir))
        else:
            _write_run(Path(out_dir), arm, int(train_config["steps"]))
        return {"steps_completed": int(train_config["steps"])}

    monkeypatch.setattr("vfates.audit.run_audit", run_audit)
    monkeypatch.setattr("vfates.experiment.train_arm", train_arm)
    monkeypatch.setattr("vfates.verifiers.calibrate.arm_validity", lambda report: {})
    return calls


def _study_config(**agreement_overrides: object) -> dict:
    return {
        "audit": {"task": {"type": "chain_sum"}},
        "study": {
            "seeds": [0],
            "arms": ["oracle"],
            "backend": "trl",
            "train": {
                "steps": 4,
                "loss_agg": "seq_mean",
                "model_id": "tiny",
                "eval_samples_per_prompt": 2,
            },
        },
        "analysis": {
            "final_step": 4,
            "hypotheses": False,
            "early": {"enabled": False},
            "n_boot": 20,
            "agreement_check": _spec(subsample=2, **agreement_overrides),
        },
    }


class _Remote:
    calls: ClassVar[int] = 0
    base_urls: ClassVar[list[str | None]] = []

    def __init__(self, model: str, base_url: str | None = None, **kwargs: object) -> None:
        assert model == "gpt-4.1"
        assert kwargs.get("rubric") == DEFAULT_RUBRIC
        assert kwargs.get("max_tokens") == 16
        _Remote.base_urls.append(base_url)

    def complete(self, prompt: str, response: str) -> str:
        del prompt, response
        _Remote.calls += 1
        return "YES"


def test_study_hook_reports_agreement_and_reanalyze_uses_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VFATES_JUDGE_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.delenv("VFATES_JUDGE_API_KEY", raising=False)
    _Remote.calls = 0
    _Remote.base_urls = []
    monkeypatch.setattr("vfates.analysis.judge_agreement.LLMJudge", _Remote)
    _install(monkeypatch)
    report = run_study(_study_config(), tmp_path)
    assert report["ok"] is True
    assert report["analysis"]["oracle_gain"]["value"] == pytest.approx(0.3)
    block = report["judge_agreement"]
    assert block["reported_not_tested"] is True
    assert block["n_frame"] == 4
    assert block["n_drawn"] == 2
    assert block["n_queries"] == 2
    assert block["agreement"] == pytest.approx(1.0)
    assert block["base_url"] == OPENAI_BASE_URL
    assert _Remote.base_urls == [OPENAI_BASE_URL]
    saved = read_json(tmp_path / "judge_agreement.json")
    assert saved["n_drawn"] == block["n_drawn"]
    text = (tmp_path / "report.md").read_text()
    assert "## Judge agreement" in text
    assert "Reported, not tested" in text
    assert "\u2013" not in text and "\u2014" not in text
    calls = _Remote.calls
    again = reanalyze(_study_config(), tmp_path)
    assert again["ok"] is True
    assert again["judge_agreement"]["n_queries"] == 0
    assert again["judge_agreement"]["agreement"] == pytest.approx(1.0)
    assert _Remote.calls == calls


def test_agreement_failure_is_recorded_and_reanalyze_completes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VFATES_JUDGE_API_KEY", raising=False)
    _install(monkeypatch)
    report = run_study(_study_config(), tmp_path)
    assert report["ok"] is True
    assert report["analysis"]["oracle_gain"]["value"] == pytest.approx(0.3)
    failed = report["judge_agreement"]
    assert failed["reported_not_tested"] is True
    assert "VFATES_JUDGE_API_KEY is not set" in failed["error"]
    assert read_json(tmp_path / "judge_agreement.json")["error"] == failed["error"]
    assert "Not computed: RuntimeError" in (tmp_path / "report.md").read_text()
    _Remote.calls = 0
    monkeypatch.setattr("vfates.analysis.judge_agreement.LLMJudge", _Remote)
    again = reanalyze(_study_config(), tmp_path)
    assert "error" not in again["judge_agreement"]
    assert again["judge_agreement"]["n_queries"] == 2
    assert again["judge_agreement"]["agreement"] == pytest.approx(1.0)


def test_incomplete_study_does_not_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _Remote.calls = 0
    monkeypatch.setattr("vfates.analysis.judge_agreement.LLMJudge", _Remote)

    def trainer(arm, train_config, out_dir):
        _write_run(out_dir, arm, int(train_config["steps"]), completed=1)

    _install(monkeypatch, trainer=trainer)
    report = run_study(_study_config(), tmp_path)
    assert report["ok"] is False
    assert "judge_agreement" not in report
    assert _Remote.calls == 0
    assert not (tmp_path / "judge_agreement.json").exists()


def test_study_rubric_mismatch_raises_and_does_not_write_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch)
    with pytest.raises(ValueError, match="rubric_sha256"):
        run_study(_study_config(rubric_sha256="0" * 64), tmp_path)
    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / "judge_agreement.json").exists()
