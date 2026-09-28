"""Content judges: heuristic rules, cached verdicts, and the HTTP judge."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from vfates.gen.base import Generator
from vfates.tasks.judges import (
    MASTER_RM_PROMPT_TEMPLATE,
    MASTER_RM_REVISION,
    MASTER_RM_SYSTEM,
    VERIFIER_PASS_TAG,
    CachedJudge,
    GeneralVerifierJudge,
    HeuristicJudge,
    HeuristicVerifierJudge,
    Judge,
    LLMJudge,
    LocalJudge,
    MasterRMJudge,
    extract_student_answer,
    format_master_rm_prompt,
    format_verifier_prompt,
    general_verifier_bit,
    judge_from_description,
    parse_master_rm,
    parse_yes_no,
    verifier_cache_key,
)
from vfates.tasks.qa import MASTER_KEYS
from vfates.types import GenOut

GOOD = (
    "IPv6 expands the address space and simplifies routing for hosts. "
    "It also improves autoconfiguration, so networks need less manual setup "
    "than they did under the previous protocol."
)
TULU = (
    "timberlike, unconnectedly Conversation 1 User : "
    "timberlike, unconnectedly Conversation 2 User : "
    "timberlike, unconnectedly Conversation 3 User : "
    "timberlike, unconnectedly Conversation 4 User : "
    "timberlike, unconnectedly Conversation 5 User : "
    "timberlike, unconnectedly Conversation 6 User : "
)
KEYWORDS = "alpha, bravo, charlie, delta, echo, foxtrot, golf, hotel, india, juliet, kilo, lima"


def test_parse_yes_no() -> None:
    assert parse_yes_no("YES") == 1
    assert parse_yes_no("no, it does not") == 0
    assert parse_yes_no("maybe") is None
    assert parse_yes_no("YES or NO") is None
    assert parse_yes_no("YES YES") is None
    assert parse_yes_no("This is not a YES") == 1


def test_heuristic_rules() -> None:
    judge = HeuristicJudge()
    assert judge.judge("Explain IPv6.", GOOD) == 1
    assert judge.judge("Write a story.", TULU) == 0
    assert judge.judge("Hi", "hello " * 30) == 0
    assert judge.judge("Hi", "!" * 80) == 0
    assert judge.judge("List words.", KEYWORDS) == 0
    assert judge.judge("Hi", "too short") == 0
    described = judge.describe()
    assert described["min_content_words"] == 12
    assert described["max_repeat_ratio"] == 0.4


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.bodies.append(body)  # type: ignore[attr-defined]
        self.server.auths.append(self.headers.get("Authorization"))  # type: ignore[attr-defined]
        if self.server.failures > 0:  # type: ignore[attr-defined]
            self.server.failures -= 1  # type: ignore[attr-defined]
            self.send_response(500)
            self.end_headers()
            return
        path = self.path.rstrip("/")
        if path.endswith("/completions") and not path.endswith("/chat/completions"):
            body = {"choices": [{"text": self.server.reply}]}  # type: ignore[attr-defined]
        else:
            body = {"choices": [{"message": {"content": self.server.reply}}]}  # type: ignore[attr-defined]
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: object) -> None:
        return


def _server(reply: str, failures: int = 0) -> tuple[ThreadingHTTPServer, threading.Thread]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.reply = reply  # type: ignore[attr-defined]
    server.failures = failures  # type: ignore[attr-defined]
    server.bodies = []  # type: ignore[attr-defined]
    server.auths = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_llm_judge_retries_and_parses() -> None:
    server, _thread = _server("YES", failures=1)
    try:
        port = server.server_address[1]
        judge = LLMJudge(
            "fake-model",
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key="sk-test",
            max_retries=3,
            backoff_s=0.01,
            timeout_s=5,
        )
        assert judge.judge("Explain tides.", "The moon pulls the oceans.") == 1
        assert len(server.bodies) == 2  # type: ignore[attr-defined]
        assert server.auths[0] == "Bearer sk-test"  # type: ignore[attr-defined]
        assert server.bodies[1]["temperature"] == 0  # type: ignore[attr-defined]
        assert server.bodies[1]["model"] == "fake-model"  # type: ignore[attr-defined]
        assert server.bodies[1]["max_tokens"] == 16  # type: ignore[attr-defined]
        described = judge.describe()
        assert described["max_tokens"] == 16
        assert described["revision"] is None
        assert described["chat_template_kwargs"] is None
        assert "chat_template_kwargs" not in server.bodies[1]  # type: ignore[attr-defined]
        assert described["rubric_sha256"]
        assert "API" not in json.dumps(described)
    finally:
        server.shutdown()
        server.server_close()


def test_llm_judge_no_and_unparseable() -> None:
    server, _thread = _server("NO")
    try:
        port = server.server_address[1]
        judge = LLMJudge(
            "fake-model",
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key="sk-test",
            max_retries=2,
            backoff_s=0.01,
            timeout_s=5,
        )
        assert judge.judge("q", "a") == 0
        server.reply = "maybe later"  # type: ignore[attr-defined]
        assert judge.judge("q", "b") == 0
        assert len(server.bodies) == 1 + 2  # type: ignore[attr-defined]
    finally:
        server.shutdown()
        server.server_close()


def test_llm_judge_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VFATES_JUDGE_API_KEY", raising=False)
    judge = LLMJudge("fake-model", api_key=None)
    with pytest.raises(RuntimeError, match="VFATES_JUDGE_API_KEY"):
        judge.judge("q", "a")


class _FakeGen(Generator):
    model_id = "fake-local"

    def generate(
        self, prompts, n, *, max_tokens, temperature=1.0, top_p=1.0, seed=0, system_prompt=None
    ):
        assert temperature == 0.0
        assert n == 1
        assert system_prompt
        return [[GenOut(text="YES", n_tokens=1)]]


def test_local_judge() -> None:
    judge = LocalJudge(_FakeGen())
    assert judge.judge("q", "a") == 1
    assert judge.describe()["temperature"] == 0


class _CountJudge(Judge):
    name = "count"

    def __init__(self) -> None:
        self.calls = 0

    def judge(self, prompt: str, response: str, **context: object) -> int:
        del context
        self.calls += 1
        return 1 if "yes" in response else 0

    def describe(self) -> dict:
        return {"name": self.name}


def test_cached_judge(tmp_path) -> None:
    inner = _CountJudge()
    path = tmp_path / "cache.jsonl"
    cached = CachedJudge(inner, path)
    assert cached.judge("q", "yes please") == 1
    assert cached.judge("q", "yes please") == 1
    assert inner.calls == 1
    assert cached.judge("q", "no") == 0
    assert inner.calls == 2
    reloaded = CachedJudge(_CountJudge(), path)
    assert reloaded.judge("q", "yes please") == 1
    assert reloaded.inner.calls == 0


def test_heuristic_format_rows_and_required_word() -> None:
    judge = HeuristicJudge()
    assert judge.judge("q", '{"answer": "Paris is the capital of France."}') == 1
    assert judge.judge("q", "I am a man of few words but I write well now today.") == 1
    twelve = "one two three four five six seven eight nine ten eleven twelve"
    assert judge.judge("q", twelve) == 1
    assert judge.judge("q", "HELLO") == 0
    assert judge.judge("q", "HELLO", func_name="validate_uppercase") == 1
    words = ["alpha"] * 17 + [f"w{i}" for i in range(25)]
    text = " ".join(words)
    assert judge.judge("q", text) == 0
    assert judge.judge("q", text, required_word="alpha") == 1
    assert (
        judge.judge(
            "q",
            text,
            func_name="verify_keyword_frequency",
            checker_args={"word": "alpha", "N": 17},
        )
        == 1
    )
    assert judge.judge("Write a story.", TULU) == 0


class _RubricJudge(Judge):
    def __init__(self, rubric: str) -> None:
        self.name = "rubric"
        self.rubric = rubric
        self.calls = 0

    def describe(self) -> dict:
        from vfates.utils.io import sha256_text

        return {"name": self.name, "rubric_sha256": sha256_text(self.rubric)}

    def judge(self, prompt: str, response: str, **context: object) -> int:
        del prompt, response, context
        self.calls += 1
        return 1


def test_cache_key_changes_with_rubric_and_context(tmp_path) -> None:
    path = tmp_path / "cache.jsonl"
    first = _RubricJudge("alpha")
    cached = CachedJudge(first, path)
    assert cached.judge("q", "a") == 1
    assert cached.judge("q", "a") == 1
    assert first.calls == 1
    second = _RubricJudge("beta")
    other = CachedJudge(second, path)
    assert other.judge("q", "a") == 1
    assert second.calls == 1
    assert cached.judge("q", "a", func_name="validate_uppercase") == 1
    assert first.calls == 2


def test_complete_prompt_reads_completion_text() -> None:
    server, _thread = _server("Final Decision: Yes")
    try:
        port = server.server_address[1]
        judge = LLMJudge(
            "fake-model",
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key="sk-test",
            max_tokens=1024,
            max_retries=1,
            timeout_s=5,
        )
        assert judge.complete_prompt("User: check this. Assistant:") == "Final Decision: Yes"
        body = server.bodies[0]  # type: ignore[attr-defined]
        assert body["prompt"] == "User: check this. Assistant:"
        assert body["temperature"] == 0
        assert body["max_tokens"] == 1024
        assert "messages" not in body
    finally:
        server.shutdown()
        server.server_close()


def test_general_verifier_prompt_and_cache_key() -> None:
    prompt = format_verifier_prompt("What is 2+2?", "4", "4")
    assert prompt.startswith("User: ### Question: What is 2+2?")
    assert "### Ground Truth Answer: 4" in prompt
    assert "### Student Answer: 4" in prompt
    assert prompt.endswith('output "Final Decision: No". Assistant:')
    assert VERIFIER_PASS_TAG in prompt
    assert extract_student_answer(r"reason \boxed{4}") == "4"
    assert extract_student_answer("The answer is: 4\n") == "4"
    assert extract_student_answer("") == "No Answer"
    assert general_verifier_bit("Final Decision: Yes") == 1
    assert general_verifier_bit("Final Decision: No") == 0
    assert verifier_cache_key("m", "rev", prompt) == verifier_cache_key("m", "rev", prompt)
    assert verifier_cache_key("m", "rev", prompt) != verifier_cache_key("m", "other", prompt)
    heuristic = HeuristicVerifierJudge()
    assert heuristic.judge("q", r"\boxed{Paris}", gold="paris") == 1
    assert heuristic.judge("q", r"\boxed{London}", gold="paris") == 0
    assert heuristic.complete_decision("q", r"\boxed{Paris}", "paris") == "Final Decision: Yes"


def test_general_verifier_judge_caches_the_filled_prompt(tmp_path) -> None:
    server, _thread = _server("Final Decision: Yes")
    try:
        port = server.server_address[1]
        judge = GeneralVerifierJudge(
            model="TIGER-Lab/general-verifier",
            revision="rev",
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key="sk-test",
            max_tokens=1024,
            max_retries=1,
            timeout_s=5,
            backoff_s=0.01,
            cache_path=tmp_path / "judge.jsonl",
        )
        assert judge.judge("What is 2+2?", r"working \boxed{4}", gold="4") == 1
        assert judge.judge("What is 2+2?", r"again \boxed{4}", gold="4") == 1
        assert len(server.bodies) == 1  # type: ignore[attr-defined]
        filled = format_verifier_prompt("What is 2+2?", "4", "4")
        assert server.bodies[0]["prompt"] == filled  # type: ignore[attr-defined]
        server.reply = "Final Decision: No"  # type: ignore[attr-defined]
        assert judge.judge("What is 2+2?", r"\boxed{5}", gold="4") == 0
        assert len(server.bodies) == 2  # type: ignore[attr-defined]
        described = judge.describe()
        assert described["revision"] == "rev"
        assert described["temperature"] == 0
        assert "sk-test" not in json.dumps(described)
    finally:
        server.shutdown()
        server.server_close()


def test_llm_judge_sends_chat_template_kwargs_on_the_chat_body() -> None:
    server, _thread = _server("YES")
    try:
        port = server.server_address[1]
        judge = LLMJudge(
            "Qwen/Qwen3-14B",
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key="sk-test",
            revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
            chat_template_kwargs={"enable_thinking": False},
            max_retries=1,
            backoff_s=0.01,
            timeout_s=5,
        )
        assert judge.complete("Explain tides.", "The moon pulls the oceans.") == "YES"
        body = server.bodies[0]  # type: ignore[attr-defined]
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert "extra_body" not in body
        assert set(body) >= {"model", "messages", "chat_template_kwargs", "temperature"}
        described = judge.describe()
        assert described["revision"] == "40c069824f4251a91eefaf281ebe4c544efd3e18"
        assert described["chat_template_kwargs"] == {"enable_thinking": False}
        judge.complete_prompt("raw prompt")
        completion = server.bodies[1]  # type: ignore[attr-defined]
        assert "chat_template_kwargs" not in completion
        assert "messages" not in completion
    finally:
        server.shutdown()
        server.server_close()


def test_master_rm_prompt_parses_yes_and_keeps_master_keys(tmp_path) -> None:
    assert parse_master_rm("YES") == 1
    assert parse_master_rm(" YES\n") == 1
    assert parse_master_rm("NO") == 0
    assert parse_master_rm("YES.") == 0
    assert parse_master_rm("yes") == 0
    lines = MASTER_RM_PROMPT_TEMPLATE.splitlines()
    assert lines[0] == ""
    assert lines[1].startswith("Given a problem, determine whether the final answer")
    assert lines[1].endswith("  ")
    assert "**Output:**  " in lines
    assert "{question}" in MASTER_RM_PROMPT_TEMPLATE
    filled = format_master_rm_prompt("What is 2+2?", "Thought process:", "4")
    assert "What is 2+2?" in filled
    assert "**Solution Process (Final Step Only):**  \nThought process:" in filled
    assert "**Reference Answer:**  \n4" in filled
    server, _thread = _server("NO")
    try:
        port = server.server_address[1]
        judge = MasterRMJudge(
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key="sk-test",
            chat_template_kwargs={"enable_thinking": False},
            max_retries=1,
            timeout_s=5,
            backoff_s=0.01,
            cache_path=tmp_path / "master.jsonl",
        )
        assert judge.describe()["revision"] == MASTER_RM_REVISION
        assert judge.describe()["type"] == "master_rm"
        for key in MASTER_KEYS:
            assert judge.judge("What is 2+2?", key, gold="4") == 0
            user = server.bodies[-1]["messages"][1]["content"]  # type: ignore[attr-defined]
            assert server.bodies[-1]["messages"][0]["content"] == MASTER_RM_SYSTEM  # type: ignore[attr-defined]
            assert user == format_master_rm_prompt("What is 2+2?", key, "4")
            assert key in user
            assert server.bodies[-1]["chat_template_kwargs"] == {"enable_thinking": False}  # type: ignore[attr-defined]
            assert "extra_body" not in server.bodies[-1]  # type: ignore[attr-defined]
        calls = len(server.bodies)  # type: ignore[attr-defined]
        assert judge.judge("What is 2+2?", MASTER_KEYS[0], gold="4") == 0
        assert len(server.bodies) == calls  # type: ignore[attr-defined]
        server.reply = "YES"  # type: ignore[attr-defined]
        assert judge.judge("What is 2+2?", "4", gold="4") == 1
        rebuilt = judge_from_description(judge.describe())
        assert rebuilt.describe() == judge.describe()
    finally:
        server.shutdown()
        server.server_close()


def test_local_judge_records_rubric_hash() -> None:
    judge = LocalJudge(_FakeGen(), rubric="alpha rubric")
    other = LocalJudge(_FakeGen(), rubric="beta rubric")
    assert judge.describe()["rubric_sha256"]
    assert judge.describe()["rubric_sha256"] != other.describe()["rubric_sha256"]
    assert judge.describe()["max_tokens"] == 16
