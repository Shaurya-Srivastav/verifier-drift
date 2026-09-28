"""Content judges: heuristic rules, cached verdicts, and the HTTP judge."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from vfates.gen.base import Generator
from vfates.tasks.judges import (
    CachedJudge,
    HeuristicJudge,
    Judge,
    LLMJudge,
    LocalJudge,
    parse_yes_no,
)
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
        data = json.dumps(
            {"choices": [{"message": {"content": self.server.reply}}]}  # type: ignore[attr-defined]
        ).encode()
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
        assert "API" not in json.dumps(judge.describe())
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

    def judge(self, prompt: str, response: str) -> int:
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
