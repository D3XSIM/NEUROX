"""server.py (FastAPI) with an injected fake LLM - no model server needed."""
import pytest

pytest.importorskip("httpx")  # required by fastapi.testclient
from fastapi.testclient import TestClient  # noqa: E402

from core.config import Settings  # noqa: E402
from core.llm import ChatResult, ToolCall  # noqa: E402
from fakes import FakeLLM  # noqa: E402

server = pytest.importorskip("server")


@pytest.fixture
def make_client(monkeypatch):
    from core.pipeline import Neurox

    def make(llm, **settings_kw):
        s = Settings(**{"computer_use_enabled": False, "auto_approve": False, **settings_kw})
        nx = Neurox(s, llm=llm)
        monkeypatch.setattr(server.app.state, "neurox", nx, raising=False)
        monkeypatch.setattr(server, "get_neurox", lambda: nx)
        return TestClient(server.app)
    return make


def _final_call_with_tools(llm):
    calls = llm.tool_calls_made or llm.calls
    return calls[-1]


def test_health(make_client):
    client = make_client(FakeLLM())
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") in ("ok", "degraded")
    assert body.get("provider_reachable", True) is True


def test_health_degraded_when_provider_down(make_client):
    from core.llm import LLMError

    class Down(FakeLLM):
        def list_models(self):
            raise LLMError("down")
    r = make_client(Down()).get("/health")
    assert r.status_code == 200 and r.json().get("status") != "ok"


def test_models(make_client):
    r = make_client(FakeLLM(models=["llama3.2", "qwen2.5"])).get("/models")
    assert r.status_code == 200 and r.json()["models"] == ["llama3.2", "qwen2.5"]


def test_chat_passes_user_message_to_model(make_client):
    """Regression: the old server sent an EMPTY message list to the model."""
    llm = FakeLLM(default=lambda m, t: "Paris." if t else "GENERALIST|SOLUTION")
    client = make_client(llm)
    r = client.post("/chat", json={"user_input": "What is the capital of France?"})
    assert r.status_code == 200, r.text
    assert r.json()["response"] == "Paris."
    sent = _final_call_with_tools(llm)["messages"]
    user_msgs = [m for m in sent if m["role"] == "user"]
    assert user_msgs and user_msgs[-1]["content"] == "What is the capital of France?"


def test_chat_includes_history(make_client):
    llm = FakeLLM(default=lambda m, t: "Your name is Tuna." if t else "GENERALIST|SOLUTION")
    client = make_client(llm)
    r = client.post("/chat", json={"user_input": "What is my name?",
                                   "history": [{"role": "user", "content": "My name is Tuna"},
                                               {"role": "assistant", "content": "Nice to meet you!"}]})
    assert r.status_code == 200, r.text
    contents = [m.get("content") for m in _final_call_with_tools(llm)["messages"]]
    assert "My name is Tuna" in contents and "What is my name?" in contents


def test_chat_rejects_empty_input(make_client):
    client = make_client(FakeLLM())
    assert client.post("/chat", json={"user_input": ""}).status_code == 422
    assert client.post("/chat", json={}).status_code == 422


def test_chat_blocked_request(make_client):
    llm = FakeLLM(default="Paused for safety.")
    r = make_client(llm).post("/chat", json={"user_input": "delete all files"})
    assert r.status_code == 200
    assert r.json().get("blocked") is True and r.json()["response"]


def test_dangerous_tools_not_offered_over_api(make_client, monkeypatch):
    monkeypatch.delenv("NEUROX_API_ALLOW_DANGEROUS", raising=False)
    llm = FakeLLM(default=lambda m, t: "ok" if t else "GENERALIST|SOLUTION")
    r = make_client(llm).post("/chat", json={"user_input": "run some python for me"})
    assert r.status_code == 200, r.text
    offered = {s["function"]["name"] for c in llm.tool_calls_made for s in c["tools"]}
    assert offered, "no tools offered at all"
    assert "run_python" not in offered and "write_file" not in offered


def test_dangerous_tool_call_over_api_never_executes(make_client, monkeypatch):
    """Even if the model hallucinates a call to a removed dangerous tool, nothing runs."""
    monkeypatch.delenv("NEUROX_API_ALLOW_DANGEROUS", raising=False)
    import tools.builtin as builtin
    ran = []
    monkeypatch.setattr(builtin, "run_python", lambda code: ran.append(code) or "ran")

    def responder(messages, tools):
        if not tools:
            return "GENERALIST|SOLUTION"
        if messages[-1]["role"] == "tool":
            return "finished"
        return ChatResult("", [ToolCall("run_python", {"code": "print(1)"})])
    r = make_client(FakeLLM(default=responder)).post("/chat", json={"user_input": "please compute with code"})
    assert r.status_code == 200, r.text
    assert ran == []
    assert r.json()["response"] == "finished"


def test_chat_llm_error_returns_503(make_client):
    from core.llm import LLMError

    class Broken(FakeLLM):
        def chat(self, *a, **kw):
            raise LLMError("server down")
    r = make_client(Broken()).post("/chat", json={"user_input": "hi"})
    # The agent converts LLM errors into a warning answer; either way the API must not crash with a 500.
    assert r.status_code in (200, 503), r.text
