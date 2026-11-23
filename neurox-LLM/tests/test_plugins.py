"""Plugin discovery/order, security gate, orchestrator fast-path and full end-to-end turns."""
import pytest

from core.config import Settings
from core.llm import ChatResult, ToolCall
from core.plugin_manager import PluginManager
from core.tools import ToolRegistry
from fakes import FakeLLM

EXPECTED_ORDER = {"SecurityPlugin": 1, "OrchestratorPlugin": 10, "SearchPlugin": 20, "SimulatorPlugin": 92,
                  "ChatPlugin": 100}


@pytest.fixture(scope="module")
def manager():
    m = PluginManager()
    m.discover_plugins()
    return m


def by_class(manager, cls_name):
    found = [p for p in manager.plugins if type(p).__name__ == cls_name]
    assert found, f"{cls_name} not loaded; loaded: {[type(p).__name__ for p in manager.plugins]}"
    return found[0]


def ctx_for(text, llm=None, settings=None, **extra):
    s = settings or Settings()
    llm = llm or FakeLLM()
    c = {"user_input": text, "messages": [{"role": "user", "content": text}], "images": [], "settings": s,
         "llm": llm, "registry": ToolRegistry(), "trace": [], "on_token_callback": None, "on_event": None}
    c.update(extra)
    return c


# ------------------------------------------------------------------------------------- discovery
def test_core_plugins_discovered_with_expected_order(manager):
    names = [type(p).__name__ for p in manager.plugins]
    for cls_name, order in EXPECTED_ORDER.items():
        assert by_class(manager, cls_name).order == order
    idx = {n: names.index(n) for n in EXPECTED_ORDER}
    assert idx["SecurityPlugin"] < idx["OrchestratorPlugin"] < idx["SearchPlugin"] < idx["SimulatorPlugin"] \
        < idx["ChatPlugin"]


def test_plugins_sorted_and_unique(manager):
    orders = [p.order for p in manager.plugins]
    assert orders == sorted(orders)
    names = [type(p).__name__ for p in manager.plugins]
    assert len(names) == len(set(names)), f"duplicate plugin instances: {names}"
    assert not any("ConceptMap" in n for n in names)
    for p in manager.plugins:
        assert p.name and p.name != "Unknown Plugin"


def test_discovery_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = PluginManager()
    m.discover_plugins()
    assert {type(p).__name__ for p in m.plugins} >= set(EXPECTED_ORDER)
    assert not (tmp_path / "plugins").exists()


def test_failing_plugin_does_not_stop_pipeline():
    from core.plugin_interface import NeuroxPlugin

    class Boom(NeuroxPlugin):
        def __init__(self):
            super().__init__()
            self.name, self.order = "Boom", 5

        def execute(self, context):
            raise RuntimeError("boom")

    class After(NeuroxPlugin):
        def __init__(self):
            super().__init__()
            self.name, self.order = "After", 6

        def execute(self, context):
            return {"after_ran": True}

    m = PluginManager()
    m.register(After())
    m.register(Boom())
    ctx = m.execute_all({"user_input": "x", "trace": []})
    assert ctx["after_ran"] is True
    assert any(e.get("name") == "Boom" and "error" in e for e in ctx["trace"])


# -------------------------------------------------------------------------------------- security
@pytest.mark.parametrize("text", [
    "delete all files",
    "Please delete all my files now",
    "tüm dosyaları sil",
    "bilgisayarımdaki bütün dosyaları sil",
    "format the hard drive",
    "rm -rf /",
    "DROP TABLE users;",
    "shut down the computer",
])
def test_security_blocks_destructive(manager, text):
    upd = by_class(manager, "SecurityPlugin").execute(ctx_for(text)) or {}
    assert upd.get("blocked") is True, text


@pytest.mark.parametrize("text", [
    "what skills do I need",
    "What skills do I need to become a data scientist?",
    "Recommend a killer app for note taking",
    "remove duplicates from this list: 1, 2, 2, 3",
    "Python'da bir dosyayı nasıl okurum?",
    "hello",
    "Bugün hava nasıl?",
])
def test_security_allows_harmless(manager, text):
    upd = by_class(manager, "SecurityPlugin").execute(ctx_for(text)) or {}
    assert not upd.get("blocked"), text


@pytest.mark.parametrize("text", ["delete all files CONFIRM", "CONFIRM delete all files",
                                  "tüm dosyaları sil ONAYLA", "ONAYLA tüm dosyaları sil"])
def test_security_passes_with_confirm_token(manager, text):
    upd = by_class(manager, "SecurityPlugin").execute(ctx_for(text)) or {}
    assert not upd.get("blocked"), text


@pytest.mark.parametrize("text", ["delete all files, confirm", "delete all files UNCONFIRMED",
                                  "delete all files CONFIRMATION", "tüm dosyaları sil onayla"])
def test_security_requires_exact_token(manager, text):
    upd = by_class(manager, "SecurityPlugin").execute(ctx_for(text)) or {}
    assert upd.get("blocked") is True, text


# ---------------------------------------------------------------------------------- orchestrator
@pytest.mark.parametrize("text", ["hello", "merhaba", "Hi!", "teşekkürler"])
def test_orchestrator_fast_path_greeting_needs_no_llm(manager, text):
    llm = FakeLLM()
    upd = by_class(manager, "OrchestratorPlugin").execute(ctx_for(text, llm)) or {}
    assert llm.calls == []
    assert upd.get("active_agent") == "GENERALIST"
    assert upd.get("system_prompt")


@pytest.mark.parametrize("text,agent", [
    ("Write a python function that reverses a string", "CODER"),
    ("What is the latest news about SpaceX?", "RESEARCHER"),
])
def test_orchestrator_fast_path_keywords(manager, text, agent):
    llm = FakeLLM()
    upd = by_class(manager, "OrchestratorPlugin").execute(ctx_for(text, llm)) or {}
    assert llm.calls == []
    assert upd.get("active_agent") == agent
    if agent == "RESEARCHER":
        assert upd.get("search_query")


def test_orchestrator_uses_llm_when_ambiguous(manager):
    llm = FakeLLM(["PLANNER|SOLUTION"])
    upd = by_class(manager, "OrchestratorPlugin").execute(ctx_for("I want to open a small bakery next year", llm)) or {}
    assert upd.get("active_agent") in {"PLANNER", "GENERALIST", "RESEARCHER", "CODER"}
    if llm.calls:
        assert upd["active_agent"] == "PLANNER"


def test_orchestrator_skips_when_blocked(manager):
    llm = FakeLLM()
    assert not by_class(manager, "OrchestratorPlugin").execute(ctx_for("hello", llm, blocked=True))


# ------------------------------------------------------------------------------------ end-to-end
def _neurox(llm, **settings_kw):
    from core.pipeline import Neurox
    s = Settings(**{"computer_use_enabled": False, "auto_approve": False, **settings_kw})
    return Neurox(s, llm=llm)


def test_run_turn_end_to_end_returns_final_response():
    llm = FakeLLM(default=lambda m, t: "The answer is 4." if t else "GENERALIST|SOLUTION")
    nx = _neurox(llm)
    msgs = [{"role": "user", "content": "What is 2+2?"}]
    ctx = nx.run_turn("What is 2+2?", msgs)
    assert ctx.get("final_response") == "The answer is 4."
    res = ctx.get("agent_result")
    assert res is not None and res.pending_call is None
    # the user message reached the model
    final_call = llm.calls[-1]
    assert any(m["role"] == "user" and "What is 2+2?" in (m.get("content") or "") for m in final_call["messages"])
    assert final_call["messages"][0]["role"] == "system"
    assert msgs == [{"role": "user", "content": "What is 2+2?"}]  # caller's list not mutated
    # trace for the "thinking" panel
    assert any(e.get("type") == "plugin" for e in ctx["trace"])


def test_run_turn_with_tool_call_end_to_end():
    def responder(messages, tools):
        if not tools:
            return "GENERALIST|SOLUTION"
        if messages[-1]["role"] == "tool":
            return "It is " + messages[-1]["content"][:20]
        return ChatResult("", [ToolCall("current_time", {})])
    llm = FakeLLM(default=responder)
    nx = _neurox(llm)
    ctx = nx.run_turn("what time is it right now", [{"role": "user", "content": "what time is it right now"}])
    res = ctx["agent_result"]
    assert [e["name"] for e in res.trace if e["type"] == "tool_call"] == ["current_time"]
    assert ctx["final_response"].startswith("It is ")


def test_run_turn_dangerous_tool_waits_for_approval():
    def responder(messages, tools):
        if not tools:
            return "GENERALIST|SOLUTION"
        if messages[-1]["role"] == "tool":
            return "done"
        return ChatResult("", [ToolCall("run_python", {"code": "print(1)"})])
    nx = _neurox(FakeLLM(default=responder))
    ctx = nx.run_turn("compute it with code", [{"role": "user", "content": "compute it with code"}])
    assert ctx["agent_result"].pending_call is not None
    assert ctx["agent_result"].pending_call.name == "run_python"


def test_run_turn_blocked_request_has_no_tools():
    llm = FakeLLM(default="I paused this request for safety.")
    nx = _neurox(llm)
    ctx = nx.run_turn("delete all files", [{"role": "user", "content": "delete all files"}])
    assert ctx.get("blocked") is True
    assert ctx.get("final_response")
    assert all(not c["tools"] for c in llm.calls)   # the model got no tools at all


def test_run_turn_streams_tokens():
    tokens = []
    llm = FakeLLM(default=lambda m, t: "streaming works fine" if t else "GENERALIST|SOLUTION")
    ctx = _neurox(llm).run_turn("hello", [{"role": "user", "content": "hello"}], on_token=tokens.append)
    assert "".join(tokens) == ctx["final_response"] == "streaming works fine"


def test_build_context_without_dangerous_tools():
    nx = _neurox(FakeLLM())
    ctx = nx.build_context("x", [{"role": "user", "content": "x"}], allow_dangerous=False)
    reg = ctx["registry"]
    assert len(reg) > 0
    assert not any(reg.get(n).dangerous for n in reg.names())
    ctx = nx.build_context("x", [{"role": "user", "content": "x"}], allow_dangerous=True)
    assert "run_python" in ctx["registry"]
