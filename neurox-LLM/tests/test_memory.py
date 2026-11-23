"""Vector memory, remember/recall tools, memory plugin, rules, XP and context compression."""
import json

import pytest

from core import config
from core.config import Settings
from core.llm import LLMError
from fakes import FakeLLM

vs = pytest.importorskip("utils.vector_store")
VectorStore = vs.VectorStore


# ------------------------------------------------------------------------------------- VectorStore
def test_add_search_delete(tmp_path):
    llm = FakeLLM()
    store = VectorStore(llm, tmp_path / "mem.json")
    a = store.add("My favourite colour is blue")
    b = store.add("The cat is called Pamuk")
    assert a and b and a != b
    hits = store.search("favourite colour blue", top_k=3, threshold=0.3)
    assert hits and hits[0]["id"] == a and "blue" in hits[0]["text"]
    assert all(set(h) >= {"id", "text", "score"} for h in hits)
    assert store.delete(a) is True
    assert store.delete(a) is False
    assert all(h["id"] != a for h in store.search("favourite colour blue", threshold=0.0))
    # persisted to disk, readable by a second instance
    again = VectorStore(llm, tmp_path / "mem.json")
    assert [e["text"] for e in again.list()] == ["The cat is called Pamuk"]


def test_duplicate_add_returns_same_id(tmp_path):
    store = VectorStore(FakeLLM(), tmp_path / "m.json")
    assert store.add("same fact") == store.add("  Same   FACT ")
    assert len(store) == 1


def test_empty_and_embed_failure_do_not_raise(tmp_path):
    llm = FakeLLM()
    store = VectorStore(llm, tmp_path / "m.json")
    assert store.add("") is None
    assert store.search("") == []

    def broken(texts, model=None):
        raise LLMError("embed model missing")
    llm.embed = broken
    assert store.add("something") is None
    assert store.search("something") == []


def test_mismatched_dimension_entries_ignored(tmp_path):
    path = tmp_path / "mem.json"
    legacy = [
        {"text": "User: old phi3 conversation | Assistant: hi", "embedding": [0.1, 0.2, 0.3]},            # legacy
        {"text": "other model fact", "embedding": [0.5] * 8, "model": "some-other-embedder", "dim": 8},
        {"text": "same model, wrong dim", "embedding": [0.5] * 8, "model": "fake-embed", "dim": 8},
    ]
    path.write_text(json.dumps(legacy), encoding="utf-8")
    llm = FakeLLM(dim=64)
    store = VectorStore(llm, path)
    assert len(store) == 3                      # kept on disk
    new_id = store.add("old phi3 conversation fact")
    hits = store.search("old phi3 conversation", top_k=10, threshold=-1.0)  # accept any score
    assert [h["id"] for h in hits] == [new_id]  # only current-model, current-dim entries are searchable
    stats = store.stats()
    assert stats["total"] == 4  # note: stats()["searchable"] only checks the model name, not the dimension


def test_reindex_reembeds_stale_entries(tmp_path):
    path = tmp_path / "mem.json"
    path.write_text(json.dumps([{"text": "stale entry", "embedding": [1.0, 0.0]}]), encoding="utf-8")
    store = VectorStore(FakeLLM(), path)
    assert store.search("stale entry") == []
    assert store.reindex() == 1
    assert store.search("stale entry")[0]["text"] == "stale entry"


def test_corrupt_memory_file_is_tolerated(tmp_path):
    path = tmp_path / "mem.json"
    path.write_text("{not json", encoding="utf-8")
    store = VectorStore(FakeLLM(), path)
    assert len(store) == 0 and store.search("x") == []


def test_default_path_is_in_test_data_dir(isolated_data_dir):
    store = VectorStore(FakeLLM())
    assert store.path.resolve().is_relative_to(isolated_data_dir)


# ------------------------------------------------------------------------------- remember / recall
def test_remember_and_recall_tools(settings):
    mt = pytest.importorskip("tools.memory_tools")
    from core.tools import ToolRegistry
    llm = FakeLLM()
    reg = ToolRegistry()
    for t in mt.get_tools(settings, llm):
        reg.register(t)
    assert {"remember", "recall"} <= set(reg.names())
    assert not reg.get("remember").dangerous and not reg.get("recall").dangerous
    out = reg.call("remember", {"text": "The user's dog is named Karabaş"}).text
    assert "error" not in out.lower()
    out = reg.call("recall", {"query": "dog named Karabaş"}).text
    assert "Karabaş" in out
    assert config.MEMORY_FILE.exists()


# ---------------------------------------------------------------------------------- memory plugin
def test_memory_plugin_stores_explicit_fact_and_recalls(settings):
    mp = pytest.importorskip("plugins.memory_plugin")
    plugin = mp.MemoryPlugin()
    llm = FakeLLM()
    ctx = {"user_input": "Remember that my birthday is on 12 May", "messages": [], "settings": settings,
           "llm": llm, "trace": []}
    upd = plugin.execute(ctx) or {}
    assert upd.get("memory_saved") is True
    ctx2 = dict(ctx, user_input="when is my birthday 12 May", trace=[])
    upd2 = plugin.execute(ctx2) or {}
    assert "birthday" in upd2.get("long_term_memory", "")


def test_memory_plugin_skips_when_blocked(settings):
    mp = pytest.importorskip("plugins.memory_plugin")
    llm = FakeLLM()
    assert not mp.MemoryPlugin().execute({"user_input": "remember that x is y", "blocked": True, "llm": llm,
                                          "settings": settings, "messages": [], "trace": []})
    assert llm.embed_calls == []


# -------------------------------------------------------------------------------------------- rules
@pytest.fixture
def misc():
    return pytest.importorskip("plugins.misc_plugins")


def test_rules_persistence(misc):
    assert misc.load_rules() == []
    r1 = misc.add_rule("Never call me Sir")
    r2 = misc.add_rule("Always answer in Turkish")
    assert r1["id"] != r2["id"]
    assert misc.add_rule("never call me sir")["id"] == r1["id"]   # no duplicates
    assert misc.add_rule("   ") is None
    assert config.RULES_FILE.exists()
    assert [r["text"] for r in misc.load_rules()] == ["Never call me Sir", "Always answer in Turkish"]
    assert misc.delete_rule(r1["id"]) is True
    assert misc.delete_rule(r1["id"]) is False
    assert [r["text"] for r in misc.load_rules()] == ["Always answer in Turkish"]


def test_rules_explicit_path(misc, tmp_path):
    p = tmp_path / "custom_rules.json"
    misc.add_rule("Be brief", path=p)
    assert misc.load_rules(path=p)[0]["text"] == "Be brief"
    assert misc.load_rules() == []


def test_rule_plugin_learns_and_injects(misc, settings, fake_llm):
    plugin = misc.RulePlugin()
    ctx = {"user_input": "From now on, don't call me Sir", "messages": [], "settings": settings,
           "llm": fake_llm, "trace": []}
    upd = plugin.execute(ctx) or {}
    assert "Sir" in upd.get("user_rules", "")
    assert len(misc.load_rules()) == 1
    upd = plugin.execute(dict(ctx, user_input="what's the weather?", trace=[])) or {}
    assert "Sir" in upd.get("user_rules", "")
    assert len(misc.load_rules()) == 1     # a normal question is not a rule


def test_rule_plugin_turkish(misc, fake_llm):
    plugin = misc.RulePlugin()
    s = Settings(language="tr")
    plugin.execute({"user_input": "Bundan sonra bana efendim deme", "messages": [], "settings": s,
                    "llm": fake_llm, "trace": []})
    assert len(misc.load_rules()) == 1


# ----------------------------------------------------------------------------------------------- XP
def _gam_ctx(settings, text="hello"):
    return {"user_input": text, "messages": [], "settings": settings, "trace": [], "final_response": "hi"}


def test_xp_level_up_message_only_on_level_change(misc, settings):
    plugin = misc.GamificationPlugin()
    start = misc.get_xp_state()
    assert start["level"] == 1 and start["xp"] == 0
    msgs = []
    for _ in range(25):
        upd = plugin.execute(_gam_ctx(settings)) or {}
        msgs.append(upd.get("gamification_msg"))
    end = misc.get_xp_state()
    assert end["xp"] > start["xp"]
    levels_gained = end["level"] - start["level"]
    assert levels_gained >= 1
    # exactly one message per level gained, never on ordinary turns
    assert sum(1 for m in msgs if m) == levels_gained
    assert config.XP_FILE.exists()


def test_xp_not_awarded_when_blocked(misc, settings):
    plugin = misc.GamificationPlugin()
    plugin.execute(dict(_gam_ctx(settings), blocked=True))
    assert misc.get_xp_state()["xp"] == 0


# -------------------------------------------------------------------------------------- compression
def test_compression_shortens_long_history(settings):
    cc = pytest.importorskip("plugins.context_compression_plugin")
    plugin = cc.ContextCompressionPlugin()
    s = Settings(history_window=6)
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(30):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"message number {i}"})
    original = [dict(m) for m in msgs]
    llm = FakeLLM(default="SUMMARY: user counted messages")
    upd = plugin.execute({"user_input": "message number 29", "messages": msgs, "settings": s, "llm": llm,
                          "trace": []}) or {}
    new = upd.get("messages")
    assert new is not None and len(new) < len(msgs)
    assert new[0] == {"role": "system", "content": "sys"}                       # system prompt kept
    assert any("SUMMARY" in (m.get("content") or "") for m in new)               # summary injected
    assert new[-1]["content"] == "message number 29"                              # newest kept verbatim
    assert msgs == original                                                       # input not mutated


def test_compression_noop_for_short_history(settings, fake_llm):
    cc = pytest.importorskip("plugins.context_compression_plugin")
    msgs = [{"role": "user", "content": "hi"}]
    upd = cc.ContextCompressionPlugin().execute({"user_input": "hi", "messages": msgs, "settings": settings,
                                                 "llm": fake_llm, "trace": []})
    assert not upd or upd.get("messages", msgs) == msgs
    assert fake_llm.calls == []


def test_compression_llm_failure_falls_back_to_truncation():
    cc = pytest.importorskip("plugins.context_compression_plugin")
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(40)]
    llm = FakeLLM(default=LLMError("server down"))
    upd = cc.ContextCompressionPlugin().execute({"user_input": "m39", "messages": msgs,
                                                 "settings": Settings(history_window=8), "llm": llm, "trace": []})
    assert upd and len(upd["messages"]) < len(msgs) and upd["messages"][-1]["content"] == "m39"


def test_compression_tail_does_not_start_with_orphan_tool_message():
    cc = pytest.importorskip("plugins.context_compression_plugin")
    msgs = []
    for i in range(12):
        msgs += [{"role": "user", "content": f"q{i}"},
                 {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "name": "calculator",
                                                                      "arguments": {}}]},
                 {"role": "tool", "tool_call_id": f"c{i}", "name": "calculator", "content": "4"},
                 {"role": "assistant", "content": f"a{i}"}]
    upd = cc.ContextCompressionPlugin().execute({"user_input": "q11", "messages": msgs,
                                                 "settings": Settings(history_window=7),
                                                 "llm": FakeLLM(default="sum"), "trace": []})
    body = [m for m in upd["messages"] if not (m["role"] == "system")]
    assert body[0]["role"] != "tool"


# ------------------------------------------------------------------------------------- chat history
def test_history_roundtrip_with_toolcall_objects():
    hm = pytest.importorskip("utils.history_manager")
    from core.llm import ToolCall
    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [ToolCall("calculator", {"expression": "1+1"}, id="c1")]},
            {"role": "tool", "tool_call_id": "c1", "name": "calculator", "content": "2"}]
    assert hm.save_history(msgs) is True
    assert config.HISTORY_FILE.exists()
    loaded = hm.load_history()
    assert [m["role"] for m in loaded] == ["user", "assistant", "tool"]
    assert loaded[1]["tool_calls"][0]["name"] == "calculator"
    hm.clear_history()
    assert hm.load_history() == []
