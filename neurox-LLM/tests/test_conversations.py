"""Saved conversations, projects (instructions + folder) and turbo mode."""
from __future__ import annotations

import json

import pytest

import core.config as config
from core.config import Settings
from core.llm import ChatResult, ToolCall
from tests.fakes import FakeLLM
from utils import conversations as convs


def _neurox(llm, **kw):
    from core.pipeline import Neurox
    return Neurox(Settings(**{"computer_use_enabled": False, "auto_approve": False, **kw}), llm=llm)


# --------------------------------------------------------------------------------------- conversations
def test_save_list_load_delete_conversation():
    conv = convs.new_conversation()
    convs.save_conversation(conv)                      # empty chats are not written
    assert convs.list_conversations() == []
    conv["messages"] += [{"role": "user", "content": "Merhaba, bana Python öğret lütfen"},
                         {"role": "assistant", "content": "Tabii!"}]
    convs.save_conversation(conv)
    listed = convs.list_conversations()
    assert [c["id"] for c in listed] == [conv["id"]]
    assert listed[0]["title"].startswith("Merhaba") and listed[0]["count"] == 2
    loaded = convs.load_conversation(conv["id"])
    assert loaded["messages"][1]["content"] == "Tabii!"
    assert convs.rename_conversation(conv["id"], "Python dersi")
    assert convs.list_conversations()[0]["title"] == "Python dersi"
    assert convs.delete_conversation(conv["id"]) and convs.list_conversations() == []


def test_newest_conversation_first():
    a, b = convs.new_conversation(), convs.new_conversation()
    a["messages"].append({"role": "user", "content": "first"})
    b["messages"].append({"role": "user", "content": "second"})
    convs.save_conversation(a)
    a_updated = convs.load_conversation(a["id"])["updated"]
    b["updated"] = a_updated
    convs.save_conversation(b)
    # same-second timestamps are possible; both must be listed
    assert {c["id"] for c in convs.list_conversations()} == {a["id"], b["id"]}


def test_title_is_truncated():
    assert convs.make_title("x" * 200).endswith("…")
    assert len(convs.make_title("x" * 200)) <= convs.TITLE_LEN + 1


def test_legacy_history_is_migrated_once():
    legacy = config.HISTORY_FILE
    legacy.write_text(json.dumps([{"role": "user", "content": "eski mesaj"},
                                  {"role": "assistant", "content": "eski cevap"}]), encoding="utf-8")
    conv_id = convs.migrate_legacy_history()
    assert conv_id and convs.load_conversation(conv_id)["messages"][0]["content"] == "eski mesaj"
    assert not legacy.exists()
    assert convs.migrate_legacy_history() is None      # second call: nothing left to migrate


# -------------------------------------------------------------------------------------------- projects
def test_project_crud_and_chat_filtering(tmp_path):
    folder = tmp_path / "myproj"
    folder.mkdir()
    proj = convs.save_project("Site", "Kısa cevap ver.", str(folder))
    assert convs.get_project(proj["id"])["folder"] == str(folder.resolve())

    in_proj = convs.new_conversation(project_id=proj["id"])
    in_proj["messages"].append({"role": "user", "content": "project chat"})
    convs.save_conversation(in_proj)
    loose = convs.new_conversation()
    loose["messages"].append({"role": "user", "content": "loose chat"})
    convs.save_conversation(loose)

    assert [c["id"] for c in convs.list_conversations(proj["id"])] == [in_proj["id"]]
    assert [c["id"] for c in convs.list_conversations(None)] == [loose["id"]]
    assert len(convs.list_conversations(all_projects=True)) == 2

    convs.save_project("Site 2", "", "", project_id=proj["id"])
    assert convs.get_project(proj["id"])["name"] == "Site 2"

    assert convs.delete_project(proj["id"])             # chats are kept, moved out of the project
    assert convs.get_project(proj["id"]) is None
    assert {c["id"] for c in convs.list_conversations(None)} == {in_proj["id"], loose["id"]}


def test_project_folder_must_exist(tmp_path):
    with pytest.raises(ValueError):
        convs.save_project("X", folder=str(tmp_path / "missing"))
    with pytest.raises(ValueError):
        convs.save_project("X", folder="relative/path")


def test_project_instructions_reach_the_model_and_file_tools_use_project_folder(tmp_path):
    folder = tmp_path / "code"
    folder.mkdir()
    (folder / "main.py").write_text("print('hi')", encoding="utf-8")
    proj = convs.save_project("Kod", "Always answer in haiku.", str(folder))

    def responder(messages, tools):
        if not tools:
            return "GENERALIST|SOLUTION"
        if messages[-1]["role"] == "tool":
            return "Files: " + messages[-1]["content"]
        return ChatResult("", [ToolCall("list_dir", {"path": "."})])

    llm = FakeLLM(default=responder)
    nx = _neurox(llm)
    nx.set_project(proj)
    try:
        ctx = nx.run_turn("show my files", [{"role": "user", "content": "show my files"}])
        system = llm.calls[-1]["messages"][0]["content"]
        assert "[PROJECT: Kod]" in system and "Always answer in haiku." in system
        assert "main.py" in ctx["final_response"]
    finally:
        nx.set_project(None)
    import tools.builtin as builtin
    assert builtin.WORKSPACE_DIR is None                # back to the default sandbox


# ----------------------------------------------------------------------------------------------- turbo
def test_turbo_mode_runs_dangerous_tools_without_approval():
    def responder(messages, tools):
        if not tools:
            return "CODER|SOLUTION"
        if messages[-1]["role"] == "tool":
            return "Done: " + messages[-1]["content"].strip()
        return ChatResult("", [ToolCall("run_python", {"code": "print(6*7)"})])

    normal = _neurox(FakeLLM(default=responder)).run_turn("run code", [{"role": "user", "content": "run code"}])
    assert normal["agent_result"].pending_call is not None   # asks for approval

    turbo = _neurox(FakeLLM(default=responder), auto_approve=True).run_turn(
        "run code", [{"role": "user", "content": "run code"}])
    assert turbo["agent_result"].pending_call is None
    assert "42" in turbo["final_response"]
