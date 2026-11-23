"""core.agent: tool loop, max_steps, approval pause/resume, ToolsNotSupported fallback, events.

Written against the behaviour contract in the core/agent.py docstring.
"""
import copy

import pytest

from core.agent import AgentResult, resume_agent, run_agent
from core.llm import ChatResult, ToolCall, ToolsNotSupported
from core.tools import Tool, ToolOutput, ToolRegistry
from fakes import FakeLLM, tool_call_names, tool_messages


class Counter:
    def __init__(self):
        self.calls = []


@pytest.fixture
def side():
    return Counter()


@pytest.fixture
def registry(side):
    reg = ToolRegistry()
    reg.register(Tool("add", "Add two numbers",
                      {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                       "required": ["a", "b"]},
                      lambda a, b: side.calls.append(("add", a, b)) or a + b))
    reg.register(Tool("delete_everything", "Dangerous",
                      {"type": "object", "properties": {"path": {"type": "string"}}},
                      lambda path="/": side.calls.append(("delete", path)) or f"deleted {path}", dangerous=True))
    reg.register(Tool("snap", "Returns an image", {"type": "object", "properties": {}},
                      lambda: ToolOutput("screenshot taken", ["aW1n"])))
    return reg


def base_messages(text="hi"):
    return [{"role": "system", "content": "You are a test."}, {"role": "user", "content": text}]


def test_plain_answer_without_tools(registry):
    llm = FakeLLM(["Hello there"])
    res = run_agent(llm, base_messages(), registry)
    assert isinstance(res, AgentResult)
    assert res.final_text == "Hello there"
    assert res.pending_call is None
    assert len(llm.calls) == 1
    # the model was offered the registry's tools
    assert llm.calls[0]["tools"] == registry.schemas()
    # final assistant message is part of the conversation
    assert res.messages[-1]["role"] == "assistant" and res.messages[-1]["content"] == "Hello there"


def test_input_messages_not_lost(registry):
    msgs = base_messages("question")
    res = run_agent(FakeLLM(["a"]), msgs, registry)
    assert res.messages[:2] == base_messages("question")


def test_tool_loop_executes_and_feeds_result_back(registry, side):
    llm = FakeLLM([ToolCall("add", {"a": 2, "b": 3}, id="call_1"), "The answer is 5."])
    events = []
    res = run_agent(llm, base_messages("2+3?"), registry, on_event=events.append)
    assert res.final_text == "The answer is 5."
    assert side.calls == [("add", 2, 3)]
    assert len(llm.calls) == 2
    # assistant message with the tool call, then the tool result, were sent back to the model
    second = llm.calls[1]["messages"]
    asst = [m for m in second if m["role"] == "assistant" and m.get("tool_calls")]
    assert asst and tool_call_names(asst[-1]) == ["add"]
    tmsg = tool_messages(second)
    assert len(tmsg) == 1
    assert tmsg[0]["content"] == "5" and tmsg[0]["tool_call_id"] == "call_1" and tmsg[0].get("name") == "add"
    # trace + on_event
    types = [e["type"] for e in res.trace]
    assert "tool_call" in types and "tool_result" in types
    call_ev = next(e for e in res.trace if e["type"] == "tool_call")
    assert call_ev["name"] == "add" and call_ev["arguments"] == {"a": 2, "b": 3}
    result_ev = next(e for e in res.trace if e["type"] == "tool_result")
    assert result_ev["name"] == "add" and "5" in result_ev["text"]
    assert [e["type"] for e in events] == types
    assert res.steps >= 1


def test_multiple_tool_calls_in_one_turn(registry, side):
    llm = FakeLLM([[ToolCall("add", {"a": 1, "b": 1}), ToolCall("add", {"a": 2, "b": 2})], "done"])
    res = run_agent(llm, base_messages(), registry)
    assert res.final_text == "done"
    assert side.calls == [("add", 1, 1), ("add", 2, 2)]
    assert [m["content"] for m in tool_messages(res.messages)] == ["2", "4"]


def test_tool_images_are_collected(registry):
    res = run_agent(FakeLLM([ToolCall("snap", {}), "I see it"]), base_messages(), registry)
    assert "aW1n" in res.images
    tm = tool_messages(res.messages)[0]
    assert tm["content"] == "screenshot taken" and tm.get("images") == ["aW1n"]


def test_unknown_tool_and_bad_args_do_not_crash(registry):
    llm = FakeLLM([ToolCall("no_such_tool", {}), ToolCall("add", {"a": 1}), "recovered"])
    res = run_agent(llm, base_messages(), registry)
    assert res.final_text == "recovered"
    texts = [m["content"] for m in tool_messages(res.messages)]
    assert "unknown tool" in texts[0] and "Error" in texts[1]


def test_max_steps_terminates(registry, side):
    llm = FakeLLM(default=lambda m, t: ChatResult("", [ToolCall("add", {"a": 1, "b": 1})]) if t else ChatResult("gave up"))
    res = run_agent(llm, base_messages(), registry, max_steps=3)
    assert res.pending_call is None
    assert isinstance(res.final_text, str)
    assert len(side.calls) <= 3
    assert len(llm.calls) <= 4  # max_steps tool rounds (+ at most one final answer without tools)
    assert res.steps <= 4  # model calls, incl. the final no-tools call
    assert not llm.calls[-1]["tools"] and res.final_text == "gave up"


def test_dangerous_tool_pauses_for_approval(registry, side):
    llm = FakeLLM([ToolCall("delete_everything", {"path": "/tmp/x"}, id="call_d")])
    events = []
    res = run_agent(llm, base_messages("clean up"), registry, on_event=events.append)
    assert res.pending_call is not None
    assert res.pending_call.name == "delete_everything" and res.pending_call.arguments == {"path": "/tmp/x"}
    assert side.calls == []                     # NOT executed
    assert any(e["type"] == "approval_needed" for e in res.trace)
    assert any(e["type"] == "approval_needed" for e in events)
    copy.deepcopy(res)                          # must be storable in st.session_state


def test_resume_approved_executes_and_continues(registry, side):
    llm = FakeLLM([ToolCall("delete_everything", {"path": "/tmp/x"}, id="call_d"), "Deleted."])
    res = run_agent(llm, base_messages(), registry)
    assert res.pending_call is not None
    res2 = resume_agent(llm, res, registry, approved=True)
    assert side.calls == [("delete", "/tmp/x")]
    assert res2.pending_call is None and res2.final_text == "Deleted."
    tm = tool_messages(res2.messages)
    assert tm and tm[-1]["tool_call_id"] == "call_d" and "deleted /tmp/x" in tm[-1]["content"]
    # the model saw the tool result on the follow-up call
    assert any(m.get("role") == "tool" for m in llm.calls[-1]["messages"])


def test_resume_denied_does_not_execute(registry, side):
    llm = FakeLLM([ToolCall("delete_everything", {}, id="call_d"), "OK, I won't."])
    res = run_agent(llm, base_messages(), registry)
    res2 = resume_agent(llm, res, registry, approved=False)
    assert side.calls == []
    assert res2.final_text == "OK, I won't."
    tm = tool_messages(res2.messages)
    assert tm[-1]["tool_call_id"] == "call_d" and "denied" in tm[-1]["content"].lower()


def test_pending_remaining_calls_run_after_approval(registry, side):
    llm = FakeLLM([[ToolCall("delete_everything", {"path": "a"}), ToolCall("add", {"a": 1, "b": 2})], "all done"])
    res = run_agent(llm, base_messages(), registry)
    assert res.pending_call.name == "delete_everything"
    assert [c.name for c in res.pending_remaining] == ["add"]
    res2 = resume_agent(llm, res, registry, approved=True)
    assert ("delete", "a") in side.calls and ("add", 1, 2) in side.calls
    assert res2.final_text == "all done"
    # every tool call of the assistant turn got exactly one tool message
    assert len(tool_messages(res2.messages)) == 2


def test_auto_approve_runs_dangerous_tools(registry, side):
    llm = FakeLLM([ToolCall("delete_everything", {"path": "p"}), "done"])
    res = run_agent(llm, base_messages(), registry, auto_approve=True)
    assert res.pending_call is None and res.final_text == "done"
    assert side.calls == [("delete", "p")]


def test_tools_not_supported_falls_back_to_plain_chat(registry):
    llm = FakeLLM(["plain answer"], reject_tools=True)
    res = run_agent(llm, base_messages(), registry)
    assert res.final_text == "plain answer"
    assert not llm.calls[-1]["tools"]  # retried without tools


def test_tools_not_supported_raised_from_script(registry):
    llm = FakeLLM([ToolsNotSupported("no tools"), "fallback"])
    res = run_agent(llm, base_messages(), registry)
    assert res.final_text == "fallback"


def test_empty_registry_still_answers():
    llm = FakeLLM(["hi"])
    res = run_agent(llm, base_messages(), ToolRegistry())
    assert res.final_text == "hi"


def test_streaming_tokens(registry):
    tokens = []
    res = run_agent(FakeLLM(["streamed answer here"]), base_messages(), registry, on_token=tokens.append)
    assert "".join(tokens) == "streamed answer here" == res.final_text
