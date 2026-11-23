"""Agent loop with native tool calling and human-in-the-loop approval.

CONTRACT (used by app.py, server.py, chat_plugin.py and tests — keep signatures stable):

    result = run_agent(llm, messages, registry, on_token=None, max_steps=8,
                       auto_approve=False, on_event=None)
    if result.pending_call:          # a dangerous tool wants to run
        ...ask the user...
        result = resume_agent(llm, result, registry, approved=True|False, on_token=..., auto_approve=..., on_event=...)

- ``messages`` is the full canonical message list (system + history + user), see core/llm.py.
- The loop calls ``llm.chat(messages, tools=registry.schemas(), ...)``; for every returned ToolCall
  it appends the assistant message (with tool_calls) and one ``{"role":"tool", ...}`` message per result,
  then calls the model again, up to ``max_steps`` times. When the model answers without tool calls,
  that text is ``final_text``.
- A tool with ``dangerous=True`` is NOT executed unless ``auto_approve``; instead the loop stops and returns
  ``AgentResult(pending_call=<ToolCall>, ...)``. ``resume_agent`` executes (approved) or rejects it
  (tool message "User denied this action.") and continues the loop.
- If the model raises ``ToolsNotSupported`` the loop falls back to a plain chat without tools.
- ``on_event(event: dict)`` is called for every step, e.g.
  {"type": "tool_call", "name": ..., "arguments": {...}} / {"type": "tool_result", "name": ..., "text": ..., "images": [...]}
  / {"type": "approval_needed", ...} / {"type": "error", "text": ...}. The same dicts are collected in ``result.trace``.
- ``AgentResult`` must be serialisable enough to be kept in ``st.session_state`` between Streamlit reruns.

Implementation notes:
- Additional event types: {"type": "info", "text"} (tools-not-supported fallback, max_steps reached),
  {"type": "approved"|"denied", "id", "name"} (from resume_agent). ``tool_call``/``tool_result``/
  ``approval_needed`` events also carry the call ``id``; ``approval_needed`` carries ``arguments`` and ``description``.
- Assistant messages store ``tool_calls`` as plain dicts (``ToolCall.to_dict()``) so the history is JSON
  serialisable; both LLM clients accept dicts or ToolCall objects.
- An ``LLMError`` never propagates: it is reported as an "error" event and ``final_text`` becomes "⚠️ LLM error: ...".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from core.llm import LLMClient, LLMError, ToolCall, ToolsNotSupported
from core.tools import ToolRegistry

log = logging.getLogger(__name__)

DENIED_TEXT = "User denied this action."
_MAX_EVENT_TEXT = 4000  # tool output kept in trace events (the full text goes to the model)


@dataclass
class AgentResult:
    messages: list[dict]                        # full conversation including tool messages
    final_text: str = ""
    pending_call: ToolCall | None = None        # set when a dangerous tool waits for approval
    pending_remaining: list[ToolCall] = field(default_factory=list)  # other calls from the same model turn, not yet run
    trace: list[dict] = field(default_factory=list)
    steps: int = 0
    images: list[str] = field(default_factory=list)  # images produced by tools (e.g. screenshots) for the UI


# ------------------------------------------------------------------------------------------ helpers
def _emit(result: AgentResult, on_event: Callable[[dict], None] | None, event: dict) -> None:
    result.trace.append(event)
    if on_event:
        try:
            on_event(event)
        except Exception:  # a broken UI callback must not kill the agent
            log.exception("on_event callback failed")


def _copy_messages(messages: list[dict]) -> list[dict]:
    """Shallow-copy each message (and its list fields) so the caller's history is never mutated."""
    out = []
    for m in messages or []:
        c = dict(m)
        for k in ("images", "tool_calls"):
            if isinstance(c.get(k), list):
                c[k] = list(c[k])
        out.append(c)
    return out


def _as_call(c) -> ToolCall:
    return c if isinstance(c, ToolCall) else ToolCall.from_dict(c)


def _call_model(llm: LLMClient, result: AgentResult, registry: ToolRegistry | None,
                on_token, on_event) -> tuple[str, list[ToolCall]] | None:
    """One model call. Returns (content, tool_calls), or None after a fatal LLMError (final_text is then set).
    Falls back to a plain call without tools if the model does not support tool calling."""
    schemas = registry.schemas() if (registry is not None and len(registry)) else None
    stream = on_token is not None
    try:
        res = llm.chat(result.messages, tools=schemas, stream=stream, on_token=on_token)
        return res.content or "", list(res.tool_calls or [])
    except ToolsNotSupported as e:
        _emit(result, on_event, {"type": "info",
                                 "text": f"Model does not support tool calling; answering without tools. ({e})"})
        try:
            res = llm.chat(result.messages, tools=None, stream=stream, on_token=on_token)
            return res.content or "", []
        except LLMError as e2:
            err = e2
    except LLMError as e:
        err = e
    msg = f"LLM error: {err}"
    _emit(result, on_event, {"type": "error", "text": msg})
    result.final_text = f"⚠️ {msg}"
    return None


def _execute(call: ToolCall, result: AgentResult, registry: ToolRegistry, on_event) -> None:
    _emit(result, on_event, {"type": "tool_call", "id": call.id, "name": call.name,
                             "arguments": dict(call.arguments or {})})
    out = registry.call(call.name, call.arguments)
    msg = {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": out.text}
    if out.images:
        msg["images"] = list(out.images)
        result.images.extend(out.images)
    result.messages.append(msg)
    text = out.text if len(out.text) <= _MAX_EVENT_TEXT else out.text[:_MAX_EVENT_TEXT] + "\n...[truncated]"
    _emit(result, on_event, {"type": "tool_result", "id": call.id, "name": call.name, "text": text,
                             "images": list(out.images)})


def _run_calls(calls: list[ToolCall], result: AgentResult, registry: ToolRegistry,
               auto_approve: bool, on_event) -> bool:
    """Run tool calls in order. Returns False if the loop paused on a dangerous tool."""
    for i, call in enumerate(calls):
        tool = registry.get(call.name)
        if tool is not None and tool.dangerous and not auto_approve:
            result.pending_call = call
            result.pending_remaining = list(calls[i + 1:])
            _emit(result, on_event, {"type": "approval_needed", "id": call.id, "name": call.name,
                                     "arguments": dict(call.arguments or {}), "description": tool.description})
            return False
        _execute(call, result, registry, on_event)
    return True


def _loop(llm: LLMClient, result: AgentResult, registry: ToolRegistry, on_token, max_steps: int,
          auto_approve: bool, on_event) -> AgentResult:
    max_steps = max(1, int(max_steps or 1))
    while result.steps < max_steps:
        out = _call_model(llm, result, registry, on_token, on_event)
        result.steps += 1
        if out is None:
            return result
        content, calls = out
        if not calls:
            result.messages.append({"role": "assistant", "content": content})
            result.final_text = content
            return result
        result.messages.append({"role": "assistant", "content": content,
                                "tool_calls": [c.to_dict() for c in calls]})
        if not _run_calls(calls, result, registry, auto_approve, on_event):
            return result  # paused for approval
    # Step budget exhausted while the model still wanted tools: one last call without tools.
    _emit(result, on_event, {"type": "info", "text": f"Reached max_steps={max_steps}; asking for a final answer."})
    out = _call_model(llm, result, None, on_token, on_event)
    result.steps += 1
    if out is not None:
        result.messages.append({"role": "assistant", "content": out[0]})
        result.final_text = out[0]
    return result


# ------------------------------------------------------------------------------------------ public API
def run_agent(llm: LLMClient, messages: list[dict], registry: ToolRegistry,
              on_token: Callable[[str], None] | None = None, max_steps: int = 8,
              auto_approve: bool = False, on_event: Callable[[dict], None] | None = None) -> AgentResult:
    """Run the tool-calling loop. ``messages`` is not mutated; ``result.messages`` is a new list."""
    registry = registry if registry is not None else ToolRegistry()
    result = AgentResult(messages=_copy_messages(messages))
    return _loop(llm, result, registry, on_token, max_steps, auto_approve, on_event)


def resume_agent(llm: LLMClient, result: AgentResult, registry: ToolRegistry, approved: bool,
                 on_token: Callable[[str], None] | None = None, auto_approve: bool = False,
                 on_event: Callable[[dict], None] | None = None, max_steps: int = 8) -> AgentResult:
    """Continue a paused run: execute (``approved``) or reject the pending dangerous call, run the other
    calls of the same model turn, then keep looping. Returns a new AgentResult; trace, images and steps
    are cumulative (``max_steps`` counts model calls across run + resumes)."""
    registry = registry if registry is not None else ToolRegistry()
    new = AgentResult(messages=_copy_messages(result.messages), final_text=result.final_text,
                      trace=list(result.trace), steps=result.steps, images=list(result.images))
    if result.pending_call is None:
        return new
    call = _as_call(result.pending_call)
    remaining = [_as_call(c) for c in result.pending_remaining]

    if approved:
        _emit(new, on_event, {"type": "approved", "id": call.id, "name": call.name})
        _execute(call, new, registry, on_event)
    else:
        _emit(new, on_event, {"type": "denied", "id": call.id, "name": call.name})
        new.messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": DENIED_TEXT})
    new.final_text = ""
    if not _run_calls(remaining, new, registry, auto_approve, on_event):
        return new
    return _loop(llm, new, registry, on_token, max_steps, auto_approve, on_event)
