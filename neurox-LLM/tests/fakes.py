"""Test doubles: a scripted, offline LLM client."""
from __future__ import annotations

import copy
import hashlib
import math
import re
from collections import deque
from typing import Callable

from core.llm import ChatResult, LLMClient, ToolCall, ToolsNotSupported

Responder = Callable[[list, "list | None"], "ChatResult | str"]


class FakeLLM(LLMClient):
    """Offline ``LLMClient``.

    * ``script``: queue of replies consumed in order. Each item is a ``ChatResult``, a ``str`` (plain text
      answer), a list of ``ToolCall`` (tool-call answer), an ``Exception`` instance (raised), or a
      callable ``(messages, tools) -> reply``.
    * ``default``: used when the queue is empty (same item types; default ``"OK"``).
    * ``reject_tools``: raise ``ToolsNotSupported`` whenever tools are passed (like a model without tool support).
    * Every call is recorded in ``self.calls`` (deep-copied messages, tools, stream flag ...).
    * ``embed`` is a deterministic hashed bag-of-words (``dim`` floats, L2 normalised).
    """

    def __init__(self, script=None, default="OK", reject_tools: bool = False, dim: int = 64,
                 models=("fake-model", "nomic-embed-text")):
        super().__init__("http://fake", "fake-model", "fake-embed", 0.0, "")
        self.script = deque(script or [])
        self.default = default
        self.reject_tools = reject_tools
        self.dim = dim
        self.models = list(models)
        self.calls: list[dict] = []
        self.embed_calls: list[list[str]] = []

    # -- helpers ---------------------------------------------------------------------------------
    def push(self, *items) -> "FakeLLM":
        self.script.extend(items)
        return self

    @staticmethod
    def _coerce(item) -> ChatResult:
        if isinstance(item, ChatResult):
            return copy.deepcopy(item)
        if isinstance(item, ToolCall):
            return ChatResult("", [copy.deepcopy(item)])
        if isinstance(item, list):
            return ChatResult("", [copy.deepcopy(tc) for tc in item])
        return ChatResult("" if item is None else str(item), [])

    @property
    def tool_calls_made(self) -> list[dict]:
        return [c for c in self.calls if c["tools"]]

    # -- LLMClient API ---------------------------------------------------------------------------
    def chat(self, messages, tools=None, stream=False, on_token=None, model=None, temperature=None):
        self.calls.append({"messages": copy.deepcopy(messages), "tools": copy.deepcopy(tools), "stream": stream,
                           "on_token": on_token is not None, "model": model, "temperature": temperature})
        if tools and self.reject_tools:
            raise ToolsNotSupported("fake-model does not support tools")
        item = self.script.popleft() if self.script else self.default
        if callable(item) and not isinstance(item, (ChatResult, ToolCall)):
            item = item(messages, tools)
        if isinstance(item, BaseException):
            raise item
        result = self._coerce(item)
        if stream and on_token and result.content:
            for tok in re.findall(r"\S+\s*|\s+", result.content):
                on_token(tok)
        return result

    def embed(self, texts, model=None):
        self.embed_calls.append(list(texts))
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"\w+", (t or "").lower()):
                h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
                v[h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out

    def list_models(self):
        return list(self.models)


def tool_call_names(msg: dict) -> list[str]:
    """Names of the tool calls of an assistant message (works for ToolCall objects and dicts)."""
    out = []
    for tc in msg.get("tool_calls") or []:
        if isinstance(tc, ToolCall):
            out.append(tc.name)
        elif "function" in tc:
            out.append(tc["function"]["name"])
        else:
            out.append(tc["name"])
    return out


def tool_messages(messages: list[dict]) -> list[dict]:
    return [m for m in messages if m.get("role") == "tool"]
