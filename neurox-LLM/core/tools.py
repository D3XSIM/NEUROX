"""Tool registry used for native LLM tool calling.

A Tool is a plain Python function plus a JSON-schema description. Tools marked
``dangerous=True`` (code execution, file writes, mouse/keyboard control...) must be
approved by the user before they run, unless ``Settings.auto_approve`` is on.

Tool functions return either a ``str`` or a ``ToolOutput`` (text + optional images,
e.g. screenshots for vision models).
"""
from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)


@dataclass
class ToolOutput:
    text: str
    images: list[str] = field(default_factory=list)   # base64-encoded PNG/JPEG (no data: prefix)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict                                   # JSON schema: {"type":"object","properties":{...},"required":[...]}
    func: Callable[..., Any]
    dangerous: bool = False
    category: str = "general"                          # "general" | "web" | "files" | "code" | "memory" | "computer"

    def schema(self) -> dict:
        """OpenAI / Ollama compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def filtered(self, allowed: list[str] | None) -> "ToolRegistry":
        """New registry containing only tools whose name is in ``allowed`` (None = all)."""
        reg = ToolRegistry()
        for t in self._tools.values():
            if allowed is None or t.name in allowed:
                reg.register(t)
        return reg

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def call(self, name: str, args: dict | str | None) -> ToolOutput:
        """Run a tool, never raising: errors are returned as text for the model."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolOutput(f"Error: unknown tool '{name}'. Available: {', '.join(self._tools)}")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                return ToolOutput(f"Error: arguments for '{name}' are not valid JSON: {args!r}")
        args = args or {}
        try:
            inspect.signature(tool.func).bind(**args)
        except TypeError as e:
            return ToolOutput(f"Error: bad arguments for '{name}': {e}")
        except ValueError:  # builtins without a signature: just try the call
            pass
        try:
            result = tool.func(**args)
        except Exception as e:  # tool failure is reported to the model
            log.exception("Tool %s failed", name)
            return ToolOutput(f"Error while running '{name}': {type(e).__name__}: {e}")
        if isinstance(result, ToolOutput):
            return result
        return ToolOutput("" if result is None else str(result))
