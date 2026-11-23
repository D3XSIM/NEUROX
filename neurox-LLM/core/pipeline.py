"""One shared entry point for a chat turn, used by app.py and server.py."""
from __future__ import annotations

from typing import Callable

from core.config import Settings
from core.llm import LLMClient, get_client
from core.plugin_manager import PluginManager
from tools import build_registry


class Neurox:
    def __init__(self, settings: Settings | None = None, llm: LLMClient | None = None,
                 manager: PluginManager | None = None):
        self.settings = settings or Settings.load()
        self.llm = llm or get_client(self.settings)
        if manager is None:
            manager = PluginManager()
            manager.discover_plugins()
        self.manager = manager

    def reconfigure(self, settings: Settings, llm: LLMClient | None = None) -> None:
        self.settings = settings
        self.llm = llm or get_client(settings)

    def build_context(self, user_input: str, messages: list[dict], images: list[str] | None = None,
                      on_token: Callable[[str], None] | None = None,
                      on_event: Callable[[dict], None] | None = None,
                      allow_dangerous: bool = True) -> dict:
        registry = build_registry(self.settings, self.llm)
        if not allow_dangerous:  # e.g. the HTTP API, which has no approval UI
            for name in list(registry.names()):
                if registry.get(name).dangerous:
                    registry.unregister(name)
        return {
            "user_input": user_input,
            "messages": messages,
            "images": images or [],
            "settings": self.settings,
            "llm": self.llm,
            "registry": registry,
            "trace": [],
            "on_token_callback": on_token,
            "on_event": on_event,
        }

    def run_turn(self, user_input: str, messages: list[dict], **kw) -> dict:
        """``messages`` must already contain the current user message. Returns the final context;
        the answer is ``ctx['final_response']`` and ``ctx.get('agent_result')`` may have a ``pending_call``."""
        return self.manager.execute_all(self.build_context(user_input, messages, **kw))

    def submit_feedback(self, text: str) -> dict:
        ctx = self.build_context("", [])
        ctx["feedback_text"] = text
        return self.manager.execute_all(ctx)
