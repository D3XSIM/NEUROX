"""Plugin base class and the shared *context* contract.

Plugins run in ascending ``order``. Each receives the shared ``context`` dict and returns
a dict of updates (or None).

Context keys (inputs, set by app.py / server.py):
    user_input: str                 current user message (already appended to ``messages``)
    messages: list[dict]            canonical chat history incl. the current user message (see core/llm.py)
    images: list[str]               base64 images attached to the current message (optional)
    settings: core.config.Settings
    llm: core.llm.LLMClient         ALWAYS use this, never import ollama directly in plugins
    registry: core.tools.ToolRegistry   tools available this turn (plugins may register more)
    trace: list[dict]               append {"type": "plugin", "name": ..., "info": ...} for the "thinking" panel
    on_token_callback: callable     streaming callback (optional)
    on_event: callable              agent event callback (optional)
    feedback_text: str              set instead of user_input when the user submits a correction

Context keys (produced by plugins):
    blocked: bool                   SecurityPlugin stopped the request; later plugins (except Chat) must skip
    active_agent, latent_need, system_prompt, search_query, search_results, tool_output,
    creative_angles, simulation_results, dynamic_persona, long_term_memory, user_rules,
    copilot_msg, inner_monologue, gamification_msg, final_response, agent_result (core.agent.AgentResult)
"""
from abc import ABC, abstractmethod


class NeuroxPlugin(ABC):
    def __init__(self):
        self.name = "Unknown Plugin"
        self.description = "No description provided."
        self.version = "0.0.1"
        self.order = 50  # Default execution order (Lower runs first)
        self.runs_when_blocked = False  # True only for plugins that must run even after SecurityPlugin blocks

    @abstractmethod
    def execute(self, context: dict) -> dict | None:
        """Execute the plugin logic with the given context. Returns a dict of updates to the context."""

    def on_load(self):
        """Called when the plugin is loaded."""
