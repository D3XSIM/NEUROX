"""Final step: builds the system prompt from everything the other plugins produced and runs the
tool-calling agent (core.agent.run_agent). Runs even when SecurityPlugin blocked the turn, but then
with NO tools, so the model can only explain why the request was paused."""
from __future__ import annotations

import datetime as _dt

from core.agent import AgentResult, run_agent
from core.plugin_interface import NeuroxPlugin
from core.tools import ToolRegistry

DEFAULT_SYSTEM_PROMPT = (
    "You are Neurox, a helpful, friendly and intelligent local AI assistant. You can use tools whenever they help. "
    "Always answer in the same language the user writes in."
)


COMPUTER_USE_MIN_STEPS = 25  # screenshot -> act -> verify loops need many model calls


def _has_computer_tools(registry: ToolRegistry | None) -> bool:
    return bool(registry) and any(registry.get(n).category == "computer" for n in registry.names())


def _first_sentence(text: str, limit: int = 140) -> str:
    s = (text or "").strip().split(". ")[0].strip()
    return s if len(s) <= limit else s[:limit - 3] + "..."


def build_system_prompt(context: dict, registry: ToolRegistry | None) -> str:
    parts = [context.get("system_prompt") or DEFAULT_SYSTEM_PROMPT]
    if context.get("dynamic_persona"):
        parts.append(f"[PERSONALITY ADAPTATION]\n{context['dynamic_persona']}")
    if context.get("copilot_msg"):
        parts.append(f"[TUTOR MODE]\n{context['copilot_msg']}")
    if context.get("user_rules"):
        rules = context["user_rules"]
        if isinstance(rules, (list, tuple)):
            rules = "\n".join(f"- {r}" for r in rules)
        parts.append(f"[USER RULES - always follow these]\n{rules}")
    if context.get("long_term_memory"):
        parts.append(f"[RECALLED MEMORIES - facts from earlier conversations, use if relevant]\n"
                     f"{context['long_term_memory']}")
    if context.get("search_results"):
        parts.append(f"[WEB SEARCH RESULTS for: {context.get('search_query') or ''}]\n{context['search_results']}\n"
                     "Use these results to answer and cite the URLs you rely on.")
    if context.get("tool_output"):
        parts.append(f"[TOOL OUTPUT]\n{context['tool_output']}")
    if context.get("creative_angles"):
        parts.append(f"[CREATIVE MUSE - consider these perspectives]\n{context['creative_angles']}")
    if context.get("simulation_results"):
        parts.append(f"[FAILURE SIMULATION - warn about the most important risks]\n{context['simulation_results']}")
    if registry is not None and len(registry):
        lines = []
        for name in registry.names():
            tool = registry.get(name)
            flag = " (asks the user for approval)" if tool.dangerous else ""
            lines.append(f"- {name}: {_first_sentence(tool.description)}{flag}")
        parts.append("[TOOLS]\nYou can call these tools. Call one only when it helps; if the user denies an "
                     "approval, do not retry it - explain instead.\n" + "\n".join(lines))
        if _has_computer_tools(registry):
            try:
                from tools.computer import COMPUTER_USE_SYSTEM_PROMPT
                parts.append(f"[COMPUTER USE]\n{COMPUTER_USE_SYSTEM_PROMPT}")
            except Exception:
                pass
    parts.append(f"Current date/time: {_dt.datetime.now().astimezone().strftime('%A, %Y-%m-%d %H:%M %Z')}")
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def prepare_messages(context: dict, system_prompt: str) -> list[dict]:
    """System prompt + a copy of the history; current images are attached to the last user message."""
    msgs = [dict(m) for m in (context.get("messages") or [])]
    user_input = context.get("user_input") or ""
    last_user = next((i for i in range(len(msgs) - 1, -1, -1) if msgs[i].get("role") == "user"), None)
    if user_input and (not msgs or msgs[-1].get("role") != "user"):
        msgs.append({"role": "user", "content": user_input})  # caller forgot to append the current message
        last_user = len(msgs) - 1
    images = list(context.get("images") or [])
    if images and last_user is not None:
        existing = list(msgs[last_user].get("images") or [])
        msgs[last_user]["images"] = existing + [b for b in images if b not in existing]
    return [{"role": "system", "content": system_prompt}] + msgs


def summarize(context: dict, result: AgentResult) -> str:
    lines = []
    if context.get("blocked"):
        lines.append(f"Security: request paused ({context.get('security_reason', 'destructive request')}); tools disabled.")
    else:
        lines.append(f"Agent: {context.get('active_agent', 'GENERALIST')} | Need: {context.get('latent_need', 'SOLUTION')}")
    plugins = [f"{e['name']} ({e.get('ms', 0):.0f} ms)" for e in context.get("trace", [])
               if e.get("type") == "plugin" and "ms" in e and e.get("updates")]
    if plugins:
        lines.append("Plugins: " + ", ".join(plugins))
    if context.get("search_results"):
        lines.append(f"Pre-search: {context.get('search_query')}")
    tools_used = [e["name"] for e in result.trace if e.get("type") == "tool_call"]
    lines.append("Tools: " + (", ".join(tools_used) if tools_used else "none"))
    if result.pending_call:
        lines.append(f"Waiting for approval: {result.pending_call.name}")
    lines.append(f"Model calls: {result.steps}")
    return "\n".join(lines)


class ChatPlugin(NeuroxPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Chat Plugin"
        self.description = "Builds the prompt and runs the tool-calling agent to produce the final answer."
        self.order = 100
        self.runs_when_blocked = True

    def execute(self, context: dict) -> dict | None:
        user_input = context.get("user_input") or ""
        llm = context.get("llm")
        if not user_input.strip() or llm is None:
            return None
        settings = context.get("settings")
        blocked = bool(context.get("blocked"))
        registry = ToolRegistry() if blocked else (context.get("registry") or ToolRegistry())

        messages = prepare_messages(context, build_system_prompt(context, registry))
        max_steps = getattr(settings, "max_agent_steps", 8)
        if _has_computer_tools(registry):
            max_steps = max(max_steps, COMPUTER_USE_MIN_STEPS)
        result = run_agent(
            llm, messages, registry,
            on_token=context.get("on_token_callback"),
            max_steps=max_steps,
            auto_approve=getattr(settings, "auto_approve", False),
            on_event=context.get("on_event"),
        )
        return {
            "final_response": result.final_text,
            "agent_result": result,
            "inner_monologue": summarize(context, result),
        }
