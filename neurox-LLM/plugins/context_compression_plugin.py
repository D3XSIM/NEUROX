"""Context compression (order 2).

When the chat history is longer than ``settings.history_window`` messages, older messages are replaced by a
single system message "Summary of earlier conversation: ..." written by ``llm.simple``. Leading system
messages are kept, and the verbatim tail never starts with an orphaned tool result.

Summaries are cached in memory keyed by (number of summarized messages, hash of those messages); a
longer prefix re-uses the best cached shorter summary and only summarizes the new messages. The cut point
moves in steps so that the LLM is not called on every turn. If the LLM fails, older messages are simply
dropped (plain truncation). The original ``messages`` list is never mutated.
"""
from __future__ import annotations

import hashlib
import logging
import math

from core.plugin_interface import NeuroxPlugin

log = logging.getLogger(__name__)

SUMMARY_PREFIX = "Summary of earlier conversation: "
MAX_CHARS_PER_MESSAGE = 600
MAX_INPUT_CHARS = 6000
MAX_CACHE = 64

_SYSTEM = ("You compress chat histories. Write a short, factual summary (max ~150 words, bullet points ok) of the "
           "conversation below: user facts and preferences, decisions, results of tool calls, open tasks. "
           "Write in the same language as the conversation. Output only the summary.")


def _msg_line(m: dict) -> str:
    role = m.get("role", "?")
    content = str(m.get("content") or "")
    if role == "tool":
        role = f"tool {m.get('name', '')}".strip()
        limit = 200
    else:
        limit = MAX_CHARS_PER_MESSAGE
    if m.get("tool_calls"):
        names = []
        for tc in m["tool_calls"]:
            names.append(getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else "") or "?")
        content = (content + f" [called tools: {', '.join(names)}]").strip()
    if m.get("images"):
        content += f" [{len(m['images'])} image(s)]"
    content = " ".join(content.split())
    if content.startswith(SUMMARY_PREFIX):
        limit = MAX_INPUT_CHARS // 2
    if len(content) > limit:
        content = content[:limit] + "…"
    return f"{role}: {content}"


def _msg_hash(m: dict) -> str:
    return f"{m.get('role')}\x00{m.get('content') or ''}\x00{m.get('tool_call_id') or ''}"


class ContextCompressionPlugin(NeuroxPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Context Compression"
        self.description = "Summarizes old messages when the history exceeds settings.history_window."
        self.version = "2.0.0"
        self.order = 2
        self._cache: dict[tuple[int, str], str] = {}

    # --------------------------------------------------------------------------------------------- helpers
    @staticmethod
    def _prefix_hashes(body: list[dict]) -> list[str]:
        """hashes[i] identifies body[:i]."""
        h = hashlib.sha1()
        out = [h.hexdigest()]
        for m in body:
            h.update(_msg_hash(m).encode("utf-8", "replace"))
            out.append(h.hexdigest())
        return out

    def _summarize(self, llm, body: list[dict], cut: int, hashes: list[str]) -> str | None:
        key = (cut, hashes[cut])
        if key in self._cache:
            return self._cache[key]
        # best cached shorter prefix of the same conversation
        prev_n, prev = 0, ""
        for n in range(cut - 1, 0, -1):
            s = self._cache.get((n, hashes[n]))
            if s:
                prev_n, prev = n, s
                break
        lines = [_msg_line(m) for m in body[prev_n:cut]]
        text = "\n".join(lines)
        if len(text) > MAX_INPUT_CHARS:  # keep the most recent part
            text = "…" + text[-MAX_INPUT_CHARS:]
        prompt = (f"Earlier summary:\n{prev}\n\nNew messages:\n{text}" if prev else f"Conversation:\n{text}")
        try:
            summary = (llm.simple(prompt, system=_SYSTEM, temperature=0.2) or "").strip()
        except Exception as e:
            log.warning("Context summary failed, falling back to truncation: %s", e)
            return None
        if not summary:
            return None
        if len(self._cache) >= MAX_CACHE:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = summary
        return summary

    # --------------------------------------------------------------------------------------------- execute
    def execute(self, context: dict) -> dict | None:
        try:
            return self._execute(context)
        except Exception as e:
            log.exception("Context compression failed: %s", e)
            return None

    def _execute(self, context: dict) -> dict | None:
        if context.get("blocked"):
            return None
        messages = context.get("messages")
        settings = context.get("settings")
        if not isinstance(messages, list) or settings is None:
            return None
        try:
            window = max(2, int(getattr(settings, "history_window", 20)))
        except (TypeError, ValueError):
            window = 20
        if len(messages) <= window:
            return None

        # keep leading system messages untouched
        lead = 0
        while lead < len(messages) and isinstance(messages[lead], dict) and messages[lead].get("role") == "system":
            lead += 1
        head, body = list(messages[:lead]), [m for m in messages[lead:] if isinstance(m, dict)]
        excess = len(body) - window
        if excess <= 0:
            return None
        step = max(2, window // 4)
        cut = min(math.ceil(excess / step) * step, len(body) - 1)
        # don't start the verbatim tail with tool results (they belong to an assistant tool_calls message)
        while cut > 0 and body[cut].get("role") == "tool":
            cut -= 1
        if cut <= 0:
            return None

        hashes = self._prefix_hashes(body)
        llm = context.get("llm")
        summary = self._summarize(llm, body, cut, hashes) if llm is not None else None
        tail = [dict(m) for m in body[cut:]]
        if summary:
            compressed = head + [{"role": "system", "content": SUMMARY_PREFIX + summary}] + tail
            info = f"summarized {cut} older messages"
        else:
            compressed = head + tail
            info = f"dropped {cut} older messages (summary unavailable)"
        trace = context.get("trace")
        if isinstance(trace, list):
            trace.append({"type": "plugin", "name": self.name, "info": info})
        return {"messages": compressed, "compressed_messages": cut}
