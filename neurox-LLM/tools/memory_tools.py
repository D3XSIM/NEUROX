"""Long-term memory tools: ``remember`` and ``recall`` (backed by utils.vector_store)."""
from __future__ import annotations

from core.tools import Tool
from utils.vector_store import get_store

RECALL_THRESHOLD = 0.3


def get_tools(settings, llm=None) -> list[Tool]:
    def _store():
        return get_store(llm) if llm is not None else None

    def remember(text: str) -> str:
        text = (text or "").strip()
        if not text:
            return "Error: nothing to remember (empty text)."
        store = _store()
        if store is None:
            return "Error: memory is unavailable (no model client)."
        entry_id = store.add(text, kind="fact", metadata={"source": "tool"})
        if not entry_id:
            return (f"Error: could not save the memory. Is the embedding model "
                    f"'{store.embed_model}' installed and the server running?")
        return f"Saved to long-term memory (id {entry_id}): {text}"

    def recall(query: str, top_k: int = 3) -> str:
        store = _store()
        if store is None:
            return "Error: memory is unavailable (no model client)."
        try:
            top_k = max(1, min(int(top_k), 10))
        except (TypeError, ValueError):
            top_k = 3
        hits = store.search(query, top_k=top_k, threshold=RECALL_THRESHOLD)
        if not hits:
            return "No relevant memories found."
        lines = []
        for h in hits:
            prefix = "User correction: " if h["kind"] == "correction" else ""
            lines.append(f"- {prefix}{h['text']} (score {h['score']:.2f}, saved {h['timestamp']})")
        return "\n".join(lines)

    return [
        Tool(
            name="remember",
            description="Save an important fact about the user or the conversation to long-term memory "
                        "so it can be recalled in future chats (e.g. names, preferences, decisions).",
            parameters={"type": "object",
                        "properties": {"text": {"type": "string", "description": "The fact to remember, as one short self-contained sentence."}},
                        "required": ["text"]},
            func=remember, dangerous=False, category="memory"),
        Tool(
            name="recall",
            description="Search long-term memory for facts saved in earlier conversations.",
            parameters={"type": "object",
                        "properties": {"query": {"type": "string", "description": "What to look for."},
                                       "top_k": {"type": "integer", "description": "Maximum number of results (default 3).", "default": 3}},
                        "required": ["query"]},
            func=recall, dangerous=False, category="memory"),
    ]
