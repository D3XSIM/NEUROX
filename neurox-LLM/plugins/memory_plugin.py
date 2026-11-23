"""Long-term memory plugin (order 15).

- ``feedback_text`` (user correction)  -> stored as kind="correction", returns ``memory_saved``.
- "remember ... / don't forget ... / hatırla ... / unutma ..." in ``user_input`` -> the fact is stored.
- Every normal turn: relevant memories are searched and injected as ``long_term_memory`` (bulleted).
Anything else is left to the ``remember`` tool; whole conversations are not stored automatically.
"""
from __future__ import annotations

import logging
import re

from core.plugin_interface import NeuroxPlugin
from utils.vector_store import get_store

log = logging.getLogger(__name__)

SEARCH_TOP_K = 4
SEARCH_THRESHOLD = 0.45

_F = re.IGNORECASE | re.DOTALL
# Patterns whose group "fact" is the thing to store.
_REMEMBER_PATTERNS = [
    # EN: "remember that X", "please remember: X", "don't forget (that) X", "keep in mind (that) X"
    re.compile(r"^\s*(?:please\s+|pls\s+)?(?:remember|memori[sz]e|don'?t\s+forget|do\s+not\s+forget|keep\s+in\s+mind|make\s+a\s+note|take\s+note|note\s+that)"
               r"(?:\s+(?:that|this))?\s*[:,\-]?\s+(?P<fact>.+)$", _F),
    # EN trailing: "X, remember that." / "X - don't forget it"
    re.compile(r"^(?P<fact>.+?)[\s,.;\-]+(?:please\s+)?(?:remember\s+(?:that|this|it)|don'?t\s+forget\s+(?:that|this|it))[\s.!]*$", _F),
    # TR: "hatırla (ki) X", "şunu hatırla: X", "unutma ki X", "aklında tut: X", "not al: X"
    re.compile(r"^\s*(?:lütfen\s+)?(?:şunu\s+|bunu\s+)?(?:hatırla|unutma|aklında\s+tut|aklında\s+bulunsun|not\s+al)"
               r"(?:\s+ki)?\s*[:,\-]?\s+(?P<fact>.+)$", _F),
    # TR trailing: "Benim adım Tuna, bunu unutma." / "X, hatırla" / "X, bunu aklında tut"
    re.compile(r"^(?P<fact>.+?)[\s,.;\-]+(?:lütfen\s+)?(?:bunu\s+|şunu\s+)?(?:hatırla|unutma|aklında\s+tut)[\s.!]*$", _F),
]
_TRAILING = {1, 3}  # indexes of the "X, remember it" patterns (need a multi-word fact)
_QUESTION = re.compile(r"(\?\s*$)|(\b(do|did)\s+you\s+remember\b)|(hatırlıyor\s+mu)|(hatırladın\s+mı)", re.IGNORECASE)


def extract_fact(text: str) -> str | None:
    """Return the fact the user explicitly asked to remember, or None."""
    text = (text or "").strip()
    if not text or len(text) > 1000 or _QUESTION.search(text):
        return None
    for i, pat in enumerate(_REMEMBER_PATTERNS):
        m = pat.match(text)
        if m:
            fact = m.group("fact").strip(" \t\n,.;:-!")
            if i in _TRAILING and len(fact.split()) < 2:
                continue
            if len(fact) >= 3 and fact.lower() not in {"this", "that", "it", "bunu", "şunu", "onu"}:
                return fact
    return None


def format_memories(hits: list[dict]) -> str:
    lines = []
    for h in hits:
        prefix = "User correction: " if h.get("kind") == "correction" else ""
        lines.append(f"- {prefix}{h.get('text', '')}")
    return "\n".join(lines)


class MemoryPlugin(NeuroxPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Memory Plugin"
        self.description = "Long-term vector memory: stores corrections/explicit facts, recalls relevant ones."
        self.version = "2.0.0"
        self.order = 15

    def execute(self, context: dict) -> dict | None:
        try:
            return self._execute(context)
        except Exception as e:  # never break the pipeline
            log.exception("Memory plugin failed")
            self._trace(context, f"error: {e}")
            return None

    def _trace(self, context: dict, info: str) -> None:
        trace = context.get("trace")
        if isinstance(trace, list):
            trace.append({"type": "plugin", "name": self.name, "info": info})

    def _execute(self, context: dict) -> dict | None:
        if context.get("blocked"):
            return None
        llm = context.get("llm")
        if llm is None:
            return None
        store = get_store(llm)
        updates: dict = {}

        feedback = (context.get("feedback_text") or "").strip()
        if feedback:
            entry_id = store.add(feedback, kind="correction", metadata={"source": "feedback"})
            self._trace(context, "correction saved" if entry_id else "could not save correction (embed model unavailable?)")
            return {"memory_saved": bool(entry_id)}

        user_input = (context.get("user_input") or "").strip()
        if not user_input:
            return None

        fact = extract_fact(user_input)
        if fact:
            entry_id = store.add(fact, kind="fact", metadata={"source": "user"})
            updates["memory_saved"] = bool(entry_id)
            self._trace(context, f"remembered: {fact[:80]}" if entry_id else "could not save memory")

        hits = store.search(user_input, top_k=SEARCH_TOP_K, threshold=SEARCH_THRESHOLD)
        if fact:  # don't echo back the fact we just stored
            hits = [h for h in hits if h["text"] != fact]
        if hits:
            updates["long_term_memory"] = format_memories(hits)
            self._trace(context, f"recalled {len(hits)} memor{'y' if len(hits) == 1 else 'ies'}")
        return updates or None
