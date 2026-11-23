"""Metacognition: judges how well-grounded the answer can be and adds an honesty instruction (no LLM call)."""
from __future__ import annotations

import re

from core.plugin_interface import NeuroxPlugin

_COMPLEX = re.compile(r"\b(?:explain|analy[sz]e|why|compare|evaluate|prove|"
                      r"açıkla\w*|analiz\w*|neden|niçin|karşılaştır\w*|değerlendir\w*)", re.IGNORECASE | re.UNICODE)


class MetacognitionPlugin(NeuroxPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Metacognition Plugin"
        self.description = "Evaluates context quality and adds honesty instructions."
        self.order = 95

    def execute(self, context: dict) -> dict | None:
        user_input = (context.get("user_input") or "").strip()
        if not user_input or context.get("blocked"):
            return None
        has_search = bool(context.get("search_results"))
        has_memory = bool(context.get("long_term_memory"))
        registry = context.get("registry")
        can_search = registry is not None and "web_search" in registry

        if has_search:
            level, meta = "HIGH", ("Verify your answer against the [WEB SEARCH RESULTS]; prefer them over training "
                                   "data when they conflict and cite the URLs you used.")
        elif _COMPLEX.search(user_input):
            level = "MEDIUM" if has_memory else "LOW"
            meta = ("No external sources were retrieved. "
                    + ("If facts may be outdated or uncertain, use the web_search tool before answering. " if can_search
                       else "Say clearly when you rely on training data that may be outdated. ")
                    + "Separate facts from assumptions.")
        else:
            level, meta = "MEDIUM", "If you are not sure about something, say so instead of inventing details."

        existing = context.get("dynamic_persona") or ""
        block = f"[METACOGNITION: {level}]\n{meta}"
        return {"dynamic_persona": f"{existing}\n\n{block}".strip(), "confidence_level": level}
