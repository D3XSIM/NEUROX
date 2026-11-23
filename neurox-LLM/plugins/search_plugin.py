"""Pre-search for RESEARCHER turns: runs one web search before the agent starts, so even models without
native tool calling get fresh results in their system prompt. Uses the shared ``tools.builtin.web_search``."""
from __future__ import annotations

import logging

from core.llm import LLMError
from core.plugin_interface import NeuroxPlugin
from tools.builtin import web_search

log = logging.getLogger(__name__)

REFINE_PROMPT = ("Rewrite the user's message as a short web search query (max 8 words, same language, no quotes). "
                 "Reply with the query only.")


class SearchPlugin(NeuroxPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Search Plugin"
        self.description = "Performs a web search for research questions."
        self.order = 20

    def execute(self, context: dict) -> dict | None:
        if not (context.get("user_input") or "").strip() or context.get("blocked"):
            return None
        query = (context.get("search_query") or "").strip()
        if not query or context.get("search_results"):
            return None

        if len(query.split()) > 12 and context.get("llm") is not None:  # long message -> condense
            try:
                refined = context["llm"].chat([{"role": "system", "content": REFINE_PROMPT},
                                               {"role": "user", "content": query[:1000]}], temperature=0).content
                refined = refined.strip().strip('"').splitlines()[0].strip() if refined.strip() else ""
                if 0 < len(refined) <= 200:
                    query = refined
            except LLMError as e:
                log.info("Query refinement skipped: %s", e)
        query = query[:300]

        results = web_search(query, max_results=5)
        ok = not (results.startswith("Error") or results.startswith("No results"))
        trace = context.get("trace")
        if isinstance(trace, list):
            trace.append({"type": "plugin", "name": self.name,
                          "info": f"web_search({query!r}) -> {'ok' if ok else results[:120]}"})
        return {"search_query": query, "search_results": results if ok else None}
