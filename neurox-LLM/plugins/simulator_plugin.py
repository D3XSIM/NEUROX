"""Pre-mortem: for plans/strategies, one short extra LLM call lists likely failure causes so the answer can
warn about them. Runs before Chat (order 92)."""
from __future__ import annotations

import logging
import re

from core.llm import LLMError
from core.plugin_interface import NeuroxPlugin

log = logging.getLogger(__name__)

_TRIGGER = re.compile(
    r"\b(?:plan|planning|strategy|roadmap|simulate|simulation|what\s+if|risks?|pre-?mortem|"
    r"planla\w*|strateji\w*|yol\s+harita\w*|simüle\w*|simülasyon\w*|risk\w*|ne\s+olur)\b", re.IGNORECASE | re.UNICODE)

PROMPT = ("Do a quick PRE-MORTEM for the user's plan or idea: list the 3 most likely reasons it could fail, "
          "one short line each (max 20 words) with a one-phrase mitigation. Numbered, in the user's language, "
          "no introduction.")


class SimulatorPlugin(NeuroxPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Simulator Plugin"
        self.description = "Simulates failure scenarios (pre-mortem) for plans and strategies."
        self.order = 92  # must run before Chat (100)

    def execute(self, context: dict) -> dict | None:
        user_input = (context.get("user_input") or "").strip()
        llm = context.get("llm")
        if not user_input or context.get("blocked") or llm is None:
            return None
        if not (_TRIGGER.search(user_input) or context.get("active_agent") == "PLANNER"):
            return None
        try:
            text = llm.chat([{"role": "system", "content": PROMPT},
                             {"role": "user", "content": user_input[:1500]}], temperature=0.3).content.strip()
        except LLMError as e:
            log.info("Simulator skipped: %s", e)
            return None
        return {"simulation_results": text[:1200]} if text else None
