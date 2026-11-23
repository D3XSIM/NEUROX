"""Divergent thinking: for brainstorming requests, one short extra LLM call suggests unusual angles."""
from __future__ import annotations

import logging
import re

from core.llm import LLMError
from core.plugin_interface import NeuroxPlugin

log = logging.getLogger(__name__)

_TRIGGER = re.compile(
    r"\b(?:brainstorm\w*|ideas?|creative|creativity|alternatives?|think\s+outside|out\s+of\s+the\s+box|"
    r"fikir\w*|yaratıcı\w*|beyin\s+fırtınası\w*|alternatif\w*|öneri\w*)", re.IGNORECASE | re.UNICODE)

PROMPT = ("Give exactly 3 divergent, non-obvious angles or perspectives for approaching the user's request. "
          "One short line each (max 20 words), numbered, in the user's language. No introduction.")


class CreativityPlugin(NeuroxPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Creativity Plugin"
        self.description = "Injects divergent thinking for brainstorming requests."
        self.order = 90

    def execute(self, context: dict) -> dict | None:
        user_input = (context.get("user_input") or "").strip()
        llm = context.get("llm")
        if not user_input or context.get("blocked") or llm is None or not _TRIGGER.search(user_input):
            return None
        try:
            text = llm.chat([{"role": "system", "content": PROMPT},
                             {"role": "user", "content": user_input[:1500]}], temperature=0.9).content.strip()
        except LLMError as e:
            log.info("Creativity skipped: %s", e)
            return None
        return {"creative_angles": text[:1200]} if text else None
