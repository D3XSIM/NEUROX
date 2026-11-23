"""Tutor mode: when the user asks to be guided/taught, instruct the model to teach step by step."""
from __future__ import annotations

import re

from core.plugin_interface import NeuroxPlugin

_TRIGGER = re.compile(
    r"\b(?:guide\s+me|tutorial|teach\s+me|walk\s+me\s+through|step[-\s]by[-\s]step|show\s+me\s+how|"
    r"öğret\w*|adım\s+adım|rehberlik\s+et\w*|bana\s+göster\w*)", re.IGNORECASE | re.UNICODE)

COPILOT_INSTRUCTION = (
    "Tutor mode is ON. Teach the user interactively instead of dumping everything at once:\n"
    "1. Briefly state the goal and what the user will be able to do at the end.\n"
    "2. Give ONE small step at a time (with a short example or command), explaining why it matters.\n"
    "3. After each step, ask a short question or give a mini exercise to check understanding, then stop and "
    "wait for the user's reply before continuing.\n"
    "4. Adapt to the user's answers: if they struggle, re-explain more simply; if they are fast, move on.\n"
    "Keep a friendly, encouraging tone and use the user's language."
)


class CoPilotPlugin(NeuroxPlugin):
    def __init__(self):
        super().__init__()
        self.name = "CoPilot Plugin"
        self.description = "Step-by-step tutor mode for 'guide me', 'tutorial', 'öğret', 'adım adım'."
        self.order = 25

    def execute(self, context: dict) -> dict | None:
        user_input = context.get("user_input") or ""
        if not user_input.strip() or context.get("blocked"):
            return None
        if _TRIGGER.search(user_input):
            return {"copilot_msg": COPILOT_INSTRUCTION}
        return None
