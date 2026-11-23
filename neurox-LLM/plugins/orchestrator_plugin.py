"""Manager agent: picks a persona (RESEARCHER / CODER / PLANNER / GENERALIST) and the latent need.

Obvious requests are classified with a keyword fast-path (EN + TR) so no LLM call is spent; otherwise the
model is asked for ``AGENT|NEED`` and the reply is parsed robustly with regexes.
"""
from __future__ import annotations

import logging
import re

from core.llm import LLMError
from core.plugin_interface import NeuroxPlugin

log = logging.getLogger(__name__)

_COMMON = (
    "You can use tools (web search, reading web pages, a calculator, the clock, files in the user's workspace, "
    "running Python, long-term memory) whenever they help - prefer a tool over guessing, and say so when you used one. "
    "Always answer in the same language the user writes in (e.g. Turkish -> Turkish, English -> English)."
)

PERSONAS = {
    "RESEARCHER": "You are Neurox in Researcher mode. Your goal is accurate, up-to-date information. Base your answer "
                  "on the [WEB SEARCH RESULTS] if present, use web_search / fetch_url for more, cite sources (URLs), "
                  "and say clearly when something is uncertain. " + _COMMON,
    "CODER": "You are Neurox in Senior Developer mode. Write clean, correct, complete code with short explanations. "
             "Use run_python to test snippets or compute results when it helps. " + _COMMON,
    "PLANNER": "You are Neurox in Strategic Planner mode. Break the problem into clear, ordered, actionable steps with "
               "priorities, risks and checkpoints. " + _COMMON,
    "GENERALIST": "You are Neurox, a helpful, friendly and intelligent local AI assistant. " + _COMMON,
}

NEED_HINTS = {
    "VALIDATION": "\n\n[INTENT: VALIDATION]\nThe user mainly wants to be heard. Be warm and supportive first, then help.",
    "EXPLANATION": "\n\n[INTENT: EXPLANATION]\nFocus on clear definitions, intuition and analogies.",
    "SOLUTION": "",
}

_W = re.IGNORECASE | re.UNICODE
_KEYWORDS = {
    "CODER": re.compile(
        r"```|\b(?:code|coding|python|javascript|typescript|java|c\+\+|c#|golang|rust|html|css|sql|regex|function|"
        r"script|bug|debug|traceback|exception|stack\s*trace|compile|compiler|refactor|unit\s*test|api|json|"
        r"class|method|variable|syntax|error\s*code|pip|npm|git)\b|"
        r"\b(?:kod\w*|fonksiyon\w*|betik\w*|derle\w*|hata\s+ayıkla\w*|programla\w*|yazılım\w*)", _W),
    "RESEARCHER": re.compile(
        r"\b(?:latest|news|today'?s|current(?:ly)?|recent|price\s+of|stock\s+price|weather|who\s+won|"
        r"search(?:\s+for)?|look\s+up|google|find\s+out|release\s+date|20[2-3]\d)\b|"
        r"\b(?:haber\w*|güncel\w*|bugün\w*|son\s+dakika|araştır\w*|internette|fiyat\w*|hava\s+durumu\w*|"
        r"kim\s+kazandı|en\s+son)", _W),
    "PLANNER": re.compile(
        r"\b(?:plan|planning|roadmap|strategy|strategic|schedule|milestones?|break\s+(?:it\s+)?down|"
        r"step[-\s]by[-\s]step\s+plan|project\s+plan)\b|"
        r"\b(?:planla\w*|plan\w*\s+yap|strateji\w*|yol\s+harita\w*|takvim\w*|program\s+oluştur\w*)", _W),
}
_GREETING = re.compile(
    r"^\W*(?:hi|hello|hey|yo|thanks|thank\s+you|good\s+(?:morning|evening|night)|how\s+are\s+you|"
    r"merhaba|selam|selamlar|sa|teşekkür\w*|sağ\s*ol\w*|günaydın|iyi\s+(?:akşamlar|geceler)|nasılsın\w*)\b", _W)
_NEED_KW = {
    "VALIDATION": re.compile(r"\b(?:i\s+feel|feeling|sad|depressed|stressed|anxious|lonely|frustrated|"
                             r"üzgün\w*|moral\w*|stres\w*|kaygı\w*|yalnız\w*|bunal\w*)", _W),
    "EXPLANATION": re.compile(r"\b(?:explain|why|how\s+does|how\s+do|what\s+is|what\s+are|meaning\s+of|"
                              r"nedir|neden|niçin|niye|açıkla\w*|nasıl\s+çalış\w*|ne\s+demek)", _W),
}

_AGENT_RE = re.compile(r"\b(RESEARCHER|CODER|PLANNER|GENERALIST)\b", re.IGNORECASE)
_NEED_RE = re.compile(r"\b(SOLUTION|VALIDATION|EXPLANATION)\b", re.IGNORECASE)

CLASSIFY_PROMPT = """Classify the user's request. Reply with exactly one line in the form AGENT|NEED and nothing else.

AGENT is one of:
- RESEARCHER: needs current information, news, facts or anything that should be looked up on the internet
- CODER: code generation, debugging, scripts, technical programming help
- PLANNER: a complex goal that needs a strategy, plan or step-by-step breakdown
- GENERALIST: casual chat, greetings, simple questions, everything else

NEED is one of:
- SOLUTION: wants a fix, an answer or an action
- VALIDATION: wants empathy or to be heard
- EXPLANATION: wants to understand a concept

Example: CODER|SOLUTION"""


def keyword_need(text: str) -> str:
    for need, rx in _NEED_KW.items():
        if rx.search(text):
            return need
    return "SOLUTION"


def keyword_agent(text: str) -> str | None:
    """Fast-path classification; None when ambiguous (then the LLM decides)."""
    stripped = text.strip()
    if _GREETING.search(stripped) and len(stripped.split()) <= 6:
        return "GENERALIST"
    hits = {agent: len(rx.findall(stripped)) for agent, rx in _KEYWORDS.items()}
    hits = {a: n for a, n in hits.items() if n}
    if not hits:
        return None
    ranked = sorted(hits.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) == 1 or ranked[0][1] >= 2 * ranked[1][1]:
        return ranked[0][0]
    return None


def parse_classification(reply: str) -> tuple[str, str]:
    a = _AGENT_RE.search(reply or "")
    n = _NEED_RE.search(reply or "")
    return (a.group(1).upper() if a else "GENERALIST", n.group(1).upper() if n else "SOLUTION")


class OrchestratorPlugin(NeuroxPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Orchestrator Plugin"
        self.description = "The Manager Agent. Analyzes intent and assigns the best persona."
        self.order = 10

    def execute(self, context: dict) -> dict | None:
        user_input = (context.get("user_input") or "").strip()
        if not user_input or context.get("blocked"):
            return None

        agent = keyword_agent(user_input)
        need = keyword_need(user_input)
        method = "keywords"
        llm = context.get("llm")
        if agent is None and llm is not None:
            method = "llm"
            try:
                reply = llm.chat([{"role": "system", "content": CLASSIFY_PROMPT},
                                  {"role": "user", "content": user_input[:1500]}], temperature=0).content
                agent, llm_need = parse_classification(reply)
                if need == "SOLUTION":
                    need = llm_need
            except LLMError as e:
                log.warning("Orchestrator classification failed: %s", e)
                method = "fallback"
        agent = agent or "GENERALIST"

        trace = context.get("trace")
        if isinstance(trace, list):
            trace.append({"type": "plugin", "name": self.name, "info": f"{agent}|{need} via {method}"})

        updates = {
            "active_agent": agent,
            "latent_need": need,
            "system_prompt": PERSONAS[agent] + NEED_HINTS.get(need, ""),
        }
        if agent == "RESEARCHER":
            updates["search_query"] = user_input
        return updates
