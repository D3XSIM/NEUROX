"""User rules (order 12) and gamification / XP (order 160).

Helpers for the UI (all take an optional ``path`` for tests):
    load_rules() -> list[dict]            [{"id": int, "text": str, "created": iso str}, ...]
    add_rule(text) -> dict | None         new (or existing identical) rule, None if empty
    delete_rule(rule_id) -> bool
    clear_rules() -> int                  number of rules removed
    format_rules(rules) -> str            the 'user_rules' prompt block ('' when there are no rules)
    get_xp_state() -> dict                {"xp", "level", "tasks_completed", "xp_in_level", "xp_for_next", "progress"}
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
from datetime import datetime
from pathlib import Path

from core.config import RULES_FILE, XP_FILE
from core.plugin_interface import NeuroxPlugin

log = logging.getLogger(__name__)

_lock = threading.RLock()

XP_PER_TURN = 10
XP_PER_TOOL_CALL = 5
MAX_TOOL_BONUS = 25
XP_PER_LEVEL = 100


# ---------------------------------------------------------------------------------------- file helpers
def _read_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8") or "null")
    except Exception as e:
        log.warning("Could not read %s: %s", path, e)
    return default


def _write_json(path: Path, data) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        return True
    except Exception as e:
        log.error("Could not write %s: %s", path, e)
        return False


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------------- rules
def _rules_path(path) -> Path:
    return Path(path) if path is not None else Path(RULES_FILE)


def load_rules(path=None) -> list[dict]:
    data = _read_json(_rules_path(path), [])
    rules = []
    for i, r in enumerate(data if isinstance(data, list) else [], start=1):
        if isinstance(r, str):          # tolerate a plain list of strings
            r = {"text": r}
        if not isinstance(r, dict) or not str(r.get("text") or "").strip():
            continue
        try:
            rid = int(r.get("id"))
        except (TypeError, ValueError):
            rid = i
        rules.append({"id": rid, "text": str(r["text"]).strip(), "created": str(r.get("created") or _now())})
    return rules


def add_rule(text: str, path=None) -> dict | None:
    text = " ".join((text or "").split())
    if not text:
        return None
    with _lock:
        rules = load_rules(path)
        for r in rules:
            if r["text"].casefold() == text.casefold():
                return r
        rule = {"id": max((r["id"] for r in rules), default=0) + 1, "text": text, "created": _now()}
        rules.append(rule)
        return rule if _write_json(_rules_path(path), rules) else None


def delete_rule(rule_id: int, path=None) -> bool:
    try:
        rule_id = int(rule_id)
    except (TypeError, ValueError):
        return False
    with _lock:
        rules = load_rules(path)
        kept = [r for r in rules if r["id"] != rule_id]
        if len(kept) == len(rules):
            return False
        return _write_json(_rules_path(path), kept)


def clear_rules(path=None) -> int:
    with _lock:
        n = len(load_rules(path))
        _write_json(_rules_path(path), [])
        return n


def format_rules(rules: list[dict]) -> str:
    if not rules:
        return ""
    lines = ["The user set these rules. Always follow them:"]
    lines += [f"{r['id']}. {r['text']}" for r in rules]
    return "\n".join(lines)


def _norm(text: str) -> str:
    return (text or "").replace("İ", "i").casefold().strip()


_RULE_PATTERNS = [re.compile(p) for p in (
    r"^(?:please\s+|pls\s+)?(?:always|never)\b",
    r"\bfrom\s+now\s+on\b",
    r"\b(?:don'?t|do\s+not|never|stop)\s+(?:ever\s+)?call(?:ing)?\s+me\b",
    r"\bbundan\s+(?:sonra|böyle)\b",
    r"\bartık\s+bana\b",
    r"^(?:lütfen\s+)?(?:her\s+zaman|asla|hiçbir\s+zaman|hiç\s+bir\s+zaman)\b",
    r"\bbana\b.{1,60}?\bdeme\b",
)]
_FORGET_ALL = re.compile(r"\b(?:forget|delete|remove|clear)\s+(?:all\s+(?:my\s+|the\s+)?|my\s+|the\s+)?rules\b"
                         r"|\b(?:tüm|bütün)\s+kuralları\s+(?:unut|sil|temizle)\b|\bkuralları\s+(?:temizle|sıfırla)\b")
_FORGET_ONE = [re.compile(p) for p in (
    r"\b(?:forget|delete|remove)\s+rule\s*(?:#|no\.?\s*|number\s*)?(\d+)",
    r"\bkural(?:ı|i)?\s*(?:unut|sil)\s*#?(\d+)",
    r"#?(\d+)\s*\.?\s*(?:numaralı\s+)?kural(?:ı|i)?\s+(?:unut|sil)",
    r"\bkural\s*#?(\d+)\s*'?\s*(?:ı|i|u|ü|yı|yi|yu|yü)?\s+(?:unut|sil)",
)]
_LIST = re.compile(r"\b(?:list|show)\s+(?:me\s+)?(?:all\s+|my\s+|the\s+)*rules\b|\bwhat\s+are\s+my\s+rules\b"
                   r"|\bkuralları(?:mı)?\s+(?:göster|listele)\b|\bkurallarım\s+(?:neler|ne)\b")


def is_rule_statement(text: str) -> bool:
    t = _norm(text)
    if not t or t.endswith("?") or len(t) > 500:
        return False
    return any(p.search(t) for p in _RULE_PATTERNS)


class RulePlugin(NeuroxPlugin):
    """Detects rule statements / rule commands and injects all rules as ``user_rules`` every turn.

    Extra context keys produced: ``rules_msg`` (short user-facing confirmation, e.g. for a toast) when a rule
    was added/removed/listed, and ``rules`` (the current list of rule dicts)."""

    def __init__(self):
        super().__init__()
        self.name = "Rule Plugin"
        self.description = "Persistent user rules (always/never/from now on...)."
        self.version = "2.0.0"
        self.order = 12

    def execute(self, context: dict) -> dict | None:
        try:
            return self._execute(context)
        except Exception as e:
            log.exception("Rule plugin failed")
            trace = context.get("trace")
            if isinstance(trace, list):
                trace.append({"type": "plugin", "name": self.name, "info": f"error: {e}"})
            return None

    def _execute(self, context: dict) -> dict | None:
        if context.get("blocked"):
            return None
        user_input = context.get("user_input") or ""
        t = _norm(user_input)
        tr = getattr(context.get("settings"), "language", "en") == "tr"
        note, msg = "", ""

        if t:
            if _FORGET_ALL.search(t):
                n = clear_rules()
                msg = f"{n} kural silindi." if tr else f"Deleted {n} rule(s)."
                note = f"All {n} user rules were just deleted at the user's request. Confirm this briefly."
            elif (m := next((mm for p in _FORGET_ONE if (mm := p.search(t))), None)):
                rid = int(m.group(1))
                if delete_rule(rid):
                    msg = f"Kural #{rid} silindi." if tr else f"Rule #{rid} deleted."
                    note = f"Rule #{rid} was just deleted at the user's request. Confirm this briefly."
                else:
                    msg = f"Kural #{rid} bulunamadı." if tr else f"Rule #{rid} not found."
                    note = f"The user asked to delete rule #{rid}, but no such rule exists. Tell them."
            elif _LIST.search(t):
                msg = "Kurallar listelendi." if tr else "Rules listed."
                note = ("The user asked to see their rules: list every rule above with its number exactly as written "
                        "(or say there are none).")
            elif is_rule_statement(user_input):
                rule = add_rule(user_input)
                if rule:
                    msg = f"Yeni kural kaydedildi (#{rule['id']})." if tr else f"New rule saved (#{rule['id']})."
                    note = f"Rule #{rule['id']} was just saved from the user's message; acknowledge it and follow it."

        rules = load_rules()
        text = format_rules(rules)
        if note:
            text = (text + "\n\n" if text else "The user has no rules.\n\n") + f"[Note: {note}]"
        trace = context.get("trace")
        if isinstance(trace, list) and (rules or msg):
            trace.append({"type": "plugin", "name": self.name,
                          "info": msg or f"{len(rules)} rule(s) active"})
        updates = {"user_rules": text, "rules": rules}
        if msg:
            updates["rules_msg"] = msg
        return updates


# ----------------------------------------------------------------------------------------- gamification
def _xp_path(path) -> Path:
    return Path(path) if path is not None else Path(XP_FILE)


def _level_for(xp: int) -> int:
    return 1 + max(0, xp) // XP_PER_LEVEL


def _load_xp(path=None) -> dict:
    data = _read_json(_xp_path(path), {})
    data = data if isinstance(data, dict) else {}
    try:
        xp = max(0, int(data.get("xp", 0)))
    except (TypeError, ValueError):
        xp = 0
    try:
        tasks = max(0, int(data.get("tasks_completed", 0)))
    except (TypeError, ValueError):
        tasks = 0
    return {"xp": xp, "level": _level_for(xp), "tasks_completed": tasks}


def get_xp_state(path=None) -> dict:
    """XP state for the sidebar: xp, level, tasks_completed, xp_in_level, xp_for_next (=100), progress (0..1)."""
    s = _load_xp(path)
    in_level = s["xp"] % XP_PER_LEVEL
    return {**s, "xp_in_level": in_level, "xp_for_next": XP_PER_LEVEL, "progress": in_level / XP_PER_LEVEL}


def _count_tool_calls(agent_result) -> int:
    if agent_result is None:
        return 0
    trace = agent_result.get("trace") if isinstance(agent_result, dict) else getattr(agent_result, "trace", None)
    return sum(1 for ev in trace or [] if isinstance(ev, dict) and ev.get("type") == "tool_call")


def award_xp(amount: int, tasks: int = 1, path=None) -> tuple[dict, dict]:
    """Add XP; returns (old_state, new_state)."""
    with _lock:
        old = _load_xp(path)
        xp = old["xp"] + max(0, int(amount))
        new = {"xp": xp, "level": _level_for(xp), "tasks_completed": old["tasks_completed"] + max(0, int(tasks))}
        _write_json(_xp_path(path), new)
        return old, new


class GamificationPlugin(NeuroxPlugin):
    """+10 XP per answered turn (+5 per tool call, max +25). ``gamification_msg`` only on level-up;
    ``xp_state`` (see get_xp_state) always."""

    def __init__(self):
        super().__init__()
        self.name = "Gamification Plugin"
        self.description = "XP and levels."
        self.version = "2.0.0"
        self.order = 160

    def execute(self, context: dict) -> dict | None:
        try:
            if context.get("blocked") or not (context.get("user_input") or "").strip():
                return None
            answer = context.get("final_response") or ""
            if not answer.strip() or answer.startswith("⚠️"):  # no XP for failed or still-pending turns
                return {"xp_state": get_xp_state()}
            tool_calls = _count_tool_calls(context.get("agent_result"))
            gained = XP_PER_TURN + min(MAX_TOOL_BONUS, XP_PER_TOOL_CALL * tool_calls)
            old, new = award_xp(gained)
            updates = {"xp_state": get_xp_state()}
            info = f"+{gained} XP ({new['xp']} total, level {new['level']})"
            if new["level"] > old["level"]:
                tr = getattr(context.get("settings"), "language", "en") == "tr"
                updates["gamification_msg"] = f"🎉 Seviye {new['level']}!" if tr else f"🎉 Level {new['level']}!"
                info += " - level up!"
            trace = context.get("trace")
            if isinstance(trace, list):
                trace.append({"type": "plugin", "name": self.name, "info": info})
            return updates
        except Exception as e:
            log.exception("Gamification plugin failed: %s", e)
            return None
