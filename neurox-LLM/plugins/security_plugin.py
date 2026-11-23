"""Blocks clearly destructive requests (delete files, format disks, shut down, drop tables...) unless the
user explicitly confirms by starting or ending the message with the exact token ``CONFIRM`` or ``ONAYLA``.

Detection is deliberately conservative to avoid false positives ("skills", "killer app", "removal of doubt",
"remove duplicates from this list"): a destructive *verb* must be close to a sensitive *object*
(files, disk, database, computer...), or the message must contain a raw destructive command (``rm -rf``, ``DROP TABLE``...).
Dangerous *tools* are additionally gated by the agent's approval step, so this is a first line of defence only.
"""
from __future__ import annotations

import re

from core.plugin_interface import NeuroxPlugin

_F = re.IGNORECASE | re.UNICODE

# --- raw commands that are dangerous on their own --------------------------------------------------------
_COMMAND_PATTERNS = [
    r"\brm\s+-(?:[a-z]*r[a-z]*f|[a-z]*f[a-z]*r)[a-z]*\b",      # rm -rf / rm -fr
    r"\brm\s+-r\b",
    r"\brmdir\s+/s\b", r"\bdel\s+/[sfq]\b", r"\bformat\s+[a-z]:", r"\bmkfs(?:\.\w+)?\b", r"\bdd\s+if=",
    r"\bdrop\s+(?:table|database|schema)\b", r"\btruncate\s+table\b", r"\bdelete\s+from\s+\w+\s*;?\s*$",
    r"\bshutdown\s+(?:[-/]\w|now\b)", r"\bkill\s+-9\b", r"\btaskkill\b", r"\bpkill\b", r"\bkillall\b",
    r"\bdiskpart\b", r"\bremove-item\b[^\n]*-recurse", r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
    r"\bchmod\s+-r\s+777\s+/", r"\bcipher\s+/w\b", r"\breg\s+delete\b",
]

# --- English: verb followed (within a few words) by a sensitive object -----------------------------------
_EN_VERBS = (r"(?:delete|deleting|erase|erasing|wipe|wiping|destroy|destroying|purge|purging|remove|removing|"
             r"format|formatting|reformat|uninstall|kill|terminate|shut\s*down|shutdown|power\s*off|reboot|"
             r"restart|drop|corrupt|encrypt)")
_EN_OBJECTS = (r"(?:files?|folders?|director(?:y|ies)|dirs?|disks?|drives?|hard\s*drives?|partitions?|c:|"
               r"system32|os|operating\s+system|databases?|db|everything|registry|users?|accounts?|computer|pc|"
               r"laptop|machine|server|process(?:es)?|pid|services?|(?:all|my|the\s+entire)\s+data|backups?|home|"
               r"documents|desktop|repo(?:sitory)?|windows|boot|mbr)")
_EN_RULE = rf"\b{_EN_VERBS}\b\W+(?:[\w'.-]+\W+){{0,4}}?{_EN_OBJECTS}(?=\W|$)"

# --- Turkish: object followed by verb (SOV word order), plus verb-first forms -----------------------------
# (verb, object) groups so that e.g. "programı kapat" (close a program) is not treated like "bilgisayarı kapat".
_TR_DESTROY_V = (r"(?:sil(?:in|iniz|mek|meyi|meni|meliyim|eyim|elim|sene|iver\w*|ebilir\w*|er\s*mi\w*|ecek\w*|di\w*)?|"
                 r"biçimlendir\w*|formatla\w*|format\s+at\w*|yok\s+et\w*|boşalt\w*|uçur\w*|imha\s+et\w*|kaldır\w*)")
_TR_DESTROY_O = (r"(?:dosya\w*|klasör\w*|dizin\w*|disk\w*|sürücü\w*|veritaban\w*|bilgisayar\w*|sistem\w*|"
                 r"sunucu\w*|her\s*şey\w*|hepsin\w*|tümün\w*|(?:tüm|bütün)\s+veri\w*|kayıt\s*defter\w*|hesab\w*|"
                 r"kullanıcı\w*|yedek\w*|belge\w*|masaüstü\w*|windows\w*|c:)")
_TR_POWER_V = r"(?:kapat\w*|yeniden\s+başlat\w*)"
_TR_POWER_O = r"(?:bilgisayar\w*|sistem\w*|sunucu\w*|makine\w*|pc\w*|laptop\w*|windows\w*)"
_TR_KILL_V = r"(?:öldür\w*|sonlandır\w*)"
_TR_KILL_O = r"(?:işlem\w*|süreç\w*|program\w*|görev\w*|servis\w*|hizmet\w*|sunucu\w*)"


def _tr_rules(verbs: str, objects: str) -> list[str]:
    return [rf"\b{objects}\W+(?:[\w'.-]+\W+){{0,3}}?{verbs}(?=\W|$)",
            rf"\b{verbs}\W+(?:[\w'.-]+\W+){{0,2}}?{objects}(?=\W|$)"]


_TR_RULES = (_tr_rules(_TR_DESTROY_V, _TR_DESTROY_O) + _tr_rules(_TR_POWER_V, _TR_POWER_O)
             + _tr_rules(_TR_KILL_V, _TR_KILL_O) + [r"\bformat\s+at\w*"])

_COMMANDS_RE = [re.compile(p, _F) for p in _COMMAND_PATTERNS]
_EN_RE = re.compile(_EN_RULE, _F)
_TR_RE = [re.compile(p, _F) for p in _TR_RULES]

_CONFIRM_RE = re.compile(r"^\W*(?:CONFIRM|ONAYLA)\b|\b(?:CONFIRM|ONAYLA)\W*$")  # case-sensitive on purpose

_TR_HINT_RE = re.compile(r"[çğışöüÇĞİŞÖÜ]|\b(?:ve|bir|bu|şu|tüm|bütün|hepsini|lütfen|mı|mi|mu|mü|benim|"
                         r"bilgisayarımı|dosyaları|sil|kapat)\b", _F)


def find_danger(text: str) -> tuple[str, str] | None:
    """Return (matched_text, lang) if ``text`` looks like a destructive request, else None. lang: 'en'|'tr'."""
    if not text:
        return None
    for rx in _COMMANDS_RE:
        m = rx.search(text)
        if m:
            return m.group(0), "en"
    for rx in _TR_RE:
        m = rx.search(text)
        if m:
            return m.group(0), "tr"
    m = _EN_RE.search(text)
    if m:
        return m.group(0), "en"
    return None


def is_confirmed(text: str) -> bool:
    """True only if the message starts or ends with the exact uppercase token CONFIRM or ONAYLA."""
    return bool(_CONFIRM_RE.search((text or "").strip()))


def _language(text: str, matched_lang: str, settings) -> str:
    if matched_lang == "tr" or _TR_HINT_RE.search(text or ""):
        return "tr"
    return "en"


_PROMPT_EN = (
    "You are the Security Officer of Neurox, a local AI assistant. The user's last message looks like a potentially "
    "destructive request (detected: \"{match}\"). It has NOT been executed and you have NO tools in this turn.\n"
    "Reply in English, briefly and politely:\n"
    "1. Say that this action could cause irreversible damage (data loss, system changes) and was paused.\n"
    "2. Tell the user that if they really want it, they must send the request again with the word CONFIRM "
    "(uppercase) at the very beginning or end of the message. Even then, every risky tool step will ask for approval.\n"
    "3. Offer a safer alternative if one exists (e.g. back up first, move to trash instead of deleting).\n"
    "Do not perform or simulate the action and do not give step-by-step destructive commands."
)
_PROMPT_TR = (
    "Sen Neurox adlı yerel yapay zekâ asistanının Güvenlik Görevlisisin. Kullanıcının son mesajı potansiyel olarak "
    "yıkıcı bir istek gibi görünüyor (tespit edilen: \"{match}\"). İstek ÇALIŞTIRILMADI ve bu turda HİÇBİR aracın yok.\n"
    "Türkçe, kısa ve kibar bir şekilde yanıt ver:\n"
    "1. Bu işlemin geri alınamaz zarara (veri kaybı, sistem değişikliği) yol açabileceğini ve durdurulduğunu söyle.\n"
    "2. Gerçekten istiyorsa isteği, mesajın en başına veya en sonuna büyük harflerle ONAYLA (veya CONFIRM) yazarak "
    "tekrar göndermesi gerektiğini söyle. O durumda bile riskli her araç adımı ayrıca onay isteyecek.\n"
    "3. Varsa daha güvenli bir alternatif öner (ör. önce yedek almak, silmek yerine çöp kutusuna taşımak).\n"
    "İşlemi gerçekleştirme veya taklit etme, yıkıcı komutları adım adım verme."
)


class SecurityPlugin(NeuroxPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Security Plugin"
        self.description = "Intercepts destructive requests and requires an explicit CONFIRM / ONAYLA."
        self.order = 1  # run first

    def execute(self, context: dict) -> dict | None:
        user_input = context.get("user_input") or ""
        if not user_input.strip():
            return None
        hit = find_danger(user_input)
        if not hit:
            return None
        match, lang = hit
        trace = context.get("trace")
        if is_confirmed(user_input):
            if isinstance(trace, list):
                trace.append({"type": "plugin", "name": self.name, "info": f"Destructive request confirmed: {match!r}"})
            return {"security_confirmed": True}
        lang = _language(user_input, lang, context.get("settings"))
        prompt = (_PROMPT_TR if lang == "tr" else _PROMPT_EN).format(match=match.strip()[:80])
        if isinstance(trace, list):
            trace.append({"type": "plugin", "name": self.name, "info": f"Blocked destructive request: {match!r}"})
        return {
            "blocked": True,
            "active_agent": "SECURITY",
            "security_reason": match.strip(),
            "search_query": None,
            "tool_output": None,
            "system_prompt": prompt,
        }
