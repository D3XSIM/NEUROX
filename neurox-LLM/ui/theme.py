"""Visual layer for the Neurox UI: global CSS + small HTML building blocks.

Streamlit's chat_message gives no way to tell user and assistant bubbles apart in CSS, so every
message renders an invisible marker (<span class="nx-user|nx-assistant">) and the CSS uses :has().
"""
from __future__ import annotations

import html

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --nx-bg: #0b0d14;
  --nx-surface: #151925;
  --nx-surface-2: #1b2031;
  --nx-border: #252b3d;
  --nx-text: #e7e9f3;
  --nx-muted: #8b91a7;
  --nx-accent: #8b5cf6;
  --nx-accent-2: #22d3ee;
  --nx-ok: #34d399;
  --nx-bad: #f87171;
  --nx-grad: linear-gradient(135deg, #8b5cf6 0%, #6366f1 50%, #22d3ee 100%);
}

/* ---------- canvas ---------- */
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1100px 520px at 75% -8%, rgba(139, 92, 246, .14), transparent 60%),
    radial-gradient(800px 480px at -5% 105%, rgba(34, 211, 238, .08), transparent 60%),
    var(--nx-bg);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stMainBlockContainer"] { max-width: 900px; padding-top: 1.6rem; padding-bottom: 7rem; }
[data-testid="stElementContainer"]:has(.nx-marker) { display: none; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: #2a3045; border-radius: 10px; border: 2px solid transparent; background-clip: content-box; }

/* ---------- header ---------- */
.nx-hero { display: flex; align-items: center; gap: 14px; margin: .2rem 0 1.4rem; }
.nx-logo {
  width: 46px; height: 46px; border-radius: 14px; display: grid; place-items: center; font-size: 24px;
  background: var(--nx-grad); box-shadow: 0 8px 28px rgba(139, 92, 246, .35);
}
.nx-hero h1 {
  margin: 0 !important; padding: 0 !important; font-size: 1.55rem !important; font-weight: 800 !important;
  letter-spacing: -.02em; line-height: 1.1 !important;
}
.nx-hero p { margin: 2px 0 0; color: var(--nx-muted); font-size: .86rem; }
.nx-hero .nx-grow { flex: 1; }
.nx-chip {
  display: inline-flex; align-items: center; gap: 7px; padding: 6px 12px; border-radius: 999px;
  background: var(--nx-surface); border: 1px solid var(--nx-border); color: var(--nx-text);
  font-size: .8rem; font-weight: 500; white-space: nowrap;
}
.nx-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--nx-bad); box-shadow: 0 0 0 3px rgba(248,113,113,.15); }
.nx-dot.ok { background: var(--nx-ok); box-shadow: 0 0 0 3px rgba(52,211,153,.15); }

/* ---------- welcome ---------- */
.nx-welcome { text-align: center; margin: 4vh 0 1.6rem; }
.nx-welcome .nx-orb {
  width: 76px; height: 76px; margin: 0 auto 18px; border-radius: 24px; display: grid; place-items: center;
  font-size: 38px; background: var(--nx-grad); box-shadow: 0 14px 50px rgba(139, 92, 246, .45);
  animation: nx-float 5s ease-in-out infinite;
}
.nx-welcome h2 {
  font-size: 2rem !important; font-weight: 800 !important; letter-spacing: -.03em; margin: 0 0 .4rem !important;
  background: linear-gradient(90deg, #fff 20%, #c4b5fd 60%, #67e8f9); -webkit-background-clip: text; background-clip: text;
  color: transparent !important;
}
.nx-welcome p { color: var(--nx-muted); max-width: 560px; margin: 0 auto; font-size: .98rem; }
@keyframes nx-float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }

.st-key-nx_suggestions button {
  min-height: 92px; height: 100%; justify-content: flex-start; align-items: flex-start; text-align: left;
  padding: 14px 16px; background: var(--nx-surface); border: 1px solid var(--nx-border); border-radius: 16px;
  transition: transform .15s ease, border-color .15s ease, background .15s ease;
}
.st-key-nx_suggestions button:hover {
  transform: translateY(-2px); border-color: rgba(139, 92, 246, .7); background: var(--nx-surface-2);
}
.st-key-nx_suggestions button p { white-space: normal; text-align: left; line-height: 1.35; }
.st-key-nx_suggestions button p strong { display: block; margin-bottom: 4px; color: var(--nx-text); }

/* ---------- chat bubbles ---------- */
[data-testid="stChatMessage"] { background: transparent !important; padding: .35rem 0; gap: .75rem; }
[data-testid="stChatMessage"] [data-testid^="stChatMessageAvatar"] {
  width: 36px; height: 36px; border-radius: 12px; background: var(--nx-surface-2); border: 1px solid var(--nx-border);
  font-size: 18px;
}
[data-testid="stChatMessage"]:has(.nx-assistant) [data-testid^="stChatMessageAvatar"] {
  background: var(--nx-grad); border: none; box-shadow: 0 4px 16px rgba(139, 92, 246, .35);
}
[data-testid="stChatMessage"]:has(.nx-assistant) [data-testid="stChatMessageContent"] {
  background: var(--nx-surface); border: 1px solid var(--nx-border); border-radius: 4px 18px 18px 18px;
  padding: .85rem 1.1rem;
}
[data-testid="stChatMessage"]:has(.nx-user) { flex-direction: row-reverse; }
[data-testid="stChatMessage"]:has(.nx-user) [data-testid="stChatMessageContent"] {
  flex: 0 1 auto; max-width: 78%; margin-left: auto; border-radius: 18px 4px 18px 18px; padding: .7rem 1.05rem;
  background: linear-gradient(135deg, #7c3aed, #5b21b6); color: #fff; box-shadow: 0 8px 24px rgba(124, 58, 237, .28);
}
[data-testid="stChatMessage"]:has(.nx-user) [data-testid="stChatMessageContent"] * { color: #fff; }
[data-testid="stChatMessageContent"] pre { border: 1px solid var(--nx-border); border-radius: 12px; }

/* thinking panel + expanders */
[data-testid="stExpander"] details {
  border: 1px solid var(--nx-border) !important; border-radius: 14px !important; background: rgba(21, 25, 37, .6);
}
[data-testid="stExpander"] summary { font-weight: 600; }
[data-testid="stExpander"] summary:hover { color: #c4b5fd; }

/* approval card */
.st-key-nx_approval {
  border: 1px solid rgba(251, 191, 36, .45) !important; border-radius: 16px; padding: 4px 6px;
  background: linear-gradient(180deg, rgba(251, 191, 36, .07), rgba(251, 191, 36, .02));
}

/* ---------- input ---------- */
[data-testid="stBottom"] > div { background: transparent; }
[data-testid="stBottomBlockContainer"] {
  max-width: 900px; padding-bottom: 1.4rem;
  background: linear-gradient(180deg, rgba(11, 13, 20, 0), var(--nx-bg) 35%);
}
[data-testid="stChatInput"] {
  border-radius: 18px !important; border: 1px solid var(--nx-border) !important; background: var(--nx-surface) !important;
  box-shadow: 0 10px 40px rgba(0, 0, 0, .35);
}
[data-testid="stChatInput"]:focus-within {
  border-color: rgba(139, 92, 246, .8) !important; box-shadow: 0 0 0 4px rgba(139, 92, 246, .15), 0 10px 40px rgba(0,0,0,.35);
}
/* Streamlit's auto-grow mis-measures the textarea (it can start ~170px tall): size it to its content. */
[data-testid="stChatInputTextArea"] {
  height: auto !important; field-sizing: content; min-height: 1.6rem !important; max-height: 220px !important;
}
[data-testid="stChatInputSubmitButton"] { background: var(--nx-grad) !important; border-radius: 12px !important; color: #fff !important; }
.nx-footer { text-align: center; color: #5d6378; font-size: .72rem; margin-top: -.6rem; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { border-right: 1px solid var(--nx-border); }
[data-testid="stSidebarHeader"] { padding-bottom: 0; }
.nx-brand { display: flex; align-items: center; gap: 10px; margin: -.6rem 0 1rem; }
.nx-brand .nx-logo { width: 38px; height: 38px; border-radius: 12px; font-size: 20px; }
.nx-brand b { font-size: 1.12rem; font-weight: 800; letter-spacing: -.02em; }
.nx-brand span { display: block; color: var(--nx-muted); font-size: .72rem; }
.nx-xp {
  background: var(--nx-surface); border: 1px solid var(--nx-border); border-radius: 14px; padding: 10px 12px; margin-bottom: .8rem;
}
.nx-xp-row { display: flex; justify-content: space-between; align-items: center; font-size: .8rem; color: var(--nx-muted); }
.nx-xp-row b { color: var(--nx-text); font-size: .9rem; }
.nx-badge {
  display: inline-grid; place-items: center; min-width: 26px; height: 22px; padding: 0 7px; border-radius: 8px;
  background: var(--nx-grad); color: #fff; font-weight: 700; font-size: .75rem; margin-right: 6px;
}
.nx-bar { height: 6px; border-radius: 6px; background: #232839; margin-top: 8px; overflow: hidden; }
.nx-bar > div { height: 100%; border-radius: 6px; background: var(--nx-grad); }
.nx-status {
  display: flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 12px; font-size: .82rem; margin-bottom: .8rem;
  background: rgba(248, 113, 113, .08); border: 1px solid rgba(248, 113, 113, .3); color: #fecaca;
}
.nx-status.ok { background: rgba(52, 211, 153, .08); border-color: rgba(52, 211, 153, .3); color: #a7f3d0; }
.st-key-nx_new_chat button { background: var(--nx-grad); border: none; color: #fff; font-weight: 600; }
.st-key-nx_new_chat button:hover { filter: brightness(1.1); color: #fff; }
[data-testid="stSidebar"] [data-testid="stExpander"] details { background: transparent; }
</style>
"""


def inject_css() -> None:
    st.html(CSS)


def marker(role: str) -> None:
    """Invisible element that lets the CSS style user vs. assistant bubbles."""
    st.markdown(f'<span class="nx-marker nx-{"user" if role == "user" else "assistant"}"></span>',
                unsafe_allow_html=True)


def _e(text) -> str:
    return html.escape(str(text))


def hero(title: str, subtitle: str, model: str, online: bool, status_text: str) -> None:
    st.markdown(
        f'<div class="nx-hero"><div class="nx-logo">🧠</div>'
        f'<div class="nx-grow"><h1>{_e(title)}</h1><p>{_e(subtitle)}</p></div>'
        f'<div class="nx-chip"><span class="nx-dot{" ok" if online else ""}"></span>{_e(model)} · {_e(status_text)}</div>'
        f'</div>', unsafe_allow_html=True)


def welcome(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="nx-welcome"><div class="nx-orb">🧠</div><h2>{_e(title)}</h2><p>{_e(subtitle)}</p></div>',
                unsafe_allow_html=True)


def brand(name: str, tagline: str) -> None:
    st.markdown(f'<div class="nx-brand"><div class="nx-logo">🧠</div><div><b>{_e(name)}</b>'
                f'<span>{_e(tagline)}</span></div></div>', unsafe_allow_html=True)


def xp_card(level_label: str, level: int, xp: int, progress: float) -> None:
    pct = max(0.0, min(1.0, progress)) * 100
    st.markdown(f'<div class="nx-xp"><div class="nx-xp-row"><span><span class="nx-badge">{level}</span>{_e(level_label)}</span>'
                f'<b>{xp} XP</b></div><div class="nx-bar"><div style="width:{pct:.0f}%"></div></div></div>',
                unsafe_allow_html=True)


def status(text: str, ok: bool) -> None:
    st.markdown(f'<div class="nx-status{" ok" if ok else ""}"><span class="nx-dot{" ok" if ok else ""}"></span>{_e(text)}</div>',
                unsafe_allow_html=True)


def footer(text: str) -> None:
    st.markdown(f'<div class="nx-footer">{_e(text)}</div>', unsafe_allow_html=True)
