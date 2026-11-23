"""Neurox — Streamlit chat UI.

Run with:  streamlit run app.py

Layout: sidebar "Neurox Cortex" (connection, models, tools, computer use, teach/rules/memory/XP),
main area = chat history + live "thinking" panel + human-in-the-loop approval card for dangerous tools.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # plugin discovery and some helpers use paths relative to the project root

import streamlit as st  # noqa: E402

from core.config import PROVIDERS, Settings  # noqa: E402
from core.llm import get_client  # noqa: E402
from ui import components as ui  # noqa: E402
from ui import theme  # noqa: E402
from ui.i18n import LANGUAGES, set_language, t  # noqa: E402
from utils.history_manager import clear_history, load_history, save_history  # noqa: E402

st.set_page_config(page_title="Neurox", page_icon="🧠", layout="wide")

ss = st.session_state


# ===================================================================================== helpers
def _optional(module: str, name: str):
    """Import ``module.name`` if it exists (other parts of Neurox may not be installed/ready)."""
    try:
        return getattr(importlib.import_module(module), name)
    except Exception:
        return None


@st.cache_data(ttl=10, show_spinner=False)
def fetch_models(provider: str, base_url: str, api_key: str) -> tuple[list[str], str]:
    """(model names, error message). Cached ~10s so the sidebar doesn't hammer the server."""
    try:
        client = get_client(Settings(provider=provider, base_url=base_url, api_key=api_key))
        return list(client.list_models()), ""
    except Exception as e:
        return [], str(e) or type(e).__name__


@st.cache_data(ttl=30, show_spinner=False)
def computer_status() -> tuple[bool, str]:
    try:
        from tools.computer import is_available
        ok, reason = is_available()
        return bool(ok), str(reason or "")
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def available_tool_names(settings: Settings, llm) -> list[str]:
    cached = ss.get("_tool_names")
    if not cached:
        found: list[str] = []
        for mod_name in ("tools.builtin", "tools.memory_tools"):
            try:
                found += [tool.name for tool in importlib.import_module(mod_name).get_tools(settings, llm)]
            except Exception:
                pass
        if found:
            ss["_tool_names"] = cached = found
    names = list(cached or [])
    for n in list(Settings().enabled_tools) + list(settings.enabled_tools):
        if n not in names:
            names.append(n)
    return names


def _match_model(name: str, models: list[str]) -> str | None:
    if name in models:
        return name
    if f"{name}:latest" in models:
        return f"{name}:latest"
    return None


def _xp_progress(state: dict) -> tuple[int, int, float]:
    level = int(state.get("level", 1) or 1)
    xp = int(state.get("xp", 0) or 0)
    progress = state.get("progress")
    if progress is None:
        nxt = state.get("next_level_xp") or state.get("xp_for_next_level") or state.get("xp_needed")
        cur = state.get("level_xp") or state.get("current_level_xp") or 0
        progress = (xp - cur) / (nxt - cur) if nxt and nxt > cur else (xp % 100) / 100
    progress = float(progress)
    if progress > 1:  # percentage
        progress /= 100
    return level, xp, min(max(progress, 0.0), 1.0)


# ===================================================================================== state
def _load_display_history() -> list[dict]:
    try:
        raw = load_history() or []
    except Exception:
        return []
    out = []
    for m in raw:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content", ""), str):
            out.append(m)
    return out


def init_state() -> None:
    if "settings" not in ss:
        ss.settings = Settings.load()
    if "neurox" not in ss:
        try:
            from core.pipeline import Neurox
            ss.neurox = Neurox(ss.settings)
            ss.init_error = None
        except Exception as e:
            ss.neurox, ss.init_error = None, f"{type(e).__name__}: {e}"
    if "messages" not in ss:
        ss.messages = _load_display_history()
    ss.setdefault("pending", None)       # approval waiting: {"result", "registry", "events", "text", "thoughts"}
    ss.setdefault("toast", None)
    ss.setdefault("mem_nonce", 0)


def apply_settings(new: Settings) -> None:
    old = ss.settings
    try:
        new.save()
    except Exception as e:
        st.sidebar.error(str(e))
    ss.settings = new
    if ss.neurox is not None:
        try:
            ss.neurox.reconfigure(new)
        except Exception as e:
            st.sidebar.error(f"reconfigure: {e}")
    if (new.tools_enabled, new.computer_use_enabled) != (old.tools_enabled, old.computer_use_enabled):
        ss.pop("_tool_names", None)
    if new.language != old.language:
        set_language(new.language)
        st.rerun()


def save_messages() -> None:
    try:
        save_history(ss.messages)
    except Exception:
        pass


# ===================================================================================== sidebar
def render_connection(s: Settings) -> dict:
    providers = list(PROVIDERS)
    provider = st.selectbox(t("provider"), providers,
                            index=providers.index(s.provider) if s.provider in providers else 0,
                            format_func=lambda p: t(f"provider_{p}"))
    url_default = PROVIDERS[provider] if provider != s.provider else (s.base_url or PROVIDERS[provider])
    base_url = st.text_input(t("base_url"), value=url_default).strip() or PROVIDERS[provider]
    api_key = s.api_key
    if provider != "ollama":
        api_key = st.text_input(t("api_key"), value=s.api_key, type="password")

    models, error = fetch_models(provider, base_url, api_key)
    if st.button(t("refresh"), key="refresh_models", type="tertiary"):
        fetch_models.clear()
        computer_status.clear()
        st.rerun()
    if error:
        st.caption(t(f"hint_{provider}"))
        with st.expander(t("error_details")):
            st.code(error[:500], language="text")
    return {"provider": provider, "base_url": base_url, "api_key": api_key,
            "models": models, "reachable": not error}


def render_models(s: Settings, conn: dict) -> dict:
    models = conn["models"]
    if conn["reachable"] and models:
        current = _match_model(s.chat_model, models)
        options = models if current else [s.chat_model] + models
        chat_model = st.selectbox(t("chat_model"), options, index=options.index(current or s.chat_model))
        if chat_model not in models:
            st.caption(t("model_missing", model=chat_model))
    else:
        if conn["reachable"]:
            st.info(t("no_models"))
        chat_model = st.text_input(t("chat_model_free"), value=s.chat_model).strip() or s.chat_model
    embed_model = st.text_input(t("embed_model"), value=s.embed_model).strip() or s.embed_model
    temperature = st.slider(t("temperature"), 0.0, 1.5, float(s.temperature), 0.05)
    return {"chat_model": chat_model, "embed_model": embed_model, "temperature": float(temperature)}


def render_tools(s: Settings, llm) -> dict:
    tools_enabled = st.toggle(t("tools_enabled"), value=s.tools_enabled)
    names = available_tool_names(s, llm)
    enabled = st.multiselect(t("enabled_tools"), names, default=[n for n in s.enabled_tools if n in names],
                             disabled=not tools_enabled)
    max_steps = st.number_input(t("max_agent_steps"), min_value=1, max_value=50, value=int(s.max_agent_steps), step=1)
    return {"tools_enabled": tools_enabled, "enabled_tools": list(enabled), "max_agent_steps": int(max_steps)}


def render_computer(s: Settings) -> dict:
    computer = st.toggle(t("computer_use"), value=s.computer_use_enabled)
    if computer:
        st.warning(t("computer_use_warning"), icon="⚠️")
    ok, reason = computer_status()
    st.caption(t("computer_available") if ok else t("computer_unavailable", reason=reason[:200]))

    auto_approve = st.checkbox(t("auto_approve"), value=s.auto_approve)
    if auto_approve:
        st.error(t("auto_approve_warning"), icon="🚨")
    return {"computer_use_enabled": computer, "auto_approve": auto_approve}


def render_display(s: Settings) -> dict:
    show_thoughts = st.toggle(t("show_thoughts"), value=s.show_thoughts)
    langs = list(LANGUAGES)
    language = st.radio(t("language"), langs, index=langs.index(s.language) if s.language in langs else 0,
                        format_func=LANGUAGES.get, horizontal=True)
    return {"show_thoughts": show_thoughts, "language": language}


def render_xp() -> None:
    get_xp_state = _optional("plugins.misc_plugins", "get_xp_state")
    if get_xp_state is None:
        return
    try:
        state = get_xp_state() or {}
        level, xp, progress = _xp_progress(state if isinstance(state, dict) else vars(state))
    except Exception:
        return
    theme.xp_card(t("level"), level, xp, progress)


def render_teach(neurox) -> None:
    with st.popover(t("teach_title"), width="stretch"):
        st.caption(t("teach_help"))
        with st.form("teach_form", clear_on_submit=True, border=False):
            text = st.text_area(t("teach_title"), label_visibility="collapsed")
            submitted = st.form_submit_button(t("teach_submit"))
        if submitted:
            if not text.strip():
                st.warning(t("teach_empty"))
            elif neurox is None:
                st.error(t("teach_error", error=ss.init_error))
            else:
                try:
                    neurox.submit_feedback(text.strip())
                    st.success(t("teach_saved"))
                except Exception as e:
                    st.error(t("teach_error", error=e))


def _rule_fields(rule, index: int) -> tuple[object, str]:
    if isinstance(rule, dict):
        rid = rule.get("id", index)
        return rid, str(rule.get("text") or rule.get("rule") or rule.get("content") or rule)
    return index, str(rule)


def render_rules() -> None:
    with st.popover(t("rules_title"), width="stretch"):
        load_rules = _optional("plugins.misc_plugins", "load_rules")
        add_rule = _optional("plugins.misc_plugins", "add_rule")
        delete_rule = _optional("plugins.misc_plugins", "delete_rule")
        if load_rules is None:
            st.caption(t("rules_unavailable"))
            return
        try:
            rules = list(load_rules() or [])
        except Exception as e:
            st.caption(f"{t('rules_unavailable')} ({e})")
            rules = []
        if not rules:
            st.caption(t("rules_empty"))
        for i, rule in enumerate(rules):
            rid, text = _rule_fields(rule, i)
            c1, c2 = st.columns([6, 1], vertical_alignment="center")
            c1.markdown(f"• {text}")
            if delete_rule is not None and c2.button("🗑️", key=f"del_rule_{rid}_{i}", help=t("delete")):
                try:
                    delete_rule(rid)
                except Exception as e:
                    st.error(str(e))
                st.rerun()
        if add_rule is not None:
            with st.form("add_rule_form", clear_on_submit=True, border=False):
                new_rule = st.text_input(t("rules_add"))
                if st.form_submit_button(t("rules_add_button")) and new_rule.strip():
                    try:
                        add_rule(new_rule.strip())
                    except Exception as e:
                        st.error(str(e))
                    st.rerun()


def _memory_store(neurox):
    llm = getattr(neurox, "llm", None)
    get_store = _optional("utils.vector_store", "get_store")  # shared instance, same one the memory plugin uses
    if get_store is not None:
        try:
            return get_store(llm)
        except Exception:
            pass
    VectorStore = _optional("utils.vector_store", "VectorStore")
    if VectorStore is None:
        return None
    try:
        return VectorStore(llm)
    except TypeError:
        return VectorStore()


def render_memory(neurox) -> None:
    with st.popover(t("memory_title"), width="stretch"):
        try:
            store = _memory_store(neurox)
            items = store.list() if store is not None and hasattr(store, "list") else None
        except Exception:
            store, items = None, None
        if store is None or items is None:
            st.caption(t("memory_unavailable"))
            return
        st.markdown(t("memory_count", count=len(items)))
        confirm = st.checkbox(t("memory_confirm"), key=f"mem_confirm_{ss.mem_nonce}")
        if st.button(t("memory_clear"), disabled=not confirm, key="mem_clear"):
            try:
                store.clear()
                st.toast(t("memory_cleared"), icon="🧹")
            except Exception as e:
                st.error(str(e))
            ss.mem_nonce += 1
            st.rerun()


def render_sidebar(neurox) -> None:
    s: Settings = ss.settings
    with st.sidebar:
        theme.brand(t("app_title"), "Cortex · v2")
        if st.button(t("new_chat"), key="nx_new_chat", width="stretch"):
            try:
                clear_history()
            except Exception:
                pass
            ss.messages = []
            ss.pending = None
            st.rerun()
        render_xp()
        status_slot = st.container()

        with st.expander(t("section_connection")):
            conn = render_connection(s)
        with status_slot:
            ss.conn_ok = conn["reachable"]
            theme.status(t("connected", count=len(conn["models"])).lstrip("✅ ") if conn["reachable"]
                         else t("not_connected").lstrip("❌ "), conn["reachable"])
        with st.expander(t("section_models"), expanded=not conn["reachable"] or not conn["models"]):
            model_cfg = render_models(s, conn)
        with st.expander(t("section_tools")):
            tool_cfg = render_tools(s, getattr(neurox, "llm", None))
        with st.expander(t("section_computer")):
            computer_cfg = render_computer(s)
        with st.expander(t("section_display")):
            display_cfg = render_display(s)
        new = replace(s, provider=conn["provider"], base_url=conn["base_url"], api_key=conn["api_key"],
                      **model_cfg, **tool_cfg, **computer_cfg, **display_cfg)
        if new != s:
            apply_settings(new)

        st.caption(t("section_learning"))
        render_teach(neurox)
        render_rules()
        render_memory(neurox)
        st.divider()
        st.caption(t("footer"))


# ===================================================================================== chat
class LiveTurn:
    """Thinking panel + streaming placeholder for one assistant turn (or one resume step)."""

    def __init__(self):
        self.status = st.status(t("thinking"), expanded=False)
        self.placeholder = st.empty()
        self.text = ""
        self.events: list = []
        self._seen: set[str] = set()

    def on_token(self, token: str) -> None:
        self.text += token or ""
        self.placeholder.markdown(self.text + "▌")

    def on_event(self, ev) -> None:
        self.events.append(ev)
        self._seen.add(ui.event_key(ev))
        with self.status:
            ui.render_event(ev)
        if isinstance(ev, dict) and ev.get("type") == "tool_call":
            self.status.update(label=f"🔧 {ev.get('name', '')}…")

    def add_trace(self, trace) -> None:
        """Render trace entries (plugin steps etc.) that did not already arrive through on_event."""
        for ev in trace or []:
            if ui.event_key(ev) not in self._seen:
                self.on_event(ev)

    def finish(self, state: str = "complete", label: str | None = None) -> None:
        self.status.update(label=label or t("done"), state=state, expanded=False)


def build_canonical(messages: list[dict]) -> list[dict]:
    """UI history -> canonical messages for the pipeline. Only the latest user message keeps its images."""
    out = [{"role": m["role"], "content": m.get("content") or ""} for m in messages
           if m.get("role") in ("user", "assistant")]
    if messages and messages[-1].get("role") == "user" and messages[-1].get("images"):
        out[-1]["images"] = list(messages[-1]["images"])
    return out


def finish_turn(text: str, images: list[str] | None = None, thoughts: str | None = None,
                toast: str | None = None) -> None:
    if text and text.startswith("⚠️ LLM error") and "not reachable" in text:
        thoughts = (str(thoughts) + "\n\n" if thoughts else "") + text  # keep the raw error in the inner monologue
        text = t("friendly_unreachable")
    msg = {"role": "assistant", "content": text or t("empty_reply")}
    if images:
        msg["images"] = list(images)[-ui.MAX_STORED_IMAGES:]
    if thoughts:
        msg["thoughts"] = str(thoughts)
    ss.messages.append(msg)
    save_messages()
    ss.toast = toast
    st.rerun()  # re-render history and refresh the sidebar (XP, memory count)


def _store_pending(result, registry, events: list, text: str, thoughts) -> None:
    if registry is None:
        try:
            from tools import build_registry
            registry = build_registry(ss.settings, ss.neurox.llm)
        except Exception:
            registry = None
    ss.pending = {"result": result, "registry": registry, "events": events, "text": text, "thoughts": thoughts}


def run_turn_ui(neurox, content: str, images: list[str]) -> None:
    canonical = build_canonical(ss.messages)
    with st.chat_message("assistant", avatar="🧠"):
        theme.marker("assistant")
        live = LiveTurn()
        try:
            ctx = neurox.run_turn(content, canonical, images=images, on_token=live.on_token, on_event=live.on_event)
        except Exception as e:
            live.finish("error", t("turn_error"))
            finish_turn(t("error_reply", error=f"{type(e).__name__}: {e}"))
            return
        live.add_trace(ctx.get("trace"))
        result = ctx.get("agent_result")
        thoughts = ctx.get("inner_monologue") if ss.settings.show_thoughts else None
        if result is not None and getattr(result, "pending_call", None) is not None:
            live.finish("running", t("waiting_approval"))
            _store_pending(result, ctx.get("registry"), live.events, live.text, thoughts)
            st.rerun()
        live.finish()
        final = ctx.get("final_response") or getattr(result, "final_text", "") or live.text
        out_images = list(getattr(result, "images", None) or []) or ui.collect_tool_images(live.events)
        finish_turn(final, out_images, thoughts, ctx.get("gamification_msg"))


def _agent_steps(s: Settings, registry) -> int:
    """Same step budget the chat plugin used for this turn (computer use needs more)."""
    try:
        from plugins.chat_plugin import COMPUTER_USE_MIN_STEPS, _has_computer_tools
        if _has_computer_tools(registry):
            return max(s.max_agent_steps, COMPUTER_USE_MIN_STEPS)
    except Exception:
        pass
    return s.max_agent_steps


def resume_ui(neurox, approved: bool) -> None:
    p = ss.pending
    s: Settings = ss.settings
    try:
        from core.agent import resume_agent
    except ImportError as e:
        ss.pending = None
        finish_turn(t("agent_missing", error=e))
        return
    live = LiveTurn()
    try:
        result = resume_agent(neurox.llm, p["result"], p["registry"], approved,
                              on_token=live.on_token, auto_approve=s.auto_approve,
                              on_event=live.on_event, max_steps=_agent_steps(s, p["registry"]))
    except Exception as e:
        live.finish("error", t("turn_error"))
        ss.pending = None
        finish_turn(t("error_reply", error=f"{type(e).__name__}: {e}"), thoughts=p.get("thoughts"))
        return
    events = list(p.get("events") or []) + live.events
    if getattr(result, "pending_call", None) is not None:
        live.finish("running", t("waiting_approval"))
        _store_pending(result, p["registry"], events, live.text or p.get("text", ""), p.get("thoughts"))
        st.rerun()
    live.finish()
    ss.pending = None
    images = list(getattr(result, "images", None) or []) or ui.collect_tool_images(events)
    finish_turn(getattr(result, "final_text", "") or live.text, images, p.get("thoughts"))


def render_approval(neurox) -> None:
    p = ss.pending
    call = getattr(p.get("result"), "pending_call", None)
    if call is None:
        ss.pending = None
        return
    name = getattr(call, "name", "?")
    args = getattr(call, "arguments", {}) or {}
    with st.chat_message("assistant", avatar="🧠"):
        theme.marker("assistant")
        if p.get("text"):
            st.markdown(p["text"])
        if p.get("events"):
            with st.expander(t("steps")):
                for ev in p["events"]:
                    ui.render_event(ev)
        with st.container(key="nx_approval"):
            st.warning(t("approval_title", tool=name), icon="⚠️")
            st.caption(t("approval_help"))
            st.code(json.dumps(args, indent=2, ensure_ascii=False, default=str), language="json")
            c1, c2 = st.columns(2)
            approve = c1.button(t("approve"), type="primary", key="approve_call", width="stretch")
            deny = c2.button(t("deny"), key="deny_call", width="stretch")
        if approve or deny:
            if neurox is None:
                st.error(ss.init_error)
                return
            resume_ui(neurox, approved=approve)


def _parse_chat_value(value) -> tuple[str, list]:
    if isinstance(value, str):
        return value.strip(), []
    text = getattr(value, "text", None)
    files = getattr(value, "files", None)
    if text is None and hasattr(value, "get"):
        text, files = value.get("text"), value.get("files")
    return (text or "").strip(), list(files or [])


def handle_submission(neurox, value) -> None:
    text, files = _parse_chat_value(value)
    images, attached, names = ui.process_uploads(files)
    if not (text or images or attached):
        return
    content = "\n\n".join(part for part in (text, attached) if part)
    user_msg: dict = {"role": "user", "content": content}
    if images:
        user_msg["images"] = images
    ss.messages.append(user_msg)
    save_messages()
    ui.render_message(user_msg)
    run_turn_ui(neurox, content, images)


def render_header(s: Settings) -> None:
    online = bool(ss.get("conn_ok"))
    theme.hero(t("app_title"), t("app_subtitle", provider=t(f"provider_{s.provider}"), model=s.chat_model).split(" · ")[0],
               s.chat_model, online, t("status_online") if online else t("status_offline"))


SUGGESTIONS = ("sugg1", "sugg2", "sugg3", "sugg4")


def render_welcome(enabled: bool) -> None:
    theme.welcome(t("welcome_title"), t("welcome_sub"))
    with st.container(key="nx_suggestions"):
        cols = st.columns(2)
        for i, key in enumerate(SUGGESTIONS):
            label = f"**{t(key + '_title')}**\n\n{t(key + '_prompt')}"
            if cols[i % 2].button(label, key=f"nx_{key}", width="stretch", disabled=not enabled):
                ss.queued_prompt = t(key + "_prompt")
                st.rerun()


def render_history(show_thoughts: bool, enabled: bool = True) -> None:
    if not ss.messages and not ss.pending and not ss.get("queued_prompt"):
        render_welcome(enabled)
    for msg in ss.messages:
        ui.render_message(msg, show_thoughts=show_thoughts)


# ===================================================================================== main
def main() -> None:
    init_state()
    theme.inject_css()
    s: Settings = ss.settings
    set_language(s.language)
    if ss.toast:
        st.toast(ss.toast, icon="🎉")
        ss.toast = None

    neurox = ss.neurox
    render_sidebar(neurox)
    s = ss.settings  # may have changed in the sidebar

    render_header(s)
    if neurox is None:
        st.error(t("init_error", error=ss.init_error))
    blocked = bool(ss.pending) or neurox is None
    render_history(s.show_thoughts, enabled=not blocked)
    if ss.pending:
        render_approval(neurox)

    value = st.chat_input(t("chat_disabled_placeholder") if ss.pending else t("chat_placeholder"),
                          accept_file="multiple", file_type=ui.UPLOAD_TYPES, disabled=blocked)
    queued = ss.pop("queued_prompt", None)
    if not blocked and (value or queued):
        handle_submission(neurox, value or queued)


main()
