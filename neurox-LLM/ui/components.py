"""Reusable Streamlit rendering helpers and attachment processing for app.py."""
from __future__ import annotations

import base64
import binascii
import io
import json
from pathlib import Path

import streamlit as st

from ui.i18n import t

IMAGE_TYPES = ["png", "jpg", "jpeg", "webp", "gif", "bmp"]
TEXT_TYPES = ["txt", "md", "py", "json", "csv"]
UPLOAD_TYPES = IMAGE_TYPES + TEXT_TYPES

MAX_TEXT_CHARS = 20_000        # per attached text file
MAX_IMAGE_SIDE = 1568          # downscale big images before sending them to a vision model
MAX_RESULT_CHARS = 1_500       # tool result text shown in the thinking panel
MAX_STORED_IMAGES = 2          # tool images (screenshots) kept on an assistant message in history


# ------------------------------------------------------------------------------------------ images
def b64_to_bytes(b64: str) -> bytes | None:
    if not b64:
        return None
    if b64.startswith("data:") and "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        return base64.b64decode(b64)
    except (binascii.Error, ValueError):
        return None


def show_images(images: list[str] | None, width: int | str = "content") -> None:
    for b64 in images or []:
        data = b64_to_bytes(b64)
        if data:
            try:
                st.image(data, width=width)
            except Exception:  # corrupt image data must never break the page
                pass


def _image_to_b64(raw: bytes, ext: str) -> str:
    """PNG/JPEG pass through (unless huge); other formats are converted to PNG for model compatibility."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(raw))
        needs_resize = max(img.size) > MAX_IMAGE_SIDE
        if ext in ("png", "jpg", "jpeg") and not needs_resize:
            return base64.b64encode(raw).decode("ascii")
        if needs_resize:
            img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return base64.b64encode(raw).decode("ascii")


# ------------------------------------------------------------------------------------- attachments
def process_uploads(files) -> tuple[list[str], str, list[str]]:
    """Turn uploaded files into (images_b64, text_to_append, display_names)."""
    images: list[str] = []
    chunks: list[str] = []
    names: list[str] = []
    for f in files or []:
        name = getattr(f, "name", "file")
        ext = Path(name).suffix.lower().lstrip(".")
        try:
            raw = f.getvalue()
        except Exception:
            continue
        if ext in IMAGE_TYPES:
            images.append(_image_to_b64(raw, ext))
            names.append(f"🖼️ {name}")
        elif ext in TEXT_TYPES:
            text = raw.decode("utf-8", errors="replace")
            if len(text) > MAX_TEXT_CHARS:
                text = text[:MAX_TEXT_CHARS] + "\n" + t("attachment_truncated")
            lang = {"py": "python", "md": "markdown"}.get(ext, ext)
            chunks.append(f"📎 **{name}**\n```{lang}\n{text}\n```")
            names.append(f"📄 {name}")
        else:
            st.toast(t("attachment_unsupported", name=name), icon="⚠️")
    return images, "\n\n".join(chunks), names


# -------------------------------------------------------------------------------------- messages
def render_message(msg: dict, show_thoughts: bool = True) -> None:
    role = msg.get("role", "assistant")
    with st.chat_message(role, avatar="🧑" if role == "user" else "🧠"):
        from ui.theme import marker
        marker(role)
        if msg.get("content"):
            st.markdown(msg["content"])
        show_images(msg.get("images"))
        if show_thoughts and msg.get("thoughts"):
            with st.expander(t("inner_monologue")):
                st.markdown(msg["thoughts"])


# ---------------------------------------------------------------------------------------- events
def _short(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + " …"


def event_key(ev) -> str:
    """Stable identity for de-duplicating events seen live and later in ctx['trace']."""
    try:
        return json.dumps(ev, sort_keys=True, default=str, ensure_ascii=False)[:2000]
    except Exception:
        return repr(ev)[:2000]


def render_event(ev) -> None:
    """Render one agent/plugin event inside the current container (e.g. an st.status panel)."""
    if not isinstance(ev, dict):
        st.caption(_short(str(ev), 300))
        return
    kind = ev.get("type", "")
    name = ev.get("name") or ev.get("tool") or ""
    if kind == "plugin":
        info = ev.get("info")
        info_txt = info if isinstance(info, str) else (json.dumps(info, ensure_ascii=False, default=str) if info else "")
        st.markdown(f"⚙️ **{name}**" + (f" — {_short(info_txt, 300)}" if info_txt else ""))
    elif kind == "tool_call":
        st.markdown(t("ev_tool_call", name=name))
        args = ev.get("arguments")
        if args:
            st.code(json.dumps(args, indent=2, ensure_ascii=False, default=str), language="json")
    elif kind == "tool_result":
        st.markdown(t("ev_tool_result", name=name))
        if ev.get("text"):
            st.code(_short(str(ev["text"])), language="text")
        show_images(ev.get("images"))
    elif kind == "approval_needed":
        st.markdown(t("ev_approval", name=name))
    elif kind in ("denied", "tool_denied"):
        st.markdown(t("ev_denied") + (f" · **{name}**" if name else ""))
    elif kind == "error":
        st.error(t("ev_error", text=_short(str(ev.get("text") or ev.get("error") or ""), 500)))
    else:
        text = ev.get("text") or ev.get("info") or ""
        label = f"• **{kind or 'event'}**" + (f" {name}" if name else "")
        st.markdown(label + (f" — {_short(str(text), 300)}" if text else ""))
        show_images(ev.get("images"))


def collect_tool_images(events: list, limit: int = MAX_STORED_IMAGES) -> list[str]:
    imgs: list[str] = []
    for ev in events or []:
        if isinstance(ev, dict) and ev.get("type") == "tool_result":
            imgs.extend(ev.get("images") or [])
    return imgs[-limit:] if limit else imgs
