"""Chat history persistence (``core.config.HISTORY_FILE``).

API: ``load_history() -> list[dict]``, ``save_history(messages) -> bool``, ``clear_history() -> None``.
Each accepts an optional ``path`` for tests.

Messages are sanitised before saving: ``ToolCall`` objects / dataclasses become dicts, values that
are not JSON-serialisable are dropped, and attached images are capped (count and size) so the file
stays small. Tool messages are kept.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import tempfile
from pathlib import Path

from core.config import HISTORY_FILE

log = logging.getLogger(__name__)

MAX_IMAGES_PER_MESSAGE = 4
MAX_IMAGE_CHARS = 1_500_000      # ~1.1 MB of binary data per base64 image
_DROP = object()


def _resolve(path) -> Path:
    return Path(path) if path is not None else Path(HISTORY_FILE)


def _jsonable(value, depth: int = 0):
    """Convert ``value`` to plain JSON types, or return ``_DROP`` if impossible."""
    if depth > 20:
        return _DROP
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _jsonable(value.to_dict(), depth + 1)
        except Exception:
            return _DROP
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return _jsonable(dataclasses.asdict(value), depth + 1)
        except Exception:
            return _DROP
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                continue
            v = _jsonable(v, depth + 1)
            if v is not _DROP:
                out[k] = v
        return out
    if isinstance(value, (list, tuple)):
        out = [_jsonable(v, depth + 1) for v in value]
        return [v for v in out if v is not _DROP]
    return _DROP


def _sanitize_message(m) -> dict | None:
    if not isinstance(m, dict) or not m.get("role"):
        return None
    msg = _jsonable({k: v for k, v in m.items() if k != "images"})
    if not isinstance(msg, dict):
        return None
    if msg.get("content") is None:
        msg["content"] = ""
    images = m.get("images") or []
    if images:
        kept = [img for img in images if isinstance(img, str) and len(img) <= MAX_IMAGE_CHARS]
        kept = kept[-MAX_IMAGES_PER_MESSAGE:]
        if kept:
            msg["images"] = kept
        if len(kept) < len(images):
            msg["images_dropped"] = len(images) - len(kept)
    return msg


def sanitize_history(messages) -> list[dict]:
    out = []
    for m in messages or []:
        s = _sanitize_message(m)
        if s is not None:
            out.append(s)
    return out


def load_history(path=None) -> list[dict]:
    p = _resolve(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8") or "[]")
    except Exception as e:
        log.warning("Could not read chat history %s: %s", p, e)
        return []
    if not isinstance(data, list):
        return []
    return [m for m in data if isinstance(m, dict) and m.get("role")]


def save_history(messages, path=None) -> bool:
    p = _resolve(path)
    try:
        data = sanitize_history(messages)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=str(p.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, p)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        return True
    except Exception as e:
        log.error("Could not save chat history %s: %s", p, e)
        return False


def clear_history(path=None) -> None:
    p = _resolve(path)
    try:
        p.unlink(missing_ok=True)
    except OSError as e:
        log.warning("Could not delete chat history %s: %s", p, e)
