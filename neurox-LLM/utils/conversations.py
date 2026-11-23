"""Saved conversations and projects.

Layout under ``core.config.DATA_DIR`` (read at call time, so tests can redirect it):

    conversations/<id>.json   {"id", "title", "project_id", "created", "updated", "messages": [...]}
    projects.json             [{"id", "name", "instructions", "folder", "created"}]

A project groups conversations and adds custom instructions to the system prompt. If it has a
``folder``, the file tools (list_dir/read_file/write_file) work inside that folder instead of the
default sandbox, so Neurox can work on a real project on disk.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path

import core.config as config
from utils.history_manager import sanitize_history

log = logging.getLogger(__name__)
_lock = threading.RLock()
TITLE_LEN = 48


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _conv_dir() -> Path:
    d = Path(config.DATA_DIR) / "conversations"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _projects_file() -> Path:
    return Path(config.DATA_DIR) / "projects.json"


def _atomic_write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except Exception as e:
        log.warning("Could not read %s: %s", path, e)
        return default


def make_title(text: str) -> str:
    line = " ".join((text or "").split())
    for prefix in ("📎",):
        line = line.split(prefix)[0].strip() or line
    return (line[:TITLE_LEN].rstrip() + "…") if len(line) > TITLE_LEN else (line or "Yeni sohbet")


# ------------------------------------------------------------------------------------ conversations
def new_conversation(project_id: str | None = None, title: str = "") -> dict:
    return {"id": uuid.uuid4().hex[:12], "title": title, "project_id": project_id,
            "created": _now(), "updated": _now(), "messages": []}


def save_conversation(conv: dict) -> None:
    """Persist a conversation (skipped while it has no messages, so empty chats don't pile up)."""
    if not conv.get("messages"):
        return
    with _lock:
        conv["updated"] = _now()
        if not conv.get("title"):
            first = next((m.get("content", "") for m in conv["messages"] if m.get("role") == "user"), "")
            conv["title"] = make_title(first)
        data = dict(conv, messages=sanitize_history(conv["messages"]))
        _atomic_write(_conv_dir() / f"{conv['id']}.json", data)


def load_conversation(conv_id: str) -> dict | None:
    data = _read_json(_conv_dir() / f"{conv_id}.json", None)
    if not isinstance(data, dict):
        return None
    data.setdefault("messages", [])
    return data


def list_conversations(project_id: str | None = None, all_projects: bool = False) -> list[dict]:
    """Newest first, without messages. ``project_id=None`` and not ``all_projects`` → chats outside projects."""
    out = []
    for p in _conv_dir().glob("*.json"):
        d = _read_json(p, None)
        if not isinstance(d, dict) or not d.get("id"):
            continue
        if not all_projects and d.get("project_id") != project_id:
            continue
        out.append({k: d.get(k) for k in ("id", "title", "project_id", "created", "updated")}
                   | {"count": len(d.get("messages") or [])})
    out.sort(key=lambda c: c.get("updated") or "", reverse=True)
    return out


def rename_conversation(conv_id: str, title: str) -> bool:
    conv = load_conversation(conv_id)
    if not conv or not title.strip():
        return False
    conv["title"] = title.strip()[:80]
    save_conversation(conv)
    return True


def delete_conversation(conv_id: str) -> bool:
    try:
        (_conv_dir() / f"{conv_id}.json").unlink()
        return True
    except FileNotFoundError:
        return False


def migrate_legacy_history() -> str | None:
    """Turn the old single chat_history.json into the first saved conversation (once)."""
    legacy = Path(config.HISTORY_FILE)
    if not legacy.exists():
        return None
    try:
        msgs = sanitize_history(_read_json(legacy, []))
        conv = None
        if msgs:
            conv = new_conversation()
            conv["messages"] = msgs
            save_conversation(conv)
        legacy.rename(legacy.with_suffix(".json.migrated"))
        return conv["id"] if conv else None
    except Exception as e:
        log.warning("History migration failed: %s", e)
        return None


# ----------------------------------------------------------------------------------------- projects
def list_projects() -> list[dict]:
    data = _read_json(_projects_file(), [])
    return [p for p in data if isinstance(p, dict) and p.get("id")] if isinstance(data, list) else []


def get_project(project_id: str | None) -> dict | None:
    if not project_id:
        return None
    return next((p for p in list_projects() if p["id"] == project_id), None)


def validate_folder(folder: str) -> tuple[bool, str]:
    """A project folder must be an existing directory (it is NOT created automatically)."""
    folder = (folder or "").strip().strip('"')
    if not folder:
        return True, ""
    p = Path(folder).expanduser()
    if not p.is_absolute():
        return False, "absolute path required"
    if not p.is_dir():
        return False, "folder not found"
    return True, str(p.resolve())


def save_project(name: str, instructions: str = "", folder: str = "", project_id: str | None = None) -> dict:
    ok, folder_or_err = validate_folder(folder)
    if not ok:
        raise ValueError(folder_or_err)
    with _lock:
        projects = list_projects()
        existing = next((p for p in projects if p["id"] == project_id), None) if project_id else None
        if existing:
            existing.update(name=name.strip() or existing["name"], instructions=instructions.strip(), folder=folder_or_err)
            proj = existing
        else:
            proj = {"id": uuid.uuid4().hex[:10], "name": name.strip() or "Proje", "instructions": instructions.strip(),
                    "folder": folder_or_err, "created": _now()}
            projects.append(proj)
        _atomic_write(_projects_file(), projects)
        return proj


def delete_project(project_id: str, delete_chats: bool = False) -> bool:
    with _lock:
        projects = list_projects()
        remaining = [p for p in projects if p["id"] != project_id]
        if len(remaining) == len(projects):
            return False
        _atomic_write(_projects_file(), remaining)
        for c in list_conversations(project_id):
            if delete_chats:
                delete_conversation(c["id"])
            else:  # keep the chats, move them out of the project
                conv = load_conversation(c["id"])
                if conv:
                    conv["project_id"] = None
                    save_conversation(conv)
        return True


def project_workspace(project: dict | None) -> Path | None:
    """Folder the file tools should use for this project (None → default sandbox)."""
    if project and project.get("folder"):
        p = Path(project["folder"])
        if p.is_dir():
            return p
    return None
