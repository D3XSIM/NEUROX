"""Long-term vector memory stored as a JSON file (default: ``core.config.MEMORY_FILE``).

Entry format::

    {"id": str, "text": str, "embedding": [float, ...], "model": str, "dim": int,
     "metadata": dict, "timestamp": ISO-8601 str, "kind": "fact" | "correction" | "conversation"}

Embeddings always come from ``llm.embed`` using ``llm.embed_model``. Entries embedded with another
model (or another dimension) are kept on disk but ignored by ``search`` until ``reindex()`` re-embeds
them with the current model. Legacy entries (old Neurox, no ``model`` field) are treated as ``phi3``.

Nothing here raises on provider errors: failures are logged and ``search`` returns ``[]``,
``add`` returns ``None`` and ``reindex`` returns ``0``.

Use :func:`get_store` to share one instance per (file, embed model) between the memory plugin,
the ``remember``/``recall`` tools and the UI. Instances also reload automatically when the file
was changed by another instance (mtime check), so separate instances stay consistent.
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

import numpy as np

from core.config import MEMORY_FILE

log = logging.getLogger(__name__)

KINDS = ("fact", "correction", "conversation")
LEGACY_MODEL = "phi3"

_file_locks: dict[str, threading.RLock] = {}
_file_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.RLock:
    key = str(path).lower()
    with _file_locks_guard:
        return _file_locks.setdefault(key, threading.RLock())


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _atomic_write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


class VectorStore:
    def __init__(self, llm, path: str | os.PathLike | None = None):
        self.llm = llm
        self.path = Path(path) if path is not None else Path(MEMORY_FILE)
        self._lock = _lock_for(self.path)
        self._entries: list[dict] = []
        self._mtime: float | None = None
        with self._lock:
            self._load()

    # ------------------------------------------------------------------ properties
    @property
    def embed_model(self) -> str:
        return str(getattr(self.llm, "embed_model", "") or "")

    def __len__(self) -> int:
        with self._lock:
            self._maybe_reload()
            return len(self._entries)

    # ------------------------------------------------------------------ persistence
    def _file_mtime(self) -> float | None:
        try:
            return self.path.stat().st_mtime
        except OSError:
            return None

    def _load(self) -> None:
        self._mtime = self._file_mtime()
        if self._mtime is None:
            self._entries = []
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8") or "[]")
        except Exception as e:
            log.error("Could not read memory file %s: %s", self.path, e)
            self._entries = []
            return
        if isinstance(raw, dict):  # tolerate {"entries": [...]}
            raw = raw.get("entries") or []
        entries = []
        for item in raw if isinstance(raw, list) else []:
            e = self._normalize(item)
            if e is not None:
                entries.append(e)
        self._entries = entries

    @staticmethod
    def _normalize(item) -> dict | None:
        if not isinstance(item, dict) or not str(item.get("text") or "").strip():
            return None
        emb = item.get("embedding") or []
        if not isinstance(emb, list):
            emb = []
        kind = item.get("kind")
        if kind not in KINDS:
            # Old Neurox stored "User: ... | Assistant: ..." conversation snippets.
            kind = "conversation" if "model" not in item else "fact"
        return {
            "id": str(item.get("id") or uuid.uuid4().hex[:12]),
            "text": str(item["text"]),
            "embedding": emb,
            "model": str(item.get("model") or LEGACY_MODEL),
            "dim": int(item.get("dim") or len(emb)),
            "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
            "timestamp": str(item.get("timestamp") or _now()),
            "kind": kind,
        }

    def _maybe_reload(self) -> None:
        if self._file_mtime() != self._mtime:
            self._load()

    def _save(self) -> bool:
        try:
            _atomic_write_json(self.path, self._entries)
            self._mtime = self._file_mtime()
            return True
        except Exception as e:
            log.error("Could not save memory file %s: %s", self.path, e)
            return False

    # ------------------------------------------------------------------ embeddings
    def _embed(self, texts: list[str]) -> list[list[float]] | None:
        if not texts:
            return []
        if self.llm is None or not self.embed_model:
            log.warning("No LLM / embed model configured; memory disabled")
            return None
        try:
            vecs = self.llm.embed(list(texts))
        except Exception as e:
            log.warning("Embedding with '%s' failed: %s", self.embed_model, e)
            return None
        if not isinstance(vecs, list) or len(vecs) != len(texts):
            log.warning("Embedding model '%s' returned an unexpected result", self.embed_model)
            return None
        out = []
        for v in vecs:
            try:
                v = [float(x) for x in v]
            except (TypeError, ValueError):
                return None
            if not v:
                return None
            out.append(v)
        return out

    def _matches(self, e: dict, dim: int | None = None) -> bool:
        if e.get("model") != self.embed_model or not e.get("embedding"):
            return False
        return dim is None or e.get("dim") == dim

    # ------------------------------------------------------------------ public API
    def add(self, text: str, kind: str = "fact", metadata: dict | None = None) -> str | None:
        """Embed and store ``text``. Returns the entry id (existing id if the same text is already stored
        with the current model), or ``None`` on failure."""
        text = (text or "").strip()
        if not text:
            return None
        if kind not in KINDS:
            kind = "fact"
        vecs = self._embed([text])
        if not vecs:
            return None
        vec = vecs[0]
        with self._lock:
            self._maybe_reload()
            norm = " ".join(text.lower().split())
            for e in self._entries:
                if " ".join(e["text"].lower().split()) == norm and self._matches(e, len(vec)):
                    if kind == "correction" and e["kind"] != "correction":
                        e["kind"] = "correction"
                        self._save()
                    return e["id"]
            entry = {
                "id": uuid.uuid4().hex[:12],
                "text": text,
                "embedding": vec,
                "model": self.embed_model,
                "dim": len(vec),
                "metadata": dict(metadata or {}),
                "timestamp": _now(),
                "kind": kind,
            }
            self._entries.append(entry)
            if not self._save():
                self._entries.pop()
                return None
            return entry["id"]

    def search(self, query: str, top_k: int = 3, threshold: float = 0.5,
               kinds: list[str] | tuple[str, ...] | None = None) -> list[dict]:
        """Cosine-similarity search over entries embedded with the current model.
        Returns ``[{id, text, kind, score, metadata, timestamp}, ...]`` best first; ``[]`` on any failure."""
        query = (query or "").strip()
        if not query or top_k <= 0:
            return []
        with self._lock:
            self._maybe_reload()
            if not any(self._matches(e) for e in self._entries):
                return []
        vecs = self._embed([query])
        if not vecs:
            return []
        q = np.asarray(vecs[0], dtype=np.float32)
        with self._lock:
            cands = [e for e in self._entries
                     if self._matches(e, len(q)) and len(e["embedding"]) == len(q)
                     and (kinds is None or e["kind"] in kinds)]
            if not cands:
                return []
            try:
                mat = np.asarray([e["embedding"] for e in cands], dtype=np.float32)
                denom = np.linalg.norm(mat, axis=1) * (np.linalg.norm(q) or 1.0)
                denom[denom == 0] = 1.0
                scores = (mat @ q) / denom
            except Exception as e:
                log.error("Memory search failed: %s", e)
                return []
            order = np.argsort(-scores)
            out = []
            for i in order:
                s = float(scores[i])
                if s < threshold or len(out) >= top_k:
                    break
                e = cands[int(i)]
                out.append({"id": e["id"], "text": e["text"], "kind": e["kind"], "score": round(s, 4),
                            "metadata": dict(e["metadata"]), "timestamp": e["timestamp"]})
            return out

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            self._maybe_reload()
            before = len(self._entries)
            self._entries = [e for e in self._entries if e["id"] != str(entry_id)]
            if len(self._entries) == before:
                return False
            return self._save()

    def list(self, kind: str | None = None) -> list[dict]:
        """All entries without their embedding vectors (newest last). ``searchable`` tells whether the entry
        matches the current embed model."""
        with self._lock:
            self._maybe_reload()
            return [{"id": e["id"], "text": e["text"], "kind": e["kind"], "model": e["model"], "dim": e["dim"],
                     "metadata": dict(e["metadata"]), "timestamp": e["timestamp"],
                     "searchable": self._matches(e)}
                    for e in self._entries if kind is None or e["kind"] == kind]

    def clear(self) -> bool:
        with self._lock:
            self._entries = []
            return self._save()

    def reindex(self, batch_size: int = 16) -> int:
        """Re-embed entries whose model/dimension differs from the current embed model.
        Returns the number of re-embedded entries (0 if the model is unavailable)."""
        with self._lock:
            self._maybe_reload()
            todo = [e for e in self._entries if e.get("model") != self.embed_model or not e.get("embedding")]
        done = 0
        for start in range(0, len(todo), batch_size):
            chunk = todo[start:start + batch_size]
            vecs = self._embed([e["text"] for e in chunk])
            if not vecs:
                break
            with self._lock:
                by_id = {e["id"]: e for e in self._entries}
                for e, v in zip(chunk, vecs):
                    target = by_id.get(e["id"])
                    if target is not None:
                        target.update(embedding=v, model=self.embed_model, dim=len(v))
                        done += 1
                self._save()
        return done

    def stats(self) -> dict:
        with self._lock:
            self._maybe_reload()
            searchable = sum(1 for e in self._entries if self._matches(e))
            return {"total": len(self._entries), "searchable": searchable,
                    "stale": len(self._entries) - searchable, "model": self.embed_model}

    # ------------------------------------------------------------------ backwards compatibility
    def add_memory(self, text, metadata=None):
        return self.add(text, kind="conversation", metadata=metadata)

    def search_memory(self, query, top_k=3, threshold=0.5):
        return [r["text"] for r in self.search(query, top_k=top_k, threshold=threshold)]


_stores: dict[tuple[str, str], VectorStore] = {}
_stores_guard = threading.Lock()


def get_store(llm, path: str | os.PathLike | None = None) -> VectorStore:
    """Shared VectorStore per (file, embed model). The store's ``llm`` is refreshed to the given client."""
    p = Path(path) if path is not None else Path(MEMORY_FILE)
    key = (str(p.resolve()).lower(), str(getattr(llm, "embed_model", "") or ""))
    with _stores_guard:
        store = _stores.get(key)
        if store is None:
            store = _stores[key] = VectorStore(llm, p)
        elif llm is not None:
            store.llm = llm
        return store
