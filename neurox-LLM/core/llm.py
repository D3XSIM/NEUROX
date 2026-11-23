"""LLM provider abstraction: Ollama and any OpenAI-compatible server (LM Studio, llama.cpp, vLLM...).

Canonical message format used everywhere in Neurox (OpenAI-like dicts)::

    {"role": "system" | "user" | "assistant" | "tool",
     "content": str,
     "images": [b64, ...],             # optional, user/tool messages, for vision models
     "tool_calls": [ToolCall, ...],    # optional, assistant messages
     "tool_call_id": str, "name": str} # tool messages

Each client converts this to its provider's wire format.
"""
from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import requests

from core.config import PROVIDERS, Settings

log = logging.getLogger(__name__)

TIMEOUT = 300  # seconds; local models can be slow


class LLMError(RuntimeError):
    """Provider unreachable or returned an error."""


class ToolsNotSupported(LLMError):
    """The selected model does not support native tool calling."""


@dataclass
class ToolCall:
    name: str
    arguments: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: "call_" + uuid.uuid4().hex[:12])

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, d: dict) -> "ToolCall":
        return cls(name=d["name"], arguments=d.get("arguments") or {}, id=d.get("id") or "call_" + uuid.uuid4().hex[:12])


@dataclass
class ChatResult:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


def _parse_args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else {"value": val}
    except (json.JSONDecodeError, TypeError):
        return {"_raw": str(raw)}


def _tool_calls_of(msg: dict) -> list[ToolCall]:
    out = []
    for tc in msg.get("tool_calls") or []:
        out.append(tc if isinstance(tc, ToolCall) else ToolCall.from_dict(tc))
    return out


class LLMClient(ABC):
    def __init__(self, base_url: str, model: str, embed_model: str = "nomic-embed-text",
                 temperature: float = 0.7, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embed_model = embed_model
        self.temperature = temperature
        self.api_key = api_key

    @abstractmethod
    def chat(self, messages: list[dict], tools: list[dict] | None = None, stream: bool = False,
             on_token: Callable[[str], None] | None = None, model: str | None = None,
             temperature: float | None = None) -> ChatResult:
        """Send a chat request. If ``stream`` and ``on_token`` are given, text tokens are pushed as they arrive.
        Raises ``ToolsNotSupported`` if tools were passed but the model can't use them, ``LLMError`` otherwise."""

    @abstractmethod
    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Return one embedding vector per text."""

    @abstractmethod
    def list_models(self) -> list[str]:
        """Names of models available on the server. Raises LLMError if unreachable."""

    def is_available(self) -> bool:
        try:
            self.list_models()
            return True
        except Exception:
            return False

    def simple(self, prompt: str, system: str | None = None, **kw) -> str:
        """Convenience: single-shot completion, returns text ('' on failure is NOT swallowed: raises LLMError)."""
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return self.chat(msgs, **kw).content.strip()


# --------------------------------------------------------------------------------------------- Ollama
class OllamaClient(LLMClient):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        import ollama
        self._ollama = ollama
        self._client = ollama.Client(host=self.base_url, timeout=TIMEOUT)

    @staticmethod
    def _to_wire(messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            w = {"role": m["role"], "content": m.get("content") or ""}
            if m.get("images"):
                w["images"] = list(m["images"])
            if m["role"] == "assistant" and m.get("tool_calls"):
                w["tool_calls"] = [{"function": {"name": tc.name, "arguments": tc.arguments}} for tc in _tool_calls_of(m)]
            if m["role"] == "tool" and m.get("name"):
                w["tool_name"] = m["name"]
            out.append(w)
        return out

    def chat(self, messages, tools=None, stream=False, on_token=None, model=None, temperature=None):
        kwargs = dict(model=model or self.model, messages=self._to_wire(messages),
                      options={"temperature": self.temperature if temperature is None else temperature})
        if tools:
            kwargs["tools"] = tools
        try:
            if stream and on_token:
                result = ChatResult()
                for chunk in self._client.chat(stream=True, **kwargs):
                    msg = chunk["message"]
                    token = msg.get("content") or ""
                    if token:
                        result.content += token
                        on_token(token)
                    for tc in msg.get("tool_calls") or []:
                        result.tool_calls.append(ToolCall(tc["function"]["name"], _parse_args(tc["function"].get("arguments"))))
                return result
            resp = self._client.chat(stream=False, **kwargs)
            msg = resp["message"]
            calls = [ToolCall(tc["function"]["name"], _parse_args(tc["function"].get("arguments")))
                     for tc in (msg.get("tool_calls") or [])]
            return ChatResult(msg.get("content") or "", calls)
        except self._ollama.ResponseError as e:
            if tools and "does not support tools" in str(e).lower():
                raise ToolsNotSupported(str(e)) from e
            raise LLMError(f"Ollama error: {e}") from e
        except Exception as e:
            raise LLMError(f"Ollama not reachable at {self.base_url}: {e}") from e

    def embed(self, texts, model=None):
        try:
            resp = self._client.embed(model=model or self.embed_model, input=texts)
            return [list(v) for v in resp["embeddings"]]
        except Exception as e:
            raise LLMError(f"Ollama embed failed: {e}") from e

    def list_models(self):
        try:
            resp = self._client.list()
            return sorted(m.get("model") or m.get("name") for m in resp["models"])
        except Exception as e:
            raise LLMError(f"Ollama not reachable at {self.base_url}: {e}") from e


# ----------------------------------------------------------------------------------- OpenAI-compatible
class OpenAICompatClient(LLMClient):
    """LM Studio (default http://localhost:1234/v1) or any OpenAI-compatible endpoint."""

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    @staticmethod
    def _to_wire(messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            role, text = m["role"], m.get("content") or ""
            images = m.get("images") or []
            if role == "tool":
                out.append({"role": "tool", "tool_call_id": m.get("tool_call_id", ""), "content": text})
                if images:  # tool messages can't carry images in the OpenAI format -> follow-up user message
                    out.append({"role": "user", "content": [{"type": "text", "text": f"[Image output of tool {m.get('name', '')}]"}]
                                + [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}} for b in images]})
                continue
            w: dict = {"role": role}
            if images:
                w["content"] = [{"type": "text", "text": text}] + [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}} for b in images]
            else:
                w["content"] = text
            if role == "assistant" and m.get("tool_calls"):
                w["tool_calls"] = [{"id": tc.id, "type": "function",
                                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}
                                   for tc in _tool_calls_of(m)]
            out.append(w)
        return out

    def chat(self, messages, tools=None, stream=False, on_token=None, model=None, temperature=None):
        body = {"model": model or self.model, "messages": self._to_wire(messages),
                "temperature": self.temperature if temperature is None else temperature,
                "stream": bool(stream and on_token)}
        if tools:
            body["tools"] = tools
        try:
            r = requests.post(f"{self.base_url}/chat/completions", json=body, headers=self._headers(),
                              stream=body["stream"], timeout=TIMEOUT)
        except requests.RequestException as e:
            raise LLMError(f"Server not reachable at {self.base_url}: {e}") from e
        if r.status_code >= 400:
            txt = r.text[:500]
            if tools and "tool" in txt.lower() and ("support" in txt.lower() or "unsupported" in txt.lower()):
                raise ToolsNotSupported(txt)
            raise LLMError(f"HTTP {r.status_code}: {txt}")

        if not body["stream"]:
            msg = r.json()["choices"][0]["message"]
            calls = []
            for tc in msg.get("tool_calls") or []:
                call = ToolCall(tc["function"]["name"], _parse_args(tc["function"].get("arguments")))
                if tc.get("id"):
                    call.id = tc["id"]
                calls.append(call)
            return ChatResult(msg.get("content") or "", calls)

        result = ChatResult()
        partial: dict[int, dict] = {}
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                delta = json.loads(data)["choices"][0].get("delta") or {}
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            token = delta.get("content") or ""
            if token:
                result.content += token
                on_token(token)
            for tc in delta.get("tool_calls") or []:
                p = partial.setdefault(tc.get("index", 0), {"id": None, "name": "", "args": ""})
                p["id"] = tc.get("id") or p["id"]
                fn = tc.get("function") or {}
                p["name"] += fn.get("name") or ""
                p["args"] += fn.get("arguments") or ""
        for _, p in sorted(partial.items()):
            call = ToolCall(p["name"], _parse_args(p["args"]))
            if p["id"]:
                call.id = p["id"]
            result.tool_calls.append(call)
        return result

    def embed(self, texts, model=None):
        try:
            r = requests.post(f"{self.base_url}/embeddings", json={"model": model or self.embed_model, "input": texts},
                              headers=self._headers(), timeout=TIMEOUT)
            r.raise_for_status()
            return [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d.get("index", 0))]
        except Exception as e:
            raise LLMError(f"Embedding failed: {e}") from e

    def list_models(self):
        try:
            r = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=5)
            r.raise_for_status()
            return sorted(m["id"] for m in r.json().get("data", []))
        except Exception as e:
            raise LLMError(f"Server not reachable at {self.base_url}: {e}") from e


def get_client(settings: Settings) -> LLMClient:
    base = settings.base_url or PROVIDERS.get(settings.provider, PROVIDERS["ollama"])
    cls = OllamaClient if settings.provider == "ollama" else OpenAICompatClient
    return cls(base, settings.chat_model, settings.embed_model, settings.temperature, settings.api_key)
