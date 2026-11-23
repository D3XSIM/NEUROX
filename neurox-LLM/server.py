"""Neurox HTTP API (FastAPI).

    POST /chat     {"user_input": str, "history": [{"role","content"}]?, "model_name": str?, "images": [b64]?}
    GET  /health   provider reachability
    GET  /models   models available on the provider

Dangerous tools (file writes, code execution, computer control) are removed because the API has no approval UI,
unless the environment variable NEUROX_API_ALLOW_DANGEROUS=1 is set - then they run without asking.

The Neurox instance is created lazily by ``get_neurox()``; tests can inject one with
``server.app.state.neurox = Neurox(settings, llm=FakeLLM(...))`` or monkeypatch ``server.get_neurox``.
Run:  python server.py   (or: uvicorn server:app --port 8000)
"""
from __future__ import annotations

import copy
import logging
import os
import threading
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.agent import resume_agent
from core.llm import LLMError
from core.pipeline import Neurox

log = logging.getLogger(__name__)

app = FastAPI(title="Neurox Core API", version="2.0")
app.state.neurox = None
_lock = threading.Lock()


def allow_dangerous() -> bool:
    return os.environ.get("NEUROX_API_ALLOW_DANGEROUS", "").strip().lower() in ("1", "true", "yes")


def get_neurox() -> Neurox:
    """Return the shared Neurox instance, creating it on first use."""
    if app.state.neurox is None:
        with _lock:
            if app.state.neurox is None:
                app.state.neurox = Neurox()
    return app.state.neurox


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = ""


class ChatRequest(BaseModel):
    user_input: str = Field(..., min_length=1)
    history: list[HistoryMessage] = Field(default_factory=list)
    model_name: str | None = None
    images: list[str] = Field(default_factory=list)


def _clean_event(e: dict) -> dict:
    return {k: (len(v) if k == "images" else v) for k, v in e.items()}  # don't echo base64 images in the trace


@app.post("/chat")
def chat(request: ChatRequest):
    if not request.user_input.strip():
        raise HTTPException(status_code=422, detail="user_input must not be empty")
    neurox = get_neurox()
    dangerous_ok = allow_dangerous()
    messages = [{"role": m.role, "content": m.content} for m in request.history]
    messages.append({"role": "user", "content": request.user_input})
    try:
        ctx = neurox.build_context(request.user_input, messages, images=request.images,
                                   allow_dangerous=dangerous_ok)
        if request.model_name and request.model_name != getattr(neurox.llm, "model", None):
            llm = copy.copy(neurox.llm)  # per-request model override, shared connection
            llm.model = request.model_name
            ctx["llm"] = llm
        ctx = neurox.manager.execute_all(ctx)

        result = ctx.get("agent_result")
        # No approval UI over HTTP: when dangerous tools are enabled, the env var is the approval.
        guard = 0
        while result is not None and result.pending_call is not None and guard < 20:
            guard += 1
            result = resume_agent(ctx["llm"], result, ctx["registry"], approved=dangerous_ok,
                                  auto_approve=dangerous_ok,
                                  max_steps=getattr(neurox.settings, "max_agent_steps", 8))
            ctx["agent_result"] = result
            ctx["final_response"] = result.final_text
    except LLMError as e:
        raise HTTPException(status_code=503, detail=f"LLM provider error: {e}") from e
    except Exception as e:
        log.exception("Chat request failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    tools_used = [e["name"] for e in (result.trace if result else []) if e.get("type") == "tool_call"]
    return {
        "response": ctx.get("final_response") or "",
        "agent": ctx.get("active_agent", "GENERALIST"),
        "blocked": bool(ctx.get("blocked")),
        "tools_used": tools_used,
        "images": list(result.images) if result else [],
        "inner_monologue": ctx.get("inner_monologue", ""),
        "trace": [_clean_event(e) for e in ctx.get("trace", [])] + ([_clean_event(e) for e in result.trace] if result else []),
    }


@app.get("/health")
def health():
    neurox = get_neurox()
    s = neurox.settings
    reachable = neurox.llm.is_available()
    return {"status": "ok" if reachable else "degraded", "provider": s.provider, "base_url": s.base_url,
            "model": getattr(neurox.llm, "model", s.chat_model), "provider_reachable": reachable,
            "plugins": [p.name for p in neurox.manager.plugins], "dangerous_tools_allowed": allow_dangerous()}


@app.get("/models")
def models():
    try:
        return {"models": get_neurox().llm.list_models()}
    except LLMError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=os.environ.get("NEUROX_API_HOST", "127.0.0.1"), port=int(os.environ.get("NEUROX_API_PORT", "8000")))
