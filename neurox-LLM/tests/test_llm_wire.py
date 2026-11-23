"""core.llm: canonical message format -> provider wire formats, streaming, error mapping."""
import json

import pytest
import requests

import core.llm as llm_mod
from core.config import Settings
from core.llm import (ChatResult, LLMError, OllamaClient, OpenAICompatClient, ToolCall, ToolsNotSupported,
                      _parse_args, get_client)

IMG = "iVBORw0KGgo="


def canonical_conversation():
    tc = ToolCall("calculator", {"expression": "2+2"}, id="call_abc")
    return [
        {"role": "system", "content": "You are Neurox."},
        {"role": "user", "content": "What is in this picture?", "images": [IMG]},
        {"role": "assistant", "content": "", "tool_calls": [tc]},
        {"role": "tool", "content": "4", "tool_call_id": "call_abc", "name": "calculator"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "call_d", "name": "screenshot", "arguments": {}}]},  # dict form also accepted
        {"role": "tool", "content": "screen", "tool_call_id": "call_d", "name": "screenshot", "images": [IMG]},
    ]


# --------------------------------------------------------------------------------------- parse args
@pytest.mark.parametrize("raw,expected", [
    ({"a": 1}, {"a": 1}),
    ('{"a": 1}', {"a": 1}),
    ("", {}),
    (None, {}),
    ("[1, 2]", {"value": [1, 2]}),
    ("not json", {"_raw": "not json"}),
])
def test_parse_args(raw, expected):
    assert _parse_args(raw) == expected


def test_toolcall_roundtrip():
    tc = ToolCall("x", {"k": "v"})
    assert tc.id.startswith("call_")
    again = ToolCall.from_dict(tc.to_dict())
    assert again == tc
    assert ToolCall.from_dict({"name": "y"}).arguments == {}


# ------------------------------------------------------------------------------------------- Ollama
def test_ollama_to_wire():
    wire = OllamaClient._to_wire(canonical_conversation())
    assert wire[0] == {"role": "system", "content": "You are Neurox."}
    assert wire[1]["images"] == [IMG]
    assert wire[2]["tool_calls"] == [{"function": {"name": "calculator", "arguments": {"expression": "2+2"}}}]
    assert wire[3]["role"] == "tool" and wire[3]["tool_name"] == "calculator" and wire[3]["content"] == "4"
    assert wire[4]["content"] == "" and wire[4]["tool_calls"][0]["function"]["name"] == "screenshot"
    assert wire[5]["images"] == [IMG]
    # the input must not be mutated
    assert isinstance(canonical_conversation()[2]["tool_calls"][0], ToolCall)


class _FakeOllamaClient:
    def __init__(self, reply=None, chunks=None, exc=None):
        self.reply, self.chunks, self.exc = reply, chunks, exc
        self.kwargs = None

    def chat(self, stream=False, **kw):
        self.kwargs = dict(kw, stream=stream)
        if self.exc:
            raise self.exc
        return iter(self.chunks) if stream else self.reply


def _ollama(fake) -> OllamaClient:
    c = OllamaClient("http://localhost:11434/", "llama3.2")
    c._client = fake
    return c


def test_ollama_chat_parses_tool_calls_and_passes_tools():
    fake = _FakeOllamaClient(reply={"message": {"content": "", "tool_calls": [
        {"function": {"name": "calculator", "arguments": {"expression": "1+1"}}},
        {"function": {"name": "web_search", "arguments": '{"query": "x"}'}},
    ]}})
    c = _ollama(fake)
    assert c.base_url == "http://localhost:11434"
    res = c.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    assert [t.name for t in res.tool_calls] == ["calculator", "web_search"]
    assert res.tool_calls[1].arguments == {"query": "x"}
    assert fake.kwargs["tools"] == [{"type": "function"}]
    assert fake.kwargs["model"] == "llama3.2"


def test_ollama_streaming_pushes_tokens():
    chunks = [{"message": {"content": "Hel"}}, {"message": {"content": "lo"}},
              {"message": {"content": "", "tool_calls": [{"function": {"name": "t", "arguments": {}}}]}}]
    tokens = []
    res = _ollama(_FakeOllamaClient(chunks=chunks)).chat([{"role": "user", "content": "x"}], stream=True,
                                                         on_token=tokens.append)
    assert tokens == ["Hel", "lo"] and res.content == "Hello" and res.tool_calls[0].name == "t"


def test_ollama_tools_not_supported():
    import ollama
    c = _ollama(_FakeOllamaClient(exc=ollama.ResponseError("registry.ollama.ai/library/phi3 does not support tools")))
    with pytest.raises(ToolsNotSupported):
        c.chat([{"role": "user", "content": "x"}], tools=[{"type": "function"}])


def test_ollama_unreachable_is_llm_error():
    c = _ollama(_FakeOllamaClient(exc=ConnectionError("refused")))
    with pytest.raises(LLMError) as ei:
        c.chat([{"role": "user", "content": "x"}])
    assert not isinstance(ei.value, ToolsNotSupported)


# ---------------------------------------------------------------------------------- OpenAI-compatible
def test_openai_to_wire():
    wire = OpenAICompatClient._to_wire(canonical_conversation())
    assert wire[0] == {"role": "system", "content": "You are Neurox."}
    user = wire[1]
    assert user["content"][0] == {"type": "text", "text": "What is in this picture?"}
    assert user["content"][1] == {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{IMG}"}}
    tc = wire[2]["tool_calls"][0]
    assert tc["id"] == "call_abc" and tc["type"] == "function"
    assert tc["function"]["name"] == "calculator"
    assert json.loads(tc["function"]["arguments"]) == {"expression": "2+2"}  # arguments are a JSON string
    assert wire[3] == {"role": "tool", "tool_call_id": "call_abc", "content": "4"}
    assert wire[4]["tool_calls"][0]["id"] == "call_d"
    # tool message with images -> plain tool message + follow-up user message carrying the image
    assert wire[5] == {"role": "tool", "tool_call_id": "call_d", "content": "screen"}
    follow = wire[6]
    assert follow["role"] == "user"
    assert any(p.get("type") == "image_url" and IMG in p["image_url"]["url"] for p in follow["content"])
    assert len(wire) == 7


class FakeResponse:
    def __init__(self, status=200, payload=None, lines=None, text=""):
        self.status_code = status
        self._payload = payload
        self._lines = lines or []
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self):
        return self._payload

    def iter_lines(self, decode_unicode=False):
        yield from self._lines

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def _sse(obj) -> str:
    return "data: " + json.dumps(obj)


@pytest.fixture
def post_spy(monkeypatch):
    calls = []

    def install(response):
        def fake_post(url, json=None, headers=None, stream=False, timeout=None):
            calls.append({"url": url, "json": json, "headers": headers, "stream": stream})
            if isinstance(response, Exception):
                raise response
            return response
        monkeypatch.setattr(llm_mod.requests, "post", fake_post)
        return calls
    return install


def test_openai_streaming_accumulates_tool_calls(post_spy):
    lines = [
        ": keep-alive",
        "",
        _sse({"choices": [{"delta": {"role": "assistant", "content": "Let me "}}]}),
        _sse({"choices": [{"delta": {"content": "check."}}]}),
        _sse({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                                     "function": {"name": "web_", "arguments": ""}}]}}]}),
        _sse({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "search",
                                                                              "arguments": '{"que'}}]}}]}),
        _sse({"choices": [{"delta": {"tool_calls": [{"index": 1, "id": "call_2",
                                                     "function": {"name": "calculator", "arguments": '{"expression"'}}]}}]}),
        _sse({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ry": "neurox"}'}}]}}]}),
        _sse({"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"arguments": ': "2*3"}'}}]}}]}),
        "data: {not json",
        _sse({"choices": []}),
        "data: [DONE]",
        _sse({"choices": [{"delta": {"content": "AFTER DONE"}}]}),
    ]
    calls = post_spy(FakeResponse(lines=lines))
    client = OpenAICompatClient("http://localhost:1234/v1/", "qwen2.5", api_key="sk-test")
    tokens = []
    res = client.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}], stream=True,
                      on_token=tokens.append)
    assert "".join(tokens) == "Let me check." == res.content
    assert [(t.id, t.name, t.arguments) for t in res.tool_calls] == [
        ("call_1", "web_search", {"query": "neurox"}),
        ("call_2", "calculator", {"expression": "2*3"}),
    ]
    sent = calls[0]
    assert sent["url"] == "http://localhost:1234/v1/chat/completions"
    assert sent["stream"] is True and sent["json"]["stream"] is True
    assert sent["json"]["tools"] == [{"type": "function"}]
    assert sent["headers"]["Authorization"] == "Bearer sk-test"


def test_openai_non_stream_keeps_tool_call_ids(post_spy):
    payload = {"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "call_x", "type": "function", "function": {"name": "current_time", "arguments": "{}"}}]}}]}
    calls = post_spy(FakeResponse(payload=payload))
    res = OpenAICompatClient("http://h/v1", "m").chat([{"role": "user", "content": "time?"}])
    assert res.content == "" and res.tool_calls[0].id == "call_x" and res.tool_calls[0].arguments == {}
    assert calls[0]["json"]["stream"] is False and "tools" not in calls[0]["json"]
    assert "Authorization" not in calls[0]["headers"]


def test_stream_without_on_token_is_not_streamed(post_spy):
    calls = post_spy(FakeResponse(payload={"choices": [{"message": {"content": "hi"}}]}))
    res = OpenAICompatClient("http://h/v1", "m").chat([{"role": "user", "content": "x"}], stream=True)
    assert res == ChatResult("hi", [])
    assert calls[0]["json"]["stream"] is False


def test_openai_tools_not_supported(post_spy):
    post_spy(FakeResponse(status=400, text='{"error": "This model does not support tools"}'))
    with pytest.raises(ToolsNotSupported):
        OpenAICompatClient("http://h/v1", "m").chat([{"role": "user", "content": "x"}], tools=[{"type": "function"}])


def test_openai_http_error_is_llm_error(post_spy):
    post_spy(FakeResponse(status=500, text="internal"))
    with pytest.raises(LLMError) as ei:
        OpenAICompatClient("http://h/v1", "m").chat([{"role": "user", "content": "x"}])
    assert "500" in str(ei.value) and not isinstance(ei.value, ToolsNotSupported)


def test_openai_unreachable_is_llm_error(post_spy):
    post_spy(requests.ConnectionError("refused"))
    with pytest.raises(LLMError):
        OpenAICompatClient("http://h/v1", "m").chat([{"role": "user", "content": "x"}])


def test_openai_embed_sorted_by_index(post_spy):
    post_spy(FakeResponse(payload={"data": [{"index": 1, "embedding": [1.0]}, {"index": 0, "embedding": [0.0]}]}))
    assert OpenAICompatClient("http://h/v1", "m").embed(["a", "b"]) == [[0.0], [1.0]]


def test_list_models_unreachable_raises_and_is_available_false(monkeypatch):
    def boom(*a, **kw):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(llm_mod.requests, "get", boom)
    c = OpenAICompatClient("http://h/v1", "m")
    with pytest.raises(LLMError):
        c.list_models()
    assert c.is_available() is False


# ------------------------------------------------------------------------------------------ factory
def test_get_client_selects_provider():
    s = Settings(provider="lmstudio", base_url="http://localhost:1234/v1/", chat_model="qwen2.5-7b")
    c = get_client(s)
    assert isinstance(c, OpenAICompatClient) and c.base_url == "http://localhost:1234/v1" and c.model == "qwen2.5-7b"
    c = get_client(Settings(provider="ollama"))
    assert isinstance(c, OllamaClient) and c.base_url == "http://localhost:11434"
    c = get_client(Settings(provider="openai_compat", base_url=""))
    assert isinstance(c, OpenAICompatClient) and c.base_url == "http://localhost:8080/v1"
