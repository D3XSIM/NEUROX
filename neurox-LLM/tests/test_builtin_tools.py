"""tools.builtin: calculator safety, workspace sandbox, run_python, web tools with a fake ``requests``."""
import time

import pytest
import requests

from core import config
from core.config import Settings
from core.tools import ToolRegistry

builtin = pytest.importorskip("tools.builtin")

EXPECTED = {"web_search", "fetch_url", "calculator", "current_time", "list_dir", "read_file", "write_file",
            "run_python"}


@pytest.fixture
def reg() -> ToolRegistry:
    r = ToolRegistry()
    for t in builtin.get_tools(Settings(), None):
        r.register(t)
    return r


def _arg_names(tool):
    props = (tool.parameters or {}).get("properties", {})
    return list((tool.parameters or {}).get("required") or props)


def call(reg, name, *values, **kw):
    """Call a tool by positional values mapped onto its required parameters (robust to parameter naming)."""
    args = dict(zip(_arg_names(reg.get(name)), values))
    args.update(kw)
    return reg.call(name, args).text


def is_error(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in ("error", "invalid", "not allowed", "unsupported", "outside", "denied", "hata"))


# --------------------------------------------------------------------------------------- registry
def test_all_builtin_tools_present_with_valid_schemas(reg):
    assert EXPECTED <= set(reg.names())
    for s in reg.schemas():
        params = s["function"]["parameters"]
        assert params["type"] == "object" and isinstance(params.get("properties", {}), dict)
        assert s["function"]["description"]


def test_dangerous_flags(reg):
    assert reg.get("run_python").dangerous is True
    assert reg.get("write_file").dangerous is True
    for safe in ("web_search", "fetch_url", "calculator", "current_time", "list_dir", "read_file"):
        assert reg.get(safe).dangerous is False, safe


def test_build_registry_respects_enabled_tools(fake_llm):
    from tools import build_registry
    r = build_registry(Settings(enabled_tools=["calculator", "current_time"]), fake_llm)
    assert set(r.names()) == {"calculator", "current_time"}
    assert len(build_registry(Settings(tools_enabled=False), fake_llm)) == 0


# ------------------------------------------------------------------------------------- calculator
@pytest.mark.parametrize("expr,expected", [("2+3*4", "14"), ("2^10", "1024"), ("sqrt(16)", "4"), ("10/4", "2.5")])
def test_calculator_math(reg, expr, expected):
    assert expected in call(reg, "calculator", expr)


@pytest.mark.parametrize("expr", [
    "__import__('os')",
    "__import__('os').system('echo pwned > pwned.txt')",
    "open('x.txt', 'w')",
    "().__class__.__bases__[0].__subclasses__()",
    "exec('print(1)')",
    "lambda: 1",
    "[x for x in range(10)]",
    "9**9**9**9",
])
def test_calculator_rejects_unsafe(reg, expr, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    start = time.time()
    out = call(reg, "calculator", expr)
    assert is_error(out), out
    assert "<module" not in out and "<built-in" not in out  # (the error may echo the input)
    assert not (tmp_path / "x.txt").exists() and not (tmp_path / "pwned.txt").exists()
    assert time.time() - start < 5


def test_calculator_division_by_zero(reg):
    assert is_error(call(reg, "calculator", "1/0"))


def test_current_time(reg):
    out = reg.call("current_time", {}).text
    assert str(time.localtime().tm_year) in out


# ------------------------------------------------------------------------------------- workspace
def test_write_read_list_roundtrip(reg):
    assert not is_error(call(reg, "write_file", "notes/a.txt", "merhaba dünya"))
    ws = config.WORKSPACE_DIR
    assert (ws / "notes" / "a.txt").read_text(encoding="utf-8") == "merhaba dünya"
    assert "merhaba dünya" in call(reg, "read_file", "notes/a.txt")
    assert "a.txt" in reg.call("list_dir", {"path": "notes"}).text
    assert "notes" in reg.call("list_dir", {}).text


def test_workspace_is_inside_test_data_dir(isolated_data_dir):
    assert config.WORKSPACE_DIR.resolve().is_relative_to(isolated_data_dir)


@pytest.mark.parametrize("bad", ["../secret.txt", "..\\secret.txt", "notes/../../secret.txt", "../../../../etc/passwd"])
def test_read_path_traversal_rejected(reg, bad):
    secret = config.WORKSPACE_DIR.parent / "secret.txt"
    secret.write_text("TOP-SECRET", encoding="utf-8")
    out = call(reg, "read_file", bad)
    assert "TOP-SECRET" not in out and is_error(out)


def test_read_absolute_path_outside_rejected(reg):
    secret = config.WORKSPACE_DIR.parent / "secret.txt"
    secret.write_text("TOP-SECRET", encoding="utf-8")
    out = call(reg, "read_file", str(secret.resolve()))
    assert "TOP-SECRET" not in out and is_error(out)


def test_write_path_traversal_rejected(reg):
    out = call(reg, "write_file", "../escape.txt", "x")
    assert is_error(out)
    assert not (config.WORKSPACE_DIR.parent / "escape.txt").exists()


def test_list_dir_traversal_rejected(reg):
    assert is_error(reg.call("list_dir", {"path": ".."}).text)


def test_read_missing_file(reg):
    assert is_error(call(reg, "read_file", "nope.txt"))


# ------------------------------------------------------------------------------------- run_python
def test_run_python_prints(reg):
    out = call(reg, "run_python", "print(2+2)")
    assert "4" in out


def test_run_python_reports_exceptions(reg):
    out = call(reg, "run_python", "raise ValueError('boom')")
    assert "ValueError" in out and "boom" in out


def test_run_python_timeout(reg, monkeypatch):
    if hasattr(builtin, "PYTHON_TIMEOUT"):
        monkeypatch.setattr(builtin, "PYTHON_TIMEOUT", 2)
    else:  # pragma: no cover - other implementations may expose a timeout parameter instead
        pytest.skip("run_python has no configurable timeout")
    start = time.time()
    out = call(reg, "run_python", "while True:\n    pass")
    assert time.time() - start < 15
    assert "time" in out.lower()  # "timed out" / "timeout"


# ---------------------------------------------------------------------------------- web (faked)
DDG_HTML = """
<html><body>
<div class="result results_links web-result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fneurox&rut=abc">Neurox Example</a>
  <a class="result__snippet">A local assistant snippet.</a>
</div>
<div class="result result--ad"><a class="result__a" href="https://ads.example">AD</a></div>
<div class="result"><a class="result__a" href="https://second.example/">Second</a></div>
</body></html>
"""


class FakeHTTPResponse:
    def __init__(self, text="", status=200, content_type="text/html; charset=utf-8"):
        self.text = text
        self.content = text.encode("utf-8")
        self.status_code = status
        self.headers = {"Content-Type": content_type}
        self.encoding = "utf-8"
        self.apparent_encoding = "utf-8"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_http(monkeypatch):
    seen = []

    def install(response):
        def fake(method):
            def _f(url, *a, **kw):
                seen.append((method, url, kw))
                if isinstance(response, Exception):
                    raise response
                return response
            return _f
        monkeypatch.setattr(requests, "get", fake("GET"))
        monkeypatch.setattr(requests, "post", fake("POST"))
        if hasattr(builtin, "requests"):
            monkeypatch.setattr(builtin.requests, "get", fake("GET"))
            monkeypatch.setattr(builtin.requests, "post", fake("POST"))
        return seen
    return install


def test_web_search_parses_results(reg, fake_http):
    seen = fake_http(FakeHTTPResponse(DDG_HTML))
    out = call(reg, "web_search", "neurox")
    assert seen, "web_search did not use requests"
    assert "Neurox Example" in out
    assert "https://example.com/neurox" in out      # DDG redirect link decoded
    assert "AD" not in out.split("\n")[0]


def test_web_search_network_error_is_reported(reg, fake_http):
    fake_http(requests.ConnectionError("offline"))
    out = call(reg, "web_search", "neurox")
    assert is_error(out)


def test_fetch_url_extracts_text(reg, fake_http):
    html = "<html><head><title>T</title><script>evil()</script></head><body><h1>Hello World</h1><p>Body text</p></body></html>"
    fake_http(FakeHTTPResponse(html))
    out = call(reg, "fetch_url", "https://example.com/")
    assert "Hello World" in out and "Body text" in out
    assert "evil()" not in out and "<h1>" not in out


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://x/y", "not a url", "javascript:alert(1)"])
def test_fetch_url_rejects_non_http(reg, fake_http, url):
    seen = fake_http(FakeHTTPResponse("SHOULD NOT BE FETCHED"))
    out = call(reg, "fetch_url", url)
    assert is_error(out) and "SHOULD NOT" not in out and not seen


def test_fetch_url_network_error(reg, fake_http):
    fake_http(requests.Timeout("slow"))
    assert is_error(call(reg, "fetch_url", "https://example.com"))
