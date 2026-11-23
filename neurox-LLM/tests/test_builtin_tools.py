"""tools.builtin: calculator safety, workspace sandbox, run_python, web tools with a fake ``requests``."""
import base64
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


# ------------------------------------------------------------------ web search fallback chain (faked)
BING_HTML = """
<html><body><ol id="b_results">
<li class="b_algo"><h2><a href="https://www.bing.com/ck/a?!&&p=abc&ptn=3&ver=2&u={u}&ntb=1">Neurox Docs</a></h2>
  <div class="b_caption"><p>Neurox is a local <strong>assistant</strong>.</p></div></li>
<li class="b_ad"><h2><a href="https://www.bing.com/aclick?ld=1">Sponsored</a></h2></li>
<li class="b_algo"><h2><a href="https://plain.example/neurox">Plain Neurox link</a></h2><p>Second snippet</p></li>
<li class="b_algo"><h2><a href="https://www.bing.com/ck/a?!&&p=zzz&u=garbage">Undecodable neurox</a></h2></li>
</ol></body></html>
"""


def _bing_wrap(url: str) -> str:
    return "a1" + base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")


WIKI_JSON = {"batchcomplete": True, "query": {"pages": [
    {"pageid": 2, "title": "Neurox engine", "index": 2, "fullurl": "https://en.wikipedia.org/wiki/Neurox_engine",
     "extract": "A second article about neurox."},
    {"pageid": 1, "title": "Neurox", "index": 1, "fullurl": "https://en.wikipedia.org/wiki/Neurox",
     "extract": "Neurox is a fictional assistant."},
]}}


class RoutedResponse(FakeHTTPResponse):
    def __init__(self, text="", status=200, content_type="text/html; charset=utf-8", url="", json_data=None,
                 raw=None):
        super().__init__(text, status, content_type)
        self.url = url
        self._json = json_data
        if raw is not None:
            self.content = raw

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


@pytest.fixture
def router(monkeypatch):
    """Fake requests.get/post answering by URL substring; unknown hosts raise ConnectionError. Disables ddgs."""
    routes: dict = {}
    seen: list = []

    def handler(method):
        def _f(url, *a, **kw):
            seen.append((method, url, kw))
            for key, resp in routes.items():
                if key in url:
                    r = resp(url, kw) if callable(resp) else resp
                    if isinstance(r, Exception):
                        raise r
                    return r
            raise requests.ConnectionError(f"no route for {url}")
        return _f
    monkeypatch.setattr(builtin.requests, "get", handler("GET"))
    monkeypatch.setattr(builtin.requests, "post", handler("POST"))
    monkeypatch.setattr(builtin, "_ddgs_class", lambda: None)
    return routes, seen


DDG_CHALLENGE = RoutedResponse('<html><div class="anomaly-modal">are you a robot?</div></html>', status=202)


def test_backend_order_and_first_failure_falls_through(monkeypatch):
    calls = []

    def failing(q, n, region, timeout):
        calls.append("a")
        raise requests.ConnectionError("down")

    def empty(q, n, region, timeout):
        calls.append("b")
        return []

    def good(q, n, region, timeout):
        calls.append("c")
        assert timeout <= builtin.HTTP_TIMEOUT
        return [{"title": "Neurox result", "url": "https://good.example/neurox", "snippet": "about neurox"}]

    def never(q, n, region, timeout):  # pragma: no cover - must not be reached
        calls.append("d")
        return [{"title": "x", "url": "https://x.example", "snippet": ""}]

    monkeypatch.setattr(builtin, "_BACKENDS", [("a", failing), ("b", empty), ("c", good), ("d", never)])
    out = builtin.web_search("neurox")
    assert calls == ["a", "b", "c"]
    assert "Neurox result" in out and "https://good.example/neurox" in out
    assert out.strip().splitlines()[-1] == "Source: c"
    assert builtin.search_web("neurox")[0]["url"] == "https://good.example/neurox"


def test_ddgs_backend_used_first(monkeypatch, router):
    routes, seen = router

    class FakeDDGS:
        def __init__(self, timeout=None, **kw):
            self.timeout = timeout

        def text(self, query, region="us-en", max_results=5, **kw):
            assert region == "tr-tr" and query == "Türkiye'nin başkenti neresidir"
            return [{"title": "Ankara - Türkiye'nin başkenti", "href": "https://tr.example/ankara",
                     "body": "Türkiye'nin başkenti Ankara'dır."}]
    monkeypatch.setattr(builtin, "_ddgs_class", lambda: FakeDDGS)
    monkeypatch.setattr(builtin, "_dns_ok", lambda host="": True)
    out = builtin.web_search("Türkiye'nin başkenti neresidir")
    assert "https://tr.example/ankara" in out and "Ankara'dır" in out
    assert out.endswith("Source: ddgs")
    assert not seen  # no scraping needed


def test_ddgs_skipped_when_offline(monkeypatch, router):
    routes, _ = router
    created = []

    class MustNotRun:
        def __init__(self, **kw):
            created.append(kw)

        def text(self, *a, **kw):
            return [{"title": "neurox via ddgs", "href": "https://ddgs.example/", "body": ""}]
    monkeypatch.setattr(builtin, "_ddgs_class", lambda: MustNotRun)
    routes["duckduckgo.com"] = RoutedResponse(DDG_HTML)
    assert builtin.web_search("neurox").endswith("Source: duckduckgo")  # conftest blocks DNS -> ddgs skipped
    assert created == []


def test_ddgs_failure_and_ddg_block_fall_back_to_bing(monkeypatch, router):
    routes, seen = router

    class BrokenDDGS:
        def __init__(self, **kw):
            pass

        def text(self, *a, **kw):
            raise RuntimeError("ratelimited")
    monkeypatch.setattr(builtin, "_ddgs_class", lambda: BrokenDDGS)
    monkeypatch.setattr(builtin, "_dns_ok", lambda host="": True)
    routes["duckduckgo.com"] = DDG_CHALLENGE
    routes["bing.com/search"] = RoutedResponse(BING_HTML.replace("{u}", _bing_wrap("https://docs.example/neurox?a=1")),
                                               url="https://www.bing.com/search?q=neurox")
    out = builtin.web_search("neurox docs", max_results=5)
    assert out.endswith("Source: bing"), out
    assert "https://docs.example/neurox?a=1" in out          # /ck/a redirect decoded
    assert "https://plain.example/neurox" in out
    assert "Sponsored" not in out and "aclick" not in out
    assert "Neurox is a local assistant." in out
    hosts = [u for _, u, _ in seen]
    assert any("html.duckduckgo.com" in u for u in hosts) and any("lite.duckduckgo.com" in u for u in hosts)
    assert all(kw.get("timeout", 99) <= builtin.HTTP_TIMEOUT for _, _, kw in seen)
    assert all("Mozilla" in kw["headers"]["User-Agent"] and kw["headers"].get("Accept-Language")
               for _, _, kw in seen)


def test_bing_parser_and_link_decoding():
    target = "https://example.org/path?q=ç&x=1"
    assert builtin._decode_bing_link("https://www.bing.com/ck/a?!&&p=1&u=" + _bing_wrap(target) + "&ntb=1") == target
    assert builtin._decode_bing_link("https://plain.example/") == "https://plain.example/"
    bad = "https://www.bing.com/ck/a?!&&p=1&u=a1%%%"
    assert builtin._decode_bing_link(bad) == bad  # undecodable -> unchanged, never raises
    res = builtin._parse_bing(BING_HTML.replace("{u}", _bing_wrap("https://docs.example/n")), 10)
    assert [r["url"] for r in res][:2] == ["https://docs.example/n", "https://plain.example/neurox"]
    assert res[0]["title"] == "Neurox Docs" and res[1]["snippet"] == "Second snippet"
    assert len(builtin._parse_bing(BING_HTML.replace("{u}", _bing_wrap("https://docs.example/n")), 1)) == 1


def test_bing_turkish_region_params(router):
    routes, seen = router
    routes["bing.com/search"] = RoutedResponse(BING_HTML.replace("{u}", _bing_wrap("https://tr.example/")))
    builtin._search_bing("neurox nedir", 5, "tr-tr", 5)
    params = seen[0][2]["params"]
    assert params["setlang"] == "tr" and params["cc"] == "TR"
    assert seen[0][2]["headers"]["Accept-Language"].startswith("tr")


def test_bing_junk_results_are_filtered_then_retried(router):
    routes, seen = router
    junk = '<li class="b_algo"><h2><a href="https://gifts.example/">Gift shop</a></h2><p>souvenirs</p></li>'
    good = '<li class="b_algo"><h2><a href="https://python.org/downloads/">Python 3.15 release</a></h2></li>'
    responses = iter([RoutedResponse(junk), RoutedResponse(good)])
    routes["bing.com/search"] = lambda url, kw: next(responses)
    res = builtin._search_bing("latest Python release", 5, "us-en", 10)
    assert [r["url"] for r in res] == ["https://python.org/downloads/"] and len(seen) == 2


def test_wikipedia_last_resort(router):
    routes, seen = router
    routes["duckduckgo.com"] = requests.Timeout("slow")
    routes["bing.com/search"] = RoutedResponse("<html><body><li class='b_no'>No results</li></body></html>")
    routes["wikipedia.org/w/api.php"] = RoutedResponse("", content_type="application/json", json_data=WIKI_JSON)
    out = builtin.web_search("what is neurox")
    assert out.endswith("Source: wikipedia"), out
    assert out.splitlines()[0] == "1. Neurox - Wikipedia"           # sorted by search index
    assert "https://en.wikipedia.org/wiki/Neurox" in out and "fictional assistant" in out
    wiki_call = [kw for _, u, kw in seen if "wikipedia.org" in u][0]
    assert wiki_call["params"]["gsrsearch"] == "neurox"              # question words stripped


def test_wikipedia_parse_formatversion1_and_missing_url():
    data = {"query": {"pages": {"7": {"title": "Ankara", "index": 1, "extract": "Başkent."}}}}
    res = builtin._parse_wikipedia(data, "tr", 5)
    assert res == [{"title": "Ankara - Wikipedia", "url": "https://tr.wikipedia.org/wiki/Ankara",
                    "snippet": "Başkent."}]
    assert builtin._parse_wikipedia({}, "en", 5) == []


def test_all_backends_fail_returns_error_string(router):
    routes, seen = router
    routes["duckduckgo.com"] = DDG_CHALLENGE
    routes["bing.com"] = RoutedResponse("blocked", status=503)
    routes["wikipedia.org"] = requests.ConnectionError("offline")
    out = builtin.web_search("neurox")
    assert out.startswith("Error:") and "SearchBlocked" in out
    with pytest.raises(builtin.SearchBlocked):
        builtin.search_web("neurox")


def test_web_search_never_raises_even_if_a_backend_is_broken(monkeypatch):
    def boom(*a, **kw):
        raise ZeroDivisionError("bug")
    monkeypatch.setattr(builtin, "_BACKENDS", [("boom", boom)])
    assert builtin.web_search("neurox").startswith("Error:")
    assert builtin.web_search("").startswith("Error:")
    monkeypatch.setattr(builtin, "_BACKENDS", [("empty", lambda *a: [])])
    assert builtin.web_search("neurox", max_results="lots").startswith("No results")


def test_search_time_budget_skips_remaining_backends(monkeypatch):
    monkeypatch.setattr(builtin, "SEARCH_TIME_BUDGET", 0)
    called = []
    monkeypatch.setattr(builtin, "_BACKENDS", [("x", lambda *a: called.append(1) or [])])
    assert builtin.web_search("neurox").startswith("Error:") and not called


@pytest.mark.parametrize("query,region", [
    ("Türkiye'nin başkenti neresidir", "tr-tr"), ("dolar kuru bugün", "tr-tr"), ("istanbul hava durumu nasıl", "tr-tr"),
    ("latest Python release", "us-en"), ("Ich möchte Bücher", "us-en"),
])
def test_region_detection(query, region):
    assert builtin._detect_region(query) == region


def test_explicit_region_is_validated():
    assert builtin._normalize_region("de-DE", "x") == "de-de"
    assert builtin._normalize_region("nonsense", "latest news") == "us-en"
    assert builtin._normalize_region(None, "bugün hava nasıl") == "tr-tr"


def test_web_search_tool_schema(reg):
    tool = reg.get("web_search")
    assert "fetch_url" in tool.description and "price" in tool.description.lower()
    assert "region" in tool.parameters["properties"]
    assert "bad arguments" not in reg.call("web_search", {"query": "", "region": "tr-tr", "max_results": 2}).text


# ------------------------------------------------------------------------------- fetch_url (faked)
def test_fetch_url_decodes_meta_charset_without_header_charset(router):
    routes, _ = router
    html = '<html><head><meta charset="windows-1254"><title>Başlık</title></head><body><p>Şişli ığdır</p></body></html>'
    routes["tr.example"] = RoutedResponse("", content_type="text/html", raw=html.encode("cp1254"))
    out = builtin.fetch_url("https://tr.example/")
    assert "Title: Başlık" in out and "Şişli ığdır" in out


def test_fetch_url_utf8_without_any_charset(router):
    routes, _ = router
    routes["u.example"] = RoutedResponse("", content_type="text/plain", raw="Türkçe metin ✓".encode("utf-8"))
    assert builtin.fetch_url("https://u.example/") == "Türkçe metin ✓"


def test_fetch_url_prefers_article_and_strips_boilerplate(router):
    routes, _ = router
    article = "<p>" + "Real article sentence about neurox. " * 20 + "</p>"
    html = (f"<html><head><title>News</title><style>.x{{}}</style></head><body><nav>Home | About</nav>"
            f"<div class='sidebar'>{'Unrelated sidebar link. ' * 5}</div><article><h1>Headline</h1>{article}</article>"
            "<footer>Copyright footer</footer><script>track()</script></body></html>")
    routes["news.example"] = RoutedResponse(html, url="https://news.example/final")
    out = builtin.fetch_url("https://news.example/start", max_chars=50_000)
    assert out.startswith("Title: News\nURL: https://news.example/final")  # redirect target reported
    assert "Headline" in out and "Real article sentence" in out
    for junk in ("Home | About", "Copyright footer", "track()", "Unrelated sidebar", ".x{"):
        assert junk not in out
    assert "  " not in out and "\n\n\n" not in out


def test_fetch_url_noscript_fallback(router):
    routes, _ = router
    html = ("<html><head><title>Forum</title></head><body><section id='main'></section><noscript>"
            "<div class='post'><p>" + "Server rendered forum answer. " * 12 + "</p></div></noscript>"
            "<script>app()</script></body></html>")
    routes["forum.example"] = RoutedResponse(html)
    out = builtin.fetch_url("https://forum.example/t/1")
    assert "Server rendered forum answer." in out and "app()" not in out


def test_fetch_url_truncates_and_rejects_binary(router):
    routes, _ = router
    routes["big.example"] = RoutedResponse("<html><body><p>" + "word " * 5000 + "</p></body></html>")
    out = builtin.fetch_url("https://big.example/", max_chars=500)
    assert "[truncated" in out and len(out) < 700
    routes["img.example"] = RoutedResponse("", content_type="image/png", raw=b"\x89PNG")
    assert is_error(builtin.fetch_url("https://img.example/x.png"))
