"""Built-in tools: web search, URL fetching, calculator, clock, sandboxed workspace files, Python execution.

Public helpers (importable by plugins / tests):
    web_search(query, max_results=5) -> str        formatted results + "Source: <backend>" line (never raises)
    search_web(query, max_results=5) -> list[dict] structured results [{"title","url","snippet"}]
                                                   via ddgs -> DuckDuckGo -> Bing -> Wikipedia (no API keys)
    fetch_url(url, max_chars=6000) -> str
    calculator(expression) -> str                  safe AST evaluator (no eval)
    current_time() -> str
    list_dir(path=".") / read_file(path) / write_file(path, content)   confined to the workspace
    run_python(code) -> str                        isolated subprocess in a temp dir, 20 s timeout
    get_tools(settings, llm=None) -> list[Tool]
"""
from __future__ import annotations

import ast
import base64
import codecs
import datetime as _dt
import logging
import math
import operator
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import requests

from core import config
from core.tools import Tool

log = logging.getLogger(__name__)

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
HTTP_TIMEOUT = 10
PYTHON_TIMEOUT = 20
MAX_TOOL_OUTPUT = 8000
MAX_READ_CHARS = 50_000
MAX_DOWNLOAD_BYTES = 3_000_000

# Override for tests; when None the workspace is core.config.WORKSPACE_DIR (read at call time).
WORKSPACE_DIR: Path | None = None


def _truncate(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated, {len(text) - limit} more characters]"


# --------------------------------------------------------------------------------------------- web
# Search is a fallback chain of key-less backends, tried in order until one returns results:
#   1. ddgs        (pip "ddgs", metasearch library that rotates its own engines; optional dependency)
#   2. duckduckgo  (html.duckduckgo.com, then lite.duckduckgo.com scraping)
#   3. bing        (www.bing.com HTML scraping)
#   4. wikipedia   (MediaWiki search API; last resort, good for factual questions)
# Each backend is `fn(query, max_results, region, timeout) -> list[{"title","url","snippet"}]` and may raise;
# a failing backend is logged and skipped. The whole chain is bounded by SEARCH_TIME_BUDGET seconds.

SEARCH_TIME_BUDGET = 30.0
_DEFAULT_REGION = "us-en"
_REGION_RE = re.compile(r"^[a-z]{2}-[a-z]{2}$")
_TR_CHARS = re.compile(r"[ğĞıİşŞ]")
_TR_WORDS = re.compile(r"\b(nedir|neresi\w*|nasıl|nasil|kaç|kac|hangi|kimdir|ne zaman|fiyatı|fiyati|haber\w*|"
                       r"bugün|bugun|güncel|guncel|türkiye|turkiye|için|icin)\b", re.IGNORECASE)


class SearchBlocked(RuntimeError):
    """Every search backend failed or was blocked (bot challenge, network error, ...)."""


def _browser_headers(lang: str = "en") -> dict:
    accept_lang = "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7" if lang == "tr" else "en-US,en;q=0.9,tr;q=0.6"
    return {"User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": accept_lang}


_SEARCH_HEADERS = _browser_headers()  # kept for backwards compatibility


def _detect_region(query: str) -> str:
    """'tr-tr' for Turkish-looking queries, otherwise 'us-en'."""
    if _TR_CHARS.search(query) or len(_TR_WORDS.findall(query)) >= 1 and re.search(r"[çöüÇÖÜ]", query):
        return "tr-tr"
    if len(_TR_WORDS.findall(query)) >= 2:
        return "tr-tr"
    return _DEFAULT_REGION


def _normalize_region(region: str | None, query: str) -> str:
    region = str(region or "").strip().lower().replace("_", "-")
    if not region or region == "auto" or not _REGION_RE.match(region):
        return _detect_region(query)
    return region


def _region_parts(region: str) -> tuple[str, str]:
    """'tr-tr' -> ('tr', 'tr'), 'us-en' -> ('us', 'en')  (country, language)."""
    country, _, lang = region.partition("-")
    return country, lang or "en"


def _clean(text: str, limit: int = 400) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


_READ_MORE = re.compile(r"\s*(?:…|\.\.\.)?\s*(?:Devamını oku|Read more|Daha fazla bilgi)\s*$", re.IGNORECASE)


_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?)\]])")


def _result(title: str, url: str, snippet: str) -> dict:
    # parsers join inline tags with " " ("local <b>assistant</b>." -> "local assistant ."); undo the stray spaces
    title = _SPACE_BEFORE_PUNCT.sub(r"\1", _clean(title, 200))
    snippet = _SPACE_BEFORE_PUNCT.sub(r"\1", _READ_MORE.sub(" …", _clean(snippet)).strip())
    return {"title": title or url, "url": url, "snippet": snippet}


# question / filler words ignored when judging relevance or building a keyword query (stored normalized)
_STOPWORDS = frozenset("""
the and for with from that this what which who whom whose where when why how are was were will can does did
about into latest today current currently now new news best top list
nedir neresi neresidir nerede nerededir kimdir nasil nasildir hangi hangisi kac kactir zaman icin gibi daha
ile veya ama bugun guncel simdi son yeni olan olarak nelerdir neden niye midir mudur nin nun dir dan den
""".split())


def _norm(text: str) -> str:
    """Lowercase, accent-free ('Türkiye' -> 'turkiye', 'İ' -> 'i', 'ı' -> 'i') for loose matching."""
    text = unicodedata.normalize("NFKD", (text or "").replace("ı", "i").replace("İ", "I"))
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def _terms(query: str) -> list[str]:
    return [w for w in re.findall(r"\w+", _norm(query)) if len(w) >= 3 and w not in _STOPWORDS]


def _relevant(results: list[dict], query: str) -> list[dict]:
    """Drop results sharing no keyword with the query (scraped engines sometimes serve unrelated junk to bots).
    Matching is by a 6-character prefix, which tolerates Turkish suffixes ('başkenti' ~ 'başkent')."""
    terms = {t[:6] for t in _terms(query)}
    if not terms:
        return results
    keep = []
    for r in results:
        hay = _norm(f"{r.get('title', '')} {r.get('snippet', '')} {unquote(r.get('url', ''))}")
        if any(t in hay for t in terms):
            keep.append(r)
    return keep


def _keyword_query(query: str) -> str:
    """'Türkiye'nin başkenti neresidir' -> "Türkiye'nin başkenti" (for keyword engines such as Wikipedia)."""
    words = [w for w in query.split() if not all(t in _STOPWORDS or len(t) < 3
                                                  for t in re.findall(r"\w+", _norm(w)))]
    return " ".join(words) or query


# ---- duckduckgo (html + lite scraping)
def _decode_ddg_link(href: str) -> str:
    """DuckDuckGo wraps result links as //duckduckgo.com/l/?uddg=<real url>&rut=...; return the real URL."""
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg")
        if target:
            return unquote(target[0])
    return href


def _is_challenge(resp) -> bool:
    return resp.status_code == 202 or "anomaly-modal" in resp.text or "anomaly.js" in resp.text


def _parse_html_endpoint(html: str, max_results: int) -> list[dict]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    results: list[dict] = []
    for div in soup.select("div.result"):
        if "result--ad" in (div.get("class") or []):
            continue
        a = div.select_one("a.result__a")
        if not a:
            continue
        url = _decode_ddg_link(a.get("href", ""))
        if not url or "duckduckgo.com/y.js" in url:  # ad redirect
            continue
        snippet_tag = div.select_one(".result__snippet")
        results.append(_result(a.get_text(" ", strip=True), url,
                               snippet_tag.get_text(" ", strip=True) if snippet_tag else ""))
        if len(results) >= max_results:
            break
    return results


def _parse_lite_endpoint(html: str, max_results: int) -> list[dict]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    results: list[dict] = []
    snippets = [td.get_text(" ", strip=True) for td in soup.select("td.result-snippet")]
    for i, a in enumerate(soup.select("a.result-link")):
        url = _decode_ddg_link(a.get("href", ""))
        if not url or "duckduckgo.com/y.js" in url:
            continue
        results.append(_result(a.get_text(" ", strip=True), url, snippets[i] if i < len(snippets) else ""))
        if len(results) >= max_results:
            break
    return results


def _search_duckduckgo(query: str, max_results: int, region: str, timeout: float) -> list[dict]:
    """DuckDuckGo html endpoint, lite endpoint as fallback. Raises SearchBlocked on a bot challenge."""
    blocked, errors = False, []
    headers = _browser_headers(_region_parts(region)[1])
    for url, parser in (("https://html.duckduckgo.com/html/", _parse_html_endpoint),
                        ("https://lite.duckduckgo.com/lite/", _parse_lite_endpoint)):
        try:
            resp = requests.post(url, data={"q": query, "kl": region}, headers=headers, timeout=timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            errors.append(e)
            log.debug("duckduckgo %s failed: %s", url, e)
            continue
        if _is_challenge(resp):
            blocked = True
            continue
        return parser(resp.text, max_results)
    if blocked:
        raise SearchBlocked("DuckDuckGo returned a bot-challenge page")
    if errors:
        raise errors[-1]
    return []


# ---- ddgs (metasearch library)
def _ddgs_class():
    """The ddgs.DDGS class, or None when the optional package is not installed."""
    try:
        from ddgs import DDGS
        return DDGS
    except Exception:  # ImportError, or a broken install
        try:  # the deprecated predecessor exposes the same API
            from duckduckgo_search import DDGS  # type: ignore
            return DDGS
        except Exception:
            return None


def _dns_ok(host: str = "duckduckgo.com") -> bool:
    """Cheap offline check (Python-level DNS lookup). ddgs uses its own native HTTP client, so this also keeps
    it from bypassing environments that disable networking at the Python socket level (e.g. the test-suite)."""
    try:
        socket.getaddrinfo(host, 443)
        return True
    except OSError:
        return False


def _search_ddgs(query: str, max_results: int, region: str, timeout: float) -> list[dict]:
    cls = _ddgs_class()
    if cls is None:
        raise RuntimeError("ddgs package not installed")
    if not _dns_ok():
        raise ConnectionError("no network (DNS lookup failed)")
    box: dict = {}

    def run():
        # ddgs shuffles its engines per call and some of them are often rate-limited, so a fast "No results"
        # is retried once (the second call usually lands on a working engine).
        start = time.monotonic()
        for attempt in range(2):
            try:
                box["r"] = cls(timeout=max(2, int(min(timeout, 6)))).text(
                    query, region=region, safesearch="moderate", max_results=max_results)
                box.pop("e", None)
                return
            except Exception as e:  # noqa: BLE001 - reported to the caller below
                box["e"] = e
                if attempt or time.monotonic() - start > 3 or "timed out" in str(e).lower():
                    return

    # ddgs may query several engines in turn; bound the total wall time with a daemon thread
    limit = min(timeout, 8.0) + 1
    t = threading.Thread(target=run, name="neurox-ddgs", daemon=True)
    t.start()
    t.join(limit)
    if t.is_alive():
        raise TimeoutError(f"ddgs did not answer within {limit:.0f}s")
    if "e" in box:
        raise box["e"]
    results = []
    for r in box.get("r") or []:
        url = r.get("href") or r.get("url") or ""
        if url.startswith(("http://", "https://")):
            results.append(_result(r.get("title", ""), url, r.get("body") or r.get("snippet") or ""))
        if len(results) >= max_results:
            break
    return results


# ---- bing (HTML scraping)
def _decode_bing_link(href: str) -> str:
    """Bing tracking links look like https://www.bing.com/ck/a?...&u=a1<base64url(real url)>&...;
    return the real URL (or the input unchanged when it is not such a link / cannot be decoded)."""
    if not href:
        return ""
    if href.startswith("/"):
        href = "https://www.bing.com" + href
    parsed = urlparse(href)
    if "bing.com" not in parsed.netloc or not parsed.path.startswith("/ck/a"):
        return href
    u = (parse_qs(parsed.query).get("u") or [""])[0]
    if len(u) > 2 and u.startswith("a1"):
        try:
            b64 = u[2:]
            decoded = base64.urlsafe_b64decode(b64 + "=" * (-len(b64) % 4)).decode("utf-8")
            if decoded.startswith(("http://", "https://")):
                return decoded
        except (ValueError, UnicodeDecodeError):
            pass
    return href


def _parse_bing(html: str, max_results: int) -> list[dict]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    results: list[dict] = []
    seen = set()
    for li in soup.select("li.b_algo"):
        a = li.select_one("h2 a") or li.select_one("a[href]")
        if not a:
            continue
        url = _decode_bing_link(a.get("href", ""))
        if not url.startswith(("http://", "https://")) or "bing.com/aclick" in url or url in seen:
            continue
        seen.add(url)
        snip = (li.select_one(".b_caption p") or li.select_one("p.b_lineclamp2") or li.select_one("p")
                or li.select_one(".b_caption"))
        results.append(_result(a.get_text(" ", strip=True), url, snip.get_text(" ", strip=True) if snip else ""))
        if len(results) >= max_results:
            break
    return results


def _search_bing(query: str, max_results: int, region: str, timeout: float) -> list[dict]:
    """Bing sometimes answers a scraper with an empty result page for one parameter set and with full results
    for another, so up to two variants are tried (the second only when the first was empty)."""
    country, lang = _region_parts(region)
    first = {"q": query, "form": "QBLH"}
    if lang != "en":
        first.update(setlang=lang, cc=country.upper())
    started = time.monotonic()
    for params in (first, {"q": query}):
        remaining = timeout - (time.monotonic() - started)
        if remaining < 1.5:
            break
        resp = requests.get("https://www.bing.com/search", params=params, headers=_browser_headers(lang),
                            timeout=remaining)
        resp.raise_for_status()
        if "/captcha" in (getattr(resp, "url", "") or "") or "b_captcha" in resp.text[:20000]:
            raise SearchBlocked("Bing returned a captcha page")
        results = _relevant(_parse_bing(resp.text, max_results), query)
        if results:
            return results
    return []


# ---- wikipedia (MediaWiki API)
def _search_wikipedia(query: str, max_results: int, region: str, timeout: float) -> list[dict]:
    lang = _region_parts(region)[1]
    langs = [lang] if lang == "en" else [lang, "en"]
    last_error: Exception | None = None
    for wl in langs:
        try:
            resp = requests.get(
                f"https://{wl}.wikipedia.org/w/api.php",
                params={"action": "query", "format": "json", "formatversion": "2", "generator": "search",
                        "gsrsearch": _keyword_query(query), "gsrlimit": max_results, "prop": "extracts|info", "exintro": "1",
                        "explaintext": "1", "exsentences": "3", "exlimit": max_results, "inprop": "url",
                        "redirects": "1"},
                headers=_browser_headers(wl), timeout=timeout)
            resp.raise_for_status()
            results = _parse_wikipedia(resp.json(), wl, max_results)
        except (requests.RequestException, ValueError) as e:
            last_error = e
            log.debug("wikipedia (%s) failed: %s", wl, e)
            continue
        if results:
            return results
    if last_error is not None:
        raise last_error
    return []


def _parse_wikipedia(data: dict, lang: str, max_results: int) -> list[dict]:
    pages = ((data or {}).get("query") or {}).get("pages") or []
    if isinstance(pages, dict):  # formatversion=1 shape
        pages = list(pages.values())
    pages = sorted((p for p in pages if isinstance(p, dict) and p.get("title")), key=lambda p: p.get("index", 0))
    results = []
    for p in pages[:max_results]:
        url = p.get("fullurl") or f"https://{lang}.wikipedia.org/wiki/{quote(p['title'].replace(' ', '_'))}"
        results.append(_result(f"{p['title']} - Wikipedia", url, p.get("extract", "")))
    return results


# name -> backend; order matters. Tests (or users) may reorder / replace entries.
_BACKENDS: list[tuple[str, object]] = [
    ("ddgs", _search_ddgs),
    ("duckduckgo", _search_duckduckgo),
    ("bing", _search_bing),
    ("wikipedia", _search_wikipedia),
]


def _run_search(query: str, max_results: int = 5, region: str | None = None) -> tuple[list[dict], str]:
    """Try every backend in order; return (results, backend name). Returns ([], "") when the backends answered
    but found nothing; raises SearchBlocked when every backend failed."""
    region = _normalize_region(region, query)
    deadline = time.monotonic() + SEARCH_TIME_BUDGET
    errors: list[str] = []
    answered = False
    for name, backend in list(_BACKENDS):
        remaining = deadline - time.monotonic()
        if remaining < 2:
            errors.append(f"{name}: skipped (time budget exhausted)")
            continue
        try:
            results = backend(query, max_results, region, min(float(HTTP_TIMEOUT), remaining))  # type: ignore[operator]
        except Exception as e:  # noqa: BLE001 - one broken backend must not break search
            log.warning("web search backend %s failed: %s: %s", name, type(e).__name__, e)
            errors.append(f"{name}: {type(e).__name__}: {e}")
            continue
        answered = True
        results = _relevant([r for r in (results or []) if r.get("url")], query)[:max_results]
        if results:
            log.debug("web search %r answered by %s (%d results, region %s)", query, name, len(results), region)
            return results, name
        log.debug("web search backend %s returned no results for %r", name, query)
    if answered:
        return [], ""
    raise SearchBlocked("all search backends failed - " + "; ".join(errors))


def search_web(query: str, max_results: int = 5, region: str | None = None) -> list[dict]:
    """Structured web search [{"title","url","snippet"}] through the backend fallback chain
    (ddgs -> DuckDuckGo html/lite -> Bing -> Wikipedia). `region` like 'tr-tr' / 'us-en'; default auto-detected.
    Raises SearchBlocked only when every backend failed or was blocked."""
    return _run_search(query, max_results, region)[0]


def web_search(query: str, max_results: int = 5, region: str = "") -> str:
    """Search the web and return numbered results with title, URL and snippet (never raises)."""
    query = (query or "").strip()
    if not query:
        return "Error: empty search query."
    try:
        max_results = max(1, min(int(max_results), 10))
    except (TypeError, ValueError):
        max_results = 5
    try:
        results, source = _run_search(query, max_results, region or None)
    except Exception as e:
        log.warning("web_search failed: %s", e)
        return f"Error: web search failed ({type(e).__name__}: {e})"
    if not results:
        return f"No results found for: {query}"
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}\n   URL: {r['url']}\n   {r['snippet']}".rstrip())
    lines.append(f"Source: {source}")
    return "\n".join(lines)


# ---- fetch_url
_META_CHARSET = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([a-zA-Z0-9_\-:.]+)""", re.IGNORECASE)
_HEADER_CHARSET = re.compile(r"charset\s*=\s*[\"']?([a-zA-Z0-9_\-:.]+)", re.IGNORECASE)
_STRIP_TAGS = ("script", "style", "noscript", "svg", "nav", "footer", "header", "form", "iframe", "aside",
               "button", "template", "canvas", "dialog", "select")
_BLOCK_TAGS = ["p", "div", "li", "ul", "ol", "dl", "dt", "dd", "h1", "h2", "h3", "h4", "h5", "h6", "br", "hr", "tr",
               "td", "th", "table", "section", "article", "main", "blockquote", "pre", "figcaption", "caption",
               "address", "details", "summary"]


def _valid_codec(name: str | None) -> str | None:
    if not name:
        return None
    try:
        return codecs.lookup(name).name
    except LookupError:
        return None


def _decode_body(raw: bytes, content_type: str) -> str:
    """Decode bytes using (in order) the HTTP header charset, a BOM, a <meta charset>, strict UTF-8, then a
    statistical guess. Never trusts requests' ISO-8859-1 default for charset-less text/* responses."""
    candidates = []
    m = _HEADER_CHARSET.search(content_type or "")
    if m:
        candidates.append(m.group(1))
    if raw.startswith(codecs.BOM_UTF8):
        return raw[len(codecs.BOM_UTF8):].decode("utf-8", errors="replace")
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return raw.decode("utf-16", errors="replace")
    m = _META_CHARSET.search(raw[:4096])
    if m:
        candidates.append(m.group(1).decode("ascii", "ignore"))
    for c in candidates:
        codec = _valid_codec(c)
        if codec:
            return raw.decode(codec, errors="replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        from charset_normalizer import from_bytes
        best = from_bytes(raw[:200_000]).best()
        if best is not None and _valid_codec(best.encoding):
            return raw.decode(best.encoding, errors="replace")
    except Exception:  # noqa: BLE001 - optional helper
        pass
    return raw.decode("cp1252", errors="replace")


def _html_to_text(html: str) -> tuple[str, str]:
    """Return (title, readable text) - boilerplate removed, <article>/<main> preferred when substantial.
    JavaScript apps that only render server-side content inside <noscript> (e.g. Discourse) are handled by a
    second pass that keeps <noscript>."""
    title, text = _extract_html(html, keep_noscript=False)
    if len(text.strip()) < 200 and "<noscript" in html.lower():
        title2, text2 = _extract_html(html, keep_noscript=True)
        if len(text2.strip()) > len(text.strip()):
            return title or title2, text2
    return title, text


def _extract_html(html: str, keep_noscript: bool) -> tuple[str, str]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    if not title:
        og = soup.find("meta", attrs={"property": "og:title"})
        title = (og.get("content") or "").strip() if og else ""
    for tag in soup([t for t in _STRIP_TAGS if not (keep_noscript and t == "noscript")]):
        tag.decompose()
    for tag in soup.select('[role="navigation"], [role="banner"], [role="contentinfo"], [aria-hidden="true"]'):
        tag.decompose()
    # line breaks only around block-level elements, so inline tags (<a>, <b>, <span>) stay inside their sentence
    for tag in soup.find_all(_BLOCK_TAGS):
        if tag.name in ("td", "th"):
            tag.insert_after(" ")
        else:
            tag.insert_before("\n")
            tag.insert_after("\n")
    body = soup.body or soup
    full = body.get_text()
    best = None
    for cand in soup.find_all(["article", "main"]) + soup.select('[role="main"]'):
        txt = cand.get_text()
        if len(txt.strip()) >= 200 and (best is None or len(txt) > len(best)):
            best = txt
    text = best if best and len(best.strip()) >= 0.25 * len(full.strip()) else full
    return title, text


def _collapse_ws(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def fetch_url(url: str, max_chars: int = 6000) -> str:
    """Download a web page (redirects followed) and return its title and readable text."""
    url = (url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return "Error: only absolute http:// or https:// URLs are allowed."
    try:
        max_chars = max(200, min(int(max_chars), 50_000))
    except (TypeError, ValueError):
        max_chars = 6000
    try:
        with requests.get(url, headers=_browser_headers(), timeout=HTTP_TIMEOUT, stream=True,
                          allow_redirects=True) as r:
            r.raise_for_status()
            ctype = (r.headers.get("Content-Type") or "").lower()
            final_url = getattr(r, "url", None) or url
            if ctype and not any(t in ctype for t in ("text", "html", "xml", "json")):
                return f"Error: unsupported content type '{ctype}' (only text/HTML pages can be read)."
            chunks, size = [], 0
            for chunk in r.iter_content(65536):
                chunks.append(chunk)
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    break
            raw = b"".join(chunks)
    except Exception as e:
        return f"Error: could not fetch {url} ({type(e).__name__}: {e})"
    text = _decode_body(raw, ctype)
    title = ""
    if "html" in ctype or "<html" in text[:2000].lower() or "<!doctype html" in text[:2000].lower():
        title, text = _html_to_text(text)
    text = _collapse_ws(text)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n...[truncated, page has {len(text)} characters]"
    header = []
    if title:
        header.append(f"Title: {_clean(title, 300)}")
    if isinstance(final_url, str) and final_url != url:
        header.append(f"URL: {final_url}")
    body = text or "(page has no readable text)"
    return ("\n".join(header) + "\n\n" + body) if header else body


# --------------------------------------------------------------------------------------- calculator
_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}
_FUNCS = {name: getattr(math, name) for name in (
    "sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "exp", "log", "log10",
    "log2", "floor", "ceil", "degrees", "radians", "hypot", "gcd", "comb", "perm", "isqrt")}
_FUNCS.update({"abs": abs, "round": round, "min": min, "max": max, "pow": pow, "factorial": math.factorial})


def _check_pow(base, exp) -> None:
    """Refuse powers whose result would be absurdly large (CPU/memory DoS)."""
    if abs(exp) > 10_000:
        raise ValueError("exponent too large")
    if isinstance(base, int) and isinstance(exp, int) and exp > 0 and base.bit_length() * exp > 4_000_000:
        raise ValueError("result too large")


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow):
            _check_pow(left, right)
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        args = [_safe_eval(a) for a in node.args]
        if node.func.id in ("factorial", "comb", "perm") and any(abs(a) > 5000 for a in args):
            raise ValueError("argument too large")
        if node.func.id == "pow" and len(args) >= 2:
            _check_pow(args[0], args[1])
        return _FUNCS[node.func.id](*args)
    raise ValueError(f"unsupported expression element: {ast.dump(node)[:60]}")


def calculator(expression: str) -> str:
    """Evaluate a math expression safely."""
    expr = (expression or "").strip().replace("^", "**").replace("×", "*").replace("÷", "/")
    if not expr:
        return "Error: empty expression."
    if len(expr) > 500:
        return "Error: expression too long."
    try:
        value = _safe_eval(ast.parse(expr, mode="eval"))
    except ZeroDivisionError:
        return "Error: division by zero."
    except Exception as e:
        return f"Error: could not evaluate '{expression}': {e}"
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e16:
        value = int(value)
    return f"{expression} = {value}"


def current_time() -> str:
    """Current local date and time."""
    now = _dt.datetime.now().astimezone()
    utc = _dt.datetime.now(_dt.timezone.utc)
    return (f"Local: {now.strftime('%A, %Y-%m-%d %H:%M:%S')} ({now.tzname()}, UTC{now.strftime('%z')})\n"
            f"UTC:   {utc.strftime('%Y-%m-%d %H:%M:%S')}")


# ------------------------------------------------------------------------------------ workspace files
def _workspace() -> Path:
    ws = Path(WORKSPACE_DIR or config.WORKSPACE_DIR).resolve()
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def _resolve(path: str) -> Path:
    """Resolve ``path`` inside the workspace; raise ValueError on traversal / absolute paths outside it."""
    ws = _workspace()
    p = (ws / (path or ".")).resolve()
    if p != ws and ws not in p.parents:
        raise ValueError(f"path '{path}' is outside the workspace; use paths relative to the workspace")
    return p


def _rel(p: Path) -> str:
    rel = p.relative_to(_workspace()).as_posix()
    return rel or "."


def list_dir(path: str = ".") -> str:
    """List a workspace directory."""
    try:
        p = _resolve(path)
    except ValueError as e:
        return f"Error: {e}"
    if not p.exists():
        return f"Error: '{path}' does not exist in the workspace."
    if not p.is_dir():
        return f"Error: '{path}' is a file, not a directory."
    entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    if not entries:
        return f"Directory '{_rel(p)}' is empty."
    lines = [f"Contents of '{_rel(p)}':"]
    for e in entries[:500]:
        lines.append(f"  {e.name}/" if e.is_dir() else f"  {e.name}  ({e.stat().st_size} bytes)")
    if len(entries) > 500:
        lines.append(f"  ... and {len(entries) - 500} more")
    return "\n".join(lines)


def read_file(path: str) -> str:
    """Read a text file from the workspace."""
    try:
        p = _resolve(path)
    except ValueError as e:
        return f"Error: {e}"
    if not p.is_file():
        return f"Error: file '{path}' not found in the workspace."
    text = p.read_text(encoding="utf-8", errors="replace")
    return _truncate(text, MAX_READ_CHARS)


def write_file(path: str, content: str) -> str:
    """Create or overwrite a text file in the workspace."""
    try:
        p = _resolve(path)
    except ValueError as e:
        return f"Error: {e}"
    if p == _workspace() or p.is_dir():
        return f"Error: '{path}' is a directory."
    p.parent.mkdir(parents=True, exist_ok=True)
    data = content if isinstance(content, str) else str(content)
    p.write_text(data, encoding="utf-8")
    return f"Wrote {len(data)} characters to '{_rel(p)}'."


# -------------------------------------------------------------------------------------------- python
def run_python(code: str) -> str:
    """Run Python code in a fresh temporary directory with an isolated interpreter."""
    if not (code or "").strip():
        return "Error: no code given."
    with tempfile.TemporaryDirectory(prefix="neurox_py_") as tmp:
        script = Path(tmp) / "script.py"
        script.write_text(code, encoding="utf-8")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            proc = subprocess.run([sys.executable, "-I", "script.py"], cwd=tmp, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=PYTHON_TIMEOUT, env=env,
                                  stdin=subprocess.DEVNULL, **kwargs)
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or "") if isinstance(e.stdout, str) else ""
            return _truncate(f"Error: timed out after {PYTHON_TIMEOUT} s.\nPartial stdout:\n{out}")
        except Exception as e:
            return f"Error: could not start Python: {e}"
    parts = [f"Exit code: {proc.returncode}"]
    if proc.stdout:
        parts.append("STDOUT:\n" + proc.stdout.rstrip())
    if proc.stderr:
        parts.append("STDERR:\n" + proc.stderr.rstrip())
    if not proc.stdout and not proc.stderr:
        parts.append("(no output - use print() to show results)")
    return _truncate("\n".join(parts))


# --------------------------------------------------------------------------------------------- registry
def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


def get_tools(settings=None, llm=None) -> list[Tool]:
    return [
        Tool("web_search",
             "Search the internet. ALWAYS use it (instead of answering from memory) for anything current, recent "
             "or time-sensitive: news, events, prices, exchange rates, weather, sports results, software versions "
             "and releases, people's current roles, and any fact you are unsure about or that may have changed "
             "since your training. Returns numbered results with title, URL and a short snippet. Snippets are "
             "often incomplete: then call fetch_url on the most relevant result URL to read the page before "
             "answering, and cite the URLs you used. Write short keyword queries in the user's language.",
             _obj({"query": {"type": "string", "description": "Search keywords, e.g. 'python latest release' or "
                                                              "'dolar kuru bugün'"},
                   "max_results": {"type": "integer", "description": "Number of results (1-10, default 5)"},
                   "region": {"type": "string", "description": "Optional region-language code such as 'tr-tr', "
                                                               "'us-en', 'de-de'. Omit for automatic detection."}},
                  ["query"]),
             web_search, category="web"),
        Tool("fetch_url",
             "Download a web page (http/https only) and return its title and main readable text (menus, scripts "
             "and boilerplate removed). Use it to read the most relevant result found with web_search, or a link "
             "the user gave, when you need details the search snippet does not contain.",
             _obj({"url": {"type": "string", "description": "Absolute URL starting with http:// or https://"},
                   "max_chars": {"type": "integer", "description": "Maximum characters to return (default 6000)"}},
                  ["url"]),
             fetch_url, category="web"),
        Tool("calculator",
             "Evaluate a math expression exactly instead of computing it in your head. Supports + - * / // % ** "
             "(or ^), parentheses, pi, e, and functions sqrt, sin, cos, tan, log, log10, log2, exp, abs, round, "
             "floor, ceil, factorial, min, max, pow, hypot, gcd, comb, perm, degrees, radians.",
             _obj({"expression": {"type": "string", "description": "e.g. '(3.5 * 12) / sqrt(2) + 2^10'"}},
                  ["expression"]),
             calculator, category="general"),
        Tool("current_time",
             "Get the current local date, time, weekday and timezone. Use it whenever the answer depends on today's "
             "date or the current time.",
             _obj({}, []), current_time, category="general"),
        Tool("list_dir",
             "List files and folders in the user's Neurox workspace folder (a sandbox). Paths are relative to the "
             "workspace root; '.' is the root.",
             _obj({"path": {"type": "string", "description": "Relative directory path, default '.'"}}, []),
             list_dir, category="files"),
        Tool("read_file",
             "Read a text file from the Neurox workspace folder. The path is relative to the workspace root.",
             _obj({"path": {"type": "string", "description": "Relative file path, e.g. 'notes/todo.txt'"}},
                  ["path"]),
             read_file, category="files"),
        Tool("write_file",
             "Create or overwrite a text file in the Neurox workspace folder (parent folders are created). The "
             "path is relative to the workspace root. Requires user approval.",
             _obj({"path": {"type": "string", "description": "Relative file path, e.g. 'report.md'"},
                   "content": {"type": "string", "description": "Full text content to write"}},
                  ["path", "content"]),
             write_file, dangerous=True, category="files"),
        Tool("run_python",
             "Execute a self-contained Python 3 script and return its exit code, stdout and stderr. Use it for "
             "data processing, precise calculations, simulations or checking code. Runs in an empty temporary "
             "directory with a 20 second timeout; only the standard library and installed packages are available. "
             "Always print() the results you need. Requires user approval.",
             _obj({"code": {"type": "string", "description": "Complete Python source code"}}, ["code"]),
             run_python, dangerous=True, category="code"),
    ]
