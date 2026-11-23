"""Built-in tools: web search, URL fetching, calculator, clock, sandboxed workspace files, Python execution.

Public helpers (importable by plugins / tests):
    web_search(query, max_results=5) -> str        formatted DuckDuckGo results
    search_web(query, max_results=5) -> list[dict] structured results [{"title","url","snippet"}]
    fetch_url(url, max_chars=6000) -> str
    calculator(expression) -> str                  safe AST evaluator (no eval)
    current_time() -> str
    list_dir(path=".") / read_file(path) / write_file(path, content)   confined to the workspace
    run_python(code) -> str                        isolated subprocess in a temp dir, 20 s timeout
    get_tools(settings, llm=None) -> list[Tool]
"""
from __future__ import annotations

import ast
import datetime as _dt
import logging
import math
import operator
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

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


class SearchBlocked(RuntimeError):
    """DuckDuckGo answered with its bot-challenge ("anomaly") page instead of results."""


_SEARCH_HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.8,tr;q=0.6"}


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
        results.append({"title": a.get_text(" ", strip=True), "url": url,
                        "snippet": snippet_tag.get_text(" ", strip=True) if snippet_tag else ""})
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
        results.append({"title": a.get_text(" ", strip=True), "url": url,
                        "snippet": snippets[i] if i < len(snippets) else ""})
        if len(results) >= max_results:
            break
    return results


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Structured DuckDuckGo search (html endpoint, lite endpoint as fallback).
    Raises requests.RequestException on network errors and SearchBlocked on a bot challenge."""
    blocked = False
    for url, parser in (("https://html.duckduckgo.com/html/", _parse_html_endpoint),
                        ("https://lite.duckduckgo.com/lite/", _parse_lite_endpoint)):
        resp = requests.post(url, data={"q": query}, headers=_SEARCH_HEADERS, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        if _is_challenge(resp):
            blocked = True
            continue
        return parser(resp.text, max_results)
    if blocked:
        raise SearchBlocked("DuckDuckGo returned a bot-challenge page (rate limited); try again in a few minutes")
    return []


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web (DuckDuckGo) and return numbered results with title, URL and snippet."""
    query = (query or "").strip()
    if not query:
        return "Error: empty search query."
    try:
        max_results = max(1, min(int(max_results), 10))
    except (TypeError, ValueError):
        max_results = 5
    try:
        results = search_web(query, max_results)
    except Exception as e:
        log.warning("web_search failed: %s", e)
        return f"Error: web search failed ({type(e).__name__}: {e})"
    if not results:
        return f"No results found for: {query}"
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}\n   URL: {r['url']}\n   {r['snippet']}".rstrip())
    return "\n".join(lines)


def fetch_url(url: str, max_chars: int = 6000) -> str:
    """Download a web page and return its readable text."""
    url = (url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return "Error: only absolute http:// or https:// URLs are allowed."
    try:
        max_chars = max(200, min(int(max_chars), 50_000))
    except (TypeError, ValueError):
        max_chars = 6000
    try:
        with requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=HTTP_TIMEOUT, stream=True) as r:
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "").lower()
            raw = b""
            for chunk in r.iter_content(65536):
                raw += chunk
                if len(raw) > MAX_DOWNLOAD_BYTES:
                    break
            encoding = r.encoding or r.apparent_encoding or "utf-8"
    except Exception as e:
        return f"Error: could not fetch {url} ({type(e).__name__}: {e})"
    if ctype and not any(t in ctype for t in ("text", "html", "xml", "json")):
        return f"Error: unsupported content type '{ctype}' (only text/HTML pages can be read)."
    text = raw.decode(encoding, errors="replace")
    if "html" in ctype or "<html" in text[:2000].lower():
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else ""
        for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "form", "iframe"]):
            tag.decompose()
        text = soup.get_text("\n")
        if title:
            text = f"Title: {title}\n\n{text}"
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n...[truncated, page has {len(text)} characters]"
    return text or "(page has no readable text)"


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
             "Search the internet (DuckDuckGo) for current information, news, facts, documentation or anything you "
             "are unsure about. Returns titles, URLs and short snippets. Use fetch_url on a result URL to read the "
             "full page. Write concise keyword queries.",
             _obj({"query": {"type": "string", "description": "Search keywords, e.g. 'python 3.13 release date'"},
                   "max_results": {"type": "integer", "description": "Number of results (1-10, default 5)"}},
                  ["query"]),
             web_search, category="web"),
        Tool("fetch_url",
             "Download a web page (http/https only) and return its readable text content, without HTML. Use it to "
             "read a page found with web_search or a link the user gave.",
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
