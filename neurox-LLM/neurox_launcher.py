"""Desktop launcher for Neurox.

Default: starts the Streamlit server in a hidden background process and shows the UI in a native
app window (pywebview / Edge WebView2). Closing the window stops Neurox. If a native window can't be
created, it falls back to the default browser.

Used as the entry point of the Windows build (``Neurox.exe``, see neurox.spec / build_exe.ps1),
but also works from source:  python neurox_launcher.py

Environment variables:
    NEUROX_HEADLESS=1   only run the server (no window, no browser) - for testing
    NEUROX_PORT=8501    fixed port (default: a free port)
    NEUROX_ADDRESS      bind address (default: localhost - Neurox can run code and control the mouse)
    NEUROX_DATA_DIR     where settings/history/memory live (frozen default: %LOCALAPPDATA%\\Neurox)
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

SERVER_FLAG = "--neurox-server"


def _app_dir() -> Path:
    # PyInstaller unpacks bundled files into sys._MEIPASS; from source it's this file's folder.
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def _data_dir() -> Path:
    # Frozen builds unpack to a temp folder that is deleted on exit, so user data must live elsewhere.
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "Neurox"


def _setup_env() -> Path:
    app_dir = _app_dir()
    if getattr(sys, "frozen", False):
        os.environ.setdefault("NEUROX_DATA_DIR", str(_data_dir()))
    os.chdir(app_dir)  # Streamlit reads .streamlit/config.toml from the working directory
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))
    return app_dir


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ----------------------------------------------------------------------------------------- server
def run_server(port: int | None, headless: bool = True) -> int:
    app_dir = _setup_env()
    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit", "run", str(app_dir / "app.py"),
        "--global.developmentMode=false",
        f"--server.headless={'true' if headless else 'false'}",
        # Only this computer may connect: Neurox can run code and control the mouse/keyboard.
        f"--server.address={os.environ.get('NEUROX_ADDRESS', 'localhost')}",
        "--browser.gatherUsageStats=false",
        "--server.fileWatcherType=none",
    ]
    if port:
        sys.argv.append(f"--server.port={port}")
    return stcli.main()


def _wait_ready(url: str, proc: subprocess.Popen, timeout: float = 90) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.4)
    return False


def _spawn_server(port: int) -> tuple[subprocess.Popen, Path]:
    data_dir = Path(os.environ.get("NEUROX_DATA_DIR", _app_dir() / "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path = data_dir / "neurox.log"
    if getattr(sys, "frozen", False):
        cmd = [sys.executable, SERVER_FLAG, str(port)]
    else:
        cmd = [sys.executable, str(Path(__file__).resolve()), SERVER_FLAG, str(port)]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            creationflags=flags, env=os.environ.copy())
    return proc, log_path


LOADING_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
html,body{height:100%;margin:0;background:#0b0d14;color:#e7e9f3;font-family:'Segoe UI',system-ui,sans-serif}
body{display:grid;place-items:center;background:radial-gradient(900px 500px at 70% -10%,rgba(139,92,246,.18),transparent 60%),#0b0d14}
.c{text-align:center}.o{width:84px;height:84px;margin:0 auto 22px;border-radius:26px;display:grid;place-items:center;font-size:42px;
background:linear-gradient(135deg,#8b5cf6,#6366f1 50%,#22d3ee);box-shadow:0 16px 60px rgba(139,92,246,.5);animation:f 2.4s ease-in-out infinite}
h1{margin:0;font-size:28px;font-weight:800;letter-spacing:-.02em}p{color:#8b91a7;margin:8px 0 0}
@keyframes f{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
.b{width:180px;height:4px;margin:22px auto 0;border-radius:4px;background:#232839;overflow:hidden}
.b i{display:block;height:100%;width:40%;border-radius:4px;background:linear-gradient(90deg,#8b5cf6,#22d3ee);animation:l 1.2s ease-in-out infinite}
@keyframes l{0%{transform:translateX(-100%)}100%{transform:translateX(260%)}}
</style></head><body><div class="c"><div class="o">🧠</div><h1>Neurox</h1><p>%MSG%</p><div class="b"><i></i></div></div></body></html>"""


def _error_html(msg: str) -> str:
    return LOADING_HTML.replace("%MSG%", msg).replace('<div class="b"><i></i></div>', "")


def run_desktop() -> int:
    _setup_env()
    port = int(os.environ.get("NEUROX_PORT") or _free_port())
    url = f"http://localhost:{port}"
    proc, log_path = _spawn_server(port)

    def stop_server() -> None:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    try:
        import webview  # pywebview -> Edge WebView2 on Windows
    except Exception:
        webview = None

    if webview is None:
        import webbrowser
        if _wait_ready(url, proc):
            webbrowser.open(url)
        try:
            return proc.wait()
        except KeyboardInterrupt:
            stop_server()
            return 0

    icon = _app_dir() / "assets" / "neurox.ico"
    window = webview.create_window(
        "Neurox", html=LOADING_HTML.replace("%MSG%", "Başlatılıyor… / Starting…"),
        width=1320, height=860, min_size=(900, 600), background_color="#0b0d14", text_select=True,
    )

    def load_when_ready() -> None:
        if _wait_ready(url, proc):
            window.load_url(url)
        else:
            window.load_html(_error_html(f"Neurox başlatılamadı. Ayrıntılar: {log_path}"))

    threading.Thread(target=load_when_ready, daemon=True).start()
    try:
        start_kwargs = {"icon": str(icon)} if icon.exists() else {}
        try:
            webview.start(**start_kwargs)
        except TypeError:  # older pywebview without the icon argument
            webview.start()
    except Exception:
        # No WebView2 runtime or GUI backend: use the browser instead.
        import webbrowser
        if proc.poll() is None and _wait_ready(url, proc):
            webbrowser.open(url)
            return proc.wait()
    finally:
        if webview is not None:
            stop_server()
    return 0


def self_test(out_path: str) -> int:
    """Diagnostics for the frozen build: checks that bundled modules import and web search works.
    Usage: Neurox.exe --neurox-selftest C:\\path\\report.txt"""
    _setup_env()
    lines = []
    try:
        import core.agent, core.pipeline, plugins, tools.computer  # noqa: F401
        from tools.builtin import search_web
        lines.append("imports: ok")
        t0 = time.time()
        results = search_web("Python programming language", max_results=3)
        lines.append(f"web_search: {len(results)} results in {time.time() - t0:.1f}s")
        lines += [f"  - {r.get('title')} | {r.get('url')}" for r in results]
    except Exception as e:
        lines.append(f"FAILED: {type(e).__name__}: {e}")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    return 0 if lines and not lines[-1].startswith("FAILED") else 1


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "--neurox-selftest":
        return self_test(sys.argv[2])
    if len(sys.argv) >= 2 and sys.argv[1] == SERVER_FLAG:
        port = int(sys.argv[2]) if len(sys.argv) > 2 else None
        return run_server(port, headless=True)
    if os.environ.get("NEUROX_HEADLESS", "").lower() in ("1", "true", "yes"):
        return run_server(int(os.environ["NEUROX_PORT"]) if os.environ.get("NEUROX_PORT") else None, headless=True)
    return run_desktop()


if __name__ == "__main__":
    sys.exit(main())
