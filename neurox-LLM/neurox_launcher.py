"""Desktop launcher for Neurox: starts the Streamlit UI and opens it in the browser.

Used as the entry point of the Windows build (``Neurox.exe``, see neurox.spec / build_exe.ps1),
but also works from source:  python neurox_launcher.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _app_dir() -> Path:
    # PyInstaller unpacks bundled files into sys._MEIPASS; from source it's this file's folder.
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def _data_dir() -> Path:
    # Frozen builds unpack to a temp folder that is deleted on exit, so user data
    # (settings, chat history, memory, rules, XP) must live in a persistent folder.
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "Neurox"


def main() -> int:
    app_dir = _app_dir()
    if getattr(sys, "frozen", False):
        os.environ.setdefault("NEUROX_DATA_DIR", str(_data_dir()))
    os.chdir(app_dir)  # Streamlit reads .streamlit/config.toml from the working directory
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))

    print("=" * 60)
    print(" Neurox is starting... the browser will open automatically.")
    print(f" Data folder: {os.environ.get('NEUROX_DATA_DIR', app_dir / 'data')}")
    print(" Close this window to stop Neurox.")
    print("=" * 60, flush=True)

    from streamlit.web import cli as stcli

    headless = os.environ.get("NEUROX_HEADLESS", "").lower() in ("1", "true", "yes")  # don't open a browser
    sys.argv = [
        "streamlit", "run", str(app_dir / "app.py"),
        "--global.developmentMode=false",
        f"--server.headless={'true' if headless else 'false'}",
        # Only this computer may connect: Neurox can run code and control the mouse/keyboard.
        f"--server.address={os.environ.get('NEUROX_ADDRESS', 'localhost')}",
        "--browser.gatherUsageStats=false",
        "--server.fileWatcherType=none",
    ]
    if os.environ.get("NEUROX_PORT"):
        sys.argv.append(f"--server.port={int(os.environ['NEUROX_PORT'])}")
    return stcli.main()


if __name__ == "__main__":
    sys.exit(main())
