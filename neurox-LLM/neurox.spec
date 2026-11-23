# PyInstaller spec for Neurox.exe (single file).  Build:  .\build_exe.ps1
# The app's own source (app.py, core/, plugins/, tools/, ui/, utils/) is shipped as plain files because
# Streamlit runs app.py as a script and plugins are discovered from the plugins/ folder at runtime.
from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

datas, binaries, hiddenimports = [], [], []

for pkg in ("streamlit", "webview", "clr_loader", "pythonnet", "ddgs", "primp"):
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
    except Exception:
        pass
hiddenimports += ["clr", "webview.platforms.edgechromium", "webview.platforms.winforms"]

for pkg in ("streamlit", "ollama", "requests", "numpy", "fastapi", "pydantic", "pyautogui", "pillow", "mss",
            "beautifulsoup4", "pyperclip"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

# Third-party packages imported only by the app source files (not by the launcher), so PyInstaller
# can't see them by analysis.
for pkg in ("ollama", "httpx", "requests", "bs4", "numpy", "PIL", "mss", "pyautogui", "pyperclip",
            "pymsgbox", "pytweening", "pyscreeze", "pygetwindow", "mouseinfo",
            "ddgs", "primp", "lxml", "charset_normalizer"):
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        pass
hiddenimports += ["ctypes.wintypes", "sqlite3", "json", "uuid", "ast", "tempfile", "subprocess"]

datas += [
    ("app.py", "."),
    ("core", "core"),
    ("plugins", "plugins"),
    ("tools", "tools"),
    ("ui", "ui"),
    ("utils", "utils"),
    (".streamlit/config.toml", ".streamlit"),
    ("assets", "assets"),
]

a = Analysis(
    ["neurox_launcher.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["pytest", "tests", "tkinter.test", "matplotlib", "IPython"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="Neurox",
    console=False,  # native window app; server logs go to %LOCALAPPDATA%\Neurox\neurox.log
    icon="assets/neurox.ico",
    upx=False,
)
