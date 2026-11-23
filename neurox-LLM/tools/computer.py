"""Computer-use tools: let a vision-capable model see the screen and drive mouse/keyboard.

Exposes ``get_tools(settings, llm=None) -> list[Tool]`` (category "computer"), plus
``COMPUTER_USE_SYSTEM_PROMPT`` and ``is_available()``.

Coordinate system
-----------------
Screenshots are downscaled so the longest side is <= ``MAX_SIDE`` (1280 px). The model
always works in SCREENSHOT pixels (origin = top-left of the image it received). Every
action converts those coordinates to real screen pixels using the module-level geometry
of the last screenshot (``_GEOMETRY``), refuses points outside the screenshot and clamps
the result to the primary monitor.

Safety
------
- Every action tool is ``dangerous=True`` (the agent asks the user before running it).
- ``pyautogui.FAILSAFE`` is on: slamming the mouse into a screen corner aborts the action;
  the tool then reports "aborted by user (failsafe)".
- Every call is appended as a JSON line to ``core.config.COMPUTER_LOG_FILE``.
- pyautogui / mss / PIL are imported lazily so importing this module never fails on a
  headless machine.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from core.config import COMPUTER_LOG_FILE
from core.tools import Tool, ToolOutput

log = logging.getLogger(__name__)

MAX_SIDE = 1280                 # longest side of screenshots sent to the model
JPEG_QUALITY = 85               # screenshots are JPEG (much smaller than PNG for photos/UI)
SETTLE_SECONDS = 0.4            # wait after an action before the verification screenshot
MAX_WAIT_SECONDS = 10.0
MAX_TYPE_CHARS = 5000

COMPUTER_USE_SYSTEM_PROMPT = (
    "You can see and control the user's computer with the computer tools "
    "(screenshot, click, type_text, press_keys, scroll, drag, open_app, ...).\n"
    "- Always start with `screenshot` to see the current screen; never guess coordinates.\n"
    "- Coordinates are pixels in the LAST screenshot you received (origin top-left), not real screen pixels.\n"
    "- Work in small steps: act, then look at the returned screenshot to verify the result before the next step.\n"
    "- Prefer keyboard shortcuts (press_keys) and open_app over hunting for small UI elements.\n"
    "- Click inside a text field before type_text. Never type passwords or payment data; ask the user to do it.\n"
    "- If something unexpected happens (dialogs, warnings, permission prompts), stop and tell the user."
)


# --------------------------------------------------------------------------- geometry (pure)

@dataclass(frozen=True)
class Geometry:
    """Mapping between screenshot pixels and real screen pixels (primary monitor)."""
    screen_w: int
    screen_h: int
    shot_w: int
    shot_h: int
    left: int = 0               # primary monitor origin in virtual-screen coordinates
    top: int = 0

    @property
    def scale_x(self) -> float:
        return self.screen_w / self.shot_w

    @property
    def scale_y(self) -> float:
        return self.screen_h / self.shot_h


def compute_shot_size(screen_w: int, screen_h: int, max_side: int | None = None) -> tuple[int, int]:
    """Downscaled size keeping aspect ratio so that the longest side is <= max_side."""
    max_side = max_side or MAX_SIDE
    longest = max(screen_w, screen_h)
    if longest <= max_side:
        return screen_w, screen_h
    f = max_side / longest
    return max(1, round(screen_w * f)), max(1, round(screen_h * f))


def make_geometry(screen_w: int, screen_h: int, left: int = 0, top: int = 0,
                  max_side: int | None = None) -> Geometry:
    sw, sh = compute_shot_size(screen_w, screen_h, max_side)
    return Geometry(screen_w, screen_h, sw, sh, left, top)


def to_screen(x: Any, y: Any, geo: Geometry) -> tuple[int, int]:
    """Convert screenshot coordinates to real screen coordinates.

    Raises ValueError for non-numeric values or points outside the screenshot. The result
    is clamped to the monitor and nudged 1 px away from the four corners so that an
    action never trips pyautogui's fail-safe by itself.
    """
    try:
        fx, fy = float(x), float(y)
    except (TypeError, ValueError):
        raise ValueError(f"coordinates must be numbers, got x={x!r}, y={y!r}")
    if fx != fx or fy != fy:  # NaN
        raise ValueError("coordinates must be numbers, got NaN")
    if not (0 <= fx <= geo.shot_w) or not (0 <= fy <= geo.shot_h):
        raise ValueError(
            f"point ({x}, {y}) is outside the screenshot; valid range is "
            f"x 0..{geo.shot_w - 1}, y 0..{geo.shot_h - 1} (screenshot pixels)")
    rx = int(round((fx + 0.5) * geo.scale_x - 0.5))
    ry = int(round((fy + 0.5) * geo.scale_y - 0.5))
    rx = min(max(rx, 0), geo.screen_w - 1)
    ry = min(max(ry, 0), geo.screen_h - 1)
    if rx in (0, geo.screen_w - 1) and ry in (0, geo.screen_h - 1):   # fail-safe corner
        rx = 1 if rx == 0 else geo.screen_w - 2
        ry = 1 if ry == 0 else geo.screen_h - 2
    return rx + geo.left, ry + geo.top


def to_shot(rx: int, ry: int, geo: Geometry) -> tuple[int, int]:
    """Convert real screen coordinates to screenshot coordinates (clamped)."""
    x = int((rx - geo.left) / geo.scale_x)
    y = int((ry - geo.top) / geo.scale_y)
    return min(max(x, 0), geo.shot_w - 1), min(max(y, 0), geo.shot_h - 1)


# --------------------------------------------------------------------------- key parsing (pure)

_KEY_ALIASES = {
    "control": "ctrl", "ctl": "ctrl", "strg": "ctrl",
    "cmd": "win", "command": "win", "super": "win", "meta": "win", "windows": "win", "start": "win",
    "option": "alt", "opt": "alt", "altgr": "altright",
    "escape": "esc", "return": "enter", "del": "delete", "ins": "insert",
    "pgup": "pageup", "page_up": "pageup", "pgdn": "pagedown", "page_down": "pagedown", "pgdown": "pagedown",
    "arrowup": "up", "arrowdown": "down", "arrowleft": "left", "arrowright": "right",
    "uparrow": "up", "downarrow": "down", "leftarrow": "left", "rightarrow": "right",
    "spacebar": "space", " ": "space", "bksp": "backspace", "back": "backspace",
    "caps": "capslock", "caps_lock": "capslock", "prtsc": "printscreen", "print_screen": "printscreen",
    "plus": "+", "minus": "-",
}


def parse_keys(keys: Any, valid: set[str] | None = None) -> list[str]:
    """Normalise "ctrl+shift+t" / ["ctrl", "c"] / "enter" into a list of pyautogui key names."""
    if isinstance(keys, str):
        s = keys.strip()
        if not s:
            raise ValueError("no keys given")
        if s == "+":
            parts = ["+"]
        else:
            parts = [p for p in re.split(r"\s*\+\s*", s) if p != ""]
            if s.endswith("++"):          # e.g. "ctrl++"
                parts.append("+")
    elif isinstance(keys, (list, tuple)):
        parts = [str(k) for k in keys]
    else:
        raise ValueError(f"keys must be a string like 'ctrl+c' or a list of key names, got {keys!r}")
    if not parts or len(parts) > 6:
        raise ValueError("give between 1 and 6 keys")
    out = []
    for p in parts:
        k = p.strip() or p
        if len(k) > 1:
            k = _KEY_ALIASES.get(k.lower(), k.lower())
        elif k.isalpha():
            k = k.lower()
        elif k == " ":
            k = "space"
        if valid is not None and k not in valid:
            raise ValueError(f"unknown key {p!r}. Use names like ctrl, shift, alt, win, enter, esc, tab, "
                             f"backspace, delete, up, down, left, right, home, end, pageup, pagedown, f1..f12, a-z, 0-9")
        out.append(k)
    return out


# --------------------------------------------------------------------------- lazy backends

_LOCK = threading.RLock()
_GEOMETRY: Geometry | None = None       # geometry of the last screenshot sent to the model
_DPI_DONE = False
_PG = None


def _ensure_dpi_awareness() -> None:
    """Make screen capture and mouse coordinates use the same (physical) pixels on Windows."""
    global _DPI_DONE
    if _DPI_DONE or sys.platform != "win32":
        _DPI_DONE = True
        return
    _DPI_DONE = True
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)   # per-monitor aware
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def _display_problem() -> str | None:
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return "no graphical display (DISPLAY / WAYLAND_DISPLAY not set)"
    if sys.platform == "win32":
        try:
            import ctypes
            if ctypes.windll.user32.GetSystemMetrics(0) <= 0:
                return "no interactive desktop (screen width is 0)"
        except Exception as e:
            return f"cannot query the desktop: {e}"
    return None


def _pyautogui():
    global _PG
    if _PG is None:
        problem = _display_problem()
        if problem:
            raise RuntimeError(problem)
        _ensure_dpi_awareness()
        import pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05
        _PG = pyautogui
    return _PG


def is_available() -> tuple[bool, str]:
    """Whether computer use can work here: (ok, reason)."""
    problem = _display_problem()
    if problem:
        return False, problem
    missing = []
    for mod in ("pyautogui", "mss", "PIL"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if "pyautogui" in missing or "PIL" in missing:
        return False, "missing packages: " + ", ".join(missing) + " (pip install pyautogui mss pillow pyperclip)"
    try:
        geo = _screen_geometry()
    except Exception as e:
        return False, f"cannot read the screen: {type(e).__name__}: {e}"
    note = " (mss missing, using slower pyautogui capture)" if missing else ""
    return True, (f"ok: primary monitor {geo.screen_w}x{geo.screen_h}, "
                  f"screenshots {geo.shot_w}x{geo.shot_h}{note}")


def _screen_geometry() -> Geometry:
    """Current geometry of the primary monitor (does not change _GEOMETRY)."""
    _ensure_dpi_awareness()
    try:
        import mss
        with mss.mss() as sct:
            mon = sct.monitors[1]
            return make_geometry(mon["width"], mon["height"], mon["left"], mon["top"])
    except ImportError:
        pass
    w, h = _pyautogui().size()
    return make_geometry(int(w), int(h))


def _capture():
    """Grab the primary monitor. Returns (PIL.Image RGB, Geometry)."""
    _ensure_dpi_awareness()
    from PIL import Image
    img = geo = None
    try:
        import mss
        with mss.mss() as sct:
            mon = sct.monitors[1]
            raw = sct.grab(mon)
            img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
            geo = make_geometry(raw.size[0], raw.size[1], mon["left"], mon["top"])
    except Exception as e:
        log.debug("mss capture failed (%s), falling back to pyautogui", e)
    if img is None:
        img = _pyautogui().screenshot().convert("RGB")
        geo = make_geometry(img.width, img.height)
    return img, geo


def _take_screenshot() -> tuple[str, Geometry]:
    """Capture, downscale, encode. Updates the module-level geometry. Returns (base64 JPEG, geo)."""
    global _GEOMETRY
    from PIL import Image
    img, geo = _capture()
    if (img.width, img.height) != (geo.shot_w, geo.shot_h):
        img = img.resize((geo.shot_w, geo.shot_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    _GEOMETRY = geo
    return base64.b64encode(buf.getvalue()).decode("ascii"), geo


def _geometry() -> Geometry:
    """Geometry the model's coordinates refer to (last screenshot, else the current screen)."""
    global _GEOMETRY
    if _GEOMETRY is None:
        _GEOMETRY = _screen_geometry()
    return _GEOMETRY


def _cursor_text(pg, geo: Geometry) -> str:
    try:
        rx, ry = pg.position()
        sx, sy = to_shot(rx, ry, geo)
        return f"Mouse cursor is at ({sx}, {sy}) in screenshot pixels."
    except Exception:
        return ""


def _describe(geo: Geometry) -> str:
    return (f"Screenshot of the primary monitor: {geo.shot_w}x{geo.shot_h} px "
            f"(real screen {geo.screen_w}x{geo.screen_h}, scale {geo.scale_x:.3f}). "
            f"Use coordinates in this image's pixels: x 0..{geo.shot_w - 1} left->right, "
            f"y 0..{geo.shot_h - 1} top->bottom.")


# --------------------------------------------------------------------------- logging / wrapper

def _log_action(tool: str, args: dict, summary: str) -> None:
    try:
        clean = {}
        for k, v in args.items():
            if isinstance(v, str) and len(v) > 200:
                v = v[:200] + f"...(+{len(v) - 200} chars)"
            clean[k] = v
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
               "tool": tool, "args": clean, "result_summary": summary[:500]}
        COMPUTER_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(COMPUTER_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        log.warning("Could not write computer log: %s", e)


def _run(tool: str, args: dict, action: Callable[[], str], return_screenshot: bool = False,
         settle: float = SETTLE_SECONDS) -> ToolOutput:
    """Run an action under the lock, handle fail-safe/errors, log, optionally attach a screenshot."""
    with _LOCK:
        try:
            text = action()
        except ValueError as e:                       # bad input -> tell the model, nothing happened
            text = f"Error: {e}"
            _log_action(tool, args, text)
            return ToolOutput(text)
        except Exception as e:
            if type(e).__name__ == "FailSafeException":
                text = ("Action aborted by user (failsafe): the mouse was moved to a screen corner. "
                        "Stop and ask the user before doing anything else on the computer.")
            else:
                text = f"Error during {tool}: {type(e).__name__}: {e}"
            _log_action(tool, args, text)
            return ToolOutput(text)
        images: list[str] = []
        if return_screenshot:
            try:
                if settle > 0:
                    time.sleep(settle)
                b64, geo = _take_screenshot()
                images.append(b64)
                text += "\n" + _describe(geo)
                cur = _cursor_text(_PG, geo) if _PG is not None else ""
                if cur:
                    text += " " + cur
            except Exception as e:
                text += f"\n(Could not take a verification screenshot: {type(e).__name__}: {e})"
        _log_action(tool, args, text)
        return ToolOutput(text, images)


def _bool(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() not in ("false", "0", "no", "off", "")
    return bool(v)


# --------------------------------------------------------------------------- tool functions

def screenshot() -> ToolOutput:
    def act() -> str:
        return "Screenshot taken."
    return _run("screenshot", {}, act, return_screenshot=True, settle=0)


def mouse_move(x: float, y: float, return_screenshot: bool = True) -> ToolOutput:
    def act() -> str:
        pg = _pyautogui()
        rx, ry = to_screen(x, y, _geometry())
        pg.moveTo(rx, ry, duration=0.15)
        return f"Moved mouse to ({x}, {y})."
    return _run("mouse_move", {"x": x, "y": y}, act, _bool(return_screenshot))


def click(x: float, y: float, button: str = "left", clicks: int = 1,
          return_screenshot: bool = True) -> ToolOutput:
    def act() -> str:
        b = str(button).lower().strip()
        if b not in ("left", "right", "middle"):
            raise ValueError("button must be 'left', 'right' or 'middle'")
        try:
            n = int(clicks)
        except (TypeError, ValueError):
            raise ValueError("clicks must be an integer 1..3")
        if not 1 <= n <= 3:
            raise ValueError("clicks must be 1, 2 or 3")
        pg = _pyautogui()
        rx, ry = to_screen(x, y, _geometry())
        pg.moveTo(rx, ry, duration=0.1)
        pg.click(rx, ry, clicks=n, interval=0.08, button=b)
        what = {1: "Clicked", 2: "Double-clicked", 3: "Triple-clicked"}[n]
        return f"{what} {b} button at ({x}, {y})."
    return _run("click", {"x": x, "y": y, "button": button, "clicks": clicks}, act, _bool(return_screenshot))


def double_click(x: float, y: float, return_screenshot: bool = True) -> ToolOutput:
    return click(x, y, button="left", clicks=2, return_screenshot=return_screenshot)


def right_click(x: float, y: float, return_screenshot: bool = True) -> ToolOutput:
    return click(x, y, button="right", clicks=1, return_screenshot=return_screenshot)


def drag(from_x: float, from_y: float, to_x: float, to_y: float, button: str = "left",
         return_screenshot: bool = True) -> ToolOutput:
    def act() -> str:
        b = str(button).lower().strip()
        if b not in ("left", "right", "middle"):
            raise ValueError("button must be 'left', 'right' or 'middle'")
        geo = _geometry()
        fx, fy = to_screen(from_x, from_y, geo)
        tx, ty = to_screen(to_x, to_y, geo)
        pg = _pyautogui()
        pg.moveTo(fx, fy, duration=0.1)
        pg.mouseDown(fx, fy, button=b)
        try:
            pg.moveTo(tx, ty, duration=0.5)
        finally:
            pg.mouseUp(tx, ty, button=b)
        return f"Dragged ({from_x}, {from_y}) -> ({to_x}, {to_y}) with the {b} button."
    return _run("drag", {"from_x": from_x, "from_y": from_y, "to_x": to_x, "to_y": to_y, "button": button},
                act, _bool(return_screenshot))


def scroll(amount: int, x: float | None = None, y: float | None = None,
           return_screenshot: bool = True) -> ToolOutput:
    def act() -> str:
        try:
            n = int(amount)
        except (TypeError, ValueError):
            raise ValueError("amount must be an integer (positive = up, negative = down)")
        if n == 0 or abs(n) > 50:
            raise ValueError("amount must be between -50 and 50 and not 0")
        pg = _pyautogui()
        where = ""
        if x is not None or y is not None:
            if x is None or y is None:
                raise ValueError("give both x and y, or neither")
            rx, ry = to_screen(x, y, _geometry())
            pg.moveTo(rx, ry, duration=0.1)
            where = f" at ({x}, {y})"
        # On Windows pyautogui sends raw wheel deltas; one notch = 120.
        pg.scroll(n * 120 if sys.platform == "win32" else n)
        return f"Scrolled {'up' if n > 0 else 'down'} {abs(n)} notch(es){where}."
    return _run("scroll", {"amount": amount, "x": x, "y": y}, act, _bool(return_screenshot))


# ---- text input ---------------------------------------------------------------

def _clipboard_get() -> str | None:
    try:
        import pyperclip
        return pyperclip.paste()
    except Exception:
        pass
    if sys.platform == "win32":
        return _win_clipboard_get()
    return None


def _clipboard_set(text: str) -> bool:
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except Exception:
        pass
    if sys.platform == "win32":
        return _win_clipboard_set(text)
    return False


def _win_user32_kernel32():
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.GetClipboardData.restype = wintypes.HANDLE
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user32.SetClipboardData.restype = wintypes.HANDLE
    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    return ctypes, user32, kernel32


def _win_clipboard_get() -> str | None:
    try:
        ctypes, user32, kernel32 = _win_user32_kernel32()
        if not user32.OpenClipboard(None):
            return None
        try:
            h = user32.GetClipboardData(13)          # CF_UNICODETEXT
            if not h:
                return None
            p = kernel32.GlobalLock(h)
            try:
                return ctypes.wstring_at(p) if p else None
            finally:
                kernel32.GlobalUnlock(h)
        finally:
            user32.CloseClipboard()
    except Exception:
        return None


def _win_clipboard_set(text: str) -> bool:
    try:
        ctypes, user32, kernel32 = _win_user32_kernel32()
        data = text.encode("utf-16-le") + b"\x00\x00"
        h = kernel32.GlobalAlloc(0x0002, len(data))  # GMEM_MOVEABLE
        if not h:
            return False
        p = kernel32.GlobalLock(h)
        ctypes.memmove(p, data, len(data))
        kernel32.GlobalUnlock(h)
        for _ in range(5):                           # clipboard may be busy
            if user32.OpenClipboard(None):
                break
            time.sleep(0.05)
        else:
            return False
        try:
            user32.EmptyClipboard()
            return bool(user32.SetClipboardData(13, h))
        finally:
            user32.CloseClipboard()
    except Exception:
        return False


def _win_send_unicode(text: str) -> None:
    """Type any Unicode text (ğüşıöç, emoji...) with SendInput/KEYEVENTF_UNICODE."""
    import ctypes
    from ctypes import wintypes

    ULONG_PTR = ctypes.c_size_t

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                    ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

    class _U(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("u", _U)]

    KEYEVENTF_KEYUP, KEYEVENTF_UNICODE = 0x0002, 0x0004
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    pg = _pyautogui()
    for ch in text:
        if ch == "\n":
            pg.press("enter")
            continue
        if ch == "\r":
            continue
        if ch == "\t":
            pg.press("tab")
            continue
        units = ch.encode("utf-16-le")
        codes = [int.from_bytes(units[i:i + 2], "little") for i in range(0, len(units), 2)]
        events = []
        for c in codes:
            events.append(INPUT(1, _U(ki=KEYBDINPUT(0, c, KEYEVENTF_UNICODE, 0, 0))))
            events.append(INPUT(1, _U(ki=KEYBDINPUT(0, c, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0, 0))))
        arr = (INPUT * len(events))(*events)
        user32.SendInput(len(events), arr, ctypes.sizeof(INPUT))
        time.sleep(0.005)


def _paste_text(pg, text: str) -> bool:
    previous = _clipboard_get()
    if not _clipboard_set(text):
        return False
    time.sleep(0.05)
    pg.hotkey("command" if sys.platform == "darwin" else "ctrl", "v")
    time.sleep(0.3)                                  # let the app read the clipboard first
    if previous is not None:
        _clipboard_set(previous)
    return True


def type_text(text: str, method: str = "auto", return_screenshot: bool = True) -> ToolOutput:
    def act() -> str:
        if not isinstance(text, str) or text == "":
            raise ValueError("text must be a non-empty string")
        if len(text) > MAX_TYPE_CHARS:
            raise ValueError(f"text too long ({len(text)} chars, max {MAX_TYPE_CHARS})")
        m = str(method).lower().strip()
        if m not in ("auto", "paste", "keys"):
            raise ValueError("method must be 'auto', 'paste' or 'keys'")
        pg = _pyautogui()
        ascii_only = all(32 <= ord(c) < 127 or c in "\n\t" for c in text)
        if m in ("auto", "paste"):
            if _paste_text(pg, text):
                return f"Typed {len(text)} characters (via clipboard paste)."
            if m == "paste":
                raise RuntimeError("clipboard is not available")
        if sys.platform == "win32":
            _win_send_unicode(text)
            return f"Typed {len(text)} characters (via key events)."
        if not ascii_only:
            raise RuntimeError("cannot type non-ASCII text: no clipboard available on this system")
        pg.write(text, interval=0.01)
        return f"Typed {len(text)} characters (via key events)."
    return _run("type_text", {"text": text, "method": method}, act, _bool(return_screenshot))


def press_keys(keys: Any, presses: int = 1, return_screenshot: bool = True) -> ToolOutput:
    def act() -> str:
        pg = _pyautogui()
        names = parse_keys(keys, set(pg.KEYBOARD_KEYS))
        try:
            n = int(presses)
        except (TypeError, ValueError):
            raise ValueError("presses must be an integer 1..20")
        if not 1 <= n <= 20:
            raise ValueError("presses must be between 1 and 20")
        for i in range(n):
            if len(names) == 1:
                pg.press(names[0])
            else:
                pg.hotkey(*names, interval=0.03)
            if i < n - 1:
                time.sleep(0.05)
        return f"Pressed {'+'.join(names)}" + (f" x{n}." if n > 1 else ".")
    return _run("press_keys", {"keys": keys, "presses": presses}, act, _bool(return_screenshot))


# ---- apps --------------------------------------------------------------------

_BLOCKED_EXT = {".bat", ".cmd", ".ps1", ".psm1", ".vbs", ".vbe", ".js", ".jse", ".wsf", ".wsh",
                ".hta", ".scr", ".pif", ".com", ".reg", ".msi", ".msp", ".lnk", ".cpl", ".inf"}
_URL_RE = re.compile(r"^(https?://[^\s\"'<>]+|ms-settings:[A-Za-z0-9\-]*)$", re.IGNORECASE)
_NAME_RE = re.compile(r"^[\w .\-()\\/:~]+$", re.UNICODE)


def _validate_app(name: Any) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string (e.g. 'notepad', 'calc', 'chrome', or a full path)")
    n = name.strip().strip('"')
    if len(n) > 260:
        raise ValueError("name is too long")
    if _URL_RE.match(n):
        return n
    if any(c in n for c in '&|<>^"%`$;\'!\n\r\t*?'):
        raise ValueError("name contains forbidden characters; give just an app name or a file path (no arguments)")
    if not _NAME_RE.match(n):
        raise ValueError("name contains forbidden characters")
    if ":" in n and not re.match(r"^[A-Za-z]:[\\/]", n):
        raise ValueError("only plain app names, absolute paths, http(s) URLs and ms-settings: links are allowed")
    if n.startswith("\\\\") or n.startswith("//"):
        raise ValueError("network paths are not allowed")
    if os.path.splitext(n)[1].lower() in _BLOCKED_EXT:
        raise ValueError("opening scripts/installers/shortcuts is not allowed with open_app")
    if n.startswith("-"):
        raise ValueError("name must not start with '-'")
    is_path = bool(re.match(r"^([A-Za-z]:[\\/]|/)", n))
    if " " in n and not is_path:
        raise ValueError("arguments are not supported; give a single app name (e.g. 'notepad') or an absolute path")
    if re.search(r"\s[/-]\w", n):
        raise ValueError("arguments are not supported; give just the app name or path")
    return n


def open_app(name: str, wait_seconds: float = 2.0, return_screenshot: bool = True) -> ToolOutput:
    def act() -> str:
        target = _validate_app(name)
        try:
            w = float(wait_seconds)
        except (TypeError, ValueError):
            raise ValueError("wait_seconds must be a number")
        w = min(max(w, 0.0), MAX_WAIT_SECONDS)
        if sys.platform == "win32":
            try:
                os.startfile(target)      # ShellExecute: no shell parsing, finds PATH / App Paths apps
            except OSError as e:
                raise RuntimeError(f"Windows could not open {target!r}: {e.strerror or e}. "
                                   "Try the executable name (e.g. 'notepad', 'calc', 'mspaint', 'chrome', "
                                   "'winword') or press_keys('win') and type the app name.")
        else:
            import shutil
            import subprocess
            if sys.platform == "darwin":
                cmd = ["open", target] if ("/" in target or ":" in target) else ["open", "-a", target]
            elif shutil.which(target):
                cmd = [target]
            else:
                cmd = ["xdg-open", target]
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        if w:
            time.sleep(w)
        return f"Opened {target!r}" + (f" and waited {w:g}s." if w else ".")
    return _run("open_app", {"name": name, "wait_seconds": wait_seconds}, act, _bool(return_screenshot))


# ---- misc --------------------------------------------------------------------

def wait(seconds: float = 1.0, return_screenshot: bool = True) -> ToolOutput:
    def act() -> str:
        try:
            s = float(seconds)
        except (TypeError, ValueError):
            raise ValueError("seconds must be a number")
        s = min(max(s, 0.0), MAX_WAIT_SECONDS)
        time.sleep(s)
        return f"Waited {s:g} seconds."
    return _run("wait", {"seconds": seconds}, act, _bool(return_screenshot), settle=0)


def cursor_position() -> ToolOutput:
    def act() -> str:
        pg = _pyautogui()
        geo = _geometry()
        rx, ry = pg.position()
        sx, sy = to_shot(rx, ry, geo)
        return (f"Mouse cursor is at ({sx}, {sy}) in screenshot pixels "
                f"(screenshot {geo.shot_w}x{geo.shot_h}; real screen ({rx}, {ry})).")
    return _run("cursor_position", {}, act, False)


# --------------------------------------------------------------------------- registration

_COORD_NOTE = ("Coordinates are pixels of the LAST screenshot you received (origin top-left), "
               "NOT real screen pixels; they are converted automatically.")
_SHOT_PARAM = {"type": "boolean",
               "description": "Attach a fresh screenshot after the action so you can verify the result (default true)."}
_X = {"type": "number", "description": "x in screenshot pixels (0 = left edge)."}
_Y = {"type": "number", "description": "y in screenshot pixels (0 = top edge)."}


def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or []}


def get_tools(settings=None, llm=None) -> list[Tool]:
    """Computer-use tools. Requires a vision-capable model to make sense of screenshots."""
    global MAX_SIDE
    max_side = getattr(settings, "computer_max_side", None)
    if isinstance(max_side, int) and 320 <= max_side <= 4096:
        MAX_SIDE = max_side
    C = "computer"
    return [
        Tool("screenshot",
             "Capture the user's primary screen and return it as an image (read-only, but it does capture "
             "whatever is visible on screen). ALWAYS call this first before any mouse/keyboard action, and again "
             "whenever you are unsure what the screen shows. The image is downscaled (longest side <= 1280 px); "
             "all other computer tools take coordinates in THIS image's pixels. Requires a vision-capable model "
             "(e.g. qwen2.5-vl, gemma3, llava, llama3.2-vision).",
             _obj({}), screenshot, dangerous=False, category=C),
        Tool("click",
             "Click the mouse at a point on the screen. " + _COORD_NOTE + " Aim at the CENTER of the button/field "
             "you saw in the screenshot. Use clicks=2 to double-click, clicks=3 to select a line. Returns a new "
             "screenshot: check that the click did what you expected.",
             _obj({"x": _X, "y": _Y,
                   "button": {"type": "string", "enum": ["left", "right", "middle"], "description": "Mouse button (default left)."},
                   "clicks": {"type": "integer", "minimum": 1, "maximum": 3, "description": "Number of clicks (default 1)."},
                   "return_screenshot": _SHOT_PARAM}, ["x", "y"]),
             click, dangerous=True, category=C),
        Tool("double_click",
             "Double-click with the left button (e.g. to open a file or desktop icon). " + _COORD_NOTE,
             _obj({"x": _X, "y": _Y, "return_screenshot": _SHOT_PARAM}, ["x", "y"]),
             double_click, dangerous=True, category=C),
        Tool("right_click",
             "Right-click to open a context menu. " + _COORD_NOTE + " Look at the returned screenshot to pick a menu item.",
             _obj({"x": _X, "y": _Y, "return_screenshot": _SHOT_PARAM}, ["x", "y"]),
             right_click, dangerous=True, category=C),
        Tool("mouse_move",
             "Move the mouse pointer without clicking (e.g. to reveal a tooltip or hover menu). " + _COORD_NOTE,
             _obj({"x": _X, "y": _Y, "return_screenshot": _SHOT_PARAM}, ["x", "y"]),
             mouse_move, dangerous=True, category=C),
        Tool("drag",
             "Press the mouse button at (from_x, from_y), move to (to_x, to_y) and release (drag & drop, "
             "selecting text, moving windows or sliders). " + _COORD_NOTE,
             _obj({"from_x": {"type": "number", "description": "Start x (screenshot pixels)."},
                   "from_y": {"type": "number", "description": "Start y (screenshot pixels)."},
                   "to_x": {"type": "number", "description": "End x (screenshot pixels)."},
                   "to_y": {"type": "number", "description": "End y (screenshot pixels)."},
                   "button": {"type": "string", "enum": ["left", "right", "middle"], "description": "Mouse button (default left)."},
                   "return_screenshot": _SHOT_PARAM}, ["from_x", "from_y", "to_x", "to_y"]),
             drag, dangerous=True, category=C),
        Tool("scroll",
             "Scroll the mouse wheel. amount = number of wheel notches: positive scrolls UP, negative scrolls DOWN "
             "(about 3 lines per notch; try 5 or -5). Optionally give x, y (screenshot pixels) to scroll over a "
             "specific area first. Keyboard (pagedown/pageup/space) is often more reliable.",
             _obj({"amount": {"type": "integer", "description": "Notches; >0 up, <0 down (range -50..50)."},
                   "x": {"type": "number", "description": "Optional x to hover before scrolling."},
                   "y": {"type": "number", "description": "Optional y to hover before scrolling."},
                   "return_screenshot": _SHOT_PARAM}, ["amount"]),
             scroll, dangerous=True, category=C),
        Tool("type_text",
             "Type text into the focused field (click the field first). Supports any language including Turkish "
             "characters (ğüşıöçİ) and newlines. method 'auto' pastes via the clipboard (fast, restores the "
             "previous clipboard text) and falls back to key events; use method 'keys' for apps where pasting "
             "does not work (terminals, games). To press Enter/Tab or shortcuts use press_keys. Never type "
             "passwords or payment details.",
             _obj({"text": {"type": "string", "description": "Text to type (max 5000 chars)."},
                   "method": {"type": "string", "enum": ["auto", "paste", "keys"], "description": "How to enter the text (default auto)."},
                   "return_screenshot": _SHOT_PARAM}, ["text"]),
             type_text, dangerous=True, category=C),
        Tool("press_keys",
             "Press a key or keyboard shortcut. Prefer shortcuts over clicking small UI elements. Examples: "
             "'enter', 'esc', 'tab', 'ctrl+c', 'ctrl+v', 'ctrl+s', 'ctrl+l' (browser address bar), 'alt+tab', "
             "'alt+f4' (close window), 'win' (Start menu: then type_text an app name and press 'enter'), 'win+d', "
             "'ctrl+shift+esc'. Accepts 'ctrl+shift+t' or a list like [\"ctrl\", \"shift\", \"t\"].",
             _obj({"keys": {"type": "string",
                            "description": "Key or combo joined with '+', e.g. 'enter', 'ctrl+c', 'ctrl+shift+t'."},
                   "presses": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Repeat count (default 1), e.g. 'down' x5."},
                   "return_screenshot": _SHOT_PARAM}, ["keys"]),
             press_keys, dangerous=True, category=C),
        Tool("open_app",
             "Open an application, file or URL on Windows (like the Run dialog, without arguments). name can be an "
             "executable name ('notepad', 'calc', 'mspaint', 'explorer', 'chrome', 'msedge', 'winword', 'excel'), "
             "an absolute path to a file/folder/program, an http(s) URL, or an ms-settings: link. Scripts and "
             "installers are refused. If it fails, press 'win', type the app name and press 'enter' instead.",
             _obj({"name": {"type": "string", "description": "App name, absolute path, URL or ms-settings: link."},
                   "wait_seconds": {"type": "number", "description": "Seconds to wait for the app to appear (default 2, max 10)."},
                   "return_screenshot": _SHOT_PARAM}, ["name"]),
             open_app, dangerous=True, category=C),
        Tool("wait",
             "Wait for something to load or finish (max 10 seconds), then return a fresh screenshot.",
             _obj({"seconds": {"type": "number", "description": "Seconds to wait (0..10, default 1)."},
                   "return_screenshot": _SHOT_PARAM}),
             wait, dangerous=False, category=C),
        Tool("cursor_position",
             "Return the current mouse pointer position in screenshot pixels.",
             _obj({}), cursor_position, dangerous=False, category=C),
    ]


__all__ = ["get_tools", "is_available", "COMPUTER_USE_SYSTEM_PROMPT", "MAX_SIDE",
           "Geometry", "compute_shot_size", "make_geometry", "to_screen", "to_shot", "parse_keys"]
