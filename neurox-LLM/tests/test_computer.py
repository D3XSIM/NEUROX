"""tools.computer: importable without a display, safety flags, coordinate mapping - with a FAKE pyautogui.

No real mouse/keyboard input and no real screenshots happen here: ``pyautogui`` is replaced by a recorder
object in ``sys.modules`` and the screen capture function is faked.
"""
import base64
import json
import types

import pytest

from core import config
from core.config import Settings

computer = pytest.importorskip("tools.computer")  # module must import on a headless machine

READ_ONLY = {"screenshot", "wait", "cursor_position", "get_screen_size", "screen_size"}
MUST_BE_DANGEROUS = {"click", "double_click", "right_click", "mouse_move", "drag", "scroll", "type_text",
                     "press_keys", "open_app", "hotkey", "key_press", "type", "move_mouse"}


class FailSafeException(Exception):
    pass


class FakePyAutoGUI(types.ModuleType):
    """Records every call instead of moving the real mouse."""

    def __init__(self, fail_on=None):
        super().__init__("pyautogui")
        self.calls = []
        self.FAILSAFE = False
        self.PAUSE = 0.0
        self.FailSafeException = FailSafeException
        self._fail_on = fail_on

    def _rec(self, name):
        def f(*a, **kw):
            if name == self._fail_on:
                raise FailSafeException("corner")
            self.calls.append((name, a, kw))
        return f

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self._rec(name)

    def size(self):
        return (1920, 1080)

    def position(self):
        return (960, 540)


@pytest.fixture
def fake_pg(monkeypatch):
    pg = FakePyAutoGUI()
    monkeypatch.setitem(__import__("sys").modules, "pyautogui", pg)
    monkeypatch.setattr(computer, "_PG", None)
    monkeypatch.setattr(computer, "_display_problem", lambda: None)
    monkeypatch.setattr(computer, "_ensure_dpi_awareness", lambda: None)
    # model coordinates refer to a 1280x720 screenshot of a 1920x1080 screen
    monkeypatch.setattr(computer, "_GEOMETRY", computer.make_geometry(1920, 1080))
    monkeypatch.setattr(computer, "SETTLE_SECONDS", 0)
    return pg


@pytest.fixture
def fake_capture(monkeypatch):
    PIL = pytest.importorskip("PIL.Image")

    def capture():
        return PIL.new("RGB", (2560, 1440), (30, 60, 90)), computer.make_geometry(2560, 1440)
    monkeypatch.setattr(computer, "_capture", capture)


@pytest.fixture
def tools():
    return {t.name: t for t in computer.get_tools(Settings(computer_use_enabled=True), None)}


# ---------------------------------------------------------------------------------------- flags
def test_get_tools_without_display(tools):
    assert "screenshot" in tools and "click" in tools
    assert all(t.category == "computer" for t in tools.values())
    for t in tools.values():
        assert t.schema()["function"]["parameters"]["type"] == "object"


def test_all_action_tools_are_dangerous(tools):
    for name, t in tools.items():
        if name not in READ_ONLY:
            assert t.dangerous is True, f"{name} can act on the computer but is not dangerous"
    for name in MUST_BE_DANGEROUS & set(tools):
        assert tools[name].dangerous is True


def test_registry_only_has_computer_tools_when_enabled(fake_llm):
    from tools import build_registry
    off = build_registry(Settings(computer_use_enabled=False), fake_llm)
    assert not any(off.get(n).category == "computer" for n in off.names())
    on = build_registry(Settings(computer_use_enabled=True), fake_llm)
    assert "click" in on and "screenshot" in on


def test_computer_off_by_default():
    assert Settings().computer_use_enabled is False
    assert Settings().auto_approve is False


# ------------------------------------------------------------------------------------- geometry
def test_coordinate_mapping_and_failsafe_corner_nudge():
    geo = computer.make_geometry(2560, 1440)
    assert (geo.shot_w, geo.shot_h) == (1280, 720)
    assert computer.to_screen(640, 360, geo) == (1280, 720)
    # never land exactly in a fail-safe corner
    assert computer.to_screen(0, 0, geo) == (1, 1)
    assert computer.to_screen(1280, 720, geo) == (2558, 1438)
    with pytest.raises(ValueError):
        computer.to_screen(5000, 10, geo)
    with pytest.raises(ValueError):
        computer.to_screen("abc", 10, geo)


# -------------------------------------------------------------------------------- fake actions
def test_click_uses_fake_pyautogui_with_scaled_coords(fake_pg):
    out = computer.click(640, 360, return_screenshot=False)
    assert "Error" not in out.text, out.text
    clicks = [c for c in fake_pg.calls if c[0] == "click"]
    assert clicks and clicks[0][1][:2] == (960, 540)
    assert fake_pg.FAILSAFE is True       # fail-safe forced on


def test_click_outside_screenshot_does_nothing(fake_pg):
    out = computer.click(99999, 10, return_screenshot=False)
    assert out.text.startswith("Error")
    assert not [c for c in fake_pg.calls if c[0] == "click"]


def test_failsafe_abort_is_reported(monkeypatch, fake_pg):
    fake_pg._fail_on = "click"
    out = computer.click(100, 100, return_screenshot=False)
    assert "failsafe" in out.text.lower()


def test_actions_are_logged_to_data_dir(fake_pg, isolated_data_dir):
    computer.click(10, 10, return_screenshot=False)
    log_file = computer.COMPUTER_LOG_FILE
    assert log_file.resolve().is_relative_to(isolated_data_dir)
    rec = json.loads(log_file.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["tool"] == "click"


def test_screenshot_returns_downscaled_image(fake_capture, fake_pg):
    out = computer.screenshot()
    assert out.images, out.text
    raw = base64.b64decode(out.images[0])
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(raw))
    assert max(img.size) <= computer.MAX_SIDE
    assert "1280" in out.text
    assert not [c for c in fake_pg.calls if c[0] in ("click", "moveTo", "write", "press", "hotkey")]


def test_type_text_via_registry_is_gated_by_agent(fake_pg, tools):
    """Through the agent, a type_text call pauses for approval and never reaches pyautogui."""
    from core.agent import run_agent
    from core.llm import ToolCall
    from core.tools import ToolRegistry
    from fakes import FakeLLM
    reg = ToolRegistry()
    for t in tools.values():
        reg.register(t)
    res = run_agent(FakeLLM([ToolCall("type_text", {"text": "hello"})]), [{"role": "user", "content": "type"}], reg)
    assert res.pending_call is not None and res.pending_call.name == "type_text"
    assert fake_pg.calls == []
