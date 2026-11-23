"""Shared pytest setup for Neurox.

Guarantees for every test:
  * The project root is importable (``import core``, ``import tools`` ...).
  * NEUROX_DATA_DIR points at a throw-away temp directory *before* ``core.config`` is imported
    (core.config resolves its paths at import time), so tests never touch the real ``data/``.
  * Each test additionally gets its own fresh data dir: the path constants in ``core.config`` and
    every already-imported project module that copied them (``from core.config import MEMORY_FILE``)
    are re-pointed at ``tmp_path``.
  * No real network: sockets may only connect to localhost (asyncio on Windows needs loopback).
"""
from __future__ import annotations

import os
import shutil
import socket
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
for p in (str(PROJECT_ROOT), str(TESTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# --- must happen before anything imports core.config -------------------------------------------
_SESSION_DATA_DIR = Path(tempfile.mkdtemp(prefix="neurox-test-data-"))
os.environ["NEUROX_DATA_DIR"] = str(_SESSION_DATA_DIR)
os.environ.pop("NEUROX_API_ALLOW_DANGEROUS", None)
# Never let a test talk to a real display / input device by accident.
os.environ.setdefault("MPLBACKEND", "Agg")

import core.config as config  # noqa: E402  (intentionally after the env var is set)

assert Path(config.DATA_DIR) == _SESSION_DATA_DIR.resolve(), "core.config was imported before conftest set NEUROX_DATA_DIR"

from fakes import FakeLLM  # noqa: E402,F401  (re-exported for convenience)

_PATH_ATTRS = ("DATA_DIR", "WORKSPACE_DIR", "SETTINGS_FILE", "HISTORY_FILE", "MEMORY_FILE",
               "RULES_FILE", "XP_FILE", "COMPUTER_LOG_FILE")
_PROJECT_PACKAGES = ("core", "tools", "plugins", "neurox_plugins", "utils", "server", "app", "ui")
# every path value core.config has ever had (session + per-test) -> constant name; lets us re-point
# module-level copies made by modules that were first imported during an earlier test.
_KNOWN_PATHS: dict[str, str] = {str(getattr(config, k)): k for k in _PATH_ATTRS if hasattr(config, k)}


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_SESSION_DATA_DIR, ignore_errors=True)


_ROOT_PREFIX = os.path.normcase(str(PROJECT_ROOT)) + os.sep
_EXCLUDED_PREFIXES = tuple(os.path.normcase(str(PROJECT_ROOT / d)) + os.sep for d in (".venv", "venv", "tests"))
_module_cache: dict[str, bool] = {}


def _is_project_module(name: str, mod) -> bool:
    cached = _module_cache.get(name)
    if cached is not None:
        return cached
    if name.split(".")[0] in _PROJECT_PACKAGES:
        result = True
    else:
        f = os.path.normcase(str(getattr(mod, "__file__", None) or ""))
        result = f.startswith(_ROOT_PREFIX) and not f.startswith(_EXCLUDED_PREFIXES)
    _module_cache[name] = result
    return result


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Fresh DATA_DIR per test; patches core.config and any module-level copies of its paths."""
    data = (tmp_path / "data").resolve()
    new = {
        "DATA_DIR": data,
        "WORKSPACE_DIR": data / "workspace",
        "SETTINGS_FILE": data / "settings.json",
        "HISTORY_FILE": data / "chat_history.json",
        "MEMORY_FILE": data / "memory.json",
        "RULES_FILE": data / "rules.json",
        "XP_FILE": data / "user_xp.json",
        "COMPUTER_LOG_FILE": data / "computer_log.jsonl",
    }
    for name, mod in list(sys.modules.items()):
        if mod is None or not _is_project_module(name, mod):
            continue
        for attr, val in list(vars(mod).items()):
            if attr.isupper() and isinstance(val, (str, Path)) and str(val) in _KNOWN_PATHS:
                key = _KNOWN_PATHS[str(val)]
                monkeypatch.setattr(mod, attr, new[key] if isinstance(val, Path) else str(new[key]))
    _KNOWN_PATHS.update({str(v): k for k, v in new.items()})
    monkeypatch.setenv("NEUROX_DATA_DIR", str(data))
    config.ensure_dirs()
    assert not str(config.DATA_DIR).startswith(str(PROJECT_ROOT / "data"))
    yield data


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", ""}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail fast instead of touching the internet. Loopback stays allowed (asyncio self-pipes)."""
    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo

    def guarded_connect(self, address, *a, **kw):
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(address, tuple) and str(host) not in _LOCAL_HOSTS:
            raise OSError(f"Network access disabled in tests (tried {address!r})")
        return real_connect(self, address, *a, **kw)

    def guarded_getaddrinfo(host, *a, **kw):
        if host is not None and str(host) not in _LOCAL_HOSTS:
            raise socket.gaierror(f"Network access disabled in tests (DNS lookup of {host!r})")
        return real_getaddrinfo(host, *a, **kw)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    yield


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def settings():
    """Default settings (no disk I/O), computer use off, nothing auto-approved."""
    from core.config import Settings
    return Settings(computer_use_enabled=False, auto_approve=False)


@pytest.fixture
def project_cwd(monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT)
    return PROJECT_ROOT
