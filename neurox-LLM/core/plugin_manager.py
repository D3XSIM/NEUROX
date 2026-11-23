"""Discovers plugins in <project>/plugins and runs them in ascending ``order``.

Each plugin run is timed and recorded in ``context['trace']`` as
``{"type": "plugin", "name": ..., "ms": ..., "updates": [keys]}`` (plus ``"error"`` on failure,
or ``"skipped": "blocked"`` when SecurityPlugin blocked the turn). A failing plugin never stops the pipeline.
"""
from __future__ import annotations

import importlib.util
import inspect
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:  # plugins import ``core.*`` / ``tools.*`` regardless of cwd
    sys.path.insert(0, str(PROJECT_ROOT))

from core.plugin_interface import NeuroxPlugin  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_PLUGIN_DIR = PROJECT_ROOT / "plugins"


class PluginManager:
    def __init__(self, plugin_dir: str | os.PathLike | None = None):
        self.plugin_dir = str(Path(plugin_dir).resolve()) if plugin_dir else str(DEFAULT_PLUGIN_DIR)
        self.plugins: list[NeuroxPlugin] = []

    def discover_plugins(self) -> None:
        """Load every ``*.py`` (not starting with ``__``) in the plugin directory, then sort by order."""
        os.makedirs(self.plugin_dir, exist_ok=True)
        for filename in sorted(os.listdir(self.plugin_dir)):
            if filename.endswith(".py") and not filename.startswith("__"):
                self.load_plugin(filename)
        self._sort()

    def _sort(self) -> None:
        self.plugins.sort(key=lambda p: getattr(p, "order", 50))

    def load_plugin(self, filename: str) -> None:
        """Dynamically load a plugin file and instantiate every NeuroxPlugin subclass defined in it."""
        plugin_path = os.path.join(self.plugin_dir, filename)
        module_name = f"neurox_plugins.{filename[:-3]}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module  # lets dataclasses/pickle inside plugins resolve their module
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            log.exception("Error loading plugin file %s", filename)
            return
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, NeuroxPlugin) and obj is not NeuroxPlugin and not inspect.isabstract(obj) \
                    and obj.__module__ == module.__name__:
                try:
                    plugin = obj()
                    plugin.on_load()
                    self.plugins.append(plugin)
                    log.info("Loaded plugin: %s (order %s)", plugin.name, plugin.order)
                except Exception:
                    log.exception("Error instantiating plugin %s from %s", obj.__name__, filename)

    def register(self, plugin: NeuroxPlugin) -> None:
        """Add a plugin instance programmatically (e.g. in tests)."""
        self.plugins.append(plugin)
        self._sort()

    def execute_all(self, context: dict) -> dict:
        """Run all plugins sequentially in order, merging their returned updates into ``context``."""
        trace = context.setdefault("trace", [])
        for plugin in self.plugins:
            name = getattr(plugin, "name", type(plugin).__name__)
            if context.get("blocked") and not getattr(plugin, "runs_when_blocked", False):
                trace.append({"type": "plugin", "name": name, "ms": 0, "updates": [], "skipped": "blocked"})
                continue
            t0 = time.perf_counter()
            entry = {"type": "plugin", "name": name, "ms": 0, "updates": []}
            try:
                result = plugin.execute(context)
                if result:
                    context.update(result)
                    entry["updates"] = list(result.keys())
            except Exception as e:
                log.exception("Plugin %s failed", name)
                entry["error"] = f"{type(e).__name__}: {e}"
            entry["ms"] = round((time.perf_counter() - t0) * 1000, 1)
            trace.append(entry)
            on_event = context.get("on_event")
            if on_event:  # live "thinking" panel in the UI
                try:
                    on_event(entry)
                except Exception:
                    log.exception("on_event callback failed")
            log.debug("Plugin %s took %.1f ms, updates=%s", name, entry["ms"], entry["updates"])
        return context
