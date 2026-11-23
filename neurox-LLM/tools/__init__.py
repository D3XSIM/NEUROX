"""Tool packages. Each module exposes ``get_tools(settings, llm=None) -> list[core.tools.Tool]``.

- tools.builtin       web_search, fetch_url, calculator, current_time, read_file, list_dir, write_file, run_python
- tools.memory_tools  remember, recall   (long-term vector memory)
- tools.computer      screenshot, mouse/keyboard control (only when settings.computer_use_enabled)
"""
from __future__ import annotations

import importlib
import logging

from core.tools import ToolRegistry

log = logging.getLogger(__name__)


def build_registry(settings, llm=None) -> ToolRegistry:
    reg = ToolRegistry()
    if not settings.tools_enabled:
        return reg
    for mod_name in ("tools.builtin", "tools.memory_tools"):
        try:
            mod = importlib.import_module(mod_name)
            for tool in mod.get_tools(settings, llm):
                if tool.name in settings.enabled_tools:
                    reg.register(tool)
        except Exception as e:
            log.warning("Could not load %s: %s", mod_name, e)
    if settings.computer_use_enabled:
        try:
            from tools import computer
            for tool in computer.get_tools(settings, llm):
                reg.register(tool)
        except Exception as e:
            log.warning("Computer-use tools unavailable: %s", e)
    return reg
