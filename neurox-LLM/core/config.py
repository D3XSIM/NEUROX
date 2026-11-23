"""Central configuration and data paths for Neurox.

All persistent files live under DATA_DIR (absolute path, independent of the
current working directory).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("NEUROX_DATA_DIR", PROJECT_ROOT / "data")).resolve()
WORKSPACE_DIR = DATA_DIR / "workspace"          # sandbox for file tools
SETTINGS_FILE = DATA_DIR / "settings.json"
HISTORY_FILE = DATA_DIR / "chat_history.json"
MEMORY_FILE = DATA_DIR / "memory.json"
RULES_FILE = DATA_DIR / "rules.json"
XP_FILE = DATA_DIR / "user_xp.json"
COMPUTER_LOG_FILE = DATA_DIR / "computer_log.jsonl"

PROVIDERS = {
    "ollama": "http://localhost:11434",
    "lmstudio": "http://localhost:1234/v1",
    "openai_compat": "http://localhost:8080/v1",
}


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    provider: str = "ollama"                 # "ollama" | "lmstudio" | "openai_compat"
    base_url: str = PROVIDERS["ollama"]
    api_key: str = ""                        # only for openai-compatible servers that need one
    chat_model: str = "llama3.2"
    embed_model: str = "nomic-embed-text"
    temperature: float = 0.7
    language: str = "tr"                     # UI language: "tr" | "en"
    tools_enabled: bool = True               # native tool calling on/off
    enabled_tools: list[str] = field(default_factory=lambda: [
        "web_search", "fetch_url", "calculator", "current_time",
        "read_file", "list_dir", "write_file", "run_python", "remember", "recall",
    ])
    computer_use_enabled: bool = False       # mouse/keyboard control (off by default)
    auto_approve: bool = False               # if False, dangerous tools need user approval
    max_agent_steps: int = 8
    history_window: int = 20                 # messages kept verbatim before compression
    show_thoughts: bool = True

    @classmethod
    def load(cls) -> "Settings":
        ensure_dirs()
        if SETTINGS_FILE.exists():
            try:
                raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
                known = {f.name for f in fields(cls)}
                return cls(**{k: v for k, v in raw.items() if k in known})
            except Exception as e:  # corrupt file -> defaults
                log.warning("Could not read settings: %s", e)
        return cls()

    def save(self) -> None:
        ensure_dirs()
        SETTINGS_FILE.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")


ensure_dirs()
