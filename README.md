# 🧠 Neurox

**Neurox** is a local AI assistant with a Streamlit chat UI. It runs on your own machine against
[Ollama](https://ollama.com), [LM Studio](https://lmstudio.ai) or any OpenAI-compatible server, and adds a
plugin-based "cognitive pipeline", native tool calling with human approval, optional computer use,
long-term vector memory and learned user rules.

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue) ![License](https://img.shields.io/badge/License-MIT-purple)

## 🇹🇷 Kısa özet

Neurox, kendi bilgisayarında çalışan bir yapay zekâ asistanıdır (Streamlit arayüzü, Ollama / LM Studio /
OpenAI uyumlu sunucular). Model gerektiğinde **araç çağırabilir**: web araması, sayfa okuma, hesap makinesi,
saat, çalışma klasöründe dosya okuma/yazma, Python çalıştırma, uzun süreli hafıza. Tehlikeli araçlar (dosya yazma,
kod çalıştırma, fare/klavye kontrolü) **her seferinde senin onayını ister**. **Bilgisayar kontrolü** varsayılan
olarak kapalıdır; açıldığında model ekran görüntüsü alıp tıklayabilir ve yazabilir (görsel/vision model gerekir,
fareyi ekranın bir köşesine çekmek işlemi anında durdurur). Söylediğin kuralları ("bundan sonra bana efendim deme")
kalıcı olarak öğrenir, önemli bilgileri vektör hafızada saklar. Arayüz Türkçe ve İngilizce'dir.

Hızlı başlangıç: `ollama pull llama3.2 && ollama pull nomic-embed-text`, ardından
`cd NEUROX/neurox-LLM && pip install -r requirements.txt && streamlit run app.py`.

## ✨ Features

- **Cognitive plugin pipeline** – every message runs through ordered plugins (security gate, context compression,
  intent/persona routing, memory, rules, web pre-search, tutor mode, creativity, pre-mortem simulation,
  metacognition) before the final agent answers. The "thinking" panel shows what each plugin did.
- **Native tool calling with approval** – the model calls tools through the provider's native tool API.
  Tools flagged *dangerous* pause the agent and show an approval card (approve / deny) before they run.
  Models without tool support automatically fall back to plain chat.
- **Computer use (optional, off by default)** – screenshot, click, type, keys, scroll, drag and open apps.
  - Disabled until you switch it on in the sidebar; every action needs your approval (unless you enable auto-approve).
  - PyAutoGUI fail-safe is on: move the mouse into a **screen corner** to abort immediately.
  - Every action is logged to `data/computer_log.jsonl`.
  - Needs a **vision-capable** model so it can read the screenshots.
- **Any local server** – Ollama, LM Studio, or any OpenAI-compatible endpoint (llama.cpp server, vLLM, ...), with
  optional API key.
- **Long-term vector memory** – facts you ask it to remember (or that the model saves with the `remember` tool)
  are embedded with `nomic-embed-text` and recalled when relevant. Entries made with another embedding model are
  ignored until re-indexed.
- **Learned rules** – statements like "from now on, answer in English" or "bundan sonra bana efendim deme" are
  stored permanently and injected into every prompt; manage them in the sidebar.
- **Turkish / English UI**, XP & levels, chat history persistence, image attachments for vision models.
- **HTTP API** (FastAPI) for scripting, with dangerous tools disabled by default.

## 🛠️ Installation

Requirements: Python 3.11+ and a local model server.

```bash
git clone https://github.com/D3XSIM/NEUROX.git
cd NEUROX/neurox-LLM

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Option A – Ollama

Install [Ollama](https://ollama.com/download), then pull a chat model and the embedding model used for memory:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Ollama listens on `http://localhost:11434` (the default provider in Neurox).

### Option B – LM Studio

1. Download and load a model in [LM Studio](https://lmstudio.ai) (pick one with tool-use support, see below).
2. Load an embedding model too (e.g. `nomic-embed-text`) if you want long-term memory.
3. Start the local server (Developer tab → *Start Server*), default port **1234**.
4. In the Neurox sidebar choose provider **LM Studio** (`http://localhost:1234/v1`) and select the model.

Any other OpenAI-compatible server works the same way with provider **OpenAI compatible** and its base URL
(e.g. `http://localhost:8080/v1`) plus an API key if the server needs one.

## 🚀 Usage

### Chat UI

```bash
streamlit run app.py
```

Pick provider, chat model and embedding model in the sidebar, choose which tools are enabled, and optionally turn
on computer use. Settings are saved to `data/settings.json`.

### Windows .exe

`Neurox.exe` is a single-file desktop app: it opens Neurox in its own native window (Edge WebView2, built into
Windows 10/11) — no Python, no browser tab, no console. If WebView2 is missing it falls back to your browser.
Download it from the [Releases](https://github.com/D3XSIM/NEUROX/releases) page, or build it yourself:

```powershell
cd NEUROX\neurox-LLM
.\build_exe.ps1          # -> dist\Neurox.exe
```

Closing the window stops Neurox. User data (settings, chat history, memory, rules, XP) and the server log
(`neurox.log`) are stored in `%LOCALAPPDATA%\Neurox`. You still need Ollama or LM Studio running for the model itself.
From source you get the same window with `python neurox_launcher.py`.
The first start takes a few seconds while the exe unpacks.

### HTTP API

```bash
python server.py          # or: uvicorn server:app --port 8000
```

The server listens on `http://127.0.0.1:8000` (override with `NEUROX_API_HOST` / `NEUROX_API_PORT`) and uses the
settings saved by the UI.

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_input": "What is 17 * 23?", "history": []}'
```

The response contains `response`, `agent`, `blocked`, `tools_used`, `images` and a `trace`.
Other endpoints: `GET /health` (provider reachability, loaded plugins) and `GET /models`.

Because the API has no approval UI, **dangerous tools are removed** from API requests. Set
`NEUROX_API_ALLOW_DANGEROUS=1` only if you understand that they will then run without asking.

## 🤖 Recommended models

| Purpose | Models |
|---|---|
| Chat + tool calling | `llama3.1`, `llama3.2`, `qwen2.5`, `qwen3`, `mistral-nemo` |
| Vision (images, computer use) | `qwen2.5-vl`, `gemma3`, `llama3.2-vision` |
| Embeddings (memory) | `nomic-embed-text` |

Small models (≤3B) work for chat but are less reliable at multi-step tool use; 7B+ is recommended for the agent
and for computer use.

## 🧩 Architecture

```
user message ──► PluginManager (plugins/ in ascending order) ──► ChatPlugin ──► agent loop ──► answer
                                                                              │  ▲
                                                                   tool call  ▼  │ tool result
                                                                        ToolRegistry (tools/)
                                                                   dangerous? → pause for approval
```

| Order | Plugin | What it does |
|---:|---|---|
| 1 | `SecurityPlugin` | Blocks destructive requests ("delete all files", "tüm dosyaları sil", `rm -rf`...) unless the message starts or ends with the exact token `CONFIRM` / `ONAYLA`. |
| 2 | `ContextCompressionPlugin` | Summarizes old messages when the history exceeds `history_window`. |
| 10 | `OrchestratorPlugin` | Picks a persona (RESEARCHER / CODER / PLANNER / GENERALIST) and the latent need; keyword fast-path, LLM only when ambiguous. |
| 12 | `RulePlugin` | Learns, lists and deletes persistent user rules and injects them into the prompt. |
| 15 | `MemoryPlugin` | Stores explicit "remember …" facts and corrections; recalls relevant memories. |
| 20 | `SearchPlugin` | Runs one web search up front for research questions. |
| 25 | `CoPilotPlugin` | Tutor mode for "guide me / teach me / adım adım". |
| 90 | `CreativityPlugin` | Adds divergent perspectives for brainstorming requests. |
| 92 | `SimulatorPlugin` | Pre-mortem: likely failure reasons for plans and strategies. |
| 95 | `MetacognitionPlugin` | Adds honesty / grounding instructions based on available evidence. |
| 100 | `ChatPlugin` | Builds the system prompt and runs the tool-calling agent (`core/agent.py`). Runs without tools when the request was blocked. |
| 160 | `GamificationPlugin` | XP and levels. |

New plugins: drop a file into `plugins/` with a `NeuroxPlugin` subclass (see `core/plugin_interface.py` for the
shared context keys).

## 🔧 Tools

| Tool | Description | Dangerous |
|---|---|:---:|
| `web_search` | DuckDuckGo search (titles, URLs, snippets) | – |
| `fetch_url` | Download a page and return its readable text (http/https only) | – |
| `calculator` | Safe math evaluator (AST based, no `eval`) | – |
| `current_time` | Local date, time and timezone | – |
| `list_dir` / `read_file` | Browse / read files inside the workspace sandbox `data/workspace/` | – |
| `write_file` | Create or overwrite a file inside the workspace sandbox | ⚠️ yes |
| `run_python` | Run a Python script in an isolated temp dir (20 s timeout) | ⚠️ yes |
| `remember` / `recall` | Save to / search long-term vector memory | – |
| `screenshot`, `cursor_position`, `wait` | Computer use: look at the screen | – |
| `click`, `double_click`, `right_click`, `mouse_move`, `drag`, `scroll`, `type_text`, `press_keys`, `open_app` | Computer use: control mouse and keyboard | ⚠️ yes |

Dangerous tools always ask for approval unless **auto-approve** is switched on in the sidebar.
Computer-use tools are only registered when computer use is enabled.

## 📁 Project layout

```
NEUROX/
├── README.md
├── .github/workflows/ci.yml   # pytest on Python 3.11 / 3.12
└── neurox-LLM/
    ├── app.py                 # Streamlit UI
    ├── server.py              # FastAPI HTTP API
    ├── core/
    │   ├── config.py          # Settings + data paths (NEUROX_DATA_DIR)
    │   ├── llm.py             # Ollama / OpenAI-compatible clients, canonical message format
    │   ├── tools.py           # Tool + ToolRegistry
    │   ├── agent.py           # tool-calling loop with approval (run_agent / resume_agent)
    │   ├── pipeline.py        # Neurox: one chat turn for UI and API
    │   ├── plugin_interface.py
    │   └── plugin_manager.py
    ├── plugins/               # cognitive pipeline (see table above)
    ├── tools/                 # builtin.py, memory_tools.py, computer.py
    ├── utils/                 # vector_store.py, history_manager.py
    ├── ui/                    # Streamlit components, TR/EN strings
    ├── tests/                 # pytest suite (fake LLM, no network)
    └── data/                  # created at runtime, git-ignored
```

All runtime data (settings, chat history, memory, rules, XP, computer-use log, workspace) lives in
`neurox-LLM/data/`, or in the directory given by the `NEUROX_DATA_DIR` environment variable.

## 🧪 Testing

```bash
cd NEUROX/neurox-LLM
pip install -r requirements-dev.txt
python -m pytest -q
```

The suite uses a scripted fake LLM, a temporary data directory and a fake `pyautogui`, so it needs no model server,
no network and no display. CI runs it on every push and pull request.

## 📜 License

MIT – see [neurox-LLM/LICENSE](neurox-LLM/LICENSE).
