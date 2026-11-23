# Neurox – application folder

This folder contains the Neurox application (Streamlit UI, HTTP API, plugins, tools and tests).
The full documentation – features, installation, models, architecture and tools – is in the
[main README](../README.md).

Quick start (from this folder):

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
ollama pull llama3.2 && ollama pull nomic-embed-text
streamlit run app.py              # chat UI
python server.py                  # optional HTTP API on http://127.0.0.1:8000
```

Tests: `pip install -r requirements-dev.txt` then `python -m pytest -q`.
