# H2 Solve UI

H2 Solve is the web page for asking H2flow a math problem. It shows the plain model's answer next to the RAG answer so you can compare them.

## Files

- `H2Solve_v2_0.html`: the page itself. Open it in a browser; there's no build step.
- `connector.py`: a small temporary stand-in that links the page to the local model. It is not the project's real backend and will be replaced by it.
- `requirements.txt`: the extra Python packages the connector needs.
- `start.bat` / `start.ps1`: one-click launcher for Windows (the `.bat` runs the `.ps1`).

## Run it

**Windows:** double-click `ui/start.bat`. It starts Ollama, sets up `.venv` and the packages if needed,
starts the connector, and opens the page. Keep its window open while you use the page; press Ctrl+C to stop.
You only need [Ollama](https://ollama.com) and Python installed.

**Manually**, from the repo root with Ollama running:

```
ollama pull gemma3:4b
pip install -r requirements.txt -r ui/requirements.txt
python -m uvicorn ui.connector:app --port 5001
```

Then open `ui/H2Solve_v2_0.html` in your browser. Run the command from the
repository root; the connector also supports direct execution from any
working directory as long as its dependencies are installed.

The RAG result card calls `POST /api/rag`. It requires Elasticsearch to be running,
the textbook to be indexed, and Ollama to have `gemma3:4b` available.
