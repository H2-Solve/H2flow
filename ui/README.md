# H2 Solve UI

H2 Solve is the web page for asking H2flow a math problem. It shows the plain model's answer next to the RAG answer so you can compare them.

## Files

- `H2Solve_v2_0.html`: the page itself. Open it in a browser; there's no build step.
- `connector.py`: a small temporary stand-in that links the page to the local model. It is not the project's real backend and will be replaced by it.
- `requirements.txt`: the extra Python packages the connector needs.

## Run it

From the repo root, with [Ollama](https://ollama.com) running:

```
ollama pull gemma3:4b
pip install -r requirements.txt -r ui/requirements.txt
python -m uvicorn ui.connector:app --port 5001
```

Then open `ui/H2Solve_v2_0.html` in your browser.

## Not done yet

The RAG result card is a placeholder; only the plain model's answer is connected so far.
