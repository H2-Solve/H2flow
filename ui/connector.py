"""Temporary connector between ui/H2Solve_v2_0.html and the local model.

Not the project backend; replace it once that is ready. See ui/README.md.

Run from the repo root (it imports benchmarking/):

    python -m uvicorn ui.connector:app --port 5001

Endpoints:
    GET  /api/model  {"model": "gemma3:4b"} if Ollama answers and the model is pulled, else 503
    POST /api/base   {"question": "..."} -> {"answer", "latency_s", "truncated"}, raw model, no retrieval
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from benchmarking.ollama_client import OllamaClient, OllamaError, default_host

MODEL = "gemma3:4b"
REPO_ROOT = Path(__file__).resolve().parent.parent

# On Windows "localhost" tries IPv6 first and Ollama listens on IPv4 only,
# which adds ~2 s to every call, so default to 127.0.0.1.
OLLAMA_URL = default_host() if os.environ.get("OLLAMA_HOST") else "http://127.0.0.1:11434"

# Same prompt and options as the benchmark's base arm, so the Default Result card
# shows what the baseline numbers measure.
BASE_PROMPT = (REPO_ROOT / "benchmarking" / "prompts" / "base_v1.txt").read_text(encoding="utf-8")
BASE_OPTIONS = {"temperature": 0.2, "num_ctx": 8192, "num_predict": 4096}

app = FastAPI(title="H2Solve connector")

# The page is opened from disk (origin "null"), so allow any origin. Local use only.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    categories: list[str] = []  # sent by the page, not used yet


@app.get("/api/model")
def model_status() -> dict:
    try:
        OllamaClient(host=OLLAMA_URL, timeout_s=2).model_info(MODEL)
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=f"Ollama or {MODEL} unavailable: {exc}") from exc
    return {"model": MODEL}


@app.post("/api/base")
def base_answer(body: Question) -> dict:
    # .replace, not .format: the prompt contains \boxed{}
    prompt = BASE_PROMPT.replace("{problem}", body.question.strip())
    try:
        res = OllamaClient(host=OLLAMA_URL).chat(MODEL, [{"role": "user", "content": prompt}], BASE_OPTIONS)
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"answer": res.content, "latency_s": round(res.latency_s, 1), "truncated": res.truncated}
