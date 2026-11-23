"""Temporary connector between ui/H2Solve_v2_0.html and the local model.

Not the project backend; see ui/README.md.

Run from the repo root (it imports benchmarking/):

    python -m uvicorn ui.connector:app --port 5001

Endpoints:
    GET  /api/model  {"model": "gemma3:4b"} if Ollama answers and the model is pulled, else 503
    POST /api/base   {"question": "..."} -> {"answer", "latency_s", "truncated"}, raw model, no retrieval
    POST /api/rag    {"question": "...", "top_k": 5} -> RAG answer, "verified" flag and citations
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

MODEL = "gemma3:4b"
REPO_ROOT = Path(__file__).resolve().parent.parent

# Make direct execution (`python ui/connector.py`) behave like the documented
# module invocation from the repository root.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sympy as sp

from benchmarking.ollama_client import OllamaClient, OllamaError, default_host
from RAG.main import (
    build_prompt,
    format_context,
    generate_answer,
    get_documents,
    rag_with_verification,
)
from verification import extract_final_answer_from_response, safe_parse

# On Windows "localhost" tries IPv6 first and Ollama listens on IPv4 only,
# which adds ~2 s to every call, so default to 127.0.0.1.
OLLAMA_URL = default_host() if os.environ.get(
    "OLLAMA_HOST") else "http://127.0.0.1:11434"

# Same prompt and options as the benchmark's base arm, so the Default Result card
# shows what the baseline numbers measure.
BASE_PROMPT = (REPO_ROOT / "benchmarking" / "prompts" /
               "base_v1.txt").read_text(encoding="utf-8")
BASE_OPTIONS = {"temperature": 0.2, "num_ctx": 8192, "num_predict": 4096}

app = FastAPI(title="H2Solve connector")

# The page is opened from disk (origin "null"), so allow any origin. Local use only.
app.add_middleware(CORSMiddleware, allow_origins=[
                   "*"], allow_methods=["*"], allow_headers=["*"])


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    categories: list[str] = []  # sent by the page, not used yet
    top_k: int = Field(default=5, ge=1, le=10)


def boxed_latex(text: str) -> str | None:
    """Contents of the last \\boxed{...}, matching nested braces. Already LaTeX."""
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    begin = start + len("\\boxed{")
    depth = 1
    for i in range(begin, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[begin:i].strip() or None
    return None


def final_answer_latex(text: str) -> str | None:
    """The FINAL_ANSWER line is SymPy syntax (e.g. 2*x**3/3); convert it to LaTeX."""
    answer = extract_final_answer_from_response(text)
    if not answer:
        return None
    try:
        return sp.latex(safe_parse(answer))
    except Exception:  # an unparseable answer just means no answer box
        return None


@app.get("/api/model")
def model_status() -> dict:
    try:
        OllamaClient(host=OLLAMA_URL, timeout_s=2).model_info(MODEL)
    except OllamaError as exc:
        raise HTTPException(
            status_code=503, detail=f"Ollama or {MODEL} unavailable: {exc}") from exc
    return {"model": MODEL}


@app.post("/api/base")
def base_answer(body: Question) -> dict:
    # .replace, not .format: the prompt contains \boxed{}
    prompt = BASE_PROMPT.replace("{problem}", body.question.strip())
    try:
        res = OllamaClient(host=OLLAMA_URL).chat(
            MODEL, [{"role": "user", "content": prompt}], BASE_OPTIONS)
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "answer": res.content,
        "final_latex": boxed_latex(res.content),
        "latency_s": round(res.latency_s, 1),
        "truncated": res.truncated,
    }


@app.post("/api/rag")
def rag_answer(body: Question) -> dict:
    """Return a verified, textbook-grounded answer with source metadata."""
    import time

    start = time.perf_counter()
    question = body.question.strip()
    verified = True
    try:
        documents = get_documents(question, top_k=body.top_k)
        try:
            answer = rag_with_verification(question, top_k=body.top_k, documents=documents)
        except ValueError as exc:
            if "Unable to verify" not in str(exc):
                raise
            # Verification can fail even when the answer is right (e.g. the intent
            # model garbles the formula SymPy checks against), and the pipeline
            # discards its answer, so generate a plain RAG answer and flag it.
            verified = False
            response = generate_answer(build_prompt(question, format_context(documents)))
            answer = str(getattr(response, "content", response))
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    citations = [
        {
            "book_id": document.get("book_id", ""),
            "module_id": document.get("module_id", ""),
            "chapter": document.get("chapter", ""),
            "title": document.get("title", ""),
            "section": document.get("section", ""),
            "excerpt": document.get("excerpt", document.get("text", "")[:500]),
        }
        for document in documents
    ]
    return {
        "answer": answer,
        "verified": verified,
        "final_latex": final_answer_latex(answer) or boxed_latex(answer),
        "citations": citations,
        "latency_s": round(time.perf_counter() - start, 1),
        "model": MODEL,
    }
