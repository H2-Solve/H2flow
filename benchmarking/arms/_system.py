"""Shared plumbing for the arms that run H2flow's own code (rag, full) without changing it.

The system code builds its own ChatOllama objects at import time (RAG/main.py `llm` and
`solver_llm`, verification.py `extractor_llm`). For the length of one solve() call the arms
swap those module attributes for HarnessModel objects, then put the originals back:

  - every call goes through the harness's Ollama client, so it gets the harness's
    num_ctx / num_predict, a recorded seed, token counts, latency and truncation;
  - each model keeps the system's own temperature (read from the original object), so the
    arm measures the system as configured, not the CLI --temperature;
  - the prompts, retrieval, SymPy check and retry loop are the system's, unchanged.

Seeds: call n of an attempt (n = 0, 1, ...) uses seed + 1000 * n. The first call therefore uses
the same seed as the base arm, and retries get different seeds, as they would in the deployed
system (where Ollama picks a random seed per call). With one fixed seed, retries 2 and 3 would
send the same prompt with the same seed and repeat retry 1 word for word.
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace

from . import ArmUnavailable, Attempt, RunContext

REPO = Path(__file__).resolve().parents[2]
SYSTEM_MODEL = "gemma3:4b"     # what RAG/main.py and verification.py hard-code
DEFAULT_INDEX = "textbook-chunks"  # RAG/main.get_documents default (H2FLOW_ES_INDEX overrides)
TOP_K = 5                      # RAG/main.py default
SYMPY_TIMEOUT_S = 30

_ROLES = {"human": "user", "user": "user", "ai": "assistant", "assistant": "assistant", "system": "system"}


# ---------- loading the system ----------

@dataclass
class System:
    rag: ModuleType            # RAG.main
    verification: ModuleType
    temperatures: dict         # role -> temperature of the system's own model object
    models: dict               # role -> model name the system hard-codes


def load_system() -> System:
    try:
        import RAG.main as rag_main
        import verification
    except ImportError as exc:
        raise ArmUnavailable(
            f"cannot import RAG/main.py: {exc}. Install the root requirements.txt and run from the repo root."
        ) from exc
    objects = {"answer": rag_main.llm, "solver": rag_main.solver_llm, "intent": verification.extractor_llm}
    return System(
        rag=rag_main,
        verification=verification,
        temperatures={role: getattr(obj, "temperature", None) for role, obj in objects.items()},
        models={role: getattr(obj, "model", None) for role, obj in objects.items()},
    )


def check_index() -> dict:
    """Confirm Elasticsearch is up and the index the system reads exists; return details for the manifest."""
    from search.index import get_client

    index = os.environ.get("H2FLOW_ES_INDEX", DEFAULT_INDEX)
    url = os.environ.get("H2FLOW_ES_URL") or "http://localhost:9201"
    base = get_client()
    client = base.options(request_timeout=10, max_retries=0)  # fail fast instead of retrying for a minute
    try:
        if not client.ping():
            raise ArmUnavailable(f"Elasticsearch is not reachable at {url} (docker compose up -d, or set H2FLOW_ES_URL)")
        if not client.indices.exists(index=index):
            raise ArmUnavailable(f"index '{index}' does not exist at {url}; build it with search/index.py "
                                 "(or set H2FLOW_ES_INDEX)")
        docs = client.count(index=index)["count"]
        exercises = client.count(index=index, query={"match_phrase": {"section": "Exercises"}})["count"]
    except ArmUnavailable:
        raise
    except Exception as exc:
        raise ArmUnavailable(f"Elasticsearch check failed at {url}: {exc.__class__.__name__}: {exc}") from exc
    finally:
        base.close()
    return {"es_url": url, "index": index, "index_docs": docs, "index_exercise_chunks": exercises}


def system_notes(system: System, ctx: RunContext, roles: tuple[str, ...]) -> dict:
    notes = {
        "model": ctx.model,
        "system_temperatures": {r: system.temperatures[r] for r in roles},
        "harness_options_applied": {k: v for k, v in ctx.options.items() if k != "temperature"},
        "seed_rule": "call n of an attempt uses seed + 1000*n",
        "top_k": TOP_K,
    }
    warnings = []
    if "temperature" in ctx.options and any(system.temperatures[r] != ctx.options["temperature"] for r in roles):
        warnings.append(f"--temperature {ctx.options['temperature']} is not used: each model keeps the system's "
                        "own temperature (system_temperatures)")
    if any(system.models[r] not in (None, ctx.model) for r in roles):
        warnings.append(f"system code hard-codes {SYSTEM_MODEL}; this run sends every call to {ctx.model}")
    if warnings:
        notes["warnings"] = warnings
        for w in warnings:
            print(f"note: {w}")
    return notes


# ---------- the stand-in model and usage recording ----------

@dataclass
class Usage:
    calls: list[dict] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    retrieval_latency_s: float = 0.0
    last_content: str | None = None
    last_truncated: bool = False
    phase: str = "start"

    def tokens(self, key: str) -> int | None:
        values = [c[key] for c in self.calls]
        return None if any(v is None for v in values) else sum(values)


def to_messages(value) -> list[dict]:
    """Convert whatever the system passes to .invoke() into Ollama chat messages, as ChatOllama does.

    RAG/main.py passes ChatPromptTemplate.format(...), which is a plain string, so it is sent as
    one user message (that is also what ChatOllama does with a string).
    """
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if hasattr(value, "to_messages"):
        value = value.to_messages()
    out = []
    for m in value:
        if isinstance(m, dict):
            out.append({"role": m["role"], "content": m["content"]})
        else:
            out.append({"role": _ROLES.get(getattr(m, "type", "human"), "user"), "content": str(m.content)})
    return out


class HarnessModel:
    """Replaces a ChatOllama object in the system code: .invoke(input) -> object with .content."""

    def __init__(self, role: str, ctx: RunContext, temperature: float | None, seed: int, usage: Usage,
                 format: str | None = None):
        self.role, self.ctx, self.seed, self.usage, self.format = role, ctx, seed, usage, format
        self.temperature = ctx.options.get("temperature") if temperature is None else temperature

    def invoke(self, value, *args, **kwargs):
        self.usage.phase = self.role
        n = len(self.usage.calls)
        options = {**self.ctx.options, "temperature": self.temperature, "seed": self.seed + 1000 * n}
        res = self.ctx.client.chat(self.ctx.model, to_messages(value), options, format=self.format)
        self.usage.calls.append({
            "role": self.role, "seed": options["seed"], "temperature": self.temperature,
            "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
            "latency_s": round(res.latency_s, 3), "done_reason": res.done_reason,
        })
        self.usage.last_content, self.usage.last_truncated = res.content, res.truncated
        return SimpleNamespace(content=res.content, response_metadata={"done_reason": res.done_reason})


def recording_retriever(original, usage: Usage):
    """Wrap RAG.main.get_documents so the attempt records what was retrieved."""
    def get_documents(*args, **kwargs):
        usage.phase = "retrieval"
        start = time.perf_counter()
        docs = original(*args, **kwargs)
        usage.retrieval_latency_s += time.perf_counter() - start
        usage.retrieved = [{k: d.get(k) for k in ("module_id", "book_id", "section", "title", "score")}
                           for d in docs]
        return docs
    return get_documents


@contextmanager
def patched(targets: list[tuple[object, str, object]]):
    """Set obj.name = value for each target, and restore the originals afterwards."""
    saved = [(obj, name, getattr(obj, name)) for obj, name, _ in targets]
    try:
        for obj, name, value in targets:
            setattr(obj, name, value)
        yield
    finally:
        for obj, name, value in saved:
            setattr(obj, name, value)


def make_attempt(usage: Usage, response: str, start: float, *, error: str | None, route: str,
                 final_text: str | None = None, extra: dict | None = None) -> Attempt:
    return Attempt(
        response=response,
        final_text=final_text,
        llm_calls=len(usage.calls),
        prompt_tokens=usage.tokens("prompt_tokens"),
        completion_tokens=usage.tokens("completion_tokens"),
        latency_s=time.perf_counter() - start,
        truncated=error is None and usage.last_truncated,
        error=error,
        route=route,
        extra={"calls": usage.calls, "retrieved": usage.retrieved,
               "retrieval_latency_s": round(usage.retrieval_latency_s, 3), **(extra or {})},
    )


# ---------- SymPy with a time limit ----------

class SympyTimeout(RuntimeError):
    pass


def _solve_raw(intent: dict):
    """Runs in a child process so a runaway integral can be killed. Returns SymPy's own result."""
    import verification

    return verification.solve_with_sympy(intent)


class SympyPool:
    def __init__(self, timeout_s: float = SYMPY_TIMEOUT_S):
        self.timeout_s = timeout_s
        self._pool: cf.ProcessPoolExecutor | None = None

    def solve(self, intent: dict):
        if self._pool is None:
            self._pool = cf.ProcessPoolExecutor(max_workers=1)
        future = self._pool.submit(_solve_raw, intent)
        try:
            return future.result(timeout=self.timeout_s)
        except cf.TimeoutError:
            for proc in list(getattr(self._pool, "_processes", {}).values()):
                proc.terminate()
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
            raise SympyTimeout(f"SymPy exceeded {self.timeout_s}s") from None


# ---------- reading the system's FINAL_ANSWER for the grader ----------

_FINAL = re.compile(r"FINAL_ANSWER:\s*(.+)", re.IGNORECASE)


def final_answer_boxed(text: str) -> str | None:
    """'FINAL_ANSWER: x**2*exp(x)' -> '\\boxed{x^{2} e^{x}}', or None if there is none or it won't parse.

    The system writes its final answer in Python/SymPy syntax, which Math-Verify does not read
    reliably, so it is converted to LaTeX here. This uses SymPy's own parser, not verification.py,
    so the grader stays independent of the system's checking code. Like the system, it reads the
    first FINAL_ANSWER line and drops a trailing "+ C".
    """
    match = _FINAL.search(text or "")
    if not match:
        return None
    answer = re.sub(r"\+\s*[cC]\b", "", match.group(1).strip().strip("`$*.")).strip()
    import sympy as sp
    from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application, parse_expr,
                                            standard_transformations)

    try:
        expr = parse_expr(answer, local_dict={"e": sp.E, "E": sp.E, "pi": sp.pi, "oo": sp.oo},
                          transformations=standard_transformations + (implicit_multiplication_application,
                                                                      convert_xor))
        if isinstance(expr, (list, tuple, set)):
            expr = sp.FiniteSet(*expr)
        return "\\boxed{" + sp.latex(expr, ln_notation=True) + "}"
    except Exception:
        return None
