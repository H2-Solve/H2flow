"""Arm C: H2flow's SymPy path on its own.

Uses the system's verification.py as it is on main:
  extract_math_intent (LLM -> JSON)  ->  solve_with_sympy  ->  answer is SymPy's result.
When the operation is "unsupported", SymPy fails, or SymPy exceeds the time limit,
the problem falls back to the base arm (same prompt and seed), as the full system would.

Limits inherited from verification.py (not changed here):
  - the intent model is hard-coded there (gemma3:4b, temperature 0), so --model and
    --temperature do not apply to the extraction call, and its token counts are not exposed;
  - the extraction is deterministic, so seeds only affect fallback answers.
"""

from __future__ import annotations

import concurrent.futures as cf
import time
from pathlib import Path

from ..datasets import Problem
from . import Arm, ArmUnavailable, Attempt, RunContext, add_usage
from .base import ask_model

REPO = Path(__file__).resolve().parents[2]
INTENT_PROMPT = REPO / "intent_extraction_prompt.txt"
SYSTEM_INTENT_MODEL = "gemma3:4b"  # mirrors verification.extractor_llm
SYMPY_TIMEOUT_S = 30


def _solve_to_latex(intent: dict) -> str | None:
    """Runs in a child process so a runaway integral can be killed."""
    import sympy as sp

    import verification

    result = verification.solve_with_sympy(intent)
    if result is None:
        return None
    if isinstance(result, (list, tuple)):
        result = sp.FiniteSet(*result)
    return sp.latex(result, ln_notation=True)


class SympyArm(Arm):
    name = "sympy"
    description = "verification.py intent extraction + SymPy; base model fallback"

    def __init__(self):
        self._pool: cf.ProcessPoolExecutor | None = None
        self._verification = None

    def setup(self, ctx: RunContext) -> dict:
        try:
            import verification  # noqa: F401  (needs sympy + langchain-ollama)
        except ImportError as exc:
            raise ArmUnavailable(f"cannot import verification.py: {exc}. "
                                 "Install requirements.txt and sympy, and run from the repo root.") from exc
        self._verification = verification
        notes = {"intent_model": SYSTEM_INTENT_MODEL, "sympy_timeout_s": SYMPY_TIMEOUT_S,
                 "fallback": "base arm", "prompt": ctx.prompt_name}
        if ctx.model != SYSTEM_INTENT_MODEL:
            notes["warning"] = f"intent extraction still uses {SYSTEM_INTENT_MODEL} (hard-coded in verification.py)"
        return notes

    def _run_sympy(self, intent: dict) -> str | None:
        if self._pool is None:
            self._pool = cf.ProcessPoolExecutor(max_workers=1)
        future = self._pool.submit(_solve_to_latex, intent)
        try:
            return future.result(timeout=SYMPY_TIMEOUT_S)
        except cf.TimeoutError:
            for proc in list(getattr(self._pool, "_processes", {}).values()):
                proc.terminate()
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
            raise

    def solve(self, problem: Problem, seed: int, ctx: RunContext) -> Attempt:
        total = Attempt(response="", llm_calls=1, prompt_tokens=None, completion_tokens=None)
        start = time.perf_counter()
        try:
            intent = self._verification.extract_math_intent(problem.problem, prompt_path=str(INTENT_PROMPT))
        except Exception as exc:  # bad JSON, connection error, ...
            intent, reason = None, f"intent extraction failed: {exc.__class__.__name__}: {exc}"
        total.latency_s = time.perf_counter() - start
        total.extra["intent"] = intent

        latex = None
        if intent is not None:
            reason = f"operation {intent.get('operation')!r} not solved by SymPy"
            if intent.get("operation") not in ("unsupported", "none", None):
                start = time.perf_counter()
                try:
                    latex = self._run_sympy(intent)
                except cf.TimeoutError:
                    reason = f"SymPy exceeded {SYMPY_TIMEOUT_S}s"
                except Exception as exc:
                    reason = f"SymPy error: {exc.__class__.__name__}: {exc}"
                total.latency_s += time.perf_counter() - start

        if latex is not None:
            total.response = f"SymPy result: \\boxed{{{latex}}}"
            total.route = "sympy"
            return total

        fallback = ask_model(problem, seed, ctx)
        add_usage(total, fallback)
        total.response, total.truncated, total.error = fallback.response, fallback.truncated, fallback.error
        total.route = "fallback"
        total.extra["fallback_reason"] = reason
        return total
