"""Arm D: H2flow end to end, RAG/main.py rag_with_verification() as it is on main.

What the system does per question: retrieve 5 chunks; extract the intent (LLM -> JSON); solve it
with SymPy. If SymPy has no answer (unsupported), return a plain RAG answer. Otherwise tell the
model the SymPy result and ask for steps ending in "FINAL_ANSWER: <expr>", retrying up to 3
times until that answer matches SymPy.

The arm runs that function unchanged. It swaps the three model objects for harness ones (see
_system.py: system temperatures, harness num_ctx/num_predict/seed, usage recorded), records
what was retrieved, and runs the system's own solve_with_sympy in a child process with a 30 s
limit, because an in-process runaway integral would hang the whole run.

Routes (shown in the report):
  verified        the answer matched SymPy (graded on its FINAL_ANSWER, converted to LaTeX)
  unsupported     SymPy had no answer; plain RAG answer, graded on the whole response
  unverified      all retries failed; the system returns no answer (the UI shows a 422), so the
                  attempt is graded "error". The last response is kept in `response`.
  sympy_timeout   SymPy exceeded the limit (the deployed system would hang)
  <phase>_error   an exception during retrieval, intent, sympy, answer or solver calls

Because the verified prompt hands the model the SymPy result, accuracy on the verified route
mostly measures intent extraction + SymPy, much like the sympy arm. Report it that way.
"""

from __future__ import annotations

import time

from ..datasets import Problem
from . import Arm, Attempt, RunContext
from ._system import (SYMPY_TIMEOUT_S, TOP_K, HarnessModel, SympyPool, SympyTimeout, Usage, check_index,
                      final_answer_boxed, load_system, make_attempt, patched, recording_retriever,
                      system_notes)

MAX_RETRIES = 3  # rag_with_verification default


class FullSystemArm(Arm):
    name = "full"
    description = "H2flow end to end: RAG/main.py rag_with_verification() on main"

    def __init__(self):
        self._system = None
        self._sympy = SympyPool(SYMPY_TIMEOUT_S)

    def setup(self, ctx: RunContext) -> dict:
        self._system = load_system()
        return {**system_notes(self._system, ctx, ("answer", "solver", "intent")), **check_index(),
                "max_retries": MAX_RETRIES, "sympy_timeout_s": SYMPY_TIMEOUT_S}

    def solve(self, problem: Problem, seed: int, ctx: RunContext) -> Attempt:
        system, usage, state = self._system, Usage(), {}
        temps = system.temperatures

        def extract(question: str) -> dict:
            usage.phase = "intent"
            state["intent"] = system.verification.extract_math_intent(question)
            return state["intent"]

        def solve_with_sympy(intent: dict):
            usage.phase = "sympy"
            state["truth"] = self._sympy.solve(intent)
            return state["truth"]

        targets = [
            (system.rag, "llm", HarnessModel("answer", ctx, temps["answer"], seed, usage)),
            (system.rag, "solver_llm", HarnessModel("solver", ctx, temps["solver"], seed, usage)),
            (system.verification, "extractor_llm",
             HarnessModel("intent", ctx, temps["intent"], seed, usage, format="json")),
            (system.rag, "get_documents", recording_retriever(system.rag.get_documents, usage)),
            (system.rag, "solve_with_sympy", solve_with_sympy),
        ]
        start = time.perf_counter()
        error, text = None, None
        try:
            with patched(targets):
                text = system.rag.rag_with_verification(problem.problem, max_retries=MAX_RETRIES, top_k=TOP_K,
                                                        intent_extractor=extract)
        except SympyTimeout as exc:
            route, error = "sympy_timeout", str(exc)
        except ValueError as exc:
            if "Unable to verify" in str(exc):
                route, error = "unverified", str(exc)
            else:
                route, error = f"{usage.phase}_error", f"ValueError: {exc}"
        except Exception as exc:
            route, error = f"{usage.phase}_error", f"{exc.__class__.__name__}: {exc}"

        truth = state.get("truth")
        extra = {
            "intent": state.get("intent"),
            "sympy_truth": None if truth is None else str(truth),
            "solver_calls": sum(c["role"] == "solver" for c in usage.calls),
        }
        if error is not None:
            return make_attempt(usage, usage.last_content or "", start, error=error, route=route, extra=extra)

        if truth is None:  # unsupported: plain RAG answer
            return make_attempt(usage, text, start, error=None, route="unsupported", extra=extra)

        boxed = final_answer_boxed(text)
        extra["final_answer_latex"] = boxed
        return make_attempt(usage, text, start, error=None, route="verified", final_text=boxed, extra=extra)
