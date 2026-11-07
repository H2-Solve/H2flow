"""Arm D: H2flow end to end, RAG/main.py rag_with_verification().

PENDING. Blocked on system code that is not written or merged yet:
  TODO(full-arm): rag_with_verification() is a stub that returns "" (RAG/main.py).
  TODO(full-arm): same import-time side effect and dummy retriever as the rag arm (see rag.py).
  TODO(full-arm): the harness needs per-question usage from the system to compare cost:
                 number of LLM calls, retries used, tokens. The function returns only a string today.
                 Either the system returns this alongside the answer, or this adapter counts calls
                 (e.g. by wrapping the Ollama client) once the implementation exists.
  TODO(full-arm): when the API (dylan/backend-api-starter) is merged, add an optional mode
                 that posts to POST /api/v1/questions so the deployed path is benchmarked too.

When unblocked, solve() should be roughly:

    from RAG.main import rag_with_verification
    start = time.perf_counter()
    text = rag_with_verification(problem.problem)
    return Attempt(response=text, latency_s=time.perf_counter() - start,
                   llm_calls=<from system>, prompt_tokens=None, completion_tokens=None)

The verification prompt asks for "FINAL_ANSWER: <sympy expression>"; Math-Verify reads that
line, so no extra answer parsing is needed here.
"""

from __future__ import annotations

from ..datasets import Problem
from . import Arm, ArmUnavailable, Attempt, RunContext

BLOCKERS = [
    "rag_with_verification() is not implemented (returns \"\")",
    "RAG/main.py executes a query on import and uses a dummy retriever",
]


class FullSystemArm(Arm):
    name = "full"
    ready = False
    description = "H2flow end to end (rag_with_verification)"

    def setup(self, ctx: RunContext) -> dict:
        raise ArmUnavailable("full arm is pending: " + "; ".join(BLOCKERS) + " (see benchmarking/arms/full.py)")

    def solve(self, problem: Problem, seed: int, ctx: RunContext) -> Attempt:
        raise ArmUnavailable("full arm is pending")
