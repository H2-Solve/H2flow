"""Arm B: model + textbook retrieval, i.e. RAG/main.py rag() with real Elasticsearch retrieval.

PENDING. Blocked on system code that is not on main yet:
  TODO(rag-arm): RAG/main.py runs a sample query at import time; it must be importable
                 without side effects (and importable as a package) before the harness can call it.
  TODO(rag-arm): rag() still uses get_documents_dummy; get_documents must call
                 search.retrieve.get_documents.
  TODO(rag-arm): the RAG prompt does not ask for a marked final answer. The grader extracts
                 \\boxed{} / FINAL_ANSWER / the last expression, so it may run without one,
                 but extraction failures should be checked when this arm first runs.
  TODO(rag-arm): rag() hard-codes the model and does not return token counts; record llm_calls=1
                 and latency only, or have the system expose usage metadata.

When unblocked, solve() should be roughly:

    from RAG.main import rag
    start = time.perf_counter()
    message = rag(problem.problem)
    return Attempt(response=message.content, llm_calls=1, latency_s=time.perf_counter() - start,
                   prompt_tokens=None, completion_tokens=None)

Leakage control (benchmark side, already decided): run against an index without
Exercises chunks when the problem set is drawn from the indexed books.
"""

from __future__ import annotations

from ..datasets import Problem
from . import Arm, ArmUnavailable, Attempt, RunContext

BLOCKERS = [
    "RAG/main.py executes a query on import",
    "RAG/main.py rag() uses get_documents_dummy, not search.retrieve.get_documents",
]


class RagArm(Arm):
    name = "rag"
    ready = False
    description = "model + textbook retrieval (RAG/main.py rag())"

    def setup(self, ctx: RunContext) -> dict:
        raise ArmUnavailable("rag arm is pending: " + "; ".join(BLOCKERS) + " (see benchmarking/arms/rag.py)")

    def solve(self, problem: Problem, seed: int, ctx: RunContext) -> Attempt:
        raise ArmUnavailable("rag arm is pending")
