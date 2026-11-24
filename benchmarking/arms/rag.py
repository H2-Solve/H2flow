"""Arm B: model + textbook retrieval, i.e. RAG/main.py rag() as it is on main.

rag() retrieves the top 5 chunks from Elasticsearch, puts them in its prompt, and makes one
model call. The arm runs that function unchanged; only the model object is swapped for one that
goes through the harness's Ollama client (see _system.py), which keeps the system's
temperature (0) and applies the harness's num_ctx, num_predict and seed.

Grading: the RAG prompt does not ask for a marked final answer, so Math-Verify reads the whole
response (last \\boxed{} or expression). Check extraction failures in the report: they show
how often the system's answer is unreadable, which is a real property of this arm.

Leakage: the index contains the textbooks' Exercises chunks with solutions. The calculus set
is written for the benchmark, so this does not apply to it; the manifest records how many
Exercises chunks the index holds.
"""

from __future__ import annotations

import time

from ..datasets import Problem
from . import Arm, Attempt, RunContext
from ._system import (TOP_K, HarnessModel, Usage, check_index, load_system, make_attempt, patched,
                      recording_retriever, system_notes)


class RagArm(Arm):
    name = "rag"
    description = "model + textbook retrieval: RAG/main.py rag() on main"

    def __init__(self):
        self._system = None

    def setup(self, ctx: RunContext) -> dict:
        self._system = load_system()
        return {**system_notes(self._system, ctx, ("answer",)), **check_index()}

    def solve(self, problem: Problem, seed: int, ctx: RunContext) -> Attempt:
        system, usage = self._system, Usage()
        model = HarnessModel("answer", ctx, system.temperatures["answer"], seed, usage)
        targets = [(system.rag, "llm", model),
                   (system.rag, "get_documents", recording_retriever(system.rag.get_documents, usage))]
        start = time.perf_counter()
        try:
            with patched(targets):
                message = system.rag.rag(problem.problem, top_k=TOP_K)
        except Exception as exc:  # Elasticsearch or Ollama down, ...
            return make_attempt(usage, usage.last_content or "", start, route=f"{usage.phase}_error",
                                error=f"{exc.__class__.__name__}: {exc}")
        return make_attempt(usage, str(getattr(message, "content", message)), start, error=None, route="rag")
