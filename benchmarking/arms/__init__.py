"""Systems under test ("arms"). Every arm answers one problem at a time through the same interface.

  base       raw model, step-by-step prompt                         ready
  sympy      intent extraction -> SymPy, base model as fallback     ready (uses verification.py)
  rag        model + textbook retrieval (RAG/main.py rag())         pending, see rag.py
  full       rag_with_verification (H2flow end to end)              pending, see full.py
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..datasets import Problem
from ..ollama_client import OllamaClient


class ArmUnavailable(RuntimeError):
    """The arm depends on system code that is not written, merged or importable yet."""


@dataclass
class RunContext:
    client: OllamaClient
    model: str
    options: dict          # Ollama options shared by every model call (temperature, num_ctx, ...)
    prompt_template: str   # contains {problem}
    prompt_name: str


@dataclass
class Attempt:
    response: str                    # full text the system returned (kept for re-grading)
    final_text: str | None = None    # text the grader reads; defaults to response
    llm_calls: int = 0
    prompt_tokens: int | None = 0
    completion_tokens: int | None = 0
    latency_s: float = 0.0
    truncated: bool = False
    error: str | None = None
    route: str = "model"             # which path answered, e.g. "model", "sympy", "fallback"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["final_text"] is None:
            d["final_text"] = d["response"]
        return d


def add_usage(total: Attempt, part: Attempt) -> None:
    """Accumulate calls/tokens/latency of a sub-attempt (e.g. a fallback) into total."""
    total.llm_calls += part.llm_calls
    for key in ("prompt_tokens", "completion_tokens"):
        a, b = getattr(total, key), getattr(part, key)
        setattr(total, key, None if a is None or b is None else a + b)
    total.latency_s += part.latency_s


class Arm:
    name = "arm"
    ready = True
    description = ""

    def setup(self, ctx: RunContext) -> dict:
        """Check dependencies; return arm details for the manifest. Raise ArmUnavailable if blocked."""
        return {}

    def solve(self, problem: Problem, seed: int, ctx: RunContext) -> Attempt:
        raise NotImplementedError


def get_arm(name: str) -> Arm:
    from .base import BaseArm
    from .full import FullSystemArm
    from .rag import RagArm
    from .sympy_only import SympyArm

    arms = {a.name: a for a in (BaseArm(), SympyArm(), RagArm(), FullSystemArm())}
    if name not in arms:
        raise SystemExit(f"unknown arm '{name}'; choose from {sorted(arms)}")
    return arms[name]
