"""Grade a final answer against the gold answer with Math-Verify.

The grader is independent of the system under test: it never imports
verification.py, which the SymPy and full-system arms use internally.

Outcome classes (everything except "correct" counts as wrong in accuracy):
  correct, wrong_answer, extraction_failure (no parseable answer),
  truncated (generation hit num_predict), error (exception/timeout in the arm).
"""

from __future__ import annotations

import importlib.metadata
import logging

OUTCOMES = ("correct", "wrong_answer", "extraction_failure", "truncated", "error")

# Math-Verify's built-in timeouts use signals (Unix) or subprocesses, which do not
# work reliably on Windows. Disable them; raw outputs are kept so any run can be
# re-graded (python -m benchmarking.report --regrade).
_PARSE_KW = {"parsing_timeout": None}
_VERIFY_KW = {"timeout_seconds": None}

logging.getLogger("math_verify").setLevel(logging.ERROR)  # silence the "timeout disabled" warning


def _mv():
    try:
        import math_verify
    except ImportError as exc:  # pragma: no cover - exercised only without the dependency
        raise SystemExit("Math-Verify is missing: pip install -r benchmarking/requirements.txt") from exc
    return math_verify


def grader_info() -> dict:
    info = {"grader": "math-verify", "extraction": "default LatexExtractionConfig + ExprExtractionConfig",
            "timeouts": "disabled"}
    for pkg in ("math-verify", "sympy", "antlr4-python3-runtime"):
        try:
            info[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            info[pkg] = None
    return info


def parse_gold(answer_latex: str):
    mv = _mv()
    text = answer_latex.strip()
    if not (text.startswith("$") or text.startswith("\\(")):
        text = f"${text}$"
    return mv.parse(text, extraction_config=[mv.LatexExtractionConfig()], **_PARSE_KW)


def parse_prediction(text: str):
    return _mv().parse(text, **_PARSE_KW)


def grade(answer_latex: str, final_text: str | None, *, truncated: bool = False,
          error: str | None = None) -> dict:
    """-> {"outcome": ..., "extracted": str | None}."""
    if error:
        return {"outcome": "error", "extracted": None}
    if truncated:
        return {"outcome": "truncated", "extracted": None}
    pred = parse_prediction(final_text or "")
    if not pred:
        return {"outcome": "extraction_failure", "extracted": None}
    gold = parse_gold(answer_latex)
    if not gold:
        raise ValueError(f"gold answer could not be parsed: {answer_latex!r}")
    ok = _mv().verify(gold, pred, **_VERIFY_KW)
    return {"outcome": "correct" if ok else "wrong_answer", "extracted": str(pred[-1])}
