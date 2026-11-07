"""Build calculus_v1.jsonl: first-year calculus problems with SymPy-computed gold answers.

  python -m benchmarking.data.build_calculus

Problems are written for this benchmark (not taken from the indexed OpenStax books), so
retrieval cannot return the problem itself. Every gold answer is computed here with SymPy
from the stated expression and was checked by hand when the problem was added.
Only problems with a single exact answer are included (no indefinite integrals: "+ C"
answers need a derivative check the grader does not do).

Fields: id, problem, answer_latex (gold), answer_sympy, topic, expected_operation
(the operation verification.py's intent extraction should produce, or "unsupported" for
word problems the system is not expected to route to SymPy directly).
"""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp

x, t, r = sp.symbols("x t r")
OUT = Path(__file__).resolve().parent / "calculus_v1.jsonl"

# (id, topic, expected_operation, problem text, gold as a SymPy computation)
PROBLEMS = [
    ("lim-01", "limits", "limit", "Evaluate the limit as x approaches 0 of sin(5x)/x.",
     lambda: sp.limit(sp.sin(5 * x) / x, x, 0)),
    ("lim-02", "limits", "limit", "Find the limit of (3x^2 - x)/(2x^2 + 7) as x approaches infinity.",
     lambda: sp.limit((3 * x**2 - x) / (2 * x**2 + 7), x, sp.oo)),
    ("lim-03", "limits", "limit", "Compute lim_{x -> 2} (x^2 - 4)/(x - 2).",
     lambda: sp.limit((x**2 - 4) / (x - 2), x, 2)),
    ("lim-04", "limits", "limit", "What is the limit of x ln(x) as x approaches 0 from the right?",
     lambda: sp.limit(x * sp.log(x), x, 0, dir="+")),
    ("lim-05", "limits", "limit", "Evaluate lim_{x -> 0} (1 - cos x)/x^2.",
     lambda: sp.limit((1 - sp.cos(x)) / x**2, x, 0)),
    ("lim-06", "limits", "limit", "Find the limit as x goes to infinity of (1 + 2/x)^x.",
     lambda: sp.limit((1 + 2 / x) ** x, x, sp.oo)),
    ("der-01", "derivatives", "derivative", "Differentiate f(x) = x^3 sin(x).",
     lambda: sp.diff(x**3 * sp.sin(x), x)),
    ("der-02", "derivatives", "derivative", "Let f(x) = e^(2x)/x. Find f'(1).",
     lambda: sp.diff(sp.exp(2 * x) / x, x).subs(x, 1)),
    ("der-03", "derivatives", "derivative", "Find the derivative of ln(x^2 + 1) with respect to x.",
     lambda: sp.diff(sp.log(x**2 + 1), x)),
    ("der-04", "derivatives", "derivative", "Find the second derivative of x^4 - 3x^2 evaluated at x = 1.",
     lambda: sp.diff(x**4 - 3 * x**2, x, 2).subs(x, 1)),
    ("der-05", "derivatives", "derivative", "What is d/dx of sqrt(1 + x^2)?",
     lambda: sp.diff(sp.sqrt(1 + x**2), x)),
    ("der-06", "derivatives", "derivative", "Find the slope of the tangent line to y = x e^x at x = 0.",
     lambda: sp.diff(x * sp.exp(x), x).subs(x, 0)),
    ("der-07", "derivatives", "derivative", "Differentiate arctan(2x).",
     lambda: sp.diff(sp.atan(2 * x), x)),
    ("int-01", "integrals", "integral", "Evaluate the integral from 0 to 2 of (3x^2 - 2x + 1) dx.",
     lambda: sp.integrate(3 * x**2 - 2 * x + 1, (x, 0, 2))),
    ("int-02", "integrals", "integral", "Compute the definite integral of sin(x) from 0 to pi.",
     lambda: sp.integrate(sp.sin(x), (x, 0, sp.pi))),
    ("int-03", "integrals", "integral", "Evaluate the integral of 1/x from 1 to e.",
     lambda: sp.integrate(1 / x, (x, 1, sp.E))),
    ("int-04", "integrals", "integral", "Find the integral from 0 to 1 of x e^x dx.",
     lambda: sp.integrate(x * sp.exp(x), (x, 0, 1))),
    ("int-05", "integrals", "integral", "Evaluate the improper integral of e^(-2x) from 0 to infinity.",
     lambda: sp.integrate(sp.exp(-2 * x), (x, 0, sp.oo))),
    ("int-06", "integrals", "integral", "Compute the integral from 0 to 1 of 1/(1 + x^2) dx.",
     lambda: sp.integrate(1 / (1 + x**2), (x, 0, 1))),
    ("app-01", "applications", "unsupported", "Find the area of the region enclosed by y = x and y = x^2.",
     lambda: sp.integrate(x - x**2, (x, 0, 1))),
    ("app-02", "applications", "unsupported",
     "Find all critical points of f(x) = x^3 - 3x. Give the answer as a set of x-values.",
     lambda: sp.FiniteSet(*sp.solve(sp.diff(x**3 - 3 * x, x), x))),
    ("app-03", "applications", "unsupported",
     "A particle moves along a line with position s(t) = t^3 - 6t^2 + 9t. What is its velocity at t = 2?",
     lambda: sp.diff(t**3 - 6 * t**2 + 9 * t, t).subs(t, 2)),
    ("app-04", "applications", "unsupported", "What is the maximum value of f(x) = x(10 - x)?",
     lambda: sp.maximum(x * (10 - x), x)),
    ("app-05", "applications", "unsupported",
     "The region under y = sqrt(x) from x = 0 to x = 4 is rotated about the x-axis. Find the volume of the solid.",
     lambda: sp.pi * sp.integrate(sp.sqrt(x) ** 2, (x, 0, 4))),
    ("app-06", "applications", "unsupported",
     "The radius of a circle increases at 2 cm/s. How fast is the area increasing, in cm^2/s, when the radius is 5 cm?",
     lambda: (sp.diff(sp.pi * r**2, r) * 2).subs(r, 5)),
    ("alg-01", "algebra", "solve", "Solve x^2 - 5x + 6 = 0. Give the answer as a set.",
     lambda: sp.FiniteSet(*sp.solve(x**2 - 5 * x + 6, x))),
    ("alg-02", "algebra", "simplify", "Simplify (x^2 - 1)/(x - 1).",
     lambda: sp.cancel((x**2 - 1) / (x - 1))),
]


def build() -> list[dict]:
    rows = []
    for pid, topic, op, text, compute in PROBLEMS:
        gold = sp.simplify(compute())
        rows.append({
            "id": pid, "problem": text,
            "answer_latex": sp.latex(gold, ln_notation=True), "answer_sympy": str(gold),
            "topic": topic, "expected_operation": op, "source": "authored for H2flow benchmarking",
        })
    return rows


def main() -> None:
    rows = build()
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} problems to {OUT}")


if __name__ == "__main__":
    main()
