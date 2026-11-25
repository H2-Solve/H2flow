"""Re-run verification.py's own functions on the intents the full run extracted, offline (no LLM)."""
import json
import os
import sys
import traceback

sys.path.insert(0, os.getcwd())
import verification as v  # noqa: E402

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
first = {}
for r in rows:
    first.setdefault(r["problem_id"], r)

print("--- SymPy errors: where in solve_with_sympy they come from")
for pid in ("lim-04", "lim-06", "app-01"):
    intent = first[pid]["extra"]["intent"]
    try:
        print(pid, "->", v.solve_with_sympy(intent))
    except Exception as exc:
        tb = traceback.extract_tb(exc.__traceback__)
        frame = next(f for f in reversed(tb) if f.filename.endswith("verification.py"))
        print(f"{pid}: {exc.__class__.__name__}: {exc}  (verification.py:{frame.lineno} `{frame.line}`)")
    fixed = dict(intent)
    if intent.get("evaluation_point") is None and intent["operation"] == "limit":
        fixed["evaluation_point"] = intent.get("upper_bound") if intent.get("upper_bound") not in (None, "oo") or pid == "lim-06" else intent.get("lower_bound")
        try:
            print(f"   same intent with the point moved to evaluation_point={fixed['evaluation_point']!r}: {v.solve_with_sympy(fixed)}")
        except Exception as exc:
            print("   still fails:", exc)

print("\n--- Wrong ground truth from the extracted intent (model answers were right)")
for pid in ("der-01", "app-02", "app-04", "app-05", "app-06"):
    r = first[pid]
    print(f"{pid}: intent op={r['extra']['intent']['operation']!r} expr={r['extra']['intent']['expression']!r} "
          f"-> SymPy truth {r['extra']['sympy_truth']}  | gold {r['gold']}")

print("\n--- alg-01: answer comparison")
truth = v.solve_with_sympy(first["alg-01"]["extra"]["intent"])
for r in rows:
    if r["problem_id"] == "alg-01":
        ans = v.extract_final_answer_from_response(r["response"])
        print(f"seed {r['seed']} route {r['route']}: FINAL_ANSWER {ans!r} vs truth {truth!r} -> "
              f"answers_are_equivalent={v.answers_are_equivalent(truth, ans, 'x')}")
for cand in ("[2, 3]", "{2, 3}", "[3, 2]", "2, 3"):
    print(f"   answers_are_equivalent([2, 3], {cand!r}) = {v.answers_are_equivalent([2, 3], cand, 'x')}")

print("\n--- app-05 seed 2: 'verified' but wrong")
r = next(x for x in rows if x["problem_id"] == "app-05" and x["seed"] == 2)
print("final answer graded:", r["final_text"], "| outcome", r["outcome"])
