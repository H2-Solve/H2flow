import json, sys
from collections import defaultdict

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
by = defaultdict(list)
for r in rows:
    by[r["problem_id"]].append(r)
print("per problem (seed: route/outcome):")
for pid, rs in by.items():
    if any(r["outcome"] != "correct" for r in rs):
        print(f"  {pid:8} gold={rs[0]['gold'][:40]!r:44}", [(r["seed"], r["route"], r["outcome"]) for r in rs])
print()
seen = set()
for r in rows:
    if r["outcome"] == "correct" or (r["problem_id"], r["route"]) in seen:
        continue
    seen.add((r["problem_id"], r["route"]))
    ex = r["extra"]
    print("=" * 110)
    print(r["problem_id"], "seed", r["seed"], "route", r["route"], "| gold", r["gold"])
    print("error:", (r["error"] or "")[:400])
    print("intent:", ex.get("intent"))
    print("sympy_truth:", ex.get("sympy_truth"), "| solver_calls:", ex.get("solver_calls"))
    print("calls:", [(c["role"], c["seed"], c["completion_tokens"], c["done_reason"]) for c in ex["calls"]])
    print("last response tail:", repr((r["response"] or "")[-500:]))
