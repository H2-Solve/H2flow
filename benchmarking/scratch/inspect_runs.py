import json, sys
from collections import Counter

for d in sys.argv[1:]:
    rows = [json.loads(l) for l in open(d + "/attempts.jsonl", encoding="utf-8")]
    print("=" * 100, "\n", d, len(rows), "attempts", Counter(r["outcome"] for r in rows), Counter(r["route"] for r in rows))
    if "base" in d:
        continue
    for r in rows:
        print("-" * 100)
        print("problem:", r["problem_id"], "| gold:", r["gold"], "| outcome:", r["outcome"], "| route:", r["route"])
        print("extracted:", r["extracted"], "| error:", r["error"], "| truncated:", r["truncated"])
        print("calls:", r["llm_calls"], "tokens:", r["prompt_tokens"], "/", r["completion_tokens"], "latency %.1fs" % r["latency_s"])
        ex = r["extra"]
        for c in ex["calls"]:
            print("   call", c)
        print("retrieved:")
        for doc in ex["retrieved"]:
            print("   ", doc)
        for k in ("intent", "sympy_truth", "solver_calls", "final_answer_latex"):
            if k in ex:
                print(k, ":", ex[k])
        print("final_text tail:", repr((r["final_text"] or "")[-300:]))
        print("response tail:", repr(r["response"][-600:]))
