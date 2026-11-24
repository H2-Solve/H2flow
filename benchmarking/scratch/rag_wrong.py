import json, re, sys

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
boxed = sum("\\boxed" in r["response"] for r in rows)
print(f"responses with \\boxed: {boxed}/{len(rows)}")
print("correct ones boxed:", sum("\\boxed" in r["response"] for r in rows if r["outcome"] == "correct"),
      "| wrong ones boxed:", sum("\\boxed" in r["response"] for r in rows if r["outcome"] != "correct"))
print("completion tokens:", [r["completion_tokens"] for r in rows])
for r in rows:
    if r["outcome"] == "correct":
        continue
    print("=" * 110)
    print(r["problem_id"], "| gold:", r["gold"], "| extracted:", r["extracted"], "| boxed:", "\\boxed" in r["response"])
    print("retrieved:", [(d["section"][:40], d["title"][:30]) for d in r["extra"]["retrieved"]])
    print("RESPONSE TAIL:\n", r["response"][-700:])
