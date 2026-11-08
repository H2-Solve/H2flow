"""Summarize benchmark runs, and compare runs on the same dataset.

  python -m benchmarking.report benchmarking/results/raw/<run>
  python -m benchmarking.report <baseline_run> <other_run> ...     # paired comparison vs the first run
  python -m benchmarking.report <runs...> --save benchmarking/results/math500-gemma3-4b
  python -m benchmarking.report <runs...> --regrade                 # re-grade raw outputs with the current grader

--save writes <name>.md and <name>.json (commit these; raw runs stay out of git).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .grading import OUTCOMES, grade, grader_info
from .stats import bootstrap_mean_ci, mcnemar_exact, mean_sd, paired_bootstrap_diff, wilson_ci

# Published scores for calibration only (different prompt/decoding/grader/precision; see README).
REFERENCE = {("math500", "gemma3:4b"): ("Gemma 3 4B IT, MATH 0-shot (Google model card)", 0.756)}
CALIBRATION_BAND = 0.10


def load_run(run_dir: Path, regrade: bool = False) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    with open(run_dir / "attempts.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if regrade:
        for r in rows:
            r.update(grade(r["gold"], r["final_text"], truncated=r["truncated"], error=r["error"]))
    return {"dir": str(run_dir), "manifest": manifest, "rows": rows}


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def per_problem_accuracy(rows: list[dict]) -> dict[str, float]:
    """Problem id -> fraction of seeds answered correctly."""
    by = defaultdict(list)
    for r in rows:
        by[r["problem_id"]].append(r["outcome"] == "correct")
    return {pid: sum(v) / len(v) for pid, v in by.items()}


def summarize_run(run: dict) -> dict:
    rows, m = run["rows"], run["manifest"]
    seeds = sorted({r["seed"] for r in rows})
    per_seed = {}
    for s in seeds:
        rs = [r for r in rows if r["seed"] == s]
        c = sum(r["outcome"] == "correct" for r in rs)
        per_seed[s] = {"correct": c, "n": len(rs), "accuracy": c / len(rs), "wilson95": wilson_ci(c, len(rs))}
    acc_mean, acc_sd = mean_sd([v["accuracy"] for v in per_seed.values()])
    pp = per_problem_accuracy(rows)

    groups = {}
    for key in ("subject", "level", "topic"):
        vals = defaultdict(list)
        for r in rows:
            if (r.get("meta") or {}).get(key) is not None:
                vals[str(r["meta"][key])].append(r["outcome"] == "correct")
        if vals:
            groups[key] = {k: {"accuracy": sum(v) / len(v), "n": len(v)} for k, v in sorted(vals.items())}

    def avg(field):
        vals = [r[field] for r in rows if r.get(field) is not None]
        return sum(vals) / len(vals) if vals else None

    intent_rows = [r for r in rows if (r.get("meta") or {}).get("expected_operation") and
                   isinstance((r.get("extra") or {}).get("intent"), dict)]
    intent_acc = None
    if intent_rows:
        hits = sum(r["extra"]["intent"].get("operation") == r["meta"]["expected_operation"] for r in intent_rows)
        intent_acc = {"operation_accuracy": hits / len(intent_rows), "n": len(intent_rows)}

    out = {
        "run_id": m["run_id"], "arm": m["config"]["arm"], "dataset": m["dataset"]["name"],
        "model": m["config"]["model"], "model_digest": m.get("model_digest"),
        "quantization": (m.get("model") or {}).get("quantization_level"),
        "options": m["config"]["options"], "prompt": m["config"]["prompt"],
        "n_problems": len(pp), "seeds": seeds, "per_seed": per_seed,
        "accuracy_mean": acc_mean, "accuracy_sd_over_seeds": acc_sd,
        "bootstrap95_over_problems": bootstrap_mean_ci(list(pp.values())),
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
        "routes": dict(Counter(r.get("route", "model") for r in rows)),
        "groups": groups,
        "cost": {"latency_s": avg("latency_s"), "llm_calls": avg("llm_calls"),
                 "prompt_tokens": avg("prompt_tokens"), "completion_tokens": avg("completion_tokens")},
        "intent": intent_acc,
        "complete": m["dataset"]["n_run"] == m["dataset"]["n_total"] and "finished_at" in m,
    }
    ref = REFERENCE.get((out["dataset"], out["model"]))
    if ref:
        out["reference"] = {"label": ref[0], "accuracy": ref[1],
                            "within_band": abs(acc_mean - ref[1]) <= CALIBRATION_BAND}
    return out


def compare(base: dict, other: dict) -> dict:
    a, b = per_problem_accuracy(base["rows"]), per_problem_accuracy(other["rows"])
    common = sorted(set(a) & set(b))
    diff = paired_bootstrap_diff([a[p] for p in common], [b[p] for p in common])
    # majority-of-seeds correctness for the fixed/broken table
    fixed = sum(a[p] < 0.5 <= b[p] for p in common)
    broken = sum(b[p] < 0.5 <= a[p] for p in common)
    both = sum(a[p] >= 0.5 and b[p] >= 0.5 for p in common)
    neither = len(common) - fixed - broken - both
    return {"baseline": base["manifest"]["run_id"], "other": other["manifest"]["run_id"], "n_common": len(common),
            "diff": diff["diff"], "diff_ci95": diff["ci"], "bootstrap_p": diff["p"],
            "fixed": fixed, "broken": broken, "both_correct": both, "both_wrong": neither,
            "mcnemar_p": mcnemar_exact(fixed, broken)}


def to_markdown(summaries: list[dict], comparisons: list[dict]) -> str:
    L = ["# Benchmark report", ""]
    L += ["| Run | Arm | Dataset | Model | Problems × seeds | Accuracy (mean ± sd) | 95% CI | Calls/problem | Latency/problem |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in summaries:
        lo, hi = s["bootstrap95_over_problems"]
        cost = s["cost"]
        calls = f"{cost['llm_calls']:.2f}" if cost["llm_calls"] is not None else "n/a"
        lat = f"{cost['latency_s']:.1f}s" if cost["latency_s"] is not None else "n/a"
        flag = "" if s["complete"] else " (partial)"
        L.append(f"| `{s['run_id']}`{flag} | {s['arm']} | {s['dataset']} | {s['model']} ({s['quantization']}) | "
                 f"{s['n_problems']} × {len(s['seeds'])} | {_pct(s['accuracy_mean'])} ± {_pct(s['accuracy_sd_over_seeds'])} | "
                 f"{_pct(lo)}–{_pct(hi)} | {calls} | {lat} |")
    L.append("")
    for s in summaries:
        L += [f"## {s['run_id']}", "",
              f"Options: `{json.dumps(s['options'])}`, prompt `{s['prompt']}`, digest `{(s['model_digest'] or 'n/a')[:19]}`", ""]
        L.append("Per seed: " + ", ".join(
            f"seed {k}: {v['correct']}/{v['n']} = {_pct(v['accuracy'])} "
            f"(Wilson {_pct(v['wilson95'][0])}–{_pct(v['wilson95'][1])})" for k, v in s["per_seed"].items()))
        L.append("")
        L.append("Outcomes: " + ", ".join(f"{o} {s['outcomes'].get(o, 0)}" for o in OUTCOMES))
        if set(s["routes"]) != {"model"}:
            L.append("Routes: " + ", ".join(f"{k} {v}" for k, v in sorted(s["routes"].items())))
        if s["intent"]:
            L.append(f"Intent operation accuracy: {_pct(s['intent']['operation_accuracy'])} (n={s['intent']['n']})")
        if s.get("reference"):
            r = s["reference"]
            verdict = "within" if r["within_band"] else "OUTSIDE"
            L.append(f"Calibration: published {r['label']} = {_pct(r['accuracy'])}; this run is {verdict} "
                     f"±{_pct(CALIBRATION_BAND)} points (a sanity check, not a comparison).")
        for key, vals in s["groups"].items():
            L += ["", f"| {key} | accuracy | n |", "| --- | --- | --- |"]
            L += [f"| {k} | {_pct(v['accuracy'])} | {v['n']} |" for k, v in vals.items()]
        L.append("")
    if comparisons:
        L += ["## Paired comparisons (vs first run)", "",
              "| Other run | Problems | Δ accuracy | 95% CI | bootstrap p | fixed | broken | McNemar p |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for c in comparisons:
            L.append(f"| `{c['other']}` | {c['n_common']} | {100 * c['diff']:+.1f} | "
                     f"{100 * c['diff_ci95'][0]:+.1f} to {100 * c['diff_ci95'][1]:+.1f} | {c['bootstrap_p']:.3f} | "
                     f"{c['fixed']} | {c['broken']} | {c['mcnemar_p']:.3f} |")
        L += ["", "fixed/broken use majority-of-seeds correctness per problem."]
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run folders; the first is the baseline for comparisons")
    ap.add_argument("--regrade", action="store_true", help="re-grade raw outputs with the installed grader")
    ap.add_argument("--save", help="path prefix for <prefix>.md and <prefix>.json")
    args = ap.parse_args(argv)

    runs = [load_run(Path(r), args.regrade) for r in args.runs]
    summaries = [summarize_run(r) for r in runs]
    comparisons = [compare(runs[0], r) for r in runs[1:]
                   if r["manifest"]["dataset"]["name"] == runs[0]["manifest"]["dataset"]["name"]]
    md = to_markdown(summaries, comparisons)
    print(md)
    if args.save:
        prefix = Path(args.save)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        prefix.with_suffix(".md").write_text(md, encoding="utf-8")
        payload = {"summaries": summaries, "comparisons": comparisons,
                   "grader": grader_info() if args.regrade else runs[0]["manifest"].get("grader")}
        prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"saved {prefix.with_suffix('.md')} and {prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
