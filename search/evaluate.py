"""Retrieval evaluation: does the right module come back in the top results?

Run it with Elasticsearch up and the index loaded:
  python -m search.evaluate
  python -m search.evaluate --save eval/baseline.json      # remember this run
  python -m search.evaluate --baseline eval/baseline.json  # show rank changes vs that run
  python -m search.evaluate --fields "text,section^3" --baseline eval/baseline.json  # try a weighting
  python -m search.evaluate --demote 0.3 --baseline eval/baseline.json  # push Exercises/Glossary down

Questions live in eval/questions.json as
  [{"question": "power rule", "expect": ["m53575"]}, ...]
"expect" lists every module that would be an acceptable answer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

DEFAULT_QUESTIONS = "eval/questions.json"
KS = (1, 3, 5)


def first_rank(hits: list[dict], expect: list[str]) -> int | None:
    """1-based position of the first hit from an expected module, or None."""
    wanted = set(expect)
    for position, hit in enumerate(hits, start=1):
        if hit.get("module_id") in wanted:
            return position
    return None


def summarize(ranks: list[int | None]) -> dict:
    """hit@1/3/5 counts and mean reciprocal rank (1 = always first, 0 = never found)."""
    total = len(ranks)
    out = {f"hit@{k}": sum(1 for r in ranks if r is not None and r <= k) for k in KS}
    out["mrr"] = sum(1.0 / r for r in ranks if r) / total if total else 0.0
    out["n"] = total
    return out


def run_eval(questions: list[dict], search, top_k: int) -> list[dict]:
    """search(query, top_k) -> list of hit dicts, as retrieve.get_documents returns."""
    results = []
    for q in questions:
        hits = search(q["question"], top_k)
        results.append(
            {"question": q["question"], "expect": q["expect"], "rank": first_rank(hits, q["expect"]), "hits": hits}
        )
    return results


def describe_search(fields: list[str] | None, match_type: str, demote: float | None = None) -> str:
    base = "text only (match)" if not fields else f"{','.join(fields)} [{match_type}]"
    return base if demote is None else f"{base}, demote exercises/glossary x{demote}"


def _fmt_rank(rank: int | None) -> str:
    return "miss" if rank is None else str(rank)


def _change(now: int | None, was: int | None) -> str:
    def score(r):  # a bigger number is better
        return 0.0 if r is None else 1.0 / r
    if score(now) > score(was):
        return "better"
    if score(now) < score(was):
        return "WORSE"
    return ""


def main() -> None:
    from .index import add_connection_arguments, get_client

    sys.stdout.reconfigure(errors="replace")  # never crash on a character the console can't show
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=DEFAULT_QUESTIONS)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--save", help="write this run's ranks to a JSON file")
    ap.add_argument("--baseline", help="compare against ranks saved by an earlier --save")
    ap.add_argument("--fields", help='comma-separated, e.g. "text,section^3,title^2" (default: text only)')
    ap.add_argument("--match-type", default="most_fields", choices=["most_fields", "best_fields"])
    ap.add_argument("--demote", type=float, help="multiply scores of Exercises/Glossary chunks by this (0 to 1)")
    add_connection_arguments(ap)
    args = ap.parse_args()

    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)
    baseline = {}
    if args.baseline:
        with open(args.baseline, encoding="utf-8") as f:
            baseline = json.load(f)
    baseline_desc = baseline.pop("_search", "unknown")

    from .retrieve import get_documents, parse_fields  # imported here so the scoring helpers need no Elasticsearch

    fields = parse_fields(args.fields)
    desc = describe_search(fields, args.match_type, args.demote)

    try:
        with get_client(args.es_url) as es:
            results = run_eval(
                questions,
                lambda query, k: get_documents(query, top_k=k, index=args.index, es=es,
                                               fields=fields, match_type=args.match_type,
                                               demote=args.demote),
                max(args.top_k, max(KS)),
            )
    except Exception as exc:
        raise SystemExit(f"search failed ({exc.__class__.__name__}). Is Elasticsearch running and the index loaded?")

    print(f"searching: {desc}" + (f"   |   baseline: {baseline_desc}" if baseline else ""))
    header = f"{'rank':>5}  " + (f"{'was':>5}  {'':7}" if baseline else "") + "question"
    print(header)
    for r in results:
        line = f"{_fmt_rank(r['rank']):>5}  "
        if baseline:
            was = baseline.get(r["question"], "n/a")
            was_rank = was if isinstance(was, int) else None
            line += f"{(_fmt_rank(was_rank) if r['question'] in baseline else 'n/a'):>5}  {_change(r['rank'], was_rank):7}"
        print(line + r["question"])
    print()
    for r in results:
        if r["rank"] is None:
            seen = "; ".join(f"{h['module_id']} ({h['section'] or h['title']})" for h in r["hits"][:3])
            print(f"MISS  {r['question']!r}  expected {r['expect']}  got: {seen}")

    s = summarize([r["rank"] for r in results])
    print(f"\nhit@1 {s['hit@1']}/{s['n']}   hit@3 {s['hit@3']}/{s['n']}   hit@5 {s['hit@5']}/{s['n']}   MRR {s['mrr']:.2f}")

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump({"_search": desc, **{r["question"]: r["rank"] for r in results}}, f, indent=2)
        print(f"saved ranks to {args.save}")


if __name__ == "__main__":
    main()
