"""Grading aid: copy each unmarked final answer into \\boxed{} so Math-Verify can read it.

The base prompt asks for \\boxed{}, but the system's RAG prompt does not, so most rag answers
(and full-system answers on the unsupported route) are plain text. Math-Verify then grades the
last expression in the text, which is often not the answer ("2x/(x^2+1)" read as "1)."). This
step asks a model to copy the response's own final answer, without solving anything, and the
report can grade that copy instead (`report --grading extracted`).

  python -m benchmarking.extract benchmarking/results/raw/<run> [<run> ...]

Which attempts: only those the grader reads from plain text, i.e. no error, not truncated, and
no \\boxed{} in the graded text. Every other attempt keeps its strict grade.

What the extractor sees: the response only. Never the gold answer, and not the question either:
with the question in view, a model asked to copy an answer will sometimes solve the problem
instead (it "found" e^2 in a response that declined to answer).

Output, next to the run's attempts.jsonl (raw attempts are never changed):
  extraction.json    extractor model, digest, prompt name and sha256, options
  extracted.jsonl    one row per extracted attempt: reply, tokens, latency, and the sha256 of the
                     response it read (a stale row is redone if the response ever changes)
Extractor calls are a grading cost, not a system cost: they are not added to the run's numbers.
Re-running continues where it stopped.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

from .grading import grade
from .ollama_client import OllamaClient, OllamaError

HERE = Path(__file__).resolve().parent
PROMPT_NAME = "extract_v1"
DEFAULT_MODEL = "gemma3:4b"
OPTIONS = {"temperature": 0.0, "seed": 0, "num_ctx": 8192, "num_predict": 256}
EXTRACTED_FILE = "extracted.jsonl"
META_FILE = "extraction.json"
_NONE = re.compile(r"\\boxed\{\s*NONE\s*\}", re.IGNORECASE)
# Function names written without a backslash ("sin(x)") parse as products of letters (s*i*n*x).
_BARE_FUNCTION = re.compile(r"(?<![\\A-Za-z])(arcsin|arccos|arctan|sinh|cosh|tanh|sin|cos|tan|sec|csc|cot|ln|log)"
                            r"(?![A-Za-z])")


def latexify(reply: str) -> str:
    """Add the missing backslash to bare function names: '3x^2 sin(x)' -> '3x^2 \\sin(x)'."""
    return _BARE_FUNCTION.sub(lambda m: "\\" + m.group(1), reply)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_extract_prompt(name: str = PROMPT_NAME) -> tuple[str, str]:
    text = (HERE / "prompts" / f"{name}.txt").read_text(encoding="utf-8")
    return text, _sha(text)


def needs_extraction(row: dict) -> bool:
    """True when the strict grade comes from plain text (no \\boxed{} in what the grader reads)."""
    if row.get("error") or row.get("truncated"):
        return False
    text = row.get("final_text") or row.get("response") or ""
    return bool(text.strip()) and "\\boxed" not in text


def load_extractions(run_dir: Path) -> tuple[dict, dict] | None:
    """-> (meta, {(problem_id, seed): row}) or None when the run has not been extracted."""
    meta_path, rows_path = run_dir / META_FILE, run_dir / EXTRACTED_FILE
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rows = {}
    if rows_path.exists():
        with open(rows_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[(r["problem_id"], r["seed"])] = r
    return meta, rows


def extracted_grade(row: dict, records: dict | None) -> dict:
    """Grade one attempt with extraction. -> {"outcome", "extracted", "source"}.

    source: "strict" (kept its strict grade), "extractor" (graded on the copied answer), or
    "missing" (needs extraction but none recorded for this response; strict grade kept).
    """
    strict = {"outcome": row["outcome"], "extracted": row.get("extracted")}
    if not needs_extraction(row):
        return {**strict, "source": "strict"}
    rec = (records or {}).get((row["problem_id"], row["seed"]))
    if rec is None or rec.get("response_sha256") != _sha(row.get("response") or ""):
        return {**strict, "source": "missing"}
    if _NONE.search(rec["reply"]):
        return {"outcome": "extraction_failure", "extracted": None, "source": "extractor"}
    g = grade(row["gold"], latexify(rec["reply"]))
    return {**g, "source": "extractor"}


def extract_run(run_dir: Path, client: OllamaClient, model: str, redo: bool = False) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    with open(run_dir / "attempts.jsonl", encoding="utf-8") as f:
        attempts = [json.loads(line) for line in f if line.strip()]
    prompt, prompt_sha = load_extract_prompt()

    meta_path, rows_path = run_dir / META_FILE, run_dir / EXTRACTED_FILE
    existing = load_extractions(run_dir)
    if existing and not redo:
        old = existing[0]
        if old["prompt_sha256"] != prompt_sha or old["model"] != model:
            raise SystemExit(f"{run_dir}: already extracted with another prompt or model; pass --redo to start over")
    if redo or not existing:
        rows_path.unlink(missing_ok=True)
        existing = None
    meta = {
        "model": model, "model_digest": client.model_info(model).get("digest"),
        "prompt": PROMPT_NAME, "prompt_sha256": prompt_sha, "prompt_text": prompt, "options": OPTIONS,
        "ollama": {"host": client.host, "version": client.version()},
        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    done = existing[1] if existing else {}
    todo = [r for r in attempts if needs_extraction(r)
            and done.get((r["problem_id"], r["seed"]), {}).get("response_sha256") != _sha(r["response"])]
    n_needed = sum(needs_extraction(r) for r in attempts)
    print(f"{run_dir.name}: {n_needed} of {len(attempts)} attempts have no \\boxed{{}} answer; "
          f"{n_needed - len(todo)} already extracted, {len(todo)} to go")
    with open(rows_path, "a", encoding="utf-8") as out:
        for i, r in enumerate(todo, 1):
            text = prompt.replace("{response}", r["response"])
            res = client.chat(model, [{"role": "user", "content": text}], OPTIONS)
            rec = {"problem_id": r["problem_id"], "seed": r["seed"], "response_sha256": _sha(r["response"]),
                   "reply": res.content.strip(), "prompt_tokens": res.prompt_tokens,
                   "completion_tokens": res.completion_tokens, "latency_s": round(res.latency_s, 3),
                   "done_reason": res.done_reason}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            done[(r["problem_id"], r["seed"])] = rec
            g = extracted_grade(r, done)
            print(f"[{i}/{len(todo)}] seed {r['seed']} {r['problem_id']}: {r['outcome']} -> {g['outcome']}  "
                  f"{rec['reply'][:60]!r}")
    return {"needed": n_needed, "extracted": len(todo)}


def main(argv=None) -> None:
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run folders under benchmarking/results/raw/")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="extractor model (default: %(default)s)")
    ap.add_argument("--ollama-host", help="default: OLLAMA_HOST or http://localhost:11434")
    ap.add_argument("--redo", action="store_true", help="discard earlier extractions for these runs")
    args = ap.parse_args(argv)

    client = OllamaClient(args.ollama_host)
    try:
        client.version()
    except OllamaError as exc:
        raise SystemExit(f"{exc}\nStart Ollama first.")
    for run in args.runs:
        extract_run(Path(run), client, args.model, args.redo)
    print("\ndone. grade with:  python -m benchmarking.report <runs...> --grading extracted")


if __name__ == "__main__":
    main()
