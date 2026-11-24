"""Faithfulness check for benchmarking/extract.py on answers whose true content we know.

Take base-arm MATH-500 attempts (which box their answers), unwrap every \\boxed{X} to plain X,
run the extractor on that text, and compare with what the strict grader read from the boxed
original. A faithful extractor:
  - on originally WRONG answers: copies the wrong answer (grade stays wrong, never "fixed")
  - on originally CORRECT answers: copies the right answer (grade stays correct)
Usage: python validate_extractor.py <base run dir> [n_correct_sample]
"""
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.getcwd())


def unbox(text: str) -> str:
    out, i = [], 0
    while True:
        j = text.find("\\boxed{", i)
        if j < 0:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:j])
        k, depth = j + len("\\boxed{"), 1
        start = k
        while k < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            k += 1
        out.append(text[start:k - 1])
        i = k


def main():
    from benchmarking.extract import OPTIONS, latexify, load_extract_prompt, _NONE
    from benchmarking.grading import grade
    from benchmarking.ollama_client import OllamaClient

    run_dir, n_correct = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 100
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "attempts.jsonl"), encoding="utf-8")]
    wrong = [r for r in rows if r["outcome"] == "wrong_answer" and "\\boxed" in r["response"]]
    correct = [r for r in rows if r["outcome"] == "correct" and "\\boxed" in r["response"]]
    random.seed(0)
    sample = [("wrong", r) for r in wrong] + [("correct", r) for r in random.sample(correct, min(n_correct, len(correct)))]
    prompt, _ = load_extract_prompt()
    client = OllamaClient()
    tally, problems = Counter(), []
    for i, (kind, r) in enumerate(sample, 1):
        plain = unbox(r["response"])
        assert "\\boxed" not in plain
        reply = client.chat("gemma3:4b", [{"role": "user", "content": prompt.replace("{response}", plain)}],
                            OPTIONS).content.strip()
        if _NONE.search(reply):
            new = "extraction_failure"
        else:
            new = grade(r["gold"], latexify(reply))["outcome"]
        # did the extractor copy the model's own (boxed) answer?
        same_as_model = (not _NONE.search(reply)) and grade(f"{r['extracted']}", latexify(reply))["outcome"] == "correct" \
            if r["extracted"] else None
        tally[(kind, new)] += 1
        tally[(kind, "copied_model_answer" if same_as_model else "differs_from_model_answer")] += 1
        if (kind == "wrong" and new == "correct") or (kind == "correct" and new != "correct"):
            problems.append((kind, r["problem_id"], r["gold"], r["extracted"], reply))
        if i % 25 == 0:
            print(f"  {i}/{len(sample)}", flush=True)
    print("\nRESULTS")
    for kind in ("wrong", "correct"):
        n = sum(1 for k, _ in sample if k == kind)
        print(f"{kind} originals (n={n}):", {k[1]: v for k, v in sorted(tally.items()) if k[0] == kind})
    print("\nMismatches (wrong->correct means the extractor fixed an answer; correct->other means it lost one):")
    for p in problems:
        print("  ", p)


if __name__ == "__main__":
    main()
