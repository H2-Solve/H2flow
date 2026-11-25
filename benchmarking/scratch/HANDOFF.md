# Handoff: rag/full benchmarking (written 2026-11-24, temporary, remove before merging)

This file, `benchmarking/scratch/`, `benchmarking/results/raw/` and `benchmarking/.cache/` are on the branch
`benchmarking/rag-full-arms` only to move the work between machines. Before the PR into main:

```powershell
git rm -r --cached benchmarking/results/raw benchmarking/.cache
```

```powershell
git rm -r benchmarking/scratch
```

## Where it stands

| Done | |
| --- | --- |
| `rag` and `full` arms | Run `RAG/main.py` unchanged; verified with a live HTTP-level trace (`scratch/trace_rag.py`). |
| Answer extraction | `benchmarking/extract.py` + `report --grading extracted`. Validated: 0/116 wrong answers turned correct, 92/100 correct kept, calculus matches hand grades. |
| Calculus v1, all three arms | Results below. |
| MATH-500 `base` at temperature 0 | 72.0% (360/500), 24 truncations (repetition loops), grader hand-checked. |

| To do, in order | |
| --- | --- |
| 1 | Set up the new machine (below). **Check the model digest first.** |
| 2 | Re-run MATH-500 `base` at temperature 0 on the new machine (about 70 min), so latency is comparable (see "Machine change"). |
| 3 | MATH-500 `rag`, 1 seed (about 45 min). |
| 4 | MATH-500 `full`, 3 seeds (3 h or more). |
| 5 | `python -m benchmarking.extract` on every run, then `report --grading extracted`. Hand-check a sample of extracted `rag` grades. |
| 6 | Write up `benchmarking/RESULTS.md`, save reports with `--save`, remove the temporary files, and open the PR. |

## Calculus v1 results (27 problems, this machine)

| Arm | Strict | Extracted | Notes |
| --- | --- | --- | --- |
| `base`, temperature 0 | 27/27 | 27/27 | Identical responses across two runs. |
| `rag` | 18/27 (66.7%) | **25/27 (92.6%)** | 17/27 answers unboxed. Real failures: lim-04 (misled by a retrieved chunk), lim-06 (declined after irrelevant retrieval). |
| `full`, 3 seeds | **56/81 (69.1%)** | same (verified answers are boxed) | 8 problems broken, 0 fixed (McNemar p = 0.008). Every failure is in the system, not the harness. |

Calculus v1 is at ceiling for `base`, so it can only show harm. `calculus_v2` (harder, about 200 problems) is still a TODO.

### Why `full` fails (system code, not the harness)

`extract_math_intent` called natively (the system's own ChatOllama) gives the same intents as the harness on 8/8
failing problems.
1. **Wrong intent leads to a wrong SymPy "truth", so a correct model answer is rejected** (15 unverified):
   der-01 (adds a cos term), app-02 (derivative instead of critical points), app-04 (solve instead of max),
   app-05 (`integrate()` inside the expression, so it is integrated twice: 32π), app-06 (drops dr/dt).
   app-05 seed 2 went further: the model said "the correct volume is 8π" and then changed the limits to reach 32π,
   which passed as **verified and wrong**.
2. **Intent format crashes SymPy** (9 errors): lim-04 and lim-06 put the limit point in `upper_bound`
   (`verification.py:91` gets `None`); app-01 puts `integrate(...)` in the expression.
3. **List comparison** (1 error): `answers_are_equivalent` compares `str(list)`, so `{2, 3}` and `[3, 2]` fail against `[2, 3]`.

Suggestions for the app team (application code, out of scope here): fall back to the model's answer when
verification fails; don't let the prompt override a model that disagrees with SymPy; validate intent fields
before SymPy; compare solution sets as sets.

## Setting up the new machine

1. Before cloning, run `git config --global core.autocrlf true` (the old machine's setting). The harness hashes
   dataset files byte for byte, and the calculus manifests were recorded with Windows (CRLF) line endings, so a
   different setting changes `calculus_v1.jsonl`'s sha256 and `--resume` on old calculus runs will refuse.
   Accuracy is unaffected. The MATH-500 cache is committed with `-text`, so its bytes match regardless. Then
   `git clone https://github.com/H2-Solve/H2flow.git` and `git checkout benchmarking/rag-full-arms`.
2. Python 3.13, then `pip install -r requirements.txt -r benchmarking/requirements.txt`.
3. Ollama, then `ollama pull gemma3:4b`. **Check the digest:** `ollama list` must show `68ceab7e468a`.
   If it differs, Ollama has updated the tag. Copy the model from the old machine
   (`%USERPROFILE%\.ollama\models`, 3.3 GB) or re-run every arm, because results across digests aren't comparable,
   and `--resume` refuses a changed digest.
4. Docker Desktop (WSL 2), then `docker compose up -d`.
5. `git clone https://github.com/openstax/osbooks-calculus-bundle ..\osbooks-calculus-bundle`, then
   `git -C ..\osbooks-calculus-bundle checkout 018da4da252bb9c56759e34396efea91852bcf5a` (the version indexed here).
   Then `python -m search.index --bundle ..\osbooks-calculus-bundle --reset`.
   Expect **4287 chunks** (volume 1: 1615, volume 2: 956, volume 3: 1716), 1121 of them Exercises.
6. On Windows, in each PowerShell window: `$env:OLLAMA_HOST = "http://127.0.0.1:11434"; $env:H2FLOW_ES_URL = "http://127.0.0.1:9201"`
   (`localhost` costs about 2 s per call).
7. Smoke test: `python -m benchmarking.run --arm full --dataset calculus --limit 2 --seeds 0`.

## Environment fingerprint (old machine)

| | |
| --- | --- |
| Model | gemma3:4b, Q4_K_M, digest `68ceab7e468a` |
| Ollama | 0.40.18 (the original 0.2 baseline used 0.40.15) |
| Grader | math-verify 0.9.0, sympy 1.14.0, antlr4 4.13.2 |
| MATH-500 file | `benchmarking/.cache/math500_test.jsonl`, sha256 `35dc4108…5a06132` (committed here) |
| Textbook bundle | osbooks-calculus-bundle `018da4d` |
| Machine | Windows 10 19044, Python 3.13, RTX 3080 10 GB, AMD CPU |

## Machine change

The protocol compares latency only between runs on one machine, and accuracy should not depend on hardware.
- Re-run MATH-500 `base` at temperature 0 on the new machine (step 2 above) before the `rag` and `full` runs.
  Compare its answers with `results/raw/math500-base-gemma3-4b-t0` as a reproducibility check. Expect near-identical
  results (temperature-0 Ollama varies slightly between runs; see `scratch/repeat_lim01.py`).
- Keep using the old calculus runs for accuracy, but re-run them on the new machine if you report calculus latency.

## Things to remember

- `rag`/`full` ignore `--temperature`; each model uses the system's own (0 answer, 0.2 solver, 0 intent).
  `rag` needs 1 seed, `full` needs 3.
- Always report strict and extracted grades for `rag`.
- 3 Elasticsearch tests in the `search/` suite (`test_cli.py`, `test_test_infrastructure.py`) fail on clean
  `main` too, now that Elasticsearch is running. They were skipped before. They are unrelated to benchmarking.
- The root `README.md` still says the `rag` and `full` arms are pending; update it in the final PR.
