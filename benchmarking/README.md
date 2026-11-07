# Accuracy benchmarking

This harness answers one question: **does H2flow solve math problems more accurately than the model it is built on?**
It runs the raw model and H2flow (the model plus textbook retrieval and SymPy verification) on the same problems, with the same
settings and the same grader. It reports accuracy with confidence intervals, a per-problem paired comparison, and cost.

The first target is **`gemma3:4b`** (Ollama). Any Ollama model can be passed with `--model`.

## Status

| Arm | What runs | Status |
| --- | --- | --- |
| `base` | Raw model, zero-shot step-by-step prompt, answer in `\boxed{}` | **Ready** |
| `sympy` | `verification.py` intent extraction → SymPy. Falls back to `base` when unsupported, failed or timed out | **Ready** (uses the system code on main) |
| `rag` | `RAG/main.py` `rag()` with real retrieval | **Pending**, see [TODO](#todo) |
| `full` | `rag_with_verification()`, H2flow end to end | **Pending**, see [TODO](#todo) |

| Dataset | Size | Purpose |
| --- | --- | --- |
| `math500` | 500 | General math ([MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)); downloaded on first use into `.cache/` |
| `calculus` | 27 (starter) | H2flow's domain: limits, derivatives, definite integrals, applications ([data/calculus_v1.jsonl](data/calculus_v1.jsonl)) |

## Baseline results

`base` arm, gemma3:4b (Q4_K_M, digest `68ceab7e468a`), default protocol, harness commit `eb0298c`, RTX 3080 10 GB, Ollama 0.40.15.
Full reports: [results/math500-base-gemma3-4b.md](results/math500-base-gemma3-4b.md) and
[results/calculus-base-gemma3-4b.md](results/calculus-base-gemma3-4b.md).

| Dataset | Problems × seeds | Accuracy (mean ± sd) | 95% CI | Per seed | Latency/problem |
| --- | --- | --- | --- | --- | --- |
| MATH-500 | 500 × 3 | **73.7** ± 2.4 | 70.2–77.1 | 73.8 / 76.0 / 71.2 | 8.5 s |
| calculus v1 | 27 × 3 | **100.0** | 100.0–100.0 | 100 / 100 / 100 | 4.5 s |

- **Calibration passes.** 73.7 is within 2 points of Google's published 75.6 (MATH, 0-shot), so the harness and grader behave as expected.
- **MATH-500 breakdown.** Accuracy falls from 94.6 (level 1) to 49.5 (level 5). Weakest subjects: Intermediate Algebra (52.2), Precalculus (55.4) and Geometry (58.5).
- **Failures.** Out of 1500 attempts, 364 are wrong answers and 0 are extraction failures. Another 31 come from repetition loops: 28 hit the
  4096-token limit (`truncated`), and 3 were stopped by Ollama with "token repeat limit reached" (logged as `error`). All 31 are model failures,
  not infrastructure faults.
- **The calculus set is at ceiling.** Every problem was solved on every seed, so there is no room left to show the system improving on it.
  Growing the set with harder problems is now the priority for calculus (see [TODO](#todo)).

**Hand-check (protocol step 4).** 147 attempts reviewed:
- 40 random MATH-500 wrong answers, plus a pattern scan of all 364 for units or text in the extracted answer.
- 40 random MATH-500 correct answers.
- All 27 calculus problems: every distinct extraction against its gold.
- All 31 truncated or error attempts.

One grader disagreement: `test/prealgebra/1865.json` seed 2 boxed `12^{\text{th}} grade` against gold `12`. It was graded wrong but is correct.
That is 0.07 points, too small to change any reported number, so it was left as graded. No false positives were found. Calculus answers matched
the gold in equivalent forms (for example, a product-rule derivative against the factored gold).

## Quick start

Requirements: [Ollama](https://ollama.com) running, with the model pulled.

```bash
ollama pull gemma3:4b
```

```bash
pip install -r benchmarking/requirements.txt
```

Smoke test (2 problems, 1 seed; don't report these numbers):

```bash
python -m benchmarking.run --arm base --dataset calculus --limit 2 --seeds 0
```

Headline baseline run (500 problems × 3 seeds):

```bash
python -m benchmarking.run --arm base --dataset math500
```

```bash
python -m benchmarking.run --arm base --dataset calculus
```

Summarize, compare and save. The first run listed is the baseline for paired comparisons:

```bash
python -m benchmarking.report benchmarking/results/raw/<math500-base-run> --save benchmarking/results/math500-base-gemma3-4b
```

```bash
python -m benchmarking.report benchmarking/results/raw/<calculus-base-run> benchmarking/results/raw/<calculus-sympy-run>
```

An interrupted run continues where it stopped, using its saved settings:

```bash
python -m benchmarking.run --resume benchmarking/results/raw/<run>
```

Run the commands from the repo root. The `sympy` arm also needs the root `requirements.txt`, because it imports `verification.py`.

## Design choices

These settle the questions the first draft of this plan left open.

| Question | Decision | Why |
| --- | --- | --- |
| Which models? | **`gemma3:4b` only** for the first results: Ollama's default tag, which is 4-bit quantized (Q4_K_M as of this writing). Each run records the exact digest and quantization. | It's the only model the system uses. The harness is model-agnostic (`--model`), so a second small model such as Qwen3:4b is just a flag later, but it isn't needed for the base-vs-system question. |
| Skip the base model and use published numbers? | **No.** Run the base arm ourselves. Published numbers serve only as a calibration check: MATH-500 should land within ±10 points of 75.6, and the report flags runs outside that band. | Published scores use different prompts, decoding, graders and full-precision weights. The grader alone can move MATH-500 by tens of points. Nothing is published for our calculus set, and a paired comparison needs both arms on the same problems. |
| Calculus difficulty | **First-year Calculus I/II**, the problems H2flow's users will ask. IntegralBench-style hard integrals are an optional stress test later, not part of the headline. | Matches the textbooks in the index and the operations SymPy handles. |
| Hardware | **One designated machine** for every reported run. The manifest records OS, CPU, GPU and Ollama version. Latency is compared only between runs on the same machine. | Latency and throughput depend on hardware. Accuracy should not, but this rules it out. |
| Decoding | **Temperature 0.2, seeds 0, 1, 2**, `num_ctx` 8192, `num_predict` 4096. Every other sampling parameter stays at the model's Modelfile defaults, which the manifest records. | 0.2 is the system's own generation temperature (`solver_llm`), so base and system differ only in the tools. A temperature above 0 makes seeds meaningful, and 3 seeds is enough on 500 problems. `num_ctx` 8192 leaves room for retrieved chunks and long reasoning without Ollama silently cutting the prompt. |
| Prompt | **`prompts/base_v1.txt`**: the problem, then "reason step by step, final answer in `\boxed{}`". Zero-shot, user message only. | The standard MATH-500 prompt, and `\boxed{}` is what the gold answers and Math-Verify expect. Changing a prompt means adding a new file (`base_v2.txt`), never editing an old one. |
| Grader | **[Math-Verify](https://github.com/huggingface/Math-Verify) 0.9.0** (antlr4 4.13.2) with its default extraction, for every arm and dataset. Gold answers are always LaTeX. The grader never imports `verification.py`. | One grader that is independent of the system under test, so the system can't grade itself. Math-Verify handles equivalent forms (`1/√2 = √2/2`, `0.5 = 1/2`, solution sets, `FINAL_ANSWER:` lines). |
| Grader timeouts | **Disabled.** Raw outputs are saved, so any run can be re-graded (`report --regrade`). | Math-Verify's timeouts rely on Unix signals or subprocesses and break on Windows. |
| Calculus gold answers | **Computed with SymPy** in [data/build_calculus.py](data/build_calculus.py) and checked by hand. Stored as LaTeX with `\ln`. Only problems with one exact answer are included, so there are no indefinite integrals. | Avoids arithmetic slips in hand-written answers. `\ln` because Math-Verify reads a bare `\log` as base 10. "+ C" answers would need a derivative check the grader doesn't do. |
| Leakage | The calculus problems are **written for this benchmark**, not taken from the indexed OpenStax books. If textbook exercises are added later, they are reported separately as "in-corpus" or run against an index without `Exercises` chunks. | The index holds the textbook exercises with their solutions, so retrieval could return the answer itself. |
| Statistics | Per seed: accuracy with a Wilson 95% CI. Per run: mean ± sd over seeds, plus a bootstrap 95% CI over problems. Between arms: a paired bootstrap of the accuracy difference, the fixed/broken counts (majority of seeds), and McNemar's exact test. | Single runs on small sets are noisy, and a paired test on the same problems is much more sensitive than comparing two separate intervals. |
| Failures | Each attempt is graded `correct`, `wrong_answer`, `extraction_failure`, `truncated` or `error`. Only `correct` counts as correct. | An answer the grader can't read, or one cut off by the token limit, is still a failure for a user, but the counts show which kind of failure it was. |
| Cost | Per problem: LLM calls, prompt and completion tokens, latency, and route taken. The report gives the average calls and latency per problem. | The full system makes several calls per question, and accuracy gains should be weighed against that cost. |

## How it works

```text
run.py ──> dataset (datasets.py) ──> arm.solve(problem, seed) ──> grading.grade(gold, final answer)
             │                         │                              │
             │                         └─ Ollama (ollama_client.py)   └─ Math-Verify
             └─ results/raw/<run_id>/manifest.json + attempts.jsonl ──> report.py ──> results/<name>.md/.json
```

| File | Role |
| --- | --- |
| `run.py` | CLI. Loads the dataset, prompt and arm, writes the manifest, runs the seeds in order (each seed finishes before the next starts), grades each attempt, and can resume. |
| `arms/` | One adapter per system under test, all with the same interface: `setup(ctx)` checks dependencies, `solve(problem, seed, ctx)` returns an `Attempt` (response, final answer text, calls, tokens, latency, route, extra). |
| `ollama_client.py` | Standard-library client for Ollama's REST API. Returns token counts, latency and `done_reason`, which is how truncation is detected. |
| `datasets.py` | MATH-500 loader (download and cache, sha256) and a JSONL loader for our own sets (`id`, `problem`, `answer_latex`, plus any metadata). |
| `grading.py` | Math-Verify wrapper and outcome classes. |
| `stats.py` | Wilson CI, bootstrap, paired bootstrap, McNemar test (no dependencies). |
| `report.py` | Summary tables (overall, per subject, level and topic, outcomes, routes, intent accuracy, cost), the calibration check, and paired comparisons. |
| `prompts/` | Versioned prompt files. |
| `data/` | Committed datasets and the calculus builder. |
| `results/` | Saved summaries (committed). `results/raw/` holds raw runs and is gitignored. |

**Run manifest** (`manifest.json`): config (arm, dataset, model, seeds, options, prompt), git SHA, branch and dirty flag,
Ollama version, model digest, quantization and Modelfile parameters, prompt text and sha256, dataset source, size and sha256,
grader and library versions, machine details, start and finish times. Resuming refuses to continue if the prompt, dataset or
model digest has changed.

**Attempt rows** (`attempts.jsonl`, one per problem per seed): problem id, seed, metadata, gold answer, full response,
graded text, outcome, extracted answer, calls, tokens, latency, truncated flag, error, route and extra details (for the `sympy`
arm: the extracted intent and the reason for any fallback).

### The `sympy` arm

This arm tests the system's SymPy path as it is on `main`, without changing it. `verification.extract_math_intent` turns the
problem into JSON. If the operation is supported, `solve_with_sympy` runs in a child process with a 30-second limit, and
its result, rendered as LaTeX, is the answer. Otherwise the problem goes to the `base` arm with the same seed. The report shows
how many problems took each route, and on the calculus set, how often the extracted operation matched `expected_operation`.
Limits inherited from `verification.py`: the intent model is fixed there (`gemma3:4b`, temperature 0), so `--model` and
`--temperature` don't apply to that call, and its token counts aren't exposed.

### Adding a dataset or arm

- **Dataset:** write a JSONL file with `id`, `problem`, `answer_latex` (no `$`), plus any metadata such as `topic` or `level`, which the report
  breaks results down by. Run it by path (`--dataset path/to/set.jsonl`), or register it in `datasets.DATASETS`.
- **Arm:** subclass `arms.Arm`, implement `setup` and `solve`, and add it to `arms.get_arm`. Put the final answer where Math-Verify can find
  it (`\boxed{}` or `FINAL_ANSWER:`), and count every LLM call in `llm_calls`.

## Protocol for reported numbers

1. Use the designated machine, a clean git tree, and Ollama not serving anything else.
2. Run the full dataset with the default settings (no `--limit`, seeds 0 1 2).
3. Check the calibration line in the report for `base` on MATH-500. If it's outside the band, look for bugs before going further:
   truncation counts, extraction failures, prompt.
4. Hand-check about 50 graded attempts (mix of correct and wrong) to confirm the grader. Record any disagreements in the results write-up.
5. Save the report with `--save benchmarking/results/<dataset>-<arm>-<model>` and commit the `.md` and `.json`.
6. Treat differences under about 2 points on MATH-500 (or a non-significant paired test) as noise.

## Background

Published Gemma 3 4B scores (for calibration, not comparison):

| Benchmark | Setting | Score | Source |
| --- | --- | --- | --- |
| MATH | IT, 0-shot | 75.6 | [Gemma 3 model card](https://ai.google.dev/gemma/docs/core/model_card_3) |
| GSM8K | IT, 0-shot | 89.2 | Gemma 3 model card |
| HiddenMath | IT, 0-shot | 43.0 | Gemma 3 model card |
| Calculus I (374 problems, unreleased) | `gemma3:4b` via Ollama, temperature 0.3 | 86.4% | [arXiv 2608.22048](https://arxiv.org/abs/2608.22048) |

- **MATH contains very little calculus.** Its "Precalculus" subject is trigonometry, vectors and complex numbers. So MATH-500 tests
  whether H2flow's tools *hurt* general problems, and the calculus set tests whether they *help* in its domain. Both results get reported.
- [MATH](https://arxiv.org/abs/2103.03874) is the original 12,500-problem benchmark. MATH-500 is OpenAI's 500-problem subset of its test set.
  The [llmdb leaderboard](https://llmdb.com/benchmarks/math) is a collection of reported scores, not a dataset, and doesn't list Gemma 3 4B.
- Further reading: [Math-Verify and the Open LLM Leaderboard](https://huggingface.co/blog/math_verify_leaderboard) (the effect of the grader),
  [A Sober Look at Progress in LM Reasoning](https://arxiv.org/abs/2504.07086) (seed variance, reproducibility),
  [IntegralBench](https://arxiv.org/abs/2507.21130) (hard definite integrals).

## TODO

**Blocked on system code** (benchmark side ready; each item names what the system has to provide):
- [ ] `rag` arm ([arms/rag.py](arms/rag.py)): needs `RAG/main.py` to be importable without running a query, and `rag()` to use
      `search.retrieve.get_documents` in place of `get_documents_dummy`. Check extraction failures on its first run, because the RAG prompt doesn't ask for
      a marked final answer.
- [ ] `full` arm ([arms/full.py](arms/full.py)): needs `rag_with_verification()` to be implemented, and per-question usage (LLM calls,
      retries, tokens) to be exposed so cost can be compared.
- [ ] Optional API mode for `full`, posting to `POST /api/v1/questions` once the API (`dylan/backend-api-starter`) is merged.
- [ ] `sympy` arm: token counts for the intent call, and running it with models other than `gemma3:4b`. Both need `verification.py` to expose them.

**Benchmark side:**
- [x] Run and commit the `base` baseline on MATH-500 and calculus (gemma3:4b, default protocol). See [Baseline results](#baseline-results).
- [x] Hand-check about 50 graded attempts for that baseline (protocol step 4).
- [ ] **Priority:** the base model scores 100% on calculus v1, so the set can't show improvement. Grow it from 27 to about 200 problems, weighted
      toward harder items (multi-step integration, integration by parts and substitution, series, related rates, optimization). Write it as
      `calculus_v2` so v1 results stay comparable. Keep the problems Calculus I/II, written for the benchmark, with `expected_operation`
      labels. Add full intent labels (expression, bounds) to score intent extraction beyond the operation.
- [ ] Add a small "unsupported" set (non-math and conceptual questions) to check the system routes them away from SymPy.
- [ ] Optional arm E: base model, best-of-n majority vote with the same number of LLM calls as `full`, to test whether the system beats just
      spending more compute.
- [ ] Optional: IntegralBench subset as a hard-integral stress test, and a second model (for example `qwen3:4b`) with `--model`.
- [ ] Write up results in `benchmarking/RESULTS.md` once `full` runs.
