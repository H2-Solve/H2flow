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
| `rag` | `RAG/main.py` `rag()` with real Elasticsearch retrieval | **Ready** (needs Elasticsearch and the index), see [The `rag` and `full` arms](#the-rag-and-full-arms) |
| `full` | `rag_with_verification()`, H2flow end to end | **Ready** (needs Elasticsearch and the index), see [The `rag` and `full` arms](#the-rag-and-full-arms) |

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

RAG and full-system runs (Elasticsearch up via `docker compose up -d`, textbooks indexed; see the root README).
Each model call keeps the system's own temperature, so `--temperature` does not apply to these arms. Compare them
against a base run at temperature 0, the temperature `rag()` uses:

```bash
python -m benchmarking.run --arm base --dataset math500 --temperature 0 --seeds 0 --run-name math500-base-gemma3-4b-t0
```

```bash
python -m benchmarking.run --arm rag --dataset math500 --seeds 0
```

```bash
python -m benchmarking.run --arm full --dataset math500
```

Most `rag` answers aren't in `oxed{}`, so extract them first, then report both ways
(see [Answer extraction](#answer-extraction)):

```bash
python -m benchmarking.extract benchmarking/results/raw/math500-base-gemma3-4b-t0 benchmarking/results/raw/<math500-rag-run> benchmarking/results/raw/<math500-full-run>
```

```bash
python -m benchmarking.report benchmarking/results/raw/math500-base-gemma3-4b-t0 benchmarking/results/raw/<math500-rag-run> benchmarking/results/raw/<math500-full-run> --grading extracted
```

Without `--grading extracted` the report uses strict grades, and each run still shows a line comparing the strict and
extracted scores.

On Windows, set `OLLAMA_HOST=http://127.0.0.1:11434` and `H2FLOW_ES_URL=http://127.0.0.1:9201` first. Calls to
`localhost` try IPv6 first and each one waits about 2 s, which inflates latency in proportion to the number of calls.

`rag` is deterministic at temperature 0, so one seed is enough. `full` retries at temperature 0.2, so keep 3 seeds.
Paired comparisons work across runs with different seed counts (they use each problem's fraction of correct seeds).

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

### The `rag` and `full` arms

Both arms run the system code on `main` unchanged (`RAG/main.py`, `verification.py`). For each problem they swap the
system's model objects (`llm`, `solver_llm`, `extractor_llm`) for stand-ins that call Ollama through the harness
client, then restore them ([arms/_system.py](arms/_system.py)). The system's prompts, retrieval, SymPy check and retry loop
are untouched.

| Setting | Value | Why |
| --- | --- | --- |
| Temperature | The system's own per model: `rag()` answer 0, verification solver 0.2, intent extractor 0. Recorded in the manifest. | Measures the system as configured. A base run at temperature 0 matches `rag()` exactly. |
| `num_ctx`, `num_predict` | Harness values (8192, 4096) | `num_ctx` 8192 keeps five chunks from being cut silently. `num_predict` stops repetition loops. |
| Seed | Call *n* of an attempt uses `seed + 1000·n` | The first call matches the base arm's seed. Retries get new seeds, as they do when deployed. With one fixed seed, retries 2 and 3 would repeat retry 1 word for word. |
| Model | `--model` for every call (default `gemma3:4b`, what the system hard-codes) | Prints a warning if they differ. |
| Retrieval | `top_k` 5, index `H2FLOW_ES_INDEX` (default `textbook-chunks`) at `H2FLOW_ES_URL` | `setup` refuses to run if Elasticsearch or the index is missing. The manifest records the index's chunk and `Exercises` counts. |
| SymPy (`full`) | The system's `solve_with_sympy`, in a child process with a 30 s limit | An in-process runaway integral would hang the run. |

Each attempt records its calls (role, seed, tokens, latency, stop reason) and the retrieved chunks, and for `full` the intent,
the SymPy result and the number of solver calls.

**Grading.**
- `rag`: the prompt asks for no marked answer, so Math-Verify reads the whole response. Watch the extraction-failure count.
- `full`, verified answers: the system writes `FINAL_ANSWER:` in SymPy syntax (`x**2*exp(x)`). The arm converts it to
  `oxed{}` LaTeX with SymPy's own parser, not `verification.py`, so the grader stays independent.
- `full`, unsupported questions: graded on the plain RAG response.

**`full` routes**, shown in the report:

| Route | Meaning | Outcome |
| --- | --- | --- |
| `verified` | The answer matched SymPy after up to 3 tries | Graded |
| `unsupported` | SymPy had no answer; plain RAG answer | Graded |
| `unverified` | Every retry failed; the system returns no answer (the UI shows a 422) | `error` (last response kept) |
| `sympy_timeout` | SymPy took longer than 30 s | `error` |
| `<phase>_error` | An exception in retrieval, intent, sympy, answer or solver | `error` |

The verified prompt gives the model the SymPy result, so accuracy on the `verified` route mostly measures intent
extraction plus SymPy, like the `sympy` arm. Say so when you report `full`.

### Answer extraction

The base prompt asks for `oxed{}`, but the system's RAG prompt doesn't. Without a box, Math-Verify grades the last
expression in the text, which is often not the answer: `2x/(x^2+1)` was read as `1).`, and `8π` as `8`. On calculus v1,
17 of 27 `rag` answers were unboxed, and strict grading marked 7 correct answers wrong (66.7% strict, 92.6% by hand).

[extract.py](extract.py) fixes this for grading only. For each attempt with no `oxed{}` in its graded text (and no
error or truncation), it asks `gemma3:4b` (temperature 0, prompt [prompts/extract_v1.txt](prompts/extract_v1.txt)) to copy
the final answer the response states into `oxed{}`, or `oxed{NONE}` if it states none.
`report --grading extracted` grades those attempts on the copy. Every other attempt keeps its strict grade.

- **The extractor sees only the response:** not the gold answer, and not the question. In an earlier draft that showed the question,
  the extractor solved a problem the response had declined to answer.
- **The raw attempts are never changed.** Extractions go to `extraction.json` and `extracted.jsonl` in the run folder,
  each tied to the sha256 of the response it read. Re-running continues; `--redo` starts over.
- **Extractor calls are not system cost** and are not added to calls, tokens or latency.
- **Function names:** names written without a backslash (`sin(x)`) get one before grading, since Math-Verify would read
  `sin` as s·i·n.

**Validation** (prompt `extract_v1`):

| Check | Result |
| --- | --- |
| Calculus v1 `rag` run, 17 unboxed answers, against hand grades | 17/17 agree; the run scores 25/27 (92.6%), the same as by hand. The response that declined to answer became `NONE`. |
| Base MATH-500 answers with `oxed{}` unwrapped to plain text: 116 wrong originals | **0 turned correct.** The extractor never fixed a wrong answer. |
| Same, 100 random correct originals | 92 kept. 8 lost: an intermediate value was picked, or a text answer (`Navin`, `(E)`) was dropped. |

So the extracted score can **miss** correct answers but did not **invent** any: treat it as a slightly conservative
estimate. Report both numbers, and look at the report's `(+fixed / −lost)` counts: a large "lost" count means the
extractor is dropping answers that strict grading happened to read correctly. Changing the prompt means
adding `extract_v2.txt` and repeating this validation.

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

**Runs:**
- [ ] Re-run `base` at temperature 0 (1 seed) on MATH-500 and calculus, as the comparison point for `rag` and `full`. Keep the
      temperature 0.2 baseline above as its own result.
- [ ] Run `rag` (1 seed) and `full` (3 seeds) on both sets, then `extract` every run and report with `--grading extracted`.
      Check the routes for `full`.
- [x] Calculus v1, `base` at temperature 0 (27/27) and `rag` (strict 66.7, extracted 92.6; the 2 real failures are lim-04,
      misled by a retrieved chunk, and lim-06, which declined to answer after irrelevant retrieval).
- [ ] Hand-check a sample of extracted `rag` grades on MATH-500. The validation above found no false positives, but
      text answers (names, multiple-choice letters) are a known gap.

**Blocked on system code** (benchmark side ready; each item names what the system has to provide):
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
