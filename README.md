# H2flow

H2flow answers calculus questions with step-by-step explanations based on OpenStax textbooks. The answers are checked with SymPy.
It finds the relevant textbook passages with Elasticsearch, has a local LLM (Ollama, `gemma3:4b`) write the
explanation from those passages, and uses SymPy to compute the correct answer. If the LLM's final answer does not
match, it tries again.

> Parts that are not on `main` yet are marked *(not on main: `branch-name`)*.

## How it works

```text
Frontend → FastAPI → answer service (RAG + verification)
                         ├─ 1. Intent extraction   LLM → JSON {operation, expression, variable, ...}
                         ├─ 2. Ground truth        SymPy solves the extracted problem
                         ├─ 3. Retrieval           Elasticsearch → textbook chunks + citation metadata
                         ├─ 4. Generation          LLM writes steps from the context, told the verified answer
                         └─ 5. Verification        FINAL_ANSWER checked against SymPy; on mismatch, retry with feedback
         ← answer + steps + citations
```

For questions with no clear formula (operation `unsupported`), the pipeline skips steps 2 and 5 and returns a
plain retrieval-based answer.

| Area | Location | Status |
| --- | --- | --- |
| Textbook ingest, indexing, retrieval, evaluation | `search/`, `eval/` | On main |
| SymPy verification | `verification.py`, `intent_extraction_prompt.txt` | On main (basic operations) |
| RAG prompting and generation | `RAG/main.py` | Elasticsearch retrieval and SymPy-verified retries |
| HTTP API | `backend/` | *(not on main: `dylan/backend-api-starter`)* |
| Frontend UI | — | *(planned; the `Frontend` branch has no UI code yet)* |
| Accuracy benchmarking (raw model vs. system) | `benchmarking/` | Base and SymPy arms ready; RAG and full-system arms pending |

## Components

### Textbook search (`search/`)

Turns the OpenStax CNXML bundles (for example
[osbooks-calculus-bundle](https://github.com/openstax/osbooks-calculus-bundle)) into searchable chunks.

- **Parsing** (`parse_osbooks.py`): converts MathML to linear text such as `2^(t/4)`, `log_3(81)`, `sqrt(x)`, `f′(x)` and
  `(1/2)ln(x)`. Each example or exercise stays in one block, with its problem and solution together. Lists, notes,
  definitions, figure captions and the glossary are kept. Each chunk gets a section path (`The Basic Rules > The Constant Rule`)
  and lists only the equations it contains. Empty cross-references become `[Figure]`, `[Example]` and so on.
- **Chunking** (`chunk.py`): neighbouring blocks in the same section are merged up to about 900 characters. Blocks over
  1500 characters are split into windows that overlap by 100 characters.
- **Bundles** (`bundle.py`): reads `collections/*.collection.xml` so each chunk gets the right book and chapter.
  A module shared by several books is indexed once.
- **Schema** (`schemas.py`): `Chunk` has the fields `book_id`, `module_id`, `title`, `section`, `chapter`, `text`, `equations` and
  `source`, plus an `excerpt` for citations.
- **Indexing** (`index.py`): index `textbook-chunks` with a `math_text` analyzer that indexes and searches math the way
  people write it: `x^2` → "x squared", `x^3` → "x cubed", `sqrt(` → "square root", `′` → "prime". The text stored
  for the LLM is unchanged. Chunk IDs are stable (`<module_id>:<n>`), so reloading replaces chunks instead of
  adding duplicates.
- **Retrieval** (`retrieve.py`): `get_documents(query, top_k=5, ...)` returns
  `[{text, book_id, title, module_id, section, chapter, excerpt, score}]`. By default it runs a plain text match. Options:
  weighted multi-field search (`fields`, `match_type`) and demoting Exercises/Glossary chunks (`demote`).
- **Evaluation** (`evaluate.py`, `eval/`): reports hit@1/3/5 and MRR for 27 tuning questions (`eval/questions.json`)
  and 9 holdout questions (`eval/holdout.json`). It can save a run and compare later runs against it. Results are in
  [`eval/EXPERIMENTS.md`](eval/EXPERIMENTS.md). `eval/benchmark_pr2.py` measures parse, index and query speed on a throwaway index.

### Verification (`verification.py`)

- `extract_math_intent(question)`: the LLM in JSON mode, using `intent_extraction_prompt.txt`, returns
  `{operation, expression, variable, order, lower_bound, upper_bound, evaluation_point, limit_direction}`.
  The expression is in SymPy syntax. Operations: `derivative`, `integral`, `limit`, `solve`, `simplify`, `evaluate`, `unsupported`.
- `solve_with_sympy(intent)`: computes the correct answer. Handles derivatives of any order, optionally at a point;
  definite and indefinite integrals; one-sided and two-sided limits; solving equations; evaluating; simplifying.
- `extract_final_answer_from_response(text)`: reads the `FINAL_ANSWER:` line from the LLM output and removes `+ C`.
- `answers_are_equivalent(truth, answer)`: checks the two answers are equal symbolically (`simplify(truth - answer) == 0`).

### RAG (`RAG/main.py`)

- `rag(query)`: retrieve → `format_context` → prompt → `gemma3:4b` (temperature 0).
- `verification_prompt` + `solver_llm` (temperature 0.2, so retries can differ): give the model the verified answer
  and ask it to end with `FINAL_ANSWER: <sympy expression>`. Any retry feedback is added to this prompt.
- `rag_with_verification(question, max_retries=3)`: the full pipeline shown above, and the function the API will call.
  It retrieves from Elasticsearch, includes source metadata in the context, and retries generation until the
  `FINAL_ANSWER` is equivalent to the SymPy result.

### HTTP API (`backend/`) *(not on main: `dylan/backend-api-starter`)*

A FastAPI app. The answer provider is chosen in `get_answer_service()`, which is where the RAG service plugs in.

- `GET /health`: shows only that the API process is running.
- `POST /api/v1/questions`

  ```json
  {"question": "Find the derivative of x^3 + 2x", "top_k": 5}
  ```

  `question`: 1–4000 characters, with whitespace trimmed. `top_k`: 1–10, default 5. The response:

  ```json
  {
    "status": "answered | no_sources | unsupported_demo_question",
    "mode": "live | demo",
    "answer": "...",
    "steps": ["..."],
    "citations": [{"book_id": "...", "title": "...", "page": 1, "excerpt": "..."}]
  }
  ```

  Invalid input returns HTTP 422. If the answer service raises `ServiceUnavailable`, the API returns HTTP 503.
  The branch has a `DemoAnswerService` that only answers "Solve 2x + 3 = 7", plus API tests.

### Frontend (`ui/`)

The standalone HTML page calls `ui/connector.py` for both the base-model result and
the Elasticsearch-backed verified RAG result. The RAG card displays retrieved textbook
source metadata and excerpts.

## Setup

Requirements: Python 3.10+, Docker (for Elasticsearch), and [Ollama](https://ollama.com) with the model pulled:

```bash
ollama pull gemma3:4b
```

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

```bash
pip install -r requirements.txt -r search/requirements.txt sympy
```

To run the API, also install `fastapi`, `uvicorn` and `httpx` *(not on main: `dylan/backend-api-starter`, `requirements-dev.txt`)*.

### Elasticsearch

```bash
docker compose up -d
```

`compose.yaml` runs Elasticsearch 8.15 as a single node with security off, on **`127.0.0.1:9201`**. The code defaults to
`http://localhost:9200`, so set the URL:

```bash
export H2FLOW_ES_URL=http://localhost:9201
```

## Usage

Load a textbook. The bundle is a clone of an OpenStax `osbooks-*-bundle`:

```bash
git clone https://github.com/openstax/osbooks-calculus-bundle
```

```bash
python -m search.index --bundle osbooks-calculus-bundle --list-books
```

```bash
python -m search.index --bundle osbooks-calculus-bundle --book calculus-volume-1 --reset
```

Leave out `--book` to index every book in the bundle. `--outline` prints chapter > module titles, and `--fixtures` indexes only the two test modules.

Search and evaluate:

```bash
python -m search.retrieve "power rule for derivatives" --top-k 3
```

```bash
python -m search.evaluate --save eval/baseline.json
```

```bash
python -m search.evaluate --fields "text^3,section" --baseline eval/baseline.json
```

Verification demo. It needs Ollama running and must be started from the repo root:

```bash
python verification.py
```

RAG demo:

```bash
python RAG/main.py
```

API *(not on main: `dylan/backend-api-starter`)*. Interactive docs are at http://127.0.0.1:8000/docs:

```bash
python -m uvicorn backend.main:app --reload
```

## Tests

```bash
python -m pytest -q
```

Tests that need Elasticsearch are skipped unless a test instance is configured. They create and delete their own
throwaway indexes:

```bash
H2FLOW_TEST_ES_URL=http://localhost:9201 python -m pytest -q
```

The tests cover `search/` and `benchmarking/` (the benchmark tests need `benchmarking/requirements.txt` and use a fake
Ollama server). API tests are in `tests/test_api.py` *(not on main: `dylan/backend-api-starter`)*.

## Benchmarking

`benchmarking/` compares the raw model with H2flow on math problem sets, using the same problems, decoding settings and
grader for both. It reports accuracy with confidence intervals, a paired per-problem comparison, and cost
(LLM calls and latency).

- **Arms:** `base` (raw `gemma3:4b`) and `sympy` (the intent extraction → SymPy path in `verification.py`) are ready.
  `rag` and `full` (`rag_with_verification`) are adapters waiting on the integration items listed under Outstanding issues.
- **Datasets:** MATH-500 (general math, downloaded on first use) and a starter calculus set written for this benchmark, with
  SymPy-computed answers.
- **Protocol:** 3 seeds at temperature 0.2. Answers are graded with Math-Verify, which is independent of the system's own
  verification code. Each run saves a manifest recording the model digest, prompt, dataset hash and machine.

Ollama must be running. Then:

```bash
pip install -r benchmarking/requirements.txt
```

```bash
python -m benchmarking.run --arm base --dataset math500
```

```bash
python -m benchmarking.report benchmarking/results/raw/<run>
```

The design choices, full protocol, output formats and benchmark TODOs are in [benchmarking/README.md](benchmarking/README.md).

## Outstanding issues / not yet implemented

**Integration**
- The RAG pipeline now calls `search.retrieve.get_documents`, includes textbook source metadata in its prompt, and is
  safe to import without running a model query.
- `rag_with_verification` now wires intent extraction, SymPy solving, generation, `FINAL_ANSWER` parsing, and retries.
- The API is not merged *(`dylan/backend-api-starter`, branched from the initial commit, so it needs a rebase)*. It only has
  the demo answer service; the live RAG adapter, timeouts, CORS origins and deployment config are still to do.
- Citation format mismatch: the API's `Citation.page` has no source, because chunks have `module_id`/`section`/`chapter`
  and no page numbers. The API contract needs to change.
- No frontend yet.

**Verification**
- For a limit, an `evaluation_point` that is present but `null` is parsed as the string `"None"`, instead of falling back to `0`.
- `solve` answers are compared as strings of the solution list, so a different order or format is marked wrong.
- `simplify(a - b) == 0` can wrongly report hard expressions as different. Spot-checking a few numeric values would be a useful fallback.
- Removing `+ C` with a regex can damage answers like `x + C*x`.
- `parse_expr` evaluates LLM output as code, so it needs to be sandboxed or restricted before the API is public.
- The intent prompt file is read relative to the current directory.
- There are no tests for `verification.py` or `RAG/`.

**Retrieval**
- On the holdout questions the right section comes first only 4 times out of 9 (MRR 0.51). The remaining misses depend on meaning
  rather than keywords ("dy/dx with x and y mixed" → implicit differentiation). The next step is hybrid search, combining
  embeddings with the current keyword search.
- Tables in the textbook are skipped during parsing.

**Project setup**
- `sympy` is missing from `requirements.txt`. The dependencies are split between `requirements.txt` and `search/requirements.txt`.
- The default Elasticsearch URL is `http://localhost:9201`, matching `compose.yaml`; `H2FLOW_ES_URL` can override it.
- No CI, lockfile, or agreed Python version.

**Benchmarking**
- The `rag` and `full` arms can't run until the integration items above are done. The benchmark's own TODOs are in
  [benchmarking/README.md](benchmarking/README.md#todo).
- Raw-model baseline (gemma3:4b): 73.7% on MATH-500, in line with the published 75.6, and 100% on the 27-problem calculus starter set.
  The calculus set needs harder problems before it can show whether the system helps. Details are in
  [benchmarking/README.md](benchmarking/README.md#baseline-results).
