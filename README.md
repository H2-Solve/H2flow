# H2flow

A capstone project for answering math questions using textbook retrieval,
Elasticsearch, and an LLM. This branch adds the FastAPI API layer.

**Current state:** runnable API with a fixed, clearly labeled demo answer.
No textbooks, Elasticsearch connection, or LLM are connected yet.

## Run locally

Use Python 3.10+ (Python 3.12 was used for development). From the repo root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m uvicorn backend.main:app --reload
```

If your Python 3.10+ executable is named `python3`, use that in the first command.
Open http://127.0.0.1:8000/docs for the interactive API demo.
No API keys or external services are needed. Stop the server with Ctrl+C.

## Tomorrow's demo (two minutes)

1. Open `/docs` and execute `GET /health` to show the backend is running.
2. Expand `POST /api/v1/questions`, click **Try it out**, and execute:

   ```json
   {"question": "Solve 2x + 3 = 7", "top_k": 5}
   ```

3. Show the answer, solution steps, citation format, and `mode: "demo"`.
   The citation is synthetic demo data, not a retrieved textbook passage.
4. Submit an empty question to show validation (HTTP 422).
5. Explain that the RAG implementation plugs into `get_answer_service()`;
   the frontend can already build against this API contract.

You can also call it from a terminal while the server is running:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/questions \
  -H 'Content-Type: application/json' \
  -d '{"question":"Solve 2x + 3 = 7","top_k":5}'
```

Other questions return `unsupported_demo_question`; the fixture does not solve
arbitrary math problems.

## What “backend” means for this team

Your part is the API that connects the user interface to your teammates' work:

```text
Frontend → FastAPI → RAG service → Elasticsearch textbook passages
                               → LLM answer using those passages
         ← answer + steps + citations
```

| Owner (proposed) | Responsibility |
| --- | --- |
| Backend / you | HTTP routes, input validation, response format, error handling, integration tests |
| Elasticsearch teammate | Textbook ingestion, chunking, indexing, search, source metadata |
| RAG / LLM teammate | Retrieve useful passages, construct prompts, generate grounded answers and citations |
| Frontend teammate | Question input and display of answers, steps, sources, and errors |

The RAG teammate should coordinate retrieval with the Elasticsearch teammate;
the API should not run a second retrieval pipeline. Confirm these ownership
boundaries with the group.

Suggested progress update:

> I set up the FastAPI backend, a validated question endpoint, a structured
> answer and citation contract, interactive API docs, and automated tests.
> It currently uses a demo fixture. We can connect the RAG service through one
> adapter without changing the frontend endpoint.

## Files to understand first

- `backend/main.py`: HTTP endpoints and mapping an unavailable service to HTTP 503.
- `backend/models.py`: accepted inputs and returned JSON structures.
- `backend/services.py`: service interface, demo implementation, and integration point.
- `tests/test_api.py`: API behavior and replacement-service tests.
- [Integration notes](docs/backend-handoff.md): proposed contract and next steps.

Run tests with the virtual environment activated:

```bash
python -m pytest -q
```

Implementation references: FastAPI's official [dependencies guide](https://fastapi.tiangolo.com/tutorial/dependencies/)
and [testing guide](https://fastapi.tiangolo.com/tutorial/testing/).
