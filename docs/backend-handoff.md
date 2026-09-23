# Backend handoff

## Branch review

On September 23, 2026, after fetching origin, `main`, `backend`,
`books_elastisearch`, and `rag-implementation` all pointed to `e1ac0bf`
(`Initial commit`). Each contained only `README.md` with the project name.
There was no implementation to reuse or conflicting API contract to preserve.
Work not pushed by teammates could not be reviewed.

Working branch: `dylan/backend-api-starter`, created from `main`.

## Proposed API contract

`POST /api/v1/questions` accepts:

- `question`: required string, trimmed, 1–4000 characters.
- `top_k`: optional integer, 1–10, defaults to 5. Requested maximum number of
  retrieved chunks; the demo has only one fixture and does not perform search.

The response contains `status`, `mode`, `answer`, `steps`, and `citations`.
Each citation has `book_id`, `title`, `page` (one-based), and `excerpt`.
Confirm with the ingestion owner whether `page` means PDF page or printed page
before implementing the real adapter.

Status values:

- `answered`: the provider produced an answer.
- `no_sources`: the live provider found no relevant textbook context.
- `unsupported_demo_question`: the fixture cannot answer this input.

HTTP 422 means invalid input. HTTP 503 means an adapter raised
`ServiceUnavailable`. `/health` reports API process liveness only.

## Connect the real RAG service

1. Agree on the citation metadata and synchronous/asynchronous calling style.
2. Implement `AnswerService.answer(request)` in a new adapter using the team's
   RAG function. The current route is synchronous; FastAPI runs it in a worker
   thread. If the team uses async clients, update the interface and route together.
3. Return `AnswerResponse(mode="live", ...)`, preserving real source metadata.
   Return `status="no_sources"` with no citations if retrieval finds nothing.
4. Translate known upstream connection/time-out errors to `ServiceUnavailable`.
   Set finite timeouts in the Elasticsearch and LLM clients; don't swallow
   programming errors as service outages.
5. Change `get_answer_service()` to return the adapter. For persistent clients,
   add application startup/shutdown management to open and close them.
6. Add integration tests with the team's actual services and sample textbook.

The dependency-override test demonstrates the seam without needing external
services. Live credentials and environment configuration should be introduced
with the actual adapter; `.env` is ignored by Git.

## Next useful backend tasks

- Agree on this contract with the frontend and RAG owners.
- Integrate the RAG adapter once its function and result shape exist.
- Configure specific CORS origins once the frontend's origin is known.
- Decide whether questions/history or textbook uploads need persistence and routes.
- Add deployment configuration and access controls before public hosting.

This starter intentionally has no database or upload pipeline. Those decisions
depend on the team's ingestion design. Dependencies use bounded version ranges;
agree on a lockfile and shared Python version before deployment.

## Share this work

The branch is local until you commit and push it. After reviewing:

```bash
git add .gitignore README.md requirements.txt requirements-dev.txt backend tests docs
git commit -m "Add FastAPI backend starter and RAG integration contract"
git push -u origin dylan/backend-api-starter
```
