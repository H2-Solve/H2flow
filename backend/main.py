"""Run from the repo root: python -m uvicorn backend.main:app --reload."""

import logging

from fastapi import Depends, FastAPI, HTTPException

from backend.models import AnswerResponse, ErrorResponse, QuestionRequest
from backend.services import AnswerService, ServiceUnavailable, get_answer_service

logger = logging.getLogger(__name__)

app = FastAPI(
    title="H2flow Math Tutor API",
    version="0.1.0",
    description=(
        "Backend starter for textbook-grounded math answers. Currently uses a "
        "fixed demo fixture; Elasticsearch and LLM integration are pending."
    ),
)


@app.get("/health", tags=["Health"])
def health() -> dict[str, str]:
    """Process liveness only; this does not test Elasticsearch or an LLM."""
    return {"status": "ok"}


@app.post(
    "/api/v1/questions",
    response_model=AnswerResponse,
    responses={503: {"model": ErrorResponse, "description": "RAG service unavailable"}},
    tags=["Questions"],
)
def ask_question(
    request: QuestionRequest,
    service: AnswerService = Depends(get_answer_service),
) -> AnswerResponse:
    """Validate a question and return an answer, explanation, and sources."""
    try:
        return service.answer(request)
    except ServiceUnavailable as exc:
        logger.warning("The answer service is unavailable")
        raise HTTPException(
            status_code=503,
            detail="The answer service is temporarily unavailable. Please try again later.",
        ) from exc
