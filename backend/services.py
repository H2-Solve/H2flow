"""Replace the demo provider with the team's RAG adapter at integration time."""

from typing import Protocol

from backend.models import AnswerResponse, Citation, QuestionRequest


class ServiceUnavailable(Exception):
    """An adapter should raise this if Elasticsearch or the LLM is unavailable."""


class AnswerService(Protocol):
    def answer(self, request: QuestionRequest) -> AnswerResponse:
        """Retrieve textbook passages, generate an answer, and retain citations."""
        ...


class DemoAnswerService:
    def answer(self, request: QuestionRequest) -> AnswerResponse:
        normalized = "".join(request.question.lower().split()).rstrip(".?")
        if normalized not in {"solve2x+3=7", "2x+3=7"}:
            return AnswerResponse(
                status="unsupported_demo_question",
                mode="demo",
                answer="This demo only supports 'Solve 2x + 3 = 7'. The real RAG service is not connected yet.",
            )
        return AnswerResponse(
            status="answered",
            mode="demo",
            answer="x = 2",
            steps=[
                "Subtract 3 from both sides: 2x = 4.",
                "Divide both sides by 2: x = 2.",
                "Check: 2(2) + 3 = 7.",
            ],
            citations=[Citation(
                book_id="demo-algebra",
                title="Demo algebra notes (synthetic fixture, not a textbook)",
                page=1,
                excerpt="To solve a linear equation, apply the same operation to both sides to isolate the variable.",
            )],
        )


def get_answer_service() -> AnswerService:
    """Integration point: return the real RAG adapter here when available."""
    return DemoAnswerService()
