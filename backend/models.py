"""The proposed JSON contract shared with the frontend and RAG team."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class QuestionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    question: str = Field(min_length=1, max_length=4000, examples=["Solve 2x + 3 = 7"])
    top_k: int = Field(default=5, ge=1, le=10, strict=True)


class Citation(BaseModel):
    book_id: str
    title: str
    page: int = Field(ge=1)
    excerpt: str


class AnswerResponse(BaseModel):
    status: Literal["answered", "no_sources", "unsupported_demo_question"]
    mode: Literal["demo", "live"]
    answer: str
    steps: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    detail: str
