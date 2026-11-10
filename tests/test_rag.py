from types import SimpleNamespace

import pytest


def test_rag_uses_elasticsearch_retrieval(monkeypatch):
    import RAG.main as rag_module

    documents = [{"text": "The power rule.", "title": "Derivatives", "section": "Rules"}]
    monkeypatch.setattr(rag_module, "get_documents", lambda query, top_k, index: documents)
    monkeypatch.setattr(
        rag_module,
        "generate_answer",
        lambda value: SimpleNamespace(content=value),
    )

    result = rag_module.rag("differentiate x^2", top_k=3, index="test-index")

    assert "The power rule." in result.content


def test_format_context_includes_citation_metadata():
    from RAG.main import format_context

    context = format_context([{
        "text": "Use the power rule.",
        "book_id": "calculus-volume-1",
        "chapter": "Derivatives",
        "title": "The Basic Rules",
        "section": "The Power Rule",
    }])

    assert "calculus-volume-1 > Derivatives > The Basic Rules > The Power Rule" in context
    assert "Use the power rule." in context


def test_verified_rag_retries_until_answer_matches(monkeypatch):
    import RAG.main as rag_module

    monkeypatch.setattr(rag_module, "get_documents", lambda *args, **kwargs: [])
    responses = iter([
        SimpleNamespace(content="Work\nFINAL_ANSWER: x"),
        SimpleNamespace(content="Work\nFINAL_ANSWER: 2*x"),
    ])
    calls = []

    def generate(prompt):
        calls.append(prompt)
        return next(responses)

    result = rag_module.rag_with_verification(
        "Find the derivative of x**2",
        max_retries=2,
        intent_extractor=lambda question: {
            "operation": "derivative",
            "expression": "x**2",
            "variable": "x",
        },
        answer_generator=generate,
    )

    assert "FINAL_ANSWER: 2*x" in result
    assert len(calls) == 2
    assert "Previous attempt was not verified" in calls[1]


def test_verified_rag_rejects_exhausted_retries(monkeypatch):
    import RAG.main as rag_module

    monkeypatch.setattr(rag_module, "get_documents", lambda *args, **kwargs: [])
    with pytest.raises(ValueError, match="Unable to verify"):
        rag_module.rag_with_verification(
            "Find the derivative of x**2",
            max_retries=1,
            intent_extractor=lambda question: {
                "operation": "derivative",
                "expression": "x**2",
                "variable": "x",
            },
            answer_generator=lambda prompt: "FINAL_ANSWER: x",
        )


def test_unsupported_question_skips_verification(monkeypatch):
    import RAG.main as rag_module

    monkeypatch.setattr(rag_module, "get_documents", lambda *args, **kwargs: [])
    result = rag_module.rag_with_verification(
        "Explain the product rule",
        intent_extractor=lambda question: {"operation": "unsupported"},
        answer_generator=lambda prompt: "retrieval answer",
    )

    assert result == "retrieval answer"
