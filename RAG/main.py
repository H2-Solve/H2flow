from __future__ import annotations

import os
from typing import Any, Callable

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from search.retrieve import get_documents as retrieve_documents
from verification import (
    answers_are_equivalent,
    extract_final_answer_from_response,
    extract_math_intent,
    solve_with_sympy,
)

llm = ChatOllama(
    model="gemma3:4b",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a specialized math question-answering assistant.
Use the provided context to answer the question.
Specifically, look for any formulas that may help you answer the question.
For each step, shortly (1 - 2 sentences) explain your reasoning.
If the context does not contain enough information,
do your best to answer the question, but state the limitations."""
    ),
    (
        "human",
        """Context:
{context}

Question:
{question}"""
    )
])

# TODO: Maybe use solver_llm and verification_prompt to implement RAG with verification
# LLM for generating the step-by-step RAG explanation
solver_llm = ChatOllama(
    model="gemma3:4b",
    temperature=0.2 # Slight temperature allows variation on retry attempts
)

verification_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a specialized math question-answering assistant.
        Use the provided textbook context to explain step-by-step how to solve the user's question.
        For each step, shortly (1-2 sentences) explain your reasoning using relevant formulas from the context.

        CRITICAL INSTRUCTION:
        The verified ground-truth mathematical answer to this problem is:
        {ground_truth}

        Your step-by-step work MUST lead to this exact result.
        At the very end of your response, on a new line, write ONLY:
        FINAL_ANSWER: <your final expression in Python/SymPy syntax>
        {retry_feedback}
        """
    ),
    (
        "human",
        """
        Context:
        {context}

        Question:
        {question}
        """
    )
])


def get_documents(query: str, top_k: int = 5, index: str | None = None) -> list[dict]:
    """Retrieve textbook chunks from Elasticsearch for a user question."""
    return retrieve_documents(
        query,
        top_k=top_k,
        index=index or os.environ.get("H2FLOW_ES_INDEX", "textbook-chunks"),
    )


def format_context(documents):
    """
    Converts Elasticsearch documents into text
    that can be placed into the LLM prompt.
    """
    context = []

    for i, document in enumerate(documents):
        source = " > ".join(
            value for value in (
                document.get("book_id"),
                document.get("chapter"),
                document.get("title"),
                document.get("section"),
            ) if value
        )
        citation = f"Source: {source}\n" if source else ""
        context.append(f"Document {i + 1}:\n{citation}{document.get('text', '')}")

    return "\n\n".join(context)


def build_prompt(query, context):
    """
    Creates the RAG prompt containing the retrieved
    context and user's question.
    """
    return prompt.format(
        context=context,
        question=query
    )


def generate_answer(prompt):
    """
    Sends the prompt to the Ollama model.
    """
    return llm.invoke(prompt)


def rag(query: str, top_k: int = 5, index: str | None = None):
    """
    Complete RAG pipeline.
    """
    documents = get_documents(query, top_k=top_k, index=index)
    context = format_context(documents)
    prompt = build_prompt(query, context)
    answer = generate_answer(prompt)

    return answer


def rag_with_verification(
    question: str,
    max_retries: int = 3,
    top_k: int = 5,
    index: str | None = None,
    documents: list[dict] | None = None,
    intent_extractor: Callable[[str], dict] = extract_math_intent,
    answer_generator: Callable[[Any], Any] | None = None,
) -> str:
    """Generate a textbook-grounded answer and retry until SymPy verifies it."""
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")

    retrieved_documents = documents if documents is not None else get_documents(
        question, top_k=top_k, index=index
    )
    context = format_context(retrieved_documents)
    intent = intent_extractor(question)
    truth = solve_with_sympy(intent)

    if truth is None:
        response = (answer_generator or generate_answer)(
            build_prompt(question, context)
        )
        return str(getattr(response, "content", response))

    ground_truth = str(truth)
    feedback = ""
    for _ in range(max_retries):
        response = (answer_generator or solver_llm.invoke)(
            verification_prompt.format(
                context=context,
                question=question,
                ground_truth=ground_truth,
                retry_feedback=feedback,
            )
        )
        text = str(getattr(response, "content", response))
        answer = extract_final_answer_from_response(text)
        variable = intent.get("variable") or "x"
        if answers_are_equivalent(truth, answer, variable):
            return text
        feedback = (
            "\nPrevious attempt was not verified. "
            f"Expected an expression equivalent to {ground_truth}. "
            "Recheck the algebra and output a new FINAL_ANSWER."
        )

    raise ValueError(
        f"Unable to verify generated answer after {max_retries} attempts"
    )


if __name__ == "__main__":
    print(rag("What is the area of a circle with radius 5?").content)
