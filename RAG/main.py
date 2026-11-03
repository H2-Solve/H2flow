from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import ollama

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


def get_documents(query):
    """
    Interface to the separate Elasticsearch project.
    Returns relevant documents.
    """
    pass


def get_documents_dummy(query):
    """
    Dummy function to simulate Elasticsearch retrieval.
    """
    return [
        {
            "text": "The area of a circle is given by the formula A = πr^2, where r is the radius."
        },
        {
            "text": "The circumference of a circle is given by C = 2πr."
        }
    ]


def format_context(documents):
    """
    Converts Elasticsearch documents into text
    that can be placed into the LLM prompt.
    """
    context = []

    for i, document in enumerate(documents):
        context.append(
            f"""Document {i + 1}:
{document["text"]}"""
        )

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


def rag(query):
    """
    Complete RAG pipeline.
    """
    documents = get_documents_dummy(query)
    context = format_context(documents)
    prompt = build_prompt(query, context)
    answer = generate_answer(prompt)

    return answer


# TODO: Implement the full RAG with SymPy verification and retry loops
#       and as a single function for future API implementation)
def rag_with_verification(question: str, max_retries: int = 3) -> str:
    """Full pipeline: Problem Intent Extraction -> SymPy Truth -> RAG -> Verification & Retry"""
    return ""


print(rag("What is the area of a circle with radius 5?").content)
