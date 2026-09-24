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


print(rag("What is the area of a circle with radius 5?").content)
