from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import ollama

llm = ChatOllama(
    model="gemma3:4b",
    temperature=0
)


def get_documents(query):
    """
    Interface to the separate Elasticsearch project.
    Returns relevant documents.
    """
    pass


def format_context(documents):
    """
    Converts Elasticsearch documents into text
    that can be placed into the LLM prompt.
    """
    pass


def build_prompt(query, context):
    """
    Creates the RAG prompt containing the retrieved
    context and user's question.
    """
    pass


def generate_answer(prompt):
    """
    Sends the prompt to the Ollama model.
    """
    pass


def rag(query):
    """
    Complete RAG pipeline.
    """
    # documents = get_documents(query)
    # context = format_context(documents)
    # prompt = build_prompt(query, context)
    # answer = generate_answer(prompt)

    # return answer
