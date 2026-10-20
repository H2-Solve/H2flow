"""Live retrieval: drop-in replacement for RAG's get_documents_dummy().

Matches elastic_example/bonsai_query.py pattern (match query), but returns
the dict shape RAG + backend need: text + citation metadata.

Usage:
  python -m search.retrieve "What is the area of a circle?" --top-k 3
"""

from __future__ import annotations

import argparse

from elasticsearch import Elasticsearch

from .index import INDEX_NAME, get_client


def parse_fields(spec: str | None) -> list[str] | None:
    """'text,section^3' -> ['text', 'section^3']. None or empty means the default query."""
    fields = [f.strip() for f in (spec or "").split(",") if f.strip()]
    return fields or None


# Chunks from these sections are lists of problems or terms: they match many words loosely
# but explain nothing, so --demote can push them down the ranking.
LOW_VALUE_SECTIONS = ("Exercises", "Glossary")


def build_query(query: str, fields: list[str] | None = None, match_type: str = "most_fields",
                demote: float | None = None) -> dict:
    """Default: a plain match on the text field. With fields: a weighted multi_match.

    Fields may carry a weight, e.g. "section^3". most_fields adds the scores of every
    field that matches, so a chunk matching in its text AND its section outranks one
    matching only in the text; best_fields uses only the single best field.
    With demote (0 < x < 1), scores of chunks in LOW_VALUE_SECTIONS are multiplied by x.
    """
    base = {"match": {"text": query}} if not fields else {
        "multi_match": {"query": query, "fields": fields, "type": match_type}
    }
    if demote is None:
        return base
    negative = {
        "bool": {
            "should": [{"match_phrase": {"section": name}} for name in LOW_VALUE_SECTIONS],
            "minimum_should_match": 1,
        }
    }
    return {"boosting": {"positive": base, "negative": negative, "negative_boost": demote}}


def get_documents(query: str, top_k: int = 5, index: str = INDEX_NAME,
                  es: Elasticsearch | None = None, fields: list[str] | None = None,
                  match_type: str = "most_fields", demote: float | None = None) -> list[dict]:
    """Query ES and return [{text, book_id, title, module_id, section, excerpt, score}]."""
    es = es or get_client()
    res = es.search(
        index=index,
        size=top_k,
        query=build_query(query, fields, match_type, demote),
    )
    out: list[dict] = []
    for hit in res["hits"]["hits"]:
        src = hit["_source"]
        out.append(
            {
                "text": src.get("text", ""),
                "book_id": src.get("book_id", ""),
                "title": src.get("title", ""),
                "module_id": src.get("module_id", ""),
                "section": src.get("section", ""),
                "chapter": src.get("chapter", ""),
                "excerpt": src.get("excerpt", src.get("text", "")[:500]),
                "score": hit.get("_score", 0.0),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--fields", help='comma-separated, e.g. "text,section^3,title^2" (default: text only)')
    ap.add_argument("--match-type", default="most_fields", choices=["most_fields", "best_fields"])
    ap.add_argument("--demote", type=float, help="multiply scores of Exercises/Glossary chunks by this (0 to 1)")
    args = ap.parse_args()
    docs = get_documents(args.query, top_k=args.top_k, fields=parse_fields(args.fields),
                         match_type=args.match_type, demote=args.demote)
    for i, d in enumerate(docs):
        print(f"--- hit {i + 1} score={d['score']:.2f} {d['module_id']} / {d['section']}")
        print(d["excerpt"][:400])
        print()


if __name__ == "__main__":
    main()