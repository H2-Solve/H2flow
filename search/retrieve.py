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


def get_documents(query: str, top_k: int = 5, index: str = INDEX_NAME,
                  es: Elasticsearch | None = None) -> list[dict]:
    """Query ES and return [{text, book_id, title, module_id, section, excerpt, score}]."""
    es = es or get_client()
    res = es.search(
        index=index,
        size=top_k,
        query={"match": {"text": query}},
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
                "excerpt": src.get("excerpt", src.get("text", "")[:500]),
                "score": hit.get("_score", 0.0),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()
    for i, d in enumerate(get_documents(args.query, top_k=args.top_k)):
        print(f"--- hit {i + 1} score={d['score']:.2f} {d['module_id']} / {d['section']}")
        print(d["excerpt"][:400])
        print()


if __name__ == "__main__":
    main()
