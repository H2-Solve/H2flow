"""Create index + bulk-load chunks (mirrors elastic_example/bonsai_load.py).

Usage:
  python -m search.index --bundle /path/to/osbooks-calculus-bundle --book calculus-volume-1
  python -m search.index --fixtures  # index the 2 test fixtures only
"""

from __future__ import annotations

import argparse

from elasticsearch import Elasticsearch, helpers

from .parse_osbooks import iter_module_files, parse_cnxml_file

INDEX_NAME = "textbook-chunks"

MAPPING = {
    "mappings": {
        "properties": {
            "book_id": {"type": "keyword"},
            "module_id": {"type": "keyword"},
            "title": {"type": "text"},
            "section": {"type": "text"},
            "text": {"type": "text"},
            "equations": {"type": "text"},
            "source": {"type": "keyword"},
            "excerpt": {"type": "text"},
        }
    }
}


def get_client(url: str = "http://localhost:9200") -> Elasticsearch:
    # No auth: matches docker run with xpack.security.enabled=false.
    return Elasticsearch(url)


def create_index(es: Elasticsearch, index: str = INDEX_NAME, reset: bool = False) -> None:
    if reset and es.indices.exists(index=index):
        es.indices.delete(index=index)
    if not es.indices.exists(index=index):
        es.indices.create(index=index, body=MAPPING)


def bulk_index(es: Elasticsearch, chunks, index: str = INDEX_NAME) -> int:
    actions = []
    for c in chunks:
        doc = c.to_es_doc()
        actions.append({"_op_type": "index", "_index": index, **doc})
    helpers.bulk(es, actions)
    return len(actions)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--book", default="calculus-volume-1")
    ap.add_argument("--fixtures", action="store_true")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    es = get_client()
    create_index(es, reset=args.reset)

    if args.fixtures:
        paths = ["tests/fixtures/m53472.cnxml", "tests/fixtures/m53481.cnxml"]
    elif args.bundle:
        paths = iter_module_files(args.bundle)
    else:
        raise SystemExit("pass --fixtures or --bundle <path>")

    chunks = []
    for p in paths:
        chunks.extend(parse_cnxml_file(p, args.book))
    n = bulk_index(es, chunks)
    print(f"indexed {n} chunks from {len(paths)} modules into {INDEX_NAME}")


if __name__ == "__main__":
    main()
