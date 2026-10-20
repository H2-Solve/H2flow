"""Create index + bulk-load chunks (mirrors elastic_example/bonsai_load.py).

Usage:
  python -m search.index --bundle /path/to/osbooks-calculus-bundle --list-books
  python -m search.index --bundle /path/to/osbooks-calculus-bundle --book calculus-volume-1 --outline
  python -m search.index --bundle /path/to/osbooks-calculus-bundle --book calculus-volume-1
  python -m search.index --bundle /path/to/osbooks-calculus-bundle   # every book in the bundle
  python -m search.index --fixtures  # index the 2 test fixtures only
"""

from __future__ import annotations

import argparse
import sys

from elasticsearch import Elasticsearch, helpers

from .bundle import collect_chunks, outline_lines, read_bundle
from .parse_osbooks import parse_cnxml_file

INDEX_NAME = "textbook-chunks"

# Text search should see math the way people say it: x^2 -> "x squared", sqrt( -> "square root",
# f' -> "f prime". These rewrites run only inside Elasticsearch's analyzer, for indexing and for
# queries alike; the stored text that the LLM reads is not changed.
MATH_CHAR_FILTERS = {
    "math_squared": (r"\^2(?![0-9])", " squared "),
    "math_cubed": (r"\^3(?![0-9])", " cubed "),
    "math_sqrt": (r"sqrt\(", " square root "),
    "math_prime": ("\u2032", " prime "),
}

ANALYSIS = {
    "char_filter": {
        name: {"type": "pattern_replace", "pattern": pattern, "replacement": replacement}
        for name, (pattern, replacement) in MATH_CHAR_FILTERS.items()
    },
    "analyzer": {
        "math_text": {
            "type": "custom",
            "tokenizer": "standard",
            "char_filter": list(MATH_CHAR_FILTERS),
            "filter": ["lowercase"],
        }
    },
}

MAPPING = {
    "settings": {"analysis": ANALYSIS},
    "mappings": {
        "properties": {
            "book_id": {"type": "keyword"},
            "module_id": {"type": "keyword"},
            "title": {"type": "text"},
            "section": {"type": "text"},
            "chapter": {"type": "text"},
            "text": {"type": "text", "analyzer": "math_text"},
            "equations": {"type": "text"},
            "source": {"type": "keyword"},
            "excerpt": {"type": "text"},
        }
    },
}


def get_client(url: str = "http://localhost:9200") -> Elasticsearch:
    # No auth: matches docker run with xpack.security.enabled=false.
    return Elasticsearch(url)


def create_index(es: Elasticsearch, index: str = INDEX_NAME, reset: bool = False) -> None:
    if reset and es.indices.exists(index=index):
        es.indices.delete(index=index)
    if not es.indices.exists(index=index):
        es.indices.create(index=index, body=MAPPING)


def build_actions(chunks, index: str = INDEX_NAME) -> list[dict]:
    """One bulk action per chunk, with a stable _id ("<module_id>:<n>").

    Stable IDs mean re-running the loader replaces chunks instead of adding
    copies. A module shared by two books is stored once. If you change the
    parser or chunker, use --reset: leftover higher-numbered chunks from an
    older run would otherwise stay behind.
    """
    counters: dict[str, int] = {}
    actions = []
    for c in chunks:
        n = counters.get(c.module_id, 0)
        counters[c.module_id] = n + 1
        actions.append({"_op_type": "index", "_index": index, "_id": f"{c.module_id}:{n}", **c.to_es_doc()})
    return actions


def bulk_index(es: Elasticsearch, chunks, index: str = INDEX_NAME) -> int:
    actions = build_actions(chunks, index)
    helpers.bulk(es, actions)
    return len(actions)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=None, help="path to a cloned osbooks-*-bundle")
    ap.add_argument("--book", default=None, help="slug of one book in the bundle (default: all books)")
    ap.add_argument("--list-books", action="store_true", help="show the books in --bundle and exit")
    ap.add_argument("--outline", action="store_true", help="print chapter > module id + title and exit")
    ap.add_argument("--fixtures", action="store_true")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    if args.list_books:
        if not args.bundle:
            raise SystemExit("--list-books needs --bundle <path>")
        books = read_bundle(args.bundle)
        if not books:
            raise SystemExit(f"no collections found under {args.bundle}")
        for b in books:
            print(f"{b.slug}\t{b.title}\t{len(b.modules)} modules")
        return

    if args.outline:
        if not args.bundle:
            raise SystemExit("--outline needs --bundle <path>")
        sys.stdout.reconfigure(errors="replace")  # titles may hold characters the console cannot show
        try:
            print("\n".join(outline_lines(args.bundle, args.book)))
        except ValueError as exc:
            raise SystemExit(str(exc))
        return

    if args.fixtures:
        label = args.book or "calculus-volume-1"
        paths = ["tests/fixtures/m53472.cnxml", "tests/fixtures/m53481.cnxml"]
        chunks = [c for p in paths for c in parse_cnxml_file(p, label)]
        print(f"parsed {len(paths)} fixture modules as '{label}'")
    elif args.bundle:
        try:
            chunks, report = collect_chunks(args.bundle, args.book)
        except ValueError as exc:
            raise SystemExit(str(exc))
        for slug, st in report.per_book.items():
            print(f"{slug}: {st['modules']} modules, {st['chunks']} chunks")
        if report.missing:
            print(f"skipped {len(report.missing)} modules listed in a collection but missing on disk")
        for module_id, err in report.failed:
            print(f"FAILED to parse {module_id}: {err}")
    else:
        raise SystemExit("pass --fixtures or --bundle <path>")

    es = get_client()
    create_index(es, reset=args.reset)
    n = bulk_index(es, chunks)
    print(f"indexed {n} chunks into {INDEX_NAME}")


if __name__ == "__main__":
    main()