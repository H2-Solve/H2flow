"""Measure PR #2 with real XML and a disposable Elasticsearch index.

Run from the repository root:
  python -m eval.benchmark_pr2 --bundle /path/to/bundle --es-url http://localhost:9201 --output eval/benchmark_current.json
Use --all-books to measure the full bundle. No existing index is modified.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
import tracemalloc
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

from elasticsearch import Elasticsearch

from search.bundle import collect_chunks, read_bundle
from search.evaluate import run_eval, summarize
from search.index import build_actions, bulk_index, create_index
from search.retrieve import get_documents


def latency_summary(values):
    ordered = sorted(values)
    return {
        "requests": len(values),
        "p50_ms": statistics.median(values),
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "max_ms": max(values),
    }


def git_sha(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--es-url", required=True)
    ap.add_argument("--book", default="calculus-volume-1")
    ap.add_argument("--all-books", action="store_true")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.repeat < 1:
        ap.error("--repeat must be positive")

    selected = None if args.all_books else args.book
    parse_seconds = []
    for _ in range(3):
        start = time.perf_counter()
        chunks, report = collect_chunks(args.bundle, selected)
        parse_seconds.append(time.perf_counter() - start)
        del chunks

    tracemalloc.start()
    chunks, report = collect_chunks(args.bundle, selected)
    _, parse_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    books = read_bundle(args.bundle)
    if selected:
        books = [book for book in books if book.slug == selected]
    memberships = Counter(module_id for book in books for module_id, _ in book.modules)
    content_hashes = Counter(
        hashlib.sha256(" ".join(chunk.text.split()).encode()).hexdigest()
        for chunk in chunks
    )
    output = {
        "pr_head": git_sha(Path.cwd()),
        "dataset_commit": git_sha(args.bundle),
        "selected_book": selected or "all",
        "machine": {
            "os": platform.system(),
            "architecture": platform.machine(),
            "logical_cpus": os.cpu_count(),
        },
        "per_book": report.per_book,
        "missing_modules": report.missing,
        "failed_modules": report.failed,
        "unique_modules": len({chunk.module_id for chunk in chunks}),
        "shared_module_ids": sum(count > 1 for count in memberships.values()),
        "chunks": len(chunks),
        "text_characters": sum(len(chunk.text) for chunk in chunks),
        "identical_normalized_chunk_excess": sum(
            count - 1 for count in content_hashes.values()
        ),
        "parse_seconds": parse_seconds,
        "parse_median_seconds": statistics.median(parse_seconds),
        "parse_python_peak_mib": parse_peak / 1024**2,
    }

    index = f"textbook-pr2-benchmark-{uuid4().hex}"
    with Elasticsearch(args.es_url, request_timeout=30, max_retries=0) as es:
        output["elasticsearch_version"] = es.info()["version"]["number"]
        create_index(es, index=index)
        try:
            tracemalloc.start()
            start = time.perf_counter()
            actions = build_actions(chunks, index)
            output["build_actions_seconds"] = time.perf_counter() - start
            _, action_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            output["build_actions_additional_python_peak_mib"] = action_peak / 1024**2
            del actions

            start = time.perf_counter()
            indexed = bulk_index(es, chunks, index=index)
            output["bulk_index_seconds"] = time.perf_counter() - start
            output["bulk_documents_per_second"] = indexed / output["bulk_index_seconds"]
            start = time.perf_counter()
            es.indices.refresh(index=index)
            output["refresh_seconds"] = time.perf_counter() - start
            output["indexed_documents"] = es.count(index=index)["count"]
            assert output["indexed_documents"] == len(chunks)

            tuning = json.loads(Path("eval/questions.json").read_text())
            holdout = json.loads(Path("eval/holdout.json").read_text())
            questions = tuning + holdout
            first_latencies = []

            def first_search(query, k):
                start = time.perf_counter()
                hits = get_documents(query, top_k=k, index=index, es=es)
                first_latencies.append((time.perf_counter() - start) * 1000)
                return hits

            results = run_eval(questions, first_search, top_k=5)
            output["first_pass_latency"] = latency_summary(first_latencies)
            output["tuning_module_metrics"] = summarize(
                [result["rank"] for result in results[: len(tuning)]]
            )
            output["holdout_module_metrics"] = summarize(
                [result["rank"] for result in results[len(tuning) :]]
            )
            output["questions_with_duplicate_top5_text"] = sum(
                len(hashes := [" ".join(hit["text"].split()) for hit in result["hits"]])
                != len(set(hashes))
                for result in results
            )
            output["results"] = [
                {
                    "question": result["question"],
                    "expected": result["expect"],
                    "rank": result["rank"],
                    "top_hits": [
                        {
                            key: hit[key]
                            for key in (
                                "module_id",
                                "book_id",
                                "title",
                                "section",
                                "excerpt",
                            )
                        }
                        for hit in result["hits"][:3]
                    ],
                }
                for result in results
            ]
            queries = [question["question"] for question in questions] * args.repeat

            def timed_search(query):
                start = time.perf_counter()
                get_documents(query, top_k=5, index=index, es=es)
                return (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            sequential = [timed_search(query) for query in queries]
            sequential_seconds = time.perf_counter() - start
            output["warm_sequential"] = {
                **latency_summary(sequential),
                "requests_per_second": len(queries) / sequential_seconds,
            }
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=4) as pool:
                concurrent = list(pool.map(timed_search, queries))
            concurrent_seconds = time.perf_counter() - start
            output["warm_four_workers"] = {
                **latency_summary(concurrent),
                "requests_per_second": len(queries) / concurrent_seconds,
            }

            fresh_client_latencies = []
            for question in questions:
                start = time.perf_counter()
                with Elasticsearch(
                    args.es_url, request_timeout=30, max_retries=0
                ) as fresh:
                    get_documents(question["question"], top_k=5, index=index, es=fresh)
                fresh_client_latencies.append((time.perf_counter() - start) * 1000)
            output["warm_fresh_client_each_request"] = latency_summary(
                fresh_client_latencies
            )
            output["primary_store_bytes"] = es.indices.stats(index=index)["_all"][
                "primaries"
            ]["store"]["size_in_bytes"]
        finally:
            es.indices.delete(index=index)

    Path(args.output).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in output.items() if key != "results"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
