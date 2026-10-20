"""Math-aware text analysis: x^2 should be searchable as "x squared"."""

import re

import pytest

pytest.importorskip("elasticsearch")

from search.index import MAPPING, MATH_CHAR_FILTERS, create_index  # noqa: E402


def apply_filters(text: str) -> str:
    """Python stand-in for the Elasticsearch char filters (same simple regex syntax)."""
    for pattern, replacement in MATH_CHAR_FILTERS.values():
        text = re.sub(pattern, replacement, text)
    return " ".join(text.split())


def test_math_symbols_become_the_words_people_search_for():
    assert "x squared" in apply_filters("f(x)=x^2")
    assert "x cubed" in apply_filters("f(x)=x^3+1")
    assert "square root 2" in apply_filters("sqrt(2)")
    assert "f prime" in apply_filters("f\u2032(x)=2x")
    assert "x^20" in apply_filters("x^20"), "only a bare ^2 means squared"
    assert "2^(t/4)" in apply_filters("2^(t/4)"), "other exponents are left alone"


def test_mapping_uses_every_filter_in_the_text_analyzer():
    analysis = MAPPING["settings"]["analysis"]
    assert set(analysis["analyzer"]["math_text"]["char_filter"]) == set(analysis["char_filter"])
    assert MAPPING["mappings"]["properties"]["text"]["analyzer"] == "math_text"


def test_elasticsearch_accepts_the_analyzer():
    """Needs Docker ES: docker start es. Skipped otherwise."""
    from elasticsearch import Elasticsearch

    client = Elasticsearch("http://localhost:9200", request_timeout=2)
    try:
        up = client.ping()
    except Exception:
        up = False
    if not up:
        pytest.skip("ES not running at localhost:9200")

    index = "textbook-analyzer-test"
    create_index(client, index=index, reset=True)
    try:
        tokens = [t["token"] for t in client.indices.analyze(index=index, analyzer="math_text", text="f(x)=x^2")["tokens"]]
        assert "squared" in tokens, tokens
        client.index(index=index, id="1", document={"text": "The derivative of f(x)=x^2 is f\u2032(x)=2x."}, refresh=True)
        for query in ("x squared", "x^2", "f prime"):
            res = client.search(index=index, query={"match": {"text": query}})
            assert res["hits"]["total"]["value"] == 1, query
    finally:
        client.indices.delete(index=index)