"""Query building for retrieval experiments (--fields / --match-type)."""

import pytest

pytest.importorskip("elasticsearch")

from search.evaluate import describe_search  # noqa: E402
from search.retrieve import build_query, parse_fields  # noqa: E402


def test_default_query_is_unchanged_text_match():
    assert build_query("power rule") == {"match": {"text": "power rule"}}
    assert build_query("power rule", fields=[]) == {"match": {"text": "power rule"}}


def test_fields_build_a_weighted_multi_match():
    q = build_query("power rule", ["text", "section^3"], "best_fields")
    assert q == {"multi_match": {"query": "power rule", "fields": ["text", "section^3"], "type": "best_fields"}}
    assert build_query("x", ["text"])["multi_match"]["type"] == "most_fields"


def test_demote_wraps_the_query_in_a_boosting_query():
    q = build_query("power rule", demote=0.3)
    assert q["boosting"]["positive"] == {"match": {"text": "power rule"}}
    assert q["boosting"]["negative_boost"] == 0.3
    phrases = [c["match_phrase"]["section"] for c in q["boosting"]["negative"]["bool"]["should"]]
    assert phrases == ["Exercises", "Glossary"]
    q2 = build_query("x", ["text", "section"], demote=0.5)
    assert "multi_match" in q2["boosting"]["positive"]


def test_parse_fields():
    assert parse_fields("text, section^3 ,title^2") == ["text", "section^3", "title^2"]
    assert parse_fields("") is None and parse_fields(None) is None


def test_describe_search():
    assert describe_search(None, "most_fields") == "text only (match)"
    assert describe_search(["text", "section^3"], "most_fields") == "text,section^3 [most_fields]"
    assert describe_search(None, "most_fields", 0.3) == "text only (match), demote exercises/glossary x0.3"


def test_section_field_finds_chunks_the_text_alone_misses():
    """Needs Docker ES: docker start es. Skipped otherwise."""
    from elasticsearch import Elasticsearch

    client = Elasticsearch("http://localhost:9200", request_timeout=2)
    try:
        up = client.ping()
    except Exception:
        up = False
    if not up:
        pytest.skip("ES not running at localhost:9200")

    from search.index import bulk_index, create_index
    from search.retrieve import get_documents
    from search.schemas import Chunk

    index = "textbook-fields-test"
    create_index(client, index=index, reset=True)
    try:
        chunks = [
            Chunk("b", "m1", "Rules", "The Quotient Rule", "Example: differentiate f(x)=x^2/(x+1) step by step.", [], "osbooks"),
            Chunk("b", "m2", "Misc", "Other", "Nothing relevant here.", [], "osbooks"),
        ]
        bulk_index(client, chunks, index=index)
        client.indices.refresh(index=index)
        assert get_documents("quotient rule", index=index, es=client) == []
        hits = get_documents("quotient rule", index=index, es=client, fields=["text", "section^3"])
        assert [h["module_id"] for h in hits] == ["m1"]
    finally:
        client.indices.delete(index=index)


def test_demoted_sections_rank_lower():
    """Needs Docker ES: docker start es. Skipped otherwise."""
    from elasticsearch import Elasticsearch

    client = Elasticsearch("http://localhost:9200", request_timeout=2)
    try:
        up = client.ping()
    except Exception:
        up = False
    if not up:
        pytest.skip("ES not running at localhost:9200")

    from search.index import bulk_index, create_index
    from search.retrieve import get_documents
    from search.schemas import Chunk

    index = "textbook-demote-test"
    create_index(client, index=index, reset=True)
    try:
        same_text = "Find the derivative of the function."
        chunks = [
            Chunk("b", "m1", "T", "Exercises", same_text, [], "osbooks"),
            Chunk("b", "m2", "T", "The Power Rule", same_text, [], "osbooks"),
        ]
        bulk_index(client, chunks, index=index)
        client.indices.refresh(index=index)
        hits = get_documents("derivative", index=index, es=client, demote=0.1)
        assert [h["module_id"] for h in hits] == ["m2", "m1"]
        assert hits[0]["score"] > hits[1]["score"]
    finally:
        client.indices.delete(index=index)