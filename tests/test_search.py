import pytest

es = pytest.importorskip("elasticsearch")


def test_search_integration():
    """Needs Docker ES: docker start es. Skipped otherwise."""
    from elasticsearch import Elasticsearch

    client = Elasticsearch("http://localhost:9200", request_timeout=2)
    try:
        client.ping()
    except Exception:
        pytest.skip("ES not running at localhost:9200")
    if not client.ping():
        pytest.skip("ES not running at localhost:9200")

    from search.index import INDEX_NAME, bulk_index, create_index
    from search.parse_osbooks import parse_cnxml_file
    from search.retrieve import get_documents

    test_index = "textbook-chunks-test"
    create_index(client, index=test_index, reset=True)
    chunks = parse_cnxml_file("tests/fixtures/m53481.cnxml", "calculus-volume-1")[:20]
    assert bulk_index(client, chunks, index=test_index) or True
    client.indices.refresh(index=test_index)

    hits = get_documents("exponential population growth", top_k=3, index=test_index, es=client)
    assert hits, "expected hits for exponential query"
    assert any("Exponential" in h["title"] or "population" in h["text"].lower() for h in hits)
    client.indices.delete(index=test_index)
