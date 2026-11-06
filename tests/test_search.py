import pytest

es = pytest.importorskip("elasticsearch")


def test_search_integration(review_index):
    from search.index import bulk_index
    from search.parse_osbooks import parse_cnxml_file
    from search.retrieve import get_documents

    client, test_index = review_index
    chunks = parse_cnxml_file("tests/fixtures/m53481.cnxml", "calculus-volume-1")[:20]
    assert bulk_index(client, chunks, index=test_index) == len(chunks)
    client.indices.refresh(index=test_index)

    hits = get_documents("exponential population growth", top_k=3, index=test_index, es=client)
    assert hits, "expected hits for exponential query"
    assert any("Exponential" in h["title"] or "population" in h["text"].lower() for h in hits)
