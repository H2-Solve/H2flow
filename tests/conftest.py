"""Isolated Elasticsearch indexes for review regression tests."""

import os
from uuid import uuid4

import pytest


@pytest.fixture(scope="session")
def review_es_client():
    Elasticsearch = pytest.importorskip("elasticsearch").Elasticsearch
    test_url = os.environ.get("H2FLOW_TEST_ES_URL")
    with Elasticsearch(
        test_url or "http://localhost:9200", request_timeout=10, max_retries=0
    ) as client:
        if not client.ping():
            if test_url:
                pytest.fail("Configured Elasticsearch test instance is unreachable")
            pytest.skip("Set H2FLOW_TEST_ES_URL to run Elasticsearch review tests")
        yield client


@pytest.fixture
def review_index(review_es_client):
    from search.index import create_index

    index = f"textbook-pr2-review-{uuid4().hex}"
    create_index(review_es_client, index=index)
    try:
        yield review_es_client, index
    finally:
        review_es_client.indices.delete(index=index)
