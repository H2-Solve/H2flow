"""Decimal math regression checks; set H2FLOW_TEST_ES_URL for Elasticsearch tests."""

import os
from uuid import uuid4

import pytest

from search.parse_osbooks import parse_cnxml_string


def parse_math(mathml: str, module_id: str = "m-decimal"):
    xml = (
        '<document xmlns="http://cnx.rice.edu/cnxml" '
        'xmlns:m="http://www.w3.org/1998/Math/MathML">'
        f"<title>Decimal math</title><content><para><m:math>{mathml}</m:math>"
        "</para></content></document>"
    )
    return parse_cnxml_string(xml, module_id, "test-book")[0]


@pytest.mark.parametrize(
    ("exponent", "expected"), [("2.5", "x^2.5"), ("3.14", "x^3.14")]
)
def test_parser_preserves_decimal_exponents(exponent, expected):
    chunk = parse_math(f"<m:msup><m:mi>x</m:mi><m:mn>{exponent}</m:mn></m:msup>")

    assert chunk.text == expected
    assert chunk.equations == [expected]


@pytest.mark.parametrize(
    ("base", "expected"), [("2.5", "log_2.5(x)"), ("3.14", "log_3.14(x)")]
)
def test_parser_preserves_decimal_logarithm_bases(base, expected):
    chunk = parse_math(
        f"<m:mrow><m:msub><m:mi>log</m:mi><m:mn>{base}</m:mn></m:msub>"
        "<m:mo>(</m:mo><m:mi>x</m:mi><m:mo>)</m:mo></m:mrow>"
    )

    assert chunk.text == expected
    assert chunk.equations == [expected]


@pytest.fixture(scope="module")
def math_index():
    Elasticsearch = pytest.importorskip("elasticsearch").Elasticsearch
    from search.index import bulk_index, create_index

    test_url = os.environ.get("H2FLOW_TEST_ES_URL")
    client = Elasticsearch(
        test_url or "http://localhost:9200", request_timeout=2, max_retries=0
    )
    try:
        if not client.ping():
            if test_url:
                pytest.fail("Configured Elasticsearch test instance is unreachable")
            pytest.skip(
                "Elasticsearch is unavailable; set H2FLOW_TEST_ES_URL to run search tests"
            )

        index = f"textbook-decimal-test-{uuid4().hex}"
        create_index(client, index=index)
        try:
            chunks = [
                parse_math(
                    f"<m:msup><m:mi>x</m:mi><m:mn>{exponent}</m:mn></m:msup>", module_id
                )
                for module_id, exponent in [
                    ("m-square", "2"),
                    ("m-cube", "3"),
                    ("m-power-25", "2.5"),
                    ("m-power-314", "3.14"),
                ]
            ]
            chunks.extend(
                parse_math(
                    f"<m:mrow><m:msub><m:mi>log</m:mi><m:mn>{base}</m:mn></m:msub>"
                    "<m:mo>(</m:mo><m:mi>x</m:mi><m:mo>)</m:mo></m:mrow>",
                    module_id,
                )
                for module_id, base in [("m-log-25", "2.5"), ("m-log-314", "3.14")]
            )
            bulk_index(client, chunks, index=index)
            client.indices.refresh(index=index)
            yield client, index
        finally:
            client.indices.delete(index=index)
    finally:
        client.close()


@pytest.mark.parametrize(("formula", "value"), [("x^2.5", "2.5"), ("x^3.14", "3.14")])
def test_elasticsearch_preserves_the_complete_decimal_exponent(
    math_index, formula, value
):
    client, index = math_index

    result = client.indices.analyze(index=index, analyzer="math_text", text=formula)
    tokens = [token["token"] for token in result["tokens"]]

    assert value in tokens, tokens


@pytest.mark.parametrize(
    ("formula", "value"), [("log_2.5(x)", "2.5"), ("log_3.14(x)", "3.14")]
)
def test_elasticsearch_preserves_the_complete_decimal_log_base(
    math_index, formula, value
):
    client, index = math_index

    result = client.indices.analyze(index=index, analyzer="math_text", text=formula)
    tokens = [token["token"] for token in result["tokens"]]

    assert any(value in token for token in tokens), tokens


@pytest.mark.parametrize(
    ("query", "module_id"), [("2.5", "m-power-25"), ("3.14", "m-power-314")]
)
def test_decimal_value_query_finds_its_exponent(math_index, query, module_id):
    from search.retrieve import get_documents

    client, index = math_index
    hits = get_documents(query, top_k=10, index=index, es=client)

    assert module_id in {hit["module_id"] for hit in hits}, hits


@pytest.mark.parametrize(
    ("query", "module_id"), [("squared", "m-square"), ("cubed", "m-cube")]
)
def test_integer_power_words_do_not_match_decimal_exponents(
    math_index, query, module_id
):
    from search.retrieve import get_documents

    client, index = math_index
    hits = get_documents(query, top_k=10, index=index, es=client)

    assert {hit["module_id"] for hit in hits} == {module_id}


@pytest.mark.parametrize(
    ("query", "module_id", "expected_text"),
    [
        ("x^2.5", "m-power-25", "x^2.5"),
        ("x^3.14", "m-power-314", "x^3.14"),
        ("log_2.5", "m-log-25", "log_2.5(x)"),
        ("log_3.14", "m-log-314", "log_3.14(x)"),
    ],
)
def test_literal_formula_search_returns_original_text(
    math_index, query, module_id, expected_text
):
    from search.retrieve import get_documents

    client, index = math_index
    hits = get_documents(query, top_k=10, index=index, es=client)

    assert hits[0]["module_id"] == module_id
    assert hits[0]["text"] == expected_text
