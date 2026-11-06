"""Presentation wrappers must preserve numeric operands and searchable powers."""

import pytest

from search.parse_osbooks import parse_cnxml_string


def parse_formula(expression, module_id="m-styled"):
    xml = (
        '<document xmlns="http://cnx.rice.edu/cnxml" '
        'xmlns:m="http://www.w3.org/1998/Math/MathML">'
        f"<title>Styled math</title><content><para><m:math>{expression}</m:math>"
        "</para></content></document>"
    )
    return parse_cnxml_string(xml, module_id, "book")[0]


@pytest.mark.parametrize("wrapper", ["mstyle", "mpadded"])
@pytest.mark.parametrize("exponent", ["2", "3", "2.5"])
def test_presentation_wrappers_keep_numeric_exponents_atomic(wrapper, exponent):
    chunk = parse_formula(
        f"<m:msup><m:mi>x</m:mi><m:{wrapper}><m:mrow>"
        f"<m:mn>{exponent}</m:mn></m:mrow></m:{wrapper}></m:msup>"
    )

    assert chunk.text == f"x^{exponent}"
    assert chunk.equations == [f"x^{exponent}"]


def test_styled_compound_exponent_stays_grouped():
    chunk = parse_formula(
        "<m:msup><m:mi>e</m:mi><m:mstyle><m:mrow>"
        "<m:mn>2</m:mn><m:mi>x</m:mi></m:mrow></m:mstyle></m:msup>"
    )

    assert chunk.text == "e^(2x)"
    assert chunk.equations == ["e^(2x)"]


def test_styled_fraction_keeps_implicit_multiplication_unambiguous():
    chunk = parse_formula(
        "<m:mrow><m:mstyle><m:mfrac><m:mn>1</m:mn><m:mn>2</m:mn>"
        "</m:mfrac></m:mstyle><m:mi>x</m:mi></m:mrow>"
    )

    assert chunk.text == "(1/2)x"


@pytest.mark.parametrize(("exponent", "query"), [("2", "squared"), ("3", "cubed")])
def test_word_queries_find_styled_integer_powers(review_index, exponent, query):
    from search.index import bulk_index
    from search.retrieve import get_documents

    client, index = review_index
    chunk = parse_formula(
        f"<m:msup><m:mi>x</m:mi><m:mstyle><m:mn>{exponent}</m:mn>"
        "</m:mstyle></m:msup>"
    )
    bulk_index(client, [chunk], index=index)
    client.indices.refresh(index=index)

    hits = get_documents(query, index=index, es=client)

    assert [hit["module_id"] for hit in hits] == ["m-styled"]
    assert hits[0]["text"] == f"x^{exponent}"
