"""Math in titles and complete equations associated with each text window."""

import pytest

from search.parse_osbooks import parse_cnxml_string


def parse(body, title="Metadata"):
    return parse_cnxml_string(
        '<document xmlns="http://cnx.rice.edu/cnxml" '
        'xmlns:m="http://www.w3.org/1998/Math/MathML">'
        f"<title>{title}</title><content>{body}</content></document>",
        "m-metadata", "book",
    )


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        (
            "<m:math><m:msup><m:mi>x</m:mi><m:mn>2</m:mn></m:msup></m:math> and its derivative",
            "x^2 and its derivative",
        ),
        (
            "Growth of <m:math><m:msup><m:mi>e</m:mi><m:mrow>"
            "<m:mn>2</m:mn><m:mi>x</m:mi></m:mrow></m:msup></m:math> over time",
            "Growth of e^(2x) over time",
        ),
    ],
)
def test_module_titles_preserve_inline_math_and_following_text(title, expected):
    chunks = parse("<para>Explanation.</para>", title)

    assert chunks[0].title == expected


def test_partial_equations_are_not_claimed_as_complete_window_equations():
    formula = "x+" * 800 + "1"
    chunks = parse(f"<para><m:math><m:mtext>{formula}</m:mtext></m:math></para>")

    assert len(chunks) > 1
    assert all(formula not in chunk.text for chunk in chunks)
    assert all(chunk.equations == [] for chunk in chunks)


def test_equations_inside_the_overlap_belong_to_both_windows():
    body = "<para>" + "padding " * 180
    body += "<m:math><m:msup><m:mi>x</m:mi><m:mn>2</m:mn></m:msup></m:math>"
    body += " words" * 50 + "</para>"
    chunks = parse(body)

    assert len(chunks) == 2
    assert all("x^2" in chunk.text for chunk in chunks)
    assert all(chunk.equations == ["x^2"] for chunk in chunks)


def test_equation_prefix_in_another_formula_is_not_a_source_occurrence():
    body = "<para><m:math><m:msup><m:mi>x</m:mi><m:mn>20</m:mn></m:msup></m:math>"
    body += " padding" * 200
    body += " <m:math><m:msup><m:mi>x</m:mi><m:mn>2</m:mn></m:msup></m:math></para>"
    chunks = parse(body)

    assert chunks[0].equations == ["x^20"]
    assert chunks[-1].equations == ["x^2"]


def test_plain_text_matching_a_later_equation_is_not_math_metadata():
    body = "<para>The string x^2 is mentioned here." + " padding" * 200
    body += " <m:math><m:msup><m:mi>x</m:mi><m:mn>2</m:mn></m:msup></m:math></para>"
    chunks = parse(body)

    assert "x^2" in chunks[0].text
    assert chunks[0].equations == []
    assert chunks[-1].equations == ["x^2"]


def test_repeated_equations_keep_their_separate_source_occurrences():
    equation = "<m:math><m:msup><m:mi>x</m:mi><m:mn>2</m:mn></m:msup></m:math>"
    chunks = parse("<para>" + equation + " padding" * 200 + " " + equation + "</para>")

    assert chunks[0].equations == ["x^2"]
    assert chunks[-1].equations == ["x^2"]
