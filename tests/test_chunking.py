"""Parser behaviour: section labels, merging, lists, definitions, links, math tweaks."""

from search.chunk import MAX_CHARS
from search.parse_osbooks import parse_cnxml_file, parse_cnxml_string

DOC = (
    '<document xmlns="http://cnx.rice.edu/cnxml" xmlns:m="http://www.w3.org/1998/Math/MathML">'
    "<title>T</title><content>{body}</content></document>"
)


def parse(body: str):
    return parse_cnxml_string(DOC.format(body=body), "m1", "book")


def test_section_path_skips_proof_and_names_exercises():
    chunks = parse(
        "<section><title>Basics</title><section><title>Constant Rule</title><para>One.</para></section></section>"
        "<section><title>Power Rule</title><section><title>Proof</title><para>Two.</para>"
        "<example><exercise><problem><para>Differentiate.</para></problem>"
        "<solution><para>Answer.</para></solution></exercise></example></section></section>"
        '<section class="section-exercises"><exercise><problem><para>Q.</para></problem></exercise></section>'
    )
    assert [c.section for c in chunks] == ["Basics > Constant Rule", "Power Rule", "Exercises"]


def test_small_blocks_merge_within_a_section_only():
    paras = "".join(f"<para>Short paragraph {i}.</para>" for i in range(6))
    chunks = parse(f"<section><title>A</title>{paras}</section><section><title>B</title><para>Other.</para></section>")
    assert len(chunks) == 2
    assert all(f"Short paragraph {i}." in chunks[0].text for i in range(6))
    assert chunks[1].section == "B"


def test_long_block_is_split_into_bounded_chunks():
    chunks = parse("<para>" + "word " * 1200 + "</para>")
    assert len(chunks) > 1
    assert all(len(c.text) <= MAX_CHARS for c in chunks)


def test_text_only_list_items_are_kept_as_bullets():
    chunks = parse(
        "<note><title>Objectives</title><list><item>Identify the form.</item>"
        "<item>Explain the difference.</item></list></note>"
    )
    assert "- Identify the form. - Explain the difference." in chunks[0].text


def test_definition_keeps_its_term():
    chunks = parse(
        "<definition><term>power rule</term><meaning><para>The derivative of x^n.</para></meaning></definition>"
    )
    assert "power rule: The derivative of x^n." in chunks[0].text


def test_empty_link_names_what_it_points_at():
    chunks = parse(
        '<figure id="fig1"><caption>A graph.</caption></figure><para>See <link target-id="fig1"/> for details.</para>'
    )
    assert "See [Figure] for details." in chunks[0].text


def test_prime_is_a_suffix_not_an_exponent():
    prime = (
        "<para>so <m:math><m:mrow><m:msup><m:mi>f</m:mi><m:mo>\u2032</m:mo></m:msup>"
        "<m:mo>(</m:mo><m:mi>x</m:mi><m:mo>)</m:mo></m:mrow></m:math> is</para>"
    )
    text = parse(prime)[0].text
    assert "f\u2032(x)" in text and "^(" not in text


def test_math_in_a_title_is_converted():
    chunks = parse(
        "<example><title>Differentiating <m:math><m:mrow><m:msup><m:mi>x</m:mi><m:mn>3</m:mn></m:msup></m:mrow>"
        "</m:math></title><exercise><problem><para>Go.</para></problem></exercise></example>"
    )
    assert "Differentiating x^3." in chunks[0].text


def test_example_problem_and_solution_share_one_chunk():
    chunks = parse_cnxml_file("tests/fixtures/m53481.cnxml", "calculus-volume-1")
    assert any("Problem: Bacterial Growth" in c.text and "2828" in c.text for c in chunks)


def test_glossary_is_indexed():
    chunks = parse_cnxml_file("tests/fixtures/m53481.cnxml", "calculus-volume-1")
    glossary = [c for c in chunks if c.section == "Glossary"]
    assert glossary and "exponent: the value x in the expression b^x" in glossary[0].text