"""Behavioral checks for PR #2; failing cases document unfixed limitations."""

from collections import Counter

import pytest

from search.bundle import collect_chunks
from search.evaluate import first_rank
from search.parse_osbooks import parse_cnxml_string


def document(body, title="Review"):
    return (
        '<document xmlns="http://cnx.rice.edu/cnxml" '
        'xmlns:m="http://www.w3.org/1998/Math/MathML">'
        f"<title>{title}</title><content>{body}</content></document>"
    )


@pytest.mark.parametrize(
    ("mathml", "expected"),
    [
        (
            "<m:msup><m:mi>e</m:mi><m:mrow><m:mn>2</m:mn><m:mi>x</m:mi></m:mrow></m:msup>",
            "e^(2x)",
        ),
        (
            "<m:msup><m:mrow><m:mi>x</m:mi><m:mi>y</m:mi></m:mrow><m:mn>2</m:mn></m:msup>",
            "(xy)^2",
        ),
        (
            "<m:mfrac><m:mn>1</m:mn><m:mrow><m:mi>x</m:mi><m:mi>y</m:mi></m:mrow></m:mfrac>",
            "1/(xy)",
        ),
        ("<m:msup><m:mi>xy</m:mi><m:mn>2</m:mn></m:msup>", "xy^2"),
        (
            "<m:mrow><m:msub><m:mi>log</m:mi>"
            "<m:mrow><m:mn>2</m:mn><m:mi>x</m:mi></m:mrow></m:msub>"
            "<m:mo>(</m:mo><m:mn>8</m:mn><m:mo>)</m:mo></m:mrow>",
            "log_(2x)(8)",
        ),
        (
            "<m:msubsup><m:mi>a</m:mi>"
            "<m:mrow><m:mi>i</m:mi><m:mi>j</m:mi></m:mrow>"
            "<m:mrow><m:mn>2</m:mn><m:mi>x</m:mi></m:mrow></m:msubsup>",
            "a_(ij)^(2x)",
        ),
        (
            "<m:mfrac><m:mrow><m:mi>x</m:mi><m:mi>y</m:mi></m:mrow><m:mn>2</m:mn></m:mfrac>",
            "(xy)/2",
        ),
        (
            "<m:msup><m:mrow><m:mi>x</m:mi></m:mrow>"
            "<m:mrow><m:mrow><m:mn>2.5</m:mn></m:mrow></m:mrow></m:msup>",
            "x^2.5",
        ),
        (
            "<m:msup><m:msup><m:mi>x</m:mi><m:mn>2</m:mn></m:msup><m:mn>3</m:mn></m:msup>",
            "(x^2)^3",
        ),
        (
            "<m:msup><m:mrow><m:mo>(</m:mo><m:mi>x</m:mi><m:mi>y</m:mi>"
            "<m:mo>)</m:mo></m:mrow><m:mn>2</m:mn></m:msup>",
            "(xy)^2",
        ),
        (
            "<m:msup><m:mrow><m:mi>f</m:mi><m:mi>g</m:mi></m:mrow><m:mo>′</m:mo></m:msup>",
            "(fg)′",
        ),
    ],
    ids=[
        "compound-exponent",
        "product-base",
        "product-denominator",
        "atomic-identifier-control",
        "compound-log-base",
        "combined-subscript-exponent",
        "product-numerator",
        "single-child-rows-preserve-decimal",
        "nested-power",
        "already-parenthesized",
        "product-prime",
    ],
)
def test_mathml_operand_grouping_survives_parsing(mathml, expected):
    chunks = parse_cnxml_string(
        document(f"<para><m:math>{mathml}</m:math></para>"), "m1", "book"
    )

    assert chunks[0].text == expected
    assert chunks[0].equations == [expected]


def test_math_in_module_title_is_preserved():
    title = (
        "Derivative of <m:math><m:msup><m:mi>x</m:mi><m:mn>3</m:mn></m:msup></m:math>"
    )
    chunks = parse_cnxml_string(
        document("<para>Explanation.</para>", title), "m1", "book"
    )

    assert chunks[0].title == "Derivative of x^3"


def test_split_chunks_only_claim_equations_in_their_own_text():
    body = (
        "<example><exercise><problem><para><m:math><m:mi>alpha</m:mi></m:math> "
        + "context " * 400
        + "</para></problem><solution><para><m:math><m:mi>omega</m:mi></m:math>"
        "</para></solution></exercise></example>"
    )
    chunks = parse_cnxml_string(document(body), "m1", "book")

    assert len(chunks) > 1
    assert all(
        equation in chunk.text for chunk in chunks for equation in chunk.equations
    )


def write_bundle(root, modules):
    (root / "collections").mkdir()
    refs = "".join(f'<module document="{module_id}"/>' for module_id in modules)
    (root / "collections" / "book.collection.xml").write_text(
        '<collection xmlns="http://cnx.rice.edu/collxml" xmlns:md="http://cnx.rice.edu/mdml">'
        f"<metadata><md:slug>book</md:slug></metadata><content>{refs}</content></collection>",
        encoding="utf-8",
    )
    for module_id, xml in modules.items():
        directory = root / "modules" / module_id
        directory.mkdir(parents=True)
        (directory / "index.cnxml").write_text(xml, encoding="utf-8")


def test_bad_module_is_reported_without_losing_the_good_module(tmp_path):
    write_bundle(
        tmp_path,
        {"m-good": document("<para>A useful explanation.</para>"), "m-bad": "<broken>"},
    )

    chunks, report = collect_chunks(str(tmp_path))

    assert {chunk.module_id for chunk in chunks} == {"m-good"}
    assert [module_id for module_id, error in report.failed] == ["m-bad"]
    assert report.per_book["book"]["modules"] == 1


def test_module_level_evaluation_does_not_establish_answer_relevance():
    unrelated_hits = [
        {"module_id": "m53575", "text": "Unrelated passage without a derivative rule."}
    ]

    assert first_rank(unrelated_hits, ["m53575"]) == 1


def test_module_metric_can_reject_an_answer_from_another_book():
    valid_alternative = [
        {
            "module_id": "m53902",
            "book_id": "calculus-volume-3",
            "text": "For a constant force in the direction of motion, work is W=Fd.",
        }
    ]

    assert first_rank(valid_alternative, ["m53648"]) is None


def test_single_load_keeps_all_distinct_modules_searchable(review_index):
    from search.index import bulk_index
    from search.retrieve import get_documents
    from search.schemas import Chunk

    client, index = review_index
    chunks = [
        Chunk("book", f"m{i}", "Title", text=f"quasar{i} explanation") for i in range(3)
    ]
    bulk_index(client, chunks, index=index)
    client.indices.refresh(index=index)

    assert client.count(index=index)["count"] == 3
    for i in range(3):
        assert (
            get_documents(f"quasar{i}", index=index, es=client)[0]["module_id"]
            == f"m{i}"
        )


def test_clean_rebuild_removes_chunks_from_an_old_parser(review_index):
    from search.index import bulk_index, create_index
    from search.retrieve import get_documents
    from search.schemas import Chunk

    client, index = review_index
    bulk_index(
        client,
        [Chunk("book", "m1", "T", text=f"obsolete{i}") for i in range(5)],
        index=index,
    )
    create_index(client, index=index, reset=True)
    bulk_index(
        client,
        [Chunk("book", "m1", "T", text=f"current{i}") for i in range(2)],
        index=index,
    )
    client.indices.refresh(index=index)

    assert client.count(index=index)["count"] == 2
    assert get_documents("obsolete4", index=index, es=client) == []


def test_identical_reload_does_not_duplicate_documents(review_index):
    from search.index import bulk_index
    from search.schemas import Chunk

    client, index = review_index
    chunks = [Chunk("book", "m1", "T", text=f"content{i}") for i in range(2)]
    bulk_index(client, chunks, index=index)
    bulk_index(client, chunks, index=index)
    client.indices.refresh(index=index)

    assert client.count(index=index)["count"] == 2


def test_conflicting_shared_module_versions_do_not_erase_distinct_content(review_index):
    from search.index import bulk_index
    from search.retrieve import get_documents
    from search.schemas import Chunk

    client, index = review_index
    bulk_index(
        client,
        [Chunk("book-a", "m-shared", "T", text="quasaralpha explanation")],
        index=index,
    )
    bulk_index(
        client,
        [Chunk("book-b", "m-shared", "T", text="quasarbeta explanation")],
        index=index,
    )
    client.indices.refresh(index=index)

    assert get_documents("quasaralpha", index=index, es=client), (
        "First book's distinct content was overwritten"
    )
    assert get_documents("quasarbeta", index=index, es=client)


def test_different_ids_can_occupy_multiple_result_slots_with_identical_text(
    review_index,
):
    from search.index import bulk_index
    from search.retrieve import get_documents
    from search.schemas import Chunk

    client, index = review_index
    bulk_index(
        client,
        [
            Chunk("book", f"m{i}", "T", text="quasarduplicate identical explanation")
            for i in range(2)
        ],
        index=index,
    )
    client.indices.refresh(index=index)
    hits = get_documents("quasarduplicate", index=index, es=client)

    assert (
        Counter(hit["text"] for hit in hits)["quasarduplicate identical explanation"]
        == 2
    )
