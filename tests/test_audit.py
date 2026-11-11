"""Bundle audit: finds what the parser skips or does not recognise. No Elasticsearch needed."""

import os
import xml.etree.ElementTree as ET

from search.audit import audit_bundle, format_report

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
NS_DOC = (
    '<document xmlns="http://cnx.rice.edu/cnxml" xmlns:m="http://www.w3.org/1998/Math/MathML">'
    "<title>T</title><content>{body}</content></document>"
)
COLLECTION = """<col:collection xmlns="http://cnx.rice.edu/collxml" xmlns:md="http://cnx.rice.edu/mdml" xmlns:col="http://cnx.rice.edu/collxml">
  <metadata mdml-version="0.5"><md:title>Audit Book</md:title><md:slug>audit-book</md:slug></metadata>
  <col:content>{modules}</col:content></col:collection>"""


def make_bundle(tmp_path, bodies: dict[str, str], extra_listed=()):
    listed = "".join(f'<col:module document="{m}" />' for m in [*bodies, *extra_listed])
    (tmp_path / "collections").mkdir()
    (tmp_path / "collections" / "audit-book.collection.xml").write_text(
        COLLECTION.format(modules=listed), encoding="utf-8")
    for module_id, body in bodies.items():
        d = tmp_path / "modules" / module_id
        d.mkdir(parents=True)
        (d / "index.cnxml").write_text(NS_DOC.format(body=body), encoding="utf-8")
    return str(tmp_path)


def test_audit_flags_skipped_and_unrecognised_content(tmp_path):
    bundle = make_bundle(
        tmp_path,
        {
            "m1": '<para>Volume is 5 ft<sup>3</sup>.</para><table id="t"><tgroup><row><entry>x</entry></row></tgroup></table>',
            "m2": "<para><m:math><m:munderover><m:mo>\u2211</m:mo><m:mi>i</m:mi><m:mi>n</m:mi></m:munderover></m:math></para>",
            "m3": "",  # a module with no text blocks
        },
        extra_listed=["m4"],  # listed in the collection but not on disk
    )
    report = audit_bundle(bundle)
    assert report["missing"] == ["m4"] and report["zero_chunk"] == ["m3"]
    assert report["modules"] == 3 and not report["failed"]
    assert report["cnxml"]["table"] == 1 and report["modules_with"]["table"] == {"m1"}
    assert report["cnxml"]["sup"] == 1 and report["where"]["cnxml"]["sup"] == ["m1"]
    assert report["math"]["munderover"] == 1
    text = "\n".join(format_report(report))
    assert "table: 1 in 1 modules" in text
    assert "sup" in text and "munderover" in text
    assert "tgroup" not in text, "elements inside a skipped table are not reported"


def test_audit_on_a_real_fixture_counts_its_tables(tmp_path):
    bundle = make_bundle(tmp_path, {})
    d = tmp_path / "modules" / "m53481"
    d.mkdir(parents=True)
    with open(os.path.join(FIXTURES, "m53481.cnxml"), encoding="utf-8") as f:
        xml = f.read()
    (d / "index.cnxml").write_text(xml, encoding="utf-8")
    (tmp_path / "collections" / "audit-book.collection.xml").write_text(
        COLLECTION.format(modules='<col:module document="m53481" />'), encoding="utf-8")
    expected = sum(1 for e in ET.fromstring(xml).iter("{http://cnx.rice.edu/cnxml}table"))
    report = audit_bundle(bundle)
    assert report["cnxml"]["table"] == expected
    assert report["modules"] == 1 and report["chunk_lengths"]