"""Tests for collection-aware loading. No Elasticsearch needed."""

import os
import shutil

import pytest

from search.bundle import collect_chunks, parse_collection_string, read_bundle
from search.index import build_actions

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

MINI_COLLECTION = """<col:collection xmlns="http://cnx.rice.edu/collxml" xmlns:md="http://cnx.rice.edu/mdml" xmlns:col="http://cnx.rice.edu/collxml">
  <metadata mdml-version="0.5"><md:title>Mini Book</md:title><md:slug>mini</md:slug></metadata>
  <col:content>
    <col:subcollection>
      <md:title>Unit 1</md:title>
      <col:content>
        <col:subcollection>
          <md:title>Chapter A</md:title>
          <col:content><col:module document="m53472" /><col:module document="m00000" /></col:content>
        </col:subcollection>
      </col:content>
    </col:subcollection>
    <col:module document="m53481" />
  </col:content>
</col:collection>"""


def test_parse_real_collection_file():
    with open(os.path.join(FIXTURES, "calculus-volume-1.collection.xml"), encoding="utf-8") as f:
        book = parse_collection_string(f.read())
    assert book.slug == "calculus-volume-1"
    assert book.title == "Calculus Volume 1"
    chapters = dict(book.modules)
    assert chapters["m53472"] == "Functions and Graphs"
    assert chapters["m53481"] == "Functions and Graphs"
    assert chapters["m53483"] == "Limits"
    assert chapters["m60027"] == ""  # front matter sits outside any chapter
    assert len(book.modules) == 55


@pytest.fixture
def mini_bundle(tmp_path):
    (tmp_path / "collections").mkdir()
    (tmp_path / "collections" / "mini.collection.xml").write_text(MINI_COLLECTION, encoding="utf-8")
    for module_id in ("m53472", "m53481", "m99999"):  # m99999 is in no collection
        d = tmp_path / "modules" / module_id
        d.mkdir(parents=True)
        shutil.copy(os.path.join(FIXTURES, f"{'m53481' if module_id == 'm99999' else module_id}.cnxml"),
                    d / "index.cnxml")
    return str(tmp_path)


def test_collect_chunks_labels_nests_and_skips(mini_bundle):
    chunks, report = collect_chunks(mini_bundle)
    assert [b.slug for b in read_bundle(mini_bundle)] == ["mini"]
    assert {c.book_id for c in chunks} == {"mini"}
    assert {c.module_id for c in chunks} == {"m53472", "m53481"}  # m99999 skipped
    assert {c.chapter for c in chunks if c.module_id == "m53472"} == {"Unit 1 > Chapter A"}
    assert {c.chapter for c in chunks if c.module_id == "m53481"} == {""}
    assert report.missing == ["m00000"]          # listed, but no file on disk
    assert report.per_book["mini"]["modules"] == 2


def test_unknown_book_lists_available(mini_bundle):
    with pytest.raises(ValueError, match="available: mini"):
        collect_chunks(mini_bundle, "nope")


def test_ids_are_stable_and_unique(mini_bundle):
    chunks, _ = collect_chunks(mini_bundle)
    ids = [a["_id"] for a in build_actions(chunks)]
    assert len(ids) == len(set(ids))
    assert ids == [a["_id"] for a in build_actions(chunks)]  # same on a second run
    assert "m53472:0" in ids