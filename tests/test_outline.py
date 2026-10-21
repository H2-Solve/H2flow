"""Outline of a bundle: chapter > module id + title. No Elasticsearch needed."""

import os
import shutil

from search.bundle import outline_lines

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

COLLECTION = """<col:collection xmlns="http://cnx.rice.edu/collxml" xmlns:md="http://cnx.rice.edu/mdml" xmlns:col="http://cnx.rice.edu/collxml">
  <metadata mdml-version="0.5"><md:title>Mini Book</md:title><md:slug>mini</md:slug></metadata>
  <col:content>
    <col:module document="m53472" />
    <col:subcollection>
      <md:title>Functions</md:title>
      <col:content><col:module document="m53481" /><col:module document="m00000" /></col:content>
    </col:subcollection>
  </col:content>
</col:collection>"""


def make_bundle(tmp_path):
    (tmp_path / "collections").mkdir()
    (tmp_path / "collections" / "mini.collection.xml").write_text(COLLECTION, encoding="utf-8")
    for module_id in ("m53472", "m53481"):
        d = tmp_path / "modules" / module_id
        d.mkdir(parents=True)
        shutil.copy(os.path.join(FIXTURES, f"{module_id}.cnxml"), d / "index.cnxml")
    return str(tmp_path)


def test_outline_groups_modules_by_chapter_with_titles(tmp_path):
    lines = outline_lines(make_bundle(tmp_path))
    assert lines[0] == "mini  (Mini Book)"
    assert lines[1] == "  (outside any chapter)"
    assert lines[2] == "    m53472  Introduction"
    assert lines[3] == "  Functions"
    assert lines[4] == "    m53481  Exponential and Logarithmic Functions"
    assert lines[5] == "    m00000  (missing on disk)"