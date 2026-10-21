"""Read an OpenStax osbooks bundle: which book and chapter each module is in.

A bundle (e.g. osbooks-calculus-bundle) keeps every module in one modules/
folder and describes its books in collections/<slug>.collection.xml. Without
that map, every module in the bundle would get the same book label.
"""

from __future__ import annotations

import glob
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from .parse_osbooks import CNXML_NS, parse_cnxml_file
from .schemas import Chunk

COL_NS = "http://cnx.rice.edu/collxml"
MD_NS = "http://cnx.rice.edu/mdml"


@dataclass
class Book:
    slug: str                 # e.g. "calculus-volume-1"
    title: str                # e.g. "Calculus Volume 1"
    modules: list[tuple[str, str]] = field(default_factory=list)  # (module_id, chapter)


@dataclass
class LoadReport:
    per_book: dict[str, dict[str, int]] = field(default_factory=dict)  # slug -> modules/chunks
    missing: list[str] = field(default_factory=list)   # listed in a collection, no file on disk
    failed: list[tuple[str, str]] = field(default_factory=list)  # (module_id, error)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _walk(content: ET.Element, path: list[str], out: list[tuple[str, str]]) -> None:
    """Collect (module_id, chapter) in book order. Nested subcollections join with ' > '."""
    for child in content:
        kind = _local(child.tag)
        if kind == "module":
            module_id = child.get("document")
            if module_id:
                out.append((module_id, " > ".join(path)))
        elif kind == "subcollection":
            title = (child.findtext(f"{{{MD_NS}}}title") or "").strip()
            inner = child.find(f"{{{COL_NS}}}content")
            if inner is not None:
                _walk(inner, (path + [title]) if title else path, out)


def parse_collection_string(xml_str: str, fallback_slug: str = "") -> Book:
    root = ET.fromstring(xml_str)
    title = slug = ""
    meta = root.find(f"{{{COL_NS}}}metadata")
    if meta is not None:
        title = (meta.findtext(f"{{{MD_NS}}}title") or "").strip()
        slug = (meta.findtext(f"{{{MD_NS}}}slug") or "").strip()
    book = Book(slug=slug or fallback_slug, title=title or slug or fallback_slug)
    content = root.find(f"{{{COL_NS}}}content")
    if content is not None:
        _walk(content, [], book.modules)
    return book


def read_bundle(bundle_root: str) -> list[Book]:
    """Every book described under <bundle_root>/collections/, sorted by file name."""
    books: list[Book] = []
    pattern = os.path.join(bundle_root, "collections", "*.collection.xml")
    for p in sorted(glob.glob(pattern)):
        fallback = os.path.basename(p)[: -len(".collection.xml")]
        with open(p, encoding="utf-8") as f:
            books.append(parse_collection_string(f.read(), fallback))
    return books


def _select_books(bundle_root: str, book_slug: str | None) -> list[Book]:
    books = read_bundle(bundle_root)
    if not books:
        raise ValueError(f"no collections found under {os.path.join(bundle_root, 'collections')}")
    if book_slug:
        wanted = [b for b in books if b.slug == book_slug]
        if not wanted:
            raise ValueError(
                f"no book '{book_slug}' in this bundle; available: " + ", ".join(b.slug for b in books)
            )
        return wanted
    return books


def module_title(bundle_root: str, module_id: str) -> str:
    """The <title> of modules/<id>/index.cnxml, for outlines."""
    path = os.path.join(bundle_root, "modules", module_id, "index.cnxml")
    if not os.path.isfile(path):
        return "(missing on disk)"
    try:
        title_el = ET.parse(path).getroot().find(f"{{{CNXML_NS}}}title")
    except ET.ParseError:
        return "(unreadable)"
    if title_el is None:
        return "(untitled)"
    return " ".join("".join(title_el.itertext()).split())


def outline_lines(bundle_root: str, book_slug: str | None = None) -> list[str]:
    """Book > chapter > module id + title, in book order. For writing test questions."""
    lines: list[str] = []
    for book in _select_books(bundle_root, book_slug):
        lines.append(f"{book.slug}  ({book.title})")
        current: str | None = None
        for module_id, chapter in book.modules:
            if chapter != current:
                lines.append(f"  {chapter or '(outside any chapter)'}")
                current = chapter
            lines.append(f"    {module_id}  {module_title(bundle_root, module_id)}")
        lines.append("")
    return lines


def collect_chunks(bundle_root: str, book_slug: str | None = None) -> tuple[list[Chunk], LoadReport]:
    """Parse the modules of one book (or all books) in a bundle, labelled correctly.

    A module shared by several books is parsed once, under the first book
    (by file name) that lists it. Modules that no collection lists are skipped.
    """
    books = _select_books(bundle_root, book_slug)

    chunks: list[Chunk] = []
    report = LoadReport()
    seen: set[str] = set()
    for book in books:
        stats = {"modules": 0, "chunks": 0}
        for module_id, chapter in book.modules:
            if module_id in seen:
                continue
            seen.add(module_id)
            path = os.path.join(bundle_root, "modules", module_id, "index.cnxml")
            if not os.path.isfile(path):
                report.missing.append(module_id)
                continue
            try:
                parsed = parse_cnxml_file(path, book.slug)
            except Exception as exc:  # one odd module must not abort a whole book
                report.failed.append((module_id, repr(exc)))
                continue
            for c in parsed:
                c.chapter = chapter
            chunks.extend(parsed)
            stats["modules"] += 1
            stats["chunks"] += len(parsed)
        report.per_book[book.slug] = stats
    return chunks, report