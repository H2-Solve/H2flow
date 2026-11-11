"""Audit a bundle before loading it: what will the parser handle, skip or misread?

  python -m search.audit --bundle C:\\path\\to\\osbooks-college-algebra-bundle
  python -m search.audit --bundle ... --book precalculus-2e --top 25

Reports load health (modules, failures, chunks), content skipped on purpose (tables, media), and
elements the parser does not recognise, in the book XML and in the math. Nothing is written to
Elasticsearch. Keep KNOWN_CNXML / KNOWN_MATHML in step with parse_osbooks.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

from .bundle import _select_books
from .parse_osbooks import CNXML_NS, MATH_NS, parse_cnxml_string

# Book-XML elements the parser handles, or whose text it reads correctly as plain text.
# "sub" and "sup" are deliberately absent: as plain text they lose their meaning
# (ft<sup>3</sup> becomes "ft3"), so they should show up in the report.
KNOWN_CNXML = {
    "document", "title", "content", "section", "para", "equation", "figure", "caption", "list", "item",
    "example", "exercise", "problem", "solution", "note", "definition", "rule", "term", "meaning",
    "glossary", "commentary", "statement", "proof", "label", "emphasis", "link", "newline", "space",
    "cite", "quote", "code", "foreign", "span", "name", "media", "image", "table", "colspec",
}
# Math elements the converter handles; anything else is just concatenated.
KNOWN_MATHML = {
    "math", "mrow", "mi", "mn", "mo", "mtext", "mspace", "msup", "msub", "msubsup",
    "mfrac", "msqrt", "mroot", "mtr", "mtd", "mtable",
}
SUBTREES_NOT_READ = ("media", "table")  # the parser skips everything inside these


def _split(tag: str) -> tuple[str, str]:
    if tag.startswith("{"):
        ns, _, name = tag[1:].partition("}")
        return ns, name
    return "", tag


def inventory(root: ET.Element) -> tuple[Counter, Counter]:
    """Element-name counts under <content> and <glossary>: (book XML, MathML)."""
    cnxml: Counter = Counter()
    math: Counter = Counter()

    def walk(el: ET.Element) -> None:
        for child in el:
            ns, name = _split(child.tag)
            if ns == MATH_NS:
                math[name] += 1
                walk(child)
            elif ns == CNXML_NS:
                cnxml[name] += 1
                if name not in SUBTREES_NOT_READ:
                    walk(child)

    for part in ("content", "glossary"):
        el = root.find(f"{{{CNXML_NS}}}{part}")
        if el is not None:
            walk(el)
    return cnxml, math


def audit_bundle(bundle_root: str, book_slug: str | None = None) -> dict:
    books = _select_books(bundle_root, book_slug)
    report: dict = {
        "root": bundle_root, "books": [], "missing": [], "failed": [], "zero_chunk": [],
        "modules": 0, "chunk_lengths": [], "cnxml": Counter(), "math": Counter(),
        "where": {"cnxml": defaultdict(list), "math": defaultdict(list)},
        "modules_with": defaultdict(set),
    }
    seen: set[str] = set()
    for book in books:
        listed = [m for m, _ in book.modules]
        shared = sum(1 for m in set(listed) if m in seen)
        report["books"].append({"slug": book.slug, "title": book.title, "listed": len(set(listed)), "shared": shared})
        for module_id in listed:
            if module_id in seen:
                continue
            seen.add(module_id)
            path = os.path.join(bundle_root, "modules", module_id, "index.cnxml")
            if not os.path.isfile(path):
                report["missing"].append(module_id)
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    xml_str = f.read()
                cnxml, math = inventory(ET.fromstring(xml_str))
                chunks = parse_cnxml_string(xml_str, module_id, book.slug)
            except Exception as exc:  # report it and keep auditing
                report["failed"].append((module_id, repr(exc)[:140]))
                continue
            report["modules"] += 1
            if not chunks:
                report["zero_chunk"].append(module_id)
            report["chunk_lengths"].extend(len(c.text) for c in chunks)
            for kind, counts in (("cnxml", cnxml), ("math", math)):
                report[kind].update(counts)
                for name in counts:
                    report["where"][kind][name].append(module_id)
            for name in SUBTREES_NOT_READ:
                if cnxml.get(name):
                    report["modules_with"][name].add(module_id)
    return report


def format_report(report: dict, top: int = 15) -> list[str]:
    lens = report["chunk_lengths"]
    lines = [f"AUDIT  {report['root']}", "", "Books"]
    for b in report["books"]:
        lines.append(f"  {b['slug']:<42} {b['listed']:>4} modules ({b['shared']} also in earlier books)")
    lines += [
        "",
        f"Modules parsed: {report['modules']}   missing on disk: {len(report['missing'])}   "
        f"failed: {len(report['failed'])}   produced no chunks: {len(report['zero_chunk'])}",
        f"Chunks: {len(lens)}" + (f"   average {sum(lens) // len(lens)} chars, longest {max(lens)}" if lens else ""),
    ]
    for module_id, err in report["failed"][:5]:
        lines.append(f"  FAILED {module_id}: {err}")
    if report["zero_chunk"]:
        lines.append("  no chunks: " + ", ".join(report["zero_chunk"][:10]))
    lines += ["", "Skipped on purpose (their text is not indexed)"]
    for name in SUBTREES_NOT_READ:
        lines.append(f"  {name}: {report['cnxml'].get(name, 0)} in {len(report['modules_with'][name])} modules")
    for kind, label, known in (("cnxml", "book XML", KNOWN_CNXML), ("math", "math (MathML)", KNOWN_MATHML)):
        unknown = [(n, c) for n, c in report[kind].most_common() if n not in known]
        lines += ["", f"Unrecognised {label} elements (read as plain text; may lose meaning)"]
        if not unknown:
            lines.append("  none")
        for name, count in unknown[:top]:
            mods = list(dict.fromkeys(report["where"][kind][name]))
            lines.append(f"  {name:<16} {count:>6}  in {len(mods):>3} modules  e.g. {', '.join(mods[:3])}")
    return lines


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--book", default=None, help="audit one book of the bundle (default: all)")
    ap.add_argument("--top", type=int, default=15, help="how many unrecognised elements to list")
    args = ap.parse_args()
    try:
        report = audit_bundle(args.bundle, args.book)
    except ValueError as exc:
        raise SystemExit(str(exc))
    print("\n".join(format_report(report, args.top)))


if __name__ == "__main__":
    main()