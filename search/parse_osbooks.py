"""Parser for openstax/osbooks-calculus-bundle CNXML.

Source layout:
  modules/<module_id>/index.cnxml   e.g. modules/m53472/index.cnxml
  collections/<book>.collection.xml  book order / titles
  media/                             images (skipped, captions kept)

CNXML namespaces:
  default http://cnx.rice.edu/cnxml  -> document/title/content/para/section/equation/figure
  m http://www.w3.org/1998/Math/MathML -> math (linearized to text)
  md http://cnx.rice.edu/mdml        -> metadata (content-id, title, uuid)

Output: list[Chunk] with source="osbooks".
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
import re

from .chunk import chunk_text
from .schemas import Chunk

CNXML_NS = "http://cnx.rice.edu/cnxml"
MATH_NS = "http://www.w3.org/1998/Math/MathML"
MD_NS = "http://cnx.rice.edu/mdml"


def _local(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


# def _linearize_math(math_el: ET.Element) -> str:
#     """Flatten MathML to readable tokens, e.g. P(1)=P0(1.02)."""
#     return "".join(math_el.itertext()).strip()


# def _block_text(el: ET.Element) -> tuple[str, list[str]]:
#     """Plain text of a <para>/<equation>/<caption> block + equations found inside."""
#     equations: list[str] = []
#     parts: list[str] = []
#     for node in el.iter():
#         if _local(node.tag) == "math" and node.tag.startswith("{" + MATH_NS):
#             eq = _linearize_math(node)
#             if eq:
#                 equations.append(eq)
#                 parts.append(f" [MATH: {eq}] ")
#         elif _local(node.tag) in {"link", "term", "emphasis", "image"}:
#             continue  # text handled via itertext at block level; skip double-count
#     text = "".join(el.itertext())
#     return " ".join(text.split()), equations

MATH_TAG = "{" + MATH_NS + "}math"
_SIMPLE = re.compile(r"^[A-Za-z0-9.]+$")


def _is_wrapped(s: str) -> bool:
    """True if s is one parenthesised group, e.g. "(t/4)" but not "(a)+(b)"."""
    if not (s.startswith("(") and s.endswith(")")):
        return False
    depth = 0
    for i, ch in enumerate(s):
        depth += (ch == "(") - (ch == ")")
        if depth == 0 and i < len(s) - 1:
            return False
    return True


def _wrap(s: str) -> str:
    """Parenthesise unless s is a plain word/number or already one group."""
    return s if _SIMPLE.match(s) or _is_wrapped(s) else f"({s})"


def _is_frac(el: ET.Element) -> bool:
    """A fraction, possibly wrapped in single-child <mrow>s."""
    while _local(el.tag) == "mrow" and len(el) == 1:
        el = el[0]
    return _local(el.tag) == "mfrac"


def _math_children(el: ET.Element) -> list[str]:
    return [_math_to_text(c) for c in el]


def _math_to_text(el: ET.Element) -> str:
    """Convert MathML to linear text, e.g. 2^(t/4), log_3(81), sqrt(x), (a)/(b).

    Tokens (mi/mn/mo/mtext) become their text; sup/sub/frac/roots become
    ^ _ / sqrt() root(); unknown elements just concatenate their children.
    """
    name = _local(el.tag)
    if name in {"mi", "mn", "mo", "mtext"}:
        return "".join(el.itertext())
    if name == "mspace":
        return " "
    kids = _math_children(el)
    if name == "msup" and len(kids) == 2:
        return f"{_wrap(kids[0])}^{_wrap(kids[1])}"
    if name == "msub" and len(kids) == 2:
        return f"{_wrap(kids[0])}_{_wrap(kids[1])}"
    if name == "msubsup" and len(kids) == 3:
        return f"{_wrap(kids[0])}_{_wrap(kids[1])}^{_wrap(kids[2])}"
    if name == "mfrac" and len(kids) == 2:
        return f"{_wrap(kids[0])}/{_wrap(kids[1])}"
    if name == "msqrt":
        return f"sqrt({''.join(kids)})"
    if name == "mroot" and len(kids) == 2:
        return f"root({kids[0]}, {kids[1]})"
    if name == "mtr":
        return " ".join(k for k in kids if k.strip())
    if name == "mtable":
        return "; ".join(k for k in kids if k.strip())
    if name == "mrow":
        out: list[str] = []
        for i, child in enumerate(el):
            text = kids[i]
            if _is_frac(child):
                prev = out[-1] if out else ""
                nxt = kids[i + 1] if i + 1 < len(kids) else ""
                # (1/2)ln(x), not 1/2ln(x): keep implicit multiplication unambiguous
                if nxt[:1].isalnum() or nxt[:1] == "(" or prev[-1:].isalnum() or prev[-1:] == ")":
                    text = f"({text})"
            out.append(text)
        return "".join(out)
    return "".join(kids)  # mtd, math, and anything unrecognised


def _linearize_math(math_el: ET.Element) -> str:
    """Faithful linear text for one <math> element, e.g. n(6)=1000·2^(6/4)≈2828."""
    return " ".join(_math_to_text(math_el).split())


def _text_with_math(el: ET.Element) -> str:
    """Like "".join(el.itertext()) but each <math> is converted, not flattened."""
    parts = [el.text or ""]
    for child in el:
        if child.tag == MATH_TAG:
            parts.append(_linearize_math(child))
        else:
            parts.append(_text_with_math(child))
        parts.append(child.tail or "")
    return "".join(parts)


def _block_text(el: ET.Element) -> tuple[str, list[str]]:
    """Plain text of a <para>/<equation>/<caption> block + equations found inside."""
    equations = [eq for m in el.iter(MATH_TAG) if (eq := _linearize_math(m))]
    return " ".join(_text_with_math(el).split()), equations

def _container_text(el: ET.Element) -> tuple[str, list[str]]:
    """Text of a whole example/exercise/list/note as ONE block.

    Walks every descendant in document order so a problem and its solution
    stay together, e.g. "Bacterial Growth. Problem: ... Solution: ...".
    """
    parts: list[str] = []
    equations: list[str] = []

    def walk(node: ET.Element) -> None:
        for child in node:
            name = _local(child.tag)
            if name in {"para", "equation"}:
                text, eqs = _block_text(child)
                if text:
                    parts.append(text)
                    equations.extend(eqs)
            elif name == "title":
                title = " ".join("".join(child.itertext()).split())
                if title:
                    parts.append(title + ".")
            elif name == "caption":
                text, eqs = _block_text(child)
                if text:
                    parts.append(f"[Figure] {text}")
                    equations.extend(eqs)
            elif name in {"problem", "solution"}:
                parts.append(name.capitalize() + ":")
                walk(child)
            elif len(child):
                walk(child)  # exercise, item, note, list, etc.

    walk(el)
    return " ".join(parts), equations

def parse_cnxml_string(xml_str: str, module_id: str, book_id: str, source: str = "osbooks") -> list[Chunk]:
    root = ET.fromstring(xml_str)
    ns = {"c": CNXML_NS, "md": MD_NS}

    title_el = root.find("c:title", ns)
    title = (title_el.text or "").strip() if title_el is not None else module_id
    meta_id_el = root.find("c:metadata/md:content-id", ns)
    if meta_id_el is not None and (meta_id_el.text or "").strip():
        module_id = meta_id_el.text.strip()

    content_el = root.find("c:content", ns)
    if content_el is None:
        return []

    # Walk content preserving section context: (section_title, block_text, equations)
    blocks: list[tuple[str, str, list[str]]] = []

    def visit(parent: ET.Element, section: str) -> None:
        for child in parent:
            name = _local(child.tag)
            if name == "title" and _local(parent.tag) == "section":
                section = "".join(child.itertext()).strip() or section
            elif name == "section":
                sec_title_el = child.find("c:title", ns)
                sec_title = "".join(sec_title_el.itertext()).strip() if sec_title_el is not None else section
                visit(child, sec_title)
            elif name in {"para", "equation"}:
                text, eqs = _block_text(child)
                if text:
                    blocks.append((section, text, eqs))
            elif name == "figure":
                for cap in child.findall("c:caption", ns):
                    text, eqs = _block_text(cap)
                    if text:
                        blocks.append((section, f"[Figure] {text}", eqs))
            elif name in {"list", "example", "exercise", "note"}:
                text, eqs = _container_text(child)
                if text:
                    blocks.append((section, text, eqs))
            else:
                # Recurse into unknown containers (e.g. content root)
                if len(child):
                    visit(child, section)

    visit(content_el, "")

    chunks: list[Chunk] = []
    for section, text, eqs in blocks:
        for piece in chunk_text(text):
            chunks.append(
                Chunk(
                    book_id=book_id,
                    module_id=module_id,
                    title=title,
                    section=section,
                    text=piece,
                    equations=eqs,
                    source=source,
                )
            )
    return chunks


def parse_cnxml_file(path: str, book_id: str) -> list[Chunk]:
    module_id = os.path.basename(os.path.dirname(path))
    with open(path, encoding="utf-8") as f:
        return parse_cnxml_string(f.read(), module_id, book_id)


def iter_module_files(bundle_root: str) -> list[str]:
    """All modules/<id>/index.cnxml under a cloned osbooks bundle."""
    out: list[str] = []
    modules_dir = os.path.join(bundle_root, "modules")
    if not os.path.isdir(modules_dir):
        return out
    for module_id in sorted(os.listdir(modules_dir)):
        p = os.path.join(modules_dir, module_id, "index.cnxml")
        if os.path.isfile(p):
            out.append(p)
    return out