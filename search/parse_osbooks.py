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

from .chunk import chunk_text
from .schemas import Chunk

CNXML_NS = "http://cnx.rice.edu/cnxml"
MATH_NS = "http://www.w3.org/1998/Math/MathML"
MD_NS = "http://cnx.rice.edu/mdml"


def _local(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _linearize_math(math_el: ET.Element) -> str:
    """Flatten MathML to readable tokens, e.g. P(1)=P0(1.02)."""
    return "".join(math_el.itertext()).strip()


def _block_text(el: ET.Element) -> tuple[str, list[str]]:
    """Plain text of a <para>/<equation>/<caption> block + equations found inside."""
    equations: list[str] = []
    parts: list[str] = []
    for node in el.iter():
        if _local(node.tag) == "math" and node.tag.startswith("{" + MATH_NS):
            eq = _linearize_math(node)
            if eq:
                equations.append(eq)
                parts.append(f" [MATH: {eq}] ")
        elif _local(node.tag) in {"link", "term", "emphasis", "image"}:
            continue  # text handled via itertext at block level; skip double-count
    text = "".join(el.itertext())
    return " ".join(text.split()), equations


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
                for sub in child.iter():
                    if _local(sub.tag) in {"para", "equation"}:
                        text, eqs = _block_text(sub)
                        if text:
                            blocks.append((section, text, eqs))
                        break
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
