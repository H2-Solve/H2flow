"""Unified chunk schema shared by both textbook parsers.

Both sources use the same module IDs (e.g. m53472), so module_id is the
dedupe key when indexing from osbooks + philschatz.
"""

from dataclasses import asdict, dataclass, field


@dataclass
class Chunk:
    book_id: str      # e.g. "calculus-volume-1"
    module_id: str    # e.g. "m53472"
    title: str        # module title
    section: str = "" # section title within the module, may be empty
    text: str = ""    # plain text, 300-800 chars after chunking
    equations: list[str] = field(default_factory=list)  # MathML-linearized strings found in this chunk
    source: str = "osbooks"  # "osbooks" | "philschatz"

    def to_es_doc(self) -> dict:
        d = asdict(self)
        d["excerpt"] = d["text"][:500]
        return d
