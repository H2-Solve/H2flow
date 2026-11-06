"""Character-window chunker and block packer used by the parser."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EquationSpan:
    text: str
    start: int
    end: int

    def shifted(self, offset: int) -> "EquationSpan":
        return EquationSpan(self.text, self.start + offset, self.end + offset)


TARGET_CHARS = 900   # merge small neighbouring blocks until a chunk is about this long
MAX_CHARS = 1500     # a single block longer than this is split into overlapping windows
OVERLAP = 100


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    text = " ".join(text.split())
    if len(text) <= size:
        return [text] if text else []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if c.strip()]


def pack_blocks(blocks):
    """Merge neighbouring blocks of the same section into chunks of ~TARGET_CHARS.

    blocks: (section, text, EquationSpan list) tuples in document order. A block is never
    cut here, so a worked example that fits stays whole; blocks longer than
    MAX_CHARS are split afterwards with chunk_text. Blocks are joined with a
    newline so paragraph boundaries stay visible to the LLM.
    """
    packed: list[tuple[str, str, list[EquationSpan]]] = []
    cur: list | None = None  # [section, text, equations]
    for section, text, eqs in blocks:
        leading = len(text) - len(text.lstrip())
        text = text.strip()
        if not text:
            continue
        eqs = [equation.shifted(-leading) for equation in eqs]
        if cur and cur[0] == section and len(cur[1]) + 1 + len(text) <= TARGET_CHARS:
            offset = len(cur[1]) + 1
            cur[1] += "\n" + text
            cur[2].extend(equation.shifted(offset) for equation in eqs)
        else:
            if cur:
                packed.append((cur[0], cur[1], cur[2]))
            cur = [section, text, eqs]
    if cur:
        packed.append((cur[0], cur[1], cur[2]))
    return packed
