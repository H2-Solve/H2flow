"""Benchmark problem sets, loaded into one shape: Problem(id, problem, answer_latex, meta).

Gold answers are always LaTeX so a single grader (Math-Verify) handles every set.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / ".cache"  # gitignored; downloaded datasets live here
DATA_DIR = HERE / "data"     # committed datasets

MATH500_URL = "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl"


@dataclass
class Problem:
    id: str
    problem: str
    answer_latex: str              # gold answer as LaTeX, without $ delimiters
    meta: dict = field(default_factory=dict)  # subject/level/topic/expected_operation, ...


@dataclass
class Dataset:
    name: str
    problems: list[Problem]
    source: str
    sha256: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_math500(path: str | Path | None = None, download: bool = True) -> Dataset:
    """MATH-500 (HuggingFaceH4). Downloaded once into benchmarking/.cache/ unless a path is given."""
    p = Path(path) if path else CACHE_DIR / "math500_test.jsonl"
    if not p.exists():
        if not download:
            raise FileNotFoundError(f"{p} not found (download disabled)")
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading MATH-500 from {MATH500_URL}")
        tmp = p.with_suffix(".part")
        urllib.request.urlretrieve(MATH500_URL, tmp)
        tmp.replace(p)
    problems = [
        Problem(
            id=row["unique_id"],
            problem=row["problem"],
            answer_latex=row["answer"],
            meta={"subject": row.get("subject"), "level": row.get("level")},
        )
        for row in _read_jsonl(p)
    ]
    return Dataset("math500", problems, source=str(path or MATH500_URL), sha256=_sha256(p))


def load_jsonl_dataset(name: str, path: str | Path) -> Dataset:
    """Our own sets: one JSON object per line with id, problem, answer_latex, and any metadata."""
    p = Path(path)
    problems = []
    for row in _read_jsonl(p):
        row = dict(row)
        problems.append(Problem(
            id=row.pop("id"), problem=row.pop("problem"), answer_latex=row.pop("answer_latex"), meta=row,
        ))
    return Dataset(name, problems, source=str(p), sha256=_sha256(p))


DATASETS = {
    "math500": lambda: load_math500(),
    "calculus": lambda: load_jsonl_dataset("calculus", DATA_DIR / "calculus_v1.jsonl"),
}


def load_dataset(name_or_path: str) -> Dataset:
    """A registered name (math500, calculus) or a path to a JSONL file in our format."""
    if name_or_path in DATASETS:
        return DATASETS[name_or_path]()
    p = Path(name_or_path)
    if p.suffix == ".jsonl" and p.exists():
        return load_jsonl_dataset(p.stem, p)
    raise SystemExit(f"unknown dataset '{name_or_path}'; use one of {sorted(DATASETS)} or a .jsonl path")
