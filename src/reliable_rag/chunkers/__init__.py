"""Structure-aware chunking for the markdown corpus.

The corpus is markdown, so it gets one structure-aware strategy:

* **Markdown** -> :func:`chunk_markdown_file` — header-aware splitting that keeps
  tables and ASCII diagrams whole and records the section breadcrumb (used both
  as the citation and as the "contextual chunk header" prepended at embed time).

It returns ``list[Chunk]`` with a uniform metadata contract, so everything
downstream (index, retrieval, trace) treats chunks identically. See ``CHUNKING.md``.

CSV/tabular sources are intentionally **out of scope** — an exact table lookup is
a structured query best served by a metadata filter, not semantic search.
"""

from __future__ import annotations

from pathlib import Path

from ..schemas import Chunk
from .markdown_header_chunker import chunk_markdown_file


def chunk_path(path: str, max_chars: int = 1200, overlap: int = 100, max_level: int = 2) -> list[Chunk]:
    """Dispatch a single file to the right chunker by extension.

    Non-markdown files (e.g. ``.csv``) are out of scope and yield no chunks.
    """
    ext = Path(path).suffix.lower()
    if ext in {".md", ".markdown"}:
        return chunk_markdown_file(path, max_chars=max_chars, overlap=overlap, max_level=max_level)
    return []


__all__ = ["chunk_path", "chunk_markdown_file"]
