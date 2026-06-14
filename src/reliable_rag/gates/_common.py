"""Helpers shared by the gates (kept separate to avoid circular imports)."""

from __future__ import annotations

from collections.abc import Sequence

from ..schemas import Chunk


def format_chunks_for_prompt(chunks: Sequence[Chunk], max_chars_each: int = 1000) -> str:
    """Render chunks for an LLM prompt: each labeled with its real ``[chunk_id]``
    and breadcrumb, so the model can cite by id and see document context.

    Shared by all gates so chunk presentation is identical everywhere.
    """
    blocks = []
    for c in chunks:
        text = c.text if len(c.text) <= max_chars_each else c.text[:max_chars_each] + " …"
        crumb = f" ({c.header_path})" if c.header_path else ""
        blocks.append(f"[{c.chunk_id}]{crumb}\n{text}")
    return "\n\n".join(blocks)
