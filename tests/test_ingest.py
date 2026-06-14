"""Chunker invariants — these run fully offline (no API keys)."""

from __future__ import annotations

from reliable_rag.chunkers import chunk_path
from reliable_rag.chunkers.markdown_header_chunker import (
    chunk_markdown_file,
    iter_sections,
    split_section_body,
    to_blocks,
)


def test_splits_only_on_h1_h2(tmp_path):
    md = "# Top\n\nintro\n\n## Sub A\n\nbody a\n\n### Deep\n\ndeep body\n"
    p = tmp_path / "Doc.md"
    p.write_text(md, encoding="utf-8")
    chunks = chunk_markdown_file(str(p))

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))  # ids are unique

    paths = {c.header_path for c in chunks}
    assert "Top" in paths
    assert "Top > Sub A" in paths
    # '###' is NOT a split point -> no deeper section is created
    assert "Top > Sub A > Deep" not in paths

    # the '### Deep' heading and its text fold into the parent '## Sub A' chunk
    sub = next(c for c in chunks if c.header_path == "Top > Sub A")
    assert "### Deep" in sub.text and "deep body" in sub.text


def test_fence_hash_is_not_a_heading():
    md = "# Real\n\n```\n# not a heading\n```\n\ntext"
    sections = list(iter_sections(md))
    # Everything stays under the single real heading; the '#' inside the fence
    # must not start a new section.
    assert all(path == "Real" for path, _level, _body in sections)


def test_to_blocks_keeps_table_and_fence_atomic():
    body = (
        "intro paragraph\n\n"
        "| h1 | h2 |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n\n"
        "```\nascii\ndiagram\n```\n\n"
        "outro paragraph"
    )
    blocks = to_blocks(body)
    assert any(b.startswith("| h1") and "| 3 | 4 |" in b for b in blocks)  # whole table = one block
    assert any(b.startswith("```") and "diagram" in b for b in blocks)      # whole fence = one block


def test_size_guard_keeps_table_rows_with_header():
    table = "\n".join(["| a | b |", "|---|---|"] + [f"| {i} | row{i} |" for i in range(40)])
    body = "some intro\n\n" + table
    pieces = split_section_body(body, max_chars=120, overlap=10)
    assert len(pieces) > 1  # it did split
    # every piece that contains table rows also carries the separator header row
    for pc in pieces:
        if "| row" in pc:
            assert "|---|" in pc


def test_contextual_header_in_embed_text(tmp_path):
    md = "# Topic\n\n## Section\n\nthe body text"
    p = tmp_path / "MyDoc.md"
    p.write_text(md, encoding="utf-8")
    chunks = chunk_markdown_file(str(p))
    chunk = next(c for c in chunks if c.header_path == "Topic > Section")
    assert chunk.embed_text.startswith("Document: MyDoc > Topic > Section")
    assert "the body text" in chunk.embed_text
    assert chunk.text == "the body text"  # raw text is untouched (used for citation)


def test_chunk_path_markdown_only(tmp_path):
    # Markdown is chunked; non-markdown (CSV is out of scope) yields nothing.
    (tmp_path / "a.md").write_text("# H\n\nbody", encoding="utf-8")
    (tmp_path / "b.csv").write_text("x,summary\n1,hello", encoding="utf-8")
    md_chunks = chunk_path(str(tmp_path / "a.md"))
    assert md_chunks and md_chunks[0].header_path == "H"
    assert chunk_path(str(tmp_path / "b.csv")) == []
