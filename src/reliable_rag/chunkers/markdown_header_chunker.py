"""Header-aware markdown chunker — chunk boundaries follow the document's structure.

Why not fixed-size windows? These docs encode meaning in their heading hierarchy
and in atomic blocks (markdown tables, ASCII diagrams in code fences, discrete
REST endpoints). A blind 512/1000-char window would slice a diagram in half,
tear table rows from their header row, merge unrelated sibling sections, and drop
the section title that gives a chunk its topic — all of which degrade the Grade
gate and the citations. So we split on headers and keep atomic blocks whole.

Strategy
--------
1. Walk the file, tracking fenced code blocks (so we never treat a ``#`` *inside*
   a fence as a heading) and a stack of headings to build the breadcrumb
   ``header_path`` (e.g. ``"Topology Management > Polarity"``).
2. One section (the text under a heading) becomes one chunk — unless it exceeds
   ``max_chars``, in which case a **block-aware** size guard splits it without
   ever breaking a code fence or a markdown table.

This module is intentionally dependency-free so it can be unit-tested offline.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..schemas import Chunk, ChunkType

HEADER_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:\-|]+\|[\s:\-|]*$")  # the |---|:--:| row
API_RE = re.compile(r"\b(GET|POST|PUT|DELETE|PATCH)\b\s+`?/")


# ---------------------------------------------------------------------------
# Section splitting (by heading, fence-aware)
# ---------------------------------------------------------------------------
def iter_sections(text: str, max_level: int = 2):
    """Yield ``(header_path, level, body)`` for each heading-delimited section.

    Only headings of level ``<= max_level`` (``#``/``##`` by default) are split
    points; deeper headings (``###`` …) are left *in* the section body so a
    procedure that uses many sub-headings stays in one coherent chunk. The body
    excludes its own heading line (which lives in ``header_path``); content
    before the first heading is yielded with an empty path so a preamble is never
    lost.
    """
    lines = text.splitlines()
    stack: list[tuple[int, str]] = []  # (level, title), shallow -> deep
    buf: list[str] = []
    in_fence = False
    fence_marker = ""
    cur_level = 0

    def path() -> str:
        return " > ".join(title for _, title in stack)

    for line in lines:
        mf = FENCE_RE.match(line)
        if mf:
            marker = mf.group(1)
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif line.lstrip().startswith(fence_marker):
                in_fence, fence_marker = False, ""
            buf.append(line)
            continue

        if not in_fence:
            mh = HEADER_RE.match(line)
            if mh and len(mh.group(1)) <= max_level:
                body = "\n".join(buf).strip("\n")
                if body.strip():
                    yield path(), cur_level, body
                buf = []
                level = len(mh.group(1))
                title = mh.group(2).strip()
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                cur_level = level
                continue
            # headings deeper than max_level are kept as content (not split points)

        buf.append(line)

    body = "\n".join(buf).strip("\n")
    if body.strip():
        yield path(), cur_level, body


# ---------------------------------------------------------------------------
# Block-aware size guard (keeps tables & fences intact)
# ---------------------------------------------------------------------------
def to_blocks(body: str) -> list[str]:
    """Break a section body into atomic blocks: each fenced code block and each
    contiguous table is one block; everything else splits on blank lines."""
    lines = body.split("\n")
    blocks: list[str] = []
    buf: list[str] = []

    def flush():
        nonlocal buf
        text = "\n".join(buf).strip("\n")
        if text.strip():
            for para in re.split(r"\n\s*\n", text):
                if para.strip():
                    blocks.append(para.rstrip())
        buf = []

    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        mf = FENCE_RE.match(line)
        if mf:  # absorb the whole fenced block as one atomic unit
            flush()
            marker = mf.group(1)
            fence = [line]
            i += 1
            while i < n:
                fence.append(lines[i])
                closed = lines[i].lstrip().startswith(marker)
                i += 1
                if closed:
                    break
            blocks.append("\n".join(fence))
            continue
        if TABLE_ROW_RE.match(line):  # absorb a contiguous table as one unit
            flush()
            tbl = []
            while i < n and TABLE_ROW_RE.match(lines[i]):
                tbl.append(lines[i])
                i += 1
            blocks.append("\n".join(tbl))
            continue
        buf.append(line)
        i += 1
    flush()
    return blocks


def _char_split(text: str, max_chars: int, overlap: int) -> list[str]:
    """Last-resort character split with overlap (only for a single block that is
    itself larger than ``max_chars`` and isn't a table)."""
    pieces, start, n = [], 0, len(text)
    step = max(1, max_chars - overlap)
    while start < n:
        pieces.append(text[start : start + max_chars])
        start += step
    return pieces


def _split_table(block: str, max_chars: int) -> list[str]:
    """Split an oversized table by rows, repeating the header (+ separator) row
    on every piece so each piece stays self-describing."""
    rows = block.split("\n")
    head = [rows[0]]
    body_start = 1
    if len(rows) > 1 and TABLE_SEP_RE.match(rows[1]):
        head.append(rows[1])
        body_start = 2
    head_text = "\n".join(head)
    pieces: list[str] = []
    cur, cur_len = list(head), len(head_text)
    for r in rows[body_start:]:
        if cur_len + len(r) + 1 > max_chars and len(cur) > len(head):
            pieces.append("\n".join(cur))
            cur, cur_len = list(head), len(head_text)
        cur.append(r)
        cur_len += len(r) + 1
    if len(cur) > len(head):
        pieces.append("\n".join(cur))
    return pieces or [block]


def split_section_body(body: str, max_chars: int, overlap: int) -> list[str]:
    """Return one piece if the section fits, else block-packed pieces."""
    if len(body) <= max_chars:
        return [body]
    pieces: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for blk in to_blocks(body):
        if len(blk) > max_chars:  # a single block is too big on its own
            if cur:
                pieces.append("\n\n".join(cur))
                cur, cur_len = [], 0
            if TABLE_ROW_RE.match(blk.split("\n", 1)[0]):
                pieces.extend(_split_table(blk, max_chars))
            else:
                pieces.extend(_char_split(blk, max_chars, overlap))
            continue
        extra = len(blk) + (2 if cur else 0)
        if cur and cur_len + extra > max_chars:
            pieces.append("\n\n".join(cur))
            cur, cur_len = [], 0
        cur.append(blk)
        cur_len += extra
    if cur:
        pieces.append("\n\n".join(cur))
    return pieces


# ---------------------------------------------------------------------------
# Classification + ids
# ---------------------------------------------------------------------------
def classify(body: str) -> ChunkType:
    """Best-effort content type — informational, drives display/filtering."""
    lines = body.split("\n")
    if any(TABLE_SEP_RE.match(l) for l in lines) or sum(bool(TABLE_ROW_RE.match(l)) for l in lines) >= 2:
        return "table"
    if API_RE.search(body):
        return "api"
    if any(FENCE_RE.match(l) for l in lines):
        return "diagram"
    return "prose"


def slugify(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s or "root"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def chunk_markdown_file(
    path: str, max_chars: int = 1200, overlap: int = 100, max_level: int = 2
) -> list[Chunk]:
    """Chunk one markdown file into ``Chunk`` objects with stable ids + breadcrumbs."""
    text = Path(path).read_text(encoding="utf-8")
    source = Path(path).as_posix()
    stem = Path(path).stem

    chunks: list[Chunk] = []
    gidx = 0  # running index makes chunk_ids unique even when sections repeat titles
    for header_path, level, body in iter_sections(text, max_level=max_level):
        ctype = classify(body)
        for piece in split_section_body(body, max_chars, overlap):
            chunk_id = f"{stem}::{slugify(header_path or 'root')}#{gidx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=piece,
                    source=source,
                    header_path=header_path,
                    chunk_type=ctype,
                    metadata={"level": level},
                )
            )
            gidx += 1
    return chunks
