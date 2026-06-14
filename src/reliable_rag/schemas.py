"""Data shapes for the whole pipeline — one place to see what flows where.

Two families live here:

1. **Domain models** (``Chunk``, ``RetrievedChunk``, ``ChunkGrade``,
   ``Verification``, ``Citation``, ``TraceEvent``, ``GateCost``) — the records
   that move through the graph and get logged.
2. **LLM structured-output schemas** (``GradeResponse``, ``VerifyResponse``) —
   the exact JSON shapes we force Claude to return via
   ``.with_structured_output(...)`` in the gates, so we never parse free text.

``RAGState`` at the bottom is the LangGraph state: every gate reads and writes
slices of it. ``trace``, ``flags`` and ``costs`` use an ``operator.add`` reducer
so each node *appends* rather than overwrites.
"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

ChunkType = Literal["prose", "table", "diagram", "api", "csv_row"]


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------
class Chunk(BaseModel):
    """One indexed unit of the corpus, with everything Trace needs to cite it."""

    chunk_id: str                       # stable, e.g. "E2E_Controller::Topology>Polarity::2"
    text: str                           # RAW chunk text — what we display & cite
    source: str                         # source file, e.g. "docs/E2E_Controller.md"
    header_path: str = ""               # breadcrumb, e.g. "Topology Management > Polarity"
    chunk_type: ChunkType = "prose"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def context_header(self) -> str:
        """Document-level context prepended before embedding (the book's
        "contextual chunk headers"). CSV rows already self-describe, so they
        get no header. Returns "" when there is nothing useful to prepend."""
        if self.chunk_type == "csv_row":
            return ""
        stem = Path(self.source).stem
        crumb = stem + (f" > {self.header_path}" if self.header_path else "")
        return f"Document: {crumb}"

    @property
    def embed_text(self) -> str:
        """The text actually embedded / BM25-indexed: contextual header + raw
        text. Keeping this separate from ``text`` means citations show the real
        content while retrieval benefits from the added context."""
        header = self.context_header
        return f"{header}\n\n{self.text}" if header else self.text


class RetrievedChunk(Chunk):
    """A chunk returned by retrieval, carrying its relevance score."""

    score: float = 0.0  # retriever score (higher = more relevant)


class ChunkGrade(BaseModel):
    """The GRADE gate's verdict on a single retrieved chunk."""

    chunk_id: str
    relevant: bool
    score: float = Field(ge=0.0, le=1.0)  # 0..1 relevance to the query
    reason: str = ""


class ClaimCheck(BaseModel):
    """One claim from the answer, checked against the retrieved chunks."""

    claim: str
    grounded: bool
    chunk_ids: list[str] = Field(default_factory=list)  # chunks that support it
    reason: str = ""


class Verification(BaseModel):
    """The VERIFY gate's overall assessment of the draft answer."""

    faithful: bool
    score: float = Field(ge=0.0, le=1.0)  # fraction of claims grounded
    claims: list[ClaimCheck] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    """A grounded citation — resolved from a real retrieved chunk, never invented."""

    chunk_id: str
    source: str
    header_path: str = ""
    snippet: str = ""


class TraceEvent(BaseModel):
    """One stage's entry in the per-query trace (what it did + how long)."""

    stage: str
    summary: str = ""
    duration_ms: float = 0.0
    data: dict[str, Any] = Field(default_factory=dict)


class GateCost(BaseModel):
    """Token + dollar cost attributed to one gate's LLM/embedding call(s)."""

    stage: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    usd: float = 0.0


# ---------------------------------------------------------------------------
# LLM structured-output schemas (forced via .with_structured_output)
# ---------------------------------------------------------------------------
class _ChunkRelevance(BaseModel):
    chunk_id: str = Field(description="The exact chunk_id being graded.")
    relevant: bool = Field(description="True if this chunk helps answer the question.")
    score: float = Field(ge=0.0, le=1.0, description="Relevance 0.0-1.0.")
    reason: str = Field(description="One short sentence justifying the score.")


class GradeResponse(BaseModel):
    """GRADE gate output: one verdict per retrieved chunk."""

    grades: list[_ChunkRelevance]


class _ClaimVerdict(BaseModel):
    claim: str = Field(description="A single atomic factual claim from the answer.")
    grounded: bool = Field(description="True if the retrieved context supports it.")
    supporting_chunk_ids: list[str] = Field(
        default_factory=list, description="chunk_ids that support the claim (may be empty)."
    )
    reason: str = Field(description="One short sentence of justification.")


class VerifyResponse(BaseModel):
    """VERIFY gate output: the answer decomposed into claims, each checked."""

    claims: list[_ClaimVerdict]


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class RAGState(TypedDict, total=False):
    """The state threaded through the LangGraph pipeline.

    ``total=False`` so nodes can return partial updates. ``trace``/``flags``/
    ``costs`` are reducer channels (``operator.add``) — returning a list appends
    to the running list instead of replacing it.
    """

    query: str
    rewritten_query: Optional[str]
    retrieved: list[RetrievedChunk]
    grades: list[ChunkGrade]
    kept_chunks: list[RetrievedChunk]
    context_chunks: list[RetrievedChunk]  # kept chunks + ±N neighbors (what Generate/Verify see)
    draft_answer: str
    answer: str
    citations: list[Citation]
    verification: Optional[Verification]
    attempt: int
    flags: Annotated[list[str], operator.add]
    trace: Annotated[list[TraceEvent], operator.add]
    costs: Annotated[list[GateCost], operator.add]
