"""GRADE gate — score retrieved chunks for relevance BEFORE generation.

Post 1's first gate. Vector similarity returns the *nearest* chunks, not
necessarily the *relevant* ones: a chunk can be on-topic yet useless for the
actual question (the post's "Chunk 2 ... Vector similarity loved it. The judge
scored it low."). Feeding such chunks to the generator poisons the answer, so we
let an LLM judge each chunk and keep only those scoring >= ``GRADE_THRESHOLD``.

All chunks are graded in a single structured call to keep latency/cost down.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import AIMessage

from ..config import settings
from ..models import chat
from ..schemas import ChunkGrade, GradeResponse, RetrievedChunk
from ._common import format_chunks_for_prompt

GRADE_SYSTEM = (
    "You are a strict relevance grader for a retrieval system over Cambium cnWave "
    "60 GHz wireless-mesh technical documentation. For EACH retrieved chunk, decide "
    "how much it helps answer the user's question and give a score from 0.0 to 1.0.\n"
    "- 1.0: directly answers or is essential to answering.\n"
    "- 0.5: related and somewhat useful.\n"
    "- 0.0: on-topic but does not help answer, or irrelevant.\n"
    "Be strict: irrelevant context poisons generation. Echo each chunk_id exactly "
    "as given and return one entry per chunk."
)


def grade_chunks(
    query: str, chunks: Sequence[RetrievedChunk]
) -> tuple[list[ChunkGrade], AIMessage | None]:
    """Grade every chunk; ``relevant`` is decided by ``GRADE_THRESHOLD`` on the score.

    Returns ``(grades, raw_message)``. Any chunk the model fails to return is
    treated as dropped (``relevant=False``) — silence never keeps a chunk.
    """
    if not chunks:
        return [], None

    llm = chat(settings.grade_model, max_tokens=1536).with_structured_output(
        GradeResponse, include_raw=True
    )
    user = (
        f"Question:\n{query}\n\n"
        f"Retrieved chunks to grade:\n\n{format_chunks_for_prompt(chunks)}\n\n"
        "Grade every chunk above."
    )
    try:
        result = llm.invoke([("system", GRADE_SYSTEM), ("human", user)])
        raw: AIMessage | None = result.get("raw")
        parsed: GradeResponse | None = result.get("parsed")
    except Exception:
        raw, parsed = None, None

    if parsed is None:
        # Grader unavailable / unparseable — fail OPEN (keep all chunks) rather
        # than drop everything and abstain on a transient glitch.
        grades = [
            ChunkGrade(chunk_id=c.chunk_id, relevant=True, score=1.0, reason="grader unavailable; kept by default")
            for c in chunks
        ]
        return grades, raw

    scored: dict[str, ChunkGrade] = {}
    for g in parsed.grades:
        scored[g.chunk_id] = ChunkGrade(
            chunk_id=g.chunk_id,
            relevant=g.score >= settings.grade_threshold,  # threshold is the source of truth
            score=g.score,
            reason=g.reason,
        )

    grades: list[ChunkGrade] = []
    for c in chunks:
        grades.append(
            scored.get(
                c.chunk_id,
                ChunkGrade(chunk_id=c.chunk_id, relevant=False, score=0.0, reason="not returned by grader"),
            )
        )
    return grades, raw
