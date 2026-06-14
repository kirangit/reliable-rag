"""The Reliable-RAG pipeline as a LangGraph ``StateGraph``.

```
            ┌───────────── rewrite_query (once, if too few chunks pass Grade)
            ▼                                          │
  retrieve ─► grade ──(enough kept?)── yes ─► generate ─► verify ──(faithful?)── yes ─► finalize ─► END
   (hybrid)   (Sonnet)        │                  ▲                         │
                              │ no               └──── regenerate ◄────── no (unsupported claims,
                              ▼                                             attempt < MAX_ATTEMPTS)
                       abstain ─► finalize
```

Each node does one job, measures its own latency, and appends a ``TraceEvent``
and a ``GateCost`` to the (reducer) state channels. The conditional edges encode
the reliability policy: rewrite-and-retry once on thin retrieval, abstain rather
than hallucinate, and regenerate once when Verify finds unsupported claims.
"""

from __future__ import annotations

import time
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from .config import settings
from .cost import estimate_embed_cost, gate_cost
from .gates import generate_answer, grade_chunks, verify_answer
from .gates.generate import _content_to_text
from .models import chat
from .retrieval import get_retriever
from .schemas import RAGState, TraceEvent, Verification
from .trace import extract_citations

ABSTAIN_MESSAGE = (
    "I don't have enough grounded information in the cnWave documentation to "
    "answer that confidently."
)

REWRITE_SYSTEM = (
    "Rewrite the user's question to maximise retrieval recall over Cambium cnWave "
    "60 GHz wireless-mesh technical documentation. Expand acronyms and add likely "
    "keywords/synonyms (model names, protocols, feature names). Return ONLY the "
    "rewritten query as a single line."
)


def _effective_query(state: RAGState) -> str:
    return state.get("rewritten_query") or state["query"]


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def retrieve_node(state: RAGState) -> dict:
    query = _effective_query(state)
    t0 = time.perf_counter()
    chunks = get_retriever().retrieve(query, state.get("k"))
    event = TraceEvent(
        stage="retrieve",
        summary=f"{len(chunks)} chunks ({settings.retrieval_mode})",
        duration_ms=_ms(t0),
        data={"query": query, "scores": {c.chunk_id: round(c.score, 3) for c in chunks}},
    )
    return {
        "retrieved": chunks,
        "trace": [event],
        "costs": [estimate_embed_cost(query, settings.embedding_model)],
    }


def grade_node(state: RAGState) -> dict:
    query = _effective_query(state)
    retrieved = state.get("retrieved", [])
    t0 = time.perf_counter()
    grades, raw = grade_chunks(query, retrieved)
    kept_ids = {g.chunk_id for g in grades if g.relevant}
    kept = [c for c in retrieved if c.chunk_id in kept_ids]
    event = TraceEvent(
        stage="grade",
        summary=f"kept {len(kept)}/{len(retrieved)} (threshold {settings.grade_threshold})",
        duration_ms=_ms(t0),
        data={"decisions": [{"chunk_id": g.chunk_id, "score": g.score, "kept": g.relevant} for g in grades]},
    )
    return {
        "grades": grades,
        "kept_chunks": kept,
        "trace": [event],
        "costs": [gate_cost("grade", settings.grade_model, raw)],
    }


def rewrite_node(state: RAGState) -> dict:
    query = state["query"]
    t0 = time.perf_counter()
    msg = chat(settings.generation_model, max_tokens=200).invoke(
        [("system", REWRITE_SYSTEM), ("human", query)]
    )
    new_query = _content_to_text(msg.content).strip() or query
    event = TraceEvent(
        stage="rewrite",
        summary="rewrote query and will re-retrieve",
        duration_ms=_ms(t0),
        data={"from": query, "to": new_query},
    )
    return {
        "rewritten_query": new_query,
        "trace": [event],
        "costs": [gate_cost("rewrite", settings.generation_model, msg)],
        "flags": ["rewrote_query"],
    }


def generate_node(state: RAGState) -> dict:
    query = _effective_query(state)
    kept = state.get("kept_chunks", [])
    # Context-Enrichment Window: pad kept chunks with their ±N same-source neighbors.
    context = get_retriever().expand_with_neighbors(kept, settings.context_window)
    attempt = state.get("attempt", 0)
    verification = state.get("verification")
    feedback = (
        verification.unsupported_claims
        if (attempt > 0 and verification and verification.unsupported_claims)
        else None
    )
    t0 = time.perf_counter()
    answer, raw = generate_answer(query, context, feedback)
    enriched = "" if len(context) == len(kept) else f" · context {len(kept)}→{len(context)} (±{settings.context_window})"
    event = TraceEvent(
        stage="generate",
        summary=f"attempt {attempt + 1}" + (" (regenerate)" if feedback else "") + enriched,
        duration_ms=_ms(t0),
        data={"regenerate_feedback": feedback or [], "context_chunk_ids": [c.chunk_id for c in context]},
    )
    return {
        "answer": answer,
        "draft_answer": answer,
        "attempt": attempt + 1,
        "context_chunks": context,
        "trace": [event],
        "costs": [gate_cost("generate", settings.generation_model, raw)],
    }


def verify_node(state: RAGState) -> dict:
    if not settings.enable_verify:
        return {
            "verification": Verification(faithful=True, score=1.0),
            "flags": ["verify_skipped"],
            "trace": [TraceEvent(stage="verify", summary="skipped (ENABLE_VERIFY=false)")],
        }
    query = _effective_query(state)
    context = state.get("context_chunks") or state.get("kept_chunks", [])
    t0 = time.perf_counter()
    verification, raw = verify_answer(query, state.get("answer", ""), context)
    summary = "faithful" if verification.faithful else f"{len(verification.unsupported_claims)} unsupported claim(s)"
    event = TraceEvent(
        stage="verify",
        summary=summary,
        duration_ms=_ms(t0),
        data={"score": verification.score, "unsupported": verification.unsupported_claims},
    )
    return {
        "verification": verification,
        "trace": [event],
        "costs": [gate_cost("verify", settings.verify_model, raw)],
    }


def abstain_node(state: RAGState) -> dict:
    return {
        "answer": ABSTAIN_MESSAGE,
        "citations": [],
        "flags": ["abstained", "low_context"],
        "trace": [TraceEvent(stage="abstain", summary="too few relevant chunks after grading")],
    }


def finalize_node(state: RAGState) -> dict:
    """The TRACE gate: assemble grounded citations from the answer's markers."""
    flags_so_far = state.get("flags", [])
    abstained = "abstained" in flags_so_far
    answer = state.get("answer", "")
    context = state.get("context_chunks") or state.get("kept_chunks", [])
    verification = state.get("verification")

    citations = [] if abstained else extract_citations(answer, context, verification)

    new_flags: list[str] = []
    if not abstained and verification is not None and not verification.faithful:
        new_flags.append("unverified")

    event = TraceEvent(
        stage="trace",
        summary=f"{len(citations)} grounded citation(s)",
        data={"citations": [c.chunk_id for c in citations]},
    )
    return {"citations": citations, "flags": new_flags, "trace": [event]}


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------
def route_after_grade(state: RAGState) -> str:
    kept = state.get("kept_chunks", [])
    if len(kept) >= settings.min_kept_chunks:
        return "generate"
    if not state.get("rewritten_query"):  # haven't tried a rewrite yet
        return "rewrite"
    return "abstain"


def route_after_verify(state: RAGState) -> str:
    if not settings.enable_verify:
        return "finalize"
    verification = state.get("verification")
    if verification and verification.faithful:
        return "finalize"
    if state.get("attempt", 0) < settings.max_attempts and verification and verification.unsupported_claims:
        return "generate"  # regenerate once with feedback
    return "finalize"


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_graph():
    """Build and compile the StateGraph once."""
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("generate", generate_node)
    g.add_node("verify", verify_node)
    g.add_node("abstain", abstain_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade",
        route_after_grade,
        {"generate": "generate", "rewrite": "rewrite", "abstain": "abstain"},
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"generate": "generate", "finalize": "finalize"},
    )
    g.add_edge("abstain", "finalize")
    g.add_edge("finalize", END)
    return g.compile()
