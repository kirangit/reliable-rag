"""``run_query`` — the one entrypoint the CLI and both UIs call.

It invokes the compiled graph, rolls the final state up into a typed
:class:`RunResult`, persists the trace to ``runs/``, and returns the result.
Keeping this thin and typed means the CLI, Chainlit, and Streamlit all share the
exact same execution path.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from pydantic import BaseModel, Field

from .condense import condense_question
from .config import settings
from .cost import gate_cost, total_usd
from .graph import get_graph
from .observability import log_run, new_run_id, setup_tracing
from .schemas import Citation, ChunkGrade, GateCost, RetrievedChunk, TraceEvent, Verification


class RunResult(BaseModel):
    """Everything one query produced — answer + full reliability trace + cost."""

    run_id: str
    query: str
    condensed_query: Optional[str] = None  # standalone rewrite of a follow-up (conversational mode)
    rewritten_query: Optional[str] = None
    answer: str = ""
    abstained: bool = False
    flags: list[str] = Field(default_factory=list)
    grades: list[ChunkGrade] = Field(default_factory=list)
    kept_chunks: list[RetrievedChunk] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    verification: Optional[Verification] = None
    costs: list[GateCost] = Field(default_factory=list)
    total_usd: float = 0.0
    trace: list[TraceEvent] = Field(default_factory=list)
    elapsed_ms: float = 0.0


def run_query(query: str, k: int | None = None, history=None) -> RunResult:
    """Run one question through the gated pipeline and return a typed result.

    If ``conversational`` is on and ``history`` is supplied, the follow-up is first
    condensed into a standalone question (condense-only) before the graph runs. The
    CLI is single-turn and passes no history, so this path is dormant there.
    """
    setup_tracing()  # no-op unless TRACING is set; safe to call repeatedly

    standalone, condense_raw = query, None
    if settings.conversational and history:
        standalone, condense_raw = condense_question(history, query)

    init: dict = {"query": standalone, "attempt": 0, "flags": [], "trace": [], "costs": []}
    if k:
        init["k"] = k

    t0 = time.perf_counter()
    final = get_graph().invoke(init, config={"recursion_limit": 15})
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    costs = list(final.get("costs", []))
    trace = list(final.get("trace", []))
    if condense_raw is not None:  # prepend the condense step to cost + trace
        costs.insert(0, gate_cost("condense", settings.generation_model, condense_raw))
        trace.insert(0, TraceEvent(stage="condense", summary="rewrote follow-up to standalone",
                                   data={"from": query, "to": standalone}))

    result = RunResult(
        run_id=new_run_id(),
        query=query,
        condensed_query=standalone if standalone != query else None,
        rewritten_query=final.get("rewritten_query"),
        answer=final.get("answer", ""),
        abstained="abstained" in final.get("flags", []),
        flags=final.get("flags", []),
        grades=final.get("grades", []),
        kept_chunks=final.get("kept_chunks", []),
        citations=final.get("citations", []),
        verification=final.get("verification"),
        costs=costs,
        total_usd=total_usd(costs),
        trace=trace,
        elapsed_ms=elapsed_ms,
    )
    log_run(result)
    return result


async def astream_query(query: str, k: int | None = None, history=None):
    """Async generator for live UIs: yields ``("step", node_name, delta)`` as each
    gate finishes, then a final ``("result", None, RunResult)``.

    The CLI uses the synchronous :func:`run_query`; the Chainlit app uses this so it
    can render each gate the moment it completes and stream the final answer. The
    graph nodes are synchronous — LangGraph runs them off the event loop.

    Conversational mode: if ``history`` is supplied (and the flag is on), the
    follow-up is condensed into a standalone query first and emitted as its own
    ``("step", "condense", ...)`` before the graph stream begins.
    """
    setup_tracing()

    standalone, condense_raw = query, None
    if settings.conversational and history:
        # run the (blocking) condense call off the event loop
        standalone, condense_raw = await asyncio.to_thread(condense_question, history, query)

    init: dict = {"query": standalone, "attempt": 0, "flags": [], "trace": [], "costs": []}
    if k:
        init["k"] = k

    # accumulate across node updates (reducer channels append; others overwrite)
    acc: dict = {
        "rewritten_query": None, "grades": [], "kept_chunks": [], "context_chunks": [],
        "answer": "", "verification": None, "citations": [], "flags": [], "costs": [], "trace": [],
    }
    t0 = time.perf_counter()

    if condense_raw is not None:  # surface the condense step + its cost first
        acc["costs"].append(gate_cost("condense", settings.generation_model, condense_raw))
        acc["trace"].append(TraceEvent(stage="condense", summary="rewrote follow-up to standalone",
                                       data={"from": query, "to": standalone}))
        yield "step", "condense", {"original": query, "standalone": standalone}

    async for chunk in get_graph().astream(init, stream_mode="updates", config={"recursion_limit": 15}):
        for node, delta in chunk.items():
            for key in ("costs", "flags", "trace"):  # reducer channels -> append
                if key in delta:
                    acc[key] = acc[key] + list(delta[key])
            for key in ("rewritten_query", "grades", "kept_chunks", "context_chunks",
                        "answer", "verification", "citations"):  # latest value wins
                if key in delta:
                    acc[key] = delta[key]
            yield "step", node, delta

    result = RunResult(
        run_id=new_run_id(),
        query=query,
        condensed_query=standalone if standalone != query else None,
        rewritten_query=acc["rewritten_query"],
        answer=acc["answer"],
        abstained="abstained" in acc["flags"],
        flags=acc["flags"],
        grades=acc["grades"],
        kept_chunks=acc["kept_chunks"],
        citations=acc["citations"],
        verification=acc["verification"],
        costs=acc["costs"],
        total_usd=total_usd(acc["costs"]),
        trace=acc["trace"],
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )
    log_run(result)
    yield "result", None, result
