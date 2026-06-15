"""Chainlit demo — the gated chat that shows its work *live*.

Instead of running the whole pipeline and then dumping the steps, this drives the
graph's async stream (:func:`reliable_rag.pipeline.astream_query`): each gate
renders as a collapsible **step the moment it finishes** (Retrieve → Grade →
[Rewrite/Abstain] → Generate → Verify → Trace), and the final *verified* answer is
then streamed in. Grounded citations live inside the Trace step; cost + 👍/👎 sit
under the answer.

Run it:  ``chainlit run app/chainlit_app.py``  (needs ANTHROPIC_API_KEY + OPENAI_API_KEY).
"""

from __future__ import annotations

import asyncio
import re

import chainlit as cl

from reliable_rag.config import settings
from reliable_rag.cost import cost_summary
from reliable_rag.feedback import record_feedback
from reliable_rag.pipeline import RunResult, astream_query

WELCOME = (
    "### Reliable RAG — cnWave docs\n"
    "Ask about the Cambium cnWave 60 GHz mesh (deployment, recovery, LED states, "
    "packet flow, the cnMaestro API, beam scanning…).\n\n"
    "You'll watch the **Grade / Verify / Trace** gates run live, then the grounded "
    "answer streams in. Out-of-scope questions are **abstained**, not guessed."
)


# ---------------------------------------------------------------------------
# Renderers (operate on raw objects, so they work from per-node stream deltas)
# ---------------------------------------------------------------------------
def _grades_table(grades) -> str:
    if not grades:
        return "_no chunks retrieved_"
    rows = ["| kept | score | chunk_id | why |", "|:---:|:---:|---|---|"]
    for g in grades:
        rows.append(f"| {'✓' if g.relevant else '·'} | {g.score:.2f} | `{g.chunk_id}` | {g.reason} |")
    return "\n".join(rows)


def _retrieved_block(chunks) -> str:
    """Raw retrieved chunks (pre-Grade): rank, id, score, location, text snippet."""
    if not chunks:
        return "_no chunks retrieved_"
    out = [f"Retrieved **{len(chunks)}** candidate chunks (pre-Grade):", ""]
    for i, c in enumerate(chunks, 1):
        snippet = " ".join(c.text.split())  # collapse newlines/whitespace for a clean preview
        if len(snippet) > 200:
            snippet = snippet[:200].rstrip() + "…"
        loc = c.source + (f" › {c.header_path}" if c.header_path else "")
        out.append(f"**{i}. `{c.chunk_id}`** · score {c.score:.2f}")
        out.append(f"*{loc}*")
        out.append(f"> {snippet}")
        out.append("")
    return "\n".join(out)


def _verify_block(verification) -> str:
    if verification is None:
        return "_verification skipped_"
    head = ("✅ **faithful**" if verification.faithful else "⚠️ **unverified**")
    head += f" — grounded {verification.score:.0%} of {len(verification.claims)} claims"
    lines = [head, "", "| grounded | claim | supporting |", "|:---:|---|---|"]
    for c in verification.claims:
        support = ", ".join(f"`{x}`" for x in c.chunk_ids) or "—"
        lines.append(f"| {'✓' if c.grounded else '✗'} | {c.claim} | {support} |")
    return "\n".join(lines)


def _citations_block(citations) -> str:
    if not citations:
        return "_no grounded citations_"
    return "\n\n---\n\n".join(
        f"**`{c.chunk_id}`**\n\n{c.source} — {c.header_path or '—'}\n\n> {c.snippet}" for c in citations
    )


def _cost_footer(result: RunResult) -> str:
    s = cost_summary(result.costs)
    by_stage = " · ".join(f"{k} ${v:.4f}" for k, v in s["by_stage_usd"].items())
    return (
        f"\n\n---\n*💰 **${result.total_usd:.4f}** ({by_stage}) · ⏱ {result.elapsed_ms:.0f} ms · "
        f"{s['input_tokens']} in / {s['output_tokens']} out tokens*"
    )


async def _render_step(node: str, delta: dict) -> None:
    """Render one gate as a collapsible step, the moment its node finishes."""
    if node == "condense":
        async with cl.Step(name="↺ Condense follow-up (using chat history)", type="llm") as step:
            step.output = (
                "Rewrote your message into a standalone question for retrieval:\n\n"
                f"> {delta.get('standalone', '')}"
            )
    elif node == "retrieve":
        mode = "dense: vector only" if settings.retrieval_mode == "dense" else "hybrid: dense + BM25"
        async with cl.Step(name=f"① Retrieve ({mode})", type="retrieval") as step:
            step.output = _retrieved_block(delta.get("retrieved", []))
    elif node == "grade":
        grades = delta.get("grades", [])
        kept = sum(1 for g in grades if g.relevant)
        async with cl.Step(name="② Grade — drop irrelevant chunks", type="llm") as step:
            step.output = f"Kept **{kept}/{len(grades)}** chunks.\n\n" + _grades_table(grades)
    elif node == "rewrite":
        async with cl.Step(name="↺ Rewrite query & re-retrieve", type="llm") as step:
            step.output = f"Too few chunks passed Grade → rewrote query to:\n\n> {delta.get('rewritten_query', '')}"
    elif node == "abstain":
        async with cl.Step(name="✋ Abstain", type="tool") as step:
            step.output = "Not enough relevant context after grading — abstaining instead of guessing."
    elif node == "generate":
        n_ctx = len(delta.get("context_chunks", []))
        async with cl.Step(name="③ Generate — grounded in retrieved context", type="llm") as step:
            step.output = f"Drafted an answer from {n_ctx} context chunks (kept + neighbors)."
    elif node == "verify":
        async with cl.Step(name="④ Verify — claim-by-claim grounding", type="llm") as step:
            step.output = _verify_block(delta.get("verification"))
    elif node == "finalize":
        async with cl.Step(name="⑤ Trace — grounded citations", type="tool") as step:
            step.output = _citations_block(delta.get("citations", []))


def _stream_pieces(text: str):
    """Split the answer into small chunks for a progressive (streaming) reveal."""
    return re.findall(r"\S+\s*|\s+", text) or [text]


# ---------------------------------------------------------------------------
# Chat lifecycle
# ---------------------------------------------------------------------------
@cl.on_chat_start
async def on_start() -> None:
    await cl.Message(content=WELCOME).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    query = message.content.strip()
    if not query:
        return

    # Prior turns (conversational mode). Used only to condense the follow-up into a
    # standalone query — the graph still runs single-turn on that query. Dormant
    # unless CONVERSATIONAL=true. Windowed to recent turns to bound tokens.
    history = cl.user_session.get("history", [])

    # Drive the graph's async stream: each gate renders the moment it finishes.
    result: RunResult | None = None
    async for kind, node, payload in astream_query(query, history=history):
        if kind == "step":
            await _render_step(node, payload)
        else:  # ("result", None, RunResult)
            result = payload

    if result is None:  # safety net — shouldn't happen
        await cl.Message(content="Something went wrong producing an answer.").send()
        return

    # Stream the final, verified answer in progressively.
    answer_msg = cl.Message(content="")
    await answer_msg.send()
    for piece in _stream_pieces(result.answer):
        await answer_msg.stream_token(piece)
        await asyncio.sleep(0.01)

    # Finalize: append the cost line + 👍/👎 actions.
    answer_msg.content = result.answer + _cost_footer(result)
    answer_msg.actions = [
        cl.Action(name="thumbs_up", label="👍 Helpful", payload={"run_id": result.run_id, "verdict": "up"}),
        cl.Action(name="thumbs_down", label="👎 Not helpful", payload={"run_id": result.run_id, "verdict": "down"}),
    ]
    cl.user_session.set(
        result.run_id,
        {"query": query, "answer": result.answer, "flags": result.flags, "total_usd": result.total_usd},
    )
    # Append this turn to history (windowed to the last 8 messages = ~4 exchanges).
    cl.user_session.set("history", (history + [("user", query), ("assistant", result.answer)])[-8:])
    await answer_msg.update()


# ---------------------------------------------------------------------------
# Feedback callbacks (👍/👎) -> persisted to feedback/feedback.jsonl
# ---------------------------------------------------------------------------
def _log_vote(action: cl.Action, verdict: str) -> None:
    run_id = action.payload.get("run_id", "")
    context = cl.user_session.get(run_id) or {}
    record_feedback({"run_id": run_id, "verdict": verdict, **context})


@cl.action_callback("thumbs_up")
async def on_thumbs_up(action: cl.Action) -> None:
    _log_vote(action, "up")
    await cl.Message(content="Thanks — logged your 👍.").send()


@cl.action_callback("thumbs_down")
async def on_thumbs_down(action: cl.Action) -> None:
    _log_vote(action, "down")
    await cl.Message(content="Thanks — logged your 👎. This feedback shows up in the eval dashboard.").send()
