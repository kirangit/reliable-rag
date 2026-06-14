"""Condense a conversational follow-up into a standalone question.

The multi-turn problem: a follow-up like "how long does it take?" is meaningless
to embed on its own — it depends on the previous turn. Before retrieval we rewrite
(history + follow-up) into a self-contained question, so the vector search makes
sense again. This is the standard "condense question" / history-aware-retrieval
step.

We keep it **condense-only**: the history is used *solely* to reformulate the
query, never injected into the answer (so generation stays grounded purely in the
retrieved docs). It runs as a pre-step before the graph — gated by
``settings.conversational`` at the call site — so the gated pipeline itself is
untouched.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import AIMessage

from .config import settings
from .gates.generate import _content_to_text
from .models import chat

CONDENSE_SYSTEM = (
    "You reformulate a user's latest message into a STANDALONE question for "
    "document retrieval, using the conversation history to resolve references "
    "(e.g. 'it', 'that', 'the unit', 'how long does it take'). Rules:\n"
    "- Output ONLY the standalone question — no preamble, no answer.\n"
    "- If the latest message is already self-contained, return it unchanged.\n"
    "- Stay faithful to the user's intent; keep it concise."
)


def _format_history(history: Sequence[tuple[str, str]], max_messages: int = 8) -> str:
    return "\n".join(f"{role}: {content}" for role, content in history[-max_messages:])


def condense_question(
    history: Sequence[tuple[str, str]], query: str
) -> tuple[str, AIMessage | None]:
    """Rewrite ``query`` into a standalone question using ``history``.

    Returns ``(standalone_query, raw_message)``. With no history it's a no-op
    passthrough (``raw_message`` is None, so it costs nothing).
    """
    if not history:
        return query, None

    llm = chat(settings.generation_model, max_tokens=256)
    user = (
        f"Conversation so far:\n{_format_history(history)}\n\n"
        f"Latest user message:\n{query}\n\nStandalone question:"
    )
    msg = llm.invoke([("system", CONDENSE_SYSTEM), ("human", user)])
    text = _content_to_text(msg.content).strip().strip('"').strip()
    return (text or query), msg
