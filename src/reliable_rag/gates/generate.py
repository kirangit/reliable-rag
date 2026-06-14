"""GENERATE — answer using ONLY the kept chunks.

The standard generation step, with two reliability constraints:

* The model uses only the provided context (no outside knowledge) and writes
  clean prose **without** inline citation markers. Sources are tracked separately
  by the Verify + Trace steps — grounded in the retrieval system, not fabricated
  by the model — which keeps the answer readable while preserving traceability.
* On a regenerate (after Verify finds unsupported claims) we pass the offending
  claims back as ``feedback`` so the model drops or qualifies them.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import AIMessage

from ..config import settings
from ..models import chat
from ..schemas import RetrievedChunk
from ._common import format_chunks_for_prompt

GEN_SYSTEM = (
    "You answer questions about Cambium cnWave 60 GHz wireless-mesh networks using "
    "ONLY the provided context chunks. Rules:\n"
    "1. Use only information present in the context. Do NOT use outside knowledge.\n"
    "2. Do NOT put citation markers, chunk_ids, or bracketed references in your "
    "answer — sources are tracked separately by the pipeline. Just write the "
    "grounded answer in clean prose.\n"
    "3. If the context does not contain the answer, say exactly that you don't have "
    "enough grounded information — do not guess.\n"
    "4. Format compactly: do NOT use level-1/level-2 Markdown headings (`#`, `##`), "
    "and don't restate the question as a title. Use short **bold** labels (or `###`) "
    "for any section structure.\n"
    "Be concise and technical."
)


def _content_to_text(content) -> str:
    """Anthropic responses are usually a string but can be a list of blocks."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, dict):
            parts.append(block.get("text", ""))
        else:
            parts.append(str(block))
    return "".join(parts)


def generate_answer(
    query: str,
    kept_chunks: Sequence[RetrievedChunk],
    feedback: list[str] | None = None,
) -> tuple[str, AIMessage | None]:
    """Generate a grounded, chunk-cited answer. Returns ``(answer_text, raw_message)``."""
    llm = chat(settings.generation_model, max_tokens=1536)
    user = (
        f"Question:\n{query}\n\n"
        f"Context chunks (cite these by [chunk_id]):\n\n{format_chunks_for_prompt(kept_chunks)}"
    )
    if feedback:
        joined = "; ".join(feedback)
        user += (
            "\n\nIMPORTANT: a previous draft made claims NOT supported by the context: "
            f"{joined}. Rewrite the answer so every claim is supported by the context "
            "above — remove or explicitly qualify anything that isn't."
        )

    msg = llm.invoke([("system", GEN_SYSTEM), ("human", user)])
    return _content_to_text(msg.content).strip(), msg
