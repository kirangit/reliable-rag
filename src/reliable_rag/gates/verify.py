"""VERIFY gate — is the answer actually grounded in the retrieved chunks?

Post 1's second gate, and the one the post argues is worth the cost on its own.
The judge (Opus — "use your strongest model as judge") decomposes the answer
into atomic claims and checks each against the context. A claim the model
believes from training data but that the context does not support is exactly the
hallucination this catches.

We deliberately judge *grounding in the context*, not real-world truth: a claim
can be true yet unsupported, and an unsupported claim is still a reliability
failure for a RAG system.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import AIMessage

from ..config import settings
from ..models import chat
from ..schemas import ClaimCheck, RetrievedChunk, Verification, VerifyResponse
from ._common import format_chunks_for_prompt

VERIFY_SYSTEM = (
    "You are a strict faithfulness checker for a RAG system. You are given a QUESTION, "
    "an ANSWER, and the CONTEXT chunks the answer was supposed to be based on.\n"
    "1. Break the ANSWER into atomic factual claims.\n"
    "2. For each claim, decide if it is supported by the CONTEXT, and list the "
    "supporting chunk_id(s).\n"
    "Judge ONLY against the context — ignore whether a claim is true in the real "
    "world. If the context does not support a claim, mark grounded=false (this is how "
    "hallucinations are caught)."
)


def verify_answer(
    query: str, answer: str, kept_chunks: Sequence[RetrievedChunk]
) -> tuple[Verification, AIMessage | None]:
    """Check the answer claim-by-claim. Returns ``(Verification, raw_message)``."""
    llm = chat(settings.verify_model, max_tokens=settings.verify_max_tokens).with_structured_output(
        VerifyResponse, include_raw=True
    )
    user = (
        f"Question:\n{query}\n\n"
        f"Answer to check:\n{answer}\n\n"
        f"Context chunks:\n\n{format_chunks_for_prompt(kept_chunks)}"
    )
    try:
        result = llm.invoke([("system", VERIFY_SYSTEM), ("human", user)])
        raw: AIMessage | None = result.get("raw")
        parsed: VerifyResponse | None = result.get("parsed")
    except Exception:
        # A malformed structured response must not crash the pipeline; treat as
        # "could not verify" -> no claims -> not faithful (finalize flags it).
        raw, parsed = None, None

    claims: list[ClaimCheck] = []
    if parsed:
        for c in parsed.claims:
            claims.append(
                ClaimCheck(
                    claim=c.claim,
                    grounded=c.grounded,
                    chunk_ids=c.supporting_chunk_ids,
                    reason=c.reason,
                )
            )

    unsupported = [c.claim for c in claims if not c.grounded]
    grounded = sum(1 for c in claims if c.grounded)
    score = grounded / len(claims) if claims else 0.0
    # Faithful only if we found claims AND none were unsupported.
    faithful = bool(claims) and not unsupported

    verification = Verification(
        faithful=faithful,
        score=score,
        claims=claims,
        unsupported_claims=unsupported,
    )
    return verification, raw
