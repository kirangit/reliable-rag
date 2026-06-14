"""Cost accounting — "count the cost" from Post 1, made literal.

Reliable RAG trades extra LLM calls for safety, so we measure that trade in
tokens and dollars, **per gate**. Token counts come straight off each
``ChatAnthropic`` response's ``usage_metadata`` — no extra API calls. Prices are
the published per-1M-token rates (update here if they change).

Embedding cost is *estimated* (the OpenAI embeddings client doesn't surface token
usage through LangChain) from a ~4-chars-per-token heuristic — fine for the tiny
per-query query embedding; clearly labeled as an estimate.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache

from langchain_core.messages import AIMessage

from .schemas import GateCost

# USD per 1,000,000 tokens, as (input, output).
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

# USD per 1,000,000 input tokens (embeddings have no output tokens).
EMBED_PRICING: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}

CACHE_READ_MULT = 0.1    # cached input is billed ~0.1x
CACHE_WRITE_MULT = 1.25  # writing the 5-min cache is ~1.25x


def _rates(model: str) -> tuple[float, float]:
    if model in PRICING:
        return PRICING[model]
    for prefix, rates in PRICING.items():  # tolerate date-suffixed ids
        if model.startswith(prefix):
            return rates
    return (0.0, 0.0)  # unknown model -> don't guess; report 0


def compute_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    cache_creation: int = 0,
) -> float:
    """Dollar cost of one Claude call. ``input_tokens`` is the LangChain total
    (includes cached), so we subtract the cache portions to avoid double-billing."""
    in_rate, out_rate = _rates(model)
    uncached = max(0, input_tokens - cache_read - cache_creation)
    cost = (
        uncached * in_rate
        + cache_read * in_rate * CACHE_READ_MULT
        + cache_creation * in_rate * CACHE_WRITE_MULT
        + output_tokens * out_rate
    )
    return cost / 1_000_000


def gate_cost(stage: str, model: str, message: AIMessage | None) -> GateCost:
    """Build a :class:`GateCost` from a gate's raw ``AIMessage`` (or zero if None)."""
    if message is None:
        return GateCost(stage=stage, model=model)
    usage = getattr(message, "usage_metadata", None) or {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    details = usage.get("input_token_details") or {}
    cache_read = int(details.get("cache_read", 0) or 0)
    cache_creation = int(details.get("cache_creation", 0) or 0)
    return GateCost(
        stage=stage,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_creation,
        usd=compute_usd(model, input_tokens, output_tokens, cache_read, cache_creation),
    )


@lru_cache(maxsize=4)
def _encoder(model: str):
    """Return the tiktoken encoder for an embedding model, or ``None`` if tiktoken
    (or its encoding file) is unavailable — in which case callers fall back to a
    ~4-chars-per-token approximation. tiktoken is the *correct* tokenizer for
    OpenAI embeddings, so token counts here are exact when it's installed."""
    try:
        import tiktoken

        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")  # the embedding-3 family
    except Exception:  # tiktoken missing, or offline and no cached encoding
        return None


def count_tokens(text: str, model: str) -> int:
    enc = _encoder(model)
    if enc is None:
        return max(1, len(text or "") // 4)
    return len(enc.encode(text or ""))


def count_tokens_for_texts(texts: Sequence[str], model: str) -> tuple[int, str]:
    """Total token count over many texts, plus the source ("tiktoken" | "approx")."""
    enc = _encoder(model)
    if enc is None:
        return sum(max(1, len(t or "") // 4) for t in texts), "approx"
    return sum(len(enc.encode(t or "")) for t in texts), "tiktoken"


def embed_usd(tokens: int, model: str) -> float:
    return tokens * EMBED_PRICING.get(model, 0.02) / 1_000_000


def estimate_embed_cost(text: str, model: str) -> GateCost:
    """Cost of embedding ``text`` (the per-query query embedding)."""
    tokens = count_tokens(text, model)
    return GateCost(stage="embed", model=model, input_tokens=tokens, usd=embed_usd(tokens, model))


def total_usd(costs: Iterable[GateCost]) -> float:
    return round(sum(c.usd for c in costs), 6)


def cost_summary(costs: Iterable[GateCost]) -> dict:
    """Per-stage and total rollup, handy for the UIs."""
    by_stage: dict[str, float] = {}
    in_tok = out_tok = 0
    for c in costs:
        by_stage[c.stage] = round(by_stage.get(c.stage, 0.0) + c.usd, 6)
        in_tok += c.input_tokens
        out_tok += c.output_tokens
    return {
        "by_stage_usd": by_stage,
        "total_usd": round(sum(by_stage.values()), 6),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }
