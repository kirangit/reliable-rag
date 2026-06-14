"""Cost accounting math — runs offline."""

from __future__ import annotations

from reliable_rag.cost import (
    compute_usd,
    count_tokens_for_texts,
    embed_usd,
    gate_cost,
    total_usd,
)
from reliable_rag.schemas import GateCost


def test_compute_usd_opus_rates():
    # 1M input + 1M output at Opus 4.8 rates ($5 / $25) = $30.
    assert abs(compute_usd("claude-opus-4-8", 1_000_000, 1_000_000) - 30.0) < 1e-6


def test_unknown_model_costs_zero():
    assert compute_usd("some-unknown-model", 1000, 1000) == 0.0


def test_cache_read_is_cheaper():
    full = compute_usd("claude-sonnet-4-6", 1000, 0)
    cached = compute_usd("claude-sonnet-4-6", 1000, 0, cache_read=1000)
    assert cached < full  # cached input billed at ~0.1x


def test_embedding_token_count_and_cost():
    tokens, source = count_tokens_for_texts(["hello world", "another chunk of text"], "text-embedding-3-small")
    assert tokens > 0
    assert source in {"tiktoken", "approx"}
    assert embed_usd(tokens, "text-embedding-3-small") >= 0.0


def test_gate_cost_from_none_message_is_zero():
    gc = gate_cost("grade", "claude-sonnet-4-6", None)
    assert gc.usd == 0.0 and gc.input_tokens == 0


def test_total_usd_sums():
    costs = [GateCost(stage="a", model="m", usd=0.01), GateCost(stage="b", model="m", usd=0.02)]
    assert abs(total_usd(costs) - 0.03) < 1e-9
