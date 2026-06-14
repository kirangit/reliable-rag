"""LLM + embedding factories — the single place models are constructed.

Centralising this means the gates never hard-code a model name or worry about
auth: they ask for ``chat(settings.grade_model)`` and get a configured client.

Note on Claude 4.x: Opus 4.8 / Sonnet 4.6 reject ``temperature``/``top_p`` and
the old fixed ``budget_tokens``. We simply don't pass them — structured output
(tool calling) gives us the determinism we need for the gates without sampling
knobs. Adaptive thinking is left off for the gates to keep latency/cost low.
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings

from .config import settings


def export_keys() -> None:
    """Make keys from ``.env``/settings visible to the langchain clients, which
    read ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` from the environment. We use
    ``setdefault`` so a real shell env var always wins over the .env file."""
    if settings.anthropic_api_key:
        os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)
    if settings.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)


@lru_cache(maxsize=8)
def chat(model: str, max_tokens: int = 2048) -> ChatAnthropic:
    """Return a cached ``ChatAnthropic`` for ``model``.

    Cached on (model, max_tokens) so repeated gate calls reuse one client.
    """
    settings.require_anthropic()
    export_keys()
    return ChatAnthropic(model=model, max_tokens=max_tokens, timeout=120)


@lru_cache(maxsize=1)
def embeddings() -> OpenAIEmbeddings:
    """Return a cached OpenAI embeddings client (the only non-Anthropic call)."""
    settings.require_openai()
    export_keys()
    return OpenAIEmbeddings(model=settings.embedding_model)


class _RagasSafeChatAnthropic(ChatAnthropic):
    """ChatAnthropic that drops sampling params RAGAS injects.

    RAGAS's ``LangchainLLMWrapper`` passes ``temperature`` (and sometimes
    ``top_p``/``top_k``) on every judge call, but Claude 4.x (Opus 4.8 /
    Sonnet 4.6) reject those parameters with a 400. We strip them so the
    judge calls succeed. Our own gates never pass these, so they're unaffected.
    """

    def _get_request_payload(self, *args, **kwargs):
        # temperature/top_p/top_k get added to the payload from model attributes
        # that RAGAS sets on the instance; strip them from the final request so
        # Claude 4.x (which rejects them) accepts the call. _generate and
        # _agenerate both route through here.
        payload = super()._get_request_payload(*args, **kwargs)
        for p in ("temperature", "top_p", "top_k"):
            payload.pop(p, None)
        return payload


def ragas_chat(model: str, max_tokens: int = 2048) -> ChatAnthropic:
    """A ChatAnthropic safe to hand to RAGAS as the evaluator/judge LLM."""
    settings.require_anthropic()
    export_keys()
    return _RagasSafeChatAnthropic(model=model, max_tokens=max_tokens, timeout=120)
