"""Pytest session setup — keep the suite hermetic.

Tests must never depend on or pollute external tracing (LangSmith / Phoenix),
even when the developer's ``.env`` has ``TRACING=langsmith``. We force it off
before anything imports the pipeline, so ``pytest`` makes no network calls and
never writes runs into a real LangSmith project.
"""

from __future__ import annotations

import os

# Belt: the LangSmith SDK reads this and won't send traces even if enabled.
os.environ["LANGSMITH_TRACING"] = "false"

# Suspenders: make setup_tracing() a no-op regardless of the .env value.
from reliable_rag.config import settings  # noqa: E402

settings.tracing = "none"
