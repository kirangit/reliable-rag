"""Central configuration — every tunable knob in one place.

Why this module exists
----------------------
A reliable system has *no magic numbers scattered through the code*. The Grade
threshold, the retrieval ``k``, which Claude model judges faithfulness — these
are reliability decisions, so they live here as named, documented settings that
can be overridden from a ``.env`` file or the environment.

Settings are loaded once into the module-level ``settings`` singleton. Import it
anywhere: ``from reliable_rag.config import settings``.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration, populated from environment / ``.env``.

    Field names map to UPPER_CASE env vars (e.g. ``grade_threshold`` <-
    ``GRADE_THRESHOLD``). See ``.env.example`` for the full list with comments.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # tolerate unrelated env vars (LANGSMITH_*, etc.)
    )

    # --- API keys (optional at import-time so tests / chunking work offline;
    #     the model factories raise a clear error if a key is actually needed) ---
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # --- Models ---
    # Sonnet for the cheap, high-volume work (generation + grading); Opus as the
    # "strongest model as judge" for Verify and the RAGAS Faithfulness metric.
    generation_model: str = "claude-sonnet-4-6"
    grade_model: str = "claude-sonnet-4-6"
    verify_model: str = "claude-opus-4-8"
    judge_model: str = "claude-opus-4-8"
    embedding_model: str = "text-embedding-3-small"

    # --- Corpus / index ---
    docs_dir: str = "docs"
    chroma_dir: str = ".chroma"
    collection_name: str = "cnwave"
    # The manifest mirrors every chunk to disk (chunk_id -> raw text + metadata).
    # It powers BM25 retrieval and the Trace/citation rendering without re-reading
    # the vector store. Derived from chroma_dir unless overridden.
    manifest_name: str = "chunks.jsonl"

    # --- Retrieval ---
    retrieval_mode: str = "hybrid"  # "hybrid" (dense + BM25) | "dense"
    retrieve_k: int = 8             # over-retrieve a little; the Grade gate filters the noise
    # Context-Enrichment Window: after grading, pad each kept chunk with its ±N
    # same-document neighbors so a hit pulls in adjacent context (procedure steps,
    # sibling list items). 0 disables.
    context_window: int = 1

    # --- Reliability gates ---
    grade_threshold: float = 0.5    # keep chunks the Grade gate scores >= this
    min_kept_chunks: int = 2        # fewer than this -> rewrite query, then abstain
    max_attempts: int = 2           # generate/verify retry budget (>=1)
    enable_verify: bool = True      # turn the Verify gate off for low-stakes/speed
    # Verify re-states every claim + its reason + chunk_ids as structured output,
    # so it needs MORE output room than the answer itself. Too low truncates the
    # JSON -> parse fails -> a good answer is wrongly flagged "unverified". A high
    # ceiling is safe: you're billed only for tokens actually generated.
    verify_max_tokens: int = 8192

    # --- Conversation (multi-turn) ---
    # When on, a follow-up question is condensed into a standalone query (using the
    # chat history) BEFORE retrieval. Condense-only: history is never put in the
    # answer prompt. Off by default — single-turn behaviour is unchanged.
    conversational: bool = False

    # --- Chunking ---
    max_chunk_chars: int = 1200     # sections larger than this get a size-guard split
    chunk_overlap: int = 100
    header_split_max_level: int = 2  # split markdown on '#'/'##' only; deeper headings stay in-content

    # --- Observability ---
    tracing: str = "none"           # "none" | "phoenix" | "langsmith"
    log_level: str = "INFO"         # DEBUG dumps full chunk text + prompts
    runs_dir: str = "runs"
    feedback_dir: str = "feedback"

    # --- LangSmith (only used when tracing == "langsmith") ---
    # Declared as real settings so a key in .env reaches the LangSmith SDK for the
    # CLI too (the SDK reads os.environ; setup_tracing bridges these across).
    langsmith_api_key: str | None = None
    langsmith_project: str = "reliable-rag"

    # --- Derived paths -----------------------------------------------------
    @property
    def manifest_path(self) -> str:
        """Absolute-ish path to the chunk manifest (lives beside the index)."""
        from pathlib import Path

        return str(Path(self.chroma_dir) / self.manifest_name)

    def require_anthropic(self) -> str:
        if not self.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return self.anthropic_api_key

    def require_openai(self) -> str:
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key "
                "(used only for embeddings)."
            )
        return self.openai_api_key


# The one instance everything imports.
settings = Settings()
