"""Ingestion — turn the corpus into a searchable index + an inspectable manifest.

Two outputs, by design:

1. **Chroma vector store** (``.chroma/``) — embeds each chunk's *contextual* text
   (breadcrumb + raw text) so retrieval benefits from document-level context.
2. **Chunk manifest** (``.chroma/chunks.jsonl``) — every chunk's RAW text +
   metadata, one JSON object per line. This powers BM25 retrieval and the
   Trace/citation rendering without re-reading the vector store, and lets you
   eyeball exactly what got indexed.

The chunk-building half is offline (no API key); only :func:`build_index` calls
the OpenAI embeddings API. That split keeps chunking unit-testable without keys.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from langchain_core.documents import Document

from .chunkers import chunk_path
from .config import settings
from .cost import count_tokens_for_texts, embed_usd
from .observability import log_ingest_cost
from .schemas import Chunk


# ---------------------------------------------------------------------------
# Chunk building (offline)
# ---------------------------------------------------------------------------
def discover_files(docs_dir: str | None = None) -> list[Path]:
    """Return the markdown files under ``docs_dir``, sorted for determinism.

    CSV/tabular sources are intentionally **out of scope**: an exact table lookup
    ("V5K MCS12 downlink throughput") is a structured query best served by a
    metadata filter over all matching rows, not by semantic top-k search — which
    can't enumerate 24 near-identical rows under a small ``k``. So we don't index
    them here. See CHUNKING.md.
    """
    root = Path(docs_dir or settings.docs_dir)
    if not root.exists():
        raise FileNotFoundError(f"DOCS_DIR not found: {root.resolve()}")
    files = [p for p in root.rglob("*") if p.suffix.lower() in {".md", ".markdown"}]
    return sorted(files, key=lambda p: p.as_posix())


def build_chunks(docs_dir: str | None = None) -> list[Chunk]:
    """Chunk every discovered file into a flat list of ``Chunk`` objects."""
    chunks: list[Chunk] = []
    for path in discover_files(docs_dir):
        chunks.extend(
            chunk_path(
                str(path), settings.max_chunk_chars, settings.chunk_overlap, settings.header_split_max_level
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Manifest (offline)
# ---------------------------------------------------------------------------
def write_manifest(chunks: list[Chunk], path: str | None = None) -> str:
    """Write one JSON object per chunk to ``chunks.jsonl``. Returns the path."""
    out = Path(path or settings.manifest_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")
    return str(out)


def load_manifest(path: str | None = None) -> list[Chunk]:
    """Read the manifest back into ``Chunk`` objects (used by BM25 + Trace)."""
    src = Path(path or settings.manifest_path)
    if not src.exists():
        raise FileNotFoundError(
            f"Chunk manifest not found: {src}. Run `reliable-rag ingest` first."
        )
    chunks: list[Chunk] = []
    with src.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(Chunk(**json.loads(line)))
    return chunks


# ---------------------------------------------------------------------------
# Vector index (needs OPENAI_API_KEY)
# ---------------------------------------------------------------------------
def _to_document(chunk: Chunk) -> Document:
    """A langchain Document whose page_content is the *embedded* text (contextual
    header + raw text), with the raw text kept in metadata for citations.

    Chroma metadata must be scalar, so we store only flat fields here; the full
    record lives in the manifest, keyed by chunk_id.
    """
    return Document(
        page_content=chunk.embed_text,
        metadata={
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "header_path": chunk.header_path,
            "chunk_type": chunk.chunk_type,
            "raw_text": chunk.text,
        },
    )


def build_index(chunks: list[Chunk]):
    """(Re)build the Chroma collection from chunks. Returns the vector store."""
    from langchain_chroma import Chroma  # imported lazily so offline tests don't need it

    from .models import embeddings

    Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)
    emb = embeddings()

    # Clean rebuild: drop any existing collection so chunking/metric changes take
    # full effect instead of upserting into a stale collection.
    try:
        Chroma(
            collection_name=settings.collection_name,
            embedding_function=emb,
            persist_directory=settings.chroma_dir,
        ).delete_collection()
    except Exception:
        pass

    return Chroma.from_documents(
        documents=[_to_document(c) for c in chunks],
        ids=[c.chunk_id for c in chunks],
        embedding=emb,
        collection_name=settings.collection_name,
        persist_directory=settings.chroma_dir,
        collection_metadata={"hnsw:space": "cosine"},  # correct metric for text-embedding-3
    )


def get_vectorstore():
    """Open the existing persisted Chroma collection (for retrieval)."""
    from langchain_chroma import Chroma

    from .models import embeddings

    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings(),
        persist_directory=settings.chroma_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def ingest(docs_dir: str | None = None) -> dict:
    """Full pipeline: build chunks -> write manifest -> build the vector index.

    Also tracks the cost of *this* ingest run (the embedding spend, the only paid
    call here) and appends it to a timestamped log. Returns a stats dict.
    """
    chunks = build_chunks(docs_dir)
    if not chunks:
        raise RuntimeError("No chunks produced — is DOCS_DIR empty?")
    manifest = write_manifest(chunks)

    # Time the part that actually calls the embeddings API.
    t0 = time.perf_counter()
    build_index(chunks)
    build_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c.chunk_type] = by_type.get(c.chunk_type, 0) + 1
    sources = {c.source for c in chunks}

    # Embedding cost: tokens of exactly what was embedded (the contextual text).
    embed_texts = [c.embed_text for c in chunks]
    embed_tokens, token_source = count_tokens_for_texts(embed_texts, settings.embedding_model)
    embed_cost = round(embed_usd(embed_tokens, settings.embedding_model), 6)

    cost_log = log_ingest_cost(
        {
            "model": settings.embedding_model,
            "chunks": len(chunks),
            "files": len(sources),
            "embed_tokens": embed_tokens,
            "token_source": token_source,
            "usd": embed_cost,
            "duration_ms": build_ms,
            "by_type": by_type,
        }
    )

    return {
        "chunks": len(chunks),
        "files": len(sources),
        "by_type": by_type,
        "manifest": manifest,
        "chroma_dir": settings.chroma_dir,
        "embed_tokens": embed_tokens,
        "embed_usd": embed_cost,
        "token_source": token_source,
        "cost_log": cost_log,
    }
