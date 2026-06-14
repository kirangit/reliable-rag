"""Hybrid retrieval — dense (Chroma) + lexical (BM25), fused with RRF.

Why hybrid? Dense embeddings capture meaning ("how do I recover a failed node")
but can miss exact tokens (model names like ``V5K``, codes like ``MCS12``, API
paths). BM25 nails those literals but misses paraphrase. Reciprocal Rank Fusion
(RRF) combines the two ranked lists into one, so a chunk that scores well on
*either* signal surfaces.

We implement the fusion ourselves (rather than ``EnsembleRetriever``) for one
reason that matters to this project: it lets us attach a **score to every
returned chunk**, which the trace, the Grade gate, and the UI all display.

Both retrievers run over the chunk's ``embed_text`` (contextual header + raw
text) so the contextual-header trick helps lexical and dense search alike.
"""

from __future__ import annotations

import re
from functools import lru_cache

from .config import settings
from .ingest import get_vectorstore, load_manifest
from .schemas import RetrievedChunk

# Standard RRF constant; larger -> flatter weighting across ranks.
RRF_K = 60

# Chunk ids end in "#<sequential index within the file>" (e.g. "Device_Recovery::…#4").
_INDEX_RE = re.compile(r"#(\d+)$")


class HybridRetriever:
    """Loads the chunk manifest once, then serves dense / hybrid retrieval."""

    def __init__(self) -> None:
        self._chunks = {c.chunk_id: c for c in load_manifest()}
        self._bm25 = None  # built lazily (no API key needed)
        self._neighbor_map: dict[str, dict[int, str]] | None = None  # built lazily

    # -- lexical ---------------------------------------------------------
    def _ensure_bm25(self):
        if self._bm25 is None:
            from langchain_community.retrievers import BM25Retriever

            ids = list(self._chunks)
            texts = [self._chunks[i].embed_text for i in ids]
            metadatas = [{"chunk_id": i} for i in ids]
            self._bm25 = BM25Retriever.from_texts(texts, metadatas=metadatas)
        return self._bm25

    def _bm25_ranked_ids(self, query: str, k: int) -> list[str]:
        retriever = self._ensure_bm25()
        retriever.k = k
        return [d.metadata["chunk_id"] for d in retriever.invoke(query)]

    # -- dense -----------------------------------------------------------
    def _dense_scored(self, query: str, k: int) -> list[tuple[str, float]]:
        # Use raw distance (not langchain's relevance_score_fn) to avoid the
        # "relevance scores must be between 0 and 1" warning, and convert cosine
        # distance ([0,2], lower = closer) to a [0,1] score ourselves.
        store = get_vectorstore()
        pairs = store.similarity_search_with_score(query, k=k)
        return [
            (d.metadata["chunk_id"], max(0.0, min(1.0, 1.0 - float(distance))))
            for d, distance in pairs
        ]

    # -- public ----------------------------------------------------------
    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        k = k or settings.retrieve_k

        if settings.retrieval_mode == "dense":
            ranked = self._dense_scored(query, k)
            return [self._make(cid, score) for cid, score in ranked if cid in self._chunks]

        # hybrid: RRF over the two ranked lists
        dense = self._dense_scored(query, k)
        bm25_ids = self._bm25_ranked_ids(query, k)

        fused: dict[str, float] = {}
        for rank, (cid, _) in enumerate(dense):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, cid in enumerate(bm25_ids):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        if not ordered:
            return []
        top = ordered[0][1] or 1.0  # normalise fused scores to 0..1 for display
        return [self._make(cid, s / top) for cid, s in ordered if cid in self._chunks]

    def _make(self, chunk_id: str, score: float) -> RetrievedChunk:
        chunk = self._chunks[chunk_id]
        return RetrievedChunk(score=max(0.0, min(1.0, score)), **chunk.model_dump())

    # -- context-enrichment window ---------------------------------------
    @staticmethod
    def _index_of(chunk_id: str) -> int | None:
        m = _INDEX_RE.search(chunk_id)
        return int(m.group(1)) if m else None

    def _ensure_neighbor_map(self) -> dict[str, dict[int, str]]:
        """source -> {sequential_index -> chunk_id}, for neighbor lookup."""
        if self._neighbor_map is None:
            mapping: dict[str, dict[int, str]] = {}
            for cid, chunk in self._chunks.items():
                idx = self._index_of(cid)
                if idx is not None:
                    mapping.setdefault(chunk.source, {})[idx] = cid
            self._neighbor_map = mapping
        return self._neighbor_map

    def expand_with_neighbors(self, chunks, window: int | None = None) -> list[RetrievedChunk]:
        """Context-Enrichment Window: pad each chunk with its ±window same-source
        neighbors. Neighbors get score 0.0; the result is sorted into document order
        (source, index) so a stitched procedure/list reads correctly. Clamped to the
        same ``source`` — it never pulls indices from another file."""
        window = settings.context_window if window is None else window
        if window <= 0:
            return list(chunks)
        nmap = self._ensure_neighbor_map()
        seen = {c.chunk_id for c in chunks}
        out = list(chunks)
        for c in list(chunks):
            idx = self._index_of(c.chunk_id)
            if idx is None:  # no '#<index>' suffix -> no sequential neighbors
                continue
            src_map = nmap.get(c.source, {})
            for j in range(idx - window, idx + window + 1):
                if j != idx and (nid := src_map.get(j)) and nid not in seen:
                    seen.add(nid)
                    out.append(RetrievedChunk(score=0.0, **self._chunks[nid].model_dump()))
        out.sort(key=lambda c: (c.source, self._index_of(c.chunk_id) or 0))
        return out


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    """Cached singleton so the manifest + BM25 index are built only once."""
    return HybridRetriever()
