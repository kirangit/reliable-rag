"""Context-Enrichment Window (neighbor expansion) — offline, no API."""

from __future__ import annotations

import reliable_rag.retrieval as R
from reliable_rag.schemas import Chunk, RetrievedChunk


def _manifest():
    chunks = [Chunk(chunk_id=f"DocA::s#{i}", text=f"a{i}", source="docs/DocA.md") for i in range(5)]
    chunks.append(Chunk(chunk_id="DocB::s#0", text="b0", source="docs/DocB.md"))
    # a chunk_id WITHOUT a '#<index>' suffix -> treated as having no neighbors
    chunks.append(Chunk(chunk_id="Gloss::intro", text="g", source="docs/Gloss.md"))
    return chunks


def _retriever(monkeypatch):
    monkeypatch.setattr(R, "load_manifest", lambda: _manifest())
    return R.HybridRetriever()


def test_expands_one_neighbor_each_side(monkeypatch):
    r = _retriever(monkeypatch)
    kept = [RetrievedChunk(score=1.0, chunk_id="DocA::s#2", text="a2", source="docs/DocA.md")]
    out = r.expand_with_neighbors(kept, window=1)
    assert [c.chunk_id for c in out] == ["DocA::s#1", "DocA::s#2", "DocA::s#3"]  # ±1, doc order


def test_clamps_to_same_source(monkeypatch):
    r = _retriever(monkeypatch)
    kept = [RetrievedChunk(score=1.0, chunk_id="DocA::s#0", text="a0", source="docs/DocA.md")]
    ids = {c.chunk_id for c in r.expand_with_neighbors(kept, window=1)}
    assert ids == {"DocA::s#0", "DocA::s#1"}  # no left neighbor at index 0; never crosses to DocB
    assert "DocB::s#0" not in ids


def test_window_zero_is_noop(monkeypatch):
    r = _retriever(monkeypatch)
    kept = [RetrievedChunk(score=1.0, chunk_id="DocA::s#2", text="a2", source="docs/DocA.md")]
    assert [c.chunk_id for c in r.expand_with_neighbors(kept, window=0)] == ["DocA::s#2"]


def test_ids_without_index_have_no_neighbors(monkeypatch):
    r = _retriever(monkeypatch)
    kept = [RetrievedChunk(score=1.0, chunk_id="Gloss::intro", text="g", source="docs/Gloss.md")]
    # no '#<index>' suffix -> no sequential neighbors to expand into.
    assert [c.chunk_id for c in r.expand_with_neighbors(kept, window=1)] == ["Gloss::intro"]
