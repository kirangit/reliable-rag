"""End-to-end pipeline — retriever + gates mocked, so it runs offline.

These exercise the graph's control flow: the happy path (Grade keeps chunks ->
Generate -> Verify faithful -> Trace builds citations) and the abstain path
(Grade keeps nothing -> rewrite -> still nothing -> abstain, never generating).
"""

from __future__ import annotations

import reliable_rag.graph as G
from reliable_rag.schemas import ChunkGrade, RetrievedChunk, Verification


class _FakeMsg:
    def __init__(self, content: str = ""):
        self.content = content
        self.usage_metadata = {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12}


class _FakeChat:
    def __init__(self, content: str = ""):
        self._content = content

    def invoke(self, _messages):
        return _FakeMsg(self._content)


class _FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, _query, _k=None):
        return self._chunks

    def expand_with_neighbors(self, chunks, window=None):
        return list(chunks)  # no neighbor expansion in the fake


def _chunks(n: int):
    return [
        RetrievedChunk(chunk_id=f"c{i}", text=f"context {i}", source="docs/x.md", header_path="H", score=0.9)
        for i in range(n)
    ]


def test_happy_path(monkeypatch):
    chunks = _chunks(3)
    monkeypatch.setattr(G, "get_retriever", lambda: _FakeRetriever(chunks))
    monkeypatch.setattr(
        G,
        "grade_chunks",
        lambda q, cs: ([ChunkGrade(chunk_id=c.chunk_id, relevant=True, score=0.9, reason="ok") for c in cs], None),
    )
    monkeypatch.setattr(
        G, "generate_answer", lambda q, kept, feedback=None: ("Bridge ports do L2 [c0][c1].", None)
    )
    monkeypatch.setattr(
        G, "verify_answer", lambda q, a, kept: (Verification(faithful=True, score=1.0), None)
    )
    monkeypatch.setattr("reliable_rag.pipeline.log_run", lambda result: None)

    from reliable_rag.pipeline import run_query

    res = run_query("what are bridge ports?")
    assert not res.abstained
    assert res.answer.startswith("Bridge ports")
    assert {c.chunk_id for c in res.citations} >= {"c0", "c1"}  # citations resolve to real chunks
    assert res.verification.faithful
    assert res.total_usd >= 0.0


def test_abstains_when_nothing_relevant(monkeypatch):
    chunks = _chunks(3)
    monkeypatch.setattr(G, "get_retriever", lambda: _FakeRetriever(chunks))
    monkeypatch.setattr(
        G,
        "grade_chunks",
        lambda q, cs: ([ChunkGrade(chunk_id=c.chunk_id, relevant=False, score=0.1, reason="irrelevant") for c in cs], None),
    )
    monkeypatch.setattr(G, "chat", lambda *a, **k: _FakeChat("rewritten query"))  # used by rewrite_node
    monkeypatch.setattr(G, "generate_answer", lambda *a, **k: ("should not be produced", None))
    monkeypatch.setattr("reliable_rag.pipeline.log_run", lambda result: None)

    from reliable_rag.pipeline import run_query

    res = run_query("what is the capital of france?")
    assert res.abstained
    assert "abstained" in res.flags
    assert res.citations == []
