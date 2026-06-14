"""Gate logic — LLM calls are mocked, so these run offline."""

from __future__ import annotations

import reliable_rag.gates.generate as gen_mod
import reliable_rag.gates.grade as grade_mod
import reliable_rag.gates.verify as verify_mod
from reliable_rag.schemas import GradeResponse, RetrievedChunk, VerifyResponse


class FakeMsg:
    """Stand-in AIMessage carrying usage_metadata for cost accounting."""

    def __init__(self, content: str = ""):
        self.content = content
        self.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


class _FakeStructured:
    def __init__(self, parsed):
        self._parsed = parsed

    def invoke(self, _messages):
        return {"raw": FakeMsg(), "parsed": self._parsed, "parsing_error": None}


class FakeChat:
    def __init__(self, parsed=None, content: str = ""):
        self._parsed = parsed
        self._content = content

    def with_structured_output(self, _schema, include_raw: bool = False):
        return _FakeStructured(self._parsed)

    def invoke(self, _messages):
        return FakeMsg(self._content)


def _chunk(cid: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, text=f"text for {cid}", source="docs/x.md", score=0.8)


def test_grade_keeps_by_threshold(monkeypatch):
    chunks = [_chunk("c1"), _chunk("c2")]
    parsed = GradeResponse(
        grades=[
            {"chunk_id": "c1", "relevant": True, "score": 0.9, "reason": "directly answers"},
            {"chunk_id": "c2", "relevant": True, "score": 0.2, "reason": "on-topic but unhelpful"},
        ]
    )
    monkeypatch.setattr(grade_mod, "chat", lambda *a, **k: FakeChat(parsed=parsed))

    grades, raw = grade_mod.grade_chunks("q", chunks)
    kept = {g.chunk_id for g in grades if g.relevant}
    assert kept == {"c1"}  # 0.2 is below the default 0.5 threshold -> dropped
    assert raw.usage_metadata["input_tokens"] == 10


def test_grade_missing_chunk_is_dropped(monkeypatch):
    chunks = [_chunk("c1")]
    monkeypatch.setattr(grade_mod, "chat", lambda *a, **k: FakeChat(parsed=GradeResponse(grades=[])))
    grades, _ = grade_mod.grade_chunks("q", chunks)
    assert grades[0].relevant is False  # silence never keeps a chunk


def test_verify_flags_unsupported(monkeypatch):
    chunks = [_chunk("c1")]
    parsed = VerifyResponse(
        claims=[
            {"claim": "supported", "grounded": True, "supporting_chunk_ids": ["c1"], "reason": "in context"},
            {"claim": "hallucinated", "grounded": False, "supporting_chunk_ids": [], "reason": "not in context"},
        ]
    )
    monkeypatch.setattr(verify_mod, "chat", lambda *a, **k: FakeChat(parsed=parsed))

    verification, _ = verify_mod.verify_answer("q", "answer", chunks)
    assert verification.faithful is False
    assert verification.unsupported_claims == ["hallucinated"]
    assert abs(verification.score - 0.5) < 1e-9


def test_generate_returns_cited_text(monkeypatch):
    chunks = [_chunk("c1")]
    monkeypatch.setattr(gen_mod, "chat", lambda *a, **k: FakeChat(content="Bridge ports do L2 [c1]."))
    text, raw = gen_mod.generate_answer("q", chunks)
    assert "[c1]" in text
    assert raw.usage_metadata["output_tokens"] == 5
