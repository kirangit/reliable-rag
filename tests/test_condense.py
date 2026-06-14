"""Tests for the conversational condense step (condense.py).

The LLM is mocked, so these run offline and cost nothing — they pin the two
behaviours that matter: no history = free passthrough; history = a standalone
rewrite (with surrounding quotes stripped).
"""

from __future__ import annotations

import reliable_rag.condense as condense_mod


class _FakeMsg:
    def __init__(self, content: str) -> None:
        self.content = content
        self.usage_metadata = {"input_tokens": 20, "output_tokens": 8}


class _FakeChat:
    def __init__(self, reply: str) -> None:
        self._reply = reply

    def invoke(self, _messages):
        return _FakeMsg(self._reply)


def test_no_history_is_passthrough_and_free():
    # Nothing to resolve -> return the query untouched and make NO LLM call
    # (raw message is None, so the caller counts zero cost).
    out, raw = condense_mod.condense_question([], "How do I recover an onboard E2E node?")
    assert out == "How do I recover an onboard E2E node?"
    assert raw is None


def test_rewrites_followup_using_history(monkeypatch):
    monkeypatch.setattr(
        condense_mod, "chat",
        lambda *a, **k: _FakeChat("How long does recovering an onboard E2E node take?"),
    )
    history = [
        ("user", "How do I recover an onboard E2E node?"),
        ("assistant", "Use the recovery-mode procedure ..."),
    ]
    out, raw = condense_mod.condense_question(history, "how long does it take?")
    assert "recover" in out.lower()
    assert raw is not None  # an LLM call happened -> cost will be counted


def test_strips_surrounding_quotes(monkeypatch):
    # Models often wrap the rewrite in quotes; we strip them so the query is clean.
    monkeypatch.setattr(
        condense_mod, "chat",
        lambda *a, **k: _FakeChat('"What are the LED states?"'),
    )
    out, _ = condense_mod.condense_question([("user", "tell me about LEDs")], "and the states?")
    assert out == "What are the LED states?"
