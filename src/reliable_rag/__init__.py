"""Reliable RAG — Grade / Verify / Trace gates + RAGAS evaluation.

A small, heavily-commented reference implementation of two ideas:

* **Reliable RAG** (runtime): the plain ``Query -> Retrieve -> Generate -> Answer``
  pipeline gains three diagnostic gates — **Grade** (drop irrelevant chunks
  before generation), **Verify** (check the answer is grounded in the chunks),
  and **Trace** (citations built from real retrieved chunk IDs, never fabricated).
* **RAGAS** (offline): a reference-free, LLM-as-judge evaluation triad —
  Context Relevance, Faithfulness, Answer Relevance.

The package is organised one-concept-per-module so it doubles as a tutorial.
Start at ``LEARN.md`` in the repo root.
"""

__version__ = "0.1.0"
