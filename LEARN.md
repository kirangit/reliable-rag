# LEARN — a guided tour

This repo is meant to be *read*. Below is the path I'd take a newcomer through,
from the two ideas to the exact code that implements them. Every module opens
with a docstring saying what it does and which idea it implements.

## 0. The two ideas (5 minutes)

1. **Reliable RAG** — the plain `Query → Retrieve → Generate → Answer` pipeline
   has no checkpoints, so a wrong answer is undebuggable. Add three gates:
   **Grade** (drop irrelevant chunks before generating), **Verify** (check the
   answer is grounded), **Trace** (cite *real* retrieved chunk IDs).
2. **RAGAS** — evaluate the whole pipeline with an LLM judge, no golden answers:
   **Context Relevance**, **Faithfulness**, **Answer Relevance**.

## 1. Configuration & data shapes

- [`config.py`](src/reliable_rag/config.py) — every knob (models, thresholds,
  retrieval mode, tracing) in one settings object. Read this first; the rest of
  the code refers back to it.
- [`schemas.py`](src/reliable_rag/schemas.py) — the records that flow through the
  system (`Chunk`, `RetrievedChunk`, `ChunkGrade`, `Verification`, `Citation`,
  `TraceEvent`, `GateCost`) and the `RAGState` the graph threads. Also the
  structured-output schemas the gates force the LLM to return.

## 2. Getting the corpus in (ingestion)

- [`chunkers/markdown_header_chunker.py`](src/reliable_rag/chunkers/markdown_header_chunker.py)
  — header-aware splitting that keeps tables and ASCII diagrams whole. See
  [CHUNKING.md](CHUNKING.md) for *why* this beats fixed-size windows here (and why
  tabular CSVs are deliberately out of scope).
- [`ingest.py`](src/reliable_rag/ingest.py) — chunks → a **Chroma** index (embeds
  the *contextual* text) + a **manifest** (`chunks.jsonl`, raw text for BM25 and
  citations). Run with `reliable-rag ingest`.

## 3. Retrieval

- [`retrieval.py`](src/reliable_rag/retrieval.py) — hybrid **dense (Chroma) + BM25**
  fused with Reciprocal Rank Fusion, so a chunk that wins on *either* meaning or
  exact tokens (model names, API paths) surfaces. Every chunk keeps a score.

## 4. The gates (the heart)

Read these in pipeline order:

- [`gates/grade.py`](src/reliable_rag/gates/grade.py) — **Grade**. One structured
  call scores every chunk; the threshold decides what's kept.
- [`gates/generate.py`](src/reliable_rag/gates/generate.py) — generation that uses
  only kept chunks (clean prose; sources tracked separately by Verify/Trace).
- [`gates/verify.py`](src/reliable_rag/gates/verify.py) — **Verify**. Opus checks
  the answer claim-by-claim against the context.
- [`trace.py`](src/reliable_rag/trace.py) — **Trace**. Builds grounded citations
  from the chunks Verify confirmed support the answer; fabricated ones can't survive.

## 5. Wiring it together

- [`graph.py`](src/reliable_rag/graph.py) — the **LangGraph** `StateGraph`. The
  nodes are the gates; the **conditional edges** are the reliability policy:
  rewrite-and-retry once on thin retrieval, **abstain** rather than hallucinate,
  **regenerate** once when Verify finds unsupported claims.
- [`pipeline.py`](src/reliable_rag/pipeline.py) — `run_query()`, the single
  entrypoint the CLI and both UIs share.

## 6. Seeing and measuring

- [`cost.py`](src/reliable_rag/cost.py) — per-gate token + dollar accounting
  ("count the cost"). Token counts come free off each response.
- [`observability.py`](src/reliable_rag/observability.py) — per-run trace logs
  (`runs/*.json`), the `rich` console view, and the optional Phoenix/LangSmith
  hookup.
- [`feedback.py`](src/reliable_rag/feedback.py) — the 👍/👎 store.

## 7. Evaluation

- [`evaluate.py`](src/reliable_rag/evaluate.py) — runs the curated questions
  ([`eval/questions.yaml`](eval/questions.yaml)) through the pipeline and scores
  them with the **RAGAS** triad. `reliable-rag eval`.

## 8. The demos

- [`app/chainlit_app.py`](app/chainlit_app.py) — the gated chat: each gate is a
  collapsible **step**, citations are grounded, cost is shown, 👍/👎 is logged.
- [`app/streamlit_dashboard.py`](app/streamlit_dashboard.py) — the RAGAS triad
  dashboard + the human-feedback panel.

## 9. Tests

- [`tests/`](tests) — fully offline (mocked LLMs). `test_ingest.py` pins the
  chunker invariants, `test_gates.py` the gate logic, `test_graph.py` the
  end-to-end control flow (happy path + abstain). Run `pytest`.

## Advanced ideas (deliberately off the default path)

- **Parent-document retrieval** — index small sections for precise matching but
  pass the full parent section to the generator. Improves completeness; kept off
  to keep the default simple.
- **Contextual *retrieval*** (Anthropic-style) — have Claude write a 1–2 sentence
  situating blurb per chunk at ingest. Higher quality than the cheap contextual
  *headers* we use, but one LLM call per chunk. See [CHUNKING.md](CHUNKING.md).
