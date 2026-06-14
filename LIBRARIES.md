# LIBRARIES & TECHNIQUES — the calls behind each step

[LEARN.md](LEARN.md) tours *our* modules; [CHUNKING.md](CHUNKING.md) explains the
chunking choice. **This file is the reference for how each technique is actually
implemented** — for every step: the call, *why* it's there, the parameters that
matter, and the gotcha we hit.

Ordered by the pipeline (ingest → retrieve → gates → eval → ops), so it doubles as
a flow map. Markers:

> 📚 = a third-party library call · ✍️ = **custom** code (and why) · 🔧 = a parameter that matters · ⚠️ = a real gotcha we debugged.

Worth knowing up front: the *reliability-defining* pieces — **chunking, RRF fusion,
the Context-Enrichment Window, contextual headers** — are deliberately **custom**.
Off-the-shelf splitters/retrievers would break tables, lose the header path, or
hide per-chunk scores. The libraries do the heavy lifting (LLM calls, vector math,
the graph runtime); the judgment lives in our code.

## Stack at a glance

| Step | Implementation | Lives in |
|---|---|---|
| Chunking | ✍️ custom (regex, fence/table-aware) | `chunkers/markdown_header_chunker.py` |
| Contextual headers | ✍️ custom (`Chunk.embed_text`) | `schemas.py` |
| Vector index | 📚 `langchain-chroma` + `chromadb` | `ingest.py` |
| Embeddings | 📚 `langchain-openai` | `models.py` |
| Dense + lexical retrieval | 📚 Chroma + `BM25Retriever` | `retrieval.py` |
| RRF fusion | ✍️ custom | `retrieval.py` |
| Context-Enrichment Window | ✍️ custom | `retrieval.py` |
| Gates (LLM) | 📚 `langchain-anthropic` structured output | `gates/` |
| Orchestration | 📚 `langgraph` | `graph.py` |
| Eval | 📚 `ragas` 0.3.x | `evaluate.py` |
| Cost | 📚 `usage_metadata` + `tiktoken` | `cost.py` |
| Config | 📚 `pydantic-settings` | `config.py` |
| UIs / tracing | 📚 `chainlit`, `streamlit`, LangSmith/Phoenix | `app/`, `observability.py` |

---

## 1. Chunking — ✍️ custom, fence/table-aware

```python
chunk_markdown_file(path, max_chars=1200, overlap=100, max_level=2)   # chunkers/markdown_header_chunker.py
iter_sections(text, max_level=2)      # split on headings (fence-aware)
to_blocks(body)                       # atomic blocks: a fence / a table = one block
split_section_body(body, max_chars, overlap)   # block-packed size guard
```

- **Why custom (not a library splitter):** LangChain's `MarkdownHeaderTextSplitter`
  doesn't keep tables/code-fences atomic, and `RecursiveCharacterTextSplitter` is
  blind to structure — both would slice an LED-state table or an ASCII packet diagram
  in half and drop the section title. We need three things at once: split on headings,
  **never break a fence or table**, and carry the **breadcrumb**. So it's ~250 lines of
  `re` over `pathlib` — *intentionally dependency-free* so it unit-tests offline.
- 🔧 **`max_level=2`** — split only on `#`/`##`; deeper headings (`###`…) stay *in* the
  body so a procedure with many sub-headings stays one coherent chunk. (This is the
  knob behind the `Overview.md` fix — promoting `### cnWave Models` to `##` made it its
  own chunk.) 🔧 `max_chars=1200` triggers the size guard; `overlap=100` is added only
  on a last-resort character split.
- **Atomicity:** `to_blocks` treats each fenced block / contiguous table as one unit;
  `_split_table` splits an oversized table *by rows, repeating the header+separator
  row* so each piece stays self-describing. `classify()` tags `table|api|diagram|prose`.
- **Stable ids:** `chunk_id = "{stem}::{slug(header_path)}#{gidx}"` — the running `gidx`
  also encodes sequential order (CEW relies on it, §4).

## 2. Contextual chunk headers — ✍️ custom (`Chunk.embed_text`)

- Before embedding, we prepend the breadcrumb: `"Document: Overview > cnWave Models\n\n"`
  + the raw text. A detached chunk (a bare table row) then carries its identity into
  the vector — better dense *and* BM25 matching.
- 🔧 Two fields, one chunk: **`embed_text`** (header + body) is what we index;
  **`text`** (raw) is what citations show. The header helps retrieval without polluting
  the quoted snippet.

## 3. Vector index — 📚 `langchain-chroma` + `chromadb` (`ingest.py`)

```python
Chroma.from_documents(documents, ids=[...], embedding=emb,
    collection_name=..., persist_directory=...,
    collection_metadata={"hnsw:space": "cosine"})
Document(page_content=chunk.embed_text, metadata={..., "raw_text": chunk.text})
OpenAIEmbeddings(model="text-embedding-3-small")   # models.py
```

- 🔧 **`collection_metadata={"hnsw:space":"cosine"}`** — `text-embedding-3` is
  cosine-normalized, so cosine is the right metric. ⚠️ Default L2 makes
  `*_with_relevance_scores` warn *"scores must be between 0 and 1."*
- **Clean rebuild:** `delete_collection()` before `from_documents`, so chunking/metric
  changes take full effect (no upsert into a stale collection).
- **Why OpenAI for embeddings:** Anthropic has no embeddings endpoint;
  `text-embedding-3-small` is ~$0.02/1M. We embed `embed_text`, keep `raw_text` in
  metadata for citations.

## 4. Retrieval — dense + lexical + fusion + window

```python
store.similarity_search_with_score(query, k)        # 📚 dense -> (doc, distance)
BM25Retriever.from_texts(texts, metadatas); r.k = k; r.invoke(query)   # 📚 lexical
# RRF + CEW are ✍️ custom, below
```

- **Dense** 📚: `similarity_search_with_score`, then we convert cosine **distance** to a
  score ourselves: `score = clamp(1 - distance, 0, 1)` (avoids the relevance-score
  warning). **BM25** 📚: built lazily, in-memory, **no API key**, over the same
  `embed_text`. Dense catches paraphrase; BM25 catches exact tokens (`V5000`, `MCS12`).
- **RRF fusion** ✍️: we fuse the two ranked lists ourselves (not `EnsembleRetriever`)
  so every returned chunk keeps a **score** the trace/Grade/UI display:
  `fused = Σ 1/(RRF_K + rank)`, 🔧 `RRF_K = 60`. `RETRIEVAL_MODE=dense` skips BM25+RRF.
- **Context-Enrichment Window (CEW)** ✍️ — `expand_with_neighbors(chunks, window)`:
  after Grade, pad each kept chunk with its ±`window` **same-document** neighbors so a
  hit pulls in adjacent steps/list-items. 🔧 `context_window` (default 1; 0 = off).
  Implementation: `_INDEX_RE = re.compile(r"#(\d+)$")` reads the sequential index from
  the `chunk_id`; a `source -> {index -> chunk_id}` map finds neighbors; ⚠️ **clamped to
  the same `source`** so it never pulls a different file's indices. No library — pure
  Python over the manifest.

## 5. Gates (LLM) — 📚 `langchain-anthropic` structured output (`gates/`)

```python
ChatAnthropic(model=model, max_tokens=max_tokens, timeout=120)        # models.py
chat(...).with_structured_output(Schema, include_raw=True).invoke([("system",...),("human",...)])
```

- **Why structured output:** `.with_structured_output(GradeResponse | VerifyResponse)`
  forces Claude to return a tool call matching a Pydantic schema → **deterministic,
  parseable** gate decisions, no string parsing. 🔧 `include_raw=True` returns
  `{"raw","parsed","parsing_error"}` so we keep the raw message (tokens/cost) *and*
  detect a bad parse instead of crashing (the gates fail-open on `parsing_error`).
- ⚠️ **Claude 4.x rejects `temperature`/`top_p`/`top_k`/`budget_tokens`** (400) — we
  never pass them; structured output gives the determinism we'd want from `temp=0`.
- 🔧 **`max_tokens`** is a hard ceiling. ⚠️ Verify restates every claim, so 2048 truncated
  long answers → parse fail → wrongly "unverified." → `verify_max_tokens=8192`. (A high
  ceiling is free — billed only for tokens generated.)
- 🔧 `timeout=120`; clients are `@lru_cache`-d on `(model, max_tokens)`.

## 6. Orchestration — 📚 `langgraph` (`graph.py`)

```python
g = StateGraph(RAGState)
g.add_conditional_edges("grade", route_after_grade, {"generate":..., "rewrite":..., "abstain":...})
g.add_conditional_edges("verify", route_after_verify, {"finalize":..., "regenerate":...})
graph.invoke(init, config={"recursion_limit": 15})        # sync (CLI)
graph.astream(init, stream_mode="updates", config=...)    # live (Chainlit)
```

- **Why a graph:** the gates branch — Grade can loop via `rewrite`, Verify via
  `regenerate`. `add_conditional_edges(node, router, mapping)` encodes that declaratively.
- 🔧 **`RAGState`** (`TypedDict`, `schemas.py`) declares `costs`/`flags`/`trace` as
  `Annotated[list, operator.add]` → each node **appends** (so cost/trace accumulate
  across loop iterations); scalar channels overwrite.
- 🔧 **`recursion_limit=15`** backstops the loops. 🔧 **`stream_mode="updates"`** yields
  each node's output as it finishes — the live Chainlit steps.

## 7. Conversational condense — 📚 one small Claude call (`condense.py`)

- `condense_question(history, query)` rewrites a follow-up into a standalone query
  *before* retrieval (history-aware retrieval, "condense-only"). One `chat(...).invoke`
  on `generation_model`, `max_tokens=256`. 🔧 Gated by `CONVERSATIONAL` (default off);
  no history → no call, no cost.

## 8. Evaluation — 📚 `ragas` 0.3.x (`evaluate.py`)

```python
LangchainLLMWrapper(ragas_chat(judge_model)); LangchainEmbeddingsWrapper(embeddings())
EvaluationDataset(samples=[SingleTurnSample(user_input=q, response=ans, retrieved_contexts=[...])])
evaluate(ds, metrics=[LLMContextPrecisionWithoutReference(), Faithfulness(), ResponseRelevancy()],
         llm=..., embeddings=..., run_config=RunConfig(timeout=600, max_workers=4))
```

- **Reference-free triad** (no golden answers): `LLMContextPrecisionWithoutReference` =
  Context Relevance, `Faithfulness` = grounding, `ResponseRelevancy` = Answer Relevance
  (the only one needing embeddings). Wrappers adapt our clients to RAGAS.
- ⚠️ RAGAS injects `temperature` → hand it `ragas_chat()` (the stripping subclass).
  ⚠️ Opus judging is slow → `RunConfig(timeout=600)` (default 180s times out).

## 9. Cost — 📚 `usage_metadata` + `tiktoken` (`cost.py`)

- **No extra API calls:** each response carries `msg.usage_metadata
  = {"input_tokens","output_tokens",...}`; `cost.py` × a per-model `PRICING` table
  (tolerates date-suffixed ids). Embeddings return no counts → **`tiktoken`** counts them
  locally at ingest.

## 10. Observability — 📚 LangSmith / Phoenix (`observability.py`)

- **LangSmith (hosted):** no code calls — auto-traced when env is set. We bridge from
  settings: `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`. ⚠️ The
  SDK reads `os.environ`, **not** our `.env` → `setup_tracing()` exports it (Chainlit
  auto-loads `.env`; the CLI needed the explicit bridge).
- **Phoenix (local):** `phoenix.otel.register(project_name=..., auto_instrument=True,
  batch=True)`. ⚠️ `batch=True` is essential — the default sync exporter **blocks the
  pipeline for minutes** retrying a dead collector.

## 11. Config — 📚 `pydantic-settings` (`config.py`)

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    retrieve_k: int = 8        # env RETRIEVE_K overrides
```
- Field ↔ `UPPER_CASE` env var; shell var > `.env` > default. 🔧 `extra="ignore"` lets
  unrelated env vars coexist. One typed import for every knob.

## 12. UIs — 📚 `chainlit`, `streamlit` (`app/`)

- **Chainlit:** `@cl.on_message`/`@cl.on_chat_start`; **`cl.Step(name, type)`** context
  manager renders each gate as a collapsible step; `msg.stream_token(piece)` streams the
  answer; `cl.Action` + `@cl.action_callback` drive 👍/👎; `cl.user_session` holds chat
  history. **Streamlit:** reads `eval/results/*` → triad metric cards + charts + drill-down.

---

## Gotcha index (the landmines, in one place)

| Symptom | Cause | Fix |
|---|---|---|
| `temperature` 400 error | Claude 4.x rejects sampling params | don't pass them; strip in `ragas_chat` |
| Long answer flagged "unverified" | Verify output truncated at `max_tokens` | `verify_max_tokens=8192` |
| "relevance scores must be between 0 and 1" | wrong Chroma distance metric | `hnsw:space=cosine` + `1-distance` |
| Pipeline hangs minutes with Phoenix | sync span export retrying a dead collector | `register(batch=True)` |
| LangSmith shows no project / traces | key in `.env` never reached `os.environ` | bridge it in `setup_tracing()` |
| RAGAS times out on Opus | default 180s too short | `RunConfig(timeout=600)` |
| Section retrieves under the wrong topic | `###`/`####` folded into a giant `##` section | promote it to `##` (`max_level=2` splits only `#`/`##`) |
| Table/diagram split mid-way | fixed-size splitter | custom fence/table-atomic chunker |
| Grade drops good chunks on messy queries | cheap grader brittle to phrasing | Sonnet for Grade; rewrite gate recovers |
