# CHUNKING — why structure-aware, not fixed-size

Chunking is where most RAG quality is won or lost. For *this* corpus the right
strategy is clear-cut, and it's worth understanding why.

## The corpus

The cnWave docs are **highly structured**: a clean `#`→`####` heading hierarchy,
and lots of **atomic blocks** — markdown tables (LED-state, Rx-sensitivity),
ASCII diagrams inside code fences (packet-flow figures), and discrete REST
endpoints (the API catalog is one endpoint per `##`).

## What we do

**Markdown → header-aware split** (`chunkers/markdown_header_chunker.py`).
Split on headings; one section = one chunk; capture the **breadcrumb**
(`Topology Management > Polarity`) as metadata. Only sections over
`MAX_CHUNK_CHARS` get a second, **block-aware** pass that never breaks a code
fence or a table (oversized tables split by rows, repeating the header row).

> **Tabular data is out of scope (on purpose).** The cnWave set also ships
> throughput CSVs, but an exact table lookup — *"V5K MCS12 downlink throughput"* —
> is a **structured** query: it wants *all* matching rows via a metadata filter,
> not a semantic top-k that can't enumerate ~24 near-identical rows under a small
> `k`. That's a different retrieval mode (structured / self-query retrieval), so
> this project stays **markdown-only** instead of faking tabular lookups with
> embeddings. Adding a metadata-filter retriever for CSVs is a natural extension.

## Why not fixed-size windows?

A blind 512/1000-character window would:

- **cut ASCII diagrams in half** (the packet-flow figures span many lines),
- **tear table rows from their header row** — a row like `Green | Red | V5000 | Negotiated at ≤ 1 Gbps` is meaningless without the header that says which column is which,
- **merge unrelated sibling sections** into one chunk, and
- **drop the section title** that tells you what the chunk is even about.

All of that degrades the **Grade** gate (it can't judge a half-table) and the
**citations** (a chunk with no section identity).

## Why not semantic (embedding-similarity) chunking?

These docs already carry author-written structure that is a *better* signal than
similarity guesses — and header-aware splitting is deterministic, cheaper, and
hands us a free, human-readable citation path. Semantic chunking shines on
unstructured walls of text (transcripts, scraped prose), not here.

## Contextual chunk headers (the book's trick)

Adopted from *RAG Made Simple*. A chunk detached from its document loses its
identity, so **before embedding** we prepend the breadcrumb:

**Before** (what a naive pipeline embeds — and what a naive splitter might cut):

```
| Left LED | Right LED | Product               | Meaning                |
|----------|-----------|-----------------------|------------------------|
| Green    | Red       | V5000 / V3000 / V2000 | Negotiated at ≤ 1 Gbps |
```

A query like *"what does a green/red main-port LED mean on a V5000?"* may not
match this well — the chunk never says "LED" or "Main Port".

**After** (what we actually embed — `Chunk.embed_text`):

```
Document: LED_Indicators > Main Port – Combined LED States

| Left LED | Right LED | Product               | Meaning                |
|----------|-----------|-----------------------|------------------------|
| Green    | Red       | V5000 / V3000 / V2000 | Negotiated at ≤ 1 Gbps |
```

One line of context, and the chunk carries its identity into the vector — better
dense *and* BM25 matching. We keep the **raw** text separately (`Chunk.text`) so
the citation snippet shows the real content, not the prepended header.

## Advanced upgrade (off by default)

The cheap version above prepends a deterministic breadcrumb. A higher-quality
variant is **contextual *retrieval*** (Anthropic's technique): have Claude write
a 1–2 sentence blurb situating each chunk in the whole document, and prepend
*that*. Better matching, but one LLM call per chunk at ingest — so it's noted
here rather than enabled by default.

## See it yourself

```bash
python -c "from reliable_rag.ingest import build_chunks; \
cs=build_chunks(); \
ex=[c for c in cs if c.chunk_type=='table'][0]; \
print(ex.header_path); print(ex.embed_text[:200])"
```
