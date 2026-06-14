# Reliable RAG — cnWave docs

This chat runs a **Reliable RAG** pipeline with three diagnostic gates:

- 🔶 **Grade** — irrelevant chunks are dropped *before* generation.
- 🔶 **Verify** — the answer is checked claim-by-claim against the retrieved context.
- 🔶 **Trace** — citations are built from the *real* retrieved chunk IDs, never fabricated.

Every answer shows these gates as expandable **steps**, lists **grounded citations**,
and reports its **token + dollar cost**. If the docs don't contain the answer, the
pipeline **abstains** rather than guessing.

Ask about deployment, node recovery, LED indicators, packet flow, the cnMaestro API,
polarity rules, or beam scanning. 👍/👎 each answer to log feedback.
