"""Streamlit dashboard — the RAGAS evaluation triad, made legible (Post 2).

Reads the scorecard written by ``reliable-rag eval`` and renders:

* the three triad scores as metric cards + a bar chart,
* a per-question drill-down (answer, scores, retrieved contexts, flags),
* a "lowest-faithfulness" failure spotlight,
* the pipeline cost for the run, and
* the human 👍/👎 feedback collected in the Chainlit demo.

Run it:  ``streamlit run app/streamlit_dashboard.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# Make ``reliable_rag`` importable when run via `streamlit run` from the repo root.
try:
    from reliable_rag.feedback import load_feedback
except Exception:  # pragma: no cover - dashboard still works without the pkg
    def load_feedback():
        return []

st.set_page_config(page_title="Reliable RAG — RAGAS dashboard", layout="wide")

# Map RAGAS metric column names -> friendly triad labels.
METRIC_LABELS = {
    "llm_context_precision_without_reference": "Context Relevance",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevance",
    "response_relevancy": "Answer Relevance",
}

st.title("Reliable RAG — RAGAS evaluation")
st.caption(
    "Reference-free LLM-as-judge triad over the cnWave docs · "
    "Context Relevance (Query↔Context) · Faithfulness (Context↔Answer) · Answer Relevance (Query↔Answer)"
)

results_dir = Path(st.sidebar.text_input("Results directory", "eval/results"))
scorecard_path = results_dir / "scorecard.csv"
summary_path = results_dir / "summary.json"

if not scorecard_path.exists():
    st.warning(
        f"No scorecard at `{scorecard_path}`. Run the evaluation first:\n\n"
        "```\nreliable-rag eval\n```"
    )
    st.stop()

df = pd.read_csv(scorecard_path)
metric_cols = [c for c in df.columns if c in METRIC_LABELS]
summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

# --- Triad metric cards -----------------------------------------------------
st.subheader("The evaluation triad")
cols = st.columns(max(len(metric_cols), 1))
for col, metric in zip(cols, metric_cols):
    mean = df[metric].dropna().mean()
    col.metric(METRIC_LABELS[metric], f"{mean:.3f}" if pd.notna(mean) else "—")

# --- Bar chart --------------------------------------------------------------
if metric_cols:
    means = pd.DataFrame(
        {"metric": [METRIC_LABELS[c] for c in metric_cols], "score": [df[c].dropna().mean() for c in metric_cols]}
    )
    try:
        import plotly.express as px

        fig = px.bar(means, x="metric", y="score", range_y=[0, 1], text_auto=".2f", color="metric")
        fig.update_layout(showlegend=False, height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.bar_chart(means.set_index("metric"))

# --- Run cost ---------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Questions", summary.get("n_questions", len(df)))
c2.metric("Pipeline cost", f"${summary.get('pipeline_usd', float('nan')):.4f}")
c3.metric("Abstained", summary.get("abstained", int(df.get("abstained", pd.Series(dtype=bool)).sum())))
st.caption("Pipeline cost = the gated runs that produced these answers. RAGAS judge calls are billed separately.")

# --- Failure spotlight ------------------------------------------------------
if "faithfulness" in df.columns:
    st.subheader("⚠️ Lowest-faithfulness answers")
    worst = df.sort_values("faithfulness", na_position="first").head(3)
    for _, row in worst.iterrows():
        q = row.get("question", row.get("user_input", ""))
        st.markdown(f"**{q}**  ·  faithfulness `{row['faithfulness']:.2f}`  ·  flags: `{row.get('flags','')}`")

# --- Per-question drill-down -----------------------------------------------
st.subheader("Per-question detail")
for i, row in df.iterrows():
    q = row.get("question", row.get("user_input", f"row {i}"))
    scores = "  ".join(f"{METRIC_LABELS[c]} `{row[c]:.2f}`" for c in metric_cols if pd.notna(row[c]))
    with st.expander(f"{i + 1}. {q}"):
        st.markdown(scores or "_no scores_")
        st.markdown("**Answer**")
        st.write(row.get("answer", row.get("response", "")))
        if row.get("flags"):
            st.caption(f"flags: {row['flags']}")
        contexts = row.get("retrieved_contexts")
        if isinstance(contexts, str) and contexts:
            st.markdown("**Retrieved contexts**")
            st.code(contexts[:2000])

# --- Human feedback panel ---------------------------------------------------
st.subheader("👍 / 👎 Human feedback")
feedback = load_feedback()
if not feedback:
    st.info("No feedback yet — run the Chainlit demo (`chainlit run app/chainlit_app.py`) and rate some answers.")
else:
    fb = pd.DataFrame(feedback)
    up = int((fb["verdict"] == "up").sum())
    down = int((fb["verdict"] == "down").sum())
    f1, f2 = st.columns(2)
    f1.metric("👍 helpful", up)
    f2.metric("👎 not helpful", down)
    show_cols = [c for c in ["ts", "verdict", "query", "total_usd", "flags"] if c in fb.columns]
    st.dataframe(fb[show_cols].iloc[::-1], use_container_width=True, hide_index=True)
