"""RAGAS evaluation — the offline measurement harness (Post 2).

Implements the **reference-free evaluation triad** — no golden answers required:

* **Context Relevance**  -> ``LLMContextPrecisionWithoutReference`` (Query <-> Context)
* **Faithfulness**       -> ``Faithfulness`` (Context <-> Answer), judged by Opus
* **Answer Relevance**   -> ``ResponseRelevancy`` (Query <-> Answer)

We run each curated question through *our own* gated pipeline, hand RAGAS the
``(question, answer, retrieved_contexts)`` triples, and let an LLM judge score
them. Results are written as a CSV scorecard + a JSON summary that the Streamlit
dashboard reads.

ragas imports are deferred into the functions so the rest of the package (and
``--help``) doesn't pay for them.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from .config import settings
from .observability import configure_logging, console


def _load_questions(path: str) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    questions = data.get("questions", [])
    if not questions:
        raise RuntimeError(f"No questions found in {path} (expected a top-level `questions:` list).")
    # Normalise: allow bare strings or {question: ..., ground_truth: ...}
    out = []
    for item in questions:
        if isinstance(item, str):
            out.append({"question": item})
        elif isinstance(item, dict) and item.get("question"):
            out.append(item)
    return out


def _run_pipeline(questions: list[dict]):
    """Run every question through our gated pipeline -> (samples, metadata)."""
    from ragas import SingleTurnSample  # type: ignore

    from .pipeline import run_query

    samples, metas = [], []
    for i, item in enumerate(questions, 1):
        q = item["question"]
        console.print(f"[dim]({i}/{len(questions)}) {q}[/dim]")
        res = run_query(q)
        contexts = [c.text for c in res.kept_chunks]
        sample_kwargs = dict(user_input=q, response=res.answer, retrieved_contexts=contexts)
        if item.get("ground_truth"):
            sample_kwargs["reference"] = item["ground_truth"]
        samples.append(SingleTurnSample(**sample_kwargs))
        metas.append(
            {
                "question": q,
                "answer": res.answer,
                "n_contexts": len(contexts),
                "abstained": res.abstained,
                "faithful": bool(res.verification and res.verification.faithful),
                "flags": ",".join(res.flags),
                "pipeline_usd": res.total_usd,
            }
        )
    return samples, metas


def run_eval(
    questions_path: str = "eval/questions.yaml",
    out_dir: str = "eval/results",
    limit: int | None = None,
) -> dict:
    """Run the reference-free RAGAS triad; write scorecard + summary; return summary.

    ``limit`` evaluates only the first N questions (handy for a cheap smoke).
    """
    configure_logging()
    # require keys early with a friendly message
    settings.require_anthropic()
    settings.require_openai()
    from .models import export_keys, ragas_chat

    export_keys()  # ensure the RAGAS judges' clients find the keys regardless of call order

    from langchain_openai import OpenAIEmbeddings
    from ragas import EvaluationDataset, evaluate  # type: ignore
    from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore
    from ragas.llms import LangchainLLMWrapper  # type: ignore
    from ragas.metrics import (  # type: ignore
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig  # type: ignore

    questions = _load_questions(questions_path)
    if limit:
        questions = questions[:limit]
        console.print(f"[dim]Evaluating only the first {len(questions)} question(s).[/dim]")
    samples, metas = _run_pipeline(questions)

    # Judges: Opus is the strong judge (faithfulness + context precision);
    # Sonnet handles answer-relevancy; OpenAI provides the embeddings it needs.
    opus = LangchainLLMWrapper(ragas_chat(settings.judge_model))        # strong judge: Faithfulness + Context
    sonnet = LangchainLLMWrapper(ragas_chat(settings.generation_model))  # Answer Relevance
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=settings.embedding_model))

    metrics = [
        LLMContextPrecisionWithoutReference(llm=opus),  # "Context Relevance"
        Faithfulness(llm=opus),
        ResponseRelevancy(llm=sonnet, embeddings=emb),  # "Answer Relevance"
    ]

    console.print("[cyan]Scoring with RAGAS (LLM-as-judge)…[/cyan]")
    # Opus judging a long, multi-claim answer can exceed RAGAS's default 180s
    # per-metric budget; raise it and dial back concurrency to avoid rate-limit
    # induced slowdowns.
    run_config = RunConfig(timeout=600, max_workers=4)
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=metrics,
        llm=opus,
        embeddings=emb,
        run_config=run_config,
    )
    df = result.to_pandas()

    # Merge our pipeline metadata in by row order (RAGAS preserves sample order).
    if metas and len(df) == len(metas):
        for key in metas[0]:
            df[key] = [m[key] for m in metas]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "scorecard.csv", index=False)

    metric_cols = [c for c in df.columns if c in {
        "llm_context_precision_without_reference", "faithfulness", "answer_relevancy", "response_relevancy",
    }]
    summary = {
        "n_questions": len(questions),
        "means": {c: float(df[c].dropna().mean()) if df[c].notna().any() else None for c in metric_cols},
        "pipeline_usd": round(float(sum(m["pipeline_usd"] for m in metas)), 6),
        "abstained": int(sum(m["abstained"] for m in metas)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    df.to_json(out / "scorecard.json", orient="records", indent=2)

    _print_summary(summary, out)
    return summary


def _print_summary(summary: dict, out: Path) -> None:
    from rich.table import Table

    table = Table(title="RAGAS — evaluation triad (means)")
    table.add_column("metric")
    table.add_column("score", justify="right")
    label = {
        "llm_context_precision_without_reference": "Context Relevance",
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevance",
        "response_relevancy": "Answer Relevance",
    }
    for key, value in summary["means"].items():
        table.add_row(label.get(key, key), f"{value:.3f}" if value is not None else "—")
    console.print(table)
    console.print(
        f"[green]Wrote[/green] {out/'scorecard.csv'} · pipeline cost "
        f"${summary['pipeline_usd']:.4f} over {summary['n_questions']} questions "
        f"({summary['abstained']} abstained)."
    )


def generate_testset(size: int = 10, out_path: str = "eval/questions.generated.yaml") -> str:
    """(Optional) Generate a synthetic question set from the docs with RAGAS.

    Uses an LLM call per node, so it's gated behind a CLI flag. Best-effort: the
    TestsetGenerator API has shifted across ragas versions, so failures are
    reported clearly rather than crashing.
    """
    configure_logging()
    settings.require_anthropic()
    settings.require_openai()
    from .models import export_keys, ragas_chat

    export_keys()

    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore
    from ragas.llms import LangchainLLMWrapper  # type: ignore
    from ragas.testset import TestsetGenerator  # type: ignore

    from .ingest import build_chunks

    docs = [
        Document(page_content=c.text, metadata={"source": c.source, "chunk_id": c.chunk_id})
        for c in build_chunks()
    ]
    generator = TestsetGenerator(
        llm=LangchainLLMWrapper(ragas_chat(settings.generation_model)),
        embedding_model=LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=settings.embedding_model)),
    )
    try:
        dataset = generator.generate_with_langchain_docs(docs, testset_size=size)
    except Exception as exc:  # noqa: BLE001 - surface API drift clearly
        raise RuntimeError(
            f"RAGAS testset generation failed ({exc}). Check your ragas version's "
            "TestsetGenerator API."
        ) from exc

    df = dataset.to_pandas()
    col = "user_input" if "user_input" in df.columns else df.columns[0]
    items = [{"question": str(q)} for q in df[col].tolist()]
    Path(out_path).write_text(
        yaml.safe_dump({"questions": items}, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    console.print(f"[green]Generated[/green] {len(items)} questions -> {out_path}")
    return out_path
