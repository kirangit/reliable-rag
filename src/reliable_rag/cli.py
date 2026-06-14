"""Command-line interface: ``ingest`` · ``ask`` · ``eval`` · ``gen-testset``.

Run via the installed entry point (``reliable-rag <cmd>``) or as a module
(``python -m reliable_rag.cli <cmd>``). Heavy imports are deferred into each
command so ``--help`` stays instant and a missing key only errors the command
that needs it.
"""

from __future__ import annotations

import typer

from .observability import configure_logging, console

app = typer.Typer(add_completion=False, help="Reliable RAG + RAGAS over the cnWave docs.")


@app.command()
def ingest() -> None:
    """Chunk DOCS_DIR, build the Chroma index, and write the chunk manifest."""
    configure_logging()
    from .ingest import ingest as run_ingest

    try:
        stats = run_ingest()
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    by_type = ", ".join(f"{k}={v}" for k, v in sorted(stats["by_type"].items()))
    console.print(
        f"[green]Ingested[/green] {stats['chunks']} chunks from {stats['files']} files "
        f"({by_type}).\n  index:    {stats['chroma_dir']}\n  manifest: {stats['manifest']}\n"
        f"  embedding cost: [bold]${stats['embed_usd']:.4f}[/bold] "
        f"({stats['embed_tokens']:,} tokens, {stats['token_source']}) -> logged to {stats['cost_log']}"
    )


@app.command()
def ask(question: str = typer.Argument(..., help="The question to answer.")) -> None:
    """Answer a question through the Grade -> Generate -> Verify -> Trace pipeline."""
    configure_logging()
    from .observability import render_run
    from .pipeline import run_query

    try:
        result = run_query(question)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    render_run(result)


@app.command("eval")
def eval_cmd(
    questions: str = typer.Option("eval/questions.yaml", help="YAML question set."),
    out: str = typer.Option("eval/results", help="Output directory for the scorecard."),
    limit: int = typer.Option(0, help="Only evaluate the first N questions (0 = all)."),
) -> None:
    """Run the reference-free RAGAS triad and write a scorecard."""
    configure_logging()
    from .evaluate import run_eval

    try:
        run_eval(questions, out, limit=limit or None)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


@app.command("gen-testset")
def gen_testset(
    size: int = typer.Option(10, help="Number of synthetic questions to generate."),
    out: str = typer.Option("eval/questions.generated.yaml", help="Where to write them."),
) -> None:
    """(Optional) Generate a synthetic test set from the docs with RAGAS."""
    configure_logging()
    from .evaluate import generate_testset

    try:
        generate_testset(size, out)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    app()
