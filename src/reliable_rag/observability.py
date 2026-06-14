"""Observability — see every chunk and what each stage did, three ways.

1. **Per-run trace log** -> ``runs/<run_id>.json``: the durable, inspectable
   record of a query (retrieved chunks + scores, grade decisions, the answer,
   verification, flags, per-gate cost + timing).
2. **`rich` console rendering** for interactive ``ask`` runs.
3. **Optional ready-made trace UI** (``TRACING=phoenix`` local, or ``langsmith``
   hosted) wired into LangChain/LangGraph — off by default so the repo runs
   offline with no extra services.

This module avoids importing :mod:`pipeline` at module load (it only needs the
``RunResult`` *shape*), so there's no import cycle.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import settings
from .cost import cost_summary

if TYPE_CHECKING:  # for type hints only; not imported at runtime
    from .pipeline import RunResult

# Make console output UTF-8 safe on legacy Windows terminals (cp1252), so the
# gate glyphs (✓, ·, —) never raise UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass

console = Console()
log = logging.getLogger("reliable_rag")

_logging_ready = False
_tracing_ready = False


def configure_logging(level: str | None = None) -> None:
    global _logging_ready
    if _logging_ready:
        return
    logging.basicConfig(
        level=(level or settings.log_level).upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _logging_ready = True


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]


def setup_tracing() -> None:
    """Best-effort wiring of the optional trace UI. Safe to call repeatedly."""
    global _tracing_ready
    if _tracing_ready:
        return
    _tracing_ready = True

    mode = (settings.tracing or "none").lower()
    if mode == "phoenix":
        try:
            # Don't let a missing collector flood the console or BLOCK the run.
            # (The default SimpleSpanProcessor exports synchronously and retries,
            # which can add minutes per query when no Phoenix server is up.)
            logging.getLogger("opentelemetry.exporter.otlp.proto.grpc.exporter").setLevel(logging.CRITICAL)
            logging.getLogger("opentelemetry.sdk.trace.export").setLevel(logging.CRITICAL)
            from phoenix.otel import register

            # batch=True -> async BatchSpanProcessor: span export happens off the
            # request path, so an unreachable collector never slows the pipeline.
            register(project_name="reliable-rag", auto_instrument=True, batch=True)
            console.print(
                "[dim]Phoenix tracing on (batched, non-blocking). Run `phoenix serve` "
                "(http://localhost:6006) to view spans; otherwise they're dropped silently.[/dim]"
            )
        except Exception as exc:  # extras not installed / server issue
            log.warning(
                'Phoenix tracing requested but unavailable (%s). Install with '
                'pip install -e ".[tracing]".',
                exc,
            )
    elif mode == "langsmith":
        # The LangSmith SDK reads these from os.environ (not our .env directly), so
        # bridge them across. setdefault -> a real shell var always wins.
        if settings.langsmith_api_key:
            os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
        if os.environ.get("LANGSMITH_API_KEY"):
            console.print(
                f"[dim]LangSmith tracing on (project={os.environ['LANGSMITH_PROJECT']}). "
                "View at https://smith.langchain.com[/dim]"
            )
        else:
            console.print(
                "[yellow]TRACING=langsmith but LANGSMITH_API_KEY is unset — traces "
                "won't be sent. Add it to .env or the shell.[/yellow]"
            )


# ---------------------------------------------------------------------------
# Run logging
# ---------------------------------------------------------------------------
def log_run(result: "RunResult") -> str:
    """Persist a full run to ``runs/<run_id>.json``. Returns the path."""
    out_dir = Path(settings.runs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result.run_id}.json"
    path.write_text(json.dumps(result.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    log.debug("Wrote run trace to %s", path)
    return str(path)


def log_ingest_cost(record: dict) -> str:
    """Append one ingest run's cost (with a UTC timestamp) to a JSONL log.

    One line per ``reliable-rag ingest`` so the embedding spend accrues a history
    you can inspect. Returns the log path.
    """
    out_dir = Path(settings.runs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ingest_cost.jsonl"
    entry = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return str(path)


# ---------------------------------------------------------------------------
# Console rendering
# ---------------------------------------------------------------------------
def render_run(result: "RunResult") -> None:
    """Pretty-print a run: answer, gate decisions, citations, and cost."""
    flags = ", ".join(result.flags) if result.flags else "none"
    header = (
        f"[bold]Q:[/bold] {result.query}\n"
        f"run_id={result.run_id}  ·  {result.elapsed_ms:.0f} ms  ·  "
        f"${result.total_usd:.4f}  ·  flags: {flags}"
    )
    if result.rewritten_query:
        header += f"\n[dim]rewritten ->[/dim] {result.rewritten_query}"
    console.print(Panel(header, title="Reliable RAG", border_style="cyan"))

    # Grade gate: every retrieved chunk + its decision.
    if result.grades:
        gt = Table(title="GRADE — retrieved chunks", show_lines=False, expand=True)
        gt.add_column("kept", justify="center")
        gt.add_column("score", justify="right")
        gt.add_column("chunk_id", overflow="fold")
        gt.add_column("why", overflow="fold")
        for g in result.grades:
            gt.add_row("✓" if g.relevant else "·", f"{g.score:.2f}", g.chunk_id, g.reason)
        console.print(gt)

    # Verify gate.
    if result.verification is not None:
        v = result.verification
        status = "[green]faithful[/green]" if v.faithful else "[red]UNVERIFIED[/red]"
        body = f"{status}  ·  grounded {v.score:.0%} of {len(v.claims)} claims"
        if v.unsupported_claims:
            body += "\n[red]unsupported:[/red]\n - " + "\n - ".join(v.unsupported_claims)
        console.print(Panel(body, title="VERIFY", border_style="magenta"))

    # Answer.
    console.print(Panel(result.answer or "(empty)", title="Answer", border_style="green"))

    # Trace / citations.
    if result.citations:
        ct = Table(title="TRACE — grounded citations", expand=True)
        ct.add_column("chunk_id", overflow="fold")
        ct.add_column("source")
        ct.add_column("section", overflow="fold")
        for c in result.citations:
            ct.add_row(c.chunk_id, c.source, c.header_path or "—")
        console.print(ct)
    else:
        console.print("[dim]No grounded citations (abstained or unsupported).[/dim]")

    # Cost.
    summary = cost_summary(result.costs)
    cost_tbl = Table(title="COST — per gate", expand=True)
    cost_tbl.add_column("stage")
    cost_tbl.add_column("model")
    cost_tbl.add_column("in", justify="right")
    cost_tbl.add_column("out", justify="right")
    cost_tbl.add_column("usd", justify="right")
    for c in result.costs:
        cost_tbl.add_row(c.stage, c.model, str(c.input_tokens), str(c.output_tokens), f"${c.usd:.4f}")
    cost_tbl.add_row("[bold]total[/bold]", "", "", "", f"[bold]${summary['total_usd']:.4f}[/bold]")
    console.print(cost_tbl)
