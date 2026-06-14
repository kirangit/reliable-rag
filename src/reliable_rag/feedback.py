"""Human feedback store — the 👍/👎 signal that complements automated RAGAS.

RAGAS judges quality automatically; real users judge it for real. We persist
each thumbs verdict (with the run context) to ``feedback/feedback.jsonl`` so the
Streamlit dashboard can show human feedback right next to the RAGAS triad — the
observability loop both posts argue for.

Plain JSONL keeps this dependency-free; swap in Chainlit's SQLite data layer if
you want threaded history.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .config import settings


def _store_path() -> Path:
    d = Path(settings.feedback_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / "feedback.jsonl"


def record_feedback(record: dict) -> None:
    """Append one feedback record (a timestamp is added automatically)."""
    record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    with _store_path().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_feedback() -> list[dict]:
    """Read all feedback records (newest last). Empty list if none yet."""
    path = _store_path()
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out
