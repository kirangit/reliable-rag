"""The three Reliable-RAG gates, one module each.

* :func:`grade.grade_chunks` — score retrieved chunks BEFORE generation (Post 1: "Grade").
* :func:`generate.generate_answer` — answer using only kept chunks, citing chunk IDs.
* :func:`verify.verify_answer` — check the answer is grounded, claim by claim (Post 1: "Verify").

Each gate returns ``(domain_result, raw_message)``. The raw ``AIMessage`` carries
``usage_metadata`` so the graph can attribute token cost per gate via ``cost.py``
— the gates themselves stay focused on their one job.
"""

from __future__ import annotations

from ._common import format_chunks_for_prompt
from .generate import generate_answer
from .grade import grade_chunks
from .verify import verify_answer

__all__ = ["grade_chunks", "generate_answer", "verify_answer", "format_chunks_for_prompt"]
