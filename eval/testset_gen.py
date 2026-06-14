"""Thin entry point for optional synthetic test-set generation.

Equivalent to ``reliable-rag gen-testset``. Usage:

    python eval/testset_gen.py [size] [out.yaml]

Uses one LLM call per knowledge-graph node, so it costs more than the eval —
that's why it's opt-in. Generated questions land in a YAML you can review and
merge into ``questions.yaml``.
"""

from __future__ import annotations

import sys

from reliable_rag.evaluate import generate_testset

if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    out_path = sys.argv[2] if len(sys.argv) > 2 else "eval/questions.generated.yaml"
    generate_testset(size, out_path)
