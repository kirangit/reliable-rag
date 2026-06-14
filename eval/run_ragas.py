"""Thin entry point so you can also run the eval as a plain script.

Equivalent to ``reliable-rag eval``. Usage:

    python eval/run_ragas.py [questions.yaml] [results_dir]
"""

from __future__ import annotations

import sys

from reliable_rag.evaluate import run_eval

if __name__ == "__main__":
    questions = sys.argv[1] if len(sys.argv) > 1 else "eval/questions.yaml"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "eval/results"
    run_eval(questions, out_dir)
