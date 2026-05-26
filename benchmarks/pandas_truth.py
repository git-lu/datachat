"""Run benchmark `pandas` snippets against the same df as app.iris_agent."""

from __future__ import annotations

import json
from pathlib import Path

from app.iris_agent import df

CASES_PATH = Path(__file__).resolve().parent / "iris_cases.json"


def run_pandas_snippet(code: str):
    """Execute a case's pandas code. `df` is in scope. Returns last expression value."""
    namespace = {"df": df}
    lines = [line for line in code.strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError("empty pandas snippet")
    if len(lines) == 1:
        return eval(lines[0], namespace)
    exec("\n".join(lines[:-1]), namespace)
    return eval(lines[-1], namespace)


def eval_case(case: dict) -> object:
    return run_pandas_snippet(case["pandas"])


def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text())


def print_all_results() -> None:
    for case in load_cases():
        result = eval_case(case)
        print(f"{case['id']}: {result!r}")
