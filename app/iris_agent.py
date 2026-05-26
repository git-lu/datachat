"""Iris dataset chatbot: Ollama tool-calling agent + friendly formatter."""

from __future__ import annotations

import json
import os

import pandas as pd
from ollama import chat
from sklearn.datasets import load_iris

AGENT_MODEL = os.getenv("AGENT_MODEL", "granite3-dense:2b")
FORMATTER_MODEL = os.getenv("FORMATTER_MODEL", "granite3-moe:1b")
MAX_TOOL_STEPS = int(os.getenv("MAX_TOOL_STEPS", "6"))
MAX_ROWS = int(os.getenv("MAX_ROWS", "30"))

iris = load_iris(as_frame=True)
df = iris.frame.rename(
    columns={
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
    }
)
df["species"] = iris.target_names[iris.target]

ALLOWED_COLUMNS = set(df.columns)
ALLOWED_AGG = {"mean", "max", "min", "sum", "count"}


def _truncate(result: pd.DataFrame | pd.Series) -> str:
    if isinstance(result, pd.Series):
        return result.to_string()
    if len(result) > MAX_ROWS:
        return (
            result.head(MAX_ROWS).to_string(index=False)
            + f"\n... ({len(result) - MAX_ROWS} more rows)"
        )
    return result.to_string(index=False)


def dataframe_info() -> str:
    """Return Iris dataset schema, row count, and summary statistics.

    Returns:
        str: Column names, dtypes, species list, and describe() output.
    """
    species = ", ".join(sorted(df["species"].unique()))
    return (
        f"rows={len(df)}, columns={list(df.columns)}\n"
        f"species: {species}\n\n"
        f"{df.describe().to_string()}\n\n"
        f"dtypes:\n{df.dtypes.to_string()}"
    )


def query_dataframe(expression: str) -> str:
    """Filter Iris rows with pandas DataFrame.query() syntax.

    Args:
        expression: Query string, e.g. "species == 'setosa' and sepal_width < 3.5".

    Returns:
        str: Matching rows as a table (capped at MAX_ROWS).
    """
    subset = df.query(expression, engine="python")
    return _truncate(subset)


def groupby_aggregate(group_by: str, column: str, agg: str) -> str:
    """Aggregate a numeric column grouped by a column (usually species).

    Args:
        group_by: Column to group by, e.g. "species".
        column: Numeric column to aggregate, e.g. "petal_length".
        agg: Aggregation name: mean, max, min, sum, or count.

    Returns:
        str: Aggregation result table.
    """
    if group_by not in ALLOWED_COLUMNS:
        raise ValueError(f"Unknown group_by: {group_by}")
    if column not in ALLOWED_COLUMNS:
        raise ValueError(f"Unknown column: {column}")
    if agg not in ALLOWED_AGG:
        raise ValueError(f"agg must be one of {ALLOWED_AGG}")

    result = df.groupby(group_by, as_index=False)[column].agg(agg)
    return _truncate(result)


def value_counts(column: str) -> str:
    """Count distinct values in a column.

    Args:
        column: Column name, e.g. "species".

    Returns:
        str: Value counts table.
    """
    if column not in ALLOWED_COLUMNS:
        raise ValueError(f"Unknown column: {column}")
    return df[column].value_counts().to_string()


PANDAS_TOOLS = [dataframe_info, query_dataframe, groupby_aggregate, value_counts]
TOOL_BY_NAME = {fn.__name__: fn for fn in PANDAS_TOOLS}


def _run_tool(name: str, arguments: dict) -> str:
    fn = TOOL_BY_NAME.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    try:
        return fn(**arguments)
    except Exception as exc:
        return f"Tool error ({name}): {exc}"


def ask_iris(question: str, verbose: bool = False) -> dict:
    """Run the agent: pandas tools + short technical draft from AGENT_MODEL."""
    messages = [
        {
            "role": "system",
            "content": (
                "You answer questions about the Iris flower dataset. "
                "Always use the pandas tools to look up data before answering. "
                "Reply briefly and factually. Do not invent numbers. "
                "Columns: sepal_length, sepal_width, petal_length, petal_width, "
                "species (setosa, versicolor, virginica)."
            ),
        },
        {"role": "user", "content": question},
    ]
    tool_trace: list[str] = []

    for _ in range(MAX_TOOL_STEPS):
        response = chat(model=AGENT_MODEL, messages=messages, tools=PANDAS_TOOLS)
        assistant = response.message
        messages.append(assistant.model_dump(exclude_none=True))

        if not assistant.tool_calls:
            return {
                "question": question,
                "draft": assistant.content or "",
                "tool_trace": tool_trace,
            }

        for call in assistant.tool_calls:
            name = call.function.name
            args = call.function.arguments
            if isinstance(args, str):
                args = json.loads(args)
            result = _run_tool(name, args)
            tool_trace.append(f"{name}({args})\n{result}")
            if verbose:
                print(f"[tool] {name}({args})\n{result}\n")
            messages.append({
                "role": "tool",
                "tool_name": name,
                "content": result,
            })

    return {
        "question": question,
        "draft": "Stopped: too many tool steps.",
        "tool_trace": tool_trace,
    }


def format_friendly(result: dict) -> str:
    """Rewrite agent output into plain language using FORMATTER_MODEL."""
    evidence = "\n\n---\n\n".join(result["tool_trace"]) or "(no tool output)"
    prompt = f"""Rewrite this data assistant reply for a non-technical user.

Rules:
- Use 2–4 short sentences, friendly and clear.
- Include the key numbers and species names from the evidence.
- Do not mention tools, pandas, SQL, or models.
- If the evidence is insufficient, say what is missing.

User question: {result['question']}

Technical draft: {result['draft']}

Data evidence:
{evidence}
"""

    response = chat(
        model=FORMATTER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.3, "num_predict": 200},
    )
    return response.message.content or ""


def ask_iris_friendly(question: str, verbose: bool = False) -> str:
    """Full pipeline: agent (tools) → formatter (readable answer)."""
    result = ask_iris(question, verbose=verbose)
    return format_friendly(result)
