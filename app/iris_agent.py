"""Iris dataset chatbot: Ollama tool-calling agent + friendly formatter."""

from __future__ import annotations

import json
import os

import pandas as pd
from ollama import chat
from sklearn.datasets import load_iris

AGENT_MODEL = os.getenv("AGENT_MODEL", "llama3.1:8b")
FORMATTER_MODEL = os.getenv("FORMATTER_MODEL", "granite3-moe:1b")
INCLUDE_DATASET_IN_PROMPT = os.getenv("INCLUDE_DATASET_IN_PROMPT", "true").lower() in {
    "1",
    "true",
    "yes",
}
# 0 = embed all rows in the system prompt (fine for Iris); set e.g. 50 to cap for larger tables.
MAX_PROMPT_DATASET_ROWS = int(os.getenv("MAX_PROMPT_DATASET_ROWS", "0"))

AGENT_SYSTEM_BASE = """You answer questions about the Iris flower dataset.

The full dataset is included below as CSV. The same table is available through tools for filters and aggregates.
For counts, means, maxima, and grouped statistics, call tools—do not do mental math over the CSV.
For questions about columns not present in the CSV header, say they are missing.

Rules:
- Prefer tool output for any aggregated or filtered numeric answer.
- Never write Python code, pandas snippets, or fake tool calls in your reply—invoke the real tools.
- After tool results return, write a brief factual draft (1–3 sentences) citing only numbers from the CSV or tool output.
- Do not use memorized Iris statistics that contradict the CSV or tools.

Tools (each runs on the complete in-memory table):
- dataframe_info — schema, row count, summary statistics.
- export_full_dataset — entire table as CSV (all rows).
- count_rows — how many rows match a filter (e.g. "species == 'setosa' and sepal_width < 3.5"). Use for "how many..." with conditions.
- filter_aggregate — filter rows, then mean/min/max/sum/count one column on that subset (e.g. versicolor min sepal width).
- query_dataframe — preview matching rows as a table (not for counting; use count_rows for counts).
- groupby_aggregate — mean/max/min/sum/count grouped by a column (usually species), no row filter.
- value_counts — distribution of values in one column across all rows (e.g. species tallies). Not for filtered counts."""

FORMATTER_SYSTEM = """You turn data assistant output into plain language
for a non-technical user. Do not include reasoning or explaination in your answer.

Rules:
- Be brief: one sentence when possible, two at most.
- State only facts supported by the evidence; include key numbers and species names.
- If the evidence is insufficient, say what is missing in one short sentence.
Example outputs:
- "The dataset has 150 rows."
- "The dataset has 3 species: setosa, versicolor, and virginica."
- "The dataset does not contain color information."
- "There is no stem length column in the dataset."
"""

MAX_TOOL_STEPS = int(os.getenv("MAX_TOOL_STEPS", "6"))
MAX_ROWS = int(os.getenv("MAX_ROWS", "150"))
BOOTSTRAP_DATAFRAME_INFO = os.getenv("BOOTSTRAP_DATAFRAME_INFO", "true").lower() in {
    "1",
    "true",
    "yes",
}
REQUIRE_AGENT_TOOL_CALL = os.getenv("REQUIRE_AGENT_TOOL_CALL", "true").lower() in {
    "1",
    "true",
    "yes",
}

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


def dataset_csv_for_prompt() -> str:
    """Serialize the in-memory table for inclusion in the agent system prompt."""
    if MAX_PROMPT_DATASET_ROWS <= 0 or len(df) <= MAX_PROMPT_DATASET_ROWS:
        return df.to_csv(index=False)
    head = df.head(MAX_PROMPT_DATASET_ROWS)
    omitted = len(df) - MAX_PROMPT_DATASET_ROWS
    return (
        head.to_csv(index=False)
        + f"\n... ({omitted} more rows omitted from prompt; use export_full_dataset tool)"
    )


def build_agent_system(*, include_dataset: bool | None = None) -> str:
    """Build the agent system prompt, optionally embedding the full dataset CSV."""
    use_dataset = INCLUDE_DATASET_IN_PROMPT if include_dataset is None else include_dataset
    if not use_dataset:
        return AGENT_SYSTEM_BASE
    csv_block = dataset_csv_for_prompt()
    row_note = len(df) if MAX_PROMPT_DATASET_ROWS <= 0 else min(len(df), MAX_PROMPT_DATASET_ROWS)
    return (
        f"{AGENT_SYSTEM_BASE}\n\n"
        f"Dataset ({row_note} of {len(df)} rows shown, CSV):\n"
        f"```csv\n{csv_block}\n```"
    )


AGENT_SYSTEM = build_agent_system()


def _truncate(result: pd.DataFrame | pd.Series) -> str:
    if isinstance(result, pd.Series):
        return result.to_string()
    if len(result) > MAX_ROWS:
        return (
            result.head(MAX_ROWS).to_string(index=False)
            + f"\n... ({len(result) - MAX_ROWS} more rows)"
        )
    return result.to_string(index=False)


def _filtered(expression: str) -> pd.DataFrame:
    return df.query(expression, engine="python")


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


def count_rows(expression: str) -> str:
    """Count rows that match a filter expression.

    Args:
        expression: pandas query string, e.g. "species == 'setosa' and sepal_width < 3.5".

    Returns:
        str: Number of matching rows (and total rows for context).
    """
    subset = _filtered(expression)
    return (
        f"matching_rows={len(subset)} (of {len(df)} total)\n"
        f"expression={expression!r}"
    )


def filter_aggregate(expression: str, column: str, agg: str) -> str:
    """Filter rows, then aggregate one numeric column on the matching subset.

    Args:
        expression: pandas query string applied before aggregating.
        column: Numeric column to aggregate, e.g. "sepal_width".
        agg: mean, max, min, sum, or count (row count of filtered subset).

    Returns:
        str: matching row count and aggregation result.
    """
    if column not in ALLOWED_COLUMNS:
        raise ValueError(f"Unknown column: {column}")
    if agg not in ALLOWED_AGG:
        raise ValueError(f"agg must be one of {ALLOWED_AGG}")

    subset = _filtered(expression)
    if len(subset) == 0:
        return f"matching_rows=0\nexpression={expression!r}\n(no rows to aggregate)"

    if agg == "count":
        value = len(subset)
    else:
        value = subset[column].agg(agg)

    return (
        f"matching_rows={len(subset)}\n"
        f"expression={expression!r}\n"
        f"{agg}({column})={value}"
    )


def query_dataframe(expression: str) -> str:
    """Preview rows matching a filter (use count_rows to count matches).

    Args:
        expression: Query string, e.g. "species == 'setosa' and sepal_width < 3.5".

    Returns:
        str: Matching rows as a table (capped at MAX_ROWS).
    """
    subset = _filtered(expression)
    header = f"matching_rows={len(subset)} (showing up to {MAX_ROWS})\n"
    return header + _truncate(subset)


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
    """Value distribution for one column across all rows (not for filtered row counts).

    Args:
        column: Column name, e.g. "species".

    Returns:
        str: Value counts table for the full dataset.
    """
    if column not in ALLOWED_COLUMNS:
        raise ValueError(f"Unknown column: {column}")
    return df[column].value_counts().to_string()


def export_full_dataset() -> str:
    """Return the complete Iris dataset as CSV (all rows, no truncation).

    Returns:
        str: Full table as CSV text.
    """
    return df.to_csv(index=False)


PANDAS_TOOLS = [
    dataframe_info,
    export_full_dataset,
    count_rows,
    filter_aggregate,
    query_dataframe,
    groupby_aggregate,
    value_counts,
]
TOOL_BY_NAME = {fn.__name__: fn for fn in PANDAS_TOOLS}


def _run_tool(name: str, arguments: dict) -> str:
    fn = TOOL_BY_NAME.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    try:
        return fn(**arguments)
    except Exception as exc:
        return f"Tool error ({name}): {exc}"


def _record_tool_step(
    *,
    name: str,
    arguments: dict,
    result: str,
    tool_trace: list[str],
    tool_steps: list[dict],
    bootstrap: bool = False,
    verbose: bool = False,
) -> None:
    tool_steps.append({
        "name": name,
        "arguments": arguments,
        "result": result,
        "bootstrap": bootstrap,
    })
    tool_trace.append(f"{name}({arguments})\n{result}")
    if verbose:
        label = "bootstrap" if bootstrap else "tool"
        print(f"[{label}] {name}({arguments})\n{result}\n")


def _append_tool_messages(messages: list[dict], name: str, result: str) -> None:
    messages.append({
        "role": "tool",
        "tool_name": name,
        "content": result,
    })


def ask_iris(question: str, verbose: bool = False) -> dict:
    """Run the agent: pandas tools on the full in-memory dataset + draft from AGENT_MODEL."""
    messages: list[dict] = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": question},
    ]
    tool_trace: list[str] = []
    tool_steps: list[dict] = []
    agent_tool_calls = 0
    nudged_for_tools = False

    if BOOTSTRAP_DATAFRAME_INFO:
        bootstrap_args: dict = {}
        bootstrap_result = _run_tool("dataframe_info", bootstrap_args)
        _record_tool_step(
            name="dataframe_info",
            arguments=bootstrap_args,
            result=bootstrap_result,
            tool_trace=tool_trace,
            tool_steps=tool_steps,
            bootstrap=True,
            verbose=verbose,
        )
        messages.append({
            "role": "user",
            "content": (
                "dataframe_info was already run on the full dataset (150 rows):\n"
                f"{bootstrap_result}\n\n"
                "Call more tools if needed, then reply with a brief draft."
            ),
        })

    for _ in range(MAX_TOOL_STEPS):
        response = chat(model=AGENT_MODEL, messages=messages, tools=PANDAS_TOOLS)
        assistant = response.message
        messages.append(assistant.model_dump(exclude_none=True))

        if not assistant.tool_calls:
            if (
                REQUIRE_AGENT_TOOL_CALL
                and agent_tool_calls == 0
                and not nudged_for_tools
            ):
                nudged_for_tools = True
                messages.append({
                    "role": "user",
                    "content": (
                        "You must invoke at least one tool "
                        "(count_rows, filter_aggregate, query_dataframe, "
                        "groupby_aggregate, value_counts, or export_full_dataset) "
                        "on the dataset before answering."
                    ),
                })
                continue

            return {
                "question": question,
                "draft": assistant.content or "",
                "tool_trace": tool_trace,
                "tool_steps": tool_steps,
            }

        for call in assistant.tool_calls:
            name = call.function.name
            args = call.function.arguments
            if isinstance(args, str):
                args = json.loads(args)
            result = _run_tool(name, args)
            agent_tool_calls += 1
            _record_tool_step(
                name=name,
                arguments=args,
                result=result,
                tool_trace=tool_trace,
                tool_steps=tool_steps,
                verbose=verbose,
            )
            _append_tool_messages(messages, name, result)

    return {
        "question": question,
        "draft": "Stopped: too many tool steps.",
        "tool_trace": tool_trace,
        "tool_steps": tool_steps,
    }


def build_formatter_input(result: dict) -> str:
    """User message sent to the formatter (question + draft + tool evidence)."""
    evidence = "\n\n---\n\n".join(result.get("tool_trace") or []) or "(no tool output)"
    return (
        f"Question: {result['question']}\n\n"
        f"Draft: {result['draft']}\n\n"
        f"Evidence:\n{evidence}"
    )


def format_friendly(result: dict) -> str:
    """Rewrite agent output into plain language using FORMATTER_MODEL."""
    user_content = build_formatter_input(result)

    response = chat(
        model=FORMATTER_MODEL,
        messages=[
            {"role": "system", "content": FORMATTER_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        options={"temperature": 0, "num_predict": 120},
    )
    return response.message.content or ""


def ask_iris_friendly(question: str, verbose: bool = False) -> str:
    """Full pipeline: agent (tools) → formatter (readable answer)."""
    result = ask_iris(question, verbose=verbose)
    return format_friendly(result)
