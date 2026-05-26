#!/usr/bin/env python3
"""Run Iris agent benchmark and generate an HTML report for manual judging."""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.iris_agent import (
    AGENT_MODEL,
    AGENT_SYSTEM,
    FORMATTER_MODEL,
    FORMATTER_SYSTEM,
    ask_iris,
    build_formatter_input,
    format_friendly,
)
from benchmarks.judge import JUDGE_MODEL, judge_result

CASES_PATH = ROOT / "benchmarks" / "iris_cases.json"
RESULTS_DIR = ROOT / "benchmarks" / "results"


def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text())


def run_case(case: dict, *, friendly: bool) -> dict:
    started = time.perf_counter()
    result = ask_iris(case["question"])
    formatter_input = build_formatter_input(result)
    answer = format_friendly(result) if friendly else result["draft"]
    elapsed = time.perf_counter() - started
    tool_steps = result.get("tool_steps") or []
    agent_tool_calls = sum(1 for step in tool_steps if not step.get("bootstrap"))
    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "expected": case["expected"],
        "pandas": case.get("pandas"),
        "answer": answer,
        "agent_system": AGENT_SYSTEM,
        "draft": result["draft"],
        "tool_trace": result["tool_trace"],
        "tool_steps": tool_steps,
        "tool_calls": len(tool_steps),
        "agent_tool_calls": agent_tool_calls,
        "bootstrap_tool_calls": len(tool_steps) - agent_tool_calls,
        "formatter_system": FORMATTER_SYSTEM,
        "formatter_input": formatter_input,
        "elapsed_s": round(elapsed, 2),
        "judge": None,
        "judge_notes": "",
        "judge_score": None,
        "judge_label": None,
        "judge_reasoning": None,
        "judge_model": None,
    }


def _render_pipeline_section(item: dict) -> str:
    tool_blocks = []
    for index, step in enumerate(item.get("tool_steps") or [], start=1):
        args = html.escape(json.dumps(step["arguments"], ensure_ascii=False))
        origin = "bootstrap" if step.get("bootstrap") else "agent"
        tool_blocks.append(
            f"""
            <div class="pipeline-step">
              <h4>Tool {index} ({origin}): {html.escape(step['name'])}({args})</h4>
              <pre>{html.escape(step['result'])}</pre>
            </div>
            """
        )
    tools_html = "".join(tool_blocks) or '<p class="muted">(no tool calls)</p>'

    formatter_block = ""
    if item.get("formatter_input"):
        formatter_block = f"""
        <div class="pipeline-step">
          <h4>Formatter system prompt</h4>
          <pre>{html.escape(item.get('formatter_system') or '')}</pre>
        </div>
        <div class="pipeline-step">
          <h4>Formatter user input (draft + evidence)</h4>
          <pre>{html.escape(item['formatter_input'])}</pre>
        </div>
        """

    return f"""
    <details class="pipeline" open>
      <summary>{html.escape(item['id'])} — agent pipeline</summary>
      <div class="pipeline-step">
        <h4>Agent system prompt</h4>
        <pre>{html.escape(item.get('agent_system') or '')}</pre>
      </div>
      <div class="pipeline-step">
        <h4>Agent draft (before formatter)</h4>
        <pre>{html.escape(item.get('draft') or '')}</pre>
      </div>
      <div class="pipeline-step">
        <h4>Tool responses</h4>
        {tools_html}
      </div>
      {formatter_block}
      <div class="pipeline-step">
        <h4>Final answer</h4>
        <pre>{html.escape(item.get('answer') or '')}</pre>
      </div>
    </details>
    """


def render_html(run: dict) -> str:
    rows = []
    for item in run["results"]:
        rows.append(
            f"""
            <tr data-id="{html.escape(item['id'])}">
              <td class="mono">{html.escape(item['id'])}</td>
              <td><span class="tag">{html.escape(item['category'])}</span></td>
              <td>{html.escape(item['question'])}</td>
              <td class="expected">{html.escape(item['expected'])}</td>
              <td class="answer">{html.escape(item['answer'])}</td>
              <td class="tools">{item['tool_calls']}</td>
              <td class="tools">{item['elapsed_s']}s</td>
              <td class="judge">
                <select class="score" aria-label="Score for {html.escape(item['id'])}">
                  <option value="">—</option>
                  <option value="correct">Correct</option>
                  <option value="partial">Partial</option>
                  <option value="wrong">Wrong</option>
                </select>
                <textarea class="notes" rows="2" placeholder="Notes"></textarea>
              </td>
            </tr>
            """
        )

    pipeline_sections = [_render_pipeline_section(item) for item in run["results"]]

    run_json = html.escape(json.dumps(run, indent=2))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Iris benchmark — {html.escape(run['run_id'])}</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #0f1419;
      --panel: #1a2332;
      --text: #e6edf3;
      --muted: #9da7b3;
      --border: #2d3a4d;
      --accent: #58a6ff;
      --good: #3fb950;
      --warn: #d29922;
      --bad: #f85149;
    }}
    @media (prefers-color-scheme: light) {{
      :root {{
        --bg: #f6f8fa;
        --panel: #ffffff;
        --text: #1f2328;
        --muted: #656d76;
        --border: #d0d7de;
        --accent: #0969da;
        --good: #1a7f37;
        --warn: #9a6700;
        --bad: #cf222e;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 14px/1.5 system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    header, main {{ max-width: 1400px; margin: 0 auto; padding: 1rem 1.25rem; }}
    header {{
      border-bottom: 1px solid var(--border);
      background: var(--panel);
    }}
    h1 {{ margin: 0 0 .25rem; font-size: 1.35rem; }}
    .meta {{ color: var(--muted); font-size: .92rem; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: .75rem;
      margin: 1rem 0;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: .85rem 1rem;
    }}
    .card .label {{ color: var(--muted); font-size: .8rem; text-transform: uppercase; letter-spacing: .04em; }}
    .card .value {{ font-size: 1.6rem; font-weight: 700; margin-top: .15rem; }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: .5rem;
      margin: 1rem 0;
    }}
    button {{
      background: var(--accent);
      color: #fff;
      border: 0;
      border-radius: 6px;
      padding: .45rem .8rem;
      cursor: pointer;
      font: inherit;
    }}
    button.secondary {{
      background: transparent;
      color: var(--text);
      border: 1px solid var(--border);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: .65rem .75rem;
      vertical-align: top;
      text-align: left;
    }}
    th {{ background: color-mix(in srgb, var(--panel) 85%, var(--accent)); font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; }}
    tr:last-child td {{ border-bottom: 0; }}
    .mono {{ font-family: ui-monospace, monospace; font-size: .85rem; }}
    .tag {{
      display: inline-block;
      padding: .1rem .45rem;
      border-radius: 999px;
      background: color-mix(in srgb, var(--accent) 18%, transparent);
      font-size: .78rem;
    }}
    .expected {{ color: var(--good); max-width: 260px; }}
    .answer {{ max-width: 320px; }}
    .tools {{ white-space: nowrap; text-align: center; }}
    select, textarea {{
      width: 100%;
      font: inherit;
      background: var(--bg);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: .25rem .35rem;
    }}
    textarea {{ margin-top: .35rem; resize: vertical; }}
    details {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: .5rem .75rem;
      margin: .5rem 0;
    }}
    details.pipeline summary {{
      font-weight: 600;
      cursor: pointer;
    }}
    .pipeline-step {{
      margin: .75rem 0;
      padding-top: .5rem;
      border-top: 1px solid var(--border);
    }}
    .pipeline-step:first-of-type {{ border-top: 0; padding-top: 0; }}
    .pipeline-step h4 {{
      margin: 0 0 .35rem;
      font-size: .82rem;
      text-transform: uppercase;
      letter-spacing: .04em;
      color: var(--muted);
    }}
    .muted {{ color: var(--muted); font-style: italic; }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      font-size: .82rem;
      background: var(--bg);
      padding: .75rem;
      border-radius: 6px;
      overflow: auto;
    }}
    #score-bar {{
      height: 10px;
      border-radius: 999px;
      background: var(--border);
      overflow: hidden;
      margin-top: .35rem;
    }}
    #score-bar > span {{
      display: block;
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--good), var(--accent));
      transition: width .2s ease;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Iris agent benchmark</h1>
    <p class="meta">
      Run <code>{html.escape(run['run_id'])}</code> ·
      Agent <code>{html.escape(run['agent_model'])}</code> ·
      Formatter <code>{html.escape(run['formatter_model'])}</code> ·
      Mode <code>{html.escape(run['mode'])}</code>
    </p>
    <div class="cards">
      <div class="card"><div class="label">Cases</div><div class="value">{run['summary']['total']}</div></div>
      <div class="card"><div class="label">Judged</div><div class="value" id="judged-count">0</div></div>
      <div class="card"><div class="label">Correct</div><div class="value" id="correct-count">0</div></div>
      <div class="card"><div class="label">Score</div><div class="value" id="score-pct">—</div><div id="score-bar"><span></span></div></div>
      <div class="card"><div class="label">Avg tools</div><div class="value">{run['summary']['avg_tool_calls']}</div></div>
      <div class="card"><div class="label">Avg time</div><div class="value">{run['summary']['avg_elapsed_s']}s</div></div>
    </div>
    <div class="toolbar">
      <button type="button" id="save-judgments">Save judgments to JSON</button>
      <button type="button" class="secondary" id="clear-judgments">Clear judgments</button>
    </div>
  </header>
  <main>
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Category</th>
          <th>Question</th>
          <th>Expected</th>
          <th>Answer</th>
          <th>Tools</th>
          <th>Time</th>
          <th>Your judgment</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
    <h2>Agent pipeline (draft → tools → formatter input → answer)</h2>
    {''.join(pipeline_sections)}
    <script type="application/json" id="run-data">{run_json}</script>
  </main>
  <script>
    const storageKey = "iris-benchmark-" + {json.dumps(run['run_id'])};
    const runData = JSON.parse(document.getElementById("run-data").textContent);

    function loadJudgments() {{
      try {{
        return JSON.parse(localStorage.getItem(storageKey) || "{{}}");
      }} catch {{
        return {{}};
      }}
    }}

    function saveJudgments(data) {{
      localStorage.setItem(storageKey, JSON.stringify(data));
    }}

    function updateSummary() {{
      const data = loadJudgments();
      const rows = [...document.querySelectorAll("tbody tr")];
      let judged = 0, correct = 0, partial = 0;
      for (const row of rows) {{
        const id = row.dataset.id;
        const score = data[id]?.score;
        if (!score) continue;
        judged += 1;
        if (score === "correct") correct += 1;
        if (score === "partial") partial += 1;
      }}
      const points = correct + partial * 0.5;
      const pct = judged ? Math.round((points / judged) * 100) : null;
      document.getElementById("judged-count").textContent = judged;
      document.getElementById("correct-count").textContent = correct;
      document.getElementById("score-pct").textContent = pct === null ? "—" : pct + "%";
      document.querySelector("#score-bar > span").style.width = (pct ?? 0) + "%";
    }}

    function bindRow(row) {{
      const id = row.dataset.id;
      const select = row.querySelector(".score");
      const notes = row.querySelector(".notes");
      const data = loadJudgments();
      if (data[id]) {{
        select.value = data[id].score || "";
        notes.value = data[id].notes || "";
      }}
      function persist() {{
        const all = loadJudgments();
        all[id] = {{ score: select.value, notes: notes.value }};
        saveJudgments(all);
        updateSummary();
      }}
      select.addEventListener("change", persist);
      notes.addEventListener("input", persist);
    }}

    document.querySelectorAll("tbody tr").forEach(bindRow);
    updateSummary();

    document.getElementById("save-judgments").addEventListener("click", () => {{
      const judgments = loadJudgments();
      const exportData = {{
        ...runData,
        judged_at: new Date().toISOString(),
        results: runData.results.map(item => ({{
          ...item,
          judge: judgments[item.id]?.score || null,
          judge_notes: judgments[item.id]?.notes || "",
        }})),
        summary: {{
          ...runData.summary,
          judged: Object.values(judgments).filter(j => j.score).length,
          correct: Object.values(judgments).filter(j => j.score === "correct").length,
          partial: Object.values(judgments).filter(j => j.score === "partial").length,
          wrong: Object.values(judgments).filter(j => j.score === "wrong").length,
        }},
      }};
      const blob = new Blob([JSON.stringify(exportData, null, 2)], {{ type: "application/json" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = runData.run_id + "-judged.json";
      a.click();
      URL.revokeObjectURL(url);
    }});

    document.getElementById("clear-judgments").addEventListener("click", () => {{
      localStorage.removeItem(storageKey);
      document.querySelectorAll(".score").forEach(el => el.value = "");
      document.querySelectorAll(".notes").forEach(el => el.value = "");
      updateSummary();
    }});
  </script>
</body>
</html>
"""


def run_benchmark(
    *,
    friendly: bool,
    limit: int | None,
    ids: list[str] | None,
    open_browser: bool,
) -> Path:
    cases = load_cases()
    if ids:
        cases = [c for c in cases if c["id"] in ids]
    if limit is not None:
        cases = cases[:limit]

    if not cases:
        raise SystemExit("No benchmark cases selected.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    results = []
    total_tools = 0
    total_elapsed = 0.0

    print(f"Running {len(cases)} cases (mode={'friendly' if friendly else 'draft'})...\n")
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']}: {case['question']}")
        item = run_case(case, friendly=friendly)
        results.append(item)
        total_tools += item["tool_calls"]
        total_elapsed += item["elapsed_s"]
        print(
            f"  -> {item['elapsed_s']}s, {item['tool_calls']} tool call(s) "
            f"(bootstrap={item['bootstrap_tool_calls']}, agent={item['agent_tool_calls']})\n"
        )

    run = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "agent_model": AGENT_MODEL,
        "formatter_model": FORMATTER_MODEL,
        "mode": "friendly" if friendly else "draft",
        "summary": {
            "total": len(results),
            "avg_tool_calls": round(total_tools / len(results), 1),
            "avg_elapsed_s": round(total_elapsed / len(results), 2),
        },
        "results": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"{run_id}.json"
    html_path = RESULTS_DIR / f"{run_id}.html"
    json_path.write_text(json.dumps(run, indent=2))
    html_path.write_text(render_html(run))

    print(f"Saved results: {json_path}")
    print(f"Report:        {html_path}")
    if open_browser:
        webbrowser.open(html_path.as_uri())

    return html_path


def print_expected_answers() -> None:
    cases = load_cases()
    print(f"{'ID':<28} {'EXPECTED'}")
    print("-" * 80)
    for case in cases:
        print(f"{case['id']:<28} {case['expected']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the Iris Ollama agent.")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Score agent draft only (skip friendly formatter)",
    )
    parser.add_argument("--limit", type=int, help="Run only the first N cases")
    parser.add_argument("--ids", nargs="+", help="Run only these case ids")
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print questions and expected answers, then exit",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the HTML report in a browser",
    )
    args = parser.parse_args()

    if args.list:
        print_expected_answers()
        return

    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    print(f"Ollama: {host}")
    print(f"Agent: {AGENT_MODEL}  |  Formatter: {FORMATTER_MODEL}\n")

    run_benchmark(
        friendly=not args.raw,
        limit=args.limit,
        ids=args.ids,
        open_browser=not args.no_open,
    )


if __name__ == "__main__":
    main()
