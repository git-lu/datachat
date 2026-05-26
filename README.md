# Data chatbot (Ollama + pandas tools)

Self-hosted Iris POC: an **agent** model calls pandas tools on the dataset; a smaller **formatter** model turns results into plain language.

## Docker (recommended)

**Requirements:** Docker Desktop (or Docker Engine + Compose v2).

### 1. Start Ollama and pull models (first run downloads ~2.5 GB)

```bash
cd data-chatbot
docker compose up -d ollama
docker compose up pull-models
```

`pull-models` is a one-shot service (downloads models **once**). Models stay in the `ollama_data` volume; you do **not** pull again per question.

Keep Ollama running in the background (`docker compose up -d ollama`). Only the old **cli** mode started a new container per question — use **interactive** below instead.

### 2. Interactive chat (many questions, one session)

**Local (simplest):** Ollama app running on the Mac, then:

```bash
source .venv/bin/activate
python -m app.main -i
```

You get a `You>` prompt. Ask as many questions as you want; type `/quit` to exit. Models stay loaded in Ollama between questions (faster after the first).

**Docker:**

```bash
docker compose up -d ollama
docker compose --profile chat run --rm chat
```

### 3. Single question (one-shot)

```bash
docker compose --profile cli run --rm chatbot "Which species has the largest average petal length?"
```

Verbose tool output:

```bash
docker compose --profile cli run --rm chatbot -v "How many setosa have sepal width under 3.5?"
```

Agent only (no formatter):

```bash
docker compose --profile cli run --rm chatbot --raw "How many rows are in the dataset?"
```

### 4. Optional: Jupyter notebook

```bash
docker compose --profile notebook up notebook
```

Open http://localhost:8888 and use `POC ollama.ipynb`. In the notebook:

```python
from app.iris_agent import ask_iris_friendly
ask_iris_friendly("Which species has the largest petal length?", verbose=True)
```

### Environment

Copy `.env.example` to `.env` to override models:

| Variable | Default | Role |
|----------|---------|------|
| `AGENT_MODEL` | `llama3.1:8b` | NL → tool choice & args (needs reliable tool calling) |
| `FORMATTER_MODEL` | `granite3-moe:1b` | Short friendly answer from tool evidence |

`granite3-dense:2b` is too weak for tool calling in practice (often answers from memory with `agent_tool_calls=0`). Use a larger instruct model for the agent; keep the formatter small.

**Pull the new agent model:**

```bash
ollama pull llama3.1:8b
# or: docker compose up pull-models
```

### Useful commands

```bash
# Ollama API on the host (for local dev outside Docker)
curl http://localhost:11434/api/tags

# Shell into Ollama container
docker compose exec ollama ollama list

# Stop everything
docker compose down

# Remove downloaded models
docker compose down -v
```

**Note:** GPU inside Docker depends on your host (NVIDIA Container Toolkit on Linux; Apple Silicon uses CPU in Docker). For fastest inference on a Mac, run Ollama on the host and point the app at `OLLAMA_HOST=http://host.docker.internal:11434`.

## Local (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.1:8b
ollama pull granite3-moe:1b
python -m app.main -i
# or one question:
python -m app.main "Which species has the largest average petal length?"
```

## Agent tools (no embeddings)

The full Iris table (150 rows) is embedded in the **agent system prompt** as CSV (`INCLUDE_DATASET_IN_PROMPT=true`) and available via pandas tools. Use tools for aggregates/filters; the CSV is ground truth for schema and raw rows.

The agent must use tools—never embeddings—for facts:

- **Bootstrap** (`BOOTSTRAP_DATAFRAME_INFO=true`, default): runs `dataframe_info` on the full dataset before the LLM turn.
- **`export_full_dataset`**: returns all rows as CSV when the model needs the raw table.
- **`count_rows`**: row counts for filter expressions (e.g. setosa with sepal_width &lt; 3.5).
- **`filter_aggregate`**: filter first, then min/max/mean on a column (e.g. versicolor min sepal width).
- **`value_counts`**: column distributions on the full table only—not for filtered “how many” questions.
- **`REQUIRE_AGENT_TOOL_CALL=true`**: nudges the model once if it tries to answer without calling a tool itself.

Aggregations (`groupby_aggregate`, filters) always run on the complete DataFrame; only displayed row previews are capped (`MAX_ROWS`).

## Usage intent agent (separate POC)

Predicts **proactive dashboards** from biologist usage logs (not Iris measurements).

| Folder | Role |
|--------|------|
| `usage_poc/data/` | 2 users (Ana, Ben), events, widget catalog, playbook |
| `usage_agent/` | RAG over session narratives → widget + question recommendations |

Uses **`USAGE_AGENT_MODEL`** (default `granite3-dense:2b`). RAG is recency + keyword over session text—no embeddings at this scale.

```bash
pip install pyyaml   # if not already installed
python -m usage_agent.main              # list users
python -m usage_agent.main ana -v       # recommend for Dr. Ana Reyes
python -m usage_agent.main ben --compare-playbook
```

See `usage_poc/README.md`. Merging with the Iris chatbot comes later.

## Benchmark

20 Iris Q&A cases with expected answers in `benchmarks/iris_cases.json`:

```bash
python scripts/run_benchmark.py --list    # print questions + expected answers
python scripts/run_benchmark.py           # run all cases, open HTML report
python scripts/run_benchmark.py --raw     # judge agent draft only (no formatter)
```

See `benchmarks/README.md` for the full case list and judging workflow.

## How it stays “warm”

| Piece | Stays running? |
|-------|----------------|
| **Ollama** (`ollama serve` or `docker compose up -d ollama`) | Yes — keep this up while you work |
| **Models on disk** | Yes — `ollama pull` only once |
| **Interactive Python** (`-i`) | One process, many questions |
| **`cli run --rm chatbot "..."`** | New container each time — fine for scripts, awkward for chatting |
