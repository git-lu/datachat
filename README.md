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

| Variable | Default |
|----------|---------|
| `AGENT_MODEL` | `granite3-dense:2b` |
| `FORMATTER_MODEL` | `granite3-moe:1b` |

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
ollama pull granite3-dense:2b
ollama pull granite3-moe:1b
python -m app.main -i
# or one question:
python -m app.main "Which species has the largest average petal length?"
```

## How it stays “warm”

| Piece | Stays running? |
|-------|----------------|
| **Ollama** (`ollama serve` or `docker compose up -d ollama`) | Yes — keep this up while you work |
| **Models on disk** | Yes — `ollama pull` only once |
| **Interactive Python** (`-i`) | One process, many questions |
| **`cli run --rm chatbot "..."`** | New container each time — fine for scripts, awkward for chatting |
