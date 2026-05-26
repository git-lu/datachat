# Iris benchmark

20+ questions with human `expected` answers and a `pandas` code snippet per case in `iris_cases.json`.

| Field | Purpose |
|-------|---------|
| `question` | What we ask the agent |
| `expected` | Your judged correct answer (edit manually) |
| `pandas` | Pandas code to compute ground truth (`df` = same table as `app.iris_agent`) |

## Run pandas snippets

```bash
python scripts/run_benchmark_pandas.py
python scripts/run_benchmark_pandas.py --ids 07-setosa-narrow-sepals
```

Or in Python:

```python
from app.iris_agent import df
from benchmarks.pandas_truth import load_cases, run_pandas_snippet

case = next(c for c in load_cases() if c["id"] == "07-setosa-narrow-sepals")
run_pandas_snippet(case["pandas"])  # np.int64(31) or similar
```

## Agent benchmark

```bash
python scripts/run_benchmark.py
```

See pipeline output in `benchmarks/results/` for agent vs `expected` judging.
