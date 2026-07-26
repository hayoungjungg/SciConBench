# SciConHarness CLI

Two CLI scripts are provided for running SciConHarness evaluations at scale across frontier models and deep research agents:

| Script | Use case |
|--------|----------|
| `query_single.py` | One-off query or interactive (REPL) mode for a single DOI |
| `query_batch.py` | Batch evaluation over a JSON file of DOI–question pairs; auto-resumes interrupted runs |

## query_single

```bash
# Interactive mode
python -m sciconharness.cli_scripts.query_single openai --interactive

# Non-interactive
python -m sciconharness.cli_scripts.query_single claude \
    --query "What are the benefits and harms of oral antibiotics for otitis media?" \
    --model claude-sonnet-4-5 \
    --doi 10.1002/14651858.CD015254.pub2 \
    --publication-date "23 October 2023" \
    --cochrane-titles data/filter_data/cochrane_titles.json \
    --enable-tool-calling --enable-filtering
```

## query_batch

`query_batch` processes a `dict[str, str]` JSON file mapping DOI → question. Already-processed DOIs are skipped automatically so interrupted runs resume safely.

```bash
python -m sciconharness.cli_scripts.query_batch openai \
    --model gpt-5.1 \
    --doi-questions data/doi_questions.json \
    --doi-dates data/filter_data/doi_dates.json \
    --cochrane-titles data/filter_data/cochrane_titles.json \
    --enable-tool-calling --enable-filtering
```

### Azure Foundry models (DeepSeek)

Same procedure as above with `provider=azure`. Uses `COCHRANE_DASHBOARD_OPENAI_KEY` and `COCHRANE_DASHBOARD_BASE_URL` in `.env` (DeepSeek-V4-Pro is deployed on its own Azure resource); falls back to `AZURE_OPENAI_KEY` / `OPENAI_BASE_URL` if those are unset.

```bash
python -m sciconharness.cli_scripts.query_batch azure \
    --model DeepSeek-V4-Pro \
    --doi-questions data/doi_questions.json \
    --doi-dates data/filter_data/doi_dates.json \
    --cochrane-titles data/filter_data/cochrane_titles.json \
    --enable-tool-calling --enable-filtering
```

### OpenRouter models (Kimi K3 / GLM-5.2 / Qwen3.5-9B / Qwen3.7-max)

Same procedure as above with `provider=openrouter`. Uses `OPENROUTER_API_KEY` in `.env`.

```bash
python -m sciconharness.cli_scripts.query_batch openrouter \
    --model moonshotai/kimi-k3 \
    --doi-questions data/doi_questions.json \
    --doi-dates data/filter_data/doi_dates.json \
    --cochrane-titles data/filter_data/cochrane_titles.json \
    --enable-tool-calling --enable-filtering

python -m sciconharness.cli_scripts.query_batch openrouter \
    --model z-ai/glm-5.2 \
    --doi-questions data/doi_questions.json \
    --doi-dates data/filter_data/doi_dates.json \
    --cochrane-titles data/filter_data/cochrane_titles.json \
    --enable-tool-calling --enable-filtering

python -m sciconharness.cli_scripts.query_batch openrouter \
    --model qwen/qwen3.5-9b \
    --doi-questions data/doi_questions.json \
    --doi-dates data/filter_data/doi_dates.json \
    --cochrane-titles data/filter_data/cochrane_titles.json \
    --enable-tool-calling --enable-filtering

python -m sciconharness.cli_scripts.query_batch openrouter \
    --model qwen/qwen3.7-max \
    --doi-questions data/doi_questions.json \
    --doi-dates data/filter_data/doi_dates.json \
    --cochrane-titles data/filter_data/cochrane_titles.json \
    --enable-tool-calling --enable-filtering
```

## Setup

To use these CLIs with the clean-room evaluation filtering enabled, they accept JSON files derived from the HuggingFace dataset. Run this once to generate them all:

```python
from datasets import load_dataset
import json, pathlib

ds = load_dataset("hayoungjung/SciConBench", "benchmark", split="test")

pathlib.Path("data").mkdir(exist_ok=True)
pathlib.Path("data/filter_data").mkdir(exist_ok=True)

# --doi-questions  (query_batch)
# dict[doi → question] — the primary input for batch runs
pathlib.Path("data/doi_questions.json").write_text(
    json.dumps({row["doi"]: row["question"] for row in ds}, indent=2)
)

# --cochrane-titles  (query_single + query_batch)
# All review titles — used to block search results that match any benchmark review
pathlib.Path("data/filter_data/cochrane_titles.json").write_text(
    json.dumps(list(ds["title"]), indent=2)
)

# --doi-dates  (query_batch)
# dict[doi → publication_date] — suppresses results published after the review date
pathlib.Path("data/filter_data/doi_dates.json").write_text(
    json.dumps({row["doi"]: row["publication_date"] for row in ds}, indent=2)
)
```

These files map to CLI flags as follows:

| File | query_single | query_batch |
|------|-------------|-------------|
| `data/doi_questions.json` | — | `--doi-questions` |
| `data/filter_data/cochrane_titles.json` | `--cochrane-titles` | `--cochrane-titles` |
| `data/filter_data/doi_dates.json` | — | `--doi-dates` |
