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

# Non-interactive — publication date and title-based filtering auto-resolve
# from --doi via the local HF benchmark cache (see "Filter data" below), so
# --publication-date / --cochrane-titles are optional overrides, not required.
python -m sciconharness.cli_scripts.query_single claude \
    --query "What are the benefits and harms of oral antibiotics for otitis media?" \
    --model claude-sonnet-4-5 \
    --doi 10.1002/14651858.CD015254.pub2 \
    --enable-tool-calling --enable-filtering
```

## query_batch

`query_batch` processes a `dict[str, str]` JSON file mapping DOI → question. Already-processed DOIs are skipped automatically so interrupted runs resume safely.

```bash
python -m sciconharness.cli_scripts.query_batch openai \
    --model gpt-5.1 \
    --doi-questions data/doi_questions.json \
    --enable-tool-calling --enable-filtering
```

### Azure Foundry models (DeepSeek)

Same procedure as above with `provider=azure`. Uses `COCHRANE_DASHBOARD_OPENAI_KEY` and `COCHRANE_DASHBOARD_BASE_URL` in `.env` (DeepSeek-V4-Pro is deployed on its own Azure resource); falls back to `AZURE_OPENAI_KEY` / `OPENAI_BASE_URL` if those are unset.

```bash
python -m sciconharness.cli_scripts.query_batch azure \
    --model DeepSeek-V4-Pro \
    --doi-questions data/doi_questions.json \
    --enable-tool-calling --enable-filtering
```

### OpenRouter models (Kimi K3 / GLM / Qwen3.8 / MiniMax M3)

Same procedure as above with `provider=openrouter`. Uses `OPENROUTER_API_KEY` in `.env`
(or `OPENROUTER_API_KEY_FILTERING` / `OPENROUTER_API_KEY_BASE_MODEL` when those are set).

```bash
python -m sciconharness.cli_scripts.query_batch openrouter \
    --model moonshotai/kimi-k3 \
    --doi-questions data/doi_questions.json \
    --enable-tool-calling --enable-filtering

python -m sciconharness.cli_scripts.query_batch openrouter \
    --model z-ai/glm-5.3 \
    --doi-questions data/doi_questions.json \
    --enable-tool-calling --enable-filtering

python -m sciconharness.cli_scripts.query_batch openrouter \
    --model qwen/qwen3.8-max \
    --doi-questions data/doi_questions.json \
    --enable-tool-calling --enable-filtering

# Controls
python -m sciconharness.cli_scripts.query_batch openrouter \
    --model qwen/qwen3.8-27b \
    --doi-questions data/doi_questions.json \
    --enable-tool-calling --enable-filtering

python -m sciconharness.cli_scripts.query_batch openrouter \
    --model minimax/minimax-m3 \
    --doi-questions data/doi_questions.json \
    --enable-tool-calling --enable-filtering
```

## Filter data (`--cochrane-titles` / `--doi-dates` / `--publication-date`)

**These flags are optional overrides, not requirements.** When `--enable-filtering` is set (the default) and you don't pass them, `SciConHarness` auto-resolves title-list filtering, source-title matching, and the publication-date cutoff straight from `--doi`, using a local JSON cache built once from the live `hayoungjung/SciConBench` HuggingFace dataset — see `sciconharness/utils/hf_benchmark_cache.py` and the "Clean Room Evaluation Protocol" section in `sciconharness/README.md`.

The only file you still need for a batch run is `--doi-questions` (there's no benchmark-wide default for the *questions* themselves):

```python
from datasets import load_dataset
import json, pathlib

ds = load_dataset("hayoungjung/SciConBench", "benchmark", split="test")
pathlib.Path("data").mkdir(exist_ok=True)
pathlib.Path("data/doi_questions.json").write_text(
    json.dumps({row["doi"]: row["question"] for row in ds}, indent=2)
)
```

Pass `--cochrane-titles <path/to/titles.json>` / `--doi-dates <path/to/doi_dates.json>` (or `--publication-date` to `query_single`) only when you need to **override** the auto-loaded cache — e.g. a custom filtering scope, or a DOI that isn't (yet) part of the tracked benchmark:

| File | query_single | query_batch |
|------|-------------|-------------|
| `data/doi_questions.json` | — | `--doi-questions` (required) |
| titles JSON (list of strings) | `--cochrane-titles` (optional) | `--cochrane-titles` (optional) |
| doi→publication_date JSON | `--publication-date` (single date, optional) | `--doi-dates` (optional) |

To force-refresh the auto-loaded cache itself (e.g. after the dashboard publishes new reviews and you want them locally without waiting for the next process to lazily rebuild it):

```bash
python -m sciconharness.utils.hf_benchmark_cache --force
```
