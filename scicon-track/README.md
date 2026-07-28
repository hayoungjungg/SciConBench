# scicon-track — SciConBench-Track CLI

`scicon-track` is the command-line interface for the **SciConBench-Track** longitudinal pipeline.  
It orchestrates the full monthly workflow: discovering new Cochrane reviews, downloading PDFs,  
extracting reference text, querying LLMs, and running factual precision/recall analysis.

---

## Installation

From the repository root:

```bash
pip install -e .
```

Verify:

```bash
scicon-track --help
```

---

## Environment variables

Copy `.env.example` to `.env` and fill in:

```dotenv
# LLM providers (same as the static benchmark)
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
# etc. — see .env.example for the full list

# Data collection
WILEY_TDM_TOKEN=...          # Wiley TDM API token for PDF downloads
CROSSREF_MAILTO=...          # Contact email for Crossref polite pool (optional)

# HuggingFace upload
HF_TOKEN=...                 # Write token from https://huggingface.co/settings/tokens
```

Paths default to `scicon-track/data_track/` inside the repo.  
Override the base directory with:

```bash
export SCICON_DATA_DIR=/path/to/your/data
```

---

## Quick start

```bash
# 1. Install
pip install -e .

# 2. Set up secrets
cp .env.example .env   # then fill in .env

# 3. Initialize the database
scicon-track init-db

# 4. Smoke-test the full pipeline with 5 DOIs
scicon-track workflow --once --max-dois 5

# 5. Full monthly run
scicon-track workflow --once
```

---

## Commands

### `scicon-track workflow`

Runs the full monthly pipeline.  Pass `--once` to execute a single run and exit;  
omit it to start a Prefect scheduler that re-triggers every 30 days.

```
Usage: scicon-track workflow [OPTIONS]

Options:
  --once            Run once immediately instead of starting the scheduler.
  --max-dois N      Limit to N DOIs per run (smoke-testing).
  --batch-size INT  Atomic-fact batch size.  [default: 500]
  --help            Show this message and exit.
```

**One-shot run:**

```bash
scicon-track workflow --once
```

**Smoke-test with 5 DOIs:**

```bash
scicon-track workflow --once --max-dois 5
```

**Start the 30-day scheduler:**

```bash
scicon-track workflow
```

---

### `scicon-track init-db`

Create the SQLite database and all tables (idempotent).

```bash
scicon-track init-db
scicon-track init-db --force    # drop and recreate all tables
```

---

### `scicon-track discover`

Fetch new Cochrane reviews from Crossref and register them as rolling reviews.

```bash
scicon-track discover
scicon-track discover --cohort-month 2026-07 --limit 20
```

---

### `scicon-track download-pdfs`

Download PDFs for all rolling reviews awaiting download via the Wiley TDM API.

```bash
scicon-track download-pdfs
```

---

### `scicon-track extract-text`

Extract reference text from downloaded PDFs and persist it to the database.

```bash
scicon-track extract-text
```

---

### `scicon-track upload`

Upload the current benchmark state to HuggingFace as a Parquet dataset.

```bash
scicon-track upload
```

---

## Pipeline stages

The `workflow` command executes these Prefect tasks in order:

| # | Task | Description |
|---|------|-------------|
| 1 | `task_init_db` | Create/verify SQLite schema |
| 2 | `task_register_core_set` | Load N=200 core DOIs from the HuggingFace benchmark |
| 3 | `task_discover_rolling` | Find 15–20 new CDSR reviews via Crossref |
| 4 | `task_download_pdfs` | Download PDFs via Wiley TDM API |
| 5 | `task_extract_text` | Extract reference text with pdfplumber |
| 6 | `task_generate_questions` | Generate clinical questions for rolling reviews |
| 7 | `task_generate_cochrane_facts_batch` | Atomic-fact decomposition of Cochrane conclusions |
| 8 | `task_upload_to_hf` | Merge + publish updated dataset to HuggingFace; also refreshes `sciconharness`'s local Cochrane-filter caches (titles / doi→title / doi→publication_date) from the same merged rows — see `huggingface/uploader.py::refresh_filter_caches()` |
| 9 | `task_run_queries` | Query all model configs with SciConHarness (picks up the just-refreshed filter caches from stage 8) |
| 10 | `task_generate_response_facts_batch` | Atomic-fact decomposition of model responses |
| 11 | `task_run_precision` | LLM-judge precision analysis |
| 12 | `task_run_recall` | LLM-judge recall analysis |

Stages 6–12 retry up to 3 times with exponential backoff.  
Email notifications are sent on pipeline start, success, and failure.

---

## Panel types

Every DOI in the database belongs to one of two panels:

| Panel | Description |
|-------|-------------|
| `core` | The stable set of N=200 reviews from the original SciConBench benchmark.  Re-evaluated every month to track model drift. |
| `rolling` | 15–20 newly published Cochrane reviews added each month.  Prevents benchmark overfitting and extends coverage. |

Each rolling review is tagged with its `cohort_month` (e.g. `"2026-07"`) to enable  
stratified temporal analysis.

---

## Configuration

All behaviour is controlled by YAML files in `scicon-track/config/`.  No code changes needed.

| File | Controls |
|------|----------|
| `config/config.yaml` | Paths, data collection settings |
| `config/llm_judge_config.yaml` | LLM judge for precision/recall labeling |
| `config/query_batch_config.yaml` | Models/providers for harness queries |
| `config/hugging_face_config.yaml` | Source + upload dataset on HuggingFace |

### `query_batch_config.yaml` — models run every month

`default_models` maps each provider to either a single model name or a
**list** of model names — the list form lets one provider run several
distinct models every run (used for OpenRouter, which hosts multiple
models behind a single provider):

```yaml
default_models:
  openai: gpt-5.6-sol
  claude: claude-opus-5
  gemini: gemini-3.1-pro
  perplexity: sonar-reasoning-pro
  azure: DeepSeek-V4-Pro          # Azure Foundry Chat Completions
  openrouter:                     # one provider, several models
    - moonshotai/kimi-k3          # Kimi K3
    - z-ai/glm-5.2                # GLM-5.2
    - qwen/qwen3.5-9b             # Qwen3.5-9B
    - qwen/qwen3.7-max            # Qwen3.7-max
```

Every `(provider, model)` pair produced by `QueryBatchConfig.iter_models()`
is queried in `task_run_queries` under all three `HARNESS_CONFIGS`
(`no_tools`, `tools`, `tools_filter`) for every DOI, so adding a model here
is enough to fully onboard it into the monthly pipeline — no other code
changes required.

`task_run_queries` never force-passes OpenAI/Azure OpenAI credentials to
`SciConHarness`; it leaves `api_key`/`base_url`/`api_version` unset and lets
each provider resolve its own credentials from `.env` (same resolution
`create_provider()` in `sciconharness/utils/query_utils.py` uses for the
CLI scripts) — DeepSeek-V4-Pro's dedicated `COCHRANE_DASHBOARD_*` resource,
the `OPENROUTER_API_KEY*` variants, Gemini's Vertex AI fallback, and
Claude's Foundry auto-detection all keep working correctly regardless of
which providers/models are listed above.
