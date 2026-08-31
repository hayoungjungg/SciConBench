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

# 4. Draw the one-time core set (up to 10 reviews/month, Jul 2025–Jun 2026, ~120)
scicon-track init-core-set

# 5. Smoke-test the full pipeline with 5 DOIs
scicon-track workflow --once --max-dois 5 --rolling-month 2026-07

# 6. Full run: ingest all new reviews; evaluate core + closed rolling months
scicon-track workflow --once
```

---

## Commands

### `scicon-track workflow`

Runs the full monthly pipeline.  Pass `--once` to execute a single run and exit;
omit it to start a Prefect scheduler that fires on the 1st of each calendar
month (`cron 0 0 1 * *`, America/New_York). Use `--interval bimonthly` for the
1st of odd months. Requires a core set created once via `scicon-track init-core-set`.

```
Usage: scicon-track workflow [OPTIONS]

Options:
  --once                  Run once immediately instead of starting the scheduler.
  --max-dois N            Limit to N DOIs per run (smoke-testing).
  --batch-size INT        Atomic-fact batch size.  [default: 500]
  --rolling-month YYYY-MM Latest closed month for evals (default: previous calendar month).
  --interval [monthly|bimonthly]
                          Scheduler cadence (ignored with --once).  [default: monthly]
  --help                  Show this message and exit.
```

**One-shot run (ingest new reviews; evaluate through the previous calendar month):**

```bash
scicon-track workflow --once
```

**Hold evals at July 2026 (August-published rolling reviews are ingested but not queried):**

```bash
scicon-track workflow --once --rolling-month 2026-07
```

**Start the calendar-month scheduler:**

```bash
scicon-track workflow
```

---

### `scicon-track init-core-set`

Draw the one-time core set from the **curated HuggingFace dataset**
(`hayoungjung/SciConBench`, config `benchmark`, split `test`) — not raw
Crossref, which also lists protocols. Up to 10 already-curated reviews from
each calendar month between 2025-07-01 and 2026-06-30 (window / per-month
cap / seed are in `config/config.yaml`). Questions and Cochrane facts are
copied from HuggingFace, so these DOIs start at `FACTS_GENERATED`.

The sample is written to the DB **and** locked in `data_track/core_set.json`
only after every month has been drawn. The monthly pipeline refuses to start
until that lock file exists. Re-running `init-core-set` is a no-op once the
lock is in place. Pass `--force` to drop the CORE panel and redraw from
HuggingFace. `scicon-track init-db --force` also refuses to drop the DB
while the lock exists, unless you also pass `--force-drop-core-set` (the
lock file itself is still not deleted).

```bash
scicon-track init-core-set
scicon-track init-core-set --force   # redraw from HuggingFace
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

Fetch new Cochrane reviews that are not already on HuggingFace, and prune stale core/rolling DOIs. Rolling `cohort_month` is assigned later from each review's publication date.

```bash
scicon-track discover
scicon-track discover --limit 20
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

Upload the current benchmark state to HuggingFace
(`hayoungjung/SciConBench`, config `benchmark`, split `test`).
Only **core** plus rolling reviews with `cohort_month` on or before the
closed month are published (default: previous calendar month). Open-month
rolling reviews stay local. After the Parquet push, a second Hub commit
puts a ``Latest update`` blurb at the top and folds older months into a
collapsible ``Previous updates`` section (before/after sample counts included).

```bash
scicon-track upload
scicon-track upload --closed-month 2026-07
```

---

## Pipeline stages

The `workflow` command executes these Prefect tasks in order:

| # | Task | Description |
|---|------|-------------|
| 1 | `task_init_db` | Create/verify SQLite schema |
| 2 | `task_load_core_set` | Read the finalized core set from `data_track/core_set.json` (fails if `init-core-set` has not finished) |
| 3 | `task_discover_and_prune` | Discover new CDSR reviews not already on HuggingFace or in the DB; if a newer `.pubN` of a tracked review appears, drop the old DOI and add the successor as rolling (never back into core). `cohort_month` is the publication month, set after text extraction. |
| 4 | `task_download_pdfs` | Download PDFs via Wiley TDM API. A 404 is retried on later runs; after 5 distinct calendar months of 404s the DOI is skipped. |
| 5 | `task_extract_text` | Extract reference text, assign rolling publication-month cohorts, and (after 12 rolling months) advance the 12-month core by one cohort |
| 6 | `task_generate_questions` | Generate clinical questions |
| 7 | `task_generate_cochrane_facts_batch` | Atomic-fact decomposition of Cochrane conclusions |
| 8 | `task_upload_to_hf` | Merge + publish **closed-month** core+rolling rows to `hayoungjung/SciConBench` (benchmark/test); refresh `sciconharness` Cochrane-filter caches; then (production only) prepend a newest-first Hub README changelog entry in a second commit. Superseded `.pubN` DOIs are dropped from the merge. |
| 9 | `task_run_queries` | Query models with SciConHarness against the current core plus the latest **4** closed rolling cohorts. By default every model queries each DOI **once** (skip if any prior response exists). Opt into per-run re-query via `reevaluate_always` in `query_batch_config.yaml`. A newly added model therefore evaluates the core and four-month rolling window, never the entire rolling history. Only `FACTS_GENERATED` DOIs are queried. Each pair is retried per-item, then leftover pairs are re-queried in whole-stage rounds; leftover pending DOI/model pairs fail the task. |
| 10 | `task_generate_response_facts_by_model` | Atomic-fact decomposition of model responses |
| 11 | `task_run_precision` | LLM-judge precision analysis |
| 12 | `task_run_recall` | LLM-judge recall analysis |

Stages 4–12 retry each incomplete item (`ITEM_RETRY_ATTEMPTS`, default 3) and then re-scan remaining work (`STAGE_ROUNDS`, default 3). Atomic-fact stages use a higher per-item budget (`FACTS_ITEM_RETRY_ATTEMPTS`, default 4) before those whole-stage rounds. Queries work the same way: try each pending DOI/model pair a few times, then re-query whatever still has no well-formed `[[[...]]]` conclusion. A stage **raises** if anything is still missing or malformed after that — it does not skip bad output. Prefect then retries the task. Wiley TDM 404s are retried on later runs and do not fail the download stage; they become `SKIPPED` only after 5 distinct calendar months of 404s.

Email notifications are sent on pipeline start, after each stage finishes
(a progressively longer digest), and on final success or failure.

---

## Panel types

Every DOI in the database belongs to one of two panels:

| Panel | Description |
|-------|-------------|
| `core` | Initially, up to 10 curated reviews per month from Jul 2025–Jun 2026, drawn once via `scicon-track init-core-set`. After 12 rolling cohorts have closed, this becomes a monthly sliding 12-cohort window: the earliest core cohort is demoted to rolling and the oldest rolling cohort is promoted to core. For example, the Jun 2027 close replaces Jul 2025 with Jul 2026; the Jul 2027 close replaces Aug 2025 with Aug 2026. This keeps the core no more than two years old. Every model queries each current core DOI once. |
| `rolling` | New CDSR reviews not already on HuggingFace, tagged by publication `cohort_month`. Open-month rows stay local until the month closes. Query evaluation includes only the latest `rolling_panel_months` closed cohorts (default **4**) outside the current core; older rolling history remains stored and published but is not backfilled for newly added models. A newer `.pubN` is assigned to rolling even when it replaces a core DOI. |

Each rolling review is tagged with its `cohort_month` (e.g. `"2026-07"`), which is the publication month, not the run date.
Panel membership is mirrored to `data_track/doi_panels.json` for inspection.

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
models behind a single provider).

Current monthly roster is **7 frontier models + 3 controls** (10 total).
Controls are all open-weight, frontier models as of the start of the study
running alongside the primary panel for comparison:

| Role | Models |
|------|--------|
| Primary | GPT-5.6 Sol, Claude Opus 5, Gemini 3.7 Flash, DeepSeek-V4-Pro, Kimi K3, GLM-5.3, Qwen3.8-max |
| Control | DeepSeek-V4-Flash-0731, Qwen3.8 27B, MiniMax M3 |

```yaml
rolling_panel_months: 4             # newest closed rolling cohorts to evaluate

default_models:
  openai: gpt-5.6-sol
  claude: claude-opus-5
  gemini: gemini-3.7-flash
  azure:
    - DeepSeek-V4-Pro              # Azure Foundry Chat Completions
    - DeepSeek-V4-Flash-0731       # control
  openrouter:                      # one provider, several models
    - moonshotai/kimi-k3           # Kimi K3
    - z-ai/glm-5.3                 # GLM-5.3
    - qwen/qwen3.8-max             # Qwen3.8-max
    - qwen/qwen3.8-27b             # control
    - minimax/minimax-m3           # control

# OpenRouter key assignment (single sequential query lane — not concurrent):
openrouter_base_model_lane:        # OPENROUTER_API_KEY_BASE_MODEL
  - z-ai/glm-5.3
  - qwen/qwen3.8-max
openrouter_generic_lane:           # OPENROUTER_API_KEY
  - moonshotai/kimi-k3             # first among this key's models
  - qwen/qwen3.8-27b               # control
  - minimax/minimax-m3             # control

reevaluate_always: []              # empty = all models query each DOI once
```
Every `(provider, model)` pair produced by `QueryBatchConfig.iter_models()`
is queried in `task_run_queries` under a single **clean-room** configuration
(`tools_filter`). By default every model skips DOIs it has already answered;
later runs only pick up newly closed rolling DOIs (and still-missing pairs
within the four-month rolling window). Put a model in `reevaluate_always` only
if you want the current core + rolling window re-queried every run. Adding a
new model name to `default_models` evaluates the current core and latest four
rolling cohorts; older rolling history is deliberately not backfilled.

### Pipeline concurrency

The pipeline does not run every model/item strictly one at a time, but it's
deliberately conservative about *where* it adds concurrency — grouping by
credential so nothing contends with itself for a rate limit — rather than
maximizing parallelism everywhere.

**Query stage (`task_run_queries` / `_run_provider_lane`, step 9).**
`(provider, model)` pairs are grouped into **lanes** (`QUERY_LANES`), and
lanes run concurrently against each other via `asyncio.gather`; *within* a
lane, models — and DOIs within a model — are queried strictly sequentially:

- `openrouter` — **all** OpenRouter models in one sequential lane (at most
  one OpenRouter request in flight, to avoid 402/429 in-flight storms).
  Per-model key still follows YAML membership:
  `openrouter_base_model_lane` → **`OPENROUTER_API_KEY_BASE_MODEL`**
  (GLM-5.3, Qwen3.8-max); everything else → **`OPENROUTER_API_KEY`**
  (Kimi K3, Qwen3.8-27B, MiniMax M3). Run order is base list then generic list.
- `azure_openai` — OpenAI GPT (`gpt-5.6-sol`) **then** DeepSeek-V4-Pro /
  DeepSeek-V4-Flash-0731 (control), all on **`COCHRANE_DASHBOARD_OPENAI_KEY`** +
  **`COCHRANE_DASHBOARD_BASE_URL`** (falls back to `AZURE_OPENAI_KEY` /
  `OPENAI_BASE_URL` if unset). GPT always runs before DeepSeek so they
  never share the Cochrane Dashboard quota concurrently.
- `azure_anthropic` — Claude on Azure Foundry
  (`AZURE_ANTHROPIC_API_KEY`, `AZURE_ANTHROPIC_BASE_URL`,
  `AZURE_ANTHROPIC_RESOURCE_NAME`).
- `gemini` — `gemini-3.7-flash` (Vertex AI / Google env vars).

A single `SciConHarness` instance is never called concurrently (it mutates
per-instance state, e.g. the OpenRouter sticky-routing `session_id`, right
before each query) — see `_run_provider_lane` in `run_workflow.py`.

The track force-passes Cochrane Dashboard credentials for `openai`/`azure`
and Azure Anthropic credentials for `claude`. OpenRouter and Gemini leave
`api_key`/`base_url` unset so `create_provider()` resolves their env vars
normally.

**Post-query stages (atomic facts, precision, recall — steps 10-11,
`_run_grouped_sharded`).** These three stages run strictly one after
another (facts → precision → recall), and only start once `task_run_queries`
has fully finished — so they're free to reuse the same two Azure credentials
(`FACTS_JUDGE_API_KEYS`: `AZURE_OPENAI_KEY` + `COCHRANE_DASHBOARD_OPENAI_KEY`)
without any cross-stage contention:

- **Atomic facts** (`task_generate_response_facts_by_model`): pending
  model-response items are grouped by generating model, then processed
  **two models at a time** (one dedicated API key per model), each model's
  items further split into `FACTS_SHARD_CONCURRENCY` (4) concurrent shards
  — up to 2 × 4 = 8 concurrent `AtomicFactGenerator` calls at once — repeating
  in batches of two until every model has been processed.
- **Precision** (`task_run_precision`), then **recall**
  (`task_run_recall`) once precision is fully done: since these aren't
  grouped by model, the full pending-item queue is instead split into two
  halves up front, one per API key, each further 4-way sharded (again up to
  8 concurrent judge calls).

Because `ThreadPoolExecutor` worker threads run genuinely concurrently
(unlike the query stage's cooperative `asyncio` scheduling), each of these
stages wraps its `populate_*` DB write in `_DB_WRITE_LOCK` — SQLite only
allows one writer at a time — so only the brief write is serialized, not the
slow LLM call before it.

### Per-DOI result files (human-browsable mirror of the DB)

The SQLite DB (`data_track/sciconbench_track.db`) is the source of truth,
but every DOI/model/run_month triple also gets a plain JSON file at:

```
data_track/results/<run_month>/<model>_tools_filter/<doi_safe>/result.json
data_track/results/<run_month>/<model>_tools_filter/<doi_safe>/mcp_client.log
```

The harness query/tool-call log (`mcp_client.log`) is written into the same
per-DOI directory as `result.json` (track sets `SciConHarness(log_dir=...)`
to `data_track/results/<run_month>/`). Static benchmark runs still default
to `sciconharness/logs/` when `log_dir` is unset.

The `<run_month>` partition matters because the pipeline runs monthly (or
bimonthly) against a sliding core plus a bounded rolling panel: the same
DOI/model pair can legitimately get queried again in a later month, and
without partitioning by month that re-run would silently overwrite the
earlier month's file even though the DB keeps them as distinct rows (unique
on `doi, model, provider, config_label, run_month`). Within one
`results/<run_month>/` directory, the layout matches what
`scripts/check_conclusions.py --logs-dir data_track/results/<run_month>`
expects (it checks each file's `"response"` field for a well-formed
`[[[...]]]` block).

Each pipeline stage merges its own top-level key into the file as that
DOI/model reaches it (see `_update_result_file` / `_results_file_path` in
`run_workflow.py`):

- `task_run_queries` writes `doi`, `model`, `provider`, `config_label`,
  `run_month`, `query`, `response`, `token_usage`.
- `task_generate_response_facts_by_model` adds `atomic_facts_pairs` and
  `total_atomic_facts`.
- `task_run_precision` adds `precision` (the full judge result dict:
  `factual_precision`, supported/contradicted/not-supported facts, etc.).
- `task_run_recall` adds `recall` (`factual_recall`, coverage details, etc.).

Since each stage only adds/overwrites its own key, a file that's missing
`atomic_facts_pairs` means fact generation hasn't run for that DOI/model
yet, one missing `precision`/`recall` means grading hasn't run, and a
`response` without a well-formed `[[[...]]]` block means generation itself
needs a retry — all inspectable directly from the file on disk (or via a
`check_conclusions.py`-style scan) without touching the DB, which also
makes it straightforward to identify and re-run just the DOIs that are
missing or malformed at any given stage.
