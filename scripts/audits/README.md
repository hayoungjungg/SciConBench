# Audit query scripts

Scripts to audit **Google AI Mode** and **Google AI Overview** (via SerpAPI) and from **OpenEvidence** (browser automation). Run them from this directory or invoke with `python scripts/audits/<script>.py`.

## Shared setup

1. **Python deps** (from repo root): `pip install -r requirements.txt` (includes `serpapi`, `python-dotenv`, `seleniumbase`, `beautifulsoup4`, etc.).

2. **SerpAPI** (`query_google_ai_mode.py`, `query_google_ai_overview.py`): set `SERPAPI_API_KEY` in the repo `.env` or in the environment. Scripts call `load_dotenv()` from the current working directory, so running from repo root or exporting the variable both work.

3. **Questions file format**: batch mode expects JSON **DOI → question** (string keys and values). Default file (both Google scripts): `scripts/experiments/main_experiment/data/querying/doi_to_question.json` (override with `-q`).

---

## Google AI Mode (`query_google_ai_mode.py`)

Uses SerpAPI’s `google_ai_mode` engine (single request per question).

```bash
python query_google_ai_mode.py                          # interactive: one question
python query_google_ai_mode.py "Your question here?"   # single question
python query_google_ai_mode.py --batch                  # all DOIs from default questions file
python query_google_ai_mode.py --batch -q path.json -o path/to/out
```

**Useful flags**: `-q` / `--questions-file`, `-o` / `--output-dir`, `--hl` (default `en`), `--max-retries`, `--retry-delay`.

**Defaults**: results under `scripts/experiments/main_experiment/data/querying/serpapi_ai_mode_results/` unless you pass `-o`.

**Customization**

- **Batch queries**: the script appends a fixed benchmark suffix (synthesis paragraph + wrap in `[[[...]]]`). **Single / interactive queries** are sent **as you typed them** (no suffix). To change batch wording, edit `BENCHMARK_QUERY_SUFFIX` / `format_benchmark_query()` near the top of the file.
- Tune retries with `--max-retries` and `--retry-delay` if you hit rate limits.

---

## Google AI Overview (`query_google_ai_overview.py`)

Uses a normal Google search via SerpAPI to obtain an AI Overview `page_token`, then fetches the overview payload. Retries when Google returns no overview. Note that queries are not guaranteed to generate an AI Oveview response.

```bash
python query_google_ai_overview.py
python query_google_ai_overview.py "Your question here?"
python query_google_ai_overview.py --batch
python query_google_ai_overview.py --batch -q path.json -o path/to/out
```

**Extra flags** (overview-specific): `--retry-if-no-overview` (default `2` → three tries total), `--retry-no-overview-delay` (default `15` seconds). Same `-q`, `-o`, `--hl`, `--max-retries`, `--retry-delay` as AI Mode.

**Defaults**: `-o` points to `scripts/audits/outputs/google_ai_overview/` so existing batch outputs are detected when you resume.

**Customization**

- Same **batch vs single** behavior as AI Mode: benchmark suffix only in **batch**; edit `BENCHMARK_QUERY_SUFFIX` if you need different instructions.
- If overviews are often missing, increase `--retry-if-no-overview` or `--retry-no-overview-delay`. Atomic-fact extraction exists in code but is not exposed on the CLI (single path uses `process_atomic_facts=False`).

---

## OpenEvidence (`query_openevidence.py`)

Opens **Chrome** (non-headless), you **log in manually** (need credentials), then the script runs searches and saves JSON (and optional HTML per DOI).

**Install Chrome**

- **Desktop**: system Chrome is enough if `CHROME_BINARY_PATH` stays `None`.
- **Linux servers / CI**: use [Chrome for Testing](https://googlechromelabs.github.io/chrome-for-testing/) — pick **stable**, **linux64**, download the **chrome** zip, unzip, `chmod +x chrome-linux64/chrome`, then set in the script:
  - `CHROME_BINARY_PATH = "/full/path/to/chrome-linux64/chrome"`
- **macOS**: usually system Chrome; if you need a pinned binary, use the same Chrome-for-Testing page (`mac-arm64` or `mac-x64`).

SeleniumBase can manage drivers; still use a Chrome build that matches your environment.

```bash
python query_openevidence.py                           # interactive prompt
python query_openevidence.py "Your research question"  # one query (all args joined)
python query_openevidence.py --batch                   # optional: path/to/questions.json
```

**Batch default questions file** if you omit the path after `--batch`: `scripts/data/sampled_100_questions.json` (DOI → question). JSON output directory: `scripts/audits/data/openevidence_results/`. Successful scrapes with a DOI also write `scripts/audits/openevidence_html/<doi_safe>.html` for later parsing.

**Customization** (top of `query_openevidence.py`)

| Constant | Role |
|----------|------|
| `CHROME_BINARY_PATH` | `None` = system Chrome; else path to the `chrome` binary |
| `USE_UNDETECTED_CHROME` | `True` = undetected Chrome (slower); `False` = faster default driver |
| `WAIT_AFTER_SEARCH` | Seconds to wait after submitting a search before scraping |
| `DELAY_BETWEEN_QUERIES` | Pause between batch queries |
| `OUTPUT_DIR` / `OPENEVIDENCE_HTML_DIR` | Where JSON and per-DOI HTML go |
| `DISABLE_IMAGES` / `DISABLE_CSS` | Speed tweaks (disabling CSS may break layout) |
| `SEARCH_SELECTORS`, `SUBMIT_SELECTORS`, `NEW_CONVERSATION_*`, `MAIN_CONTENT_SELECTORS`, `RESULT_ITEM_SELECTORS` | Update if the site’s DOM changes |
| `BENCHMARK_QUERY_SUFFIX` | **Single and batch** searches use `format_benchmark_query()` (question + suffix) |

If the search box or results break after a site update, adjust the selector lists.

---

## Output and resume

- **Google scripts**: batch mode skips DOIs whose output JSON already exists in `-o`.
- **OpenEvidence**: batch skips when the expected JSON for that DOI already exists; failed DOIs can be listed in `openevidence_failed_dois.json` under the output directory. You can retry for these failed DOIs

Filenames are typically `serpapi_ai_mode_<doi_safe>.json`, `serpapi_ai_overview_<doi_safe>.json`, or `openevidence_<doi_safe>.json`.
