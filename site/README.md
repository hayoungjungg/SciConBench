# SciConBench project website

The public dashboard for the live benchmark: <https://sciconbench.cs.princeton.edu>

It is a **static bundle** — no server, no build step, no third-party requests at
runtime. Every number on the page is baked into a single JSON file that
`export_data.py` generates from the tracking database.

```
site/
├── export_data.py       SQLite + workflow logs  ->  public/data/dashboard.json
├── site.config.json     editable copy: links, news, team, FAQ, citation
├── publish.sh           rsync public/ -> /n/fs/sciconbench/www
└── public/              the deployable web root
    ├── index.html
    ├── assets/{styles.css, app.js, charts.js}
    └── data/dashboard.json   (generated — safe to delete and rebuild)
```

## Publishing

```bash
./site/publish.sh              # regenerate data, then deploy
./site/publish.sh --dry-run    # preview the file changes only
./site/publish.sh --no-export  # deploy public/ without touching the data
```

Anything placed in `/n/fs/sciconbench/www` is served at the public URL, so
`publish.sh` is the whole deployment story. Set `SCICON_WEB_ROOT` to stage
somewhere else.

## Refreshing after a monthly run

The pipeline runs on the 1st of each month and the judging stages finish hours
later, so the site should be regenerated once grading has landed:

```bash
cd /n/fs/hamcore/hayoung/SciConBench
./site/publish.sh
```

To keep it hands-off, add a crontab entry a day after the pipeline fires:

```cron
0 6 2 * *  cd /n/fs/hamcore/hayoung/SciConBench && ./site/publish.sh >> data_track/logs/site-publish.log 2>&1
```

## What gets read

| Source | Used for |
|--------|----------|
| `data_track/sciconbench_track.db` | reviews, panels, atomic facts, model responses, precision/recall scores |
| `data_track/logs/workflow-*.log` (newest) | live stage-by-stage status of the latest run |
| `scicon-track/config/query_batch_config.yaml` | model roster and `always` vs `once` re-eval policy |
| `site.config.json` | links, news items, team, FAQ, citation |

Only responses with `config_label = tools_filter` are scored, matching what the
pipeline actually runs. Ungraded responses render as `pending` badges rather
than being dropped, so a partially complete run is still an honest page.

## Previewing locally

```bash
python3 -m http.server 8899 --directory site/public
# open http://127.0.0.1:8899
```

`file://` will not work — the page fetches its data over HTTP.

## Previewing the layout before grades exist

```bash
python3 site/export_data.py --demo
```

This fills ungraded rows with synthetic scores so the leaderboard and trend
chart can be judged visually. The page then renders a prominent orange
**"Preview build"** banner and sets `"demo": true` in the JSON. Re-run
`export_data.py` without the flag before publishing.

## Editing content

Prose that isn't derived from data lives in two places:

- **`site.config.json`** — tagline, description, links, news, team, FAQ,
  citation. No code changes needed.
- **`public/index.html`** — section headings and the explanatory copy about the
  live loop and the metrics.

Model display names and provider colours are in the `DISPLAY_NAMES` and
`PROVIDER_META` tables at the top of `export_data.py`; a model missing from
them falls back to its raw API name and a neutral grey.
