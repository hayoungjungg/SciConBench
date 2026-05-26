# Analysis Scripts

Scripts for reproducing the analysis figures and tables in the paper.

## 1. Download Data

From the repo root, run:

```bash
pip install -r requirements.txt
python scripts/download_data.py
```

This downloads and extracts four archives from Google Drive into `data/`:

| Archive | Contents |
|---|---|
| `labeled_facts.zip` | LLM-judge-labeled facts for factual precision & recall (**required** for all analysis scripts) |
| `preprocessed_facts.zip` | Decomposed atomic facts per model/DOI (required for `length_vs_metrics.py`) |
| `model_response.zip` | Raw generated conclusions |
| `llm-judge-human-annotations.zip` | Expert annotations used to validate the LLM judge |

## 2. Running the Analysis Scripts

All scripts are run from the repo root (or any directory — paths are resolved relative to the script location).

### Metrics table — `compute_macro_metrics_table.py`

Computes macro (per-DOI) recall, precision, and F1 for every model and writes three CSVs to `data/labeled_facts/`. The other plotting scripts call this automatically if the CSVs are missing, but you can also run it explicitly:

```bash
python scripts/analysis/compute_macro_metrics_table.py          # prints table + writes CSVs
python scripts/analysis/compute_macro_metrics_table.py --no-table  # CSVs only
```

Outputs:
- `data/labeled_facts/metrics_macro_per_doi_by_model.csv`
- `data/labeled_facts/metrics_macro_variance_by_model.csv`
- `data/labeled_facts/metrics_macro_variance_grouped_by_family.csv`

### Label distribution — `compute_label_distribution.py`

Prints the percentage of SUPPORTED / NOT SUPPORTED / CONTRADICTED labels for each model variant.

```bash
python scripts/analysis/compute_label_distribution.py
```

### Cost vs. metrics — `plot_cost_vs_metrics.py`

Scatter plots of cost (USD, log scale) vs. F1/precision/recall with Pareto frontiers. Uses `data/model_cost.json` bundled in this directory.

```bash
python scripts/analysis/plot_cost_vs_metrics.py
```

Figures are saved under `scripts/analysis/figures/tools_plus_filter/` and `scripts/analysis/figures/tools_only/`.

### Latency vs. metrics — `plot_time_vs_metrics.py`

Same layout as the cost plot but with average query time (seconds, log scale) on the x-axis. Uses `data/model_time.json`.

```bash
python scripts/analysis/plot_time_vs_metrics.py
```

Figures are saved under `scripts/analysis/figures/tools_plus_filter/` and `scripts/analysis/figures/tools_only/`.

### Conclusion length vs. metrics — `length_vs_metrics.py`

Regression plots of conclusion length (words) vs. recall and precision for filter-only model variants. Requires `data/preprocessed_facts/`.

```bash
python scripts/analysis/length_vs_metrics.py
```

Figures are saved under `scripts/analysis/figures/length_vs_metrics/`.
