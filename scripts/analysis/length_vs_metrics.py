"""
Figure for # atomic facts vs. recall + precision
for filter-only model variants.

Key story: generating more atomic facts trades recall for precision —
recall gains are weak/inconsistent, precision loss is consistent.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_macro_metrics_table import compute_metrics, DEFAULT_BASE_DIR  # noqa: E402

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

# ── Matplotlib global style ──────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":           "serif",
    "font.serif":            ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":             11,
    "axes.titlesize":        12,
    "axes.labelsize":        11,
    "xtick.labelsize":       10,
    "ytick.labelsize":       10,
    "legend.fontsize":       9,
    "legend.title_fontsize": 9.5,
    "axes.linewidth":        0.8,
    "xtick.major.width":     0.8,
    "ytick.major.width":     0.8,
    "xtick.minor.visible":   False,
    "ytick.minor.visible":   False,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.grid":             True,
    "grid.color":            "#e4e4e4",
    "grid.linewidth":        0.6,
    "grid.alpha":            1.0,
    "figure.dpi":            150,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "savefig.pad_inches":    0.06,
})

_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _REPO_ROOT / "data"
FACTS_DIR = DATA_DIR / "preprocessed_facts"
METRICS_CSV = DATA_DIR / "labeled_facts" / "metrics_macro_per_doi_by_model.csv"
OUT_DIR = Path(__file__).resolve().parent / "figures" / "length_vs_metrics"
os.makedirs(OUT_DIR, exist_ok=True)

if not METRICS_CSV.exists():
    print(f"[Auto] {METRICS_CSV.name} not found — computing metrics first...")
    compute_metrics(DEFAULT_BASE_DIR)

# ── Filter-only models & display config ─────────────────────────────────────
FILTER_MODELS = {
    "claude-sonnet-4-5_tools_filter":               ("Claude Sonnet 4.5",        "#D95F02"),
    "gemini-3-pro-preview_tools_filter":            ("Gemini 3 Pro",             "#1F78B4"),
    "gpt-5.1_tools_filter":                         ("GPT-5.1",                  "#1A9850"),
    "o3-deep-research-2025-06-26_tools_filter":     ("o3 Deep Research",         "#6B3FA0"),
    "o4-mini-deep-research-2025-06-26_tools_filter":("o4-mini Deep Research",    "#9970CB"),
    "dr-tulu_tools_filter":                         ("DR-Tulu",                  "#B2182B"),
    "sonar-deep-research_filter":                   ("Sonar Deep Research",      "#E07722"),
    "sonar-reasoning-pro_filter":                   ("Sonar Reasoning Pro",      "#B8860B"),
}

# ── Load data ────────────────────────────────────────────────────────────────
def load_facts(facts_dir, model_set):
    rows = []
    for fname in sorted(os.listdir(facts_dir)):
        if not fname.endswith("_atomic_facts.json"):
            continue
        model = fname.replace("_atomic_facts.json", "")
        if model not in model_set:
            continue
        with open(os.path.join(facts_dir, fname)) as f:
            data = json.load(f)
        for doi_key, entry in data.items():
            raw = entry.get("doi", doi_key)
            doi = raw.replace("_", "/", 1) if raw.startswith("10.1002_") else raw
            conclusion_text = entry.get("conclusion_text", "")
            n_words = len(conclusion_text.split())
            rows.append({"model": model, "doi": doi, "n_words": n_words})
    return pd.DataFrame(rows)


facts_df = load_facts(FACTS_DIR, set(FILTER_MODELS))
metrics_df = pd.read_csv(METRICS_CSV)
merged = metrics_df.merge(facts_df, on=["model", "doi"], how="inner")
merged = merged[merged["model"].isin(FILTER_MODELS)]

# ── Compute per-model regression stats for both metrics ──────────────────────
stats_table = {}
for model in FILTER_MODELS:
    sub = merged[merged["model"] == model].dropna(subset=["n_words", "recall", "precision"])
    if len(sub) < 10:
        continue
    x = sub["n_words"].values
    sl_r, ic_r, r_r, p_r, _ = stats.linregress(x, sub["recall"].values)
    sl_p, ic_p, r_p, p_p, _ = stats.linregress(x, sub["precision"].values)
    stats_table[model] = dict(
        r_recall=r_r, p_recall=p_r,
        r_prec=r_p,   p_prec=p_p,
        avg_words=x.mean(), n=len(sub),
    )

print(f"\n{'Model':<45}  {'r(recall)':>9}  {'p':>6}  {'r(prec)':>9}  {'p':>6}  avg_words")
for model, s in stats_table.items():
    lbl = FILTER_MODELS[model][0]
    print(f"  {lbl:<43}  {s['r_recall']:>+9.3f}  {s['p_recall']:>6.3f}  "
          f"{s['r_prec']:>+9.3f}  {s['p_prec']:>6.3f}  {s['avg_words']:>6.1f}")

# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5), sharey=False)

panel_cfg = [
    ("recall",     "Factual Recall",    axes[0]),
    ("precision",  "Factual Precision", axes[1]),
]

legend_handles = []   # for shared legend (model name + color only)

for ycol, ylabel, ax in panel_cfg:
    # Sort models by their regression slope so annotation offsets avoid overlap
    model_regressions = []
    for model, (lbl, color) in FILTER_MODELS.items():
        sub = merged[merged["model"] == model].dropna(subset=["n_words", ycol])  # ycol is "recall" or "precision"
        if len(sub) < 20:
            continue
        x = sub["n_words"].values
        y = sub[ycol].values
        sl, ic, r, p, _ = stats.linregress(x, y)
        model_regressions.append((model, lbl, color, x, y, sl, ic, r, p))

    # Sort by y-value at x=400 (right side of plot) so stagger offsets make sense
    model_regressions.sort(key=lambda t: t[5] * 400 + t[6], reverse=True)

    for idx, (model, lbl, color, x, y, sl, ic, r, p) in enumerate(model_regressions):
        # Regression line only
        xr = np.linspace(x.min(), x.max(), 200)
        line, = ax.plot(
            xr, sl * xr + ic,
            color=color, lw=2.0, alpha=0.92, zorder=3,
            label=lbl,
        )
        if ycol == "recall":
            legend_handles.append(line)

    ax.set_xlabel("Conclusion Length (words)", labelpad=4)
    ax.set_ylabel(ylabel, labelpad=4)
    ax.set_xlim(left=0)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(length=3)

# Shared legend — model names + colors, below both panels
fig.legend(
    legend_handles,
    [h.get_label() for h in legend_handles],
    loc="lower center",
    ncol=4,
    fontsize=8.5,
    bbox_to_anchor=(0.5, -0.1),
    frameon=True,
    framealpha=0.97,
    edgecolor="#cccccc",
    handlelength=1.4,
    columnspacing=1.0,
)

plt.tight_layout(w_pad=1.5)
out_png = os.path.join(OUT_DIR, "facts_vs_precision_recall.png")
plt.savefig(out_png, dpi=200)
print(f"\nSaved: {out_png}")
plt.close()
