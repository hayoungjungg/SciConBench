#!/usr/bin/env python3
"""
Scatter average time per query (seconds, log x-axis) vs factual metrics from labeled-facts results.
Uses model_time.json avg_time_seconds joined to metrics_macro_variance_by_model.csv.

Produces tools-only and tools+filter panel sets x three metrics (F1, precision, recall).
Pareto fill: light yellow region in the top-left (between the Pareto step hull and y-max).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_macro_metrics_table import compute_metrics, DEFAULT_BASE_DIR  # noqa: E402

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Polygon
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None  # type: ignore[misc, assignment]

CSV_MODEL_TO_COST_KEY: Dict[str, Tuple[str, str]] = {
    "gpt-5.1": ("non_mcp_models", "GPT-5.1"),
    "gpt-5.1_tools": ("mcp_no_filter", "GPT-5.1"),
    "gpt-5.1_tools_filter": ("mcp_filter", "GPT-5.1"),
    "claude-sonnet-4-5": ("non_mcp_models", "Claude Sonnet 4.5"),
    "claude-sonnet-4-5_tools": ("mcp_no_filter", "Claude Sonnet 4.5"),
    "claude-sonnet-4-5_tools_filter": ("mcp_filter", "Claude Sonnet 4.5"),
    "gemini-3-pro-preview": ("non_mcp_models", "Gemini 3"),
    "gemini-3-pro-preview_tools": ("mcp_no_filter", "Gemini 3"),
    "gemini-3-pro-preview_tools_filter": ("mcp_filter", "Gemini 3"),
    "dr-tulu_tools": ("deep_research_no_filter", "Dr Tulu"),
    "dr-tulu_tools_filter": ("deep_research_filter", "Dr Tulu"),
    "o3-deep-research-2025-06-26_tools": ("deep_research_no_filter", "o3 Deep Research"),
    "o3-deep-research-2025-06-26_tools_filter": ("deep_research_filter", "o3 Deep Research"),
    "o4-mini-deep-research-2025-06-26_tools": ("deep_research_no_filter", "o4 Mini Deep Research"),
    "o4-mini-deep-research-2025-06-26_tools_filter": ("deep_research_filter", "o4 Mini Deep Research"),
    "sonar-deep-research": ("deep_research_no_filter", "Perplexity Deep Research"),
    "sonar-deep-research_filter": ("deep_research_filter", "Perplexity Deep Research"),
    "sonar-reasoning-pro": ("mcp_no_filter", "Perplexity Sonar Reasoning Pro"),
    "sonar-reasoning-pro_filter": ("mcp_filter", "Perplexity Sonar Reasoning Pro"),
}

PanelKind = Literal["tools_filter", "tools_only"]

PARETO_FACE = (1.0, 0.96, 0.82)
PARETO_ALPHA = 0.65
SPINE_LW = 1.
ANNOT_FONTSIZE = 12.0
AXIS_LABEL_FONTSIZE = 13.5
AXIS_TICK_FONTSIZE = 12.5
LABEL_OFFSET_X = 9
LABEL_OFFSET_Y = 5
POINT_COLOR = "#3d3d3d"
USE_ADJUST_TEXT = False
TIME_X_SYMLOG_LINTHRESH = 10.0
TIME_X_SYMLOG_LINSCALE = 0.15
OPENAI_ICON_PATH = Path(__file__).resolve().parent / "figures" / "icons" / "image.png"
OPENAI_ICON_ZOOM = 0.018
CLAUDE_ICON_PATH = Path(__file__).resolve().parent / "figures" / "icons" / "claude-ai-icon.webp"
CLAUDE_ICON_ZOOM = 0.034
PERPLEXITY_ICON_PATH = Path(__file__).resolve().parent / "figures" / "icons" / "perplexity-ai-icon.webp"
PERPLEXITY_ICON_ZOOM = 0.034
DR_TULU_ICON_PATH = Path(__file__).resolve().parent / "figures" / "icons" / "Ai2_icon_pink_CMYK.jpg"
DR_TULU_ICON_ZOOM = 0.035
GEMINI_ICON_PATH = Path(__file__).resolve().parent / "figures" / "icons" / "google-gemini-icon.webp"
GEMINI_ICON_ZOOM = 0.038


DISPLAY_NAME_BY_BASE: Dict[str, str] = {
    "dr-tulu": "DR Tulu-8B",
    "sonar-reasoning-pro": "Sonar Reasoning Pro",
    "gpt-5.1": "GPT-5.1",
    "claude-sonnet-4-5": "Claude Sonnet 4.5",
    "gemini-3-pro-preview": "Gemini 3 Pro",
    "sonar-deep-research": "Sonar DR",
    "o4-mini-deep-research-2025-06-26": "OpenAI DR (o4-mini)",
    "o3-deep-research-2025-06-26": "OpenAI DR (o3)",
}


def load_time_table(path: Path) -> Dict[Tuple[str, str], float]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[Tuple[str, str], float] = {}
    for section, rows in data.items():
        for row in rows:
            name = row["model"]
            out[(section, name)] = float(row["avg_time_seconds"])
    return out


def load_metrics(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def join_time_and_metrics(
    time_table: Dict[Tuple[str, str], float],
    rows: Sequence[Dict[str, str]],
) -> List[Tuple[str, float, float, float, float]]:
    points: List[Tuple[str, float, float, float, float]] = []
    for r in rows:
        mid = r["model"]
        if mid not in CSV_MODEL_TO_COST_KEY:
            continue
        section, display = CSV_MODEL_TO_COST_KEY[mid]
        key = (section, display)
        if key not in time_table:
            raise KeyError(f"Missing time for {key}")
        t_sec = time_table[key]
        recall = float(r["macro_recall_mean"])
        prec = float(r["macro_precision_mean"])
        f1 = float(r["macro_f1_mean"])
        points.append((mid, t_sec, recall, prec, f1))
    return points


def strip_variant_suffix(model_id: str) -> str:
    if model_id.endswith("_tools_filter"):
        return model_id[: -len("_tools_filter")]
    if model_id.endswith("_tools"):
        return model_id[: -len("_tools")]
    if model_id.endswith("_filter"):
        return model_id[: -len("_filter")]
    return model_id


def display_model_name(model_id: str) -> str:
    base = strip_variant_suffix(model_id)
    if base in DISPLAY_NAME_BY_BASE:
        return DISPLAY_NAME_BY_BASE[base]
    return base.replace("-", " ").title()


def is_openai_model(model_id: str) -> bool:
    base = strip_variant_suffix(model_id)
    return base in {"gpt-5.1", "o3-deep-research-2025-06-26", "o4-mini-deep-research-2025-06-26"}


def is_claude_model(model_id: str) -> bool:
    return strip_variant_suffix(model_id) == "claude-sonnet-4-5"


def is_sonar_model(model_id: str) -> bool:
    return strip_variant_suffix(model_id) in {"sonar-reasoning-pro", "sonar-deep-research"}


def is_dr_tulu_model(model_id: str) -> bool:
    return strip_variant_suffix(model_id) == "dr-tulu"


def is_gemini_model(model_id: str) -> bool:
    return strip_variant_suffix(model_id) == "gemini-3-pro-preview"


def is_tools_filter_variant(model_id: str) -> bool:
    if model_id.endswith("_tools_filter"):
        return True
    return model_id.endswith("_filter") and not model_id.endswith("_tools_filter")


def is_tools_only_no_filter(model_id: str) -> bool:
    if model_id.endswith("_tools_filter"):
        return False
    if model_id.endswith("_tools"):
        return True
    return model_id in ("sonar-reasoning-pro", "sonar-deep-research")


def filter_points(
    points: List[Tuple[str, float, float, float, float]],
    kind: PanelKind,
) -> List[Tuple[str, float, float, float, float]]:
    if kind == "tools_filter":
        return [p for p in points if is_tools_filter_variant(p[0])]
    if kind == "tools_only":
        return [p for p in points if is_tools_only_no_filter(p[0])]
    raise ValueError(kind)


def pareto_frontier_sorted(costs: List[float], scores: List[float]) -> List[int]:
    """
    Return frontier point indices sorted by increasing cost.

    We first collapse equal-cost points to the best score at that cost, then sweep
    left-to-right and keep strict score improvements. This produces the same
    non-dominated frontier while making the plotted staircase stable to tiny
    floating-point differences.
    """
    if len(costs) != len(scores):
        raise ValueError("costs and scores must have the same length")
    if not costs:
        return []

    eps = 1e-12
    ordered = sorted(range(len(costs)), key=lambda i: (costs[i], -scores[i], i))

    # Keep only the best score for each (near-)identical cost.
    dedup_idx: List[int] = []
    for i in ordered:
        if not dedup_idx or abs(costs[i] - costs[dedup_idx[-1]]) > eps:
            dedup_idx.append(i)
        elif scores[i] > scores[dedup_idx[-1]] + eps:
            dedup_idx[-1] = i

    frontier: List[int] = []
    best_score = -float("inf")
    for i in dedup_idx:
        if scores[i] > best_score + eps:
            frontier.append(i)
            best_score = scores[i]
    return frontier


def pareto_bottom_polygon_axes(xmin: float, xmax: float, fc: List[float], fv: List[float]) -> np.ndarray:
    if not fc:
        return np.zeros((0, 2))
    # Start on x-axis and move right to the first frontier point.
    x_pts: List[float] = [xmin, fc[0], fc[0]]
    y_pts: List[float] = [0.0, 0.0, fv[0]]
    for i in range(len(fc) - 1):
        x_pts.extend([fc[i + 1], fc[i + 1]])
        y_pts.extend([fv[i], fv[i + 1]])
    # Continue flat to xmax, then close back on x-axis.
    x_pts.extend([xmax, xmax, xmin])
    y_pts.extend([fv[-1], 0.0, 0.0])
    return np.column_stack([x_pts, y_pts])


def plot_metric(
    points: List[Tuple[str, float, float, float, float]],
    value_index: int,
    ylabel: str,
    out_path: Path,
    xlim: Tuple[float, float],
    ymax: float,
    panel_slug: str,
) -> None:
    labels = [p[0] for p in points]
    times = [p[1] for p in points]
    values = [p[value_index] for p in points]

    frontier_idx = pareto_frontier_sorted(times, values)
    fc_fill = [times[i] for i in frontier_idx]
    fv_fill = [values[i] for i in frontier_idx]

    fig, ax = plt.subplots(figsize=(9.2, 6.0))

    xmin, xmax = xlim
    xmin_eff = max(0.0, xlim[0])

    if len(fc_fill) >= 1:
        # Inverted highlight: fill full plotting box, then carve out the
        # under-frontier staircase back to white.
        full_box = np.array(
            [
                [xmin_eff, 0.0],
                [xmax, 0.0],
                [xmax, ymax],
                [xmin_eff, ymax],
            ]
        )
        highlight_patch = Polygon(
            full_box,
            closed=True,
            facecolor=(*PARETO_FACE, PARETO_ALPHA),
            edgecolor="none",
            linewidth=0.0,
            zorder=1,
        )
        ax.add_patch(highlight_patch)

        white_cutout = pareto_bottom_polygon_axes(xmin_eff, xmax, fc_fill, fv_fill)
        cutout_patch = Polygon(
            white_cutout,
            closed=True,
            facecolor="white",
            edgecolor="none",
            linewidth=0.0,
            zorder=1.05,
        )
        ax.add_patch(cutout_patch)

    openai_idx = [i for i, mid in enumerate(labels) if is_openai_model(mid)]
    claude_idx = [i for i, mid in enumerate(labels) if is_claude_model(mid)]
    sonar_idx = [i for i, mid in enumerate(labels) if is_sonar_model(mid)]
    dr_tulu_idx = [i for i, mid in enumerate(labels) if is_dr_tulu_model(mid)]
    gemini_idx = [i for i, mid in enumerate(labels) if is_gemini_model(mid)]
    icon_idx = sorted(set(openai_idx + claude_idx + sonar_idx + dr_tulu_idx + gemini_idx))
    non_icon_idx = [i for i in range(len(labels)) if i not in icon_idx]

    if non_icon_idx:
        ax.scatter(
            [times[i] for i in non_icon_idx],
            [values[i] for i in non_icon_idx],
            s=52,
            alpha=0.92,
            color=POINT_COLOR,
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )

    if openai_idx:
        try:
            icon = plt.imread(OPENAI_ICON_PATH)
            for i in openai_idx:
                ab = AnnotationBbox(
                    OffsetImage(icon, zoom=OPENAI_ICON_ZOOM),
                    (times[i], values[i]),
                    frameon=False,
                    pad=0.0,
                    box_alignment=(0.5, 0.5),
                    zorder=3.2,
                )
                ax.add_artist(ab)
        except Exception:
            # Fallback to regular dots if icon load fails.
            ax.scatter(
                [times[i] for i in openai_idx],
                [values[i] for i in openai_idx],
                s=52,
                alpha=0.92,
                color=POINT_COLOR,
                edgecolors="white",
                linewidths=0.7,
                zorder=3,
            )

    if claude_idx:
        try:
            claude_icon = plt.imread(CLAUDE_ICON_PATH)
            for i in claude_idx:
                ab = AnnotationBbox(
                    OffsetImage(claude_icon, zoom=CLAUDE_ICON_ZOOM),
                    (times[i], values[i]),
                    frameon=False,
                    pad=0.0,
                    box_alignment=(0.5, 0.5),
                    zorder=3.2,
                )
                ax.add_artist(ab)
        except Exception:
            ax.scatter(
                [times[i] for i in claude_idx],
                [values[i] for i in claude_idx],
                s=52,
                alpha=0.92,
                color=POINT_COLOR,
                edgecolors="white",
                linewidths=0.7,
                zorder=3,
            )

    if sonar_idx:
        try:
            sonar_icon = plt.imread(PERPLEXITY_ICON_PATH)
            for i in sonar_idx:
                ab = AnnotationBbox(
                    OffsetImage(sonar_icon, zoom=PERPLEXITY_ICON_ZOOM),
                    (times[i], values[i]),
                    frameon=False,
                    pad=0.0,
                    box_alignment=(0.5, 0.5),
                    zorder=3.2,
                )
                ax.add_artist(ab)
        except Exception:
            ax.scatter(
                [times[i] for i in sonar_idx],
                [values[i] for i in sonar_idx],
                s=52,
                alpha=0.92,
                color=POINT_COLOR,
                edgecolors="white",
                linewidths=0.7,
                zorder=3,
            )

    if dr_tulu_idx:
        try:
            dr_tulu_icon = plt.imread(DR_TULU_ICON_PATH)
            for i in dr_tulu_idx:
                ab = AnnotationBbox(
                    OffsetImage(dr_tulu_icon, zoom=DR_TULU_ICON_ZOOM),
                    (times[i], values[i]),
                    frameon=False,
                    pad=0.0,
                    box_alignment=(0.5, 0.5),
                    zorder=3.2,
                )
                ax.add_artist(ab)
        except Exception:
            ax.scatter(
                [times[i] for i in dr_tulu_idx],
                [values[i] for i in dr_tulu_idx],
                s=52,
                alpha=0.92,
                color=POINT_COLOR,
                edgecolors="white",
                linewidths=0.7,
                zorder=3,
            )

    if gemini_idx:
        try:
            gemini_icon = plt.imread(GEMINI_ICON_PATH)
            for i in gemini_idx:
                ab = AnnotationBbox(
                    OffsetImage(gemini_icon, zoom=GEMINI_ICON_ZOOM),
                    (times[i], values[i]),
                    frameon=False,
                    pad=0.0,
                    box_alignment=(0.5, 0.5),
                    zorder=3.2,
                )
                ax.add_artist(ab)
        except Exception:
            ax.scatter(
                [times[i] for i in gemini_idx],
                [values[i] for i in gemini_idx],
                s=52,
                alpha=0.92,
                color=POINT_COLOR,
                edgecolors="white",
                linewidths=0.7,
                zorder=3,
            )

    texts: List[object] = []
    for x, y, lab in zip(times, values, labels):
        txt = display_model_name(lab)
        ox, oy = LABEL_OFFSET_X, LABEL_OFFSET_Y
        if txt in {"Sonar Reasoning Pro", "Sonar DR"}:
            ox += 2
            oy -= 4
        if panel_slug == "tools_only" and out_path.name == "time_vs_macro_precision.png":
            if txt in {"Gemini 3 Pro"}:
                oy = -13
                ox = 5
            if txt in {"OpenAI DR (o4-mini)"}:
                ox, oy = -40, -20
            if txt in {"DR Tulu-8B"}:
                oy -= 17

        if panel_slug == "tools_only" and out_path.name == "time_vs_macro_recall.png":
            if txt in {"OpenAI DR (o4-mini)"}:
                ox, oy = -25, -22
            if txt in {"DR Tulu-8B"}:
                oy = -12
            if txt in {"Gemini 3 Pro"}:
                ox, oy = 8, -10
            if txt in {"Sonar Reasoning Pro", "Sonar DR"}:
                oy = -8.5
        if panel_slug == "tools_only" and out_path.name == "time_vs_macro_f1.png":
            if txt == "OpenAI DR (o4-mini)":
                ox, oy = -40, 11
        
        if panel_slug == "tools_plus_filter" and out_path.name == "time_vs_macro_f1.png":
            if txt == "GPT-5.1":
                ox, oy = 7, -13
            if txt == "OpenAI DR (o4-mini)":
                ox, oy = 9, -5
            if txt == "Sonar Reasoning Pro":
                ox, oy = -15, 12
            if txt == "Gemini 3 Pro":
                ox, oy = 9, -8
            if txt == "Claude Sonnet 4.5":
                oy += 5
                ox -= 5

        if panel_slug == "tools_plus_filter" and out_path.name == "time_vs_macro_precision.png":

            if txt == "Gemini 3 Pro":
                ox, oy = 10, -9
            if txt == "Claude Sonnet 4.5":
                oy = -16
            if txt == "Sonar DR":
                ox, oy = 10, 10
            if txt == "DR Tulu-8B":
                oy -= 18
            if txt == "OpenAI DR (o4-mini)":
                ox -= 20
                oy += 8
        if panel_slug == "tools_plus_filter" and out_path.name == "time_vs_macro_recall.png":
            if txt == "Sonar Reasoning Pro":
                oy = 7
            if txt == "OpenAI DR (o4-mini)":
                oy -= 24
                ox -= 23
            if txt == "Claude Sonnet 4.5":
                oy = -10
                ox += 1
            if txt == "Gemini 3 Pro":
                oy -= 15
                ox += 1
            if txt == "Sonar Reasoning Pro":
                oy += 7
                ox -= 35

        t = ax.annotate(
            txt,
            (x, y),
            textcoords="offset points",
            xytext=(ox, oy),
            fontsize=ANNOT_FONTSIZE,
            color="#2c2c2c",
            fontfamily="sans-serif",
            zorder=4,
        )
        texts.append(t)

    if USE_ADJUST_TEXT and adjust_text is not None and texts:
        adjust_text(
            texts,
            ax=ax,
            expand_points=(1.45, 1.45),
            expand_text=(1.35, 1.35),
            force_points=(0.4, 0.6),
            force_text=(0.5, 0.7),
            lim=520,
        )

    ax.set_xscale(
        "symlog",
        linthresh=TIME_X_SYMLOG_LINTHRESH,
        linscale=TIME_X_SYMLOG_LINSCALE,
        base=10.0,
    )
    ax.set_xlim(xmin_eff, xmax)
    ax.xaxis.set_major_locator(FixedLocator([0.0, 10.0, 100.0, 1000.0]))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _pos: str(int(x)) if x in {0.0, 10.0, 100.0, 1000.0} else "")
    )
    ax.xaxis.set_minor_locator(
        FixedLocator([20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0])
    )
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(
        axis="x",
        which="major",
        length=6.0,
        width=1.0,
        colors="black",
        pad=0,
        labelsize=AXIS_TICK_FONTSIZE,
        direction="inout",
        bottom=True,
        top=False,
        labelbottom=True,
    )
    ax.tick_params(
        axis="x",
        which="minor",
        length=3.5,
        width=0.8,
        colors="black",
        labelsize=AXIS_TICK_FONTSIZE,
        direction="inout",
        bottom=True,
        top=False,
    )
    ax.tick_params(
        axis="y",
        which="major",
        length=4.5,
        width=0.9,
        colors="black",
        pad=0,
        labelsize=AXIS_TICK_FONTSIZE,
    )

    ax.set_ylim(0.0, ymax)
    ax.yaxis.set_major_locator(FixedLocator([0.1, 0.3, 0.5, 0.7]))

    ax.set_xlabel(
        "Time Per Query (seconds)",
        fontweight="bold",
        fontsize=AXIS_LABEL_FONTSIZE,
        color="#1a1a1a",
        labelpad=5,
    )
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=AXIS_LABEL_FONTSIZE, color="#1a1a1a", labelpad=7)

    # Explicitly draw x/y axis lines.
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_linewidth(SPINE_LW)
    ax.spines["bottom"].set_linewidth(SPINE_LW)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Plot average time per query vs factual metrics with Pareto frontiers.")
    parser.add_argument("--time-json", type=Path, default=here / "data" / "model_time.json")
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=here.parents[1] / "data" / "labeled_facts" / "metrics_macro_variance_by_model.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=here / "figures")
    parser.add_argument("--xmin", type=float, default=0.0)
    parser.add_argument("--xmax", type=float, default=1000.0)
    parser.add_argument("--ymax", type=float, default=0.7)
    args = parser.parse_args()

    sns.set_theme(
        style="white",
        rc={
            "axes.spines.right": False,
            "axes.spines.top": False,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica Neue",
                "Helvetica",
                "Arial",
                "DejaVu Sans",
                "Liberation Sans",
                "sans-serif",
            ],
        },
    )

    if not args.metrics_csv.exists():
        print(f"[Auto] {args.metrics_csv.name} not found — computing metrics first...")
        compute_metrics(DEFAULT_BASE_DIR)

    time_table = load_time_table(args.time_json)
    rows = load_metrics(args.metrics_csv)
    all_points = join_time_and_metrics(time_table, rows)
    if not all_points:
        raise SystemExit("No points after joining metrics with model_time.json; check mappings.")

    panels: List[Tuple[PanelKind, str]] = [
        ("tools_filter", "tools_plus_filter"),
        ("tools_only", "tools_only"),
    ]
    for kind, slug in panels:
        pts = filter_points(all_points, kind)
        if len(pts) != 8:
            raise SystemExit(
                f"Panel {kind!r}: expected 8 points, got {len(pts)}. IDs: {[p[0] for p in pts]}"
            )

    metrics_spec: List[Tuple[int, str, str]] = [
        (4, "Factual F1-Score", "f1"),
        (3, "Factual Precision", "precision"),
        (2, "Factual Recall", "recall"),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    xlim = (args.xmin, args.xmax)
    ymax = float(args.ymax)

    for kind, slug in panels:
        pts = filter_points(all_points, kind)
        subdir = args.output_dir / slug
        subdir.mkdir(parents=True, exist_ok=True)
        for value_index, ylabel, fname in metrics_spec:
            plot_metric(
                pts,
                value_index=value_index,
                ylabel=ylabel,
                out_path=subdir / f"time_vs_macro_{fname}.png",
                xlim=xlim,
                ymax=ymax,
                panel_slug=slug,
            )

    print(f"Wrote figures under {args.output_dir.resolve()} (subfolders: tools_plus_filter, tools_only)")


if __name__ == "__main__":
    main()
