#!/usr/bin/env python3
"""Export the SciConBench-Track database into static JSON for the public site.

The dashboard at https://sciconbench.cs.princeton.edu is a purely static bundle,
so every number it shows has to be baked into ``public/data/dashboard.json``
ahead of time. Run this after a monthly pipeline run, then ``./publish.sh``.

    python site/export_data.py              # real data only
    python site/export_data.py --demo       # fill ungraded rows with fake
                                            # scores to preview the layout

Reads (all read-only):
    data_track/sciconbench_track.db        model responses + judge scores
    data_track/sciconbench_track.parquet   size of the published HuggingFace release
    data_track/logs/workflow-*.log         stage-by-stage status of latest run
    scicon-track/config/query_batch_config.yaml   model roster + re-eval policy
    site/site.config.json                  editable metadata (links, news, team)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SITE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SITE_DIR.parent
DATA_DIR = Path(os.environ.get("SCICON_DATA_DIR", REPO_ROOT / "data_track"))
DB_PATH = DATA_DIR / "sciconbench_track.db"
PARQUET_PATH = DATA_DIR / "sciconbench_track.parquet"
LOG_DIR = DATA_DIR / "logs"
RESULTS_DIR = DATA_DIR / "results"
CONFIG_PATH = REPO_ROOT / "scicon-track" / "config" / "query_batch_config.yaml"
SITE_CONFIG_PATH = SITE_DIR / "site.config.json"
OUT_PATH = SITE_DIR / "public" / "data" / "dashboard.json"

EVAL_CONFIG_LABEL = "tools_filter"

# The workflow's stage labels, in run order. Must match the strings passed to
# PipelineReport.begin() in scicon-track/run_workflow.py.
PIPELINE_STAGES = [
    ("1. Initialize database", "Verify the tracking schema."),
    ("2. Load core set", "Load the frozen core panel of Cochrane reviews."),
    ("3. Discover + prune", "Find newly published reviews; drop superseded versions."),
    ("4. Download PDFs", "Fetch full text through the Wiley TDM API."),
    ("5. Extract reference text", "Parse conclusions and metadata out of each PDF."),
    ("6. Generate clinical questions", "Write the question an agent must answer."),
    ("7. Cochrane atomic facts", "Decompose expert conclusions into atomic facts."),
    ("8. Upload to HuggingFace", "Merge the new month into the public dataset."),
    ("9. Query models", "Run every model through the clean-room harness."),
    ("10. Response atomic facts", "Decompose model conclusions into atomic facts."),
    ("11. Precision analysis", "Judge each model fact against the review."),
    ("12. Recall analysis", "Judge how many expert facts the model recovered."),
]

# Brand colours. Most of these keys are *labs* (the company that actually
# built the model), keyed the same way as resolve_icon()'s logomark keys —
# a couple (azure, openrouter) are API hosts, kept only as a last-resort
# fallback for a model resolve_family() can't otherwise place.
PROVIDER_META = {
    "openai": {"label": "OpenAI", "color": "#10a37f"},
    "claude": {"label": "Anthropic", "color": "#d97757"},
    "anthropic": {"label": "Anthropic", "color": "#d97757"},
    "gemini": {"label": "Google DeepMind", "color": "#4285f4"},
    "deepseek": {"label": "DeepSeek", "color": "#4d6bfe"},
    "moonshot": {"label": "Moonshot AI", "color": "#6f69f7"},
    "kimi": {"label": "Moonshot AI", "color": "#6f69f7"},
    "glm": {"label": "Z.ai (GLM)", "color": "#0ea5a3"},
    "qwen": {"label": "Alibaba (Qwen)", "color": "#eab308"},
    "minimax": {"label": "MiniMax", "color": "#dc2626"},
    "meta": {"label": "Meta", "color": "#0668e1"},
    "mistral": {"label": "Mistral AI", "color": "#f97316"},
    "xai": {"label": "xAI", "color": "#111827"},
    "perplexity": {"label": "Perplexity", "color": "#20b8cd"},
    "ai2": {"label": "Ai2", "color": "#e8478b"},
    "azure": {"label": "Azure AI Foundry", "color": "#8b5cf6"},
    "openrouter": {"label": "OpenRouter", "color": "#f59e0b"},
}

DISPLAY_NAMES = {
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "claude-opus-5": "Claude Opus 5",
    "gemini-3.7-flash": "Gemini 3.7 Flash",
    "DeepSeek-V4-Pro": "DeepSeek-V4-Pro",
    "DeepSeek-V4-Flash-0731": "DeepSeek-V4-Flash",
    "moonshotai/kimi-k3": "Kimi K3",
    "z-ai/glm-5.3": "GLM-5.3",
    "z-ai/glm-5.2": "GLM-5.2",
    "qwen/qwen3.8-max": "Qwen3.8-Max",
    "qwen/qwen3.8-27b": "Qwen3.8 27B",
    "qwen/qwen3.7-max": "Qwen3.7-Max",
    "minimax/minimax-m3": "MiniMax M3",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def display_name(model: str) -> str:
    return DISPLAY_NAMES.get(model, model.split("/")[-1])


def provider_meta(provider: str) -> dict[str, str]:
    return PROVIDER_META.get(
        provider.lower(), {"label": provider.title(), "color": "#94a3b8"}
    )


# Model-name substrings mapped to a logomark key, so a model is badged (and,
# via resolve_family() below, labelled and coloured) by the lab that
# actually built it rather than whichever API host happens to serve it —
# e.g. a DeepSeek model called via Azure still reads as DeepSeek, and a
# Kimi/GLM/Qwen model called through OpenRouter still reads as its own lab.
_ICON_HINTS: list[tuple[str, str]] = [
    ("deepseek", "deepseek"),
    ("claude", "anthropic"),
    ("opus", "anthropic"),
    ("sonnet", "anthropic"),
    ("haiku", "anthropic"),
    ("gemini", "gemini"),
    ("grok", "xai"),
    ("llama", "meta"),
    ("mistral", "mistral"),
    ("qwen", "qwen"),
    ("kimi", "kimi"),
    ("glm", "glm"),
    ("minimax", "minimax"),
    ("tulu", "ai2"),
    ("gpt", "openai"),
]


def resolve_icon(model: str, provider: str) -> str:
    m = model.lower()
    for hint, icon in _ICON_HINTS:
        if hint in m:
            return icon
    if re.match(r"^o[0-9](-|$|-mini)", m):
        return "openai"
    return provider.lower()


def resolve_family(model: str, provider: str) -> dict[str, str]:
    """Which lab actually built a model — inferred from its *name*, not
    whichever API host happens to be serving it this month (an aggregator
    like OpenRouter or Azure AI Foundry can front models from any lab, e.g.
    Kimi/GLM/Qwen via OpenRouter, or DeepSeek via Azure).

    This reuses resolve_icon()'s name-hint list, so it needs zero code
    changes when a new model shows up: "gpt-6" already contains "gpt" and
    is classified as OpenAI the moment its first result lands, same as
    "gpt-5.1" and "gpt-5.6-sol" are today.
    """
    key = resolve_icon(model, provider)
    meta = provider_meta(key)
    return {"key": key, "label": meta["label"], "color": meta["color"]}


# --------------------------------------------------------------------------- #
# paper-release baselines
# --------------------------------------------------------------------------- #
# Numbers from the SciConBench paper's own evaluation run, hand-transcribed
# here rather than pulled from the tracking DB: unlike the monthly pipeline,
# this was a one-off snapshot (some models graded in Jan 2026, others in Jul
# 2026) that isn't itself continuously re-run. Some model names overlap with
# the live roster (e.g. DeepSeek-V4-Pro) — those are deliberately kept as a
# separate point so the site can show how much a score moved since the paper.
PAPER_BASELINES: list[dict[str, Any]] = [
    {"group": "Models", "model": "gpt-5.1", "display_name": "GPT-5.1", "provider": "openai",
     "precision": 0.294, "precision_std": 0.017, "recall": 0.408, "recall_std": 0.065, "f1": 0.300, "f1_std": 0.024},
    {"group": "Models", "model": "claude-sonnet-4.5", "display_name": "Claude Sonnet 4.5", "provider": "anthropic",
     "precision": 0.350, "precision_std": 0.020, "recall": 0.329, "recall_std": 0.057, "f1": 0.297, "f1_std": 0.030},
    {"group": "Models", "model": "gemini-3-pro", "display_name": "Gemini 3 Pro", "provider": "gemini",
     "precision": 0.294, "precision_std": 0.034, "recall": 0.206, "recall_std": 0.042, "f1": 0.194, "f1_std": 0.029},
    {"group": "Models", "model": "sonar-reasoning-pro", "display_name": "Sonar Reasoning Pro", "provider": "perplexity",
     "precision": 0.384, "precision_std": 0.032, "recall": 0.205, "recall_std": 0.044, "f1": 0.220, "f1_std": 0.035},
    {"group": "Models", "model": "deepseek-v4-pro-paper", "display_name": "DeepSeek-V4-Pro", "provider": "deepseek",
     "precision": 0.3995, "precision_std": 0.0236, "recall": 0.3437, "recall_std": 0.0553, "f1": 0.3258, "f1_std": 0.0323},
    {"group": "Models", "model": "kimi-k3-paper", "display_name": "Kimi K3", "provider": "moonshot",
     "precision": 0.3212, "precision_std": 0.0154, "recall": 0.4014, "recall_std": 0.0559, "f1": 0.3079, "f1_std": 0.0230},
    {"group": "Models", "model": "qwen3.5-9b-paper", "display_name": "Qwen3.5 9B", "provider": "openrouter",
     "precision": 0.3932, "precision_std": 0.0280, "recall": 0.2469, "recall_std": 0.0462, "f1": 0.2482, "f1_std": 0.0306},
    {"group": "Models", "model": "glm-5.2-paper", "display_name": "GLM-5.2", "provider": "openrouter",
     "precision": 0.3179, "precision_std": 0.0178, "recall": 0.4025, "recall_std": 0.0663, "f1": 0.3198, "f1_std": 0.0295},
    {"group": "DR", "model": "dr-tulu", "display_name": "DR Tulu", "provider": "ai2",
     "precision": 0.259, "precision_std": 0.038, "recall": 0.168, "recall_std": 0.034, "f1": 0.145, "f1_std": 0.023},
    {"group": "DR", "model": "sonar-deep-research", "display_name": "Sonar Deep Research", "provider": "perplexity",
     "precision": 0.357, "precision_std": 0.036, "recall": 0.243, "recall_std": 0.047, "f1": 0.237, "f1_std": 0.034},
    {"group": "DR", "model": "o4-mini-deep-research", "display_name": "o4-mini Deep Research", "provider": "openai",
     "precision": 0.467, "precision_std": 0.028, "recall": 0.298, "recall_std": 0.051, "f1": 0.315, "f1_std": 0.039},
    {"group": "DR", "model": "o3-deep-research", "display_name": "o3 Deep Research", "provider": "openai",
     "precision": 0.441, "precision_std": 0.033, "recall": 0.342, "recall_std": 0.054, "f1": 0.337, "f1_std": 0.035},
]


def build_paper_baselines() -> list[dict[str, Any]]:
    out = []
    for b in PAPER_BASELINES:
        meta = resolve_family(b["model"], b["provider"])
        out.append(
            {
                "model": b["model"],
                "display_name": b["display_name"],
                "group": b["group"],
                "provider": b["provider"],
                "family": meta["key"],
                "provider_label": meta["label"],
                "color": meta["color"],
                "icon": meta["key"],
                "precision": b["precision"],
                "precision_std": b.get("precision_std"),
                "recall": b["recall"],
                "recall_std": b.get("recall_std"),
                "f1": b["f1"],
                "f1_std": b.get("f1_std"),
            }
        )
    return out


def count_atomic_facts(pairs: Any) -> int:
    """Count leaf facts in the ``[[sentence, [fact, ...]], ...]`` payload."""
    if isinstance(pairs, str):
        try:
            pairs = json.loads(pairs)
        except json.JSONDecodeError:
            return 0
    total = 0
    if isinstance(pairs, dict):
        pairs = pairs.get("atomic_facts_pairs") or pairs.get("pairs") or []
    for item in pairs or []:
        if isinstance(item, dict):
            facts = item.get("facts") or item.get("atomic_facts") or []
            total += len(facts) if isinstance(facts, list) else 0
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            total += len(item[1]) if isinstance(item[1], list) else 0
    return total


def load_json(raw: Any) -> Any:
    if isinstance(raw, (bytes, str)):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
    return raw


def month_label(month: str) -> str:
    try:
        return datetime.strptime(month, "%Y-%m").strftime("%b %Y")
    except ValueError:
        return month


# --------------------------------------------------------------------------- #
# model roster
# --------------------------------------------------------------------------- #

def parse_model_roster() -> dict[str, str]:
    """Map model name -> re-eval policy ('always' | 'once') from the run config.

    Hand-parsed rather than imported so this script stays dependency-free and
    runnable outside the pipeline's conda environment.
    """
    if not CONFIG_PATH.exists():
        return {}
    text = CONFIG_PATH.read_text()

    def block(key: str) -> list[str]:
        match = re.search(rf"^{key}:\s*$(.*?)(?=^\S|\Z)", text, re.M | re.S)
        if not match:
            return []
        # Trailing "# control" style annotations are common in this config and
        # would otherwise become part of the model name.
        lines = (raw.split("#", 1)[0].strip() for raw in match.group(1).splitlines())
        return [line for line in lines if line]

    always = {line.lstrip("- ").strip() for line in block("reevaluate_always")}
    # Legacy YAMLs listed open-weight models under evaluate_once; invert that.
    legacy_once = {line.lstrip("- ").strip() for line in block("evaluate_once")}
    roster: dict[str, str] = {}
    for line in block("default_models"):
        if ":" in line and not line.startswith("-"):
            _, _, value = line.partition(":")
            value = value.strip()
            if value:
                if always:
                    roster[value] = "always" if value in always else "once"
                elif legacy_once:
                    roster[value] = "once" if value in legacy_once else "always"
                else:
                    roster[value] = "once"
        elif line.startswith("-"):
            name = line.lstrip("- ").strip()
            if always:
                roster[name] = "always" if name in always else "once"
            elif legacy_once:
                roster[name] = "once" if name in legacy_once else "always"
            else:
                roster[name] = "once"
    return roster


# --------------------------------------------------------------------------- #
# workflow log
# --------------------------------------------------------------------------- #

BEGIN_RE = re.compile(r"^\[(.*?)\]\s+>>\s+BEGIN\s+(.*)$")
DONE_RE = re.compile(r"^\[(.*?)\]\s+<<\s+(OK|FAIL)\s+(.*?)(?:\s+—\s+(.*))?$")


def parse_latest_run() -> dict[str, Any]:
    """Reconstruct stage status for the most recent workflow log."""
    logs = sorted(LOG_DIR.glob("workflow-*.log")) if LOG_DIR.exists() else []
    if not logs:
        return {"available": False, "stages": []}

    log_path = logs[-1]
    lines = log_path.read_text(errors="replace").splitlines()

    header: dict[str, str] = {}
    for line in lines[:12]:
        match = re.match(r"^\s{2}(\w+):\s*(.*)$", line)
        if match:
            header[match.group(1)] = match.group(2).strip()

    state: dict[str, dict[str, Any]] = {}
    for line in lines:
        if begin := BEGIN_RE.match(line):
            state.setdefault(begin.group(2).strip(), {}).update(
                status="running", started_at=begin.group(1)
            )
        elif done := DONE_RE.match(line):
            state.setdefault(done.group(3).strip(), {}).update(
                status="ok" if done.group(2) == "OK" else "failed",
                finished_at=done.group(1),
                detail=(done.group(4) or "").strip() or None,
            )

    stages = []
    for name, blurb in PIPELINE_STAGES:
        entry = state.get(name, {})
        stages.append(
            {
                "name": name,
                "short": name.split(". ", 1)[-1],
                "description": blurb,
                "status": entry.get("status", "pending"),
                "started_at": entry.get("started_at"),
                "finished_at": entry.get("finished_at"),
                "detail": entry.get("detail"),
            }
        )

    started = re.search(r"started (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", lines[1] if len(lines) > 1 else "")
    return {
        "available": True,
        "log_file": log_path.name,
        "mode": header.get("mode"),
        "target_month": header.get("target_month"),
        "started_at": started.group(1) if started else header.get("started_at"),
        "log_mtime": datetime.fromtimestamp(
            log_path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        "stages": stages,
    }


# --------------------------------------------------------------------------- #
# database
# --------------------------------------------------------------------------- #

def export_benchmark() -> dict[str, Any]:
    """Size of the published HuggingFace release.

    This is the whole benchmark (thousands of reviews), which is a different
    quantity from the live evaluation panel that the monthly runs score. The
    site shows both so the two are never confused.
    """
    empty = {"available": False, "reviews": None, "atomic_facts": None}
    if not PARQUET_PATH.exists():
        return empty
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return empty

    try:
        table = pq.read_table(PARQUET_PATH, columns=["total_atomic_facts"])
    except Exception:
        return empty

    counts = [n for n in table.column("total_atomic_facts").to_pylist() if n]
    reviews = table.num_rows
    return {
        "available": True,
        "reviews": reviews,
        "atomic_facts": sum(counts),
        "facts_per_review": round(sum(counts) / len(counts), 1) if counts else None,
        "updated": datetime.fromtimestamp(
            PARQUET_PATH.stat().st_mtime, tz=timezone.utc
        ).strftime("%Y-%m-%d"),
    }


def connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise SystemExit(f"tracking database not found: {DB_PATH}")
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def export_dataset(conn: sqlite3.Connection) -> dict[str, Any]:
    panels: dict[str, int] = defaultdict(int)
    cohorts: dict[str, int] = defaultdict(int)
    statuses: dict[str, int] = defaultdict(int)
    for row in conn.execute("SELECT panel_type, cohort_month, processing_status FROM doi_info"):
        panel = (row["panel_type"] or "").lower()
        panels[panel] += 1
        statuses[(row["processing_status"] or "").lower()] += 1
        if panel == "rolling" and row["cohort_month"]:
            cohorts[row["cohort_month"]] += 1

    review_types: dict[str, int] = defaultdict(int)
    for row in conn.execute(
        "SELECT review_type, COUNT(*) n FROM review_metadata GROUP BY review_type"
    ):
        label = (row["review_type"] or "Unspecified").replace("Review - ", "")
        review_types[label] += row["n"]

    total_facts = 0
    reviews_with_facts = 0
    for row in conn.execute(
        "SELECT atomic_facts_pairs FROM atomic_facts WHERE LOWER(source) = 'cochrane'"
    ):
        n = count_atomic_facts(row["atomic_facts_pairs"])
        total_facts += n
        reviews_with_facts += 1 if n else 0

    dates = conn.execute(
        "SELECT MIN(publication_date) lo, MAX(publication_date) hi FROM review_metadata"
    ).fetchone()

    core = panels.get("core", 0)
    # Cumulative dataset size after each monthly cohort lands.
    growth, running = [], core
    for month in sorted(cohorts):
        running += cohorts[month]
        growth.append(
            {
                "month": month,
                "label": month_label(month),
                "added": cohorts[month],
                "total": running,
                "core": core,
                "rolling": running - core,
            }
        )

    return {
        "total_reviews": sum(panels.values()),
        "core_reviews": core,
        "rolling_reviews": panels.get("rolling", 0),
        "questions": conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0],
        "atomic_facts": total_facts,
        "reviews_with_facts": reviews_with_facts,
        "facts_per_review": round(total_facts / reviews_with_facts, 1) if reviews_with_facts else None,
        "publication_range": {"from": dates["lo"], "to": dates["hi"]},
        "cohorts": [
            {"month": m, "label": month_label(m), "count": cohorts[m]} for m in sorted(cohorts)
        ],
        "growth": growth,
        "review_types": [
            {"label": k, "count": v}
            for k, v in sorted(review_types.items(), key=lambda kv: -kv[1])
        ],
        "processing_status": [
            {"label": k, "count": v}
            for k, v in sorted(statuses.items(), key=lambda kv: -kv[1])
        ],
    }


def panel_key_of(panel_type: str | None, cohort_month: str | None) -> str | None:
    """Slice key for the core-vs-rolling breakdown: 'core', 'rolling' (every
    non-core review, regardless of which month's cohort it joined in — the
    site only ever needs to filter core vs. rolling, not month by month),
    or None if the DOI isn't classified."""
    panel = (panel_type or "").upper()
    if panel == "CORE":
        return "core"
    if panel == "ROLLING" and cohort_month:
        return "rolling"
    return None


def panel_label_of(panel_key: str) -> str:
    return "Core set" if panel_key == "core" else "Rolling set"


def new_score_bucket() -> dict[str, Any]:
    return {"precision": [], "recall": [], "f1": [], "responses": 0, "graded": 0, "dois": set()}


def export_evaluations(conn: sqlite3.Connection, demo: bool) -> dict[str, Any]:
    """Aggregate per-(model, run_month) scores, cost, and tool-use statistics."""
    rows = conn.execute(
        """
        SELECT r.id, r.doi, r.model, r.provider, r.run_month, r.token_usage,
               p.factual_precision, p.total_llm_facts, p.supported_facts AS p_supported,
               p.contradicted_facts, p.not_supported_facts,
               c.factual_recall, c.total_article_facts, c.supported_facts AS r_supported,
               d.panel_type, d.cohort_month
          FROM model_responses r
          LEFT JOIN factual_precision_results p ON p.model_response_id = r.id
          LEFT JOIN factual_recall_results   c ON c.model_response_id = r.id
          LEFT JOIN doi_info d ON d.doi = r.doi
         WHERE r.config_label = ?
        """,
        (EVAL_CONFIG_LABEL,),
    ).fetchall()

    rng = random.Random(20260826)
    demo_baseline: dict[str, tuple[float, float]] = {}

    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    # Same aggregation, but sliced by core-set vs individual rolling cohort —
    # powers the leaderboard's "core set vs rolling panel" filter.
    panel_buckets: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    panel_keys_seen: set[str] = set()
    for row in rows:
        key = (row["model"], row["provider"], row["run_month"] or "unknown")
        bucket = buckets.setdefault(
            key,
            {
                "precision": [], "recall": [], "f1": [],
                "responses": 0, "graded": 0, "dois": set(),
                "input_tokens": [], "output_tokens": [], "tool_calls": [],
                "iterations": [], "tool_usage": defaultdict(int),
                "contradicted": 0, "model_facts": 0, "article_facts": 0,
                "recovered_facts": 0,
            },
        )
        bucket["responses"] += 1
        bucket["dois"].add(row["doi"])

        panel_key = panel_key_of(row["panel_type"], row["cohort_month"])
        panel_bucket = None
        if panel_key is not None:
            panel_keys_seen.add(panel_key)
            panel_bucket = panel_buckets.setdefault((*key, panel_key), new_score_bucket())
            panel_bucket["responses"] += 1
            panel_bucket["dois"].add(row["doi"])

        usage = load_json(row["token_usage"]) or {}
        if isinstance(usage, dict):
            for field, target in (
                ("input_tokens", "input_tokens"),
                ("output_tokens", "output_tokens"),
                ("tool_call_count", "tool_calls"),
                ("iterations", "iterations"),
            ):
                value = usage.get(field)
                if isinstance(value, (int, float)):
                    bucket[target].append(float(value))
            for tool, n in (usage.get("tool_usage") or {}).items():
                if isinstance(n, (int, float)):
                    bucket["tool_usage"][tool] += int(n)

        precision, recall = row["factual_precision"], row["factual_recall"]
        if demo and precision is None and recall is None:
            base_p, base_r = demo_baseline.setdefault(
                row["model"], (rng.uniform(0.45, 0.78), rng.uniform(0.30, 0.62))
            )
            precision = min(1.0, max(0.0, rng.gauss(base_p, 0.13)))
            recall = min(1.0, max(0.0, rng.gauss(base_r, 0.13)))

        if precision is not None:
            bucket["precision"].append(precision)
            bucket["model_facts"] += row["total_llm_facts"] or 0
            bucket["contradicted"] += row["contradicted_facts"] or 0
            if panel_bucket is not None:
                panel_bucket["precision"].append(precision)
        if recall is not None:
            bucket["recall"].append(recall)
            bucket["article_facts"] += row["total_article_facts"] or 0
            bucket["recovered_facts"] += row["r_supported"] or 0
            if panel_bucket is not None:
                panel_bucket["recall"].append(recall)
        if precision is not None and recall is not None:
            bucket["graded"] += 1
            denom = precision + recall
            f1 = 2 * precision * recall / denom if denom else 0.0
            bucket["f1"].append(f1)
            if panel_bucket is not None:
                panel_bucket["graded"] += 1
                panel_bucket["f1"].append(f1)

    entries = []
    for (model, provider, run_month), b in buckets.items():
        meta = resolve_family(model, provider)
        tools = sorted(b["tool_usage"].items(), key=lambda kv: -kv[1])
        entries.append(
            {
                "model": model,
                "display_name": display_name(model),
                "provider": provider,
                "family": meta["key"],
                "provider_label": meta["label"],
                "color": meta["color"],
                "icon": meta["key"],
                "run_month": run_month,
                "run_month_label": month_label(run_month),
                "responses": b["responses"],
                "reviews": len(b["dois"]),
                "graded": b["graded"],
                "precision": mean(b["precision"]),
                "recall": mean(b["recall"]),
                "f1": mean(b["f1"]),
                "contradiction_rate": (
                    b["contradicted"] / b["model_facts"] if b["model_facts"] else None
                ),
                "avg_input_tokens": mean(b["input_tokens"]),
                "avg_output_tokens": mean(b["output_tokens"]),
                "avg_tool_calls": mean(b["tool_calls"]),
                "avg_iterations": mean(b["iterations"]),
                "tool_usage": [{"tool": t, "count": n} for t, n in tools],
                "panels": {
                    panel_key: {
                        "precision": mean(pb["precision"]),
                        "recall": mean(pb["recall"]),
                        "f1": mean(pb["f1"]),
                        "reviews": len(pb["dois"]),
                        "responses": pb["responses"],
                        "graded": pb["graded"],
                    }
                    for panel_key in panel_keys_seen
                    if (pb := panel_buckets.get((model, provider, run_month, panel_key)))
                },
            }
        )

    entries.sort(key=lambda e: (e["run_month"], e["model"]))
    panel_views = [{"key": "core", "label": panel_label_of("core")}] + [
        {"key": k, "label": panel_label_of(k)}
        for k in sorted(k for k in panel_keys_seen if k != "core")
    ]
    return {
        "entries": entries,
        "run_months": sorted({e["run_month"] for e in entries}),
        "panel_views": panel_views if panel_keys_seen else [],
    }


def build_leaderboard(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per model, using its most recent run month that has data.

    Open-weight and proprietary models default to an ``once`` policy, so a
    model's newest row may be older than the current run month; showing each
    model's latest available evaluation keeps them comparable instead of
    silently dropping them.
    """
    latest: dict[str, dict[str, Any]] = {}
    for entry in entries:
        current = latest.get(entry["model"])
        if current is None or entry["run_month"] > current["run_month"]:
            latest[entry["model"]] = entry

    board = list(latest.values())
    board.sort(
        key=lambda e: (e["f1"] is None, -(e["f1"] or 0), -(e["precision"] or 0))
    )
    for rank, entry in enumerate(board, start=1):
        entry["rank"] = rank
    return board


def build_series(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-model time series of F1 across run months, for the trend chart."""
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_model[entry["model"]].append(entry)

    series = []
    for model, rows in by_model.items():
        rows.sort(key=lambda e: e["run_month"])
        points = [
            {
                "month": r["run_month"],
                "label": r["run_month_label"],
                "f1": r["f1"],
                "precision": r["precision"],
                "recall": r["recall"],
                "reviews": r["reviews"],
                # Per-panel (core vs. each rolling cohort) precision/recall/f1
                # and sample counts, so the trend chart can both slice scores
                # by panel and show "N samples (core vs. rolling)" on hover.
                "panels": {
                    k: {
                        "precision": v["precision"],
                        "recall": v["recall"],
                        "f1": v["f1"],
                        "reviews": v["reviews"],
                    }
                    for k, v in (r.get("panels") or {}).items()
                },
            }
            for r in rows
        ]
        if any(p["f1"] is not None for p in points):
            series.append(
                {
                    "model": model,
                    "display_name": rows[0]["display_name"],
                    "provider": rows[0]["provider"],
                    "family": rows[0]["family"],
                    "provider_label": rows[0]["provider_label"],
                    "color": rows[0]["color"],
                    "icon": rows[0]["icon"],
                    "points": points,
                }
            )
    series.sort(key=lambda s: s["display_name"])
    return series


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="synthesize scores for ungraded responses to preview the layout; "
             "the site renders a prominent warning banner when this is set",
    )
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    site_config = json.loads(SITE_CONFIG_PATH.read_text()) if SITE_CONFIG_PATH.exists() else {}
    roster = parse_model_roster()
    benchmark = export_benchmark()

    with connect() as conn:
        dataset = export_dataset(conn)
        evaluations = export_evaluations(conn, demo=args.demo)
        registry = [
            {
                "model": row["name"],
                "display_name": display_name(row["name"]),
                "provider": row["provider"],
                "family": resolve_family(row["name"], row["provider"])["key"],
                "provider_label": resolve_family(row["name"], row["provider"])["label"],
                "color": resolve_family(row["name"], row["provider"])["color"],
                "policy": roster.get(row["name"]),
                "active": row["name"] in roster,
            }
            for row in conn.execute("SELECT name, provider FROM models ORDER BY id")
        ]

    entries = evaluations["entries"]
    leaderboard = build_leaderboard(entries)
    run_months = evaluations["run_months"]
    latest_month = run_months[-1] if run_months else None

    total_responses = sum(e["responses"] for e in entries)
    total_graded = sum(e["graded"] for e in entries)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "demo": args.demo,
        "site": site_config,
        "eval_config": EVAL_CONFIG_LABEL,
        "summary": {
            "total_reviews": dataset["total_reviews"],
            "atomic_facts": dataset["atomic_facts"],
            "benchmark_reviews": benchmark["reviews"],
            "benchmark_atomic_facts": benchmark["atomic_facts"],
            "models_tracked": len([m for m in registry if m["active"]]),
            "total_responses": total_responses,
            "graded_responses": total_graded,
            "grading_complete": total_responses > 0 and total_graded == total_responses,
            "latest_run_month": latest_month,
            "latest_run_month_label": month_label(latest_month) if latest_month else None,
            "run_months": run_months,
        },
        "benchmark": benchmark,
        "dataset": dataset,
        "leaderboard": leaderboard,
        "panel_views": evaluations["panel_views"],
        "series": build_series(entries),
        "paper_baselines": build_paper_baselines(),
        "entries": entries,
        "models": registry,
        "pipeline": parse_latest_run(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"wrote {args.out}")
    if benchmark["available"]:
        print(
            f"  benchmark: {benchmark['reviews']} reviews · "
            f"{benchmark['atomic_facts']} atomic facts"
        )
    print(
        f"  live panel: {dataset['total_reviews']} reviews · {dataset['atomic_facts']} atomic facts · "
        f"{len(leaderboard)} models on the board"
    )
    print(f"  {total_graded}/{total_responses} responses graded"
          + ("  [DEMO SCORES]" if args.demo else ""))


if __name__ == "__main__":
    main()
