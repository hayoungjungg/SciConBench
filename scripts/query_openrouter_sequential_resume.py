#!/usr/bin/env python3
"""Query-only: finish OpenRouter models one at a time (no precision/recall).

Order (skip any model with nothing pending):
  1. z-ai/glm-5.3          → OPENROUTER_API_KEY_BASE_MODEL
  2. qwen/qwen3.8-max      → OPENROUTER_API_KEY_BASE_MODEL
  3. qwen/qwen3.8-27b      → OPENROUTER_API_KEY
  4. minimax/minimax-m3    → OPENROUTER_API_KEY

Kimi K3 is omitted (already complete for 2026-07). Uses the normal
task_run_queries path (max_tokens, Jina Azure summarization, etc.).
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scicon-track"), str(ROOT / "scripts")]

os.environ.setdefault(
    "PREFECT_HOME", f"/tmp/{os.environ.get('USER', 'user')}-prefect"
)

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from config import query_cfg
from data_collection.utils import previous_year_month
from db.utils import get_eval_dois
from run_workflow import (
    OPENROUTER_BASE_MODEL_LANE,
    OPENROUTER_GENERIC_LANE,
    OPENROUTER_LANE_KEY_ENV,
    _pending_dois_for_model,
    task_run_queries,
)

# (model, key_env) — sequential resume order
JOBS: list[tuple[str, str]] = [
    ("z-ai/glm-5.3", "OPENROUTER_API_KEY_BASE_MODEL"),
    ("qwen/qwen3.8-max", "OPENROUTER_API_KEY_BASE_MODEL"),
    ("qwen/qwen3.8-27b", "OPENROUTER_API_KEY"),
    ("minimax/minimax-m3", "OPENROUTER_API_KEY"),
]

KEY_TO_LANE = {
    "OPENROUTER_API_KEY_BASE_MODEL": OPENROUTER_BASE_MODEL_LANE,
    "OPENROUTER_API_KEY": OPENROUTER_GENERIC_LANE,
}


def _run_one(model: str, key_env: str, run_month: str) -> None:
    if not os.getenv(key_env):
        raise SystemExit(f"Missing required env var: {key_env}")

    lane = KEY_TO_LANE[key_env]
    eval_dois, _held = get_eval_dois(run_month)
    pend = _pending_dois_for_model(
        provider="openrouter",
        model=model,
        eval_dois=eval_dois,
        run_month=run_month,
        config_label="tools_filter",
    )
    print("=" * 60)
    print(f"MODEL {model}")
    print(f"  key_env={key_env} lane={lane}")
    print(f"  max_tokens={query_cfg.max_tokens}")
    print(f"  pending={len(pend)}/{len(eval_dois)}")
    print("=" * 60)
    if not pend:
        print(f"[skip] {model}: nothing pending")
        return

    orig_models = dict(query_cfg.default_models)
    orig_base = list(query_cfg.openrouter_base_model_lane)
    orig_generic = list(query_cfg.openrouter_generic_lane)

    query_cfg.default_models = {"openrouter": [model]}
    if lane == OPENROUTER_BASE_MODEL_LANE:
        query_cfg.openrouter_base_model_lane = [model]
        query_cfg.openrouter_generic_lane = []
    else:
        query_cfg.openrouter_base_model_lane = []
        query_cfg.openrouter_generic_lane = [model]

    try:
        asyncio.run(
            task_run_queries.fn(
                eval_dois, run_month=run_month, providers=["openrouter"]
            )
        )
        print(f"Complete: {model}")
    finally:
        query_cfg.default_models = orig_models
        query_cfg.openrouter_base_model_lane = orig_base
        query_cfg.openrouter_generic_lane = orig_generic


def main() -> None:
    run_month = previous_year_month()
    print(f"OpenRouter sequential resume: run_month={run_month}")
    print(f"Jobs: {JOBS}")
    print(f"lane_key_map={OPENROUTER_LANE_KEY_ENV}")
    for model, key_env in JOBS:
        _run_one(model, key_env, run_month)
    print("All sequential OpenRouter jobs finished.")


if __name__ == "__main__":
    main()
