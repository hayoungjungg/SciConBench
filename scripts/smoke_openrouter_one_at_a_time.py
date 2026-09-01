#!/usr/bin/env python3
"""Sequential smoke: one OpenRouter model at a time, N pending DOIs each.

Uses the normal task_run_queries path (incl. max_tokens completion cap).
Default: 1 DOI per model, models run strictly one after another.
"""
from __future__ import annotations

import argparse
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
    _pending_dois_for_model,
    task_run_queries,
)

KEY_TO_LANE = {
    "OPENROUTER_API_KEY_BASE_MODEL": OPENROUTER_BASE_MODEL_LANE,
    "OPENROUTER_API_KEY": OPENROUTER_GENERIC_LANE,
}

# One model at a time; pair that shared a key runs sequentially here.
DEFAULT_JOBS = [
    ("moonshotai/kimi-k3", "OPENROUTER_API_KEY"),
    ("qwen/qwen3.8-27b", "OPENROUTER_API_KEY"),
    ("minimax/minimax-m3", "OPENROUTER_API_KEY"),
    ("z-ai/glm-5.3", "OPENROUTER_API_KEY_BASE_MODEL"),
    ("qwen/qwen3.8-max", "OPENROUTER_API_KEY_BASE_MODEL"),
]


def _run_one(model: str, key_env: str, run_month: str, limit: int) -> None:
    if not os.getenv(key_env):
        raise SystemExit(f"Missing env var: {key_env}")

    lane = KEY_TO_LANE[key_env]
    eval_dois, held = get_eval_dois(run_month)
    pend = _pending_dois_for_model(
        provider="openrouter",
        model=model,
        eval_dois=eval_dois,
        run_month=run_month,
        config_label="tools_filter",
    )
    smoke_dois = pend[:limit]
    if not smoke_dois:
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

    print("=" * 60)
    print(f"SMOKE model={model}")
    print(f"  key_env={key_env} lane={lane}")
    print(f"  max_tokens={query_cfg.max_tokens}")
    print(f"  dois={smoke_dois} ({len(smoke_dois)} of {len(pend)} pending)")
    print("=" * 60)

    try:
        # Pass only the smoke DOI list so leftover checks don't fail on the rest.
        asyncio.run(
            task_run_queries.fn(
                smoke_dois, run_month=run_month, providers=["openrouter"]
            )
        )
        print(f"SMOKE OK: {model}")
    finally:
        query_cfg.default_models = orig_models
        query_cfg.openrouter_base_model_lane = orig_base
        query_cfg.openrouter_generic_lane = orig_generic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1, help="DOIs per model")
    parser.add_argument(
        "--run-month",
        default=None,
        help="Default: previous closed month",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="If set, smoke only this model (still one-at-a-time)",
    )
    args = parser.parse_args()

    # Keep AZURE_OPENAI_KEY / OPENAI_BASE_URL for Jina summarization.
    # OpenRouter billing is forced per-job via key-env / lane config.

    run_month = args.run_month or previous_year_month()
    jobs = DEFAULT_JOBS
    if args.model:
        jobs = [j for j in DEFAULT_JOBS if j[0] == args.model]
        if not jobs:
            raise SystemExit(f"Unknown model {args.model!r}; choose from {[m for m,_ in DEFAULT_JOBS]}")

    print(f"Sequential OpenRouter smoke: run_month={run_month} limit={args.limit}")
    print(f"max_tokens={query_cfg.max_tokens}")
    for model, key_env in jobs:
        _run_one(model, key_env, run_month, args.limit)
    print("All smoke jobs finished.")


if __name__ == "__main__":
    main()
