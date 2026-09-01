#!/usr/bin/env python3
"""Run post-query stages for one model: atomic facts → precision → recall.

Uses the same Prefect task implementations as ``run_workflow.py``, but
restricts pending items to a single model so partially complete models (e.g.
an in-flight Qwen rerun) are not picked up.

Example:
  python scripts/run_postquery_for_model.py --model z-ai/glm-5.3 --run-month 2026-07
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scicon-track"), str(ROOT / "scripts")]

os.environ.setdefault(
    "PREFECT_HOME", f"/tmp/{os.environ.get('USER', 'user')}-prefect"
)

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import db.utils as db_utils
from data_collection.utils import previous_year_month
from run_workflow import (
    _run_task,
    task_generate_response_facts_by_model,
    task_run_precision,
    task_run_recall,
)


def _filter_responses(
    responses: dict[int, dict[str, Any]], model: str
) -> dict[int, dict[str, Any]]:
    return {
        response_id: data
        for response_id, data in responses.items()
        if data.get("model") == model
    }


def _install_model_filter(model: str) -> tuple[Any, Any]:
    orig_unprocessed = db_utils.get_unprocessed_model_responses
    orig_all = db_utils.get_all_model_responses

    def filtered_unprocessed(run_month: str | None = None) -> dict[int, dict[str, Any]]:
        return _filter_responses(orig_unprocessed(run_month), model)

    def filtered_all(run_month: str | None = None) -> dict[int, dict[str, Any]]:
        return _filter_responses(orig_all(run_month), model)

    db_utils.get_unprocessed_model_responses = filtered_unprocessed
    db_utils.get_all_model_responses = filtered_all
    return orig_unprocessed, orig_all


def _restore_model_filter(orig_unprocessed: Any, orig_all: Any) -> None:
    db_utils.get_unprocessed_model_responses = orig_unprocessed
    db_utils.get_all_model_responses = orig_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run atomic facts, precision, and recall for one model."
    )
    parser.add_argument(
        "--model",
        default="z-ai/glm-5.3",
        help="Model name as stored in model_responses (default: z-ai/glm-5.3).",
    )
    parser.add_argument(
        "--run-month",
        default=None,
        help="Run month partition (default: previous calendar month).",
    )
    args = parser.parse_args()

    run_month = args.run_month or previous_year_month()
    model = args.model

    pending_facts = _filter_responses(
        db_utils.get_unprocessed_model_responses(run_month), model
    )
    all_responses = _filter_responses(
        db_utils.get_all_model_responses(run_month), model
    )
    graded_precision = db_utils.get_graded_response_ids(precision=True)
    graded_recall = db_utils.get_graded_response_ids(precision=False)
    pending_precision = [
        rid for rid in all_responses if rid not in graded_precision
    ]
    pending_recall = [
        rid for rid in all_responses if rid not in graded_recall
    ]

    print("=" * 60)
    print(f"Post-query pipeline: model={model} run_month={run_month}")
    print(f"  atomic facts pending: {len(pending_facts)}/{len(all_responses)}")
    print(f"  precision pending:    {len(pending_precision)}/{len(all_responses)}")
    print(f"  recall pending:       {len(pending_recall)}/{len(all_responses)}")
    print("=" * 60)

    if not all_responses:
        raise SystemExit(f"No model responses found for {model!r} in {run_month}.")

    orig_unprocessed, orig_all = _install_model_filter(model)
    try:
        if pending_facts:
            print("Stage 1/3: model-response atomic facts")
            _run_task(task_generate_response_facts_by_model, run_month=run_month)
        else:
            print("Stage 1/3: model-response atomic facts — nothing pending, skipping.")

        if pending_precision:
            print("Stage 2/3: factual precision")
            _run_task(task_run_precision, run_month=run_month)
        else:
            print("Stage 2/3: factual precision — nothing pending, skipping.")

        if pending_recall:
            print("Stage 3/3: factual recall")
            _run_task(task_run_recall, run_month=run_month)
        else:
            print("Stage 3/3: factual recall — nothing pending, skipping.")
    finally:
        _restore_model_filter(orig_unprocessed, orig_all)

    leftover_facts = _filter_responses(
        db_utils.get_unprocessed_model_responses(run_month), model
    )
    leftover_precision = [
        rid
        for rid, _ in _filter_responses(
            db_utils.get_all_model_responses(run_month), model
        ).items()
        if rid not in db_utils.get_graded_response_ids(precision=True)
    ]
    leftover_recall = [
        rid
        for rid, _ in _filter_responses(
            db_utils.get_all_model_responses(run_month), model
        ).items()
        if rid not in db_utils.get_graded_response_ids(precision=False)
    ]

    print("=" * 60)
    print(f"Done: {model}")
    print(f"  atomic facts remaining: {len(leftover_facts)}")
    print(f"  precision remaining:    {len(leftover_precision)}")
    print(f"  recall remaining:       {len(leftover_recall)}")
    print("=" * 60)

    if leftover_facts or leftover_precision or leftover_recall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
