#!/usr/bin/env python3
"""Resume post-query stages for a run month: atomic facts → precision → recall.

Idempotent: Stage 10 only processes responses still missing atomic facts;
Stages 11–12 only grade responses that are not yet graded.
"""
from __future__ import annotations

import argparse
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

import db.utils as db_utils
from data_collection.utils import previous_year_month
from run_workflow import (
    _run_task,
    task_generate_response_facts_by_model,
    task_run_precision,
    task_run_recall,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-month",
        default=None,
        help="Run month partition (default: previous calendar month).",
    )
    args = parser.parse_args()
    run_month = args.run_month or previous_year_month()

    all_responses = db_utils.get_all_model_responses(run_month)
    pending_facts = db_utils.get_unprocessed_model_responses(run_month)
    graded_p = db_utils.get_graded_response_ids(precision=True)
    graded_r = db_utils.get_graded_response_ids(precision=False)
    pending_p = [rid for rid in all_responses if rid not in graded_p]
    pending_r = [rid for rid in all_responses if rid not in graded_r]

    print("=" * 60)
    print(f"Resume stages 10–12: run_month={run_month}")
    print(f"  atomic facts pending: {len(pending_facts)}/{len(all_responses)}")
    if pending_facts:
        for rid, data in sorted(pending_facts.items(), key=lambda x: x[1]["model"]):
            print(f"    id={rid}  {data['model']}  {data['doi']}")
    print(f"  precision pending:    {len(pending_p)}/{len(all_responses)}")
    print(f"  recall pending:       {len(pending_r)}/{len(all_responses)}")
    print("=" * 60)

    if not pending_facts and not pending_p and not pending_r:
        print("Nothing pending — exiting.")
        return

    if pending_facts:
        print("Stage 10: model-response atomic facts")
        _run_task(task_generate_response_facts_by_model, run_month=run_month)
    else:
        print("Stage 10: nothing pending, skipping.")

    if pending_p:
        print("Stage 11: factual precision")
        _run_task(task_run_precision, run_month=run_month)
    else:
        print("Stage 11: nothing pending, skipping.")

    if pending_r:
        print("Stage 12: factual recall")
        _run_task(task_run_recall, run_month=run_month)
    else:
        print("Stage 12: nothing pending, skipping.")

    leftover_facts = db_utils.get_unprocessed_model_responses(run_month)
    leftover_p = [
        rid
        for rid in db_utils.get_all_model_responses(run_month)
        if rid not in db_utils.get_graded_response_ids(precision=True)
    ]
    leftover_r = [
        rid
        for rid in db_utils.get_all_model_responses(run_month)
        if rid not in db_utils.get_graded_response_ids(precision=False)
    ]
    print("=" * 60)
    print("Done.")
    print(f"  atomic facts remaining: {len(leftover_facts)}")
    print(f"  precision remaining:    {len(leftover_p)}")
    print(f"  recall remaining:       {len(leftover_r)}")
    print("=" * 60)
    if leftover_facts or leftover_p or leftover_r:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
