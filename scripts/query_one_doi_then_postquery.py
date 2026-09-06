#!/usr/bin/env python3
"""Query one DOI across all configured models, then run postquery stages.

Intended for catch-up cases (e.g. a core-set replacement that landed after
the monthly query stage finished).

Example:
  python scripts/query_one_doi_then_postquery.py \\
      --doi 10.1002/14651858.CD000247.pub4 --run-month 2026-08
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

import db.utils as db_utils
from data_collection.utils import previous_year_month
from run_workflow import (
    _run_task,
    task_generate_response_facts_by_model,
    task_run_precision,
    task_run_queries,
    task_run_recall,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doi", required=True, help="DOI to query + grade")
    parser.add_argument(
        "--run-month",
        default=None,
        help="Run month partition (default: previous calendar month)",
    )
    parser.add_argument(
        "--skip-query",
        action="store_true",
        help="Skip Stage 9; only run facts/precision/recall for this DOI",
    )
    args = parser.parse_args()

    run_month = args.run_month or previous_year_month()
    doi = args.doi

    questions = db_utils.get_questions()
    if doi not in questions:
        raise SystemExit(f"No question in DB for {doi}")

    print("=" * 60)
    print(f"Catch-up: doi={doi} run_month={run_month}")
    print(f"  question: {questions[doi][:140]}...")
    print("=" * 60)

    if not args.skip_query:
        print("Stage 9: query this DOI across all configured models")
        asyncio.run(task_run_queries.fn([doi], run_month=run_month))
    else:
        print("Stage 9: skipped (--skip-query)")

    # Restrict postquery stages to responses for this DOI only, so leftover
    # checks don't wait on unrelated unfinished work.
    orig_unprocessed = db_utils.get_unprocessed_model_responses
    orig_all = db_utils.get_all_model_responses

    def filtered_unprocessed(run_month: str | None = None):
        return {
            rid: data
            for rid, data in orig_unprocessed(run_month).items()
            if data.get("doi") == doi
        }

    def filtered_all(run_month: str | None = None):
        return {
            rid: data
            for rid, data in orig_all(run_month).items()
            if data.get("doi") == doi
        }

    db_utils.get_unprocessed_model_responses = filtered_unprocessed
    db_utils.get_all_model_responses = filtered_all
    try:
        pending_facts = filtered_unprocessed(run_month)
        all_for_doi = filtered_all(run_month)
        graded_p = db_utils.get_graded_response_ids(precision=True)
        graded_r = db_utils.get_graded_response_ids(precision=False)
        pending_p = [rid for rid in all_for_doi if rid not in graded_p]
        pending_r = [rid for rid in all_for_doi if rid not in graded_r]

        print(
            f"Postquery scope: responses={len(all_for_doi)} "
            f"facts_pending={len(pending_facts)} "
            f"precision_pending={len(pending_p)} "
            f"recall_pending={len(pending_r)}"
        )
        for rid, data in sorted(all_for_doi.items(), key=lambda x: x[1]["model"]):
            print(f"  id={rid}  {data['model']}")

        if not all_for_doi:
            raise SystemExit(f"No model responses saved for {doi} in {run_month}")

        if pending_facts:
            print("Stage 10: model-response atomic facts")
            _run_task(task_generate_response_facts_by_model, run_month=run_month)
        else:
            print("Stage 10: nothing pending")

        if pending_p:
            print("Stage 11: factual precision")
            _run_task(task_run_precision, run_month=run_month)
        else:
            print("Stage 11: nothing pending")

        if pending_r:
            print("Stage 12: factual recall")
            _run_task(task_run_recall, run_month=run_month)
        else:
            print("Stage 12: nothing pending")
    finally:
        db_utils.get_unprocessed_model_responses = orig_unprocessed
        db_utils.get_all_model_responses = orig_all

    leftover = {
        rid: data
        for rid, data in db_utils.get_all_model_responses(run_month).items()
        if data.get("doi") == doi
    }
    unproc = {
        rid: data
        for rid, data in db_utils.get_unprocessed_model_responses(run_month).items()
        if data.get("doi") == doi
    }
    gp = db_utils.get_graded_response_ids(precision=True)
    gr = db_utils.get_graded_response_ids(precision=False)
    p_left = [rid for rid in leftover if rid not in gp]
    r_left = [rid for rid in leftover if rid not in gr]

    print("=" * 60)
    print("Done.")
    print(f"  responses: {len(leftover)}")
    print(f"  atomic facts remaining: {len(unproc)}")
    print(f"  precision remaining:    {len(p_left)}")
    print(f"  recall remaining:       {len(r_left)}")
    print("=" * 60)
    if unproc or p_left or r_left:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
