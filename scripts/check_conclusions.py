#!/usr/bin/env python3
"""
Check whether each result.json under sciconharness/logs/<model>/<run_id>/
contains a properly-delimited conclusion, i.e. a [[[ ... ]]] block inside
the "response" field (see `extract_conclusion` in evaluate.py).

Usage
-----
  python scripts/check_conclusions.py                  # summary per model
  python scripts/check_conclusions.py -v                # also list offending run_ids
  python scripts/check_conclusions.py --logs-dir PATH   # override logs directory
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CONCLUSION_RE = re.compile(r"\[\[\[(.*?)\]\]\]", re.DOTALL)


def has_conclusion(response: str | None) -> bool:
    return bool(CONCLUSION_RE.search(response or ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-dir",
        default=Path(__file__).resolve().parent.parent / "sciconharness" / "logs",
        type=Path,
        help="Path to sciconharness/logs directory",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="List each run_id missing a conclusion (and any unreadable/missing result.json)",
    )
    args = parser.parse_args()

    logs_dir: Path = args.logs_dir
    if not logs_dir.is_dir():
        raise SystemExit(f"Logs directory not found: {logs_dir}")

    model_dirs = sorted(
        d for d in logs_dir.iterdir()
        if d.is_dir() and d.name != "all_logs"
    )

    grand_total = grand_ok = grand_missing = grand_no_result = grand_bad_json = 0

    for model_dir in model_dirs:
        run_dirs = sorted(d for d in model_dir.iterdir() if d.is_dir())

        total = ok = missing = no_result = bad_json = 0
        missing_run_ids: list[str] = []
        no_result_run_ids: list[str] = []
        bad_json_run_ids: list[str] = []

        for run_dir in run_dirs:
            total += 1
            result_path = run_dir / "result.json"
            if not result_path.is_file():
                no_result += 1
                no_result_run_ids.append(run_dir.name)
                continue

            try:
                data = json.loads(result_path.read_text())
            except (json.JSONDecodeError, OSError):
                bad_json += 1
                bad_json_run_ids.append(run_dir.name)
                continue

            if has_conclusion(data.get("response")):
                ok += 1
            else:
                missing += 1
                missing_run_ids.append(run_dir.name)

        grand_total += total
        grand_ok += ok
        grand_missing += missing
        grand_no_result += no_result
        grand_bad_json += bad_json

        print(
            f"{model_dir.name:35s} total={total:4d}  "
            f"has_conclusion={ok:4d}  missing_conclusion={missing:3d}  "
            f"no_result_json={no_result:3d}  bad_json={bad_json:3d}"
        )

        if args.verbose:
            for run_id in missing_run_ids:
                print(f"    [missing conclusion] {run_id}")
            for run_id in no_result_run_ids:
                print(f"    [no result.json]     {run_id}")
            for run_id in bad_json_run_ids:
                print(f"    [bad json]            {run_id}")

    print("-" * 100)
    print(
        f"{'TOTAL':35s} total={grand_total:4d}  "
        f"has_conclusion={grand_ok:4d}  missing_conclusion={grand_missing:3d}  "
        f"no_result_json={grand_no_result:3d}  bad_json={grand_bad_json:3d}"
    )


if __name__ == "__main__":
    main()
