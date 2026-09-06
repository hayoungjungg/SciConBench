#!/usr/bin/env python3
"""
Check whether each result.json under sciconharness/logs/<model>/<run_id>/
contains a properly-delimited conclusion, i.e. a [[[ ... ]]] block inside
the "response" field (see `extract_conclusion` in evaluate.py).

Usage
-----
  python scripts/check_conclusions.py                  # summary per model
  python scripts/check_conclusions.py -v                # also list offending run_ids

Retry mode
----------
Passing --retry additionally re-runs response generation (via SciConHarness)
for every run_id that is missing a well-formed conclusion, has unreadable
JSON, or has no result.json at all. The stale/bad run directory is deleted
first, since SciConHarness's own "already processed" skip-check only
verifies that result.json is valid JSON -- not that it contains a
well-formed [[[...]]] conclusion -- so simply re-running would otherwise
just skip the bad ones again.

  python scripts/check_conclusions.py --retry --dry-run        # preview only
  python scripts/check_conclusions.py --retry                  # actually retry
  python scripts/check_conclusions.py --retry --models qwen_qwen3.5-9b_tools
  python scripts/check_conclusions.py --retry \\
      --models some_new_model_dir --provider openrouter --model foo/bar-9000

Retry targets are resolved back to (provider, model, enable_tools,
enable_filtering) using scicon-track/config/query_batch_config.yaml's
default_models, plus a small legacy table (LEGACY_PROVIDER_MODEL_MAP) for
model dirs that are no longer in that config. Model dirs that can't be
resolved either way are skipped with a warning unless --provider/--model
are given explicitly (only valid together with a single --models entry).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONCLUSION_RE = re.compile(r"\[\[\[(.*?)\]\]\]", re.DOTALL)

DEFAULT_LOGS_DIR = PROJECT_ROOT / "sciconharness" / "logs"
DEFAULT_DOI_QUESTIONS = PROJECT_ROOT / "data_track" / "doi_to_question.json"
DEFAULT_DOI_DATES = (
    PROJECT_ROOT / "data_track" / "hf_benchmark_cache" / "doi_to_publication_date.json"
)
DEFAULT_QUERY_BATCH_CONFIG = (
    PROJECT_ROOT / "scicon-track" / "config" / "query_batch_config.yaml"
)

# Models that were run into sciconharness/logs/ but have since been removed
# from query_batch_config.yaml's default_models, so --retry can't discover
# their (provider, model) automatically from that file. Add entries here as
# needed, or use --provider/--model with --models for a one-off directory.
LEGACY_PROVIDER_MODEL_MAP: dict[str, tuple[str, str]] = {
    "qwen_qwen3.5-9b": ("openrouter", "qwen/qwen3.5-9b"),
}


def has_conclusion(response: str | None) -> bool:
    return bool(CONCLUSION_RE.search(response or ""))


def split_config_suffix(model_dir_name: str) -> tuple[str, bool, bool]:
    """Split a model_dir name into (base_name, enable_tools, enable_filtering)."""
    if model_dir_name.endswith("_tools_filter"):
        return model_dir_name[: -len("_tools_filter")], True, True
    if model_dir_name.endswith("_tools"):
        return model_dir_name[: -len("_tools")], True, False
    return model_dir_name, False, False


def load_provider_model_map(query_batch_config: Path) -> dict[str, tuple[str, str]]:
    """Map a sanitized base model name -> (provider, real_model_name).

    Sourced from query_batch_config.yaml's default_models (current models)
    plus LEGACY_PROVIDER_MODEL_MAP (models retired from that config but
    still present under sciconharness/logs/).
    """
    mapping = dict(LEGACY_PROVIDER_MODEL_MAP)
    if query_batch_config.is_file():
        import yaml

        try:
            cfg = yaml.safe_load(query_batch_config.read_text()) or {}
        except Exception as exc:
            print(f"warning: could not parse {query_batch_config}: {exc}", file=sys.stderr)
            cfg = {}
        for provider, models in (cfg.get("default_models") or {}).items():
            if isinstance(models, str):
                models = [models]
            for model in models:
                mapping[model.replace("/", "_")] = (provider, model)
    return mapping


def load_doi_lookup(doi_questions_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Return (doi_safe -> real_doi, real_doi -> question)."""
    from sciconharness.utils.query_utils import sanitize_doi_for_path

    doi_to_question: dict[str, str] = json.loads(doi_questions_path.read_text())
    doi_safe_to_real = {sanitize_doi_for_path(doi): doi for doi in doi_to_question}
    return doi_safe_to_real, doi_to_question


async def retry_model_dir(
    *,
    model_dir_name: str,
    provider: str,
    model: str,
    enable_tools: bool,
    enable_filtering: bool,
    bad_run_ids: list[str],
    doi_safe_to_real: dict[str, str],
    doi_to_question: dict[str, str],
    doi_to_date: dict[str, str],
    logs_dir: Path,
    max_format_retries: int,
    min_conclusion_length: int,
    dry_run: bool,
) -> None:
    from sciconharness.harness import SciConHarness
    from sciconharness.utils.query_utils import sanitize_doi_for_path

    retry_doi_to_question: dict[str, str] = {}
    unmapped: list[str] = []
    for run_id in bad_run_ids:
        real_doi = doi_safe_to_real.get(run_id)
        if not real_doi or real_doi not in doi_to_question:
            unmapped.append(run_id)
            continue
        retry_doi_to_question[real_doi] = doi_to_question[real_doi]

    if unmapped:
        print(
            f"  [{model_dir_name}] {len(unmapped)} run_id(s) not found in "
            f"{DEFAULT_DOI_QUESTIONS.name} -- skipping: {unmapped}"
        )

    if not retry_doi_to_question:
        return

    print(
        f"  [{model_dir_name}] retrying {len(retry_doi_to_question)} DOI(s) "
        f"(provider={provider}, model={model}, tools={enable_tools}, "
        f"filter={enable_filtering})"
    )
    if dry_run:
        for doi in retry_doi_to_question:
            print(f"    would retry: {doi}")
        return

    # Delete stale/bad run dirs first: SciConHarness._is_doi_already_processed()
    # only checks that result.json is valid JSON, not that it contains a
    # well-formed [[[...]]] conclusion, so it would otherwise just skip these
    # again instead of retrying.
    model_dir = logs_dir / model_dir_name
    for doi in retry_doi_to_question:
        run_dir = model_dir / sanitize_doi_for_path(doi)
        if run_dir.is_dir():
            shutil.rmtree(run_dir)

    harness = SciConHarness(
        provider=provider,
        model=model,
        enable_tools=enable_tools,
        enable_filtering=enable_filtering,
        max_format_retries=max_format_retries,
        min_conclusion_length=min_conclusion_length,
    )
    async with harness:
        results = await harness.query_batch(retry_doi_to_question, doi_to_date)

    still_bad = [doi for doi, r in results.items() if r.get("error")]
    if still_bad:
        print(
            f"  [{model_dir_name}] {len(still_bad)}/{len(retry_doi_to_question)} "
            f"still not well-formatted after retry: {still_bad}"
        )
    else:
        print(f"  [{model_dir_name}] all {len(retry_doi_to_question)} retried DOI(s) fixed")


async def run_retries(
    *,
    retry_targets: dict[str, list[str]],
    logs_dir: Path,
    provider_model_map: dict[str, tuple[str, str]],
    override_provider: str | None,
    override_model: str | None,
    doi_questions_path: Path,
    doi_dates_path: Path,
    max_format_retries: int,
    min_conclusion_length: int,
    dry_run: bool,
) -> None:
    doi_safe_to_real, doi_to_question = load_doi_lookup(doi_questions_path)
    doi_to_date: dict[str, str] = {}
    if doi_dates_path.is_file():
        doi_to_date = json.loads(doi_dates_path.read_text())

    for model_dir_name, bad_run_ids in retry_targets.items():
        base_name, enable_tools, enable_filtering = split_config_suffix(model_dir_name)

        if override_provider and override_model:
            provider, model = override_provider, override_model
        else:
            resolved = provider_model_map.get(base_name)
            if not resolved:
                print(
                    f"  [{model_dir_name}] SKIPPED -- couldn't resolve (provider, model); "
                    f"add it to LEGACY_PROVIDER_MODEL_MAP or pass "
                    f"--models {model_dir_name} --provider ... --model ..."
                )
                continue
            provider, model = resolved

        await retry_model_dir(
            model_dir_name=model_dir_name,
            provider=provider,
            model=model,
            enable_tools=enable_tools,
            enable_filtering=enable_filtering,
            bad_run_ids=bad_run_ids,
            doi_safe_to_real=doi_safe_to_real,
            doi_to_question=doi_to_question,
            doi_to_date=doi_to_date,
            logs_dir=logs_dir,
            max_format_retries=max_format_retries,
            min_conclusion_length=min_conclusion_length,
            dry_run=dry_run,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--logs-dir", default=DEFAULT_LOGS_DIR, type=Path,
        help="Path to sciconharness/logs directory",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="List each run_id missing a conclusion (and any unreadable/missing result.json)",
    )
    parser.add_argument(
        "--retry", action="store_true",
        help="Re-run response generation for every bad run_id found (missing "
             "conclusion, bad json, or no result.json)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="With --retry, only print what would be retried; make no API calls "
             "or deletions",
    )
    parser.add_argument(
        "--models", nargs="+", metavar="MODEL_DIR",
        help="With --retry, restrict retries to these model_dir names only "
             "(default: all model dirs with bad run_ids)",
    )
    parser.add_argument(
        "--provider", type=str,
        help="With --retry --models <one dir>, override the auto-resolved provider",
    )
    parser.add_argument(
        "--model", type=str,
        help="With --retry --models <one dir>, override the auto-resolved model name",
    )
    parser.add_argument(
        "--doi-questions", default=DEFAULT_DOI_QUESTIONS, type=Path,
        help="JSON file mapping DOI -> question, used to resolve run_ids back to "
             "DOIs/questions for retries",
    )
    parser.add_argument(
        "--doi-dates", default=DEFAULT_DOI_DATES, type=Path,
        help="JSON file mapping DOI -> publication date, used for retries",
    )
    parser.add_argument(
        "--query-batch-config", default=DEFAULT_QUERY_BATCH_CONFIG, type=Path,
        help="query_batch_config.yaml used to auto-resolve (provider, model) per "
             "model_dir for retries",
    )
    parser.add_argument(
        "--max-format-retries", type=int, default=4,
        help="Max attempts per DOI if the response is not well-formatted (default: 4)",
    )
    parser.add_argument(
        "--min-conclusion-length", type=int, default=50,
        help="Min chars inside [[[...]]] to be well-formatted (default: 50)",
    )
    args = parser.parse_args()

    if args.provider and args.model and (not args.models or len(args.models) != 1):
        parser.error("--provider/--model require --models with exactly one MODEL_DIR")

    logs_dir: Path = args.logs_dir
    if not logs_dir.is_dir():
        raise SystemExit(f"Logs directory not found: {logs_dir}")

    model_dirs = sorted(
        d for d in logs_dir.iterdir()
        if d.is_dir() and d.name != "all_logs"
    )
    if args.models:
        wanted = set(args.models)
        model_dirs = [d for d in model_dirs if d.name in wanted]

    grand_total = grand_ok = grand_missing = grand_no_result = grand_bad_json = 0
    retry_targets: dict[str, list[str]] = {}

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

        bad_run_ids = missing_run_ids + no_result_run_ids + bad_json_run_ids
        if bad_run_ids:
            retry_targets[model_dir.name] = bad_run_ids

    print("-" * 100)
    print(
        f"{'TOTAL':35s} total={grand_total:4d}  "
        f"has_conclusion={grand_ok:4d}  missing_conclusion={grand_missing:3d}  "
        f"no_result_json={grand_no_result:3d}  bad_json={grand_bad_json:3d}"
    )

    if not args.retry:
        return

    if not retry_targets:
        print("\nNo bad run_ids found -- nothing to retry.")
        return

    print(f"\n{'DRY RUN: ' if args.dry_run else ''}Retrying response generation...")
    provider_model_map = load_provider_model_map(args.query_batch_config)
    asyncio.run(
        run_retries(
            retry_targets=retry_targets,
            logs_dir=logs_dir,
            provider_model_map=provider_model_map,
            override_provider=args.provider,
            override_model=args.model,
            doi_questions_path=args.doi_questions,
            doi_dates_path=args.doi_dates,
            max_format_retries=args.max_format_retries,
            min_conclusion_length=args.min_conclusion_length,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
