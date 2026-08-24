"""Prefect workflow for the SciConBench-Track pipeline.

Run once:                       python scicon-track/run_workflow.py --once
Run on the 1st of each month:   python scicon-track/run_workflow.py
Run on the 1st of odd months:   python scicon-track/run_workflow.py --interval bimonthly
Limit to N DOIs (smoke test):   python scicon-track/run_workflow.py --once --max-dois 5
Rolling panel month override:   python scicon-track/run_workflow.py --once --rolling-month 2026-07

The core set must be created once via ``scicon-track init-core-set`` before
any monthly run. Every run processes the (possibly shrunken) core panel plus
every accumulated rolling panel; open-weight models skip DOIs they have
already answered, proprietary models are re-queried in full. Adding a model
to query_batch_config.yaml is enough — the next run backfills it.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make scicon-track/ sub-packages (config, db, huggingface) importable,
# and scripts/ sub-packages (data_preprocessing, data_labeling, data_collection).
_track_dir = str(Path(__file__).resolve().parent)
_scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
for _p in (_track_dir, _scripts_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import asyncio
import json
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from itertools import islice
from pathlib import Path
from typing import Callable, List, Tuple, TypeVar
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

# Prefect's SQLite must not live on NFS — configure *before* importing Prefect.
from prefect_env import configure_prefect_home  # noqa: E402
configure_prefect_home()

from prefect import flow, task  # noqa: E402
from prefect.tasks import exponential_backoff  # noqa: E402

import logging as _logging
_logging.basicConfig(
    level=_logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

# SQLite (scicon-track/db) only allows one writer at a time process-wide, and
# the ThreadPoolExecutor-based sharding below (atomic facts, precision,
# recall) runs several worker *threads* truly concurrently -- unlike the
# asyncio-based query-stage sharding, where cooperative scheduling means
# sync DB calls never actually overlap. This lock serializes just the DB
# write itself (not the slow LLM call before it), so "database is locked"
# errors can't happen while still keeping almost all of the concurrency win.
_DB_WRITE_LOCK = threading.Lock()


# ── Per-DOI result files (SQLite is the source of truth; these are a
#    human-browsable, easy-to-diff mirror on disk) ──────────────────────────
#
# Every DOI/model gets one JSON file at
# data_track/results/<model>_tools_filter/<doi_safe>/result.json -- the same
# <model_dir>/<doi_safe>/result.json layout sciconharness/logs/ uses for the
# static benchmark (see scripts/check_conclusions.py), so the same kind of
# "does this have a well-formed [[[...]]] response" check can point at this
# directory too. Each pipeline stage (query, atomic facts, precision,
# recall) merges its own top-level key into the file as that DOI/model pair
# reaches it, so a partially-processed DOI is visible on disk exactly as far
# as it got -- e.g. a file with "response" but no "atomic_facts" key means
# generation succeeded but fact decomposition hasn't run yet, which is
# exactly the signal needed to find & re-run only what's missing/malformed.
def _results_file_path(
    *, doi: str, model: str, run_month: str | None, use_tools: bool = True, use_filter: bool = True,
) -> Path:
    from config import path_cfg
    from sciconharness.utils.query_utils import sanitize_doi_for_path

    run_month = run_month or "unspecified-month"
    parts = [model.replace("/", "_")]
    if use_tools:
        parts.append("tools")
    if use_filter:
        parts.append("filter")
    model_dir = "_".join(parts)
    # run_month partitions the tree (data_track/results/<run_month>/...) so
    # the same DOI/model pair queried again in a later month's run -- which
    # happens for the fixed "core" sample, and possibly for DOIs that recur
    # in the rolling panel -- gets its own file instead of silently
    # overwriting the previous month's, matching the DB's uniqueness
    # constraint on (doi, model, provider, config_label, run_month). Pointing
    # `check_conclusions.py --logs-dir` at one `results/<run_month>/`
    # subdirectory still gives the same <model_dir>/<doi_safe>/result.json
    # layout it expects.
    return (
        path_cfg.data_dir / "results" / run_month / model_dir
        / sanitize_doi_for_path(doi) / "result.json"
    )


def _update_result_file(path: Path, updates: dict) -> None:
    """Merge `updates` into the JSON file at `path`, creating it if needed.

    Safe to call repeatedly across pipeline stages for the same DOI/model:
    each call only adds/overwrites its own top-level keys (e.g. "response",
    "atomic_facts", "precision", "recall"), so later stages layer onto
    whatever earlier stages already wrote instead of clobbering them. Writes
    via a temp-file + rename so a crash mid-write can't corrupt the file.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing: dict = {}
        if path.is_file():
            try:
                existing = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                existing = {}
        existing.update(updates)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(existing, indent=2, default=str))
        tmp_path.replace(path)
    except Exception as exc:
        print(f"Failed to write results file {path}: {exc}")

# Per-item retry knobs, shared by every stage that processes items one at a
# time inside a batch task (questions, precision, recall, queries).
# Distinct from (and on top of) each stage's own lower-level retries — e.g.
# SciConHarness.query()'s max_format_retries, AtomicFactGenerator's
# per-LLM-call max_retries, or the judge analyzers' max_attempts for
# unparseable output. This layer exists so a transient failure that exhausts
# those (rate limit, network blip, etc.) still gets a few more whole-item
# attempts within the same run. A stage then re-scans whatever is still
# incomplete (up to STAGE_ROUNDS) and raises rather than skipping it.
ITEM_RETRY_ATTEMPTS = 3
ITEM_RETRY_BACKOFF_SECS = 10.0
# Extra outer passes over whatever is still incomplete after the per-item
# retries. A stage raises if anything is still unfinished after this many
# rounds so Prefect can retry the whole task rather than silently skipping.
STAGE_ROUNDS = 3
# Atomic-fact stages (Cochrane + model-response) use a higher per-item
# budget, then the same STAGE_ROUNDS re-scan.
FACTS_ITEM_RETRY_ATTEMPTS = 4
MIN_QUESTION_LENGTH = 20
MIN_ATOMIC_FACTS = 1

_T = TypeVar("_T")


def _retry_item(fn: Callable[[], _T], *, label: str,
                 attempts: int = ITEM_RETRY_ATTEMPTS,
                 backoff_secs: float = ITEM_RETRY_BACKOFF_SECS) -> _T:
    """Call ``fn()``, retrying up to ``attempts`` times on any exception.

    Re-raises the last exception if every attempt fails so the caller can
    log it. Stages then re-scan remaining items and raise if anything is
    still incomplete — they do not permanently skip malformed output.
    """
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < attempts:
                print(f"{label} failed (attempt {attempt}/{attempts}): {exc}; "
                      f"retrying in {backoff_secs:.0f}s...")
                time.sleep(backoff_secs)
            else:
                print(f"{label} failed after {attempts} attempts: {exc}")
                try:
                    from workflow_log import get_workflow_log
                    wlog = get_workflow_log()
                    if wlog is not None:
                        wlog.warn(f"{label} failed after {attempts} attempts: {exc}")
                except Exception:
                    pass
    assert last_exc is not None
    raise last_exc


def _workflow_warn(msg: str) -> None:
    """Append a WARN line to the active workflow log (if any)."""
    try:
        from workflow_log import get_workflow_log
        wlog = get_workflow_log()
        if wlog is not None:
            wlog.warn(msg)
    except Exception:
        pass


def _workflow_error(msg: str) -> None:
    try:
        from workflow_log import get_workflow_log
        wlog = get_workflow_log()
        if wlog is not None:
            wlog.error(msg)
    except Exception:
        pass


def _workflow_info(msg: str) -> None:
    try:
        from workflow_log import get_workflow_log
        wlog = get_workflow_log()
        if wlog is not None:
            wlog.info(msg)
    except Exception:
        pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _extract_conclusion(text: str) -> str:
    """Return the [[[...]]] conclusion, or '' if the markers are missing."""
    import re
    m = re.search(r"\[\[\[(.*?)\]\]\]", text or "", re.DOTALL)
    return m.group(1).strip() if m else ""


def _fact_count(pairs) -> int:
    n = 0
    for p in pairs or []:
        if isinstance(p, dict):
            n += len(p.get("atomic_facts") or [])
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            n += len(p[1] or [])
    return n


def _stage_leftover_error(label: str, leftover) -> RuntimeError:
    sample = list(leftover)[:8]
    msg = (
        f"{label}: {len(leftover)} item(s) still incomplete after "
        f"{STAGE_ROUNDS} rounds: {sample}"
    )
    _workflow_error(msg)
    return RuntimeError(msg)


def _grade_result_ok(result: dict, *, n_facts: int, kind: str) -> None:
    """Reject empty or error-filled precision/recall judge output."""
    _require(isinstance(result, dict), f"{kind} result is not a dict")
    if kind == "precision":
        _require("factual_precision" in result, "missing factual_precision")
        _require(
            isinstance(result.get("factual_precision"), (int, float)),
            "factual_precision is not numeric",
        )
        details = result.get("precision_details") or []
        _require(
            len(details) == n_facts,
            f"precision_details has {len(details)} entries, expected {n_facts}",
        )
        for detail in details:
            just = str(detail.get("justification") or "")
            _require("Error in LLM call" not in just, "judge error in precision detail")
            _require(
                detail.get("judgment") in ("SUPPORTED", "CONTRADICTED", "NOT SUPPORTED"),
                f"bad precision judgment {detail.get('judgment')!r}",
            )
        return
    _require("factual_recall" in result, "missing factual_recall")
    _require(
        isinstance(result.get("factual_recall"), (int, float)),
        "factual_recall is not numeric",
    )
    details = result.get("coverage_details") or []
    _require(
        len(details) == n_facts,
        f"coverage_details has {len(details)} entries, expected {n_facts}",
    )
    for detail in details:
        just = str(detail.get("justification") or "")
        _require("Error in LLM call" not in just, "judge error in recall detail")
        _require(isinstance(detail.get("is_supported"), bool), "is_supported is not bool")


# ── Notification helpers ───────────────────────────────────────────────────────
# Rich stage digests live in notifications.py (PipelineReport). Emails are
# best-effort and never abort the pipeline.


def _now() -> str:
    return datetime.now(ZoneInfo("America/New_York")).strftime("%a, %d %b %Y %H:%M:%S EST")


def _in_flow_run() -> bool:
    """True when executing inside an active Prefect flow-run context."""
    try:
        from prefect.context import FlowRunContext
        return FlowRunContext.get() is not None
    except Exception:
        return False


def _run_task(task, *args, **kwargs):
    """Run a Prefect ``@task`` — via the engine in a flow run, else ``.fn()``.

    ``--once`` calls ``sciconbench_track_pipeline.fn(...)`` so we never spin
    up Prefect's ephemeral API server (which often times out on this cluster
    under the default 20s budget). Nested ``@task`` calls must also use
    ``.fn()`` in that mode, otherwise they re-trigger the ephemeral server.
    """
    if _in_flow_run():
        return task(*args, **kwargs)
    return task.fn(*args, **kwargs)


async def _run_task_async(task, *args, **kwargs):
    """Async variant of :func:`_run_task` for ``async def`` tasks."""
    if _in_flow_run():
        return await task(*args, **kwargs)
    return await task.fn(*args, **kwargs)


@task(name="Notify: pipeline started")
def task_notify_start(report) -> None:
    from notifications import notify_start
    notify_start(report)


@task(name="Notify: pipeline succeeded")
def task_notify_success(report) -> None:
    from notifications import notify_success
    notify_success(report)


@task(name="Notify: pipeline failed")
def task_notify_error(report, error: str) -> None:
    from notifications import notify_failure
    notify_failure(report, error)


# ── Pipeline tasks ─────────────────────────────────────────────────────────────


@task(name="Initialize database")
def task_init_db() -> None:
    from db import init_db
    init_db(force=False)


@task(name="Load core set")
def task_load_core_set() -> list[str]:
    """Return the finalized core set. Does not sample or backfill.

    Refuses to start if ``data_track/core_set.json`` is missing — that file
    is written only after ``scicon-track init-core-set`` finishes the full
    sample. CORE rows without the lock are treated as unofficial.
    """
    from db.utils import require_core_set_finalized

    core_dois = require_core_set_finalized()
    print(f"Core set: {len(core_dois)} DOI(s) (finalized lock; never resampled).")
    return core_dois


@task(name="Discover rolling reviews and prune stale DOIs")
def task_discover_and_prune(target_month: str, max_dois: int | None = None) -> list[str]:
    """Discover new reviews and drop stale DOIs.

    New reviews are those not already on HuggingFace or in the local DB.
    Each is registered as rolling; ``cohort_month`` is filled in later from
    the publication date. If a tracked DOI is superseded by a newer
    ``.pubN``, the old DOI is removed and the successor is registered as
    rolling (never added back to core).
    """
    from data_collection.collector import DataCollector

    collector = DataCollector()
    new_dois = collector.discover_rolling_for_month(target_month, limit=max_dois)
    events = collector.prune_stale_dois(target_month)
    print(
        f"Discovered {len(new_dois)} new DOI(s); "
        f"{len(events)} stale event(s)."
    )
    return new_dois


@task(
    name="Download PDFs via Wiley TDM",
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=10),
)
def task_download_pdfs(dois: list[str] | None = None) -> dict[str, str]:
    from data_collection.collector import DataCollector
    from data_collection.utils import current_year_month
    from db.utils import get_dois_needing_pdf

    collector = DataCollector()
    wanted = set(dois) if dois else None
    calendar_month = current_year_month()
    unavailable: set[str] = set()
    paths: dict[str, str] = {}
    for round_i in range(1, STAGE_ROUNDS + 1):
        pending = get_dois_needing_pdf()
        if wanted is not None:
            pending = [d for d in pending if d in wanted]
        pending = [d for d in pending if d not in unavailable]
        if not pending:
            return paths
        print(f"PDF download round {round_i}/{STAGE_ROUNDS}: {len(pending)} DOI(s)...")
        paths.update(collector.download_pdfs(pending, calendar_month=calendar_month))
        unavailable |= getattr(collector, "last_tdm_unavailable", set())
    leftover = get_dois_needing_pdf()
    if wanted is not None:
        leftover = [d for d in leftover if d in wanted]
    leftover = [d for d in leftover if d not in unavailable]
    if leftover:
        raise _stage_leftover_error("PDF download", leftover)
    return paths


@task(
    name="Extract reference text from PDFs",
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=10),
)
def task_extract_text(dois: list[str] | None = None) -> int:
    from data_collection.collector import DataCollector
    from db.utils import get_dois_needing_text_extraction

    collector = DataCollector()
    wanted = set(dois) if dois else None
    extracted = 0
    for round_i in range(1, STAGE_ROUNDS + 1):
        pending = get_dois_needing_text_extraction()
        if wanted is not None:
            pending = [d for d in pending if d in wanted]
        if not pending:
            return extracted
        print(f"Text extraction round {round_i}/{STAGE_ROUNDS}: {len(pending)} DOI(s)...")
        extracted += collector.extract_and_store_text(pending)
    leftover = get_dois_needing_text_extraction()
    if wanted is not None:
        leftover = [d for d in leftover if d in wanted]
    if leftover:
        raise _stage_leftover_error("Text extraction", leftover)
    return extracted


@task(
    name="Generate questions",
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=10),
)
def task_generate_questions(dois: list[str] | None = None) -> None:
    from data_preprocessing.question_generation import QuestionGenerator
    from db.db import ProcessingStatus
    from db.utils import (
        get_dois_needing_question,
        get_reviews_from_db,
        populate_questions,
        update_doi_status,
    )

    wanted = set(dois) if dois else None
    generator = QuestionGenerator()
    import re as _re

    def _parse_background(text: str) -> str:
        """Extract Background or Rationale section for use as background_context."""
        for heading in (r"Background", r"Rationale"):
            m = _re.search(
                rf"(?m)^{heading}\s*\n(.*?)(?=\n[A-Z][a-zA-Z'\u2019 ]+\n|\n\[PLAIN|\Z)",
                text, _re.DOTALL,
            )
            if m:
                return m.group(1).strip()
        return ""

    def _pending_questions() -> list[str]:
        pending = get_dois_needing_question()
        if wanted is not None:
            pending = [d for d in pending if d in wanted]
        return pending

    for round_i in range(1, STAGE_ROUNDS + 1):
        pending = _pending_questions()
        if not pending:
            print("No DOIs need question generation.")
            return
        print(f"Question generation round {round_i}/{STAGE_ROUNDS}: {len(pending)} DOI(s)...")
        reviews = get_reviews_from_db()
        for doi in pending:
            review = reviews.get(doi)
            if not review:
                print(f"Question generation: no review row for {doi}")
                continue
            # Use the structured objectives section when available; fall back to full reference text.
            objective_text = review.get("objectives") or review.get("reference_text") or ""
            if not objective_text:
                print(f"Question generation: empty objectives/text for {doi}")
                continue
            ref_text = review.get("reference_text") or ""
            background_ctx = _parse_background(ref_text)
            try:
                def _one_question():
                    result = generator.run(
                        objective=objective_text,
                        background_context=background_ctx,
                    )
                    q = (result.get("question") or "").strip()
                    _require(
                        len(q) >= MIN_QUESTION_LENGTH,
                        f"question too short ({len(q)} chars)",
                    )
                    return q

                question = _retry_item(_one_question, label=f"Question generation for {doi}")
                populate_questions(doi, question)
                update_doi_status(doi, ProcessingStatus.QUESTION_GENERATED)
            except Exception as exc:
                print(f"Question generation failed for {doi}: {exc}")

    leftover = _pending_questions()
    if leftover:
        raise _stage_leftover_error("Question generation", leftover)


@task(
    name="Generate Cochrane atomic facts (batch)",
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=10),
)
def task_generate_cochrane_facts_batch(batch: dict, questions: dict) -> None:
    from data_preprocessing.atomic_fact_generation import AtomicFactGenerator
    from db.db import ProcessingStatus
    from db.utils import populate_atomic_facts, update_doi_status

    from db.utils import get_dois_needing_cochrane_facts

    still_needed = set(get_dois_needing_cochrane_facts())
    generator = AtomicFactGenerator()
    for doi, review in batch.items():
        if doi not in still_needed:
            continue
        question = questions.get(doi, "")
        # Use authors' conclusions section when available; fall back to full reference text.
        generation_text = review.get("authors_conclusions") or review.get("reference_text", "")
        if not generation_text or not question:
            print(f"Cochrane atomic facts: missing question or conclusions for {doi}")
            continue
        try:
            def _one_facts():
                facts_pairs, _para_breaks, _meta = generator.run(
                    generation=generation_text, question=question,
                )
                _require(
                    _fact_count(facts_pairs) >= MIN_ATOMIC_FACTS,
                    f"Cochrane atomic facts empty for {doi}",
                )
                return facts_pairs

            facts_pairs = _retry_item(
                _one_facts,
                label=f"Cochrane atomic facts for {doi}",
                attempts=FACTS_ITEM_RETRY_ATTEMPTS,
            )
            populate_atomic_facts(doi=doi, source="cochrane", atomic_facts_pairs=facts_pairs)
            update_doi_status(doi, ProcessingStatus.FACTS_GENERATED)
        except Exception as exc:
            print(
                f"Cochrane atomic facts failed for {doi} after "
                f"{FACTS_ITEM_RETRY_ATTEMPTS} attempts: {exc}"
            )


# ── Query-stage concurrency (provider lanes + credential assignment) ─────────
#
# (provider, model) pairs are grouped into "lanes"; every lane runs
# concurrently against every other lane (via asyncio.gather), but *within* a
# lane, models are queried strictly sequentially, one full DOI pass at a
# time. Four lanes, each with its own credential / rate-limit domain:
#
#   openrouter       — Kimi / GLM / Qwen (OPENROUTER_API_KEY*)
#   azure_openai     — OpenAI GPT then DeepSeek (COCHRANE_DASHBOARD_*)
#   azure_anthropic  — Claude on Azure (AZURE_ANTHROPIC_*)
#   gemini           — Gemini (Vertex / GOOGLE_*)
#
# Within azure_openai, openai GPT always runs before azure DeepSeek models
# so the shared Cochrane Dashboard quota is not contended by two models at
# once. A single SciConHarness instance is never called concurrently (it
# mutates per-instance state, e.g. the OpenRouter sticky-routing session_id).
QUERY_LANES: dict[str, tuple[str, ...]] = {
    "openrouter": ("openrouter",),
    "azure_openai": ("openai", "azure"),
    "azure_anthropic": ("claude",),
    "gemini": ("gemini",),
}

# Within the azure_openai lane, force GPT before DeepSeek regardless of
# YAML key order in query_batch_config.yaml.
_AZURE_OPENAI_LANE_ORDER: dict[str, int] = {"openai": 0, "azure": 1}


def _resolve_query_credentials(
    provider: str, model: str,
) -> tuple[str | None, str | None, str | None]:
    """Return (api_key, base_url, api_version) for a query-stage provider.

    - ``openai`` / ``azure`` (DeepSeek): Cochrane Dashboard Azure OpenAI
      (``COCHRANE_DASHBOARD_*``), falling back to ``AZURE_OPENAI_KEY`` /
      ``OPENAI_BASE_URL`` if the dashboard vars are unset.
    - ``claude``: Azure Anthropic Foundry (``AZURE_ANTHROPIC_*``).
      ``AZURE_ANTHROPIC_RESOURCE_NAME`` is still read from the env inside
      ``create_provider()`` / ``ClaudeProvider``.
    - Everything else: leave unset so ``create_provider()`` resolves from env.
    """
    del model  # reserved for future per-model overrides
    if provider in ("openai", "azure"):
        return (
            os.environ.get("COCHRANE_DASHBOARD_OPENAI_KEY")
            or os.environ.get("AZURE_OPENAI_KEY"),
            os.environ.get("COCHRANE_DASHBOARD_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL"),
            os.environ.get("OPENAI_API_VERSION"),
        )
    if provider == "claude":
        return (
            os.environ.get("AZURE_ANTHROPIC_API_KEY"),
            os.environ.get("AZURE_ANTHROPIC_BASE_URL"),
            None,
        )
    return None, None, None


def _round_robin_shards(items: list, n: int) -> list[list]:
    """Split *items* into up to *n* round-robin shards, dropping empty ones."""
    shards = [items[i::n] for i in range(n)]
    return [s for s in shards if s]


def _pending_dois_for_model(
    *,
    provider: str,
    model: str,
    eval_dois: list[str],
    run_month: str,
    config_label: str = "tools_filter",
) -> list[str]:
    """DOIs this model still needs to query this run.

    Open-weight (``evaluate_once``): skip any DOI that already has a
    response in *any* month. Proprietary: skip only DOIs already queried
    *this* ``run_month``. A brand-new model has zero rows, so it backfills
    the full universe automatically.
    """
    from config import query_cfg
    from db.utils import get_dois_with_response

    once = query_cfg.reeval_policy(model) == "once"
    done = get_dois_with_response(
        model=model,
        provider=provider,
        config_label=config_label,
        run_month=None if once else run_month,
    )
    return [doi for doi in eval_dois if doi not in done]


async def _run_provider_lane(
    *,
    lane_name: str,
    lane_models: list[tuple[str, str]],
    config_label: str,
    use_tools: bool,
    use_filter: bool,
    all_titles: list[str],
    doi_to_title: dict[str, str],
    eval_dois: list[str],
    questions_map: dict[str, str],
    reviews: dict[str, dict],
    run_month: str,
    max_format_retries: int,
    min_conclusion_length: int,
) -> None:
    """Sequentially query every (provider, model) in one lane, over all DOIs.

    Different lanes run concurrently with each other (see task_run_queries);
    within a lane, models -- and DOIs within a model -- are processed one at
    a time, so a single SciConHarness instance per model is safe to reuse
    (it is *not* safe to call .query() on concurrently -- it mutates shared
    per-instance state, e.g. the OpenRouter sticky-routing session_id, right
    before each call).
    """
    from config import query_cfg
    from db.utils import populate_model_response
    from sciconharness import SciConHarness

    for provider, model in lane_models:
        dois = _pending_dois_for_model(
            provider=provider,
            model=model,
            eval_dois=eval_dois,
            run_month=run_month,
            config_label=config_label,
        )
        print(
            f"[{lane_name}] {provider}/{model}: {len(dois)} pending of "
            f"{len(eval_dois)} eval DOI(s) (policy={query_cfg.reeval_policy(model)})..."
        )
        _workflow_info(
            f"[{lane_name}] {provider}/{model}: {len(dois)} pending of "
            f"{len(eval_dois)} (policy={query_cfg.reeval_policy(model)})"
        )
        if not dois:
            continue

        api_key, base_url, api_version = _resolve_query_credentials(provider, model)

        # OpenRouter / Gemini leave api_key/base_url unset so create_provider()
        # resolves OPENROUTER_API_KEY* / Vertex AI from env. Azure OpenAI
        # (openai + DeepSeek) and Azure Anthropic (claude) get credentials
        # forced here so the track never silently falls back to the wrong
        # Azure resource.
        harness = SciConHarness(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
            enable_tools=use_tools,
            enable_filtering=use_filter,
            cochrane_titles=all_titles,
            doi_to_title=doi_to_title,
            save_results=False,
            # Response quality-check retry: re-query up to this many times if
            # the response lacks a well-formed [[[...]]] conclusion of at
            # least min_conclusion_length chars (see SciConHarness.query()).
            max_format_retries=max_format_retries,
            min_conclusion_length=min_conclusion_length,
        )
        async with harness:
            for doi in dois:
                question = questions_map.get(doi)
                review = reviews.get(doi, {})
                pub_date_raw = review.get("publication_date")
                pub_date_str = (
                    pub_date_raw.strftime("%d %B %Y")
                    if pub_date_raw and hasattr(pub_date_raw, "strftime")
                    else str(pub_date_raw or "")
                )
                if not question:
                    continue
                last_err: Exception | None = None
                saved = False
                for attempt in range(1, ITEM_RETRY_ATTEMPTS + 1):
                    try:
                        response, usage = await harness.query(
                            question,
                            doi=doi,
                            publication_date=pub_date_str,
                        )
                        conclusion = _extract_conclusion(response)
                        _require(
                            len(conclusion) >= min_conclusion_length,
                            f"missing/short [[[...]]] conclusion "
                            f"({len(conclusion)} chars)",
                        )
                        populate_model_response(
                            doi=doi,
                            model=model,
                            provider=provider,
                            config_label=config_label,
                            query=question,
                            response=response,
                            token_usage=usage,
                            enable_tool_calling=use_tools,
                            enable_filtering=use_filter,
                            publication_date=pub_date_raw,
                            run_month=run_month,
                        )
                        _update_result_file(
                            _results_file_path(
                                doi=doi, model=model, run_month=run_month,
                                use_tools=use_tools, use_filter=use_filter,
                            ),
                            {
                                "doi": doi,
                                "model": model,
                                "provider": provider,
                                "config_label": config_label,
                                "run_month": run_month,
                                "query": question,
                                "response": response,
                                "token_usage": usage,
                            },
                        )
                        saved = True
                        break
                    except Exception as exc:
                        last_err = exc
                        print(
                            f"Query failed for doi={doi} {provider}/{model} "
                            f"(attempt {attempt}/{ITEM_RETRY_ATTEMPTS}): {exc}"
                        )
                        if attempt < ITEM_RETRY_ATTEMPTS:
                            await asyncio.sleep(ITEM_RETRY_BACKOFF_SECS)
                if not saved:
                    msg = (
                        f"Query still incomplete for doi={doi} {provider}/{model} "
                        f"after {ITEM_RETRY_ATTEMPTS} attempts: {last_err}"
                    )
                    print(msg)
                    _workflow_warn(msg)


def _leftover_queries(
    models: list[tuple[str, str]],
    dois: list[str],
    run_month: str,
    config_label: str,
) -> list[tuple[str, str, str]]:
    leftover: list[tuple[str, str, str]] = []
    for provider, model in models:
        for doi in _pending_dois_for_model(
            provider=provider,
            model=model,
            eval_dois=dois,
            run_month=run_month,
            config_label=config_label,
        ):
            leftover.append((provider, model, doi))
    return leftover


@task(
    name="Run model queries",
    retries=2,
    retry_delay_seconds=exponential_backoff(backoff_factor=15),
)
async def task_run_queries(
    dois: list[str],
    run_month: str,
    providers: list[str] | None = None,
) -> None:
    """Query all configured model configs for the given DOIs.

    Each DOI/model pair is retried ``ITEM_RETRY_ATTEMPTS`` times. After a
    full pass, leftover pairs with no saved well-formed response are
    re-queried up to ``STAGE_ROUNDS`` times (same idea as other stages:
    finish the universe, then go back over what is still missing).

    (provider, model) pairs are grouped into lanes (QUERY_LANES) that run
    concurrently against each other; within a lane, models are queried
    strictly sequentially (see _run_provider_lane). Four lanes:
    openrouter, azure_openai (GPT then DeepSeek on COCHRANE_DASHBOARD_*),
    azure_anthropic (Claude on AZURE_ANTHROPIC_*), gemini.
    """
    from config import query_cfg
    from db.utils import get_questions, get_reviews_from_db
    from sciconharness.utils.hf_benchmark_cache import (
        load_cochrane_titles_cached,
        load_doi_to_title_cached,
        standardize_title,
    )

    # Load clean-room filtering variables from the local HF benchmark cache
    # (see sciconharness/utils/hf_benchmark_cache.py) instead of pulling the
    # full dataset again here — task_upload_to_hf (the previous pipeline
    # stage) already refreshed this cache from the exact rows it just
    # merged/published, so this is just a disk read, no second HF pull.
    all_titles = load_cochrane_titles_cached()
    doi_to_title = load_doi_to_title_cached()

    reviews = get_reviews_from_db()
    questions_map = get_questions()

    # Also include rolling DOIs not yet in the HF dataset (i.e. not yet
    # covered by task_upload_to_hf's just-refreshed cache — e.g. brand new
    # this month and still missing a question/atomic-facts, which
    # get_sciconbench_rows() requires before a DOI is eligible for upload).
    for doi in dois:
        if doi not in doi_to_title:
            review = reviews.get(doi, {})
            if review.get("name"):
                doi_to_title[doi] = review["name"]
                all_titles.append(standardize_title(review["name"]))

    # Clean-room evaluations only: tool calling AND result filtering both on.
    # (Previously this also ran "no_tools" / "tools"-only variants; the
    # pipeline now scores exactly one configuration per model.)
    config_label, use_tools, use_filter = "tools_filter", True, True

    models = query_cfg.iter_models()
    if providers:
        want = {p.strip().lower() for p in providers if p.strip()}
        models = [(p, m) for p, m in models if p.lower() in want]
        print(
            f"Query providers restricted to {sorted(want)}: "
            f"{len(models)} model(s) {models}"
        )
        if not models:
            raise ValueError(
                f"No configured models match providers={sorted(want)}. "
                f"Available: {query_cfg.iter_models()}"
            )

    # Bucket (provider, model) pairs into lanes; anything not covered by
    # QUERY_LANES falls into its own "other" lane so nothing is silently
    # dropped if a new provider is added later without updating the map.
    # azure_openai is sorted so openai GPT always precedes azure DeepSeek.
    lane_of_provider: dict[str, str] = {
        provider: lane_name
        for lane_name, lane_providers in QUERY_LANES.items()
        for provider in lane_providers
    }
    lanes: dict[str, list[tuple[str, str]]] = {}
    for provider, model in models:
        lane_name = lane_of_provider.get(provider, "other")
        lanes.setdefault(lane_name, []).append((provider, model))
    if "azure_openai" in lanes:
        # Stable sort: openai GPT before azure DeepSeek; keep YAML order
        # among DeepSeek models.
        lanes["azure_openai"].sort(
            key=lambda pm: _AZURE_OPENAI_LANE_ORDER.get(pm[0], 99),
        )

    print(
        f"Running {len(lanes)} query lane(s) concurrently "
        f"({', '.join(f'{name}: {len(m)} model(s)' for name, m in lanes.items())})..."
    )
    leftover: list[tuple[str, str, str]] = []
    for round_i in range(1, STAGE_ROUNDS + 1):
        leftover = _leftover_queries(models, dois, run_month, config_label)
        if not leftover:
            return
        print(
            f"Query round {round_i}/{STAGE_ROUNDS}: "
            f"{len(leftover)} pending DOI/model pair(s)..."
        )
        _workflow_info(
            f"Query round {round_i}/{STAGE_ROUNDS}: "
            f"{len(leftover)} pending DOI/model pair(s)"
        )
        await asyncio.gather(*(
            _run_provider_lane(
                lane_name=lane_name,
                lane_models=lane_models,
                config_label=config_label,
                use_tools=use_tools,
                use_filter=use_filter,
                all_titles=all_titles,
                doi_to_title=doi_to_title,
                eval_dois=dois,
                questions_map=questions_map,
                reviews=reviews,
                run_month=run_month,
                max_format_retries=query_cfg.max_format_retries,
                min_conclusion_length=query_cfg.min_conclusion_length,
            )
            for lane_name, lane_models in lanes.items()
        ))
    leftover = _leftover_queries(models, dois, run_month, config_label)
    if leftover:
        raise _stage_leftover_error("Model queries", leftover)


# ── Post-query sharding (atomic facts, precision, recall) ──────────────────────
#
# These three stages run strictly one after another (facts -> precision ->
# recall), *after* task_run_queries has fully finished -- so they're free to
# reuse the same two Azure credentials without any cross-stage contention.
# Within each stage: work is split into up to FACTS_MODELS_PER_BATCH (2)
# top-level groups processed concurrently, one dedicated API key per group
# (round-robin over FACTS_JUDGE_API_KEYS), and each group's items are
# further split into FACTS_SHARD_CONCURRENCY (4) concurrent shards sharing
# that group's key -- so up to 2 x 4 = 8 concurrent LLM calls at a time.
# Groups run via a ThreadPoolExecutor (these stages are sync/blocking calls,
# not async), so batches of groups are processed one batch at a time,
# repeating until every group has been handled ("two models at a time...
# repeat sequentially until you finish" for atomic facts; a single batch of
# 2 for precision/recall, since those aren't grouped by model).
FACTS_JUDGE_API_KEYS: list[tuple[str, str, str | None]] = [
    ("AZURE_OPENAI_KEY", "OPENAI_BASE_URL", "OPENAI_API_VERSION"),
    ("COCHRANE_DASHBOARD_OPENAI_KEY", "COCHRANE_DASHBOARD_BASE_URL", None),
]
FACTS_MODELS_PER_BATCH = 2
FACTS_SHARD_CONCURRENCY = 4


def _run_grouped_sharded(
    item_groups: list[list],
    shard_worker: Callable[[list, str | None, str | None], None],
    *,
    group_label: str,
) -> None:
    """Process item_groups (already split however the caller wants) in
    batches of FACTS_MODELS_PER_BATCH concurrent groups, each bound to its
    own API key (round-robin over FACTS_JUDGE_API_KEYS) and each further
    split into FACTS_SHARD_CONCURRENCY concurrent shards sharing that key.
    """
    item_groups = [g for g in item_groups if g]
    if not item_groups:
        return
    for batch_start in range(0, len(item_groups), FACTS_MODELS_PER_BATCH):
        batch = item_groups[batch_start: batch_start + FACTS_MODELS_PER_BATCH]
        jobs: list[tuple[list, str | None, str | None]] = []
        for slot, group_items in enumerate(batch):
            api_key_env, base_url_env, _ = FACTS_JUDGE_API_KEYS[slot % len(FACTS_JUDGE_API_KEYS)]
            api_key = os.environ.get(api_key_env)
            base_url = os.environ.get(base_url_env)
            for shard in _round_robin_shards(group_items, FACTS_SHARD_CONCURRENCY):
                jobs.append((shard, api_key, base_url))
        print(
            f"[{group_label}] batch {batch_start // FACTS_MODELS_PER_BATCH + 1}: "
            f"{len(batch)} group(s) x up to {FACTS_SHARD_CONCURRENCY} shard(s) "
            f"= {len(jobs)} concurrent worker(s)..."
        )
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = [pool.submit(shard_worker, shard, api_key, base_url) for shard, api_key, base_url in jobs]
            for f in futures:
                f.result()


@task(
    name="Generate model-response atomic facts (by model)",
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=10),
)
def task_generate_response_facts_by_model(run_month: str | None = None) -> None:
    """Generate atomic facts for pending model responses, grouped by model.

    Two models at a time (one dedicated API key each), each model's pending
    items sharded 4-way; repeats in batches of two until every model with
    pending responses has been processed (see _run_grouped_sharded).
    Re-scans remaining items up to STAGE_ROUNDS and raises if any are still
    missing well-formed facts.
    """
    from data_preprocessing.atomic_fact_generation import AtomicFactGenerator
    from db.utils import get_unprocessed_model_responses, populate_atomic_facts

    def _run_shard(shard_items: list[tuple[int, dict]], api_key: str | None, base_url: str | None) -> None:
        generator = AtomicFactGenerator(api_key=api_key, base_url=base_url)
        for _response_id, resp_data in shard_items:
            doi = resp_data["doi"]
            question = resp_data.get("query", "")
            conclusion = _extract_conclusion(resp_data.get("response", ""))
            if not conclusion or not question:
                print(
                    f"Response atomic facts: missing question or [[[...]]] "
                    f"conclusion for doi={doi} response_id={_response_id}"
                )
                continue
            try:
                def _one_facts():
                    facts_pairs, _para_breaks, _meta = generator.run(
                        generation=conclusion, question=question,
                    )
                    _require(
                        _fact_count(facts_pairs) >= MIN_ATOMIC_FACTS,
                        f"response atomic facts empty for {doi}",
                    )
                    return facts_pairs

                facts_pairs = _retry_item(
                    _one_facts,
                    label=f"Response atomic facts for doi={doi}",
                    attempts=FACTS_ITEM_RETRY_ATTEMPTS,
                )
                with _DB_WRITE_LOCK:
                    populate_atomic_facts(
                        doi=doi, source="model_response",
                        atomic_facts_pairs=facts_pairs, record_id=_response_id,
                    )
                total_facts = sum(len(decontextualized) for _, decontextualized in facts_pairs)
                _update_result_file(
                    _results_file_path(
                        doi=doi, model=resp_data["model"], run_month=resp_data["run_month"],
                        use_tools=resp_data.get("enable_tool_calling", True),
                        use_filter=resp_data.get("enable_filtering", True),
                    ),
                    {"atomic_facts_pairs": facts_pairs, "total_atomic_facts": total_facts},
                )
            except Exception as exc:
                print(
                    f"Response atomic facts failed for doi={doi} after "
                    f"{FACTS_ITEM_RETRY_ATTEMPTS} attempts: {exc}"
                )

    for round_i in range(1, STAGE_ROUNDS + 1):
        unprocessed = get_unprocessed_model_responses(run_month=run_month)
        if not unprocessed:
            print("Model-response atomic facts: nothing pending.")
            return
        by_model: dict[str, list[tuple[int, dict]]] = {}
        for response_id, resp_data in unprocessed.items():
            by_model.setdefault(resp_data["model"], []).append((response_id, resp_data))
        print(
            f"Model-response atomic facts round {round_i}/{STAGE_ROUNDS}: "
            f"{len(by_model)} model(s), {len(unprocessed)} pending item(s)."
        )
        _run_grouped_sharded(list(by_model.values()), _run_shard, group_label="atomic facts")

    leftover = get_unprocessed_model_responses(run_month=run_month)
    if leftover:
        raise _stage_leftover_error("Model-response atomic facts", leftover.keys())


@task(
    name="Run factual precision analysis",
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=10),
)
def task_run_precision(run_month: str | None = None) -> None:
    from config import llm_judge_cfg
    from db.utils import (
        get_all_model_responses,
        get_graded_response_ids,
        get_model_response_atomic_facts,
        get_reviews_from_db,
        populate_precision_result,
    )

    j = llm_judge_cfg["llm_judge"]
    prec = llm_judge_cfg["factual_precision_analyzer"]
    os.environ["DATA_LABELING_JUDGE_DELAY_S"] = str(prec.get("delay", 0))

    def _pending() -> list[tuple[int, dict]]:
        graded = get_graded_response_ids(precision=True)
        return [
            (rid, data)
            for rid, data in get_all_model_responses(run_month=run_month).items()
            if rid not in graded
        ]

    def _run_shard(shard_items: list[tuple[int, dict]], api_key: str | None, base_url: str | None) -> None:
        from data_labeling import FactualPrecisionAnalyzer, ModelJudgeFactory
        from data_labeling.utils.prompts import factual_precision_few_shot_prompt, factual_precision_zero_shot_prompt

        judge = ModelJudgeFactory.create(
            model=j["model"],
            api_key=api_key,
            base_url=base_url,
            openai_reasoning_effort=j.get("openai_reasoning_effort", "none"),
        )
        prompt = (
            factual_precision_few_shot_prompt
            if prec.get("prompt_mode", "few-shot") == "few-shot"
            else factual_precision_zero_shot_prompt
        )
        precision_judge = FactualPrecisionAnalyzer(
            llm_judge=judge, precision_prompt_template=prompt, temperature=prec.get("temperature", 0.2),
        )
        model_response_facts = get_model_response_atomic_facts()
        reviews = get_reviews_from_db()
        for response_id, resp_data in shard_items:
            doi = resp_data["doi"]
            model_facts = model_response_facts.get(response_id, {}).get("all_facts", [])
            reference_text = reviews.get(doi, {}).get("reference_text", "")
            if not model_facts or not reference_text:
                print(
                    f"Precision: missing model facts or reference text for "
                    f"doi={doi} response_id={response_id}"
                )
                continue
            try:
                def _one_precision():
                    result = precision_judge.compute_factual_precision(
                        llm_atomic_facts=model_facts,
                        ground_truth_text=reference_text,
                    )
                    _grade_result_ok(result, n_facts=len(model_facts), kind="precision")
                    return result

                result = _retry_item(
                    _one_precision, label=f"Precision analysis for doi={doi}",
                )
                with _DB_WRITE_LOCK:
                    populate_precision_result(response_id, doi, result)
                _update_result_file(
                    _results_file_path(
                        doi=doi, model=resp_data["model"], run_month=resp_data["run_month"],
                        use_tools=resp_data.get("enable_tool_calling", True),
                        use_filter=resp_data.get("enable_filtering", True),
                    ),
                    {"precision": result},
                )
            except Exception as exc:
                print(f"Precision failed for doi={doi} after {ITEM_RETRY_ATTEMPTS} attempts: {exc}")

    for round_i in range(1, STAGE_ROUNDS + 1):
        items = _pending()
        if not items:
            print("Factual precision: nothing pending.")
            return
        print(f"Factual precision round {round_i}/{STAGE_ROUNDS}: {len(items)} response(s).")
        _run_grouped_sharded(
            _round_robin_shards(items, FACTS_MODELS_PER_BATCH), _run_shard, group_label="precision",
        )

    leftover = _pending()
    if leftover:
        raise _stage_leftover_error("Factual precision", [rid for rid, _ in leftover])


@task(
    name="Run factual recall analysis",
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=10),
)
def task_run_recall(run_month: str | None = None) -> None:
    from config import llm_judge_cfg
    from db.utils import (
        get_all_model_responses,
        get_atomic_facts,
        get_graded_response_ids,
        populate_recall_result,
    )

    j = llm_judge_cfg["llm_judge"]
    rec = llm_judge_cfg["factual_recall_analyzer"]
    os.environ["DATA_LABELING_JUDGE_DELAY_S"] = str(rec.get("delay", 0))

    def _pending() -> list[tuple[int, dict]]:
        graded = get_graded_response_ids(precision=False)
        return [
            (rid, data)
            for rid, data in get_all_model_responses(run_month=run_month).items()
            if rid not in graded
        ]

    def _run_shard(shard_items: list[tuple[int, dict]], api_key: str | None, base_url: str | None) -> None:
        from data_labeling import FactualRecallAnalyzer, ModelJudgeFactory
        from data_labeling.utils.prompts import factual_recall_few_shot_prompt, factual_recall_zero_shot_prompt

        judge = ModelJudgeFactory.create(
            model=j["model"],
            api_key=api_key,
            base_url=base_url,
            openai_reasoning_effort=j.get("openai_reasoning_effort", "none"),
        )
        prompt = (
            factual_recall_few_shot_prompt
            if rec.get("prompt_mode", "zero-shot") == "few-shot"
            else factual_recall_zero_shot_prompt
        )
        recall_judge = FactualRecallAnalyzer(
            llm_judge=judge, recall_prompt_template=prompt, temperature=rec.get("temperature", 1.0),
        )
        cochrane_facts = get_atomic_facts("cochrane")
        for response_id, resp_data in shard_items:
            doi = resp_data["doi"]
            conclusion = _extract_conclusion(resp_data.get("response", ""))
            gt_facts = cochrane_facts.get(doi, {}).get("all_facts", [])
            if not conclusion or not gt_facts:
                print(
                    f"Recall: missing [[[...]]] conclusion or Cochrane facts for "
                    f"doi={doi} response_id={response_id}"
                )
                continue
            try:
                def _one_recall():
                    result = recall_judge.compute_factual_recall(
                        llm_response_text=conclusion,
                        article_atomic_facts=gt_facts,
                    )
                    _grade_result_ok(result, n_facts=len(gt_facts), kind="recall")
                    return result

                result = _retry_item(
                    _one_recall, label=f"Recall analysis for doi={doi}",
                )
                with _DB_WRITE_LOCK:
                    populate_recall_result(response_id, doi, result)
                _update_result_file(
                    _results_file_path(
                        doi=doi, model=resp_data["model"], run_month=resp_data["run_month"],
                        use_tools=resp_data.get("enable_tool_calling", True),
                        use_filter=resp_data.get("enable_filtering", True),
                    ),
                    {"recall": result},
                )
            except Exception as exc:
                print(f"Recall failed for doi={doi} after {ITEM_RETRY_ATTEMPTS} attempts: {exc}")

    for round_i in range(1, STAGE_ROUNDS + 1):
        items = _pending()
        if not items:
            print("Factual recall: nothing pending.")
            return
        print(f"Factual recall round {round_i}/{STAGE_ROUNDS}: {len(items)} response(s).")
        _run_grouped_sharded(
            _round_robin_shards(items, FACTS_MODELS_PER_BATCH), _run_shard, group_label="recall",
        )

    leftover = _pending()
    if leftover:
        raise _stage_leftover_error("Factual recall", [rid for rid, _ in leftover])


@task(name="Print eval metrics")
def task_print_eval_metrics(
    run_month: str,
    dois: list[str] | None = None,
) -> None:
    from db.utils import get_eval_metrics

    rows = get_eval_metrics(run_month=run_month, dois=dois)
    if not rows:
        print("Eval metrics: no scored responses.")
        return
    print(
        f"{'doi':<44} {'model':<18} {'P':>6} {'R':>6} {'F1':>6}"
    )
    print("-" * 86)
    f1s, ps, rs = [], [], []
    for row in rows:
        p, r, f1 = row["precision"], row["recall"], row["f1"]
        if p is not None:
            ps.append(p)
        if r is not None:
            rs.append(r)
        if f1 is not None:
            f1s.append(f1)
        print(
            f"{row['doi']:<44} {row['model']:<18} "
            f"{p if p is not None else float('nan'):6.3f} "
            f"{r if r is not None else float('nan'):6.3f} "
            f"{f1 if f1 is not None else float('nan'):6.3f}"
        )
    if ps or rs or f1s:
        def _mean(xs: list[float]) -> float:
            return sum(xs) / len(xs) if xs else float("nan")
        print("-" * 86)
        print(
            f"{'macro-average':<44} {'':<18} "
            f"{_mean(ps):6.3f} {_mean(rs):6.3f} {_mean(f1s):6.3f}"
        )


@task(name="Upload to HuggingFace")
def task_upload_to_hf(
    trial: bool = False,
    include_dois: list[str] | None = None,
) -> None:
    from config import hf_cfg
    from huggingface.uploader import SciConBenchUploader

    # Always rebuild the local Cochrane-filter cache from the full
    # production merge (live benchmark + every FACTS_GENERATED DB row).
    # Trial mode still needs that complete title list; it just must not
    # push the merged parquet onto the live ``benchmark`` shard.
    cache_uploader = SciConBenchUploader()
    cache_uploader.save_to_parquet()
    cache_uploader.refresh_filter_caches()

    if trial:
        trial_cfg = hf_cfg.trial
        if trial_cfg is None:
            raise RuntimeError(
                "Trial HuggingFace config is missing from "
                "hugging_face_config.yaml (expected a `trial:` block)."
            )
        uploader = SciConBenchUploader(
            output=trial_cfg.output,
            path_in_repo=trial_cfg.path_in_repo,
            source_config=trial_cfg.config,
            allow_missing_source=True,
            include_dois=include_dois,
        )
        uploader.save_to_parquet()
        url = uploader.upload(
            commit_message="Trial track: append new Cochrane reviews (practice run)",
        )
        try:
            uploader.ensure_hub_config()
        except Exception as exc:
            print(
                f"Trial parquet uploaded but HuggingFace README config "
                f"update failed (file is still at {trial_cfg.path_in_repo}): {exc}"
            )
        print(
            f"Trial upload: {url} "
            f"({len(include_dois or [])} DOI filter(s); "
            f"production benchmark parquet was NOT overwritten)."
        )
        return

    cache_uploader.upload()


# ── Batch helpers ──────────────────────────────────────────────────────────────


def _batches(d: dict, size: int) -> List[Tuple[int, int]]:
    n = len(d)
    num = (n + size - 1) // size
    return [(i * size, min((i + 1) * size, n)) for i in range(num)]


# ── Main flow ──────────────────────────────────────────────────────────────────


@flow(name="SciConBench-Track Monthly Pipeline", log_prints=True)
def sciconbench_track_pipeline(
    batch_size: int = 500,
    max_dois: int | None = None,
    rolling_month: str | None = None,
    trial: bool = False,
    providers: list[str] | None = None,
) -> None:
    """End-to-end monthly pipeline.

    Stages:
      1.  Initialize DB
      2.  Load the already-registered core set (must exist; never resampled)
      3.  Discover all new reviews not already on HuggingFace
          + drop tracked DOIs replaced by a newer .pubN (successor goes
          into rolling, never back into core). Rolling cohort_month is the
          review's publication month, assigned after text extraction.
      4.  Download PDFs for anything still at REGISTERED
      5.  Extract reference text from PDFs
      6.  Generate clinical questions
      7.  Generate Cochrane atomic facts
      8.  Upload to HuggingFace (hayoungjung/SciConBench, benchmark/test;
          ``--trial`` writes config ``trial`` instead and never overwrites
          the live benchmark shard)
      9.  Query models against core + rolling panels whose publication month
          is on or before the last closed calendar month (open-month reviews
          are ingested but not evaluated until that month ends).
          Per-model: open-weight once-ever, proprietary every run
      10. Generate model-response atomic facts
      11. Run precision & recall analysis
    """
    from data_collection.utils import current_year_month, previous_year_month
    from db.utils import (
        get_dois_by_panel,
        get_dois_needing_cochrane_facts,
        get_dois_needing_pdf,
        get_dois_needing_question,
        get_dois_needing_text_extraction,
        get_eval_dois,
        get_questions,
        get_reviews_from_db,
    )
    from db.db import PanelType
    from notifications import PipelineReport
    from workflow_log import create_workflow_log

    if trial and providers is None:
        providers = ["openai"]
    # Practice runs need to score the samples just found this month, so
    # default the closed-month cutoff to the current calendar month.
    if trial and rolling_month is None:
        target_month = current_year_month()
    else:
        target_month = rolling_month or previous_year_month()

    wlog = create_workflow_log(trial=trial, target_month=target_month)
    report = PipelineReport(
        trial=trial,
        target_month=target_month,
        max_dois=max_dois,
        providers=list(providers) if providers else None,
        started_at=_now(),
        workflow_log=wlog,
    )
    wlog.open_run(
        mode="trial" if trial else "production",
        target_month=target_month,
        max_dois=max_dois if max_dois is not None else "none",
        providers=", ".join(providers) if providers else "all",
        started_at=report.started_at,
    )
    print(f"Workflow log: {wlog.path}")
    _run_task(task_notify_start, report)
    if trial:
        print(
            "TRIAL MODE: HuggingFace upload goes to config 'trial' "
            "(production benchmark/test will NOT be overwritten); "
            f"query providers={providers}."
        )
        wlog.info(
            f"TRIAL MODE — HF upload → config 'trial'; providers={providers}"
        )
    print(f"Pipeline closed-month for evals: {target_month}")

    run_status = "failed"
    try:
        # 1. Init DB
        report.begin("1. Initialize database")
        _run_task(task_init_db)
        report.finish("schema ready")

        # 2. Core set (read-only — created once via init-core-set)
        report.begin("2. Load core set")
        core_dois = _run_task(task_load_core_set)
        report.n_core = len(core_dois)
        report.finish(f"{len(core_dois)} DOI(s)")

        # 3. Discover new reviews + prune stale core/rolling
        report.begin("3. Discover + prune")
        new_dois = _run_task(
            task_discover_and_prune,
            target_month=target_month, max_dois=max_dois,
        )
        report.new_dois = list(new_dois)
        report.n_rolling = len(get_dois_by_panel(PanelType.ROLLING))
        report.finish(
            f"{len(new_dois)} new rolling DOI(s); "
            f"{report.n_rolling} rolling total"
        )
        if new_dois:
            wlog.info(
                "New DOIs: " + ", ".join(new_dois[:20])
                + (f" … (+{len(new_dois) - 20} more)" if len(new_dois) > 20 else "")
            )

        # 4. Download PDFs for anything still REGISTERED (new rolling reviews)
        report.begin("4. Download PDFs")
        pending_pdf = get_dois_needing_pdf()
        paths = _run_task(task_download_pdfs)
        report.finish(
            f"{len(paths)} downloaded/existing; "
            f"{len(pending_pdf)} were pending at stage start"
        )

        # 5. Extract text
        report.begin("5. Extract reference text")
        pending_text = get_dois_needing_text_extraction()
        n_extracted = _run_task(task_extract_text)
        report.finish(
            f"{n_extracted} extracted; "
            f"{len(pending_text)} were pending at stage start"
        )

        # 6. Generate questions for reviews without one
        report.begin("6. Generate clinical questions")
        pending_q = get_dois_needing_question()
        _run_task(task_generate_questions)
        still_q = get_dois_needing_question()
        report.finish(
            f"{len(pending_q) - len(still_q)} generated "
            f"({len(pending_q)} pending at start)"
        )

        # 7. Cochrane atomic facts — re-scan remaining DOIs until none are left
        report.begin("7. Cochrane atomic facts")
        n_facts_rounds = 0
        n_facts_start = len(get_dois_needing_cochrane_facts())
        for round_i in range(1, STAGE_ROUNDS + 1):
            needing_facts = get_dois_needing_cochrane_facts()
            if not needing_facts:
                break
            n_facts_rounds += 1
            print(
                f"Cochrane atomic facts round {round_i}/{STAGE_ROUNDS}: "
                f"{len(needing_facts)} DOI(s)..."
            )
            wlog.info(
                f"Cochrane facts round {round_i}/{STAGE_ROUNDS}: "
                f"{len(needing_facts)} DOI(s) remaining"
            )
            reviews = get_reviews_from_db()
            questions_map = get_questions()
            to_process = {
                doi: reviews[doi] for doi in needing_facts if doi in reviews
            }
            if not to_process:
                break
            for start, end in _batches(to_process, batch_size):
                batch = dict(islice(to_process.items(), start, end))
                _run_task(task_generate_cochrane_facts_batch, batch, questions_map)
        leftover_facts = get_dois_needing_cochrane_facts()
        if leftover_facts:
            raise _stage_leftover_error("Cochrane atomic facts", leftover_facts)
        report.finish(
            f"{n_facts_start} pending at start; complete after "
            f"{n_facts_rounds} round(s)"
        )

        # 8. Upload to HuggingFace
        report.begin("8. Upload to HuggingFace")
        upload_dois = list(new_dois) if trial else None
        _run_task(task_upload_to_hf, trial=trial, include_dois=upload_dois)
        if trial:
            report.upload_note = (
                f"trial track only ({len(upload_dois or [])} DOI filter(s)); "
                "production benchmark/test NOT overwritten"
            )
        else:
            report.upload_note = "production benchmark/test merged + uploaded"
        report.finish(report.upload_note)

        # 9. Query models against core + every *closed* rolling panel.
        report.begin("9. Query models")
        eval_dois, held_back = get_eval_dois(target_month)
        if trial:
            new_set = set(new_dois)
            eval_dois = [d for d in eval_dois if d in new_set]
            print(
                f"Trial eval: {len(eval_dois)} of {len(new_dois)} newly "
                f"discovered DOI(s) are ready (cohort_month <= {target_month})."
            )
        elif max_dois is not None:
            eval_dois = eval_dois[:max_dois]
        report.eval_dois = list(eval_dois)
        report.held_back = list(held_back)
        report.n_rolling = len(get_dois_by_panel(PanelType.ROLLING))
        print(
            f"Eval universe: {len(eval_dois)} DOI(s) with questions+facts "
            f"(core + rolling cohort_month <= {target_month}; "
            f"{len(get_dois_by_panel(PanelType.CORE))} core, "
            f"{report.n_rolling} rolling total; "
            f"{len(held_back)} rolling held back as still-open)."
        )
        wlog.info(
            f"Eval universe: {len(eval_dois)} DOI(s); "
            f"held back={len(held_back)}; providers={providers or 'all'}"
        )

        asyncio.run(_run_task_async(
            task_run_queries,
            eval_dois, run_month=target_month, providers=providers,
        ))
        report.finish(
            f"{len(eval_dois)} DOI(s); providers="
            f"{providers or 'all'}"
        )

        # 10. Model-response atomic facts
        report.begin("10. Response atomic facts")
        from db.utils import get_unprocessed_model_responses
        n_resp_pending = len(get_unprocessed_model_responses(run_month=target_month))
        wlog.info(f"Response atomic facts pending at start: {n_resp_pending}")
        _run_task(task_generate_response_facts_by_model, run_month=target_month)
        n_resp_left = len(get_unprocessed_model_responses(run_month=target_month))
        report.finish(
            f"{n_resp_pending} pending at start; {n_resp_left} remaining"
        )

        # 11. Precision, then recall
        report.begin("11. Precision analysis")
        from db.utils import get_all_model_responses, get_graded_response_ids
        n_prec_pending = sum(
            1 for rid in get_all_model_responses(run_month=target_month)
            if rid not in get_graded_response_ids(precision=True)
        )
        wlog.info(f"Precision pending at start: {n_prec_pending}")
        _run_task(task_run_precision, run_month=target_month)
        report.finish(f"{n_prec_pending} pending at start; complete")

        report.begin("12. Recall analysis")
        n_rec_pending = sum(
            1 for rid in get_all_model_responses(run_month=target_month)
            if rid not in get_graded_response_ids(precision=False)
        )
        wlog.info(f"Recall pending at start: {n_rec_pending}")
        _run_task(task_run_recall, run_month=target_month)
        report.finish(f"{n_rec_pending} pending at start; complete")

        _run_task(task_print_eval_metrics, run_month=target_month, dois=eval_dois)

        report.ended_at = _now()
        run_status = "success"
        _run_task(task_notify_success, report)

    except Exception:
        err = traceback.format_exc()
        if report.current_stage:
            report.fail_stage(summary="raised", detail=err.splitlines()[-1] if err else "")
        wlog.exception(err)
        report.ended_at = _now()
        _run_task(task_notify_error, report, err)
        raise
    finally:
        try:
            wlog.close_run(status=run_status, ended_at=report.ended_at or _now())
            print(f"Workflow log written: {wlog.path}")
        except Exception as exc:
            print(f"Failed to close workflow log: {exc}")


# ── Entry point ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SciConBench-Track Monthly Pipeline")
    parser.add_argument("--once", action="store_true",
                        help="Run immediately once instead of starting the scheduler.")
    parser.add_argument("--max-dois", type=int, default=None, metavar="N",
                        help="Limit to N DOIs (for smoke-testing).")
    parser.add_argument("--batch-size", type=int, default=500, metavar="N",
                        help="Atomic-fact batch size.")
    parser.add_argument("--rolling-month", default=None, metavar="YYYY-MM",
                        help="Latest closed month for evals (default: previous "
                             "calendar month; current month with --trial). New "
                             "reviews from any month are still ingested; rolling "
                             "DOIs published after this month are not queried "
                             "until that month has ended.")
    parser.add_argument("--trial", action="store_true",
                        help="Practice run: upload to HuggingFace config 'trial' "
                             "(never overwrites production benchmark/test), "
                             "query OpenAI only, and evaluate only newly "
                             "discovered rolling DOIs.")
    parser.add_argument("--providers", default=None, metavar="LIST",
                        help="Comma-separated providers to query (default: all, "
                             "or openai-only with --trial).")
    parser.add_argument("--interval", choices=["monthly", "bimonthly"], default="monthly",
                        help="Recurring-schedule cadence (ignored with --once): "
                             "'monthly' (1st of each month) or 'bimonthly' "
                             "(1st of odd months). Calendar-aligned, not a rolling interval.")
    args = parser.parse_args()

    providers = (
        [p.strip() for p in args.providers.split(",") if p.strip()]
        if args.providers else None
    )

    if args.once:
        print("Running pipeline once (Prefect ephemeral server bypassed).")
        sciconbench_track_pipeline.fn(
            batch_size=args.batch_size,
            max_dois=args.max_dois,
            rolling_month=args.rolling_month,
            trial=args.trial,
            providers=providers,
        )
    else:
        from prefect.schedules import Cron
        cron = "0 0 1 * *" if args.interval == "monthly" else "0 0 1 1,3,5,7,9,11 *"
        sciconbench_track_pipeline.serve(
            name=f"sciconbench-track-{args.interval}",
            schedules=[Cron(cron, timezone="America/New_York")],
        )
