"""Database helper functions for SciConBench-Track.

Provides idempotent upsert helpers for every table in the schema, plus
query helpers used by the workflow and HuggingFace uploader.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from sqlalchemy import select

import db as _db  # use late-binding so Session always reflects the live engine
from db.db import (
    AtomicFacts,
    AtomicFactSource,
    DOIInfo,
    FactualPrecisionResult,
    FactualRecallResult,
    Model,
    ModelResponse,
    PanelType,
    ProcessingStatus,
    Questions,
    ReviewMetaData,
)

logger = logging.getLogger(__name__)


def _session():
    """Return the current module-level Session factory (late-bound)."""
    return _db.Session


# ── DOI registration ───────────────────────────────────────────────────────────


def register_doi(
    doi: str,
    panel_type: PanelType = PanelType.ROLLING,
    cohort_month: str | None = None,
    status: ProcessingStatus = ProcessingStatus.REGISTERED,
) -> None:
    """Insert a DOI record if it does not already exist (idempotent)."""
    with _session().begin() as session:
        existing = session.get(DOIInfo, doi)
        if existing is None:
            session.add(DOIInfo(
                doi=doi,
                panel_type=panel_type,
                cohort_month=cohort_month,
                processing_status=status,
            ))
            logger.debug("Registered DOI: %s  panel=%s  cohort=%s", doi, panel_type.value, cohort_month)


def register_doi_batch(
    dois: list[str],
    panel_type: PanelType = PanelType.ROLLING,
    cohort_month: str | None = None,
    status: ProcessingStatus = ProcessingStatus.REGISTERED,
) -> int:
    """Register multiple DOIs; returns number of newly inserted records."""
    inserted = 0
    with _session().begin() as session:
        existing = {
            row.doi for row in session.query(DOIInfo.doi).filter(DOIInfo.doi.in_(dois)).all()
        }
        for doi in dois:
            if doi not in existing:
                session.add(DOIInfo(
                    doi=doi,
                    panel_type=panel_type,
                    cohort_month=cohort_month,
                    processing_status=status,
                ))
                inserted += 1
    logger.info("Registered %d new DOI(s) (panel=%s, cohort=%s)", inserted, panel_type.value, cohort_month)
    return inserted


def update_doi_status(doi: str, status: ProcessingStatus, pdf_path: str | None = None) -> None:
    """Update the processing_status (and optionally pdf_path) for a DOI."""
    with _session().begin() as session:
        record = session.get(DOIInfo, doi)
        if record is None:
            raise ValueError(f"DOI not found in database: {doi!r}")
        record.processing_status = status
        if pdf_path is not None:
            record.pdf_path = pdf_path


def update_doi_cohort_month(doi: str, cohort_month: str) -> None:
    """Set ``cohort_month`` on a rolling DOI (no-op for core)."""
    with _session().begin() as session:
        record = session.get(DOIInfo, doi)
        if record is None:
            raise ValueError(f"DOI not found in database: {doi!r}")
        if record.panel_type != PanelType.ROLLING:
            return
        record.cohort_month = cohort_month


def remove_doi(doi: str) -> bool:
    """Delete a DOIInfo row (cascades to review/questions/facts/responses/scores).

    Returns True if a row was deleted.
    """
    with _session().begin() as session:
        record = session.get(DOIInfo, doi)
        if record is None:
            return False
        session.delete(record)
        logger.info("Removed DOI %s from the database (cascade delete).", doi)
        return True


def clear_rolling_panel(*, delete_pdfs: bool = True) -> list[str]:
    """Remove every rolling-panel DOI and optionally its downloaded PDF."""
    from pathlib import Path

    from config import path_cfg
    from data_collection.utils import sanitize_doi_for_filename

    rolling_dois = sorted(get_dois_by_panel(PanelType.ROLLING))
    removed: list[str] = []
    for doi in rolling_dois:
        pdf_path: Path | None = None
        with _session()() as session:
            record = session.get(DOIInfo, doi)
            if record and record.pdf_path:
                pdf_path = Path(record.pdf_path)
        if not remove_doi(doi):
            continue
        removed.append(doi)
        if delete_pdfs and pdf_path and pdf_path.exists():
            pdf_path.unlink()
        safe = sanitize_doi_for_filename(doi)
        for subdir in ("supplemental_html/articles", "supplemental_html/failed"):
            html_path = path_cfg.data_dir / "tdm_extraction" / subdir / f"{safe}.html"
            if html_path.exists():
                html_path.unlink()
    if removed:
        write_doi_panels()
    return removed


def clear_query_artifacts(*, run_months: list[str] | None = None) -> int:
    """Delete model responses and linked response-fact rows.

    When *run_months* is ``None``, every stored response is removed.
    """
    with _session().begin() as session:
        q = session.query(ModelResponse)
        if run_months is not None:
            q = q.filter(ModelResponse.run_month.in_(run_months))
        responses = q.all()
        response_ids = [r.id for r in responses]
        if response_ids:
            session.query(AtomicFacts).filter(
                AtomicFacts.source == AtomicFactSource.MODEL_RESPONSE,
                AtomicFacts.id.in_(response_ids),
            ).delete(synchronize_session=False)
        for response in responses:
            session.delete(response)
        return len(responses)


def reset_for_production(*, refresh_parquet: bool = True) -> dict[str, Any]:
    """Clear rolling panel and practice-run artifacts for a clean prod deployment."""
    import shutil

    from config import hf_cfg, path_cfg

    removed_rolling = clear_rolling_panel()
    cleared_responses = clear_query_artifacts()

    results_root = path_cfg.data_dir / "results"
    cleared_result_months: list[str] = []
    if results_root.exists():
        for month_dir in list(results_root.iterdir()):
            if not month_dir.is_dir():
                continue
            cleared_result_months.append(month_dir.name)
            shutil.rmtree(month_dir, ignore_errors=True)

    cleared_logs: list[str] = []
    logs_root = path_cfg.data_dir / "logs"
    if logs_root.exists():
        for pattern in ("workflow-trial-*", "query-smoke-*"):
            for path in logs_root.glob(pattern):
                path.unlink()
                cleared_logs.append(path.name)

    trial_removed = False
    if hf_cfg.trial is not None and hf_cfg.trial.output.exists():
        hf_cfg.trial.output.unlink()
        trial_removed = True

    if refresh_parquet:
        from data_collection.utils import previous_year_month
        from huggingface.uploader import SciConBenchUploader

        # Local parquet/cache should match what production would publish.
        uploader = SciConBenchUploader()
        uploader.save_to_parquet(closed_month=previous_year_month())
        uploader.refresh_filter_caches()

    write_doi_panels()

    return {
        "removed_rolling": removed_rolling,
        "cleared_responses": cleared_responses,
        "cleared_result_months": cleared_result_months,
        "cleared_logs": cleared_logs,
        "trial_parquet_removed": trial_removed,
    }


def get_doi_info_rows() -> list[dict[str, Any]]:
    """Return every DOIInfo row as a plain dict (panel membership snapshot)."""
    with _session()() as session:
        return [
            {
                "doi": row.doi,
                "panel_type": row.panel_type.value,
                "cohort_month": row.cohort_month,
                "processing_status": row.processing_status.value,
            }
            for row in session.query(DOIInfo).all()
        ]


def append_stale_log(entries: dict[str, str | None]) -> None:
    """Merge *entries* into ``data_track/stale_dois.json``.

    Keys are removed DOIs. Values are the superseding DOI, or ``None`` if
    the DOI was withdrawn with no replacement.
    """
    if not entries:
        return
    from config import path_cfg
    import json

    path = path_cfg.data_dir / "stale_dois.json"
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except Exception:
            existing = {}
    existing.update(entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    logger.info("Stale DOI log updated (%d new): %s", len(entries), path)


TDM_404_SKIP_AFTER = 5
TDM_404_LOG_NAME = "tdm_404.json"


def _tdm_404_path():
    from config import path_cfg
    return path_cfg.data_dir / TDM_404_LOG_NAME


def _read_tdm_404() -> dict[str, Any]:
    import json
    path = _tdm_404_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_tdm_404(data: dict[str, Any]) -> None:
    import json
    path = _tdm_404_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def record_tdm_404(doi: str, calendar_month: str) -> int:
    """Count a Wiley TDM 404 once per calendar month. Return the updated count."""
    data = _read_tdm_404()
    rec = data.get(doi) or {}
    if isinstance(rec, int):
        rec = {"count": rec, "last_month": None}
    if rec.get("last_month") == calendar_month:
        return int(rec.get("count") or 0)
    count = int(rec.get("count") or 0) + 1
    data[doi] = {"count": count, "last_month": calendar_month}
    _write_tdm_404(data)
    return count


def clear_tdm_404(doi: str) -> None:
    """Drop the 404 counter after a successful TDM download."""
    data = _read_tdm_404()
    if doi not in data:
        return
    del data[doi]
    _write_tdm_404(data)


def write_doi_panels(history_events: list[dict[str, Any]] | None = None) -> None:
    """Write ``data_track/doi_panels.json`` mirroring current panel membership."""
    import json
    from config import dc_cfg, path_cfg

    path = path_cfg.data_dir / "doi_panels.json"
    existing_history: list = []
    if path.exists():
        try:
            existing_history = json.loads(path.read_text()).get("history") or []
        except Exception:
            existing_history = []
    if history_events:
        existing_history.extend(history_events)

    rows = get_doi_info_rows()
    core = sorted(r["doi"] for r in rows if r["panel_type"] == PanelType.CORE.value)
    rolling: dict[str, list[str]] = {}
    for r in rows:
        if r["panel_type"] != PanelType.ROLLING.value:
            continue
        month = r["cohort_month"] or "unknown"
        rolling.setdefault(month, []).append(r["doi"])
    for month in rolling:
        rolling[month] = sorted(rolling[month])

    payload = {
        "core": core,
        "rolling": dict(sorted(rolling.items())),
        "core_window": [
            dc_cfg.core_window_start.isoformat(),
            dc_cfg.core_window_end.isoformat(),
        ],
        "core_per_month": dc_cfg.core_per_month,
        "core_sample_seed": dc_cfg.core_sample_seed,
        "history": existing_history,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info(
        "Wrote panel snapshot: %d core, %d rolling months → %s",
        len(core), len(rolling), path,
    )


CORE_SET_LOCK_NAME = "core_set.json"


def _core_set_lock_path():
    from config import path_cfg
    return path_cfg.data_dir / CORE_SET_LOCK_NAME


def read_core_set_lock() -> dict[str, Any] | None:
    """Return the finalized core-set lock file, or None if it does not exist."""
    import json
    path = _core_set_lock_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(
            f"Core-set lock file {path} exists but is unreadable: {exc}. "
            "Refusing to treat the core set as initialized."
        ) from exc
    if not data.get("finalized"):
        return None
    return data


def delete_core_set_lock() -> None:
    """Remove ``data_track/core_set.json`` so a forced redraw can rewrite it."""
    path = _core_set_lock_path()
    if path.is_file():
        path.unlink()
        logger.info("Deleted core-set lock: %s", path)


def write_core_set_lock(
    *,
    dois: list[str],
    counts_by_month: dict[str, int],
    per_month: int,
    source: str = "huggingface",
) -> None:
    """Write the finalized core-set lock. Call only after sampling has finished."""
    import json
    from datetime import datetime, timezone
    from config import dc_cfg

    path = _core_set_lock_path()
    if path.is_file():
        raise RuntimeError(
            f"Core-set lock already exists at {path}. Refusing to overwrite."
        )
    payload = {
        "finalized": True,
        "finalized_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "core_window": [
            dc_cfg.core_window_start.isoformat(),
            dc_cfg.core_window_end.isoformat(),
        ],
        "core_per_month": per_month,
        "core_sample_seed": dc_cfg.core_sample_seed,
        "n_dois": len(dois),
        "counts_by_month": counts_by_month,
        "dois": sorted(dois),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp.replace(path)
    logger.info("Finalized core-set lock: %d DOI(s) → %s", len(dois), path)


def require_core_set_finalized() -> list[str]:
    """Return locked core DOIs, or raise if init-core-set has not finished."""
    lock = read_core_set_lock()
    if lock is None:
        raise RuntimeError(
            "Core set is not finalized. Run `scicon-track init-core-set` and "
            "wait for it to finish (it writes data_track/core_set.json only "
            "after the full sample is complete). The monthly pipeline will "
            "not start until that lock exists."
        )
    return list(lock.get("dois") or [])


# ── DOI queries ────────────────────────────────────────────────────────────────


def get_all_dois() -> set[str]:
    """Return the set of all DOIs currently in the database."""
    with _session()() as session:
        return {row.doi for row in session.query(DOIInfo.doi).all()}


def get_dois_by_panel(panel_type: PanelType) -> set[str]:
    """Return DOIs for the given panel type (CORE or ROLLING)."""
    with _session()() as session:
        return {
            row.doi
            for row in session.query(DOIInfo.doi)
            .filter(DOIInfo.panel_type == panel_type)
            .all()
        }


def get_dois_by_status(status: ProcessingStatus) -> set[str]:
    """Return DOIs at a given processing status."""
    with _session()() as session:
        return {
            row.doi
            for row in session.query(DOIInfo.doi)
            .filter(DOIInfo.processing_status == status)
            .all()
        }


def get_eval_dois(closed_month: str) -> tuple[list[str], list[str]]:
    """Return ``(eval_dois, held_back_rolling)`` for a pipeline run.

    Evals cover core plus rolling reviews whose ``cohort_month`` is on or
    before *closed_month* (the last finished calendar month). Rolling DOIs
    with no cohort yet, or published in a still-open month, are held back
    so a mid-month run does not score an incomplete panel.
    """
    ready = get_dois_by_status(ProcessingStatus.FACTS_GENERATED)
    eval_dois: list[str] = []
    held_back: list[str] = []
    with _session()() as session:
        rows = session.query(DOIInfo).all()
    for row in rows:
        if row.panel_type == PanelType.CORE:
            if row.doi in ready:
                eval_dois.append(row.doi)
            continue
        if row.panel_type != PanelType.ROLLING:
            continue
        cohort = row.cohort_month
        if cohort and cohort <= closed_month:
            if row.doi in ready:
                eval_dois.append(row.doi)
        else:
            held_back.append(row.doi)
    return sorted(eval_dois), sorted(held_back)


def get_dois_needing_pdf() -> list[str]:
    """Return DOIs that still need a PDF (REGISTERED, or FAILED with no file)."""
    with _session()() as session:
        rows = (
            session.query(DOIInfo)
            .filter(
                DOIInfo.processing_status.in_(
                    [ProcessingStatus.REGISTERED, ProcessingStatus.FAILED]
                )
            )
            .all()
        )
        out: list[str] = []
        for row in rows:
            if row.processing_status == ProcessingStatus.REGISTERED:
                out.append(row.doi)
            elif not row.pdf_path:
                out.append(row.doi)
        return out


def get_dois_needing_text_extraction() -> list[str]:
    """Return DOIs that still need reference text extracted.

    Includes rows with a downloaded PDF, failed PDF/extraction retries, and
    failed downloads with no PDF (supplemental-only path).
    """
    with _session()() as session:
        rows = (
            session.query(DOIInfo)
            .filter(
                DOIInfo.processing_status.in_(
                    [ProcessingStatus.PDF_DOWNLOADED, ProcessingStatus.FAILED]
                )
            )
            .all()
        )
        out: list[str] = []
        for row in rows:
            if row.processing_status == ProcessingStatus.PDF_DOWNLOADED:
                out.append(row.doi)
            elif row.processing_status == ProcessingStatus.FAILED:
                out.append(row.doi)
        return out


def get_dois_needing_question() -> list[str]:
    """Return DOIs with extracted text but no generated question."""
    with _session()() as session:
        return [
            row.doi
            for row in session.query(DOIInfo.doi)
            .filter(DOIInfo.processing_status == ProcessingStatus.TEXT_EXTRACTED)
            .all()
        ]


def get_dois_needing_cochrane_facts() -> list[str]:
    """Return DOIs that have a question but no Cochrane atomic facts yet."""
    with _session()() as session:
        fact_dois = {
            row.doi
            for row in session.query(AtomicFacts.doi)
            .filter(AtomicFacts.source == AtomicFactSource.COCHRANE)
            .all()
        }
        return [
            row.doi
            for row in session.query(DOIInfo.doi)
            .filter(DOIInfo.processing_status == ProcessingStatus.QUESTION_GENERATED)
            .all()
            if row.doi not in fact_dois
        ]


def get_graded_response_ids(*, precision: bool) -> set[int]:
    """Return ModelResponse ids that already have a precision or recall row."""
    with _session()() as session:
        if precision:
            return {
                row.model_response_id
                for row in session.query(FactualPrecisionResult.model_response_id).all()
            }
        return {
            row.model_response_id
            for row in session.query(FactualRecallResult.model_response_id).all()
        }


# ── Review metadata ────────────────────────────────────────────────────────────


def populate_review(
    doi: str,
    name: str | None = None,
    reference_text: str | None = None,
    review_type: str | None = None,
    publication_date: date | None = None,
    new_search: bool = False,
    conclusion_changed: bool = False,
    version: int | None = None,
    objectives: str | None = None,
    authors_conclusions: str | None = None,
    citations: str | None = None,
) -> None:
    """Insert or update a ReviewMetaData row for *doi* (idempotent on DOI)."""
    with _session().begin() as session:
        existing = session.get(ReviewMetaData, doi)
        if existing is None:
            session.add(ReviewMetaData(
                doi=doi,
                name=name,
                reference_text=reference_text,
                review_type=review_type,
                publication_date=publication_date,
                new_search=new_search,
                conclusion_changed=conclusion_changed,
                version=version,
                objectives=objectives,
                authors_conclusions=authors_conclusions,
                citations=citations,
            ))
        else:
            if name is not None:
                existing.name = name
            if reference_text is not None:
                existing.reference_text = reference_text
            if review_type is not None:
                existing.review_type = review_type
            if publication_date is not None:
                existing.publication_date = publication_date
            existing.new_search = new_search
            existing.conclusion_changed = conclusion_changed
            if version is not None:
                existing.version = version
            if objectives is not None:
                existing.objectives = objectives
            if authors_conclusions is not None:
                existing.authors_conclusions = authors_conclusions
            if citations is not None:
                existing.citations = citations


def get_reviews_from_db() -> dict[str, dict[str, Any]]:
    """Return all ReviewMetaData rows as {doi: {column: value}} dicts."""
    with _session()() as session:
        rows = session.query(ReviewMetaData).all()
        return {
            row.doi: {
                col.name: getattr(row, col.name)
                for col in ReviewMetaData.__table__.columns
                if col.name != "doi"
            }
            for row in rows
        }


# ── Questions ──────────────────────────────────────────────────────────────────


def populate_questions(doi: str, question: str) -> None:
    """Insert or update the clinical question for *doi*."""
    with _session().begin() as session:
        existing = session.get(Questions, doi)
        if existing is None:
            session.add(Questions(doi=doi, question=question))
        else:
            existing.question = question


def get_questions() -> dict[str, str]:
    """Return {doi: question} for all rows in the questions table."""
    with _session()() as session:
        return {row.doi: row.question for row in session.query(Questions).all()}


# ── Atomic facts ───────────────────────────────────────────────────────────────


def _all_facts_from_pairs(pairs: list) -> list:
    """Flatten decontextualized facts from either tuple or dict pair format."""
    all_facts: list = []
    for p in pairs or []:
        if isinstance(p, dict):
            all_facts.extend(p.get("atomic_facts") or [])
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            all_facts.extend(p[1] or [])
    return all_facts


def populate_atomic_facts(
    doi: str,
    source: str,
    atomic_facts_pairs: list,
    overwrite: bool = False,
    record_id: int | None = None,
) -> None:
    """Insert an AtomicFacts record; skip (or overwrite) if one already exists.

    For ``source="cochrane"`` the row is unique per DOI. For
    ``source="model_response"`` *record_id* must be the originating
    ``ModelResponse.id`` — one facts row per model response, so re-queries
    in a later month (and different models in the same month) don't clobber
    each other.
    """
    source_enum = AtomicFactSource(source)
    with _session().begin() as session:
        if source_enum == AtomicFactSource.MODEL_RESPONSE:
            if record_id is None:
                raise ValueError(
                    "record_id (ModelResponse.id) is required for source='model_response'"
                )
            existing = (
                session.query(AtomicFacts)
                .filter(AtomicFacts.id == record_id, AtomicFacts.source == source_enum)
                .first()
            )
            if existing is not None:
                if not overwrite:
                    return
                session.delete(existing)
                session.flush()
            session.add(AtomicFacts(
                id=record_id, doi=doi, source=source_enum, atomic_facts_pairs=atomic_facts_pairs,
            ))
            return

        existing = (
            session.query(AtomicFacts)
            .filter(AtomicFacts.doi == doi, AtomicFacts.source == source_enum)
            .first()
        )
        if existing is not None:
            if not overwrite:
                return
            existing_id = existing.id
            session.delete(existing)
            session.flush()
            session.add(AtomicFacts(
                id=existing_id, doi=doi, source=source_enum, atomic_facts_pairs=atomic_facts_pairs
            ))
        else:
            import hashlib
            stable_id = int(hashlib.md5(f"{doi}:{source}".encode()).hexdigest(), 16) % (2**31)
            session.add(AtomicFacts(
                id=stable_id, doi=doi, source=source_enum, atomic_facts_pairs=atomic_facts_pairs
            ))


def get_atomic_facts(source: str) -> dict[str, dict[str, Any]]:
    """Return atomic facts for a given source ("cochrane" or "model_response").

    Keyed by DOI. For ``source="model_response"`` prefer
    :func:`get_model_response_atomic_facts` when you need per-response facts
    (multiple models / run months share a DOI).
    """
    source_enum = AtomicFactSource(source)
    with _session()() as session:
        records = (
            session.query(AtomicFacts)
            .filter(AtomicFacts.source == source_enum)
            .all()
        )
        questions = {row.doi: row.question for row in session.query(Questions).all()}
        result = {}
        for row in records:
            pairs = row.atomic_facts_pairs or []
            all_facts = _all_facts_from_pairs(pairs)
            result[row.doi] = {
                "atomic_facts_pairs": pairs,
                "all_facts": all_facts,
                "question": questions.get(row.doi, ""),
                "total_atomic_facts": len(all_facts),
            }
        return result


def get_model_response_atomic_facts() -> dict[int, dict[str, Any]]:
    """Return model-response atomic facts keyed by ``ModelResponse.id``."""
    with _session()() as session:
        records = (
            session.query(AtomicFacts)
            .filter(AtomicFacts.source == AtomicFactSource.MODEL_RESPONSE)
            .all()
        )
        result: dict[int, dict[str, Any]] = {}
        for row in records:
            pairs = row.atomic_facts_pairs or []
            all_facts = _all_facts_from_pairs(pairs)
            result[row.id] = {
                "doi": row.doi,
                "atomic_facts_pairs": pairs,
                "all_facts": all_facts,
                "total_atomic_facts": len(all_facts),
            }
        return result


# ── Model responses ────────────────────────────────────────────────────────────


def _get_or_create_model(session, name: str, provider: str) -> Model:
    instance = (
        session.query(Model).filter(Model.name == name, Model.provider == provider).first()
    )
    if instance is None:
        instance = Model(name=name, provider=provider)
        session.add(instance)
        session.flush()
    return instance


def populate_model_response(
    doi: str,
    model: str,
    provider: str,
    config_label: str,
    query: str,
    response: str,
    token_usage: dict | None = None,
    enable_tool_calling: bool = True,
    enable_filtering: bool = True,
    publication_date: date | None = None,
    run_month: str | None = None,
    overwrite: bool = False,
) -> int | None:
    """Insert a ModelResponse row; returns the new row id, or None if skipped."""
    with _session().begin() as session:
        existing = (
            session.query(ModelResponse)
            .filter(
                ModelResponse.doi == doi,
                ModelResponse.model == model,
                ModelResponse.provider == provider,
                ModelResponse.config_label == config_label,
                ModelResponse.run_month == run_month,
            )
            .first()
        )
        if existing is not None:
            if not overwrite:
                logger.debug("ModelResponse exists for doi=%s model=%s month=%s — skipping.", doi, model, run_month)
                return None
            session.delete(existing)
            session.flush()

        model_record = _get_or_create_model(session, name=model, provider=provider)
        record = ModelResponse(
            doi=doi,
            model=model,
            provider=provider,
            config_label=config_label,
            query=query,
            response=response,
            token_usage=token_usage,
            enable_tool_calling=enable_tool_calling,
            enable_filtering=enable_filtering,
            publication_date=publication_date,
            run_month=run_month,
            model_id=model_record.id,
        )
        session.add(record)
        session.flush()
        return record.id


def get_dois_with_response(
    *,
    model: str,
    provider: str,
    config_label: str = "tools_filter",
    run_month: str | None = None,
) -> set[str]:
    """Return DOIs that already have a ModelResponse for this model/config.

    If *run_month* is given, only rows from that month count (used for
    proprietary re-eval). If omitted, any prior month counts (used for
    open-weight evaluate-once).
    """
    with _session()() as session:
        q = session.query(ModelResponse.doi).filter(
            ModelResponse.model == model,
            ModelResponse.provider == provider,
            ModelResponse.config_label == config_label,
        )
        if run_month is not None:
            q = q.filter(ModelResponse.run_month == run_month)
        return {row.doi for row in q.all()}


def get_all_model_responses(run_month: str | None = None) -> dict[int, Any]:
    """Return all ModelResponse rows as a dict keyed by response id.

    Each value is a plain dict of column name → value, matching the shape
    returned by :func:`get_unprocessed_model_responses`.  Optionally filters
    to a specific *run_month*.
    """
    with _session()() as session:
        q = session.query(ModelResponse)
        if run_month is not None:
            q = q.filter(ModelResponse.run_month == run_month)
        return {
            row.id: {
                col.name: getattr(row, col.name)
                for col in ModelResponse.__table__.columns
            }
            for row in q.all()
        }


def get_unprocessed_model_responses(run_month: str | None = None) -> dict[str, Any]:
    """Return model responses that do not yet have atomic facts generated.

    Optionally filters to a specific *run_month*.
    """
    with _session()() as session:
        processed_response_ids = {
            row.id
            for row in session.scalars(
                select(AtomicFacts).where(
                    AtomicFacts.source == AtomicFactSource.MODEL_RESPONSE
                )
            ).all()
        }
        q = session.query(ModelResponse)
        if run_month is not None:
            q = q.filter(ModelResponse.run_month == run_month)
        return {
            row.id: {
                col.name: getattr(row, col.name)
                for col in ModelResponse.__table__.columns
            }
            for row in q.all()
            if row.id not in processed_response_ids
        }


# ── Precision / recall results ─────────────────────────────────────────────────


def populate_precision_result(
    model_response_id: int,
    doi: str,
    result: dict[str, Any],
    overwrite: bool = False,
) -> None:
    with _session().begin() as session:
        existing = (
            session.query(FactualPrecisionResult)
            .filter(FactualPrecisionResult.model_response_id == model_response_id)
            .first()
        )
        if existing is not None:
            if not overwrite:
                return
            session.delete(existing)
            session.flush()
        session.add(FactualPrecisionResult(
            model_response_id=model_response_id,
            doi=doi,
            factual_precision=result.get("factual_precision", 0.0),
            precision_function_used=result.get("precision_function_used", ""),
            total_llm_facts=result.get("total_llm_facts", 0),
            supported_facts=result.get("supported_facts", 0),
            contradicted_facts=result.get("contradicted_facts", 0),
            not_supported_facts=result.get("not_supported_facts", 0),
            supported_facts_list=result.get("supported_facts_list", []),
            contradicted_facts_list=result.get("contradicted_facts_list", []),
            not_supported_facts_list=result.get("not_supported_facts_list", []),
            precision_details=result.get("precision_details", []),
            all_precision_metrics=result.get("all_precision_metrics", {}),
            ground_truth_source=result.get("ground_truth_source", ""),
            token_usage=result.get("token_usage", {}),
        ))


def populate_recall_result(
    model_response_id: int,
    doi: str,
    result: dict[str, Any],
    overwrite: bool = False,
) -> None:
    with _session().begin() as session:
        existing = (
            session.query(FactualRecallResult)
            .filter(FactualRecallResult.model_response_id == model_response_id)
            .first()
        )
        if existing is not None:
            if not overwrite:
                return
            session.delete(existing)
            session.flush()
        session.add(FactualRecallResult(
            model_response_id=model_response_id,
            doi=doi,
            factual_recall=result.get("factual_recall", 0.0),
            total_article_facts=result.get("total_article_facts", 0),
            supported_facts=result.get("supported_facts", 0),
            not_supported_facts_list=result.get("not_supported_facts", []),
            coverage_details=result.get("coverage_details", []),
            token_usage=result.get("token_usage", {}),
        ))


# ── HuggingFace export ─────────────────────────────────────────────────────────


def get_sciconbench_rows() -> list[dict[str, Any]]:
    """Join all tables into benchmark rows for the HuggingFace upload.

    Returns one row per DOI, sorted alphabetically.  DOIs without a question
    or Cochrane atomic facts are omitted.
    """
    with _session()() as session:
        reviews = {r.doi: r for r in session.query(ReviewMetaData).all()}
        questions_map = {q.doi: q.question for q in session.query(Questions).all()}
        facts_map = {
            f.doi: f
            for f in session.query(AtomicFacts)
            .filter(AtomicFacts.source == AtomicFactSource.COCHRANE)
            .all()
        }
        doi_info_map = {d.doi: d for d in session.query(DOIInfo).all()}

        rows: list[dict[str, Any]] = []
        for doi in sorted(reviews):
            review = reviews[doi]
            question = questions_map.get(doi)
            fact = facts_map.get(doi)
            doi_info = doi_info_map.get(doi)
            if not question or not fact:
                continue
            pairs = fact.atomic_facts_pairs or []
            all_facts = _all_facts_from_pairs(pairs)
            # Format publication_date as "D Month YYYY" (no leading zero on day)
            # to match the canonical HuggingFace dataset format.
            if review.publication_date:
                _d = review.publication_date
                pub_date_str = f"{_d.day} {_d.strftime('%B')} {_d.year}"
            else:
                pub_date_str = None

            rows.append({
                "doi": doi,
                "title": review.name,
                "reference_text": review.reference_text or "",
                "objectives": review.objectives or "",
                "authors_conclusions": review.authors_conclusions or "",
                "question": question,
                "all_facts": all_facts,
                "atomic_facts_pairs": pairs,
                "publication_date": pub_date_str,
                "total_atomic_facts": len(all_facts),
                "review_type": review.review_type or "",
                "new_search": review.new_search,
                "conclusion_changed": review.conclusion_changed,
                "citations": review.citations or "",
                "panel_type": doi_info.panel_type.value if doi_info else "rolling",
                "cohort_month": doi_info.cohort_month if doi_info else None,
            })
        return rows


def get_eval_metrics(
    run_month: str | None = None,
    dois: list[str] | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Join model responses with precision/recall scores for a summary table.

    Returns one dict per matching response. Missing grades are left as
    ``None`` so a partial run is still printable.
    """
    wanted = set(dois) if dois is not None else None
    with _session()() as session:
        q = session.query(ModelResponse)
        if run_month is not None:
            q = q.filter(ModelResponse.run_month == run_month)
        if model is not None:
            q = q.filter(ModelResponse.model == model)
        rows: list[dict[str, Any]] = []
        for resp in q.all():
            if wanted is not None and resp.doi not in wanted:
                continue
            prec = resp.precision_result
            rec = resp.recall_result
            p = prec.factual_precision if prec is not None else None
            r = rec.factual_recall if rec is not None else None
            if p is not None and r is not None and (p + r) > 0:
                f1 = (2.0 * p * r) / (p + r)
            elif p is not None and r is not None:
                f1 = 0.0
            else:
                f1 = None
            rows.append({
                "doi": resp.doi,
                "model": resp.model,
                "provider": resp.provider,
                "run_month": resp.run_month,
                "precision": p,
                "recall": r,
                "f1": f1,
                "supported_llm_facts": prec.supported_facts if prec is not None else None,
                "total_llm_facts": prec.total_llm_facts if prec is not None else None,
                "supported_article_facts": rec.supported_facts if rec is not None else None,
                "total_article_facts": rec.total_article_facts if rec is not None else None,
            })
        return rows
