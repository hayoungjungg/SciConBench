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


def get_dois_needing_pdf() -> list[str]:
    """Return rolling DOIs that have been registered but not yet PDF-downloaded."""
    with _session()() as session:
        return [
            row.doi
            for row in session.query(DOIInfo.doi)
            .filter(
                DOIInfo.panel_type == PanelType.ROLLING,
                DOIInfo.processing_status == ProcessingStatus.REGISTERED,
            )
            .all()
        ]


def get_dois_needing_text_extraction() -> list[str]:
    """Return DOIs with a PDF but no extracted reference text yet."""
    with _session()() as session:
        return [
            row.doi
            for row in session.query(DOIInfo.doi)
            .filter(DOIInfo.processing_status == ProcessingStatus.PDF_DOWNLOADED)
            .all()
        ]


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
    """Return DOIs with a question but no Cochrane atomic facts."""
    with _session()() as session:
        return [
            row.doi
            for row in session.query(DOIInfo.doi)
            .filter(DOIInfo.processing_status == ProcessingStatus.QUESTION_GENERATED)
            .all()
        ]


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


def populate_atomic_facts(
    doi: str,
    source: str,
    atomic_facts_pairs: list,
    overwrite: bool = False,
) -> None:
    """Insert an AtomicFacts record; skip (or overwrite) if one already exists."""
    source_enum = AtomicFactSource(source)
    with _session().begin() as session:
        existing = (
            session.query(AtomicFacts)
            .filter(AtomicFacts.doi == doi, AtomicFacts.source == source_enum)
            .first()
        )
        if existing is not None:
            if not overwrite:
                return
            # Re-use the existing id when overwriting to avoid gaps
            existing_id = existing.id
            session.delete(existing)
            session.flush()
            session.add(AtomicFacts(
                id=existing_id, doi=doi, source=source_enum, atomic_facts_pairs=atomic_facts_pairs
            ))
        else:
            # Derive a stable id: hash(doi + source) truncated to a positive int
            import hashlib
            stable_id = int(hashlib.md5(f"{doi}:{source}".encode()).hexdigest(), 16) % (2**31)
            session.add(AtomicFacts(
                id=stable_id, doi=doi, source=source_enum, atomic_facts_pairs=atomic_facts_pairs
            ))


def get_atomic_facts(source: str) -> dict[str, dict[str, Any]]:
    """Return atomic facts for a given source ("cochrane" or "model_response")."""
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
            all_facts = [fact for _, decontextualized in pairs for fact in decontextualized]
            result[row.doi] = {
                "atomic_facts_pairs": pairs,
                "all_facts": all_facts,
                "question": questions.get(row.doi, ""),
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
            all_facts = [f for _, decontextualized in pairs for f in decontextualized]
            rows.append({
                "doi": doi,
                "title": review.name,
                "reference_text": review.reference_text or "",
                "question": question,
                "all_facts": all_facts,
                "atomic_facts_pairs": pairs,
                "publication_date": (
                    review.publication_date.strftime("%d %B %Y")
                    if review.publication_date else None
                ),
                "total_atomic_facts": len(all_facts),
                "review_type": review.review_type or "",
                "new_search": review.new_search,
                "conclusion_changed": review.conclusion_changed,
                "panel_type": doi_info.panel_type.value if doi_info else "rolling",
                "cohort_month": doi_info.cohort_month if doi_info else None,
            })
        return rows
