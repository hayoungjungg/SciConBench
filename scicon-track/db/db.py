"""SQLAlchemy ORM models for SciConBench-Track.

Schema additions vs the static benchmark:
  - DOIInfo.panel_type   – CORE (stable set) or ROLLING (monthly additions)
  - DOIInfo.cohort_month – "YYYY-MM" for rolling reviews; NULL for core
  - DOIInfo.processing_status – tracks progress through the TDM-based pipeline
  - DOIInfo.pdf_path     – local path to the downloaded PDF (rolling only)
  - DOIInfo.added_at     – when the DOI was registered
"""

from __future__ import annotations

import enum
from datetime import date, datetime
from typing import List, Optional

from sqlalchemy import CheckConstraint, ForeignKey, Index, JSON, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db import Base


# ── Enumerations ───────────────────────────────────────────────────────────────


class PanelType(enum.Enum):
    CORE = "core"
    ROLLING = "rolling"


class ProcessingStatus(enum.Enum):
    """Tracks where a DOI is in the TDM-based ingestion pipeline."""
    REGISTERED = "registered"           # DOI known, nothing done yet
    PDF_DOWNLOADED = "pdf_downloaded"   # PDF downloaded from Wiley TDM
    TEXT_EXTRACTED = "text_extracted"   # Reference text extracted from PDF
    QUESTION_GENERATED = "question_generated"  # Clinical question generated
    FACTS_GENERATED = "facts_generated"  # Cochrane atomic facts generated
    COMPLETED = "completed"             # All pipeline steps finished
    FAILED = "failed"                   # A pipeline step failed
    SKIPPED = "skipped"                 # Not eligible (protocol, unavailable, etc.)


class AtomicFactSource(enum.Enum):
    COCHRANE = "cochrane"
    MODEL_RESPONSE = "model_response"


# ── Core tables ────────────────────────────────────────────────────────────────


class DOIInfo(Base):
    __tablename__ = "doi_info"

    doi: Mapped[str] = mapped_column(primary_key=True)
    panel_type: Mapped[PanelType] = mapped_column(default=PanelType.ROLLING)
    cohort_month: Mapped[Optional[str]] = mapped_column(
        nullable=True, default=None,
        comment="YYYY-MM string for rolling reviews; NULL for core set",
    )
    processing_status: Mapped[ProcessingStatus] = mapped_column(
        default=ProcessingStatus.REGISTERED
    )
    pdf_path: Mapped[Optional[str]] = mapped_column(nullable=True, default=None)
    added_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    review_metadata: Mapped[Optional["ReviewMetaData"]] = relationship(
        back_populates="doi_info", cascade="all, delete-orphan", uselist=False
    )
    questions: Mapped[Optional["Questions"]] = relationship(
        back_populates="doi_info", cascade="all, delete-orphan", uselist=False
    )
    atomic_facts: Mapped[List["AtomicFacts"]] = relationship(
        back_populates="doi_info", cascade="all, delete-orphan"
    )
    model_responses: Mapped[List["ModelResponse"]] = relationship(
        back_populates="doi_info", cascade="all, delete-orphan"
    )
    precision_results: Mapped[List["FactualPrecisionResult"]] = relationship(
        back_populates="doi_info", cascade="all, delete-orphan"
    )
    recall_results: Mapped[List["FactualRecallResult"]] = relationship(
        back_populates="doi_info", cascade="all, delete-orphan"
    )


class ReviewMetaData(Base):
    __tablename__ = "review_metadata"

    doi: Mapped[str] = mapped_column(ForeignKey("doi_info.doi"), primary_key=True)
    version: Mapped[Optional[int]] = mapped_column(nullable=True)
    name: Mapped[Optional[str]] = mapped_column(nullable=True)
    reference_text: Mapped[Optional[str]] = mapped_column(nullable=True)
    review_type: Mapped[Optional[str]] = mapped_column(nullable=True)
    publication_date: Mapped[Optional[date]] = mapped_column(nullable=True)
    new_search: Mapped[bool] = mapped_column(default=False)
    conclusion_changed: Mapped[bool] = mapped_column(default=False)
    objectives: Mapped[Optional[str]] = mapped_column(nullable=True)
    authors_conclusions: Mapped[Optional[str]] = mapped_column(nullable=True)
    citations: Mapped[Optional[str]] = mapped_column(nullable=True)

    doi_info: Mapped["DOIInfo"] = relationship(back_populates="review_metadata")


class Questions(Base):
    __tablename__ = "questions"

    doi: Mapped[str] = mapped_column(ForeignKey("doi_info.doi"), primary_key=True)
    question: Mapped[Optional[str]] = mapped_column(nullable=True)

    doi_info: Mapped["DOIInfo"] = relationship(back_populates="questions")


class AtomicFacts(Base):
    __tablename__ = "atomic_facts"

    # id is caller-supplied (e.g. ModelResponse.id for model_response source),
    # not auto-generated — SQLite does not support autoincrement on composite PKs.
    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[AtomicFactSource] = mapped_column(
        default=AtomicFactSource.COCHRANE, primary_key=True
    )
    doi: Mapped[str] = mapped_column(ForeignKey("doi_info.doi"))
    atomic_facts_pairs: Mapped[list] = mapped_column(JSON)

    doi_info: Mapped["DOIInfo"] = relationship(back_populates="atomic_facts")

    __table_args__ = (
        Index("ix_atomic_facts_doi_source", "doi", "source"),
    )


class Model(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str]
    provider: Mapped[str]
    display_name: Mapped[Optional[str]] = mapped_column(nullable=True)
    description: Mapped[Optional[str]] = mapped_column(nullable=True)
    is_deep_research: Mapped[bool] = mapped_column(default=False)

    __table_args__ = (
        UniqueConstraint("name", "provider", name="uq_models_name_provider"),
    )

    responses: Mapped[List["ModelResponse"]] = relationship(
        back_populates="model_info", cascade="all, delete-orphan"
    )


class ModelResponse(Base):
    __tablename__ = "model_responses"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    doi: Mapped[str] = mapped_column(ForeignKey("doi_info.doi"))
    model: Mapped[str]
    provider: Mapped[str]
    # Harness config: no_tools | tools | tools_filter
    config_label: Mapped[str] = mapped_column(default="tools_filter")
    query: Mapped[str]
    response: Mapped[str]
    token_usage: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    enable_tool_calling: Mapped[bool] = mapped_column(default=True)
    enable_filtering: Mapped[bool] = mapped_column(default=True)
    publication_date: Mapped[Optional[date]] = mapped_column(nullable=True)
    # ISO date string "YYYY-MM-DD" for the monthly run that produced this response
    run_month: Mapped[Optional[str]] = mapped_column(nullable=True)
    model_id: Mapped[Optional[int]] = mapped_column(ForeignKey("models.id"), nullable=True)

    doi_info: Mapped["DOIInfo"] = relationship(back_populates="model_responses")
    model_info: Mapped[Optional["Model"]] = relationship(back_populates="responses")
    precision_result: Mapped[Optional["FactualPrecisionResult"]] = relationship(
        back_populates="model_response", uselist=False, cascade="all, delete-orphan"
    )
    recall_result: Mapped[Optional["FactualRecallResult"]] = relationship(
        back_populates="model_response", uselist=False, cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_model_responses_doi", "doi"),
        Index("ix_model_responses_model_provider", "model", "provider"),
        Index("ix_model_responses_run_month", "run_month"),
        UniqueConstraint(
            "doi", "model", "provider", "config_label", "run_month",
            name="uq_model_responses_doi_model_config_month",
        ),
    )


class FactualPrecisionResult(Base):
    __tablename__ = "factual_precision_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_response_id: Mapped[int] = mapped_column(
        ForeignKey("model_responses.id"), unique=True
    )
    doi: Mapped[str] = mapped_column(ForeignKey("doi_info.doi"))
    factual_precision: Mapped[float]
    precision_function_used: Mapped[str]
    total_llm_facts: Mapped[int]
    supported_facts: Mapped[int]
    contradicted_facts: Mapped[int]
    not_supported_facts: Mapped[int]
    supported_facts_list: Mapped[list] = mapped_column(JSON)
    contradicted_facts_list: Mapped[list] = mapped_column(JSON)
    not_supported_facts_list: Mapped[list] = mapped_column(JSON)
    precision_details: Mapped[list] = mapped_column(JSON)
    all_precision_metrics: Mapped[dict] = mapped_column(JSON)
    ground_truth_source: Mapped[str]
    token_usage: Mapped[dict] = mapped_column(JSON)

    model_response: Mapped["ModelResponse"] = relationship(back_populates="precision_result")
    doi_info: Mapped["DOIInfo"] = relationship(back_populates="precision_results")

    __table_args__ = (
        Index("ix_fp_results_doi", "doi"),
        Index("ix_fp_results_model_response_id", "model_response_id"),
    )


class FactualRecallResult(Base):
    __tablename__ = "factual_recall_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_response_id: Mapped[int] = mapped_column(
        ForeignKey("model_responses.id"), unique=True
    )
    doi: Mapped[str] = mapped_column(ForeignKey("doi_info.doi"))
    factual_recall: Mapped[float]
    total_article_facts: Mapped[int]
    supported_facts: Mapped[int]
    not_supported_facts_list: Mapped[list] = mapped_column(JSON)
    coverage_details: Mapped[list] = mapped_column(JSON)
    token_usage: Mapped[dict] = mapped_column(JSON)

    model_response: Mapped["ModelResponse"] = relationship(back_populates="recall_result")
    doi_info: Mapped["DOIInfo"] = relationship(back_populates="recall_results")

    __table_args__ = (
        Index("ix_fr_results_doi", "doi"),
        Index("ix_fr_results_model_response_id", "model_response_id"),
    )
