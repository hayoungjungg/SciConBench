"""Pydantic configuration dataclasses for SciConBench-Track.

Only types that are new to the longitudinal tracking pipeline live here.
Model, judge, and preprocessing configs reuse the dataclasses already
defined in ``data_preprocessing`` and ``data_labeling``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel


class PathConfig(BaseModel):
    data_dir: Path
    db_path: Path
    pdf_dir: Path
    logs_dir: Path


class DataCollectionConfig(BaseModel):
    wiley_tdm_token: Optional[str] = None
    rate_limit_secs: float = 10.0
    max_retries: int = 3
    retry_backoff_secs: float = 30.0
    crossref_mailto: Optional[str] = None
    # One-time core-set sample: up to ``core_per_month`` reviews from each
    # calendar month in this closed window, drawn once via
    # ``scicon-track init-core-set`` and never resampled.
    core_window_start: date = date(2025, 7, 1)
    core_window_end: date = date(2026, 6, 30)
    core_per_month: int = 10
    core_sample_seed: int = 42


class QueryBatchConfig(BaseModel):
    """Models evaluated each monthly harness run.

    All providers in ``default_models`` are queried every month. Add or
    remove entries there to control which models are benchmarked. Each
    provider maps to either a single model name (``str``) or a list of
    model names (``list[str]``) — the latter lets a single provider (e.g.
    ``openrouter``, which hosts Kimi K3, GLM-5.2, Qwen3.5-9B, Qwen3.7-max,
    etc.) run several distinct models every run.
    """

    enable_tool_calling: bool = True
    enable_filtering: bool = True
    default_models: dict[str, Union[str, list[str]]]
    # Response quality-check retry (see SciConHarness.query()): re-queries a
    # DOI up to this many times if the response is missing a well-formed
    # ``[[[...]]]`` conclusion of at least ``min_conclusion_length`` chars.
    max_format_retries: int = 4
    min_conclusion_length: int = 20
    # Open-weight models listed here are queried once per DOI and then
    # skipped on later runs. Every other model in ``default_models`` is
    # treated as proprietary and is re-queried against the full
    # core + accumulated-rolling universe every run. A brand-new model
    # (zero response rows) is always queried against that universe on
    # its first appearance, regardless of this list.
    evaluate_once: list[str] = []

    def iter_models(self) -> list[tuple[str, str]]:
        """Flatten ``default_models`` into a ``(provider, model)`` pair list."""
        pairs: list[tuple[str, str]] = []
        for provider, models in self.default_models.items():
            model_list = [models] if isinstance(models, str) else models
            pairs.extend((provider, model) for model in model_list)
        return pairs

    def reeval_policy(self, model: str) -> str:
        """Return ``"once"`` (open-weight) or ``"always"`` (proprietary)."""
        return "once" if model in self.evaluate_once else "always"


class HuggingFaceSourceConfig(BaseModel):
    repo_id: str
    config: str
    split: str


class HuggingFaceUploadConfig(BaseModel):
    repo_id: str
    output: Path
    path_in_repo: str


class HuggingFaceTrialConfig(BaseModel):
    """Practice-run upload target. Never overwrites the live ``benchmark`` shard."""

    config: str = "trial"
    output: Path
    path_in_repo: str


class HuggingFaceConfig(BaseModel):
    source: HuggingFaceSourceConfig
    upload: HuggingFaceUploadConfig
    trial: HuggingFaceTrialConfig | None = None

