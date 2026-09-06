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
    # Once this many rolling months have closed, advance the 12-month core
    # window by one cohort each month.
    core_rotation_after_months: int = 12


class QueryBatchConfig(BaseModel):
    """Models evaluated each monthly harness run.

    All providers in ``default_models`` are considered each run. Add or
    remove entries there to control which models are benchmarked. Each
    provider maps to either a single model name (``str``) or a list of
    model names (``list[str]``) — the latter lets a single provider (e.g.
    ``openrouter``) run several distinct models every run.
    """

    enable_tool_calling: bool = True
    enable_filtering: bool = True
    default_models: dict[str, Union[str, list[str]]]
    # Response quality-check retry (see SciConHarness.query()): re-queries a
    # DOI up to this many times if the response is missing a well-formed
    # ``[[[...]]]`` conclusion of at least ``min_conclusion_length`` chars.
    max_format_retries: int = 4
    min_conclusion_length: int = 50
    # Per-turn output cap for every provider. Provider adapters map this to
    # their native max_tokens / max_output_tokens field.
    max_tokens: Optional[int] = 8192
    # Query only this many most-recent closed rolling cohorts, in addition to
    # the current core panel. This bounds first-time evaluation for new models.
    rolling_panel_months: int = 4
    # OpenRouter key assignment (not concurrent lanes). Models in
    # ``openrouter_base_model_lane`` bill to OPENROUTER_API_KEY_BASE_MODEL;
    # everything else uses OPENROUTER_API_KEY. All OpenRouter models run in
    # one sequential query lane (at most one OpenRouter request in flight).
    openrouter_base_model_lane: list[str] = []
    openrouter_generic_lane: list[str] = []
    # Default: every model queries each DOI once. Models listed here are
    # re-queried against the current core + rolling window every run. A
    # brand-new model (zero response rows) evaluates that bounded universe
    # on first appearance, regardless of this list.
    reevaluate_always: list[str] = []
    # Deprecated alias kept for old YAMLs; ignored if ``reevaluate_always``
    # is the source of truth. Prefer leaving this empty.
    evaluate_once: list[str] = []

    def iter_models(self) -> list[tuple[str, str]]:
        """Flatten ``default_models`` into a ``(provider, model)`` pair list."""
        pairs: list[tuple[str, str]] = []
        for provider, models in self.default_models.items():
            model_list = [models] if isinstance(models, str) else models
            pairs.extend((provider, model) for model in model_list)
        return pairs

    def reeval_policy(self, model: str) -> str:
        """Return ``"once"`` (default) or ``"always"`` (opt-in re-query)."""
        return "always" if model in self.reevaluate_always else "once"


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

