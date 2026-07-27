"""Pydantic configuration dataclasses for SciConBench-Track.

Only types that are new to the longitudinal tracking pipeline live here.
Model, judge, and preprocessing configs reuse the dataclasses already
defined in ``data_preprocessing`` and ``data_labeling``.
"""

from __future__ import annotations

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

    def iter_models(self) -> list[tuple[str, str]]:
        """Flatten ``default_models`` into a ``(provider, model)`` pair list."""
        pairs: list[tuple[str, str]] = []
        for provider, models in self.default_models.items():
            model_list = [models] if isinstance(models, str) else models
            pairs.extend((provider, model) for model in model_list)
        return pairs


class HuggingFaceSourceConfig(BaseModel):
    repo_id: str
    config: str
    split: str


class HuggingFaceUploadConfig(BaseModel):
    repo_id: str
    output: Path
    path_in_repo: str


class HuggingFaceConfig(BaseModel):
    source: HuggingFaceSourceConfig
    upload: HuggingFaceUploadConfig

