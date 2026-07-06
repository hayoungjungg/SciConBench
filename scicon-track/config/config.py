"""Pydantic configuration dataclasses for SciConBench-Track.

Only types that are new to the longitudinal tracking pipeline live here.
Model, judge, and preprocessing configs reuse the dataclasses already
defined in ``data_preprocessing`` and ``data_labeling``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

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

    All providers in ``default_models`` are queried every month.
    Add or remove entries there to control which models are benchmarked.
    """

    enable_tool_calling: bool = True
    enable_filtering: bool = True
    default_models: dict[str, str]


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

