"""Config package for SciConBench-Track.

All configuration objects are loaded once at import time from the YAML files
in this directory.  Paths are resolved relative to the repository root; the
base data directory can be overridden via the ``SCICON_DATA_DIR`` env var.

Model configs (``model_cfgs``) are loaded from the existing
``scripts/data_preprocessing/atomic_fact_generation/config/`` directory so
there is no duplication.  LLM-judge and question-generator settings are
managed internally by ``data_labeling.make_precision_judge()`` /
``make_recall_judge()`` and ``data_preprocessing.question_generation``
respectively; they are not re-exported from here.

Exported names
--------------
path_cfg      PathConfig             – file-system paths for tracking data
dc_cfg        DataCollectionConfig   – Crossref / Wiley TDM settings
model_cfgs    dict[str, ModelConfig] – per-stage atomic-fact models
query_cfg     QueryBatchConfig       – model/provider for the monthly harness run
hf_cfg        HuggingFaceConfig      – source benchmark + upload target
REPO_ROOT     Path                   – absolute path to the repository root
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from config.config import (
    DataCollectionConfig,
    HuggingFaceConfig,
    HuggingFaceUploadConfig,
    HuggingFaceSourceConfig,
    PathConfig,
    QueryBatchConfig,
)

# Reuse the existing ModelConfig from data_preprocessing — no duplication.
from data_preprocessing.atomic_fact_generation.config.model_config import (
    ModelConfig,
    create_default_configs,
)

_THIS_DIR = Path(__file__).parent
REPO_ROOT = _THIS_DIR.parent.parent  # scicon-track/config/ -> scicon-track/ -> repo root


# ── Path resolution ────────────────────────────────────────────────────────────


def _resolve_paths(cfg: PathConfig) -> PathConfig:
    """Make all PathConfig paths absolute, honouring SCICON_DATA_DIR."""
    base_override = os.environ.get("SCICON_DATA_DIR")
    if base_override:
        base = Path(base_override).expanduser().resolve()
        return PathConfig(
            data_dir=base,
            db_path=base / "sciconbench_track.db",
            pdf_dir=base / "pdfs",
            logs_dir=base / "logs",
        )
    return PathConfig(
        data_dir=REPO_ROOT / cfg.data_dir,
        db_path=REPO_ROOT / cfg.db_path,
        pdf_dir=REPO_ROOT / cfg.pdf_dir,
        logs_dir=REPO_ROOT / cfg.logs_dir,
    )


def _resolve_hf_output(cfg: HuggingFaceConfig) -> HuggingFaceConfig:
    """Make HF output / readme paths absolute."""
    upload = cfg.upload
    return HuggingFaceConfig(
        source=cfg.source,
        upload=HuggingFaceUploadConfig(
            repo_id=upload.repo_id,
            output=REPO_ROOT / upload.output,
            path_in_repo=upload.path_in_repo,
        ),
    )


# ── Load YAML files ────────────────────────────────────────────────────────────

_raw = yaml.safe_load((_THIS_DIR / "config.yaml").read_text())
_query_raw = yaml.safe_load((_THIS_DIR / "query_batch_config.yaml").read_text())
_hf_raw = yaml.safe_load((_THIS_DIR / "hugging_face_config.yaml").read_text())
_judge_raw = yaml.safe_load((_THIS_DIR / "llm_judge_config.yaml").read_text())

# ── Instantiate config objects ─────────────────────────────────────────────────

path_cfg: PathConfig = _resolve_paths(PathConfig(**_raw["paths"]))

dc_cfg: DataCollectionConfig = DataCollectionConfig(**_raw["data_collection"])
if not dc_cfg.wiley_tdm_token:
    dc_cfg = dc_cfg.model_copy(update={"wiley_tdm_token": os.environ.get("WILEY_TDM_TOKEN")})
if not dc_cfg.crossref_mailto:
    dc_cfg = dc_cfg.model_copy(update={"crossref_mailto": os.environ.get("CROSSREF_MAILTO")})

# Load atomic-fact generation model configs from the canonical location in
# data_preprocessing — avoids duplicating the YAML and ModelConfig class.
_atomic_fact_config_path = (
    REPO_ROOT
    / "scripts"
    / "data_preprocessing"
    / "atomic_fact_generation"
    / "config"
    / "model_config.yaml"
)
model_cfgs: dict[str, ModelConfig] = create_default_configs(_atomic_fact_config_path)

query_cfg: QueryBatchConfig = QueryBatchConfig(**_query_raw)

hf_cfg: HuggingFaceConfig = _resolve_hf_output(HuggingFaceConfig(**_hf_raw))

# Raw dict — consumed by the workflow to build judges with the right params.
llm_judge_cfg: dict = _judge_raw

__all__ = [
    "path_cfg",
    "dc_cfg",
    "model_cfgs",
    "query_cfg",
    "hf_cfg",
    "llm_judge_cfg",
    "ModelConfig",
    "REPO_ROOT",
]
