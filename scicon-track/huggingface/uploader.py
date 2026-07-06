"""Append new benchmark examples to the live hayoungjung/SciConBench dataset.

Each monthly run:
  1. Pulls the current Parquet from the HuggingFace repo.
  2. Fetches new rows from the local SQLite database.
  3. Merges: existing rows are kept; new DOIs are appended; duplicate DOIs
     (already in HF) are skipped so the core benchmark is never overwritten.
  4. Writes the merged Parquet locally and pushes it back to the same repo.

Usage
-----
    from huggingface.uploader import SciConBenchUploader

    uploader = SciConBenchUploader()
    uploader.save_to_parquet()   # merge + write local Parquet
    uploader.upload()            # push to HuggingFace Hub

Requires:
    pip install huggingface_hub pyarrow pandas datasets
    HF_TOKEN env var set to a write token from https://huggingface.co/settings/tokens
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi

from config import hf_cfg
from db.utils import get_sciconbench_rows
from huggingface.utils import DATASET_COLUMN_ORDER, normalize_atomic_facts_pairs, write_parquet

logger = logging.getLogger(__name__)


class SciConBenchUploader:
    """Append new monthly examples to the live SciConBench HuggingFace dataset.

    Existing rows (core benchmark) are never modified — only new DOIs absent
    from the current HF dataset are appended.
    """

    def __init__(
        self,
        repo_id: str | None = None,
        output: Path | None = None,
        path_in_repo: str | None = None,
    ) -> None:
        upload = hf_cfg.upload
        self.repo_id = repo_id or upload.repo_id
        self.output = Path(output or upload.output)
        self.path_in_repo = path_in_repo or upload.path_in_repo
        self.token = os.getenv("HF_TOKEN")

        src = hf_cfg.source
        self._src_repo_id = src.repo_id
        self._src_config = src.config
        self._src_split = src.split

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_existing_hf_rows(self) -> list[dict]:
        """Download the current live dataset from HuggingFace as a list of dicts."""
        logger.info("Loading existing dataset from %s …", self._src_repo_id)
        ds = load_dataset(
            self._src_repo_id,
            self._src_config,
            split=self._src_split,
            token=self.token,
        )
        return [dict(row) for row in ds]

    def _merge_rows(
        self, existing: list[dict], new_rows: list[dict]
    ) -> list[dict]:
        """Return existing rows + any new_rows whose DOI is not already present."""
        existing_dois = {row["doi"] for row in existing}
        appended = [r for r in new_rows if r["doi"] not in existing_dois]
        logger.info(
            "Existing rows: %d  |  New from DB: %d  |  To append: %d",
            len(existing),
            len(new_rows),
            len(appended),
        )
        return existing + appended

    # ── Public API ─────────────────────────────────────────────────────────────

    def save_to_parquet(self) -> None:
        """Pull HF dataset, merge with DB rows, write merged Parquet locally."""
        existing_rows = self._load_existing_hf_rows()
        db_rows = get_sciconbench_rows()

        # Normalise DB rows to match DATASET_COLUMN_ORDER
        packed_db = [
            {
                "doi": r["doi"],
                "title": r.get("title") or "",
                "reference_text": r.get("reference_text") or "",
                "question": r.get("question") or "",
                "all_facts": [str(x) for x in (r.get("all_facts") or [])],
                "atomic_facts_pairs": normalize_atomic_facts_pairs(
                    r.get("atomic_facts_pairs")
                ),
                "publication_date": r.get("publication_date") or "",
                "total_atomic_facts": r.get("total_atomic_facts") or 0,
                "review_type": r.get("review_type") or "",
                "new_search": bool(r.get("new_search")),
                "conclusion_changed": bool(r.get("conclusion_changed")),
                "citations": r.get("citations") or "",
            }
            for r in db_rows
        ]

        merged = self._merge_rows(existing_rows, packed_db)

        self.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(merged, columns=list(DATASET_COLUMN_ORDER)).to_parquet(
            self.output, index=False, compression="snappy", engine="pyarrow"
        )
        logger.info("Merged Parquet written: %s  (%d rows)", self.output, len(merged))

    def upload(self) -> str:
        """Push the merged local Parquet back to HuggingFace.

        The dataset README is intentionally left untouched.
        Returns the URL of the uploaded Parquet file.
        """
        if not self.token:
            raise RuntimeError(
                "HF_TOKEN environment variable is not set. "
                "Get a write token from https://huggingface.co/settings/tokens"
            )

        api = HfApi(token=self.token)

        url = api.upload_file(
            path_or_fileobj=str(self.output),
            path_in_repo=self.path_in_repo,
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message="Monthly update: append new Cochrane reviews",
            token=self.token,
        )
        logger.info("Uploaded: %s → %s/%s", self.output, self.repo_id, self.path_in_repo)
        return url
