"""Append new benchmark examples to the live hayoungjung/SciConBench dataset.

Each monthly run:
  1. Pulls the current Parquet from the HuggingFace repo.
  2. Fetches new rows from the local SQLite database.
  3. Merges with stale-DOI handling:
       - If a new DOI supersedes an existing one (same base CD number, higher
         .pubN version), the old row is removed and the new one is inserted.
         Removed DOIs are logged to data_track/stale_dois.json.
       - New DOIs absent from HF are appended.
       - Existing DOIs with no superseding update are kept unchanged.
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

import json
import logging
import os
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi

from config import hf_cfg, path_cfg
from db.utils import get_sciconbench_rows
from huggingface.utils import DATASET_COLUMN_ORDER, normalize_atomic_facts_pairs, write_parquet

logger = logging.getLogger(__name__)

# Path to persistent stale-DOI audit log (inside the gitignored data_track/ dir).
_STALE_LOG = Path(__file__).resolve().parent.parent.parent / "data_track" / "stale_dois.json"


def _base_doi(doi: str) -> str:
    """Strip .pubN suffix so different versions map to the same base."""
    return re.sub(r"\.pub\d+$", "", doi.lower())


def _pub_version(doi: str) -> int:
    m = re.search(r"\.pub(\d+)$", doi.lower())
    return int(m.group(1)) if m else 1


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

    def _find_stale(
        self, existing: list[dict], new_rows: list[dict]
    ) -> dict[str, str]:
        """Return {stale_doi: superseding_doi} for existing rows outdated by new_rows."""
        new_by_base: dict[str, tuple[str, int]] = {}
        for r in new_rows:
            base = _base_doi(r["doi"])
            ver  = _pub_version(r["doi"])
            prev = new_by_base.get(base)
            if prev is None or ver > prev[1]:
                new_by_base[base] = (r["doi"], ver)

        stale: dict[str, str] = {}
        for row in existing:
            base = _base_doi(row["doi"])
            if base in new_by_base:
                new_doi, new_ver = new_by_base[base]
                if new_ver > _pub_version(row["doi"]):
                    stale[row["doi"]] = new_doi
        return stale

    def _log_stale(self, stale: dict[str, str]) -> None:
        """Append newly stale DOIs to the persistent audit log."""
        if not stale:
            return
        existing: dict = {}
        if _STALE_LOG.exists():
            try:
                existing = json.loads(_STALE_LOG.read_text())
            except Exception:
                pass
        existing.update(stale)
        _STALE_LOG.parent.mkdir(parents=True, exist_ok=True)
        _STALE_LOG.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
        logger.info("Stale DOIs logged to %s: %s", _STALE_LOG, list(stale))

    def _merge_rows(
        self, existing: list[dict], new_rows: list[dict]
    ) -> list[dict]:
        """Merge existing HF rows with new DB rows, handling stale-DOI replacement.

        Rules:
        * Stale existing rows (superseded by a higher .pubN in new_rows) are dropped
          and their DOIs are written to the stale audit log.
        * Existing rows whose DOI already appears in new_rows are replaced (updated).
        * All other existing rows are kept as-is (backfilled with new columns if absent).
        * Remaining new rows (not already in HF) are appended.
        """
        stale = self._find_stale(existing, new_rows)
        if stale:
            for old, new in stale.items():
                logger.warning("Stale DOI %s superseded by %s — removing.", old, new)
            self._log_stale(stale)

        new_by_doi = {r["doi"]: r for r in new_rows}
        kept: list[dict] = []
        for row in existing:
            if row["doi"] in stale:
                continue   # drop stale
            if row["doi"] in new_by_doi:
                continue   # will be replaced by the incoming version
            kept.append(row)

        to_add = list(new_rows)  # all incoming rows replace-or-append
        logger.info(
            "Existing kept: %d  |  New/updated from DB: %d  |  Stale removed: %d",
            len(kept),
            len(to_add),
            len(stale),
        )
        return kept + to_add

    # ── Public API ─────────────────────────────────────────────────────────────

    def save_to_parquet(self) -> None:
        """Pull HF dataset, merge with DB rows, write merged Parquet locally."""
        existing_rows = self._load_existing_hf_rows()
        db_rows = get_sciconbench_rows()

        # Normalise DB rows to match DATASET_COLUMN_ORDER.
        # Internal fields (objectives, authors_conclusions) are used for
        # preprocessing only and are intentionally excluded from the upload.
        packed_db = [
            {
                "doi":                r["doi"],
                "title":              r.get("title") or "",
                "reference_text":     r.get("reference_text") or "",
                "question":           r.get("question") or "",
                "all_facts":          [str(x) for x in (r.get("all_facts") or [])],
                "atomic_facts_pairs": normalize_atomic_facts_pairs(
                    r.get("atomic_facts_pairs")
                ),
                "publication_date":   r.get("publication_date") or "",
                "total_atomic_facts": int(r.get("total_atomic_facts") or 0),
                "review_type":        r.get("review_type") or "",
                "new_search":         bool(r.get("new_search")),
                "conclusion_changed": bool(r.get("conclusion_changed")),
                "citations":          r.get("citations") or "",
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
