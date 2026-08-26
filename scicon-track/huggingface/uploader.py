"""Append new benchmark examples to the live hayoungjung/SciConBench dataset.

Each monthly run:
  1. Pulls the current Parquet from the HuggingFace repo.
  2. Fetches new rows from the local SQLite database (closed months only
     for production uploads — open-month rolling reviews stay local).
  3. Merges with stale-DOI handling:
       - If a new DOI supersedes an existing one (same base CD number, higher
         .pubN version), the old row is removed and the new one is inserted.
         Removed DOIs are logged to data_track/stale_dois.json.
       - New DOIs absent from HF are appended.
       - Existing DOIs with no superseding update are kept unchanged.
  4. Writes the merged Parquet locally.
  5. Refreshes sciconharness's local title/DOI-mapping filter caches from
     those same merged rows (see ``refresh_filter_caches()``), so the caches
     ``CochraneResultFilter`` reads from never lag behind what's about to be
     published.
  6. Pushes the merged Parquet back to the same repo.
  7. (Production only) Updates the Hub dataset README with a newest-first
     monthly changelog entry in a second commit.

Usage
-----
    from huggingface.uploader import SciConBenchUploader

    uploader = SciConBenchUploader()
    uploader.save_to_parquet(closed_month="2026-07")
    uploader.refresh_filter_caches()
    uploader.upload()
    uploader.update_hub_readme(closed_month="2026-07")

Requires:
    pip install huggingface_hub pyarrow pandas datasets
    HF_TOKEN env var set to a write token from https://huggingface.co/settings/tokens
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi

from config import hf_cfg
from db.utils import get_sciconbench_rows
from huggingface.utils import DATASET_COLUMN_ORDER, normalize_atomic_facts_pairs

logger = logging.getLogger(__name__)

# Path to persistent stale-DOI audit log (inside the gitignored data_track/ dir).
_STALE_LOG = Path(__file__).resolve().parent.parent.parent / "data_track" / "stale_dois.json"

# Leading Hub README changelog lines, e.g.
#   > Latest update — **July 2026**. SciConBench is a *live* …
_UPDATE_LINE_RE = re.compile(
    r"^> (?:Latest )?update — \*\*[^*]+\*\*\..*$",
    re.IGNORECASE,
)
_PREVIOUS_ITEM_RE = re.compile(
    r"^- \*\*[^*]+\*\* — .+$",
)
_PREVIOUS_UPDATES_HEADING_RE = re.compile(
    r"^#{1,6}\s+Previous updates\s*$",
    re.IGNORECASE,
)
_DETAILS_OPEN_RE = re.compile(r"^<details>\s*$", re.IGNORECASE)
_DETAILS_CLOSE_RE = re.compile(r"^</details>\s*$", re.IGNORECASE)
_PREVIOUS_SUMMARY_RE = re.compile(
    r"^<summary>\s*Previous updates\s*</summary>\s*$",
    re.IGNORECASE,
)


def _base_doi(doi: str) -> str:
    """Strip .pubN suffix so different versions map to the same base."""
    return re.sub(r"\.pub\d+$", "", doi.lower())


def _pub_version(doi: str) -> int:
    m = re.search(r"\.pub(\d+)$", doi.lower())
    return int(m.group(1)) if m else 1


def _format_sample_count(n: int) -> str:
    """Format a question-row count like the Hub card (e.g. 9110 → ``9.11K``)."""
    return f"{n / 1000:.2f}K"


def _month_label(year_month: str) -> str:
    """``2026-07`` → ``July 2026``."""
    return datetime.strptime(year_month, "%Y-%m").strftime("%B %Y")


def _build_update_line(
    *,
    latest: bool,
    month_label: str,
    before: int,
    after: int,
) -> str:
    """Build a Latest blockquote or a Previous-updates list item."""
    before_s = _format_sample_count(before)
    after_s = _format_sample_count(after)
    if latest:
        return (
            f"> Latest update — **{month_label}**. SciConBench is a *live* "
            f"benchmark updated monthly with newly released CDSR reviews. "
            f"This release updates the benchmark from {before_s} to {after_s} "
            f"samples and replaces superseded reviews with their latest "
            f"editions through **{month_label}**."
        )
    return (
        f"- **{month_label}** — from {before_s} to {after_s} samples; "
        f"superseded reviews replaced through {month_label}."
    )


def _month_from_update_line(line: str) -> str | None:
    m = re.search(r"\*\*([^*]+)\*\*", line)
    return m.group(1) if m else None


def _to_previous_update_item(line: str) -> str:
    """Normalize a Latest/Update blockquote (or list item) for Previous updates."""
    stripped = line.strip()
    if _PREVIOUS_ITEM_RE.match(stripped):
        return stripped

    month = _month_from_update_line(stripped)
    counts = re.search(
        r"from\s+(\d+\.\d+K)\s+to\s+(\d+\.\d+K)",
        stripped,
        flags=re.IGNORECASE,
    )
    if month and counts:
        return (
            f"- **{month}** — from {counts.group(1)} to {counts.group(2)} "
            f"samples; superseded reviews replaced through {month}."
        )
    if month:
        rest = re.sub(
            r"^>\s*(?:Latest )?update — \*\*[^*]+\*\*\.\s*",
            "",
            stripped,
            flags=re.IGNORECASE,
        )
        return f"- **{month}** — {rest}"
    return f"- {stripped.lstrip('> ').lstrip('- ')}"


def _consume_previous_entries(lines: list[str], start: int) -> tuple[list[str], int]:
    """Collect previous-update entries from a heading or ``<details>`` block."""
    updates: list[str] = []
    i = start
    if i >= len(lines):
        return updates, i

    # Collapsible: <details> … </details>
    if _DETAILS_OPEN_RE.match(lines[i]):
        i += 1
        while i < len(lines) and not _DETAILS_CLOSE_RE.match(lines[i]):
            if (
                not lines[i].strip()
                or _PREVIOUS_SUMMARY_RE.match(lines[i])
            ):
                i += 1
                continue
            if _UPDATE_LINE_RE.match(lines[i]) or _PREVIOUS_ITEM_RE.match(lines[i]):
                updates.append(_to_previous_update_item(lines[i]))
            i += 1
        if i < len(lines) and _DETAILS_CLOSE_RE.match(lines[i]):
            i += 1
        return updates, i

    # Legacy: ## Previous updates + blockquotes/list items
    if _PREVIOUS_UPDATES_HEADING_RE.match(lines[i]):
        i += 1
        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue
            if _UPDATE_LINE_RE.match(lines[i]) or _PREVIOUS_ITEM_RE.match(lines[i]):
                updates.append(_to_previous_update_item(lines[i]))
                i += 1
                continue
            break
    return updates, i


def _split_update_preamble(text: str) -> tuple[list[str], str]:
    """Split leading latest + previous-updates changelog from the card body."""
    lines = (text or "").splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1

    updates: list[str] = []
    # Optional leading "Latest update" / loose update blockquotes.
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        if _UPDATE_LINE_RE.match(lines[i]):
            updates.append(lines[i])
            i += 1
            continue
        break

    prior, i = _consume_previous_entries(lines, i)
    updates.extend(prior)

    rest = "\n".join(lines[i:])
    return updates, rest


def _render_changelog(latest_line: str, prior_lines: list[str], rest: str) -> str:
    """Render latest blurb, then a collapsible Previous updates block, then body."""
    parts: list[str] = [latest_line]
    if prior_lines:
        items = "\n".join(prior_lines)
        prior_block = (
            "<details>\n"
            "<summary>Previous updates</summary>\n\n"
            f"{items}\n\n"
            "</details>"
        )
        parts.append(prior_block)
    if rest.strip():
        parts.append(rest.lstrip("\n"))
    return "\n\n".join(parts) + "\n"


def _eligible_for_closed_month(row: dict, closed_month: str) -> bool:
    """Core always publishes; rolling only when ``cohort_month <= closed_month``."""
    if row.get("panel_type") == "core":
        return True
    cohort = row.get("cohort_month")
    return bool(cohort and cohort <= closed_month)


class SciConBenchUploader:
    """Append new monthly examples to the live SciConBench HuggingFace dataset.

    Existing rows are kept unless superseded by a newer ``.pubN`` or present in
    the stale-DOI audit log. Production uploads only publish closed-month
    rolling rows (plus core).
    """

    def __init__(
        self,
        repo_id: str | None = None,
        output: Path | None = None,
        path_in_repo: str | None = None,
        source_config: str | None = None,
        allow_missing_source: bool = False,
        include_dois: list[str] | None = None,
    ) -> None:
        upload = hf_cfg.upload
        self.repo_id = repo_id or upload.repo_id
        self.output = Path(output or upload.output)
        self.path_in_repo = path_in_repo or upload.path_in_repo
        self.token = os.getenv("HF_TOKEN")
        self.allow_missing_source = allow_missing_source
        self.include_dois = set(include_dois) if include_dois is not None else None

        src = hf_cfg.source
        self._src_repo_id = src.repo_id
        self._src_config = source_config or src.config
        self._src_split = src.split

        # Populated by save_to_parquet(); reused by refresh_filter_caches()
        # and update_hub_readme().
        self._merged_rows: list[dict] | None = None
        self._existing_row_count: int | None = None
        self._merged_row_count: int | None = None

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_existing_hf_rows(self) -> list[dict]:
        """Download the current live dataset from HuggingFace as a list of dicts."""
        logger.info(
            "Loading existing dataset from %s (%s/%s) …",
            self._src_repo_id, self._src_config, self._src_split,
        )
        try:
            ds = load_dataset(
                self._src_repo_id,
                self._src_config,
                split=self._src_split,
                token=self.token,
                # Appended rows change the Parquet shard's actual size without
                # updating the dataset card's recorded split sizes (by design —
                # we intentionally leave YAML sizes untouched), so HF's cached
                # metadata is stale immediately after every monthly push. Skip
                # that check rather than failing to load valid data.
                verification_mode="no_checks",
            )
        except Exception as exc:
            if self.allow_missing_source:
                logger.warning(
                    "No existing HuggingFace config %s/%s — starting empty (%s).",
                    self._src_repo_id, self._src_config, exc,
                )
                return []
            raise
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

        # Also drop DOIs previously logged as superseded/withdrawn (e.g. a
        # core/rolling prune that hasn't produced a replacement row yet).
        logged_stale: dict = {}
        if _STALE_LOG.exists():
            try:
                logged_stale = json.loads(_STALE_LOG.read_text())
            except Exception:
                logged_stale = {}

        new_by_doi = {r["doi"]: r for r in new_rows}
        kept: list[dict] = []
        for row in existing:
            if row["doi"] in stale or row["doi"] in logged_stale:
                continue   # drop stale / withdrawn
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

    def save_to_parquet(self, closed_month: str | None = None) -> None:
        """Pull HF dataset, merge with DB rows, write merged Parquet locally.

        When *closed_month* is set (production), only core rows and rolling
        rows with ``cohort_month <= closed_month`` are merged in from the DB.
        Open-month rolling reviews remain local until that month closes.
        """
        existing_rows = self._load_existing_hf_rows()
        self._existing_row_count = len(existing_rows)
        db_rows = get_sciconbench_rows()

        # Normalise DB rows to match DATASET_COLUMN_ORDER.
        # Internal fields (objectives, authors_conclusions) are used for
        # preprocessing only and are intentionally excluded from the upload.
        if self.include_dois is not None:
            db_rows = [r for r in db_rows if r.get("doi") in self.include_dois]

        if closed_month is not None:
            before_filter = len(db_rows)
            db_rows = [
                r for r in db_rows if _eligible_for_closed_month(r, closed_month)
            ]
            logger.info(
                "Closed-month filter %s: %d → %d DB row(s) eligible for upload",
                closed_month,
                before_filter,
                len(db_rows),
            )

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
        self._merged_rows = merged
        self._merged_row_count = len(merged)

        self.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(merged, columns=list(DATASET_COLUMN_ORDER)).to_parquet(
            self.output, index=False, compression="snappy", engine="pyarrow"
        )
        logger.info("Merged Parquet written: %s  (%d rows)", self.output, len(merged))

    def refresh_filter_caches(self) -> dict:
        """Regenerate sciconharness's local Cochrane-filter caches.

        ``sciconharness.mcp_client.filters.CochraneResultFilter`` relies on a
        local JSON cache of every review title plus DOI→title / DOI→
        publication_date mappings (see
        ``sciconharness.utils.hf_benchmark_cache``) so it doesn't need a
        HuggingFace pull on every harness run. Call this right after
        ``save_to_parquet()`` (and before ``upload()``) so those caches are
        rebuilt from the *exact* rows about to be published — no extra
        network pull needed, since we already have them in memory.

        Returns the dict of cache-file paths from ``build_hf_benchmark_cache``.
        """
        if self._merged_rows is None:
            raise RuntimeError(
                "refresh_filter_caches() requires save_to_parquet() to have "
                "been called first (no merged rows to build the cache from)."
            )
        from sciconharness.utils.hf_benchmark_cache import build_hf_benchmark_cache

        paths = build_hf_benchmark_cache(force=True, rows=self._merged_rows)
        logger.info("Refreshed sciconharness filter caches: %s", paths)
        return paths

    def ensure_hub_config(self) -> None:
        """Add this uploader's dataset config to the Hub README if missing.

        Production monthly uploads leave YAML configs alone and instead call
        :meth:`update_hub_readme` for the prose changelog. Practice-run
        configs (e.g. ``trial``) need an entry so
        ``load_dataset(repo, "trial")`` works after the first parquet push.
        """
        from huggingface_hub import DatasetCard

        if not self.token:
            raise RuntimeError(
                "HF_TOKEN environment variable is not set. "
                "Get a write token from https://huggingface.co/settings/tokens"
            )

        card = DatasetCard.load(self.repo_id, repo_type="dataset", token=self.token)
        configs = list(card.data.configs or [])
        already = {
            (c.get("config_name") if isinstance(c, dict) else getattr(c, "config_name", None))
            for c in configs
        }
        if self._src_config in already:
            logger.info("HuggingFace README already has config %s.", self._src_config)
            return

        glob_path = str(Path(self.path_in_repo).parent / "test-*")
        configs.append({
            "config_name": self._src_config,
            "data_files": [{"split": self._src_split, "path": glob_path}],
        })
        card.data.configs = configs
        card.push_to_hub(
            self.repo_id,
            repo_type="dataset",
            token=self.token,
            commit_message=f"Add HuggingFace config {self._src_config}",
        )
        logger.info(
            "Added HuggingFace config %s → %s", self._src_config, glob_path,
        )

    def update_hub_readme(self, closed_month: str) -> None:
        """Prepend a newest-first monthly changelog entry on the Hub README.

        Production only. Runs as a *second* commit after :meth:`upload`.
        The newest month stays as a top-level ``Latest update`` blurb; older
        months move under a collapsible ``Previous updates`` ``<details>``
        block as compact list items. Re-running the same ``closed_month``
        replaces that month's entry.
        """
        from huggingface_hub import DatasetCard

        if not self.token:
            raise RuntimeError(
                "HF_TOKEN environment variable is not set. "
                "Get a write token from https://huggingface.co/settings/tokens"
            )
        if self._existing_row_count is None or self._merged_row_count is None:
            raise RuntimeError(
                "update_hub_readme() requires save_to_parquet() first "
                "(no before/after row counts)."
            )

        month_label = _month_label(closed_month)
        new_line = _build_update_line(
            latest=True,
            month_label=month_label,
            before=self._existing_row_count,
            after=self._merged_row_count,
        )

        card = DatasetCard.load(self.repo_id, repo_type="dataset", token=self.token)
        prior_updates, rest = _split_update_preamble(card.text or "")

        # Drop a prior entry for this same month (idempotent re-runs), then
        # fold any remaining Latest/Update blurbs into Previous updates.
        kept = [
            _to_previous_update_item(line)
            for line in prior_updates
            if _month_from_update_line(line) != month_label
        ]
        card.text = _render_changelog(new_line, kept, rest)

        card.push_to_hub(
            self.repo_id,
            repo_type="dataset",
            token=self.token,
            commit_message=f"Update dataset card: {month_label} release notes",
        )
        logger.info(
            "Hub README updated for %s (%s → %s samples).",
            month_label,
            _format_sample_count(self._existing_row_count),
            _format_sample_count(self._merged_row_count),
        )

    def upload(self, commit_message: str | None = None) -> str:
        """Push the merged local Parquet back to HuggingFace.

        Does not modify the dataset README — call :meth:`update_hub_readme`
        (production) or :meth:`ensure_hub_config` (trial) afterwards.
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
            commit_message=commit_message or "Monthly update: append new Cochrane reviews",
            token=self.token,
        )
        logger.info("Uploaded: %s → %s/%s", self.output, self.repo_id, self.path_in_repo)
        return url
