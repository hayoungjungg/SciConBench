"""Local JSON caches derived from the SciConBench HuggingFace dataset.

``sciconharness``'s ``CochraneResultFilter`` needs three pieces of metadata
about *every* Cochrane review in the benchmark (not just the one DOI being
queried) to filter search results properly:

- a flat list of every review title (``title_filter_list``), so results that
  are mirrors/reproductions of *any* known review get caught even when their
  URL/title doesn't literally contain the word "Cochrane";
- a ``{doi: title}`` mapping, so the ``source_title`` for the review actually
  being queried can be resolved automatically instead of requiring every
  caller to pass it explicitly;
- a ``{doi: publication_date}`` mapping, for the same reason.

Previously these had to be supplied manually (``--cochrane-titles`` /
``doi_to_title=`` / an explicit ``--publication-date``).
This module makes the live ``hayoungjung/SciConBench`` dataset the
single source of truth for all three, cached locally so normal runs never
need a network call or the ``datasets`` package after the first one:

    from sciconharness.utils.hf_benchmark_cache import (
        load_cochrane_titles_cached,
        load_doi_to_title_cached,
        load_doi_to_publication_date_cached,
    )

    titles          = load_cochrane_titles_cached()           # list[str], lowercased
    doi_to_title    = load_doi_to_title_cached()               # dict[str, str], original casing
    doi_to_pubdate  = load_doi_to_publication_date_cached()    # dict[str, str]

Titles in ``cochrane_titles.json`` are standardized to lowercase + collapsed
whitespace (see ``_standardize_title``) since it's a blanket blocklist, not
displayed anywhere — ``doi_to_title.json`` keeps the dataset's original
casing since it's used for human-readable ``source_title`` logging.

The first call that finds no cache on disk pulls the full dataset from
HuggingFace once (``datasets.load_dataset(...)``) and writes all three JSON
files under ``data_track/hf_benchmark_cache/`` (override with
``SCICONBENCH_CACHE_DIR``); every subsequent call — in this process or a
later one — just reads JSON off disk.

Refreshing
----------
``scicon-track``'s HuggingFace uploader (``huggingface/uploader.py``) calls
``build_hf_benchmark_cache(force=True, rows=<already-merged rows>)`` right
after merging and before pushing to HuggingFace on every monthly run, so the
cache is regenerated from the *exact* rows about to be published — no extra
network pull needed, and the cache never drifts more than one dashboard run
behind the live dataset. You can also run this module directly to force a
refresh from the live HF dataset (a plain pull, not the uploader's in-memory
merge):

    python -m sciconharness.utils.hf_benchmark_cache --force
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

HF_REPO_ID = "hayoungjung/SciConBench"
HF_CONFIG = "benchmark"
HF_SPLIT = "test"

# sciconharness/utils/hf_benchmark_cache.py -> sciconharness/ -> repo root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _default_cache_dir() -> Path:
    override = os.getenv("SCICONBENCH_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return _PROJECT_ROOT / "data_track" / "hf_benchmark_cache"


CACHE_DIR = _default_cache_dir()
TITLES_FILE = CACHE_DIR / "cochrane_titles.json"
DOI_TO_TITLE_FILE = CACHE_DIR / "doi_to_title.json"
DOI_TO_PUBDATE_FILE = CACHE_DIR / "doi_to_publication_date.json"


def _rows_from_hf() -> List[Dict[str, Any]]:
    """Pull the full benchmark dataset from HuggingFace as a list of dicts."""
    from datasets import load_dataset  # optional dep — only needed on cache miss/refresh

    logger.info(
        "Pulling %s (%s/%s) from HuggingFace to build local benchmark cache…",
        HF_REPO_ID, HF_CONFIG, HF_SPLIT,
    )
    ds = load_dataset(
        HF_REPO_ID,
        HF_CONFIG,
        split=HF_SPLIT,
        token=os.getenv("HF_TOKEN"),
        # The uploader appends rows to the Parquet shard without touching the
        # dataset card's recorded split sizes (by design — see
        # scicon-track/huggingface/uploader.py), so the "expected" size in
        # HF's cached metadata is stale immediately after every monthly
        # push. Skip that check rather than failing to load valid data.
        verification_mode="no_checks",
    )
    return [dict(row) for row in ds]


def _cache_complete() -> bool:
    return TITLES_FILE.exists() and DOI_TO_TITLE_FILE.exists() and DOI_TO_PUBDATE_FILE.exists()


def _standardize_publication_date(doi: str, pubdate: str) -> Optional[str]:
    """Normalize a publication date to the canonical ``"D Month YYYY"`` string
    (e.g. ``"13 June 2012"``) that ``CochraneResultFilter``/``harness.py``
    use everywhere, by round-tripping it through the *exact same* parser
    (``cochrane.py::_parse_date``) that later applies the cutoff comparison.

    This guarantees every date written to ``doi_to_publication_date.json``
    is guaranteed-parseable by the filter, regardless of which of
    ``_parse_date``'s several accepted input formats the raw dataset value
    happened to use. Returns ``None`` (and logs a warning) for a date that
    doesn't parse at all, rather than caching a value that would silently
    fail to filter anything at query time.
    """
    from sciconharness.mcp_client.filters.cochrane import _parse_date

    parsed = _parse_date(pubdate)
    if parsed is None:
        logger.warning(
            "Skipping unparseable publication_date for %s: %r — this DOI's "
            "results won't get a date cutoff unless publication_date is "
            "passed explicitly.", doi, pubdate,
        )
        return None
    return parsed.strftime("%d %B %Y")


def standardize_title(title: str) -> str:
    """Lowercase + collapse whitespace for the flat title list.

    ``cochrane_titles.json`` is matched against tool-result titles for exact
    (case-insensitive) blocking, so a single normalized casing/whitespace
    convention avoids near-duplicate entries differing only in Title Case vs
    lowercase. Matches the lowercase style of the benchmark's own title
    examples (e.g. ``"abdominal decompression for suspected fetal
    compromise/pre‐eclampsia"``). Does not touch Unicode hyphen variants —
    those are preserved verbatim, exactly as they appear in the dataset.

    Public (not module-private) because callers that append titles to an
    already-cached list — e.g. ``scicon-track``'s ``task_run_queries``
    appending rolling reviews not yet in the HF dataset — should apply the
    exact same standardization so the combined list stays consistent.
    """
    return " ".join(title.strip().lower().split())


def build_hf_benchmark_cache(
    force: bool = False,
    rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Path]:
    """(Re)build the three cache files from SciConBench dataset rows.

    Parameters
    ----------
    force : bool
        If False (default) and all three cache files already exist on disk,
        this is a no-op (no network call). If True, always regenerates —
        pulling fresh rows from HuggingFace unless ``rows`` is supplied.
    rows : sequence of dict, optional
        Pre-loaded dataset rows, each with at least ``doi``/``title``/
        ``publication_date`` keys (e.g. ``scicon-track``'s already-merged
        in-memory rows right before a HuggingFace push). Skips the network
        pull entirely when provided.

    Returns
    -------
    dict with keys ``"titles"``, ``"doi_to_title"``, ``"doi_to_publication_date"``
    mapping to their on-disk :class:`~pathlib.Path`.
    """
    paths = {
        "titles": TITLES_FILE,
        "doi_to_title": DOI_TO_TITLE_FILE,
        "doi_to_publication_date": DOI_TO_PUBDATE_FILE,
    }
    if not force and _cache_complete():
        logger.debug("HF benchmark cache already present at %s; skipping rebuild.", CACHE_DIR)
        return paths

    if rows is None:
        rows = _rows_from_hf()

    titles: List[str] = []
    doi_to_title: Dict[str, str] = {}
    doi_to_pubdate: Dict[str, str] = {}
    for row in rows:
        doi = row.get("doi")
        title = row.get("title")
        pubdate = row.get("publication_date")
        if title:
            titles.append(standardize_title(title))
        # doi_to_title keeps the dataset's original casing (used for
        # human-readable source-title logging); only the flat titles list
        # used for blanket title-list blocking is standardized to lowercase.
        if doi and title:
            doi_to_title[doi] = title
        if doi and pubdate:
            normalized_pubdate = _standardize_publication_date(doi, pubdate)
            if normalized_pubdate:
                doi_to_pubdate[doi] = normalized_pubdate

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TITLES_FILE.write_text(json.dumps(titles, indent=2, ensure_ascii=False), encoding="utf-8")
    DOI_TO_TITLE_FILE.write_text(json.dumps(doi_to_title, indent=2, ensure_ascii=False), encoding="utf-8")
    DOI_TO_PUBDATE_FILE.write_text(json.dumps(doi_to_pubdate, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "Built HF benchmark cache at %s: %d titles, %d doi->title, %d doi->publication_date",
        CACHE_DIR, len(titles), len(doi_to_title), len(doi_to_pubdate),
    )
    return paths


def load_cochrane_titles_cached() -> List[str]:
    """Return the flat list of every review title, building the cache first if needed."""
    build_hf_benchmark_cache(force=False)
    return json.loads(TITLES_FILE.read_text(encoding="utf-8"))


def load_doi_to_title_cached() -> Dict[str, str]:
    """Return ``{doi: title}``, building the cache first if needed."""
    build_hf_benchmark_cache(force=False)
    return json.loads(DOI_TO_TITLE_FILE.read_text(encoding="utf-8"))


def load_doi_to_publication_date_cached() -> Dict[str, str]:
    """Return ``{doi: publication_date}``, building the cache first if needed."""
    build_hf_benchmark_cache(force=False)
    return json.loads(DOI_TO_PUBDATE_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="Force a fresh pull from HuggingFace even if a cache already exists.",
    )
    args = parser.parse_args()
    result = build_hf_benchmark_cache(force=args.force)
    for name, path in result.items():
        print(f"{name}: {path}")
