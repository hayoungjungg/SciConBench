"""Utilities for serializing SciConBench-Track rows to Parquet for HuggingFace upload."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

# Canonical column order — must remain stable across monthly releases.
# Matches exactly the live HuggingFace benchmark parquet schema.
# Internal-only fields (objectives, authors_conclusions) are intentionally
# excluded; they live in the DB for preprocessing but are not published.
DATASET_COLUMN_ORDER = (
    "doi",
    "title",
    "reference_text",
    "question",
    "all_facts",
    "atomic_facts_pairs",
    "publication_date",
    "total_atomic_facts",
    "review_type",
    "new_search",
    "conclusion_changed",
    "citations",
)


def normalize_atomic_facts_pairs(pairs: Any) -> list[dict[str, Any]]:
    """Convert raw atomic-facts pairs (list of [sentence, [fact, ...]]) to dicts.

    Each output dict has keys ``sentence`` (str) and ``atomic_facts`` (list[str]).
    """
    out: list[dict[str, Any]] = []
    if not pairs:
        return out
    for p in pairs:
        if isinstance(p, dict):
            # Already normalized (e.g. loaded back from Parquet/JSON)
            sentence = str(p.get("sentence", ""))
            atoms    = p.get("atomic_facts") or []
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            sentence, atoms = p[0], p[1]
        else:
            continue
        if not isinstance(atoms, list):
            atoms = [atoms]
        # Key order matches the live HuggingFace dataset: atomic_facts first.
        out.append({"atomic_facts": [str(x) for x in atoms], "sentence": str(sentence)})
    return out


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    """Serialize benchmark rows to a Snappy-compressed Parquet file at *path*.

    Columns are reordered to :data:`DATASET_COLUMN_ORDER` and atomic-facts pairs
    are normalized before writing.  Parent directories are created if absent.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    packed = [
        {
            "doi":                r["doi"],
            "title":              r.get("title") or "",
            "reference_text":     r.get("reference_text") or "",
            "question":           r.get("question") or "",
            "all_facts":          [str(x) for x in (r.get("all_facts") or [])],
            "atomic_facts_pairs": normalize_atomic_facts_pairs(r.get("atomic_facts_pairs")),
            "publication_date":   r.get("publication_date") or "",
            "total_atomic_facts": int(r.get("total_atomic_facts") or 0),
            "review_type":        r.get("review_type") or "",
            "new_search":         bool(r.get("new_search")),
            "conclusion_changed": bool(r.get("conclusion_changed")),
            "citations":          r.get("citations") or "",
        }
        for r in rows
    ]

    pd.DataFrame(packed, columns=list(DATASET_COLUMN_ORDER)).to_parquet(
        path, index=False, compression="snappy", engine="pyarrow"
    )
