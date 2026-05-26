#!/usr/bin/env python3
"""
Compute macro (per-DOI) recall/precision/F1 metrics for all labeled-fact models.

Reads:
  - recall/recall_*.jsonl
  - precision/precision_*.jsonl

Outputs:
  - metrics_macro_per_doi_by_model.csv
  - metrics_macro_variance_by_model.csv
  - metrics_macro_variance_grouped_by_family.csv

Also prints a grouped terminal table:
  family -> variant (base/tools/tools_filter/filter)
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SPECIAL_TABLE_FAMILIES = {"google_ai_overview", "google_ai_mode", "openevidence"}


def f1_score(precision: float, recall: float) -> float:
    return 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)


def precision_fancy(supported: int, contradicted: int, not_supported: int, total: int) -> float:
    del not_supported
    if total == 0:
        return 0.0
    support_rate = supported / total
    no_contradiction = (total - contradicted) / total
    return support_rate * no_contradiction


def parse_model_variant(model_name: str) -> Tuple[str, str]:
    if model_name.endswith("_tools_filter"):
        return model_name[: -len("_tools_filter")], "tools_filter"
    if model_name.endswith("_tools"):
        return model_name[: -len("_tools")], "tools"
    if model_name.endswith("_filter"):
        return model_name[: -len("_filter")], "filter"
    return model_name, "base"


def load_recall_counts(path: Path) -> Dict[str, Dict[str, int]]:
    per_doi: Dict[str, Dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            doi = row.get("doi") or row.get("doi_key_preprocessed")
            if not doi:
                continue
            label = (row.get("llm_label") or "").strip().upper()
            bucket = per_doi.setdefault(doi, {"supported": 0, "total": 0})
            bucket["total"] += 1
            if label == "SUPPORTED":
                bucket["supported"] += 1
    return per_doi


def load_precision_counts(path: Path) -> Dict[str, Dict[str, int]]:
    per_doi: Dict[str, Dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            doi = row.get("doi") or row.get("doi_key_preprocessed")
            if not doi:
                continue
            label = (row.get("llm_label") or "").strip().upper()
            bucket = per_doi.setdefault(doi, {"supported": 0, "contradicted": 0, "not_supported": 0, "total": 0})
            bucket["total"] += 1
            if label == "SUPPORTED":
                bucket["supported"] += 1
            elif label == "CONTRADICTED":
                bucket["contradicted"] += 1
            else:
                bucket["not_supported"] += 1
    return per_doi


def safe_mean(values: List[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def safe_variance(values: List[float]) -> float:
    return statistics.variance(values) if len(values) > 1 else 0.0


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def format_metric(mean_v: float, var_v: float) -> str:
    return f"{mean_v:.4f} ({var_v:.4f})"


def _print_table(summary_rows: List[Dict[str, object]], title: str | None = None) -> None:
    variant_order = {"base": 0, "tools": 1, "tools_filter": 2, "filter": 3}
    sorted_rows = sorted(
        summary_rows,
        key=lambda r: (
            str(r["family"]),
            variant_order.get(str(r["variant"]), 99),
            str(r["model"]),
        ),
    )

    headers = [
        "Family",
        "Variant",
        "N",
        "Recall mean(var)",
        "Precision mean(var)",
        "F1 mean(var)",
    ]
    data_rows: List[List[str]] = []
    for row in sorted_rows:
        data_rows.append(
            [
                str(row["family"]),
                str(row["variant"]),
                str(row["n_dois"]),
                format_metric(float(row["macro_recall_mean"]), float(row["macro_recall_variance"])),
                format_metric(float(row["macro_precision_mean"]), float(row["macro_precision_variance"])),
                format_metric(float(row["macro_f1_mean"]), float(row["macro_f1_variance"])),
            ]
        )

    widths = [len(h) for h in headers]
    for row in data_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(sep: str = "-") -> str:
        return "+" + "+".join(sep * (w + 2) for w in widths) + "+"

    def render_row(cells: List[str]) -> str:
        return "| " + " | ".join(cells[i].ljust(widths[i]) for i in range(len(cells))) + " |"

    if title:
        print(title)
    print(line("="))
    print(render_row(headers))
    print(line("="))

    prev_family = None
    for row in data_rows:
        if prev_family is not None and row[0] != prev_family:
            print(line("-"))
        print(render_row(row))
        prev_family = row[0]
    print(line("="))


DEFAULT_BASE_DIR = Path(__file__).resolve().parents[2] / "data" / "labeled_facts"


def compute_metrics(base_dir: Path, print_table: bool = False) -> None:
    """Compute macro per-DOI metrics and write the three CSVs to ``base_dir``.

    Can be called directly from other scripts to ensure the CSVs exist without
    requiring a separate manual invocation.
    """
    base_dir = base_dir.resolve()
    recall_dir = base_dir / "recall"
    precision_dir = base_dir / "precision"

    recall_map = {p.name[len("recall_") : -len(".jsonl")]: p for p in recall_dir.glob("recall_*.jsonl")}
    precision_map = {p.name[len("precision_") : -len(".jsonl")]: p for p in precision_dir.glob("precision_*.jsonl")}
    models = sorted(set(recall_map) & set(precision_map))

    if not models:
        raise SystemExit("No matching recall_/precision_ model pairs found.")

    per_doi_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for model in models:
        recall_counts = load_recall_counts(recall_map[model])
        precision_counts = load_precision_counts(precision_map[model])
        matched_dois = sorted(set(recall_counts) & set(precision_counts))

        family, variant = parse_model_variant(model)

        recalls: List[float] = []
        precisions: List[float] = []
        f1s: List[float] = []

        for doi in matched_dois:
            r = recall_counts[doi]["supported"] / recall_counts[doi]["total"] if recall_counts[doi]["total"] else 0.0
            s = precision_counts[doi]["supported"]
            c = precision_counts[doi]["contradicted"]
            ns = precision_counts[doi]["not_supported"]
            t = precision_counts[doi]["total"]

            p = precision_fancy(s, c, ns, t)
            f1 = f1_score(p, r)

            recalls.append(r)
            precisions.append(p)
            f1s.append(f1)

            per_doi_rows.append(
                {
                    "family": family,
                    "variant": variant,
                    "model": model,
                    "doi": doi,
                    "recall": r,
                    "precision": p,
                    "f1": f1,
                }
            )

        summary_rows.append(
            {
                "family": family,
                "variant": variant,
                "model": model,
                "n_dois": len(matched_dois),
                "macro_recall_mean": safe_mean(recalls),
                "macro_recall_variance": safe_variance(recalls),
                "macro_precision_mean": safe_mean(precisions),
                "macro_precision_variance": safe_variance(precisions),
                "macro_f1_mean": safe_mean(f1s),
                "macro_f1_variance": safe_variance(f1s),
            }
        )

    variant_order = {"base": 0, "tools": 1, "tools_filter": 2, "filter": 3}
    per_doi_rows.sort(key=lambda r: (str(r["family"]), variant_order.get(str(r["variant"]), 99), str(r["model"]), str(r["doi"])))
    summary_rows.sort(key=lambda r: (str(r["family"]), variant_order.get(str(r["variant"]), 99), str(r["model"])))

    out_doi = base_dir / "metrics_macro_per_doi_by_model.csv"
    out_summary = base_dir / "metrics_macro_variance_by_model.csv"
    out_grouped = base_dir / "metrics_macro_variance_grouped_by_family.csv"

    write_csv(
        out_doi,
        per_doi_rows,
        ["family", "variant", "model", "doi", "recall", "precision", "f1"],
    )
    write_csv(
        out_summary,
        summary_rows,
        [
            "family", "variant", "model", "n_dois",
            "macro_recall_mean", "macro_recall_variance",
            "macro_precision_mean", "macro_precision_variance",
            "macro_f1_mean", "macro_f1_variance",
        ],
    )
    write_csv(
        out_grouped,
        summary_rows,
        [
            "family", "variant", "model", "n_dois",
            "macro_recall_mean", "macro_recall_variance",
            "macro_precision_mean", "macro_precision_variance",
            "macro_f1_mean", "macro_f1_variance",
        ],
    )

    print(f"Wrote: {out_doi}")
    print(f"Wrote: {out_summary}")
    print(f"Wrote: {out_grouped}")
    print(f"Models processed: {len(models)}")
    counts = defaultdict(int)
    for r in summary_rows:
        counts[int(r["n_dois"])] += 1
    print("n_dois distribution:", dict(sorted(counts.items())))

    if print_table:
        print()
        main_rows = [r for r in summary_rows if str(r["family"]) not in SPECIAL_TABLE_FAMILIES]
        special_rows = [r for r in summary_rows if str(r["family"]) in SPECIAL_TABLE_FAMILIES]
        if main_rows:
            _print_table(main_rows)
        if special_rows:
            if main_rows:
                print()
            _print_table(special_rows, title="Google AI Overview, AI Mode, and OpenEvidence")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute macro per-DOI recall/precision/F1 metrics for all models.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Directory containing recall/ and precision/ folders.",
    )
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="Skip terminal table printing; still writes CSV files.",
    )
    args = parser.parse_args()
    compute_metrics(args.base_dir.resolve(), print_table=not args.no_table)


if __name__ == "__main__":
    main()

