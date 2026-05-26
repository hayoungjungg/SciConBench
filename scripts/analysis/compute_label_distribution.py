"""
Print label distribution (%) for factual precision and recall across all models,
grouped by model family and variant (base / tools / tools_filter / filter).
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "labeled_facts"

LABELS = ["SUPPORTED", "NOT SUPPORTED", "CONTRADICTED"]

VARIANT_ORDER = {"base": 0, "tools": 1, "tools_filter": 2, "filter": 3}


def parse_model_variant(model_name: str) -> Tuple[str, str]:
    if model_name.endswith("_tools_filter"):
        return model_name[: -len("_tools_filter")], "tools_filter"
    if model_name.endswith("_tools"):
        return model_name[: -len("_tools")], "tools"
    if model_name.endswith("_filter"):
        return model_name[: -len("_filter")], "filter"
    return model_name, "base"


def load_label_counts(task: str, data_dir: Path) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    pattern = os.path.join(data_dir, task, f"{task}_*.jsonl")
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                model = record["source_model"]
                label = record.get("llm_label", "")
                if label in LABELS:
                    counts[model][label] += 1
    return {m: dict(c) for m, c in counts.items()}


def print_table(task: str, counts_by_model: Dict[str, Dict[str, int]]) -> None:
    rows: List[Dict[str, object]] = []
    for model, counts in counts_by_model.items():
        total = sum(counts.values())
        if total == 0:
            continue
        family, variant = parse_model_variant(model)
        rows.append({
            "family": family,
            "variant": variant,
            "total": total,
            "supported_pct": counts.get("SUPPORTED", 0) / total * 100,
            "not_supported_pct": counts.get("NOT SUPPORTED", 0) / total * 100,
            "contradicted_pct": counts.get("CONTRADICTED", 0) / total * 100,
        })

    rows.sort(key=lambda r: (
        str(r["family"]),
        VARIANT_ORDER.get(str(r["variant"]), 99),
    ))

    show_contradicted = task == "precision"
    headers = ["Family", "Variant", "N facts", "Supported %", "Not Supported %"]
    if show_contradicted:
        headers.append("Contradicted %")

    col_data: List[List[str]] = []
    for r in rows:
        row = [
            str(r["family"]),
            str(r["variant"]),
            str(r["total"]),
            f"{r['supported_pct']:.1f}",
            f"{r['not_supported_pct']:.1f}",
        ]
        if show_contradicted:
            row.append(f"{r['contradicted_pct']:.1f}")
        col_data.append(row)

    widths = [len(h) for h in headers]
    for row in col_data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(sep: str = "-") -> str:
        return "+" + "+".join(sep * (w + 2) for w in widths) + "+"

    def render(cells: List[str]) -> str:
        return "| " + " | ".join(cells[i].ljust(widths[i]) for i in range(len(cells))) + " |"

    print(f"\nLabel Distribution — {task.capitalize()}")
    print(line("="))
    print(render(headers))
    print(line("="))

    prev_family = None
    for row in col_data:
        if prev_family is not None and row[0] != prev_family:
            print(line("-"))
        print(render(row))
        prev_family = row[0]

    print(line("="))


def main() -> None:
    parser = argparse.ArgumentParser(description="Print label distribution for precision and recall.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing precision/ and recall/ folders. Defaults to {DEFAULT_DATA_DIR}.",
    )
    args = parser.parse_args()

    for task in ["precision", "recall"]:
        counts = load_label_counts(task, args.data_dir)
        if not counts:
            print(f"[Warning] No data found for task '{task}' in {args.data_dir}")
            continue
        print_table(task, counts)


if __name__ == "__main__":
    main()
