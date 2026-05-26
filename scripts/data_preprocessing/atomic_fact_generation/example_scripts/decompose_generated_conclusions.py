#!/usr/bin/env python3
"""
decompose_generated_conclusions.py

Generate atomic facts from generated conclusions stored in query-log result files from
running the models and agents using SciConHarness.

Expected log structure::

    <logs_dir>/
      <model_name>/
        <doi>/
          result.json     # or <doi>.json

Each ``result.json`` must contain:
  - ``response`` (str): full model response, OR
  - ``conclusion`` (str): pre-extracted conclusion text
  - ``query`` / ``question`` (str): the question that was answered

Extracted conclusions must be wrapped in triple brackets ``[[[...]]]`` inside
the response field; the ``conclusion`` field is used as-is (stripped of
``[[[...]]]`` if present).

Usage::

    # Process ALL models found in logs_dir (default behaviour when --model is omitted)
    python decompose_generated_conclusions.py \\
        --logs-dir  path/to/data_querying/logs \\
        --output-dir path/to/output

    # Process a single model directory
    python decompose_generated_conclusions.py \\
        --logs-dir  path/to/data_querying/logs \\
        --output-dir path/to/output \\
        --model gpt-5.1_tools_filter \\
        [--model-configs path/to/model_config.yaml]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup — add the atomic_fact_generation package parent to sys.path
# ---------------------------------------------------------------------------
_PKG_PARENT = str(Path(__file__).resolve().parent.parent.parent)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from atomic_fact_generation import AtomicFactGenerator
from atomic_fact_generation.config.model_config import create_default_configs, load_configs_from_yaml, load_configs_from_json

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Conclusion extraction
# ---------------------------------------------------------------------------

def extract_conclusion(text: str) -> Optional[str]:
    """Extract the longest ``[[[...]]]``-wrapped conclusion from *text*.

    Falls back to ``[[...]]`` then ``[...]`` if triple brackets are absent.
    Returns ``None`` if no match is found.
    """
    candidates = []
    for m in re.finditer(r"\[\[\[(.*?)\]\]\]", text, re.DOTALL):
        candidates.append(m.group(1).strip())
    for m in re.finditer(r"\[\[(.*?)\]\]", text, re.DOTALL):
        candidates.append(m.group(1).strip())
    for m in re.finditer(r"\[([^\[\]]+)\]", text, re.DOTALL):
        candidates.append(m.group(1).strip())

    valid = [c for c in candidates if len(c) > 10]
    return max(valid, key=len) if valid else None


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def load_result_file(path: Path) -> Optional[Dict[str, Any]]:
    """Load a result JSON file, normalising list-wrapped results."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            print(f"  Warning: {path} is a list — using first element")
            return data[0] if data and isinstance(data[0], dict) else None
        return data if isinstance(data, dict) else None
    except Exception as e:
        print(f"  Error loading {path}: {e}")
        return None


def find_result_files(
    logs_dir: Path,
    model_name: Optional[str] = None,
) -> List[Tuple[str, str, Path]]:
    """Return ``(model_name, doi, result_path)`` tuples from *logs_dir*.

    Skips directories whose names start with ``query_batch``.
    """
    results = []
    if not logs_dir.exists():
        print(f"Error: logs directory not found: {logs_dir}")
        return results

    for model_dir in sorted(logs_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("query_batch"):
            continue
        if model_name and model_dir.name != model_name:
            continue

        for doi_dir in sorted(model_dir.iterdir()):
            if not doi_dir.is_dir():
                continue
            result_path = doi_dir / "result.json"
            if not result_path.exists():
                safe_doi = doi_dir.name.replace("/", "_")
                result_path = doi_dir / f"{safe_doi}.json"
            if result_path.exists():
                results.append((model_dir.name, doi_dir.name, result_path))

    return results


def load_existing_output(path: Path) -> Dict[str, Any]:
    """Load an existing output file; return empty dict if not found."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: could not load {path}: {e}")
        return {}


def is_processed(doi: str, existing: Dict[str, Any]) -> bool:
    """Return True if *doi* already has a valid (or error) result recorded."""
    entry = existing.get(doi, {})
    return "total_atomic_facts" in entry or "error" in entry


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_output(atomic_facts_pairs, para_breaks, metadata) -> Dict[str, Any]:
    """Flatten pipeline output into a serialisable dict."""
    if not isinstance(metadata, dict):
        metadata = {}

    all_facts = [f for _, facts in atomic_facts_pairs for f in facts]
    token_usage = metadata.get("token_usage", {}) if isinstance(metadata, dict) else {}

    initial_pairs = metadata.get("initial_atomic_facts_pairs", [])
    dectx_pairs = metadata.get("decontextualized_atomic_facts_pairs", [])
    dep_meta = metadata.get("dependent_facts_metadata", {})
    irr_meta = metadata.get("irrelevant_facts_metadata", {})
    red_meta = metadata.get("redundant_facts_metadata", {})

    return {
        "atomic_facts_pairs": [[s, f] for s, f in atomic_facts_pairs],
        "all_facts": all_facts,
        "total_atomic_facts": len(all_facts),
        "paragraph_breaks": para_breaks,
        "metadata": {
            "initial_facts_count": sum(len(f) for _, f in initial_pairs),
            "decontextualized_facts_count": sum(len(f) for _, f in dectx_pairs),
            "final_facts_count": len(all_facts),
            "dependent_facts_count": len(dep_meta) if isinstance(dep_meta, dict) else 0,
            "irrelevant_facts_filtered": len(irr_meta) if isinstance(irr_meta, dict) else 0,
            "redundant_facts_filtered": len(red_meta) if isinstance(red_meta, dict) else 0,
            "token_usage": token_usage,
        },
    }


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_model(
    model_name: str,
    logs_dir: Path,
    output_dir: Path,
    generator: AtomicFactGenerator,
) -> Dict[str, Any]:
    """Process all result files for one model; returns a summary dict."""
    result_files = find_result_files(logs_dir, model_name=model_name)
    if not result_files:
        print(f"No result files found for model '{model_name}'.")
        return {"processed": 0, "skipped": 0, "errors": 0, "no_conclusion": 0}

    print(f"Found {len(result_files)} result files for '{model_name}'")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_safe = model_name.replace("/", "_").replace("\\", "_")
    out_path = output_dir / f"{model_safe}_atomic_facts.json"
    model_results = load_existing_output(out_path)

    if model_results:
        print(f"  Loaded {len(model_results)} existing results (resume mode)")

    stats = {"processed": 0, "skipped": 0, "errors": 0, "no_conclusion": 0, "already": 0}

    for idx, (_, doi, result_path) in enumerate(result_files, 1):
        label = f"[{idx}/{len(result_files)}] {model_name}/{doi}"

        if is_processed(doi, model_results):
            print(f"{label} — already processed, skipping")
            stats["already"] += 1
            continue

        print(f"\n{label}")
        data = load_result_file(result_path)
        if not data or not isinstance(data, dict):
            print("  Error: could not load result file")
            stats["errors"] += 1
            continue

        response = data.get("response", "")
        query = data.get("query", "") or data.get("question", "")
        direct_conclusion = data.get("conclusion", "")

        if direct_conclusion:
            conclusion = extract_conclusion(direct_conclusion) or direct_conclusion.strip()
            print(f"  Using 'conclusion' field ({len(conclusion)} chars)")
        elif response:
            conclusion = extract_conclusion(response)
            if not conclusion:
                print("  No [[[...]]] pattern found, skipping")
                stats["no_conclusion"] += 1
                continue
            print(f"  Extracted from 'response' field ({len(conclusion)} chars)")
        else:
            print("  No response field, skipping")
            stats["skipped"] += 1
            continue

        if not query or not query.strip():
            print("  No query/question, skipping")
            stats["skipped"] += 1
            continue

        try:
            print("  Generating atomic facts...")
            atomic_facts_pairs, para_breaks, metadata = generator.run(
                generation=conclusion,
                question=query,
                verbose=False,
                enable_incomplete_detection=True,
                enable_irrelevant_filtering=True,
                enable_redundant_filtering=True,
            )

            out = format_output(atomic_facts_pairs, para_breaks, metadata)
            out.update({
                "model": model_name,
                "doi": doi,
                "query": query,
                "conclusion_text": conclusion,
                "full_response": response,
                "result_metadata": {
                    "timestamp": data.get("timestamp"),
                    "provider": data.get("provider"),
                    "model": data.get("model"),
                    "token_usage": data.get("token_usage") or {},
                    "enable_tool_calling": data.get("enable_tool_calling"),
                    "enable_filtering": data.get("enable_filtering"),
                },
            })

            model_results[doi] = out
            stats["processed"] += 1
            print(f"  Generated {out['total_atomic_facts']} atomic facts")

            if stats["processed"] % 5 == 0:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(model_results, f, indent=2, ensure_ascii=False)
                print(f"  Progress saved ({stats['processed']} new)")

        except Exception as e:
            err = str(e)
            print(f"  Error: {err}")
            model_results[doi] = {
                "error": err,
                "error_type": type(e).__name__,
                "model": model_name,
                "doi": doi,
                "query": query,
                "conclusion_text": conclusion,
            }
            stats["errors"] += 1

            if stats["errors"] % 5 == 0:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(model_results, f, indent=2, ensure_ascii=False)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(model_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved {len(model_results)} total results to: {out_path}")
    if stats["already"] > 0:
        print(f"  Skipped {stats['already']} already-processed DOIs")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate atomic facts from model query-log conclusions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--logs-dir",
        required=True,
        help="Path to the query logs directory with the result.json files containing the generated conclusions.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write atomic-fact JSON files.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Name of a single model subdirectory to process (e.g. 'gpt-5.1_tools_filter'). "
            "If omitted, every model directory found under --logs-dir is processed."
        ),
    )
    parser.add_argument(
        "--model-configs",
        default=None,
        help="Path to a YAML or JSON model-config file (default: config/model_config.yaml).",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)

    # Load model configs
    if args.model_configs:
        cfg_path = Path(args.model_configs)
        if cfg_path.suffix.lower() == ".json":
            model_configs = load_configs_from_json(cfg_path)
        else:
            model_configs = load_configs_from_yaml(cfg_path)
        print(f"Loaded model configs from: {cfg_path}")
    else:
        model_configs = None
        print("Using default model configs (config/model_config.yaml)")

    generator = AtomicFactGenerator(model_configs=model_configs)

    result_files = find_result_files(logs_dir, model_name=args.model)
    if not result_files:
        print("No result files found. Check --logs-dir.")
        sys.exit(1)

    models = sorted({m for m, _, _ in result_files})
    if args.model and args.model not in models:
        print(f"Model '{args.model}' not found. Available: {', '.join(models)}")
        sys.exit(1)

    print("=" * 70)
    print("Generating Atomic Facts from Query-Log Results")
    print(f"Models to process: {', '.join(models)}")
    print("=" * 70)

    total_stats = {"processed": 0, "skipped": 0, "errors": 0, "no_conclusion": 0}
    for model in models:
        print(f"\n{'='*70}\nModel: {model}\n{'='*70}")
        stats = process_model(model, logs_dir, output_dir, generator)
        for k in total_stats:
            total_stats[k] += stats.get(k, 0)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Processed:       {total_stats['processed']}")
    print(f"  Skipped:         {total_stats['skipped']}")
    print(f"  No conclusion:   {total_stats['no_conclusion']}")
    print(f"  Errors:          {total_stats['errors']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
