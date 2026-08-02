#!/usr/bin/env python3
"""
run_bulk_atomic_facts.py

Bulk, shardable atomic-fact generation over SciConHarness query-log responses,
built for running **multiple shards in parallel** (optionally against different
API keys / endpoints), analogous to ``archive/run_bulk_precision_recall_labels.py``.

Expected log structure (see ``sciconharness/logs/``)::

    <logs-dir>/
      <model-name>/                 # e.g. "qwen_qwen3.5-9b_tools_filter"
        <doi>/
          result.json                # must contain "response" and "query"

The conclusion is extracted from the ``[[[...]]]``-wrapped span inside
``response`` (same convention as ``decompose_generated_conclusions.py``).

Each shard writes its own resumable JSONL file — one line per DOI — so shards
never contend for the same file while running concurrently. Once all shards
for a model finish, merge them into the single dict-format
``<model>_atomic_facts.json`` expected by downstream tooling (e.g.
``--preprocessed-facts`` in ``run_bulk_precision_recall_labels.py``).

Usage
-----
List model directories available under a logs dir::

    python run_bulk_atomic_facts.py --logs-dir ../../../../sciconharness/logs --list-models

Run shard 0 of 4 for one model (repeat with different --shard-id/--api-key/--base-url
in parallel terminals or background jobs to fan out the work)::

    python run_bulk_atomic_facts.py \\
        --logs-dir  ../../../../sciconharness/logs \\
        --model     qwen_qwen3.5-9b_tools_filter \\
        --api-key   "$AZURE_KEY_A" --base-url "https://YOUR_RESOURCE_A.openai.azure.com/" \\
        --shard-id 0 --num-shards 4 \\
        --output-dir output/

    python run_bulk_atomic_facts.py \\
        --logs-dir  ../../../../sciconharness/logs \\
        --model     qwen_qwen3.5-9b_tools_filter \\
        --api-key   "$AZURE_KEY_B" --base-url "https://YOUR_RESOURCE_B.openai.azure.com/" \\
        --shard-id 1 --num-shards 4 \\
        --output-dir output/

    # ... shard-id 2, 3 similarly (each can use its own key/endpoint) ...

Once every shard has completed, merge them into the single per-model JSON file::

    python run_bulk_atomic_facts.py \\
        --model qwen_qwen3.5-9b_tools_filter \\
        --num-shards 4 \\
        --output-dir output/ \\
        --merge

Resume: re-run a shard with the same ``--output-dir``/``--model``/``--shard-id``/
``--num-shards``; DOIs already present in that shard's JSONL are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterator, Optional

# ---------------------------------------------------------------------------
# Path setup — add the atomic_fact_generation package parent to sys.path
# ---------------------------------------------------------------------------
_PKG_PARENT = str(Path(__file__).resolve().parent.parent.parent)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from atomic_fact_generation import AtomicFactGenerator
from atomic_fact_generation.config.model_config import load_configs_from_json, load_configs_from_yaml

from dotenv import load_dotenv  # type: ignore[import-not-found]

load_dotenv()


# ---------------------------------------------------------------------------
# Conclusion extraction (mirrors decompose_generated_conclusions.py)
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
# Log discovery
# ---------------------------------------------------------------------------

def list_model_dirs(logs_dir: Path) -> list[str]:
    """Return sorted model directory names under *logs_dir* that contain DOI/result.json entries."""
    models = []
    for d in sorted(logs_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("query_batch"):
            continue
        has_result = any(
            (sub / "result.json").exists() or list(sub.glob("*.json"))
            for sub in d.iterdir()
            if sub.is_dir()
        )
        if has_result:
            models.append(d.name)
    return models


def find_result_files(logs_dir: Path, model_name: str) -> list[tuple[str, Path]]:
    """Return sorted ``(doi, result_path)`` tuples for a single model directory."""
    model_dir = logs_dir / model_name
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    results = []
    for doi_dir in sorted(model_dir.iterdir()):
        if not doi_dir.is_dir():
            continue
        result_path = doi_dir / "result.json"
        if not result_path.exists():
            safe_doi = doi_dir.name.replace("/", "_")
            result_path = doi_dir / f"{safe_doi}.json"
        if result_path.exists():
            results.append((doi_dir.name, result_path))
    return results


def load_result_file(path: Path) -> Optional[dict[str, Any]]:
    """Load a result JSON file, normalising list-wrapped results."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data[0] if data and isinstance(data[0], dict) else None
        return data if isinstance(data, dict) else None
    except Exception as e:
        print(f"  Error loading {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Output formatting (mirrors decompose_generated_conclusions.format_output)
# ---------------------------------------------------------------------------

def format_output(atomic_facts_pairs, para_breaks, metadata) -> dict[str, Any]:
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
# Sharding + resume helpers (mirrors run_bulk_precision_recall_labels.py)
# ---------------------------------------------------------------------------

def _iter_shard(items: list[Any], shard_id: int, num_shards: int) -> Iterator[Any]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards")
    for i, x in enumerate(items):
        if i % num_shards == shard_id:
            yield x


def _resume_dois_jsonl(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.is_file():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            doi = row.get("doi")
            if doi:
                done.add(str(doi))
    return done


def shard_output_path(output_dir: Path, model_safe: str, shard_id: int, num_shards: int) -> Path:
    return output_dir / f"{model_safe}_shard{shard_id}of{num_shards}_atomic_facts.jsonl"


def _prune_errored_dois(output_path: Path, error_log_path: Path) -> int:
    """Remove DOIs recorded in *error_log_path* from *output_path* so they get
    reprocessed instead of being treated as already-done.

    Only DOIs that appear in the ``.errors.jsonl`` sidecar are pruned — this
    intentionally excludes rows that were skipped for having no parseable
    ``[[[...]]]`` conclusion or no query/question field, since those aren't
    transient failures (e.g. rate limits) and retrying won't change the
    outcome. The errors file itself is cleared; any DOI that fails again this
    run will get a fresh entry.

    Returns the number of DOIs pruned (and thus queued for retry).
    """
    if not error_log_path.is_file():
        return 0

    errored_dois: set[str] = set()
    with error_log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            doi = row.get("doi")
            if doi:
                errored_dois.add(str(doi))

    if not errored_dois:
        return 0

    if output_path.is_file():
        kept_lines = []
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    kept_lines.append(line if line.endswith("\n") else line + "\n")
                    continue
                if str(row.get("doi")) in errored_dois:
                    continue
                kept_lines.append(line if line.endswith("\n") else line + "\n")
        with output_path.open("w", encoding="utf-8") as f:
            f.writelines(kept_lines)

    # Clear the errors file; DOIs that fail again this run will be re-appended.
    error_log_path.write_text("", encoding="utf-8")

    return len(errored_dois)


# ---------------------------------------------------------------------------
# Merge shards -> single dict-format {model}_atomic_facts.json
# ---------------------------------------------------------------------------

def merge_shards(output_dir: Path, model_safe: str, num_shards: int) -> Path:
    merged: dict[str, Any] = {}
    n_shard_files_found = 0
    n_rows = 0
    n_errors = 0

    for shard_id in range(num_shards):
        shard_path = shard_output_path(output_dir, model_safe, shard_id, num_shards)
        if not shard_path.is_file():
            print(f"  Warning: missing shard file: {shard_path}")
            continue
        n_shard_files_found += 1
        with shard_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                doi = row.pop("doi")
                row.pop("shard", None)
                if "error" in row:
                    n_errors += 1
                merged[doi] = row
                n_rows += 1

    out_path = output_dir / f"{model_safe}_atomic_facts.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {
                "action": "merge",
                "model": model_safe,
                "num_shards": num_shards,
                "shard_files_found": n_shard_files_found,
                "dois_merged": len(merged),
                "rows_seen": n_rows,
                "errors": n_errors,
                "output": str(out_path),
            },
            indent=2,
        )
    )
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Bulk, shardable atomic-fact generation over SciConHarness query logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--logs-dir",
        type=Path,
        default=None,
        help="Path to sciconharness/logs (required unless --merge).",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model directory name within --logs-dir, e.g. 'qwen_qwen3.5-9b_tools_filter'.",
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        help="List available model directories under --logs-dir and exit.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for shard JSONL files and the merged JSON output (required unless --list-models).",
    )
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument(
        "--merge",
        action="store_true",
        help="Merge all shard JSONL files for --model into '<model>_atomic_facts.json' and exit.",
    )
    p.add_argument("--doi", type=str, default=None, help="If set, process only this DOI.")
    p.add_argument("--limit", type=int, default=None, help="Cap items after sharding (debug).")
    p.add_argument(
        "--retry-errors",
        action="store_true",
        help=(
            "Before processing, drop any DOIs recorded in this shard's "
            "'<output>.errors.jsonl' from the resumable output file (and clear "
            "that errors file), so they are re-attempted this run instead of "
            "being treated as already-done. Only rows that hit a real error "
            "(e.g. rate limits, exceptions) are retried — DOIs skipped for "
            "having no parseable conclusion/query are left alone."
        ),
    )

    p.add_argument(
        "--api-key",
        "--azure-openai-key",
        type=str,
        default=None,
        dest="api_key",
        help="API key applied to all pipeline components (overrides env vars). "
        "Pass distinct keys per shard to spread parallel shards across resources.",
    )
    p.add_argument(
        "--base-url",
        "--azure-endpoint",
        type=str,
        default=None,
        dest="base_url",
        help="Endpoint applied to all pipeline components (overrides env vars).",
    )
    p.add_argument(
        "--model-configs",
        type=Path,
        default=None,
        help="Path to a YAML or JSON model-config file (default: config/model_config.yaml).",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per LLM call when the response can't be parsed, before falling back (default: 3).",
    )
    p.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        help="Seconds to sleep between retry attempts (default: 1.0).",
    )
    p.add_argument("--disable-incomplete-detection", action="store_true")
    p.add_argument("--disable-irrelevant-filtering", action="store_true")
    p.add_argument("--disable-redundant-filtering", action="store_true")
    p.add_argument("--verbose", action="store_true")

    args = p.parse_args()

    if args.list_models:
        if not args.logs_dir:
            p.error("--list-models requires --logs-dir")
        for m in list_model_dirs(args.logs_dir):
            print(m)
        return

    if not args.model:
        p.error("--model is required (use --list-models to see available directories)")
    if not args.output_dir:
        p.error("--output-dir is required")
    model_safe = args.model.replace("/", "_").replace("\\", "_")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge_shards(args.output_dir, model_safe, args.num_shards)
        return

    if not args.logs_dir:
        p.error("--logs-dir is required unless --merge is set")

    # ------------------------------------------------------------------
    # Build the generator (component-level model config + credentials)
    # ------------------------------------------------------------------
    model_configs = None
    if args.model_configs:
        if args.model_configs.suffix.lower() == ".json":
            model_configs = load_configs_from_json(args.model_configs)
        else:
            model_configs = load_configs_from_yaml(args.model_configs)
        print(f"Loaded model configs from: {args.model_configs}")
    else:
        print("Using default model configs (config/model_config.yaml)")

    generator = AtomicFactGenerator(
        model_configs=model_configs,
        api_key=args.api_key,
        base_url=args.base_url,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )

    # ------------------------------------------------------------------
    # Discover + shard jobs
    # ------------------------------------------------------------------
    result_files = find_result_files(args.logs_dir, args.model)
    if args.doi:
        result_files = [(doi, path) for doi, path in result_files if doi == args.doi]
    if not result_files:
        print(f"No result files found for model '{args.model}' under {args.logs_dir}")
        sys.exit(1)

    jobs = list(_iter_shard(result_files, args.shard_id, args.num_shards))
    if args.limit is not None:
        jobs = jobs[: max(0, args.limit)]

    output_path = shard_output_path(args.output_dir, model_safe, args.shard_id, args.num_shards)
    error_log_path = output_path.with_suffix(output_path.suffix + ".errors.jsonl")

    n_retry_pruned = 0
    if args.retry_errors:
        n_retry_pruned = _prune_errored_dois(output_path, error_log_path)

    done = _resume_dois_jsonl(output_path)

    print(
        f"Model '{args.model}': {len(result_files)} total DOIs, "
        f"{len(jobs)} in shard {args.shard_id}/{args.num_shards} "
        f"({len(done)} already done, resuming)."
        + (f" Retrying {n_retry_pruned} previously-errored DOI(s)." if args.retry_errors else "")
    )

    from tqdm import tqdm  # type: ignore[import-not-found]

    n_done = 0
    n_skip = 0
    n_err = 0
    n_no_conclusion = 0
    n_since_checkpoint = 0

    with (
        output_path.open("a", encoding="utf-8") as out_f,
        error_log_path.open("a", encoding="utf-8") as err_f,
        tqdm(total=len(jobs), desc=f"{args.model} atomic facts") as bar,
    ):
        for doi, result_path in jobs:
            if doi in done:
                n_skip += 1
                bar.update(1)
                continue

            data = load_result_file(result_path)
            row_base: dict[str, Any] = {
                "doi": doi,
                "model": args.model,
                "shard": {"id": args.shard_id, "num_shards": args.num_shards},
            }

            if not data:
                row = {**row_base, "error": "could not load result file"}
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                err_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                err_f.flush()
                done.add(doi)
                n_err += 1
                bar.update(1)
                continue

            response = data.get("response", "")
            query = data.get("query", "") or data.get("question", "")
            direct_conclusion = data.get("conclusion", "")

            if direct_conclusion:
                conclusion = extract_conclusion(direct_conclusion) or direct_conclusion.strip()
            elif response:
                conclusion = extract_conclusion(response)
            else:
                conclusion = None

            if not conclusion:
                row = {**row_base, "error": "no [[[...]]] conclusion found", "query": query}
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                done.add(doi)
                n_no_conclusion += 1
                bar.update(1)
                continue

            if not query or not query.strip():
                row = {**row_base, "error": "no query/question field", "conclusion_text": conclusion}
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                done.add(doi)
                n_no_conclusion += 1
                bar.update(1)
                continue

            try:
                atomic_facts_pairs, para_breaks, metadata = generator.run(
                    generation=conclusion,
                    question=query,
                    verbose=args.verbose,
                    enable_incomplete_detection=not args.disable_incomplete_detection,
                    enable_irrelevant_filtering=not args.disable_irrelevant_filtering,
                    enable_redundant_filtering=not args.disable_redundant_filtering,
                )
                out = format_output(atomic_facts_pairs, para_breaks, metadata)
                row = {
                    **row_base,
                    **out,
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
                }
                row_had_error = False
            except Exception as e:
                row = {
                    **row_base,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "query": query,
                    "conclusion_text": conclusion,
                }
                row_had_error = True

            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            if row_had_error:
                err_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                err_f.flush()
                n_err += 1

            done.add(doi)
            n_done += 1
            n_since_checkpoint += 1
            if n_since_checkpoint >= 5:
                try:
                    os.fsync(out_f.fileno())
                except OSError:
                    pass
                n_since_checkpoint = 0
            bar.update(1)

        try:
            os.fsync(out_f.fileno())
        except OSError:
            pass

    print(
        json.dumps(
            {
                "model": args.model,
                "shard_id": args.shard_id,
                "num_shards": args.num_shards,
                "jobs_in_shard": len(jobs),
                "written_this_run": n_done,
                "skipped_resume": n_skip,
                "no_conclusion_this_run": n_no_conclusion,
                "errors_this_run": n_err,
                "output": str(output_path),
                "error_log": str(error_log_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nInterrupted — partial results may be in the JSONL (resume with the same --output-dir/--shard-id).",
            file=sys.stderr,
        )
        raise SystemExit(130)
