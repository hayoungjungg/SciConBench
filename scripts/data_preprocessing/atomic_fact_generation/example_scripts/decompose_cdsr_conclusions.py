#!/usr/bin/env python3
"""
decompose_cdsr_conclusions.py

Batch-process Cochrane systematic review authors' conclusions into atomic facts.

Reads review articles from a ``data.json`` file and pairs each conclusion with
a pre-generated question from ``generated_questions.json``.  Results are written
in batches so large datasets can be processed incrementally.

Usage::

    # List available batches
    python decompose_cdsr_conclusions.py --batch 0 --list-batches \\
        --data-path path/to/data.json

    # Process batch 0 (default: 500 articles per batch)
    python decompose_cdsr_conclusions.py \\
        --batch 0 \\
        --data-path  path/to/data.json \\
        --output-dir path/to/output \\
        --questions-path path/to/generated_questions.json

    # Custom model config and batch size
    python decompose_cdsr_conclusions.py \\
        --batch 1 --batch-size 200 \\
        --data-path  path/to/data.json \\
        --output-dir path/to/output \\
        --model-configs path/to/model_config.yaml
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PKG_PARENT = str(Path(__file__).resolve().parent.parent.parent)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from atomic_fact_generation import AtomicFactGenerator
from atomic_fact_generation.config.model_config import (
    create_default_configs,
    load_configs_from_json,
    load_configs_from_yaml,
)

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BatchConfig:
    """All paths and settings for one batch run."""
    batch_num: int
    batch_size: int
    data_path: Path
    output_dir: Path
    output_stem: str
    questions_path: Path
    model_configs_path: Optional[Path] = None

    @property
    def log_dir(self) -> Path:
        return self.output_dir / "logs"

    @property
    def batch_output_path(self) -> Path:
        return self.output_dir / f"{self.output_stem}_batch_{self.batch_num}.json"

    @property
    def log_file_path(self) -> Path:
        return self.log_dir / f"{self.output_stem}_batch_{self.batch_num}.log"


@dataclass
class ProcessingStats:
    processed: int = 0
    skipped: int = 0
    errors: int = 0

    def total(self) -> int:
        return self.processed + self.skipped + self.errors


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(batch_num: int, log_dir: Path, stem: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"batch_{batch_num}")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    fh = logging.FileHandler(log_dir / f"{stem}_batch_{batch_num}.log", mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    return logger


def _log(logger: Optional[logging.Logger], msg: str, level: str = "info"):
    if logger:
        getattr(logger, level)(msg)
    else:
        print(msg)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def extract_conclusions(article: Dict[str, Any]) -> Optional[str]:
    """Return the 'Authors' conclusions' section text, or None."""
    for section in article.get("abstract", []):
        heading = section.get("heading", "").lower()
        if "conclusions" in heading or "authors' conclusions" in heading:
            return section.get("text", "") or None
    return None


def load_articles(data_path: Path, logger=None) -> List[Dict[str, Any]]:
    if not data_path.exists():
        _log(logger, f"Error: data file not found: {data_path}", "error")
        return []
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            articles = json.load(f)
        _log(logger, f"Loaded {len(articles)} articles from {data_path}")
        return articles
    except Exception as e:
        _log(logger, f"Error loading {data_path}: {e}", "error")
        return []


def load_questions(path: Path, logger=None) -> Dict[str, str]:
    if not path.exists():
        _log(logger, f"Warning: questions file not found: {path}", "warning")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            q = json.load(f)
        _log(logger, f"Loaded {len(q)} questions from {path}")
        return q
    except Exception as e:
        _log(logger, f"Error loading questions: {e}", "error")
        return {}


def load_existing_results(path: Path, logger=None) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        _log(logger, f"Warning: could not load existing results: {e}", "warning")
        return {}


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_output(
    atomic_facts_pairs,
    para_breaks,
    metadata,
    question: str,
    conclusions_text: str,
) -> Dict[str, Any]:
    all_facts = [f for _, facts in atomic_facts_pairs for f in facts]
    return {
        "atomic_facts_pairs": [[s, f] for s, f in atomic_facts_pairs],
        "all_facts": all_facts,
        "total_atomic_facts": len(all_facts),
        "paragraph_breaks": para_breaks,
        "metadata": {
            "initial_facts_count": sum(len(f) for _, f in metadata.get("initial_atomic_facts_pairs", [])),
            "decontextualized_facts_count": sum(len(f) for _, f in metadata.get("decontextualized_atomic_facts_pairs", [])),
            "final_facts_count": len(all_facts),
            "dependent_facts_count": len(metadata.get("dependent_facts_metadata", {})),
            "irrelevant_facts_filtered": len(metadata.get("irrelevant_facts_metadata", {})),
            "redundant_facts_filtered": len(metadata.get("redundant_facts_metadata", {})),
            "token_usage": metadata.get("token_usage", {}),
        },
        "question": question,
        "conclusions_text": conclusions_text,
    }


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def calculate_batch_ranges(total: int, batch_size: int) -> List[Tuple[int, int, int]]:
    """Return list of (batch_num, start_idx, end_idx)."""
    ranges = []
    n_batches = (total + batch_size - 1) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, total)
        ranges.append((i, start, end))
    return ranges


def save_results(path: Path, new_results: Dict[str, Any], logger=None):
    """Merge *new_results* into any existing file and write back."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception as e:
            _log(logger, f"Warning: could not read existing results for merge: {e}", "warning")

    merged = {**existing, **new_results}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_article(
    article: Dict[str, Any],
    questions: Dict[str, str],
    generator: AtomicFactGenerator,
    logger=None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Process one article; returns ``(result_dict, status)``."""
    doi = article.get("doi", "")
    name = article.get("name") or "Unknown"

    conclusions = extract_conclusions(article)
    if not conclusions:
        _log(logger, "  No conclusions found, skipping", "warning")
        return {"error": "No conclusions found"}, "skipped"

    question = questions.get(doi)
    if not question:
        question = f"What are the findings and conclusions regarding {name}?"
        _log(logger, "  Question not found — using fallback", "warning")

    _log(logger, f"  Conclusions: {len(conclusions)} chars")
    _log(logger, f"  Question:    {question[:100]}{'...' if len(question) > 100 else ''}")

    try:
        if isinstance(conclusions, bytes):
            conclusions = conclusions.decode("utf-8", errors="replace")
        if isinstance(question, bytes):
            question = question.decode("utf-8", errors="replace")

        atomic_facts_pairs, para_breaks, metadata = generator.run(
            generation=conclusions,
            question=question,
            verbose=False,
            enable_incomplete_detection=True,
            enable_irrelevant_filtering=False,  # CDSR conclusions are high-quality and do not contain irrelevant information; skip this step to save cost
            enable_redundant_filtering=True,
        )

        result = format_output(atomic_facts_pairs, para_breaks, metadata, question, conclusions)
        _log(logger, f"  Generated {result['total_atomic_facts']} atomic facts")
        return result, "processed"

    except Exception as e:
        _log(logger, f"  Error: {e}", "error")
        return {"error": str(e), "conclusions_text": conclusions}, "error"


def process_batch(
    batch_articles: List[Dict[str, Any]],
    config: BatchConfig,
    questions: Dict[str, str],
    generator: AtomicFactGenerator,
    existing_results: Dict[str, Any],
    logger=None,
) -> Tuple[Dict[str, Any], ProcessingStats]:
    results = {}
    stats = ProcessingStats()

    _log(logger, f"\nProcessing {len(batch_articles)} articles...")
    _log(logger, "=" * 70)

    for idx, article in enumerate(batch_articles):
        doi = article.get("doi", "")
        name = article.get("name") or "Unknown"
        display_name = name[:80] + ("..." if len(name) > 80 else "")

        _log(logger, f"\n[{idx+1}/{len(batch_articles)}] {doi}")
        _log(logger, f"  Article: {display_name}")

        if doi in existing_results:
            entry = existing_results[doi]
            if "error" not in entry and "total_atomic_facts" in entry:
                _log(logger, "  Already processed, skipping")
                results[doi] = entry
                stats.skipped += 1
                if (idx + 1) % 10 == 0:
                    save_results(config.batch_output_path, results, logger)
                continue

        result, status = process_article(article, questions, generator, logger)
        results[doi] = result

        if status == "processed":
            stats.processed += 1
        elif status == "skipped":
            stats.skipped += 1
        else:
            stats.errors += 1

        if (idx + 1) % 10 == 0:
            save_results(config.batch_output_path, results, logger)
            _log(logger, f"  Progress saved ({idx+1}/{len(batch_articles)})")

    return results, stats


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(config: BatchConfig, logger=None) -> bool:
    _log(logger, "=" * 70)
    _log(logger, "Batch Processing Authors' Conclusions -> Atomic Facts")
    _log(logger, f"Batch {config.batch_num} | size {config.batch_size}")
    _log(logger, f"Data:    {config.data_path}")
    _log(logger, f"Output:  {config.batch_output_path}")
    _log(logger, f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    _log(logger, "=" * 70)

    articles = load_articles(config.data_path, logger)
    if not articles:
        return False

    with_conclusions = [a for a in articles if a.get("doi") and extract_conclusions(a)]
    _log(logger, f"Articles with conclusions: {len(with_conclusions)}")

    batch_ranges = calculate_batch_ranges(len(with_conclusions), config.batch_size)
    if config.batch_num >= len(batch_ranges):
        _log(logger, f"Error: batch {config.batch_num} out of range (total: {len(batch_ranges)})", "error")
        return False

    _, start, end = batch_ranges[config.batch_num]
    batch_articles = with_conclusions[start:end]
    _log(logger, f"Batch {config.batch_num}: articles {start}–{end-1} ({len(batch_articles)} total)")

    existing = load_existing_results(config.batch_output_path, logger)
    if existing:
        already = sum(
            1 for a in batch_articles
            if a.get("doi") in existing
            and "total_atomic_facts" in existing[a["doi"]]
        )
        if already:
            _log(logger, f"  {already} articles already processed — will skip")

    questions = load_questions(config.questions_path, logger)
    if not questions:
        _log(logger, "Warning: no questions loaded — will use fallback questions", "warning")

    # Load model configs
    if config.model_configs_path and config.model_configs_path.exists():
        if config.model_configs_path.suffix.lower() == ".json":
            model_configs = load_configs_from_json(config.model_configs_path)
        else:
            model_configs = load_configs_from_yaml(config.model_configs_path)
        _log(logger, f"Loaded model configs from: {config.model_configs_path}")
    else:
        model_configs = None
        _log(logger, "Using default model configs")

    generator = AtomicFactGenerator(model_configs=model_configs)

    results, stats = process_batch(batch_articles, config, questions, generator, existing, logger)

    _log(logger, "\n" + "=" * 70)
    _log(logger, "Saving final results...")
    save_results(config.batch_output_path, results, logger)
    _log(logger, f"Saved to: {config.batch_output_path}")

    _log(logger, "\n" + "=" * 70)
    _log(logger, f"Batch {config.batch_num} Summary")
    _log(logger, "=" * 70)
    _log(logger, f"  Processed: {stats.processed}")
    _log(logger, f"  Skipped:   {stats.skipped}")
    _log(logger, f"  Errors:    {stats.errors}")
    _log(logger, f"  Completed: {datetime.now():%Y-%m-%d %H:%M:%S}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch-process CDSR authors' conclusions into atomic facts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--batch", type=int, required=True, help="Batch number (0-indexed).")
    parser.add_argument("--batch-size", type=int, default=500, help="Articles per batch (default: 500).")
    parser.add_argument("--data-path", required=True, help="Path to data.json.")
    parser.add_argument("--output-dir", required=True, help="Directory for output JSON files.")
    parser.add_argument("--questions-path", required=True, help="Path to generated_questions.json.")
    parser.add_argument(
        "--output-stem",
        default="conclusions_atomic_facts",
        help="Base filename stem for output files (default: conclusions_atomic_facts).",
    )
    parser.add_argument(
        "--model-configs",
        default=None,
        help="Path to YAML or JSON model-config file (default: config/model_config.yaml).",
    )
    parser.add_argument(
        "--list-batches",
        action="store_true",
        help="Print batch information and exit.",
    )

    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    questions_path = Path(args.questions_path)
    model_configs_path = Path(args.model_configs) if args.model_configs else None

    if args.list_batches:
        articles = load_articles(data_path)
        with_conclusions = [a for a in articles if a.get("doi") and extract_conclusions(a)]
        batch_ranges = calculate_batch_ranges(len(with_conclusions), args.batch_size)
        print(f"\nTotal articles with conclusions: {len(with_conclusions)}")
        print(f"Batch size: {args.batch_size} | Total batches: {len(batch_ranges)}")
        for b, s, e in batch_ranges:
            print(f"  Batch {b}: articles {s}–{e-1} ({e-s})")
        return

    config = BatchConfig(
        batch_num=args.batch,
        batch_size=args.batch_size,
        data_path=data_path,
        output_dir=output_dir,
        output_stem=args.output_stem,
        questions_path=questions_path,
        model_configs_path=model_configs_path,
    )

    logger = setup_logging(config.batch_num, config.log_dir, config.output_stem)
    success = run(config, logger)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
