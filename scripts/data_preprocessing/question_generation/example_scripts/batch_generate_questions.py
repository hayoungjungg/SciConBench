"""
Batch Question Generator

Processes a JSON data file and generates a research question for every entry
that has not yet been processed.  Results are written to an output JSON file
that maps DOIs to generated questions.  Already-processed entries are skipped,
so the script is safe to re-run on partial outputs.

Usage (from the repo root):
    python scripts/data_preprocessing/question_generation/example_scripts/batch_generate_questions.py \\
        --data-file  data/review_articles/data.json \\
        --output-file data/preprocessed_qa/generated_questions.json

All remaining flags are optional and override the YAML config:
    --config            Path to preprocessing_config.yaml
    --model             Azure-deployed model name
    --few-shot          Enable few-shot prompting
    --no-few-shot       Disable few-shot prompting (default)
    --batch-save-interval  Save progress every N entries (default: 10)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

# Fallback sys.path for running without pip install -e .
_scripts_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _scripts_root not in sys.path:
    sys.path.insert(0, _scripts_root)

from data_preprocessing.question_generation.generate_questions import QuestionGenerator

_DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "preprocessing_config.yaml"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str] = None) -> dict:
    """Load the YAML config, falling back to the bundled default."""
    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    if not path.exists():
        print(f"Warning: Config file not found at {path}. Using hard-coded defaults.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def extract_objectives_and_background(abstract: list) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract objectives and background sections from the structured abstract.

    Args:
        abstract: List of dicts with 'heading' and 'text' keys.

    Returns:
        Tuple of (objectives_text, background_text); either can be None.
    """
    objectives = None
    background = None

    for section in abstract:
        heading = section.get("heading", "").lower()
        text = section.get("text", "")
        if heading == "objectives":
            objectives = text
        elif heading == "background":
            background = text

    return objectives, background


def load_existing_questions(questions_file: str) -> Dict[str, str]:
    """
    Load previously generated questions from disk.

    Args:
        questions_file: Path to the questions JSON file.

    Returns:
        Dict mapping DOI → question string.
    """
    if os.path.exists(questions_file):
        with open(questions_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_questions(questions: Dict[str, str], questions_file: str) -> None:
    """
    Persist questions to a JSON file, creating parent directories as needed.

    Args:
        questions: Dict mapping DOI → question string.
        questions_file: Destination file path.
    """
    Path(questions_file).parent.mkdir(parents=True, exist_ok=True)
    with open(questions_file, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print(f"Questions saved to {questions_file}")


def remove_outdated_dois(questions: Dict[str, str]) -> Dict[str, str]:
    """
    Keep only the highest-version DOI for each base DOI.

    Cochrane DOIs end with .pub, .pub2, .pub3, etc.  When multiple versions
    exist in the output file, this function retains only the latest one.

    Args:
        questions: Dict mapping DOI → question string.

    Returns:
        Cleaned dict with lower-version duplicates removed.
    """
    pub_pattern = re.compile(r"\.pub(\d*)$")
    base_to_versions: Dict[str, list] = {}

    for doi in questions:
        match = pub_pattern.search(doi)
        if match:
            base = doi[: match.start()]
            version = int(match.group(1)) if match.group(1) else 1
            base_to_versions.setdefault(base, []).append((doi, version))
        else:
            base_to_versions[doi] = [(doi, 0)]

    cleaned: Dict[str, str] = {}
    removed_count = 0

    for base, versions in base_to_versions.items():
        if len(versions) == 1:
            cleaned[versions[0][0]] = questions[versions[0][0]]
        else:
            versions.sort(key=lambda x: -x[1])
            best_doi = versions[0][0]
            cleaned[best_doi] = questions[best_doi]
            removed_count += len(versions) - 1
            for doi, _ in versions[1:]:
                print(f"  Removing outdated DOI: {doi} (keeping {best_doi})")

    if removed_count:
        print(f"\nRemoved {removed_count} outdated DOI(s)")
        print(f"  Before: {len(questions)}  After: {len(cleaned)}")

    return cleaned


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_data_file(
    data_file: str,
    questions_file: str,
    generator: QuestionGenerator,
    batch_save_interval: int = 10,
) -> None:
    """
    Generate questions for all unprocessed entries in *data_file*.

    Args:
        data_file: Path to the source JSON file (list of article dicts with 'doi' and 'abstract').
        questions_file: Path to the output JSON file (DOI → question).
        generator: Initialised QuestionGenerator instance.
        batch_save_interval: Persist progress every N successfully generated questions.
    """
    existing_questions = load_existing_questions(questions_file)
    print(f"Loaded {len(existing_questions)} existing questions from {questions_file}")

    print(f"Loading data from {data_file}...")
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Found {len(data)} entries in data file")

    all_dois = {entry.get("doi") for entry in data if entry.get("doi")}
    missing_dois = all_dois - set(existing_questions)

    print(f"Total DOIs        : {len(all_dois)}")
    print(f"Already processed : {len(existing_questions)}")
    print(f"Remaining         : {len(missing_dois)}")

    if not missing_dois:
        print("All questions already generated. Running DOI version cleanup...")
        existing_questions = remove_outdated_dois(existing_questions)
        save_questions(existing_questions, questions_file)
        print(f"Final count: {len(existing_questions)} questions")
        return

    doi_to_entry = {entry.get("doi"): entry for entry in data if entry.get("doi")}
    processed = skipped = errors = 0

    for doi in missing_dois:
        entry = doi_to_entry.get(doi)
        if not entry:
            skipped += 1
            continue

        abstract = entry.get("abstract", [])
        objectives, background = extract_objectives_and_background(abstract)

        if not objectives:
            print(f"  Skipping {doi}: no 'objectives' section in abstract")
            skipped += 1
            continue

        if not background:
            print(f"  Warning: no 'background' for {doi}, using empty string")
            background = ""

        try:
            print(f"  Processing: {doi}")
            result = generator.run(objectives, background)
            question = result.get("question", "")

            if question and question.strip():
                existing_questions[doi] = question
                processed += 1
                if processed % batch_save_interval == 0:
                    save_questions(existing_questions, questions_file)
                    print(f"  Progress: {processed} generated, {errors} errors, {skipped} skipped")
            else:
                print(f"  Warning: empty/invalid question for {doi}")
                errors += 1

        except Exception as e:
            import traceback
            print(f"  Error processing {doi}: {e}")
            print(f"  {traceback.format_exc()}")
            errors += 1

    print("\n" + "=" * 50)
    print("Removing outdated DOI versions...")
    existing_questions = remove_outdated_dois(existing_questions)

    save_questions(existing_questions, questions_file)

    print("\nProcessing complete!")
    print(f"  Generated : {processed}")
    print(f"  Errors    : {errors}")
    print(f"  Skipped   : {skipped}")
    print(f"  Total in file: {len(existing_questions)}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-generate research questions from a JSON data file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-file",
        required=True,
        help="Path to the input JSON file (list of article dicts).",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Path to the output JSON file mapping DOI → question.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=f"Path to the YAML config file (default: {_DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model deployment/API name (overrides config).",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["azure", "openai"],
        help="API provider: 'azure' for Azure OpenAI, 'openai' for OpenAI API (overrides config).",
    )
    few_shot_group = parser.add_mutually_exclusive_group()
    few_shot_group.add_argument(
        "--few-shot",
        dest="few_shot",
        action="store_true",
        default=None,
        help="Enable few-shot prompting (overrides config).",
    )
    few_shot_group.add_argument(
        "--no-few-shot",
        dest="few_shot",
        action="store_false",
        help="Disable few-shot prompting (overrides config).",
    )
    # Chat-completions model params
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for chat-completions models (overrides config).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max response tokens for chat-completions models (overrides config).",
    )
    # Responses-API / reasoning model params
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["none", "minimal", "low", "medium", "high"],
        help="Reasoning effort for responses-API models (overrides config).",
    )
    parser.add_argument(
        "--verbosity",
        default=None,
        choices=["low", "medium", "high"],
        help="Output verbosity for responses-API models (overrides config).",
    )

    parser.add_argument(
        "--batch-save-interval",
        type=int,
        default=None,
        help="Save progress every N questions (overrides config default of 10).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    gen_cfg = config.get("question_generator", {})

    def _cfg(attr, key, default):
        """Return CLI arg if set, otherwise fall back to config, then hard default."""
        val = getattr(args, attr, None)
        return val if val is not None else gen_cfg.get(key, default)

    model            = _cfg("model",            "model",            "gpt-5-chat")
    provider         = _cfg("provider",         "provider",         "azure")
    include_few_shot = _cfg("few_shot",         "include_few_shot", False)
    temperature      = _cfg("temperature",      "temperature",      0)
    max_tokens       = _cfg("max_tokens",       "max_tokens",       1024)
    reasoning_effort = _cfg("reasoning_effort", "reasoning_effort", "low")
    verbosity        = _cfg("verbosity",        "verbosity",        "low")
    batch_save_interval = _cfg("batch_save_interval", None, 10)

    generator = QuestionGenerator(
        model=model,
        provider=provider,
        include_few_shot=include_few_shot,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
    )

    process_data_file(
        data_file=args.data_file,
        questions_file=args.output_file,
        generator=generator,
        batch_save_interval=batch_save_interval,
    )


if __name__ == "__main__":
    main()
