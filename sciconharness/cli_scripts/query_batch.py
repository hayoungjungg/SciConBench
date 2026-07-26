"""
Batch processing CLI for SciConHarness.

Processes multiple queries from DOI-to-question dictionaries.  Each DOI gets
its own directory for logs and results.  Already-processed DOIs are skipped
automatically so interrupted runs can be resumed safely.

Usage:
    python -m sciconharness.cli_scripts.query_batch [openai|gemini|claude|perplexity|azure|openrouter] \\
        --doi-questions path/to/doi_questions.json \\
        [--doi-dates path/to/doi_dates.json] \\
        --model "gpt-4" \\
        [--cochrane-titles path/to/titles.json] \\
        [--enable-tool-calling] \\
        [--enable-filtering] \\
        [--no-save-results] \\
        [--temperature 0.2] \\
        [--max-tokens 8192] \\
        [--base-url https://<resource>.services.ai.azure.com/openai/v1/] \\
        [--filtered-links-json path/to/top_18_filtered_links_from_logs.json]

    Example (OpenRouter — GLM-5.2):
    python -m sciconharness.cli_scripts.query_batch openrouter \\
        --model z-ai/glm-5.2 \\
        --doi-questions data/doi_questions.json \\
        --doi-dates data/filter_data/doi_dates.json \\
        --cochrane-titles data/filter_data/cochrane_titles.json \\
        --enable-tool-calling --enable-filtering

    Note: Perplexity has built-in search, so tool calling is disabled automatically.

Parameters:
    --doi-questions:       Required. JSON file mapping DOI to questions (dict[str, str])
    --doi-dates:           Optional. JSON file mapping DOI to publication dates
    --filtered-links-json: Optional. Perplexity per-DOI domain denylist JSON
    --temperature:         Optional. Sampling temperature
    --max-tokens:          Optional. Maximum output tokens
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Add project root to sys.path for editable installs
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from sciconharness.harness import SciConHarness
from sciconharness.utils.query_utils import load_doi_dict
from sciconharness.mcp_client.llm_providers.perplexity_provider import (
    prepare_doi_domain_filters_from_json,
)


async def main() -> None:
    """Entry point for batch processing."""
    parser = argparse.ArgumentParser(description="SciConHarness — Batch Processing")
    parser.add_argument(
        "provider",
        nargs="?",
        default="openai",
        choices=["openai", "gemini", "claude", "perplexity", "azure", "openrouter"],
        help="LLM provider (default: openai)",
    )
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--doi-questions", type=str, required=True,
                        help="Path to JSON file mapping DOI → question")
    parser.add_argument("--doi-dates", type=str,
                        help="Path to JSON file mapping DOI → publication date")
    parser.add_argument("--cochrane-titles", type=str,
                        help="Path to Cochrane titles JSON file")
    parser.add_argument("--enable-tool-calling", action="store_true", default=True,
                        help="Enable MCP tool calling (default: True)")
    parser.add_argument("--no-enable-tool-calling", action="store_false",
                        dest="enable_tool_calling", help="Disable tool calling")
    parser.add_argument("--enable-filtering", action="store_true", default=True,
                        help="Enable result filtering (default: True)")
    parser.add_argument("--no-enable-filtering", action="store_false",
                        dest="enable_filtering", help="Disable filtering")
    parser.add_argument("--no-save-results", action="store_true",
                        help="Disable saving results to disk")
    parser.add_argument("--max-format-retries", type=int, default=3,
                        help="Max attempts if response is not well-formatted (default: 3)")
    parser.add_argument("--min-conclusion-length", type=int, default=20,
                        help="Min chars inside [[[...]]] to be well-formatted (default: 20)")
    parser.add_argument("--log-dir", type=str,
                        help="Custom base log/output directory")
    parser.add_argument("--temperature", type=float,
                        help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int,
                        help="Maximum output tokens")
    parser.add_argument("--filtered-links-json", type=str,
                        help="Path to top_18_filtered_links JSON for Perplexity domain filtering")
    parser.add_argument("--max-tool-calls", type=int, default=30,
                        help="Max tool calls for deep-research models (default: 30)")
    parser.add_argument("--max-samples", type=int,
                        help="Cap on DOIs to process (useful for smoke tests)")
    parser.add_argument("--api-key", type=str,
                        help="API key (overrides environment variable)")
    parser.add_argument("--base-url", type=str,
                        help="Azure OpenAI / Foundry endpoint URL")
    parser.add_argument("--api-version", type=str,
                        help="Azure API version (default: 2025-04-01-preview)")
    args = parser.parse_args()

    # Apply custom log directory globally before anything else so all
    # subsequent setup_directories() calls read the correct root.
    if args.log_dir:
        from sciconharness.mcp_client.utils.utils import set_custom_log_dir
        custom_log_path = Path(args.log_dir).resolve()
        set_custom_log_dir(custom_log_path)
        logger.info("Using custom log directory: %s", custom_log_path)

    # Perplexity uses built-in search; MCP tool calling must be off for its
    # domain filter / iterative filtering to work.
    if args.provider == "perplexity" and args.enable_tool_calling:
        logger.warning(
            "Perplexity uses built-in search — disabling MCP tool calling so "
            "domain filtering works. (--enable-tool-calling → --no-enable-tool-calling)"
        )
        args.enable_tool_calling = False

    # ── load DOI questions ────────────────────────────────────────────────────
    doi_questions_file = Path(args.doi_questions)
    if not doi_questions_file.exists():
        logger.error("DOI questions file not found: %s", doi_questions_file)
        return

    # ── resolve Perplexity domain filter from file ────────────────────────────
    doi_to_domain_filter = None
    if args.filtered_links_json and args.provider == "perplexity":
        filtered_links_file = Path(args.filtered_links_json)
        if not filtered_links_file.exists():
            logger.warning("Filtered links JSON not found: %s", filtered_links_file)
        else:
            try:
                logger.info("Loading filtered links from %s", filtered_links_file)
                doi_to_domain_filter = prepare_doi_domain_filters_from_json(
                    filtered_links_file,
                    max_links_per_doi=18,
                    max_total_links=20,
                )
                logger.info(
                    "Prepared domain filters for %d DOIs", len(doi_to_domain_filter)
                )
            except Exception as exc:
                logger.warning("Error loading filtered links JSON: %s", exc)

    # ── create harness ────────────────────────────────────────────────────────
    harness = SciConHarness(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        api_version=args.api_version,
        enable_tools=args.enable_tool_calling,
        enable_filtering=args.enable_filtering,
        cochrane_titles=Path(args.cochrane_titles) if args.cochrane_titles else None,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_tool_calls=args.max_tool_calls,
        max_format_retries=args.max_format_retries,
        min_conclusion_length=args.min_conclusion_length,
        save_results=not args.no_save_results,
    )

    # ── load question / date dicts, run batch ─────────────────────────────────
    doi_to_question = load_doi_dict(doi_questions_file)
    logger.info("Loaded %d DOI-question pairs from %s",
                len(doi_to_question), doi_questions_file)

    doi_to_date = None
    if args.doi_dates:
        doi_dates_file = Path(args.doi_dates)
        if not doi_dates_file.exists():
            logger.warning("DOI dates file not found: %s", doi_dates_file)
        else:
            doi_to_date = load_doi_dict(doi_dates_file)
            logger.info("Loaded %d DOI-date pairs from %s",
                        len(doi_to_date), doi_dates_file)

    async with harness:
        await harness.query_batch(
            doi_to_question,
            doi_to_date,
            max_samples=args.max_samples,
            doi_to_domain_filter=doi_to_domain_filter,
        )


if __name__ == "__main__":
    asyncio.run(main())
