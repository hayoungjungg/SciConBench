"""
Single-query CLI for SciConHarness.

Supports interactive (REPL) or non-interactive single-query processing.

Usage:
    # Interactive mode:
    python -m sciconharness.cli_scripts.query_single [openai|gemini|claude|perplexity|azure|openrouter] [--interactive]

    # Non-interactive mode:
    python -m sciconharness.cli_scripts.query_single [openai|gemini|claude|perplexity|azure|openrouter]
        --query "your query" --model "gpt-4" --doi "10.1002/doi1"
        [--publication-date "date"]
        [--cochrane-titles path/to/titles.json]
        [--enable-tool-calling] [--enable-filtering] [--no-save-results]
        [--base-url https://<resource>.services.ai.azure.com/openai/v1/]

    Example (OpenRouter — Kimi K3):
    python -m sciconharness.cli_scripts.query_single openrouter
        --query "What are the benefits and harms of oral antibiotics for otitis media?"
        --model 'moonshotai/kimi-k3' --doi '10.1002.14651858.CD015254.pub2'
        --publication-date "23 October 2023"
        --cochrane-titles "data/cochrane_titles.json"
        --enable-tool-calling --enable-filtering

    Example (Claude):
    python -m sciconharness.cli_scripts.query_single claude
        --query "What are the benefits and harms of oral antibiotics for otitis media?"
        --model 'claude-sonnet-4-5' --doi '10.1002.14651858.CD015254.pub2'
        --publication-date "23 October 2023"
        --cochrane-titles "data/cochrane_titles.json"
        --enable-tool-calling --enable-filtering

    Example (Perplexity with domain filter file):
    python -m sciconharness.cli_scripts.query_single perplexity
        --query "Your query here" --model 'sonar-pro' --doi "10.1002/doi1"
        --filtered-links "path/to/filtered_links.json" --no-enable-tool-calling
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
from sciconharness.utils.perplexity_filtering import build_perplexity_domain_filter


async def main() -> None:
    """Entry point for single-query processing."""
    parser = argparse.ArgumentParser(description="SciConHarness - Single Query")
    parser.add_argument(
        "provider",
        nargs="?",
        default="openai",
        choices=["openai", "gemini", "claude", "perplexity", "azure", "openrouter"],
        help="LLM provider (default: openai)",
    )
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--doi", type=str, help="DOI of the Cochrane review")
    parser.add_argument("--query", type=str,
                        help="Query string (omit for interactive mode)")
    parser.add_argument("--publication-date", type=str,
                        help="Publication date cutoff (e.g. '23 October 2023')")
    parser.add_argument("--cochrane-titles", type=str,
                        help="Path to Cochrane titles JSON file")
    parser.add_argument("--enable-tool-calling", action="store_true", default=True,
                        help="Enable MCP tool calling (default: True)")
    parser.add_argument("--no-enable-tool-calling", action="store_false",
                        dest="enable_tool_calling",
                        help="Disable tool calling (required for Perplexity)")
    parser.add_argument("--enable-filtering", action="store_true", default=True,
                        help="Enable result filtering (default: True)")
    parser.add_argument("--no-enable-filtering", action="store_false",
                        dest="enable_filtering", help="Disable filtering")
    parser.add_argument("--no-save-results", action="store_true",
                        help="Disable saving results to disk")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive REPL mode")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, help="Maximum output tokens")
    parser.add_argument("--domain-filter", nargs="+",
                        help="Perplexity domain filter list. "
                             "Use '-' prefix for denylist (e.g. '-example.com')")
    parser.add_argument("--filtered-links", type=str,
                        help="Path to JSON with 'filtered_links' array "
                             "(Perplexity domain denylist source)")
    parser.add_argument("--max-tool-calls", type=int, default=30,
                        help="Max tool calls for deep-research models (default: 30)")
    parser.add_argument("--api-key", type=str,
                        help="API key (overrides environment variable)")
    parser.add_argument("--base-url", type=str,
                        help="Azure OpenAI / Foundry endpoint URL")
    parser.add_argument("--api-version", type=str,
                        help="Azure API version (classic AzureOpenAI path only)")
    args = parser.parse_args()

    # Perplexity uses built-in search; MCP tool calling must be off so domain
    # filtering and iterative filtering work correctly.
    if args.provider == "perplexity" and args.enable_tool_calling:
        logger.warning(
            "Perplexity uses built-in search - disabling MCP tool calling so "
            "domain filtering works. (--enable-tool-calling -> --no-enable-tool-calling)"
        )
        args.enable_tool_calling = False

    is_interactive = args.interactive or not args.query

    # Resolve Perplexity domain filter before constructing the harness.
    domain_filter = None
    if args.provider == "perplexity":
        domain_filter = build_perplexity_domain_filter(
            domain_filter=args.domain_filter,
            filtered_links_path=args.filtered_links,
        )

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
        save_results=not args.no_save_results,
    )

    async with harness:
        if is_interactive:
            await harness.chat_loop(publication_date=args.publication_date)
        else:
            response, token_usage = await harness.query(
                question=args.query,
                doi=args.doi,
                publication_date=args.publication_date,
                domain_filter=domain_filter,
            )
            print(f"\nResponse:\n{response}")
            print(f"\nToken usage: {token_usage.get('total_tokens', 0)} total tokens")


if __name__ == "__main__":
    asyncio.run(main())
