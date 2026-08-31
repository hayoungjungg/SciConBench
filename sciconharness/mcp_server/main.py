import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Annotated, Optional

import dotenv
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse

logger = logging.getLogger(__name__)

# Suppress noisy low-level MCP framework loggers
logging.getLogger("mcp.server").setLevel(logging.WARNING)
logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)


def _setup_server_logging() -> None:
    """
    If the client passed MCP_SERVER_LOG_FILE, attach a FileHandler to the root
    logger so that mcp_server.* INFO messages (including Jina summarization
    stats, cache hits, etc.) land in the same log file as the client.
    This makes the mcp_client.log more informative with what's going on in the MCP server.

    The FileHandler uses the same timestamp format as the client's handler so
    the entries are interleaved cleanly when reading the log.
    """
    log_file = os.environ.get("MCP_SERVER_LOG_FILE")
    if not log_file:
        return

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.addHandler(handler)
    # Ensure root level doesn't swallow INFO before it reaches the handler.
    # Individual noisy loggers (mcp.server.*) are already set to WARNING above.
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)


_setup_server_logging()

from .apis.semantic_scholar_apis import (
    SemanticScholarSnippetSearchQueryParams,
    search_semantic_scholar_snippets,
)
from .apis.serper_apis import (
    SearchResponse,
    WebpageContentResponse,
    fetch_webpage_content,
    search_serper,
)
from .apis.jina_apis import (
    JinaWebpageResponse,
    fetch_webpage_content_jina,
    require_jina_summarization_ready,
)
from .cache import set_cache_enabled

dotenv.load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")  # SciConBench/.env


def _check_required_api_keys() -> None:
    """Raise EnvironmentError at startup if any required API key is missing."""
    missing = []
    if not os.getenv("SERPER_API_KEY"):
        missing.append("SERPER_API_KEY  # https://serper.dev/")
    if not os.getenv("S2_API_KEY"):
        missing.append("S2_API_KEY      # https://api.semanticscholar.org/")
    if not os.getenv("JINA_API_KEY"):
        missing.append("JINA_API_KEY    # https://jina.ai/reader/")
    if missing:
        raise EnvironmentError(
            "MCP server cannot start: the following API keys are missing from your .env file:\n\n"
            + "\n".join(f"  {k}" for k in missing)
            + "\n\nAdd them to your .env file at the project root and restart."
        )
    # Jina browse tool summarizes via Azure gpt-5-mini — fail fast if that
    # path cannot work (do not start and later return raw multi-MB pages).
    try:
        require_jina_summarization_ready()
    except Exception as exc:
        raise EnvironmentError(
            "MCP server cannot start: Jina summarization is not configured.\n\n"
            f"  {exc}\n\n"
            "Set AZURE_OPENAI_KEY and OPENAI_BASE_URL in your .env "
            "(used only for gpt-5-mini page summarization)."
        ) from exc


mcp = FastMCP(
    "BenchmarkMCP",
    include_tags=os.environ.get("MCP_INCLUDE_TAGS", "search,browse").split(","),
)

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    """
    Check if the MCP server is running.
    curl http://127.0.0.1:8000/health
    """
    return PlainTextResponse("OK")



@mcp.tool(tags={"search"})
def semantic_scholar_snippet_search(
    query: Annotated[str, "Search query string to find within paper content"],
    year: Annotated[
        Optional[str],
        "Optional publication year filter. If not provided, searches all years. Format: single number (e.g., '2024') or range (e.g., '2022-2025', '2020-', '-2023')",
    ] = None,
    paper_ids: Annotated[
        Optional[str], "Comma-separated list of specific paper IDs to search within"
    ] = None,
    venue: Annotated[Optional[str], "Venue filter (e.g., 'ACL', 'EMNLP')"] = None,
    limit: Annotated[int, "Number of snippets to retrieve (default: 10)"] = 10,
) -> dict:
    """
    Focused snippet retrieval from scientific papers using Semantic Scholar API.

    Purpose: Search for specific text snippets within academic papers to find relevant passages, quotes,
    or mentions from scientific literature. Returns focused snippets from existing papers rather than
    full paper metadata.

    Returns:
        Dictionary containing snippets from existing papers with text passages and their source papers.
        Each snippet includes the relevant text passage and metadata about the source paper.

    Example:
        Search for LLM evaluation snippets published between 2021-2025 in Medicine:
        query="efficacy of surgical interventions for neurotrophic keratopathy compared to no treatment", year="2021-2025", limit=8
    """
    # Convert comma-separated string to list if provided
    paper_ids_list = None
    if paper_ids:
        paper_ids_list = [pid.strip() for pid in paper_ids.split(",")]

    # Only include year in params if it's provided (None values are excluded)
    query_params = SemanticScholarSnippetSearchQueryParams(
        query=query,
        year=year,  # None means no year restriction
        paperIds=paper_ids_list,
        venue=venue,
    )

    try:
        results = search_semantic_scholar_snippets(
            query_params=query_params,
            limit=limit,
        )
    except Exception as e:
        logger.warning("semantic_scholar_snippet_search failed (will return empty): %s", e)
        return {"data": [], "error": str(e)}

    return results



@mcp.tool(tags={"search", "necessary"})
def serper_google_webpage_search(
    query: Annotated[str, "Search query string"],
    num_results: Annotated[int, "Number of results to return"] = 10,
    gl: Annotated[
        str,
        "Geolocation - country code to boost search results whose country of origin matches the parameter value",
    ] = "us",
    hl: Annotated[str, "Host language of user interface"] = "en",
    page: Annotated[Optional[int], "Page number (1-indexed). If not provided, fetches 2 pages."] = None,
):
    """
    General web search using Google Search (based on Serper.dev API). Perform general web search to find relevant webpages, articles, and online resources, including academic papers and research publications.

    Returns:
        Dictionary containing web search snippets with the following fields:
        - organic: List of organic search results with title, link, and snippet
        - knowledgeGraph: Knowledge graph information (if available)
        - peopleAlsoAsk: List of related questions
        - relatedSearches: List of related searches
    """
    results = search_serper(
        query=query, search_type="search", gl=gl, hl=hl, page=page
    )

    return results


@mcp.tool(tags={"browse"})
def jina_fetch_webpage_content(
    webpage_url: Annotated[str, "The URL of the webpage to fetch."],
    timeout: Annotated[int, "Request timeout in seconds"] = 30,
) -> JinaWebpageResponse:
    """
    Fetch the full content of a webpage using Jina Reader API with timeout support.
    
    Purpose: Retrieves the full content of a webpage using Jina Reader API for specific URL links. For example, you can use this tool after retrieving search results from serper_google_webpage_search when you identify relevant URL links in the Google Search results (from the 'organic' results list, specifically the 'link' field) or from snippet results from semantic_scholar_snippet_search. This allows you to read the complete article or webpage content beyond just the search snippet.

    Args:
        webpage_url: The URL of the webpage to fetch. Should be a URL link.
        timeout: Request timeout in seconds (default: 30)

    Returns:
        Dictionary containing the webpage content with the following fields:
        - url: The original URL that was fetched
        - title: Page title
        - content: The webpage content as clean text/markdown
        - description: Page description (if available)
        - publishedTime: Published time (if available)
        - metadata: Additional metadata (lang, viewport, etc.)
        - success: Boolean indicating if the fetch was successful
        - error: Error message if fetch failed
    """
    result = fetch_webpage_content_jina(url=webpage_url, timeout=timeout)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MCP server")
    parser.add_argument(
        "--transport",
        type=str,
        default="http",
        choices=["stdio", "http", "sse", "streamable-http"],
        help="Transport protocol to use (default: stdio for local, http for web)",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to (for HTTP transports)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind to (for HTTP transports)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="/mcp",
        help="Path for the HTTP endpoint (default: /mcp)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level for the server",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable API response caching",
    )

    args = parser.parse_args()

    # Validate required API keys before starting
    _check_required_api_keys()

    # Set cache enabled/disabled based on argument
    if args.no_cache:
        set_cache_enabled(False)
    else:
        set_cache_enabled(True)

    # Configure mcp.server logging to suppress INFO level messages
    # Only show WARNING and above
    mcp_server_logger = logging.getLogger("mcp.server")
    mcp_server_logger.setLevel(logging.WARNING)
    mcp_server_lowlevel_logger = logging.getLogger("mcp.server.lowlevel.server")
    mcp_server_lowlevel_logger.setLevel(logging.WARNING)

    # Run the server with the provided arguments
    if args.transport == "stdio":
        # stdio transport doesn't accept host/port/path arguments
        # For stdio, we can omit the transport argument since it's the default
        mcp.run(transport="stdio")
    else:
        # HTTP-based transports accept host/port/path/log_level arguments
        mcp.run(
            transport=args.transport,
            host=args.host,
            port=args.port,
            path=args.path,
            log_level=args.log_level,
        )