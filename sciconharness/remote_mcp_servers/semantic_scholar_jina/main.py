import argparse
import json
import logging
import os
import re
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional, Callable, Dict, Any, List, Tuple

import dotenv
from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

# Suppress mcp.server logging
logging.getLogger("mcp.server").setLevel(logging.WARNING)
logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)

# Set up logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Enable debug logging for authentication

# Remove any existing handlers (especially console handlers)
logger.handlers = []
# Prevent propagation to root logger to avoid console output
logger.propagate = False

# Ensure project root is on sys.path so sciconharness is importable when run as a script
current_dir = Path(__file__).resolve().parent
_project_root = current_dir.parent.parent.parent  # SciConBench/
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from sciconharness.remote_mcp_servers.utils import (
    get_global_filter_config,
    set_global_filter_config,
    get_allowed_fetch_urls,
    track_search_urls,
    validate_fetch_url,
    flush_log_handler,
    setup_session_logging,
    load_cochrane_titles,
    get_cochrane_titles_file,
    AuthMiddleware,
    ToolSchemaFixMiddleware,
    check_auth,
)

from sciconharness.mcp_server.apis.semantic_scholar_apis import (
    SemanticScholarSnippetSearchQueryParams,
    search_semantic_scholar_snippets,
)
from sciconharness.mcp_server.apis.jina_apis import JinaWebpageResponse, fetch_webpage_content_jina, normalize_metadata
from sciconharness.mcp_server.cache import set_cache_enabled
from sciconharness.mcp_client.filters.cochrane import CochraneResultFilter

# Load environment variables
dotenv.load_dotenv(_project_root / ".env")

# Validate required API keys at startup
_missing_keys = []
if not os.getenv("S2_API_KEY"):
    _missing_keys.append("S2_API_KEY    # https://api.semanticscholar.org/")
if not os.getenv("JINA_API_KEY"):
    _missing_keys.append("JINA_API_KEY  # https://jina.ai/reader/")
if _missing_keys:
    raise EnvironmentError(
        "Semantic Scholar+Jina MCP server cannot start: the following API keys are missing from your .env file:\n\n"
        + "\n".join(f"  {k}" for k in _missing_keys)
        + "\n\nAdd them to your .env file at the project root and restart."
    )

# Get authentication token from environment
MCP_AUTH_TOKEN = os.getenv("MCP_AUTH_TOKEN")

# Load base title filter list from cochrane_titles.json - FAIL FAST if cannot load
COCHRANE_TITLES_FILE = get_cochrane_titles_file()
try:
    COCHRANE_TITLE_FILTER_LIST = load_cochrane_titles(COCHRANE_TITLES_FILE)
except (FileNotFoundError, ValueError, RuntimeError) as e:
    logger.critical(f"CRITICAL: Cannot start server without Cochrane titles file. {e}")
    sys.exit(1)



server_instructions = """
This MCP server provides academic paper search and full web content fetch capabilities
for OpenAI Deep Research Agents. Use the search tool to find relevant academic papers and 
research publications using Semantic Scholar API, then use the fetch tool
to retrieve complete webpage content using Jina Reader API.
"""

mcp = FastMCP(
    name="SemanticScholarJinaMCP",
    instructions=server_instructions,
    include_tags=os.environ.get("MCP_INCLUDE_TAGS", "search,fetch").split(","),
)




# FilterConfigMiddleware and AuthMiddleware are imported from utils - use them directly


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    """
    Check if the MCP server is running.
    curl http://127.0.0.1:8002/health
    """
    return PlainTextResponse("OK")


async def configure_filter(request: Request) -> JSONResponse:
    """
    Configure filter settings globally.
    
    Client should call this endpoint before making OpenAI API calls.
    This sets global filter configuration that will be used for all subsequent requests
    until a new /configure request is made.
    
    To DISABLE filtering, send empty strings or omit both fields:
    {
        "source_title": "",
        "publication_date": ""
    }
    
    To ENABLE filtering, send both fields:
    {
        "source_title": "Cochrane Review Title",
        "publication_date": "2024-01-15"
    }
    
    Response:
    {
        "message": "Filter configuration set successfully",
        "source_title": "...",
        "publication_date": "...",
        "filtering_enabled": true/false
    }
    """
    try:
        body = await request.json()
        source_title = body.get("source_title")
        publication_date = body.get("publication_date")
        log_dir = body.get("log_dir")
        
        # Normalize empty strings to None
        if source_title == "":
            source_title = None
        if publication_date == "":
            publication_date = None
        
        # Check if we're disabling filtering (both None or empty)
        if not source_title and not publication_date:
            # Disable filtering
            set_global_filter_config(None, None)
            
            # Set up file logging even when filtering is disabled (if log_dir is provided)
            log_file = None
            if log_dir:
                try:
                    log_dir_path = Path(log_dir)
                    log_file = setup_session_logging(logger, log_dir=log_dir_path, source_title=None, publication_date=None)
                    # Verify file was created
                    if not log_file.exists():
                        logger.error(f"ERROR: Log file was not created at {log_file}")
                    else:
                        logger.info(f"✓ Log file created successfully at {log_file}")
                except Exception as e:
                    logger.error(f"ERROR setting up session logging: {e}", exc_info=True)
                    log_file = None
            else:
                logger.warning("⚠ log_dir not provided - remote_mcps.log will not be created")
            
            logger.info("=" * 80)
            logger.info("FILTER CONFIGURATION CLEARED (FILTERING DISABLED)")
            logger.info("=" * 80)
            logger.info("Server: SEMANTIC SCHOLAR+JINA")
            logger.info("Filtering: DISABLED")
            if log_file:
                logger.info(f"Log File: {log_file}")
            logger.info("=" * 80)
            
            flush_log_handler()
            
            return JSONResponse({
                "message": "Filter configuration cleared successfully (filtering disabled)",
                "source_title": None,
                "publication_date": None,
                "filtering_enabled": False,
                "log_file": str(log_file) if log_file else None
            })
        
        # If only one is provided, that's an error
        if not source_title or not publication_date:
            return JSONResponse(
                {
                    "error": "Missing required fields",
                    "message": "Both 'source_title' and 'publication_date' are required to enable filtering. To disable filtering, send empty strings for both."
                },
                status_code=400
            )
        
        # Set global filter configuration
        set_global_filter_config(source_title, publication_date)
        
        # Set up file logging (always create log file if log_dir is provided)
        log_file = None
        if log_dir:
            try:
                log_dir_path = Path(log_dir)
                log_file = setup_session_logging(logger, log_dir=log_dir_path, source_title=source_title, publication_date=publication_date)
                # Verify file was created
                if not log_file.exists():
                    logger.error(f"ERROR: Log file was not created at {log_file}")
                else:
                    logger.info(f"✓ Log file created successfully at {log_file}")
            except Exception as e:
                logger.error(f"ERROR setting up session logging: {e}", exc_info=True)
                log_file = None
        else:
            logger.warning("⚠ log_dir not provided - remote_mcps.log will not be created")
        
        logger.info("=" * 80)
        logger.info("FILTER CONFIGURATION SET")
        logger.info("=" * 80)
        logger.info("Server: SEMANTIC SCHOLAR+JINA")
        logger.info(f"Source Title: {source_title}")
        logger.info(f"Publication Date: {publication_date}")
        if log_file:
            logger.info(f"Log File: {log_file}")
        logger.info("=" * 80)
        
        # Flush log handler to ensure logs are written immediately
        flush_log_handler()
        
        return JSONResponse({
            "message": "Filter configuration set successfully",
            "source_title": source_title,
            "publication_date": publication_date,
            "filtering_enabled": True,
            "log_file": str(log_file) if log_file else None
        })
    except Exception as e:
        logger.error(f"Error configuring filter: {e}", exc_info=True)
        flush_log_handler()
        return JSONResponse(
            {"error": "Internal server error", "message": str(e)},
            status_code=500
        )


# Register the configure route using custom_route decorator
@mcp.custom_route("/configure", methods=["POST"])
async def configure_filter_route(request: Request) -> JSONResponse:
    """Wrapper for configure_filter to use with FastMCP custom_route."""
    return await configure_filter(request)


async def verify_config(request: Request) -> JSONResponse:
    """
    Verify the current filter configuration.
    
    Returns the current configuration that was set via /configure endpoint.
    This allows clients to verify that configuration was set correctly.
    
    Response:
    {
        "source_title": "...",
        "publication_date": "...",
        "configured": true/false
    }
    """
    try:
        config = get_global_filter_config()
        source_title = config.get('source_title')
        publication_date = config.get('publication_date')
        
        is_configured = source_title is not None and publication_date is not None
        
        # Log verification request
        logger.info("=" * 80)
        logger.info("CONFIGURATION VERIFICATION REQUESTED")
        logger.info("=" * 80)
        logger.info("Server: SEMANTIC SCHOLAR+JINA")
        logger.info(f"Current Source Title: {source_title}")
        logger.info(f"Current Publication Date: {publication_date}")
        logger.info(f"Configuration Status: {'CONFIGURED' if is_configured else 'NOT CONFIGURED'}")
        logger.info("=" * 80)
        
        # Flush log handler to ensure logs are written immediately
        flush_log_handler()
        
        return JSONResponse({
            "source_title": source_title,
            "publication_date": publication_date,
            "configured": is_configured
        })
    except Exception as e:
        logger.error(f"Error verifying config: {e}", exc_info=True)
        flush_log_handler()
        return JSONResponse(
            {"error": "Internal server error", "message": str(e)},
            status_code=500
        )


# Register the verify-config route
@mcp.custom_route("/verify-config", methods=["GET"])
async def verify_config_route(request: Request) -> JSONResponse:
    """Wrapper for verify_config to use with FastMCP custom_route."""
    return await verify_config(request)


# _get_session_token_from_context is now get_session_token_from_context from utils


def _create_result_filter() -> Optional[CochraneResultFilter]:
    """Create a result filter from global filter configuration.
    
    Returns:
        CochraneResultFilter if filter configuration is available, None otherwise.
    """
    # Get global filter configuration
    filter_config = get_global_filter_config()
    source_title = filter_config.get('source_title')
    publication_date = filter_config.get('publication_date')
    
    # Debug logging
    logger.info(f"_create_result_filter: source_title = {source_title}")
    logger.info(f"_create_result_filter: publication_date = {publication_date}")
    flush_log_handler()
    
    # Check if configuration is set
    if not source_title or not publication_date:
        logger.info(f"_create_result_filter: Global filter configuration not set. Filtering disabled.")
        logger.info(f"_create_result_filter: source_title={source_title}, publication_date={publication_date}")
        flush_log_handler()
        return None  # Return None instead of raising exception to allow unfiltered operation
    
    # Use the base title filter list (all Cochrane titles) plus the specific source_title
    # The source_title is the specific review being queried, but we filter out all Cochrane titles
    title_filter_list = COCHRANE_TITLE_FILTER_LIST.copy() if COCHRANE_TITLE_FILTER_LIST else []
   
    return CochraneResultFilter(
        title_filter_list=title_filter_list,
        source_title=source_title,
        publication_date=publication_date
    )


# _flush_session_log_handler is now flush_session_log_handler from utils


def _apply_filtering_to_semantic_scholar_results(
    api_results: Dict[str, Any],
    result_filter: CochraneResultFilter
) -> Dict[str, Any]:
    """Apply filtering to Semantic Scholar API results.
    
    Args:
        api_results: Raw API results with "data" key
        result_filter: Filter to apply
        
    Returns:
        Filtered results dictionary.
    """
    logger.info("-" * 80)
    logger.info("RAW RESULTS (before filtering): %d items", len(api_results.get("data", [])))
    for idx, item in enumerate(api_results.get("data", []), start=1):
        paper = item.get("paper", {})
        title = paper.get("title", "")
        corpus_id = paper.get("corpusId", "")
        paper_id = paper.get("paperId", "")
        
        # Extract URL from disclaimer (preferred)
        disclaimer_url = None
        open_access_info = paper.get("openAccessInfo", {})
        disclaimer_text = open_access_info.get("disclaimer", "")
        
        if disclaimer_text:
            import re
            url_pattern = r'https?://[^\s,)]+'
            urls = re.findall(url_pattern, disclaimer_text)
            if urls:
                # Prefer PMC URLs, then DOI URLs, then other URLs
                for found_url in urls:
                    if 'pmc.ncbi.nlm.nih.gov' in found_url:
                        disclaimer_url = found_url
                        break
                    elif 'doi.org' in found_url:
                        disclaimer_url = found_url
                        break
                    elif 'unpaywall.org' not in found_url:
                        disclaimer_url = found_url
                        break
                if not disclaimer_url and urls:
                    disclaimer_url = urls[0]
        
        # Determine final URL: disclaimer URL, or construct from corpus ID/paper ID
        if disclaimer_url:
            url = disclaimer_url
            url_source = "disclaimer"
        elif paper_id:
            url = f"https://www.semanticscholar.org/paper/{paper_id}"
            url_source = "paperId"
        elif corpus_id:
            url = f"https://www.semanticscholar.org/paper/{corpus_id}"
            url_source = "corpusId"
        else:
            url = paper.get("url", "")
            url_source = "paper API" if url else "none"
        
        logger.info("  [RAW] Item %d:", idx)
        logger.info("    Title: %s", title)
        logger.info("    URL (%s): %s", url_source, url)
    
    logger.info("-" * 80)
    logger.info("")
    logger.info("APPLYING FILTER to %d results", len(api_results.get("data", [])))
    logger.info("Filter Configuration:")
    logger.info("  Source Title: %s", result_filter.source_title)
    logger.info("  Publication Date: %s", result_filter.publication_date_cutoff.strftime('%Y-%m-%d') if result_filter.publication_date_cutoff else "None")
    
    # Apply filter
    filtered_result = result_filter.filter(api_results, "semantic_scholar_snippet_search")
    
    # Log filtering results
    original_count = len(api_results.get("data", []))
    filtered_count = len(filtered_result.get("data", []))
    filtered_out_count = original_count - filtered_count
    logger.info("FILTERING RESULTS:")
    logger.info("  Total items before filtering: %d", original_count)
    logger.info("  Items after filtering: %d", filtered_count)
    logger.info("  Items filtered out: %d", filtered_out_count)
    
    if filtered_out_count > 0:
        logger.info("")
        logger.info("FILTERED OUT %d item(s):", filtered_out_count)
        # Find which items were filtered by comparing original and filtered lists
        filtered_corpus_ids = {item.get("paper", {}).get("corpusId", "") for item in filtered_result.get("data", [])}
        for item in api_results.get("data", []):
            corpus_id = item.get("paper", {}).get("corpusId", "")
            if corpus_id not in filtered_corpus_ids:
                title = item.get("paper", {}).get("title", "")
                logger.info("  [FILTERED OUT] %s", title)
                logger.info("    Corpus ID: %s", corpus_id)
                # Log the reason for filtering if available
                paper = item.get("paper", {})
                title_text = paper.get("title", "")
                if title_text and "cochrane" in title_text.lower():
                    logger.info("    Reason: Title contains 'Cochrane'")
                elif result_filter.title_filter and result_filter.title_filter(title_text):
                    logger.info("    Reason: Title matched filter list")
    else:
        logger.info("  No items were filtered out")
    logger.info("")
    
    return filtered_result


def _format_semantic_scholar_results(api_results: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Format Semantic Scholar API results to the required output format.
    
    Args:
        api_results: Raw API results with "data" key
        
    Returns:
        Dictionary with "results" key containing formatted results.
    """
    formatted_results = []
    data = api_results.get("data", [])
    
    for idx, item in enumerate(data):
        paper = item.get("paper", {})
        snippet = item.get("snippet", {})
        
        # Extract URL from disclaimer (preferred source for ID)
        disclaimer_url = None
        open_access_info = paper.get("openAccessInfo", {})
        disclaimer_text = open_access_info.get("disclaimer", "")
        
        if disclaimer_text:
            # Extract URLs from disclaimer text using regex
            import re
            # Match URLs in the disclaimer (prefer PMC, DOI, or other direct URLs over unpaywall API URLs)
            url_pattern = r'https?://[^\s,)]+'
            urls = re.findall(url_pattern, disclaimer_text)
            
            if urls:
                logger.debug(f"Result {idx + 1}: Found {len(urls)} URL(s) in disclaimer: {urls[:3]}...")  # Log first 3 URLs
                # Prefer PMC URLs, then DOI URLs, then other URLs (skip unpaywall API URLs)
                for found_url in urls:
                    if 'pmc.ncbi.nlm.nih.gov' in found_url:
                        disclaimer_url = found_url
                        break
                    elif 'doi.org' in found_url and 'unpaywall.org' not in found_url:
                        disclaimer_url = found_url
                        break
                    elif 'unpaywall.org' not in found_url:
                        disclaimer_url = found_url
                        break
                
                # If no preferred URL found, use first non-unpaywall URL, or first URL as last resort
                if not disclaimer_url:
                    for found_url in urls:
                        if 'unpaywall.org' not in found_url:
                            disclaimer_url = found_url
                            break
                    if not disclaimer_url and urls:
                        disclaimer_url = urls[0]  # Last resort: use first URL even if unpaywall
            else:
                logger.debug(f"Result {idx + 1}: Disclaimer present but no URLs found in: {disclaimer_text[:200]}...")
        else:
            logger.debug(f"Result {idx + 1}: No disclaimer found in openAccessInfo")
        
        # Get corpus ID and paper ID for fallback
        corpus_id = paper.get("corpusId")
        paper_id = paper.get("paperId")
        
        # Determine final URL and ID:
        # 1. Prefer URL from disclaimer (PMC, DOI, etc.)
        # 2. Fallback to Semantic Scholar URL constructed from corpusId or paperId
        if disclaimer_url:
            # Use disclaimer URL as both ID and URL
            result_id = disclaimer_url
            url = disclaimer_url
            logger.info(f"Result {idx + 1}: Using URL from disclaimer as ID and URL: {disclaimer_url}")
        else:
            # Fallback: construct Semantic Scholar URL from paperId or corpusId
            if paper_id:
                url = f"https://www.semanticscholar.org/paper/{paper_id}"
                result_id = url
                logger.info(f"Result {idx + 1}: Constructed Semantic Scholar URL from paperId as ID and URL: {url}")
            elif corpus_id:
                url = f"https://www.semanticscholar.org/paper/{corpus_id}"
                result_id = url
                logger.info(f"Result {idx + 1}: Constructed Semantic Scholar URL from corpusId as ID and URL: {url}")
            else:
                # Last resort: use direct URL from paper API if available
                url = paper.get("url", "")
                if url:
                    result_id = url
                    logger.info(f"Result {idx + 1}: Using direct URL from paper API as ID and URL: {url}")
                else:
                    # Final fallback
                    result_id = f"result_{idx}"
                    url = ""
                    logger.warning(f"Result {idx + 1}: Missing URL and cannot construct from ID, using fallback: {result_id}")
        
        formatted_results.append({
            "id": result_id,  # ID is always a URL (from disclaimer or constructed from corpusId/paperId)
            "title": paper.get("title", ""),
            "url": url,  # URL matches ID (from disclaimer or constructed from corpusId/paperId)
            "snippet": snippet if isinstance(snippet, dict) else {"text": str(snippet)},
            "position": idx + 1,
        })
    
    return {"results": formatted_results}

@mcp.tool(tags={"search"})
def search(
    query: Annotated[str, "Search query string to find within paper content"],
) -> dict:
    """
    Focused snippet retrieval from scientific papers using Semantic Scholar API.

    Purpose: Search for specific text snippets within academic papers to find relevant passages, quotes,
    or mentions from scientific literature. Returns focused snippets from existing papers rather than
    full paper metadata.

    Returns:
        Dictionary containing snippets from existing papers with text passages and their source papers.
        Each snippet includes the relevant text passage and metadata about the source paper.
    """
    # Use default values for internal parameters (not exposed in schema)
    year = None
    paper_ids = None
    venue = None
    limit = 10
    
    logger.info("=" * 80)
    logger.info("SEMANTIC SCHOLAR+JINA SEARCH - Tool called")
    logger.info("=" * 80)
    logger.info("Query: %s", query)
    logger.info("Parameters: year=%s, paper_ids=%s, venue=%s, limit=%d", year, paper_ids, venue, limit)
    
    # Check filter status
    result_filter = _create_result_filter()
    if result_filter:
        logger.info("FILTER STATUS: ACTIVE")
        logger.info("  Filtering for Source Title: %s", result_filter.source_title)
        logger.info("  Filtering for Publication Date: %s", result_filter.publication_date_cutoff.strftime('%Y-%m-%d') if result_filter.publication_date_cutoff else "None")
    else:
        logger.info("FILTER STATUS: INACTIVE (no filter configuration)")
    logger.info("")
    
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
    
    # Fetch results from API
    api_results = search_semantic_scholar_snippets(
        query_params=query_params,
        limit=limit,
    )
    
    # Apply filtering if available
    if result_filter:
        api_results = _apply_filtering_to_semantic_scholar_results(api_results, result_filter)
    else:
        # Log raw results when filtering is disabled (same format as when filtering is enabled)
        logger.info("-" * 80)
        logger.info("RAW RESULTS (no filtering): %d items", len(api_results.get("data", [])))
        for idx, item in enumerate(api_results.get("data", []), start=1):
            paper = item.get("paper", {})
            title = paper.get("title", "")
            corpus_id = paper.get("corpusId", "")
            paper_id = paper.get("paperId", "")
            
            # Extract URL from disclaimer (preferred)
            disclaimer_url = None
            open_access_info = paper.get("openAccessInfo", {})
            disclaimer_text = open_access_info.get("disclaimer", "")
            
            if disclaimer_text:
                import re
                url_pattern = r'https?://[^\s,)]+'
                urls = re.findall(url_pattern, disclaimer_text)
                if urls:
                    # Prefer PMC URLs, then DOI URLs, then other URLs
                    for found_url in urls:
                        if 'pmc.ncbi.nlm.nih.gov' in found_url:
                            disclaimer_url = found_url
                            break
                        elif 'doi.org' in found_url:
                            disclaimer_url = found_url
                            break
                        elif 'unpaywall.org' not in found_url:
                            disclaimer_url = found_url
                            break
                    if not disclaimer_url and urls:
                        disclaimer_url = urls[0]
            
            # Determine final URL: disclaimer URL, or construct from corpus ID/paper ID
            if disclaimer_url:
                url = disclaimer_url
                url_source = "disclaimer"
            elif paper_id:
                url = f"https://www.semanticscholar.org/paper/{paper_id}"
                url_source = "paperId"
            elif corpus_id:
                url = f"https://www.semanticscholar.org/paper/{corpus_id}"
                url_source = "corpusId"
            else:
                url = paper.get("url", "")
                url_source = "paper API" if url else "none"
            
            logger.info("  [RAW] Item %d:", idx)
            logger.info("    Title: %s", title)
            logger.info("    URL (%s): %s", url_source, url)
        logger.info("-" * 80)
        logger.info("")
    
    # Format results to required output format
    formatted_results = _format_semantic_scholar_results(api_results)
    
    # Track URLs from search results so they can be fetched later
    results_list = formatted_results.get("results", [])
    # Track URLs from search results
    urls = [result.get("url", "") for result in results_list if result.get("url")]
    track_search_urls(urls)
    if urls:
        allowed_urls = get_allowed_fetch_urls()
        logger.info("Tracked %d URLs for fetch validation", len(allowed_urls))
    
    logger.info("")
    logger.info("After filtering: %d results remaining (requested: %d, fetched: %d)", 
              len(results_list), limit, len(api_results.get("data", [])))
    logger.info("-" * 80)
    allowed_urls = get_allowed_fetch_urls()
    if allowed_urls:
        logger.info("Tracked %d URLs for fetch validation", len(allowed_urls))
    logger.info("=" * 80)
    logger.info("SEMANTIC SCHOLAR+JINA SEARCH - Tool completed")
    logger.info("=" * 80)
    logger.info("Total results returned: %d", len(results_list))
    logger.info("=" * 80)
    logger.info("")
    
    # Log filtering summary if filter was used
    if result_filter and hasattr(result_filter, 'log_filtering_summary'):
        result_filter.log_filtering_summary()
    
    # Flush log handler to ensure logs are written immediately
    flush_log_handler()
    
    return formatted_results


@mcp.tool(tags={"fetch", "browse"})
async def fetch(id: str) -> Dict[str, Any]:
    """
    Fetch the full content of a webpage using Jina Reader API with timeout support.
    
    Purpose: Retrieves the full content of a webpage using Jina Reader API by the specific URL links/ID. For example, you can use this tool after retrieving search results from semantic_scholar_snippet_search when you identify relevant URL links in the Semantic Scholar results. This allows you to read the complete article or webpage content beyond just the search snippet.

    Args:
        id: A unique identifier for the search document (typically a URL link).

    Returns:
        Dictionary containing the document with the following fields:
        - id: Unique identifier for the document
        - title: Document title
        - text: Full text content of the document
        - url: URL to the document
        - metadata: Optional metadata about the document
    """
    if not id:
        raise ValueError("Document ID is required")
    
    # Use default timeout value (not exposed in schema)
    timeout = 30
    
    logger.info("=" * 80)
    logger.info("SEMANTIC SCHOLAR+JINA FETCH - Tool called")
    logger.info("=" * 80)
    logger.info("ID/URL: %s", id)
    logger.info("Timeout: %d seconds", timeout)
    
    # Log if URL was not returned from a search result (for monitoring, but don't block)
    is_valid = validate_fetch_url(id)
    allowed_urls = get_allowed_fetch_urls()
    if not is_valid:
        logger.warning("=" * 80)
        logger.warning("WARNING: Fetching URL that was NOT returned from search results")
        logger.warning("=" * 80)
        logger.warning("Requested URL: %s", id)
        logger.warning("Allowed URLs count: %d", len(allowed_urls))
        logger.warning("This URL was NOT returned from any search results.")
        logger.warning("=" * 80)
        logger.warning("")
    else:
        logger.info("URL validated - found in search results")
    
    # Check filter status and create filter check callback to use BEFORE summarization
    result_filter = _create_result_filter()
    if result_filter:
        logger.info("FILTER STATUS: ACTIVE")
        logger.info("  Filtering for Source Title: %s", result_filter.source_title)
        logger.info("  Filtering for Publication Date: %s", result_filter.publication_date_cutoff.strftime('%Y-%m-%d') if result_filter.publication_date_cutoff else "None")
    else:
        logger.info("FILTER STATUS: INACTIVE (no filter configuration)")
    logger.info("")
    
    # Create filter check callback to use BEFORE summarization
    should_filter_callback = None
    if result_filter:
        logger.info("Creating filter callback for Jina fetch (before summarization)")
        def filter_check(content: str, title: str, url: str) -> Tuple[bool, Optional[str]]:
            """Check if content should be filtered before summarization."""
            logger.info(f"Filter callback invoked: content_length={len(content) if content else 0}, title={title[:80] if title else 'None'}...")
            try:
                result = result_filter._should_filter_jina_content(content, title, url)
                logger.info(f"Filter callback result: {result}")
                return result
            except Exception as e:
                logger.error(f"Error in filter callback: {e}", exc_info=True)
                return False, None
        should_filter_callback = filter_check
        logger.info("Filter callback created and will be passed to fetch_webpage_content_jina")
    else:
        logger.warning("No result filter available - filter callback will not be created")
    
    result = fetch_webpage_content_jina(url=id, timeout=timeout, should_filter_callback=should_filter_callback)
    
    # Check if fetch was successful
    if not result.get("success", False):
        # Log fetch failure
        logger.info("JINA FETCH FAILED")
        logger.info("  URL: %s", id)
        logger.info("  Title: %s", result.get("title", ""))
        logger.info("  Error: %s", result.get("error", "Unknown error"))
        logger.info("=" * 80)
        logger.info("SEMANTIC SCHOLAR+JINA FETCH - Tool completed (with error)")
        logger.info("=" * 80)
        logger.info("")
        flush_log_handler()
        # Return error in the required format
        return {
            "id": id,
            "title": result.get("title", ""),
            "text": "",
            "url": id,
            "metadata": {
                "error": result.get("error", "Unknown error"),
                "success": False
            }
        }
    
    # Check if content was filtered (empty content means filtered)
    # Filtering already happened BEFORE summarization via callback
    if result_filter:
        if result.get("content") == "" and result.get("success", False):
            logger.info("JINA CONTENT FILTERED OUT (before summarization)")
            logger.info("  URL: %s", id)
            logger.info("  Title: %s", result.get("title", ""))
        else:
            logger.info("JINA CONTENT PASSED FILTER")
            logger.info("  URL: %s", id)
            logger.info("  Title: %s", result.get("title", ""))
            logger.info("  Content length: %d characters", len(result.get("content", "")))
    else:
        # Log fetch results when filtering is disabled (same format as when filtering is enabled)
        logger.info("JINA FETCH COMPLETED (no filtering)")
        logger.info("  URL: %s", id)
        logger.info("  Title: %s", result.get("title", ""))
        logger.info("  Content length: %d characters", len(result.get("content", "")))
        if result.get("description"):
            logger.info("  Description: %s", result.get("description", "")[:200] + ("..." if len(result.get("description", "")) > 200 else ""))
        if result.get("publishedTime"):
            logger.info("  Published Time: %s", result.get("publishedTime", ""))
    
    # Log fetch completion
    logger.info("=" * 80)
    logger.info("SEMANTIC SCHOLAR+JINA FETCH - Tool completed")
    logger.info("=" * 80)
    logger.info("")
    flush_log_handler()
    
    # Transform to required format: {"id": "...", "title": "...", "text": "...", "url": "...", "metadata": {...}}
    # Normalize metadata to ensure type safety (handles viewport as list -> string conversion)
    metadata = normalize_metadata(result.get("metadata", {}))
    
    # Add additional fields to metadata if available
    if result.get("description"):
        metadata["description"] = result.get("description")
    if result.get("publishedTime"):
        metadata["publishedTime"] = result.get("publishedTime")
    metadata["success"] = True
    
    return {
        "id": id,  # Use ID/URL as unique identifier
        "title": result.get("title", ""),
        "text": result.get("content", ""),
        "url": result.get("url", id),
        "metadata": metadata
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Semantic Scholar+Jina Remote MCP server")
    parser.add_argument(
        "--transport",
        type=str,
        default="http",
        choices=["http", "sse", "streamable-http"],
        help="Transport protocol to use (default: http for remote MCP server)",
    )
    parser.add_argument(
        "--port", type=int, default=8002, help="Port to bind to (for HTTP transports)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind to (for HTTP transports, default: 0.0.0.0 for remote access)",
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

    # No middleware needed - we use global variables set by /configure endpoint
    logger.info("Using global filter configuration (set via /configure endpoint)")
    
    # Add configure and verify-config routes directly to the Starlette app if custom_route didn't work
    def add_custom_routes():
        """Try to add custom routes directly to the Starlette app."""
        try:
            from starlette.routing import Route
            
            app = None
            if hasattr(mcp, 'app') and isinstance(mcp.app, Starlette):
                app = mcp.app
            elif hasattr(mcp, '_app') and isinstance(mcp._app, Starlette):
                app = mcp._app
            elif hasattr(mcp, 'server') and hasattr(mcp.server, 'app'):
                app = mcp.server.app
            
            if app:
                # Check if routes already exist
                existing_routes = [route.path for route in app.routes if hasattr(route, 'path')]
                
                if "/configure" not in existing_routes:
                    app.add_route("/configure", configure_filter, methods=["POST"])
                    logger.info("✓ Configure route added directly to Starlette app")
                
                if "/verify-config" not in existing_routes:
                    app.add_route("/verify-config", verify_config, methods=["GET"])
                    logger.info("✓ Verify-config route added directly to Starlette app")
                
                return True
        except Exception as e:
            logger.warning(f"Could not add custom routes directly: {e}")
        return False
    
    # Try to add custom routes before run
    add_custom_routes()
    
    # Fix tool schemas directly by modifying tool schemas
    # This is more reliable than middleware since it modifies the source
    def fix_tool_schemas_directly():
        """Fix tool schemas to match OpenAI MCP requirements.
        
        Ensures:
        1. Schema has type: "object" at root
        2. Schema has properties object
        3. additionalProperties: false is set
        """
        try:
            if hasattr(mcp, '_tool_manager') and hasattr(mcp._tool_manager, '_tools'):
                tools_fixed = 0
                for tool_name, tool in mcp._tool_manager._tools.items():
                    # FastMCP stores schema in parameters (not inputSchema)
                    # We need to fix parameters, which will be used when converting to MCP format
                    schema = None
                    schema_attr = None
                    
                    # FastMCP uses 'parameters' attribute (not inputSchema)
                    if hasattr(tool, 'parameters'):
                        schema_attr = 'parameters'
                        if isinstance(tool.parameters, dict):
                            schema = tool.parameters
                        else:
                            logger.debug(f"Tool '{tool_name}': parameters exists but is not a dict: {type(tool.parameters)}")
                    
                    # Also check inputSchema if it exists (for compatibility)
                    if schema is None and hasattr(tool, 'inputSchema'):
                        schema_attr = 'inputSchema'
                        if isinstance(tool.inputSchema, dict):
                            schema = tool.inputSchema
                        else:
                            logger.debug(f"Tool '{tool_name}': inputSchema exists but is not a dict: {type(tool.inputSchema)}")
                    
                    if schema:
                        # Log original schema structure for debugging
                        logger.info(f"Tool '{tool_name}': Original schema: {json.dumps(schema, indent=2)}")
                        logger.debug(f"Tool '{tool_name}': Original schema keys: {list(schema.keys())}")
                        
                        # Ensure schema has type: "object" at root
                        if "type" not in schema:
                            schema["type"] = "object"
                            logger.debug(f"Tool '{tool_name}': Added type=object")
                        
                        # Ensure schema has properties (even if empty)
                        if "properties" not in schema:
                            schema["properties"] = {}
                            logger.debug(f"Tool '{tool_name}': Added properties={{}}")
                        
                        # Ensure required fields are set if properties exist
                        if "properties" in schema and schema["properties"] and "required" not in schema:
                            # Set all properties as required
                            schema["required"] = list(schema["properties"].keys())
                            logger.debug(f"Tool '{tool_name}': Added required fields: {schema['required']}")
                        
                        # Set additionalProperties to false (required by OpenAI)
                        schema["additionalProperties"] = False
                        
                        # Log final schema structure
                        logger.info(f"Tool '{tool_name}': Fixed schema: {json.dumps(schema, indent=2)}")
                        
                        tools_fixed += 1
                        logger.info(f"✓ Fixed schema for tool '{tool_name}' (via {schema_attr}): type=object, additionalProperties=false")
                    else:
                        logger.warning(f"Tool '{tool_name}': Could not find inputSchema or parameters attribute")
                        # Log all attributes for debugging
                        logger.debug(f"Tool '{tool_name}': Available attributes: {dir(tool)}")
                
                if tools_fixed > 0:
                    logger.info(f"✓ Fixed schemas for {tools_fixed} tools for OpenAI compatibility")
                    return True
                else:
                    logger.warning("No tools found to fix")
            else:
                logger.warning("MCP tool manager or tools not found")
                if hasattr(mcp, '_tool_manager'):
                    logger.debug(f"MCP has _tool_manager: {type(mcp._tool_manager)}")
                    if hasattr(mcp._tool_manager, '_tools'):
                        logger.debug(f"_tools type: {type(mcp._tool_manager._tools)}")
        except Exception as e:
            logger.error(f"Could not fix tool schemas directly: {e}", exc_info=True)
        return False
    
    # Fix tool schemas before server starts
    # This must run after all tools are registered but before server starts
    schema_fix_result = fix_tool_schemas_directly()
    if not schema_fix_result:
        logger.warning("WARNING: Tool schema fix may not have been applied. Check logs above.")
    else:
        logger.info("✓ Tool schema fix completed successfully")
    
    # Add middleware to fix tool schemas and authentication
    def add_middlewares():
        """Add middleware to FastMCP app for schema fixing and authentication."""
        try:
            app = None
            if hasattr(mcp, 'app') and isinstance(mcp.app, Starlette):
                app = mcp.app
            elif hasattr(mcp, '_app') and isinstance(mcp._app, Starlette):
                app = mcp._app
            elif hasattr(mcp, 'server') and hasattr(mcp.server, 'app'):
                app = mcp.server.app
            
            if app:
                # Add schema fix middleware as backup (runs last, modifies responses)
                app.add_middleware(ToolSchemaFixMiddleware)
                logger.info("✓ ToolSchemaFixMiddleware added (backup)")
                
                # Add authentication middleware if token is configured
                if MCP_AUTH_TOKEN:
                    app.add_middleware(AuthMiddleware, auth_token=MCP_AUTH_TOKEN)
                    logger.info("✓ AuthMiddleware added")
                
                return True
        except Exception as e:
            logger.warning(f"Could not add middleware: {e}")
        return False
    
    # Try to add middleware before run (as backup)
    add_middlewares()
    
    # Run the server with the provided arguments
    # Remote MCP servers should use HTTP transport
    mcp.run(
        transport=args.transport,
        host=args.host,
        port=args.port,
        path=args.path,
        log_level=args.log_level,
    )

