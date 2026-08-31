import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Callable, Tuple

import dotenv
import requests
from requests.exceptions import ReadTimeout, RequestException, Timeout, ConnectionError, ConnectTimeout
from typing_extensions import TypedDict

from openai import AzureOpenAI

logger = logging.getLogger(__name__)
# Also try to get __main__ logger for remote MCP servers (so logs go to the same file)
_main_logger = logging.getLogger("__main__")


# Handle both module import and direct execution
from ..cache import cached, DEFAULT_CACHE, is_cache_enabled

# Try to load .env from multiple possible locations
# apis/ -> mcp_server/ -> sciconharness/ -> SciConBench/ (project root)
env_paths = [
    ".env",  # Current directory (works when CWD is project root)
    Path(__file__).resolve().parent.parent.parent.parent / ".env",  # SciConBench/.env
]
for env_path in env_paths:
    if Path(env_path).exists():
        dotenv.load_dotenv(env_path)
        break
else:
    dotenv.load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY")
TIMEOUT = int(os.getenv("API_TIMEOUT", 30))

# Project root .env — used as a durable source for Azure summarization creds
# even when process env was cleared (e.g. OpenRouter-only resume scripts).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PROJECT_ENV_FILE = _PROJECT_ROOT / ".env"


def _azure_openai_creds_for_summarization() -> Tuple[Optional[str], Optional[str]]:
    """Resolve Azure OpenAI credentials for Jina page summarization.

    Always uses ``AZURE_OPENAI_KEY`` + ``OPENAI_BASE_URL`` (never OpenRouter
    or other provider keys). Checks process env first, then re-reads the
    project ``.env`` so summarization still works if those vars were popped
    from the environment for query routing.
    """
    api_key = os.getenv("AZURE_OPENAI_KEY")
    endpoint = os.getenv("OPENAI_BASE_URL")
    if api_key and endpoint:
        return api_key, endpoint

    if _PROJECT_ENV_FILE.is_file():
        file_vals = dotenv.dotenv_values(_PROJECT_ENV_FILE)
        api_key = api_key or (file_vals.get("AZURE_OPENAI_KEY") or None)
        endpoint = endpoint or (file_vals.get("OPENAI_BASE_URL") or None)
        if api_key:
            api_key = api_key.strip().strip("'\"")
        if endpoint:
            endpoint = endpoint.strip().strip("'\"")

    return api_key or None, endpoint or None


class JinaSummarizationError(RuntimeError):
    """Raised only when Azure OpenAI creds for Jina summarization are missing.

    Transient Azure API failures are retried inside ``summarize_content`` and
    fall back to truncated content — they do not raise this error.
    """


class JinaMetadata(TypedDict, total=False):
    lang: str
    viewport: str


class JinaWebpageResponse(TypedDict, total=False):
    url: str
    title: str
    content: str
    description: str
    publishedTime: str
    metadata: JinaMetadata
    success: bool
    error: str


def require_jina_summarization_ready() -> Tuple[str, str]:
    """Validate Azure OpenAI creds for Jina summarization; raise if missing.

    Returns ``(api_key, endpoint)`` on success. This is the only hard stop —
    without these keys pages would be returned unsummarized.
    """
    api_key, endpoint = _azure_openai_creds_for_summarization()
    if not api_key or not endpoint:
        msg = (
            "Jina webpage summarization requires Azure OpenAI credentials "
            "(AZURE_OPENAI_KEY + OPENAI_BASE_URL in env or project .env). "
            "Refusing to continue without summarization."
        )
        logger.error(msg)
        _main_logger.error(msg)
        print(f"ERROR: {msg}", file=sys.stderr)
        raise JinaSummarizationError(msg)
    return api_key, endpoint


def summarize_content(
    content: str,
    model: str = "gpt-5-mini",
    *,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> str:
    """
    Summarize content using Azure OpenAI (gpt-5-mini) focusing on main details.

    Content is always truncated to MAX_CONTENT_LENGTH before being sent to the
    API so that the summarizer never hits a context-length error.

    Credentials are always ``AZURE_OPENAI_KEY`` / ``OPENAI_BASE_URL`` (env or
    project ``.env``), independent of whichever key the query model uses.

    Hard-stops (raises ``JinaSummarizationError``) only when Azure creds are
    missing. Transient Azure API failures are retried, then fall back to the
    truncated page text with a warning.

    Args:
        content: The content to summarize
        model: Azure OpenAI model to use (default: gpt-5-mini)
        max_retries: Attempts for transient Azure failures (default: 3)
        retry_delay: Base delay seconds between retries (exponential backoff)
    """
    original_length = len(content)
    logger.info(f"[summarize_content] Starting summarization with {model}. Original content length: {original_length:,} characters")

    # gpt-5-mini context window ≈ 400K tokens
    # We stay well under that limit so reasoning tokens and prompt overhead
    # never push us over the edge.
    MAX_CONTENT_LENGTH = 375_000

    # Always truncate before sending — this is the primary safeguard against
    # context-length errors in the summarizer.
    if original_length > MAX_CONTENT_LENGTH:
        logger.warning(
            f"[summarize_content] Content too long ({original_length:,} chars), "
            f"truncating to {MAX_CONTENT_LENGTH:,} chars before summarization"
        )
        content = (
            content[:MAX_CONTENT_LENGTH]
            + f"\n\n[Note: Content was truncated from {original_length:,} to "
            f"{MAX_CONTENT_LENGTH:,} characters before summarization.]"
        )
        logger.info(f"[summarize_content] Content truncated to {len(content):,} characters (including truncation note)")

    # Hard stop only if Azure is not configured — otherwise we would silently
    # ship unsummarized Jina pages into the query context.
    azure_api_key, azure_endpoint = require_jina_summarization_ready()

    # Responses API requires api-version 2025-03-01-preview or later
    required_version = "2025-03-01-preview"
    client = AzureOpenAI(
        api_version=required_version,
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
    )

    prompt = f"""Summarize the following web content, focusing only on the main details and key information. Preserve important facts, numbers, dates, and conclusions. Aim to filter out any noisy characters (e.g., HTML tags, social media links, random strings, etc.) and outputting only important information. Specific details, including but not limited to metrics, deltas, definitions, settings, limitations, and citations and references should be preserved. Make sure not to lose any key information. 

Content to summarize:
{content}"""

    reasoning = {
        "effort": "medium",
        "summary": "auto",
    }
    text = {
        "verbosity": "medium",
    }
    input_list = [
        {"role": "user", "content": prompt}
    ]

    last_error: Optional[BaseException] = None
    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                instructions="You are a helpful assistant that creates summaries of web content focusing on main details.",
                model=model,
                input=input_list,
                reasoning=reasoning,
                text=text,
            )

            summary = str(response.output_text) if response.output_text else None
            if not summary:
                for item in getattr(response, "output", []):
                    if hasattr(item, "text"):
                        summary = str(item.text)
                        break

            if not summary or not str(summary).strip():
                raise RuntimeError(
                    f"Azure {model} returned an empty summary "
                    f"(input was {original_length:,} characters)."
                )

            final_summary = str(summary).strip()
            summarized_length = len(final_summary)
            reduction = original_length - summarized_length
            reduction_pct = (reduction / original_length * 100) if original_length > 0 else 0
            logger.info(
                f"[summarize_content] Summarization completed. Final length: {summarized_length:,} characters "
                f"(reduced by {reduction:,} chars, {reduction_pct:.1f}%)"
            )
            return final_summary

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = retry_delay * (2 ** attempt)
                logger.warning(
                    "[summarize_content] Azure summarization failed "
                    "(attempt %d/%d): %s; retrying in %.1fs",
                    attempt + 1, max_retries, e, wait,
                )
                time.sleep(wait)
                continue
            logger.error(
                f"[summarize_content] Summarization failed after {max_retries} attempts: {e}. "
                f"Returning truncated content ({len(content):,} chars)."
            )
            _main_logger.error(
                f"[summarize_content] Summarization failed after {max_retries} attempts: {last_error}. "
                f"Returning truncated content ({len(content):,} chars)."
            )
            return content

    return content


def normalize_metadata(metadata):
    """
    Normalize metadata to ensure type safety for JinaMetadata TypedDict.
    
    This function converts viewport from list to string format and ensures
    all required fields are properly typed. Used by both fetch_webpage_content_jina
    and remote MCP servers.
    
    Args:
        metadata: Dictionary containing metadata (may have viewport as list)
        
    Returns:
        Normalized metadata dictionary with viewport as string
    """
    if not isinstance(metadata, dict):
        return {}
    
    normalized = {}
    for key, value in metadata.items():
        if key == "viewport":
            # Handle viewport: convert list to string if needed
            if isinstance(value, list):
                normalized[key] = ", ".join(str(item) for item in value)
            elif value is not None:
                normalized[key] = str(value)
            else:
                normalized[key] = ""
        elif key == "lang":
            normalized[key] = str(value) if value is not None else ""
        else:
            normalized[key] = value
    return normalized


def fetch_webpage_content_jina(
    url: str,
    api_key: str = None,
    timeout: int = TIMEOUT,
    max_retries: int = 3,
    should_filter_callback: Optional[Callable[[str, str, str], Tuple[bool, Optional[str]]]] = None,
    summarize: bool = True,
) -> JinaWebpageResponse:
    """
    Fetch webpage content using Jina Reader API with JSON format.
    Automatically retries on timeout errors, bot protection pages, and server errors.
    Only caches successful responses to avoid caching transient errors.

    Args:
        url: The URL of the webpage to fetch
        api_key: Jina API key (if not provided, will use JINA_API_KEY env var)
        timeout: Request timeout in seconds (if not provided, will use TIMEOUT env var or default 30)
        max_retries: Maximum number of retry attempts for retryable errors (default: 3)
        should_filter_callback: Optional callback function(content, title, url) -> (should_filter, reason)
                                If provided, will be called BEFORE summarization to check if content should be filtered.
                                If should_filter is True, summarization is skipped and content is returned as empty.
        summarize: If True (default), long content is passed through Azure OpenAI summarization.
            If False, returns raw Jina Reader text with no LLM summarization step.

    Returns:
        JinaWebpageResponse containing:
        - url: The original URL that was fetched
        - title: The webpage title
        - content: The webpage content as clean text/markdown
        - description: The webpage description (if available)
        - publishedTime: Publication timestamp (if available)
        - metadata: Additional metadata (lang, viewport, etc.)
        - success: Boolean indicating if the fetch was successful
        - error: Error message if fetch failed
    """
    # Check cache first (only for successful responses)
    # BUT: If a filter callback is provided, we need to check the filter even on cached results
    # because the filter configuration might have changed
    # ALSO: If cached content is too long, we may summarize it (old cached entries might not be summarized)
    cache_key = None
    if is_cache_enabled():
        cache_key = DEFAULT_CACHE._get_cache_key(
            fetch_webpage_content_jina,
            (),
            {"url": url, "timeout": timeout, "max_retries": max_retries, "summarize": summarize},
        )
        cached_result = DEFAULT_CACHE.get(cache_key)
        if cached_result is not None:
            # Only return cached result if it was successful
            if isinstance(cached_result, dict) and cached_result.get("success", False):
                # Normalize metadata in cached result before returning (fixes old cached data)
                if "metadata" in cached_result:
                    cached_result["metadata"] = normalize_metadata(cached_result["metadata"])
                
                cached_content = cached_result.get("content", "")
                cached_title = cached_result.get("title", "")
                cached_url = cached_result.get("url", url)
                
                # If filter callback is provided, check the cached content too
                if should_filter_callback:
                    if cached_content:
                        try:
                            should_filter, filter_reason = should_filter_callback(cached_content, cached_title, cached_url)
                            if should_filter:
                                _main_logger.info(f"Cached content filtered: {filter_reason}")
                                _main_logger.info(f"  URL: {cached_url}")
                                _main_logger.info(f"  Title: {cached_title}")
                                # Return filtered result (empty content)
                                cached_result["content"] = ""
                                return cached_result
                        except Exception as e:
                            _main_logger.error(f"Error in filter callback for cached content: {e}", exc_info=True)
                            # Continue and return cached result anyway
                
                # Check if cached content needs summarization (might be old cache entry without summarization)
                # Summarize if content is longer than 1M characters (likely not summarized) and likely to
                # exceed the model's context length
                MAX_UNSUMMARIZED_LENGTH = 1_000_000
                if summarize and cached_content and len(cached_content) > MAX_UNSUMMARIZED_LENGTH:
                    _main_logger.info(
                        f"Cached content is very long ({len(cached_content):,} chars), likely not summarized. "
                        f"Summarizing cached content for URL: {cached_url}"
                    )
                    try:
                        summarized_content = summarize_content(cached_content)
                        cached_result["content"] = summarized_content
                        # Update cache with summarized content
                        try:
                            DEFAULT_CACHE.set(cache_key, cached_result)
                            _main_logger.info(f"Updated cache with summarized content ({len(summarized_content):,} chars)")
                        except Exception as e:
                            _main_logger.warning(f"Failed to update cache with summarized content: {e}")
                    except JinaSummarizationError:
                        # Missing Azure creds — hard stop (do not serve raw pages).
                        raise
                
                return cached_result
            # If cached result is an error, don't use it - allow retry
    
    if not api_key:
        api_key = os.getenv("JINA_API_KEY")
        if not api_key:
            raise ValueError(
                "JINA_API_KEY environment variable is not set or api_key parameter not provided"
            )

    if timeout is None:
        timeout = TIMEOUT

    jina_url = f"https://r.jina.ai/{url}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    # Helper functions
    def error_response(error_msg: str, data: Optional[Dict] = None) -> JinaWebpageResponse:
        """Create an error response."""
        if data:
            return {
                "url": data.get("url", url),
                "title": data.get("title", ""),
                "content": data.get("content", ""),
                "description": data.get("description", ""),
                "publishedTime": data.get("publishedTime", ""),
                "metadata": data.get("metadata", {}),
                "success": False,
                "error": error_msg,
            }
        return {
            "url": url,
            "title": "",
            "content": "",
            "description": "",
            "publishedTime": "",
            "metadata": {},
            "success": False,
            "error": error_msg,
        }

    def is_bot_protection(title: str, content: str) -> bool:
        """Check if page is a bot protection page."""
        indicators = ["just a moment", "verify you are human", "checking your browser",
                     "please wait", "security check", "access denied", "cloudflare"]
        text = f"{title} {content}".lower()
        return any(indicator in text for indicator in indicators)

    # Retry loop
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(jina_url, headers=headers, timeout=timeout)
            
            # Handle non-200 status codes
            if response.status_code != 200:
                # Retry on server errors (500+), 402 (InsufficientBalanceError), and specific gateway errors
                # 402 errors can occur due to rate limiting or temporary quota checks even with sufficient balance
                # 502, 503, 504 are gateway/service unavailable errors that are often transient
                retryable_status_codes = {402, 502, 503, 504}  # Specific retryable client/server errors
                is_server_error = response.status_code >= 500
                is_retryable = is_server_error or response.status_code in retryable_status_codes
                
                if is_retryable and attempt < max_retries - 1:
                    last_error = f"API error {response.status_code}: {response.text[:200]}"
                    time.sleep(random.uniform(10, 15))
                    continue
                return error_response(f"API request failed with status {response.status_code}: {response.text[:500]}")

            # Parse JSON
            try:
                data = response.json().get("data", {})
            except Exception as e:
                return error_response(f"Failed to parse JSON response: {str(e)}")

            # Check for bot protection
            title = data.get("title", "")
            content = data.get("content", "")
            if is_bot_protection(title, content):
                if attempt < max_retries - 1:
                    last_error = "Bot protection page detected"
                    time.sleep(random.uniform(10, 15))
                    continue
                return error_response("Bot protection page detected - unable to fetch actual content", data)

            # Check filter BEFORE summarization (if callback provided)
            should_filter = False
            filter_reason = None
            if should_filter_callback:
                # Use __main__ logger so logs go to the same file as remote MCP servers
                _main_logger.info(f"Filter callback provided, checking content (length: {len(content) if content else 0} chars)")
                if content:
                    try:
                        should_filter, filter_reason = should_filter_callback(content, title, url)
                        _main_logger.info(f"Filter check result: should_filter={should_filter}, reason={filter_reason}")
                        if should_filter:
                            _main_logger.info(f"Content filtered BEFORE summarization: {filter_reason}")
                            _main_logger.info(f"  URL: {url}")
                            _main_logger.info(f"  Title: {title}")
                            _main_logger.info(f"  Content length: {len(content):,} characters")
                            # Return result with empty content (filtered out)
                            metadata = normalize_metadata(data.get("metadata", {}))
                            return {
                                "url": data.get("url", url),
                                "title": title,
                                "content": "",  # Empty content indicates filtered
                                "description": data.get("description", ""),
                                "publishedTime": data.get("publishedTime", ""),
                                "metadata": metadata,
                                "success": True,
                            }
                    except Exception as e:
                        _main_logger.error(f"Error in filter callback: {e}", exc_info=True)
                        _main_logger.warning("Continuing with summarization despite filter error.")
                else:
                    _main_logger.warning("Filter callback provided but content is empty, skipping filter check")
            else:
                _main_logger.debug("No filter callback provided, skipping filter check")

            # Summarize content (only when there's no JINA API error, i.e., success case)
            # AND only if not filtered AND summarize=True
            if summarize and content and not should_filter:
                original_length = len(content)
                logger.info(f"BEFORE summarization: Content length = {original_length:,} characters")
                content = summarize_content(content)
                summarized_length = len(content)
                reduction = original_length - summarized_length
                reduction_pct = (reduction / original_length * 100) if original_length > 0 else 0
                logger.info(f"AFTER summarization: Content length = {summarized_length:,} characters (reduced by {reduction:,} chars, {reduction_pct:.1f}%)")

            # Normalize metadata to ensure type safety
            # Jina API sometimes returns viewport as a list, but we expect a string
            metadata = normalize_metadata(data.get("metadata", {}))
            
            # Success - cache only successful responses
            result = {
                "url": data.get("url", url),
                "title": title,
                "content": content,
                "description": data.get("description", ""),
                "publishedTime": data.get("publishedTime", ""),
                "metadata": metadata,
                "success": True,
            }
            # Only cache successful responses
            if is_cache_enabled() and cache_key is not None:
                try:
                    DEFAULT_CACHE.set(cache_key, result)
                except Exception as e:
                    # Log but don't fail if caching fails
                    pass
            return result

        except (ReadTimeout, Timeout, ConnectTimeout) as e:
            last_error = f"Request timed out after {timeout} seconds"
            if attempt < max_retries - 1:
                time.sleep(random.uniform(10, 15))
                continue
            return error_response(f"{last_error}: {str(e)}")
        except ConnectionError as e:
            # Connection errors (network issues, upstream errors) should be retried
            last_error = f"Connection error: {str(e)}"
            if attempt < max_retries - 1:
                time.sleep(random.uniform(10, 15))
                continue
            return error_response(last_error)
        except RequestException as e:
            # Other request exceptions should also be retried
            last_error = f"Request failed: {str(e)}"
            if attempt < max_retries - 1:
                time.sleep(random.uniform(10, 15))
                continue
            return error_response(last_error)

    return error_response(f"Failed after {max_retries} attempts: {last_error or 'Unknown error'}")
