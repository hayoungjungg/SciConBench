"""Perplexity provider implementation with built-in search capabilities.

Maintains the same filtering mechanisms to mitigate leakage of ground-truth answer.
- search_before_date_filter limits the search to articles before a certain date
- domain_filter filters out domains/URLs that leaks the ground-truth answer, preventing rigorous synthesis. 
    * Crowdsources URLs filtered from other models' tool usages. This is used as a starting point
    * Appends cochrane.org and cochranelibrary.com URLs to the filter list since our benchmark is based on Cochrane reviews.
    * For Sonar Reasoning Pro: Iteratively runs Perplexity and detects URLs from search results that share the same title as a Cochrane review, 
    appending to the initial list of domains/URLs we are filtering for. Repeat until no more Cochrane-related titles are found from the Perplexity search results that informed their generated conclusions OR the 20 denylist domain limit is hit
    * For Sonar Deep Research: Uses pre-loaded domain filter list from top_18_filtered_links_from_logs.json 
    (per-DOI, max 18 links + 2 default Cochrane domains = max 20 total). Curated from mcp_client.log files 
    of 3 models (claude-sonnet-4-5, gemini-3-pro-preview, gpt-5.1) by extracting "FILTERED OUT" URLs, 
    cleaning/formatting them, and selecting top 18 most frequent per DOI. Iterative filtering is disabled 
    when domain_filter is provided - filters are applied directly.

Parameters for sonar-reasoning-pro:
- model: "sonar-reasoning-pro"
- temperature:  0.2 (default on perplexity)
- reasoning_effort: "high"
- search_mode: "academic" (used in web_search_options)
- web_search_context_size: "high" (used in web_search_options)
- search_type: "auto" (always set to "auto", not configurable)
- domain_filter: Optional[List[str]], per-call parameter
    * List of domains/URLs to filter. Use "-" prefix for denylist mode
    * Example: ["-example.com", "https://allowed.com"]
- search_before_date_filter: Optional[str], per-call parameter
    * Date filter in MM/DD/YYYY format (e.g., "3/5/2025")
    * Only include results before this date
- stream: True (automatically set, not configurable)

Parameters for sonar-deep-research:
- model: "sonar-deep-research"
- temperature: 0.2
- reasoning_effort: "high"
- search_mode: "academic" (used in web_search_options)
- web_search_context_size: "high" (used in web_search_options)
- search_type: "auto" (always set to "auto", not configurable)
- domain_filter: Optional[List[str]], per-call parameter
    * Pre-loaded from top_18_filtered_links_from_logs.json (per-DOI, max 18 links from 3 models' logs)
    * Automatically adds cochrane.org and cochranelibrary.com if not present (max 20 total)
    * All entries have "-" prefix for denylist mode
    * Iterative filtering disabled when provided (uses pre-loaded filters directly)
- search_before_date_filter: Optional[str], per-call parameter
- stream: True (automatically set, not configurable)

API request structure (async mode, if use_async_deep_research=True):
    - Submit via: client.async_.chat.completions.create(**async_kwargs)
    - Poll via: client.async_.chat.completions.get(request_id)
    - Same parameters as sync mode, but uses async endpoint with polling
"""

import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from perplexity import Perplexity
    # Import exceptions from perplexity SDK if available
    try:
        from perplexity import (
            RateLimitError,
            APIConnectionError,
            APIStatusError,
            APIError,
            APITimeoutError,
        )
    except ImportError:
        # If specific exceptions aren't available, use generic Exception
        # or check if they're in a different module
        try:
            from perplexity._exceptions import (
                RateLimitError,
                APIConnectionError,
                APIStatusError,
                APIError,
                APITimeoutError,
            )
        except ImportError:
            # Fallback: create dummy exception classes
            class RateLimitError(Exception):
                pass
            class APIConnectionError(Exception):
                pass
            class APIStatusError(Exception):
                pass
            class APIError(Exception):
                pass
            class APITimeoutError(Exception):
                pass
    PERPLEXITY_AVAILABLE = True
except ImportError:
    PERPLEXITY_AVAILABLE = False
    # Create dummy exception classes for type checking
    class RateLimitError(Exception):
        pass
    class APIConnectionError(Exception):
        pass
    class APIStatusError(Exception):
        pass
    class APIError(Exception):
        pass
    class APITimeoutError(Exception):
        pass

from .base import LLMProvider, ContextLengthExceededError
from ..prompts import RESEARCH_ASSISTANT_PROMPT
from ..filters.cochrane import create_title_filter_from_list, CochraneResultFilter, _is_cochrane_url

logger = logging.getLogger(__name__)


def _get_tool_attr(tool: Any, attr: str, default: Any = None) -> Any:
    """Get attribute from tool (handles both object and dict)."""
    return getattr(tool, attr, None) or (tool.get(attr, default) if isinstance(tool, dict) else default)


def _validate_date_filter(date_str: str) -> bool:
    """
    Validate date filter format (MM/DD/YYYY).
    
    Args:
        date_str: Date string to validate
    
    Returns:
        True if valid, False otherwise
    """
    if not date_str:
        return False
    # Check format: M/D/YYYY or MM/DD/YYYY
    pattern = r'^\d{1,2}/\d{1,2}/\d{4}$'
    if not re.match(pattern, date_str):
        return False
    try:
        month, day, year = map(int, date_str.split('/'))
        if month < 1 or month > 12:
            return False
        if day < 1 or day > 31:
            return False
        if year < 1900 or year > 2100:
            return False
        return True
    except ValueError:
        return False


def _is_date_before_filter(result_date: Optional[str], filter_date: Optional[str]) -> bool:
    """
    Check if a result date is before the filter date.
    
    Args:
        result_date: Date from search result (various formats)
        filter_date: Filter date in MM/DD/YYYY format
    
    Returns:
        True if result_date is before filter_date, False otherwise.
        Returns True if either date is missing (assume valid if can't verify).
    """
    if not filter_date or not result_date:
        # If no filter date, allow all. If no result date, allow it (can't verify).
        return True
    
    if not _validate_date_filter(filter_date):
        # Invalid filter date, allow result
        return True
    
    # Try to parse result_date and compare
    try:
        from dateutil import parser
        result_dt = parser.parse(result_date)
        filter_dt = parser.parse(filter_date)
        return result_dt < filter_dt
    except (ImportError, ValueError):
        # Can't parse result date, allow it (assume valid)
        logger.debug("Could not parse result date '%s' for comparison, allowing it", result_date)
        return True


def _is_pubmed_pmc_domain(domain: str) -> bool:
    """
    Check if a domain is a PubMed or PMC domain that should never be sanitized.
    
    PubMed/PMC domains must always be filtered at the article level (e.g., 
    https://pmc.ncbi.nlm.nih.gov/articles/PMC12324103/), never sanitized to 
    just the domain.
    
    Args:
        domain: A domain entry (may have denylist prefix)
    
    Returns:
        True if this is a PubMed/PMC domain, False otherwise
    """
    domain_lower = domain.lstrip('-').lower()
    
    # Check for PubMed/PMC domains
    pubmed_pmc_domains = [
        'pubmed.ncbi.nlm.nih.gov',
        'pmc.ncbi.nlm.nih.gov',
        'www.pubmed.ncbi.nlm.nih.gov',
        'www.pmc.ncbi.nlm.nih.gov',
    ]
    
    # Remove protocol if present
    if domain_lower.startswith('http://'):
        domain_lower = domain_lower[7:]
    elif domain_lower.startswith('https://'):
        domain_lower = domain_lower[8:]
    
    # Check if domain starts with any PubMed/PMC domain
    for pubmed_domain in pubmed_pmc_domains:
        if domain_lower.startswith(pubmed_domain):
            return True
    
    return False


def _is_pubmed_search_url(url: str) -> bool:
    """
    Check if a URL is a PubMed search URL (e.g., https://pubmed.ncbi.nlm.nih.gov/?term=...).
    These should be excluded from domain filters.
    
    Args:
        url: A URL string (may have denylist prefix)
    
    Returns:
        True if this is a PubMed search URL, False otherwise
    """
    cleaned = url.lstrip('-').lower()
    
    # Remove protocol if present
    if cleaned.startswith('http://'):
        cleaned = cleaned[7:]
    elif cleaned.startswith('https://'):
        cleaned = cleaned[8:]
    
    # Check if it's a PubMed search URL (has ?term= or just ? with no article ID path)
    if 'pubmed.ncbi.nlm.nih.gov' in cleaned:
        # Check if it has a query parameter (search URL) and no article ID in path
        if '?' in url:
            # Extract the path before the query
            path_before_query = url.split('?')[0].lower()
            # Check if path is just domain or domain/ (no article ID)
            if (path_before_query.endswith('pubmed.ncbi.nlm.nih.gov') or 
                path_before_query.endswith('pubmed.ncbi.nlm.nih.gov/')):
                return True
            # Also check for ?term= specifically
            if '?term=' in url.lower() or '&term=' in url.lower():
                return True
    
    return False


def _normalize_url_for_comparison(url: str) -> str:
    """
    Normalize a URL for comparison by removing protocol, query parameters, fragments,
    and denylist prefix. Used to match URLs that might be in different formats.
    
    Args:
        url: A URL string (may have denylist prefix, protocol, query params)
    
    Returns:
        Normalized URL string for comparison
    """
    # Remove denylist prefix
    normalized = url.lstrip('-')
    
    # Remove protocol
    if normalized.startswith('http://'):
        normalized = normalized[7:]
    elif normalized.startswith('https://'):
        normalized = normalized[8:]
    
    # Remove www. prefix
    if normalized.startswith('www.'):
        normalized = normalized[4:]
    
    # Remove query parameters and fragments
    normalized = normalized.split('?')[0]  # Remove query parameters
    normalized = normalized.split('#')[0]   # Remove fragments
    
    # Normalize to lowercase
    normalized = normalized.lower()
    
    # Ensure trailing slash for consistency
    if '/' in normalized and not normalized.endswith('/'):
        normalized = normalized + '/'
    
    return normalized


def _sanitize_pubmed_pmc_url(url: str) -> Optional[str]:
    """
    Sanitize a PubMed/PMC URL by removing query parameters and fragments.
    Keeps the protocol, domain, and path up to the article ID.
    
    Examples:
        https://pubmed.ncbi.nlm.nih.gov/40919710/?fc=None&ff=20251211211425&v=2.18.0.post22+67771e2
        -> https://pubmed.ncbi.nlm.nih.gov/40919710/
        
        https://pmc.ncbi.nlm.nih.gov/articles/PMC12324103/?some=param
        -> https://pmc.ncbi.nlm.nih.gov/articles/PMC12324103/
        
        https://pubmed.ncbi.nlm.nih.gov/?term=40105375
        -> None (search URLs are excluded)
    
    Args:
        url: A PubMed/PMC URL (may have denylist prefix)
    
    Returns:
        Sanitized URL with query parameters and fragments removed, preserving protocol.
        Returns None if the URL is a search URL (should be excluded).
    """
    # Check if this is a search URL - exclude it
    if _is_pubmed_search_url(url):
        return None
    
    # Preserve denylist prefix
    is_denylist = url.startswith('-')
    cleaned = url.lstrip('-')
    
    # Preserve protocol
    protocol = ''
    if cleaned.startswith('http://'):
        protocol = 'http://'
        cleaned = cleaned[7:]
    elif cleaned.startswith('https://'):
        protocol = 'https://'
        cleaned = cleaned[8:]
    
    # Remove www. prefix
    if cleaned.startswith('www.'):
        cleaned = cleaned[4:]
    
    # Remove query parameters and fragments
    cleaned = cleaned.split('?')[0]  # Remove query parameters
    cleaned = cleaned.split('#')[0]   # Remove fragments
    
    # Ensure it ends with / if it has a path
    if '/' in cleaned and not cleaned.endswith('/'):
        # Extract domain and path
        parts = cleaned.split('/', 1)
        if len(parts) == 2 and parts[1]:  # Has a path component
            cleaned = f"{parts[0]}/{parts[1]}/"
    
    # Reconstruct with protocol
    result = f"{protocol}{cleaned}" if protocol else cleaned
    return f"-{result}" if is_denylist else result


def _is_article_level_url(url: str) -> bool:
    """
    Check if a URL is an article-level URL that should be kept intact (not truncated).
    
    Article-level URLs typically have patterns like:
    - academic.oup.com/{journal}/article/{volume}/{issue}/{page}/{article_id}
    - Other journal sites with article identifiers
    
    Args:
        url: A URL (may have denylist prefix)
    
    Returns:
        True if this is an article-level URL that should be kept intact, False otherwise
    """
    cleaned = url.lstrip('-').lower()
    
    # Remove protocol if present
    if cleaned.startswith('http://'):
        cleaned = cleaned[7:]
    elif cleaned.startswith('https://'):
        cleaned = cleaned[8:]

    
    return False


def _sanitize_single_domain_progressively(domain: str) -> str:
    """
    Progressively sanitize a single domain by removing the last path segment.
    
    This removes one path segment at a time from the end, rather than removing all paths.
    For example: "avalonhcs.com/policypublishing/bcbsm/G2063%20Testing.pdf" 
    -> "avalonhcs.com/policypublishing/bcbsm/"
    
    IMPORTANT: PubMed/PMC domains are NEVER sanitized - they must always be filtered 
    at the article level (e.g., https://pmc.ncbi.nlm.nih.gov/articles/PMC12324103/).
    
    Args:
        domain: A domain entry (may contain paths, e.g., "avalonhcs.com/path/to/file")
    
    Returns:
        Sanitized domain with last path segment removed (e.g., "avalonhcs.com/path/to/" or "-avalonhcs.com/path/to/" if it had a "-" prefix).
        Returns the domain unchanged if it's a PubMed/PMC domain.
    """
    # Never sanitize PubMed/PMC domains - they must always be at article level
    if _is_pubmed_pmc_domain(domain):
        logger.warning("Skipping sanitization for PubMed/PMC domain (must be filtered at article level): %s", domain)
        return domain
    
    # Preserve denylist prefix
    is_denylist = domain.startswith('-')
    cleaned = domain.lstrip('-')
    
    # Remove protocol prefixes
    if cleaned.startswith('http://'):
        cleaned = cleaned[7:]
    elif cleaned.startswith('https://'):
        cleaned = cleaned[8:]
    
    # Remove www. prefix
    if cleaned.startswith('www.'):
        cleaned = cleaned[4:]
    
    # Remove query parameters and fragments first
    cleaned = cleaned.split('?')[0]  # Remove query parameters
    cleaned = cleaned.split('#')[0]   # Remove fragments
    
    # Progressively remove the last path segment
    if '/' in cleaned:
        # Split by '/' and remove the last segment
        parts = cleaned.split('/')
        # Keep domain and all path segments except the last one
        if len(parts) > 1:
            # Remove last segment, but keep trailing slash if there were multiple segments
            cleaned = '/'.join(parts[:-1]) + '/'
        else:
            # No path segments, just domain
            cleaned = parts[0]
    else:
        # No path, just domain
        pass
    
    # Remove trailing dots
    cleaned = cleaned.rstrip('.')
    
    # Re-add denylist prefix if it was present
    return f"-{cleaned}" if is_denylist else cleaned


def _sanitize_url_aggressively(url: str, max_path_segments: int = 3) -> str:
    """
    Aggressively sanitize a URL by keeping only the domain and first few path segments.
    
    This is used when errors occur to shorten URLs to a base path.
    For example: "https://adacyte.com/professional/scientific-corner/long/path/to/article"
    -> "https://adacyte.com/professional/scientific-corner/"
    
    IMPORTANT: PubMed/PMC domains are NEVER sanitized - they must always be filtered 
    at the article level (e.g., https://pmc.ncbi.nlm.nih.gov/articles/PMC12324103/).
    
    Args:
        url: A URL (may have "-" prefix for denylist)
        max_path_segments: Maximum number of path segments to keep (default: 3, meaning domain + 2 path segments)
    
    Returns:
        Sanitized URL with only base path kept (e.g., "https://adacyte.com/professional/scientific-corner/").
        Returns the URL unchanged if it's a PubMed/PMC domain.
    """
    # Never sanitize PubMed/PMC domains - they must always be at article level
    if _is_pubmed_pmc_domain(url):
        logger.warning("Skipping aggressive sanitization for PubMed/PMC domain (must be filtered at article level): %s", url)
        return url
    
    # Preserve denylist prefix
    is_denylist = url.startswith('-')
    cleaned = url.lstrip('-')
    
    # Determine protocol
    has_https = cleaned.startswith('https://')
    has_http = cleaned.startswith('http://')
    protocol = 'https://' if has_https else ('http://' if has_http else '')
    
    if has_https:
        cleaned = cleaned[8:]
    elif has_http:
        cleaned = cleaned[7:]
    
    # Remove www. prefix
    if cleaned.startswith('www.'):
        cleaned = cleaned[4:]
    
    # Remove query parameters and fragments first
    cleaned = cleaned.split('?')[0]  # Remove query parameters
    cleaned = cleaned.split('#')[0]   # Remove fragments
    
    # Split by '/' to get path segments
    if '/' in cleaned:
        parts = cleaned.split('/')
        # Filter out empty parts (from leading/trailing slashes)
        parts = [p for p in parts if p]
        
        if len(parts) > 1:
            # Keep domain + first (max_path_segments - 1) path segments
            # max_path_segments=3 means: domain + 2 path segments (e.g., "adacyte.com/professional/scientific-corner")
            # Total segments to keep = max_path_segments (domain + path segments)
            total_segments_to_keep = min(len(parts), max_path_segments)
            if total_segments_to_keep > 1:
                # Keep domain + first (total_segments_to_keep - 1) path segments
                kept_parts = parts[:total_segments_to_keep]
                cleaned = '/'.join(kept_parts) + '/'
            else:
                # No path segments to keep, just domain
                cleaned = parts[0] + '/'
        else:
            # Just domain, no path
            cleaned = parts[0] + '/'
    else:
        # No path, just domain
        cleaned = cleaned + '/'
    
    # Remove trailing dots
    cleaned = cleaned.rstrip('.')
    
    # Reconstruct with protocol
    result = protocol + cleaned if protocol else cleaned
    
    # Re-add denylist prefix if it was present
    return f"-{result}" if is_denylist else result


def _extract_problematic_domain_from_error(error_msg: str) -> Optional[str]:
    """
    Extract the problematic domain from Perplexity API error message.
    
    Error formats:
    - "Validation error: search_domain_filters must be a valid domain name, but got avalonhcs.com/path/to/file"
    - "invalid_search_domain_filter: <domain>"
    - Error messages containing domain-like strings
    
    Args:
        error_msg: Error message from Perplexity API
    
    Returns:
        The problematic domain string if found, None otherwise
    """
    import re
    
    # Pattern 1: Match "but got <domain>" in the error message
    match = re.search(r'but got\s+([^\s,]+)', error_msg, re.IGNORECASE)
    if match:
        domain = match.group(1).strip('"\'')
        if domain:
            return domain
    
    # Pattern 2: Match "invalid_search_domain_filter: <domain>"
    match = re.search(r'invalid_search_domain_filter[:\s]+([^\s,]+)', error_msg, re.IGNORECASE)
    if match:
        domain = match.group(1).strip('"\'')
        if domain:
            return domain
    
    # Pattern 3: Try to find URLs/domains in the error message (look for http:// or https://)
    match = re.search(r'(https?://[^\s,)]+)', error_msg, re.IGNORECASE)
    if match:
        domain = match.group(1).strip('"\'')
        if domain:
            return domain
    
    # Pattern 4: Try to find domain-like strings (domain.com/path)
    match = re.search(r'([a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]*\.[a-zA-Z]{2,}[^\s,)]*)', error_msg)
    if match:
        domain = match.group(1).strip('"\'')
        # Only return if it looks like a domain with optional path
        if '/' in domain or '.' in domain:
            return domain
    
    return None


def _extract_domain_from_url(url: str) -> str:
    """
    Extract domain and path from a URL for use in Perplexity domain filter.
    
    Perplexity accepts specific URLs (with full paths), so we keep the full URL.
    If Perplexity rejects a URL, progressive sanitization will remove path segments.
    
    For URLs with paths, keeps the full URL with https://.
    For domain-level entries (just domain or domain/), removes https:// prefix.
    
    Examples:
    - "https://pubmed.ncbi.nlm.nih.gov/40484402/" -> "https://pubmed.ncbi.nlm.nih.gov/40484402/"
    - "https://nds.ox.ac.uk/publications/file.pdf" -> "https://nds.ox.ac.uk/publications/file.pdf"
    - "https://cochrane.org/" -> "cochrane.org/"
    
    Args:
        url: Full URL (may have "-" prefix for denylist)
    
    Returns:
        Full URL with path (preserving "-" prefix if present), or domain-level entry without https://
    """
    # Preserve denylist prefix
    is_denylist = url.startswith('-')
    cleaned = url.lstrip('-')
    
    # Check if it's a PubMed/PMC domain - keep at article level with https://
    if _is_pubmed_pmc_domain(cleaned):
        return url  # Return as-is (with or without "-" prefix)
    
    # Check if URL has https://
    has_https = cleaned.startswith('https://')
    has_http = cleaned.startswith('http://')
    protocol = ''
    if has_https:
        protocol = 'https://'
        cleaned = cleaned[8:]
    elif has_http:
        protocol = 'http://'
        cleaned = cleaned[7:]
    
    # Remove www. prefix
    if cleaned.startswith('www.'):
        cleaned = cleaned[4:]
    
    # Remove query parameters and fragments first
    cleaned = cleaned.split('?')[0]  # Remove query parameters
    cleaned = cleaned.split('#')[0]   # Remove fragments
    
    # Check if URL has a path (not just domain)
    if '/' in cleaned:
        # Split by '/' 
        parts = cleaned.split('/')
        non_empty_parts = [p for p in parts if p]
        
        if len(non_empty_parts) > 1:
            # Has actual path segments - keep full URL with https:// (including filename)
            # Reconstruct with all path segments
            result = '/'.join(parts)
            # Preserve trailing slash if it was in the original (after removing query params)
            original_no_query = url.lstrip('-').split('?')[0].split('#')[0]
            if original_no_query.rstrip('/').endswith('/') or (len(parts) > 1 and parts[-1] == ''):
                if not result.endswith('/'):
                    result += '/'
            result = 'https://' + result
        else:
            # Just domain/ (domain-level) - no https://
            result = '/'.join(parts[:-1]) + '/' if len(parts) > 1 else parts[0]
    else:
        # No path, just domain
        result = cleaned
    
    # Re-add denylist prefix if it was present
    return f"-{result}" if is_denylist else result


def _url_matches_domain_filter(url: str, domain_filter: List[str]) -> bool:
    """
    Check if a URL matches any entry in the domain filter.
    
    This is used for client-side filtering to avoid processing duplicate URLs
    in iterative filtering, even though URLs are also sent to Perplexity API.
    
    Args:
        url: URL to check (e.g., "https://pmc.ncbi.nlm.nih.gov/articles/PMC12324103/")
        domain_filter: List of domain filter entries (may have "-" prefix)
    
    Returns:
        True if the URL matches any filter entry, False otherwise
    """
    if not url or not domain_filter:
        return False
    
    # Normalize the URL for comparison
    url_lower = url.lower().strip()
    url_without_protocol = url_lower
    if url_without_protocol.startswith('http://'):
        url_without_protocol = url_without_protocol[7:]
    elif url_without_protocol.startswith('https://'):
        url_without_protocol = url_without_protocol[8:]
    
    # Remove query parameters and fragments for comparison
    url_without_protocol = url_without_protocol.split('?')[0].split('#')[0]
    
    for filter_entry in domain_filter:
        # Remove "-" prefix if present
        filter_entry_clean = filter_entry.lstrip('-').lower().strip()
        
        # For PMC/PubMed URLs, do exact match (they're stored as full article URLs)
        if _is_pubmed_pmc_domain(filter_entry_clean):
            # Compare full URLs (normalized)
            filter_url_without_protocol = filter_entry_clean
            if filter_url_without_protocol.startswith('http://'):
                filter_url_without_protocol = filter_url_without_protocol[7:]
            elif filter_url_without_protocol.startswith('https://'):
                filter_url_without_protocol = filter_url_without_protocol[8:]
            filter_url_without_protocol = filter_url_without_protocol.split('?')[0].split('#')[0]
            
            # Exact match for PMC/PubMed URLs
            if url_without_protocol == filter_url_without_protocol:
                return True
        else:
            # For other domains, check if URL starts with the filter entry
            filter_without_protocol = filter_entry_clean
            if filter_without_protocol.startswith('http://'):
                filter_without_protocol = filter_without_protocol[7:]
            elif filter_without_protocol.startswith('https://'):
                filter_without_protocol = filter_without_protocol[8:]
            filter_without_protocol = filter_without_protocol.split('?')[0].split('#')[0]
            
            # Check if URL starts with filter entry (domain/path matching)
            if url_without_protocol.startswith(filter_without_protocol):
                return True
    
    return False


def _extract_search_results_list(response: Any) -> Optional[List[Dict[str, Any]]]:
    """
    Extract search_results from Perplexity API response and convert to list of dicts.
    
    Args:
        response: Response object from Perplexity API (may have search_results attribute)
    
    Returns:
        List of dictionaries with 'title', 'url', 'date' keys, or None if not available
    """
    if not response or not hasattr(response, 'search_results'):
        return None
    
    search_results = response.search_results
    if not search_results:
        return None
    
    search_results_list = []
    for result in search_results:
        result_dict = {
            'title': getattr(result, 'title', None) if hasattr(result, 'title') else (result.get('title') if isinstance(result, dict) else None),
            'url': getattr(result, 'url', None) if hasattr(result, 'url') else (result.get('url') if isinstance(result, dict) else None),
            'date': getattr(result, 'date', None) if hasattr(result, 'date') else (result.get('date') if isinstance(result, dict) else None),
        }
        search_results_list.append(result_dict)
    
    return search_results_list


def _has_any_cochrane_domain(domain_list: List[str]) -> tuple[bool, bool]:
    """
    Check if any cochrane.org or cochranelibrary.com domain exists in the filter list.
    
    Returns:
        Tuple of (has_cochrane_org, has_cochranelibrary)
    """
    has_cochrane_org = False
    has_cochranelibrary = False
    for domain in domain_list:
        domain_clean = domain.lstrip('-').lower()
        # Remove protocol if present
        if domain_clean.startswith('http://'):
            domain_clean = domain_clean[7:]
        elif domain_clean.startswith('https://'):
            domain_clean = domain_clean[8:]
        # Check for generic or specific cochrane.org
        if 'cochrane.org' in domain_clean:
            has_cochrane_org = True
        # Check for generic or specific cochranelibrary.com
        if 'cochranelibrary.com' in domain_clean:
            has_cochranelibrary = True
    return has_cochrane_org, has_cochranelibrary


def _deduplicate_and_sanitize_domains(domain_list: List[str]) -> List[str]:
    """
    Deduplicate and sanitize domain filter list.
    
    Removes:
    - Duplicate URLs
    - Redundant URLs (e.g., specific paths when a more generic domain/path is already filtered)
      Example: If "-cochrane.org/" exists, remove "-cochrane.org/evidence/" and "-cochrane.org/ru/evidence/"
      Example: If "-cochranelibrary.com/" exists, remove "-cochranelibrary.com/cdsr/doi/..."
    
    Args:
        domain_list: List of domains/URLs (may have "-" prefix for denylist)
    
    Returns:
        Deduplicated and sanitized list
    """
    if not domain_list:
        return []
    
    # Separate denylist and allowlist entries
    denylist = []
    allowlist = []
    
    for domain in domain_list:
        if domain.startswith('-'):
            denylist.append(domain)
        else:
            allowlist.append(domain)
    
    # Deduplicate within each list
    denylist = list(dict.fromkeys(denylist))  # Preserves order
    allowlist = list(dict.fromkeys(allowlist))
    
    # Remove redundant entries: if a more generic domain/path exists, remove more specific ones
    # Sort by length (shorter = more generic) to process generic ones first
    denylist_sorted = sorted(denylist, key=lambda x: len(x.lstrip('-')))
    
    filtered_denylist = []
    for domain in denylist_sorted:
        domain_clean = domain.lstrip('-').lower()
        # Remove protocol if present for comparison
        if domain_clean.startswith('http://'):
            domain_clean = domain_clean[7:]
        elif domain_clean.startswith('https://'):
            domain_clean = domain_clean[8:]
        
        # Check if this domain is already covered by a more generic one
        is_redundant = False
        for existing in filtered_denylist:
            existing_clean = existing.lstrip('-').lower()
            # Remove protocol if present for comparison
            if existing_clean.startswith('http://'):
                existing_clean = existing_clean[7:]
            elif existing_clean.startswith('https://'):
                existing_clean = existing_clean[8:]
            
            # Check if the current domain starts with the existing (more generic) one
            # This means the existing domain is more generic and covers the current one
            if domain_clean.startswith(existing_clean):
                # Additional check: ensure it's at a path boundary (not just a substring match)
                # For example, "cochrane.org" should match "cochrane.org/evidence" but not "cochrane.orgx"
                if (len(domain_clean) == len(existing_clean) or 
                    domain_clean[len(existing_clean):len(existing_clean)+1] in ['/', '?', '#', '']):
                    is_redundant = True
                    logger.debug("Removing redundant domain filter: %s (covered by: %s)", domain, existing)
                    break
        
        if not is_redundant:
            filtered_denylist.append(domain)
    
    # SPECIAL HANDLING: Always ensure generic Cochrane domains are present and remove specific ones
    # First pass: identify generic domains and collect all domains
    generic_cochrane_org = None
    generic_cochranelibrary = None
    all_domains = []
    
    for domain in filtered_denylist:
        domain_clean = domain.lstrip('-').lower()
        # Remove protocol if present
        if domain_clean.startswith('http://'):
            domain_clean = domain_clean[7:]
        elif domain_clean.startswith('https://'):
            domain_clean = domain_clean[8:]
        
        # Check if it's a generic cochrane.org (just domain, no path or ending with /)
        if domain_clean == 'cochrane.org' or domain_clean == 'cochrane.org/':
            generic_cochrane_org = '-cochrane.org/'
        # Check if it's a generic cochranelibrary.com
        elif domain_clean == 'cochranelibrary.com' or domain_clean == 'cochranelibrary.com/':
            generic_cochranelibrary = '-cochranelibrary.com/'
        
        all_domains.append((domain, domain_clean))
    
    # Check if we have any Cochrane domains (generic or specific) and ensure generic ones exist
    has_cochrane_org = any('cochrane.org' in d[1] for d in all_domains)
    has_cochranelibrary = any('cochranelibrary.com' in d[1] for d in all_domains)
    
    # If we have any Cochrane domains but no generic, set the generic flag
    if has_cochrane_org and not generic_cochrane_org:
        generic_cochrane_org = '-cochrane.org/'
        logger.debug("No generic cochrane.org found, will add it and remove specific paths")
    if has_cochranelibrary and not generic_cochranelibrary:
        generic_cochranelibrary = '-cochranelibrary.com/'
        logger.debug("No generic cochranelibrary.com found, will add it and remove specific paths")
    
    # Second pass: build final list, removing redundant specific paths
    final_denylist = []
    for domain, domain_clean in all_domains:
        # Check if it's a specific cochrane.org path
        if domain_clean.startswith('cochrane.org/') and domain_clean != 'cochrane.org/':
            # Skip if we have generic cochrane.org (always use generic to cover all paths)
            if generic_cochrane_org:
                logger.debug("Removing redundant cochrane.org path: %s (covered by %s)", domain, generic_cochrane_org)
                continue
            else:
                final_denylist.append(domain)
        # Check if it's a specific cochranelibrary.com path
        elif domain_clean.startswith('cochranelibrary.com/') and domain_clean != 'cochranelibrary.com/':
            # Skip if we have generic cochranelibrary.com (always use generic to cover all paths)
            if generic_cochranelibrary:
                logger.debug("Removing redundant cochranelibrary.com path: %s (covered by %s)", domain, generic_cochranelibrary)
                continue
            else:
                final_denylist.append(domain)
        # Check if it's generic cochrane.org - add canonical form
        elif (domain_clean == 'cochrane.org' or domain_clean == 'cochrane.org/') and generic_cochrane_org:
            if generic_cochrane_org not in final_denylist:
                final_denylist.append(generic_cochrane_org)
        # Check if it's generic cochranelibrary.com - add canonical form
        elif (domain_clean == 'cochranelibrary.com' or domain_clean == 'cochranelibrary.com/') and generic_cochranelibrary:
            if generic_cochranelibrary not in final_denylist:
                final_denylist.append(generic_cochranelibrary)
        else:
            # Not a Cochrane domain, keep it
            final_denylist.append(domain)
    
    # Always ensure generic Cochrane domains are added if any Cochrane domain was present
    if generic_cochrane_org and generic_cochrane_org not in final_denylist:
        final_denylist.append(generic_cochrane_org)
        logger.debug("Added generic cochrane.org domain to ensure all paths are covered")
    if generic_cochranelibrary and generic_cochranelibrary not in final_denylist:
        final_denylist.append(generic_cochranelibrary)
        logger.debug("Added generic cochranelibrary.com domain to ensure all paths are covered")
    
    # Convert any allowlist entries to denylist (all entries must have "-" prefix)
    denylist_from_allowlist = []
    for entry in allowlist:
        if not entry.startswith('-'):
            denylist_entry = f"-{entry}"
            logger.warning(
                "Converting allowlist entry to denylist (added '-' prefix): %s -> %s",
                entry, denylist_entry
            )
            denylist_from_allowlist.append(denylist_entry)
        else:
            # Already has "-" prefix, shouldn't happen but handle it
            denylist_from_allowlist.append(entry)
    
    # Combine: all entries are now denylist with "-" prefix
    result = final_denylist + denylist_from_allowlist
    return result


def _normalize_doi_for_lookup(doi: str) -> str:
    """
    Normalize DOI for consistent lookup (handles both slash and underscore formats).
    
    Args:
        doi: DOI string (may have "/" or "_" separators)
        
    Returns:
        Normalized DOI with underscores (consistent with JSON format)
    """
    # Replace slashes with underscores to match JSON format
    return doi.replace('/', '_')


def load_filtered_links_from_top18_json(json_path: Path, max_links_per_doi: int = 18) -> Dict[str, List[str]]:
    """
    Load filtered links from top_18_filtered_links_from_logs.json format.
    
    This function extracts per-DOI filtered links from the JSON structure generated by
    collect_filtered_links_from_logs.py. DOIs in the JSON use underscores (e.g., 
    "10.1002_14651858.CD015721.pub2"), but this function normalizes lookup keys to handle
    both slash and underscore formats.
    
    Args:
        json_path: Path to the JSON file (top_18_filtered_links_from_logs.json)
        max_links_per_doi: Maximum number of links to extract per DOI (default: 18)
        
    Returns:
        Dictionary mapping normalized DOI (with underscores) to list of filtered URLs (with "-" prefix)
        
    Example:
        >>> doi_to_links = load_filtered_links_from_top18_json(
        ...     Path("logs/top_18_filtered_links_from_logs.json")
        ... )
        >>> # Can lookup with either format
        >>> print(doi_to_links["10.1002_14651858.CD015721.pub2"])
        >>> print(doi_to_links["10.1002/14651858.CD015721.pub2"])  # Also works after normalization
        ['-https://pubmed.ncbi.nlm.nih.gov/38597945/', ...]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    doi_to_filtered_links = {}
    per_doi = data.get('per_doi', {})
    
    for doi, doi_data in per_doi.items():
        top_urls = doi_data.get('top_18_urls', [])
        # Extract URLs from the list of dicts with 'url' key
        urls = []
        for item in top_urls[:max_links_per_doi]:  # Limit to max_links_per_doi
            if isinstance(item, dict) and 'url' in item:
                urls.append(item['url'])
            elif isinstance(item, str):
                urls.append(item)
        
        # Store with normalized DOI (JSON already has underscores, but normalize for consistency)
        normalized_doi = _normalize_doi_for_lookup(doi)
        doi_to_filtered_links[normalized_doi] = urls
    
    return doi_to_filtered_links


def prepare_domain_filter_from_filtered_links(
    filtered_links: List[str],
    max_total_links: int = 20
) -> List[str]:
    """
    Prepare domain filter list from filtered links, adding Cochrane domains.
    
    This function takes a list of filtered URLs and prepares a domain filter list
    suitable for Perplexity's search_domain_filter parameter. It ensures:
    - All entries have "-" prefix for denylist mode
    - Cochrane domains are included if not already present
    - Total number of links doesn't exceed max_total_links (default: 20)
    - Domains are deduplicated and sanitized
    
    Args:
        filtered_links: List of filtered URLs (should already have "-" prefix)
        max_total_links: Maximum total number of links including Cochrane domains (default: 20)
        
    Returns:
        List of domain filter entries (with "-" prefix), max max_total_links items
        
    Example:
        >>> filtered_links = [
        ...     "-https://pubmed.ncbi.nlm.nih.gov/38597945/",
        ...     "-https://pmc.ncbi.nlm.nih.gov/articles/PMC123456/"
        ... ]
        >>> domain_filter = prepare_domain_filter_from_filtered_links(filtered_links, max_total_links=20)
        >>> print(len(domain_filter))  # Will be <= 20
        >>> print(any("cochrane" in d.lower() for d in domain_filter))  # True
    """
    domain_filter = []
    seen = set()
    
    # Add filtered links (should already have "-" prefix)
    for url in filtered_links:
        if url not in seen:
            domain_filter.append(url)
            seen.add(url)
            if len(domain_filter) >= max_total_links - 2:  # Reserve 2 spots for Cochrane domains
                break
    
    # Add Cochrane domains if not already present
    cochrane_domains = [
        "-cochranelibrary.com/",
        "-cochrane.org/"
    ]
    
    has_cochrane_org, has_cochranelibrary = _has_any_cochrane_domain(domain_filter)
    
    if not has_cochranelibrary and len(domain_filter) < max_total_links:
        domain_filter.append(cochrane_domains[0])
        seen.add(cochrane_domains[0])
    
    if not has_cochrane_org and len(domain_filter) < max_total_links:
        domain_filter.append(cochrane_domains[1])
        seen.add(cochrane_domains[1])
    
    # Deduplicate and sanitize
    domain_filter = _deduplicate_and_sanitize_domains(domain_filter)
    
    return domain_filter


def prepare_doi_domain_filters_from_json(
    json_path: Path,
    max_links_per_doi: int = 18,
    max_total_links: int = 20
) -> Dict[str, List[str]]:
    """
    Load filtered links from JSON and prepare domain filters for all DOIs.
    
    This is a convenience function that combines loading filtered links and preparing
    domain filters for batch processing. DOIs are normalized to handle both slash and
    underscore formats.
    
    Args:
        json_path: Path to top_18_filtered_links_from_logs.json
        max_links_per_doi: Maximum number of links to extract per DOI from JSON (default: 18)
        max_total_links: Maximum total links per DOI including Cochrane domains (default: 20)
        
    Returns:
        Dictionary mapping normalized DOI (with underscores) to list of domain filter entries (with "-" prefix)
        The dictionary keys use underscores to match the JSON format, but lookups will work
        with either format when normalized.
        
    Example:
        >>> doi_to_domain_filters = prepare_doi_domain_filters_from_json(
        ...     Path("logs/top_18_filtered_links_from_logs.json")
        ... )
        >>> # Use with BatchQueryRunner
        >>> runner = BatchQueryRunner(..., domain_filter=doi_to_domain_filters)
    """
    # Load filtered links from JSON (DOIs are normalized to underscore format)
    doi_to_filtered_links = load_filtered_links_from_top18_json(json_path, max_links_per_doi)
    
    # Prepare domain filters for each DOI
    doi_to_domain_filters = {}
    for doi, filtered_links in doi_to_filtered_links.items():
        domain_filter = prepare_domain_filter_from_filtered_links(filtered_links, max_total_links)
        # Store with normalized DOI (already normalized in load_filtered_links_from_top18_json)
        doi_to_domain_filters[doi] = domain_filter
    
    return doi_to_domain_filters


def _convert_publication_date_to_mm_dd_yyyy(publication_date: Optional[str]) -> Optional[str]:
    """
    Convert publication date from various formats to MM/DD/YYYY format for Perplexity.
    
    Supports formats like:
    - "23 October 2023" -> "10/23/2023"
    - "Oct 23, 2023" -> "10/23/2023"
    - "2023-10-23" -> "10/23/2023"
    - Already in MM/DD/YYYY format -> returns as-is
    
    Args:
        publication_date: Date string in various formats
    
    Returns:
        Date string in MM/DD/YYYY format, or None if parsing fails
    """
    if not publication_date:
        return None
    
    # If already in MM/DD/YYYY format, validate and return
    if _validate_date_filter(publication_date):
        return publication_date
    
    # Try to parse the date using common formats
    date_formats = [
        "%d %B %Y",      # "23 October 2023"
        "%B %d, %Y",     # "October 23, 2023"
        "%b %d, %Y",     # "Oct 23, 2023"
        "%d %b %Y",      # "23 Oct 2023"
        "%Y-%m-%d",      # "2023-10-23"
        "%Y-%m",         # "2023-10" (fallback to first day of month)
        "%Y",            # "2023" (fallback to Jan 1)
    ]
    
    for fmt in date_formats:
        try:
            dt = datetime.strptime(publication_date.strip(), fmt)
            # Convert to MM/DD/YYYY format (handle Windows vs Unix)
            # Use format without leading zeros for month/day
            try:
                # Try Unix format first (no leading zeros)
                return dt.strftime("%-m/%-d/%Y")
            except ValueError:
                # Fall back to Windows format or format with leading zeros
                return dt.strftime("%m/%d/%Y")
        except ValueError:
            continue
    
    # Try dateutil parser as fallback
    try:
        from dateutil import parser
        dt = parser.parse(publication_date)
        try:
            return dt.strftime("%-m/%-d/%Y")
        except ValueError:
            return dt.strftime("%m/%d/%Y")
    except (ImportError, ValueError):
        pass
    
    logger.warning(
        "Could not parse publication date '%s' for Perplexity filter. "
        "Expected format: MM/DD/YYYY or common date formats.",
        publication_date
    )
    return None


async def _call_perplexity_with_retry(
    client: Any,
    kwargs: Dict[str, Any],
    max_retries: int = 3,
) -> Any:
    """
    Call Perplexity API with retry logic for rate limits and connection errors.
    
    Implements exponential backoff with jitter for rate limits and shorter delays
    for connection errors.
    
    Handles both streaming (stream=True) and non-streaming (stream=False) responses.
    """
    loop = asyncio.get_event_loop()
    is_streaming = kwargs.get("stream", False)
    
    for attempt in range(max_retries):
        try:
            if is_streaming:
                # For streaming, collect all chunks and reconstruct response
                stream = await loop.run_in_executor(
                    None, lambda: client.chat.completions.create(**kwargs)
                )
                # Collect all chunks from the stream
                chunks = []
                accumulated_content = ""
                
                try:
                    for chunk in stream:
                        chunks.append(chunk)
                        
                        # Check for errors in the chunk first
                        if hasattr(chunk, 'error') and chunk.error:
                            error_info = chunk.error
                            error_msg = error_info.get('message', 'Unknown error') if isinstance(error_info, dict) else str(error_info)
                            error_type = error_info.get('type', 'APIError') if isinstance(error_info, dict) else 'APIError'
                            error_code = error_info.get('code', 400) if isinstance(error_info, dict) else 400
                            
                            logger.error("Perplexity API error in stream chunk: %s (type: %s, code: %s)", 
                                       error_msg, error_type, error_code)
                            
                            # Check if it's an invalid domain filter error
                            if error_type == "invalid_search_domain_filter" or "search_domain_filters must be a valid domain name" in error_msg:
                                # This will be caught and handled in call_llm
                                raise ValueError(f"Perplexity API error: {error_msg} (type: {error_type}, code: {error_code})")
                            
                            raise ValueError(f"Perplexity API error: {error_msg} (type: {error_type}, code: {error_code})")
                        
                        # Perplexity SDK may use different chunk structures
                        # Try multiple ways to extract content
                        content_piece = None
                        
                        # Method 1: Standard OpenAI-compatible format (choices[0].delta.content)
                        if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                            choice = chunk.choices[0]
                            # Check for delta (streaming format)
                            if hasattr(choice, 'delta'):
                                delta = choice.delta
                                if hasattr(delta, 'content') and delta.content:
                                    content_piece = delta.content
                            # Check for message (final chunk format)
                            elif hasattr(choice, 'message'):
                                message = choice.message
                                if hasattr(message, 'content') and message.content:
                                    content_piece = message.content
                        
                        # Method 2: Direct content attribute (Perplexity SDK might use this)
                        if not content_piece and hasattr(chunk, 'content'):
                            content_piece = chunk.content
                        
                        # Method 3: Check for text attribute
                        if not content_piece and hasattr(chunk, 'text'):
                            content_piece = chunk.text
                        
                        # Method 4: Check for data attribute (some SDKs use this)
                        if not content_piece and hasattr(chunk, 'data'):
                            data = chunk.data
                            if isinstance(data, dict) and 'content' in data:
                                content_piece = data['content']
                            elif isinstance(data, str):
                                content_piece = data
                        
                        # Method 5: Try to access via __dict__ or getattr for debugging
                        if not content_piece:
                            # Log chunk structure for debugging
                            chunk_attrs = [attr for attr in dir(chunk) if not attr.startswith('_')]
                            logger.debug("Chunk type: %s, attributes: %s", type(chunk).__name__, chunk_attrs[:10])
                            
                            # Try common attribute names
                            for attr in ['content', 'text', 'message', 'delta', 'data', 'body']:
                                if hasattr(chunk, attr):
                                    val = getattr(chunk, attr)
                                    if val and isinstance(val, str):
                                        content_piece = val
                                        break
                                    elif val and isinstance(val, dict) and 'content' in val:
                                        content_piece = val['content']
                                        break
                        
                        if content_piece:
                            if isinstance(content_piece, str):
                                accumulated_content += content_piece
                            else:
                                accumulated_content += str(content_piece)
                    
                    logger.debug("Collected %d chunks, accumulated %d characters", len(chunks), len(accumulated_content))
                    
                    if not accumulated_content and chunks:
                        # Log the structure of first chunk for debugging
                        first_chunk = chunks[0]
                        logger.warning("No content accumulated from %d chunks. First chunk type: %s", 
                                     len(chunks), type(first_chunk).__name__)
                        
                        # Try to inspect chunk structure more deeply
                        try:
                            # Check if chunk has __dict__ or can be converted to dict
                            if hasattr(first_chunk, '__dict__'):
                                logger.warning("First chunk __dict__: %s", first_chunk.__dict__)
                            
                            # Try to access common Perplexity SDK attributes
                            for attr_name in ['delta', 'choices', 'content', 'text', 'message', 'data', 'body', 'response']:
                                if hasattr(first_chunk, attr_name):
                                    attr_val = getattr(first_chunk, attr_name)
                                    logger.warning("First chunk.%s = %s (type: %s)", 
                                                 attr_name, str(attr_val)[:100], type(attr_val).__name__)
                            
                            # Try to get string representation
                            chunk_str = str(first_chunk)
                            logger.warning("First chunk string representation: %s", chunk_str[:500])
                            
                            # Try JSON serialization if possible
                            try:
                                import json
                                if hasattr(first_chunk, '__dict__'):
                                    chunk_dict = first_chunk.__dict__
                                    logger.warning("First chunk as dict (JSON): %s", json.dumps(chunk_dict, default=str)[:500])
                            except:
                                pass
                        except Exception as inspect_error:
                            logger.warning("Error inspecting chunk: %s", str(inspect_error))
                        
                        # Last resort: try to extract from all chunks using any available method
                        for i, chunk in enumerate(chunks):
                            # Try all possible extraction methods
                            for method in [
                                lambda c: getattr(c, 'content', None) if hasattr(c, 'content') else None,
                                lambda c: getattr(c, 'text', None) if hasattr(c, 'text') else None,
                                lambda c: getattr(c.choices[0].delta, 'content', None) if hasattr(c, 'choices') and c.choices and hasattr(c.choices[0], 'delta') else None,
                                lambda c: getattr(c.choices[0].message, 'content', None) if hasattr(c, 'choices') and c.choices and hasattr(c.choices[0], 'message') else None,
                            ]:
                                try:
                                    val = method(chunk)
                                    if val and isinstance(val, str) and val.strip():
                                        accumulated_content += val
                                        logger.info("Extracted content from chunk %d using method: %s", i, method.__name__ if hasattr(method, '__name__') else str(method))
                                        break
                                except:
                                    continue
                    
                    # Create a mock response object with accumulated content
                    class StreamedResponse:
                        def __init__(self, content, last_chunk):
                            self.choices = [type('Choice', (), {
                                'message': type('Message', (), {'content': content})(),
                            })()]
                            self._last_chunk = last_chunk
                            # Try to extract citations from last chunk
                            self.citations = None
                            self.search_results = None
                            if last_chunk:
                                if hasattr(last_chunk, 'citations'):
                                    self.citations = last_chunk.citations
                                    logger.info("Found citations in last chunk: %s", self.citations)
                                    print(f"Citations from Perplexity API (streaming): {self.citations}")
                                if hasattr(last_chunk, 'search_results'):
                                    self.search_results = last_chunk.search_results
                                    logger.info("Found search_results in last chunk: %s", self.search_results)
                                    if self.search_results:
                                        print(f"\nSearch results from Perplexity API (streaming) ({len(self.search_results)} results):")
                                        for i, result in enumerate(self.search_results, 1):
                                            title = getattr(result, 'title', 'N/A') if hasattr(result, 'title') else (result.get('title', 'N/A') if isinstance(result, dict) else 'N/A')
                                            url = getattr(result, 'url', 'N/A') if hasattr(result, 'url') else (result.get('url', 'N/A') if isinstance(result, dict) else 'N/A')
                                            date = getattr(result, 'date', None) if hasattr(result, 'date') else (result.get('date', None) if isinstance(result, dict) else None)
                                            print(f"  [{i}] {title}")
                                            print(f"      URL: {url}")
                                            if date:
                                                print(f"      Date: {date}")
                                    else:
                                        print("Search results from Perplexity API (streaming): None")
                                # Log all attributes of last chunk
                                chunk_attrs = [attr for attr in dir(last_chunk) if not attr.startswith('_')]
                                logger.debug("Last chunk attributes: %s", chunk_attrs)
                                if hasattr(last_chunk, '__dict__'):
                                    logger.debug("Last chunk __dict__: %s", last_chunk.__dict__)
                    
                    return StreamedResponse(accumulated_content, chunks[-1] if chunks else None)
                except Exception as stream_error:
                    logger.error("Error processing stream: %s: %s", type(stream_error).__name__, str(stream_error))
                    import traceback
                    logger.error("Stream error traceback: %s", traceback.format_exc())
                    raise
            else:
                # Non-streaming response
                response = await loop.run_in_executor(
                    None, lambda: client.chat.completions.create(**kwargs)
                )
                return response
            
        except Exception as e:
            # Check if it's a domain filter error - these should be immediately propagated
            # to the sanitization logic, not retried here
            error_type = type(e).__name__
            error_msg = str(e)
            error_msg_lower = error_msg.lower()
            
            # Check for domain filter errors (both in ValueError and in the error message)
            if ("invalid_search_domain_filter" in error_msg_lower or 
                "search_domain_filters must be a valid domain name" in error_msg_lower or
                (isinstance(e, ValueError) and "perplexity api error" in error_msg_lower and 
                 ("invalid_search_domain_filter" in error_msg_lower or "search_domain_filters" in error_msg_lower))):
                # Don't retry - let the sanitization logic handle this
                logger.warning("Domain filter error detected - propagating to sanitization logic: %s", str(e))
                raise
            
            # Try to catch RateLimitError if available
            if error_type == "RateLimitError" or "rate limit" in error_msg or "429" in str(e):
                if attempt == max_retries - 1:
                    raise
                delay = (2 ** attempt) + random.uniform(0, 1)
                logger.warning("Rate limited (attempt %d/%d). Retrying in %.2f seconds...", 
                              attempt + 1, max_retries, delay)
                await asyncio.sleep(delay)
                continue
            
            # Check for connection/timeout errors
            if error_type in ("APIConnectionError", "APITimeoutError", "ConnectionError", "TimeoutError") or \
               "connection" in error_msg or "timeout" in error_msg:
                if attempt == max_retries - 1:
                    raise
                delay = 1 + random.uniform(0, 1)
                logger.warning("Connection/timeout error (attempt %d/%d). Retrying in %.2f seconds...",
                              attempt + 1, max_retries, delay)
                await asyncio.sleep(delay)
                continue
            
            # Check for API status errors (server errors)
            if error_type == "APIStatusError" or (hasattr(e, 'status_code') and e.status_code >= 500):
                status_code = getattr(e, 'status_code', None)
                # Don't retry on client errors (4xx)
                if status_code and 400 <= status_code < 500:
                    raise
                if attempt == max_retries - 1:
                    raise
                delay = (2 ** attempt) + random.uniform(0, 1)
                logger.warning("Server error (attempt %d/%d). Retrying in %.2f seconds...",
                              attempt + 1, max_retries, delay)
                await asyncio.sleep(delay)
                continue
            
            # Check for context length errors
            if "context_length_exceeded" in error_msg or "maximum context length" in error_msg:
                raise ContextLengthExceededError(
                    f"Input exceeds context window for model {kwargs.get('model', 'unknown')}"
                ) from e
            
            # For other errors, retry with exponential backoff
            if attempt == max_retries - 1:
                raise
            delay = (2 ** attempt) + random.uniform(0, 1)
            logger.warning("API error (attempt %d/%d): %s. Retrying in %.2f seconds...",
                          attempt + 1, max_retries, error_type, delay)
            await asyncio.sleep(delay)


class PerplexityProvider(LLMProvider):
    """
    Perplexity provider with built-in search capabilities.
    
    Perplexity has its own search and tooling, so MCP integration is not needed.
    The system prompt is integrated directly into the API calls.
    """
    
    def __init__(
        self,
        model: str = "sonar-reasoning-pro",
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries: int = 3,
        timeout: Optional[float] = 30.0,
        use_async_deep_research: bool = False,
        async_poll_interval: float = 2.0,
        async_max_wait: float = 300.0,
        search_mode: str = "academic",
        web_search_context_size: str = "high",
        reasoning_effort: str = "high",
        search_type: str = "auto",
    ):
        """
        Initialize Perplexity provider.
        
        Args:
            model: Model name (default: "sonar-reasoning-pro", etc.)
            api_key: API key for Perplexity API
            temperature: Sampling temperature (0.0 to 2.0). Default: 0.2
            max_retries: Maximum number of retry attempts for rate limits and connection errors (default: 3)
            timeout: Request timeout in seconds (default: 30.0)
            use_async_deep_research: If True, use async/deep research API with polling (default: False)
            async_poll_interval: Seconds between status checks when using async mode (default: 2.0)
            async_max_wait: Maximum seconds to wait for async request completion (default: 300.0)
            search_mode: Search mode (default: "academic")
            web_search_context_size: Web search context size (default: "high")
            reasoning_effort: Computational effort for reasoning ("low", "medium", "high"). Default: "high"
                - "low": Faster, simpler answers with reduced token usage
                - "medium": Balanced approach 
                - "high": Deeper, more thorough responses with increased token usage
            search_type: Search type for search configurations ("fast", "pro", or "auto"). Default: "auto"
                - "fast": Faster search mode
                - "pro": More thorough search mode
                - "auto": Automatic search mode selection
            
        Note: The "sonar-deep-research" and "sonar-reasoning-pro" models automatically use stream=True.
        """
        super().__init__(model, api_key)
        
        if not PERPLEXITY_AVAILABLE:
            raise ImportError(
                "Perplexity package not installed. Install with: pip install perplexity"
            )
        
        api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise ValueError(
                "Perplexity API key required. Set PERPLEXITY_API_KEY environment variable "
                "or provide api_key parameter."
            )
        
        # Use Perplexity SDK
        # Note: Perplexity SDK may not support timeout/max_retries in constructor
        # We'll handle retries in _call_perplexity_with_retry
        self.client = Perplexity(api_key=api_key)
        # Default temperature is 0.2 for Perplexity
        self.temperature = temperature if temperature is not None else 0.2
        self.max_retries = max_retries
        self.timeout = timeout
        self.use_async_deep_research = use_async_deep_research
        self.async_poll_interval = async_poll_interval
        self.async_max_wait = async_max_wait
        self.search_mode = search_mode
        self.web_search_context_size = web_search_context_size
        
        # Validate search_type
        valid_search_types = ["fast", "pro", "auto"]
        if search_type not in valid_search_types:
            raise ValueError(
                f"Invalid search_type: {search_type}. "
                f"Must be one of: {valid_search_types}"
            )
        self.search_type = search_type
        
        # Validate and set reasoning_effort
        valid_effort_levels = ["low", "medium", "high"]
        if reasoning_effort not in valid_effort_levels:
            raise ValueError(
                f"Invalid reasoning_effort: {reasoning_effort}. "
                f"Must be one of: {valid_effort_levels}"
            )
        self.reasoning_effort = reasoning_effort
    
    def format_tools(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """
        Format tools for Perplexity API.
        
        Note: Perplexity has built-in search, so tools are typically not needed.
        This method is implemented for interface compatibility but returns empty list.
        """
        logger.info("Perplexity has built-in search capabilities, no external tools needed")
        return []
    
    async def call_llm(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        domain_filter: Optional[List[str]] = None,
        search_before_date_filter: Optional[str] = None,
        cochrane_titles: Optional[List[str]] = None,
        log_dir: Optional[str] = None,
        result_filter: Optional[CochraneResultFilter] = None,
        enable_filtering: bool = True,
    ) -> Tuple[Any, Optional[str], List[Dict[str, Any]], Optional[str]]:
        """
        Call Perplexity API with messages.
        
        Perplexity has built-in search, so tools parameter is ignored.
        The system prompt (RESEARCH_ASSISTANT_PROMPT) is automatically included.
        
        For sonar-pro and sonar-deep-research models, if cochrane_titles is provided, performs iterative domain filtering:
        - Extracts search results from each response
        - Filters results based on Cochrane titles
        - Adds matching URLs to domain filter
        - Repeats until no more Cochrane-related titles are found or 20 entries reached
        - Saves final filter list to log directory
        
        Domain filtering is always applied when domain_filter is provided, regardless of model.
        
        Args:
            messages: Conversation history
            tools: Optional list of tools (ignored for Perplexity)
            max_tokens: Ignored (Perplexity uses model defaults)
            domain_filter: List of domains/URLs to filter for this call. Use "-" prefix for denylist mode (e.g., ["-example.com", "https://allowed.com"])
            search_before_date_filter: Date filter in MM/DD/YYYY format (e.g., "3/5/2025"). Only include results before this date.
            cochrane_titles: Optional list of Cochrane review titles for iterative filtering (for sonar-reasoning-pro and sonar-deep-research)
            log_dir: Optional log directory path to save the final filter list (for sonar-reasoning-pro and sonar-deep-research with cochrane_titles)
            enable_filtering: Whether filtering is enabled. If False, warnings about missing domain filters will be suppressed.
        
        Returns:
            Tuple of (response, text_content, tool_calls, reasoning_summary)
            - response: Raw response object from Perplexity
            - text_content: Text response from Perplexity
            - tool_calls: Empty list (Perplexity doesn't use external tool calls)
            - reasoning_summary: None (Perplexity doesn't provide reasoning summaries)
        """
        # Validate date filter if provided
        if search_before_date_filter and not _validate_date_filter(search_before_date_filter):
            raise ValueError(
                f"Invalid date filter format: {search_before_date_filter}. "
                "Expected format: MM/DD/YYYY (e.g., '3/5/2025')"
            )
        
        # Prepare messages (needed for both regular and iterative filtering)
        # Check if system message is already present
        has_system_message = any(msg.get("role") == "system" for msg in messages)
        
        # Prepare messages with system prompt if not present
        api_messages = []
        if not has_system_message:
            # Add system prompt at the beginning
            api_messages.append({
                "role": "system",
                "content": RESEARCH_ASSISTANT_PROMPT
            })
        
        # Add all other messages
        for msg in messages:
            role = msg.get("role")
            if role == "system" and has_system_message:
                # If system message exists in messages, use it instead of default
                api_messages.append({
                    "role": "system",
                    "content": msg.get("content", "")
                })
            elif role in ["user", "assistant"]:
                api_messages.append({
                    "role": role,
                    "content": msg.get("content", "")
                })
            # Skip tool messages as Perplexity doesn't use external tools
        
        # For sonar-reasoning-pro and sonar-deep-research with Cochrane titles, perform iterative domain filtering
        # BUT: Skip iterative filtering for sonar-deep-research if domain_filter is already provided
        # (this means we're using pre-loaded filters from JSON, so no need for iterative filtering)
        should_use_iterative_filtering = (
            (self.model == "sonar-reasoning-pro" or self.model == "sonar-deep-research") and 
            cochrane_titles and 
            not self.use_async_deep_research and
            # For deep research, skip iterative filtering if domain_filter is provided (pre-loaded from JSON)
            not (self.model == "sonar-deep-research" and domain_filter is not None)
        )
        
        if should_use_iterative_filtering:
            return await self._call_llm_with_iterative_filtering(
                api_messages=api_messages,
                domain_filter=domain_filter,
                search_before_date_filter=search_before_date_filter,
                cochrane_titles=cochrane_titles,
                log_dir=log_dir,
                result_filter=result_filter,
            )
        
        # Build API call parameters for standard sync mode
        # sonar-deep-research and sonar-reasoning-pro require stream=True
        stream_value = self.model in ["sonar-deep-research", "sonar-reasoning-pro"]
        
        kwargs = {
            "model": self.model,
            "messages": api_messages,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
        }
        
        # Only add stream if needed (sonar-deep-research or sonar-reasoning-pro)
        if stream_value:
            kwargs["stream"] = True
        
        # Build web search options for sonar-deep-research and sonar-reasoning-pro
        web_search_options = {}
        if search_before_date_filter:
            web_search_options["search_before_date_filter"] = search_before_date_filter
        
        # Add search_type, search_mode, and web_search_context_size for sonar-deep-research and sonar-reasoning-pro
        if self.model in ["sonar-deep-research", "sonar-reasoning-pro"]:
            web_search_options["search_type"] = self.search_type
            web_search_options["search_mode"] = self.search_mode
            web_search_options["web_search_context_size"] = self.web_search_context_size
        
        # Add web_search_options as top-level parameter only if it has content
        if web_search_options:
            kwargs["web_search_options"] = web_search_options
        
        # Add domain filter if provided (per-call parameter)
        # Use search_domain_filter (not domain_filter) - Perplexity uses search_domain_filter parameter name
        final_domain_filter = None
        if domain_filter:
            # Deduplicate and sanitize the domain filter (same logic as iterative filtering)
            # This ensures consistent behavior whether using iterative filtering or pre-combined filters
            # First, ensure all entries have "-" prefix for denylist mode
            normalized_filter = [
                entry if entry.startswith('-') else f"-{entry}"
                for entry in domain_filter
            ]
            
            # Deduplicate and sanitize
            sanitized_domain_filter = _deduplicate_and_sanitize_domains(normalized_filter)
            
            # Ensure default Cochrane domains are present
            has_cochrane_org, has_cochranelibrary = _has_any_cochrane_domain(sanitized_domain_filter)
            
            default_cochrane_domains = [
                "-cochranelibrary.com/",
                "-cochrane.org/"
            ]
            if not has_cochranelibrary:
                sanitized_domain_filter.append(default_cochrane_domains[0])
                logger.debug("Added default Cochrane domain: %s", default_cochrane_domains[0])
            if not has_cochrane_org:
                sanitized_domain_filter.append(default_cochrane_domains[1])
                logger.debug("Added default Cochrane domain: %s", default_cochrane_domains[1])
            
            # Final deduplication after adding defaults
            final_domain_filter = _deduplicate_and_sanitize_domains(sanitized_domain_filter)
            
            kwargs["search_domain_filter"] = final_domain_filter
            
            # Log the complete domain filter list before querying (especially important for deep research)
            if self.model == "sonar-deep-research":
                logger.info("=" * 80)
                logger.info("Domain filter for %s (pre-loaded, no iterative filtering):", self.model)
                logger.info("Total domain filter entries: %d (from %d original)", len(final_domain_filter), len(domain_filter))
                logger.info("Complete domain filter list:")
                for i, domain in enumerate(final_domain_filter, 1):
                    logger.info("  %2d. %s", i, domain)
                logger.info("=" * 80)
            else:
                logger.info(
                    "Domain filter applied for %s: %d entries (after deduplication and sanitization from %d original)",
                    self.model, len(final_domain_filter), len(domain_filter)
                )
                logger.debug("Domain filter entries: %s", final_domain_filter)
        else:
            # Only warn if filtering is enabled but no domain filter was provided
            # If filtering is disabled (enable_filtering=False), this is intentional and not a warning
            if self.model in ["sonar-deep-research", "sonar-reasoning-pro"] and enable_filtering:
                logger.warning(
                    "No domain filter provided for %s. "
                    "Consider providing domain_filter to prevent leakage of ground-truth answers.",
                    self.model
                )
        
        # Log API call parameters
        # Use final_domain_filter if it was created, otherwise use domain_filter
        domain_filter_for_log = final_domain_filter if final_domain_filter is not None else domain_filter
        logger.debug(
            "Perplexity API call: model=%s, messages=%d, temperature=%s, reasoning_effort=%s, "
            "stream=%s, date_filter=%s, domain_filter=%s, max_retries=%d, timeout=%s",
            self.model, len(api_messages), self.temperature, self.reasoning_effort, 
            stream_value, search_before_date_filter, domain_filter_for_log, self.max_retries, self.timeout
        )
        if self.model in ["sonar-deep-research", "sonar-reasoning-pro"]:
            logger.info("Using %s model with stream=True and search_type=auto", self.model)
            if domain_filter_for_log:
                logger.info("Domain filtering enabled with %d entries", len(domain_filter_for_log))
        
        # Call API with retry logic and domain filter sanitization
        max_domain_sanitization_attempts = 3
        sanitization_attempt = 0
        
        while sanitization_attempt < max_domain_sanitization_attempts:
            try:
                # Verify search_domain_filter is in kwargs before API call (for deep research)
                # Only warn if filtering is enabled - if filtering is disabled, this is intentional
                if self.model == "sonar-deep-research":
                    if "search_domain_filter" in kwargs:
                        logger.info(
                            "✅ VERIFIED: search_domain_filter is in kwargs for API call: %d entries",
                            len(kwargs["search_domain_filter"])
                        )
                        logger.debug("search_domain_filter entries: %s", kwargs["search_domain_filter"])
                    elif enable_filtering:
                        # Only warn if filtering is enabled - if disabled, this is intentional
                        logger.warning(
                            "❌ WARNING: search_domain_filter NOT found in kwargs for deep research API call!"
                        )
                        logger.debug("Available kwargs keys: %s", list(kwargs.keys()))
                
                response = await _call_perplexity_with_retry(
                    client=self.client,
                    kwargs=kwargs,
                    max_retries=self.max_retries,
                )
                # Success - break out of sanitization loop
                break
            except ContextLengthExceededError:
                # Re-raise context length errors as-is
                raise
            except ValueError as e:
                error_msg = str(e)
                # Check if it's an invalid domain filter error
                if "invalid_search_domain_filter" in error_msg or "search_domain_filters must be a valid domain name" in error_msg:
                    if domain_filter and sanitization_attempt < max_domain_sanitization_attempts - 1:
                        # Extract the problematic domain from the error message
                        problematic_domain = _extract_problematic_domain_from_error(error_msg)
                        
                        if problematic_domain:
                            # Check if the problematic domain is a PubMed/PMC domain
                            if _is_pubmed_pmc_domain(problematic_domain):
                                logger.warning(
                                    "Invalid domain filter error for PubMed/PMC domain: %s. "
                                    "Removing query parameters and fragments to sanitize the URL.",
                                    problematic_domain
                                )
                                # Sanitize by removing query parameters and fragments
                                sanitized_pubmed = _sanitize_pubmed_pmc_url(problematic_domain)
                                
                                # If sanitization returns None, it's a search URL - exclude it
                                if sanitized_pubmed is None:
                                    logger.warning(
                                        "PubMed search URL detected and excluded: %s",
                                        problematic_domain
                                    )
                                    # Remove the problematic domain from the filter list
                                    sanitized_domain_filter = []
                                    found_and_removed = False
                                    
                                    for domain in domain_filter:
                                        # Normalize both for comparison
                                        normalized_domain = _normalize_url_for_comparison(domain)
                                        normalized_problematic = _normalize_url_for_comparison(problematic_domain)
                                        
                                        # Check if this domain matches the problematic one
                                        if (domain == problematic_domain or 
                                            normalized_domain == normalized_problematic):
                                            # Skip this domain (don't add it to the new list)
                                            found_and_removed = True
                                            logger.info("Removed PubMed search URL from filter list")
                                        else:
                                            sanitized_domain_filter.append(domain)
                                    
                                    if found_and_removed:
                                        domain_filter = sanitized_domain_filter
                                        sanitization_attempt += 1
                                        logger.info("Retrying without the excluded search URL...")
                                        continue  # Retry without the search URL
                                    else:
                                        # Fallback: try normalized comparison and remove if found
                                        normalized_problematic = _normalize_url_for_comparison(problematic_domain)
                                        filtered_list = []
                                        removed_any = False
                                        for d in domain_filter:
                                            normalized_d = _normalize_url_for_comparison(d)
                                            if normalized_d == normalized_problematic:
                                                removed_any = True
                                                logger.info("Removed search URL using normalized comparison")
                                            else:
                                                filtered_list.append(d)
                                        
                                        if removed_any:
                                            domain_filter = filtered_list
                                            sanitization_attempt += 1
                                            logger.info("Retrying without the excluded search URL...")
                                            continue
                                        else:
                                            logger.warning(
                                                "Search URL not found in filter list. It may have already been removed. Continuing..."
                                            )
                                            sanitization_attempt += 1
                                            continue
                                
                                logger.info(
                                    "Sanitized PubMed/PMC URL: %s -> %s",
                                    problematic_domain, sanitized_pubmed
                                )
                                
                                # Find and replace the problematic domain in the filter list
                                sanitized_domain_filter = []
                                found_and_sanitized = False
                                
                                for domain in domain_filter:
                                    # Normalize both for comparison
                                    normalized_domain = _normalize_url_for_comparison(domain)
                                    normalized_problematic = _normalize_url_for_comparison(problematic_domain)
                                    
                                    # Check if this domain matches the problematic one
                                    if (domain == problematic_domain or 
                                        normalized_domain == normalized_problematic):
                                        # Ensure sanitized version has "-" prefix if original had it
                                        if domain.startswith('-') and not sanitized_pubmed.startswith('-'):
                                            sanitized_pubmed = f"-{sanitized_pubmed}"
                                        elif not domain.startswith('-') and sanitized_pubmed.startswith('-'):
                                            sanitized_pubmed = sanitized_pubmed.lstrip('-')
                                        sanitized_domain_filter.append(sanitized_pubmed)
                                        found_and_sanitized = True
                                    else:
                                        sanitized_domain_filter.append(domain)
                                
                                if found_and_sanitized:
                                    domain_filter = sanitized_domain_filter
                                    sanitization_attempt += 1
                                    logger.info("Retrying with sanitized PubMed/PMC URL...")
                                    continue  # Retry with sanitized domain
                                else:
                                    # Fallback: if we can't find the domain in the list, 
                                    # try adding the sanitized version directly
                                    normalized_problematic = _normalize_url_for_comparison(problematic_domain)
                                    logger.warning(
                                        "Could not find problematic domain in filter list to sanitize. "
                                        "Adding sanitized version directly: %s",
                                        sanitized_pubmed
                                    )
                                    # Ensure sanitized version has "-" prefix for consistency
                                    if not sanitized_pubmed.startswith('-'):
                                        sanitized_pubmed = f"-{sanitized_pubmed}"
                                    # Check if it's already in the list (normalized comparison)
                                    normalized_sanitized = _normalize_url_for_comparison(sanitized_pubmed)
                                    already_present = any(
                                        _normalize_url_for_comparison(d) == normalized_sanitized 
                                        for d in domain_filter
                                    )
                                    if not already_present:
                                        domain_filter.append(sanitized_pubmed)
                                        logger.info("Added sanitized PubMed/PMC URL to filter list")
                                        sanitization_attempt += 1
                                        continue  # Retry with sanitized domain added
                                    else:
                                        logger.warning("Sanitized version already in filter list, removing problematic one")
                                        # Remove the problematic one by normalizing and comparing
                                        filtered_list = []
                                        for d in domain_filter:
                                            if _normalize_url_for_comparison(d) != normalized_problematic:
                                                filtered_list.append(d)
                                        domain_filter = filtered_list
                                        sanitization_attempt += 1
                                        continue  # Retry with problematic domain removed
                            
                            logger.warning(
                                "Invalid domain filter error detected for domain: %s. Sanitizing only this domain (attempt %d/%d)...",
                                problematic_domain, sanitization_attempt + 1, max_domain_sanitization_attempts
                            )
                            
                            # Find and sanitize only the problematic domain in the filter list
                            sanitized_domain_filter = []
                            found_and_sanitized = False
                            
                            for domain in domain_filter:
                                # Never sanitize PubMed/PMC domains - skip them
                                if _is_pubmed_pmc_domain(domain):
                                    sanitized_domain_filter.append(domain)
                                    continue
                                
                                # Check if this domain matches the problematic one (with or without denylist prefix)
                                domain_without_prefix = domain.lstrip('-')
                                problematic_without_prefix = problematic_domain.lstrip('-')
                                
                                # Check if the problematic domain starts with this domain (to handle path cases)
                                # This handles cases where the domain in the list has a path that matches the problematic domain
                                if (domain_without_prefix == problematic_without_prefix or 
                                    problematic_without_prefix.startswith(domain_without_prefix + '/') or
                                    domain_without_prefix.startswith(problematic_without_prefix.split('/')[0])):
                                    # This is the problematic domain - sanitize it aggressively (shorten to base path)
                                    sanitized = _sanitize_url_aggressively(domain, max_path_segments=3)
                                    sanitized_domain_filter.append(sanitized)
                                    found_and_sanitized = True
                                    logger.info("Sanitized domain aggressively (shortened to base path): %s -> %s", domain, sanitized)
                                    print(f"Sanitized domain: {domain} -> {sanitized}")
                                else:
                                    # Keep other domains unchanged
                                    sanitized_domain_filter.append(domain)
                            
                            if not found_and_sanitized:
                                # If we couldn't match, try sanitizing the problematic domain directly
                                # Handle PubMed/PMC domains specially
                                if _is_pubmed_pmc_domain(problematic_domain):
                                    sanitized_pubmed = _sanitize_pubmed_pmc_url(problematic_domain)
                                    
                                    # If sanitization returns None, it's a search URL - exclude it
                                    if sanitized_pubmed is None:
                                        logger.warning(
                                            "PubMed search URL detected and will be excluded: %s",
                                            problematic_domain
                                        )
                                        # Don't add it to the list - it's already been filtered out
                                        sanitization_attempt += 1
                                        continue
                                    
                                    # Add the sanitized version
                                    if not sanitized_pubmed.startswith('-'):
                                        sanitized_pubmed = f"-{sanitized_pubmed}"
                                    sanitized_domain_filter.append(sanitized_pubmed)
                                    logger.info("Added sanitized PubMed/PMC URL to filter list: %s", sanitized_pubmed)
                                    domain_filter = sanitized_domain_filter
                                    sanitization_attempt += 1
                                    continue
                                
                                sanitized_problematic = _sanitize_url_aggressively(problematic_domain, max_path_segments=3)
                                # Add it if not already in the list, or replace if we can find a match
                                logger.warning("Could not find exact match for problematic domain. Adding aggressively sanitized version: %s", sanitized_problematic)
                                print(f"Sanitized problematic domain: {problematic_domain} -> {sanitized_problematic}")
                                if sanitized_problematic not in sanitized_domain_filter:
                                    sanitized_domain_filter.append(sanitized_problematic)
                            
                            domain_filter = sanitized_domain_filter
                            kwargs["search_domain_filter"] = sanitized_domain_filter
                            
                            logger.info("Retrying with sanitized domain filter: %s", sanitized_domain_filter)
                            print(f"Domain filter list (after sanitization): {sanitized_domain_filter}")
                            sanitization_attempt += 1
                            continue
                        else:
                            # Could not extract problematic domain from error message
                            # Try one more time with a more aggressive extraction attempt
                            logger.warning("Could not extract problematic domain from error message. Error: %s", error_msg)
                            logger.warning("Attempting to find domain-like strings in error message...")
                            
                            # Try to find any domain-like string in the error and match it against our filter list
                            import re
                            # Look for any URL or domain pattern in the error
                            potential_domains = re.findall(r'(https?://[^\s,)]+|[a-zA-Z0-9][a-zA-Z0-9-]*\.[a-zA-Z]{2,}[^\s,)]*)', error_msg)
                            
                            found_match = False
                            if potential_domains:
                                sanitized_domain_filter = []
                                for domain in domain_filter:
                                    domain_without_prefix = domain.lstrip('-')
                                    # Check if any potential domain from error matches this filter entry
                                    matched = False
                                    for potential in potential_domains:
                                        potential_clean = potential.strip('"\'')
                                        # Check if potential domain matches or is contained in filter entry
                                        if (potential_clean == domain_without_prefix or 
                                            domain_without_prefix.startswith(potential_clean) or
                                            potential_clean.startswith(domain_without_prefix.split('/')[0])):
                                            # This might be the problematic domain - sanitize it
                                            if _is_pubmed_pmc_domain(domain):
                                                sanitized_domain_filter.append(domain)
                                            else:
                                                sanitized = _sanitize_url_aggressively(domain, max_path_segments=3)
                                                sanitized_domain_filter.append(sanitized)
                                                logger.info("Sanitized potential problematic domain: %s -> %s", domain, sanitized)
                                            matched = True
                                            found_match = True
                                            break
                                    
                                    if not matched:
                                        # Keep unchanged if no match
                                        sanitized_domain_filter.append(domain)
                                
                                if found_match:
                                    domain_filter = sanitized_domain_filter
                                    kwargs["search_domain_filter"] = sanitized_domain_filter
                                    logger.info("Retrying with sanitized domain filter based on potential matches: %s", sanitized_domain_filter)
                                    sanitization_attempt += 1
                                    continue
                            
                            # If we still can't identify the problematic domain, raise an error
                            # rather than blanket sanitizing all domains
                            logger.error(
                                "Could not extract problematic domain from error message after multiple attempts. "
                                "Error: %s. Domain filter: %s",
                                error_msg, domain_filter
                            )
                            raise ValueError(
                                f"Could not identify problematic domain in error message: {error_msg}. "
                                "Please check the domain filter list for invalid entries."
                            )
                    else:
                        # Max sanitization attempts reached or no domain filter
                        logger.error(
                            "Failed to sanitize domain filter after %d attempts. Error: %s",
                            max_domain_sanitization_attempts, error_msg
                        )
                        raise
                else:
                    # Not a domain filter error, re-raise
                    raise
            except Exception as e:
                # Log any other unexpected errors
                logger.error(
                    "Unexpected error calling Perplexity API: %s: %s",
                    type(e).__name__, str(e)
                )
                raise
        
        # Parse response
        text_content = None
        logger.debug("Parsing response. Response type: %s", type(response).__name__)
        
        # Check for citations/references in the response object
        citations = None
        search_results = None
        if response:
            # Check for citations attribute
            if hasattr(response, 'citations'):
                citations = response.citations
                logger.info("Found citations in response: %s", citations)
                print(f"Citations from Perplexity API: {citations}")
            # Check for search_results attribute (ApiPublicSearchResult[])
            if hasattr(response, 'search_results'):
                search_results = response.search_results
                logger.info("Found search_results in response: %s", search_results)
                # Don't print here - already printed in streaming path or will be handled by mcp_client
            # Log all non-private attributes to see what's available
            response_attrs = [attr for attr in dir(response) if not attr.startswith('_')]
            logger.debug("Response attributes: %s", response_attrs)
            # Check if response has a __dict__ to inspect
            if hasattr(response, '__dict__'):
                logger.debug("Response __dict__ keys: %s", list(response.__dict__.keys()))
        
        if response and hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            logger.debug("Found choice, type: %s", type(choice).__name__)
            if hasattr(choice, 'message'):
                message = choice.message
                logger.debug("Found message, type: %s, has content: %s", 
                           type(message).__name__, hasattr(message, 'content'))
                if hasattr(message, 'content') and message.content:
                    text_content = message.content
                    logger.debug("Extracted content length: %d", len(text_content))
                else:
                    logger.warning("Response message has no content attribute or content is None/empty")
                    # Try to get content as string
                    if hasattr(message, 'content'):
                        content_val = message.content
                        logger.warning("Content value: %s (type: %s)", content_val, type(content_val).__name__)
            else:
                logger.warning("Response choice has no message attribute. Choice attributes: %s", 
                             dir(choice) if hasattr(choice, '__dict__') else "N/A")
        else:
            logger.warning("Response has no choices or choices is empty. Response type: %s, attributes: %s", 
                         type(response).__name__, 
                         [attr for attr in dir(response) if not attr.startswith('_')] if response else "None")
            # Try to extract content directly if response has a different structure
            if hasattr(response, 'content'):
                text_content = response.content
                logger.info("Extracted content from response.content: %d chars", len(text_content) if text_content else 0)
            elif hasattr(response, 'text'):
                text_content = response.text
                logger.info("Extracted content from response.text: %d chars", len(text_content) if text_content else 0)
            elif hasattr(response, 'output_text'):
                text_content = response.output_text
                logger.info("Extracted content from response.output_text: %d chars", len(text_content) if text_content else 0)
        
        if not text_content:
            logger.error("Failed to extract text content from response. Response structure: %s", 
                       str(response)[:500] if response else "None")
        
        # Perplexity doesn't use external tool calls (has built-in search)
        tool_calls = []
        
        return response, text_content, tool_calls, None
    
    async def _call_llm_with_iterative_filtering(
        self,
        api_messages: List[Dict[str, Any]],
        domain_filter: Optional[List[str]] = None,
        search_before_date_filter: Optional[str] = None,
        cochrane_titles: Optional[List[str]] = None,
        log_dir: Optional[str] = None,
        result_filter: Optional[CochraneResultFilter] = None,
    ) -> Tuple[Any, Optional[str], List[Dict[str, Any]], Optional[str]]:
        """
        Call Perplexity API with iterative domain filtering based on Cochrane titles.
        
        This method:
        1. Makes an initial API call with the provided domain_filter
        2. Extracts search results from the response
        3. Filters search results based on Cochrane titles
        4. Adds matching URLs to the domain filter (deduplicated and sanitized)
        5. Repeats steps 1-4 until no more Cochrane-related titles are found OR 20 entries reached
        6. Saves the final filter list to a file in log_dir
        
        Best-effort filtering: Caps at 20 entries total. Prioritizes:
        - Manually provided URLs (from domain_filter parameter)
        - Default Cochrane domains (cochrane.org, cochranelibrary.com)
        - Most frequently detected URLs from iterative process
        
        Args:
            api_messages: Messages in API format
            domain_filter: Initial list of domains/URLs to filter
            search_before_date_filter: Date filter in MM/DD/YYYY format
            cochrane_titles: List of Cochrane review titles for filtering
            log_dir: Log directory path to save the final filter list
            result_filter: Optional CochraneResultFilter with full filtering mechanism (URL, title keyword, title match, date)
        
        Returns:
            Tuple of (response, text_content, tool_calls, reasoning_summary) from the final call
        """
        MAX_DOMAIN_FILTER_ENTRIES = 20  # Cap at 20 entries
        
        # Use full CochraneResultFilter if provided, otherwise create title filter from list
        if result_filter:
            # Use the provided filter which includes all checks (URL, title keyword, title match, date)
            cochrane_filter = result_filter
            logger.info("Using provided CochraneResultFilter for iterative filtering")
        elif cochrane_titles:
            # Create a full CochraneResultFilter with title list and date cutoff
            # Convert search_before_date_filter from MM/DD/YYYY to the format expected by CochraneResultFilter
            publication_date_str = None
            if search_before_date_filter:
                # CochraneResultFilter expects format like "09 June 2025"
                try:
                    from dateutil import parser
                    dt = parser.parse(search_before_date_filter)
                    publication_date_str = dt.strftime("%d %B %Y")
                except (ImportError, ValueError):
                    logger.warning("Could not parse search_before_date_filter for CochraneResultFilter: %s", search_before_date_filter)
            
            cochrane_filter = CochraneResultFilter(
                title_filter_list=cochrane_titles,
                publication_date=publication_date_str
            )
            logger.info("Created CochraneResultFilter with %d titles and date cutoff: %s", 
                       len(cochrane_titles), publication_date_str or "none")
        else:
            cochrane_filter = None
            logger.warning("No Cochrane filter available for iterative filtering")
        
        # Helper function to check if a domain is a default Cochrane domain
        def _is_default_cochrane_domain(domain: str) -> bool:
            """Check if domain is a default Cochrane domain."""
            domain_lower = domain.lstrip('-').lower()
            return "cochrane" in domain_lower
        
        # Use the module-level _has_any_cochrane_domain helper function (defined above)
        
        # Try to load previously saved domain_filter_list.json if log_dir is provided
        saved_domain_filter = None
        if log_dir:
            try:
                from pathlib import Path
                log_path = Path(log_dir)
                filter_file = log_path / "domain_filter_list.json"
                if filter_file.exists():
                    with open(filter_file, 'r', encoding='utf-8') as f:
                        saved_data = json.load(f)
                        if 'domain_filter' in saved_data and isinstance(saved_data['domain_filter'], list):
                            saved_domain_filter = saved_data['domain_filter']
                            # Ensure all entries have "-" prefix
                            saved_domain_filter = [
                                entry if entry.startswith('-') else f"-{entry}"
                                for entry in saved_domain_filter
                            ]
                            logger.info(
                                "Loaded saved domain filter list from %s: %d entries (all with '-' prefix)",
                                filter_file, len(saved_domain_filter)
                            )
            except Exception as e:
                logger.warning("Failed to load saved domain filter list from %s: %s", log_dir, e)
        
        # Merge saved domain filter with provided domain_filter
        # Saved filter takes precedence (it's the result of previous iterations)
        # Ensure all entries have "-" prefix
        initial_domain_filter = saved_domain_filter if saved_domain_filter else domain_filter
        if saved_domain_filter and domain_filter:
            # Merge: combine both, with saved filter entries first (they're more complete)
            merged_filter = list(saved_domain_filter)  # Already has "-" prefix from loading
            seen = set(saved_domain_filter)
            for entry in domain_filter:
                # Ensure entry has "-" prefix before adding
                entry_with_prefix = entry if entry.startswith('-') else f"-{entry}"
                if entry_with_prefix not in seen:
                    merged_filter.append(entry_with_prefix)
                    seen.add(entry_with_prefix)
            initial_domain_filter = merged_filter
            logger.info(
                "Merged saved domain filter (%d entries) with provided filter (%d entries): %d total (all with '-' prefix)",
                len(saved_domain_filter), len(domain_filter), len(initial_domain_filter)
            )
        elif initial_domain_filter:
            # Ensure all entries have "-" prefix even if not merging
            initial_domain_filter = [
                entry if entry.startswith('-') else f"-{entry}"
                for entry in initial_domain_filter
            ]
        
        # Initialize domain filter list (copy to avoid modifying original)
        # Sanitize initial domain filter entries to ensure they're in the correct format
        manually_provided_urls = set()  # Track manually provided URLs
        default_cochrane_urls = set()   # Track default Cochrane domains
        current_domain_filter = []
        
        if initial_domain_filter:
            for domain_entry in initial_domain_filter:
                # Ensure it has "-" prefix for denylist mode
                if not domain_entry.startswith('-'):
                    domain_entry = f"-{domain_entry}"
                    logger.debug("Added '-' prefix to domain filter entry: %s", domain_entry)
                
                # If it's already a simple domain (no path), keep it as-is
                # Otherwise, extract domain + path up to directory
                if '/' in domain_entry.lstrip('-') and not _is_pubmed_pmc_domain(domain_entry):
                    # Has a path - sanitize it
                    sanitized = _extract_domain_from_url(domain_entry)
                    # Ensure sanitized version also has "-" prefix
                    if not sanitized.startswith('-'):
                        sanitized = f"-{sanitized}"
                    current_domain_filter.append(sanitized)
                    manually_provided_urls.add(sanitized)
                    logger.debug("Sanitized initial domain filter entry: %s -> %s", domain_entry, sanitized)
                else:
                    # Already a domain or PubMed/PMC - keep as-is (with "-" prefix)
                    current_domain_filter.append(domain_entry)
                    manually_provided_urls.add(domain_entry)
                    # Check if it's a default Cochrane domain
                    if _is_default_cochrane_domain(domain_entry):
                        default_cochrane_urls.add(domain_entry)
        
        # Check if any Cochrane domains exist (before adding defaults)
        has_cochrane_org, has_cochranelibrary = _has_any_cochrane_domain(current_domain_filter)
        
        # Always add default Cochrane domains if not already present (check by domain, not exact match)
        # This ensures Cochrane domains are always blocked, even if no initial domain_filter was provided
        default_cochrane_domains = [
            "-cochranelibrary.com/",
            "-cochrane.org/"
        ]
        if not has_cochranelibrary:
            current_domain_filter.append(default_cochrane_domains[0])
            default_cochrane_urls.add(default_cochrane_domains[0])
            logger.info("Added default Cochrane domain: %s", default_cochrane_domains[0])
        if not has_cochrane_org:
            current_domain_filter.append(default_cochrane_domains[1])
            default_cochrane_urls.add(default_cochrane_domains[1])
            logger.info("Added default Cochrane domain: %s", default_cochrane_domains[1])
        
        # Deduplicate and sanitize the initial filter
        current_domain_filter = _deduplicate_and_sanitize_domains(current_domain_filter)
        
        # After deduplication, ensure default Cochrane domains are still present
        # (deduplication may have removed specific paths, but generic domains should remain)
        has_cochrane_org, has_cochranelibrary = _has_any_cochrane_domain(current_domain_filter)
        if not has_cochranelibrary:
            current_domain_filter.append(default_cochrane_domains[0])
            default_cochrane_urls.add(default_cochrane_domains[0])
            logger.info("Added default Cochrane domain after deduplication: %s", default_cochrane_domains[0])
        if not has_cochrane_org:
            current_domain_filter.append(default_cochrane_domains[1])
            default_cochrane_urls.add(default_cochrane_domains[1])
            logger.info("Added default Cochrane domain after deduplication: %s", default_cochrane_domains[1])
        
        # Update sets after deduplication (in case deduplication changed entries)
        manually_provided_urls = {url for url in manually_provided_urls if url in current_domain_filter}
        default_cochrane_urls = {url for url in default_cochrane_urls if url in current_domain_filter}
        
        # Track frequency of iteratively detected URLs
        iterative_url_frequency = {}  # Maps URL -> count
        
        # Track all iterations
        max_iterations = 10  # Safety limit
        iteration = 0
        final_response = None
        final_text_content = None
        
        logger.info("Starting iterative domain filtering for %s with %d Cochrane titles (max %d entries)", self.model, 
                   len(cochrane_titles) if cochrane_titles else 0, MAX_DOMAIN_FILTER_ENTRIES)
        
        # If we already have 20+ entries, cap it and make one final call
        if len(current_domain_filter) > MAX_DOMAIN_FILTER_ENTRIES:
            logger.info("Initial domain filter has %d entries, capping to %d most important entries", 
                       len(current_domain_filter), MAX_DOMAIN_FILTER_ENTRIES)
            # Build prioritized list (manually provided + default Cochrane + most frequent iterative)
            prioritized_filter = []
            prioritized_filter_set = set()
            
            # Add manually provided URLs first
            for url in current_domain_filter:
                if url in manually_provided_urls:
                    prioritized_filter.append(url)
                    prioritized_filter_set.add(url)
            
            # Add default Cochrane URLs
            for url in current_domain_filter:
                if url in default_cochrane_urls and url not in prioritized_filter_set:
                    prioritized_filter.append(url)
                    prioritized_filter_set.add(url)
            
            # Add any remaining URLs until we hit the cap (if we have space)
            for url in current_domain_filter:
                if len(prioritized_filter) >= MAX_DOMAIN_FILTER_ENTRIES:
                    break
                if url not in prioritized_filter_set:
                    prioritized_filter.append(url)
                    prioritized_filter_set.add(url)
            
            current_domain_filter = _deduplicate_and_sanitize_domains(prioritized_filter)
            logger.info("Capped domain filter to %d entries", len(current_domain_filter))
        
        while iteration < max_iterations:
            iteration += 1
            logger.info("Iteration %d/%d: Domain filter has %d/%d entries", iteration, max_iterations, len(current_domain_filter), MAX_DOMAIN_FILTER_ENTRIES)
            
            # Make API call with current domain filter
            # Build API call parameters
            # sonar-deep-research and sonar-reasoning-pro require stream=True
            stream_value = self.model in ["sonar-deep-research", "sonar-reasoning-pro"]
            
            kwargs = {
                "model": self.model,
                "messages": api_messages,
                "temperature": self.temperature,
                "reasoning_effort": self.reasoning_effort,
            }
            
            # Only add stream if needed (sonar-deep-research or sonar-reasoning-pro)
            if stream_value:
                kwargs["stream"] = True
            
            # Build web search options for sonar-deep-research and sonar-reasoning-pro
            web_search_options = {}
            if search_before_date_filter:
                web_search_options["search_before_date_filter"] = search_before_date_filter
            
            # Add search_type, search_mode, and web_search_context_size for sonar-deep-research and sonar-reasoning-pro
            if self.model in ["sonar-deep-research", "sonar-reasoning-pro"]:
                web_search_options["search_type"] = self.search_type
                web_search_options["search_mode"] = self.search_mode
                web_search_options["web_search_context_size"] = self.web_search_context_size
            
            if web_search_options:
                kwargs["web_search_options"] = web_search_options
            
            # Add domain filter - ALWAYS use current_domain_filter (updated iteratively)
            # Final safety check: ensure all entries have "-" prefix
            # Perplexity accepts specific URLs, so we include all entries including PubMed/PMC URLs
            if current_domain_filter:
                # Ensure all entries have "-" prefix before sending to API
                sanitized_filter = []
                for entry in current_domain_filter:
                    if not entry.startswith('-'):
                        entry = f"-{entry}"
                        logger.warning(
                            "Iteration %d: Added missing '-' prefix to domain filter entry: %s",
                            iteration, entry
                        )
                    sanitized_filter.append(entry)
                
                current_domain_filter = sanitized_filter
                
                # Send all entries to Perplexity API (including PubMed/PMC URLs)
                kwargs["search_domain_filter"] = current_domain_filter
                pmc_count = sum(1 for entry in current_domain_filter if _is_pubmed_pmc_domain(entry.lstrip('-')))
                logger.info(
                    "Iteration %d: Using domain filter with %d entries for API call (%d PubMed/PMC URLs included)",
                    iteration, len(current_domain_filter), pmc_count
                )
                logger.info(
                    "Iteration %d: Domain filter entries sent to Perplexity API: %s",
                    iteration, current_domain_filter
                )
            else:
                logger.warning(
                    "Iteration %d: No domain filter available - API call will proceed without filtering",
                    iteration
                )
            
            # Call API with retry logic and domain filter sanitization
            max_domain_sanitization_attempts = 3
            sanitization_attempt = 0
            
            while sanitization_attempt < max_domain_sanitization_attempts:
                try:
                    response = await _call_perplexity_with_retry(
                        client=self.client,
                        kwargs=kwargs,
                        max_retries=self.max_retries,
                    )
                    # Success - break out of sanitization loop
                    break
                except ValueError as e:
                    error_msg = str(e)
                    # Check if it's an invalid domain filter error
                    if "invalid_search_domain_filter" in error_msg or "search_domain_filters must be a valid domain name" in error_msg:
                        if sanitization_attempt < max_domain_sanitization_attempts - 1:
                            # Extract the problematic domain from the error message
                            problematic_domain = _extract_problematic_domain_from_error(error_msg)
                            
                            if problematic_domain:
                                logger.warning(
                                    "Invalid domain filter error detected for domain: %s. Sanitizing (attempt %d/%d)...",
                                    problematic_domain, sanitization_attempt + 1, max_domain_sanitization_attempts
                                )
                                
                                # Sanitize the problematic domain
                                if _is_pubmed_pmc_domain(problematic_domain):
                                    logger.warning(
                                        "Invalid domain filter error for PubMed/PMC domain: %s. "
                                        "Removing query parameters and fragments to sanitize the URL.",
                                        problematic_domain
                                    )
                                    # Sanitize by removing query parameters and fragments
                                    sanitized_pubmed = _sanitize_pubmed_pmc_url(problematic_domain)
                                    
                                    # If sanitization returns None, it's a search URL - exclude it
                                    if sanitized_pubmed is None:
                                        logger.warning(
                                            "PubMed search URL detected and excluded: %s",
                                            problematic_domain
                                        )
                                        # Remove the problematic domain from the filter list
                                        sanitized_domain_filter = []
                                        found_and_removed = False
                                        
                                        for domain in current_domain_filter:
                                            # Normalize both for comparison
                                            normalized_domain = _normalize_url_for_comparison(domain)
                                            normalized_problematic = _normalize_url_for_comparison(problematic_domain)
                                            
                                            # Check if this domain matches the problematic one
                                            if (domain == problematic_domain or 
                                                normalized_domain == normalized_problematic):
                                                # Skip this domain (don't add it to the new list)
                                                found_and_removed = True
                                                logger.info("Removed PubMed search URL from filter list")
                                            else:
                                                sanitized_domain_filter.append(domain)
                                        
                                        if found_and_removed:
                                            current_domain_filter = sanitized_domain_filter
                                            kwargs["search_domain_filter"] = current_domain_filter
                                            sanitization_attempt += 1
                                            logger.info("Retrying without the excluded search URL...")
                                            continue  # Retry without the search URL
                                        else:
                                            # Fallback: if we can't find the search URL in the list,
                                            # it might already be filtered out or in a different format.
                                            # Just log a warning and try to continue by removing any matching normalized version
                                            logger.warning(
                                                "Could not find search URL in filter list to remove. "
                                                "Attempting to filter by normalized comparison: %s",
                                                problematic_domain
                                            )
                                            normalized_problematic = _normalize_url_for_comparison(problematic_domain)
                                            filtered_list = []
                                            removed_any = False
                                            for d in current_domain_filter:
                                                normalized_d = _normalize_url_for_comparison(d)
                                                if normalized_d == normalized_problematic:
                                                    removed_any = True
                                                    logger.info("Removed search URL using normalized comparison")
                                                else:
                                                    filtered_list.append(d)
                                            
                                            if removed_any:
                                                current_domain_filter = filtered_list
                                                kwargs["search_domain_filter"] = current_domain_filter
                                                sanitization_attempt += 1
                                                logger.info("Retrying without the excluded search URL...")
                                                continue  # Retry without the search URL
                                            else:
                                                # If we still can't find it, it's likely not in the list.
                                                # This shouldn't happen, but if it does, just log and continue
                                                logger.warning(
                                                    "Search URL not found in filter list even after normalization. "
                                                    "It may have already been removed. Continuing..."
                                                )
                                                # Don't raise an error - just continue with the current filter
                                                sanitization_attempt += 1
                                                continue
                                    
                                    logger.info(
                                        "Sanitized PubMed/PMC URL: %s -> %s",
                                        problematic_domain, sanitized_pubmed
                                    )
                                    
                                    # Find and replace the problematic domain in the filter list
                                    sanitized_domain_filter = []
                                    found_and_sanitized = False
                                    
                                    for domain in current_domain_filter:
                                        # Normalize both for comparison
                                        normalized_domain = _normalize_url_for_comparison(domain)
                                        normalized_problematic = _normalize_url_for_comparison(problematic_domain)
                                        
                                        # Check if this domain matches the problematic one
                                        if (domain == problematic_domain or 
                                            normalized_domain == normalized_problematic):
                                            # Ensure sanitized version has "-" prefix if original had it
                                            if domain.startswith('-') and not sanitized_pubmed.startswith('-'):
                                                sanitized_pubmed = f"-{sanitized_pubmed}"
                                            elif not domain.startswith('-') and sanitized_pubmed.startswith('-'):
                                                sanitized_pubmed = sanitized_pubmed.lstrip('-')
                                            sanitized_domain_filter.append(sanitized_pubmed)
                                            found_and_sanitized = True
                                        else:
                                            sanitized_domain_filter.append(domain)
                                    
                                    if found_and_sanitized:
                                        current_domain_filter = sanitized_domain_filter
                                        kwargs["search_domain_filter"] = current_domain_filter
                                        sanitization_attempt += 1
                                        logger.info("Retrying with sanitized PubMed/PMC URL...")
                                        continue  # Retry with sanitized domain
                                    else:
                                        # Fallback: if we can't find the domain in the list, 
                                        # try adding the sanitized version directly
                                        normalized_problematic = _normalize_url_for_comparison(problematic_domain)
                                        logger.warning(
                                            "Could not find problematic domain in filter list to sanitize. "
                                            "Adding sanitized version directly: %s",
                                            sanitized_pubmed
                                        )
                                        # Ensure sanitized version has "-" prefix for consistency
                                        if not sanitized_pubmed.startswith('-'):
                                            sanitized_pubmed = f"-{sanitized_pubmed}"
                                        # Check if it's already in the list (normalized comparison)
                                        normalized_sanitized = _normalize_url_for_comparison(sanitized_pubmed)
                                        already_present = any(
                                            _normalize_url_for_comparison(d) == normalized_sanitized 
                                            for d in current_domain_filter
                                        )
                                        if not already_present:
                                            current_domain_filter.append(sanitized_pubmed)
                                            logger.info("Added sanitized PubMed/PMC URL to filter list")
                                            kwargs["search_domain_filter"] = current_domain_filter
                                            sanitization_attempt += 1
                                            continue  # Retry with sanitized domain added
                                        else:
                                            logger.warning("Sanitized version already in filter list, removing problematic one")
                                            # Remove the problematic one by normalizing and comparing
                                            filtered_list = []
                                            for d in current_domain_filter:
                                                if _normalize_url_for_comparison(d) != normalized_problematic:
                                                    filtered_list.append(d)
                                            current_domain_filter = filtered_list
                                            kwargs["search_domain_filter"] = current_domain_filter
                                            sanitization_attempt += 1
                                            continue  # Retry with problematic domain removed
                                
                                # Find and sanitize the problematic domain in the filter list
                                sanitized_domain_filter = []
                                found_and_sanitized = False
                                
                                for domain in current_domain_filter:
                                    domain_without_prefix = domain.lstrip('-')
                                    problematic_without_prefix = problematic_domain.lstrip('-')
                                    
                                    # Check if this domain matches the problematic one
                                    if (domain_without_prefix == problematic_without_prefix or 
                                        problematic_without_prefix.startswith(domain_without_prefix + '/') or
                                        domain_without_prefix.startswith(problematic_without_prefix.split('/')[0])):
                                        # This is the problematic domain - sanitize it aggressively
                                        sanitized = _sanitize_url_aggressively(domain, max_path_segments=3)
                                        sanitized_domain_filter.append(sanitized)
                                        found_and_sanitized = True
                                        logger.info("Sanitized domain aggressively: %s -> %s", domain, sanitized)
                                    else:
                                        sanitized_domain_filter.append(domain)
                                
                                if not found_and_sanitized:
                                    # Try sanitizing the problematic domain directly aggressively
                                    sanitized_problematic = _sanitize_url_aggressively(problematic_domain, max_path_segments=3)
                                    if sanitized_problematic not in sanitized_domain_filter:
                                        sanitized_domain_filter.append(sanitized_problematic)
                                
                                current_domain_filter = sanitized_domain_filter
                                kwargs["search_domain_filter"] = current_domain_filter
                                sanitization_attempt += 1
                                continue
                            else:
                                # Couldn't extract problematic domain from error message
                                # Try one more time with a more aggressive extraction attempt
                                logger.warning("Could not extract problematic domain. Error: %s", error_msg)
                                logger.warning("Attempting to find domain-like strings in error message...")
                                
                                # Try to find any domain-like string in the error and match it against our filter list
                                import re
                                # Look for any URL or domain pattern in the error
                                potential_domains = re.findall(r'(https?://[^\s,)]+|[a-zA-Z0-9][a-zA-Z0-9-]*\.[a-zA-Z]{2,}[^\s,)]*)', error_msg)
                                
                                found_match = False
                                if potential_domains:
                                    sanitized_domain_filter = []
                                    for domain in current_domain_filter:
                                        domain_without_prefix = domain.lstrip('-')
                                        # Check if any potential domain from error matches this filter entry
                                        matched = False
                                        for potential in potential_domains:
                                            potential_clean = potential.strip('"\'')
                                            # Check if potential domain matches or is contained in filter entry
                                            if (potential_clean == domain_without_prefix or 
                                                domain_without_prefix.startswith(potential_clean) or
                                                potential_clean.startswith(domain_without_prefix.split('/')[0])):
                                                # This might be the problematic domain - sanitize it
                                                if _is_pubmed_pmc_domain(domain):
                                                    sanitized_domain_filter.append(domain)
                                                else:
                                                    sanitized = _sanitize_url_aggressively(domain, max_path_segments=3)
                                                    sanitized_domain_filter.append(sanitized)
                                                    logger.info("Sanitized potential problematic domain: %s -> %s", domain, sanitized)
                                                matched = True
                                                found_match = True
                                                break
                                        
                                        if not matched:
                                            # Keep unchanged if no match
                                            sanitized_domain_filter.append(domain)
                                    
                                    if found_match:
                                        current_domain_filter = sanitized_domain_filter
                                        kwargs["search_domain_filter"] = current_domain_filter
                                        logger.info("Retrying with sanitized domain filter based on potential matches: %s", current_domain_filter)
                                        sanitization_attempt += 1
                                        continue
                                
                                # If we still can't identify the problematic domain, raise an error
                                # rather than blanket sanitizing all domains
                                logger.error(
                                    "Could not extract problematic domain from error message after multiple attempts. "
                                    "Error: %s. Domain filter: %s",
                                    error_msg, current_domain_filter
                                )
                                raise ValueError(
                                    f"Could not identify problematic domain in error message: {error_msg}. "
                                    "Please check the domain filter list for invalid entries."
                                )
                        else:
                            # Max sanitization attempts reached - try one final fallback
                            problematic_domain = _extract_problematic_domain_from_error(error_msg)
                            if problematic_domain:
                                logger.warning(
                                    "Max sanitization attempts reached. Attempting final fallback: removing problematic domain: %s",
                                    problematic_domain
                                )
                                # Try to remove the problematic domain as a last resort
                                normalized_problematic = _normalize_url_for_comparison(problematic_domain)
                                filtered_list = []
                                for d in current_domain_filter:
                                    normalized_d = _normalize_url_for_comparison(d)
                                    if normalized_d != normalized_problematic:
                                        filtered_list.append(d)
                                    else:
                                        logger.info("Removed problematic domain in final fallback: %s", d)
                                
                                if len(filtered_list) < len(current_domain_filter):
                                    # Successfully removed at least one domain
                                    current_domain_filter = filtered_list
                                    kwargs["search_domain_filter"] = current_domain_filter
                                    logger.warning(
                                        "Final fallback: Removed problematic domain. Retrying with %d entries...",
                                        len(current_domain_filter)
                                    )
                                    # Make one more attempt with the cleaned filter
                                    try:
                                        response = await _call_perplexity_with_retry(
                                            client=self.client,
                                            kwargs=kwargs,
                                            max_retries=self.max_retries,
                                        )
                                        break  # Success
                                    except Exception as final_error:
                                        logger.error(
                                            "Final fallback also failed. Error: %s",
                                            str(final_error)
                                        )
                                        raise ValueError(
                                            f"Failed to sanitize domain filter after {max_domain_sanitization_attempts} attempts and final fallback. "
                                            f"Last error: {str(final_error)}"
                                        )
                                else:
                                    logger.error("Final fallback: Could not find problematic domain to remove")
                                    raise ValueError(
                                        f"Failed to sanitize domain filter after {max_domain_sanitization_attempts} attempts. "
                                        f"Problematic domain: {problematic_domain}"
                                    )
                            else:
                                # Couldn't extract problematic domain
                                logger.error("Failed to sanitize domain filter after %d attempts", max_domain_sanitization_attempts)
                                raise ValueError(
                                    f"Failed to sanitize domain filter after {max_domain_sanitization_attempts} attempts. "
                                    f"Could not identify problematic domain from error: {error_msg}"
                                )
                    else:
                        # Not a domain filter error, re-raise
                        raise
                except Exception as e:
                    logger.error("Error in iterative filtering iteration %d: %s", iteration, str(e))
                    # Save filter list before returning/raising, even on error
                    if log_dir:
                        try:
                            from pathlib import Path
                            log_path = Path(log_dir)
                            log_path.mkdir(parents=True, exist_ok=True)
                            filter_file = log_path / "domain_filter_list.json"
                            filter_info = {
                                "domain_filter": current_domain_filter,
                                "total_entries": len(current_domain_filter),
                                "max_entries": MAX_DOMAIN_FILTER_ENTRIES,
                                "iterations": iteration,
                                "cochrane_titles_count": len(cochrane_titles) if cochrane_titles else 0,
                                "manually_provided_count": len(manually_provided_urls),
                                "default_cochrane_count": len(default_cochrane_urls),
                                "iterative_detected_count": len(iterative_url_frequency),
                                "iterative_url_frequencies": dict(sorted(iterative_url_frequency.items(), key=lambda x: x[1], reverse=True)),
                                "error": str(e),
                                "error_iteration": iteration,
                            }
                            with open(filter_file, 'w', encoding='utf-8') as f:
                                json.dump(filter_info, f, indent=2, ensure_ascii=False)
                            logger.info("Saved domain filter list (%d/%d entries) after error to %s", 
                                       len(current_domain_filter), MAX_DOMAIN_FILTER_ENTRIES, filter_file)
                        except Exception as save_error:
                            logger.error("Failed to save domain filter list after error: %s", str(save_error))
                    # If we have a previous response, return it
                    if final_response:
                        return final_response, final_text_content, [], None
                    raise
            
            # Extract text content
            text_content = None
            if response and hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    text_content = choice.message.content
            
            # Extract search results
            search_results = _extract_search_results_list(response)
            
            if not search_results:
                logger.info("No search results found in iteration %d. Stopping iterative filtering.", iteration)
                final_response = response
                final_text_content = text_content
                break
            
            # Filter search results using full CochraneResultFilter mechanism
            matching_urls = []
            if cochrane_filter:
                logger.info("Checking %d search results using full CochraneResultFilter (titles: %d, date cutoff: %s)", 
                           len(search_results), 
                           len(cochrane_filter.title_filter_list) if cochrane_filter.title_filter_list else 0,
                           cochrane_filter.publication_date_cutoff.strftime("%d %B %Y") if cochrane_filter.publication_date_cutoff else "none")
                for result in search_results:
                    title = result.get('title', '')
                    url = result.get('url', '')
                    result_date = result.get('date', None)
                    
                    if title and url:
                        # FIRST: Check if this URL is already in the domain filter (client-side filtering)
                        if current_domain_filter and _url_matches_domain_filter(url, current_domain_filter):
                            is_pubmed_pmc = _is_pubmed_pmc_domain(url)
                            if is_pubmed_pmc:
                                logger.debug("Skipping PMC article already in domain filter (client-side): %s -> %s", title[:80], url)
                            else:
                                logger.debug("Skipping result already in domain filter (client-side): %s -> %s", title[:80], url)
                            continue
                        # Use full CochraneResultFilter._should_filter_item to check all criteria:
                        # 1. URL is a Cochrane URL
                        # 2. Title contains "Cochrane" (case-insensitive)
                        # 3. Title matches any Cochrane title from list
                        # 4. Publication date is after cutoff
                        should_filter, reason = cochrane_filter._should_filter_item(
                            title=title,
                            urls=[url],
                            publication_date=result_date
                        )
                        
                        if should_filter:
                            # This result should be filtered - extract domain and add to denylist
                            # Extract domain from URL (or keep full URL for PubMed/PMC)
                            domain_entry = _extract_domain_from_url(url)
                            
                            # For PubMed/PMC URLs, keep the full URL (Perplexity accepts specific URLs)
                            # These will be sent to Perplexity API as part of the domain filter
                            is_pubmed_pmc = _is_pubmed_pmc_domain(domain_entry.lstrip('-'))
                            
                            # Ensure it has "-" prefix for denylist
                            if not domain_entry.startswith('-'):
                                domain_entry = f"-{domain_entry}"
                            
                            # Skip if this is a default Cochrane domain (already filtered)
                            if _is_default_cochrane_domain(domain_entry):
                                logger.debug("Skipping default Cochrane domain (already in filter): %s", domain_entry)
                                continue
                            
                            # Add to matching_urls for tracking (including PubMed/PMC URLs which are sent to API)
                            matching_urls.append(domain_entry)
                            if is_pubmed_pmc:
                                logger.info("Found matching Cochrane result on PubMed/PMC (reason: %s): %s -> %s (will be added to domain filter and sent to Perplexity API, date: %s)", 
                                          reason, title, url, result_date or "unknown")
                            else:
                                logger.info("Found matching Cochrane result (reason: %s): %s -> %s (extracted domain: %s, date: %s)", 
                                          reason, title, url, domain_entry, result_date or "unknown")
                        else:
                            # Log non-matching results for debugging
                            logger.debug("Non-matching result (not filtered): %s", title)
            else:
                logger.warning("No Cochrane filter available for iterative filtering")
            
            # Stop conditions:
            # 1) No search results "need filtering" (i.e., no Cochrane-title matches found)
            # 2) We cannot expand the denylist further (i.e., all matches are already covered by current_domain_filter)
            if not matching_urls:
                logger.info(
                    "Iteration %d: No Cochrane-related titles found in search results. Stopping iterative filtering.",
                    iteration,
                )
                final_response = response
                final_text_content = text_content
                break
            
            # Only proceed if we discovered NEW denylist entries this iteration.
            #
            # IMPORTANT: If Perplexity returns results that match titles we are filtering AND those URLs are already
            # present in our denylist, then (per Perplexity docs) those matches should NOT have appeared in the returned
            # search_results at all. This indicates server-side filtering leakage or URL normalization mismatch.
            # Iterating further cannot fix this (denylist won't grow), so we stop and log loudly.
            new_matching_urls = [u for u in matching_urls if u not in current_domain_filter]
            if not new_matching_urls:
                logger.warning(
                    "Iteration %d: Perplexity returned %d Cochrane-title matches even though ALL are already in the "
                    "denylist. This suggests search_domain_filter leakage or URL normalization mismatch. "
                    "Stopping iterative filtering. matching_urls=%s",
                    iteration,
                    len(matching_urls),
                    matching_urls,
                )
                final_response = response
                final_text_content = text_content
                break
            
            # Update frequency tracking for iteratively detected URLs (track only newly discovered)
            for url in new_matching_urls:
                iterative_url_frequency[url] = iterative_url_frequency.get(url, 0) + 1
            
            # Build prioritized filter list:
            # 1. Keep all manually provided URLs
            # 2. Keep all default Cochrane URLs
            # 3. Add most frequent iteratively detected URLs until we hit the cap
            
            # Start with manually provided and default URLs (these are always kept)
            prioritized_filter = []
            prioritized_filter_set = set()
            
            # Add manually provided URLs first
            for url in current_domain_filter:
                if url in manually_provided_urls:
                    prioritized_filter.append(url)
                    prioritized_filter_set.add(url)
            
            # Add default Cochrane URLs (if not already added)
            for url in current_domain_filter:
                if url in default_cochrane_urls and url not in prioritized_filter_set:
                    prioritized_filter.append(url)
                    prioritized_filter_set.add(url)
            
            # Sort iteratively detected URLs by frequency (most frequent first)
            # Only consider URLs that were detected in this iteration or previous iterations
            sorted_iterative_urls = sorted(
                iterative_url_frequency.items(),
                key=lambda x: x[1],  # Sort by frequency
                reverse=True
            )
            
            # Add most frequent iteratively detected URLs until we hit the cap
            for url, frequency in sorted_iterative_urls:
                if len(prioritized_filter) >= MAX_DOMAIN_FILTER_ENTRIES:
                    break
                if url not in prioritized_filter_set:
                    # Ensure URL has "-" prefix (should already have it, but verify)
                    url_with_prefix = url if url.startswith('-') else f"-{url}"
                    prioritized_filter.append(url_with_prefix)
                    prioritized_filter_set.add(url_with_prefix)
            
            # Deduplicate and sanitize the prioritized filter
            current_domain_filter = _deduplicate_and_sanitize_domains(prioritized_filter)
            
            # After deduplication, ensure default Cochrane domains are still present
            # (deduplication should preserve them, but verify to be safe)
            has_cochrane_org, has_cochranelibrary = _has_any_cochrane_domain(current_domain_filter)
            default_cochrane_domains = [
                "-cochranelibrary.com/",
                "-cochrane.org/"
            ]
            if not has_cochranelibrary:
                current_domain_filter.append(default_cochrane_domains[0])
                default_cochrane_urls.add(default_cochrane_domains[0])
                logger.debug("Re-added default Cochrane domain after iteration deduplication: %s", default_cochrane_domains[0])
            if not has_cochrane_org:
                current_domain_filter.append(default_cochrane_domains[1])
                default_cochrane_urls.add(default_cochrane_domains[1])
                logger.debug("Re-added default Cochrane domain after iteration deduplication: %s", default_cochrane_domains[1])
            
            # Update sets after deduplication
            manually_provided_urls = {url for url in manually_provided_urls if url in current_domain_filter}
            default_cochrane_urls = {url for url in default_cochrane_urls if url in current_domain_filter}
            
            logger.info(
                "Added %d NEW denylist entries this iteration (tracking %d unique total). Domain filter now has %d/%d entries",
                len(new_matching_urls),
                len(iterative_url_frequency),
                len(current_domain_filter),
                MAX_DOMAIN_FILTER_ENTRIES,
            )
            
            # Check if we've reached the cap after adding new URLs
            if len(current_domain_filter) >= MAX_DOMAIN_FILTER_ENTRIES:
                logger.info("Domain filter has reached maximum of %d entries. Stopping iterative filtering.", MAX_DOMAIN_FILTER_ENTRIES)
                final_response = response
                final_text_content = text_content
                break
            
            # Store response for potential return if next iteration fails
            final_response = response
            final_text_content = text_content
        
        if iteration >= max_iterations:
            logger.warning("Reached maximum iterations (%d) for iterative filtering", max_iterations)
        
        # Save final filter list to file (always save, even if there was an error)
        # This ensures both sonar-reasoning-pro and sonar-deep-research track the filtering list
        if log_dir:
            try:
                from pathlib import Path
                log_path = Path(log_dir)
                log_path.mkdir(parents=True, exist_ok=True)
                
                # Save as JSON file with frequency information
                filter_file = log_path / "domain_filter_list.json"
                
                # Build detailed filter info
                filter_info = {
                    "domain_filter": current_domain_filter,
                    "total_entries": len(current_domain_filter),
                    "max_entries": MAX_DOMAIN_FILTER_ENTRIES,
                    "iterations": iteration,
                    "cochrane_titles_count": len(cochrane_titles) if cochrane_titles else 0,
                    "manually_provided_count": len(manually_provided_urls),
                    "default_cochrane_count": len(default_cochrane_urls),
                    "iterative_detected_count": len(iterative_url_frequency),
                    "iterative_url_frequencies": dict(sorted(iterative_url_frequency.items(), key=lambda x: x[1], reverse=True)),
                }
                
                with open(filter_file, 'w', encoding='utf-8') as f:
                    json.dump(filter_info, f, indent=2, ensure_ascii=False)
                
                logger.info("Saved final domain filter list (%d/%d entries) to %s", 
                           len(current_domain_filter), MAX_DOMAIN_FILTER_ENTRIES, filter_file)
                print(f"Saved final domain filter list ({len(current_domain_filter)}/{MAX_DOMAIN_FILTER_ENTRIES} entries) to {filter_file}")
            except Exception as e:
                logger.error("Failed to save domain filter list to %s: %s", log_dir, str(e))
        
        return final_response, final_text_content, [], None
    
    async def _call_async_deep_research(
        self,
        api_messages: List[Dict[str, Any]],
        domain_filter: Optional[List[str]] = None,
        search_before_date_filter: Optional[str] = None,
        enable_filtering: bool = True,
    ) -> Tuple[Any, Optional[str], List[Dict[str, Any]], Optional[str]]:
        """
        Call Perplexity async/deep research API with polling.
        
        Submits an async request and polls until completion.
        
        Args:
            api_messages: Messages in API format
            domain_filter: List of domains/URLs to filter for this call
            search_before_date_filter: Date filter in MM/DD/YYYY format
        
        Returns:
            Tuple of (response, text_content, tool_calls, reasoning_summary)
        """
        loop = asyncio.get_event_loop()
        
        # sonar-deep-research and sonar-reasoning-pro require stream=True
        stream_value = self.model in ["sonar-deep-research", "sonar-reasoning-pro"]
        
        # Build async request kwargs
        async_kwargs = {
            "messages": api_messages,
            "model": self.model,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
        }
        
        # Only add stream if needed (sonar-deep-research or sonar-reasoning-pro)
        if stream_value:
            async_kwargs["stream"] = True
        
        # Build web search options for sonar-deep-research and sonar-reasoning-pro
        web_search_options = {}
        if search_before_date_filter:
            web_search_options["search_before_date_filter"] = search_before_date_filter
        
        # Add search_type, search_mode, and web_search_context_size for sonar-deep-research and sonar-reasoning-pro
        if self.model in ["sonar-deep-research", "sonar-reasoning-pro"]:
            web_search_options["search_type"] = self.search_type
            web_search_options["search_mode"] = self.search_mode
            web_search_options["web_search_context_size"] = self.web_search_context_size
        
        # Add web_search_options as top-level parameter only if it has content
        if web_search_options:
            async_kwargs["web_search_options"] = web_search_options
        
        # Add domain filter if provided (per-call parameter)
        # Use search_domain_filter (not domain_filter) - Perplexity uses search_domain_filter parameter name
        if domain_filter:
            async_kwargs["search_domain_filter"] = domain_filter
            logger.info(
                "Domain filter applied for %s (async mode): %d entries",
                self.model, len(domain_filter)
            )
        else:
            # Only warn if filtering is enabled but no domain filter was provided
            # If filtering is disabled (enable_filtering=False), this is intentional and not a warning
            if self.model in ["sonar-deep-research", "sonar-reasoning-pro"] and enable_filtering:
                logger.warning(
                    "No domain filter provided for %s (async mode). "
                    "Consider providing domain_filter to prevent leakage of ground-truth answers.",
                    self.model
                )
        
        # Submit async request
        try:
            async_request = await loop.run_in_executor(
                None,
                lambda: self.client.async_.chat.completions.create(**async_kwargs)
            )
            
            request_id = async_request.request_id
            logger.info(
                "Async request submitted with ID: %s, Status: %s",
                request_id, async_request.status
            )
        except Exception as e:
            logger.error("Failed to submit async request: %s: %s", type(e).__name__, str(e))
            raise
        
        # Poll for completion
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed > self.async_max_wait:
                raise TimeoutError(
                    f"Async request {request_id} did not complete within {self.async_max_wait} seconds"
                )
            
            try:
                status = await loop.run_in_executor(
                    None,
                    lambda: self.client.async_.chat.completions.get(request_id)
                )
                
                logger.debug("Request %s status: %s", request_id, status.status)
                
                if status.status == "completed":
                    # Extract response
                    if hasattr(status, 'result') and status.result:
                        response = status.result
                        text_content = None
                        if response.choices and len(response.choices) > 0:
                            message = response.choices[0].message
                            if hasattr(message, 'content') and message.content:
                                text_content = message.content
                        
                        logger.info("Async request %s completed successfully", request_id)
                        return response, text_content, [], None
                    else:
                        raise ValueError(f"Completed request {request_id} has no result")
                
                elif status.status == "failed":
                    error_msg = getattr(status, 'error', 'Unknown error')
                    raise RuntimeError(f"Async request {request_id} failed: {error_msg}")
                
                # Status is "pending" or "processing" - continue polling
                await asyncio.sleep(self.async_poll_interval)
                
            except Exception as e:
                if isinstance(e, (RuntimeError, ValueError)):
                    raise
                logger.warning(
                    "Error checking async request status: %s: %s. Retrying...",
                    type(e).__name__, str(e)
                )
                await asyncio.sleep(self.async_poll_interval)
    
    def format_tool_response_message(
        self, tool_call_id: str, tool_name: str, tool_result: str
    ) -> Dict[str, Any]:
        """
        Format a tool response message for Perplexity.
        
        Note: Perplexity doesn't use external tools, so this is mainly for interface compatibility.
        """
        logger.warning(
            "format_tool_response_message called for Perplexity provider, "
            "but Perplexity uses built-in search and doesn't require external tools"
        )
        # Return a message format that won't break the interface
        return {
            "role": "user",
            "content": f"Tool result from {tool_name}: {tool_result}"
        }

