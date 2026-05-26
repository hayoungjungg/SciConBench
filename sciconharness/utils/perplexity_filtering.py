"""
Perplexity domain filtering utilities. 

This module provides utilities for building and sanitizing domain filters
for Perplexity models (sonar-pro and sonar-deep-research).

Key functions:
- build_perplexity_domain_filter: Build domain filter from CLI args and JSON files
- process_doi_domain_filters: Process DOI-to-domain-filter dictionaries for batch processing
- clean_domain: Clean domains/URLs by removing protocols and paths
- sanitize_domain_filter: Sanitize domain filter lists
- filter_search_results_by_domain: Filter search results based on domain filter list
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _is_pubmed_pmc_domain(domain: str) -> bool:
    """
    Check if a domain is a PubMed or PMC domain that should never have paths removed.
    
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


def clean_domain(domain_or_url: str, remove_path: bool = True) -> str:
    """
    Clean a domain or URL by removing http://, https://, www. prefixes and optionally paths.
    
    IMPORTANT: PubMed/PMC domains are NEVER sanitized - they must always be filtered 
    at the article level (e.g., https://pmc.ncbi.nlm.nih.gov/articles/PMC12324103/).
    
    Args:
        domain_or_url: A domain or URL (e.g., "https://www.example.com/path" or "-http://example.com")
        remove_path: If True, remove path components. If False, keep them (for progressive sanitization).
    
    Returns:
        Cleaned domain name (e.g., "example.com" or "-example.com" if it had a "-" prefix).
        Returns the domain unchanged if it's a PubMed/PMC domain.
    """
    # Preserve denylist prefix
    is_denylist = domain_or_url.startswith('-')
    cleaned = domain_or_url.lstrip('-')
    
    # Never remove paths from PubMed/PMC domains - they must always be at article level
    if _is_pubmed_pmc_domain(cleaned):
        # Only remove protocol and www, but keep all paths
        if cleaned.startswith('http://'):
            cleaned = cleaned[7:]
        elif cleaned.startswith('https://'):
            cleaned = cleaned[8:]
        if cleaned.startswith('www.'):
            cleaned = cleaned[4:]
        # Keep everything else (paths, query params, etc.) for PubMed/PMC
        return f"-{cleaned}" if is_denylist else cleaned
    
    # For non-PubMed/PMC domains, proceed with normal cleaning
    # Remove protocol prefixes
    if cleaned.startswith('http://'):
        cleaned = cleaned[7:]
    elif cleaned.startswith('https://'):
        cleaned = cleaned[8:]
    
    # Remove www. prefix
    if cleaned.startswith('www.'):
        cleaned = cleaned[4:]
    
    # Remove trailing slashes and paths if requested
    if remove_path:
        cleaned = cleaned.split('/')[0]
        cleaned = cleaned.split('?')[0]  # Remove query parameters
        cleaned = cleaned.split('#')[0]   # Remove fragments
    
    # Remove trailing dots
    cleaned = cleaned.rstrip('.')
    
    # Re-add denylist prefix if it was present
    return f"-{cleaned}" if is_denylist else cleaned


def _minimal_clean(url: str) -> str:
    """
    Minimal cleaning: only remove protocol and www, keep paths.
    
    This is used when building domain filters to preserve URL-level filtering
    while still normalizing basic elements.
    
    Args:
        url: URL string (may have "-" prefix)
    
    Returns:
        Cleaned URL with protocol/www removed but paths preserved
    """
    # Preserve denylist prefix
    is_denylist = url.startswith('-')
    cleaned = url.lstrip('-')
    
    # Remove protocol
    if cleaned.startswith('http://'):
        cleaned = cleaned[7:]
    elif cleaned.startswith('https://'):
        cleaned = cleaned[8:]
    
    # Remove www.
    if cleaned.startswith('www.'):
        cleaned = cleaned[4:]
    
    # Re-add denylist prefix if it was present
    return f"-{cleaned}" if is_denylist else cleaned


def _ensure_denylist_prefix(domain_entry: str) -> str:
    """
    Ensure a domain/URL entry has the "-" prefix for denylist mode.
    
    All domain filter entries should use denylist mode (exclusion) with "-" prefix.
    
    Args:
        domain_entry: Domain or URL entry (may or may not have "-" prefix)
    
    Returns:
        Domain entry with "-" prefix (if not already present)
    """
    if not domain_entry.startswith('-'):
        return f"-{domain_entry}"
    return domain_entry


def build_perplexity_domain_filter(
    domain_filter: Optional[List[str]] = None,
    filtered_links_path: Optional[str] = None,
) -> Optional[List[str]]:
    """
    Build domain filter list for Perplexity from command-line arguments and JSON file.
    
    This function:
    1. Merges user-provided domain filters
    2. Loads filtered links from JSON file (if provided) and adds them as denylist entries
    3. Automatically adds Cochrane domains as denylist entries when domain filtering is enabled
    4. Cleans all domains by removing http://, https://, www. prefixes
    
    Args:
        domain_filter: Optional list of domains/URLs from --domain-filter argument.
                       Use '-' prefix for denylist (e.g., '-example.com').
        filtered_links_path: Optional path to JSON file with 'filtered_links' array.
                            Links will be added as denylist entries (with '-' prefix).
    
    Returns:
        List of cleaned domain filter entries (with '-' prefix for denylist), or None if no filters.
        Always includes Cochrane domains when any filtering is enabled.
    
    Example:
        >>> filter_list = build_perplexity_domain_filter(
        ...     domain_filter=["-example.com", "pubmed.ncbi.nlm.nih.gov"],
        ...     filtered_links_path="path/to/filtered_links.json"
        ... )
    """
    domain_filter_list = []
    seen_domains = set()  # Track unique domains to avoid duplicates
    
    # Add user-provided domain filter
    if domain_filter:
        for item in domain_filter:
            # Only remove protocol/www, keep paths - don't clean to domain name yet
            cleaned = _minimal_clean(item)
            # Ensure it has "-" prefix for denylist mode
            cleaned = _ensure_denylist_prefix(cleaned)
            if cleaned not in seen_domains:
                domain_filter_list.append(cleaned)
                seen_domains.add(cleaned)
    
    # Load filtered links from JSON file if provided
    if filtered_links_path:
        try:
            filtered_links_file = Path(filtered_links_path)
            if not filtered_links_file.exists():
                raise FileNotFoundError(f"Filtered links file not found: {filtered_links_file}")
            
            with open(filtered_links_file, 'r') as f:
                filtered_data = json.load(f)
            
            # Extract filtered_links array
            if 'filtered_links' in filtered_data and isinstance(filtered_data['filtered_links'], list):
                # Add all links as denylist entries (with "-" prefix)
                for link in filtered_data['filtered_links']:
                    # Only remove protocol/www, keep all paths - don't clean to domain name yet
                    cleaned = _minimal_clean(link)
                    # Ensure it has "-" prefix for denylist mode
                    cleaned = _ensure_denylist_prefix(cleaned)
                    
                    if cleaned not in seen_domains:
                        domain_filter_list.append(cleaned)
                        seen_domains.add(cleaned)
            else:
                raise ValueError(
                    f"Invalid JSON structure: 'filtered_links' array not found in {filtered_links_file}"
                )
        except Exception as e:
            print(f"Warning: Failed to load filtered links from {filtered_links_path}: {e}")
    
    # Always add Cochrane domains as denylist entries when domain filter is enabled
    if domain_filter_list:
        cochrane_domains = [
            "-cochranelibrary.com/",
            "-cochrane.org/"
        ]
        for domain in cochrane_domains:
            if domain not in seen_domains:
                domain_filter_list.append(domain)
                seen_domains.add(domain)
    
    return domain_filter_list if domain_filter_list else None


def merge_domain_filter_with_filtered_links(
    domain_filter: Optional[List[str]],
    filtered_links_path: Optional[Path] = None,
) -> Optional[List[str]]:
    """
    Merge a domain filter list with filtered_links from a JSON file.
    
    This is useful when you have a base domain filter and want to add
    filtered_links from a previously saved filtered_links.json file.
    
    Args:
        domain_filter: Optional base domain filter list
        filtered_links_path: Optional path to filtered_links.json file
    
    Returns:
        Merged domain filter list, or None if both inputs are None/empty
    """
    if not filtered_links_path or not filtered_links_path.exists():
        return domain_filter
    
    try:
        with open(filtered_links_path, 'r') as f:
            filtered_data = json.load(f)
        
        if 'filtered_links' not in filtered_data or not isinstance(filtered_data['filtered_links'], list):
            return domain_filter
        
        # Start with existing domain filter or empty list
        # Ensure all existing entries have "-" prefix
        merged_filter = []
        if domain_filter:
            for entry in domain_filter:
                entry_with_prefix = _ensure_denylist_prefix(entry)
                merged_filter.append(entry_with_prefix)
        seen_domains = set(merged_filter)
        
        # Add filtered_links as denylist entries
        for link in filtered_data['filtered_links']:
            cleaned = _minimal_clean(link)
            # Ensure it has "-" prefix for denylist mode
            cleaned = _ensure_denylist_prefix(cleaned)
            
            if cleaned not in seen_domains:
                merged_filter.append(cleaned)
                seen_domains.add(cleaned)
        
        # Add Cochrane domains if not already present
        cochrane_domains = [
            _minimal_clean("https://www.cochranelibrary.com/"),
            _minimal_clean("https://www.cochrane.org/")
        ]
        cochrane_domains = [f"-{d}" if not d.startswith('-') else d for d in cochrane_domains]
        
        for domain in cochrane_domains:
            if domain not in seen_domains:
                merged_filter.append(domain)
                seen_domains.add(domain)
        
        return merged_filter if merged_filter else None
    except Exception as e:
        logger.warning("Failed to merge filtered links: %s", e)
        return domain_filter


def process_doi_domain_filters(
    doi_to_domain_filter: Dict[str, List[str]],
    filtered_links_base_dir: Optional[Path] = None,
) -> Dict[str, List[str]]:
    """
    Process domain filters for multiple DOIs, optionally loading filtered_links from DOI-specific files.
    
    This function processes a dictionary mapping DOIs to domain filter lists, and optionally
    merges in filtered_links from DOI-specific JSON files.
    
    Args:
        doi_to_domain_filter: Dictionary mapping DOI strings to domain filter lists
        filtered_links_base_dir: Optional base directory where filtered_links.json files are stored
                                (expected structure: {base_dir}/{doi}/filtered_links.json)
    
    Returns:
        Dictionary mapping DOIs to processed domain filter lists
    
    Example:
        >>> filters = process_doi_domain_filters(
        ...     {"10.1002/doi1": ["-example.com"], "10.1002/doi2": ["-test.com"]},
        ...     filtered_links_base_dir=Path("data/filtered")
        ... )
    """
    processed_filters = {}
    seen_domains_per_doi = {}  # Track unique domains per DOI
    
    for doi, domain_list in doi_to_domain_filter.items():
        if not domain_list:
            # Empty list - still add Cochrane domains
            domain_filter_list = []
            seen_domains = set()
        else:
            domain_filter_list = []
            seen_domains = set()
            
            # Clean and add user-provided domain filters
            for item in domain_list:
                cleaned = _minimal_clean(item)
                # Ensure it has "-" prefix for denylist mode
                cleaned = _ensure_denylist_prefix(cleaned)
                if cleaned not in seen_domains:
                    domain_filter_list.append(cleaned)
                    seen_domains.add(cleaned)
        
        # Optionally load filtered_links from DOI-specific JSON file
        if filtered_links_base_dir:
            filtered_links_file = filtered_links_base_dir / doi / "filtered_links.json"
            if filtered_links_file.exists():
                try:
                    with open(filtered_links_file, 'r') as f:
                        filtered_data = json.load(f)
                    
                    if 'filtered_links' in filtered_data and isinstance(filtered_data['filtered_links'], list):
                        for link in filtered_data['filtered_links']:
                            cleaned = _minimal_clean(link)
                            # Ensure it has "-" prefix for denylist mode
                            cleaned = _ensure_denylist_prefix(cleaned)
                            
                            if cleaned not in seen_domains:
                                domain_filter_list.append(cleaned)
                                seen_domains.add(cleaned)
                except Exception as e:
                    print(f"Warning: Failed to load filtered links from {filtered_links_file}: {e}")
        
        # Always add Cochrane domains as denylist entries
        cochrane_domains = [
            _minimal_clean("https://www.cochranelibrary.com/"),
            _minimal_clean("https://www.cochrane.org/")
        ]
        cochrane_domains = [f"-{d}" if not d.startswith('-') else d for d in cochrane_domains]
        for domain in cochrane_domains:
            if domain not in seen_domains:
                domain_filter_list.append(domain)
                seen_domains.add(domain)
        
        processed_filters[doi] = domain_filter_list if domain_filter_list else []
        seen_domains_per_doi[doi] = seen_domains
    
    return processed_filters


def sanitize_domain_filter(domain_filter: List[str]) -> List[str]:
    """
    Sanitize a domain filter list by cleaning all entries.
    
    Args:
        domain_filter: List of domain/URL entries (may have "-" prefix)
    
    Returns:
        Sanitized list with cleaned domains
    """
    sanitized = []
    seen = set()
    
    for entry in domain_filter:
        cleaned = _minimal_clean(entry)
        if cleaned not in seen:
            sanitized.append(cleaned)
            seen.add(cleaned)
    
    return sanitized


def _normalize_url_for_matching(url: str) -> str:
    """
    Normalize a URL for matching against domain filter entries.
    
    Removes protocol, www, trailing slashes, and query parameters for comparison.
    Preserves paths for URL-level matching.
    
    Args:
        url: URL string to normalize
    
    Returns:
        Normalized URL string
    """
    normalized = url.lower().strip()
    
    # Remove protocol
    if normalized.startswith('http://'):
        normalized = normalized[7:]
    elif normalized.startswith('https://'):
        normalized = normalized[8:]
    
    # Remove www.
    if normalized.startswith('www.'):
        normalized = normalized[4:]
    
    # Remove query parameters and fragments
    normalized = normalized.split('?')[0]
    normalized = normalized.split('#')[0]
    
    # Remove trailing slash (but keep path structure)
    normalized = normalized.rstrip('/')
    
    return normalized


def _url_matches_filter_entry(url: str, filter_entry: str) -> bool:
    """
    Check if a URL matches a domain filter entry.
    
    Handles:
    - Exact URL matches (for URL-level filtering)
    - Domain matches (for domain-level filtering)
    - Path prefix matches (for directory-level filtering)
    
    Args:
        url: URL to check
        filter_entry: Filter entry (may have "-" prefix, may be domain or full URL)
    
    Returns:
        True if URL matches the filter entry, False otherwise
    """
    # Remove denylist prefix for comparison
    filter_pattern = filter_entry.lstrip('-')
    
    # Normalize both for comparison
    url_normalized = _normalize_url_for_matching(url)
    filter_normalized = _normalize_url_for_matching(filter_pattern)
    
    # Exact match
    if url_normalized == filter_normalized:
        return True
    
    # For PubMed/PMC domains, require exact URL match (article-level filtering)
    if _is_pubmed_pmc_domain(filter_pattern):
        # PubMed/PMC entries should be full URLs, check exact match
        return url_normalized == filter_normalized
    
    # Domain-level match (filter entry is just domain, URL starts with that domain)
    # Check if URL starts with the filter pattern
    if url_normalized.startswith(filter_normalized):
        # Ensure it's a proper domain/path match (not partial string match)
        # Check if filter is a domain (no path) or if URL path starts with filter path
        if '/' not in filter_normalized:
            # Domain-level filter: URL must start with domain followed by / or end of string
            if url_normalized == filter_normalized or url_normalized.startswith(filter_normalized + '/'):
                return True
        else:
            # Path-level filter: URL path must start with filter path
            return True
    
    return False


def filter_search_results_by_domain(
    search_results: List[Dict[str, Any]],
    domain_filter: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Filter search results to remove URLs that match domain filter entries.
    
    This function removes search results whose URLs match any entry in the domain filter
    list (entries with "-" prefix are exclusion filters).
    
    Args:
        search_results: List of search result dictionaries with 'url' key
        domain_filter: Optional list of domain/URL filter entries (with "-" prefix for exclusion)
    
    Returns:
        Filtered list of search results with matching URLs removed
    
    Example:
        >>> results = [
        ...     {"title": "Test", "url": "https://example.com/page"},
        ...     {"title": "Test2", "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC123/"}
        ... ]
        >>> filter_list = ["-example.com", "-https://pmc.ncbi.nlm.nih.gov/articles/PMC123/"]
        >>> filtered = filter_search_results_by_domain(results, filter_list)
        >>> # Returns empty list (both URLs match filters)
    """
    if not domain_filter or not search_results:
        return search_results
    
    # Separate denylist entries (with "-" prefix)
    denylist_entries = [entry for entry in domain_filter if entry.startswith('-')]
    
    if not denylist_entries:
        # No exclusion filters, return all results
        return search_results
    
    filtered_results = []
    filtered_count = 0
    
    for result in search_results:
        url = result.get('url', '')
        if not url:
            # Keep results without URLs
            filtered_results.append(result)
            continue
        
        # Check if URL matches any denylist entry
        should_exclude = False
        matching_filter = None
        
        for filter_entry in denylist_entries:
            if _url_matches_filter_entry(url, filter_entry):
                should_exclude = True
                matching_filter = filter_entry
                break
        
        if should_exclude:
            filtered_count += 1
            logger.info(
                "CLIENT-SIDE FILTER: Excluding search result matching filter entry '%s': %s",
                matching_filter, url
            )
        else:
            filtered_results.append(result)
    
    if filtered_count > 0:
        logger.info(
            "CLIENT-SIDE FILTER: Filtered out %d/%d search results based on domain filter",
            filtered_count, len(search_results)
        )
    
    return filtered_results


def combine_filtered_links_from_models(
    doi: str,
    base_log_dir: Path,
    model_dirs: List[str],
) -> List[str]:
    """
    Combine filtered_links from multiple model directories for a given DOI.
    
    This function:
    1. Loads filtered_links.json from each model directory for the given DOI
    2. Combines all filtered_links lists
    3. Deduplicates the combined list
    4. Removes Cochrane URLs (since they're automatically appended)
    
    Args:
        doi: DOI string (e.g., "10.1002/14651858.CD000247.pub4")
        base_log_dir: Base log directory path (e.g., Path("logs"))
        model_dirs: List of model directory names (e.g., ["claude-sonnet-4-5_tools_filter", "gpt-5.1_tools_filter", "gemini-3-pro-preview_tools_filter"])
    
    Returns:
        Combined and deduplicated list of filtered links (with Cochrane URLs removed)
    
    Example:
        >>> base_dir = Path("logs")
        >>> models = ["claude-sonnet-4-5_tools_filter", "gpt-5.1_tools_filter", "gemini-3-pro-preview_tools_filter"]
        >>> links = combine_filtered_links_from_models("10.1002/14651858.CD000247.pub4", base_dir, models)
    """
    # Convert DOI to directory format (replace / with _)
    doi_dir = doi.replace("/", "_")
    
    all_links = []
    
    # Load filtered_links from each model directory
    for model_dir in model_dirs:
        filtered_links_file = base_log_dir / model_dir / doi_dir / "filtered_links.json"
        
        if filtered_links_file.exists():
            try:
                with open(filtered_links_file, 'r', encoding='utf-8') as f:
                    filtered_data = json.load(f)
                
                if 'filtered_links' in filtered_data and isinstance(filtered_data['filtered_links'], list):
                    all_links.extend(filtered_data['filtered_links'])
                    logger.debug(
                        "Loaded %d filtered links from %s",
                        len(filtered_data['filtered_links']), filtered_links_file
                    )
            except Exception as e:
                logger.warning(
                    "Failed to load filtered links from %s: %s",
                    filtered_links_file, e
                )
        else:
            logger.debug(
                "Filtered links file not found: %s",
                filtered_links_file
            )
    
    # Deduplicate while preserving order
    seen = set()
    deduplicated_links = []
    for link in all_links:
        if link not in seen:
            seen.add(link)
            deduplicated_links.append(link)
    
    # Remove Cochrane URLs (they're automatically appended)
    # Check for common Cochrane domain patterns
    cochrane_patterns = [
        "cochranelibrary.com",
        "cochrane.org",
        "abstracts.cochrane.org",
    ]
    
    filtered_links = []
    for link in deduplicated_links:
        link_lower = link.lower()
        is_cochrane = any(pattern in link_lower for pattern in cochrane_patterns)
        if not is_cochrane:
            filtered_links.append(link)
        else:
            logger.debug("Removed Cochrane URL from combined filtered links: %s", link)
    
    logger.info(
        "Combined filtered links for DOI %s: %d total from %d models, %d after deduplication, %d after removing Cochrane URLs",
        doi, len(all_links), len(model_dirs), len(deduplicated_links), len(filtered_links)
    )
    
    return filtered_links


def get_combined_filtered_links_for_doi(
    doi: str,
    base_log_dir: Optional[Path] = None,
    model_dirs: Optional[List[str]] = None,
) -> Optional[List[str]]:
    """
    Get combined filtered links for a DOI, with default model directories.
    
    This is a convenience function that uses default model directories if not provided.
    
    Args:
        doi: DOI string
        base_log_dir: Optional base log directory (defaults to sciconharness/logs relative to this file)
        model_dirs: Optional list of model directory names (defaults to the three specified models)
    
    Returns:
        Combined filtered links list, or None if no links found
    """
    if base_log_dir is None:
        # Default to logs/ relative to the project root (3 levels up from sciconharness/utils/)
        current_file = Path(__file__)
        base_log_dir = current_file.parent.parent.parent / "logs"
    
    if model_dirs is None:
        # Default to the three model directories
        model_dirs = [
            "claude-sonnet-4-5_tools_filter",
            "gpt-5.1_tools_filter",
            "gemini-3-pro-preview_tools_filter",
        ]
    
    combined_links = combine_filtered_links_from_models(doi, base_log_dir, model_dirs)
    
    return combined_links if combined_links else None


def filtered_links_to_domain_filter(filtered_links: List[str]) -> List[str]:
    """
    Convert a list of filtered links to domain filter format.
    
    This function:
    1. Cleans each link (removes protocol/www, keeps paths)
    2. Adds "-" prefix for denylist mode
    3. Deduplicates
    4. Adds default Cochrane domains
    
    Args:
        filtered_links: List of URLs to convert to domain filter format
    
    Returns:
        List of domain filter entries (with "-" prefix)
    """
    domain_filter_list = []
    seen_domains = set()
    
    for link in filtered_links:
        cleaned = _minimal_clean(link)
        cleaned = _ensure_denylist_prefix(cleaned)
        if cleaned not in seen_domains:
            domain_filter_list.append(cleaned)
            seen_domains.add(cleaned)
    
    # Add default Cochrane domains
    cochrane_domains = [
        "-cochranelibrary.com/",
        "-cochrane.org/"
    ]
    for domain in cochrane_domains:
        if domain not in seen_domains:
            domain_filter_list.append(domain)
            seen_domains.add(domain)
    
    return domain_filter_list
