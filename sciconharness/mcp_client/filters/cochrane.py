"""Cochrane-specific filter for MCP tool results."""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

from .base import BaseResultFilter

logger = logging.getLogger(__name__)


def _is_cochrane_url(url: str) -> bool:
    """Check if URL is a Cochrane-related URL."""
    if not url:
        return False
    url_lower = url.lower()
    return "cochrane" in url_lower

def _extract_urls_from_text(text: str) -> List[str]:
    """Extract URLs from text using regex."""
    if not text:
        return []
    url_pattern = r'https?://[^\s\)]+'
    return re.findall(url_pattern, text)

def normalize_title_for_matching(title: str) -> str:
    """
    Normalize a title for matching by extracting the core meaningful words.
    
    This comprehensive function removes all noise and standardizes the title to focus on actual words:
    - Converts to lowercase
    - Removes truncation markers (..., …, trailing dots)
    - Normalizes ALL Unicode hyphen/dash variants to regular hyphens (preserves hyphenated words)
    - Removes common prefixes ([PDF], [Review], [Article], Web Annex B.8., etc.)
    - Removes common suffixes (- PubMed, - PMC, | PubMed, | PMC, etc.)
    - Normalizes whitespace (multiple spaces to single space)
    - Removes trailing punctuation and ellipsis
    
    This function consolidates all title normalization logic used throughout the codebase to ensure
    consistent matching regardless of formatting variations.
    
    Args:
        title: The title string to normalize
        
    Returns:
        Normalized title string with only the core meaningful words
        
    Example:
        Input:  "Low-complexity automated nucleic acid amplification tests - PubMed"
        Output: "low-complexity automated nucleic acid amplification tests"
        
        Input:  "[PDF] Web Annex B.8. Parallel use of low‐complexity tests..."
        Output: "parallel use of low-complexity tests"
        
        Input:  "Parallel Use of Low-Complexity Automated Nucleic Acid ..."
        Output: "parallel use of low-complexity automated nucleic acid"
    """
    if not title:
        return ""
    
    # Convert to lowercase and strip
    normalized = title.lower().strip()
    
    # STEP 1: Remove truncation markers (ellipsis, trailing dots)
    # Handle both "..." and Unicode ellipsis "…"
    normalized = normalized.replace('...', ' ').replace('…', ' ').strip()
    # Remove trailing dots and ellipsis patterns (e.g., "title....")
    normalized = re.sub(r'\.{2,}\s*$', '', normalized).strip()
    # Remove trailing dots and spaces
    normalized = normalized.rstrip('. ').strip()
    
    # STEP 2: Normalize ALL Unicode hyphens/dashes to regular hyphens FIRST
    # This must be done before other operations to ensure consistent hyphen handling
    # Handles: ‐ (U+2010 hyphen), − (U+2212 minus), – (U+2013 en dash), — (U+2014 em dash),
    #          ― (U+2015 horizontal bar), ‒ (U+2012 figure dash), ­ (U+00AD soft hyphen),
    #          ﹣ (U+FE63 small hyphen-minus), ֊ (U+058A Armenian hyphen), ־ (U+05BE Hebrew maqaf)
    normalized = re.sub(r'[\u00AD\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uFE63\u058A\u05BE-]', '-', normalized)
    
    # STEP 3: Remove common prefixes (bracketed tags, Web Annex, etc.)
    # Remove [PDF], [Review], [Article], [Meta-analysis], etc. (case-insensitive)
    normalized = re.sub(r'^\[[^\]]+\]\s*', '', normalized).strip()
    
    # Remove "Web Annex" prefixes with various formats:
    # - "Web Annex B.8."
    # - "Web Annex 4.12."
    # - "Web Annex B.8" (without period)
    # - "Web Annex 4.12" (without period)
    normalized = re.sub(r'^web\s+annex\s+[a-z0-9.]+\.?\s*', '', normalized, flags=re.IGNORECASE).strip()
    # More general pattern for Web Annex variations (handles edge cases)
    if normalized.startswith('web annex'):
        # Remove everything up to and including the first period or space after "annex"
        normalized = re.sub(r'^web\s+annex[^a-z]*', '', normalized, flags=re.IGNORECASE).strip()
    
    # STEP 4: Remove common suffixes (database names, separators, etc.)
    # Remove patterns like:
    # - " - PubMed"
    # - " - PMC"
    # - " | PubMed"
    # - " | PMC"
    # - " - NCBI"
    # - " - ..." (with ellipsis)
    # Handles both regular hyphen and pipe separator
    normalized = re.sub(r'\s*[-|]\s*(PubMed|PMC|NCBI|pubmed|pmc|ncbi|\.\.\.|…).*$', '', normalized).strip()
    
    # STEP 5: Remove trailing ellipsis, dots, and other punctuation that might remain
    normalized = normalized.rstrip('.… ').strip()
    
    # STEP 6: Normalize whitespace (multiple spaces to single space)
    # Preserve hyphens as they're part of the word structure (e.g., "low-complexity" stays as one unit)
    # This ensures "low-complexity" and "low complexity" are treated differently for matching
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse date string into datetime object.
    
    Supports formats:
    - "23 October 2023"
    - "Oct 23, 2023"
    - "2023-10-23"
    - "Jun 12, 2016"
    - "Mar 8, 2021"
    
    Args:
        date_str: Date string to parse
        
    Returns:
        datetime object or None if parsing fails
    """
    if not date_str:
        return None
    
    # Try common date formats
    date_formats = [
        "%d %B %Y",      # "23 October 2023"
        "%B %d, %Y",     # "October 23, 2023"
        "%b %d, %Y",     # "Oct 23, 2023", "Jun 12, 2016"
        "%d %b %Y",      # "23 Oct 2023"
        "%Y-%m-%d",      # "2023-10-23"
        "%Y-%m",         # "2023-10" (fallback to first day of month)
        "%Y",            # "2023" (fallback to Jan 1)
    ]
    
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    
    # Try parsing with dateutil if available (more flexible)
    try:
        from dateutil import parser
        return parser.parse(date_str)
    except (ImportError, ValueError):
        pass
    
    return None


def create_title_filter_from_list(title_list: Union[List[str], Set[str]]) -> callable:
    """Create a title filter function from a list or set of titles (case-insensitive).
    
    Args:
        title_list: List or set of titles to filter. If a result title contains (case-insensitive)
                    any title from this list/set, it will be filtered. This handles cases where
                    search results have suffixes like " - PubMed" or " - PMC".
                    Matching is done at word boundaries to avoid false matches (e.g., "chronic" 
                    won't match "chronically").
    
    Returns:
        A function that takes a title string and returns True if it should be filtered
    """
    if not title_list:
        return None
    
    # Pre-normalize all titles in the list for efficient matching
    normalized_title_list = [normalize_title_for_matching(title) for title in title_list if title]
    
    def title_filter(title: str) -> bool:
        # Normalize the input title using the comprehensive normalization function
        title_clean = normalize_title_for_matching(title)
        
        if not title_clean:
            return False
        
        for list_title_normalized in normalized_title_list:
            if not list_title_normalized:
                continue
            
            list_title_clean = list_title_normalized
            
            if not title_clean or not list_title_clean:
                continue
            
            # Method 1: Exact substring match (handles suffixes like " - PubMed" that weren't caught)
            # Check if list title appears as a complete phrase in cleaned result title
            # Require at least 4 words to avoid false positives with short titles
            list_title_words = list_title_clean.split()
            if len(list_title_words) >= 4 and list_title_clean in title_clean:
                # Verify it's at word boundaries (not part of another word)
                escaped = re.escape(list_title_clean)
                # Match if it's at word boundaries or is the exact string
                if re.search(rf'\b{escaped}\b|^{escaped}$', title_clean):
                    return True
            
            # Method 1b: Check if result title appears in list title (reverse direction)
            # This handles truncated search result titles - check if truncated title matches start of full title
            # This is the PRIMARY method for truncated titles like "Parallel Use of Low-Complexity Automated Nucleic Acid ..."
            title_words = title_clean.split()
            if len(title_words) >= 3:
                # PRIMARY CHECK: Check if ALL words in the filtered title match the start of the Cochrane title
                # This is the most accurate check - compare all available words, not just first N
                if list_title_clean.startswith(title_clean):
                    return True
                
                # PRIMARY CHECK VARIANT: Handle cases where result title is missing words
                # Example: "parallel use of low-complexity nucleic acid amplification tests" 
                # should match "parallel use of low-complexity automated nucleic acid amplification tests..."
                # by checking if all words in result title appear in order in the source title
                # This handles cases where search results omit words like "automated"
                list_title_words = list_title_clean.split()
                if len(list_title_words) >= len(title_words) and len(title_words) >= 5:
                    # Check if all words from result title appear in order in source title
                    # Allow for missing words in between (like "automated"), but require words to be reasonably close
                    # Require at least 5 words to avoid false positives with short titles
                    title_idx = 0
                    skipped_words = 0
                    max_skipped = 2  # Allow up to 2 words to be skipped between matches
                    
                    for list_word in list_title_words:
                        if title_idx < len(title_words):
                            if title_words[title_idx].lower() == list_word.lower():
                                title_idx += 1
                                skipped_words = 0  # Reset skip counter on match
                                # If we've matched all words, it's a match
                                if title_idx == len(title_words):
                                    return True
                            else:
                                skipped_words += 1
                                # If we've skipped too many words, this probably isn't a match
                                if skipped_words > max_skipped:
                                    break
                    
                    # If we matched all words (allowing for some skipped words), it's likely the same title
                    if title_idx == len(title_words):
                        return True
                
                # SECONDARY CHECK: Only use progressively shorter prefixes if we suspect truncation
                # Check if title appears truncated (ends with ellipsis, is unusually short, etc.)
                # Otherwise, require ALL words to match
                appears_truncated = (
                    title_clean.endswith('...') or 
                    title_clean.endswith('…') or
                    len(title_words) < 8  # Titles with fewer than 8 words might be truncated
                )
                
                if appears_truncated:
                    # For truncated titles, check progressively shorter prefixes
                    # Check 7, 6 words to catch different truncation points
                    for prefix_len in [7, 6]:
                        if len(title_words) >= prefix_len:
                            title_prefix = ' '.join(title_words[:prefix_len])
                            if list_title_clean.startswith(title_prefix):
                                # Additional validation: if we have more words, check that the next word also matches
                                # This prevents false positives where titles share a common prefix but diverge
                                if len(title_words) > prefix_len:
                                    # Get the next word after the prefix
                                    next_word = title_words[prefix_len] if prefix_len < len(title_words) else None
                                    if next_word:
                                        # Check if the Cochrane title has the same next word
                                        list_title_words = list_title_clean.split()
                                        if len(list_title_words) > prefix_len:
                                            list_next_word = list_title_words[prefix_len]
                                            if next_word.lower() == list_next_word.lower():
                                                return True
                                        # If next word doesn't match, don't consider it a match (likely different papers)
                                else:
                                    # If the filtered title is exactly this length, it's a match
                                    return True
                # If title doesn't appear truncated, we already checked full match above, so no need for prefix fallback
            
            # Method 1c: Check if significant portion of result title matches start of list title
            # This is a fallback for cases where truncation happens mid-word or with slight variations
            # Only use this if the title appears truncated (otherwise Method 1b should have caught it)
            appears_truncated = (
                title_clean.endswith('...') or 
                title_clean.endswith('…') or
                len(title_words) < 8  # Titles with fewer than 8 words might be truncated
            )
            
            if appears_truncated and len(title_words) >= 6:
                # Take first 6 words of result title
                title_prefix = ' '.join(title_words[:6])
                if list_title_clean.startswith(title_prefix):
                    # Additional validation: check that the next word also matches (if available)
                    # This prevents false positives where titles share a common prefix but diverge
                    if len(title_words) > 6:
                        next_word = title_words[6]
                        list_title_words = list_title_clean.split()
                        if len(list_title_words) > 6:
                            list_next_word = list_title_words[6]
                            if next_word.lower() != list_next_word.lower():
                                # Next word doesn't match - likely different papers, don't filter
                                pass
                            else:
                                # Next word matches - ensure it's at word boundaries
                                escaped_prefix = re.escape(title_prefix)
                                if re.search(rf'^{escaped_prefix}\b', list_title_clean):
                                    return True
                    else:
                        # Filtered title is exactly 6 words - check word boundaries
                        escaped_prefix = re.escape(title_prefix)
                        if re.search(rf'^{escaped_prefix}\b', list_title_clean):
                            return True
            
            # Method 2: Check if cleaned result title appears at the beginning of list title (handles truncation)
            # Require at least 3 words (reduced from 4) to catch more truncated titles
            title_words = title_clean.split()
            if len(title_words) >= 3 and list_title_clean.startswith(title_clean):
                # Additional check: ensure the next character (if any) is a word boundary
                if len(list_title_clean) == len(title_clean) or not list_title_clean[len(title_clean):len(title_clean)+1].isalnum():
                    return True
            
            # Method 3: List title starts with cleaned result title (handles truncation at end)
            # Require at least 3 words (reduced from 4) to catch more truncated titles
            if len(title_clean.split()) >= 3 and list_title_clean.startswith(title_clean):
                # Additional check: ensure the next character (if any) is a word boundary
                if len(list_title_clean) == len(title_clean) or not list_title_clean[len(title_clean):len(title_clean)+1].isalnum():
                    return True
        
        return False
    
    return title_filter


class CochraneResultFilter(BaseResultFilter):
    """Filter that removes Cochrane-related results from search tools.
    
    This filter applies to:
    - serper_google_webpage_search: Filters organic search results
    - semantic_scholar_snippet_search: Filters semantic scholar snippets
    - jina_fetch_webpage_content: Filters webpage content based on Cochrane mentions and titles
    
    Filters out items with:
    - URLs matching Cochrane domains (cochranelibrary.com, cochrane.org, etc.)
    - Titles containing "Cochrane" (case-insensitive)
    - Titles matching optional custom title filter (e.g., matches one of Cochrane review titles word-for-word)
    - Jina API content that contains "Cochrane" AND the source_title (highly indicates content cites the source)
    - Search Result Published after the publication_date cutoff
    
    Example:
        # Basic usage (filters Cochrane URLs only)
        filter = CochraneResultFilter()
        
        # With title filter list (filters titles that exactly match any title in the list)
        filter = CochraneResultFilter(title_filter_list=["Title 1", "Title 2", "Title 3"])
    """
    
    def __init__(
        self, 
        title_filter_list: Optional[Union[List[str], Set[str]]] = None,
        source_title: Optional[str] = None,
        publication_date: Optional[str] = None
    ):
        """Initialize the filter.
        
        Args:
            title_filter_list: Optional list or set of titles. If provided, creates a filter that
                              filters titles that exactly match (case-insensitive) any title in
                              this list/set. Used for search result filtering.
            source_title: Optional specific source title to filter. Used for Jina API content filtering.
                         If not provided but title_filter_list has one item, uses that as source_title.
            publication_date: Optional date string (e.g., "23 October 2023"). Items published
                             after this date will be filtered out. Supports various date formats.
        """
        self.title_filter_list = title_filter_list if title_filter_list else None
        
        if title_filter_list:
            self.title_filter = create_title_filter_from_list(title_filter_list)
        else:
            self.title_filter = None
        
        # Set source_title for Jina content filtering
        # Use provided source_title, or if title_filter_list has exactly one item, use that
        if source_title:
            self.source_title = source_title
        elif title_filter_list and len(title_filter_list) == 1:
            self.source_title = list(title_filter_list)[0]
        else:
            self.source_title = None
        
        # Parse publication date
        self.publication_date_cutoff = None
        if publication_date:
            parsed_date = _parse_date(publication_date)
            if parsed_date:
                self.publication_date_cutoff = parsed_date
            else:
                logger.warning("Could not parse publication_date: %s", publication_date)
        
        # Tools that this filter applies to
        self.filtered_tools = {
            "serper_google_webpage_search",
            "semantic_scholar_snippet_search",
            "jina_fetch_webpage_content"
        }
        
        # Track unique filtered links throughout the tool calling process
        self.filtered_links: Set[str] = set()
        
        # Track filtered items with reasons for summary reporting
        # Structure: {tool_name: [{"title": str, "url": str, "reason": str, "date": Optional[str]}, ...]}
        self.filtered_items: Dict[str, List[Dict[str, Any]]] = {
            "jina_fetch_webpage_content": [],
            "semantic_scholar_snippet_search": [],
            "serper_google_webpage_search": [],
        }
    
    def should_filter_tool(self, tool_name: str) -> bool:
        """Check if this filter should be applied to the given tool.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if the tool should be filtered, False otherwise
        """
        return tool_name in self.filtered_tools
    
    def _should_filter_item(
        self, 
        title: str, 
        urls: List[str], 
        publication_date: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """Determine if an item should be filtered and return reason.
        
        Args:
            title: Item title to check
            urls: List of URLs associated with the item
            publication_date: Optional publication date string to check
            
        Returns:
            Tuple of (should_filter, reason)
        """
        # FIRST: Check if any URL has been filtered before (most robust check)
        # This handles cases where the same URL appears with different titles
        try:
            from ..utils.utils import is_url_filtered
            for url in urls:
                if url and is_url_filtered(url):
                    return True, f"URL was previously filtered: {url}"
        except ImportError:
            # Try remote_mcp_servers utils as fallback (for remote MCP servers)
            try:
                from sciconharness.remote_mcp_servers.utils import is_url_filtered
                for url in urls:
                    if url and is_url_filtered(url):
                        return True, f"URL was previously filtered: {url}"
            except ImportError:
                # If utils module not available, skip this check
                pass
        
        # Check URLs for Cochrane
        for url in urls:
            if _is_cochrane_url(url):
                return True, f"Cochrane URL: {url}"
        
        # Check if title contains "Cochrane" (case-insensitive)
        if title and "cochrane" in title.lower():
            return True, f"Title contains 'Cochrane': {title}"
        
        # Check title with custom filter if provided
        if self.title_filter and self.title_filter(title):
            return True, f"Title filter matched: {title}"
        
        # Check publication date if cutoff is set
        # Only filter if the item has a date - items without dates are kept
        if self.publication_date_cutoff and publication_date:
            item_date = _parse_date(publication_date)
            if item_date and item_date > self.publication_date_cutoff:
                return True, f"Published after cutoff date ({self.publication_date_cutoff.strftime('%d %B %Y')}): {publication_date}"
            # If date exists but couldn't be parsed, log a warning but don't filter
            elif publication_date and not item_date:
                logger.debug("Could not parse date '%s' for item '%s'", publication_date, title)
        
        return False, None
    
    def _should_filter_jina_content(
        self,
        content: str,
        title: str,
        url: str
    ) -> tuple[bool, Optional[str]]:
        """Determine if Jina API content should be filtered.
        
        Filters content if:
        1. Title matches any Cochrane title from title_filter_list (using full title from fetch), OR
        2. BOTH conditions are met:
           - Content contains "Cochrane" keyword (case-insensitive)
           - Content contains the source article title
        
        Args:
            content: The webpage content text to check
            title: The webpage title (full title from fetch - use this for filtering!)
            url: The webpage URL (not used for filtering)
            
        Returns:
            Tuple of (should_filter, reason)
        """
        # FIRST: Check if URL has been filtered before (most robust check)
        if url:
            try:
                from ..utils.utils import is_url_filtered
                if is_url_filtered(url):
                    return True, f"URL was previously filtered: {url}"
            except ImportError:
                # Try remote_mcp_servers utils as fallback (for remote MCP servers)
                try:
                    from sciconharness.remote_mcp_servers.utils import is_url_filtered
                    if is_url_filtered(url):
                        return True, f"URL was previously filtered: {url}"
                except ImportError:
                    # If utils module not available, skip this check
                    pass
        
        # SECOND: Check title against Cochrane titles list (full title from fetch)
        # This is the primary check since fetch provides the full title
        if title and self.title_filter:
            # Normalize title for logging and comparison
            title_normalized = normalize_title_for_matching(title)
            
            logger.info(f"Filter check: Checking title against Cochrane titles list")
            logger.info(f"  Original title: {title[:120]}...")
            logger.info(f"  Normalized title: {title_normalized[:120]}...")
            
            # title_filter uses normalize_title_for_matching internally and checks against all Cochrane titles
            if self.title_filter(title):
                logger.info(f"Filter matched: Title matches a Cochrane title from filter list")
                return True, f"Title matches a Cochrane title from filter list: {title}"
            logger.debug(f"Filter check: Title did not match any Cochrane titles")
        
        # THIRD: Check content: requires BOTH "Cochrane" keyword AND source title
        if not content:
            logger.debug("Filter check: No content provided")
            return False, None
        
        # Both conditions required - if no source title, can't filter by content
        if not self.source_title:
            logger.debug("Filter check: No source title configured")
            return False, None
        
        content_lower = content.lower()
        source_title_lower = self.source_title.lower()
        
        # Check if content contains "Cochrane" keyword
        contains_cochrane = "cochrane" in content_lower
        
        logger.info(f"Filter check: Content length={len(content)}, contains_cochrane={contains_cochrane}, source_title={self.source_title[:80]}...")
        
        if not contains_cochrane:
            logger.debug("Filter check: Content does not contain 'Cochrane' keyword")
            return False, None
        
        # Normalize both strings for comparison (handle different dash/hyphen characters and spaces)
        # For content matching, we normalize the source title using the same function as title matching
        # This ensures consistent normalization across all filtering operations
        normalized_source_title = normalize_title_for_matching(self.source_title)
        
        # For content, normalize similarly but keep all words (don't remove suffixes/prefixes from content)
        # Just normalize hyphens and spaces to make it searchable
        def normalize_content_for_matching(text: str) -> str:
            """Normalize content text for matching by standardizing hyphens and spaces, keeping all words."""
            if not text:
                return ""
            # Normalize ALL Unicode hyphens to regular hyphens (same as title normalization)
            text = re.sub(r'[\u00AD\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uFE63\u058A\u05BE-]', '-', text)
            # Normalize multiple spaces/hyphens to single space
            text = re.sub(r'[\s\-]+', ' ', text)
            # Remove punctuation but keep alphanumeric and spaces (for word matching)
            text = re.sub(r'[^\w\s]', '', text)
            # Normalize multiple spaces to single space (final pass)
            text = re.sub(r'\s+', ' ', text)
            return text.lower().strip()
        
        normalized_content = normalize_content_for_matching(content_lower)
        
        # Log for debugging
        logger.debug(f"Filter check - Source title (normalized): {normalized_source_title[:150]}")
        logger.debug(f"Filter check - Content snippet (normalized): {normalized_content[:300]}")
        
        # EXACT MATCH ONLY: Check if normalized source title appears in normalized content
        # This requires the full title to appear in the correct order
        if normalized_source_title in normalized_content:
            logger.info(f"Filter matched: Exact normalized source title found in content (in order)")
            return True, f"Content contains 'Cochrane' keyword and exact source title match: {self.source_title}"
        
        logger.warning(f"Filter did not match: Source title not found in content (exact match required)")
        logger.warning(f"  Source title (normalized, first 200 chars): {normalized_source_title[:200]}")
        logger.warning(f"  Content snippet (normalized, first 500 chars): {normalized_content[:500]}")
        logger.warning(f"  Source title length: {len(normalized_source_title)}, Content length: {len(normalized_content)}")
        return False, None
    
    def _filter_list_items(
        self,
        items: List[Dict[str, Any]],
        get_title: callable,
        get_urls: callable,
        get_metadata: callable,
        get_date: callable,
        tool_name: str = ""
    ) -> List[Dict[str, Any]]:
        """Filter a list of items based on title, URL, and date criteria.
        
        Args:
            items: List of items to filter
            get_title: Function(item) -> str to extract title
            get_urls: Function(item) -> List[str] to extract URLs
            get_metadata: Function(item) -> Dict for logging metadata
            get_date: Function(item) -> Optional[str] to extract publication date
            tool_name: Name of tool for logging
            
        Returns:
            Filtered list of items
        """
        filtered_items = []
        original_count = len(items)
        
        # Iterate through items in order and preserve order in filtered list
        for item in items:
            title = get_title(item)
            urls = get_urls(item)
            date = get_date(item)
            should_filter, reason = self._should_filter_item(title, urls, date)
            
            if should_filter:
                metadata = get_metadata(item)
                logger.info("FILTERED OUT - %s", reason)
                logger.info("  Title: %s", title)
                if date:
                    logger.info("  Date: %s", date)
                for key, value in metadata.items():
                    if value:
                        logger.info("  %s: %s", key, value)
                
                # Track filtered URLs
                for url in urls:
                    if url:  # Only track non-empty URLs
                        self.filtered_links.add(url)
                
                # Track filtered item with reason for summary
                filtered_item = {
                    "title": title,
                    "url": urls[0] if urls else "",
                    "reason": reason or "Unknown reason",
                    "date": date,
                }
                if tool_name in self.filtered_items:
                    self.filtered_items[tool_name].append(filtered_item)
            else:
                # Append items in original order (no reordering)
                filtered_items.append(item)
        
        # Update position indices for filtered items
        for idx, item in enumerate(filtered_items, start=1):
            if "position" in item:
                item["position"] = idx
        
        filtered_count = len(filtered_items)
        if original_count != filtered_count:
            logger.info("Filtered %d out of %d %s results", 
                      original_count - filtered_count, original_count, tool_name)
        
        return filtered_items
    
    def filter(self, tool_result: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        """Filter search results from supported tools.
        
        Supported tools:
        - serper_google_webpage_search: Filters organic search results
        - semantic_scholar_snippet_search: Filters semantic scholar snippets
        - jina_fetch_webpage_content: Filters webpage content based on Cochrane mentions and titles
        
        Args:
            tool_result: The tool result dictionary to filter
            tool_name: Name of the tool that produced the result
            
        Returns:
            Filtered tool result dictionary
        """
        filtered_result = tool_result.copy()
        
        if tool_name == "jina_fetch_webpage_content":
            # Jina returns a single result, not a list
            if isinstance(filtered_result, dict) and filtered_result.get("success", False):
                content = filtered_result.get("content", "")
                title = filtered_result.get("title", "")
                url = filtered_result.get("url", "")
                
                should_filter, reason = self._should_filter_jina_content(content, title, url)
                
                if should_filter:
                    logger.info("FILTERED OUT - %s", reason)
                    logger.info("  Title: %s", title)
                    logger.info("  URL: %s", url)
                    
                    # Track filtered URL
                    if url:  # Only track non-empty URLs
                        self.filtered_links.add(url)
                    
                    # Track filtered item with reason for summary
                    filtered_item = {
                        "title": title,
                        "url": url,
                        "reason": reason or "Unknown reason",
                        "date": filtered_result.get("publishedTime", ""),
                    }
                    if tool_name in self.filtered_items:
                        self.filtered_items[tool_name].append(filtered_item)
                    
                    # Return result with empty content (success=True but content is empty)
                    return {
                        "url": url,
                        "title": title,
                        "content": "",
                        "description": "",
                        "publishedTime": filtered_result.get("publishedTime", ""),
                        "metadata": filtered_result.get("metadata", {}),
                        "success": True,
                    }
        
        elif tool_name == "serper_google_webpage_search":
            if "organic" in filtered_result and isinstance(filtered_result["organic"], list):
                def get_title(item): return item.get("title", "")
                def get_urls(item): return [item.get("link", "")]
                def get_metadata(item): return {"Link": item.get("link", "")}
                def get_date(item): return item.get("date")
                
                filtered_result["organic"] = self._filter_list_items(
                    filtered_result["organic"],
                    get_title, get_urls, get_metadata, get_date,
                    "serper_google_webpage_search"
                )
        
        elif tool_name == "semantic_scholar_snippet_search":
            if "data" in filtered_result and isinstance(filtered_result["data"], list):
                def get_title(item):
                    return item.get("paper", {}).get("title", "")
                
                def get_urls(item):
                    """Extract URLs from Semantic Scholar API result item.
                    
                    Checks:
                    1. openAccessInfo.disclaimer for URLs (PMC, DOI, etc.)
                    2. paper.url field
                    3. Constructs from paperId or corpusId if available
                    """
                    urls = []
                    paper = item.get("paper", {})
                    
                    # First, try to extract URL from disclaimer
                    open_access_info = paper.get("openAccessInfo", {})
                    disclaimer_text = open_access_info.get("disclaimer", "")
                    
                    if disclaimer_text:
                        import re
                        url_pattern = r'https?://[^\s,)]+'
                        found_urls = re.findall(url_pattern, disclaimer_text)
                        
                        if found_urls:
                            # Prefer PMC URLs, then DOI URLs, then other URLs (skip unpaywall API URLs)
                            for found_url in found_urls:
                                if 'pmc.ncbi.nlm.nih.gov' in found_url:
                                    urls.append(found_url)
                                    break
                                elif 'doi.org' in found_url and 'unpaywall.org' not in found_url:
                                    urls.append(found_url)
                                    break
                                elif 'unpaywall.org' not in found_url:
                                    urls.append(found_url)
                                    break
                            
                            # If no preferred URL found, use first non-unpaywall URL
                            if not urls:
                                for found_url in found_urls:
                                    if 'unpaywall.org' not in found_url:
                                        urls.append(found_url)
                                        break
                    
                    # Second, try direct URL from paper
                    if not urls:
                        paper_url = paper.get("url", "")
                        if paper_url:
                            urls.append(paper_url)
                    
                    # Third, construct from paperId or corpusId
                    if not urls:
                        paper_id = paper.get("paperId")
                        corpus_id = paper.get("corpusId")
                        if paper_id:
                            urls.append(f"https://www.semanticscholar.org/paper/{paper_id}")
                        elif corpus_id:
                            urls.append(f"https://www.semanticscholar.org/paper/{corpus_id}")
                    
                    return urls
                
                def get_metadata(item):
                    paper = item.get("paper", {})
                    return {"Corpus ID": paper.get("corpusId", "")}
                
                # No date filtering for semantic scholar - use None as get_date
                def get_date(item):
                    return None
                
                filtered_result["data"] = self._filter_list_items(
                    filtered_result["data"],
                    get_title, get_urls, get_metadata, get_date,
                    "semantic_scholar_snippet_search"
                )
        
        return filtered_result
    
    def get_filtered_links(self) -> List[str]:
        """Get list of unique filtered links.
        
        Returns:
            Sorted list of unique filtered URLs
        """
        return sorted(list(self.filtered_links))
    
    def reset_filtered_links(self):
        """Reset the filtered links tracking (useful for testing or reusing filter instances)."""
        self.filtered_links.clear()
    
    def log_filtering_summary(self):
        """Log a summary of all filtered items grouped by API type."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("FILTERING SUMMARY")
        logger.info("=" * 80)
        
        total_filtered = 0
        api_names = {
            "jina_fetch_webpage_content": "Jina API",
            "semantic_scholar_snippet_search": "Semantic Scholar",
            "serper_google_webpage_search": "Google Serper",
        }
        
        # Group all filtered items for a comprehensive view
        all_filtered = []
        for tool_name, items in self.filtered_items.items():
            if items:
                api_name = api_names.get(tool_name, tool_name)
                for item in items:
                    all_filtered.append({
                        "api": api_name,
                        "tool_name": tool_name,
                        **item
                    })
                total_filtered += len(items)
        
        if total_filtered == 0:
            logger.info("No items were filtered during this query.")
        else:
            # Show breakdown by API first
            logger.info("")
            logger.info("Breakdown by API:")
            for tool_name, items in self.filtered_items.items():
                if items:
                    api_name = api_names.get(tool_name, tool_name)
                    logger.info("  - %s: %d item(s)", api_name, len(items))
            
            # Show detailed list grouped by API
            logger.info("")
            logger.info("Detailed Filtering Results:")
            logger.info("-" * 80)
            
            for tool_name, items in self.filtered_items.items():
                if items:
                    api_name = api_names.get(tool_name, tool_name)
                    logger.info("")
                    logger.info("%s (%d item(s)):", api_name, len(items))
                    logger.info("-" * 80)
                    for idx, item in enumerate(items, start=1):
                        logger.info("  %d. Filtered by: %s", idx, api_name)
                        logger.info("     URL: %s", item.get("url", "N/A"))
                        logger.info("     Title: %s", item.get("title", "N/A"))
                        if item.get("date"):
                            logger.info("     Date: %s", item.get("date"))
                        logger.info("     How filtered: %s", item.get("reason", "Unknown reason"))
                        logger.info("")
            
            logger.info("")
            logger.info("Total items filtered: %d", total_filtered)
        
        logger.info("=" * 80)
        logger.info("")
    
    def get_filtering_summary(self) -> Dict[str, Any]:
        """Get a dictionary summary of all filtered items.
        
        Returns:
            Dictionary with filtering statistics and details
        """
        summary = {
            "total_filtered": 0,
            "by_api": {},
        }
        
        api_names = {
            "jina_fetch_webpage_content": "Jina API",
            "semantic_scholar_snippet_search": "Semantic Scholar",
            "serper_google_webpage_search": "Google Serper",
        }
        
        for tool_name, items in self.filtered_items.items():
            api_name = api_names.get(tool_name, tool_name)
            summary["by_api"][api_name] = {
                "count": len(items),
                "items": items,
            }
            summary["total_filtered"] += len(items)
        
        return summary


# Convenience: default Cochrane filter instance (URLs only, no title filter)
custom_cochrane_filter_search_results = CochraneResultFilter()


