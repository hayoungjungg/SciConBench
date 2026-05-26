#!/usr/bin/env python3
"""
Collect and process filtered links from mcp_client.log files for Perplexity deep research.

This script:
1. Parses mcp_client.log files from claude-sonnet-4-5_tools_filter, gemini-3-pro-preview_tools_filter, and gpt-5.1_tools_filter
2. Extracts filtered links by parsing "FILTERED OUT" entries and matching them with search results
3. Uses existing code to deduplicate, clean up URLs, and format them (adding "-" prefix)
4. Returns the top 20 most frequently occurring URLs per DOI and across all DOIs
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add project root to path for imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from sciconharness.mcp_client.llm_providers.perplexity_provider import (
    _sanitize_pubmed_pmc_url,
    _extract_domain_from_url,
)


def extract_urls_from_semantic_scholar_item(item: Dict) -> List[str]:
    """
    Extract URLs from a Semantic Scholar API result item (same logic as get_urls in cochrane.py).
    
    Args:
        item: Semantic Scholar result item
    
    Returns:
        List of URLs extracted from the item
    """
    urls = []
    paper = item.get("paper", {})
    
    # First, try to extract URL from disclaimer
    open_access_info = paper.get("openAccessInfo", {})
    disclaimer_text = open_access_info.get("disclaimer", "")
    
    if disclaimer_text:
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


def extract_urls_from_serper_item(item: Dict) -> List[str]:
    """
    Extract URLs from a Serper Google search result item.
    
    Args:
        item: Serper search result item
    
    Returns:
        List of URLs (typically just the link field)
    """
    urls = []
    link = item.get("link", "")
    if link:
        urls.append(link)
    return urls


def parse_mcp_client_log(log_file: Path) -> List[str]:
    """
    Parse a mcp_client.log file to extract filtered URLs.
    
    This function uses multiple strategies:
    1. Extract URLs directly from "FILTERED OUT" lines that contain URLs
    2. Extract URLs from metadata lines (Link, Corpus ID -> construct URL)
    3. Use regex to find URLs near "FILTERED OUT" entries
    
    Args:
        log_file: Path to mcp_client.log file
    
    Returns:
        List of filtered URLs found in the log
    """
    if not log_file.exists():
        return []
    
    filtered_urls = set()  # Use set to avoid duplicates
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Warning: Failed to read {log_file}: {e}")
        return []
    
    # URL pattern for matching URLs in log
    # Match URLs - handle parentheses in URLs but stop at whitespace, quotes, or angle brackets
    # This is a more robust pattern that handles URLs with parentheses
    url_pattern = re.compile(r'https?://[^\s<>"\']+')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for "FILTERED OUT" entries
        if "FILTERED OUT" in line:
            # Strategy 1: Extract URL directly from the reason field
            # Examples: "FILTERED OUT - Cochrane URL: https://..."
            #           "FILTERED OUT - URL was previously filtered: https://..."
            url_matches = url_pattern.findall(line)
            for url in url_matches:
                # Clean up URL (remove trailing punctuation)
                url = url.rstrip('.,;:')
                filtered_urls.add(url)
            
            # Strategy 2: Look at following lines for metadata with URLs
            # Only extract metadata if it's part of this FILTERED OUT entry
            # Metadata lines have "  " after the log prefix (e.g., "INFO -   Title:")
            j = i + 1
            found_metadata = False
            while j < len(lines) and j < i + 10:  # Look at next 10 lines
                next_line = lines[j]
                
                # Check if this is a metadata line
                # Metadata lines contain "  Title:", "  Corpus ID:", "  Link:" after the log prefix
                is_metadata_line = ("  Title:" in next_line or "  Corpus ID:" in next_line or 
                                   "  Link:" in next_line or "  Date:" in next_line)
                
                # Stop if we hit a new log entry that's NOT metadata
                # A new log entry starts with timestamp pattern
                if not is_metadata_line:
                    # Check if this is a new log entry (starts with timestamp)
                    if re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - ', next_line.strip()):
                        # Check if it's another FILTERED OUT entry (which we'll process separately)
                        # or a different log entry (which means we've moved past this FILTERED OUT's metadata)
                        if "FILTERED OUT" not in next_line:
                            # This is a new non-FILTERED OUT log entry, stop processing metadata
                            break
                        else:
                            # This is a new FILTERED OUT entry, stop processing metadata for previous one
                            break
                
                # Only extract metadata if we're still in the metadata section of this FILTERED OUT entry
                if is_metadata_line:
                    # Extract Link (for Serper)
                    if "  Link:" in next_line:
                        link_match = url_pattern.search(next_line)
                        if link_match:
                            url = link_match.group(0).rstrip('.,;:')
                            # Skip Semantic Scholar corpus ID URLs
                            if not (re.search(r'semanticscholar\.(org|com)/paper/\d+', url, re.IGNORECASE) or
                                    re.search(r'semanticscholar\.(org|com)/corpusid', url, re.IGNORECASE)):
                                filtered_urls.add(url)
                                found_metadata = True
                    
                    # Extract Corpus ID - but DON'T construct Semantic Scholar URL
                    # (we filter those out anyway)
                    # Note: We're NOT constructing Semantic Scholar URLs from Corpus ID anymore
                    # as those will be filtered out
                    
                    # Look for URLs in any metadata line (but skip Semantic Scholar corpus ID URLs)
                    url_matches = url_pattern.findall(next_line)
                    for url in url_matches:
                        url = url.rstrip('.,;:')
                        # Only add if it looks like a real URL (not just part of text)
                        # Skip Semantic Scholar corpus ID URLs
                        if (len(url) > 10 and ('http://' in url or 'https://' in url) and
                            not re.search(r'semanticscholar\.(org|com)/paper/\d+', url, re.IGNORECASE) and
                            not re.search(r'semanticscholar\.(org|com)/corpusid', url, re.IGNORECASE)):
                            filtered_urls.add(url)
                            found_metadata = True
                else:
                    # If we're past metadata and haven't hit a new log entry yet, continue a bit
                    # but break if we've moved too far
                    if j > i + 5:
                        break
                
                j += 1
        
        i += 1
    
    return list(filtered_urls)


def clean_and_format_url(url: str) -> Optional[str]:
    """
    Clean and format a URL using existing code from perplexity provider.
    
    Args:
        url: Raw URL string
    
    Returns:
        Cleaned and formatted URL with "-" prefix, or None if URL should be excluded
    """
    if not url or not url.strip():
        return None
    
    url_lower = url.lower()
    
    # Filter out Cochrane URLs
    if 'cochrane.org' in url_lower or 'cochranelibrary.com' in url_lower:
        return None
    
    # Filter out Semantic Scholar URLs with corpusID patterns
    # These are URLs like:
    # - https://www.semanticscholar.org/paper/{corpus_id}
    # - https://semanticscholar.com/paper/{corpus_id}
    # - https://www.semanticscholar.org/corpusID/{id}
    # - https://semanticscholar.com/corpusID/{id}
    if 'semanticscholar' in url_lower:
        # Check for corpus ID patterns: /paper/ followed by digits, or /corpusid
        if (re.search(r'/paper/\d+', url_lower) or 
            re.search(r'/corpusid', url_lower) or
            re.search(r'/corpus-id', url_lower)):
            return None
    
    # Check if it's a PubMed/PMC URL
    if 'pubmed.ncbi.nlm.nih.gov' in url_lower or 'pmc.ncbi.nlm.nih.gov' in url_lower:
        # Use PubMed/PMC sanitization
        cleaned = _sanitize_pubmed_pmc_url(url)
        if cleaned is None:
            return None
        # Ensure denylist prefix
        if not cleaned.startswith('-'):
            cleaned = f"-{cleaned}"
        # Double-check for Cochrane URLs after cleaning
        if 'cochrane.org' in cleaned.lower() or 'cochranelibrary.com' in cleaned.lower():
            return None
        return cleaned
    else:
        # Use domain extraction for other URLs
        cleaned = _extract_domain_from_url(url)
        # Ensure denylist prefix
        if not cleaned.startswith('-'):
            cleaned = f"-{cleaned}"
        # Double-check for Cochrane URLs after cleaning
        if 'cochrane.org' in cleaned.lower() or 'cochranelibrary.com' in cleaned.lower():
            return None
        return cleaned


def collect_filtered_links_per_doi(
    base_log_dir: Path,
    model_dirs: List[str]
) -> Tuple[Dict[str, List[str]], int]:
    """
    Collect filtered links per DOI from mcp_client.log files.
    
    Args:
        base_log_dir: Base log directory
        model_dirs: List of model directory names
    
    Returns:
        Tuple of (dictionary mapping DOI to filtered URLs, total DOIs processed)
    """
    doi_to_urls = defaultdict(list)
    all_dois_seen = set()
    
    for model_dir in model_dirs:
        model_path = base_log_dir / model_dir
        if not model_path.exists():
            print(f"Warning: Model directory not found: {model_path}")
            continue
        
        # Find all DOI directories (subdirectories) - only count those starting with 10.1002
        doi_dirs = [d for d in model_path.iterdir() if d.is_dir() and d.name.startswith("10.1002")]
        print(f"Found {len(doi_dirs)} DOI directories in {model_dir}")
        
        for doi_dir in doi_dirs:
            log_file = doi_dir / "mcp_client.log"
            filtered_links_file = doi_dir / "filtered_links.json"
            doi = doi_dir.name
            all_dois_seen.add(doi)  # Track all DOIs we've seen
            
            # Always initialize DOI in dict, even if no URLs found
            if doi not in doi_to_urls:
                doi_to_urls[doi] = []
            
            # Parse mcp_client.log
            if log_file.exists():
                print(f"  Parsing {doi}...")
                filtered_urls = parse_mcp_client_log(log_file)
                if filtered_urls:
                    doi_to_urls[doi].extend(filtered_urls)
                    print(f"    Found {len(filtered_urls)} filtered URLs from mcp_client.log")
            
            # Also load from filtered_links.json if it exists
            if filtered_links_file.exists():
                try:
                    with open(filtered_links_file, 'r', encoding='utf-8') as f:
                        filtered_data = json.load(f)
                    
                    if 'filtered_links' in filtered_data and isinstance(filtered_data['filtered_links'], list):
                        json_urls = filtered_data['filtered_links']
                        if json_urls:
                            doi_to_urls[doi].extend(json_urls)
                            print(f"    Found {len(json_urls)} filtered URLs from filtered_links.json")
                except Exception as e:
                    print(f"    Warning: Failed to load filtered_links.json: {e}")
    
    return dict(doi_to_urls), len(all_dois_seen)


def get_top_filtered_links(
    base_log_dir: Path,
    model_dirs: List[str],
    top_n: int = 18
) -> Tuple[List[Tuple[str, int]], Dict[str, List[Tuple[str, int]]], int]:
    """
    Get the top N most frequently occurring filtered links across all DOIs and per DOI.
    
    Args:
        base_log_dir: Base log directory
        model_dirs: List of model directory names
        top_n: Number of top URLs to return (default: 20)
    
    Returns:
        Tuple of (global_top_urls, per_doi_top_urls)
        - global_top_urls: List of (url, count) tuples sorted by frequency
        - per_doi_top_urls: Dict mapping DOI to list of (url, count) tuples
    """
    # Collect filtered links per DOI
    print("Collecting filtered links from mcp_client.log files...")
    doi_to_urls, total_dois_processed = collect_filtered_links_per_doi(base_log_dir, model_dirs)
    
    # Clean and format URLs
    print("\nCleaning and formatting URLs...")
    all_cleaned_urls = []
    doi_to_cleaned_urls = {}
    
    for doi, urls in doi_to_urls.items():
        cleaned_urls = []
        for url in urls:
            cleaned = clean_and_format_url(url)
            if cleaned:
                cleaned_urls.append(cleaned)
                all_cleaned_urls.append(cleaned)
        doi_to_cleaned_urls[doi] = cleaned_urls
    
    print(f"Total cleaned URLs across all DOIs: {len(all_cleaned_urls)}")
    
    # Count frequencies globally
    global_counter = Counter(all_cleaned_urls)
    global_top_urls = global_counter.most_common(top_n)
    
    # Count frequencies per DOI (include all DOIs, even if they have 0 filtered URLs)
    per_doi_top_urls = {}
    # Ensure all DOIs from doi_to_urls are in doi_to_cleaned_urls (even if empty)
    for doi in doi_to_urls.keys():
        if doi not in doi_to_cleaned_urls:
            doi_to_cleaned_urls[doi] = []
    
    for doi, cleaned_urls in doi_to_cleaned_urls.items():
        if cleaned_urls:  # DOIs with filtered URLs
            doi_counter = Counter(cleaned_urls)
            per_doi_top_urls[doi] = doi_counter.most_common(top_n)
        else:  # DOIs with no filtered URLs - include with empty list
            per_doi_top_urls[doi] = []
    
    return global_top_urls, per_doi_top_urls, total_dois_processed


def main():
    """Main function to collect and display top filtered links."""
    # Set up paths — logs live at sciconharness/logs/ (matches mcp_client/utils/utils.py _log_dir)
    base_log_dir = Path(__file__).resolve().parent.parent / "logs"
    model_dirs = [
        "claude-sonnet-4-5_tools_filter",
        "gemini-3-pro-preview_tools_filter",
        "gpt-5.1_tools_filter",
    ]
    
    # Get top filtered links
    global_top_urls, per_doi_top_urls, total_dois_processed = get_top_filtered_links(base_log_dir, model_dirs, top_n=18)
    
    # Print global results
    print("\n" + "="*80)
    print("TOP 18 MOST FREQUENTLY OCCURRING FILTERED LINKS (GLOBAL)")
    print("="*80)
    print(f"{'Rank':<6} {'Count':<8} {'URL'}")
    print("-"*80)
    
    for rank, (url, count) in enumerate(global_top_urls, 1):
        print(f"{rank:<6} {count:<8} {url}")
    
    print("\n" + "="*80)
    print(f"Total unique URLs: {len(set(url for url, _ in global_top_urls))}")
    print("="*80)
    
    # Save results alongside this script so it can be committed and reused
    output_file = Path(__file__).resolve().parent / "filtered_links_from_logs.json"
    output_data = {
        "global": {
            "total_unique_urls": len(set(url for url, _ in global_top_urls)),
            "top_18_urls": [
                {"rank": rank, "url": url, "count": count}
                for rank, (url, count) in enumerate(global_top_urls, 1)
            ]
        },
        "per_doi": {
            doi: {
                "total_unique_urls": len(set(url for url, _ in urls)),
                "top_18_urls": [
                    {"rank": rank, "url": url, "count": count}
                    for rank, (url, count) in enumerate(urls, 1)
                ]
            }
            for doi, urls in per_doi_top_urls.items()
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Pass this file to --filtered-links-json when running query_batch for Perplexity.")
    print(f"Total DOIs processed: {total_dois_processed}")
    dois_with_urls = sum(1 for urls in per_doi_top_urls.values() if len(urls) > 0)
    print(f"DOIs in output: {len(per_doi_top_urls)}")
    print(f"DOIs with filtered URLs (after filtering Cochrane): {dois_with_urls}")
    print(f"DOIs with 0 filtered URLs: {len(per_doi_top_urls) - dois_with_urls}")


if __name__ == "__main__":
    main()
