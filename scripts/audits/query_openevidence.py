#!/usr/bin/env python3
"""
Web scraper for OpenEvidence using SeleniumBase.

This script:
1. Opens a browser (non-headless) for manual login to OpenEvidence
2. After login, automatically searches queries, waits, and scrapes results
3. Saves results to JSON files

Prerequisites:
    - seleniumbase package installed
    - Chrome browser (or specify path to Chrome binary)

Usage:
    # Interactive mode (enter queries one by one):
    python query_openevidence.py
    
    # Batch mode (process queries from file):
    python query_openevidence.py --batch [questions_file.json]
    
    # Single query:
    python query_openevidence.py "Your research question here"
"""

import sys
import os
import json
import time
import re
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from seleniumbase import SB
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By


# ============================================================================
# CONFIGURATION - Customize these based on OpenEvidence website structure
# ============================================================================

OPENEVIDENCE_URL = "https://openevidence.com"  # Update if different
WAIT_AFTER_SEARCH = 250  # Wait time in seconds after search (default: 4 minutes)
DELAY_BETWEEN_QUERIES = 15  # Delay in seconds between queries in batch mode
OUTPUT_DIR = os.path.join(current_dir, 'data', 'openevidence_results')
# Directory for HTML files named by DOI (for later parsing). DOI→question mapping: experiments/main_experiment/data/querying/doi_to_question.json
OPENEVIDENCE_HTML_DIR = os.path.join(current_dir, 'openevidence_html')

# Performance settings
USE_UNDETECTED_CHROME = True  # Set to True if you need anti-detection (slower), False for faster performance
DISABLE_IMAGES = False  # Disable images for faster page loads
DISABLE_CSS = False  # Disable CSS for faster page loads (may break layout)

# Chrome binary path (set to None to use system Chrome)
# Default path matches scraper.py configuration
CHROME_BINARY_PATH = None  # Use system Chrome

# Search box selectors (try these in order, add more as needed)
# Updated based on OpenEvidence's actual HTML structure
SEARCH_SELECTORS = [
    'textarea[placeholder="Ask a medical question..."]',  # Most specific - OpenEvidence
    'textarea[aria-label="Ask a medical question"]',  # Also specific - OpenEvidence
    '[data-testid="ask--query-bar"] textarea',  # Using test id - OpenEvidence
    'textarea.MuiInputBase-inputMultiline',  # Material-UI class - OpenEvidence
    'textarea[placeholder*="medical question" i]',  # Partial match
    'textarea[placeholder*="ask" i]',  # Partial match
    'input[type="search"]',  # Generic fallback
    'input[name="q"]',  # Generic fallback
    'input[name="query"]',  # Generic fallback
]

# Submit button selectors (for clicking search button if Enter key doesn't work)
SUBMIT_SELECTORS = [
    'button[type="submit"]',
    'button[class*="search" i]',
    'input[type="submit"]',
    'button[aria-label*="search" i]',
    # Add custom selectors here
]

# "New Conversation" link selectors (click after each search to return to query entry page)
# OpenEvidence: <a href="/"> with inner text "New Conversation" (nav-item-text, title="New Conversation")
NEW_CONVERSATION_SELECTORS_CSS = [
    'a[href="/"]',  # Link to home / new conversation
    'p[title="New Conversation"]',  # Fallback: get parent <a> in code
]
# XPath via raw driver when CSS is insufficient (finds the <a> for "New Conversation")
NEW_CONVERSATION_SELECTORS_XPATH = [
    "//a[@href='/' and contains(., 'New Conversation')]",
    "//p[@title='New Conversation']/ancestor::a[@href='/']",
    "//a[@href='/']//p[contains(@class,'nav-item-text') and contains(text(),'New Conversation')]/ancestor::a",
]

# Main content area selectors (for extracting main response)
MAIN_CONTENT_SELECTORS = [
    'main',
    'div[class*="content" i]',
    'div[class*="result" i]',
    'article',
    'div[class*="response" i]',
    'div[class*="answer" i]',
    # Add custom selectors here
]

# Individual result item selectors (for extracting multiple results)
RESULT_ITEM_SELECTORS = [
    'div[class*="result" i]',
    'div[class*="item" i]',
    'article',
    'li[class*="result" i]',
    'div[class*="card" i]',
    # Add custom selectors here
]

MAX_RESULTS_TO_SCRAPE = 20  # Maximum number of result items to scrape per query

# Benchmark query suffix: requests a synthesized, evidence-focused paragraph (used for N=268 benchmark).
BENCHMARK_QUERY_SUFFIX = (
    " Synthesize a concise paragraph-long conclusion using the highest-quality and most up-to-date"
    " scientific evidence available, and explicitly discuss the strengths, "
    "limitations, uncertainty, and contradictions across the body of evidence. "
    "Wrap the conclusion paragraph in three square brackets."
)


def format_benchmark_query(question: str) -> str:
    """Format a question for benchmark evals: ensure trailing '?' then append synthesis instruction."""
    q = question.strip()
    if not q.endswith("?"):
        q = q + "?"
    return q + BENCHMARK_QUERY_SUFFIX


def ensure_output_directory(output_dir: str):
    """Ensure output directory exists."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def sanitize_filename(text: str, max_length: int = 100) -> str:
    """Sanitize text for use in filename."""
    # Remove special characters
    text = re.sub(r'[^\w\s-]', '', text)
    # Replace spaces with underscores
    text = re.sub(r'[-\s]+', '_', text)
    # Limit length
    if len(text) > max_length:
        text = text[:max_length]
    return text


def get_output_filename(doi: Optional[str] = None, question: Optional[str] = None) -> str:
    """
    Generate output filename based on DOI or question.
    
    Args:
        doi: DOI string (optional)
        question: Question string (optional)
    
    Returns:
        Filename string
    """
    if doi:
        safe_doi = doi.replace('/', '_').replace('.', '_')
        filename = f"openevidence_{safe_doi}.json"
    elif question:
        safe_question = sanitize_filename(question)
        filename = f"openevidence_{safe_question}.json"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"openevidence_{timestamp}.json"
    
    return filename


def result_exists(output_dir: str, filename: str) -> bool:
    """Check if a result file already exists."""
    output_path = os.path.join(output_dir, filename)
    return os.path.exists(output_path)


def wait_for_manual_login(driver, logger=None):
    """
    Wait for user to manually log in to OpenEvidence.
    
    Args:
        driver: SeleniumBase driver instance
        logger: Optional logger instance
    
    Returns:
        bool: True if login appears successful, False otherwise
    """
    if logger:
        logger.info("Waiting for manual login...")
    else:
        print("\n" + "=" * 70)
        print("MANUAL LOGIN REQUIRED")
        print("=" * 70)
        print("Please log in to OpenEvidence in the browser window.")
        print("Once logged in, press ENTER in this terminal to continue...")
        print("=" * 70)
    
    # Wait for user to press Enter
    try:
        input()
    except KeyboardInterrupt:
        print("\nLogin cancelled by user.")
        return False
    
    # Give a moment for any final page loads after login
    time.sleep(2)
    
    if logger:
        logger.info("Proceeding with automated queries...")
    else:
        print("✓ Login confirmed. Starting automated queries...\n")
    
    return True


def search_query(driver, query: str, logger=None) -> bool:
    """
    Search for a query on OpenEvidence.
    
    Args:
        driver: SeleniumBase driver instance
        query: Search query string
        logger: Optional logger instance
    
    Returns:
        bool: True if search was successful, False otherwise
    """
    try:
        if logger:
            logger.info(f"Searching for: {query}")
        else:
            print(f"🔍 Searching for: {query}")
        
        # Find and interact with search box
        # Adjust selectors based on actual OpenEvidence website structure
        # Common patterns:
        # - input[type="search"]
        # - input[name="q"] or input[name="query"]
        # - textarea for search
        # - div with contenteditable
        
        # Try multiple common selectors (from configuration)
        search_element = None
        for selector in SEARCH_SELECTORS:
            try:
                search_element = driver.find_element(selector, timeout=5)
                if search_element:
                    break
            except:
                continue
        
        if not search_element:
            # If no search box found, try to find by text or other methods
            if logger:
                logger.warning("Could not find search box with common selectors. Trying alternative methods...")
            else:
                print("⚠️  Could not find search box automatically. Please check the selectors.")
            return False
        
        # Scroll to element to ensure it's visible
        try:
            driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", search_element)
            time.sleep(0.5)
        except:
            pass
        
        # Click on the search element to focus it (important for Material-UI components)
        try:
            search_element.click()
            time.sleep(0.3)
        except:
            pass
        
        # Clear and enter query
        # For Material-UI textareas, clear existing content
        search_element.clear()
        time.sleep(0.2)
        search_element.send_keys(query)
        time.sleep(0.5)
        
        # Submit search by pressing Enter (OpenEvidence's textarea submits on Enter key)
        # This is the primary method for OpenEvidence
        try:
            search_element.send_keys("\n")  # Press Enter to submit
            time.sleep(0.5)
            if logger:
                logger.info("Search submitted via Enter key")
        except Exception as e:
            # Fallback: Try to find and click submit button (usually not needed for OpenEvidence)
            if logger:
                logger.warning(f"Enter key submission failed, trying submit button: {e}")
            for selector in SUBMIT_SELECTORS:
                try:
                    submit_button = driver.find_element(selector, timeout=2)
                    submit_button.click()
                    if logger:
                        logger.info("Search submitted via submit button")
                    break
                except:
                    continue
        
        if logger:
            logger.info(f"Search submitted. Waiting {WAIT_AFTER_SEARCH} seconds for results...")
        else:
            print(f"⏳ Waiting {WAIT_AFTER_SEARCH} seconds for results to load...")
        
        time.sleep(WAIT_AFTER_SEARCH)
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"Error searching for query '{query}': {e}")
        else:
            print(f"❌ Error searching: {e}")
        return False


def click_new_conversation_link(driver, logger=None) -> bool:
    """
    Click the "New Conversation" link to return to the query entry page (openevidence.com/).
    Uses the sidebar link: <a href="/"> with "New Conversation" text (not /visits).
    
    Args:
        driver: SeleniumBase driver instance
        logger: Optional logger instance
    
    Returns:
        bool: True if link was found and clicked, False otherwise
    """
    try:
        if logger:
            logger.info("Looking for New Conversation link...")
        else:
            print("🔄 Clicking New Conversation link...")
        
        # Prefer raw driver + XPath to reliably find the <a href="/"> that contains "New Conversation"
        new_conversation_link = None
        raw_driver = driver.driver if hasattr(driver, 'driver') else driver
        
        for xpath in NEW_CONVERSATION_SELECTORS_XPATH:
            try:
                new_conversation_link = raw_driver.find_element(By.XPATH, xpath)
                if new_conversation_link and new_conversation_link.is_displayed():
                    break
            except Exception:
                continue
        
        # Fallback: CSS selectors (may match first a[href="/"] which might be correct)
        if not new_conversation_link:
            for selector in NEW_CONVERSATION_SELECTORS_CSS:
                try:
                    elem = driver.find_element(selector, timeout=2)
                    if elem:
                        # If we got the <p>, get the parent <a>
                        if elem.tag_name.lower() == 'p' and elem.get_attribute('title') == 'New Conversation':
                            try:
                                new_conversation_link = elem.find_element(By.XPATH, './ancestor::a[@href="/"]')
                            except Exception:
                                continue
                        else:
                            new_conversation_link = elem
                        if new_conversation_link and new_conversation_link.is_displayed():
                            break
                except Exception:
                    continue
        
        if not new_conversation_link:
            if logger:
                logger.warning("Could not find New Conversation link. Continuing anyway...")
            else:
                print("⚠️  Could not find New Conversation link. Continuing anyway...")
            return False
        
        # Scroll into view and click
        try:
            raw_driver.execute_script(
                "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});",
                new_conversation_link
            )
            time.sleep(0.3)
        except Exception:
            pass
        
        try:
            new_conversation_link.click()
            time.sleep(1.5)  # Wait for navigation to query entry page
            
            if logger:
                logger.info("New Conversation link clicked successfully")
            else:
                print("✓ New Conversation link clicked")
            return True
        except Exception as e:
            if logger:
                logger.warning(f"Could not click New Conversation link: {e}")
            else:
                print(f"⚠️  Could not click New Conversation link: {e}")
            return False
        
    except Exception as e:
        if logger:
            logger.warning(f"Error clicking New Conversation link: {e}")
        else:
            print(f"⚠️  Error clicking New Conversation link: {e}")
        return False


def scrape_results(driver, logger=None) -> Dict[str, Any]:
    """
    Scrape search results from OpenEvidence page.
    
    Args:
        driver: SeleniumBase driver instance
        logger: Optional logger instance
    
    Returns:
        dict: Scraped results with metadata
    """
    try:
        if logger:
            logger.info("Scraping results from page...")
        else:
            print("📄 Scraping results...")
        
        # Get page source
        page_source = driver.get_page_source()
        soup = BeautifulSoup(page_source, 'html.parser')
        
        # Extract results - adjust selectors based on actual OpenEvidence structure
        # Common patterns for search results:
        results = {
            'url': driver.driver.current_url,
            'title': None,
            'main_content': None,
            'results': [],
            'raw_html': page_source,
            'timestamp': datetime.now().isoformat()
        }
        
        # Try to extract page title
        try:
            title_elem = soup.find('title')
            if title_elem:
                results['title'] = title_elem.get_text(strip=True)
        except:
            pass
        
        # Try to find main content area (from configuration)
        main_content = None
        for selector in MAIN_CONTENT_SELECTORS:
            try:
                main_content = soup.select_one(selector)
                if main_content:
                    # Get text content
                    results['main_content'] = main_content.get_text(separator='\n', strip=True)
                    break
            except:
                continue
        
        # Try to find individual result items (from configuration)
        for selector in RESULT_ITEM_SELECTORS:
            try:
                items = soup.select(selector)
                if items:
                    for item in items[:MAX_RESULTS_TO_SCRAPE]:
                        item_text = item.get_text(separator='\n', strip=True)
                        if item_text and len(item_text) > 50:  # Filter out very short items
                            results['results'].append({
                                'text': item_text,
                                'html': str(item)
                            })
                    if results['results']:
                        break
            except:
                continue
        
        # If no structured results found, extract all text from body
        if not results['main_content'] and not results['results']:
            try:
                body = soup.find('body')
                if body:
                    results['main_content'] = body.get_text(separator='\n', strip=True)
            except:
                pass
        
        if logger:
            logger.info(f"Scraped {len(results['results'])} result items")
        else:
            print(f"✓ Scraped {len(results['results'])} result items")
        
        return results
        
    except Exception as e:
        if logger:
            logger.error(f"Error scraping results: {e}")
        else:
            print(f"❌ Error scraping results: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def process_single_query(
    driver,
    query: str,
    doi: Optional[str] = None,
    output_dir: str = OUTPUT_DIR,
    logger=None
) -> bool:
    """
    Process a single query: search, wait, and scrape.
    
    Args:
        driver: SeleniumBase driver instance
        query: Search query string
        doi: Optional DOI associated with the query
        output_dir: Directory to save results
        logger: Optional logger instance
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Format query for benchmark (suffix + trailing ?)
        formatted_query = format_benchmark_query(query)
        
        # Search for the formatted query
        if not search_query(driver, formatted_query, logger):
            return False
        
        # Scrape results
        results = scrape_results(driver, logger)
        
        # If scraping failed, scrape_results returns dict with 'error' and no 'raw_html'
        if 'error' in results:
            if logger:
                logger.error(f"Scraping failed for query; saving error payload only: {results['error']}")
            else:
                print(f"⚠️  Scraping failed (saving error to JSON): {results['error']}")
        
        # Add metadata (original question + formatted query sent)
        results['metadata'] = {
            'query': query,
            'formatted_query': formatted_query,
            'doi': doi,
            'timestamp': datetime.now().isoformat(),
            'url': driver.driver.current_url
        }
        
        # Generate filenames
        json_filename = get_output_filename(doi=doi, question=query)
        json_path = os.path.join(output_dir, json_filename)
        
        # Save JSON results (includes raw_html when present, or error payload)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save HTML only in openevidence_html as {DOI}.html when scrape succeeded (for parsing; DOI→question in doi_to_question.json)
        if 'raw_html' in results and results['raw_html'] and doi and 'error' not in results:
            os.makedirs(OPENEVIDENCE_HTML_DIR, exist_ok=True)
            doi_safe = doi.replace('/', '_')  # e.g. 10.1002_14651858.CD000028.pub4
            doi_html_path = os.path.join(OPENEVIDENCE_HTML_DIR, f"{doi_safe}.html")
            with open(doi_html_path, 'w', encoding='utf-8') as f:
                f.write(results['raw_html'])
            if logger:
                logger.info(f"HTML saved to openevidence_html: {doi_safe}.html")
            else:
                print(f"✓ HTML saved to openevidence_html: {doi_safe}.html")
        
        if logger:
            logger.info(f"Results saved to: {json_filename}")
        else:
            print(f"✓ Results saved to: {json_filename}")
        
        # Click "New Conversation" link to return to query entry page
        click_new_conversation_link(driver, logger)
        
        # Report failure if scraping returned an error (so batch mode counts it)
        return 'error' not in results
        
    except Exception as e:
        if logger:
            logger.error(f"Error processing query '{query}': {e}")
        else:
            print(f"❌ Error processing query: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_batch_queries(
    questions_file: str,
    output_dir: str = OUTPUT_DIR,
    chrome_binary_path: Optional[str] = None,
    headless: bool = False
) -> None:
    """
    Process all queries from a questions file.
    
    Args:
        questions_file: Path to JSON file with questions (DOI -> question mapping)
        output_dir: Directory to save results
        chrome_binary_path: Optional path to Chrome binary
        headless: Whether to run browser in headless mode (should be False for manual login)
    """
    print("=" * 70)
    print("Batch Processing: OpenEvidence Web Scraper")
    print("=" * 70)
    
    # Load questions
    if not os.path.exists(questions_file):
        print(f"Error: Questions file not found: {questions_file}")
        sys.exit(1)
    
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions_dict = json.load(f)
    
    print(f"Loaded {len(questions_dict)} questions from {questions_file}")
    
    # Ensure output directory exists
    ensure_output_directory(output_dir)
    
    # Check which ones are already processed
    processed = set()
    for doi, question in questions_dict.items():
        if not doi:
            continue
        filename = get_output_filename(doi=doi, question=question)
        if result_exists(output_dir, filename):
            processed.add(doi)
    
    print(f"Already processed: {len(processed)}")
    print(f"Remaining to process: {len(questions_dict) - len(processed)}")
    print("=" * 70)
    
    # Set up Chrome binary path (from configuration)
    if chrome_binary_path is None:
        chrome_binary_path = CHROME_BINARY_PATH
        # Check if it exists, if not, use system Chrome
        if chrome_binary_path and not os.path.exists(chrome_binary_path):
            chrome_binary_path = None
    
    # Initialize browser (non-headless for manual login)
    print("\nOpening browser for OpenEvidence...")
    print("You will need to log in manually, then press ENTER to continue.\n")
    
    # Initialize browser with performance optimizations
    sb_kwargs = {
        'uc': USE_UNDETECTED_CHROME,  # Only use if needed (slower but avoids detection)
        'headless': headless,
        'browser': 'chrome'
    }
    if chrome_binary_path:
        sb_kwargs['binary_location'] = chrome_binary_path
    
    print(f"Initializing browser (headless={headless})...")
    print(f"Browser args: {sb_kwargs}")
    try:
        with SB(**sb_kwargs) as driver:
            print("✓ Browser opened successfully!")
            
            # Navigate to OpenEvidence
            print(f"Navigating to {OPENEVIDENCE_URL}...")
            driver.get(OPENEVIDENCE_URL)
            time.sleep(2)
            print("✓ Page loaded")
            
            # Wait for manual login
            if not wait_for_manual_login(driver):
                print("Login cancelled. Exiting.")
                return
            
            # Process each question
            success_count = 0
            fail_count = 0
            skip_count = len(processed)
            total_to_process = len(questions_dict) - skip_count
            current_index = 0
            failed_dois = []  # Track DOIs that failed for retry or inspection
            
            for doi, question in questions_dict.items():
                if doi in processed:
                    continue
                
                if not doi:
                    print(f"❌ Error: Entry found without DOI: {question[:100]}...")
                    fail_count += 1
                    continue
                
                current_index += 1
                print(f"\n[{current_index}/{total_to_process}] Processing: {doi}")
                print(f"Question: {question[:100]}..." if len(question) > 100 else f"Question: {question}")
                
                try:
                    success = process_single_query(
                        driver=driver,
                        query=question,
                        doi=doi,
                        output_dir=output_dir
                    )
                    
                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                        failed_dois.append(doi)
                    
                    # Add delay between queries to avoid rate limiting
                    if current_index < total_to_process:
                        print(f"Waiting {DELAY_BETWEEN_QUERIES} seconds before next query...")
                        time.sleep(DELAY_BETWEEN_QUERIES)
                        
                except KeyboardInterrupt:
                    print(f"\n\n⚠️  Batch processing interrupted by user.")
                    print(f"Progress: {success_count} succeeded, {fail_count} failed, {skip_count} skipped")
                    print(f"Remaining: {len(questions_dict) - success_count - fail_count - skip_count}")
                    if failed_dois:
                        print(f"Failed DOIs: {failed_dois}")
                    break
                except Exception as e:
                    print(f"❌ Unexpected error processing {doi}: {e}")
                    fail_count += 1
                    failed_dois.append(doi)
                    import traceback
                    traceback.print_exc()
            
            # Summary
            print("\n" + "=" * 70)
            print("Batch Processing Complete")
            print("=" * 70)
            print(f"Total questions: {len(questions_dict)}")
            print(f"✓ Succeeded: {success_count}")
            print(f"❌ Failed: {fail_count}")
            print(f"⏭  Skipped (already processed): {skip_count}")
            if failed_dois:
                print(f"Failed DOIs ({len(failed_dois)}): {failed_dois}")
                failed_path = os.path.join(output_dir, 'openevidence_failed_dois.json')
                with open(failed_path, 'w', encoding='utf-8') as f:
                    json.dump(failed_dois, f, indent=2)
                print(f"Failed DOIs saved to: {failed_path}")
            print("=" * 70)
        
    except Exception as e:
        print(f"❌ Error initializing browser: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run the script."""
    # Check for batch mode
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        # Batch processing mode
        questions_file = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
            parent_dir, 'data', 'sampled_100_questions.json'
        )
        
        process_batch_queries(
            questions_file=questions_file,
            output_dir=OUTPUT_DIR,
            headless=False  # Must be False for manual login
        )
        return
    
    # Ensure output directory exists
    ensure_output_directory(OUTPUT_DIR)
    
    # Set up Chrome binary path (from configuration)
    chrome_binary_path = CHROME_BINARY_PATH
    if chrome_binary_path and not os.path.exists(chrome_binary_path):
        chrome_binary_path = None
    
    # Check if query is provided as command line argument
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        # Interactive mode
        print("OpenEvidence Web Scraper")
        print("=" * 70)
        print("\nEnter your research question (or press Ctrl+C to exit):\n")
        query = input("Question: ").strip()
        
        if not query:
            print("Error: No question provided.")
            sys.exit(1)
    
    # Try to find DOI from questions file
    doi = None
    questions_file = os.path.join(parent_dir, 'data', 'sampled_100_questions.json')
    if os.path.exists(questions_file):
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions_dict = json.load(f)
            # Reverse lookup: find DOI by question
            for key, value in questions_dict.items():
                if value == query:
                    doi = key
                    break
        except Exception as e:
            print(f"Warning: Could not load questions file to find DOI: {e}")
    
    # Initialize browser (non-headless for manual login)
    print("\nOpening browser for OpenEvidence...")
    print("You will need to log in manually, then press ENTER to continue.\n")
    
    # Initialize browser (non-headless for manual login)
    sb_kwargs = {
        'uc': USE_UNDETECTED_CHROME,  # Only use if needed (slower but avoids detection)
        'headless': False,
        'browser': 'chrome'
    }
    if chrome_binary_path:
        sb_kwargs['binary_location'] = chrome_binary_path
    
    print(f"Initializing browser with headless=False...")
    print(f"Browser args: {sb_kwargs}")
    try:
        with SB(**sb_kwargs) as driver:
            print("✓ Browser opened successfully!")
            
            # Navigate to OpenEvidence
            print(f"Navigating to {OPENEVIDENCE_URL}...")
            driver.get(OPENEVIDENCE_URL)
            time.sleep(2)
            print("✓ Page loaded")
            
            # Wait for manual login
            if not wait_for_manual_login(driver):
                print("Login cancelled. Exiting.")
                return
            
            # Process the query
            try:
                success = process_single_query(
                    driver=driver,
                    query=query,
                    doi=doi,
                    output_dir=OUTPUT_DIR
                )
                
                if success:
                    print("\n✓ Query processed successfully!")
                else:
                    print("\n❌ Failed to process query")
                    
            except KeyboardInterrupt:
                print("\n\nQuery cancelled by user.")
                sys.exit(0)
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error initializing browser: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

