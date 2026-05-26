#!/usr/bin/env python3
"""
Runnable script to query SerpAPI's Google AI Mode API.

Prerequisites:
    - Set SERPAPI_API_KEY in your .env file or as an environment variable
    - Install required packages: serpapi, python-dotenv

Usage:
    python query_google_ai_mode.py [OPTIONS] [question...]
    python query_google_ai_mode.py --batch [OPTIONS]

    Single:   query_google_ai_mode.py "Your question?"
    Interactive:  query_google_ai_mode.py
    Batch:   query_google_ai_mode.py --batch -q questions.json -o ./results
    Use -q/--questions-file for the DOI->question JSON and -o/--output-dir for where to save results.
"""

import argparse
import sys
import os
import json
import re
import time
from typing import Optional, Dict, Any
from datetime import datetime

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Default paths; override with -q and -o
_defaults_base = os.path.join(parent_dir, "experiments", "main_experiment", "data", "querying")
DEFAULT_QUESTIONS_FILE = os.path.join(_defaults_base, "doi_to_question.json")
DEFAULT_OUTPUT_DIR = os.path.join(_defaults_base, "serpapi_ai_mode_results")

from dotenv import load_dotenv
from serpapi import GoogleSearch

# Load environment variables from .env file
load_dotenv()

# Benchmark query suffix: requests a synthesized, evidence-focused paragraph (used for N=268 benchmark).
BENCHMARK_QUERY_SUFFIX = (
    " Synthesize a paragraph-long conclusion using the highest-quality and most up-to-date"
    "scientific evidence available, and explicitly discuss the strengths, "
    "limitations, uncertainty, and contradictions across the body of evidence."
    "Wrap the conclusion paragraph in three square brackets."
)


def format_benchmark_query(question: str) -> str:
    """Format a question for benchmark evals: ensure trailing '?' then append synthesis instruction."""
    q = question.strip()
    if not q.endswith("?"):
        q = q + "?"
    return q + BENCHMARK_QUERY_SUFFIX


def extract_ai_mode_text(ai_mode_data: Dict[str, Any]) -> str:
    """
    Extract the main text content from AI Mode response.
    
    Args:
        ai_mode_data: The ai_mode dictionary from SerpAPI response
        
    Returns:
        str: Combined text from all text blocks
    """
    if not ai_mode_data:
        return ""
    
    text_blocks = ai_mode_data.get("text_blocks", [])
    if not text_blocks:
        return ""
    
    text_parts = []
    
    def extract_list_items(list_items, indent_level=0):
        """Recursively extract list items, handling nested lists."""
        items = []
        for item in list_items:
            item_snippet = item.get("snippet", "")
            nested_list = item.get("list", [])
            
            if item_snippet:
                indent = "  " * indent_level
                items.append(f"{indent} * {item_snippet}")
            
            # Handle nested lists
            if nested_list:
                nested_items = extract_list_items(nested_list, indent_level + 1)
                items.extend(nested_items)
        
        return items
    
    for block in text_blocks:
        block_type = block.get("type", "")
        
        if block_type == "paragraph":
            snippet = block.get("snippet", "")
            if snippet:
                text_parts.append(snippet)
        
        elif block_type == "heading":
            snippet = block.get("snippet", "")
            if snippet:
                text_parts.append(f"\n{snippet}")
        
        elif block_type == "expandable":
            title = block.get("title", "")
            inner_blocks = block.get("text_blocks", [])
            
            if title:
                text_parts.append(f"\n{title}:")
            
            for inner_block in inner_blocks:
                inner_type = inner_block.get("type", "")
                if inner_type == "paragraph":
                    snippet = inner_block.get("snippet", "")
                    if snippet:
                        text_parts.append(snippet)
                elif inner_type == "list":
                    list_items = inner_block.get("list", [])
                    extracted_items = extract_list_items(list_items, indent_level=1)
                    text_parts.extend(extracted_items)
                elif inner_type == "comparison":
                    # For comparison blocks, we can extract feature comparisons
                    product_labels = inner_block.get("product_labels", [])
                    comparison = inner_block.get("comparison", [])
                    if product_labels and comparison:
                        text_parts.append(f"\nComparison: {', '.join(product_labels)}")
                        for comp_item in comparison:
                            feature = comp_item.get("feature", "")
                            values = comp_item.get("values", [])
                            if feature and values:
                                text_parts.append(f"  {feature}: {', '.join(str(v) for v in values)}")
        
        elif block_type == "list":
            list_items = block.get("list", [])
            extracted_items = extract_list_items(list_items, indent_level=0)
            text_parts.extend(extracted_items)
        
        elif block_type == "code_block":
            language = block.get("language", "")
            code = block.get("code", "")
            if code:
                lang_label = f" ({language})" if language else ""
                text_parts.append(f"\nCode{lang_label}:\n{code}")
        
        elif block_type == "table":
            table = block.get("table", [])
            if table:
                text_parts.append("\nTable:")
                for row in table:
                    if isinstance(row, list):
                        text_parts.append(" | ".join(str(cell) for cell in row))
    
    return "\n\n".join(text_parts)


def query_serpapi_ai_mode(
    question: str,
    hl: str = "en",
    no_cache: bool = True,
    doi: Optional[str] = None,
    max_retries: int = 5,
    retry_delay: int = 20
) -> dict:
    """
    Query SerpAPI's Google AI Mode API with a question.
    
    This is simpler than AI Overview - it requires only a single API call.
    
    Args:
        question: The research question to ask
        hl: Language code (default: "en")
        no_cache: Whether to bypass cache (default: True)
        doi: Optional DOI associated with the question (default: None)
        max_retries: Maximum number of retry attempts (default: 5)
        retry_delay: Base delay in seconds between retries (default: 20)
    
    Returns:
        dict: Query results
    """
    print("=" * 70)
    print("SerpAPI Google AI Mode Query")
    print("=" * 70)
    print(f"Question: {question[:100]}..." if len(question) > 100 else f"Question: {question}")
    print(f"Language: {hl}")
    print("=" * 70)
    print("\nQuerying SerpAPI...\n")
    
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise ValueError("SERPAPI_API_KEY environment variable is not set")
    
    # Single API call for AI Mode
    search_params = {
        "engine": "google_ai_mode",
        "q": question,
        "api_key": api_key,
        "hl": hl,
    }
    
    if no_cache:
        search_params["no_cache"] = "true"
    
    # Retry with exponential backoff for rate limits and other errors
    ai_results = None
    for attempt in range(max_retries):
        try:
            search = GoogleSearch(search_params)
            ai_results = search.get_dict()
            break  # Success, exit retry loop
        except Exception as e:
            error_str = str(e).lower()
            # Check if it's a rate limit error or other retryable error
            is_rate_limit = "rate_limit" in error_str or "rate limit" in error_str or "429" in error_str
            is_retryable = is_rate_limit or "timeout" in error_str or "connection" in error_str or "temporary" in error_str
            
            if is_retryable and attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                error_type = "Rate limit" if is_rate_limit else "Error"
                print(f"{error_type} encountered. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                print(f"Error: {str(e)[:200]}")
                time.sleep(wait_time)
            else:
                if attempt < max_retries - 1:
                    # For non-retryable errors, still retry but with longer delay
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"Error encountered. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                    print(f"Error: {str(e)[:200]}")
                    time.sleep(wait_time)
                else:
                    print(f"Error after {max_retries} attempts: {e}")
                    return {
                        'metadata': {
                            'timestamp': datetime.now().isoformat(),
                            'question': question,
                            'doi': doi,
                            'error': str(e)
                        },
                        'response': {
                            'main_text': "",
                            'text_length': 0
                        }
                    }
    
    if ai_results is None:
        return {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'doi': doi,
                'error': "Failed to get response from API"
            },
            'response': {
                'main_text': "",
                'text_length': 0
            }
        }
    
    # Check for errors in response
    if "error" in ai_results:
        error_msg = ai_results.get("error", "Unknown error")
        print(f"Error from AI Mode API: {error_msg}")
        return {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'doi': doi,
                'error': error_msg
            },
            'response': {
                'main_text': "",
                'text_length': 0
            }
        }
    
    # Extract AI Mode data - text_blocks are directly in the response, not under "ai_mode" key
    text_blocks = ai_results.get("text_blocks", [])
    references = ai_results.get("references", [])
    
    # Check if AI Mode results are available
    if not text_blocks:
        error_msg = "No text_blocks in response"
        print(f"Warning: {error_msg}")
        print("This may mean:")
        print("  - Google didn't generate AI Mode results for this query")
        print("  - The query is too specific or too new")
        print("  - Regional restrictions may apply")
        
        return {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'doi': doi,
                'hl': hl,
                'ai_mode_available': False,
                'error': error_msg
            },
            'response': {
                'main_text': "",
                'text_length': 0
            },
            'full_response_structure': ai_results
        }
    
    # Create a structure similar to ai_mode for compatibility with extract_ai_mode_text
    ai_mode_data = {
        "text_blocks": text_blocks,
        "references": references
    }
    
    main_text = extract_ai_mode_text(ai_mode_data)
    
    # Build results dictionary
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'doi': doi,
            'hl': hl,
            'ai_mode_available': True
        },
        'response': {
            'main_text': main_text,
            'text_length': len(main_text) if main_text else 0
        },
        'full_response_structure': {
            'text_blocks': text_blocks,
            'references': references,
            'quick_results': ai_results.get("quick_results", []),
            'shopping_results': ai_results.get("shopping_results", []),
            'local_results': ai_results.get("local_results", []),
            'inline_videos': ai_results.get("inline_videos", []),
            'search_metadata': ai_results.get("search_metadata", {}),
            'search_parameters': ai_results.get("search_parameters", {})
        }
    }
    
    print("\n" + "=" * 70)
    print("RESPONSE")
    print("=" * 70)
    if main_text:
        print(main_text)
    else:
        print("(No text content extracted)")
    print("=" * 70)
    
    # Print references if available
    if references:
        print(f"\nReferences ({len(references)}):")
        for i, ref in enumerate(references[:5], 1):  # Show first 5
            title = ref.get("title", "No title")
            link = ref.get("link", "")
            print(f"  {i}. {title}")
            if link:
                print(f"     {link}")
        if len(references) > 5:
            print(f"  ... and {len(references) - 5} more")
    
    return results


def get_output_filename(doi: str, question: Optional[str] = None) -> str:
    """
    Generate output filename based on DOI.
    
    Args:
        doi: DOI string (required)
        question: Question string (optional, for logging only - not used in filename)
    
    Returns:
        Filename string using DOI
    """
    if not doi:
        raise ValueError("DOI is required for filename generation but was not provided")
    
    safe_doi = doi.replace('/', '_').replace('.', '_')
    filename = f"serpapi_ai_mode_{safe_doi}.json"
    return filename


def result_exists(output_dir: str, filename: str) -> bool:
    """Check if a result file already exists."""
    output_path = os.path.join(output_dir, filename)
    return os.path.exists(output_path)


def process_single_question(
    doi: str,
    question: str,
    output_dir: str,
    max_retries: int = 5,
    retry_delay: int = 20
) -> bool:
    """
    Process a single question and save results.
    
    Args:
        doi: DOI associated with the question
        question: The question to process
        output_dir: Directory to save results
        max_retries: Maximum number of retry attempts
        retry_delay: Base delay in seconds between retries
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        results = query_serpapi_ai_mode(
            question=question,
            doi=doi,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        
        if results is None:
            print("❌ Failed to get response")
            return False
        
        # Generate filename
        filename = get_output_filename(doi, question)
        output_path = os.path.join(output_dir, filename)
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Results saved to: {filename}")
        return True
        
    except Exception as e:
        print(f"❌ Error processing question: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_batch_questions(
    questions_file: str,
    output_dir: str,
    max_retries: int = 5,
    retry_delay: int = 20,
) -> None:
    """
    Process all questions from a DOI->question JSON file.
    Always uses the benchmark query format (question + synthesis instruction). Skips already processed, retries on failure.
    """
    print("=" * 70)
    print("Batch Processing: SerpAPI Google AI Mode")
    print("=" * 70)
    
    # Load questions
    if not os.path.exists(questions_file):
        print(f"Error: Questions file not found: {questions_file}")
        sys.exit(1)
    
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions_dict = json.load(f)
    
    print(f"Loaded {len(questions_dict)} questions from {questions_file}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check which ones are already processed (questions file must be DOI -> question)
    processed = set()
    for doi, question in questions_dict.items():
        if not doi:
            print(f"❌ Error: Entry without DOI in questions file: {question[:100]}...")
            continue
        filename = get_output_filename(doi, question)
        if result_exists(output_dir, filename):
            processed.add(doi)
    
    print(f"Already processed: {len(processed)}")
    print(f"Remaining to process: {len(questions_dict) - len(processed)}")
    print("=" * 70)
    
    # Process each question
    success_count = 0
    fail_count = 0
    skip_count = len(processed)
    total_to_process = len(questions_dict) - skip_count
    current_index = 0
    
    for doi, question in questions_dict.items():
        if doi in processed:
            continue
        if not doi:
            print(f"❌ Error: Entry without DOI in questions file: {question[:100]}...")
            fail_count += 1
            continue
        
        current_index += 1
        print(f"\n[{current_index}/{total_to_process}] Processing: {doi}")
        query_text = format_benchmark_query(question)

        try:
            success = process_single_question(
                doi=doi,
                question=query_text,
                output_dir=output_dir,
                max_retries=max_retries,
                retry_delay=retry_delay
            )
            
            if success:
                success_count += 1
            else:
                fail_count += 1
                
        except KeyboardInterrupt:
            print(f"\n\n⚠  Batch processing interrupted by user.")
            print(f"Progress: {success_count} succeeded, {fail_count} failed, {skip_count} skipped")
            print(f"Remaining: {len(questions_dict) - success_count - fail_count - skip_count}")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Unexpected error processing {doi}: {e}")
            fail_count += 1
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
    print("=" * 70)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Query SerpAPI Google AI Mode with research questions (single or batch).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single question (interactive):  python query_google_ai_mode.py
  Single question (CLI):           python query_google_ai_mode.py "Your question here?"
  Batch (default N=268 file):      python query_google_ai_mode.py --batch
  Batch (custom file & output):    python query_google_ai_mode.py --batch -q my_questions.json -o ./out
        """,
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Research question (single-query mode). If omitted, runs interactive prompt.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: process all DOI->question pairs from --questions-file.",
    )
    parser.add_argument(
        "-q",
        "--questions-file",
        default=DEFAULT_QUESTIONS_FILE,
        help="Path to JSON file mapping DOI -> question (default: %(default)s).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for result JSON files (single and batch). Default: %(default)s",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries per API call (default: 5).",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=20,
        help="Base delay in seconds between retries (default: 20).",
    )
    parser.add_argument(
        "--hl",
        default="en",
        help="Language code for the API (default: en).",
    )
    return parser.parse_args()


def main():
    """Main function to run the script."""
    args = parse_args()

    if not os.getenv("SERPAPI_API_KEY"):
        print("Error: SERPAPI_API_KEY not found in environment variables.")
        print("Please set it in your .env file or export it:")
        print("  export SERPAPI_API_KEY='your-api-key-here'")
        sys.exit(1)

    if args.batch:
        process_batch_questions(
            questions_file=args.questions_file,
            output_dir=args.output_dir,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )
        return

    # Single-query or interactive mode
    if args.question:
        question = " ".join(args.question).strip()
    else:
        print("SerpAPI Google AI Mode Query Tool")
        print("=" * 70)
        print("\nEnter your research question (or press Ctrl+C to exit):\n")
        question = input("Question: ").strip()
        if not question:
            print("Error: No question provided.")
            sys.exit(1)

    doi = None
    if os.path.exists(args.questions_file):
        try:
            with open(args.questions_file, "r", encoding="utf-8") as f:
                questions_dict = json.load(f)
            for key, value in questions_dict.items():
                if value == question:
                    doi = key
                    break
        except Exception as e:
            print(f"Warning: Could not load questions file for DOI lookup: {e}")

    try:
        results = query_serpapi_ai_mode(
            question=question,
            doi=doi,
            hl=args.hl,
        )
        if doi:
            filename = f"serpapi_ai_mode_{doi.replace('/', '_').replace('.', '_')}.json"
        else:
            safe_question = re.sub(r"[^\w\s-]", "", question)[:50]
            safe_question = re.sub(r"[-\s]+", "_", safe_question)
            filename = f"serpapi_ai_mode_{safe_question}.json"
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to: {output_path}")
        print(f"  File size: {os.path.getsize(output_path) / 1024:.2f} KB")
    except KeyboardInterrupt:
        print("\n\nQuery cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
