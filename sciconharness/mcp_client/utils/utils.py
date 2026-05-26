"""Utility functions for token counting and logging."""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import tiktoken

logger = logging.getLogger(__name__)

# Set up logging for mcp_client package
_log_dir = Path(__file__).parent.parent.parent / "logs"
_log_dir.mkdir(exist_ok=True)

# Configure root mcp_client logger - all child loggers will inherit this
_mcp_logger = logging.getLogger("mcp_client")
_log_handler = None  # Store handler reference for cleanup

# Global variable to allow overriding the base log directory
_custom_log_dir: Optional[Path] = None

# Track URLs that have been filtered (persists across search results for same query)
_filtered_urls: set = set()


def set_custom_log_dir(base_log_dir: Optional[Path]):
    """
    Set a custom base log directory for all subsequent logging.
    
    Args:
        base_log_dir: Custom base log directory path, or None to use default
    """
    global _custom_log_dir
    _custom_log_dir = base_log_dir
    if base_log_dir:
        base_log_dir.mkdir(parents=True, exist_ok=True)


def setup_logging_for_run(model_name: str, doi: Optional[str] = None):
    """
    Set up logging with run/model directory structure.
    
    Args:
        model_name: Name of the model directory (e.g., "gpt-5.1_tools_filter", "claude-3-5-sonnet_tools")
        doi: Optional DOI. If None, generates one from timestamp.
    
    Returns:
        Path to the log directory
    """
    global _mcp_logger, _log_handler
    
    # Sanitize DOI for filesystem (replace slashes and other unsafe chars with underscores)
    if doi:
        doi_safe = re.sub(r'[^\w\-_\.]', '_', doi)
        if len(doi_safe) > 200:
            doi_safe = doi_safe[:200]
    else:
        # Generate timestamp-based directory name when DOI is not provided
        doi_safe = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    
    # Use custom log directory if set, otherwise use default
    base_log_dir = _custom_log_dir if _custom_log_dir else _log_dir
    
    # Create log directory structure: {base_log_dir}/{model_name}/{doi}/
    log_dir = base_log_dir / model_name / doi_safe
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove existing handlers from parent logger
    if _log_handler:
        _mcp_logger.removeHandler(_log_handler)
        _log_handler.close()
        _log_handler = None
    
    # Remove handlers from all existing child loggers in mcp_client and mcp_server.apis packages
    # Capture mcp_client and mcp_server.apis loggers (e.g. jina_apis for summarized content)
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if "mcp_client" in name or "mcp_server.apis" in name:
            child_logger = logging.getLogger(name)
            for handler in child_logger.handlers[:]:
                child_logger.removeHandler(handler)
                if hasattr(handler, 'close'):
                    handler.close()
    
    # Create new log file
    log_file = log_dir / f"mcp_client.log"
    _log_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    _log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    # Ensure immediate flushing
    _log_handler.setLevel(logging.INFO)
    
    # Create a custom filter to ensure all mcp_client and mcp_server.apis loggers use this handler
    class MCPClientFilter(logging.Filter):
        """Filter that ensures mcp_client and mcp_server.apis loggers get the handler."""
        def filter(self, record):
            # This filter doesn't filter anything, but we use it to attach handler to new loggers
            # Capture mcp_client and mcp_server.apis loggers (e.g. jina_apis for summarized content)
            if "mcp_client" in record.name or "mcp_server.apis" in record.name:
                logger_instance = logging.getLogger(record.name)
                if _log_handler not in logger_instance.handlers:
                    logger_instance.addHandler(_log_handler)
                    logger_instance.setLevel(logging.INFO)
                    logger_instance.propagate = False
            return True
    
    mcp_filter = MCPClientFilter()
    _log_handler.addFilter(mcp_filter)
    
    # Configure parent logger
    _mcp_logger.setLevel(logging.INFO)
    _mcp_logger.addHandler(_log_handler)
    _mcp_logger.propagate = False
    
    # Configure all existing child loggers - do this AFTER monkey patch so new gets work
    # But also reconfigure existing ones - IMPORTANT: getLogger returns the SAME instance
    # so we need to reconfigure the existing instances
    # Capture mcp_client and mcp_server.apis loggers (e.g. jina_apis for summarized content)
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if "mcp_client" in name or "mcp_server.apis" in name:
            child_logger = logging.getLogger(name)  # This returns the existing instance
            child_logger.setLevel(logging.INFO)
            # Remove ALL existing handlers first (including old file handlers)
            for handler in list(child_logger.handlers):
                child_logger.removeHandler(handler)
                if hasattr(handler, 'close') and handler != _log_handler:
                    try:
                        handler.close()
                    except:
                        pass
            # Add our new handler
            child_logger.addHandler(_log_handler)
            child_logger.propagate = False
            # Force the logger to be enabled
            child_logger.disabled = False
    
    # Override logging.getLogger to automatically configure mcp_client loggers
    original_getLogger = logging.getLogger
    
    def getLogger_with_mcp_config(name=None):
        logger_instance = original_getLogger(name)
        # Capture mcp_client and mcp_server.apis loggers
        if name and ("mcp_client" in name or "mcp_server.apis" in name) and _log_handler:
            if _log_handler not in logger_instance.handlers:
                logger_instance.setLevel(logging.INFO)
                logger_instance.addHandler(_log_handler)
                logger_instance.propagate = False
        return logger_instance
    
    # Monkey patch logging.getLogger (but only if not already patched)
    if not hasattr(logging, '_mcp_patched'):
        logging._original_getLogger = original_getLogger
        logging.getLogger = getLogger_with_mcp_config
        logging._mcp_patched = True
    
    # Test logging
    _mcp_logger.info("Logging configured for model: %s, doi: %s", model_name, doi)
    _mcp_logger.info("Log file: %s", log_file)
    logger.info("Child logger test - logging configured")
    
    # Force flush to ensure it's written
    _log_handler.flush()
    
    return log_dir


def get_log_file_path() -> Optional[Path]:
    """
    Get the current log file path if logging is configured.
    
    Returns:
        Path to the log file, or None if logging is not configured
    """
    global _log_handler
    if _log_handler and hasattr(_log_handler, 'baseFilename'):
        return Path(_log_handler.baseFilename)
    return None


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    if not text:
        return 0
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def count_message_content_tokens(content: Any) -> int:
    """Count tokens in message content (handles string or list format)."""
    if isinstance(content, str):
        return count_tokens(content)
    elif isinstance(content, list):
        total = 0
        for item in content:
            if isinstance(item, dict) and "text" in item:
                total += count_tokens(item["text"])
        return total
    return 0


def count_initial_input_tokens(messages: List[Dict[str, Any]]) -> int:
    """Count tokens in initial messages (conversation history + user query)."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        total += count_message_content_tokens(content)
    return total


def count_tool_definitions_tokens(tools: List[Dict[str, Any]]) -> int:
    """Count tokens in tool definitions."""
    if not tools:
        return 0
    tools_json = json.dumps(tools, ensure_ascii=False)
    return count_tokens(tools_json)


def count_tool_calls_tokens(tool_calls: List[Dict[str, Any]]) -> int:
    """Count tokens in tool calls."""
    total = 0
    for tool_call in tool_calls:
        # Create a copy without bytes objects for JSON serialization
        tool_call_copy = {k: v for k, v in tool_call.items() if k != "_thought_signature_bytes"}
        # Also remove extra_content if it contains bytes
        if "extra_content" in tool_call_copy:
            extra = tool_call_copy["extra_content"]
            if isinstance(extra, dict) and "google" in extra:
                google_extra = extra["google"]
                if isinstance(google_extra, dict) and "thought_signature" in google_extra:
                    # Check if it's bytes and exclude it
                    if isinstance(google_extra["thought_signature"], bytes):
                        # Remove the bytes thought_signature for JSON serialization
                        google_extra_copy = {k: v for k, v in google_extra.items() if k != "thought_signature"}
                        if google_extra_copy:
                            tool_call_copy["extra_content"] = {"google": google_extra_copy}
                        else:
                            tool_call_copy.pop("extra_content", None)
        tool_call_str = json.dumps(tool_call_copy, ensure_ascii=False)
        total += count_tokens(tool_call_str)
    return total


def log_token_usage(
    input_tokens: int, 
    output_tokens: int, 
    iteration: int, 
    tool_call_count: int,
    initial_message_tokens: Optional[int] = None,
    tool_def_tokens: Optional[int] = None,
    tool_result_tokens: Optional[int] = None,
    reasoning_tokens: Optional[int] = None,
    tool_call_tokens: Optional[int] = None
):
    """Log token usage summary with optional breakdown."""
    total_tokens = input_tokens + output_tokens
    logger.info("=" * 60)
    logger.info("TOKEN USAGE SUMMARY")
    logger.info("=" * 60)
    logger.info("Input tokens: %d", input_tokens)
    logger.info("Output tokens: %d", output_tokens)
    logger.info("Total tokens: %d", total_tokens)
    logger.info("Iterations: %d", iteration)
    logger.info("Tool calls: %d", tool_call_count)
    
    # Show breakdown if available
    if any([initial_message_tokens is not None, tool_def_tokens is not None, 
            tool_result_tokens is not None, reasoning_tokens is not None, 
            tool_call_tokens is not None]):
        logger.info("-" * 60)
        logger.info("INPUT TOKEN BREAKDOWN:")
        if initial_message_tokens is not None:
            logger.info("  Initial messages: %d (%.1f%%)", 
                       initial_message_tokens, 
                       100 * initial_message_tokens / input_tokens if input_tokens > 0 else 0)
        if tool_def_tokens is not None:
            logger.info("  Tool definitions: %d (%.1f%%) [counted %d times]", 
                       tool_def_tokens,
                       100 * tool_def_tokens / input_tokens if input_tokens > 0 else 0,
                       iteration if tool_def_tokens > 0 else 0)
        if tool_result_tokens is not None:
            logger.info("  Tool results: %d (%.1f%%)", 
                       tool_result_tokens,
                       100 * tool_result_tokens / input_tokens if input_tokens > 0 else 0)
        if reasoning_tokens is not None:
            logger.info("  Reasoning summaries: %d (%.1f%%)", 
                       reasoning_tokens,
                       100 * reasoning_tokens / input_tokens if input_tokens > 0 else 0)
        if tool_call_tokens is not None:
            logger.info("  Tool calls: %d (%.1f%%)", 
                       tool_call_tokens,
                       100 * tool_call_tokens / input_tokens if input_tokens > 0 else 0)
        
        # Calculate unaccounted tokens (messages resent each iteration)
        accounted = sum(filter(None, [
            initial_message_tokens, tool_def_tokens, tool_result_tokens, 
            reasoning_tokens, tool_call_tokens
        ]))
        unaccounted = input_tokens - accounted
        if unaccounted > 0:
            logger.info("  Other (message overhead, etc.): %d (%.1f%%)", 
                       unaccounted,
                       100 * unaccounted / input_tokens if input_tokens > 0 else 0)
        
        logger.info("")
    
    logger.info("=" * 60)


def cap_year_parameter(existing_year: Optional[str], cutoff_year: int) -> str:
    """Cap the upper bound of a year parameter to the cutoff year.
    
    Args:
        existing_year: Existing year parameter (e.g., "2008-2024", "2020-", "-2025", "2024")
        cutoff_year: Year to cap at (e.g., 2023)
        
    Returns:
        Modified year parameter with upper bound capped
        
    Examples:
        "2008-2024" with cutoff 2023 -> "2008-2023"
        "2020-" with cutoff 2023 -> "2020-2023"
        "-2025" with cutoff 2023 -> "-2023"
        "2024" with cutoff 2023 -> "-2023" (single year after cutoff becomes before-only)
        "2022" with cutoff 2023 -> "2022" (single year before cutoff stays the same)
    """
    if not existing_year:
        # No existing year parameter, return before-only format
        return f"-{cutoff_year}"
    
    existing_year = existing_year.strip()
    
    # Check if it's a range (contains "-")
    if "-" in existing_year:
        parts = existing_year.split("-", 1)
        start_part = parts[0].strip()
        end_part = parts[1].strip() if len(parts) > 1 else ""
        
        # Case: "2008-2024" or "2008-2024"
        if start_part and end_part:
            try:
                start_year = int(start_part)
                end_year = int(end_part)
                # Cap the end year
                capped_end = min(end_year, cutoff_year)
                return f"{start_year}-{capped_end}"
            except ValueError:
                # If parsing fails, just cap to cutoff
                return f"{start_part}-{cutoff_year}"
        
        # Case: "2020-" (open-ended range)
        elif start_part and not end_part:
            try:
                start_year = int(start_part)
                # Cap to cutoff year
                return f"{start_year}-{cutoff_year}"
            except ValueError:
                # If parsing fails, just use cutoff
                return f"-{cutoff_year}"
        
        # Case: "-2025" (before-only)
        elif not start_part and end_part:
            try:
                end_year = int(end_part)
                # Cap the end year
                capped_end = min(end_year, cutoff_year)
                return f"-{capped_end}"
            except ValueError:
                return f"-{cutoff_year}"
    
    # Case: Single year "2024" or "2022"
    try:
        single_year = int(existing_year)
        if single_year > cutoff_year:
            # Single year after cutoff becomes before-only
            return f"-{cutoff_year}"
        else:
            # Single year before/at cutoff stays the same
            return existing_year
    except ValueError:
        # If parsing fails, just use cutoff
        return f"-{cutoff_year}"


def log_tool_result(tool_name: str, parsed_args: Dict[str, Any], tool_result_str: str):
    """Log detailed results for specific tools (search results, Jina API results, Semantic Scholar snippets).
    
    Args:
        tool_name: Name of the tool that was called
        parsed_args: Parsed arguments that were passed to the tool
        tool_result_str: JSON string representation of the tool result
    """
    if tool_name == "serper_google_webpage_search":
        logger.info("=" * 80)
        logger.info("SEARCH RESULTS (serper_google_webpage_search)")
        logger.info("=" * 80)
        logger.info("Query: %s", parsed_args.get("query", "N/A"))
        logger.info("Full Results:\n%s", tool_result_str)
        logger.info("=" * 80)
    elif tool_name == "jina_fetch_webpage_content":
        logger.info("=" * 80)
        logger.info("JINA API RESULTS (jina_fetch_webpage_content)")
        logger.info("=" * 80)
        logger.info("URL: %s", parsed_args.get("webpage_url", "N/A"))
        # Parse JSON to get content length
        try:
            import json
            result_dict = json.loads(tool_result_str)
            content = result_dict.get("content", "")
            content_length = len(content) if content else 0
            logger.info("Content length: %s characters", f"{content_length:,}")
            if content_length > 0:
                logger.info("Content preview (first 500 chars): %s", content[:500])
        except (json.JSONDecodeError, TypeError):
            pass
        logger.info("Full Results:\n%s", tool_result_str)
        logger.info("=" * 80)
    elif tool_name == "semantic_scholar_snippet_search":
        logger.info("=" * 80)
        logger.info("SEMANTIC SCHOLAR SNIPPET RESULTS (semantic_scholar_snippet_search)")
        logger.info("=" * 80)
        logger.info("Query: %s", parsed_args.get("query", "N/A"))
        logger.info("Full Results:\n%s", tool_result_str)
        logger.info("=" * 80)


def log_query_run_summary(
    query: str,
    response_text: str,
    messages: List[Dict[str, Any]],
    iterations: int,
    tool_call_count: int,
    total_tokens: int,
    publication_date: Optional[str] = None,
    result_filter: Optional[Any] = None
):
    """Log summary of query run before final answer is returned.
    
    Args:
        query: The original query string
        response_text: The final response text
        messages: The complete message context (conversation history)
        iterations: Number of iterations/rounds
        tool_call_count: Total number of tool calls made
        total_tokens: Total tokens used
        publication_date: Optional publication date cutoff that was applied
        result_filter: Optional result filter to log filtering summary
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("QUERY RUN SUMMARY")
    logger.info("=" * 80)
    logger.info("Original Query: %s", query)
    if publication_date:
        logger.info("Publication Date Cutoff: %s", publication_date)
    logger.info("Iterations: %d", iterations)
    logger.info("Tool Calls Made: %d", tool_call_count)
    logger.info("Total Tokens: %d", total_tokens)
    logger.info("Response Length: %d characters", len(response_text))
    logger.info("")
    logger.info("COMPLETE MESSAGE CONTEXT:")
    logger.info("-" * 80)
    
    # Log each message in the conversation
    for idx, msg in enumerate(messages, 1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
        msg_type = msg.get("type")
        
        logger.info("")
        logger.info("[Message %d]", idx)
        
        if msg_type == "function_call_output":
            # Tool response message (OpenAI format)
            logger.info("Type: Tool Response")
            logger.info("Tool Call ID: %s", msg.get("call_id", "N/A"))
            # OpenAI format doesn't include tool name in the response message
            # Try to get it from the message if available, otherwise show N/A
            logger.info("Tool Name: %s", msg.get("name", "N/A"))
            # OpenAI format uses "output" field, but check both for compatibility
            tool_content = msg.get("output") or msg.get("content", "")
            if isinstance(tool_content, str):
                if tool_content:
                    try:
                        # Try to pretty-print JSON if it's JSON
                        parsed = json.loads(tool_content)
                        logger.info("Tool Result (JSON):\n%s", json.dumps(parsed, indent=2))
                    except (json.JSONDecodeError, TypeError):
                        logger.info("Tool Result:\n%s", tool_content[:1000] + ("..." if len(tool_content) > 1000 else ""))
                else:
                    logger.info("Tool Result: (empty)")
            else:
                logger.info("Tool Result:\n%s", str(tool_content)[:1000] + ("..." if len(str(tool_content)) > 1000 else ""))
        elif role == "user":
            logger.info("Role: User")
            # Check for Gemini format with parts containing function_response
            parts = msg.get("parts", [])
            if parts and isinstance(parts, list):
                # Gemini format: user message with parts containing function_response
                for part in parts:
                    if isinstance(part, dict) and "function_response" in part:
                        func_resp = part["function_response"]
                        tool_name = func_resp.get("name", "N/A")
                        tool_result = func_resp.get("response", {})
                        logger.info("Tool Result (Name: %s):\n%s", 
                                  tool_name,
                                  json.dumps(tool_result, indent=2)[:2000] + ("..." if len(json.dumps(tool_result, indent=2)) > 2000 else ""))
                    elif isinstance(part, dict) and "text" in part:
                        logger.info("Content: %s", part["text"][:1000] + ("..." if len(part["text"]) > 1000 else ""))
                    else:
                        logger.info("Part: %s", str(part)[:1000] + ("..." if len(str(part)) > 1000 else ""))
            elif isinstance(content, list):
                # Handle structured content (e.g., from Claude tool responses)
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        logger.info("Tool Result (ID: %s, Name: %s):\n%s", 
                                  item.get("tool_use_id", "N/A"),
                                  item.get("name", "N/A"),
                                  str(item.get("content", ""))[:1000] + ("..." if len(str(item.get("content", ""))) > 1000 else ""))
                    else:
                        logger.info("Content: %s", str(item)[:1000] + ("..." if len(str(item)) > 1000 else ""))
            else:
                logger.info("Content: %s", content[:1000] + ("..." if len(content) > 1000 else ""))
        elif role == "assistant":
            logger.info("Role: Assistant")
            if content:
                logger.info("Content: %s", content[:1000] + ("..." if len(content) > 1000 else ""))
            if tool_calls:
                logger.info("Tool Calls (%d):", len(tool_calls))
                for tool_call in tool_calls:
                    # Handle both OpenAI format (function dict) and Gemini format (direct fields)
                    func = tool_call.get("function", {})
                    if func and isinstance(func, dict):
                        # OpenAI format
                        tool_name = func.get("name", "unknown")
                        tool_args = func.get("arguments", "{}")
                    else:
                        # Gemini format (direct fields)
                        tool_name = tool_call.get("name", "unknown")
                        tool_args = tool_call.get("arguments", "{}")
                    tool_id = tool_call.get("id", "unknown")
                    logger.info("  - Tool: %s (ID: %s)", tool_name, tool_id)
                    if isinstance(tool_args, str):
                        try:
                            parsed_args = json.loads(tool_args)
                            logger.info("    Arguments: %s", json.dumps(parsed_args, indent=4))
                        except json.JSONDecodeError:
                            logger.info("    Arguments: %s", tool_args[:500])
                    else:
                        logger.info("    Arguments: %s", json.dumps(tool_args, indent=4))
        elif role == "tool" or role == "function":
            logger.info("Role: Tool/Function")
            # Handle both OpenAI format (name in message) and Gemini format (parts with function_response)
            parts = msg.get("parts", [])
            if parts and isinstance(parts, list):
                # Gemini format: multiple parts with function_response (for parallel tool calls)
                function_responses = [p for p in parts if isinstance(p, dict) and "function_response" in p]
                if function_responses:
                    logger.info("Function Responses (%d):", len(function_responses))
                    for idx, part in enumerate(function_responses):
                        func_resp = part["function_response"]
                        tool_name = func_resp.get("name", "N/A")
                        tool_result = func_resp.get("response", {})
                        logger.info("  [%d] Tool Name: %s", idx + 1, tool_name)
                        if isinstance(tool_result, dict):
                            result_str = json.dumps(tool_result, indent=2)
                            logger.info("      Tool Result (JSON):\n%s", result_str[:2000] + ("..." if len(result_str) > 2000 else ""))
                        elif isinstance(tool_result, str):
                            if tool_result:
                                try:
                                    parsed = json.loads(tool_result)
                                    result_str = json.dumps(parsed, indent=2)
                                    logger.info("      Tool Result (JSON):\n%s", result_str[:2000] + ("..." if len(result_str) > 2000 else ""))
                                except (json.JSONDecodeError, TypeError):
                                    logger.info("      Tool Result:\n%s", tool_result[:1000] + ("..." if len(tool_result) > 1000 else ""))
                            else:
                                logger.info("      Tool Result: (empty)")
                        else:
                            logger.info("      Tool Result:\n%s", str(tool_result)[:1000] + ("..." if len(str(tool_result)) > 1000 else ""))
                else:
                    # No function_response parts, check for other content
                    tool_name = msg.get("name", "N/A")
                    logger.info("Tool Name: %s", tool_name)
                    tool_result = msg.get("content", "")
                    if isinstance(tool_result, str):
                        if tool_result:
                            try:
                                parsed = json.loads(tool_result)
                                logger.info("Tool Result (JSON):\n%s", json.dumps(parsed, indent=2)[:2000] + ("..." if len(json.dumps(parsed, indent=2)) > 2000 else ""))
                            except (json.JSONDecodeError, TypeError):
                                logger.info("Tool Result:\n%s", tool_result[:1000] + ("..." if len(tool_result) > 1000 else ""))
                        else:
                            logger.info("Tool Result: (empty)")
                    else:
                        logger.info("Tool Result:\n%s", str(tool_result)[:1000] + ("..." if len(str(tool_result)) > 1000 else ""))
            else:
                # OpenAI format or other format
                tool_name = msg.get("name", "N/A")
                logger.info("Tool Name: %s", tool_name)
                tool_result = msg.get("content", "")
                if isinstance(tool_result, str):
                    if tool_result:
                        try:
                            parsed = json.loads(tool_result)
                            logger.info("Tool Result (JSON):\n%s", json.dumps(parsed, indent=2)[:2000] + ("..." if len(json.dumps(parsed, indent=2)) > 2000 else ""))
                        except (json.JSONDecodeError, TypeError):
                            logger.info("Tool Result:\n%s", tool_result[:1000] + ("..." if len(tool_result) > 1000 else ""))
                    else:
                        logger.info("Tool Result: (empty)")
                else:
                    logger.info("Tool Result:\n%s", str(tool_result)[:1000] + ("..." if len(str(tool_result)) > 1000 else ""))
        else:
            logger.info("Role: %s", role)
            logger.info("Content: %s", str(msg)[:1000] + ("..." if len(str(msg)) > 1000 else ""))
    
    logger.info("")
    logger.info("-" * 80)
    logger.info("Final Answer (about to be returned):")
    logger.info("-" * 80)
    logger.info("%s", response_text)
    logger.info("-" * 80)
    logger.info("=" * 80)
    logger.info("")
    
    # Log filtering summary if filter is provided
    if result_filter and hasattr(result_filter, 'log_filtering_summary'):
        result_filter.log_filtering_summary()


def track_filtered_urls(urls: List[str]):
    """Track URLs that have been filtered.
    
    Args:
        urls: List of URLs to add to the filtered set
    """
    global _filtered_urls
    _filtered_urls.update(urls)


def get_filtered_urls() -> set:
    """Get the filtered URLs set.
    
    Returns:
        Copy of the filtered URLs set
    """
    return _filtered_urls.copy()


def is_url_filtered(url: str) -> bool:
    """Check if a URL has been filtered.
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL has been filtered, False otherwise
    """
    return url in _filtered_urls


def reset_filtered_urls():
    """Reset the filtered URLs set (call when starting a new query)."""
    global _filtered_urls
    count_before = len(_filtered_urls)
    _filtered_urls.clear()
    if count_before > 0:
        logger.debug(f"Reset {count_before} filtered URLs at start of new query")



