"""Common functionality shared between remote MCP servers."""

import json
import logging
import os
import re
import sys
import uuid
from contextvars import ContextVar, copy_context
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

import dotenv
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class ImmediateFlushFileHandler(logging.FileHandler):
    """File handler that flushes after every log record to ensure immediate writes."""
    
    def emit(self, record):
        """Emit a record and immediately flush."""
        super().emit(record)
        self.flush()

# Global filter configuration - set by /configure endpoint, used by all requests
_global_filter_config: Dict[str, Optional[str]] = {
    'source_title': None,
    'publication_date': None
}

# Lock for thread-safe updates to global config
import threading
_filter_config_lock = threading.Lock()

# Track URLs returned from search results (no longer per-session, just global)
_allowed_fetch_urls: set = set()

# Track URLs that have been filtered (persists across search results for same configuration)
_filtered_urls: set = set()


def get_sciconharness_dir() -> Path:
    """Get the sciconharness package directory path."""
    current_dir = Path(__file__).resolve().parent
    return current_dir.parent


def get_cochrane_titles_file() -> Path:
    """Get the path to the Cochrane titles JSON file."""
    # Look for cochrane_titles.json in data/filter_data/ relative to project root
    current_dir = Path(__file__).resolve().parent
    # current_dir is: .../SciConBench/sciconharness/remote_mcp_servers
    # Go up: remote_mcp_servers -> sciconharness -> SciConBench (project root)
    project_root = current_dir.parent.parent
    titles_file = project_root / "data" / "filter_data" / "cochrane_titles.json"
    
    if not titles_file.exists():
        # Return the expected path (will fail when loading if not found)
        return titles_file
    
    return titles_file


def load_cochrane_titles(titles_file: Path) -> List[str]:
    """Load Cochrane review titles from JSON file. FAILS FAST if file cannot be loaded."""
    logger = logging.getLogger(__name__)
    
    if not titles_file.exists():
        error_msg = f"Cochrane titles file not found: {titles_file}. Filtering cannot proceed without this file."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        with open(titles_file, 'r', encoding='utf-8') as f:
            titles = json.load(f)
            if not isinstance(titles, list):
                error_msg = f"Cochrane titles file must contain a JSON array, got {type(titles)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            logger.info(f"Loaded {len(titles)} Cochrane review titles for filtering from {titles_file}")
            return titles
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in Cochrane titles file {titles_file}: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e
    except Exception as e:
        error_msg = f"Error loading Cochrane titles file {titles_file}: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def create_log_filename_from_title(source_title: str, publication_date: str) -> str:
    """Create a safe filename from source title and publication date."""
    # Remove special characters and limit length
    title_safe = re.sub(r'[^\w\s-]', '', source_title)[:100]
    title_safe = re.sub(r'[-\s]+', '_', title_safe)
    
    # Format date as YYYY_MM_DD
    date_safe = publication_date.replace('-', '_')
    
    return f"{title_safe}_{date_safe}"


def set_global_filter_config(source_title: Optional[str], publication_date: Optional[str]):
    """Set the global filter configuration. Thread-safe.
    
    Args:
        source_title: Source title to filter out, or None to disable filtering
        publication_date: Publication date cutoff, or None to disable filtering
    """
    with _filter_config_lock:
        _global_filter_config['source_title'] = source_title
        _global_filter_config['publication_date'] = publication_date
        # Clear allowed fetch URLs when configuration changes
        # This ensures each configuration has its own isolated set of URLs
        _allowed_fetch_urls.clear()
        # Clear filtered URLs when configuration changes
        # This ensures each configuration has its own isolated set of filtered URLs
        _filtered_urls.clear()


def get_global_filter_config() -> Dict[str, Optional[str]]:
    """Get the current global filter configuration. Thread-safe."""
    with _filter_config_lock:
        return _global_filter_config.copy()


def setup_session_logging(module_logger: logging.Logger, log_dir: Optional[Path] = None, source_title: Optional[str] = None, publication_date: Optional[str] = None) -> Path:
    """Set up file logging for the current configuration.
    
    Creates a log file in the specified log_dir (if provided) or in 
    sciconharness/remote_mcp_servers/logs/{config_id}/ directory.
    Both serper_jina and semantic_scholar_jina write to the same log file.
    
    Args:
        source_title: Optional source title for the configuration (required if log_dir not provided)
        publication_date: Optional publication date (required if log_dir not provided)
        module_logger: The module logger to configure (e.g., logger from __main__)
        log_dir: Optional log directory path. If provided, logs will be written to {log_dir}/remote_mcps.log
    """
    if log_dir:
        # Use provided log directory and write to remote_mcps.log
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "remote_mcps.log"
    else:
        # Create base log directory (shared between both servers)
        current_dir = Path(__file__).resolve().parent
        base_log_dir = current_dir / "logs"
        base_log_dir.mkdir(parents=True, exist_ok=True)
    
        # Create config-specific directory (requires both source_title and publication_date)
        if not source_title or not publication_date:
            # Use "unfiltered" as default when filter params are not provided
            config_id = "unfiltered"
        else:
            config_id = create_log_filename_from_title(source_title, publication_date)
        log_dir = base_log_dir / config_id
        log_dir.mkdir(parents=True, exist_ok=True)
    
        # Use a fixed filename per config (shared between both servers)
        log_file = log_dir / "server.log"
    
    # Create file handler (append mode so both servers can write to same file)
    # Use ImmediateFlushFileHandler to ensure logs are written immediately
    # Ensure parent directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = ImmediateFlushFileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                         datefmt='%Y-%m-%d %H:%M:%S')
    )
    file_handler.setLevel(logging.INFO)
    
    # Configure module logger (file output only, no console)
    module_logger.handlers = []
    module_logger.propagate = False
    module_logger.addHandler(file_handler)
    module_logger.setLevel(logging.INFO)
    
    # Configure utils logger to write to file (for middleware logs)
    utils_logger = logging.getLogger("utils")
    utils_logger.handlers = []
    utils_logger.addHandler(file_handler)
    utils_logger.setLevel(logging.INFO)
    utils_logger.propagate = False
    
    # Configure filter logger to write to file only (no console)
    filter_logger = logging.getLogger("sciconharness.mcp_client.filters.cochrane")
    filter_logger.handlers = []
    filter_logger.addHandler(file_handler)
    filter_logger.setLevel(logging.INFO)
    filter_logger.propagate = False
    
    # Configure apis logger to write to file only (no console)
    apis_logger = logging.getLogger("sciconharness.mcp_server.apis")
    apis_logger.handlers = []
    apis_logger.addHandler(file_handler)
    apis_logger.setLevel(logging.INFO)
    apis_logger.propagate = False
    
    # Also configure __main__ logger if the server is run as a script
    # This ensures all logging from the main module goes to the file
    # Note: When server runs as script, __name__ is "__main__", so module_logger is already __main__
    # But we configure it explicitly here to be safe
    main_logger = logging.getLogger("__main__")
    # Always configure __main__ logger (it might be used by tool functions)
    if main_logger.name == "__main__":
        # Check if it already has our handler to avoid duplicates
        has_handler = any(
            isinstance(h, ImmediateFlushFileHandler) and h.baseFilename == str(log_file)
            for h in main_logger.handlers
        )
        if not has_handler:
            main_logger.handlers = []
            main_logger.addHandler(file_handler)
            main_logger.setLevel(logging.INFO)
            main_logger.propagate = False
    
    # Write initial log entry to verify file creation
    module_logger.info("=" * 80)
    module_logger.info("REMOTE MCP SERVER LOGGING INITIALIZED")
    module_logger.info("=" * 80)
    module_logger.info(f"Log file: {log_file}")
    module_logger.info(f"Filtering: {'ENABLED' if (source_title and publication_date) else 'DISABLED'}")
    if source_title:
        module_logger.info(f"Source Title: {source_title}")
    if publication_date:
        module_logger.info(f"Publication Date: {publication_date}")
    module_logger.info("=" * 80)
    module_logger.info("")
    
    # Force flush to ensure initial log is written
    file_handler.flush()
    
    # Verify file was actually created
    if not log_file.exists():
        raise RuntimeError(f"Failed to create log file at {log_file}")
    
    return log_file


def track_search_urls(urls: List[str]):
    """Track URLs from search results."""
    _allowed_fetch_urls.update(urls)


def get_allowed_fetch_urls() -> set:
    """Get the allowed fetch URLs set."""
    return _allowed_fetch_urls.copy()


def validate_fetch_url(webpage_url: str) -> bool:
    """Validate that a URL was returned from a search result."""
    return webpage_url in _allowed_fetch_urls


def track_filtered_urls(urls: List[str]):
    """Track URLs that have been filtered."""
    with _filter_config_lock:
        _filtered_urls.update(urls)


def get_filtered_urls() -> set:
    """Get the filtered URLs set."""
    with _filter_config_lock:
        return _filtered_urls.copy()


def is_url_filtered(url: str) -> bool:
    """Check if a URL has been filtered."""
    with _filter_config_lock:
        return url in _filtered_urls


def flush_log_handler():
    """Flush all log handlers to ensure logs are written immediately."""
    # Get the main logger and flush all its handlers
    main_logger = logging.getLogger("__main__")
    for handler in main_logger.handlers:
        if hasattr(handler, 'flush'):
            handler.flush()


def check_auth(request: Request, auth_token: Optional[str]) -> Optional[JSONResponse]:
    """Check if request has valid authorization token."""
    if not auth_token:
        return None  # No auth required
    
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return JSONResponse(
            {
                "error": "Unauthorized",
                "message": "Missing or invalid Authorization header. Expected: Bearer <token>"
            },
            status_code=401
        )
    
    token = auth_header[7:]  # Remove "Bearer " prefix
    if token != auth_token:
        return JSONResponse(
            {
                "error": "Unauthorized",
                "message": "Invalid authorization token"
            },
            status_code=401
        )
    
    return None


def extract_session_token_from_request(request: Request) -> Optional[str]:
    """Extract session token from request (query params or headers)."""
    return request.headers.get("X-Session-Token") or request.query_params.get("session_token")


class FilterConfigMiddleware(BaseHTTPMiddleware):
    """Middleware to extract and validate filter configuration from requests.
    
    Extracts filter configuration from session token (query param or header) or headers.
    FAILS FAST if required headers are missing for tool execution requests.
    """
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Use __main__ logger to ensure logs go to the file handler
        # The utils logger might not have the file handler set up yet
        logger = logging.getLogger("__main__")
        
        # Log that middleware is running (this helps debug if middleware is being called)
        logger.info(f"FilterConfigMiddleware: DISPATCH CALLED - Path: {request.url.path}, Method: {request.method}")
        flush_log_handler()  # Flush immediately to ensure this log is written
        
        # Skip validation for health check and configuration endpoints
        if request.url.path in ["/health", "/configure"]:
            logger.info(f"FilterConfigMiddleware: Skipping {request.url.path} (health/configure endpoint)")
            return await call_next(request)
        
        # Skip validation for GET requests (typically used for tool discovery)
        if request.method == "GET":
            return await call_next(request)
        
        # For POST requests, check if it's a tool discovery request
        is_tool_discovery = False
        if request.method == "POST":
            try:
                body = await request.body()
                if body:
                    body_data = json.loads(body)
                    method = body_data.get("method", "")
                    logger.info(f"FilterConfigMiddleware: POST request method = {method}")
                    if method in ["tools/list", "initialize", "ping"]:
                        is_tool_discovery = True
                        logger.info(f"FilterConfigMiddleware: Detected tool discovery request (method: {method}), skipping filter validation")
                    elif method == "tools/call":
                        logger.info(f"FilterConfigMiddleware: Detected tool call request (method: {method}), proceeding with filter validation")
                
                # Re-create the request body so it can be read again by the handler
                async def receive():
                    return {"type": "http.request", "body": body}
                request._receive = receive
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"FilterConfigMiddleware: Error parsing request body: {e}")
        
        # Skip validation for tool discovery requests
        if is_tool_discovery:
            return await call_next(request)
        
        # Try to get filter configuration from session token (preferred method)
        session_token = extract_session_token_from_request(request)
        source_title = None
        publication_date = None
        
        logger.info(f"FilterConfigMiddleware: Request path = {request.url.path}")
        logger.info(f"FilterConfigMiddleware: Query params = {dict(request.query_params)}")
        logger.info(f"FilterConfigMiddleware: Session token from headers/query = {session_token}")
        
        # Get global filter configuration (no longer using session tokens)
        filter_config = get_global_filter_config()
        source_title = filter_config.get('source_title')
        publication_date = filter_config.get('publication_date')
        
        logger.info(f"FilterConfigMiddleware: Using global filter config - source_title={source_title[:50] if source_title else None}..., publication_date={publication_date}")
        
        # If global config is not set, that's okay - tools will handle it
        # We just pass through the request
        flush_log_handler()
        
        return await call_next(request)


class ToolSchemaFixMiddleware(BaseHTTPMiddleware):
    """Middleware to fix tool schemas for OpenAI compatibility.
    
    Adds additionalProperties: false to all tool inputSchema objects
    in tools/list responses, as required by OpenAI's responses API.
    """
    
    async def dispatch(self, request: Request, call_next: Callable):
        logger = logging.getLogger("__main__")
        
        # Check if this is a tools/list request by reading body once
        is_tools_list = False
        if request.method == "POST" and request.url.path.endswith("/mcp"):
            try:
                body = await request.body()
                if body:
                    body_data = json.loads(body)
                    is_tools_list = body_data.get("method") == "tools/list"
                    if is_tools_list:
                        logger.info("ToolSchemaFixMiddleware: Detected tools/list request")
                    
                    # Re-create the request body so it can be read again by the handler
                    async def receive():
                        return {"type": "http.request", "body": body}
                    request._receive = receive
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"ToolSchemaFixMiddleware: Error parsing request: {e}")
        
        response = await call_next(request)
        
        # Only modify JSON responses for tools/list method
        if is_tools_list:
            try:
                logger.info("ToolSchemaFixMiddleware: Processing tools/list response")
                logger.info(f"ToolSchemaFixMiddleware: Response type: {type(response)}")
                logger.info(f"ToolSchemaFixMiddleware: Response status: {response.status_code}")
                
                # Read response body
                response_body = b""
                if hasattr(response, 'body_iterator'):
                    logger.info("ToolSchemaFixMiddleware: Reading from body_iterator")
                    async for chunk in response.body_iterator:
                        response_body += chunk
                elif hasattr(response, 'body'):
                    logger.info("ToolSchemaFixMiddleware: Reading from body")
                    response_body = response.body if isinstance(response.body, bytes) else b""
                else:
                    logger.warning(f"ToolSchemaFixMiddleware: No body or body_iterator found. Response attributes: {dir(response)}")
                
                if not response_body:
                    logger.warning("ToolSchemaFixMiddleware: Empty response body")
                    return response
                
                logger.info(f"ToolSchemaFixMiddleware: Response body length: {len(response_body)} bytes")
                
                # Parse JSON response
                response_data = json.loads(response_body.decode('utf-8'))
                logger.info(f"ToolSchemaFixMiddleware: Response keys: {list(response_data.keys())}")
                
                # Fix tool schemas in the response
                if "result" in response_data and "tools" in response_data["result"]:
                    modified = False
                    tools_count = len(response_data["result"]["tools"])
                    logger.info(f"ToolSchemaFixMiddleware: Found {tools_count} tools to fix")
                    
                    for tool in response_data["result"]["tools"]:
                        tool_name = tool.get('name', 'unknown')
                        logger.info(f"ToolSchemaFixMiddleware: Processing tool '{tool_name}'")
                        
                        if "inputSchema" in tool and isinstance(tool["inputSchema"], dict):
                            schema = tool["inputSchema"]
                            logger.info(f"ToolSchemaFixMiddleware: Tool '{tool_name}' original schema: {json.dumps(schema, indent=2)}")
                            schema_modified = False
                            
                            # Ensure schema has type: "object" at root
                            if "type" not in schema:
                                schema["type"] = "object"
                                schema_modified = True
                                logger.info(f"ToolSchemaFixMiddleware: Added type=object to tool '{tool_name}'")
                            
                            # Ensure schema has properties (even if empty)
                            if "properties" not in schema:
                                schema["properties"] = {}
                                schema_modified = True
                                logger.info(f"ToolSchemaFixMiddleware: Added properties={{}} to tool '{tool_name}'")
                            
                            # Ensure required fields are set if properties exist
                            if "properties" in schema and schema["properties"] and "required" not in schema:
                                # Set all properties as required
                                schema["required"] = list(schema["properties"].keys())
                                schema_modified = True
                                logger.info(f"ToolSchemaFixMiddleware: Added required fields for tool '{tool_name}': {schema['required']}")
                            
                            # Set additionalProperties to false (required by OpenAI)
                            if "additionalProperties" not in schema or schema.get("additionalProperties") is not False:
                                schema["additionalProperties"] = False
                                schema_modified = True
                                logger.info(f"ToolSchemaFixMiddleware: Set additionalProperties=false for tool '{tool_name}'")
                            
                            if schema_modified:
                                modified = True
                                logger.info(f"ToolSchemaFixMiddleware: Fixed schema for tool '{tool_name}': {json.dumps(schema, indent=2)}")
                        else:
                            logger.warning(f"ToolSchemaFixMiddleware: Tool '{tool_name}' has no inputSchema or it's not a dict")
                            logger.warning(f"ToolSchemaFixMiddleware: Tool keys: {list(tool.keys())}")
                    
                    if modified:
                        logger.info("ToolSchemaFixMiddleware: Returning modified response")
                        return JSONResponse(response_data, status_code=response.status_code)
                    else:
                        logger.info("ToolSchemaFixMiddleware: No modifications needed (additionalProperties already set)")
                
                # Return original response data if no modification needed
                return JSONResponse(response_data, status_code=response.status_code)
            except (json.JSONDecodeError, KeyError, AttributeError, TypeError) as e:
                # If we can't parse or modify, log and return original response
                logger.error(f"ToolSchemaFixMiddleware: Error modifying response: {e}", exc_info=True)
                # Note: Once body_iterator is consumed, we can't return original response
                # So we return an error response or empty response
                return JSONResponse({"error": "Failed to process response"}, status_code=500)
        
        return response


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware to validate Authorization token for MCP endpoints."""
    
    def __init__(self, app, auth_token: Optional[str]):
        super().__init__(app)
        self.auth_token = auth_token
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Skip authentication for health check endpoint
        if request.url.path == "/health":
            return await call_next(request)
        
        # Check authentication
        auth_error = check_auth(request, self.auth_token)
        if auth_error:
            return auth_error
        
        return await call_next(request)
