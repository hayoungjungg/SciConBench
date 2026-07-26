"""
Utility functions for query processing.

Common functionality shared between query_single.py and query_batch.py.
"""

import json
import logging
import os
import re
import time
import requests
from requests.exceptions import Timeout, ConnectionError as RequestsConnectionError
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from sciconharness.mcp_client import MCPClient
from sciconharness.mcp_client.filters.base import BaseResultFilter
from sciconharness.mcp_client.llm_providers import (
    AzureChatCompletionsProvider,
    ClaudeProvider,
    GeminiProvider,
    OpenAIProvider,
    OpenRouterProvider,
    PerplexityProvider,
)
from sciconharness.mcp_client.utils import setup_logging_for_run

logger = logging.getLogger(__name__)


def create_provider(
    provider_name: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_version: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Any:
    """
    Create and return an LLM provider instance.
    
    Args:
        provider_name: Provider name ("openai", "gemini", "claude", "perplexity", "azure", or "openrouter")
        model: Model name
        api_key: Optional API key (will try environment variables if not provided)
        base_url: Base URL for Azure OpenAI or Foundry (auto-detects Azure/Foundry mode when provided)
        api_version: API version for Azure OpenAI (default: 2025-04-01-preview)
        temperature: Sampling temperature (0.0 to 1.0 for Claude, 0.0 to 2.0 for Perplexity). If None, uses model default.
        max_tokens: Maximum number of tokens to generate. If None, uses provider default. (Claude and Perplexity)
    
    Returns:
        Provider instance (OpenAIProvider, GeminiProvider, ClaudeProvider,
        PerplexityProvider, AzureChatCompletionsProvider, or OpenRouterProvider)
    """
    provider_name = provider_name.lower()
    
    if provider_name == "gemini":
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return GeminiProvider(model=model, api_key=api_key)
    elif provider_name == "perplexity":
        # Set default model based on provider if not specified
        if not model or model == "sonar":
            model = "sonar"  # Default Perplexity model
        
        api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        
        perplexity_kwargs = {
            "model": model,
            "api_key": api_key,
        }
        
        # Add temperature if provided (default is 0.2)
        if temperature is not None:
            perplexity_kwargs["temperature"] = temperature
        
        
        return PerplexityProvider(**perplexity_kwargs)
    elif provider_name == "azure":
        # Azure Foundry Chat Completions (DeepSeek-V4-Pro, etc.)
        azure_kwargs = {
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            "api_version": api_version,
        }
        if temperature is not None:
            azure_kwargs["temperature"] = temperature
        if max_tokens is not None:
            azure_kwargs["max_tokens"] = max_tokens
        return AzureChatCompletionsProvider(**azure_kwargs)
    elif provider_name == "openrouter":
        # OpenRouter Chat Completions (Kimi K3, GLM-5.2, Qwen3.5-9B, Qwen3.7-max, etc.)
        openrouter_kwargs = {
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
        }
        if temperature is not None:
            openrouter_kwargs["temperature"] = temperature
        if max_tokens is not None:
            openrouter_kwargs["max_tokens"] = max_tokens
        return OpenRouterProvider(**openrouter_kwargs)
    elif provider_name == "claude":
        # Auto-detect Foundry if base_url is provided or AZURE_ANTHROPIC_API_KEY is set
        foundry_base_url = base_url or os.getenv("AZURE_ANTHROPIC_BASE_URL") or os.getenv("ANTHROPIC_FOUNDRY_BASE_URL")
        use_foundry = foundry_base_url is not None or (os.getenv("AZURE_ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_FOUNDRY_API_KEY"))
        
        # Build kwargs for ClaudeProvider
        claude_kwargs = {
            "model": model,
            "use_foundry": use_foundry,
        }
        
        if use_foundry:
            # Microsoft Foundry initialization
            # Check for resource name (alternative to base_url)
            resource = os.getenv("AZURE_ANTHROPIC_RESOURCE_NAME") or os.getenv("ANTHROPIC_FOUNDRY_RESOURCE")
            api_key = api_key or os.getenv("AZURE_ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_FOUNDRY_API_KEY")
            
            # Extract resource from base_url if it matches the pattern https://<resource>.services.ai...
            if foundry_base_url and not resource:
                # Pattern: https://<resource>.services.ai...
                match = re.search(r'https://([^/]+)\.services\.ai', foundry_base_url)
                if match:
                    resource = match.group(1)
                    logger.info(f"Extracted resource name '{resource}' from base URL")
            
            claude_kwargs["api_key"] = api_key
            # Only pass base_url or resource, not both (they are mutually exclusive)
            # Prefer resource if available (recommended for Azure Foundry)
            if resource:
                claude_kwargs["resource"] = resource
            elif foundry_base_url:
                claude_kwargs["base_url"] = foundry_base_url
        else:
            # Standard Anthropic API
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            claude_kwargs["api_key"] = api_key
        
        # Add temperature and max_tokens if provided
        if temperature is not None:
            claude_kwargs["temperature"] = temperature
        if max_tokens is not None:
            claude_kwargs["max_tokens"] = max_tokens
        
        return ClaudeProvider(**claude_kwargs)
    else:
        # OpenAI/GPT provider
        api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
        openai_base_url = base_url or os.getenv("OPENAI_BASE_URL")
        
        # Deep research models (o4-mini-deep-research, o3-deep-research) are only available on regular OpenAI API, not Azure
        # Always use regular OpenAI for these models, ignore base_url
        is_deep_research = "o4-mini-deep-research" in model.lower() or "o3-deep-research" in model.lower()
        
        if is_deep_research:
            # Force regular OpenAI API for deep research models
            logger.info(f"Using regular OpenAI API for deep research model: {model} (Azure not supported)")
            return OpenAIProvider(model=model, api_key=api_key)
        elif openai_base_url:
            # For other models, use Azure OpenAI if base_url is provided
            openai_api_version = api_version or os.getenv("OPENAI_API_VERSION", "2025-04-01-preview")
            return OpenAIProvider(
                model=model,
                api_key=api_key,
                base_url=openai_base_url,
                api_version=openai_api_version
            )
        else:
            # Standard OpenAI API
            return OpenAIProvider(model=model, api_key=api_key)


def load_cochrane_titles(titles_file: Path) -> list:
    """
    Load Cochrane review titles from JSON file.
    
    Args:
        titles_file: Path to JSON file containing list of titles
    
    Returns:
        List of titles
    """
    try:
        with open(titles_file, 'r', encoding='utf-8') as f:
            titles = json.load(f)
            print(f"Loaded {len(titles)} Cochrane review titles for filtering")
            return titles
    except Exception as e:
        print(f"Warning: Could not load titles from {titles_file}: {e}")
        return []


def load_doi_to_title_mapping(review_articles_file: Optional[Path] = None) -> Dict[str, str]:
    """
    Load DOI-to-title mapping.

    Args:
        review_articles_file: Optional explicit path to a legacy scraped
            ``review_articles/data.json`` file (list of ``{"doi", "name"}``
            objects). If None (the common case), falls back to the
            HuggingFace-backed cache built from the live SciConBench dataset
            — see ``sciconharness.utils.hf_benchmark_cache``.

    Returns:
        Dictionary mapping DOI to title
    """
    if review_articles_file is not None:
        doi_to_title = {}
        try:
            with open(review_articles_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
                for article in articles:
                    doi = article.get("doi")
                    title = article.get("name")
                    if doi and title:
                        doi_to_title[doi] = title
            logger.debug("Loaded %d DOI-to-title mappings from %s", len(doi_to_title), review_articles_file)
            return doi_to_title
        except Exception as e:
            logger.debug("DOI-to-title mapping not found at %s (optional, skipping): %s", review_articles_file, e)
            return {}

    try:
        from sciconharness.utils.hf_benchmark_cache import load_doi_to_title_cached
        doi_to_title = load_doi_to_title_cached()
        logger.debug("Loaded %d DOI-to-title mappings from HF benchmark cache", len(doi_to_title))
        return doi_to_title
    except Exception as e:
        logger.debug("HF benchmark DOI-to-title cache unavailable (optional, skipping): %s", e)
        return {}


def load_doi_to_publication_date_mapping() -> Dict[str, str]:
    """
    Load DOI-to-publication-date mapping from the HuggingFace-backed cache
    built from the live SciConBench dataset — see
    ``sciconharness.utils.hf_benchmark_cache``.

    Returns:
        Dictionary mapping DOI to publication date string (e.g. ``"13 June 2012"``)
    """
    try:
        from sciconharness.utils.hf_benchmark_cache import load_doi_to_publication_date_cached
        doi_to_pubdate = load_doi_to_publication_date_cached()
        logger.debug("Loaded %d DOI-to-publication-date mappings from HF benchmark cache", len(doi_to_pubdate))
        return doi_to_pubdate
    except Exception as e:
        logger.debug("HF benchmark DOI-to-publication-date cache unavailable (optional, skipping): %s", e)
        return {}


def get_title_for_doi(doi: str, doi_to_title_mapping: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    Get the title for a specific DOI.
    
    Args:
        doi: The DOI to look up
        doi_to_title_mapping: Optional pre-loaded mapping. If None, will load from default location.
    
    Returns:
        Title string if found, None otherwise
    """
    if doi_to_title_mapping is None:
        doi_to_title_mapping = load_doi_to_title_mapping()
    
    return doi_to_title_mapping.get(doi)


def load_doi_dict(file_path: Path) -> Dict[str, str]:
    """
    Load DOI dictionary from JSON file.
    
    Args:
        file_path: Path to JSON file containing dictionary
    
    Returns:
        Dictionary mapping DOI to value
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            else:
                raise ValueError(f"Expected dictionary, got {type(data)}")
    except Exception as e:
        raise ValueError(f"Could not load DOI dictionary from {file_path}: {e}")


def sanitize_doi_for_path(doi: str) -> str:
    """
    Sanitize DOI for use in file paths.
    
    Args:
        doi: DOI string
    
    Returns:
        Sanitized DOI safe for filesystem
    """
    sanitized = re.sub(r'[^\w\-_\.]', '_', doi)
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    return sanitized


def setup_directories(
    model: str,
    doi: str,
    save_results: bool,
    enable_tool_calling: bool = True,
    enable_filtering: bool = True,
    base_path: Optional[Path] = None,
) -> tuple[Path, Optional[Path]]:
    """
    Set up logging and data directories.
    
    Creates structure: {base_log_dir}/{model_name}/{doi}/
    All logs (query_batch, query_single, mcp_client, remote_mcps) go to this directory.
    
    Args:
        model: Model name
        doi: DOI
        save_results: Whether to create data directory
        enable_tool_calling: Whether tool calling is enabled
        enable_filtering: Whether filtering is enabled
        base_path: Base path for data directory (default: project root)
    
    Returns:
        Tuple of (log_dir, data_dir) where data_dir is None if save_results=False
    """
    from sciconharness.mcp_client.utils.utils import set_custom_log_dir
    
    model_name_safe = model.replace("/", "_")
    
    # Build directory name with tool calling and filtering status
    dir_parts = [model_name_safe]
    if enable_tool_calling:
        dir_parts.append("tools")
    if enable_filtering:
        dir_parts.append("filter")
    model_dir_name = "_".join(dir_parts)
    
    # Set up logging - this creates {base_log_dir}/{model_dir_name}/{doi}/
    log_dir = setup_logging_for_run(model_dir_name, doi)
    
    # Set custom log dir to the base logs directory (not DOI-specific) so that
    # subsequent DOIs in batch processing are siblings, not nested.
    # Remote MCP servers get log_dir passed directly via configure_remote_mcp_servers,
    # so they don't need _custom_log_dir to be set to the DOI directory.
    # Calculate the base logs directory (parent of model directory, which is parent of log_dir)
    base_logs_dir = log_dir.parent.parent
    set_custom_log_dir(base_logs_dir)
    
    # Save results in the same directory as logs
    data_dir = None
    if save_results:
        data_dir = log_dir  # Use log directory for saving results
        data_dir.mkdir(parents=True, exist_ok=True)
    
    return log_dir, data_dir


def extract_model_parameters(provider: Any) -> Dict[str, Any]:
    """
    Extract model-specific parameters from provider.
    
    Args:
        provider: LLM provider instance
    
    Returns:
        Dictionary of model-specific parameters
    """
    params = {}
    
    # OpenAI-specific parameters
    if isinstance(provider, OpenAIProvider):
        params["reasoning_effort"] = getattr(provider, 'reasoning_effort', None)
        params["verbosity"] = getattr(provider, 'verbosity', None)
        params["reasoning_summary"] = getattr(provider, 'reasoning_summary', None)
    
    # Perplexity-specific parameters
    if isinstance(provider, PerplexityProvider):
        params["temperature"] = getattr(provider, 'temperature', None)
        params["reasoning_effort"] = getattr(provider, 'reasoning_effort', None)
        params["search_type"] = getattr(provider, 'search_type', None)
        params["search_mode"] = getattr(provider, 'search_mode', None)
        params["web_search_context_size"] = getattr(provider, 'web_search_context_size', None)
        params["use_async_deep_research"] = getattr(provider, 'use_async_deep_research', None)
        # Note: domain_filter and search_before_date_filter are per-call parameters, not instance-level
    
    # Claude-specific parameters
    if isinstance(provider, ClaudeProvider):
        params["thinking_mode"] = getattr(provider, 'thinking_mode', None)
        params["thinking_budget_tokens"] = getattr(provider, 'thinking_budget_tokens', None)
        params["max_tokens"] = getattr(provider, 'max_tokens', None)
        params["temperature"] = getattr(provider, 'temperature', None)
        params["adaptive_effort"] = getattr(provider, 'adaptive_effort', None)

    # Gemini-specific parameters
    if isinstance(provider, GeminiProvider):
        thinking_level = getattr(provider, 'thinking_level', None)
        params["thinking_level"] = thinking_level.name if hasattr(thinking_level, 'name') else thinking_level
        params["thinking_budget_tokens"] = getattr(provider, 'thinking_budget_tokens', None)
        params["max_output_tokens"] = getattr(provider, 'max_output_tokens', None)
        params["temperature"] = getattr(provider, 'temperature', None)
        params["top_p"] = getattr(provider, 'top_p', None)
        params["top_k"] = getattr(provider, 'top_k', None)

    # Azure Chat Completions parameters
    if isinstance(provider, AzureChatCompletionsProvider):
        params["temperature"] = getattr(provider, 'temperature', None)
        params["max_tokens"] = getattr(provider, 'max_tokens', None)
        params["reasoning_effort"] = getattr(provider, 'reasoning_effort', None)
        params["base_url"] = getattr(provider, 'base_url', None)

    # OpenRouter parameters
    if isinstance(provider, OpenRouterProvider):
        params["temperature"] = getattr(provider, 'temperature', None)
        params["max_tokens"] = getattr(provider, 'max_tokens', None)
        # reasoning_effort may be None even when reasoning is enabled, for
        # models that don't expose effort selection (e.g. Qwen3.5-9B,
        # Qwen3.7-max) — those instead get an explicit reasoning_max_tokens
        # budget. See reasoning_enabled for whether the "reasoning":
        # {"enabled": True} payload was sent at all.
        params["reasoning_effort"] = getattr(provider, 'reasoning_effort', None)
        params["reasoning_max_tokens"] = getattr(provider, 'reasoning_max_tokens', None)
        params["reasoning_enabled"] = getattr(provider, '_reasoning_enabled', None)
        params["base_url"] = getattr(provider, 'base_url', None)

    return params


def save_query_result(
    data_dir: Path,
    query: str,
    response: str,
    token_usage: Dict[str, Any],
    provider_name: str,
    model: str,
    doi: str,
    provider: Any,
    client: Optional[MCPClient],
    publication_date: Optional[str] = None,
    filename: Optional[str] = None,
    result_filter: Optional[BaseResultFilter] = None,
) -> Path:
    """
    Save query result to data folder.
    
    Args:
        data_dir: Directory to save result
        query: Query string
        response: Response string
        token_usage: Token usage dictionary
        provider_name: Provider name
        model: Model name
        doi: DOI
        provider: Provider instance (for extracting model parameters)
        client: MCPClient instance (for extracting client settings). 
                Can be None for deep research models that use remote MCP servers.
        publication_date: Optional publication date
        filename: Optional custom filename (default: based on DOI or timestamp)
    
    Returns:
        Path to the saved result file
    """
    # Extract model-specific parameters
    model_params = extract_model_parameters(provider)
    
    # Extract search_results from token_usage if present (for Perplexity)
    search_results = None
    if isinstance(token_usage, dict) and "search_results" in token_usage:
        search_results = token_usage.pop("search_results")  # Remove from token_usage to keep it clean
    
    # Handle client being None (for deep research models)
    # Deep research models always use tool calling and filtering via remote MCP servers
    is_deep_research = "o4-mini-deep-research" in model.lower() or "o1-mini-deep-research" in model.lower()
    if client is None:
        # For deep research models, tool calling and filtering are always enabled
        enable_tool_calling = True
        enable_filtering = True
    else:
        enable_tool_calling = client.enable_tool_calling
        enable_filtering = client.enable_filtering
    
    # Create result entry
    result_entry = {
        "timestamp": datetime.now().isoformat(),
        "provider": provider_name,
        "model": model,
        "run_id": doi,  # Keep run_id for backward compatibility
        "query": query,
        "response": response,
        "token_usage": token_usage,
        "model_parameters": model_params,
        "doi": doi,
        "publication_date": publication_date,
        "enable_tool_calling": enable_tool_calling,
        "enable_filtering": enable_filtering,
    }
    
    # Add search_results if available (for Perplexity)
    if search_results is not None:
        result_entry["search_results"] = search_results
    
    # Determine filename (use DOI)
    if filename:
        result_file = data_dir / filename
    elif doi:
        doi_safe = sanitize_doi_for_path(doi)
        result_file = data_dir / f"{doi_safe}.json"
    else:
        timestamp_safe = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        result_file = data_dir / f"query_{timestamp_safe}.json"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_entry, f, indent=2, ensure_ascii=False)
    
    # Save filtered links if filter is provided and filtering is enabled
    if enable_filtering and result_filter:
        # Check if filter has get_filtered_links method (CochraneResultFilter)
        if hasattr(result_filter, 'get_filtered_links'):
            filtered_links = result_filter.get_filtered_links()
            if filtered_links:
                # Always use default "filtered_links.json" for consistency
                save_filtered_links(
                    data_dir=data_dir,
                    filtered_links=filtered_links,
                    doi=doi,
                    filename=None,  # Will use default "filtered_links.json"
                )
    
    return result_file


def verify_result_file_exists(result_file: Path) -> bool:
    """
    Verify that a result file exists and is readable.
    
    Args:
        result_file: Path to the result file to verify
    
    Returns:
        True if file exists and is readable, False otherwise
    """
    if not result_file.exists():
        return False
    
    try:
        # Try to read and parse the file to ensure it's valid JSON
        with open(result_file, 'r', encoding='utf-8') as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, IOError, OSError):
        return False


def save_filtered_links(
    data_dir: Path,
    filtered_links: list,
    doi: str,
    filename: Optional[str] = None,
) -> Optional[Path]:
    """
    Save filtered links to a JSON file.
    
    Args:
        data_dir: Directory to save result
        filtered_links: List of unique filtered URLs
        doi: DOI
        filename: Optional custom filename (default: filtered_links.json)
    
    Returns:
        Path to the saved filtered links file, or None if no links to save
    """
    if not data_dir or not filtered_links:
        return None
    
    # Create filtered links entry
    filtered_links_entry = {
        "timestamp": datetime.now().isoformat(),
        "doi": doi,
        "total_filtered_links": len(filtered_links),
        "filtered_links": filtered_links,
    }
    
    # Determine filename
    if filename:
        filtered_links_file = data_dir / filename
    else:
        filtered_links_file = data_dir / "filtered_links.json"
    
    try:
        with open(filtered_links_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_links_entry, f, indent=2, ensure_ascii=False)
        return filtered_links_file
    except Exception as e:
        print(f"Warning: Could not save filtered links file: {e}")
        return None


def configure_remote_mcp_servers(
    source_title: Optional[str] = None,
    publication_date: Optional[str] = None,
    serper_server_base: Optional[str] = None,
    semantic_server_base: Optional[str] = None,
    mcp_auth_token: Optional[str] = None,
    log_dir: Optional[Path] = None,
    max_retries: int = 5,
    custom_logger: Optional[logging.Logger] = None,
) -> None:
    """
    Configure remote MCP servers with filter settings before querying.
    
    This function sends configuration requests to both Serper+Jina and Semantic Scholar+Jina
    servers to set up filtering based on source title and publication date.
    It verifies the configuration was set correctly and retries up to max_retries times
    if verification fails.
    
    Args:
        source_title: Optional source title to filter out
        publication_date: Optional publication date cutoff (format: YYYY-MM-DD)
        serper_server_base: Base URL for Serper+Jina server (without /mcp path)
        semantic_server_base: Base URL for Semantic Scholar+Jina server (without /mcp path)
        mcp_auth_token: Optional authentication token for MCP servers
        log_dir: Optional log directory path
        max_retries: Maximum number of retry attempts (default: 5)
    
    Raises:
        requests.exceptions.RequestException: If configuration fails after all retries
        ValueError: If configuration verification fails after all retries
    """
    # Get server URLs from environment or use defaults
    if serper_server_base is None:
        serper_server_base = os.getenv("SERPER_SERVER_BASE", "https://patriarchical-sherri-burdenedly.ngrok-free.dev/serper")
    if semantic_server_base is None:
        semantic_server_base = os.getenv("SEMANTIC_SERVER_BASE", "https://patriarchical-sherri-burdenedly.ngrok-free.dev/semantic")
    if mcp_auth_token is None:
        mcp_auth_token = os.getenv("MCP_AUTH_TOKEN", "")
    
    # Use custom logger if provided, otherwise use module logger
    log = custom_logger if custom_logger else logger
    
    # Only include Authorization header if token is provided
    auth_headers = {}
    if mcp_auth_token:
        auth_headers["Authorization"] = f"Bearer {mcp_auth_token}"
    
    # If log_dir is provided, we can set up logging even without filter settings
    # Otherwise, we need both filter settings to configure
    if not log_dir and (not source_title or not publication_date):
        log.info("Incomplete filter settings provided (both source_title and publication_date are required), skipping MCP server configuration")
        log.info(f"  source_title: {source_title is not None}, publication_date: {publication_date is not None}")
        return
    
    # If we don't have filter settings but have log_dir, we're just setting up logging (filtering disabled)
    if not source_title or not publication_date:
        log.info("Setting up logging without filter configuration (filtering disabled)")
        log.info(f"  log_dir: {log_dir}")
    
    log.info("=" * 80)
    log.info("CONFIGURING FILTER SETTINGS FOR MCP SERVERS")
    log.info("=" * 80)
    if source_title:
        log.info(f"Source Title: {source_title}")
    if publication_date:
        log.info(f"Publication Date: {publication_date}")
    log.info("")
    
    def _configure_and_verify_server(
        server_name: str,
        server_base: str,
        source_title: Optional[str],
        publication_date: Optional[str],
        log_dir: Optional[Path],
        auth_headers: Dict[str, str],
        max_retries: int,
    ) -> None:
        """Configure and verify a single MCP server with retry logic."""
        config_url = f"{server_base}/configure"
        verify_url = f"{server_base}/verify-config"
        
        # Build payload
        # When filtering is disabled, send empty strings for source_title and publication_date
        # When filtering is enabled, send the actual values
        config_payload = {}
        if source_title:
            config_payload["source_title"] = source_title
        else:
            config_payload["source_title"] = ""  # Empty string to disable filtering
        if publication_date:
            config_payload["publication_date"] = publication_date
        else:
            config_payload["publication_date"] = ""  # Empty string to disable filtering
        if log_dir:
            config_payload["log_dir"] = str(log_dir)
        
        for attempt in range(max_retries):
            try:
                log.info(f"Configuring {server_name} server (attempt {attempt + 1}/{max_retries})...")
                
                # Health check (non-blocking - proceed even if it fails)
                # Increased timeout to 20s for ngrok connections which can be slow
                health_url = f"{server_base}/health"
                try:
                    health_response = requests.get(health_url, timeout=20)
                    if health_response.status_code == 200:
                        log.info(f"✓ {server_name} server is reachable")
                    else:
                        log.warning(f"⚠ {server_name} health check returned status {health_response.status_code} (proceeding anyway)")
                except Exception as health_error:
                    # Health check failure is not critical - log warning but proceed with configuration
                    log.warning(f"⚠ {server_name} health check failed: {health_error}")
                    log.info("  Proceeding with configuration attempt anyway (health check is non-blocking)...")
                
                # Send configuration
                log.info("")
                log.info("-" * 80)
                log.info(f"SETTING CONFIGURATION: {server_name}")
                log.info("-" * 80)
                log.info(f"POST {config_url}")
                log.info(f"  Source Title: {source_title if source_title else '(empty - filtering disabled)'}")
                log.info(f"  Publication Date: {publication_date if publication_date else '(empty - filtering disabled)'}")
                log.info(f"  Log Directory: {log_dir}")
                config_response = requests.post(
                    config_url,
                    json=config_payload,
                    headers=auth_headers,
                    timeout=30
                )
                
                # Log response details for debugging
                if config_response.status_code != 200:
                    log.error(f"✗ Configuration request failed with status {config_response.status_code}")
                    try:
                        error_body = config_response.json()
                        log.error(f"  Error response: {error_body}")
                    except:
                        log.error(f"  Error response (text): {config_response.text[:500]}")
                
                config_response.raise_for_status()
                
                # Extract log_file path from response if available
                try:
                    response_data = config_response.json()
                    log_file_path = response_data.get("log_file")
                    if log_file_path:
                        log.info(f"✓ Remote MCP log file: {log_file_path}")
                        # Verify file exists
                        if Path(log_file_path).exists():
                            log.info(f"✓ Verified: remote_mcps.log file exists at {log_file_path}")
                        else:
                            log.warning(f"⚠ Warning: remote_mcps.log file not found at {log_file_path} (may be created later)")
                    else:
                        log.warning(f"⚠ Warning: Server response does not include log_file path")
                        log.warning(f"  Response: {response_data}")
                except Exception as e:
                    log.warning(f"⚠ Could not parse response for log_file: {e}")
                
                log.info(f"✓ Configuration SET successfully on {server_name}")
                log.info("-" * 80)
                log.info("")
                
                # Verify configuration was set correctly
                log.info("-" * 80)
                log.info(f"VERIFYING CONFIGURATION: {server_name}")
                log.info("-" * 80)
                log.info(f"GET {verify_url}")
                verify_response = requests.get(
                    verify_url,
                    headers=auth_headers,
                    timeout=10
                )
                verify_response.raise_for_status()
                verify_data = verify_response.json()
                
                # Check if configuration matches
                # If we're disabling filtering (empty strings), configured will be False, which is expected
                is_disabling_filtering = not source_title and not publication_date
                is_configured = verify_data.get("configured", False)
                
                if not is_disabling_filtering and not is_configured:
                    log.error(f"✗ Verification failed: {server_name} server reports configuration not set")
                    raise ValueError(f"{server_name} server reports configuration not set")
                elif is_disabling_filtering and is_configured:
                    log.error(f"✗ Verification failed: {server_name} server reports filtering is enabled when it should be disabled")
                    raise ValueError(f"{server_name} server reports filtering is enabled when it should be disabled")
                
                configured_title = verify_data.get("source_title")
                configured_date = verify_data.get("publication_date")
                
                log.info(f"  Verified Source Title: {configured_title}")
                log.info(f"  Verified Publication Date: {configured_date}")
                
                # Normalize empty strings and None for comparison (both mean filtering is disabled)
                expected_title = source_title if source_title else None
                expected_date = publication_date if publication_date else None
                actual_title = configured_title if configured_title else None
                actual_date = configured_date if configured_date else None
                
                if actual_title != expected_title or actual_date != expected_date:
                    log.error(f"✗ Verification failed: Configuration mismatch")
                    log.error(f"  Expected: title='{expected_title}', date='{expected_date}'")
                    log.error(f"  Got: title='{actual_title}', date='{actual_date}'")
                    raise ValueError(
                        f"{server_name} configuration mismatch: "
                        f"expected title='{expected_title}', date='{expected_date}', "
                        f"got title='{actual_title}', date='{actual_date}'"
                    )
                
                log.info(f"✓ Configuration VERIFIED successfully on {server_name}")
                log.info(f"  Configuration matches expected values")
                log.info("-" * 80)
                log.info("")
                return  # Success!
                
            except Timeout as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    log.warning(f"✗ Timeout configuring {server_name} (attempt {attempt + 1}/{max_retries})")
                    log.warning(f"  Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    log.error(f"✗ Timeout connecting to {server_name} server after {max_retries} attempts")
                    log.error(f"  URL: {config_url}")
                    raise
                    
            except RequestsConnectionError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    log.warning(f"✗ Connection error to {server_name} (attempt {attempt + 1}/{max_retries})")
                    log.warning(f"  Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    log.error(f"✗ Connection error to {server_name} server after {max_retries} attempts")
                    log.error(f"  URL: {config_url}")
                    raise
                    
            except (ValueError, KeyError) as e:
                # Configuration verification failed - break fast (no retry)
                log.error(f"✗ {server_name} configuration verification failed: {e}")
                log.error(f"  This indicates the configuration was not set correctly")
                # Re-raise the original exception to break fast
                raise
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    log.warning(f"✗ Error configuring {server_name} (attempt {attempt + 1}/{max_retries}): {e}")
                    log.warning(f"  Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    log.error(f"✗ Failed to configure {server_name} server after {max_retries} attempts: {e}", exc_info=True)
                    raise
        
        # Should never reach here, but just in case
        raise RuntimeError(f"Failed to configure {server_name} server after {max_retries} attempts")
    
    # Configure Serper+Jina server
    if serper_server_base:
        _configure_and_verify_server(
            server_name="Serper+Jina",
            server_base=serper_server_base,
            source_title=source_title,
            publication_date=publication_date,
            log_dir=log_dir,
            auth_headers=auth_headers,
            max_retries=max_retries,
        )
    
    # Configure Semantic Scholar+Jina server
    if semantic_server_base:
        _configure_and_verify_server(
            server_name="Semantic Scholar+Jina",
            server_base=semantic_server_base,
            source_title=source_title,
            publication_date=publication_date,
            log_dir=log_dir,
            auth_headers=auth_headers,
            max_retries=max_retries,
        )
    
    logger.info("")
    logger.info("=" * 80)


def build_mcp_tools_for_deep_research(
    serper_server_base: Optional[str] = None,
    semantic_server_base: Optional[str] = None,
    mcp_auth_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Build MCP tool configurations for OpenAI Deep Research API.
    
    Args:
        serper_server_base: Base URL for Serper+Jina server (without /mcp path)
        semantic_server_base: Base URL for Semantic Scholar+Jina server (without /mcp path)
        mcp_auth_token: Optional authentication token for MCP servers
    
    Returns:
        List of MCP tool configuration dictionaries
    """
    # Get server URLs from environment or use defaults
    if serper_server_base is None:
        serper_server_base = os.getenv("SERPER_SERVER_BASE", "https://patriarchical-sherri-burdenedly.ngrok-free.dev/serper")
    if semantic_server_base is None:
        semantic_server_base = os.getenv("SEMANTIC_SERVER_BASE", "https://patriarchical-sherri-burdenedly.ngrok-free.dev/semantic")
    if mcp_auth_token is None:
        mcp_auth_token = os.getenv("MCP_AUTH_TOKEN", "")
    
    tools = []
    
    if serper_server_base:
        tools.append({
            "type": "mcp",
            "server_label": "SerperJinaMCP",
            "server_url": f"{serper_server_base}/mcp",
            "allowed_tools": ["search", "fetch"],
            "require_approval": "never",
            "authorization": mcp_auth_token
        })
    
    if semantic_server_base:
        tools.append({
            "type": "mcp",
            "server_label": "SemanticScholarJinaMCP",
            "server_url": f"{semantic_server_base}/mcp",
            "allowed_tools": ["search", "fetch"],
            "require_approval": "never",
            "authorization": mcp_auth_token
        })
    
    return tools

