"""
Modular MCP Client Setup for Multiple LLMs

This module provides integration between various LLMs (OpenAI, Gemini, Claude) and the MCP server,
exposing three tools:
1. Google Search (serper_google_webpage_search)
2. Browse Web via JINA API (jina_fetch_webpage_content)
3. Semantic Scholar Snippet Search (semantic_scholar_snippet_search)
"""

from .filters import (
    BaseResultFilter,
    CochraneResultFilter,
    custom_cochrane_filter_search_results,
)
from .llm_providers import (
    ClaudeProvider,
    GeminiProvider,
    LLMProvider,
    OpenAIProvider,
    PerplexityProvider,
)
from .mcp_client import MCPClient

__all__ = [
    # Core classes
    "MCPClient",
    "LLMProvider",
    # Providers
    "OpenAIProvider",
    "GeminiProvider",
    "ClaudeProvider",
    "PerplexityProvider",
    # Filters
    "BaseResultFilter",
    "CochraneResultFilter",
    "custom_cochrane_filter_search_results",
]


