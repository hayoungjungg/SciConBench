"""MCP tool result filters.

This module provides filter classes for filtering tool results from MCP tools.
"""

from .base import BaseResultFilter
from .cochrane import (
    CochraneResultFilter,
    create_title_filter_from_list,
    custom_cochrane_filter_search_results,
)

__all__ = [
    "BaseResultFilter",
    "CochraneResultFilter",
    "create_title_filter_from_list",
    "custom_cochrane_filter_search_results",
]


