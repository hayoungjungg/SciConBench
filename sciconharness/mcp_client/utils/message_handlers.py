"""Message handling utilities for MCP client."""

from typing import Dict, List, Optional

from ..filters.base import BaseResultFilter

class MessageHandler:
    """Handles message formatting and management for MCP client."""
    
    @staticmethod
    def create_query_filter(
        base_filter: Optional[BaseResultFilter],
        publication_date: Optional[str] = None
    ) -> Optional[BaseResultFilter]:
        """
        Create or combine filters for a query.
        
        Args:
            base_filter: Base filter to combine with (or None to create new)
            publication_date: Optional publication date to add
            
        Returns:
            Combined filter or new filter instance
        """
        from ..filters import CochraneResultFilter
        
        # If no additional filters needed, return base filter
        if not publication_date:
            return base_filter
        
        # Create new filter if base is None
        if base_filter is None:
            return CochraneResultFilter(
                publication_date=publication_date
            )
        
        # Combine with base filter
        base_title_filter_list = getattr(base_filter, 'title_filter_list', None)
        base_source_title = getattr(base_filter, 'source_title', None)
        
        # Handle publication date: use new if provided, otherwise preserve base
        pub_date_str = publication_date
        if not pub_date_str and hasattr(base_filter, 'publication_date_cutoff') and base_filter.publication_date_cutoff:
            pub_date_str = base_filter.publication_date_cutoff.strftime('%d %B %Y')
        
        # Create combined filter
        query_filter = CochraneResultFilter(
            title_filter_list=base_title_filter_list,
            source_title=base_source_title,
            publication_date=pub_date_str
        )
        
        # Copy filtered_tools from base filter
        if hasattr(base_filter, 'filtered_tools'):
            query_filter.filtered_tools = base_filter.filtered_tools
        
        return query_filter
