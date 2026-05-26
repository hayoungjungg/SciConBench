"""Base filter class for MCP tool results."""

from typing import Any, Dict


class BaseResultFilter:
    """Base class for filtering tool results.
    
    Subclasses should implement:
    - should_filter_tool() to specify which tools to filter
    - filter() to customize filtering behavior
    
    This class makes filters callable like functions for backward compatibility.
    
    Example:
        class MyFilter(BaseResultFilter):
            def should_filter_tool(self, tool_name: str) -> bool:
                return tool_name in ["my_tool_1", "my_tool_2"]
            
            def filter(self, tool_result: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
                # Custom filtering logic
                return tool_result
    """
    
    def __call__(self, tool_result: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        """Make the filter callable like a function.
        
        Args:
            tool_result: The tool result dictionary to filter
            tool_name: Name of the tool that produced the result
            
        Returns:
            Filtered tool result dictionary
        """
        return self.filter(tool_result, tool_name)
    
    def should_filter_tool(self, tool_name: str) -> bool:
        """Determine if this filter should be applied to the given tool.
        
        Override this method in subclasses to specify which tools should be filtered.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if this filter should be applied to the tool, False otherwise.
            Default: False (no tools filtered by default)
        """
        return False
    
    def filter(self, tool_result: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        """Filter the tool result. Override this method in subclasses.
        
        Args:
            tool_result: The tool result dictionary to filter
            tool_name: Name of the tool that produced the result
            
        Returns:
            Filtered tool result dictionary (default: no filtering)
        """
        return tool_result


