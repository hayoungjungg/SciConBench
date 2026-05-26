"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class ContextLengthExceededError(Exception):
    """Exception raised when the input exceeds the model's context window."""
    pass


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    Each provider must implement methods to:
    - Format tools for the LLM's function calling format
    - Make API calls with function calling support
    - Extract tool calls and text from responses
    """
    
    def __init__(self, model: str, api_key: Optional[str] = None):
        """
        Initialize the LLM provider.
        
        Args:
            model: Model name/identifier
            api_key: API key (if None, will try to get from environment)
        """
        self.model = model
        self.api_key = api_key
    
    @abstractmethod
    def format_tools(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert MCP tools to the LLM's function calling format.
        
        Args:
            tools: List of MCP tool objects
            
        Returns:
            List of tools in the LLM's format
        """
        pass
    
    @abstractmethod
    async def call_llm(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Any, Optional[str], List[Dict[str, Any]], Optional[str]]:
        """
        Call the LLM API with messages and optional tools.
        
        Args:
            messages: Conversation history
            tools: Optional list of tools in LLM format
            
        Returns:
            Tuple of (response, text_content, tool_calls, reasoning_summary)
            - response: Raw response object from LLM (can be None for providers that don't need it)
            - text_content: Text response from LLM (None if only tool calls)
            - tool_calls: List of tool call dicts with 'id', 'name', 'arguments' keys
            - reasoning_summary: Reasoning/summary text if available (None for providers that don't support it)
        """
        pass
    
    @abstractmethod
    def format_tool_response_message(
        self, tool_call_id: str, tool_name: str, tool_result: str
    ) -> Dict[str, Any]:
        """
        Format a tool response message for the LLM.
        
        Args:
            tool_call_id: ID of the tool call
            tool_name: Name of the tool
            tool_result: JSON string of tool result
            
        Returns:
            Message dict in LLM format
        """
        pass


