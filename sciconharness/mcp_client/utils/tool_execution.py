"""Tool execution logic for MCP client."""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..filters.base import BaseResultFilter
from ..llm_providers.base import LLMProvider
from .utils import (
    cap_year_parameter,
    count_tokens,
    log_tool_result,
)

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Handles tool execution for MCP client."""
    
    def __init__(
        self,
        session: Any,
        llm_provider: LLMProvider,
        enable_filtering: bool = True,
    ):
        """
        Initialize tool executor.
        
        Args:
            session: MCP client session for calling tools
            llm_provider: LLM provider for formatting tool responses
            enable_filtering: Whether filtering is enabled
        """
        self.session = session
        self.llm_provider = llm_provider
        self.enable_filtering = enable_filtering
    
    def extract_tool_result(self, result: Any) -> dict:
        """Extract tool result from MCP response."""
        if hasattr(result, "content") and result.content:
            if isinstance(result.content[0], dict):
                return result.content[0]
            elif hasattr(result.content[0], "text"):
                text_content = result.content[0].text
                # Handle empty or invalid JSON
                if not text_content or not text_content.strip():
                    logger.warning("Empty text content in tool result, returning empty dict")
                    return {}
                
                # Check if this looks like a validation error from MCP server
                # FastMCP validation errors often start with "Output validation error:"
                if text_content.startswith("Output validation error:"):
                    logger.error("MCP server validation error detected: %s", text_content[:200])
                    # Extract the actual error message for better context
                    error_msg = text_content
                    # Try to extract field name if present (e.g., "['field'] is not of type 'string'")
                    return {
                        "success": False,
                        "error": f"MCP server validation error: {error_msg}",
                        "raw_content": text_content
                    }
                
                try:
                    return json.loads(text_content)
                except json.JSONDecodeError as e:
                    logger.error("Failed to parse tool result JSON: %s. Content: %r", e, text_content[:200] if text_content else "None")
                    # Return error dict instead of raising
                    # Include more context in the error message
                    return {
                        "success": False,
                        "error": f"Failed to parse tool result JSON: {e}",
                        "raw_content": text_content[:500] if text_content else ""
                    }
            else:
                return {"data": str(result.content[0])}
        return {"error": "No content in response", "data": []}
    
    def extract_tool_call_info(self, tool_call: Dict[str, Any]) -> tuple[str, str, str]:
        """Extract tool call information from standardized format.
        
        Handles multiple formats:
        - OpenAI format: {"function": {"name": "...", "arguments": "..."}, "id": "..."}
        - Gemini format: {"name": "...", "arguments": "...", "id": "..."}
        """
        # Try OpenAI format first (function dict)
        func = tool_call.get("function", {})
        if func and isinstance(func, dict):
            tool_name = func.get("name", "")
            arguments = func.get("arguments", "{}")
        else:
            # Try direct format (Gemini format)
            tool_name = tool_call.get("name", "")
            arguments = tool_call.get("arguments", "{}")
        
        # Log what we're extracting for debugging
        logger.debug("Extracting tool call info - arguments type: %s, value: %r", 
                    type(arguments).__name__, arguments)
        
        # Handle different argument types
        if arguments is None:
            logger.debug("Arguments is None, normalizing to '{}'")
            arguments = "{}"
        elif isinstance(arguments, dict):
            # If already a dict, convert to JSON string
            arguments = json.dumps(arguments)
        elif isinstance(arguments, str):
            # Handle empty string or whitespace-only string
            if not arguments.strip():
                logger.debug("Empty string arguments, normalizing to '{}'")
                arguments = "{}"
        else:
            # For other types, try to convert to string then JSON
            try:
                arguments = json.dumps(arguments)
            except (TypeError, ValueError):
                logger.warning("Could not serialize arguments type %s, using '{}'", type(arguments).__name__)
                arguments = "{}"
        
        tool_call_id = tool_call.get("id", "")
        
        # Validate tool_name is not empty - should always be present in tool calls
        if not tool_name or not tool_name.strip():
            raise ValueError(f"tool_name cannot be empty. tool_call: {tool_call}, tool_call_id: {tool_call_id}. This indicates a bug in tool call extraction or formatting.")
        
        logger.debug("Extracted - tool_name: %s, tool_call_id: %s, arguments: %r", 
                    tool_name, tool_call_id, arguments)
        
        return (tool_name, arguments, tool_call_id)
    
    async def execute_tool_call(
        self, 
        tool_call: Dict[str, Any], 
        messages: List[Dict[str, Any]],
        input_tokens: int,
        tool_result_tokens_tracker: Optional[List[int]] = None,
        tool_usage_tracker: Optional[List[str]] = None,
        result_filter: Optional[BaseResultFilter] = None
    ) -> tuple[int, bool]:
        """
        Execute a single tool call and add result to messages.
        
        Args:
            tool_call: The tool call dictionary to execute
            messages: The list of messages to add to the conversation
            input_tokens: The number of input tokens to add to the conversation
            tool_result_tokens_tracker: The list of tool result tokens to track
            tool_usage_tracker: The list to track which tools were called
            result_filter: The result filter to apply to the tool result (optional)
            
        Returns:
            Tuple of (updated_input_tokens, success)
        """
        tool_name, tool_args_str, tool_call_id = self.extract_tool_call_info(tool_call)
        
        # Track tool usage
        if tool_usage_tracker is not None:
            tool_usage_tracker.append(tool_name)
        
        tool_result = None  # Initialize to avoid UnboundLocalError
        try:
            # Parse and execute tool
            if isinstance(tool_args_str, str):
                # Normalize empty strings before parsing
                if not tool_args_str.strip():
                    logger.warning("Empty tool arguments string for tool %s, using empty dict", tool_name)
                    tool_args_str = "{}"
                try:
                    parsed_args = json.loads(tool_args_str)
                except json.JSONDecodeError as e:
                    logger.error("Failed to parse tool arguments as JSON. Tool: %s, Arguments string: %r (length: %d), Type: %s, Error: %s", 
                               tool_name, tool_args_str, len(tool_args_str) if tool_args_str else 0, 
                               type(tool_args_str).__name__, str(e))
                    # Try to use empty dict as fallback
                    parsed_args = {}
                    logger.warning("Using empty dict as fallback for tool %s", tool_name)
            elif isinstance(tool_args_str, dict):
                # Arguments are already a dict/object
                parsed_args = tool_args_str
                logger.debug("Tool arguments already parsed (dict): %s", type(tool_args_str).__name__)
            else:
                # Unexpected type, log and use empty dict
                logger.warning("Unexpected tool arguments type %s for tool %s, using empty dict", 
                             type(tool_args_str).__name__, tool_name)
                parsed_args = {}
            
            # For semantic_scholar_snippet_search, add/modify year parameter if publication_date cutoff is set
            # to prevent data leakage from sources published *after* the Cochrane article 
            # Only apply if filtering is enabled
            if (tool_name == "semantic_scholar_snippet_search" and 
                self.enable_filtering and 
                result_filter):
                if hasattr(result_filter, 'publication_date_cutoff') and result_filter.publication_date_cutoff:
                    cutoff_year = result_filter.publication_date_cutoff.year
                    existing_year = parsed_args.get("year")
                    
                    # Cap the existing year parameter or create a new one
                    parsed_args["year"] = cap_year_parameter(existing_year, cutoff_year)
                    logger.debug("Calling tool: %s with args: %s", tool_name, parsed_args)
            
            # Execute tool and get result
            tool_result = await self._execute_tool_with_filtering(
                tool_name, parsed_args, result_filter
            )
            
            tool_result_str = json.dumps(tool_result, indent=2)
            logger.debug("Tool result received (%d chars)", len(tool_result_str))
            
            # Log detailed results for specific tools
            log_tool_result(tool_name, parsed_args, tool_result_str)
            
            input_tokens = self._add_tool_response_to_messages(
                tool_call_id, tool_name, tool_result_str, messages, input_tokens, tool_result_tokens_tracker
            )
            return input_tokens, True
            
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse tool arguments: {e}"
            return self._handle_tool_error(tool_call_id, tool_name, error_msg, messages, input_tokens, tool_result_tokens_tracker)
            
        except Exception as e:
            error_msg = f"Tool execution error: {str(e)}"
            logger.error("Error: %s", error_msg, exc_info=True)
            return self._handle_tool_error(tool_call_id, tool_name, error_msg, messages, input_tokens, tool_result_tokens_tracker)
    
    async def _execute_tool_with_filtering(
        self,
        tool_name: str,
        parsed_args: Dict[str, Any],
        result_filter: Optional[BaseResultFilter]
    ) -> Dict[str, Any]:
        """Execute tool with pagination and filtering logic."""
        # Handle pagination for serper_google_webpage_search when filtering is enabled
        if (tool_name == "serper_google_webpage_search" and 
            self.enable_filtering and
            result_filter and 
            result_filter.should_filter_tool(tool_name)):
            return await self._execute_paginated_search(tool_name, parsed_args, result_filter)
        else:
            # Execute tool call normally (no pagination)
            result = await self.session.call_tool(tool_name, parsed_args)
            tool_result = self.extract_tool_result(result)
            
            # Apply filtering if the filter is responsible for this tool and filtering is enabled
            if (self.enable_filtering and 
                result_filter and 
                result_filter.should_filter_tool(tool_name)):
                
                # Handle Jina API filtering separately (returns single dict, not list)
                if tool_name == "jina_fetch_webpage_content":
                    # Apply filter to the Jina result
                    filtered_result = result_filter.filter(tool_result, tool_name)
                    tool_result = filtered_result
                    # Check if content was filtered (empty content means filtered)
                    if tool_result.get("content") == "" and tool_result.get("success", False):
                        logger.info("Jina content filtered out")
                else:
                    # Handle list-based results (search tools)
                    # Get requested number - check both num_results and limit (different tools use different params)
                    requested_num_results = parsed_args.get("num_results") or parsed_args.get("limit") or 10
                    
                    # Tool-specific result key and logging
                    if tool_name == "semantic_scholar_snippet_search":
                        result_key = "data"
                        fetched_count = len(tool_result.get(result_key, []))
                        logger.info("Fetched %d results from API", fetched_count)
                    else:
                        # Default to "organic" for other tools
                        result_key = "organic"
                        fetched_count = len(tool_result.get(result_key, []))
                        logger.info("Fetched %d results from API", fetched_count)
                    
                    # Apply filter to the results
                    filtered_result = result_filter.filter(tool_result, tool_name)
                    filtered_items = filtered_result.get(result_key, [])
                    
                    # Return all filtered results (don't limit after filtering)
                    # The requested_num_results is used to determine how many to fetch, not how many to return
                    tool_result[result_key] = filtered_items
                    logger.info("After filtering: %d results remaining (requested: %d, fetched: %d)", 
                              len(tool_result[result_key]), requested_num_results, fetched_count)
            
            return tool_result
    
    async def _execute_paginated_search(
        self,
        tool_name: str,
        parsed_args: Dict[str, Any],
        result_filter: BaseResultFilter
    ) -> Dict[str, Any]:
        """Execute paginated search with filtering."""
        # Get requested number - check both num_results and limit (different tools use different params)
        requested_num_results = parsed_args.get("num_results") or parsed_args.get("limit") or 10
        result_key = "organic"
        
        # Fetch pages incrementally until we have enough filtered results
        # Results are returned in order: page 1 items first, then page 2, etc.
        # Within each page, items maintain their original order after filtering
        all_filtered_items = []
        all_organic = []
        knowledge_graph = None
        people_also_ask = []
        related_searches = []
        search_parameters = None
        page = 1
        total_fetched = 0
        
        while len(all_filtered_items) < requested_num_results:
           
            # Create args for this page
            page_args = parsed_args.copy()
            page_args["page"] = page
            
            # Fetch this page (pages are fetched sequentially: 1, 2, 3...)
            logger.info("Fetching page %d to get more filtered results (currently have %d/%d)", 
                       page, len(all_filtered_items), requested_num_results)
            result = await self.session.call_tool(tool_name, page_args)
            page_result = self.extract_tool_result(result)
            
            page_organic = page_result.get(result_key, [])
            if not page_organic:
                # No more results available from API
                logger.info("Page %d returned no results, stopping pagination (have %d/%d filtered results)", 
                          page, len(all_filtered_items), requested_num_results)
                break
            
            # If API returns fewer results than requested, it means there are no more results available
            # Stop pagination in this case (works for both first and subsequent pages)
            if len(page_organic) < requested_num_results:
                logger.info("Page %d returned %d results (less than requested %d), stopping pagination (have %d/%d filtered results)", 
                          page, len(page_organic), requested_num_results, len(all_filtered_items), requested_num_results)
                total_fetched += len(page_organic)
                all_organic.extend(page_organic)
                
                # Filter this page's results before breaking
                filtered_page_result = result_filter.filter(page_result, tool_name)
                filtered_page_items = filtered_page_result.get(result_key, [])
                all_filtered_items.extend(filtered_page_items)
                break
            
            total_fetched += len(page_organic)
            all_organic.extend(page_organic)
            
            # Store metadata from first page
            if page == 1:
                knowledge_graph = page_result.get("knowledgeGraph")
                people_also_ask = page_result.get("peopleAlsoAsk", [])
                related_searches = page_result.get("relatedSearches", [])
                search_parameters = page_result.get("searchParameters", {})
            
            # Filter this page's results (filter preserves order within page)
            filtered_page_result = result_filter.filter(page_result, tool_name)
            filtered_page_items = filtered_page_result.get(result_key, [])
            
            # Add filtered items to our collection in order
            # extend() preserves order: page 1 items come before page 2 items
            all_filtered_items.extend(filtered_page_items)
            
            logger.info("Page %d: Fetched %d results, %d passed filter (total filtered: %d/%d)", 
                      page, len(page_organic), len(filtered_page_items), 
                      len(all_filtered_items), requested_num_results)
            
            page += 1
        
        # Update position indices globally across all pages (not per page)
        final_items = all_filtered_items[:requested_num_results]
        for idx, item in enumerate(final_items, start=1):
            if "position" in item:
                item["position"] = idx
        
        # Build final result with filtered items
        tool_result = {
            result_key: final_items,
        }
        if knowledge_graph:
            tool_result["knowledgeGraph"] = knowledge_graph
        if people_also_ask:
            tool_result["peopleAlsoAsk"] = people_also_ask
        if related_searches:
            tool_result["relatedSearches"] = related_searches
        if search_parameters:
            tool_result["searchParameters"] = search_parameters
        
        logger.info("Fetched %d total results across %d page(s)", total_fetched, page - 1)
        logger.info("After filtering: %d results remaining (requested: %d)", 
                  len(tool_result[result_key]), requested_num_results)
        
        return tool_result
    
    def _add_tool_response_to_messages(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_result_str: str,
        messages: List[Dict[str, Any]],
        input_tokens: int,
        tool_result_tokens_tracker: Optional[List[int]] = None
    ) -> int:
        """Add tool response to messages and return updated token count."""
        tool_response = self.llm_provider.format_tool_response_message(
            tool_call_id, tool_name, tool_result_str
        )
        messages.append(tool_response)
        result_tokens = count_tokens(tool_result_str)
        if tool_result_tokens_tracker is not None:
            tool_result_tokens_tracker.append(result_tokens)
        return input_tokens + result_tokens
    
    def _handle_tool_error(
        self,
        tool_call_id: str,
        tool_name: str,
        error_msg: str,
        messages: List[Dict[str, Any]],
        input_tokens: int,
        tool_result_tokens_tracker: Optional[List[int]] = None
    ) -> tuple[int, bool]:
        """Handle tool execution error and add error response to messages."""
        logger.error("Error: %s", error_msg)
        error_msg_json = json.dumps({"error": error_msg})
        input_tokens = self._add_tool_response_to_messages(
            tool_call_id, tool_name, error_msg_json, messages, input_tokens, tool_result_tokens_tracker
        )
        return input_tokens, False
    
    async def execute_tool_call_for_collection(
        self,
        tool_call: Dict[str, Any],
        input_tokens: int,
        tool_result_tokens_tracker: Optional[List[int]] = None,
        tool_usage_tracker: Optional[List[str]] = None,
        result_filter: Optional[BaseResultFilter] = None
    ) -> tuple[str, bool]:
        """
        Execute a tool call and return the result string without adding to messages.
        Used for collecting multiple tool results before formatting them together for Claude/Gemini.
        
        This method reuses the execution logic from execute_tool_call but extracts
        the result string from the formatted message instead of adding it to the main messages list.
        
        Args:
            tool_call: The tool call dictionary to execute
            input_tokens: Current input token count (for tracking, not modified)
            tool_result_tokens_tracker: The list to track tool result tokens
            tool_usage_tracker: The list to track which tools were called
            result_filter: The result filter to apply to the tool result (optional)
        
        Returns:
            Tuple of (tool_result_str, success)
        """
        # Use a temporary messages list to reuse execute_tool_call logic
        # We'll extract the result from the formatted message
        temp_messages = []
        try:
            _, success = await self.execute_tool_call(
                tool_call, temp_messages, input_tokens,
                tool_result_tokens_tracker=tool_result_tokens_tracker,
                tool_usage_tracker=tool_usage_tracker,
                result_filter=result_filter
            )
            
            if success and temp_messages:
                # Extract the tool result from the last message (which was just added)
                last_message = temp_messages[-1]
                
                # Handle Gemini format: role is "function" with "parts" array
                if last_message.get("role") == "function" and "parts" in last_message:
                    parts = last_message.get("parts", [])
                    if parts and "function_response" in parts[0]:
                        func_response = parts[0]["function_response"]
                        response = func_response.get("response", {})
                        # Serialize the response dict back to JSON string
                        return json.dumps(response), True
                
                # Handle Claude format: content is a list with tool_result blocks
                elif last_message.get("role") == "user" and isinstance(last_message.get("content"), list):
                    content_blocks = last_message["content"]
                    for block in content_blocks:
                        if block.get("type") == "tool_result":
                            return block.get("content", ""), True
                
                # Unexpected format - log warning
                # Note: This method is only called for Claude and Gemini providers
                logger.warning(
                    "Unexpected message format in execute_tool_call_for_collection. "
                    "Expected Claude or Gemini format. Message keys: %s, Message: %s",
                    list(last_message.keys()), last_message
                )
            
            return "", False
            
        except Exception as e:
            error_msg = f"Tool execution error: {str(e)}"
            logger.error("Error in execute_tool_call_for_collection: %s", error_msg, exc_info=True)
            error_msg_json = json.dumps({"error": error_msg})
            return error_msg_json, False

