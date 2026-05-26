"""MCP Client for LLM integration."""

import json
import logging
import os
import re
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Tuple

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .filters.base import BaseResultFilter
from .llm_providers.base import LLMProvider
from .utils import (
    MessageHandler,
    ToolExecutor,
    count_initial_input_tokens,
    count_tool_calls_tokens,
    count_tool_definitions_tokens,
    count_tokens,
    get_log_file_path,
    log_token_usage,
    log_query_run_summary,
)

# Import utils to set up logging (logging is configured in utils.py)
from . import utils  

logger = logging.getLogger(__name__)

class MCPClient:
    """
    MCP client that works with any LLM provider for our evaluation case against Cochrane reviews.
    
    This client connects to the MCP server, lists available tools, and uses
    the provided LLM provider to execute tools via the MCP protocol.
    """
    
    # Default allowed tools (can be overridden via constructor)
    DEFAULT_ALLOWED_TOOLS = {
        "serper_google_webpage_search",
        "jina_fetch_webpage_content",
        "semantic_scholar_snippet_search",
    }
    
    @classmethod
    def _get_default_mcp_server_path(cls) -> Path:
        """Get the default MCP server path relative to this file."""
        # Get the sciconharness directory (parent of mcp_client)
        sciconharness_dir = Path(__file__).parent.parent
        # Server is at sciconharness/mcp_server/main.py
        server_path = sciconharness_dir / "mcp_server" / "main.py"
        return server_path
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        mcp_server_path: Optional[Path] = None,
        allowed_tools: Optional[Set[str]] = None,
        enable_tool_calling: bool = True,
        enable_filtering: bool = True,
    ):
        """
        Initialize the MCP client.
        
        Args:
            llm_provider: LLM provider instance (OpenAIProvider, GeminiProvider, etc.)
            mcp_server_path: Path to MCP server script (default: auto-detected from class location)
            allowed_tools: Set of tool names to allow (default: DEFAULT_ALLOWED_TOOLS)
            enable_tool_calling: If False, disables MCP tool calling and uses LLM directly (default: True)
            enable_filtering: If False, disables result filtering even if filter is provided (default: True)
        """
        self.llm_provider = llm_provider
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack[bool | None]()
        self.mcp_server_path = mcp_server_path or self._get_default_mcp_server_path()
        self.allowed_tools = allowed_tools or self.DEFAULT_ALLOWED_TOOLS
        self.enable_tool_calling = enable_tool_calling
        self.enable_filtering = enable_filtering
        
        # Initialize tool executor (will be set up after connection)
        self.tool_executor: Optional[ToolExecutor] = None
        self.message_handler = MessageHandler()
        
        # Verify MCP server path exists (only if tool calling is enabled)
        if self.enable_tool_calling and not self.mcp_server_path.exists():
            raise FileNotFoundError(
                f"MCP server not found at {self.mcp_server_path}. "
                "Please ensure the server file exists."
            )
    
    async def connect_to_server(self):
        """Connect to the MCP server using stdio transport."""
        if not self.enable_tool_calling:
            logger.info("Tool calling disabled, skipping MCP server connection")
            return
        
        if not self.mcp_server_path.suffix == ".py":
            raise ValueError("MCP server must be a Python (.py) file")
        
        # Run as module: mcp_server/main.py -> mcp_server.main
        # Get parent directory (sciconharness) to add to PYTHONPATH
        # mcp_server_path is sciconharness/mcp_server/main.py
        # parent.parent is sciconharness/
        lib_dir = str(self.mcp_server_path.parent.parent)
        
        # Set PYTHONPATH so mcp_server module can be found
        env = os.environ.copy()
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{lib_dir}:{current_pythonpath}" if current_pythonpath else lib_dir
        
        # Pass log file path to server so it can log to the same file
        log_file_path = get_log_file_path()
        if log_file_path:
            env["MCP_SERVER_LOG_FILE"] = str(log_file_path)
        
        # Determine module name from path: mcp_server/main.py -> mcp_server.main
        server_module = str(self.mcp_server_path.parent.name) + "." + self.mcp_server_path.stem
        
        server_params = StdioServerParameters(
            command="python",
            args=["-m", server_module, "--transport", "stdio", "--log-level", "warning"],
            env=env,
        )
        
        # Establish stdio transport connection and create client session
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        
        await self.session.initialize()
        
        # List available tools and filter to allowed ones
        response = await self.session.list_tools()
        all_tools = response.tools
        self.available_tools = [
            tool for tool in all_tools if tool.name in self.allowed_tools
        ]
        
        tool_names = [tool.name for tool in self.available_tools]
        logger.info("Connected to MCP server")
        logger.info("Available tools: %s", tool_names)
        logger.info("Using LLM: %s (%s)", self.llm_provider.__class__.__name__, self.llm_provider.model)
        
        # Initialize tool executor after session is established
        self.tool_executor = ToolExecutor(
            session=self.session,
            llm_provider=self.llm_provider,
            enable_filtering=self.enable_filtering,
        )
    
    async def disconnect(self):
        """Disconnect from the MCP server."""
        if self.enable_tool_calling and self.session:
            await self.exit_stack.aclose()
            self.session = None
    
    # ============================================================================
    # Tool Execution (delegated to ToolExecutor)
    # ============================================================================
    
    async def _execute_tool_call(
        self, 
        tool_call: Dict[str, Any], 
        messages: List[Dict[str, Any]],
        input_tokens: int,
        tool_result_tokens_tracker: Optional[List[int]] = None,
        tool_usage_tracker: Optional[List[str]] = None,
        result_filter: Optional[BaseResultFilter] = None
    ) -> tuple[int, bool]:
        """Execute a single tool call and add result to messages."""
        if not self.tool_executor:
            raise RuntimeError("Tool executor not initialized. Call connect_to_server() first.")
        return await self.tool_executor.execute_tool_call(
            tool_call, messages, input_tokens,
            tool_result_tokens_tracker=tool_result_tokens_tracker,
            tool_usage_tracker=tool_usage_tracker,
            result_filter=result_filter
        )
    
    async def _execute_tool_call_for_collection(
        self,
        tool_call: Dict[str, Any],
        input_tokens: int,
        tool_result_tokens_tracker: Optional[List[int]] = None,
        tool_usage_tracker: Optional[List[str]] = None,
        result_filter: Optional[BaseResultFilter] = None
    ) -> tuple[str, bool]:
        """Execute a tool call and return the result string without adding to messages."""
        if not self.tool_executor:
            raise RuntimeError("Tool executor not initialized. Call connect_to_server() first.")
        return await self.tool_executor.execute_tool_call_for_collection(
            tool_call, input_tokens,
            tool_result_tokens_tracker=tool_result_tokens_tracker,
            tool_usage_tracker=tool_usage_tracker,
            result_filter=result_filter
        )
    
    def _extract_tool_call_info(self, tool_call: Dict[str, Any]) -> tuple[str, str, str]:
        """Extract tool call information from standardized format."""
        if not self.tool_executor:
            raise RuntimeError("Tool executor not initialized. Call connect_to_server() first.")
        return self.tool_executor.extract_tool_call_info(tool_call)
    
    # ============================================================================
    # Filter Management (delegated to MessageHandler)
    # ============================================================================
    
    def _create_query_filter(
        self,
        base_filter: Optional[BaseResultFilter],
        publication_date: Optional[str] = None
    ) -> Optional[BaseResultFilter]:
        """Create or combine filters for a query."""
        return self.message_handler.create_query_filter(base_filter, publication_date)
    
    # ============================================================================
    # Helper Functions
    # ============================================================================
    
    def _extract_search_results_from_response(self, response: Any) -> Optional[List[Dict[str, Any]]]:
        """
        Extract search_results from Perplexity API response and convert to list of dicts.
        
        Args:
            response: Response object from LLM provider (may have search_results attribute)
        
        Returns:
            List of dictionaries with 'title', 'url', 'date' keys, or None if not available
        """
        if not response or not hasattr(response, 'search_results'):
            return None
        
        search_results = response.search_results
        if not search_results:
            return None
        
        search_results_list = []
        for result in search_results:
            result_dict = {
                'title': getattr(result, 'title', None) if hasattr(result, 'title') else (result.get('title') if isinstance(result, dict) else None),
                'url': getattr(result, 'url', None) if hasattr(result, 'url') else (result.get('url') if isinstance(result, dict) else None),
                'date': getattr(result, 'date', None) if hasattr(result, 'date') else (result.get('date') if isinstance(result, dict) else None),
            }
            search_results_list.append(result_dict)
        
        return search_results_list
    
    def _has_triple_brackets(self, response_text: str) -> bool:
        """
        Check if response contains triple square brackets [[[...]]].
        
        Args:
            response_text: The response text to check
        
        Returns:
            True if triple brackets are found, False otherwise
        """
        if not response_text or not isinstance(response_text, str):
            return False
        
        pattern = r'\[\[\[(.*?)\]\]\]'
        matches = re.findall(pattern, response_text, re.DOTALL)
        return len(matches) > 0
    
    # ============================================================================
    # Main Query Processing
    # ============================================================================
    
    async def process_query(
        self, 
        query: str, 
        conversation_history: Optional[List[dict]] = None,
        result_filter: Optional[BaseResultFilter] = None,
        return_token_usage: bool = False,
        domain_filter: Optional[List[str]] = None,
    ) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """
        Process a query using the LLM and available MCP tools.
        
        Args:
            query: User query string
            conversation_history: Optional list of previous messages for context
            result_filter: Optional BaseResultFilter instance (or subclass) to filter tool results.
                          The filter determines which tools to filter via its should_filter_tool() method.
                          For example, CochraneResultFilter filters serper_google_webpage_search 
                          and semantic_scholar_snippet_search.
                          Note: Filtering is only applied if enable_filtering=True.
            return_token_usage: If True, returns tuple of (response, token_usage_dict). If False, returns just response.
                          
        Returns:
            Final response string from LLM, or tuple of (response, token_usage_dict) if return_token_usage=True
        """
        # Reset filtered URLs at the start of a new query
        # This ensures each query starts with a clean slate for URL-based filtering
        from .utils.utils import reset_filtered_urls
        reset_filtered_urls()
        
        # If tool calling is disabled, use LLM directly without tools
        if not self.enable_tool_calling:
            logger.info("Tool calling disabled, using LLM directly")
            messages = conversation_history.copy() if conversation_history else []
            messages.append({"role": "user", "content": query})
            
            # For Perplexity, extract and pass domain_filter, search_before_date_filter, cochrane_titles, and log_dir
            # BUT: Only pass these if filtering is enabled (skip if --no-enable-filtering is set)
            call_kwargs = {}
            from .llm_providers.perplexity_provider import PerplexityProvider
            if isinstance(self.llm_provider, PerplexityProvider):
                # Always pass enable_filtering flag so provider can suppress warnings when filtering is disabled
                call_kwargs['enable_filtering'] = self.enable_filtering
                
                # Only apply filtering if enable_filtering is True
                if self.enable_filtering:
                    from .llm_providers.perplexity_provider import _convert_publication_date_to_mm_dd_yyyy
                    from .utils.utils import get_log_file_path
                    
                    # Extract publication date and convert to MM/DD/YYYY format
                    search_before_date_filter = None
                    if result_filter and hasattr(result_filter, 'publication_date_cutoff') and result_filter.publication_date_cutoff:
                        pub_date_str = result_filter.publication_date_cutoff.strftime('%d %B %Y')
                        search_before_date_filter = _convert_publication_date_to_mm_dd_yyyy(pub_date_str)
                        if search_before_date_filter:
                            call_kwargs['search_before_date_filter'] = search_before_date_filter
                    
                    # Pass domain_filter if provided
                    if domain_filter:
                        call_kwargs['domain_filter'] = domain_filter
                    
                    # Extract Cochrane titles from result_filter if available (for iterative filtering)
                    if result_filter and hasattr(result_filter, 'title_filter_list') and result_filter.title_filter_list:
                        call_kwargs['cochrane_titles'] = list(result_filter.title_filter_list)
                    
                    # Pass result_filter directly for full Cochrane filtering mechanism
                    if result_filter:
                        call_kwargs['result_filter'] = result_filter
                    
                    # Get log directory path for saving filter list
                    log_file_path = get_log_file_path()
                    if log_file_path:
                        call_kwargs['log_dir'] = str(log_file_path.parent)
                else:
                    # Filtering disabled - don't pass any filter parameters
                    logger.info("Filtering disabled for Perplexity - skipping domain_filter, cochrane_titles, and iterative filtering")
            
            # Retry loop for triple bracket validation
            max_retries = 5
            response = None
            text_content = None
            response_text = None
            last_exception = None
            
            for retry_attempt in range(max_retries):
                try:
                    # Call LLM without tools
                    response, text_content, tool_calls, reasoning_summary = await self.llm_provider.call_llm(
                        messages, 
                        tools=None,
                        **call_kwargs
                    )
                    
                    response_text = text_content if text_content else "No response generated."
                    logger.debug("LLM call completed - text_content length: %d, response_text length: %d", 
                               len(text_content) if text_content else 0, len(response_text))
                    
                    # Check if response has triple brackets
                    if self._has_triple_brackets(response_text):
                        # Response is valid, break out of retry loop
                        logger.info("Response has triple brackets, validation passed")
                        break
                    else:
                        # Response missing triple brackets, retry
                        if retry_attempt < max_retries - 1:
                            logger.warning(
                                "Response missing triple brackets [[[...]]] (attempt %d/%d). Retrying...",
                                retry_attempt + 1, max_retries
                            )
                            print(f"Warning: Response missing triple brackets [[[...]]]. Retrying (attempt {retry_attempt + 1}/{max_retries})...")
                            # Continue to next retry
                        else:
                            # Max retries reached, log warning but use the response anyway
                            logger.warning(
                                "Response still missing triple brackets [[[...]]] after %d attempts. Using response anyway.",
                                max_retries
                            )
                            print(f"Warning: Response still missing triple brackets [[[...]]] after {max_retries} attempts. Using response anyway.")
                except Exception as e:
                    last_exception = e
                    logger.error(
                        "Error calling LLM (attempt %d/%d): %s",
                        retry_attempt + 1, max_retries, e, exc_info=True
                    )
                    if retry_attempt < max_retries - 1:
                        logger.info("Retrying LLM call...")
                    else:
                        logger.error("Failed to get response from LLM after %d attempts", max_retries)
                        raise
            
            if not response_text or response_text == "No response generated.":
                error_msg = "No response text generated from LLM"
                logger.error(error_msg)
                if last_exception:
                    raise last_exception
                raise RuntimeError(error_msg)
            
            # Extract search_results from Perplexity response if available
            search_results = self._extract_search_results_from_response(response)
            
            # Apply client-side filtering to search_results if domain_filter is provided AND filtering is enabled
            # This is a safety measure in case Perplexity API doesn't filter correctly
            if search_results and domain_filter and self.enable_filtering:
                from sciconharness.utils.perplexity_filtering import filter_search_results_by_domain
                original_count = len(search_results)
                search_results = filter_search_results_by_domain(search_results, domain_filter)
                filtered_count = original_count - len(search_results)
                if filtered_count > 0:
                    logger.warning(
                        "Client-side filter removed %d search results that matched domain filter. "
                        "This suggests Perplexity API may not be filtering correctly server-side.",
                        filtered_count
                    )
            
            # Extract publication date from filter if available
            publication_date = None
            if result_filter and hasattr(result_filter, 'publication_date_cutoff') and result_filter.publication_date_cutoff:
                publication_date = result_filter.publication_date_cutoff.strftime('%d %B %Y')
            
            # Extract token usage - use API counts for Gemini, estimate for others
            from .llm_providers import GeminiProvider
            is_gemini = isinstance(self.llm_provider, GeminiProvider)
            
            if is_gemini and response and hasattr(response, 'gemini_token_usage') and response.gemini_token_usage:
                # Use Gemini API's accurate token counts
                api_usage = response.gemini_token_usage
                input_tokens = api_usage.get('prompt_tokens', 0)
                output_tokens = api_usage.get('completion_tokens', 0)
                thoughts_tokens = api_usage.get('thoughts_tokens', 0)
                total_tokens = api_usage.get('total_tokens', 0) or (input_tokens + output_tokens)
                logger.info("Gemini API token counts - prompt: %d → input_tokens: %d, candidates: %d → output_tokens: %d, thoughts: %d → thoughts_tokens: %d, total: %d",
                           api_usage.get('prompt_tokens', 0), input_tokens,
                           api_usage.get('completion_tokens', 0), output_tokens,
                           api_usage.get('thoughts_tokens', 0), thoughts_tokens,
                           total_tokens)
            else:
                # Estimate token usage for non-Gemini providers
                input_tokens = count_initial_input_tokens(messages)
                output_tokens = count_tokens(response_text)
                total_tokens = input_tokens + output_tokens
                thoughts_tokens = 0
            
            # Log query run summary before returning final answer
            log_query_run_summary(
                query=query,
                response_text=response_text,
                messages=messages,
                iterations=1,
                tool_call_count=0,
                total_tokens=total_tokens,
                publication_date=publication_date,
                result_filter=result_filter
            )
            
            if return_token_usage:
                token_usage = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "iterations": 1,
                    "tool_call_count": 0,
                    "tool_usage": {},
                    "initial_message_tokens": input_tokens,
                    "tool_def_tokens": 0,
                    "tool_result_tokens": 0,
                    "tool_call_tokens": 0,
                }
                # Add Gemini-specific token counts
                if is_gemini:
                    token_usage["thoughts_tokens"] = thoughts_tokens
                
                # Apply title-based filtering using result_filter if available (for Cochrane reviews)
                # Only apply if filtering is enabled
                if search_results and result_filter and self.enable_filtering:
                    from .filters.cochrane import CochraneResultFilter
                    if isinstance(result_filter, CochraneResultFilter):
                        original_count = len(search_results)
                        filtered_search_results = []
                        for result in search_results:
                            title = result.get('title', '')
                            url = result.get('url', '')
                            result_date = result.get('date', None)
                            
                            if title and url:
                                should_filter, reason = result_filter._should_filter_item(
                                    title=title,
                                    urls=[url],
                                    publication_date=result_date
                                )
                                if should_filter:
                                    logger.info(
                                        "CLIENT-SIDE TITLE FILTER: Excluding search result matching Cochrane filter (reason: %s): %s -> %s",
                                        reason, title, url
                                    )
                                    continue
                            filtered_search_results.append(result)
                        
                        search_results = filtered_search_results
                        title_filtered_count = original_count - len(search_results)
                        if title_filtered_count > 0:
                            logger.info(
                                "CLIENT-SIDE TITLE FILTER: Filtered out %d/%d search results based on Cochrane title filter",
                                title_filtered_count, original_count
                            )
                
                # Add search_results if available (for Perplexity)
                if search_results is not None:
                    token_usage["search_results"] = search_results
                return response_text, token_usage
            return response_text
        
        # Tool calling is enabled - proceed with MCP workflow
        if not self.session:
            raise RuntimeError("Not connected to MCP server. Call connect_to_server() first.")
        
        # Initialize
        messages = conversation_history.copy() if conversation_history else []
        messages.append({"role": "user", "content": query})
        available_tools = self.llm_provider.format_tools(self.available_tools)
        
        # Disable filtering if toggle is off
        if not self.enable_filtering:
            result_filter = None
            logger.info("Filtering disabled by toggle")
        
        # Initialize token tracking
        initial_message_tokens = count_initial_input_tokens(messages)
        input_tokens = 0  # Will be set based on provider type
        output_tokens = 0
        tool_def_tokens = count_tool_definitions_tokens(available_tools)
        
        # Track breakdown for detailed logging
        total_tool_def_tokens = 0
        tool_result_tokens_list = []  # List to collect individual tool result token counts
        total_reasoning_tokens = 0
        total_tool_call_tokens = 0
        tool_usage_list = []  # List to track which tools were called
        total_cached_content_tokens = 0  # Track cached content tokens for Gemini
        # Track final iteration's Gemini-specific token counts for result.json
        final_gemini_prompt_tokens = 0
        final_gemini_candidates_tokens = 0
        final_gemini_thoughts_tokens = 0
        final_gemini_total_tokens = 0
        
        # Check if using Gemini (which provides accurate API token counts)
        from .llm_providers import GeminiProvider
        is_gemini = isinstance(self.llm_provider, GeminiProvider)
        if not is_gemini:
            # For non-Gemini providers, initialize with initial message tokens
            input_tokens = initial_message_tokens
        
        # Initalizing variables to query with tool calling
        final_text = []
        iteration = 0
        tool_call_count = 0
        
        while True:
            iteration += 1
            logger.info("Starting iteration %d", iteration)
            
            # Count tool definitions as input tokens
            # NOTE: For non-Gemini providers, tools are sent with EVERY API call
            # For Gemini, prompt_token_count already includes tools, so we don't add them separately
            if not is_gemini and available_tools:
                input_tokens += tool_def_tokens
                total_tool_def_tokens += tool_def_tokens
            
            # Log message count and approximate size before LLM call
            # Calculate approximate total size in characters
            total_chars = sum(len(str(msg.get("content", ""))) + len(str(msg.get("output", ""))) for msg in messages)
            logger.info("Calling LLM with %d messages (approx %d input tokens, ~%d chars)", len(messages), input_tokens, total_chars)
            
            # Call LLM (tools are passed and sent with this API call)
            response, text_content, tool_calls, reasoning_summary = await self.llm_provider.call_llm(messages, tools=available_tools)
            
            logger.info("LLM call completed - text_content: %s, tool_calls: %d", 
                       "present" if text_content else "None", 
                       len(tool_calls) if tool_calls else 0)
            
            # Use API's token counts for Gemini instead of manual counting
            # According to Gemini API docs: prompt_token_count includes ALL input tokens for that specific API call
            # (messages, system instructions, tools). Each API call's prompt_token_count includes the full
            # conversation history up to that point, so we should use the LAST iteration's prompt_token_count
            # as the total input tokens, not sum them (which would double-count).
            if hasattr(response, 'gemini_token_usage') and response.gemini_token_usage:
                api_usage = response.gemini_token_usage
                # Use API's token counts directly
                # prompt_token_count already includes: messages, system instructions, and tools
                iteration_input_tokens = api_usage.get('prompt_tokens', 0)
                iteration_output_tokens = api_usage.get('completion_tokens', 0)
                iteration_thoughts_tokens = api_usage.get('thoughts_tokens', 0)
                iteration_cached_tokens = api_usage.get('cached_content_tokens', 0)
                iteration_total_tokens = api_usage.get('total_tokens', 0)
                
                # Store final iteration's values for result.json
                final_gemini_prompt_tokens = iteration_input_tokens
                final_gemini_candidates_tokens = iteration_output_tokens
                final_gemini_thoughts_tokens = iteration_thoughts_tokens
                final_gemini_total_tokens = iteration_total_tokens
                
                # For Gemini, prompt_token_count includes the full prompt for that API call
                # (all conversation history + tools). So we should use the current iteration's
                # prompt_token_count as the total input tokens (it includes everything up to this point).
                # We accumulate output tokens and thoughts tokens across iterations.
                input_tokens = iteration_input_tokens  # Use current iteration's prompt count (includes all history)
                output_tokens += iteration_output_tokens
                total_reasoning_tokens += iteration_thoughts_tokens
                total_cached_content_tokens += iteration_cached_tokens
                
                # Track tool definition tokens for logging (even though they're included in prompt_token_count)
                # This is just for reporting purposes - estimate based on tool definitions
                if available_tools and iteration == 1:
                    total_tool_def_tokens = tool_def_tokens
                
                logger.debug("Using Gemini API token counts - input: %d (full prompt), output: %d, thoughts: %d, cached: %d", 
                           iteration_input_tokens, iteration_output_tokens, iteration_thoughts_tokens, iteration_cached_tokens)
            else:
                # Fall back to manual counting for other providers
                # Check if done
                if text_content and not tool_calls:
                    output_tokens += count_tokens(text_content)
                        
            if text_content and not tool_calls:
                final_text.append(text_content)
                
                # Preserve thinking blocks from Claude response if present
                assistant_msg = {"role": "assistant", "content": text_content}
                if hasattr(response, 'thinking_blocks') and response.thinking_blocks:
                    assistant_msg["thinking_blocks"] = response.thinking_blocks
                    logger.debug("Preserving %d thinking blocks in assistant message", len(response.thinking_blocks))
                
                messages.append(assistant_msg)
                break
            
            # Handle tool calls (may also include text content)
            if tool_calls:
                logger.info('Tool Calls: %s', tool_calls)
                logger.info('Reasoning Summary: %s', reasoning_summary)
                
                # Count and append reasoning summary
                if reasoning_summary:
                    reasoning_tokens = count_tokens(reasoning_summary)
                    input_tokens += reasoning_tokens
                    total_reasoning_tokens += reasoning_tokens
                    messages.append({
                        "role": "assistant",
                        "content": reasoning_summary
                    })
                    logger.debug("Reasoning summary tokens: %d", reasoning_tokens)
                
                # Count and append tool call information
                tool_call_tokens = count_tool_calls_tokens(tool_calls)
                input_tokens += tool_call_tokens
                total_tool_call_tokens += tool_call_tokens
                
                # Preserve thinking blocks from Claude response if present
                assistant_msg = {
                    "role": "assistant",
                    "tool_calls": tool_calls
                }
                if hasattr(response, 'thinking_blocks') and response.thinking_blocks:
                    assistant_msg["thinking_blocks"] = response.thinking_blocks
                    logger.debug("Preserving %d thinking blocks in assistant message with tool calls", len(response.thinking_blocks))
                
                messages.append(assistant_msg)
                logger.debug("Tool call tokens: %d", tool_call_tokens)
                
                # Execute all tool calls
                # For Claude and Gemini, we need to collect all results and format them together
                # Both require ALL function calls from an assistant message to have
                # corresponding function responses in the NEXT user message, even for single tool calls
                # Gemini specifically requires the number of function response parts to equal the number of function call parts
                supports_multiple = hasattr(self.llm_provider, 'format_multiple_tool_response_message')
                multiple_tool_results = [] if supports_multiple else None
                
                for idx, tool_call in enumerate(tool_calls):
                    tool_call_count += 1
                    # Apply filter only if filtering is enabled
                    effective_filter = result_filter if self.enable_filtering else None
                    
                    if multiple_tool_results is not None:
                        # For Claude and Gemini: collect all results first (even single tool calls)
                        # IMPORTANT: Both require ALL function calls to have corresponding function responses,
                        # even if the tool execution failed. We must always add a result for every tool_call.
                        tool_result_str, success = await self._execute_tool_call_for_collection(
                            tool_call, input_tokens,
                            tool_result_tokens_tracker=tool_result_tokens_list,
                            tool_usage_tracker=tool_usage_list,
                            result_filter=effective_filter
                        )
                        # Always add result, even if execution failed (required for API compliance)
                        tool_name, _, tool_call_id = self._extract_tool_call_info(tool_call)
                        # Extract thought signature from first tool call if available (for Gemini)
                        # According to docs: "Thought signatures should always be used with function calling for best results."
                        thought_sig = tool_call.get("_thought_signature_bytes") if idx == 0 else None
                        multiple_tool_results.append((tool_call_id, tool_name, tool_result_str, thought_sig))
                        if not success:
                            logger.warning(
                                "Tool call %s (%s) failed but result added for Claude API compliance",
                                tool_call_id, tool_name
                            )
                    else:
                        # Standard execution for non-Claude/Gemini providers (e.g., OpenAI): add result to messages immediately
                        input_tokens, _ = await self._execute_tool_call(
                            tool_call, messages, input_tokens, 
                            tool_result_tokens_tracker=tool_result_tokens_list,
                            tool_usage_tracker=tool_usage_list,
                            result_filter=effective_filter
                        )
                
                # For Claude and Gemini: format all results together in a single user message
                # Both require ALL function calls to have corresponding function responses.
                # Gemini specifically requires the number of function response parts to equal the number of function call parts.
                # We must ensure we have results for every tool_call.
                if multiple_tool_results is not None:
                    if len(multiple_tool_results) != len(tool_calls):
                        logger.error(
                            "Mismatch: %d tool calls but %d tool results. This will cause API error (Claude/Gemini require matching counts).",
                            len(tool_calls), len(multiple_tool_results)
                        )
                        # This should never happen with the fix above, but add error results for missing ones
                        tool_call_ids_in_results = {result[0] for result in multiple_tool_results}
                        for tool_call in tool_calls:
                            # Extract tool_call_id using the same method as in the main loop
                            tool_name, _, tool_call_id = self._extract_tool_call_info(tool_call)
                            if tool_call_id not in tool_call_ids_in_results:
                                error_result = json.dumps({"error": "Tool result missing - this should not happen"})
                                # Use 4-tuple format for consistency (thought_sig is None for error cases)
                                multiple_tool_results.append((tool_call_id, tool_name, error_result, None))
                                logger.warning("Added error result for missing tool_call_id: %s", tool_call_id)
                    
                    if len(multiple_tool_results) > 0:
                        tool_response = self.llm_provider.format_multiple_tool_response_message(multiple_tool_results)
                        messages.append(tool_response)
                        # Count tokens for all results
                        # Handle both 3-tuple and 4-tuple formats (with/without thought signature)
                        total_result_tokens = sum(count_tokens(result[2] if len(result) >= 3 else result[1]) for result in multiple_tool_results)
                        input_tokens += total_result_tokens
                        logger.debug(
                            "Added %d tool result(s) in single message for parallel tool calls (%d tokens)",
                            len(multiple_tool_results), total_result_tokens
                        )
                    else:
                        logger.error(
                            "No tool results collected for %d tool calls. This will cause Claude API error.",
                            len(tool_calls)
                        )
                
                continue
            else:
                break
        
        # Calculate total tool result tokens from tracker
        total_tool_result_tokens = sum(tool_result_tokens_list)
        
        # Log token usage with breakdown
        log_token_usage(
            input_tokens, output_tokens, iteration, tool_call_count,
            initial_message_tokens=initial_message_tokens,
            tool_def_tokens=total_tool_def_tokens,
            tool_result_tokens=total_tool_result_tokens,
            reasoning_tokens=total_reasoning_tokens,
            tool_call_tokens=total_tool_call_tokens
        )
        
        response_text = "\n".join(final_text) if final_text else "No response generated."
        
        # Extract publication date from filter if available
        publication_date = None
        if result_filter and hasattr(result_filter, 'publication_date_cutoff') and result_filter.publication_date_cutoff:
            publication_date = result_filter.publication_date_cutoff.strftime('%d %B %Y')
        
        # Log query run summary before returning final answer
        log_query_run_summary(
            query=query,
            response_text=response_text,
            messages=messages,
            iterations=iteration,
            tool_call_count=tool_call_count,
            total_tokens=input_tokens + output_tokens,
            publication_date=publication_date,
            result_filter=result_filter
        )
        
        if return_token_usage:
            # Count tool usage
            tool_usage_counts = {}
            for tool_name in tool_usage_list:
                tool_usage_counts[tool_name] = tool_usage_counts.get(tool_name, 0) + 1
            
            # For Gemini with tool calling, use the final API token counts instead of accumulated values
            # This ensures result.json matches the actual API token counts shown in terminal output
            if is_gemini and final_gemini_total_tokens > 0:
                # Use final iteration's API token counts (these match what's printed in terminal)
                token_usage = {
                    "input_tokens": final_gemini_prompt_tokens,
                    "output_tokens": final_gemini_candidates_tokens,
                    "total_tokens": final_gemini_total_tokens,
                    "iterations": iteration,
                    "tool_call_count": tool_call_count,
                    "tool_usage": tool_usage_counts,
                    "initial_message_tokens": initial_message_tokens,
                    "tool_def_tokens": total_tool_def_tokens,
                    "tool_result_tokens": total_tool_result_tokens,
                    "tool_call_tokens": total_tool_call_tokens,
                    "thoughts_tokens": final_gemini_thoughts_tokens,
                }
                # Add cached content tokens for Gemini if available
                if total_cached_content_tokens > 0:
                    token_usage["cached_content_tokens"] = total_cached_content_tokens
            else:
                # For non-Gemini providers or Gemini without tool calling, use accumulated values
                token_usage = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "iterations": iteration,
                    "tool_call_count": tool_call_count,
                    "tool_usage": tool_usage_counts,
                    "initial_message_tokens": initial_message_tokens,
                    "tool_def_tokens": total_tool_def_tokens,
                    "tool_result_tokens": total_tool_result_tokens,
                    "tool_call_tokens": total_tool_call_tokens,
                }
                # Add Gemini-specific token counts
                if is_gemini:
                    token_usage["thoughts_tokens"] = total_reasoning_tokens
                # Add cached content tokens for Gemini if available
                if total_cached_content_tokens > 0:
                    token_usage["cached_content_tokens"] = total_cached_content_tokens
            # Extract search_results from last response if available (for Perplexity)
            # Note: In tool-calling mode, we use the last response from the LLM
            search_results = self._extract_search_results_from_response(response)
            
            # Apply client-side filtering to search_results if domain_filter is provided AND filtering is enabled
            # This is a safety measure in case Perplexity API doesn't filter correctly
            if search_results and domain_filter and self.enable_filtering:
                from sciconharness.utils.perplexity_filtering import filter_search_results_by_domain
                original_count = len(search_results)
                search_results = filter_search_results_by_domain(search_results, domain_filter)
                filtered_count = original_count - len(search_results)
                if filtered_count > 0:
                    logger.warning(
                        "Client-side filter removed %d search results that matched domain filter. "
                        "This suggests Perplexity API may not be filtering correctly server-side.",
                        filtered_count
                    )
            
            # Apply title-based filtering using result_filter if available (for Cochrane reviews)
            # This catches cases where domain filtering can't work (e.g., PubMed/PMC URLs)
            # NOTE: This is for logging only - we do NOT remove results from the final output
            # Only apply if filtering is enabled
            if search_results and result_filter and self.enable_filtering:
                from .filters.cochrane import CochraneResultFilter
                if isinstance(result_filter, CochraneResultFilter):
                    original_count = len(search_results)
                    filtered_count = 0
                    for result in search_results:
                        title = result.get('title', '')
                        url = result.get('url', '')
                        result_date = result.get('date', None)
                        
                        if title and url:
                            should_filter, reason = result_filter._should_filter_item(
                                title=title,
                                urls=[url],
                                publication_date=result_date
                            )
                            if should_filter:
                                # Log what would be filtered, but don't actually remove it
                                logger.info(
                                    "CLIENT-SIDE TITLE FILTER: Would exclude search result matching Cochrane filter (reason: %s): %s -> %s",
                                    reason, title, url
                                )
                                filtered_count += 1
                    
                    if filtered_count > 0:
                        logger.info(
                            "CLIENT-SIDE TITLE FILTER: Would filter out %d/%d search results based on Cochrane title filter (keeping all in final output)",
                            filtered_count, original_count
                        )
            
            if search_results is not None:
                token_usage["search_results"] = search_results
            return response_text, token_usage
        
        return response_text
    
    async def chat_loop(
        self, 
        result_filter: Optional[BaseResultFilter] = None,
        interactive: bool = True,
        query: Optional[str] = None,
        publication_date: Optional[str] = None
    ):
        """Run a chat loop (interactive or non-interactive).
        
        Args:
            result_filter: Optional BaseResultFilter instance to filter tool results.
                          Filters search results from serper_google_webpage_search 
                          and semantic_scholar_snippet_search.
            interactive: If True, prompts for user input. If False, uses provided parameters.
            query: Query string (only used if interactive=False)
            publication_date: Publication date cutoff (only used if interactive=False)
        """
        provider_name = self.llm_provider.__class__.__name__
        print("\n" + "=" * 60)
        print(f"{provider_name} MCP Client")
        print("=" * 60)
        if interactive:
            print("Type your queries or 'quit' to exit.")
        else:
            print("Non-interactive mode.")
        print("Available tools: Google Search, JINA Browse, Semantic Scholar")
        if result_filter:
            print(f"Result filter: {result_filter.__class__.__name__}")
        print("=" * 60)
        
        conversation_history = []
        
        while True:
            try:
                if interactive:
                    query_input = input("\n💬 Query: ").strip()
                    
                    if query_input.lower() in ["quit", "exit", "q"]:
                        print("\n👋 Goodbye!")
                        break
                    
                    if not query_input:
                        continue
                    
                    query = query_input
                    # Ask for publication date cutoff
                    date_input = input("📅 Publication date cutoff (press Enter to skip): ").strip()
                else:
                    # Non-interactive mode: use provided parameters
                    if not query:
                        print("Error: Query is required in non-interactive mode.")
                        break
                    
                    date_input = publication_date or ""
                
                # Create or combine filters for this query
                query_filter = self._create_query_filter(
                    result_filter,
                    publication_date=date_input if date_input else None
                )
                
                # Print filter status
                if date_input:
                    print(f"  ✓ Filtering items published after: '{date_input}'")
                if not date_input:
                    print("  ✓ No additional filters")
                
                print("\n🤔 Processing...")
                response = await self.process_query(
                    query, conversation_history, 
                    result_filter=query_filter
                )
                
                print(f"\n🤖 Response:\n{response}")
                
                # Update conversation history
                conversation_history.append({"role": "user", "content": query})
                conversation_history.append({"role": "assistant", "content": response})
                
                # In non-interactive mode, exit after processing one query
                if not interactive:
                    break
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error("Error in chat loop: %s", str(e), exc_info=True)
                print(f"\n✗ Error: {str(e)}")
                # In non-interactive mode, exit on error
                if not interactive:
                    break

