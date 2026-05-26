"""OpenAI GPT provider implementation.

This provider also supports OpenAI's Deep Research models via the responses API in background mode.
For Deep research models use remote MCP servers for tool calling and do NOT use local context management.

API Request Parameters (default mode):
--------------------------------------
- model: "o4-mini-deep-research-2025-06-26"
- input: The query string (single message, no conversation history)
- instructions: RESEARCH_ASSISTANT_PROMPT (system prompt for research assistant behavior)
- tools: List of MCP tool configurations (required for deep research)
- background: True (enables background/async processing mode)
- max_tool_calls: Optional[int] (default: 30 when called from query_batch/query_single, None otherwise)

MCP Tools Configuration (default):
----------------------------------
Tools are configured via build_mcp_tools_for_deep_research() with:
- SerperJinaMCP:
  * type: "mcp"
  * server_label: "SerperJinaMCP"
  * server_url: "{SERPER_SERVER_BASE}/mcp" (from env or default ngrok URL)
  * allowed_tools: ["search", "fetch"]
  * require_approval: "never"
  * authorization: MCP_AUTH_TOKEN (from env or empty string)
- SemanticScholarJinaMCP:
  * type: "mcp"
  * server_label: "SemanticScholarJinaMCP"
  * server_url: "{SEMANTIC_SERVER_BASE}/mcp" (from env or default ngrok URL)
  * allowed_tools: ["search", "fetch"]
  * require_approval: "never"
  * authorization: MCP_AUTH_TOKEN (from env or empty string)

"""

import asyncio
import json
import logging
import os
import re
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI, AzureOpenAI
from openai import BadRequestError
from .base import LLMProvider, ContextLengthExceededError
from ..prompts import RESEARCH_ASSISTANT_PROMPT

logger = logging.getLogger(__name__)

# ============================================================================
# Helper Functions for Rate Limit Handling
# ============================================================================

def parse_rate_limit_wait_time(error_message: str) -> Optional[float]:
    """
    Parse the wait time from a rate limit error message.
    Example: "Please try again in 3.646s" -> 3.646
    """
    match = re.search(r'Please try again in ([\d.]+)s', error_message)
    if match:
        return float(match.group(1))
    return None

def is_rate_limit_error(error: Any) -> bool:
    """Check if an error is a rate limit error."""
    if error is None:
        return False
    error_str = str(error).lower()
    return "rate limit" in error_str or "rate_limit" in error_str or "429" in error_str

def _get_tool_attr(tool: Any, attr: str, default: Any = None) -> Any:
    """Get attribute from tool (handles both object and dict)."""
    return getattr(tool, attr, None) or (tool.get(attr, default) if isinstance(tool, dict) else default)

def messages_to_input_list(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert standard message format to responses API input list format."""
    input_list = []
    for msg in messages:
        # Check if this is a function_call_output message (no role, just type)
        if msg.get("type") == "function_call_output":
            # Tool results are already in responses API format from format_tool_response_message
            input_list.append(msg)
            continue
        
        role = msg.get("role")
        
        if role == "user":
            input_list.append({
                "role": "user",
                "content": msg.get("content", "")
            })
        elif role == "assistant":
            content = msg.get("content", "")
            if content:
                input_list.append({
                    "role": "assistant",
                    "content": content
                })
            # Convert tool_calls to function_call format
            for tool_call in msg.get("tool_calls", []):
                func = tool_call.get("function", {})
                input_list.append({
                    "type": "function_call",
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", "{}"),
                    "call_id": tool_call.get("id", "")
                })
        elif role == "tool":
            # Tool results are already in responses API format from format_tool_response_message
            input_list.append(msg)
    
    return input_list


def parse_gpt5_response(response: Any) -> Tuple[Optional[str], List[Dict[str, Any]], Optional[str]]:
    """Parse GPT-5 responses API response to standard format.
    
    This function extracts reasoning summaries from GPT-5 responses to preserve
    reasoning continuity across tool calls. Reasoning summaries are critical for
    maintaining the model's reasoning state when tools are invoked.
    
    Returns:
        Tuple of (text_content, tool_calls, reasoning_summary)
    """
    text_content = str(response.output_text) if response.output_text else None
    tool_calls = []
    reasoning_summaries = []
    
    # Track items processed for logging
    reasoning_items_found = 0
    function_call_items_found = 0
    
    for item in response.output or []:
        item_type = getattr(item, 'type', None)
        if item_type == "function_call":
            function_call_items_found += 1
            arguments = getattr(item, 'arguments', '{}')
            # Normalize arguments: handle None, empty string, or dict
            if arguments is None or (isinstance(arguments, str) and not arguments.strip()):
                arguments = '{}'
            elif not isinstance(arguments, str):
                # Convert dict/object to JSON string
                arguments = json.dumps(arguments) if arguments else '{}'
            
            tool_calls.append({
                "id": getattr(item, 'call_id', ''),
                "type": "function",
                "function": {
                    "name": getattr(item, 'name', ''),
                    "arguments": arguments,
                },
            })
        elif item_type == "reasoning":
            # Extract summary text from reasoning items
            # This is critical for preserving reasoning state across tool calls
            reasoning_items_found += 1
            summary_list = getattr(item, 'summary', [])
            if not summary_list:
                logger.debug("Found reasoning item but summary list is empty")
            for summary in summary_list:
                summary_text = getattr(summary, 'text', None)
                if summary_text:
                    reasoning_summaries.append(summary_text)
                    logger.debug("Extracted reasoning summary (length: %d chars)", len(summary_text))
                else:
                    logger.debug("Found summary item but text is None or empty")
    
    reasoning_content = "\n\n".join(reasoning_summaries) if reasoning_summaries else None
    
    # Log extraction results for verification
    if tool_calls:
        if reasoning_content:
            logger.info(
                "✓ Extracted reasoning summary for %d tool call(s): %d reasoning item(s) found, "
                "summary length: %d chars",
                len(tool_calls), reasoning_items_found, len(reasoning_content)
            )
        else:
            if reasoning_items_found > 0:
                logger.warning(
                    "No reasoning summary extracted for %d tool call(s): reasoning item(s) found (%d) "
                    "but summary was empty — the LLM may not have included a summary "
                    "(reasoning_summary='auto' allows the model to omit it).",
                    len(tool_calls), reasoning_items_found
                )
            else:
                logger.warning(
                    "No reasoning summary extracted for %d tool call(s): no reasoning items found "
                    "in the response — the LLM may not have included reasoning.",
                    len(tool_calls)
                )
    
    return text_content, tool_calls, reasoning_content


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider using responses API."""
    
    def __init__(
        self, 
        model: str, 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        reasoning_effort: str = "high",
        verbosity: str = "medium",
        reasoning_summary: str = "auto"
    ):
        super().__init__(model, api_key)
        
        api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or AZURE_OPENAI_KEY environment variable.")

        # If base_url is provided, use Azure OpenAI
        if base_url:
            api_version = api_version or os.getenv("OPENAI_API_VERSION", "2025-04-01-preview")
            self.client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_url,
                api_version=api_version
            )
        else:
            # Standard OpenAI API
            self.client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")
        
        # Set defaults based on model type
        # For GPT-5.1: verbosity=medium, reasoning_summary=auto, reasoning_effort=high
        # Note: is_gpt51 could be used for model-specific defaults in the future
        
        self.verbosity = verbosity 
        self.reasoning_summary = reasoning_summary 
        self.reasoning_effort = reasoning_effort 
    
    def format_tools(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """Format tools for OpenAI responses API function calling."""
        formatted = []
        for tool in tools:
            tool_name = _get_tool_attr(tool, 'name')
            if not tool_name:
                continue
            
            tool_description = _get_tool_attr(tool, 'description')
            schema = _get_tool_attr(tool, 'inputSchema') or (_get_tool_attr(tool, 'parameters', {}))
            
            if not isinstance(schema, dict):
                schema = {}
            if "type" not in schema:
                schema = {"type": "object", "properties": schema} if schema else {"type": "object", "properties": {}}
            
            formatted.append({
                "type": "function",
                "name": tool_name,
                "description": tool_description or f"Tool: {tool_name}",
                "parameters": schema,
            })
        
        return formatted
    
    async def call_llm_background(
        self,
        query: str,
        mcp_tools: Optional[List[Dict[str, Any]]] = None,
        max_retries: int = 5,
        initial_retry_delay: float = 5.0,
        poll_interval: int = 5,
        max_tool_calls: Optional[int] = None,
    ) -> Tuple[Any, Optional[str], List[Dict[str, Any]], Optional[str]]:
        """Call OpenAI responses API in background mode with polling and rate limit handling.

        This method is specifically for o4-mini-deep-research and does NOT do context management for tool-calling.
        It simply runs a single query in background mode, polls for completion, and handles rate limits.
        
        Args:
            query: The query string (single message, no conversation history)
            mcp_tools: List of MCP tool configurations (required for o4-mini-deep-research)
            max_retries: Maximum number of retry attempts (default: 5)
            initial_retry_delay: Base delay in seconds for retries (default: 5.0)
            poll_interval: Interval in seconds between polling attempts (default: 5)
            max_tool_calls: Maximum number of tool calls allowed (default: None, no limit; typically 30 from query_batch/query_single)
        
        Returns:
            Tuple of (response, text_content, tool_calls, reasoning_summary)
            - response: Raw OpenAI responses API response object
            - text_content: Extracted text content from response
            - tool_calls: List of tool calls made during research
            - reasoning_summary: Reasoning summary if available
        """
        # Use MCP tools (required for o4-mini-deep-research)
        tools_to_use = mcp_tools if mcp_tools else []
        
        loop = asyncio.get_event_loop()
        max_poll_retries = max_retries
        
        # Initial API call with retry logic
        # For o4-mini-deep-research, use simple format: input (string), instructions, model, tools, background
        resp = None
        for attempt in range(max_retries):
            try:
                # Build API call parameters
                # IMPORTANT: RESEARCH_ASSISTANT_PROMPT is used here to ensure proper formatting
                api_params = {
                    "model": self.model,
                    "input": query,
                    "instructions": RESEARCH_ASSISTANT_PROMPT,
                    "tools": tools_to_use,
                    "background": True,
                }
                # Add max_tool_calls if specified
                if max_tool_calls is not None:
                    api_params["max_tool_calls"] = max_tool_calls
                
                # Log that we're using RESEARCH_ASSISTANT_PROMPT
                logger.info("=" * 80)
                logger.info("USING RESEARCH_ASSISTANT_PROMPT FOR INSTRUCTIONS")
                logger.info("=" * 80)
                logger.info(f"Instructions length: {len(RESEARCH_ASSISTANT_PROMPT)} characters")
                logger.info(f"Instructions preview: {RESEARCH_ASSISTANT_PROMPT[:300]}...")
                logger.info("")
                
                # Use a function instead of lambda to avoid closure issues
                def make_api_call():
                    return self.client.responses.create(**api_params)
                
                resp = await loop.run_in_executor(None, make_api_call)
                break  # Success, exit retry loop
            except Exception as e:
                error_str = str(e)
                if is_rate_limit_error(e) and attempt < max_retries - 1:
                    wait_time = parse_rate_limit_wait_time(error_str)
                    if wait_time is None:
                        wait_time = initial_retry_delay * (2 ** attempt)
                    else:
                        wait_time = wait_time + 1.0
                    
                    logger.warning("=" * 80)
                    logger.warning("RATE LIMIT ERROR ON INITIAL API CALL")
                    logger.warning("=" * 80)
                    logger.warning(f"Attempt {attempt + 1}/{max_retries}")
                    logger.warning(f"Error: {error_str[:500]}")
                    logger.warning(f"Waiting {wait_time:.2f} seconds before retry...")
                    logger.warning("=" * 80)
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Non-rate-limit error or last attempt
                    raise
        
        if resp is None:
            raise Exception("Failed to create API request after all retries")
        
        logger.info("=" * 80)
        logger.info("OPENAI API REQUEST SUBMITTED (BACKGROUND MODE)")
        logger.info("=" * 80)
        logger.info(f"Response ID: {resp.id}")
        logger.info(f"Initial Status: {resp.status}")
        logger.info("")
        
        # Poll for completion
        logger.info("Polling for completion (background mode)...")
        logger.info(f"Poll interval: {poll_interval} seconds")
        logger.info("")
        
        while resp.status in {"queued", "in_progress"}:
            logger.info(f"Current status: {resp.status} - waiting {poll_interval} seconds...")
            await asyncio.sleep(poll_interval)
            
            # Retrieve with retry logic for rate limits
            for poll_attempt in range(max_poll_retries):
                try:
                    resp = await loop.run_in_executor(
                        None,
                        lambda: self.client.responses.retrieve(resp.id)
                    )
                    break
                except Exception as e:
                    error_str = str(e)
                    if is_rate_limit_error(e) and poll_attempt < max_poll_retries - 1:
                        wait_time = parse_rate_limit_wait_time(error_str)
                        if wait_time is None:
                            wait_time = initial_retry_delay * (2 ** poll_attempt)
                        else:
                            wait_time = wait_time + 1.0
                        
                        logger.warning("=" * 80)
                        logger.warning("RATE LIMIT ERROR WHILE POLLING")
                        logger.warning("=" * 80)
                        logger.warning(f"Poll attempt {poll_attempt + 1}/{max_poll_retries}")
                        logger.warning(f"Error: {error_str[:500]}")
                        logger.warning(f"Waiting {wait_time:.2f} seconds before retry...")
                        logger.warning("=" * 80)
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        # Non-rate-limit error or last attempt
                        raise
        
        logger.info(f"Final status: {resp.status}")
        logger.info("")
        
        if resp.status == "completed":
            logger.info("=" * 80)
            logger.info("OPENAI API RESPONSE RECEIVED")
            logger.info("=" * 80)
            # Parse response
            text_content, tool_calls, reasoning_summary = parse_gpt5_response(resp)
            
            # Log response details for debugging
            logger.info(f"Response text length: {len(text_content) if text_content else 0} characters")
            logger.info(f"Tool calls made: {len(tool_calls)}")
            if text_content:
                # Log first and last 200 chars to help debug formatting issues
                logger.info(f"Response text preview (first 200 chars): {text_content[:200]}")
                if len(text_content) > 400:
                    logger.info(f"Response text preview (last 200 chars): {text_content[-200:]}")
                # Check for triple brackets
                if "[[[" in text_content and "]]]" in text_content:
                    logger.info("✓ Triple square brackets found in response")
                else:
                    logger.warning("⚠ Triple square brackets NOT found in response")
                    logger.warning(f"Full response text: {text_content}")
            
            return resp, text_content, tool_calls, reasoning_summary
        elif resp.status == "failed":
            logger.error("=" * 80)
            logger.error("OPENAI API RESPONSE FAILED")
            logger.error("=" * 80)
            logger.error(f"Response ID: {resp.id}")
            logger.error(f"Status: {resp.status}")
            
            # Log all available error information
            error_details = {}
            if hasattr(resp, 'error'):
                error = resp.error
                error_details['error'] = error
                logger.error(f"Error object: {error}")
                
                # Try to extract error message if it's an object
                if hasattr(error, 'message'):
                    error_details['message'] = error.message
                    logger.error(f"Error message: {error.message}")
                if hasattr(error, 'code'):
                    error_details['code'] = error.code
                    logger.error(f"Error code: {error.code}")
                if hasattr(error, 'type'):
                    error_details['type'] = error.type
                    logger.error(f"Error type: {error.type}")
            
            # Log any output that might be available even on failure
            if hasattr(resp, 'output_text') and resp.output_text:
                logger.error(f"Output text (partial): {str(resp.output_text)[:500]}")
            if hasattr(resp, 'output') and resp.output:
                logger.error(f"Output items count: {len(resp.output)}")
            
            # Log the query and instructions used
            logger.error(f"Query: {query[:200]}...")
            logger.error(f"Instructions used: {RESEARCH_ASSISTANT_PROMPT[:200]}...")
            logger.error(f"Model: {self.model}")
            logger.error(f"Tools count: {len(tools_to_use)}")
            if max_tool_calls is not None:
                logger.error(f"Max tool calls: {max_tool_calls}")
            
            # Check if it's a rate limit error and we should retry
            if hasattr(resp, 'error') and is_rate_limit_error(resp.error):
                logger.warning("=" * 80)
                logger.warning("RATE LIMIT ERROR - WAITING 60 SECONDS THEN RESTARTING")
                logger.warning("=" * 80)
                logger.warning("Waiting 60 seconds before retrying from scratch...")
                logger.warning("=" * 80)
                await asyncio.sleep(60)
                
                # Retry the entire request from scratch
                logger.info("Retrying API request from scratch...")
                return await self.call_llm_background(
                    query=query,
                    mcp_tools=mcp_tools,
                    max_retries=max_retries,
                    initial_retry_delay=initial_retry_delay,
                    poll_interval=poll_interval,
                    max_tool_calls=max_tool_calls,
                )
            else:
                # Build detailed error message
                error_msg = f"OpenAI API response failed with status: {resp.status}"
                if error_details:
                    error_msg += f"\nError details: {error_details}"
                raise Exception(error_msg)
        else:
            logger.warning(f"Unexpected response status: {resp.status}")
            # Try to parse anyway
            text_content, tool_calls, reasoning_summary = parse_gpt5_response(resp)
            return resp, text_content, tool_calls, reasoning_summary
    
    async def call_llm(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_retries: int = 3,
        retry_delay: float = 10.0,
        mcp_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Any, Optional[str], List[Dict[str, Any]], Optional[str]]:
        """Call OpenAI responses API with retry logic for timeouts.
        
        According to OpenAI responses API:
        - input should be a list of message dictionaries
        - tools should be a list of tool definitions with "type", "name", "description", "parameters"
        - For o4-mini-deep-research, MCP tools can be passed directly via mcp_tools parameter
        
        Args:
            messages: List of message dictionaries
            tools: Optional list of tool definitions (for function calling)
            max_retries: Maximum number of retry attempts for timeout errors (default: 3)
            retry_delay: Base delay in seconds between retries, uses exponential backoff (default: 10.0)
            mcp_tools: Optional list of MCP tool configurations (for o4-mini-deep-research)
        
        Returns:
            Tuple of (response, text_content, tool_calls, reasoning_summary)
        """
        # Convert messages to the model-provider-specific input list format
        input_list = messages_to_input_list(messages)
        
        # Build reasoning and text parameters.
        # reasoning_effort=None is passed as-is so the API receives {"effort": None, ...}.
        reasoning = {
            "effort": self.reasoning_effort,
            "summary": self.reasoning_summary,
        }

        text = {
            "verbosity": self.verbosity
        }
        
        # For o4-mini-deep-research, use MCP tools if provided, otherwise format regular tools
        tools_to_use = None
        if mcp_tools:
            # MCP tools are passed directly (for o4-mini-deep-research)
            tools_to_use = mcp_tools
        elif tools:
            # Format regular function tools
            tools_to_use = self.format_tools(tools)
        
        # If tools are not available, explicitly instruct the model to answer directly.
        # Keep this aligned with ClaudeProvider and GeminiProvider's no-tools instruction wording.
        instructions = RESEARCH_ASSISTANT_PROMPT
        if not tools_to_use:
            no_tools_instruction = "\n\nIMPORTANT: Tool calls are NOT available. Do NOT attempt to make any function calls or tool calls. Answer the question directly using only your knowledge and reasoning."
            instructions = instructions + no_tools_instruction
        
        # Calculate input size for logging
        input_size = len(json.dumps(input_list)) if input_list else 0
        num_messages = len(messages)
        num_tools = len(tools_to_use) if tools_to_use else 0
        
        # Set timeout to 5 minutes (300 seconds) for long-running API calls
        timeout_seconds = 300
        
        # Retry loop for timeout errors
        loop = asyncio.get_event_loop()
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                logger.debug(
                    "Calling OpenAI responses API (attempt %d/%d, timeout: %d seconds)",
                    attempt + 1, max_retries, timeout_seconds
                )
                
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self.client.responses.create(
                            instructions=instructions,
                            model=self.model,
                            input=input_list,
                            reasoning=reasoning,
                            tools=tools_to_use,
                            text=text,
                        )
                    ),
                    timeout=timeout_seconds
                )
                
                logger.debug("OpenAI responses API call completed successfully")
                # Parse response
                text_content, tool_calls, reasoning_summary = parse_gpt5_response(response)
                return response, text_content, tool_calls, reasoning_summary
                
            except asyncio.TimeoutError as e:
                last_exception = e
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                
                # Build detailed error message
                error_details = []
                error_details.append("=" * 80)
                error_details.append("OPENAI API TIMEOUT ERROR - DETAILED INFORMATION")
                error_details.append("=" * 80)
                error_details.append(f"Exception Type: {type(e).__name__}")
                error_details.append(f"Exception Message: {str(e)}")
                error_details.append(f"Exception Args: {repr(e.args)}")
                error_details.append(f"Timeout Duration: {timeout_seconds} seconds")
                error_details.append(f"Retry Attempt: {attempt + 1}/{max_retries}")
                error_details.append("")
                error_details.append("API Call Details:")
                error_details.append(f"  Model: {self.model}")
                error_details.append(f"  Number of Messages: {num_messages}")
                error_details.append(f"  Input Size (chars): {input_size}")
                error_details.append(f"  Number of Tools: {num_tools}")
                error_details.append(f"  Reasoning Effort: {self.reasoning_effort}")
                error_details.append(f"  Reasoning Summary: {self.reasoning_summary}")
                error_details.append(f"  Verbosity: {self.verbosity}")
                error_details.append("")
                error_details.append("Full Stack Trace:")
                error_details.append(traceback.format_exc())
                error_details.append("=" * 80)
                
                # Print to console AND log
                error_msg_full = "\n".join(error_details)
                print(error_msg_full, file=sys.stderr)
                logger.error(error_msg_full)
                
                # If this is not the last attempt, retry
                if attempt < max_retries - 1:
                    retry_msg = f"Timeout occurred. Retrying in {wait_time:.1f} seconds (attempt {attempt + 2}/{max_retries})..."
                    print(retry_msg, file=sys.stderr)
                    logger.warning(retry_msg)
                    await asyncio.sleep(wait_time)
                else:
                    final_error_msg = f"All {max_retries} retry attempts exhausted. Raising timeout error."
                    print(final_error_msg, file=sys.stderr)
                    logger.error(final_error_msg)
                    error_msg = (
                        f"OpenAI API call timed out after {timeout_seconds} seconds. "
                        f"Failed after {max_retries} attempts."
                    )
                    raise TimeoutError(error_msg) from e
                    
            except BadRequestError as e:
                # Check if this is a context length exceeded error
                error_msg = str(e)
                is_context_length_error = 'context_length_exceeded' in error_msg
                
                # Also try to extract from error body if available
                if not is_context_length_error and hasattr(e, 'body') and e.body:
                    try:
                        if isinstance(e.body, dict):
                            error_obj = e.body.get('error', {})
                            is_context_length_error = error_obj.get('code') == 'context_length_exceeded'
                        elif isinstance(e.body, str):
                            error_body = json.loads(e.body)
                            error_obj = error_body.get('error', {})
                            is_context_length_error = error_obj.get('code') == 'context_length_exceeded'
                    except Exception:
                        pass
                
                if is_context_length_error:
                    logger.warning(
                        "Context length exceeded for model %s. Input size: %d chars, Messages: %d, Tools: %d",
                        self.model, input_size, num_messages, num_tools
                    )
                    raise ContextLengthExceededError(
                        f"Input exceeds context window for model {self.model}. "
                        f"Input size: {input_size} chars, Messages: {num_messages}, Tools: {num_tools}"
                    ) from e
                
                # For other BadRequestErrors, check for rate limits
                last_exception = e
                error_msg = str(e)
                
                # Check for rate limit errors
                if is_rate_limit_error(e) and attempt < max_retries - 1:
                    wait_time = parse_rate_limit_wait_time(error_msg)
                    if wait_time is None:
                        wait_time = retry_delay * (2 ** attempt)
                    else:
                        wait_time = wait_time + 1.0
                    
                    logger.warning("=" * 80)
                    logger.warning("RATE LIMIT ERROR ON API CALL")
                    logger.warning("=" * 80)
                    logger.warning(f"Attempt {attempt + 1}/{max_retries}")
                    logger.warning(f"Error: {error_msg[:500]}")
                    logger.warning(f"Waiting {wait_time:.2f} seconds before retry...")
                    logger.warning("=" * 80)
                    await asyncio.sleep(wait_time)
                    continue
                
                # Build detailed error message
                error_details = []
                error_details.append("=" * 80)
                error_details.append("OPENAI API ERROR - DETAILED INFORMATION")
                error_details.append("=" * 80)
                error_details.append(f"Exception Type: {type(e).__name__}")
                error_details.append(f"Exception Message: {error_msg}")
                error_details.append(f"Exception Args: {repr(e.args)}")
                error_details.append(f"Exception Repr: {repr(e)}")
                error_details.append("")
                error_details.append("API Call Details:")
                error_details.append(f"  Model: {self.model}")
                error_details.append(f"  Number of Messages: {num_messages}")
                error_details.append(f"  Input Size (chars): {input_size}")
                error_details.append(f"  Number of Tools: {num_tools}")
                error_details.append(f"  Reasoning Effort: {self.reasoning_effort}")
                error_details.append(f"  Reasoning Summary: {self.reasoning_summary}")
                error_details.append(f"  Verbosity: {self.verbosity}")
                error_details.append("")
                error_details.append("Full Stack Trace:")
                error_details.append(traceback.format_exc())
                error_details.append("=" * 80)
                
                # Print to console AND log
                error_msg_full = "\n".join(error_details)
                print(error_msg_full, file=sys.stderr)
                logger.error(error_msg_full)
                
                # Handle specific error types
                if "404" in error_msg or "not found" in error_msg.lower():
                    raise ValueError(
                        f"Model '{self.model}' not found or responses API not available. "
                        f"Original error: {error_msg}"
                    ) from e
                
                # For other errors, don't retry (only timeout and rate limit errors are retried)
                raise
                
            except Exception as e:
                last_exception = e
                error_msg = str(e)
                
                # Check for rate limit errors
                if is_rate_limit_error(e) and attempt < max_retries - 1:
                    wait_time = parse_rate_limit_wait_time(error_msg)
                    if wait_time is None:
                        wait_time = retry_delay * (2 ** attempt)
                    else:
                        wait_time = wait_time + 1.0
                    
                    logger.warning("=" * 80)
                    logger.warning("RATE LIMIT ERROR ON API CALL")
                    logger.warning("=" * 80)
                    logger.warning(f"Attempt {attempt + 1}/{max_retries}")
                    logger.warning(f"Error: {error_msg[:500]}")
                    logger.warning(f"Waiting {wait_time:.2f} seconds before retry...")
                    logger.warning("=" * 80)
                    await asyncio.sleep(wait_time)
                    continue
                
                # Build detailed error message
                error_details = []
                error_details.append("=" * 80)
                error_details.append("OPENAI API ERROR - DETAILED INFORMATION")
                error_details.append("=" * 80)
                error_details.append(f"Exception Type: {type(e).__name__}")
                error_details.append(f"Exception Message: {error_msg}")
                error_details.append(f"Exception Args: {repr(e.args)}")
                error_details.append(f"Exception Repr: {repr(e)}")
                error_details.append("")
                error_details.append("API Call Details:")
                error_details.append(f"  Model: {self.model}")
                error_details.append(f"  Number of Messages: {num_messages}")
                error_details.append(f"  Input Size (chars): {input_size}")
                error_details.append(f"  Number of Tools: {num_tools}")
                error_details.append(f"  Reasoning Effort: {self.reasoning_effort}")
                error_details.append(f"  Reasoning Summary: {self.reasoning_summary}")
                error_details.append(f"  Verbosity: {self.verbosity}")
                error_details.append("")
                error_details.append("Full Stack Trace:")
                error_details.append(traceback.format_exc())
                error_details.append("=" * 80)
                
                # Print to console AND log
                error_msg_full = "\n".join(error_details)
                print(error_msg_full, file=sys.stderr)
                logger.error(error_msg_full)
                
                # Handle specific error types
                if "404" in error_msg or "not found" in error_msg.lower():
                    raise ValueError(
                        f"Model '{self.model}' not found or responses API not available. "
                        f"Original error: {error_msg}"
                    ) from e
                
                # For other errors, don't retry (only timeout and rate limit errors are retried)
                raise
        
        # This should never be reached, but just in case
        if last_exception:
            raise last_exception
    
    def format_tool_response_message(
        self, tool_call_id: str, tool_name: str, tool_result: str
    ) -> Dict[str, Any]:
        """Format tool response for OpenAI responses API.
        
        According to the reference, function call outputs should be formatted as:
        {
            "type": "function_call_output",
            "call_id": "...",
            "output": "..."  # JSON string
        }
        """
        return {
            "type": "function_call_output",
            "call_id": tool_call_id,
            "output": tool_result,  # Should already be a JSON string
        }


