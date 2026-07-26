"""Google Gemini provider implementation."""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types

from ..prompts import RESEARCH_ASSISTANT_PROMPT
from .base import LLMProvider
from .reasoning_discovery import (
    candidate_openrouter_slugs,
    discover_reasoning_config,
    gemini_thinking_level_for_effort,
    highest_supported_effort,
)

logger = logging.getLogger(__name__)


def is_rate_limit_error(error: Any) -> bool:
    """Check if an error is a rate limit error."""
    if error is None:
        return False
    error_str = str(error).lower()
    # Check for common rate limit indicators
    return (
        "rate limit" in error_str
        or "rate_limit" in error_str
        or "429" in error_str
        or "quota" in error_str
        or "resource exhausted" in error_str
        or "too many requests" in error_str
    )


def parse_rate_limit_wait_time(error_message: str) -> Optional[float]:
    """Parse wait time from rate limit error message."""
    # Look for patterns like "try again in X seconds" or "retry after X"
    patterns = [
        r"try again in ([\d.]+) seconds?",
        r"retry after ([\d.]+) seconds?",
        r"wait ([\d.]+) seconds?",
        r"in ([\d.]+) seconds?",
    ]
    for pattern in patterns:
        match = re.search(pattern, error_message.lower())
        if match:
            return float(match.group(1))
    return None

class GeminiProvider(LLMProvider):
    """Google Gemini provider (gemini-3-pro, gemini-3-flash, etc.)"""
    
    def __init__(
        self,
        model: str = "gemini-3-pro-preview",
        api_key: Optional[str] = None,
        # Common generation controls (wrapped from google.genai.GenerateContentConfig).
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[float] = None,
        seed: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        # Thinking controls.
        thinking_level: Optional[types.ThinkingLevel | str] = None,
        thinking_budget_tokens: Optional[int] = None,
    ):
        super().__init__(model, api_key)

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.max_output_tokens = max_output_tokens
        self.thinking_budget_tokens = thinking_budget_tokens

        if thinking_level is None:
            # Max out reasoning: discover this model's highest supported
            # effort via OpenRouter's GET /models catalog (same mechanism
            # OpenRouterProvider uses) and map it to Gemini's native
            # `thinking_level` enum — no inference traffic goes through
            # OpenRouter, we only use its catalog as a reference. Gemini has
            # no tier above HIGH today (OpenRouter maps "xhigh"/"max" down to
            # it too), so this mainly future-proofs against Google adding a
            # higher tier later; falls back to the previous hardcoded HIGH
            # default if discovery is unavailable or the model isn't listed.
            slugs = candidate_openrouter_slugs(["google"], model)
            reasoning_cfg = discover_reasoning_config(slugs)
            discovered_effort, supports_effort = highest_supported_effort(reasoning_cfg)
            mapped_level = (
                gemini_thinking_level_for_effort(discovered_effort) if supports_effort else None
            )
            thinking_level = mapped_level or "HIGH"
            logger.info(
                "GeminiProvider thinking_level auto-discovered for %s: %s "
                "(reasoning_cfg=%s)",
                model, thinking_level, reasoning_cfg,
            )
        self.thinking_level = thinking_level
     
        # Try to get API key first (simpler option)
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if api_key:
            # Use API key authentication (Google AI API)
            self.client = genai.Client(api_key=api_key)
        else:
            # Fall back to Vertex AI (requires Google Cloud credentials)
            use_vertexai = os.getenv('GOOGLE_GENAI_USE_VERTEXAI', 'true').lower() == 'true'
            project = os.getenv('GOOGLE_CLOUD_PROJECT', 'cos-image-embeddings')
            location = os.getenv('GOOGLE_CLOUD_LOCATION', 'global')
            
            self.client = genai.Client(
                vertexai=use_vertexai,
                project=project,
                location=location,
            )

        # Normalize thinking_level if user passed a string.
        if isinstance(self.thinking_level, str):
            lvl = self.thinking_level.strip().upper()
            # Support common inputs like "high" / "minimal" / "medium".
            mapping = {
                "LOW": types.ThinkingLevel.LOW,
                "MEDIUM": types.ThinkingLevel.MEDIUM,
                "HIGH": types.ThinkingLevel.HIGH,
                "MINIMAL": types.ThinkingLevel.MINIMAL,
                "THINKING_LEVEL_UNSPECIFIED": types.ThinkingLevel.THINKING_LEVEL_UNSPECIFIED,
                "UNSPECIFIED": types.ThinkingLevel.THINKING_LEVEL_UNSPECIFIED,
            }
            self.thinking_level = mapping.get(lvl, types.ThinkingLevel.HIGH)   # defaults to HIGH
    
    def format_tools(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """Format tools for Gemini function calling."""
        # Convert to FunctionDeclaration format for new API
        return [
            {
                "name": tool.name,
                "description": tool.description or f"Tool: {tool.name}",
                "parameters": tool.inputSchema or {},
            }
            for tool in tools
        ]
    
    async def call_llm(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_retries: int = 3,
        retry_delay: float = 10.0,
    ) -> Tuple[Any, Optional[str], List[Dict[str, Any]], Optional[str]]:
        """
        Call Gemini API using the new google.genai package.
        
        Supports MCP integration:
        - tools can be MCP session objects (e.g., mcp_client.session from FastMCP)
        - tools can be dictionaries with tool definitions
        - Thought signatures are automatically preserved when using Content objects
        
        Args:
            messages: Conversation history in standard format
            tools: Optional list of tools - can be MCP session objects or tool dictionaries
            max_retries: Maximum number of retry attempts for rate limit errors (default: 3)
            retry_delay: Base delay in seconds between retries, uses exponential backoff (default: 10.0)
            
        Returns:
            Tuple of (response, text_content, tool_calls, reasoning_summary)
        """
        # Convert messages to Content format for new API
        contents: List[types.Content] = []
        system_instruction_list = [RESEARCH_ASSISTANT_PROMPT]  # Always include RESEARCH_ASSISTANT_PROMPT
        
        for msg in messages:
            role = msg["role"]
            if role == "system":
                # Append additional system messages to the system_instruction list
                # RESEARCH_ASSISTANT_PROMPT is always first
                system_content = msg["content"]
                if system_content and system_content != RESEARCH_ASSISTANT_PROMPT:
                    system_instruction_list.append(system_content)
            elif role == "assistant":
                # Handle assistant messages - could contain tool calls with thought signatures
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls", [])
                parts = []
                
                # Add text content if present
                if content:
                    parts.append(types.Part.from_text(text=content))
                
                # Add function calls if present
                # Note: Thought signatures MUST be preserved when reconstructing messages
                # The API requires thought_signature on function calls in subsequent requests
                # According to docs, only the first function call in parallel calls has the thought signature
                for idx, tool_call in enumerate(tool_calls):
                    func_call_dict = tool_call.get("function", {}) if isinstance(tool_call.get("function"), dict) else {}
                    if not func_call_dict and "name" in tool_call:
                        func_call_dict = {
                            "name": tool_call.get("name", ""),
                            "arguments": tool_call.get("arguments", "{}")
                        }
                    
                    # Parse arguments
                    try:
                        args = json.loads(func_call_dict.get("arguments", "{}")) if isinstance(func_call_dict.get("arguments"), str) else func_call_dict.get("arguments", {})
                    except json.JSONDecodeError:
                        args = {}
                    
                    func_call_obj = types.FunctionCall(
                        name=func_call_dict.get("name", tool_call.get("name", "")),
                        args=args
                    )
                    
                    part = types.Part(function_call=func_call_obj)
                    
                    # Extract and set thought signature if present
                    # According to the format: only the FIRST tool call in parallel calls has the thought signature
                    # We need to set it on the Part object when reconstructing messages
                    thought_sig = None
                    if idx == 0:  # Only the first function call should have thought signature
                        # Try to get from _thought_signature_bytes first (stored separately to avoid JSON issues)
                        thought_sig = tool_call.get("_thought_signature_bytes")
                        if not thought_sig and "extra_content" in tool_call:
                            extra = tool_call.get("extra_content", {})
                            if isinstance(extra, dict) and "google" in extra:
                                google_extra = extra["google"]
                                if isinstance(google_extra, dict) and "thought_signature" in google_extra:
                                    thought_sig = google_extra["thought_signature"]
                    
                    # Set thought signature on the part if we have it (only for first function call)
                    if thought_sig:
                        # Use object.__setattr__ to set thought_signature on Part object
                        object.__setattr__(part, 'thought_signature', thought_sig)
                    
                    parts.append(part)
                
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
            elif role == "user":
                content = msg.get("content", "")
                parts = []
                if isinstance(content, list):
                    # Handle structured content (e.g., from tool responses)
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "tool_result":
                            # Convert tool result to function_response part
                            result = json.loads(item["content"]) if isinstance(item["content"], str) else item["content"]
                            tool_name = item.get("name", "")
                            if tool_name:  # Only add if name is present
                                parts.append(types.Part(
                                    function_response=types.FunctionResponse(
                                        name=tool_name,
                                        response=result
                                    )
                                ))
                        else:
                            parts.append(types.Part.from_text(text=str(item)))
                else:
                    parts = [types.Part.from_text(text=content)]
                contents.append(types.Content(role="user", parts=parts))
            elif role == "function":
                # Handle function responses - convert parts format and preserve thought signatures
                # Gemini API expects function responses in a "user" role message with function_response parts
                msg_parts = msg.get("parts", [])
                parts = []
                for idx, part_dict in enumerate(msg_parts):
                    if "function_response" in part_dict:
                        func_resp = part_dict["function_response"]
                        # Ensure name is present and not empty (required field)
                        func_name = func_resp.get("name")
                        if not func_name:
                            # Try to get name from the message itself if not in function_response
                            func_name = msg.get("name", "")
                        if not func_name:
                            # Skip function responses with empty names to avoid "Tool '' not listed" warnings
                            continue
                        func_response_obj = types.FunctionResponse(
                            name=func_name,
                            response=func_resp.get("response", {})
                        )
                        
                        # Create part with function response
                        part = types.Part(function_response=func_response_obj)
                        
                        # Preserve thought signature if present (should be on first function response part)
                        # According to docs: "Thought signatures should always be used with function calling for best results."
                        if "thought_signature" in part_dict and part_dict["thought_signature"]:
                            thought_sig = part_dict["thought_signature"]
                            # Use object.__setattr__ to set thought_signature on Part object
                            object.__setattr__(part, 'thought_signature', thought_sig)
                        
                        parts.append(part)
                    elif "text" in part_dict:
                        parts.append(types.Part.from_text(text=part_dict["text"]))
                if parts:
                    # Gemini API expects function responses in "user" role, not "function" role
                    contents.append(types.Content(role="user", parts=parts))
        
        # Prepare generation config with tools and thinking config.
        # NOTE: We set thinking explicitly here to match your existing helper behavior (HIGH by default).
        thinking_kwargs: Dict[str, Any] = {"thinking_level": self.thinking_level}
        if self.thinking_budget_tokens is not None:
            thinking_kwargs["thinkingBudget"] = self.thinking_budget_tokens

        config_kwargs: Dict[str, Any] = {
            "thinking_config": types.ThinkingConfig(**thinking_kwargs),
        }

        # Add sampling/generation controls if provided.
        if self.temperature is not None:
            config_kwargs["temperature"] = float(self.temperature)
        if self.top_p is not None:
            config_kwargs["top_p"] = float(self.top_p)
        if self.top_k is not None:
            config_kwargs["top_k"] = float(self.top_k)
        if self.seed is not None:
            config_kwargs["seed"] = int(self.seed)
        if self.max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = int(self.max_output_tokens)

        if tools:
            # Handle tools - support both MCP session objects and tool dictionaries
            # Based on MCP integration pattern: tools can be MCP session objects (like mcp_client.session)
            # or dictionaries with tool definitions
            tool_list = []
            
            for tool in tools:
                # Check if tool is an MCP session object (has session-like attributes)
                # This allows passing mcp_client.session directly as shown in play.py
                if hasattr(tool, 'list_tools') or hasattr(tool, 'call_tool'):
                    # This is likely an MCP session object - pass it directly
                    tool_list.append(tool)
                elif isinstance(tool, dict):
                    # Convert dictionary to FunctionDeclaration format
                    tool_list.append(
                        types.FunctionDeclaration(
                            name=tool["name"],
                            description=tool.get("description", ""),
                            parameters=tool.get("parameters", {})
                        )
                    )
                else:
                    # Try to extract tool info from object (MCP tool object)
                    if hasattr(tool, 'name') and hasattr(tool, 'inputSchema'):
                        tool_list.append(
                            types.FunctionDeclaration(
                                name=tool.name,
                                description=getattr(tool, 'description', '') or f"Tool: {tool.name}",
                                parameters=tool.inputSchema or {}
                            )
                        )

            # If all tools are FunctionDeclarations, wrap in Tool object
            # Otherwise, pass directly (for MCP session objects)
            if all(isinstance(t, types.FunctionDeclaration) for t in tool_list):
                config_kwargs["tools"] = [types.Tool(function_declarations=tool_list)]
            else:
                # Mixed or MCP session objects - pass directly
                config_kwargs["tools"] = tool_list
        else:
            # When tools are disabled, explicitly tell the model not to use tool calls
            # This prevents the model from trying to make function calls when none are available
            # which would result in MALFORMED_FUNCTION_CALL errors
            no_tools_instruction = "\n\nIMPORTANT: Tool calls are NOT available. Do NOT attempt to make any function calls or tool calls. Answer the question directly using only your knowledge and reasoning."
            # Append to the last system instruction (or create a new one if only RESEARCH_ASSISTANT_PROMPT exists)
            if len(system_instruction_list) == 1:
                # Only RESEARCH_ASSISTANT_PROMPT exists, append the no-tools instruction
                system_instruction_list.append(no_tools_instruction)
            else:
                # Append to the last system instruction
                system_instruction_list[-1] = system_instruction_list[-1] + no_tools_instruction
        
        # Always set system_instruction as a list (RESEARCH_ASSISTANT_PROMPT is always included)
        config_kwargs["system_instruction"] = system_instruction_list
        
        config = types.GenerateContentConfig(**config_kwargs)
        
        # Client is sync, so we run it in executor
        loop = asyncio.get_event_loop()
        
        # Retry loop for rate limit errors
        response = None
        last_exception = None
        for attempt in range(max_retries):
            try:
                # Use generate_content with contents list
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=config
                    )
                )
                # Success - break out of retry loop
                break
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
                    logger.warning("RATE LIMIT ERROR ON GEMINI API CALL")
                    logger.warning("=" * 80)
                    logger.warning(f"Attempt {attempt + 1}/{max_retries}")
                    logger.warning(f"Error: {error_msg[:500]}")
                    logger.warning(f"Waiting {wait_time:.2f} seconds before retry...")
                    logger.warning("=" * 80)
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Non-rate-limit error or last attempt - log and re-raise
                    logger.error("=" * 80)
                    logger.error("GEMINI API ERROR")
                    logger.error("=" * 80)
                    logger.error(f"Exception Type: {type(e).__name__}")
                    logger.error(f"Exception Message: {error_msg}")
                    logger.error(f"Attempt: {attempt + 1}/{max_retries}")
                    logger.error("=" * 80)
                    print("Error:", e)
                    raise e
        
        # If we exhausted retries without success, raise the last exception
        if response is None and last_exception is not None:
            raise last_exception

        print("Candidates token count:", response.usage_metadata.candidates_token_count)
        print("Prompt token count:", response.usage_metadata.prompt_token_count)
        print("Thoughts token count:", response.usage_metadata.thoughts_token_count)
        print("Total token count:", response.usage_metadata.total_token_count)
        print("Cached content token count:", response.usage_metadata.cached_content_token_count)

        # Extract text content
        text_content = None
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts:
                text_parts = []
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                if text_parts:
                    text_content = "".join(text_parts)
            elif hasattr(candidate, 'finish_reason'):
                # Check if response was blocked or filtered
                finish_reason = candidate.finish_reason
                if finish_reason:
                    logger.warning(f"Gemini response finished with reason: {finish_reason}")
                    # For certain finish reasons, provide a response with triple brackets so it passes validation
                    if finish_reason in ['SAFETY', 'RECITATION', 'OTHER']:
                        text_content = f"[[[Response blocked by Gemini API: {finish_reason}. The model was unable to generate a response due to content filtering or safety restrictions.]]]"
                    elif finish_reason == 'MALFORMED_FUNCTION_CALL':
                        # This shouldn't happen when tool calling is disabled, but handle it gracefully
                        text_content = f"[[[Response error: MALFORMED_FUNCTION_CALL. The model attempted to make a function call but it was malformed. This may occur if the model was configured for function calling but the call format was invalid.]]]"
                    else:
                        text_content = f"[[[Response finished with reason: {finish_reason}. The model completed but may not have generated the expected content format.]]]"
        
        # Extract token usage from response (use API's token counts instead of manual counting)
        # Store usage_metadata for mcp_client to access
        # Use object.__setattr__ to bypass Pydantic validation since response is a Pydantic model
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            # Directly access the attributes - they exist as shown by the print statements
            # Convert to int and handle None values
            prompt_token_count = int(usage.prompt_token_count) if usage.prompt_token_count is not None else 0
            candidates_token_count = int(usage.candidates_token_count) if usage.candidates_token_count is not None else 0
            thoughts_token_count = int(usage.thoughts_token_count) if usage.thoughts_token_count is not None else 0
            total_token_count = int(usage.total_token_count) if usage.total_token_count is not None else 0
            cached_content_token_count = int(usage.cached_content_token_count) if usage.cached_content_token_count is not None else 0
            
            # Store in response object for mcp_client to access
            object.__setattr__(response, 'gemini_token_usage', {
                'prompt_tokens': prompt_token_count,
                'completion_tokens': candidates_token_count,
                'thoughts_tokens': thoughts_token_count,
                'total_tokens': total_token_count,
                'cached_content_tokens': cached_content_token_count,
            })
            
            # Log token usage for debugging/verification
            logger.info(
                "Gemini API token usage extracted - prompt: %d, candidates: %d, thoughts: %d, total: %d, cached: %d",
                prompt_token_count, candidates_token_count, thoughts_token_count, total_token_count, cached_content_token_count
            )
        else:
            # No usage_metadata available - set defaults
            logger.warning("No usage_metadata found in Gemini response")
            object.__setattr__(response, 'gemini_token_usage', {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'thoughts_tokens': 0,
                'total_tokens': 0,
                'cached_content_tokens': 0,
            })

        #print("messages:", messages)
        
        # Extract tool calls and thought signatures
        tool_calls = []
        thought_signature = None
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        func_call = part.function_call
                        
                        # Extract arguments - convert to JSON string
                        args_dict = {}
                        if hasattr(func_call, 'args') and func_call.args:
                            if isinstance(func_call.args, dict):
                                args_dict = func_call.args
                            else:
                                # Try to convert to dict if it's not already
                                try:
                                    if isinstance(func_call.args, str):
                                        args_dict = json.loads(func_call.args)
                                    else:
                                        args_dict = dict(func_call.args) if hasattr(func_call.args, '__iter__') else {}
                                except (json.JSONDecodeError, TypeError, ValueError):
                                    args_dict = {}
                        
                        # Extract name - should always be present in Gemini response
                        if not hasattr(func_call, 'name') or not func_call.name:
                            logger.error("Function call missing name attribute: %s", func_call)
                            raise ValueError(f"Function call from Gemini response is missing required 'name' attribute: {func_call}")
                        func_name = func_call.name
                        
                        # Ensure arguments is valid JSON string
                        # Always use json.dumps to ensure valid JSON, even for empty dicts
                        try:
                            if args_dict is None:
                                args_json = "{}"
                            elif isinstance(args_dict, dict):
                                args_json = json.dumps(args_dict, ensure_ascii=False)
                            else:
                                # Try to serialize, fallback to empty dict if it fails
                                args_json = json.dumps(args_dict, ensure_ascii=False)
                        except (TypeError, ValueError) as e:
                            # If serialization fails, use empty dict
                            args_json = "{}"
                        
                        tool_call_dict = {
                            "id": f"call_{len(tool_calls)}",  # Gemini doesn't provide IDs
                            "name": func_name,
                            "arguments": args_json,
                        }
                        
                        # Extract thought signature from the first function call part
                        # According to docs, the first function call in parallel calls has the thought signature
                        # Note: thought_signature is expected to be bytes and should NOT be converted
                        # We store it separately to avoid JSON serialization issues
                        if thought_signature is None and hasattr(part, 'thought_signature') and part.thought_signature:
                            thought_signature = part.thought_signature
                            # Store thought signature separately (not in tool_call_dict) to avoid JSON serialization
                            # The thought signature will be added back when reconstructing messages for the API
                            tool_call_dict["_thought_signature_bytes"] = thought_signature  # Internal use only
                        
                        tool_calls.append(tool_call_dict)
                    
                    # Also check for thought signature in non-function-call parts (final part may have it)
                    # Note: thought_signature is expected to be bytes and should not be converted
                    if thought_signature is None and hasattr(part, 'thought_signature') and part.thought_signature:
                        thought_signature = part.thought_signature
        
        return None, text_content, tool_calls, None
    
    def format_tool_response_message(
        self, tool_call_id: str, tool_name: str, tool_result: str, thought_signature: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format tool response for Gemini.
        
        Args:
            tool_call_id: ID of the tool call
            tool_name: Name of the tool
            tool_result: JSON string of tool result
            thought_signature: Optional thought signature to preserve from the original function call
        """
        # Validate tool_name is not empty (required field)
        # Tool name should always be present as it's explicitly provided in Gemini response
        if not tool_name or not tool_name.strip():
            raise ValueError(f"tool_name cannot be empty. tool_call_id: {tool_call_id}. This indicates a bug in tool call extraction.")
        
        # Gemini uses function_response format in parts
        try:
            result_dict = json.loads(tool_result) if isinstance(tool_result, str) else tool_result
        except json.JSONDecodeError:
            result_dict = {"result": tool_result}
        
        part_dict = {
            "function_response": {
                "name": tool_name,
                "response": result_dict
            }
        }
        
        # Include thought signature if provided (should be from the first function call)
        if thought_signature:
            part_dict["thought_signature"] = thought_signature
        
        return {
            "role": "function",
            "parts": [part_dict]
        }
    
    def format_multiple_tool_response_message(
        self,
        tool_results: List[Tuple[str, str, str, Optional[bytes]]]
    ) -> Dict[str, Any]:
        """
        Format multiple tool responses into a single message for parallel tool calls.
        
        For Gemini, all function responses must be in a single user message with multiple parts,
        matching the number of function calls in the assistant's message.
        
        According to the documentation: "Thought signatures should always be used with function calling for best results."
        The thought signature should be included in the first function response part.
        
        Args:
            tool_results: List of tuples (tool_call_id, tool_name, tool_result_str, thought_signature)
                         where thought_signature is optional bytes (only for first result)
        
        Returns:
            Message dict with role "function" (converted to "user" role in call_llm when building Content objects)
            and parts containing all function responses
        """
        parts = []
        
        for idx, result_tuple in enumerate(tool_results):
            # Handle both 3-tuple (backward compatibility) and 4-tuple (with thought signature)
            if len(result_tuple) == 4:
                tool_call_id, tool_name, tool_result_str, thought_signature = result_tuple
            else:
                tool_call_id, tool_name, tool_result_str = result_tuple
                thought_signature = None
            
            # Validate tool_name is not empty
            if not tool_name or not tool_name.strip():
                raise ValueError(f"tool_name cannot be empty. tool_call_id: {tool_call_id}. This indicates a bug in tool call extraction.")
            
            try:
                result_dict = json.loads(tool_result_str) if isinstance(tool_result_str, str) else tool_result_str
            except json.JSONDecodeError:
                result_dict = {"result": tool_result_str}
            
            part_dict = {
                "function_response": {
                    "name": tool_name,
                    "response": result_dict
                }
            }
            
            # Only the first function response should have the thought signature
            # (matching the pattern where only the first function call has it)
            # According to docs: "Thought signatures should always be used with function calling for best results."
            if idx == 0 and thought_signature:
                part_dict["thought_signature"] = thought_signature
            
            parts.append(part_dict)
        
        # Return with "function" role - this will be converted to "user" role in call_llm
        # when building Content objects for the Gemini API (see line 188 in call_llm)
        return {
            "role": "function",
            "parts": parts
        }


