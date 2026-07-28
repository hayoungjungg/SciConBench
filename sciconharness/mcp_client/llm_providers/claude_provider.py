"""
Anthropic Claude provider implementation with MCP framework support.
Has extended thinking support (4,096 budget tokens) for reasoning and tool calling.
8,192 max tokens per turn. No temperature used.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

# Try to import Azure identity for Foundry support
try:
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    AZURE_IDENTITY_AVAILABLE = True
except ImportError:
    AZURE_IDENTITY_AVAILABLE = False

from .base import LLMProvider, ContextLengthExceededError
from .reasoning_discovery import (
    anthropic_effort_ratio,
    candidate_openrouter_slugs,
    discover_reasoning_config,
    highest_supported_effort,
)
from ..prompts import RESEARCH_ASSISTANT_PROMPT

logger = logging.getLogger(__name__)

# Sentinel distinguishing "caller didn't pass adaptive_effort at all" (-> discover
# the model's highest supported effort, same mechanism used for thinking_budget_tokens
# below) from "caller explicitly passed a specific adaptive_effort" (-> respect it as-is).
_UNSET = object()


def _get_tool_attr(tool: Any, attr: str, default: Any = None) -> Any:
    """Get attribute from tool (handles both object and dict)."""
    return getattr(tool, attr, None) or (tool.get(attr, default) if isinstance(tool, dict) else default)


class ClaudeProvider(LLMProvider):
    """
    Anthropic Claude provider with support for:
    - Standard Anthropic API
    - Microsoft Foundry (Azure) endpoints
    - MCP framework integration
    """
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        resource: Optional[str] = None,
        base_url: Optional[str] = None,
        use_foundry: bool = False,
        azure_ad_token_provider: Optional[Any] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 8192,    # maximum *output* token per turn
        thinking_budget_tokens: Optional[int] = None,  # budget for extended thinking; None -> auto-discover (see below)
        # Extended thinking configuration.
        # - "enabled": fixed budget (budget_tokens)
        # - "adaptive": dynamic budget/usage (effort)
        # - "disabled": no extended thinking; temperature (if provided) is used normally
        thinking_mode: str = "enabled",
        adaptive_effort: str = _UNSET,
    ):
        """
        Initialize Claude provider.
        
        Args:
            model: Model name (e.g., "claude-sonnet-4-5")
            api_key: API key for standard Anthropic API (or ANTHROPIC_FOUNDRY_API_KEY for Foundry)
            resource: Foundry resource name (for Foundry endpoints)
            base_url: Custom base URL (alternative to resource for Foundry)
            use_foundry: If True, use Microsoft Foundry endpoints
            azure_ad_token_provider: Azure AD token provider for Foundry authentication
            temperature: Sampling temperature (0.0 to 1.0). Default: None (disabled when thinking is enabled).
            max_tokens: Maximum number of tokens to generate. Default: 8192.
            thinking_budget_tokens: Budget for extended thinking. Must be less than max_tokens.
                Default: None, which auto-discovers this model's highest reasoning effort via
                OpenRouter's GET /models catalog (same mechanism OpenRouterProvider uses) and
                scales the budget accordingly (capped at 80% of max_tokens to leave headroom for
                the final answer + tool calls); falls back to a flat 4096 if no data is found for
                the model (this is the case for claude-sonnet-4-5 today).
            thinking_mode: Extended thinking mode: "enabled" (fixed budget), "adaptive" (dynamic), or "disabled".
            adaptive_effort: Adaptive thinking effort sent via the Messages API's
                ``output_config.effort`` field — one of "low", "medium", "high"
        """
        super().__init__(model, api_key)
        
        if not CLAUDE_AVAILABLE:
            raise ImportError(
                "Anthropic package not installed. Install with: pip install anthropic"
            )
        
        self.use_foundry = use_foundry
        self.resource = resource
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking_mode = thinking_mode
        
        # Validate thinking configuration.
        self.thinking_mode = str(self.thinking_mode or "enabled").lower()
        if self.thinking_mode not in {"enabled", "adaptive", "disabled"}:
            raise ValueError('thinking_mode must be one of {"enabled","adaptive","disabled"}')

        # Discover this model's highest supported reasoning effort once via
        # OpenRouter's GET /models catalog (same mechanism OpenRouterProvider
        # uses) — reused below both to pick a maxed-out `adaptive_effort` for
        # adaptive thinking (native `output_config.effort`) and, when
        # `thinking_budget_tokens` isn't explicitly passed, to size the fixed
        # thinking budget for "enabled" mode. No inference traffic goes
        # through OpenRouter, we only use its catalog as a reference.
        slugs = candidate_openrouter_slugs(["anthropic"], model)
        reasoning_cfg = discover_reasoning_config(slugs)
        discovered_effort, supports_effort = highest_supported_effort(reasoning_cfg)

        if adaptive_effort is _UNSET:
            self.adaptive_effort = discovered_effort if supports_effort else "high"
            logger.info(
                "ClaudeProvider adaptive_effort auto-discovered for %s: %s "
                "(reasoning_cfg=%s)",
                model, self.adaptive_effort, reasoning_cfg,
            )
        else:
            self.adaptive_effort = str(adaptive_effort or "high").lower()
        if self.adaptive_effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError(
                'adaptive_effort must be one of {"low","medium","high","xhigh","max"}'
            )

        if thinking_budget_tokens is None:
            if supports_effort:
                ratio = min(anthropic_effort_ratio(discovered_effort), 0.8)
                thinking_budget_tokens = max(1024, int(max_tokens * ratio))
                logger.info(
                    "ClaudeProvider thinking_budget_tokens auto-discovered for %s: "
                    "effort=%s ratio=%.2f -> budget=%d tokens of max_tokens=%d "
                    "(reasoning_cfg=%s)",
                    model, discovered_effort, ratio, thinking_budget_tokens, max_tokens, reasoning_cfg,
                )
            else:
                # No OpenRouter effort data for this model (true today for
                # claude-sonnet-4-5) — keep the previous hardcoded default.
                thinking_budget_tokens = 4096
                logger.debug(
                    "No OpenRouter reasoning config found for %s; using default "
                    "thinking_budget_tokens=4096.",
                    model,
                )

        # Store requested thinking budget. Do not cap by constructor max_tokens here:
        # per-call max_tokens (e.g. from the judge runner --max-tokens) can be larger and is applied in call_llm().
        if self.thinking_mode != "disabled":
            if thinking_budget_tokens < 1024 and self.thinking_mode == "enabled":
                raise ValueError("thinking_budget_tokens must be at least 1024 for thinking_mode='enabled'")
            self.thinking_budget_tokens = max(1024, int(thinking_budget_tokens))
        else:
            self.thinking_budget_tokens = max(1024, int(thinking_budget_tokens))
        
        # Determine authentication method
        if use_foundry:
            # Use API key authentication (check both AZURE_* and ANTHROPIC_FOUNDRY_* for compatibility)
            api_key = api_key or os.getenv("AZURE_ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_FOUNDRY_API_KEY")
            if not api_key:
                raise ValueError(
                    "Foundry API key required. Set AZURE_ANTHROPIC_API_KEY or ANTHROPIC_FOUNDRY_API_KEY environment variable "
                    "or provide api_key parameter."
                )
            # base_url and resource are mutually exclusive - prefer resource if available (recommended for Azure Foundry)
            foundry_kwargs = {"api_key": api_key}
            if resource:
                foundry_kwargs["resource"] = resource
            elif base_url:
                # If base_url is provided, remove /messages suffix if present (SDK adds it)
                clean_base_url = base_url.rstrip('/messages').rstrip('/v1/messages')
                foundry_kwargs["base_url"] = clean_base_url
            
            self.client = anthropic.AnthropicFoundry(**foundry_kwargs)
        else:
            # Standard Anthropic API
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                    "or provide api_key parameter."
                )
            self.client = anthropic.Anthropic(api_key=api_key)
    
    def format_tools(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """
        Format tools for Claude API.
        
        Claude uses `input_schema` instead of `parameters` for tool definitions.
        According to MCP and Claude documentation, tools should have:
        - name: Tool name
        - description: Tool description
        - input_schema: JSON schema for tool inputs
        
        Args:
            tools: List of MCP tool objects from mcp_session.list_tools()
        """
        logger.info("Formatting %d tools for Claude API (from mcp_session.list_tools())", len(tools))
        
        # Log tool names being formatted
        tool_names = [_get_tool_attr(tool, 'name', 'unknown') for tool in tools]
        logger.debug("Tool names received: %s", tool_names)
        
        formatted = []
        for tool in tools:
            tool_name = _get_tool_attr(tool, 'name')
            if not tool_name:
                logger.warning("Skipping tool with no name: %s", tool)
                continue
            
            tool_description = _get_tool_attr(tool, 'description')
            schema = _get_tool_attr(tool, 'inputSchema') or _get_tool_attr(tool, 'parameters', {})
            
            # Ensure schema is a dict
            if not isinstance(schema, dict):
                schema = {}
            
            # Claude expects input_schema to be a JSON schema object
            # If schema doesn't have "type", wrap it properly
            if "type" not in schema:
                if schema:
                    # If schema has properties, it's likely already a schema structure
                    schema = {"type": "object", "properties": schema} if "properties" not in schema else schema
                else:
                    schema = {"type": "object", "properties": {}}
            
            formatted_tool = {
                "name": tool_name,
                "description": tool_description or f"Tool: {tool_name}",
                "input_schema": schema,
            }
            
            formatted.append(formatted_tool)
        
        # Log detailed tool information for each formatted tool
        logger.info("=" * 80)
        logger.info("TOOLS PROVIDED TO CLAUDE API")
        logger.info("=" * 80)
        for i, tool in enumerate(formatted, 1):
            logger.info("Tool %d:", i)
            logger.info("  Name: %s", tool["name"])
            logger.info("  Description: %s", tool["description"] or "(no description)")
            logger.info("  Input Schema: %s", json.dumps(tool["input_schema"], indent=4))
            logger.info("")
        logger.info("=" * 80)
        logger.info("Total: %d tools formatted for Claude API", len(formatted))
        logger.info("=" * 80)
        
        return formatted
    
    def _convert_messages_to_claude_format(self, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Convert standard message format to Claude API format.
        
        Claude uses a different message format:
        - System messages are separate (not in messages list)
        - Tool results are embedded in user messages as content blocks
        - Assistant messages with tool calls need special handling
        
        Returns:
            Tuple of (claude_messages, system_message)
        """
        claude_messages = []
        system_message = None
        
        for msg in messages:
            role = msg.get("role")
            
            if role == "system":
                # Extract system message (Claude handles it separately)
                system_message = msg.get("content", "")
                continue
            
            elif role == "user":
                content = msg.get("content", "")
                
                # Check if this is already a Claude-formatted message with content blocks
                if isinstance(content, list):
                    # Already in Claude format (e.g., tool results)
                    claude_messages.append({
                        "role": "user",
                        "content": content
                    })
                else:
                    # Plain text user message
                    claude_messages.append({
                        "role": "user",
                        "content": content
                    })
            
            elif role == "assistant":
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls", [])
                thinking_blocks = msg.get("thinking_blocks", [])
                
                # Check if this message is already in Claude format with content blocks
                # (e.g., from a previous API response that includes thinking blocks)
                if isinstance(content, list):
                    # Already in Claude format with content blocks
                    # IMPORTANT: If content list contains tool_use blocks, we must ensure tool_calls
                    # are also processed (they might be duplicates) or that tool_result blocks follow.
                    # Check if content has tool_use blocks
                    has_tool_use_in_content = any(
                        isinstance(block, dict) and block.get("type") == "tool_use"
                        for block in content
                    )
                    
                    # If content has tool_use blocks but we also have tool_calls separately,
                    # we should use tool_calls (which will be processed below) and filter out
                    # tool_use blocks from content to avoid duplication.
                    # However, if content has tool_use and no tool_calls, we preserve content as-is
                    # and rely on validation to ensure tool_result blocks follow.
                    if has_tool_use_in_content and tool_calls:
                        # Content has tool_use blocks AND we have tool_calls separately
                        # This suggests duplication - filter out tool_use blocks from content
                        # and let tool_calls be processed below
                        logger.warning(
                            "Assistant message has both content list with tool_use blocks and separate tool_calls. "
                            "Filtering tool_use blocks from content list to avoid duplication."
                        )
                        filtered_content = [
                            block for block in content
                            if not (isinstance(block, dict) and block.get("type") == "tool_use")
                        ]
                        # If filtered_content is empty or only has thinking blocks, use empty string
                        # so tool_calls are processed below
                        if not filtered_content or all(
                            isinstance(block, dict) and block.get("type") == "thinking"
                            for block in filtered_content
                        ):
                            content = ""  # Will be processed below with tool_calls
                        else:
                            # Keep non-tool_use, non-thinking blocks (e.g., text)
                            content = filtered_content
                            # Still need to process tool_calls below, so don't continue yet
                    
                    # The content list may already contain thinking blocks in their original sequence. 
                    # If thinking_blocks are provided separately, they should already be in the content list.
                    # Only merge if content doesn't already have thinking blocks AND we have separate thinking_blocks.
                    has_thinking_in_content = any(
                        isinstance(block, dict) and block.get("type") == "thinking" 
                        for block in content
                    )
                    
                    # Only use content as-is if we don't have tool_calls to process
                    if not tool_calls:
                        if thinking_blocks and not has_thinking_in_content:
                            # Content list doesn't have thinking blocks, but we have them separately
                            # Add them at the start, preserving their exact sequence
                            merged_content = []
                            for thinking_block in thinking_blocks:
                                if isinstance(thinking_block, dict):
                                    if thinking_block.get("type") == "thinking":
                                        # Pass back the complete, unmodified block
                                        merged_content.append(thinking_block)
                                    elif "thinking" in thinking_block:
                                        # Preserve all fields from the thinking block
                                        preserved_block = {
                                            "type": "thinking",
                                            "thinking": thinking_block.get("thinking", ""),
                                            "signature": thinking_block.get("signature", "")
                                        }
                                        # Preserve any additional fields
                                        for key, value in thinking_block.items():
                                            if key not in preserved_block:
                                                preserved_block[key] = value
                                        merged_content.append(preserved_block)
                            # Then add existing content blocks (preserving their order)
                            merged_content.extend(content)
                            claude_messages.append({
                                "role": "assistant",
                                "content": merged_content
                            })
                        else:
                            # Use content as-is (may already include thinking blocks in correct sequence)
                            # OR we don't have separate thinking blocks to merge
                            claude_messages.append({
                                "role": "assistant",
                                "content": content
                            })
                        continue
                    # If we have tool_calls, fall through to process them below
                
                # Build content blocks for assistant message
                content_blocks = []
                
                # Preserve thinking blocks from previous turns if present
                # According to Anthropic docs:
                # - "During tool use, you must pass thinking blocks back to the API for the last assistant message.
                #   Include the complete unmodified block back to the API to maintain reasoning continuity.".
                if thinking_blocks:
                    for thinking_block in thinking_blocks:
                        # Ensure thinking blocks are in the correct format
                        if isinstance(thinking_block, dict):
                            if thinking_block.get("type") == "thinking":
                                # Pass back the complete, unmodified block in its original position
                                content_blocks.append(thinking_block)
                            elif "thinking" in thinking_block:
                                # Handle summarized thinking format - preserve all fields
                                preserved_block = {
                                    "type": "thinking",
                                    "thinking": thinking_block.get("thinking", ""),
                                    "signature": thinking_block.get("signature", "")
                                }
                                # Preserve any additional fields
                                for key, value in thinking_block.items():
                                    if key not in preserved_block:
                                        preserved_block[key] = value
                                content_blocks.append(preserved_block)
                
                # Add tool use blocks if present (before text content)
                # According to Anthropic docs: "If thinking is enabled, the final assistant turn must start with a thinking block."
                # The order should be: thinking blocks -> tool_use blocks -> text content
                for tool_call in tool_calls:
                    # Handle both OpenAI format and Claude format
                    if "function" in tool_call:
                        # OpenAI format: {"id": "...", "function": {"name": "...", "arguments": "..."}}
                        func = tool_call.get("function", {})
                        tool_name = func.get("name", "")
                        arguments_str = func.get("arguments", "{}")
                        
                        # Parse arguments JSON string
                        try:
                            arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse tool arguments as JSON, using empty dict")
                            arguments = {}
                        
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tool_call.get("id", ""),
                            "name": tool_name,
                            "input": arguments,
                        })
                    elif "name" in tool_call:
                        # Already in Claude format
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tool_call.get("id", ""),
                            "name": tool_call.get("name", ""),
                            "input": tool_call.get("input", {}),
                        })
                
                # Add text content if present (after tool use blocks)
                # Note: When tool calls are present, text content typically comes in a later assistant message
                # after tool results are provided, but we handle it here for completeness.
                if content:
                    content_blocks.append({
                        "type": "text",
                        "text": content
                    })
                
                if content_blocks:
                    claude_messages.append({
                        "role": "assistant",
                        "content": content_blocks
                    })
            
            elif role == "tool":
                # Tool results should be formatted as user messages with tool_result content blocks
                # This format is handled by format_tool_response_message
                tool_call_id = msg.get("tool_call_id") or msg.get("call_id", "")
                tool_result = msg.get("content", "")
                
                # Check if already in Claude format
                if isinstance(tool_result, list):
                    claude_messages.append({
                        "role": "user",
                        "content": tool_result
                    })
                else:
                    claude_messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": tool_result
                        }]
                    })
        
        # CRITICAL VALIDATION: Ensure all tool_use blocks have corresponding tool_result blocks
        # Claude API requires that every tool_use block in an assistant message must be immediately
        # followed by a user message with tool_result blocks for ALL tool_use IDs.
        logger.debug("Validating Claude message format: checking %d messages", len(claude_messages))
        for i, msg in enumerate(claude_messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list):
                    # Extract all tool_use IDs from this assistant message
                    tool_use_ids = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            tool_use_id = block.get("id")
                            if tool_use_id:
                                tool_use_ids.append(tool_use_id)
                    
                    # If there are tool_use blocks, the next message must be a user message with tool_result blocks
                    if tool_use_ids:
                        logger.debug(
                            "Found assistant message at index %d with %d tool_use blocks: %s",
                            i, len(tool_use_ids), tool_use_ids[:3]  # Log first 3 IDs
                        )
                        if i + 1 < len(claude_messages):
                            next_msg = claude_messages[i + 1]
                            if next_msg.get("role") != "user":
                                logger.error(
                                    "Assistant message at index %d has tool_use blocks but next message is not a user message. "
                                    "This will cause Claude API error.",
                                    i
                                )
                                raise ValueError(
                                    f"Assistant message with tool_use blocks must be followed by a user message with tool_result blocks. "
                                    f"Found {next_msg.get('role')} message instead."
                                )
                        
                        # Extract all tool_use_ids from the next user message's tool_result blocks
                        next_content = next_msg.get("content", [])
                        if isinstance(next_content, list):
                            tool_result_ids = []
                            for block in next_content:
                                if isinstance(block, dict) and block.get("type") == "tool_result":
                                    tool_result_id = block.get("tool_use_id")
                                    if tool_result_id:
                                        tool_result_ids.append(tool_result_id)
                            
                            # Check if all tool_use_ids have corresponding tool_result blocks
                            missing_ids = set(tool_use_ids) - set(tool_result_ids)
                            if not missing_ids:
                                logger.debug(
                                    "✓ All %d tool_use blocks at index %d have corresponding tool_result blocks",
                                    len(tool_use_ids), i
                                )
                            if missing_ids:
                                logger.error(
                                    "Assistant message at index %d has tool_use blocks without corresponding tool_result blocks: %s. "
                                    "This will cause Claude API error.",
                                    i, missing_ids
                                )
                                raise ValueError(
                                    f"Missing tool_result blocks for tool_use IDs: {missing_ids}. "
                                    f"All tool_use blocks must have corresponding tool_result blocks in the next message."
                                )
                        else:
                            # Next message doesn't have tool_result blocks
                            logger.error(
                                "Assistant message at index %d has tool_use blocks but next user message doesn't have tool_result blocks. "
                                "This will cause Claude API error.",
                                i
                            )
                            raise ValueError(
                                f"Assistant message with tool_use blocks must be followed by a user message with tool_result content blocks."
                            )
                    elif tool_use_ids:
                        # Has tool_use blocks but no next message
                        logger.error(
                            "Assistant message at index %d has tool_use blocks but no following message with tool_result blocks. "
                            "This will cause Claude API error.",
                            i
                        )
                        raise ValueError(
                            f"Assistant message with tool_use blocks must be followed by a user message with tool_result blocks."
                        )
        
        return claude_messages, system_message
    
    async def call_llm(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[Any, Optional[str], List[Dict[str, Any]], Optional[str]]:
        """
        Call Claude API with messages and optional tools.
        
        Args:
            messages: Conversation history in standard format
            tools: Optional list of tools in Claude format
            max_tokens: Maximum tokens to generate (default: uses self.max_tokens from __init__)
        
        Returns:
            Tuple of (response, text_content, tool_calls, reasoning_summary)
            - response: Raw response object from Claude
            - text_content: Text response from Claude (None if only tool calls)
            - tool_calls: List of tool call dicts with 'id', 'name', 'arguments' keys
            - reasoning_summary: None (Claude doesn't provide reasoning summaries like GPT-5.1)
        """
        # Convert messages to Claude format
        claude_messages, system_message = self._convert_messages_to_claude_format(messages)
        
        # Use RESEARCH_ASSISTANT_PROMPT as system message if no system message is present
        if not system_message:
            system_message = RESEARCH_ASSISTANT_PROMPT

        # If tools are not available, explicitly instruct the model to answer directly.
        # Keep this aligned with GeminiProvider's no-tools instruction wording.
        if not tools:
            no_tools_instruction = "\n\nIMPORTANT: Tool calls are NOT available. Do NOT attempt to make any function calls or tool calls. Answer the question directly using only your knowledge and reasoning."
            system_message = system_message + no_tools_instruction
        
        # Log which system prompt is being used (helps verify RESEARCH_ASSISTANT_PROMPT usage)
        try:
            logger.info("=" * 80)
            logger.info("USING SYSTEM PROMPT FOR CLAUDE")
            logger.info("=" * 80)
            logger.info("System prompt length: %d characters", len(system_message))
            logger.info("System prompt preview: %s", system_message[:300].replace("\n", "\\n") + ("..." if len(system_message) > 300 else ""))
            logger.info("Tools passed: %d", len(tools) if tools else 0)
            logger.info("=" * 80)
        except Exception:
            # Never fail the request due to logging issues
            pass

        # Use provided max_tokens or fall back to instance default
        output_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        # API requires thinking budget_tokens >= 1024 when using type "enabled" (including adaptive fallback).
        thinking_budget = max(1024, min(self.thinking_budget_tokens, output_max_tokens - 1))
        
        # Build API call parameters
        kwargs = {
            "model": self.model,
            "max_tokens": output_max_tokens,
            "messages": claude_messages,
        }
        
        # Extended thinking
        if self.thinking_mode != "disabled":
            if self.temperature is not None:
                # According to docs: thinking isn't compatible with temperature modifications.
                logger.warning(
                    "Temperature specified but extended thinking is enabled. "
                    "Temperature is not compatible with thinking and will be ignored."
                )
        else:
            if self.temperature is not None:
                # Temperature is applied only when extended thinking is disabled.
                kwargs["temperature"] = float(self.temperature)

        # Always add system message (RESEARCH_ASSISTANT_PROMPT if not present in messages)
        kwargs["system"] = system_message
        if self.thinking_mode == "enabled":
            # Fixed budget extended thinking.
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
        elif self.thinking_mode == "adaptive":
            # Adaptive thinking: model decides when/how much to think. 
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": self.adaptive_effort}
        else:
            # thinking_mode == "disabled": do not set kwargs["thinking"].
            pass
        
        # Add tools if present
        if tools:
            kwargs["tools"] = tools
            tool_names = [tool.get("name", "unknown") for tool in tools]
            logger.info("Passing %d tools to Claude API: %s", len(tools), tool_names)
            
            # Note: tool_choice is not set, so it defaults to "auto" which is required for extended thinking.
            # Extended thinking only supports tool_choice: "auto" (default) or "none".
            # Forced tool use (tool_choice: "any" or tool_choice: {"type": "tool", "name": "..."}) 
            # is incompatible with extended thinking and will result in an error.
        else:
            logger.info("No tools passed to Claude API")
        
        # Log API call parameters (without sensitive data)
        logger.debug(
            "Claude API call: model=%s, messages=%d, tools=%d, system_prompt_length=%d, thinking_budget=%d",
            self.model, len(claude_messages), len(tools) if tools else 0, len(system_message), thinking_budget
        )
        
        # Anthropic client is sync, so we run it in executor
        loop = asyncio.get_event_loop()
        
        try:
            response = await loop.run_in_executor(None, lambda: self.client.messages.create(**kwargs))
        except anthropic.APIError as e:
            # Some models/resources don't support adaptive thinking. If we asked for adaptive and the API rejects it,
            # retry once with fixed-budget thinking (enabled) so callers don't have to special-case by model.
            try:
                err_msg = str(e)
            except Exception:
                err_msg = ""
            if (
                self.thinking_mode == "adaptive"
                and "adaptive thinking is not supported" in err_msg.lower()
            ):
                logger.warning(
                    "Adaptive thinking not supported for model %s; retrying with thinking_mode='enabled'.",
                    self.model,
                )
                kwargs_retry = dict(kwargs)
                kwargs_retry["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
                kwargs_retry.pop("output_config", None)  # only valid alongside adaptive thinking
                response = await loop.run_in_executor(None, lambda: self.client.messages.create(**kwargs_retry))
            elif (
                self.thinking_mode == "enabled"
                and '"thinking.type.enabled" is not supported' in err_msg
            ):
                # Newer models (e.g. claude-opus-5) only support adaptive
                # thinking — fixed budget_tokens is rejected outright. Retry
                # once with adaptive so callers don't have to special-case by
                # model; mirrors the opposite fallback above.
                logger.warning(
                    "Fixed-budget thinking not supported for model %s; retrying with "
                    "thinking_mode='adaptive' (output_config.effort=%s).",
                    self.model, self.adaptive_effort,
                )
                kwargs_retry = dict(kwargs)
                kwargs_retry["thinking"] = {"type": "adaptive"}
                kwargs_retry["output_config"] = {"effort": self.adaptive_effort}
                response = await loop.run_in_executor(None, lambda: self.client.messages.create(**kwargs_retry))
            else:
                # Handle context length errors
                error_msg = str(e)
                if 'context_length_exceeded' in error_msg.lower() or 'maximum context length' in error_msg.lower():
                    logger.warning(
                        "Context length exceeded for model %s. Messages: %d, Tools: %d",
                        self.model, len(messages), len(tools) if tools else 0
                    )
                    raise ContextLengthExceededError(
                        f"Input exceeds context window for model {self.model}. "
                        f"Messages: {len(messages)}, Tools: {len(tools) if tools else 0}"
                    ) from e
                raise
        
        # Parse response
        text_content = None
        tool_calls = []
        thinking_blocks = []
        
        # Preserve thinking blocks in their EXACT SEQUENCE as generated by the model.
        for content_block in response.content:
            if content_block.type == "thinking":
                # Preserve thinking blocks COMPLETELY and UNMODIFIED for subsequent requests.
                # This enables cache optimization and preserves reasoning across turns.
                
                # Extract all attributes from the content block to preserve it completely
                # The thinking block should include: type, thinking, and signature at minimum
                thinking_block = {
                    "type": "thinking",
                    "thinking": getattr(content_block, 'thinking', ''),
                    "signature": getattr(content_block, 'signature', '')
                }
                
                # Preserve any additional attributes that might be present
                # (e.g., if the SDK adds other fields in the future)
                if hasattr(content_block, 'model_dump'):
                    # Pydantic model - use model_dump to get all fields
                    full_block = content_block.model_dump()
                    thinking_block.update(full_block)
                elif hasattr(content_block, '__dict__'):
                    # Regular object - preserve all attributes
                    for key, value in content_block.__dict__.items():
                        if key not in thinking_block:
                            thinking_block[key] = value
                
                thinking_blocks.append(thinking_block)
                
                # Log thinking block content at INFO level for visibility
                thinking_text = thinking_block.get("thinking", "")
                thinking_length = len(thinking_text)
                signature = thinking_block.get("signature", "")
                
                # Log summary of thinking block
                block_index = len(thinking_blocks)
                logger.info("=" * 80)
                logger.info("THINKING BLOCK %d", block_index)
                logger.info("=" * 80)
                logger.info("Length: %d characters", thinking_length)
                if signature:
                    logger.info("Signature: %s", signature[:50] + "..." if len(signature) > 50 else signature)
                
                # Show preview of thinking content (first 1000 chars or full if shorter)
                preview_length = 1000
                if thinking_length <= preview_length:
                    logger.info("Thinking content (full):")
                    logger.info("-" * 80)
                    logger.info(thinking_text)
                else:
                    logger.info("Thinking content (first %d chars of %d):", preview_length, thinking_length)
                    logger.info("-" * 80)
                    logger.info(thinking_text[:preview_length])
                    logger.info("... [truncated, %d more characters]", thinking_length - preview_length)
                logger.info("=" * 80)
                
                # Also log at debug level with signature details
                logger.debug("Extracted thinking block (length: %d chars, signature: %s)", 
                           thinking_length, 
                           signature[:20] + "..." if signature else "None")
            elif content_block.type == "text":
                if text_content is None:
                    text_content = content_block.text
                else:
                    # Multiple text blocks - concatenate
                    text_content += "\n" + content_block.text
            elif content_block.type == "tool_use":
                # Claude returns tool_use blocks with id, name, and input
                # Convert to OpenAI-compatible format for MCP client compatibility
                # MCP client expects: {"id": "...", "function": {"name": "...", "arguments": "..."}}
                tool_calls.append({
                    "id": content_block.id,
                    "type": "function",
                    "function": {
                        "name": content_block.name,
                        "arguments": json.dumps(content_block.input),
                    },
                })
        
        # Store thinking blocks in response object for preservation
        if thinking_blocks:
            response.thinking_blocks = thinking_blocks
            total_thinking_chars = sum(len(block.get("thinking", "")) for block in thinking_blocks)
            logger.info("Preserved %d thinking block(s) from response (total: %d characters)", 
                       len(thinking_blocks), total_thinking_chars)
        
        return response, text_content, tool_calls, None
    
    def format_tool_response_message(
        self, tool_call_id: str, tool_name: str, tool_result: str
    ) -> Dict[str, Any]:
        """
        Format a tool response message for Claude.
        
        Claude expects tool results as user messages with tool_result content blocks:
        {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "...",
                "content": "..."  # JSON string of tool result
            }]
        }
        
        Note: For parallel tool calls, use format_multiple_tool_response_message instead.
        """
        return {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": tool_result  # Should already be a JSON string
            }]
        }
    
    def format_multiple_tool_response_message(
        self, tool_results: List[Union[Tuple[str, str, str], Tuple[str, str, str, Optional[bytes]]]]
    ) -> Dict[str, Any]:
        """
        Format multiple tool results into a single user message for Claude parallel tool calling.
        
        Claude can call multiple tools in parallel. All tool results must be provided
        in a single user message with multiple tool_result blocks.
        
        Args:
            tool_results: List of tuples (tool_call_id, tool_name, tool_result_str) or 
                         (tool_call_id, tool_name, tool_result_str, thought_signature)
                         The 4th element (thought_signature) is optional and ignored for Claude
        
        Returns:
            Single user message with all tool_result blocks in content array
        
        Example:
            tool_results = [
                ("toolu_01", "weather", '{"temp": 68}'),
                ("toolu_02", "time", '{"time": "2:30 PM"}')
            ]
            Returns:
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_01", "content": '{"temp": 68}'},
                    {"type": "tool_result", "tool_use_id": "toolu_02", "content": '{"time": "2:30 PM"}'}
                ]
            }
        """
        content_blocks = []
        for result_tuple in tool_results:
            # Handle both 3-tuple and 4-tuple formats (with/without thought_signature)
            # Claude doesn't use thought_signature, so we ignore the 4th element if present
            if len(result_tuple) >= 3:
                tool_call_id, tool_name, tool_result_str = result_tuple[0], result_tuple[1], result_tuple[2]
            else:
                logger.error(
                    "Invalid tool result tuple format: expected 3 or 4 elements, got %d: %s",
                    len(result_tuple), result_tuple
                )
                continue
            
            content_blocks.append({
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": tool_result_str  # Should already be a JSON string
            })
        
        logger.debug(
            "Formatting %d tool results into single Claude message for parallel tool calls",
            len(tool_results)
        )
        
        return {
            "role": "user",
            "content": content_blocks
        }


