"""Azure Foundry Chat Completions provider.

Supports OpenAI-compatible Chat Completions models hosted on Microsoft Foundry,
including:

- DeepSeek-V4-Pro

This model uses ``client.chat.completions.create`` (not the OpenAI Responses
API used by ``OpenAIProvider``). The wire format differs (flat vs. nested
function schema, ``role=tool`` messages vs. ``function_call_output`` items,
top-level ``instructions`` vs. an injected ``system`` message), but the
end-to-end contract matches ``OpenAIProvider`` exactly:

- Same ``RESEARCH_ASSISTANT_PROMPT`` system prompt, plus the same "tool calls
  are NOT available" fallback wording when no tools are supplied.
- Same tool-call loop shape: no ``format_multiple_tool_response_message`` is
  defined, so ``MCPClient`` executes/feeds back tool results one-by-one
  exactly like it does for ``OpenAIProvider`` (Chat Completions expects one
  ``role=tool`` message per ``tool_call_id``, matching how OpenAI's Responses
  API expects one ``function_call_output`` per ``call_id``).
- Same retry/error contract: rate-limit detection + backoff, timeout retry
  with exponential backoff, ``openai.BadRequestError`` inspection (via
  ``e.body``) to raise ``ContextLengthExceededError`` on context-window
  overflows, and a ``ValueError`` on 404/"model not found" so callers get the
  same exception types regardless of provider.

Reasoning defaults (maxed; sampling left at API defaults)
---------------------------------------------------------
**DeepSeek-V4-Pro**

- Thinking is on by default server-side.
- Effort: ``reasoning_effort="max"`` (API default is only ``"high"``).
  Values: ``"high"`` | ``"max"``.
- Do **not** send ``thinking`` / ``extra_body`` on Azure Foundry — Azure rejects
  them. Pass top-level ``reasoning_effort`` only.
- ``temperature`` / ``top_p`` / penalties are ignored while thinking is on;
  we omit them unless the caller sets them explicitly.

Credentials (same as Azure OpenAI / ``OpenAIProvider``)::

    AZURE_OPENAI_KEY=...
    OPENAI_BASE_URL=https://<resource>.services.ai.azure.com/openai/v1/
    OPENAI_API_VERSION=2025-04-01-preview   # classic AzureOpenAI path only

**Foundry Models v1** (``.../openai/v1/``): OpenAI SDK pointed at the Azure URL.
**Classic Azure OpenAI**: ``AzureOpenAI`` with ``azure_endpoint`` + ``api_version``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import traceback
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from openai import AzureOpenAI, BadRequestError, OpenAI

from .base import ContextLengthExceededError, LLMProvider
from ..prompts import RESEARCH_ASSISTANT_PROMPT

logger = logging.getLogger(__name__)

def parse_rate_limit_wait_time(error_message: str) -> Optional[float]:
    match = re.search(r"Please try again in ([\d.]+)s", error_message)
    if match:
        return float(match.group(1))
    return None


def is_rate_limit_error(error: Any) -> bool:
    if error is None:
        return False
    error_str = str(error).lower()
    return "rate limit" in error_str or "rate_limit" in error_str or "429" in error_str


def _get_tool_attr(tool: Any, attr: str, default: Any = None) -> Any:
    return getattr(tool, attr, None) or (tool.get(attr, default) if isinstance(tool, dict) else default)


class AzureChatCompletionsProvider(LLMProvider):
    """Azure Foundry provider using the OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ):
        # Pass deployment / catalog name through unchanged (must match Azure).
        super().__init__(model, api_key)

        api_key = (
            api_key
            or os.getenv("AZURE_OPENAI_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "Azure API key required. Set AZURE_OPENAI_KEY "
                "(or OPENAI_API_KEY)."
            )

        base_url = base_url or os.getenv("OPENAI_BASE_URL")
        if not base_url:
            raise ValueError(
                "Azure base URL required. Set OPENAI_BASE_URL (e.g. "
                "https://<resource>.services.ai.azure.com/openai/v1/) "
                "or pass --base-url."
            )

        self.base_url = base_url.rstrip("/")
        self.api_version = api_version or os.getenv(
            "OPENAI_API_VERSION", "2025-04-01-preview"
        )
        # Leave sampling at API defaults unless the caller overrides.
        self.temperature = temperature
        self.max_tokens = max_tokens

        model_lower = model.lower()
        self._is_deepseek = "deepseek" in model_lower

        # DeepSeek: max out reasoning (API default is only "high").
        # Azure Foundry rejects a "thinking" / extra_body payload for this model —
        # reasoning_effort must be passed as a top-level parameter only.
        if reasoning_effort is not None:
            self.reasoning_effort = reasoning_effort
        elif self._is_deepseek:
            self.reasoning_effort = "max"
        else:
            self.reasoning_effort = None

        self.client = self._build_client(api_key)
        logger.info(
            "AzureChatCompletionsProvider ready: model=%s endpoint=%s "
            "client=%s reasoning_effort=%s temperature=%s",
            self.model,
            self.base_url,
            type(self.client).__name__,
            self.reasoning_effort,
            self.temperature,
        )

    def _build_client(self, api_key: str):
        """Build OpenAI or AzureOpenAI client based on endpoint style."""
        # Foundry Models v1 API: .../openai/v1  — use OpenAI SDK against Azure URL.
        if "/openai/v1" in self.base_url:
            return OpenAI(api_key=api_key, base_url=self.base_url + "/")

        # Classic Azure OpenAI deployment path.
        api_version = self.api_version or "2025-04-01-preview"
        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=self.base_url,
            api_version=api_version,
        )

    def format_tools(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """Format MCP tools for Chat Completions function calling.

        Same contract as Claude/Gemini ``format_tools``: accept MCP tool objects
        from ``mcp_session.list_tools()`` and return provider-native schemas that
        ``MCPClient`` later passes into ``call_llm(..., tools=...)``.
        """
        logger.info(
            "Formatting %d tools for Azure Chat Completions (from mcp_session.list_tools())",
            len(tools),
        )
        tool_names = [_get_tool_attr(tool, "name", "unknown") for tool in tools]
        logger.debug("Tool names received: %s", tool_names)

        formatted = []
        for tool in tools:
            tool_name = _get_tool_attr(tool, "name")
            if not tool_name:
                logger.warning("Skipping tool with no name: %s", tool)
                continue

            tool_description = _get_tool_attr(tool, "description")
            schema = _get_tool_attr(tool, "inputSchema") or (
                _get_tool_attr(tool, "parameters", {})
            )

            if not isinstance(schema, dict):
                schema = {}
            if "type" not in schema:
                schema = (
                    {"type": "object", "properties": schema}
                    if schema
                    else {"type": "object", "properties": {}}
                )

            formatted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_description or f"Tool: {tool_name}",
                        "parameters": schema,
                    },
                }
            )

        logger.info(
            "Total: %d tools formatted for Azure Chat Completions API", len(formatted)
        )
        return formatted

    @staticmethod
    def _tools_already_formatted(tools: List[Any]) -> bool:
        """True when tools are already Chat Completions function schemas."""
        if not tools:
            return False
        first = tools[0]
        return isinstance(first, dict) and (
            ("function" in first and isinstance(first.get("function"), dict))
            or first.get("type") == "function"
        )

    def _prepare_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert harness message history into Chat Completions format.

        Mirrors Claude/Gemini:
        - Always injects ``RESEARCH_ASSISTANT_PROMPT`` as the system message.
        - Preserves assistant ``tool_calls`` + ``reasoning_content`` for the
          MCPClient tool loop (execute tools → feed role=tool results back).
        - Merges a content-only assistant message immediately followed by an
          assistant tool-call message (MCPClient reasoning-summary record).
        """
        prepared: List[Dict[str, Any]] = [
            {"role": "system", "content": RESEARCH_ASSISTANT_PROMPT}
        ]
        # Extra system messages (rare) are appended after the research prompt,
        # matching GeminiProvider's system_instruction_list behaviour.
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if content and content != RESEARCH_ASSISTANT_PROMPT:
                    prepared.append({"role": "system", "content": content})

        i = 0
        pending_reasoning: Optional[str] = None
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role")

            # Responses-API style tool outputs (should not appear for this provider).
            if msg.get("type") == "function_call_output":
                prepared.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.get("call_id", ""),
                        "content": msg.get("output", ""),
                    }
                )
                i += 1
                continue

            if role == "system":
                # Already handled above.
                i += 1
                continue

            if role == "user":
                pending_reasoning = None
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Flatten Claude-style content blocks if they ever appear.
                    text_parts = [
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    prepared.append(
                        {"role": "user", "content": "\n".join(text_parts) or str(content)}
                    )
                else:
                    prepared.append({"role": "user", "content": content})
                i += 1
                continue

            if role == "tool":
                # MCPClient → format_tool_response_message → role=tool
                prepared.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.get("tool_call_id", msg.get("id", "")),
                        "content": msg.get("content", ""),
                    }
                )
                i += 1
                continue

            if role == "assistant":
                tool_calls = msg.get("tool_calls") or []
                content = msg.get("content")
                reasoning_content = msg.get("reasoning_content")

                # Content-only assistant immediately before a tool-call turn is
                # the MCPClient reasoning-summary record — stash and fold in.
                next_msg = messages[i + 1] if i + 1 < len(messages) else None
                if (
                    not tool_calls
                    and next_msg
                    and next_msg.get("role") == "assistant"
                    and next_msg.get("tool_calls")
                ):
                    pending_reasoning = content or reasoning_content
                    i += 1
                    continue

                if pending_reasoning and not reasoning_content:
                    reasoning_content = pending_reasoning
                pending_reasoning = None

                # If previous prepared message was a content-only assistant and
                # this one has tool_calls, merge (handles already-appended case).
                if (
                    tool_calls
                    and prepared
                    and prepared[-1].get("role") == "assistant"
                    and not prepared[-1].get("tool_calls")
                ):
                    prev = prepared.pop()
                    if not reasoning_content and prev.get("content"):
                        reasoning_content = prev.get("content")
                    elif not content and prev.get("content"):
                        content = prev.get("content")

                # Chat Completions: use "" with tool_calls when content is absent
                # (DeepSeek samples use empty string, not null).
                if tool_calls and not content:
                    content = ""
                assistant_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": content if content is not None else None,
                }
                if reasoning_content:
                    assistant_msg["reasoning_content"] = reasoning_content

                if tool_calls:
                    normalized = []
                    for tc in tool_calls:
                        if "function" in tc:
                            func = tc.get("function", {})
                            args = func.get("arguments", "{}")
                            if not isinstance(args, str):
                                args = json.dumps(args) if args else "{}"
                            normalized.append(
                                {
                                    "id": tc.get("id", ""),
                                    "type": "function",
                                    "function": {
                                        "name": func.get("name", ""),
                                        "arguments": args or "{}",
                                    },
                                }
                            )
                        else:
                            # Gemini-flat / Claude-converted shape
                            args = tc.get("arguments", tc.get("input", "{}"))
                            if not isinstance(args, str):
                                args = json.dumps(args) if args else "{}"
                            normalized.append(
                                {
                                    "id": tc.get("id", ""),
                                    "type": "function",
                                    "function": {
                                        "name": tc.get("name", ""),
                                        "arguments": args or "{}",
                                    },
                                }
                            )
                    assistant_msg["tool_calls"] = normalized

                prepared.append(assistant_msg)
                i += 1
                continue

            # Unknown role — pass through best-effort.
            prepared.append(msg)
            i += 1

        return prepared

    def _parse_response(
        self, response: Any
    ) -> Tuple[Optional[str], List[Dict[str, Any]], Optional[str]]:
        """Extract text, tool calls, and reasoning from a Chat Completions response."""
        choice = response.choices[0] if response.choices else None
        if choice is None:
            return None, [], None

        message = choice.message
        text_content = message.content if message.content else None
        reasoning_content = getattr(message, "reasoning_content", None)

        tool_calls: List[Dict[str, Any]] = []
        for tc in message.tool_calls or []:
            arguments = getattr(tc.function, "arguments", None) or "{}"
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments) if arguments else "{}"
            if not arguments.strip():
                arguments = "{}"
            tool_calls.append(
                {
                    "id": getattr(tc, "id", "") or "",
                    "type": "function",
                    "function": {
                        "name": getattr(tc.function, "name", "") or "",
                        "arguments": arguments,
                    },
                }
            )

        return text_content, tool_calls, reasoning_content

    async def call_llm(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_retries: int = 3,
        retry_delay: float = 10.0,
    ) -> Tuple[Any, Optional[str], List[Dict[str, Any]], Optional[str]]:
        """Call Azure Chat Completions with retries for timeouts / rate limits.

        Contract matches Claude/Gemini ``call_llm``:
        - Injects ``RESEARCH_ASSISTANT_PROMPT``
        - Accepts MCPClient-formatted tools (or raw MCP tools)
        - Returns ``(response, text_content, tool_calls, reasoning_summary)``
          where ``tool_calls`` use OpenAI shape
          ``{"id", "type": "function", "function": {"name", "arguments"}}``
          so ``ToolExecutor.extract_tool_call_info`` / MCP execution work unchanged.
        """
        api_messages = self._prepare_messages(messages)

        tools_to_use = None
        if tools:
            # MCPClient normally pre-formats via format_tools(); accept either shape.
            if self._tools_already_formatted(tools):
                tools_to_use = tools
            else:
                tools_to_use = self.format_tools(tools)

        # Same no-tools instruction wording as ClaudeProvider / GeminiProvider / OpenAIProvider.
        if not tools_to_use:
            no_tools = (
                "\n\nIMPORTANT: Tool calls are NOT available. Do NOT attempt to "
                "make any function calls or tool calls. Answer the question "
                "directly using only your knowledge and reasoning."
            )
            if api_messages and api_messages[0].get("role") == "system":
                api_messages[0]["content"] = (
                    (api_messages[0].get("content") or "") + no_tools
                )

        system_message = next(
            (m.get("content", "") for m in api_messages if m.get("role") == "system"),
            "",
        )
        try:
            logger.info("=" * 80)
            logger.info("USING SYSTEM PROMPT FOR AZURE CHAT COMPLETIONS")
            logger.info("=" * 80)
            logger.info("System prompt length: %d characters", len(system_message))
            logger.info(
                "System prompt preview: %s",
                system_message[:300].replace("\n", "\\n")
                + ("..." if len(system_message) > 300 else ""),
            )
            logger.info("Tools passed: %d", len(tools_to_use) if tools_to_use else 0)
            logger.info("=" * 80)
        except Exception:
            pass

        api_params: Dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
        }
        if tools_to_use:
            api_params["tools"] = tools_to_use
            api_params["tool_choice"] = "auto"
            tool_names = [
                (t.get("function") or {}).get("name", t.get("name", "unknown"))
                for t in tools_to_use
                if isinstance(t, dict)
            ]
            logger.info(
                "Passing %d tools to Azure Chat Completions API: %s",
                len(tools_to_use),
                tool_names,
            )
        else:
            logger.info("No tools passed to Azure Chat Completions API")

        # Sampling: omit unless explicitly set (API defaults; ignored while thinking is on).
        if self.temperature is not None:
            api_params["temperature"] = self.temperature
        if self.max_tokens is not None:
            api_params["max_tokens"] = self.max_tokens
        # DeepSeek-V4: top-level reasoning_effort (Azure rejects thinking/extra_body).
        if self.reasoning_effort:
            api_params["reasoning_effort"] = self.reasoning_effort

        logger.debug(
            "Azure API call: model=%s, messages=%d, tools=%d, system_prompt_length=%d, "
            "reasoning_effort=%s",
            self.model,
            len(api_messages),
            len(tools_to_use) if tools_to_use else 0,
            len(system_message),
            self.reasoning_effort,
        )

        # Max reasoning / long tool loops can exceed 5 minutes.
        timeout_seconds = 600 if self._is_deepseek else 300
        loop = asyncio.get_event_loop()
        last_exception: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                logger.debug(
                    "Calling Azure chat.completions (attempt %d/%d, model=%s)",
                    attempt + 1,
                    max_retries,
                    self.model,
                )

                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self.client.chat.completions.create(**api_params),
                    ),
                    timeout=timeout_seconds,
                )

                text_content, tool_calls, reasoning_content = self._parse_response(
                    response
                )

                # Attach reasoning_content so MCPClient can preserve it on the
                # assistant message (mirrors Claude thinking_blocks).
                wrapped = SimpleNamespace(
                    usage=getattr(response, "usage", None),
                    reasoning_content=reasoning_content,
                    raw=response,
                )

                if reasoning_content:
                    logger.info(
                        "Azure reasoning_content length: %d chars",
                        len(reasoning_content),
                    )
                if tool_calls:
                    logger.info(
                        "Azure returned %d tool call(s): %s",
                        len(tool_calls),
                        [tc["function"]["name"] for tc in tool_calls],
                    )

                return wrapped, text_content, tool_calls, reasoning_content

            except asyncio.TimeoutError as e:
                last_exception = e
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(
                    "Azure chat.completions timeout (attempt %d/%d, timeout=%ds); "
                    "model=%s messages=%d tools=%d",
                    attempt + 1,
                    max_retries,
                    timeout_seconds,
                    self.model,
                    len(api_messages),
                    len(tools_to_use) if tools_to_use else 0,
                )
                if attempt < max_retries - 1:
                    logger.warning("Retrying in %.1fs...", wait_time)
                    await asyncio.sleep(wait_time)
                    continue
                error_msg = (
                    f"Azure chat.completions call timed out after {timeout_seconds}s. "
                    f"Failed after {max_retries} attempts."
                )
                logger.error(error_msg)
                raise TimeoutError(error_msg) from e

            except BadRequestError as e:
                # Same detection strategy as OpenAIProvider: prefer the structured
                # error body (Chat Completions returns the same
                # {"error": {"code": "context_length_exceeded", ...}} shape as the
                # Responses API), and fall back to substring matching on str(e).
                error_msg = str(e)
                is_context_length_error = "context_length_exceeded" in error_msg

                if not is_context_length_error and hasattr(e, "body") and e.body:
                    try:
                        if isinstance(e.body, dict):
                            error_obj = e.body.get("error", {})
                            is_context_length_error = (
                                error_obj.get("code") == "context_length_exceeded"
                            )
                        elif isinstance(e.body, str):
                            error_body = json.loads(e.body)
                            error_obj = error_body.get("error", {})
                            is_context_length_error = (
                                error_obj.get("code") == "context_length_exceeded"
                            )
                    except Exception:
                        pass

                if not is_context_length_error:
                    is_context_length_error = any(
                        phrase in error_msg.lower()
                        for phrase in (
                            "maximum context length",
                            "context window",
                            "too many tokens",
                        )
                    )

                if is_context_length_error:
                    logger.warning(
                        "Context length exceeded for model %s. Messages: %d, Tools: %d",
                        self.model, len(api_messages), len(tools_to_use) if tools_to_use else 0,
                    )
                    raise ContextLengthExceededError(
                        f"Input exceeds context window for model {self.model}. "
                        f"Messages: {len(api_messages)}, Tools: {len(tools_to_use) if tools_to_use else 0}"
                    ) from e

                last_exception = e
                if is_rate_limit_error(e) and attempt < max_retries - 1:
                    wait_time = parse_rate_limit_wait_time(error_msg)
                    if wait_time is None:
                        wait_time = retry_delay * (2 ** attempt)
                    else:
                        wait_time = wait_time + 1.0
                    logger.warning(
                        "Azure rate limit (attempt %d/%d); waiting %.1fs",
                        attempt + 1, max_retries, wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    continue

                self._log_error(e, error_msg, api_messages, tools_to_use)

                if "404" in error_msg or "not found" in error_msg.lower():
                    raise ValueError(
                        f"Model '{self.model}' not found or not deployed on this "
                        f"Azure endpoint. Original error: {error_msg}"
                    ) from e
                raise

            except Exception as e:
                last_exception = e
                error_msg = str(e)

                if is_rate_limit_error(e) and attempt < max_retries - 1:
                    wait_time = parse_rate_limit_wait_time(error_msg)
                    if wait_time is None:
                        wait_time = retry_delay * (2 ** attempt)
                    else:
                        wait_time = wait_time + 1.0
                    logger.warning(
                        "Azure rate limit (attempt %d/%d); waiting %.1fs",
                        attempt + 1,
                        max_retries,
                        wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    continue

                # Fallback string-match in case a non-BadRequestError transport
                # error still encodes a context-window overflow.
                if any(
                    phrase in error_msg.lower()
                    for phrase in (
                        "context_length_exceeded",
                        "maximum context length",
                        "context window",
                        "too many tokens",
                    )
                ):
                    raise ContextLengthExceededError(error_msg) from e

                self._log_error(e, error_msg, api_messages, tools_to_use)

                if "404" in error_msg or "not found" in error_msg.lower():
                    raise ValueError(
                        f"Model '{self.model}' not found or not deployed on this "
                        f"Azure endpoint. Original error: {error_msg}"
                    ) from e

                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Azure chat.completions call failed without exception")

    def _log_error(
        self,
        e: Exception,
        error_msg: str,
        api_messages: List[Dict[str, Any]],
        tools_to_use: Optional[List[Dict[str, Any]]],
    ) -> None:
        """Log a detailed error block, mirroring OpenAIProvider's diagnostics."""
        error_details = [
            "=" * 80,
            "AZURE CHAT COMPLETIONS ERROR",
            "=" * 80,
            f"Exception Type: {type(e).__name__}",
            f"Exception Message: {error_msg}",
            "",
            "API Call Details:",
            f"  Model: {self.model}",
            f"  Endpoint: {self.base_url}",
            f"  Messages: {len(api_messages)}",
            f"  Tools: {len(tools_to_use) if tools_to_use else 0}",
            f"  Reasoning Effort: {self.reasoning_effort}",
            f"  Temperature: {self.temperature}",
            "",
            "Full Stack Trace:",
            traceback.format_exc(),
            "=" * 80,
        ]
        error_msg_full = "\n".join(error_details)
        print(error_msg_full, file=sys.stderr)
        logger.error(error_msg_full)

    def format_tool_response_message(
        self, tool_call_id: str, tool_name: str, tool_result: str
    ) -> Dict[str, Any]:
        """Format a tool result for Chat Completions (role=tool)."""
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": tool_result,
        }
