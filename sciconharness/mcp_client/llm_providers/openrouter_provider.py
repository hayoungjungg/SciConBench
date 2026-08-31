"""OpenRouter provider (OpenAI-compatible Chat Completions API).

Supports models hosted on OpenRouter, including:

- Kimi K3       (``moonshotai/kimi-k3``)
- GLM-5.2 / 5.3 (``z-ai/glm-5.2``, ``z-ai/glm-5.3``)
- Qwen3.5-9B    (``qwen/qwen3.5-9b``)
- Qwen3.7-max   (``qwen/qwen3.7-max``)
- Qwen3.8-max   (``qwen/qwen3.8-max``)
- Qwen3.8 27B   (``qwen/qwen3.8-27b``)   # SciConBench-Track control
- MiniMax M3    (``minimax/minimax-m3``)  # SciConBench-Track control

OpenRouter exposes an OpenAI-compatible ``client.chat.completions.create``
endpoint at ``https://openrouter.ai/api/v1`` (same SDK call shape as
``AzureChatCompletionsProvider`` / ``OpenAIProvider``'s Chat Completions
path). The wire format matches ``AzureChatCompletionsProvider`` exactly
(nested ``function`` tool schema, ``role=tool`` result messages), so this
provider reuses the same message-preparation and tool-loop logic. The
end-to-end contract also matches ``OpenAIProvider``:

- Same ``RESEARCH_ASSISTANT_PROMPT`` system prompt, plus the same "tool calls
  are NOT available" fallback wording when no tools are supplied.
- Same tool-call loop shape: no ``format_multiple_tool_response_message`` is
  defined, so ``MCPClient`` executes/feeds back tool results one-by-one.
- Same retry/error contract: rate-limit detection + backoff, timeout retry
  with exponential backoff, ``openai.BadRequestError`` inspection (via
  ``e.body``) to raise ``ContextLengthExceededError`` on context-window
  overflows, and a ``ValueError`` on 404/"model not found".

Reasoning (systematically maxed; sampling left at API defaults)
-----------------------------------------------------------------
OpenRouter exposes a unified ``reasoning`` object across every provider it
routes to::

    extra_body={"reasoning": {"enabled": True, "effort": "max"}}

(The OpenAI Python SDK has no native ``reasoning`` field, so — exactly like
Kimi's ``thinking`` param previously did for Azure — it must be passed via
``extra_body``.)

Rather than hardcoding ``effort="max"`` and hoping OpenRouter's clamping
picks the right ceiling, ``__init__`` *discovers* each model's actual
supported effort levels via ``GET /models`` (see ``_discover_reasoning_config``
/ ``_highest_supported_effort`` below, per
https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#discovering-per-model-reasoning-options)
and sends the true highest level explicitly, cached per model for the
process lifetime::

    moonshotai/kimi-k3  -> supported_efforts=["max", "high", "low"]   -> enabled + effort="max"
    z-ai/glm-5.3        -> supported_efforts=["max", "high", "low"]   -> enabled + effort="max"
    qwen/qwen3.8-max    -> supported_efforts=["xhigh", "high", ...]   -> enabled + effort="xhigh"
    qwen/qwen3.8-27b    -> supported_efforts=["xhigh", "medium", "low"] -> enabled + effort="xhigh"
    minimax/minimax-m3  -> no `supported_efforts` / no reasoning_effort -> enabled only

If discovery fails (network error, model not listed, etc.) we fall back to
``effort="max"``, which OpenRouter clamps down to whatever the model
actually supports.

Reasoning is always turned on (``reasoning.enabled=True``) for these models.
Top-level completion ``max_tokens`` defaults to 8192 (``_DEFAULT_MAX_TOKENS``;
pipeline YAML should match). The nested reasoning object never includes
``max_tokens``: effort-capable models send their discovered ceiling
(``max`` / ``xhigh``), while MiniMax sends only ``enabled=True`` because it
does not expose effort selection.

We only enable reasoning at all for the reasoning-capable models this
provider was built for (matched by a case-insensitive substring on the model
slug: ``kimi``, ``glm``, ``qwen``, ``minimax``) so that passing some other,
non-reasoning OpenRouter model through this same class doesn't send a
``reasoning`` payload it doesn't understand. Pass ``reasoning_effort``
explicitly to override discovery for any model.

Response parsing: OpenRouter returns reasoning text on ``message.reasoning``
(``message.reasoning_content`` is documented as an equivalent alias) plus the
full structured ``message.reasoning_details`` array. We read both. Per
OpenRouter's "Preserving Reasoning" guidance
(https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#preserving-reasoning-blocks),
``reasoning_details`` is the more robust of the two to echo back on the next
turn — it's the exact, unmodified sequence of reasoning blocks the model
produced, so tool-calling round trips resume reasoning state precisely
instead of relying on a lossy plaintext re-summary. We therefore prefer
``reasoning_details`` when present and only fall back to the plaintext
``reasoning_content`` alias for models/turns that don't return it.
``MCPClient`` preserves both fields on the assistant message the same way it
already preserves Claude's ``thinking_blocks`` / Azure's ``reasoning_content``
(a plain ``hasattr`` check — harmless no-op for every other provider).

Credentials::

    OPENROUTER_API_KEY=...
    OPENROUTER_BASE_URL=https://openrouter.ai/api/v1   # optional override
    OPENROUTER_SITE_URL=...   # optional, for openrouter.ai leaderboard attribution
    OPENROUTER_SITE_NAME=...  # optional, ditto

Prompt caching / sticky routing (``session_id``)
-------------------------------------------------
OpenRouter routes each request independently by default, which means the
provider serving turn N of a tool-calling conversation may differ from the
one that served turn N-1, defeating that provider's own prompt cache even
though every turn resends the (growing) full message history. Passing a
stable ``session_id`` opts a conversation into *sticky routing*: OpenRouter
prefers the same upstream provider for every request sharing that id, which
is what actually gives the growing shared prefix (system prompt + tool defs
+ earlier turns) a chance to hit cache.

``SciConHarness.query()`` calls ``set_session_id()`` once per query attempt
(before the tool-calling loop starts) with an id scoped to
``{provider}:{model}:{doi}:{attempt}:{random}`` — see
``harness.py::_build_session_id``. Every ``call_llm()`` invocation within
that one query's loop reuses the same id (sent via ``extra_body["session_id"]``),
so all its turns route together, while a different query — even the exact
same DOI against a different model/config, or a rerun of the same
(doi, model) pair — gets a fresh, unrelated id. Callers that don't use
``SciConHarness`` (e.g. direct ``OpenRouterProvider`` use) can call
``set_session_id()`` themselves, or leave it unset to fall back to
OpenRouter's default per-request routing.
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

from openai import BadRequestError, OpenAI

from .base import ContextLengthExceededError, LLMProvider
from .reasoning_discovery import discover_reasoning_config, highest_supported_effort
from ..prompts import RESEARCH_ASSISTANT_PROMPT

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# Reasoning-capable models this provider was built for. Substring match
# (case-insensitive) against the OpenRouter model slug. ``qwen`` covers
# qwen3.5 / 3.7 / 3.8 (max and 27B); ``minimax`` covers MiniMax M3.
_REASONING_MODEL_HINTS = ("kimi", "glm", "qwen", "minimax")

# Default top-level completion cap when the caller does not pass ``max_tokens``.
_DEFAULT_MAX_TOKENS = 8192


def parse_rate_limit_wait_time(error_message: str) -> Optional[float]:
    """Extract a wait duration (seconds) from an OpenRouter / OpenAI error string.

    Handles the usual ``Please try again in Ns`` prose plus OpenRouter's
    ``Retry-After`` field embedded in 402 ``in_flight_budget_exhausted`` /
    429 payloads (often as ``'Retry-After': '120'`` inside the message).
    """
    if not error_message:
        return None
    match = re.search(r"Please try again in ([\d.]+)s", error_message, re.I)
    if match:
        return float(match.group(1))
    # OpenRouter embeds headers in the error body / message, e.g.
    # {'Retry-After': '120'} or "Retry-After": "120"
    match = re.search(
        r"Retry-After['\"\s:=]+['\"]?(\d+(?:\.\d+)?)",
        error_message,
        re.I,
    )
    if match:
        return float(match.group(1))
    return None


# Floor / stretch for 429 and 402 in_flight retries. In-flight generations can
# hold budget for many minutes; short waits just amplify the storm.
RATE_LIMIT_MIN_WAIT_SECS = 300.0  # 5 minutes
RATE_LIMIT_WAIT_MULTIPLIER = 2.0  # stretch Retry-After hints


def rate_limit_backoff_secs(
    error_message: str,
    *,
    attempt: int = 0,
    retry_delay: float = 60.0,
) -> float:
    """Long backoff for OpenRouter 429 / in-flight budget errors.

    Uses ``max(Retry-After * multiplier, MIN)``, or exponential from the min
    floor when no hint is present.
    """
    hinted = parse_rate_limit_wait_time(error_message)
    if hinted is not None:
        wait = float(hinted) * RATE_LIMIT_WAIT_MULTIPLIER + 1.0
    else:
        wait = max(retry_delay, RATE_LIMIT_MIN_WAIT_SECS) * (2 ** attempt)
    return max(wait, RATE_LIMIT_MIN_WAIT_SECS)


def is_rate_limit_error(error: Any) -> bool:
    """True for transient OpenRouter throttles that should back off + retry.

    Includes classic 429s and OpenRouter's 402 ``in_flight_budget_exhausted``
    (credits reserved by concurrent requests — Retry-After applies; hammering
    makes it worse). Also treats nested ``limit_rpd`` / daily-limit prose as
    retryable so we at least wait instead of spinning.
    """
    if error is None:
        return False
    # Prefer structured status when the OpenAI SDK attached one.
    status = getattr(error, "status_code", None)
    if status in (429, 503):
        return True
    if status == 402:
        body = str(getattr(error, "body", "") or "") + str(error)
        body_l = body.lower()
        if (
            "in_flight" in body_l
            or "in-flight" in body_l
            or "retry-after" in body_l
            or "rate limit" in body_l
            or "limit_rpd" in body_l
        ):
            return True
    error_str = str(error).lower()
    return (
        "rate limit" in error_str
        or "rate_limit" in error_str
        or "429" in error_str
        or "in_flight_budget" in error_str
        or "in-flight requests" in error_str
        or "limit_rpd" in error_str
    )


def _get_tool_attr(tool: Any, attr: str, default: Any = None) -> Any:
    return getattr(tool, attr, None) or (tool.get(attr, default) if isinstance(tool, dict) else default)


class OpenRouterProvider(LLMProvider):
    """OpenRouter provider using the OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        # Pass the OpenRouter model slug through unchanged (e.g. "moonshotai/kimi-k3").
        super().__init__(model, api_key)

        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY."
            )

        self.base_url = (base_url or os.getenv("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")

        # Leave sampling at API defaults unless the caller overrides.
        self.temperature = temperature
        # Top-level completion cap. This is distinct from the nested
        # reasoning.max_tokens field, which this provider never sends.
        self.max_tokens = max_tokens if max_tokens is not None else _DEFAULT_MAX_TOKENS

        # Sticky-routing key for prompt-cache affinity across a single
        # conversation's tool-calling loop (see module docstring). Usually
        # set per-query via set_session_id() rather than at construction time.
        self.session_id = session_id

        model_lower = model.lower()
        self._is_reasoning_model = any(hint in model_lower for hint in _REASONING_MODEL_HINTS)

        # Optional attribution headers for the OpenRouter leaderboard (harmless if unset).
        extra_headers = {}
        site_url = os.getenv("OPENROUTER_SITE_URL")
        site_name = os.getenv("OPENROUTER_SITE_NAME")
        if site_url:
            extra_headers["HTTP-Referer"] = site_url
        if site_name:
            extra_headers["X-Title"] = site_name

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            default_headers=extra_headers or None,
        )

        # Max out reasoning for the models this provider targets via
        # GET /models discovery (see reasoning_discovery.py). Current
        # SciConBench OpenRouter slugs (verified against OpenRouter catalog):
        #   kimi-k3 / glm-5.3     -> supported_efforts incl. "max"
        #   qwen3.8-max / 27b     -> supported_efforts incl. "xhigh"
        #   minimax-m3            -> reasoning yes, no reasoning_effort
        self._reasoning_enabled = False
        if reasoning_effort is not None:
            self.reasoning_effort = reasoning_effort
            self._reasoning_enabled = True
        elif self._is_reasoning_model:
            self._reasoning_enabled = True
            reasoning_cfg = discover_reasoning_config([self.model], client=self.client)
            discovered_effort, supports_effort = highest_supported_effort(reasoning_cfg)
            self.reasoning_effort = discovered_effort if supports_effort else None
            if reasoning_cfg:
                logger.info(
                    "Discovered OpenRouter reasoning config for %s: %s -> using effort=%s",
                    self.model,
                    reasoning_cfg,
                    self.reasoning_effort or "(none — reasoning enabled only)",
                )
            else:
                # Discovery failed — "max" is a safe default; OpenRouter clamps
                # unsupported effort values to the nearest supported one.
                self.reasoning_effort = "max"
                logger.info(
                    "No reasoning config discovered for %s via GET /models; "
                    "defaulting to effort=max (OpenRouter clamps unsupported values).",
                    self.model,
                )
        else:
            self.reasoning_effort = None

        logger.info(
            "OpenRouterProvider ready: model=%s endpoint=%s reasoning_enabled=%s "
            "reasoning_effort=%s temperature=%s",
            self.model,
            self.base_url,
            self._reasoning_enabled,
            self.reasoning_effort,
            self.temperature,
        )

    def set_session_id(self, session_id: Optional[str]) -> None:
        """Set the sticky-routing ``session_id`` used by subsequent ``call_llm()``
        calls (see module docstring). Call this once per logical conversation
        — e.g. once per ``SciConHarness.query()`` attempt — *before* its
        tool-calling loop starts, so every turn in that loop routes to the
        same upstream provider and can actually hit that provider's prompt
        cache. Pass ``None`` to disable sticky routing again.
        """
        self.session_id = session_id

    def format_tools(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """Format MCP tools for Chat Completions function calling.

        Same contract as Claude/Gemini/Azure ``format_tools``: accept MCP tool
        objects from ``mcp_session.list_tools()`` and return provider-native
        schemas that ``MCPClient`` later passes into ``call_llm(..., tools=...)``.
        """
        logger.info(
            "Formatting %d tools for OpenRouter (from mcp_session.list_tools())",
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

        logger.info("Total: %d tools formatted for OpenRouter API", len(formatted))
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

        Identical logic to ``AzureChatCompletionsProvider._prepare_messages``:
        - Always injects ``RESEARCH_ASSISTANT_PROMPT`` as the system message.
        - Preserves assistant ``tool_calls`` + ``reasoning_content`` for the
          MCPClient tool loop (execute tools → feed role=tool results back).
        - Merges a content-only assistant message immediately followed by an
          assistant tool-call message (MCPClient reasoning-summary record).
        """
        prepared: List[Dict[str, Any]] = [
            {"role": "system", "content": RESEARCH_ASSISTANT_PROMPT}
        ]
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if content and content != RESEARCH_ASSISTANT_PROMPT:
                    prepared.append({"role": "system", "content": content})

        i = 0
        pending_reasoning: Optional[str] = None
        pending_reasoning_details: Optional[List[Dict[str, Any]]] = None
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
                i += 1
                continue

            if role == "user":
                pending_reasoning = None
                pending_reasoning_details = None
                content = msg.get("content", "")
                if isinstance(content, list):
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
                reasoning_details = msg.get("reasoning_details")

                next_msg = messages[i + 1] if i + 1 < len(messages) else None
                if (
                    not tool_calls
                    and next_msg
                    and next_msg.get("role") == "assistant"
                    and next_msg.get("tool_calls")
                ):
                    pending_reasoning = content or reasoning_content
                    pending_reasoning_details = reasoning_details
                    i += 1
                    continue

                if pending_reasoning and not reasoning_content:
                    reasoning_content = pending_reasoning
                if pending_reasoning_details and not reasoning_details:
                    reasoning_details = pending_reasoning_details
                pending_reasoning = None
                pending_reasoning_details = None

                if (
                    tool_calls
                    and prepared
                    and prepared[-1].get("role") == "assistant"
                    and not prepared[-1].get("tool_calls")
                ):
                    prev = prepared.pop()
                    if not reasoning_details and prev.get("reasoning_details"):
                        reasoning_details = prev.get("reasoning_details")
                    if not reasoning_content and prev.get("content"):
                        reasoning_content = prev.get("content")
                    elif not content and prev.get("content"):
                        content = prev.get("content")

                if tool_calls and not content:
                    content = ""
                assistant_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": content if content is not None else None,
                }
                if reasoning_details:
                    # Preferred per OpenRouter's docs: pass the full structured
                    # reasoning_details block back verbatim (unmodified, same order)
                    # for exact reasoning-state continuity across tool-call turns.
                    # https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#preserving-reasoning-blocks
                    assistant_msg["reasoning_details"] = reasoning_details
                elif reasoning_content:
                    # "reasoning_content" is documented by OpenRouter as an alias
                    # for "reasoning" when sending messages back on later turns.
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
    ) -> Tuple[Optional[str], List[Dict[str, Any]], Optional[str], Optional[List[Dict[str, Any]]]]:
        """Extract text, tool calls, and reasoning from a Chat Completions response.

        Returns ``(text_content, tool_calls, reasoning_content, reasoning_details)``.
        ``reasoning_details`` is the full structured block OpenRouter recommends
        preserving verbatim for exact reasoning-state continuity (see
        https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#preserving-reasoning-blocks);
        ``reasoning_content`` is the simpler plaintext fallback.
        """
        choice = response.choices[0] if response.choices else None
        if choice is None:
            return None, [], None, None

        message = choice.message
        text_content = message.content if message.content else None
        # OpenRouter's own field is "reasoning"; "reasoning_content" is a
        # documented alias some upstream providers also populate. Check both.
        reasoning_content = getattr(message, "reasoning", None) or getattr(
            message, "reasoning_content", None
        )
        reasoning_details = getattr(message, "reasoning_details", None) or None

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

        return text_content, tool_calls, reasoning_content, reasoning_details

    async def call_llm(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_retries: int = 3,
        retry_delay: float = 60.0,
    ) -> Tuple[Any, Optional[str], List[Dict[str, Any]], Optional[str]]:
        """Call OpenRouter Chat Completions with retries for timeouts / rate limits.

        Rate/in-flight limits use a long backoff floor (see
        ``rate_limit_backoff_secs``) so retries do not hammer the account.

        Contract matches ``AzureChatCompletionsProvider`` / ``OpenAIProvider``
        ``call_llm``:
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
            if self._tools_already_formatted(tools):
                tools_to_use = tools
            else:
                tools_to_use = self.format_tools(tools)

        # Same no-tools instruction wording as ClaudeProvider / GeminiProvider / OpenAIProvider / AzureChatCompletionsProvider.
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
            logger.info("USING SYSTEM PROMPT FOR OPENROUTER")
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
                "Passing %d tools to OpenRouter API: %s",
                len(tools_to_use),
                tool_names,
            )
        else:
            logger.info("No tools passed to OpenRouter API")

        # Sampling: omit unless explicitly set (API defaults; ignored while thinking is on).
        if self.temperature is not None:
            api_params["temperature"] = self.temperature
        if self.max_tokens is not None:
            api_params["max_tokens"] = self.max_tokens
        # Unified OpenRouter reasoning object — OpenAI SDK has no native field for
        # this, so it must go through extra_body. Always send "enabled": True when
        # reasoning is on. Never send nested reasoning.max_tokens: OpenRouter
        # rejects it when effort is present. Budget-only models send enabled
        # without effort and use the gateway/model's native reasoning budget.
        extra_body: Dict[str, Any] = {}
        if self._reasoning_enabled:
            reasoning_obj: Dict[str, Any] = {"enabled": True}
            if self.reasoning_effort:
                reasoning_obj["effort"] = self.reasoning_effort
            extra_body["reasoning"] = reasoning_obj
        # Sticky routing so a multi-turn tool-calling conversation keeps
        # landing on the same upstream provider — see module docstring and
        # https://openrouter.ai/docs/features/prompt-caching
        if self.session_id:
            extra_body["session_id"] = self.session_id
        if extra_body:
            api_params["extra_body"] = extra_body

        logger.info(
            "OpenRouter API call: model=%s, messages=%d, tools=%d, system_prompt_length=%d, "
            "max_tokens=%s, reasoning_effort=%s, session_id=%s",
            self.model,
            len(api_messages),
            len(tools_to_use) if tools_to_use else 0,
            len(system_message),
            self.max_tokens,
            self.reasoning_effort,
            self.session_id,
        )

        # Max reasoning / long tool loops can exceed 5 minutes.
        timeout_seconds = 600 if self._is_reasoning_model else 300
        loop = asyncio.get_event_loop()
        last_exception: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                logger.debug(
                    "Calling OpenRouter chat.completions (attempt %d/%d, model=%s)",
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

                text_content, tool_calls, reasoning_content, reasoning_details = (
                    self._parse_response(response)
                )

                # Prompt-caching visibility: OpenRouter mirrors OpenAI's
                # usage.prompt_tokens_details.cached_tokens shape (tokens of
                # *this* call's prompt that were served from cache — cheaper,
                # per https://openrouter.ai/docs/features/prompt-caching).
                # Surface it on `wrapped` so MCPClient can roll it into
                # result.json the same way it already does for Gemini's
                # cached_content_token_count.
                usage = getattr(response, "usage", None)
                cached_tokens = 0
                reasoning_tokens_used = 0
                if usage is not None:
                    prompt_details = getattr(usage, "prompt_tokens_details", None)
                    if prompt_details is not None:
                        cached_tokens = getattr(prompt_details, "cached_tokens", 0) or 0
                    completion_details = getattr(usage, "completion_tokens_details", None)
                    if completion_details is not None:
                        reasoning_tokens_used = (
                            getattr(completion_details, "reasoning_tokens", 0) or 0
                        )
                    logger.info(
                        "OpenRouter usage: prompt_tokens=%s completion_tokens=%s "
                        "cached_tokens=%s reasoning_tokens=%s cost=%s",
                        getattr(usage, "prompt_tokens", None),
                        getattr(usage, "completion_tokens", None),
                        cached_tokens,
                        reasoning_tokens_used,
                        getattr(usage, "cost", None),
                    )

                # Attach reasoning_content/reasoning_details so MCPClient can preserve
                # them on the assistant message (mirrors Claude thinking_blocks / Azure
                # reasoning_content). reasoning_details is the full structured block
                # OpenRouter recommends for exact reasoning-state continuity; we send
                # it back verbatim on the next turn's assistant message in
                # _prepare_messages, falling back to the plaintext reasoning_content
                # alias when a model only returns that.
                wrapped = SimpleNamespace(
                    usage=usage,
                    cached_tokens=cached_tokens,
                    reasoning_content=reasoning_content,
                    reasoning_details=reasoning_details,
                    raw=response,
                )

                if reasoning_content:
                    logger.info(
                        "OpenRouter reasoning_content length: %d chars",
                        len(reasoning_content),
                    )
                if reasoning_details:
                    logger.info(
                        "OpenRouter reasoning_details: %d block(s)",
                        len(reasoning_details),
                    )
                if tool_calls:
                    logger.info(
                        "OpenRouter returned %d tool call(s): %s",
                        len(tool_calls),
                        [tc["function"]["name"] for tc in tool_calls],
                    )

                return wrapped, text_content, tool_calls, reasoning_content

            except asyncio.TimeoutError as e:
                last_exception = e
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(
                    "OpenRouter chat.completions timeout (attempt %d/%d, timeout=%ds); "
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
                    f"OpenRouter chat.completions call timed out after {timeout_seconds}s. "
                    f"Failed after {max_retries} attempts."
                )
                logger.error(error_msg)
                raise TimeoutError(error_msg) from e

            except BadRequestError as e:
                # Same detection strategy as OpenAIProvider/AzureChatCompletionsProvider:
                # prefer the structured error body, fall back to substring matching.
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
                    wait_time = rate_limit_backoff_secs(
                        error_msg, attempt=attempt, retry_delay=retry_delay,
                    )
                    logger.warning(
                        "OpenRouter rate limit (attempt %d/%d); waiting %.1fs",
                        attempt + 1, max_retries, wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    continue

                self._log_error(e, error_msg, api_messages, tools_to_use)

                if "404" in error_msg or "not found" in error_msg.lower():
                    raise ValueError(
                        f"Model '{self.model}' not found on OpenRouter. "
                        f"Check the model slug (e.g. 'moonshotai/kimi-k3'). "
                        f"Original error: {error_msg}"
                    ) from e
                raise

            except Exception as e:
                last_exception = e
                error_msg = str(e)

                if is_rate_limit_error(e) and attempt < max_retries - 1:
                    wait_time = rate_limit_backoff_secs(
                        error_msg, attempt=attempt, retry_delay=retry_delay,
                    )
                    logger.warning(
                        "OpenRouter rate limit (attempt %d/%d); waiting %.1fs",
                        attempt + 1,
                        max_retries,
                        wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    continue

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
                        f"Model '{self.model}' not found on OpenRouter. "
                        f"Check the model slug (e.g. 'moonshotai/kimi-k3'). "
                        f"Original error: {error_msg}"
                    ) from e

                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("OpenRouter chat.completions call failed without exception")

    def _log_error(
        self,
        e: Exception,
        error_msg: str,
        api_messages: List[Dict[str, Any]],
        tools_to_use: Optional[List[Dict[str, Any]]],
    ) -> None:
        """Log a detailed error block, mirroring OpenAIProvider/Azure diagnostics."""
        error_details = [
            "=" * 80,
            "OPENROUTER CHAT COMPLETIONS ERROR",
            "=" * 80,
            f"Exception Type: {type(e).__name__}",
            f"Exception Message: {error_msg}",
            "",
            "API Call Details:",
            f"  Model: {self.model}",
            f"  Endpoint: {self.base_url}",
            f"  Messages: {len(api_messages)}",
            f"  Tools: {len(tools_to_use) if tools_to_use else 0}",
            f"  Max Tokens (completion): {self.max_tokens}",
            f"  Reasoning Effort: {self.reasoning_effort}",
            f"  Session ID: {self.session_id}",
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
