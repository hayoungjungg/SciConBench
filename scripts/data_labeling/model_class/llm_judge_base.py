"""
LLM Judge interface for factual precision and recall evaluation.
Any model backend can implement LLMJudgeBase so analyzers work with any LLM.
"""
from abc import ABC, abstractmethod
import asyncio
import re
import os
import importlib
import warnings
from typing import Any, Dict, List, Optional, Tuple

# Sentinel used to distinguish "caller never passed this arg" from "caller explicitly passed None".
_UNSET: Any = object()


class LLMJudgeBase(ABC):
    """Abstract base for an LLM judge. Implement complete() for your backend."""

    @abstractmethod
    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict[str, int]]:
        """Run chat completion. Returns (content, token_usage with input/output/total_tokens)."""
        pass


class OpenAICompatibleJudge(LLMJudgeBase):
    """
    Judge that uses OpenAI or Azure OpenAI chat completions.
    Works with any model name; uses env OPENAI_API_KEY or AZURE_OPENAI_KEY + OPENAI_BASE_URL.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_temperature: float = 0.0,
        default_max_tokens: int = 1024,
    ):
        self.model = model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        base_url = base_url or os.getenv("OPENAI_BASE_URL")
        api_key = api_key or os.getenv("AZURE_OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAICompatibleJudge: no API key found. "
                "Set OPENAI_API_KEY for standard OpenAI, or AZURE_OPENAI_KEY + OPENAI_BASE_URL for Azure."
            )
        if base_url and "azure" in base_url.lower():
            from openai import AzureOpenAI  # type: ignore[import-not-found]
            api_version = os.getenv("OPENAI_API_VERSION", "2025-04-01-preview")
            self._client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=base_url,
                api_key=api_key,
            )
        else:
            from openai import OpenAI  # type: ignore[import-not-found]
            self._client = OpenAI(api_key=api_key, base_url=base_url or "https://api.openai.com/v1")

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict[str, int]]:
        effective_max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.default_temperature,
                max_tokens=effective_max_tokens,
            )
        except Exception as e:
            # Some newer chat models reject max_tokens and require max_completion_tokens.
            if "max_completion_tokens" in str(e) and "max_tokens" in str(e):
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature if temperature is not None else self.default_temperature,
                    max_completion_tokens=effective_max_tokens,
                )
            else:
                raise
        content = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None) or {}
        token_usage = {
            "input_tokens": getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }
        if not token_usage["total_tokens"]:
            token_usage["total_tokens"] = token_usage["input_tokens"] + token_usage["output_tokens"]
        return content, token_usage


class _BaseModelJudge(LLMJudgeBase):
    """
    Shared utilities for model-specific judges.
    """

    @staticmethod
    def _split_system_and_user_messages(messages: List[Dict[str, str]]) -> tuple[str, str]:
        system_parts: List[str] = []
        user_parts: List[str] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content") or ""
            if role == "system":
                if content.strip():
                    system_parts.append(content.strip())
            elif role == "user":
                if content.strip():
                    user_parts.append(content.strip())
        system_prompt = "\n\n".join(system_parts).strip()
        user_prompt = "\n\n".join(user_parts).strip()
        return system_prompt, user_prompt

    @staticmethod
    def _extract_token_usage(response: Any) -> Dict[str, int]:
        """
        Best-effort extraction of token usage from provider response objects.
        Falls back to zeros when usage is unavailable.
        """
        zeros = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        if response is None:
            return zeros

        usage = getattr(response, "usage", None) or {}

        def _get(key: str) -> Optional[int]:
            val = None
            if isinstance(usage, dict):
                val = usage.get(key)
            else:
                val = getattr(usage, key, None)
            try:
                return int(val) if val is not None else None
            except (TypeError, ValueError):
                return None

        input_tokens = _get("input_tokens") or _get("prompt_tokens")
        output_tokens = _get("output_tokens") or _get("completion_tokens")
        total_tokens = _get("total_tokens")
        if total_tokens is None:
            total_tokens = (input_tokens or 0) + (output_tokens or 0)

        return {
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "total_tokens": int(total_tokens or 0),
        }

    @staticmethod
    def _run_async_safely(coro: Any) -> Any:
        try:
            asyncio.get_running_loop()
            # Already inside a running event loop (e.g. called from asyncio.run(main())).
            # Run the coroutine in a fresh thread so it gets its own event loop.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        except RuntimeError:
            # No running event loop — safe to call asyncio.run() directly.
            return asyncio.run(coro)


class OpenAIResponsesJudge(_BaseModelJudge):
    """
    OpenAI GPT-5 "responses"-style judge using your MCP OpenAIProvider.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str],
        base_url: Optional[str],
        openai_verbosity: Optional[str] = None,
        openai_reasoning_effort: Any = _UNSET,
        openai_reasoning_summary: Optional[str] = None,
    ):
        """
        ``openai_reasoning_effort``:
          - ``_UNSET`` (default) — auto-detect from model name (sets ``"high"`` for GPT-5).
          - ``None`` — explicitly **disable** reasoning (passes ``None`` to the provider).
          - ``"low"`` / ``"medium"`` / ``"high"`` — use that effort level.
        """
        effective_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
        if not effective_key:
            warnings.warn(
                "OpenAIResponsesJudge: no API key found. "
                "Set OPENAI_API_KEY for standard OpenAI, or AZURE_OPENAI_KEY + OPENAI_BASE_URL for Azure.",
                UserWarning,
                stacklevel=2,
            )

        # Fall back to OPENAI_BASE_URL env var so Azure setups work without passing base_url explicitly.
        effective_base_url = base_url or os.getenv("OPENAI_BASE_URL")

        openai_provider_mod = importlib.import_module("sciconharness.mcp_client.llm_providers.openai_provider")
        OpenAIProvider = getattr(openai_provider_mod, "OpenAIProvider")

        m = (model or "").lower()
        if openai_reasoning_effort is _UNSET:
            # Auto-detect: pick effort and verbosity based on model family.
            if "gpt-5.4" in m:
                verbosity = openai_verbosity or "high"
                reasoning_effort = "high"
            else:
                verbosity = openai_verbosity or "medium"
                reasoning_effort = "high"
        else:
            # Caller explicitly set reasoning_effort (None = disabled, string = use it).
            reasoning_effort = openai_reasoning_effort
            verbosity = openai_verbosity or "medium"

        self._provider = OpenAIProvider(
            model=model,
            api_key=api_key,
            base_url=effective_base_url,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            reasoning_summary=openai_reasoning_summary or "auto",
        )

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict[str, int]]:
        # In your OpenAIProvider implementation, system messages aren't forwarded as separate roles.
        # So we fold system+user into a single user content message.
        system_prompt, user_prompt = self._split_system_and_user_messages(messages)
        combined_user_content = "\n\n".join([p for p in [system_prompt, user_prompt] if p]).strip()
        provider_messages = [{"role": "user", "content": combined_user_content}]

        async def _call() -> Tuple[str, Dict[str, int]]:
            response, text_content, _tool_calls, _reasoning_summary = await self._provider.call_llm(
                messages=provider_messages,
                tools=None,
            )
            content = (text_content or "").strip()
            return content, self._extract_token_usage(response)

        return self._run_async_safely(_call())


class ClaudeJudge(_BaseModelJudge):
    """
    Anthropic Claude judge using your MCP ClaudeProvider.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str],
        claude_thinking_budget_tokens: Optional[int] = None,
        claude_thinking_mode: Optional[str] = None,
        claude_adaptive_effort: Optional[str] = None,
        default_max_tokens: int = 1024,
    ):
        claude_provider_mod = importlib.import_module("sciconharness.mcp_client.llm_providers.claude_provider")
        ClaudeProvider = getattr(claude_provider_mod, "ClaudeProvider")

        self._thinking_mode = (claude_thinking_mode or "enabled").lower()

        # Mirror `sciconharness.utils.query_utils.create_provider()` behavior:
        # - If AZURE_ANTHROPIC_API_KEY (or Foundry base URL) is present, default to Foundry.
        # - Otherwise fall back to standard Anthropic API.
        foundry_base_url = os.getenv("AZURE_ANTHROPIC_BASE_URL") or os.getenv("ANTHROPIC_FOUNDRY_BASE_URL")
        foundry_resource = os.getenv("AZURE_ANTHROPIC_RESOURCE_NAME") or os.getenv("ANTHROPIC_FOUNDRY_RESOURCE")
        has_foundry_key = bool(os.getenv("AZURE_ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_FOUNDRY_API_KEY"))
        use_foundry = bool(foundry_base_url or foundry_resource or has_foundry_key)

        if use_foundry:
            effective_key = api_key or os.getenv("AZURE_ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_FOUNDRY_API_KEY")
            if not effective_key:
                warnings.warn(
                    "ClaudeJudge (Azure/Foundry mode): no API key found. "
                    "Set AZURE_ANTHROPIC_API_KEY or ANTHROPIC_FOUNDRY_API_KEY.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            effective_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not effective_key:
                warnings.warn(
                    "ClaudeJudge: no API key found. Set ANTHROPIC_API_KEY.",
                    UserWarning,
                    stacklevel=2,
                )

        provider_kwargs: Dict[str, Any] = {
            "model": model,
            # Temperature is set per-call only when thinking_mode == disabled.
            "temperature": None,
            "max_tokens": default_max_tokens,
            "thinking_budget_tokens": claude_thinking_budget_tokens if claude_thinking_budget_tokens is not None else 4096,
            "thinking_mode": self._thinking_mode,
            "adaptive_effort": claude_adaptive_effort or "high",
        }

        if use_foundry:
            provider_kwargs["use_foundry"] = True
            provider_kwargs["api_key"] = api_key or os.getenv("AZURE_ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_FOUNDRY_API_KEY")
            # Prefer explicit base URL from .env (AZURE_ANTHROPIC_BASE_URL) so the exact endpoint is used;
            # otherwise use resource name so the SDK can construct the URL.
            if foundry_base_url:
                provider_kwargs["base_url"] = foundry_base_url
            elif foundry_resource:
                provider_kwargs["resource"] = foundry_resource
        else:
            provider_kwargs["api_key"] = api_key or os.getenv("ANTHROPIC_API_KEY")

        self._provider = ClaudeProvider(**provider_kwargs)

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict[str, int]]:
        # Preserve earlier behavior: temperature only applies when thinking is disabled.
        self._provider.temperature = float(temperature) if (self._thinking_mode == "disabled" and temperature is not None) else None

        async def _call() -> Tuple[str, Dict[str, int]]:
            response, text_content, _tool_calls, _reasoning_summary = await self._provider.call_llm(
                messages=messages,
                tools=None,
                max_tokens=max_tokens,
            )
            content = (text_content or "").strip()
            return content, self._extract_token_usage(response)

        return self._run_async_safely(_call())


class GeminiJudge(_BaseModelJudge):
    """
    Google Gemini judge using your MCP GeminiProvider.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str],
        gemini_temperature: Optional[float] = None,
        gemini_thinking_level: Optional[str] = None,
        gemini_thinking_budget_tokens: Optional[int] = None,
        default_max_tokens: int = 1024,
    ):
        effective_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not effective_key:
            warnings.warn(
                "GeminiJudge: no API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY.",
                UserWarning,
                stacklevel=2,
            )

        gemini_provider_mod = importlib.import_module("sciconharness.mcp_client.llm_providers.gemini_provider")
        GeminiProvider = getattr(gemini_provider_mod, "GeminiProvider")

        self._gemini_temperature_override = gemini_temperature
        self._provider = GeminiProvider(
            model=model,
            api_key=api_key,
            temperature=gemini_temperature,
            max_output_tokens=default_max_tokens,
            thinking_level=gemini_thinking_level,
            thinking_budget_tokens=gemini_thinking_budget_tokens,
        )

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict[str, int]]:
        # GeminiProvider requires max_output_tokens at construction time, but we can safely mutate the attribute.
        if max_tokens is not None:
            self._provider.max_output_tokens = int(max_tokens)
        effective_temp = self._gemini_temperature_override if self._gemini_temperature_override is not None else temperature
        self._provider.temperature = float(effective_temp) if effective_temp is not None else None

        async def _call() -> Tuple[str, Dict[str, int]]:
            _response, text_content, _tool_calls, _reasoning_summary = await self._provider.call_llm(
                messages=messages,
                tools=None,
            )
            content = (text_content or "").strip()
            # Preserve earlier behavior: GeminiJudge returns zeros for token usage.
            return content, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        return self._run_async_safely(_call())


class ModelJudgeFactory:
    """
    Instantiate the correct model-specific judge for `data_labeling`.
    """

    @staticmethod
    def _normalize_model_name(model: str) -> str:
        m = (model or "").strip()
        if not m:
            return m
        ml = m.lower()

        # "Haiku 4.5s" / "Sonnet 4.5s" -> canonical Claude ids.
        if re.search(r"\bhaiku\b.*\b4[.\-]5s?\b", ml):
            return "claude-haiku-4-5"
        if re.search(r"\bsonnet\b.*\b4[.\-]5s?\b", ml):
            return "claude-sonnet-4-5"

        ml = ml.replace("claude-haiku-4-5s", "claude-haiku-4-5")
        ml = ml.replace("claude-sonnet-4-5s", "claude-sonnet-4-5")
        return ml

    @staticmethod
    def _detect_backend_kind(model: str) -> str:
        m = (model or "").lower()
        if m.startswith("gemini-") or "gemini" in m:
            return "gemini"
        if m.startswith("claude-") or "sonnet" in m or "haiku" in m or "claude" in m:
            return "claude"
        if "chat" in m or "o1" in m or "o3" in m or m.startswith("gpt-4"):
            return "openai_chat"
        if "gpt-5." in m or "gpt-5" in m:
            return "openai_responses"
        return "openai_chat"

    @classmethod
    def create(
        cls,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        openai_verbosity: Optional[str] = None,
        openai_reasoning_effort: Any = _UNSET,
        openai_reasoning_summary: Optional[str] = None,
        claude_thinking_budget_tokens: Optional[int] = None,
        claude_thinking_mode: Optional[str] = None,
        claude_adaptive_effort: Optional[str] = None,
        gemini_temperature: Optional[float] = None,
        gemini_thinking_level: Optional[str] = None,
        gemini_thinking_budget_tokens: Optional[int] = None,
        default_temperature: float = 0.0,
        default_max_tokens: int = 1024,
    ) -> LLMJudgeBase:
        normalized_model = cls._normalize_model_name(model)
        backend_kind = cls._detect_backend_kind(normalized_model)

        if backend_kind == "openai_chat":
            return OpenAICompatibleJudge(
                model=normalized_model,
                api_key=api_key,
                base_url=base_url,
                default_temperature=default_temperature,
                default_max_tokens=default_max_tokens,
            )

        if backend_kind == "openai_responses":
            return OpenAIResponsesJudge(
                model=normalized_model,
                api_key=api_key,
                base_url=base_url,
                openai_verbosity=openai_verbosity,
                openai_reasoning_effort=openai_reasoning_effort,
                openai_reasoning_summary=openai_reasoning_summary,
            )

        if backend_kind == "claude":
            return ClaudeJudge(
                model=normalized_model,
                api_key=api_key,
                claude_thinking_budget_tokens=claude_thinking_budget_tokens,
                claude_thinking_mode=claude_thinking_mode,
                claude_adaptive_effort=claude_adaptive_effort,
                default_max_tokens=default_max_tokens,
            )

        # gemini
        return GeminiJudge(
            model=normalized_model,
            api_key=api_key,
            gemini_temperature=gemini_temperature,
            gemini_thinking_level=gemini_thinking_level,
            gemini_thinking_budget_tokens=gemini_thinking_budget_tokens,
            default_max_tokens=default_max_tokens,
        )


class MultiModelJudge(_BaseModelJudge):
    """
    Backward-compatible wrapper around `ModelJudgeFactory`.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        openai_verbosity: Optional[str] = None,
        openai_reasoning_effort: Any = _UNSET,
        openai_reasoning_summary: Optional[str] = None,
        claude_thinking_budget_tokens: Optional[int] = None,
        claude_thinking_mode: Optional[str] = None,
        claude_adaptive_effort: Optional[str] = None,
        gemini_temperature: Optional[float] = None,
        gemini_thinking_level: Optional[str] = None,
        gemini_thinking_budget_tokens: Optional[int] = None,
        default_temperature: float = 0.0,
        default_max_tokens: int = 1024,
    ):
        self._judge = ModelJudgeFactory.create(
            model=model,
            api_key=api_key,
            base_url=base_url,
            openai_verbosity=openai_verbosity,
            openai_reasoning_effort=openai_reasoning_effort,
            openai_reasoning_summary=openai_reasoning_summary,
            claude_thinking_budget_tokens=claude_thinking_budget_tokens,
            claude_thinking_mode=claude_thinking_mode,
            claude_adaptive_effort=claude_adaptive_effort,
            gemini_temperature=gemini_temperature,
            gemini_thinking_level=gemini_thinking_level,
            gemini_thinking_budget_tokens=gemini_thinking_budget_tokens,
            default_temperature=default_temperature,
            default_max_tokens=default_max_tokens,
        )

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict[str, int]]:
        return self._judge.complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
