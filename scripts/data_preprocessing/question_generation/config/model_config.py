"""
Model configuration for question generation.

Two API styles are supported:

1. **Chat Completions API** — gpt-5-chat, gpt-5-chat-latest, or any non-GPT-5 model.
   Uses ``temperature`` and ``max_tokens``.

2. **Responses API** — gpt-5, gpt-5.1, gpt-5-mini, gpt-5-nano, and other GPT-5 reasoning models.
   Uses ``reasoning_effort``, ``verbosity``, and ``reasoning_summary``.
   Temperature and max_tokens are ignored.

Both Azure OpenAI and the standard OpenAI API are supported via the ``provider`` field.

Environment variables required:
    Azure  : AZURE_OPENAI_KEY, OPENAI_BASE_URL, OPENAI_API_VERSION
    OpenAI : OPENAI_API_KEY
"""

import os
from dataclasses import dataclass
from typing import Any, Literal

from openai import AzureOpenAI, OpenAI


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for the question-generation model.

    Attributes:
        model: Model deployment/API name (e.g. ``'gpt-5-chat'``, ``'gpt-5-mini'``).
        provider: ``'azure'`` for Azure OpenAI or ``'openai'`` for OpenAI API.
        temperature: Sampling temperature for chat-completions models (default: 0).
        max_tokens: Maximum response tokens for chat-completions models (default: 1024).
        reasoning_effort: Effort level for responses-API models.
            Valid values: ``'none'``, ``'minimal'``, ``'low'``, ``'medium'``, ``'high'``.
        verbosity: Verbosity for responses-API models: ``'low'``, ``'medium'``, ``'high'``.
        reasoning_summary: Summary mode for responses-API models:
            ``'auto'``, ``'none'``, ``'brief'``, ``'detailed'``.
    """

    model: str
    provider: Literal["azure", "openai"] = "azure"
    temperature: float = 0
    max_tokens: int = 1024
    reasoning_effort: str = "low"
    verbosity: str = "low"
    reasoning_summary: str = "auto"

    def is_responses_api_model(self) -> bool:
        """Return True if this model uses the Responses API (GPT-5 family, not gpt-5-chat)."""
        name = self.model.lower()
        if name in ("gpt-5-chat", "gpt-5-chat-latest"):
            return False
        return name.startswith("gpt-5")


# ---------------------------------------------------------------------------
# UnifiedLLMClient
# ---------------------------------------------------------------------------

class UnifiedLLMClient:
    """Dispatches API calls to Azure OpenAI or OpenAI based on the ModelConfig."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._client = self._build_client(config.provider)

    @staticmethod
    def _build_client(provider: str):
        match provider.lower():
            case "azure":
                return AzureOpenAI(
                    api_version=os.getenv("OPENAI_API_VERSION"),
                    azure_endpoint=os.getenv("OPENAI_BASE_URL"),
                    api_key=os.getenv("AZURE_OPENAI_KEY"),
                )
            case "openai":
                return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            case _:
                raise ValueError(
                    f"Unsupported provider '{provider}'. Use 'azure' or 'openai'."
                )

    def create_completion(self, messages: list) -> Any:
        """Send a chat-style messages list and return the raw API response."""
        cfg = self.config

        if not cfg.is_responses_api_model():
            return self._client.chat.completions.create(
                model=cfg.model,
                messages=messages,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )

        # Responses API — convert system message to instructions
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        user_messages = [
            {"role": "user", "content": m["content"]}
            for m in messages
            if m.get("role") != "system"
        ]

        params: dict = {
            "model": cfg.model,
            "input": user_messages,
            "reasoning": {
                "effort": cfg.reasoning_effort,
                "summary": cfg.reasoning_summary,
            },
            "text": {"verbosity": cfg.verbosity},
        }
        if system_parts:
            params["instructions"] = "\n".join(system_parts)

        return self._client.responses.create(**params)

    def extract_content(self, response: Any) -> str:
        """Return the text content of a response, regardless of API style."""
        if hasattr(response, "output_text"):
            return str(response.output_text) if response.output_text else ""
        return response.choices[0].message.content or ""
