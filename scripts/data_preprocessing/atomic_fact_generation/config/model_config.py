"""
Model configuration for atomic fact generation.

Two API styles are supported depending on the model:

1. **Responses API** (gpt-5, gpt-5.1, gpt-5-mini, gpt-5-nano, etc.)
   Uses reasoning parameters: reasoning_effort, verbosity, reasoning_summary.
   Temperature is not supported.

2. **Chat Completions API** (gpt-5-chat, gpt-5-chat-latest, or any non-GPT-5 model)
   Uses temperature. Reasoning parameters are ignored.

Default configurations are loaded from ``model_config.yaml`` in this directory.
Override per-component by passing a ``model_configs`` dict to ``AtomicFactGenerator``.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH: Path = Path(__file__).parent / "model_config.yaml"

COMPONENTS = [
    "decomposition",
    "decontextualization",
    "incomplete_detection",
    "irrelevant_filtering",
    "redundant_filtering",
]


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    """Configuration for a single pipeline component's model.

    Attributes:
        model: Model deployment name (e.g. ``'gpt-5.1'``, ``'gpt-5-mini'``).
        provider: ``'azure'`` for Azure OpenAI or ``'openai'`` for OpenAI API.
        temperature: Used only for chat-completions models (non-GPT-5 or gpt-5-chat).
        reasoning_effort: Responses-API effort level. Valid values:
            ``'none'``, ``'minimal'``, ``'low'``, ``'medium'``, ``'high'``.
        verbosity: Responses-API verbosity: ``'low'``, ``'medium'``, ``'high'``.
        reasoning_summary: Responses-API summary mode:
            ``'auto'``, ``'none'``, ``'brief'``, ``'detailed'``.
    """

    model: str
    provider: Literal["azure", "openai"]
    temperature: Optional[float] = 0
    reasoning_effort: Optional[str] = "low"
    verbosity: Optional[str] = "low"
    reasoning_summary: Optional[str] = "auto"

    def is_responses_api_model(self) -> bool:
        """Return True if this model uses the Responses API (GPT-5 family, excluding gpt-5-chat)."""
        if self.model.lower() in ("gpt-5-chat", "gpt-5-chat-latest"):
            return False
        return self.model.lower().startswith("gpt-5")


# ---------------------------------------------------------------------------
# UnifiedLLMClient
# ---------------------------------------------------------------------------

class UnifiedLLMClient:
    """Thin wrapper that dispatches to either Azure OpenAI or OpenAI Responses API."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.provider = config.provider
        self.client = self._setup_client(config.provider)

    @staticmethod
    def _setup_client(provider: str):
        match provider.lower():
            case "azure":
                return AzureOpenAI(
                    api_version=os.getenv("OPENAI_API_VERSION"),
                    azure_endpoint=os.getenv("OPENAI_BASE_URL"),
                    api_key=os.getenv("AZURE_OPENAI_KEY"),
                )
            case "openai":
                return OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url="https://api.openai.com/v1",
                )
            case _:
                raise ValueError(
                    f"Unsupported provider '{provider}'. Must be 'azure' or 'openai'."
                )

    def _messages_to_input_list(self, messages: list) -> list:
        """Convert chat-style messages to Responses API input format."""
        input_list = []
        for msg in messages:
            role = "user" if msg.get("role") == "system" else msg.get("role")
            content = msg.get("content", "")
            if role == "assistant" and not content:
                continue
            input_list.append({"role": role, "content": content})
        return input_list

    def create_completion(self, messages: list, instructions: str = None) -> Any:
        """Create a completion using the configured provider and model."""
        if not self.config.is_responses_api_model():
            # Newer Azure/OpenAI models require max_completion_tokens and
            # reject non-default temperature values.  Build params dynamically.
            params: dict = {"model": self.config.model, "messages": messages}
            max_tok = self.config.max_tokens if hasattr(self.config, "max_tokens") else None
            if max_tok:
                params["max_completion_tokens"] = max_tok
            if self.config.temperature != 1:
                try:
                    return self.client.chat.completions.create(**params, temperature=self.config.temperature)
                except Exception as e:
                    if "temperature" not in str(e):
                        raise
            return self.client.chat.completions.create(**params)

        system_messages = [
            msg.get("content", "") for msg in messages if msg.get("role") == "system"
        ]

        api_params = {
            "model": self.config.model,
            "input": self._messages_to_input_list(messages),
            "reasoning": {
                "summary": self.config.reasoning_summary,
                "effort": self.config.reasoning_effort or "low",
            },
            "text": {"verbosity": self.config.verbosity},
        }

        if instructions or system_messages:
            api_params["instructions"] = instructions or "\n".join(system_messages)

        return self.client.responses.create(**api_params)

    def extract_content(self, response: Any) -> str:
        """Extract text content from a response object."""
        if self.provider == "azure" and not hasattr(response, "output_text"):
            return response.choices[0].message.content
        return str(response.output_text) if response.output_text else ""

    def extract_token_usage(self, response: Any) -> Dict[str, int]:
        """Normalize token usage across both API styles.

        Returns a dict with keys:
        ``prompt_tokens``, ``completion_tokens``, ``total_tokens``, ``cached_tokens``.
        """

        def _get(obj, key, default=0):
            return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

        if not hasattr(response, "usage") or not response.usage:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}

        usage = response.usage
        total = _get(usage, "total_tokens")

        if hasattr(usage, "input_tokens") or (isinstance(usage, dict) and "input_tokens" in usage):
            input_tokens = _get(usage, "input_tokens")
            output_tokens = _get(usage, "output_tokens")
            if total < 0:
                total = input_tokens + output_tokens
            cached = 0
            details = _get(usage, "input_tokens_details", None)
            if details is not None:
                cached = _get(details, "cached_tokens", 0)
            return {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": total,
                "cached_tokens": cached,
            }

        prompt = _get(usage, "prompt_tokens")
        completion = _get(usage, "completion_tokens")
        if total < 0:
            total = prompt + completion
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "cached_tokens": 0,
        }

    def normalize_response(self, response: Any) -> Any:
        """Wrap Responses-API objects to expose a ``.choices[0].message.content`` interface."""
        if hasattr(response, "choices") and hasattr(response.choices[0], "message"):
            return response

        class NormalizedResponse:
            def __init__(self, orig):
                self.original_response = orig
                self.choices = [self._Choice(orig)]
                self.usage = getattr(orig, "usage", None)

            class _Choice:
                def __init__(self, orig):
                    self.message = self._Message(orig)

                class _Message:
                    def __init__(self, orig):
                        if hasattr(orig, "output_text"):
                            self.content = str(orig.output_text) if orig.output_text else ""
                        elif hasattr(orig, "choices") and orig.choices:
                            self.content = orig.choices[0].message.content
                        else:
                            self.content = ""

        return NormalizedResponse(response)


# ---------------------------------------------------------------------------
# Config loading helpers
# ---------------------------------------------------------------------------

def load_configs_from_yaml(path: str | Path) -> Dict[str, ModelConfig]:
    """Load per-component ``ModelConfig`` objects from a YAML file.

    The YAML must be a mapping of component names to parameter dicts, e.g.::

        decomposition:
          model: gpt-5.1
          provider: azure
          reasoning_effort: none
          verbosity: low
          reasoning_summary: auto

    Args:
        path: Path to a YAML configuration file.

    Returns:
        Dict mapping component name to :class:`ModelConfig`.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {comp: ModelConfig(**params) for comp, params in raw.items()}


def load_configs_from_json(path: str | Path) -> Dict[str, ModelConfig]:
    """Load per-component ``ModelConfig`` objects from a JSON file.

    The JSON must be a mapping of component names to parameter dicts, e.g.::

        {
          "decomposition": {"model": "gpt-5.1", "provider": "azure", ...},
          ...
        }

    Args:
        path: Path to a JSON configuration file.

    Returns:
        Dict mapping component name to :class:`ModelConfig`.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        comp: ModelConfig(**cfg) if isinstance(cfg, dict) else cfg
        for comp, cfg in raw.items()
    }


def create_default_configs(config_path: str | Path = None) -> Dict[str, ModelConfig]:
    """Return per-component :class:`ModelConfig` objects from a config file.

    Defaults to ``config/model_config.yaml`` if no path is given.  Accepts
    both ``.yaml`` / ``.yml`` and ``.json`` files.

    Args:
        config_path: Optional path to a YAML or JSON config file.

    Returns:
        Dict mapping component name to :class:`ModelConfig`.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if path.suffix.lower() == ".json":
        return load_configs_from_json(path)
    return load_configs_from_yaml(path)
