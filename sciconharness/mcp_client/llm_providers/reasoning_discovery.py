"""Cross-provider reasoning-effort discovery, backed by OpenRouter's ``GET /models``.

This module is used by ``OpenRouterProvider`` **and** the native
``ClaudeProvider`` / ``OpenAIProvider`` / ``GeminiProvider`` classes. For the
native providers, no inference traffic ever goes through OpenRouter — we only
use OpenRouter's model catalog as a *reference/discovery* source, because it
aggregates the same per-model ``reasoning`` capability data
(``supported_efforts``, in descending/highest-first order) across virtually
every vendor's models in one place:
https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#discovering-per-model-reasoning-options

Each provider still talks to its own vendor's native API directly and sends
the discovered effort through whatever mechanism that vendor's own API
actually accepts (OpenAI: native ``reasoning.effort`` string; Gemini: native
``thinking_level`` enum; Anthropic: no native effort parameter at all today,
only fixed ``budget_tokens`` — see ``anthropic_effort_ratio`` below, mirroring
OpenRouter's own documented conversion formula for Claude).

Public API:
- ``candidate_openrouter_slugs(vendor_prefixes, model)``: build plausible
  OpenRouter slugs for a native (unprefixed) model name.
- ``discover_reasoning_config(model_slugs)``: cached ``GET /models`` lookup;
  tries each candidate slug in order, returns the first match's raw
  ``reasoning`` dict, or ``None``.
- ``highest_supported_effort(reasoning_cfg)``: ``(effort_or_None, supports_effort)``.
- ``anthropic_effort_ratio(effort)``: OpenRouter's documented
  ``budget_tokens = max_tokens * ratio`` multiplier for a given effort label.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# OpenRouter's own documented effort_ratio table for Anthropic models
# (https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#reasoning-max-tokens-for-anthropic-models):
# budget_tokens = max(min(max_tokens * ratio, 128000), 1024)
_ANTHROPIC_EFFORT_RATIOS: Dict[str, float] = {
    "max": 0.95,
    "xhigh": 0.95,
    "high": 0.8,
    "medium": 0.5,
    "low": 0.2,
    "minimal": 0.1,
    "none": 0.0,
}

# Google's documented effort -> thinkingLevel mapping for Gemini 3 models
# (https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#google-gemini-3-models-with-thinking-levels).
# Gemini has no level above HIGH, so "xhigh"/"max" both map down to "high".
_GEMINI_EFFORT_TO_THINKING_LEVEL: Dict[str, str] = {
    "max": "high",
    "xhigh": "high",
    "high": "high",
    "medium": "medium",
    "low": "low",
    "minimal": "minimal",
}

_MODELS_CACHE: Optional[List[Any]] = None
_MODELS_CACHE_LOCK = threading.Lock()
_REASONING_CONFIG_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}


def _get_discovery_client():
    """Build a throwaway OpenAI-SDK client pointed at OpenRouter, purely for
    ``GET /models`` discovery. Returns ``None`` if no OpenRouter key is
    configured (discovery is best-effort and never required)."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    base_url = os.getenv("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL
    return OpenAI(api_key=api_key, base_url=base_url)


def _list_models(client: Optional[Any] = None) -> List[Any]:
    """Fetch (and cache) OpenRouter's model catalog. ``client`` lets
    ``OpenRouterProvider`` pass its own already-constructed OpenAI-SDK client
    in to avoid building a second one; native providers (Claude/OpenAI/Gemini)
    have no OpenRouter client of their own, so we build a throwaway one from
    ``OPENROUTER_API_KEY`` on demand. The result is cached globally either way
    since the catalog is the same regardless of which client fetched it."""
    global _MODELS_CACHE
    with _MODELS_CACHE_LOCK:
        if _MODELS_CACHE is not None:
            return _MODELS_CACHE
        active_client = client or _get_discovery_client()
        if active_client is None:
            logger.debug(
                "OPENROUTER_API_KEY not set; skipping cross-provider reasoning-effort "
                "discovery (providers will fall back to their own hardcoded defaults)."
            )
            _MODELS_CACHE = []
            return _MODELS_CACHE
        try:
            resp = active_client.models.list()
            _MODELS_CACHE = list(resp.data)
        except Exception as e:
            logger.warning(
                "Could not fetch OpenRouter GET /models for reasoning-effort discovery "
                "(falling back to conservative per-provider defaults): %s",
                e,
            )
            _MODELS_CACHE = []
        return _MODELS_CACHE


def candidate_openrouter_slugs(vendor_prefixes: Iterable[str], model: str) -> List[str]:
    """Build plausible OpenRouter slugs for a native (unprefixed) model name.

    Native provider ``model`` strings (e.g. ``"claude-sonnet-4-5"``,
    ``"gpt-5.1"``, ``"gemini-3-pro-preview"``) don't carry the
    ``vendor/`` prefix OpenRouter uses, and vendors are inconsistent about
    dashes vs. dots in version numbers (``"claude-sonnet-4-5"`` vs.
    OpenRouter's ``"claude-sonnet-4.5"``). Try the model as given plus a
    dash<->dot swap on trailing ``-N-M`` version segments, under each vendor
    prefix, in order.
    """
    variants = [model]
    dash_to_dot = re.sub(r"-(\d+)-(\d+)(?=$|-)", r"-\1.\2", model)
    if dash_to_dot != model and dash_to_dot not in variants:
        variants.append(dash_to_dot)
    dot_to_dash = model.replace(".", "-")
    if dot_to_dash not in variants:
        variants.append(dot_to_dash)

    slugs: List[str] = []
    for prefix in vendor_prefixes:
        for variant in variants:
            slug = f"{prefix}/{variant}"
            if slug not in slugs:
                slugs.append(slug)
    return slugs


def discover_reasoning_config(
    model_slugs: Iterable[str], client: Optional[Any] = None
) -> Optional[Dict[str, Any]]:
    """Try each candidate OpenRouter slug (in order) and return the first
    match's ``reasoning`` capability dict, e.g.::

        {"supported_efforts": ["xhigh", "high"], "default_effort": "high",
         "default_enabled": True, "mandatory": False}

    Returns ``None`` if none of the slugs are listed, none expose a
    ``reasoning`` object, or the lookup failed. Cached per slug-list for the
    process lifetime. ``client``: optional pre-built OpenAI-SDK client
    pointed at OpenRouter to reuse (see ``_list_models``).
    """
    slugs = [s for s in model_slugs if s]
    if not slugs:
        return None
    cache_key = "|".join(slugs)
    if cache_key in _REASONING_CONFIG_CACHE:
        return _REASONING_CONFIG_CACHE[cache_key]

    result: Optional[Dict[str, Any]] = None
    models = _list_models(client)
    if models:
        by_id = {getattr(m, "id", None): m for m in models}
        for slug in slugs:
            m = by_id.get(slug)
            if m is not None:
                cfg = getattr(m, "reasoning", None)
                if isinstance(cfg, dict):
                    result = cfg
                    logger.info("Discovered OpenRouter reasoning config via slug %s: %s", slug, cfg)
                break

    _REASONING_CONFIG_CACHE[cache_key] = result
    return result


def highest_supported_effort(reasoning_cfg: Optional[Dict[str, Any]]) -> Tuple[Optional[str], bool]:
    """Pick the highest reasoning effort a model supports from its discovered config.

    Returns ``(effort_or_None, supports_effort_selection)``. Per OpenRouter's
    docs, ``supported_efforts`` is returned in descending effort order
    (highest first). If the key is *omitted* entirely, the model doesn't
    expose effort selection at all -> ``(None, False)``. If present but
    ``null``/empty, "all gateway effort values are accepted" -> the ceiling
    ``"max"`` applies.
    """
    if not reasoning_cfg or "supported_efforts" not in reasoning_cfg:
        return None, False
    supported = reasoning_cfg["supported_efforts"]
    if not supported:
        return "max", True
    return supported[0], True


def anthropic_effort_ratio(effort: Optional[str]) -> float:
    """OpenRouter's documented ``budget_tokens = max_tokens * ratio`` multiplier
    for a given effort label (falls back to the "medium" ratio for unknown
    labels, matching Anthropic's historical implicit default)."""
    if not effort:
        return _ANTHROPIC_EFFORT_RATIOS["medium"]
    return _ANTHROPIC_EFFORT_RATIOS.get(effort.lower(), _ANTHROPIC_EFFORT_RATIOS["medium"])


def gemini_thinking_level_for_effort(effort: Optional[str]) -> Optional[str]:
    """Map a discovered OpenRouter effort label to Gemini's native
    ``thinking_level`` string (``"minimal"|"low"|"medium"|"high"``), per
    Google/OpenRouter's documented mapping. Gemini has no tier above
    ``"high"``, so ``"max"``/``"xhigh"`` both map down to it."""
    if not effort:
        return None
    return _GEMINI_EFFORT_TO_THINKING_LEVEL.get(effort.lower())
