"""
atomic_fact_generation
======================

Pipeline for decomposing scientific text into independent, verifiable atomic facts.

Quick start::

    from atomic_fact_generation import AtomicFactGenerator

    generator = AtomicFactGenerator()
    facts, para_breaks, metadata = generator.run(
        generation="Some scientific text ...",
        question="What are the effects of X on Y?",
    )

See ``README.md`` for full documentation.
"""

from .generate_atomic_facts import AtomicFactGenerator, format_output_for_json
from .config.model_config import (
    ModelConfig,
    UnifiedLLMClient,
    DEFAULT_CONFIG_PATH,
    COMPONENTS,
    create_default_configs,
    load_configs_from_yaml,
    load_configs_from_json,
)

__all__ = [
    # Core generator
    "AtomicFactGenerator",
    "format_output_for_json",
    # Model configuration
    "ModelConfig",
    "UnifiedLLMClient",
    "DEFAULT_CONFIG_PATH",
    "COMPONENTS",
    "create_default_configs",
    "load_configs_from_yaml",
    "load_configs_from_json",
]
