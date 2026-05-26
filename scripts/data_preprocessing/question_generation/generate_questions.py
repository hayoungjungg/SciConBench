"""
Question Generator

Converts research objectives into answerable research questions.
Supports Azure OpenAI and the standard OpenAI API via a ModelConfig.
"""

import os
import sys
import time
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv

# Fallback sys.path for running without pip install -e .
_scripts_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _scripts_root not in sys.path:
    sys.path.insert(0, _scripts_root)

from data_preprocessing.question_generation.config.model_config import (
    ModelConfig,
    UnifiedLLMClient,
)
from data_preprocessing.question_generation.utils.helper_questions import (
    parse_question_response,
    validate_question,
)
from data_preprocessing.question_generation.utils.prompts import (
    persona_description,
    zero_shot_prompt,
    few_shot_prompt,
)

# Load environment variables from .env file
load_dotenv()


class QuestionGenerator:
    """
    Generates research questions from research objectives using an Azure OpenAI model.

    Accepts either a :class:`~data_preprocessing.question_generation.config.model_config.ModelConfig`
    for full control, or individual keyword arguments for quick setup.

    Examples::

        # Azure OpenAI (default)
        gen = QuestionGenerator(model="gpt-5-chat", provider="azure")

        # Standard OpenAI API
        gen = QuestionGenerator(model="gpt-4o", provider="openai")

        # OpenAI reasoning model
        from data_preprocessing.question_generation.config.model_config import ModelConfig
        gen = QuestionGenerator(
            model_config=ModelConfig(
                model="gpt-5-mini",
                provider="openai",
                reasoning_effort="low",
            )
        )
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        *,
        model: Optional[str] = None,
        provider: str = "azure",
        include_few_shot: Optional[bool] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
    ):
        """
        Args:
            model_config: Full :class:`ModelConfig` object.  When provided, all
                individual model kwargs (``model``, ``provider``, ``temperature``, etc.)
                are ignored.
            model: Model deployment/API name (default: ``'gpt-5-chat'``).
            provider: ``'azure'`` (default) or ``'openai'``.
            include_few_shot: Use few-shot prompt examples (default: ``False``).
            temperature: Sampling temperature for chat-completions models.
            max_tokens: Response token limit for chat-completions models.
            reasoning_effort: Effort level for responses-API models
                (``'none'``, ``'minimal'``, ``'low'``, ``'medium'``, ``'high'``).
            verbosity: Output verbosity for responses-API models
                (``'low'``, ``'medium'``, ``'high'``).
        """
        if model_config is not None:
            self._model_config = model_config
        else:
            self._model_config = ModelConfig(
                model=model or "gpt-5-chat",
                provider=provider or "azure",
                temperature=temperature if temperature is not None else 0,
                max_tokens=max_tokens if max_tokens is not None else 1024,
                reasoning_effort=reasoning_effort or "low",
                verbosity=verbosity or "low",
                reasoning_summary="auto",
            )

        self.include_few_shot = include_few_shot if include_few_shot is not None else False

        self._client = UnifiedLLMClient(self._model_config)

    def run(self, objective: str, background_context: str) -> Dict[str, Any]:
        """
        Convert a research objective into a single research question.

        Args:
            objective: Research objective to convert.
            background_context: Background context for the objective.

        Returns:
            Dict with keys ``'original_objective'`` and ``'question'``.
        """
        assert isinstance(objective, str), "objective must be a string"
        assert isinstance(background_context, str), "background_context must be a string"

        content = self._generate_content(objective, background_context)
        raw_question = parse_question_response(content)
        question = validate_question(raw_question)

        return {
            "original_objective": objective,
            "question": question,
        }

    def _generate_content(self, objective: str, background_context: str) -> str:
        """Call the API and return the raw response text."""
        prompt_content = (
            few_shot_prompt if self.include_few_shot else zero_shot_prompt
        ).format(objective=objective, background_context=background_context)

        messages = [
            {"role": "system", "content": persona_description},
            {"role": "user", "content": prompt_content},
        ]

        try:
            response = self._client.create_completion(messages)
            return self._client.extract_content(response)
        except Exception as e:
            print(f"Error generating question: {e}")
            return ""

    def batch_run(
        self,
        objectives: List[str],
        background_contexts: Optional[List[str]] = None,
        delay: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """
        Convert multiple objectives to research questions.

        Args:
            objectives: List of research objectives.
            background_contexts: Corresponding background contexts (optional).
            delay: Seconds to wait between API calls to avoid rate limiting.

        Returns:
            List of result dicts (same structure as :meth:`run`).
        """
        results = []
        for i, objective in enumerate(objectives):
            background = (
                background_contexts[i]
                if background_contexts and i < len(background_contexts)
                else ""
            )
            results.append(self.run(objective, background))
            time.sleep(delay)
        return results
