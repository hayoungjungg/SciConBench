# data_labeling package
from typing import Optional

from .factual_recall_analyzer import FactualRecallAnalyzer
from .factual_precision_analyzer import FactualPrecisionAnalyzer
from .model_class.llm_judge_base import (
    LLMJudgeBase,
    OpenAICompatibleJudge,
    MultiModelJudge,
    ModelJudgeFactory,
    OpenAIResponsesJudge,
    ClaudeJudge,
    GeminiJudge,
)
from .utils.prompts import (
    system_prompt,
    factual_precision_few_shot_prompt,
    factual_precision_zero_shot_prompt,
    factual_recall_zero_shot_prompt,
    factual_recall_few_shot_prompt,
)


def make_precision_judge(
    model: str = "gpt-5.4-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> FactualPrecisionAnalyzer:
    """
    Return a FactualPrecisionAnalyzer with recommended defaults.

    Configuration: gpt-5.4-mini, few-shot prompt, no reasoning, temperature 0.2.

    Args:
        model: OpenAI model name (default: ``"gpt-5.4-mini"``).
        api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.
        base_url: Optional base URL for Azure or compatible endpoints.
                  Falls back to ``OPENAI_BASE_URL`` env var.

    Returns:
        A ready-to-use :class:`FactualPrecisionAnalyzer`.

    Example::

        from data_labeling import make_precision_judge

        analyzer = make_precision_judge()
        result = analyzer.compute_factual_precision(
            llm_atomic_facts=["Aspirin reduces fever."],
            ground_truth_text="Aspirin is an antipyretic...",
        )
        print(result["factual_precision"])
    """
    judge = ModelJudgeFactory.create(
        model=model,
        api_key=api_key,
        base_url=base_url,
        openai_reasoning_effort="none",
    )
    return FactualPrecisionAnalyzer(
        llm_judge=judge,
        precision_prompt_template=factual_precision_few_shot_prompt,
        temperature=0.2,
    )


def make_recall_judge(
    model: str = "gpt-5.4-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> FactualRecallAnalyzer:
    """
    Return a FactualRecallAnalyzer with recommended defaults.

    Configuration: gpt-5.4-mini, zero-shot prompt, no reasoning, temperature 1.0.

    Args:
        model: OpenAI model name (default: ``"gpt-5.4-mini"``).
        api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.
        base_url: Optional base URL for Azure or compatible endpoints.
                  Falls back to ``OPENAI_BASE_URL`` env var.

    Returns:
        A ready-to-use :class:`FactualRecallAnalyzer`.

    Example::

        from data_labeling import make_recall_judge

        analyzer = make_recall_judge()
        result = analyzer.compute_factual_recall(
            llm_response_text="The study found aspirin reduces fever...",
            article_atomic_facts=["Aspirin reduces fever.", "Pain relief was significant."],
        )
        print(result["factual_recall"])
    """
    judge = ModelJudgeFactory.create(
        model=model,
        api_key=api_key,
        base_url=base_url,
        openai_reasoning_effort="none",
    )
    return FactualRecallAnalyzer(
        llm_judge=judge,
        recall_prompt_template=factual_recall_zero_shot_prompt,
        temperature=1.0,
    )


__all__ = [
    # Core analyzers
    "FactualRecallAnalyzer",
    "FactualPrecisionAnalyzer",
    # Convenience factories (recommended entry points)
    "make_precision_judge",
    "make_recall_judge",
    # Judge backends
    "LLMJudgeBase",
    "OpenAICompatibleJudge",
    "MultiModelJudge",
    "ModelJudgeFactory",
    "OpenAIResponsesJudge",
    "ClaudeJudge",
    "GeminiJudge",
    # Prompts
    "system_prompt",
    "factual_precision_few_shot_prompt",
    "factual_precision_zero_shot_prompt",
    "factual_recall_zero_shot_prompt",
    "factual_recall_few_shot_prompt",
]
