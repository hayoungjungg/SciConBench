"""
Modular precision calculation functions for factual precision analysis.

Each function takes judgment counts (supported, contradicted, not_supported, total)
and returns a precision score in [0.0, 1.0].

Default metric: 'main'  →  (S / T) * (1 - C / T)
"""

from typing import Any, Dict


def precision_no_contradiction(supported: int, contradicted: int, _not_supported: int, total: int) -> float:
    """
    No-contradiction score: (T - C) / T

    Measures the proportion of facts that are NOT contradicted.
    This metric focuses on avoiding contradictions rather than requiring support.
    
    Args:
        supported: Number of facts that are supported
        contradicted: Number of facts that are contradicted
        _not_supported: Number of facts that are not supported
        total: Total number of facts
        
    Returns:
        Precision score between 0.0 and 1.0
    """
    if total == 0:
        return 0.0
    return (total - contradicted) / total


def precision_entailment(supported: int, contradicted: int, _not_supported: int, total: int) -> float:
    """
    Support-based score: S / T
    
    Measures the proportion of facts that are supported by the ground truth.
    This is the standard precision metric.
    
    Args:
        supported: Number of facts that are supported
        contradicted: Number of facts that are contradicted
        _not_supported: Number of facts that are not supported
        total: Total number of facts
        
    Returns:
        Precision score between 0.0 and 1.0
    """
    if total == 0:
        return 0.0
    return supported / total


def precision_main(supported: int, contradicted: int, _not_supported: int, total: int) -> float:
    """
    Main factual precision score: (S / T) * (1 - C / T). Main metric used in the paper.
    
    Multiplies the support rate by the no-contradiction rate.
    This metric requires both high support AND low contradiction.
    It penalizes contradictions more heavily than the average metric.
    
    Args:
        supported: Number of facts that are supported
        contradicted: Number of facts that are contradicted
        _not_supported: Number of facts that are not supported
        total: Total number of facts
        
    Returns:
        Precision score between 0.0 and 1.0
    """
    if total == 0:
        return 0.0
    return (supported / total) * ((total - contradicted) / total)


PRECISION_FUNCTIONS: Dict[str, Any] = {
    "main": precision_main,
    "entailment": precision_entailment,
    "no_contradiction": precision_no_contradiction,
}


def get_precision_function(name: str):
    """
    Return a precision function by name.

    Args:
        name: One of ``'main'`` (default), ``'entailment'``, ``'no_contradiction'``.

    Raises:
        ValueError: If the name is not in the registry.
    """
    if name not in PRECISION_FUNCTIONS:
        available = ", ".join(PRECISION_FUNCTIONS.keys())
        raise ValueError(f"Unknown precision function '{name}'. Available: {available}")
    return PRECISION_FUNCTIONS[name]


def compute_f1(precision: float | None, recall: float | None) -> float | None:
    """
    Harmonic mean of precision and recall.

    Returns ``None`` if either input is ``None`` or both are zero.
    """
    if precision is None or recall is None:
        return None
    denom = precision + recall
    if denom == 0:
        return None
    return 2 * precision * recall / denom


def compute_all_precision_metrics(supported: int, contradicted: int, _not_supported: int, total: int) -> Dict[str, float]:
    """
    Compute all registered precision metrics and return them as a dict.
    """
    results = {}
    for name, func in PRECISION_FUNCTIONS.items():
        try:
            results[name] = func(supported, contradicted, _not_supported, total)
        except Exception:
            results[name] = None
    return results
