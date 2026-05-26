#!/usr/bin/env python3
"""
Fact Coverage Analyzer

A tool for analyzing the coverage of atomic facts between LLM responses and article conclusions.
Uses LLM to determine whether scientific conclusion facts are supported by LLM-generated facts.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

from .model_class.llm_judge_base import LLMJudgeBase
from .utils.prompts import (
    factual_recall_zero_shot_prompt,
    system_prompt as default_system_prompt,
)


def _strip_markdown_json_fence(text: str) -> str:
    """Remove optional ``` / ```json fences from model output."""
    s = text.strip()
    if not s.startswith("```"):
        return s
    lines = s.split("\n")
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_recall_json_object(text: str) -> Optional[dict]:
    """
    Parse the first JSON object from model output.

    Handles:
    - Trailing prose after a valid object (JSONDecodeError: Extra data)
    - Optional ```json ... ``` fences
    - A leading preamble before the first '{'
    - Top-level JSON array containing one object (uses first dict with LABEL)
    """
    s = _strip_markdown_json_fence((text or "").strip())
    if not s:
        return None
    dec = json.JSONDecoder()

    def _coerce(val: Any) -> Optional[dict]:
        if isinstance(val, dict):
            return val
        if isinstance(val, list):
            for item in val:
                if isinstance(item, dict) and "LABEL" in item:
                    return item
        return None

    # Try decode from start, then from each '{' (preamble / multiple objects).
    starts = [0]
    starts.extend(i for i, c in enumerate(s) if c == "{")
    seen: set[int] = set()
    for start in starts:
        if start in seen:
            continue
        seen.add(start)
        try:
            val, _end = dec.raw_decode(s, start)
        except json.JSONDecodeError:
            continue
        got = _coerce(val)
        if got is not None:
            return got
    return None


class FactualRecallAnalyzer:
    """
    Analyzer for factual recall (support of conclusion facts by an LLM response).
    """

    def __init__(
        self,
        llm_judge: LLMJudgeBase,
        system_prompt: str = default_system_prompt,
        recall_prompt_template: str = factual_recall_zero_shot_prompt,
        temperature: float = 1.0,
        max_tokens: int = 1024,
    ):
        self._llm_judge = llm_judge
        self._system_prompt = system_prompt
        self._recall_prompt_template = recall_prompt_template
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)

    @staticmethod
    def _merge_token_usage(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
        return {
            "input_tokens": int(a.get("input_tokens", 0) or 0) + int(b.get("input_tokens", 0) or 0),
            "output_tokens": int(a.get("output_tokens", 0) or 0) + int(b.get("output_tokens", 0) or 0),
            "total_tokens": int(a.get("total_tokens", 0) or 0) + int(b.get("total_tokens", 0) or 0),
        }

    def check_fact_support(self, llm_response_text: str, article_fact: str) -> Dict[str, Any]:
        """
        Use LLM to check if a scientific conclusion fact is supported by LLM response text.
        
        Args:
            llm_response_text (str): Full text of LLM-generated response
            article_fact (str): Single atomic fact from scientific conclusions
            
        Returns:
            Dict[str, Any]: Support judgment and reasoning
        """
        if not llm_response_text or not llm_response_text.strip():
            return {
                'support': False,
                'justification': 'No LLM response text provided',
                'token_usage': {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
            }
        
        prompt = self._recall_prompt_template.format(
            article_facts_text=article_fact,
            llm_response_text=llm_response_text
        )
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt}
        ]
        try:
            response_content, usage = self._llm_judge.complete(
                messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            token_usage = {
                'input_tokens': usage.get('input_tokens', 0),
                'output_tokens': usage.get('output_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0) or usage.get('input_tokens', 0) + usage.get('output_tokens', 0),
            }

            if not response_content:
                return {
                    'support': False,
                    'justification': 'Empty response from LLM',
                    'token_usage': token_usage
                }

            result = _parse_recall_json_object(response_content)
            repair_attempts = int(os.getenv("DATA_LABELING_RECALL_JSON_REPAIR_MAX", "1") or "1")
            repair_attempts = max(0, min(repair_attempts, 3))

            if result is None and repair_attempts > 0:
                repair_user = (
                    "Your previous answer could not be parsed as a single JSON object. "
                    "Reply with ONLY one JSON object and no other text (no markdown fences). "
                    "Required keys: LABEL (string, exactly SUPPORTED or NOT SUPPORTED), "
                    "EXCERPTS (JSON array of strings), JUSTIFICATION (string)."
                )
                retry_messages = messages + [
                    {"role": "assistant", "content": response_content},
                    {"role": "user", "content": repair_user},
                ]
                for _ in range(repair_attempts):
                    retry_content, retry_usage = self._llm_judge.complete(
                        retry_messages,
                        temperature=self._temperature,
                        max_tokens=self._max_tokens,
                    )
                    token_usage = self._merge_token_usage(token_usage, {
                        "input_tokens": retry_usage.get("input_tokens", 0),
                        "output_tokens": retry_usage.get("output_tokens", 0),
                        "total_tokens": retry_usage.get("total_tokens", 0)
                        or retry_usage.get("input_tokens", 0) + retry_usage.get("output_tokens", 0),
                    })
                    result = _parse_recall_json_object(retry_content or "")
                    if result is not None:
                        break
                    retry_messages = retry_messages + [
                        {"role": "assistant", "content": retry_content or ""},
                        {"role": "user", "content": repair_user},
                    ]

            if result is None:
                preview = (response_content or "")[:400].replace("\n", " ")
                return {
                    "support": False,
                    "justification": f"No valid JSON object found in response (after repair attempts): {preview!r}...",
                    "token_usage": token_usage,
                }

            # Strictly follow the JSON schema defined by `factual_recall_zero_shot_prompt`:
            # { "LABEL": "SUPPORTED" | "NOT SUPPORTED", "EXCERPTS": [...], "JUSTIFICATION": "..." }
            required_keys = ("LABEL", "EXCERPTS", "JUSTIFICATION")
            missing = [k for k in required_keys if k not in result]
            if missing:
                return {
                    "support": False,
                    "justification": f"Invalid response format: missing keys {missing}",
                    "token_usage": token_usage,
                }

            raw_label = result.get("LABEL")
            if not isinstance(raw_label, str):
                return {
                    "support": False,
                    "justification": "Invalid response format: LABEL must be a string",
                    "token_usage": token_usage,
                }

            # Normalize whitespace/casing, but only accept the two label values from the prompt.
            label = " ".join(raw_label.strip().upper().split())
            if label == "SUPPORTED":
                support = True
            elif label == "NOT SUPPORTED":
                support = False
            else:
                return {
                    "support": False,
                    "justification": f"Invalid response format: unexpected LABEL value {raw_label!r}",
                    "token_usage": token_usage,
                }

            excerpts = result.get("EXCERPTS")
            if not isinstance(excerpts, list) or not all(isinstance(x, str) for x in excerpts):
                return {
                    "support": False,
                    "justification": "Invalid response format: EXCERPTS must be a list of strings",
                    "token_usage": token_usage,
                }

            justification = result.get("JUSTIFICATION")
            if not isinstance(justification, str):
                return {
                    "support": False,
                    "justification": "Invalid response format: JUSTIFICATION must be a string",
                    "token_usage": token_usage,
                }

            return {
                "support": support,
                "justification": justification,
                "excerpts": excerpts,
                "token_usage": token_usage,
            }

        except Exception as e:
            # Surface judge/provider failures during runs (otherwise this only shows up in output JSON).
            print(f"  ERROR: Exception during fact support check: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'support': False,
                'justification': f'Error in LLM call: {type(e).__name__}: {str(e)}',
                'token_usage': {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
            }
    
    def compute_factual_recall(self, llm_response_text: str, article_atomic_facts: List[str]) -> Dict[str, Any]:
        """
        Compute factual recall - whether scientific conclusion facts are supported by LLM response text.
        
        Args:
            llm_response_text (str): Full text of LLM-generated response
            article_atomic_facts (List[str]): List of atomic facts from scientific conclusions
            
        Returns:
            Dict[str, Any]: Dictionary containing recall metrics and detailed analysis
        """
        if not llm_response_text or not article_atomic_facts:
            return {
                'factual_recall': 0.0,
                'total_article_facts': len(article_atomic_facts) if article_atomic_facts else 0,
                'supported_facts': 0,
                'not_supported_facts': article_atomic_facts if article_atomic_facts else [],
                'coverage_details': [],
                'token_usage': {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
            }
        
        print(f"Checking fact support for {len(article_atomic_facts)} scientific conclusion facts...")
        
        supported_facts = []
        not_supported_facts = []
        coverage_details = []
        
        # Track total token usage
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens = 0
        
        for i, article_fact in enumerate(article_atomic_facts):
            print(f"  Checking fact {i+1}/{len(article_atomic_facts)}: {article_fact[:50]}...", end='', flush=True)
            
            support_result = self.check_fact_support(llm_response_text, article_fact)
            is_supported = support_result.get('support', False)
            justification = support_result.get('justification', '')

            status = "supported" if is_supported else "not supported"
            print(f" → {status}", flush=True)

            if justification:
                justification_preview = justification[:200] + "..." if len(justification) > 200 else justification
                print(f"    Justification: {justification_preview}", flush=True)
            
            # Extract token usage for this fact
            fact_token_usage = support_result.get('token_usage', {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0})
            fact_input_tokens = fact_token_usage.get('input_tokens', 0)
            fact_output_tokens = fact_token_usage.get('output_tokens', 0)
            fact_total_tokens = fact_token_usage.get('total_tokens', 0)
            
            # Accumulate totals
            total_input_tokens += fact_input_tokens
            total_output_tokens += fact_output_tokens
            total_tokens += fact_total_tokens
            
            if support_result['support']:
                supported_facts.append(article_fact)
            else:
                not_supported_facts.append(article_fact)
            
            coverage_details.append({
                'article_fact': article_fact,
                'is_supported': support_result['support'],
                'excerpts': support_result.get('excerpts', []),
                'justification': support_result['justification'],
                'token_usage': {
                    'input_tokens': fact_input_tokens,
                    'output_tokens': fact_output_tokens,
                    'total_tokens': fact_total_tokens
                }
            })
            
            # Optional delay (set via env if you want to throttle)
            delay_s = float(os.getenv("DATA_LABELING_JUDGE_DELAY_S", "0") or 0)
            if delay_s > 0:
                time.sleep(delay_s)
        
        factual_recall = len(supported_facts) / len(article_atomic_facts) if article_atomic_facts else 0.0
        
        return {
            'factual_recall': factual_recall,
            'total_article_facts': len(article_atomic_facts),
            'supported_facts': len(supported_facts),
            'not_supported_facts': not_supported_facts,
            'coverage_details': coverage_details,
            'token_usage': {
                'input_tokens': total_input_tokens,
                'output_tokens': total_output_tokens,
                'total_tokens': total_tokens
            }
        }
