#!/usr/bin/env python3
"""
Factual Precision Analyzer

A tool for analyzing the precision of atomic facts extracted from LLM responses
against ground-truth reference text (e.g., from Cochrane articles).
Uses LLM to determine whether LLM-generated facts are supported, contradicted, or not supported
with respect to the ground truth.
"""

import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

from .model_class.llm_judge_base import LLMJudgeBase
from .utils.prompts import (
    factual_precision_few_shot_prompt,
    system_prompt as default_system_prompt,
)
from .utils.precision_equations import (
    get_precision_function,
    compute_all_precision_metrics,
)

def resolve_ground_truth_text(
    reference_text: str,
    doi: Optional[str] = None,
) -> tuple[str, str]:
    """
    Choose ground-truth text for judging.

    This analyzer intentionally does **not** fetch external ground-truth text (e.g. via Jina).
    If a DOI is provided, it is ignored and the supplied ``reference_text`` is always used.

    Returns:
        (text, source) where source is always ``"reference"``.
    """
    _ = doi  # kept for backward compatibility with older call sites
    return (reference_text or "", "reference")


def _fill_precision_prompt_template(
    template: str, llm_fact: str, ground_truth_text: str
) -> str:
    """
    Insert fact and reference text into the judge template.

    Uses str.replace instead of str.format so that literal braces in few-shot prompts
    (e.g. [{'heading': ...}] in Plain language summaries) are not treated as format fields.
    """
    return template.replace("{ground_truth_text}", ground_truth_text).replace(
        "{llm_fact}", llm_fact
    )


def _strip_markdown_json_fence(text: str) -> str:
    """Remove optional ``` / ```json fences from model output (common with Gemini)."""
    s = (text or "").strip()
    if not s.startswith("```"):
        return s
    lines = s.split("\n")
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_precision_markdown_fallback(text: str) -> Optional[Dict[str, Any]]:
    """
    When the model returns markdown sections instead of JSON, e.g.:

        ## LABEL
        **CONTRADICTED**

        ## EXCERPTS
        1. "quote one"

        ## JUSTIFICATION
        ...

    Returns a dict with LABEL / EXCERPTS / JUSTIFICATION keys compatible with JSON parsing,
    or None if no usable LABEL section is found.
    """
    if not (text or "").strip():
        return None
    header_re = re.compile(r"^#{1,2}\s+(\w+)\s*$", re.MULTILINE | re.IGNORECASE)
    matches = list(header_re.finditer(text))
    if not matches:
        return None
    sections: Dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if name in ("LABEL", "EXCERPTS", "JUSTIFICATION"):
            sections[name] = body
    if "LABEL" not in sections:
        return None

    label_raw = sections["LABEL"]
    label_flat = re.sub(r"[*_`#]", "", label_raw)
    label_line = label_flat.strip().split("\n")[0].strip()
    tl = " ".join(label_line.upper().replace("_", " ").split())
    if tl == "NOT SUPPORTED" or tl.startswith("NOT SUPPORTED"):
        label_norm = "NOT SUPPORTED"
    elif "CONTRADICTED" in tl:
        label_norm = "CONTRADICTED"
    elif tl == "SUPPORTED" or (tl.startswith("SUPPORTED") and not tl.startswith("NOT")):
        label_norm = "SUPPORTED"
    else:
        return None

    ex_body = sections.get("EXCERPTS", "")
    excerpts: List[str] = []
    quoted = re.findall(r'"((?:[^"\\]|\\.)*)"', ex_body, re.DOTALL)
    if quoted:
        excerpts = [q.replace('\\"', '"').replace("\\n", "\n").strip() for q in quoted if q.strip()]
    if not excerpts:
        for line in ex_body.split("\n"):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^(\d+\.|[-*+])\s+", "", line).strip()
            if len(line) >= 2 and line[0] == '"' and line[-1] == '"':
                excerpts.append(line[1:-1].strip())
            elif line:
                excerpts.append(line)

    justification = sections.get("JUSTIFICATION", "").strip() or "No justification provided"

    return {
        "LABEL": label_norm,
        "EXCERPTS": excerpts,
        "JUSTIFICATION": justification,
    }


def _extract_first_json_object(text: str) -> Optional[str]:
    """
    Find the first balanced `{ ... }` span in text, respecting string literals.
    Helps when the model wraps JSON in prose or when greedy `\\{.*\\}` fails.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _precision_try_raw_decode_object(s: str) -> Optional[Dict[str, Any]]:
    """
    First JSON object in text via JSONDecoder.raw_decode (handles trailing prose / 'Extra data').
    """
    s = (s or "").strip()
    if not s:
        return None
    dec = json.JSONDecoder()
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
        if isinstance(val, dict) and any(
            k in val for k in ("LABEL", "judgment", "label", "EXCERPTS", "JUSTIFICATION")
        ):
            return val
        if isinstance(val, list):
            for item in val:
                if isinstance(item, dict) and ("LABEL" in item or "label" in item):
                    return item
    return None


def _parse_precision_truncated_json_fallback(text: str) -> Optional[Dict[str, Any]]:
    """
    Recover LABEL (and any fully closed EXCERPTS strings) when output is cut off mid-JSON
    (typical when few-shot context is long and max_tokens is tight).
    """
    if not (text or "").strip():
        return None
    label_m = re.search(r'"LABEL"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if not label_m:
        return None
    raw_label = " ".join(label_m.group(1).strip().upper().split())
    if raw_label in ("NOT_SUPPORTED", "NOT-SUPPORTED"):
        raw_label = "NOT SUPPORTED"
    if raw_label == "NOT SUPPORTED":
        label_norm = "NOT SUPPORTED"
    elif raw_label == "CONTRADICTED":
        label_norm = "CONTRADICTED"
    elif raw_label == "SUPPORTED":
        label_norm = "SUPPORTED"
    else:
        return None

    excerpts: List[str] = []
    ex_block_m = re.search(r'"EXCERPTS"\s*:\s*\[(.*)', text, re.DOTALL | re.IGNORECASE)
    if ex_block_m:
        block = ex_block_m.group(1)
        end_m = re.search(r']\s*,\s*"JUSTIFICATION"\s*:', block, re.IGNORECASE)
        if end_m:
            block = block[: end_m.start()]
        for qm in re.finditer(r'"((?:[^"\\]|\\.)*)"', block, re.DOTALL):
            piece = qm.group(1).replace('\\"', '"').replace("\\n", "\n").strip()
            if piece:
                excerpts.append(piece)

    justification = (
        "Recovered from truncated JSON; EXCERPTS/JUSTIFICATION may be incomplete."
        if not excerpts
        else "Recovered from truncated JSON; JUSTIFICATION may be incomplete."
    )
    just_val_m = re.search(
        r'"JUSTIFICATION"\s*:\s*"((?:[^"\\]|\\.)*)"',
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if just_val_m:
        justification = just_val_m.group(1).replace('\\"', '"').replace("\\n", "\n").strip()

    return {"LABEL": label_norm, "EXCERPTS": excerpts, "JUSTIFICATION": justification}


def _parse_precision_judge_response(response_content: str) -> Optional[Dict[str, Any]]:
    """
    Parse judge output: full JSON, raw_decode (trailing prose), fenced JSON, balanced object,
    greedy regex, markdown sections, or truncated JSON with at least LABEL present.
    Returns None if nothing usable was parsed.
    """
    if not (response_content or "").strip():
        return None

    work = _strip_markdown_json_fence(response_content)

    try:
        parsed = json.loads(work)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    raw_obj = _precision_try_raw_decode_object(work)
    if raw_obj is not None:
        return raw_obj

    subset = _extract_first_json_object(work)
    if subset:
        try:
            parsed = json.loads(subset)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    try:
        json_match = re.search(r"\{.*\}", work, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, dict):
                return parsed
    except json.JSONDecodeError:
        pass

    md_parsed = _parse_precision_markdown_fallback(work)
    if md_parsed is not None:
        return md_parsed

    partial = _parse_precision_truncated_json_fallback(work)
    if partial is not None:
        return partial

    return None


class FactualPrecisionAnalyzer:
    """
    An analyzer for computing factual precision metrics between LLM response facts
    and ground-truth reference text.
    
    Uses an LLM to determine whether atomic facts from LLM responses are supported,
    contradicted, or not supported with respect to the ground-truth reference text.
    """
    
    def __init__(
        self,
        llm_judge: LLMJudgeBase,
        system_prompt: str = default_system_prompt,
        precision_prompt_template: str = factual_precision_few_shot_prompt,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        """
        Initialize the FactualPrecisionAnalyzer.

        Args:
            llm_judge: LLMJudgeBase implementation provided by the caller.
            system_prompt: System prompt for the judge (default: `system_prompt` from `data_labeling.utils.prompts`).
            precision_prompt_template: Prompt template containing {llm_fact} and {ground_truth_text}.
            temperature: Sampling temperature for the judge.
            max_tokens: Max tokens for the judge response.
        """
        self._llm_judge = llm_judge
        self._system_prompt = system_prompt
        self._precision_prompt_template = precision_prompt_template
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
    
    def check_fact_judgment(
        self,
        llm_fact: str,
        ground_truth_text: str,
        doi: Optional[str] = None,
        *,
        preresolved: Optional[tuple[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Use LLM to check if an LLM-generated atomic fact is supported, contradicted, or not supported
        with respect to the ground-truth reference text.
        
        Args:
            llm_fact (str): Single atomic fact from LLM response
            ground_truth_text (str): Reference text (e.g., from article metadata).
            doi (str, optional): Accepted for backward compatibility; ignored.
            preresolved: If set, ``(resolved_text, source)`` from a prior call to
                :func:`resolve_ground_truth_text`.
            
        Returns:
            Dict[str, Any]: Judgment (supported/contradicted/not supported) and reasoning

        On empty responses or unparseable output, the judge is called again up to
        ``DATA_LABELING_PRECISION_JUDGE_MAX_ATTEMPTS`` times (default ``3``). Token counts
        include all attempts.
        """
        if not llm_fact or not llm_fact.strip():
            return {
                'judgment': 'NOT SUPPORTED',
                'justification': 'Empty LLM fact provided',
                'token_usage': {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0},
                'ground_truth_source': 'reference',
            }
        
        if preresolved is not None:
            resolved_text, gt_source = preresolved
        else:
            resolved_text, gt_source = resolve_ground_truth_text(ground_truth_text, doi)
        if not resolved_text or not resolved_text.strip():
            return {
                'judgment': 'NOT SUPPORTED',
                'justification': 'Empty ground truth text provided',
                'token_usage': {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0},
                'ground_truth_source': gt_source,
            }
        
        prompt = _fill_precision_prompt_template(
            self._precision_prompt_template, llm_fact, resolved_text
        )
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            try:
                max_attempts = max(
                    1,
                    int(os.getenv("DATA_LABELING_PRECISION_JUDGE_MAX_ATTEMPTS", "3")),
                )
            except ValueError:
                max_attempts = 3
            total_input_tokens = 0
            total_output_tokens = 0
            total_tokens_acc = 0
            response_content = ""
            result: Optional[Dict[str, Any]] = None

            for attempt in range(max_attempts):
                response_content, usage = self._llm_judge.complete(
                    messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
                in_tok = usage.get("input_tokens", 0)
                out_tok = usage.get("output_tokens", 0)
                tot_tok = usage.get("total_tokens", 0) or (
                    usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                )
                total_input_tokens += in_tok
                total_output_tokens += out_tok
                total_tokens_acc += tot_tok

                token_usage = {
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                    "total_tokens": total_tokens_acc,
                }

                if not (response_content or "").strip():
                    if attempt < max_attempts - 1:
                        print(
                            f"  WARNING: Empty response from LLM (attempt {attempt + 1}/{max_attempts}), retrying..."
                        )
                        continue
                    print(f"  WARNING: Empty response from LLM after {max_attempts} attempt(s)")
                    return {
                        "judgment": "NOT SUPPORTED",
                        "justification": "Empty response from LLM",
                        "raw_json_response": "",
                        "token_usage": token_usage,
                        "ground_truth_source": gt_source,
                    }

                result = _parse_precision_judge_response(response_content)
                if result is not None:
                    break

                if attempt < max_attempts - 1:
                    print(
                        f"  WARNING: Unparseable judge output (attempt {attempt + 1}/{max_attempts}), retrying..."
                    )
                    print(f"  Response preview (first 500 chars): {response_content[:500]}")
                    continue

                print(
                    f"  WARNING: No JSON or markdown LABEL/EXCERPTS sections found after {max_attempts} attempt(s)"
                )
                print(f"  Response preview (first 500 chars): {response_content[:500]}")
                return {
                    "judgment": "NOT SUPPORTED",
                    "justification": f"No parseable judge output: {response_content[:200]}",
                    "raw_json_response": response_content,
                    "token_usage": token_usage,
                    "ground_truth_source": gt_source,
                }

            assert result is not None  # loop exits with break only when parse succeeded

            # Validate and normalize the judgment
            # We stick to: SUPPORTED / NOT SUPPORTED / CONTRADICTED
            # Prompt schema uses LABEL / EXCERPTS / JUSTIFICATION; some models also emit judgment / excerpts / justification.
            judgment_raw = str(
                result.get("judgment")
                or result.get("LABEL")
                or result.get("label")
                or ""
            ).strip()
            judgment = judgment_raw.lower().strip()
            
            # Debug: print what we got
            if not judgment:
                print(f"  WARNING: No judgment/LABEL field in result. Keys: {list(result.keys())}")
                print(f"  Result preview: {str(result)[:300]}")
            
            # Check "not supported" first so it is not misclassified by substring matching.
            if judgment in ('not supported', 'not_supported', 'not-supported'):
                judgment = 'NOT SUPPORTED'
            elif judgment in ('supported',):
                judgment = 'SUPPORTED'
            elif judgment in ('contradicted',):
                judgment = 'CONTRADICTED'
            else:
                # Heuristics for imperfect outputs
                if judgment.startswith('not ') or 'not_support' in judgment or 'not support' in judgment:
                    judgment = 'NOT SUPPORTED'
                elif 'contradict' in judgment or 'refute' in judgment:
                    judgment = 'CONTRADICTED'
                elif 'support' in judgment or 'entail' in judgment:
                    judgment = 'SUPPORTED'
                else:
                    print(f"  WARNING: Unexpected judgment value: '{judgment_raw}'. Defaulting to NOT SUPPORTED.")
                    print(f"  Full result keys: {list(result.keys())}")
                    print(f"  Full result preview: {str(result)[:400]}")
                    judgment = 'NOT SUPPORTED'
            
            justification = (
                result.get("justification")
                or result.get("JUSTIFICATION")
                or "No justification provided"
            )
            excerpts = result.get("excerpts") or result.get("EXCERPTS") or []
            if not isinstance(excerpts, list):
                excerpts = [excerpts] if excerpts else []
            
            return {
                'judgment': judgment,
                'excerpts': excerpts,
                'justification': justification,
                'raw_json_response': response_content,  # Save the raw JSON output from the prompt
                'token_usage': token_usage,
                'ground_truth_source': gt_source,
            }
            
        except Exception as e:
            print(f"  ERROR: Exception during fact judgment: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'judgment': 'NOT SUPPORTED',
                'justification': f'Error in LLM call: {type(e).__name__}: {str(e)}',
                'raw_json_response': '',  # No response due to error
                'token_usage': {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0},
                'ground_truth_source': gt_source,
            }
    
    def compute_factual_precision(self, 
                                 llm_atomic_facts: List[str],
                                 ground_truth_text: str,
                                 precision_function: Optional[str] = 'main',
                                 custom_precision_func: Optional[Callable] = None,
                                 doi: Optional[str] = None) -> Dict[str, Any]:
        """
        Compute factual precision - whether LLM facts are supported, contradicted, or not supported
        with respect to the ground-truth reference text.
        
        Args:
            llm_atomic_facts (List[str]): List of atomic facts from LLM response
            ground_truth_text (str): Reference text (e.g., from Cochrane article metadata).
            precision_function (str): Name of the precision function to use (default: 'main').
                Options: 'main', 'entailment', 'no_contradiction'
            custom_precision_func (Callable): Optional custom precision function that takes
                (supported, contradicted, not supported, total) and returns a float
            doi (str, optional): Accepted for backward compatibility; ignored.
        
        Returns:
            Dict[str, Any]: Dictionary containing precision metrics and detailed analysis
        """
        preresolved = resolve_ground_truth_text(ground_truth_text, doi)
        resolved_gt, gt_source = preresolved
        if not llm_atomic_facts or not resolved_gt:
            return {
                'factual_precision': 0.0,
                'total_llm_facts': len(llm_atomic_facts) if llm_atomic_facts else 0,
                'supported_facts': 0,
                'contradicted_facts': 0,
                'not_supported_facts': 0,
                'precision_details': [],
                'all_precision_metrics': {},
                'ground_truth_source': gt_source,
            }
        
        print(f"Checking fact judgments for {len(llm_atomic_facts)} LLM facts...")
        
        supported_facts = []
        contradicted_facts = []
        not_supported_facts = []
        precision_details = []
        
        # Track total token usage
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens = 0
        
        for i, llm_fact in enumerate(llm_atomic_facts):
            print(f"  Checking fact {i+1}/{len(llm_atomic_facts)}: {llm_fact[:50]}...", end='', flush=True)
            
            judgment_result = self.check_fact_judgment(
                llm_fact, ground_truth_text, doi=doi, preresolved=preresolved
            )
            judgment = judgment_result.get('judgment', 'NOT SUPPORTED')
            justification = judgment_result.get('justification', '')
            
            # Print judgment immediately on same line
            print(f" → {judgment}", flush=True)
            
            # Print justification if available (truncate if too long)
            if justification:
                justification_preview = justification[:200] + "..." if len(justification) > 200 else justification
                print(f"    Justification: {justification_preview}", flush=True)
 
            # Extract token usage for this fact
            fact_token_usage = judgment_result.get('token_usage', {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0})
            fact_input_tokens = fact_token_usage.get('input_tokens', 0)
            fact_output_tokens = fact_token_usage.get('output_tokens', 0)
            fact_total_tokens = fact_token_usage.get('total_tokens', 0)
            
            # Accumulate totals
            total_input_tokens += fact_input_tokens
            total_output_tokens += fact_output_tokens
            total_tokens += fact_total_tokens
            
            if judgment == 'SUPPORTED':
                supported_facts.append(llm_fact)
            elif judgment == 'CONTRADICTED':
                contradicted_facts.append(llm_fact)
            else:  # NOT SUPPORTED
                not_supported_facts.append(llm_fact)
            
            precision_details.append({
                'llm_fact': llm_fact,
                'judgment': judgment,
                'excerpts': judgment_result.get('excerpts', []),
                'justification': judgment_result.get('justification', ''),
                'raw_json_response': judgment_result.get('raw_json_response', ''),  # Save raw JSON from prompt
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
        
        # Compute precision using the specified function
        total = len(llm_atomic_facts)
        supported_count = len(supported_facts)
        contradicted_count = len(contradicted_facts)
        not_supported_count = len(not_supported_facts)
        
        if custom_precision_func:
            factual_precision = custom_precision_func(
                supported_count, contradicted_count, not_supported_count, total
            )
        else:
            precision_func = get_precision_function(precision_function)
            factual_precision = precision_func(
                supported_count, contradicted_count, not_supported_count, total
            )
        
        # Compute all precision metrics for comparison
        all_precision_metrics = compute_all_precision_metrics(
            supported_count, contradicted_count, not_supported_count, total
        )
        
        return {
            'factual_precision': factual_precision,
            'precision_function_used': precision_function if not custom_precision_func else 'custom',
            'total_llm_facts': total,
            'supported_facts': supported_count,
            'contradicted_facts': contradicted_count,
            'not_supported_facts': not_supported_count,
            'supported_facts_list': supported_facts,
            'contradicted_facts_list': contradicted_facts,
            'not_supported_facts_list': not_supported_facts,
            'precision_details': precision_details,
            'all_precision_metrics': all_precision_metrics,
            'ground_truth_source': gt_source,
            'token_usage': {
                'input_tokens': total_input_tokens,
                'output_tokens': total_output_tokens,
                'total_tokens': total_tokens
            }
        }
    
