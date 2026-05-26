"""
Helper functions for atomic fact generation.

Covers:
* Sentence preprocessing  (splitting, initials, bullet-point merging, non-content filter)
* LLM response parsing    (decomposition, decontextualization, incomplete-detection, filtering)
* Verbose output          (fact comparisons, pipeline summary)
* JSON serialization      (tuple-keyed metadata to string keys)
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentence preprocessing
# ---------------------------------------------------------------------------

def detect_initials(text: str) -> list:
    """Return list of initial patterns (e.g. 'J. K.') found in *text*."""
    return re.findall(r"[A-Z]\. ?[A-Z]\.", text)


def fix_sentence_splitter(curr_sentences: list, initials: list) -> list:
    """Fix NLTK sentence-splitter errors caused by initials and lowercase continuations."""
    # Pass 1: re-merge sentences split by initials
    for initial in initials:
        if not np.any([initial in sent for sent in curr_sentences]):
            parts = [t.strip() for t in initial.split(".") if t.strip()]
            if len(parts) < 2:
                continue
            alpha1, alpha2 = parts[0], parts[1]
            for i, (s1, s2) in enumerate(zip(curr_sentences, curr_sentences[1:])):
                if s1.endswith(alpha1 + ".") and s2.startswith(alpha2 + "."):
                    curr_sentences = (
                        curr_sentences[:i]
                        + [curr_sentences[i] + " " + curr_sentences[i + 1]]
                        + curr_sentences[i + 2:]
                    )
                    break

    # Pass 2: merge single-word sentences and lowercase continuations
    sentences = []
    combine_with_previous = None

    for idx, sent in enumerate(curr_sentences):
        if len(sent.split()) <= 1 and idx == 0:
            combine_with_previous = True
            sentences.append(sent)
        elif len(sent.split()) <= 1:
            sentences[-1] += " " + sent
            combine_with_previous = False
        elif sent[0].isalpha() and not sent[0].isupper() and idx > 0:
            sentences[-1] += " " + sent
            combine_with_previous = False
        elif combine_with_previous:
            sentences[-1] += " " + sent
            combine_with_previous = False
        else:
            assert not combine_with_previous
            sentences.append(sent)

    return sentences


def is_non_content_sentence(sentence: str) -> bool:
    """Return True if *sentence* is a non-content element (markdown, pleasantries, etc.)."""
    s = sentence.lower().strip()

    if len(s) < 10:
        return True
    if re.match(r"^[^\w\s]*$", sentence):
        return True
    if re.match(r"^#{1,6}\s+", sentence):
        return True
    if re.match(r"^[-*_]{3,}$", sentence) or sentence in ("---", "***", "___"):
        return True

    conversational_patterns = [
        r"^(of course|happy to help|you're welcome|thank you|thanks)$",
        r"^(this question is|this is an excellent)$",
        r"^(here is a detailed breakdown)$",
        r"^(let me|i'll|i will)$",
        r"^(as you can see|as mentioned|as stated)$",
        r"^(in summary|to summarize|in conclusion)$",
        r"^(disclaimer|note:|important:)$",
        r"^(please note|it should be noted)$",
        r"^(for your information|fyi)$",
        r"^(i hope this helps|hope this helps)$",
        r"^(feel free to|don't hesitate to)$",
        r"^(if you have any|if you need any)$",
        r"^(let me know|please let me know)$",
        r"^(i'm here to|i am here to)$",
        r"^(glad to help|happy to assist)$",
        r"^(no problem|not a problem)$",
        r"^(you're right|you are right)$",
        r"^(exactly|precisely|indeed)$",
        r"^(absolutely|definitely|certainly)$",
        r"^(i understand|i see|i get it)$",
        r"^(that makes sense|that's clear)$",
        r"^(good question|great question)$",
        r"^(excellent point|good point)$",
        r"^(i agree|i would agree)$",
        r"^(that's correct|that is correct)$",
        r"^(you're absolutely right)$",
        r"^(i couldn't agree more)$",
        r"^(well said|nicely put)$",
        r"^(that's a great|that is a great)$",
        r"^(i appreciate|thanks for)$",
        r"^(welcome|you're welcome)$",
        r"^(my pleasure|it's my pleasure)$",
        r"^(anytime|any time)$",
        r"^(sure thing|of course)$",
        r"^(no worries|no problem)$",
        r"^(you got it|got it)$",
        r"^(sounds good|sounds great)$",
        r"^(perfect|excellent|great)$",
        r"^(awesome|fantastic|wonderful)$",
        r"^(i'm glad|i am glad)$",
        r"^(that's helpful|that is helpful)$",
        r"^(i hope|hope that)$",
        r"^(let's|let us)$",
        r"^(we can|we could)$",
        r"^(i can|i could)$",
        r"^(i'd be happy|i would be happy)$",
        r"^(i'd love to|i would love to)$",
        r"^(i'm happy to|i am happy to)$",
        r"^(i'm here|i am here)$",
        r"^(i'm available|i am available)$",
        r"^(i'm ready|i am ready)$",
    ]
    for pattern in conversational_patterns:
        if re.match(pattern, s):
            return True

    if (
        re.match(r"^[A-Z\s]+$", sentence)
        and len(sentence.split()) <= 4
        and len(sentence) <= 40
        and s in ("introduction", "conclusion", "summary", "overview",
                  "background", "methods", "results", "discussion")
    ):
        return True

    if len(sentence.split()) <= 2 and not any(c.isdigit() for c in sentence):
        return True
    if re.match(r"^[^\w\s]+$", sentence):
        return True

    return False


def combine_bullet_point_lists(
    sentences: list,
    sentences_to_parent_paragraph: Optional[dict] = None,
) -> Tuple[list, dict]:
    """Merge consecutive bullet-point sentences into a single sentence."""
    if not sentences:
        return sentences, sentences_to_parent_paragraph or {}

    combined = []
    updated_map = {}
    i = 0

    while i < len(sentences):
        curr = sentences[i].strip()
        if is_bullet_point_sentence(curr):
            bullet_group = [curr]
            j = i + 1
            while j < len(sentences) and is_bullet_point_sentence(sentences[j].strip()):
                bullet_group.append(sentences[j].strip())
                j += 1

            if len(bullet_group) > 1:
                merged = _combine_bullet_points(bullet_group)
                combined.append(merged)
                logger.debug("Combined %d bullet points: '%s...'", len(bullet_group), merged[:80])
                if sentences_to_parent_paragraph:
                    updated_map[merged] = sentences_to_parent_paragraph.get(bullet_group[0], "")
            else:
                combined.append(curr)
                if sentences_to_parent_paragraph:
                    updated_map[curr] = sentences_to_parent_paragraph.get(curr, "")
            i = j
        else:
            combined.append(curr)
            if sentences_to_parent_paragraph:
                updated_map[curr] = sentences_to_parent_paragraph.get(curr, "")
            i += 1

    return combined, updated_map


def is_bullet_point_sentence(sentence: str) -> bool:
    """Return True if *sentence* looks like a list item."""
    patterns = [
        r"^[-]\s+",
        r"^\*\s+",
        r"^\d+\.\s+",
        r"^[a-z]\.\s+",
        r"^\([a-z]\)\s+",
        r"^\([ivx]+\)\s+",
    ]
    for pattern in patterns:
        if re.match(pattern, sentence):
            if pattern == r"^\*\s+":
                if not (
                    re.search(r"\*[^*]+\*", sentence)
                    or re.search(r"\*\*[^*]+\*\*", sentence)
                    or sentence.count("*") > 2
                ):
                    return True
            else:
                return True
    return False


def _combine_bullet_points(bullet_points: list) -> str:
    """Join cleaned bullet-point strings into one sentence."""
    if not bullet_points:
        return ""
    if len(bullet_points) == 1:
        return bullet_points[0]

    cleaned = []
    for point in bullet_points:
        p = re.sub(r"^[-*]\s+", "", point)
        p = re.sub(r"^\d+\.\s+", "", p)
        p = re.sub(r"^[a-z]\.\s+", "", p)
        p = re.sub(r"^\([a-z]\)\s+", "", p)
        p = re.sub(r"^\([ivx]+\)\s+", "", p)
        cleaned.append(p.strip())

    if len(cleaned) == 2:
        return "{} and {}".format(cleaned[0], cleaned[1])
    return ", ".join(cleaned[:-1]) + ", and {}".format(cleaned[-1])


# ---------------------------------------------------------------------------
# JSON extraction helpers (internal)
# ---------------------------------------------------------------------------

def _extract_json_from_response(content: str, context: str = "response") -> Optional[str]:
    """Extract a JSON object string from a raw LLM response."""
    if not content:
        return None

    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content, re.DOTALL)
    if match:
        inner = match.group(1).strip()
        json_match = re.search(r"\{[\s\S]*\}", inner, re.DOTALL)
        return json_match.group(0).strip() if json_match else inner

    stripped = content.strip()
    for prefix in ("```json", "```"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            if stripped.endswith("```"):
                stripped = stripped[:-3]
            return stripped.strip()

    json_match = re.search(r"\{[\s\S]*\}", content, re.DOTALL)
    return json_match.group(0).strip() if json_match else stripped


def _parse_json_safe(json_content: str, context: str = "parsing") -> Optional[dict]:
    """Parse JSON, logging an error on failure."""
    if not json_content:
        return None
    try:
        return json.loads(json_content)
    except json.JSONDecodeError as e:
        logger.error("JSON parse error [%s]: %s", context, e)
        logger.debug("Content (first 500 chars): %r", json_content[:500])
        return None


# ---------------------------------------------------------------------------
# LLM response parsers
# ---------------------------------------------------------------------------

def text_to_sentences(response) -> list:
    """Extract the ``atomic_facts`` list from a decomposition response object."""
    content = response.choices[0].message.content
    if not content:
        return []

    json_str = _extract_json_from_response(content, "decomposition")
    if json_str:
        data = _parse_json_safe(json_str, "decomposition")
        if data:
            facts = data.get("atomic_facts", [])
            return facts if isinstance(facts, list) else []

    facts = []
    for line in content.split("\n"):
        line = line.strip()
        line = re.sub(r"^[-*]\s+", "", line)
        line = re.sub(r"^\d+\.\s+", "", line)
        if line and len(line) > 10:
            facts.append(line)
    return facts


def parse_decontextualization_response(response) -> Tuple[Optional[str], Optional[str]]:
    """Parse a decontextualization response.

    Returns:
        ``(decontextualized_fact, justification)`` or ``(None, None)`` on failure.
    """
    content = response.choices[0].message.content
    if not content:
        return None, None

    json_str = _extract_json_from_response(content, "decontextualization")
    data = _parse_json_safe(json_str, "decontextualization") if json_str else None
    if not data:
        return None, None

    return data.get("decontextualized_fact"), data.get("justification")


def parse_incomplete_fact_response(response) -> dict:
    """Parse an incomplete-fact-detection response.

    Returns:
        Dict with keys ``classification``, ``dependent_type``, ``explanation``,
        ``rewritten_claim``. Values are ``None`` on parse failure.
    """
    empty = {
        "classification": None, "dependent_type": None,
        "explanation": None, "rewritten_claim": None,
    }
    content = response.choices[0].message.content
    if not content:
        logger.error("Empty response for incomplete fact detection")
        return empty

    json_str = _extract_json_from_response(content, "incomplete fact detection")
    data = _parse_json_safe(json_str, "incomplete fact detection") if json_str else None
    if not data:
        return empty

    dependent_type = data.get("dependent_type")
    if dependent_type and str(dependent_type).lower() == "none":
        dependent_type = None

    return {
        "classification": data.get("classification"),
        "dependent_type": dependent_type,
        "explanation": data.get("explanation"),
        "rewritten_claim": data.get("rewritten_claim"),
    }


def parse_filter_irrelevant_facts_response(response) -> dict:
    """Parse an irrelevant-fact-filtering response (Foo / Not Foo).

    Returns:
        Dict with keys ``reasoning``, ``classification``.
    """
    empty = {"reasoning": None, "classification": None}
    content = response.choices[0].message.content
    if not content:
        logger.error("Empty response for irrelevant fact filtering")
        return empty

    json_str = _extract_json_from_response(content, "irrelevant fact filtering")
    data = _parse_json_safe(json_str, "irrelevant fact filtering") if json_str else None
    if not data:
        return empty

    return {"reasoning": data.get("reasoning"), "classification": data.get("classification")}


def parse_filter_redundant_facts_response(response) -> dict:
    """Parse a single-fact redundancy-check response.

    Returns:
        Dict with keys ``reasoning``, ``classification``, ``redundant_with``.
    """
    empty = {"reasoning": None, "classification": None, "redundant_with": []}
    content = response.choices[0].message.content
    if not content:
        logger.error("Empty response for redundant fact filtering")
        return empty

    json_str = _extract_json_from_response(content, "redundant fact filtering")
    data = _parse_json_safe(json_str, "redundant fact filtering") if json_str else None
    if not data:
        return empty

    rw = data.get("redundant_with", [])
    if not isinstance(rw, list):
        rw = []
    else:
        rw = [int(x) if str(x).isdigit() else x for x in rw]

    return {
        "reasoning": data.get("reasoning"),
        "classification": data.get("classification"),
        "redundant_with": rw,
    }


def parse_filter_redundant_facts_batch_response(response) -> dict:
    """Parse a batch redundancy-selection response.

    Returns:
        Dict with keys ``reasoning``, ``selected_statements`` (1-based int list).
    """
    empty = {"reasoning": None, "selected_statements": []}
    content = response.choices[0].message.content
    if not content:
        logger.error("Empty response for batch redundant fact filtering")
        return empty

    json_str = _extract_json_from_response(content, "batch redundant fact filtering")
    data = _parse_json_safe(json_str, "batch redundant fact filtering") if json_str else None
    if not data:
        return empty

    sel = data.get("selected_statements", [])
    if not isinstance(sel, list):
        sel = []
    else:
        sel = [int(x) if str(x).isdigit() else x for x in sel]

    return {"reasoning": data.get("reasoning"), "selected_statements": sel}


# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------

def print_fact_comparison(
    sentence_index: int,
    pairs_1: list,
    pairs_2: list,
    title_1: str,
    title_2: str,
):
    """Print a side-by-side diff of two fact sets for one sentence index."""
    if not (0 <= sentence_index < len(pairs_1) and sentence_index < len(pairs_2)):
        logger.error("Sentence index %d out of range.", sentence_index)
        return

    _, facts_1 = pairs_1[sentence_index]
    _, facts_2 = pairs_2[sentence_index]
    set_1, set_2 = set(facts_1 or []), set(facts_2 or [])

    logger.info("\n%s vs %s:", title_1, title_2)
    logger.info("-" * 80)
    for f in sorted(set_1 & set_2):
        logger.info("  [both] %s", f)
    for f in sorted(set_1 - set_2):
        logger.info("  [%s only] %s", title_1, f)
    for f in sorted(set_2 - set_1):
        logger.info("  [%s only] %s", title_2, f)
    logger.info("")


def print_verbose_comparisons(
    initial_pairs, dectx_pairs, rewritten_pairs, final_pairs,
    dep_meta, irr_meta, red_meta, token_usage,
):
    """Log a full verbose summary of the pipeline run.

    Note: output is sent via logging.info; configure logging to see it.
    """
    logger.info("\n" + "=" * 80)
    logger.info("VERBOSE: DETAILED COMPARISONS AND METADATA")
    logger.info("=" * 80)

    print_fact_comparison(1, initial_pairs, dectx_pairs, "INITIAL", "DECONTEXTUALIZED")
    print_fact_comparison(1, dectx_pairs, rewritten_pairs, "DECONTEXTUALIZED", "REWRITTEN")

    logger.info("-" * 80)
    logger.info("Dependent facts: %d", len(dep_meta))
    for i, ((_, fact), meta) in enumerate(list(dep_meta.items())[:3]):
        dep_type, explanation, classification, rewritten = meta
        logger.info("\n  %d. Type: %s | Original: %s", i + 1, dep_type, fact)
        logger.info("       Rewritten: %s", rewritten)
        if explanation:
            logger.info("       Explanation: %s", explanation)

    logger.info("-" * 80)
    logger.info("Irrelevant facts filtered: %d", len(irr_meta))
    for i, ((_, fact), meta) in enumerate(list(irr_meta.items())[:5]):
        question, statement, reasoning, classification = meta
        logger.info("\n  %d. [%s] %s", i + 1, classification, statement)
        if reasoning:
            logger.info("       Reasoning: %s", reasoning)

    logger.info("-" * 80)
    logger.info("Redundant facts filtered: %d", len(red_meta))
    for i, ((_, fact), meta) in enumerate(list(red_meta.items())[:3]):
        statement, reasoning, classification = meta[0], meta[1], meta[2]
        redundant_with = meta[3] if len(meta) > 3 else []
        logger.info("\n  %d. [%s] %s", i + 1, classification, statement)
        if redundant_with:
            logger.info("       Redundant with: %s", ", ".join(repr(f) for f in redundant_with[:3]))
        if reasoning:
            logger.info("       Reasoning: %s", reasoning)

    logger.info("-" * 80)
    logger.info("TOKEN USAGE")
    steps = (
        "decomposition", "decontextualization", "incomplete_detection",
        "irrelevant_filtering", "redundant_filtering", "total",
    )
    for step in steps:
        u = token_usage.get(step, {})
        logger.info(
            "  %-25s  in=%s  out=%s  total=%s",
            step,
            "{:,}".format(u.get("prompt_tokens", 0)),
            "{:,}".format(u.get("completion_tokens", 0)),
            "{:,}".format(u.get("total_tokens", 0)),
        )
    logger.info("\n" + "=" * 80 + "\n")


# ---------------------------------------------------------------------------
# JSON serialization utilities
# ---------------------------------------------------------------------------

def convert_metadata_for_json(metadata_dict: dict) -> dict:
    """Convert tuple-keyed metadata to JSON-safe string keys ('sent|||fact' format)."""
    return {"{}|||{}".format(k[0], k[1]): v for k, v in metadata_dict.items()}


def convert_justifications_for_json(justifications_dict: dict) -> dict:
    """Convert tuple or string keys to JSON-safe string keys."""
    result = {}
    for k, v in justifications_dict.items():
        key_str = "{}|||{}".format(k[0], k[1]) if isinstance(k, tuple) else str(k)
        result[key_str] = v
    return result
