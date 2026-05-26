"""
Helper functions for question generation.
"""

import json
from typing import List, Optional

_DEFAULT_QUESTION_WORDS: List[str] = [
    "what", "how", "why", "when", "where", "which", "who",
    "is", "are", "does", "do", "can", "will", "should", "would", "could", "may", "might",
]
_DEFAULT_MIN_LENGTH: int = 10


def parse_question_response(content: str) -> str:
    """
    Parse the model's text response into a question string.

    Expects JSON with a ``"question"`` key, optionally wrapped in a markdown
    code fence.  Returns an empty string on any parse failure.

    Args:
        content: Raw text content returned by the model.

    Returns:
        Extracted question string, or ``""`` on failure.
    """
    if not content:
        print("ERROR: Empty response content")
        return ""

    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
    elif content.startswith("```"):
        content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    try:
        json_data = json.loads(content)
        return json_data["question"]
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Content that failed to parse: {repr(content)}")
        return ""
    except KeyError:
        print(f"Missing 'question' key. Available keys: {list(json_data.keys())}")
        return ""


def validate_question(
    question: str,
    min_length: Optional[int] = None,
    question_words: Optional[List[str]] = None,
) -> str:
    """
    Validate and clean a generated question.

    Args:
        question: Generated question to validate.
        min_length: Minimum character count after stripping whitespace.
                    Defaults to ``_DEFAULT_MIN_LENGTH`` (10).
        question_words: List of interrogative/auxiliary words the question must
                        contain at least one of.  Defaults to ``_DEFAULT_QUESTION_WORDS``.

    Returns:
        Cleaned question string, or ``""`` if validation fails.
    """
    if min_length is None:
        min_length = _DEFAULT_MIN_LENGTH
    if question_words is None:
        question_words = _DEFAULT_QUESTION_WORDS

    if not question or not isinstance(question, str):
        return ""

    question = question.strip()

    if not question.endswith("?"):
        question += "?"

    if len(question) < min_length:
        return ""

    if not any(word in question.lower() for word in question_words):
        return ""

    return question
