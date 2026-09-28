"""Context compaction for the long-horizon MCP tool-calling loop.

Adapted from Terminus-2's summarization handoff (``reference_terminus2.py``),
specialized for SciConBench's evidence-synthesis task:

- The system prompt (``RESEARCH_ASSISTANT_PROMPT``) is never part of
  ``messages`` — every provider injects it on each call — so it is always
  preserved. Compaction only rewrites the conversation after it.
- Trigger: proactively when the estimated free context drops below
  ``free_tokens_threshold``, and reactively when a provider raises
  ``ContextLengthExceededError``.
- Procedure (Terminus-style three steps, each a tool-free LLM call):
    1. Summary: the model summarizes the research transcript into an
       evidence-focused handoff (actions, relevant sources with verbatim
       details, evidence quality, synthesis so far, remaining gaps).
    2. Questions: a fresh model that sees only the question, summary, and
       source ledger asks what it still needs to write the final conclusion.
    3. Answers: the model with the full transcript answers those questions.
  Falls back to a single short summary, then to a no-LLM handoff.
- A deterministic source ledger (title, URL, date, verbatim snippets) is
  extracted from every tool result, accumulated across compactions, ranked
  by relevance to the question, and appended to the handoff so citations
  survive compaction even if the summary omits them.
- The compacted history is plain ``user``/``assistant`` text, so it works
  unchanged with every provider's message format.

The transcript given to the summarization calls is a flattened text rendering
of the history (rather than the raw provider messages) so it can be budgeted
to fit the context window and never violates tool-call/tool-result pairing
rules when sent without tools.
"""

import json
import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from ..llm_providers.base import ContextLengthExceededError, LLMProvider
from ..llm_providers.reasoning_discovery import (
    candidate_openrouter_slugs,
    discover_context_length,
)
from ..prompts import RESEARCH_ASSISTANT_PROMPT
from .utils import count_tokens, get_log_file_path

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_LIMIT = 128_000
DEFAULT_FREE_TOKENS_THRESHOLD = 32_000
DEFAULT_MAX_COMPACTIONS = 10
DEFAULT_OUTPUT_RESERVE = 8192
# tiktoken's cl100k_base undercounts Claude/Gemini/DeepSeek tokenizers; scale
# estimates up so proactive compaction fires before a real overflow.
TOKEN_ESTIMATE_SAFETY_FACTOR = 1.2
# Summaries are much longer than agent turns; with max reasoning effort the
# agent's per-turn cap (8192) can be spent entirely on reasoning, leaving no text.
COMPACTION_MAX_OUTPUT_TOKENS = 32_768
_OUTPUT_TOKEN_ATTRS = ("max_tokens", "max_output_tokens")

LEDGER_TOKEN_BUDGET = 8192
_LEDGER_SNIPPETS_PER_SOURCE = 3
_LEDGER_SNIPPET_CHARS = 400
_LEDGER_MIN_RELEVANCE = 0.2  # fraction of question terms a source must mention to keep its snippets
_TRANSCRIPT_SAFETY_MARGIN = 2_000
_TRANSCRIPT_TOOL_RESULT_CAPS: List[Optional[int]] = [None, 16_000, 8_000, 4_000, 2_000, 1_000, 500]

_VENDOR_PREFIXES: Dict[str, List[str]] = {
    "ClaudeProvider": ["anthropic"],
    "OpenAIProvider": ["openai"],
    "GeminiProvider": ["google"],
}
# OpenRouter lists Claude models with the 1M-token beta context, which the
# native ClaudeProvider does not request (no ``context-1m`` beta header).
_NATIVE_CONTEXT_CAPS: Dict[str, int] = {"ClaudeProvider": 1_000_000}
# Used when the model is missing from OpenRouter's catalog (e.g. Gemini previews).
_NATIVE_CONTEXT_DEFAULTS: Dict[str, int] = {"GeminiProvider": 1_048_576}
_FALLBACK_VENDOR_PREFIXES = [
    "deepseek", "openai", "anthropic", "google", "moonshotai", "qwen",
    "z-ai", "minimax", "meta-llama", "mistralai", "x-ai",
]

# Opaque replay tokens (Claude thinking signatures, Gemini thought signatures,
# OpenAI encrypted reasoning) that are not billed as readable context text.
_NON_CONTEXT_KEYS = {
    "signature", "_thought_signature_bytes", "thought_signature", "encrypted_content",
}

_STOPWORDS = {
    "the", "and", "for", "are", "with", "that", "this", "from", "what", "which",
    "who", "whom", "whose", "when", "where", "why", "how", "does", "did", "was",
    "were", "been", "being", "have", "has", "had", "not", "but", "can", "could",
    "should", "would", "may", "might", "will", "shall", "their", "there", "these",
    "those", "than", "then", "into", "onto", "about", "over", "under", "between",
    "among", "such", "other", "any", "all", "each", "more", "most", "less", "some",
    "its", "our", "your", "his", "her", "they", "them", "you", "also", "use",
    "used", "using", "versus", "compared", "effect", "effects", "people", "patients",
}

_CONTINUE_RESEARCH_INSTRUCTION = (
    "Continue working on the original question from where the previous research assistant "
    "left off. You can no longer ask them questions. Use the tools to fill the remaining "
    "evidence gaps without repeating searches that were already done. "
)
_FINAL_ANSWER_INSTRUCTION = (
    "The research budget for this question is exhausted: tools are no longer available and "
    "you cannot ask the previous research assistant further questions. Write the final answer "
    "now, using only the evidence in this handoff and the source ledger. "
)

_HANDOFF_FORMAT_REMINDER = (
    "When the evidence is sufficient, provide the final answer following the task "
    "requirements: a comprehensive, paragraph-long, evidence-backed conclusion wrapped in "
    "exactly three square brackets on each side ([[[...]]]), describing strengths and "
    "limitations of the evidence, followed by references for the sources cited. Please follow "
    "the Task Requirments strictly. Only cite sources that appear in this handoff, the source "
    "ledger, or new tool results; never fabricate snippets or citations."
)


# ============================================================================
# Context window / token estimation
# ============================================================================

def resolve_context_limit(llm_provider: LLMProvider, override: Optional[int] = None) -> int:
    """Context window (tokens) for ``llm_provider``'s model: ``override`` if
    given, else OpenRouter's catalog ``context_length``, else
    ``DEFAULT_CONTEXT_LIMIT``."""
    if override:
        return int(override)
    model = getattr(llm_provider, "model", "") or ""
    provider_cls = type(llm_provider).__name__
    client = None
    if provider_cls == "OpenRouterProvider":
        slugs = [model]
        client = getattr(llm_provider, "client", None)
    else:
        prefixes = _VENDOR_PREFIXES.get(provider_cls, _FALLBACK_VENDOR_PREFIXES)
        slugs = candidate_openrouter_slugs(prefixes, model)
    try:
        context_limit = discover_context_length(slugs, client=client)
    except Exception as e:
        logger.warning("Context-length discovery failed for %s: %s", model, e)
        context_limit = None
    if context_limit:
        native_cap = _NATIVE_CONTEXT_CAPS.get(provider_cls)
        return min(context_limit, native_cap) if native_cap else context_limit
    if provider_cls in _NATIVE_CONTEXT_DEFAULTS:
        return _NATIVE_CONTEXT_DEFAULTS[provider_cls]
    logger.warning(
        "Could not discover context window for model %s; assuming %d tokens "
        "(override with context_limit / --context-limit).",
        model, DEFAULT_CONTEXT_LIMIT,
    )
    return DEFAULT_CONTEXT_LIMIT


def _provider_max_output_tokens(llm_provider: LLMProvider) -> int:
    for attr in ("max_tokens", "max_output_tokens"):
        value = getattr(llm_provider, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return DEFAULT_OUTPUT_RESERVE


def _iter_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if key in _NON_CONTEXT_KEYS:
                continue
            yield from _iter_strings(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _iter_strings(value)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield str(obj)


def estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """Approximate prompt tokens for ``messages`` (all readable text, in any
    provider's message format), plus a small per-message overhead."""
    text = "\n".join(_iter_strings(messages))
    return count_tokens(text) + 4 * len(messages)


# ============================================================================
# Transcript rendering
# ============================================================================

@dataclass
class _TranscriptEntry:
    kind: str  # "user" | "assistant" | "reasoning" | "tool_call" | "tool_result"
    text: str
    label: str = ""


def _limit_middle(text: str, max_chars: Optional[int]) -> str:
    """Keep the first and last halves of ``text`` if it exceeds ``max_chars``."""
    if max_chars is None or len(text) <= max_chars:
        return text
    half = max_chars // 2
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n[... {omitted} characters omitted ...]\n{text[-half:]}"


def _compact_json_text(payload: Any) -> str:
    """Render a tool result without JSON indentation to save tokens."""
    parsed = payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return payload
    try:
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return str(payload)


def _reasoning_texts(msg: Dict[str, Any]) -> List[str]:
    texts = [
        block["thinking"]
        for block in msg.get("thinking_blocks") or []
        if isinstance(block, dict) and isinstance(block.get("thinking"), str) and block["thinking"]
    ]
    reasoning_content = msg.get("reasoning_content")
    if isinstance(reasoning_content, str) and reasoning_content:
        texts.append(reasoning_content)
    if not texts:
        # reasoning_details usually duplicates reasoning_content; only use it as a fallback.
        for detail in msg.get("reasoning_details") or []:
            if isinstance(detail, dict):
                text = detail.get("text") or detail.get("summary")
                if isinstance(text, str) and text:
                    texts.append(text)
    return texts


def _tool_call_name_args(tool_call: Dict[str, Any]) -> Tuple[str, Any]:
    func = tool_call.get("function")
    if isinstance(func, dict) and func:
        return func.get("name", ""), func.get("arguments", "{}")
    return tool_call.get("name", ""), tool_call.get("arguments", "{}")


def _parse_args(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments or "{}")
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


@dataclass
class _ToolResult:
    tool_name: str
    tool_args: Dict[str, Any]
    payload: Any


def _walk_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[List[_TranscriptEntry], List[_ToolResult]]:
    """Flatten provider-specific messages into transcript entries, and pair
    every tool result with the tool name/arguments that produced it."""
    entries: List[_TranscriptEntry] = []
    results: List[_ToolResult] = []
    calls_by_id: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    pending_calls: List[Tuple[str, Dict[str, Any]]] = []

    def add_result(call_id: Optional[str], name: Optional[str], payload: Any) -> None:
        tool_name, tool_args = "", {}
        if call_id and call_id in calls_by_id:
            tool_name, tool_args = calls_by_id[call_id]
        elif name:
            match = next((c for c in pending_calls if c[0] == name), None)
            tool_name, tool_args = match if match else (name, {})
            if match:
                pending_calls.remove(match)
        tool_name = tool_name or name or ""
        results.append(_ToolResult(tool_name, tool_args, payload))
        entries.append(_TranscriptEntry("tool_result", _compact_json_text(payload), tool_name))

    for msg in messages:
        if msg.get("type") == "function_call_output":
            add_result(msg.get("call_id"), None, msg.get("output", ""))
            continue

        role = msg.get("role")
        if role == "system":
            continue

        if role == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        add_result(block.get("tool_use_id"), block.get("name"), block.get("content", ""))
                    elif isinstance(block, dict) and isinstance(block.get("text"), str):
                        entries.append(_TranscriptEntry("user", block["text"]))
            elif content:
                entries.append(_TranscriptEntry("user", str(content)))

        elif role == "assistant":
            for text in _reasoning_texts(msg):
                entries.append(_TranscriptEntry("reasoning", text))
            content = msg.get("content")
            if isinstance(content, str) and content:
                entries.append(_TranscriptEntry("assistant", content))
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                        entries.append(_TranscriptEntry("assistant", block["text"]))
            for tool_call in msg.get("tool_calls") or []:
                name, arguments = _tool_call_name_args(tool_call)
                args = _parse_args(arguments)
                call_id = tool_call.get("id") or tool_call.get("call_id")
                if call_id:
                    calls_by_id[call_id] = (name, args)
                pending_calls.append((name, args))
                entries.append(_TranscriptEntry(
                    "tool_call", json.dumps(args, ensure_ascii=False, default=str), name
                ))

        elif role == "tool":
            add_result(msg.get("tool_call_id") or msg.get("id"), msg.get("name"), msg.get("content", ""))

        elif role == "function":
            for part in msg.get("parts") or []:
                if "function_response" in part:
                    func_response = part["function_response"]
                    add_result(None, func_response.get("name"), func_response.get("response", {}))
                elif isinstance(part.get("text"), str):
                    entries.append(_TranscriptEntry("user", part["text"]))

    return entries, results


def _render_entries(
    entries: List[_TranscriptEntry],
    tool_result_cap: Optional[int],
    omitted_steps: int = 0,
) -> str:
    reasoning_cap = tool_result_cap * 2 if tool_result_cap else None
    lines: List[str] = []
    for idx, entry in enumerate(entries):
        if idx == 1 and omitted_steps:
            lines.append(f"[... {omitted_steps} earlier steps omitted to fit the context window ...]")
        if entry.kind == "user":
            lines.append(f"[USER]\n{entry.text}")
        elif entry.kind == "assistant":
            lines.append(f"[ASSISTANT]\n{entry.text}")
        elif entry.kind == "reasoning":
            lines.append(f"[ASSISTANT REASONING]\n{_limit_middle(entry.text, reasoning_cap)}")
        elif entry.kind == "tool_call":
            lines.append(f"[TOOL CALL] {entry.label}({entry.text})")
        elif entry.kind == "tool_result":
            lines.append(f"[TOOL RESULT: {entry.label}]\n{_limit_middle(entry.text, tool_result_cap)}")
    return "\n\n".join(lines)


def _fit_transcript(entries: List[_TranscriptEntry], budget_tokens: int) -> str:
    """Render the transcript within ``budget_tokens``: first by truncating
    long tool results/reasoning, then by dropping the oldest steps (the
    first entry, the task, is always kept)."""
    if not entries:
        return ""
    for cap in _TRANSCRIPT_TOOL_RESULT_CAPS:
        text = _render_entries(entries, cap)
        if count_tokens(text) <= budget_tokens:
            return text

    tail = entries[1:]
    while tail:
        drop = max(1, len(tail) // 10)
        tail = tail[drop:]
        omitted = len(entries) - 1 - len(tail)
        text = _render_entries([entries[0]] + tail, _TRANSCRIPT_TOOL_RESULT_CAPS[-1], omitted)
        if count_tokens(text) <= budget_tokens:
            return text
    return _limit_middle(_render_entries(entries[:1], _TRANSCRIPT_TOOL_RESULT_CAPS[-1]), budget_tokens * 3)


# ============================================================================
# Source ledger
# ============================================================================

@dataclass
class _Source:
    key: str
    title: str = ""
    url: str = ""
    date: str = ""
    authors: str = ""
    tools: List[str] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    snippets: List[str] = field(default_factory=list)
    opened: bool = False
    seen: int = 0


def _normalize_url(url: str) -> str:
    url = url.strip().lower()
    url = re.sub(r"^https?://", "", url)
    url = re.sub(r"^www\.", "", url)
    url = url.split("#", 1)[0]
    return url.rstrip("/")


def _terms(text: str) -> set:
    return {
        word for word in re.findall(r"[a-z0-9]+", text.lower())
        if len(word) > 2 and word not in _STOPWORDS
    }


def _overlap(question_terms: set, text: str) -> float:
    if not question_terms:
        return 0.0
    return len(question_terms & _terms(text)) / len(question_terms)


def _best_passages(content: str, question_terms: set, k: int, max_chars: int) -> List[str]:
    """The ``k`` passages of ``content`` most relevant to the question, verbatim."""
    chunks: List[str] = []
    for paragraph in re.split(r"\n\s*\n", content):
        paragraph = " ".join(paragraph.split())
        if len(paragraph) < 40:
            continue
        for start in range(0, len(paragraph), max_chars):
            chunks.append(paragraph[start:start + max_chars])
    scored = sorted(
        ((_overlap(question_terms, chunk), idx, chunk) for idx, chunk in enumerate(chunks)),
        key=lambda item: (-item[0], item[1]),
    )
    return [chunk for score, _, chunk in scored[:k] if score > 0]


def _payload_dict(payload: Any) -> Optional[Dict[str, Any]]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


class SourceLedger:
    """Deterministic record of every source surfaced by tool results."""

    def __init__(self, question: str):
        self._question_terms = _terms(question)
        self._sources: Dict[str, _Source] = {}

    def __len__(self) -> int:
        return len(self._sources)

    def _upsert(
        self,
        key: str,
        tool_name: str,
        query: str,
        title: str = "",
        url: str = "",
        date: str = "",
        authors: str = "",
        snippets: Iterable[str] = (),
        opened: bool = False,
    ) -> None:
        source = self._sources.setdefault(key, _Source(key=key))
        source.title = source.title or (title or "").strip()
        source.url = source.url or (url or "").strip()
        source.date = source.date or (str(date) if date else "")
        source.authors = source.authors or authors
        if tool_name and tool_name not in source.tools:
            source.tools.append(tool_name)
        if query and query not in source.queries:
            source.queries.append(query)
        for snippet in snippets:
            snippet = " ".join(str(snippet).split())
            if snippet and snippet not in source.snippets:
                source.snippets.append(snippet)
        source.opened = source.opened or opened
        source.seen += 1

    def ingest(self, tool_results: List[_ToolResult]) -> None:
        for result in tool_results:
            data = _payload_dict(result.payload)
            if not data:
                continue
            query = str(result.tool_args.get("query") or result.tool_args.get("webpage_url") or "")
            try:
                self._ingest_one(result.tool_name, query, data)
            except Exception as e:
                logger.debug("Skipping unparseable tool result for ledger (%s): %s", result.tool_name, e)

    def _ingest_one(self, tool_name: str, query: str, data: Dict[str, Any]) -> None:
        if isinstance(data.get("organic"), list):
            for item in data["organic"]:
                if isinstance(item, dict) and item.get("link"):
                    self._upsert(
                        _normalize_url(item["link"]), tool_name, query,
                        title=item.get("title", ""), url=item["link"], date=item.get("date", ""),
                        snippets=[item["snippet"]] if item.get("snippet") else [],
                    )
            return

        if isinstance(data.get("data"), list):
            for item in data["data"]:
                if not isinstance(item, dict):
                    continue
                paper = item.get("paper") or {}
                snippet = item.get("snippet") or {}
                corpus_id = paper.get("corpusId")
                if not corpus_id:
                    continue
                authors = paper.get("authors") or []
                author_names = [a if isinstance(a, str) else a.get("name", "") for a in authors[:3]]
                snippet_text = snippet.get("text", "") if isinstance(snippet, dict) else str(snippet)
                self._upsert(
                    f"corpusid:{corpus_id}", tool_name, query,
                    title=paper.get("title", ""),
                    url=f"https://api.semanticscholar.org/CorpusId:{corpus_id}",
                    authors=", ".join(n for n in author_names if n) + (" et al." if len(authors) > 3 else ""),
                    snippets=[snippet_text] if snippet_text else [],
                )
            return

        if data.get("url") and "content" in data:
            content = data.get("content") or ""
            self._upsert(
                _normalize_url(data["url"]), tool_name, query,
                title=data.get("title", ""), url=data["url"], date=data.get("publishedTime", ""),
                snippets=_best_passages(
                    content, self._question_terms, _LEDGER_SNIPPETS_PER_SOURCE, _LEDGER_SNIPPET_CHARS
                ),
                opened=bool(content.strip()),
            )

    def _score(self, source: _Source, cited_text: str) -> Tuple[float, bool]:
        """(ranking score, whether the source is relevant enough to keep its snippets)."""
        overlap = _overlap(self._question_terms, " ".join([source.title] + source.snippets))
        cited = False
        if cited_text:
            url_key = _normalize_url(source.url) if source.url else ""
            cited = bool(url_key and url_key in cited_text) or (
                len(source.title) >= 20 and source.title.lower() in cited_text
            )
        score = overlap + (0.5 if source.opened else 0.0) + (1.0 if cited else 0.0)
        relevant = cited or overlap >= _LEDGER_MIN_RELEVANCE
        return score + 0.02 * min(source.seen, 5), relevant

    def render(self, cited_text: str = "", budget_tokens: int = LEDGER_TOKEN_BUDGET) -> str:
        """Sources ranked by relevance to the question (and whether the
        handoff cites or the agent opened them). Relevant top sources keep
        verbatim snippets; the rest are listed by title/URL only, and the
        lowest-ranked are counted but omitted to stay within ``budget_tokens``."""
        if not self._sources:
            return "(no sources retrieved yet)"
        cited_text = cited_text.lower()
        scored = [(source, *self._score(source, cited_text)) for source in self._sources.values()]
        scored.sort(key=lambda item: -item[1])

        lines: List[str] = []
        used = 0
        detailed_budget = int(budget_tokens * 0.75)
        omitted = 0
        for rank, (source, _, relevant) in enumerate(scored, start=1):
            header = f"[S{rank}] {source.title or '(untitled)'}"
            meta = [f"URL: {source.url}" if source.url else "", source.authors, source.date]
            flags = ", ".join(source.tools) + (" | opened" if source.opened else "")
            detail_lines = [header, "    " + " | ".join(m for m in meta if m) + f" | via {flags}"]
            for snippet in source.snippets[:_LEDGER_SNIPPETS_PER_SOURCE]:
                detail_lines.append(f'    > "{_limit_middle(snippet, _LEDGER_SNIPPET_CHARS)}"')
            detailed = "\n".join(detail_lines)
            brief = f"[S{rank}] {source.title or '(untitled)'} — {source.url}"

            detailed_tokens = count_tokens(detailed)
            if relevant and used + detailed_tokens <= detailed_budget:
                lines.append(detailed)
                used += detailed_tokens
                continue
            brief_tokens = count_tokens(brief)
            if used + brief_tokens <= budget_tokens:
                lines.append(brief)
                used += brief_tokens
            else:
                omitted += 1
        if omitted:
            lines.append(f"[... {omitted} lower-relevance sources omitted ...]")
        return "\n".join(lines)


# ============================================================================
# Prompts
# ============================================================================

_SUMMARY_PROMPT = """You are a research assistant partway through a long evidence-gathering session for a scientific question. Your context window is nearly full, so you are about to hand off your work to another research assistant who will continue the research and write the final conclusion. The other assistant will NOT see the research transcript, only your handoff, so anything you omit is lost.

**Original Question:**
{question}

**Research transcript so far** (your reasoning, tool calls, and tool results; long results may be truncated):
<transcript>
{transcript}
</transcript>

**Source ledger** (extracted automatically from all tool results so far, ranked by relevance to the question):
<source_ledger>
{ledger}
</source_ledger>

This is an intermediate handoff, NOT the final answer: do NOT write the final conclusion and do NOT use triple square brackets. Using ONLY information that appears in the transcript (never invent studies, numbers, quotes, or citations), write a detailed handoff with these sections:

1. **Research Actions So Far** - Each significant search or page fetch (tool and query/URL), what it returned, and whether it was useful. List dead ends and queries that returned nothing relevant so they are not repeated.
2. **Relevant Evidence Gathered** - For EVERY source relevant to the question (skip irrelevant ones): title, authors/year if known, exact URL copied from the transcript, study type (e.g., systematic review, meta-analysis, RCT, cohort, guideline), population, intervention/exposure and comparator, outcomes (benefits and harms), sample sizes, effect estimates with confidence intervals or p-values, and key findings quoted verbatim in quotation marks. Order sources from most to least relevant, preferring higher-quality evidence.
3. **Evidence Quality Assessment** - Strengths and limitations of the evidence so far: risk of bias, imprecision, inconsistency across sources, indirectness, publication bias, recency, and overall certainty.
4. **Synthesis So Far** - How the sources relate: where they agree, where they conflict, and the tentative direction of the answer to the question.
5. **Remaining Gaps and Next Steps** - What evidence is still missing to answer the question well (e.g., specific outcomes, populations, comparators, harms, higher-quality or more recent studies) and concrete searches or URLs to pursue next.

Be comprehensive and precise. The next assistant must be able to write a well-cited, evidence-backed conclusion from your handoff without re-running searches you already did."""

_QUESTIONS_PROMPT = """You are a research assistant picking up an evidence-gathering task from a previous research assistant who ran out of context. After this step you will continue the research on your own and ultimately write a comprehensive, well-cited, evidence-backed conclusion to the original question.

**Original Question:**
{question}

**Handoff Summary from the Previous Assistant:**
{summary}

**Source Ledger** (extracted automatically from all tool results so far, ranked by relevance to the question):
{ledger}

Before continuing, ask several questions (at least five, more if necessary) about information that is missing or unclear in the handoff and that you would need to write the final conclusion: for example exact effect sizes and confidence intervals, study designs and sample sizes, which source supports which claim, conflicting findings, evidence quality and certainty, verbatim wording of key findings, or which searches were already tried. The previous assistant can answer from the full research history. After you ask these questions you will be on your own, so ask everything you need to know. Only ask questions in this step; do not call tools and do not write the final conclusion."""

_ANSWERS_PROMPT = """You are the research assistant who conducted the research below on this question:
{question}

**Research transcript** (your reasoning, tool calls, and tool results; long results may be truncated):
<transcript>
{transcript}
</transcript>

**Your handoff summary:**
<handoff_summary>
{summary}
</handoff_summary>

The next research assistant has a few questions for you. Answer each of them one by one in detail, using ONLY information from the transcript. Quote tool results verbatim where possible and give the exact source title and URL for every factual claim. If the transcript does not contain the answer, say so explicitly rather than guessing. Do not write the final conclusion and do not use triple square brackets.

**Questions:**
{questions}"""

_SHORT_SUMMARY_PROMPT = """You are handing off an evidence-gathering task on this scientific question to another research assistant:
{question}

**Research transcript** (heavily truncated):
<transcript>
{transcript}
</transcript>

Using ONLY information from the transcript, briefly summarize (at most about 500 words): the most relevant evidence found so far with exact source titles and URLs and key numbers quoted verbatim, the quality of that evidence, searches already tried, and what evidence is still needed. Do not write the final conclusion and do not use triple square brackets."""


def _sanitize(text: str) -> str:
    """Handoff text must not look like a final ``[[[...]]]`` conclusion."""
    return text.replace("[[[", "[").replace("]]]", "]").strip()


# ============================================================================
# Compactor
# ============================================================================

class ContextCompactor:
    """Compacts one query's tool-calling history when it nears the context window."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        original_query: str,
        context_limit: int,
        free_tokens_threshold: int = DEFAULT_FREE_TOKENS_THRESHOLD,
        max_compactions: int = DEFAULT_MAX_COMPACTIONS,
        static_prompt_tokens: int = 0,
    ):
        """
        Args:
            llm_provider: Provider used for the summarization calls (same model as the agent).
            original_query: The user's question; restated verbatim in every handoff.
            context_limit: Model context window in tokens (see ``resolve_context_limit``).
            free_tokens_threshold: Compact proactively when estimated free tokens drop below this.
            max_compactions: Upper bound on compactions per query (guards against runaway loops).
            static_prompt_tokens: Tokens sent on every call outside ``messages`` (tool definitions).
        """
        self.llm_provider = llm_provider
        self.original_query = original_query
        self.context_limit = context_limit
        self.free_tokens_threshold = free_tokens_threshold
        self.max_compactions = max_compactions
        self._static_tokens = static_prompt_tokens + count_tokens(RESEARCH_ASSISTANT_PROMPT)
        self._output_reserve = max(
            _provider_max_output_tokens(llm_provider),
            min(COMPACTION_MAX_OUTPUT_TOKENS, context_limit // 4),
        )
        self._ledger_budget = min(LEDGER_TOKEN_BUDGET, context_limit // 20)
        self.ledger = SourceLedger(original_query)

        self.compaction_count = 0
        self._final = False
        self._messages_len_after_compaction = 1
        self._llm_calls = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._records: List[Dict[str, Any]] = []

    # ── triggering ───────────────────────────────────────────────────────────

    def estimate_prompt_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Conservative (safety-factor-scaled) prompt size for ``messages``."""
        return int((estimate_messages_tokens(messages) + self._static_tokens) * TOKEN_ESTIMATE_SAFETY_FACTOR)

    def can_compact(self) -> bool:
        return self.compaction_count < self.max_compactions

    def _near_limit(self, messages: List[Dict[str, Any]], reported_prompt_tokens: int) -> bool:
        if len(messages) <= self._messages_len_after_compaction:
            return False
        used = max(self.estimate_prompt_tokens(messages), reported_prompt_tokens)
        free_tokens = self.context_limit - used
        if free_tokens < self.free_tokens_threshold:
            logger.info(
                "Context nearly full: ~%d free tokens (context %d, used ~%d) below threshold %d",
                free_tokens, self.context_limit, used, self.free_tokens_threshold,
            )
            return True
        return False

    def should_compact(self, messages: List[Dict[str, Any]], reported_prompt_tokens: int = 0) -> bool:
        """Proactive trigger. ``reported_prompt_tokens`` is the provider's own
        prompt-token count for the latest call, when available."""
        return self.can_compact() and self._near_limit(messages, reported_prompt_tokens)

    def should_force_final_answer(self, messages: List[Dict[str, Any]], reported_prompt_tokens: int = 0) -> bool:
        """The context is nearly full but ``max_compactions`` is exhausted."""
        return not self.can_compact() and self._near_limit(messages, reported_prompt_tokens)

    # ── compaction ───────────────────────────────────────────────────────────

    async def compact(
        self, messages: List[Dict[str, Any]], reason: str, final: bool = False
    ) -> List[Dict[str, Any]]:
        """Return a replacement history: [task + summary + ledger, questions, answers handoff].

        With ``final=True`` (used once ``max_compactions`` is exhausted), the
        handoff tells the model to write the final answer without tools; the
        caller must then query the provider with ``tools=None``."""
        self.compaction_count += 1
        self._final = final
        entries, tool_results = _walk_messages(messages)
        self.ledger.ingest(tool_results)
        record: Dict[str, Any] = {
            "index": self.compaction_count,
            "reason": reason,
            "final": final,
            "timestamp": datetime.now().isoformat(),
            "num_messages_before": len(messages),
            "estimated_tokens_before": self.estimate_prompt_tokens(messages),
            "context_limit": self.context_limit,
            "ledger_sources": len(self.ledger),
        }
        logger.info(
            "Compaction %d (%s): %d messages, ~%d tokens, %d ledger sources",
            self.compaction_count, reason, len(messages),
            record["estimated_tokens_before"], len(self.ledger),
        )

        try:
            new_messages = await self._three_step_handoff(entries, record)
            record["method"] = "three_step"
        except Exception as e:
            logger.warning("Compaction %d: full summary failed: %s", self.compaction_count, e)
            record["three_step_error"] = str(e)
            try:
                new_messages = await self._short_summary_handoff(entries, record)
                record["method"] = "short_summary"
            except Exception as e2:
                logger.error("Compaction %d: short summary failed: %s", self.compaction_count, e2)
                record["short_summary_error"] = str(e2)
                new_messages = self._no_llm_handoff(entries)
                record["method"] = "no_llm_fallback"

        record["num_messages_after"] = len(new_messages)
        record["estimated_tokens_after"] = self.estimate_prompt_tokens(new_messages)
        record["messages_before"] = messages
        record["handoff_messages"] = new_messages
        self._records.append(record)
        self._save_record(record)
        self._messages_len_after_compaction = len(new_messages)
        logger.info(
            "Compaction %d done via %s: ~%d -> ~%d tokens",
            self.compaction_count, record["method"],
            record["estimated_tokens_before"], record["estimated_tokens_after"],
        )
        return new_messages

    async def _three_step_handoff(
        self, entries: List[_TranscriptEntry], record: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        question = self.original_query

        # Step 1: summary (full transcript).
        pre_ledger = self.ledger.render(budget_tokens=self._ledger_budget)
        summary = await self._call_with_transcript(
            entries,
            lambda transcript: _SUMMARY_PROMPT.format(
                question=question, transcript=transcript, ledger=pre_ledger
            ),
        )
        record["summary"] = summary

        # Step 2: questions (fresh context: question + summary + ledger only).
        ledger = self.ledger.render(cited_text=summary, budget_tokens=self._ledger_budget)
        question_prompt = _QUESTIONS_PROMPT.format(question=question, summary=summary, ledger=ledger)
        model_questions = await self._call(question_prompt)
        record["questions"] = model_questions

        # Step 3: answers (full transcript + summary).
        answers = await self._call_with_transcript(
            entries,
            lambda transcript: _ANSWERS_PROMPT.format(
                question=question, transcript=transcript, summary=summary, questions=model_questions
            ),
        )
        record["answers"] = answers

        handoff = (
            "Here are the answers the previous research assistant provided.\n\n"
            f"{answers}\n\n"
            + self._next_step_instruction()
        )
        return [
            {"role": "user", "content": question_prompt},
            {"role": "assistant", "content": model_questions},
            {"role": "user", "content": handoff},
        ]

    async def _short_summary_handoff(
        self, entries: List[_TranscriptEntry], record: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        summary = await self._call_with_transcript(
            entries,
            lambda transcript: _SHORT_SUMMARY_PROMPT.format(question=self.original_query, transcript=transcript),
            budget_fraction=0.5,
        )
        record["summary"] = summary
        return [{"role": "user", "content": self._standalone_handoff(summary, self.ledger.render(summary, self._ledger_budget))}]

    def _no_llm_handoff(self, entries: List[_TranscriptEntry]) -> List[Dict[str, Any]]:
        recent_reasoning = [e.text for e in entries if e.kind in ("reasoning", "assistant")][-3:]
        notes = _limit_middle("\n\n".join(recent_reasoning), 6000) or "(none)"
        return [{
            "role": "user",
            "content": self._standalone_handoff(
                f"(Automatic summary unavailable.) Most recent reasoning from the previous assistant:\n{_sanitize(notes)}",
                self.ledger.render(budget_tokens=self._ledger_budget),
            ),
        }]

    def _standalone_handoff(self, summary: str, ledger: str) -> str:
        return (
            f"**Original Question:**\n{self.original_query}\n\n"
            "You are continuing an evidence-gathering task from a previous research assistant who "
            "ran out of context.\n\n"
            f"**Handoff Summary from the Previous Assistant:**\n{summary}\n\n"
            "**Source Ledger** (extracted automatically from all tool results so far, ranked by "
            f"relevance to the question):\n{ledger}\n\n"
            + self._next_step_instruction()
        )

    def _next_step_instruction(self) -> str:
        if self._final:
            return _FINAL_ANSWER_INSTRUCTION + _HANDOFF_FORMAT_REMINDER
        return _CONTINUE_RESEARCH_INSTRUCTION + _HANDOFF_FORMAT_REMINDER

    # ── LLM calls ────────────────────────────────────────────────────────────

    @contextmanager
    def _compaction_output_limit(self) -> Iterator[None]:
        """Temporarily raise the provider's per-call output cap to ``_output_reserve``."""
        saved: Dict[str, int] = {}
        for attr in _OUTPUT_TOKEN_ATTRS:
            value = getattr(self.llm_provider, attr, None)
            if isinstance(value, int) and value < self._output_reserve:
                saved[attr] = value
                setattr(self.llm_provider, attr, self._output_reserve)
        try:
            yield
        finally:
            for attr, value in saved.items():
                setattr(self.llm_provider, attr, value)

    async def _call(self, prompt: str) -> str:
        with self._compaction_output_limit():
            _, text_content, _, _ = await self.llm_provider.call_llm(
                [{"role": "user", "content": prompt}], tools=None
            )
        self._llm_calls += 1
        self._input_tokens += count_tokens(prompt) + self._static_tokens
        self._output_tokens += count_tokens(text_content or "")
        if not text_content or not text_content.strip():
            raise RuntimeError("Compaction LLM call returned no text")
        return _sanitize(text_content)

    async def _call_with_transcript(
        self,
        entries: List[_TranscriptEntry],
        build_prompt: Callable[[str], str],
        budget_fraction: float = 1.0,
    ) -> str:
        """Call the LLM with a transcript sized to fit the context window,
        halving the budget if the provider still reports an overflow."""
        usable = self.context_limit - self._output_reserve - _TRANSCRIPT_SAFETY_MARGIN
        overhead = count_tokens(build_prompt("")) + self._static_tokens
        budget = int((usable / TOKEN_ESTIMATE_SAFETY_FACTOR - overhead) * budget_fraction)
        for _ in range(3):
            if budget <= 0:
                break
            transcript = _fit_transcript(entries, budget)
            try:
                return await self._call(build_prompt(transcript))
            except ContextLengthExceededError:
                budget = min(budget, count_tokens(transcript)) // 2
                logger.warning("Compaction call exceeded context; retrying with transcript budget %d", budget)
        raise ContextLengthExceededError("Compaction transcript could not fit the context window")

    # ── reporting ────────────────────────────────────────────────────────────

    def usage_summary(self) -> Dict[str, Any]:
        """Stats for ``token_usage`` (token counts are tiktoken estimates)."""
        return {
            "compaction_count": self.compaction_count,
            "compaction_llm_calls": self._llm_calls,
            "compaction_input_tokens": self._input_tokens,
            "compaction_output_tokens": self._output_tokens,
            "compactions": [
                {
                    key: record.get(key)
                    for key in (
                        "index", "reason", "final", "method", "estimated_tokens_before",
                        "estimated_tokens_after", "ledger_sources",
                    )
                }
                for record in self._records
            ],
        }

    def _save_record(self, record: Dict[str, Any]) -> None:
        log_file = get_log_file_path()
        if not log_file:
            return
        try:
            with open(log_file.parent / "compactions.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as e:
            logger.warning("Could not save compaction record: %s", e)
