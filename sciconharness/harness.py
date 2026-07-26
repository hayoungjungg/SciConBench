"""
SciConHarness — top-level Python API.

Use this instead of the CLI when you want to drive queries programmatically.

Example — single query::

    import asyncio
    from sciconharness import SciConHarness

    harness = SciConHarness(
        provider="openai",
        model="gpt-5.1",
        enable_tools=True,
        enable_filtering=True,
        cochrane_titles=["Oral antibiotics for otitis media in children", ...],  
        temperature=None,        # model default
        max_tool_calls=30,
    )

    async def main():
        async with harness:
            response, usage = await harness.query(
                question="What are the benefits and harms of oral antibiotics for otitis media?",
                doi="10.1002/14651858.CD015254.pub2",
                publication_date="23 October 2023",
            )
            print(response)

    asyncio.run(main())

Example — batch query::

    import asyncio, json
    from sciconharness import SciConHarness

    harness = SciConHarness(provider="openai", model="gpt-5.1", enable_tools=True, enable_filtering=True)

    doi_to_question = json.load(open("doi_questions.json"))
    doi_to_date    = json.load(open("doi_dates.json"))

    async def main():
        async with harness:
            results = await harness.query_batch(doi_to_question, doi_to_date)

    asyncio.run(main())

Example — interactive chat (non-deep-research models only)::

    async def main():
        async with harness:
            await harness.chat_loop()   # reads from stdin interactively
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import dotenv

logger = logging.getLogger(__name__)

# Load .env from project root (SciConBench/.env) relative to this file
dotenv.load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Accumulated batch log directory (mirrors query_batch.py behaviour)
_batch_log_dir = Path(__file__).parent / "logs"
_batch_log_dir.mkdir(exist_ok=True)


class SciConHarness:
    """
    Programmatic interface to SciConHarness.

    All configuration is provided at construction time.  After construction,
    call ``query()`` for a single question or ``query_batch()`` for many.
    Use as an async context manager to manage the MCP server connection
    automatically, or call ``connect()`` / ``disconnect()`` yourself.

    Parameters
    ----------
    provider : str
        LLM backend: ``"openai"`` | ``"claude"`` | ``"gemini"`` | ``"perplexity"``
        | ``"azure"`` | ``"openrouter"``.
    model : str, optional
        Model name.  Defaults: openai→``gpt-5.1``, claude→``claude-sonnet-4-5``,
        gemini→``gemini-3-pro-preview``, perplexity→``sonar-reasoning-pro``,
        azure→``DeepSeek-V4-Pro``, openrouter→``moonshotai/kimi-k3``.
        Other OpenRouter models used in this project: ``z-ai/glm-5.2``
        (GLM-5.2), ``qwen/qwen3.5-9b`` (Qwen3.5-9B).
    api_key : str, optional
        Override the provider API key (otherwise read from env).
    base_url : str, optional
        Azure OpenAI / Foundry endpoint.  Required for ``provider="azure"``;
        for ``provider="openai"`` / ``"claude"`` activates Azure mode when set.
        Ignored for ``provider="openrouter"`` (always ``https://openrouter.ai/api/v1``
        unless ``OPENROUTER_BASE_URL`` is set).
    api_version : str, optional
        Azure API version (default ``2025-04-01-preview``). Classic AzureOpenAI
        path only; Foundry ``/openai/v1/`` endpoints do not need it.

    enable_tools : bool
        Enable MCP tool calling (search + browse).  Default ``True``.
        Perplexity (``sonar-deep-research``), OpenAI deep research
        (``o3-deep-research``, ``o4-mini-deep-research``), and similar
        built-in-search models ignore this and always use their own search.
    enable_filtering : bool
        Filter results that post-date the review's publication date or that
        come from the review itself.  Default ``True``.

    cochrane_titles : list[str] or Path, optional
        Cochrane review titles used for title-based source filtering.
        Pass either a Python list of title strings or a ``Path`` to a JSON
        file that contains a list of strings.
    doi_to_title : dict[str, str], optional
        Mapping of DOI → source title.  Used to automatically derive the
        ``source_title`` filter for each DOI without you having to pass it
        explicitly on every call.  Falls back to the bundled
        ``data/review_articles/data.json`` if not provided.

    temperature : float, optional
        Sampling temperature.  ``None`` uses the model's default.
    max_tokens : int, optional
        Max output tokens.  ``None`` uses the provider default.
    max_tool_calls : int
        Hard cap on tool calls per query for [o3, o4-mini]-deep-research models.  Default 30.
    max_format_retries : int
        How many total attempts to make if the response is missing
        ``[[[...]]]``.  Default 3 (matches ``query_batch.py``).
    min_conclusion_length : int
        Minimum character count inside ``[[[...]]]`` to be considered
        well-formatted.  Default 20.

    save_results : bool
        Whether to save the query result to a ``result.json`` file under ``logs/<model>_[tools]_[filter]/<doi>/``.
        Default ``True``.
    log_dir : Path, optional
        Override the base log/output directory.
    """

    def __init__(
        self,
        *,
        # Provider
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        # Feature flags
        enable_tools: bool = True,
        enable_filtering: bool = True,
        # Filter data — accept list directly or a path to a JSON file
        cochrane_titles: Optional[Union[List[str], Path]] = None,
        doi_to_title: Optional[Dict[str, str]] = None,
        # LLM hyperparameters
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_tool_calls: int = 30,
        # Batch-specific
        max_format_retries: int = 3,
        min_conclusion_length: int = 20,
        # Output
        save_results: bool = True,
        log_dir: Optional[Path] = None,
    ):
        from sciconharness.utils.query_utils import (
            create_provider,
            load_cochrane_titles,
            load_doi_to_title_mapping,
        )
        from sciconharness.mcp_client import MCPClient

        self.provider_name = provider.lower()
        self.enable_tools = enable_tools
        self.enable_filtering = enable_filtering
        self.save_results = save_results
        self.max_tool_calls = max_tool_calls
        self.max_format_retries = max_format_retries
        self.min_conclusion_length = min_conclusion_length
        self._log_dir_override = Path(log_dir) if log_dir else None
        self._connected = False
        self._batch_file_handler: Optional[logging.FileHandler] = None

        # ── resolve model default ────────────────────────────────────────────
        import os
        if model is None:
            model = {
                "claude": "claude-sonnet-4-5",
                "gemini": "gemini-3-pro-preview",
                "perplexity": "sonar-reasoning-pro",
                "azure": "DeepSeek-V4-Pro",
                "openrouter": "moonshotai/kimi-k3",
            }.get(self.provider_name, os.getenv("OPENAI_MODEL", "gpt-5.1"))
        self.model = model

        # ── is this a deep-research model? ──────────────────────────────────
        self._is_deep_research = any(
            tag in self.model.lower()
            for tag in ("o4-mini-deep-research", "o3-deep-research")
        )
        if self._is_deep_research:
            self.enable_tools = True  # always on for deep-research

        # ── Perplexity always uses its own search ────────────────────────────
        if self.provider_name == "perplexity":
            self.enable_tools = False

        # ── build LLM provider ───────────────────────────────────────────────
        self._llm_provider = create_provider(
            provider_name=self.provider_name,
            model=self.model,
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # ── build MCP client (not needed for deep-research) ──────────────────
        if not self._is_deep_research:
            self._mcp_client = MCPClient(
                self._llm_provider,
                enable_tool_calling=self.enable_tools,
                enable_filtering=self.enable_filtering,
            )
        else:
            self._mcp_client = None

        # ── cochrane titles ──────────────────────────────────────────────────
        if isinstance(cochrane_titles, (str, Path)):
            self.cochrane_titles: List[str] = load_cochrane_titles(Path(cochrane_titles))
            self._cochrane_titles_file: Optional[Path] = Path(cochrane_titles)
        elif isinstance(cochrane_titles, list):
            self.cochrane_titles = cochrane_titles
            self._cochrane_titles_file = None
        else:
            self.cochrane_titles = []
            self._cochrane_titles_file = None

        # ── DOI → title mapping (only needed when filtering is enabled) ─────────
        if enable_filtering:
            self._doi_to_title: Dict[str, str] = (
                doi_to_title if doi_to_title is not None else load_doi_to_title_mapping()
            )
        else:
            self._doi_to_title = doi_to_title or {}

    # ── async context manager ────────────────────────────────────────────────

    async def __aenter__(self) -> "SciConHarness":
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.disconnect()

    async def connect(self) -> None:
        """Start the MCP server subprocess (no-op for deep-research models)."""
        if self._mcp_client and not self._connected:
            await self._mcp_client.connect_to_server()
            self._connected = True

    async def disconnect(self) -> None:
        """Stop the MCP server subprocess and clean up logging handlers."""
        if self._mcp_client and self._connected:
            await self._mcp_client.disconnect()
            self._connected = False
        # Clean up batch log file handler (mirrors BatchQueryRunner.disconnect)
        if self._batch_file_handler:
            logger.removeHandler(self._batch_file_handler)
            self._batch_file_handler.close()
            self._batch_file_handler = None

    # ── private helpers ───────────────────────────────────────────────────────

    def _source_title_for(self, doi: Optional[str]) -> Optional[str]:
        if not doi:
            return None
        from sciconharness.utils.query_utils import get_title_for_doi
        return get_title_for_doi(doi, self._doi_to_title)

    def _make_filter(
        self,
        doi: Optional[str],
        publication_date: Optional[str],
        source_title: Optional[str] = None,
    ):
        from sciconharness.mcp_client.filters import CochraneResultFilter
        if not (self.cochrane_titles or publication_date or source_title):
            return None
        resolved_title = source_title or self._source_title_for(doi)
        return CochraneResultFilter(
            title_filter_list=self.cochrane_titles or None,
            source_title=resolved_title,
            publication_date=publication_date,
        )

    def _setup_dirs(self, doi: Optional[str]) -> Tuple[Path, Optional[Path]]:
        from sciconharness.utils.query_utils import sanitize_doi_for_path, setup_directories
        from sciconharness.mcp_client.utils.utils import set_custom_log_dir

        # Apply the override before setup_directories reads the global _custom_log_dir.
        if self._log_dir_override:
            set_custom_log_dir(self._log_dir_override)

        doi_safe = sanitize_doi_for_path(doi) if doi else "no_doi"
        log_dir, data_dir = setup_directories(
            model=self.model,
            doi=doi_safe,
            save_results=self.save_results,
            enable_tool_calling=self.enable_tools,
            enable_filtering=self.enable_filtering,
        )
        return log_dir, data_dir

    def _save(
        self,
        *,
        data_dir: Optional[Path],
        question: str,
        response: str,
        token_usage: Dict[str, Any],
        doi: Optional[str],
        publication_date: Optional[str],
        result_filter: Any,
        max_retries: int = 3,
    ) -> Optional[Path]:
        """Save result.json with up to max_retries attempts (mirrors _save_query_result)."""
        from sciconharness.utils.query_utils import save_query_result, verify_result_file_exists

        if not self.save_results or not data_dir:
            return None

        result_file = None
        for attempt in range(max_retries):
            try:
                result_file = save_query_result(
                    data_dir=data_dir,
                    query=question,
                    response=response,
                    token_usage=token_usage,
                    provider_name=self.provider_name,
                    model=self.model,
                    doi=doi,
                    provider=self._llm_provider,
                    client=self._mcp_client,
                    publication_date=publication_date,
                    filename="result.json",
                    result_filter=result_filter,
                )
                if verify_result_file_exists(result_file):
                    return result_file
                logger.warning(
                    "Result file verification failed (attempt %d/%d): %s",
                    attempt + 1, max_retries, result_file,
                )
            except Exception as e:
                logger.warning(
                    "Error saving result (attempt %d/%d): %s",
                    attempt + 1, max_retries, e,
                )

        logger.error("Failed to save result file after %d attempts", max_retries)
        return None

    def _is_well_formatted(self, response: str) -> Tuple[bool, str]:
        """Return (ok, reason) — mirrors BatchQueryRunner._is_response_well_formatted."""
        if not response or not isinstance(response, str):
            return False, "Response is empty or not a string"
        matches = re.findall(r'\[\[\[(.*?)\]\]\]', response, re.DOTALL)
        if not matches:
            return False, "No triple square brackets [[[...]]] found in response"
        for content in matches:
            if len(content.strip()) >= self.min_conclusion_length:
                return True, f"Found well-formatted conclusion ({len(content.strip())} chars)"
        max_len = max(len(m.strip()) for m in matches)
        return False, (
            f"Triple bracket content is too short "
            f"({max_len} chars, minimum {self.min_conclusion_length} required)"
        )

    def _is_doi_already_processed(self, doi: str) -> bool:
        """
        Return True if result.json already exists for this DOI.
        Allows resuming interrupted batch runs (mirrors _is_doi_already_processed).
        """
        if not self.save_results:
            return False

        from sciconharness.utils.query_utils import sanitize_doi_for_path, verify_result_file_exists
        from sciconharness.mcp_client.utils import utils as mcp_utils

        doi_safe = sanitize_doi_for_path(doi)
        model_safe = self.model.replace("/", "_")
        parts = [model_safe]
        if self.enable_tools:
            parts.append("tools")
        if self.enable_filtering:
            parts.append("filter")
        model_dir = "_".join(parts)

        base = mcp_utils._custom_log_dir if mcp_utils._custom_log_dir else mcp_utils._log_dir
        result_file = base / model_dir / doi_safe / "result.json"
        return result_file.exists() and verify_result_file_exists(result_file)

    def _setup_batch_logging(self) -> None:
        """Set up a per-batch accumulated log file (mirrors _setup_query_batch_logging)."""
        if self._batch_file_handler:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe = self.model.replace("/", "_")
        log_file = _batch_log_dir / f"query_batch_{model_safe}_{timestamp}.log"
        self._batch_file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        self._batch_file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(self._batch_file_handler)
        logger.setLevel(logging.INFO)

    @staticmethod
    def _extract_token_usage(response_obj: Any, query: str, response_text: str) -> Dict[str, Any]:
        """
        Extract token usage from a provider response object.
        Falls back to direct attribute lookup, then to a word-count estimate.
        Mirrors the token extraction in query_single.py / query_batch.py.
        """
        token_usage = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

        if hasattr(response_obj, "usage") and response_obj.usage:
            u = response_obj.usage
            token_usage = {
                "total_tokens": getattr(u, "total_tokens", 0) or 0,
                "prompt_tokens": getattr(u, "input_tokens", 0) or getattr(u, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(u, "output_tokens", 0) or getattr(u, "completion_tokens", 0) or 0,
            }
        elif hasattr(response_obj, "total_tokens"):
            token_usage = {
                "total_tokens": getattr(response_obj, "total_tokens", 0) or 0,
                "prompt_tokens": getattr(response_obj, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(response_obj, "completion_tokens", 0) or 0,
            }

        if token_usage["total_tokens"] == 0:
            logger.warning("Token usage not available from API response, using estimate")
            word_count = len(response_text.split())
            token_usage = {
                "total_tokens": int(round(word_count * 1.3)),
                "prompt_tokens": int(round(len(query.split()) * 1.3)),
                "completion_tokens": int(round(word_count * 1.3)),
            }
        else:
            token_usage = {k: int(v) for k, v in token_usage.items()}

        return token_usage

    # ── public API ────────────────────────────────────────────────────────────

    async def query(
        self,
        question: str,
        *,
        doi: Optional[str] = None,
        publication_date: Optional[str] = None,
        source_title: Optional[str] = None,
        domain_filter: Optional[List[str]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Run a query and return ``(response_text, token_usage)``.

        Retries up to ``max_format_retries`` times if the response is missing
        the required ``[[[...]]]`` conclusion marker.  Can be called directly
        for a single question or is called per-DOI by ``query_batch()``.

        Parameters
        ----------
        question : str
            The research question to answer.
        doi : str, optional
            DOI of the Cochrane review. Used to name the output directory and
            derive the source title for filtering.
        publication_date : str, optional
            Publication date of the review (e.g. ``"23 October 2023"``).
            Results published after this date are filtered out.
        source_title : str, optional
            Explicit source title override. When omitted the title is looked
            up from ``doi_to_title``.
        domain_filter : list[str], optional
            Perplexity-only: pre-resolved domain allow/deny list. Callers are
            responsible for loading and processing any source files before
            passing this in (e.g. via ``build_perplexity_domain_filter()``).
        """
        if not self._connected:
            await self.connect()

        log_dir, data_dir = self._setup_dirs(doi)
        logger.info("Log directory: %s", log_dir)
        if self.save_results and data_dir:
            logger.info("Results will be saved to: %s", data_dir)

        result_filter = self._make_filter(doi, publication_date, source_title)

        # ── Resolve Perplexity domain filter ─────────────────────────────────
        df: Optional[List[str]] = domain_filter
        if self.provider_name == "perplexity":
            # Merge with filtered_links.json left by a previous run
            if data_dir:
                filtered_links_file = data_dir / "filtered_links.json"
                if filtered_links_file.exists():
                    from sciconharness.utils.perplexity_filtering import (
                        merge_domain_filter_with_filtered_links,
                    )
                    df = merge_domain_filter_with_filtered_links(
                        domain_filter=df,
                        filtered_links_path=filtered_links_file,
                    )
                    logger.info(
                        "Merged domain filter with filtered_links from %s", filtered_links_file
                    )

            if df:
                logger.info("=" * 80)
                logger.info("Domain filter for DOI: %s", doi)
                logger.info("Total domain filter entries: %d", len(df))
                for i, d in enumerate(df, 1):
                    logger.info("  %2d. %s", i, d)
                logger.info("=" * 80)
            else:
                logger.info("No domain filter for DOI: %s", doi)

        # ── Query with format retries (covers both deep-research and MCPClient) ─
        response: Optional[str] = None
        token_usage: Optional[Dict[str, Any]] = None
        format_error: Optional[str] = None

        for attempt in range(self.max_format_retries):
            try:
                if self._is_deep_research:
                    # Pass data_dir=None to defer saving until after all retries
                    response, token_usage = await self._query_deep_research(
                        question=question,
                        doi=doi,
                        publication_date=publication_date,
                        source_title=source_title,
                        log_dir=log_dir,
                        data_dir=None,
                    )
                else:
                    response, token_usage = await self._mcp_client.process_query(
                        question,
                        conversation_history=None,
                        result_filter=result_filter,
                        return_token_usage=True,
                        domain_filter=df,
                    )

                ok, reason = self._is_well_formatted(response)
                if ok:
                    logger.info("Response is well-formatted: %s", reason)
                    break
                logger.warning(
                    "Response not well-formatted for %s (attempt %d/%d): %s",
                    doi, attempt + 1, self.max_format_retries, reason,
                )
                if attempt < self.max_format_retries - 1:
                    logger.info("Retrying query for %s...", doi)
                else:
                    format_error = (
                        f"Response not well-formatted after "
                        f"{self.max_format_retries} attempts: {reason}"
                    )
                    logger.error(
                        "Response still not well-formatted after %d attempts for %s: %s",
                        self.max_format_retries, doi, reason,
                    )

            except Exception as e:
                logger.error(
                    "Error processing query for %s (attempt %d/%d): %s",
                    doi, attempt + 1, self.max_format_retries, e, exc_info=True,
                )
                if attempt < self.max_format_retries - 1:
                    logger.info("Retrying query for %s...", doi)
                else:
                    raise  # propagate on final attempt so query_batch can capture it

        # ── Save once after all retries ───────────────────────────────────────
        if response:
            result_file = self._save(
                data_dir=data_dir,
                question=question,
                response=response,
                token_usage=token_usage or {},
                doi=doi,
                publication_date=publication_date,
                result_filter=result_filter,
            )
            if self.save_results:
                from sciconharness.utils.query_utils import verify_result_file_exists
                if result_file and verify_result_file_exists(result_file):
                    ok, reason = self._is_well_formatted(response)
                    status = "✓" if ok else "⚠"
                    logger.info(
                        "%s Completed: %s (%d characters, %d tokens)",
                        status, doi, len(response), (token_usage or {}).get("total_tokens", 0),
                    )
                    logger.info("Result saved to: %s", result_file)
                    if not ok:
                        logger.warning("Response saved but not well-formatted: %s", reason)
                else:
                    logger.error("Result file verification failed for %s", doi)

        if format_error:
            logger.warning("query() completed with formatting issue: %s", format_error)

        return response or "", token_usage or {}

    async def chat_loop(
        self,
        *,
        publication_date: Optional[str] = None,
    ) -> None:
        """
        Run an interactive terminal chat session.

        Only supported for non-deep-research models.  Connects automatically
        if not already connected.

        Parameters
        ----------
        publication_date : str, optional
            Date cutoff applied to search results during the session.
        """
        if self._is_deep_research:
            raise ValueError(
                "chat_loop() is not supported for deep-research models. Use query() instead."
            )
        from sciconharness.mcp_client.filters import CochraneResultFilter

        if not self._connected:
            await self.connect()

        base_filter = None
        if self.cochrane_titles:
            base_filter = CochraneResultFilter(title_filter_list=self.cochrane_titles)

        await self._mcp_client.chat_loop(
            result_filter=base_filter,
            interactive=True,
            query=None,
            publication_date=publication_date,
        )

    async def query_batch(
        self,
        doi_to_question: Dict[str, str],
        doi_to_date: Optional[Dict[str, str]] = None,
        *,
        max_samples: Optional[int] = None,
        doi_to_domain_filter: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run queries for every DOI in *doi_to_question*.

        Already-processed DOIs (those with a valid ``result.json``) are skipped
        automatically so interrupted runs can be resumed safely.

        Parameters
        ----------
        doi_to_question : dict[str, str]
            Mapping ``doi → question``.
        doi_to_date : dict[str, str], optional
            Mapping ``doi → publication_date``.
        max_samples : int, optional
            Cap on the number of *remaining* (not already processed) DOIs to
            run.  Useful for smoke tests.
        doi_to_domain_filter : dict[str, list[str]], optional
            Per-DOI Perplexity domain filters (pre-resolved).  Callers are
            responsible for loading any source files before passing this in
            (e.g. via ``prepare_doi_domain_filters_from_json()``).

        Returns
        -------
        dict[str, dict]
            ``doi → {"response": str, "token_usage": dict, "error": str|None,
            "skipped": bool}``
        """
        if not self._connected:
            await self.connect()

        from sciconharness.mcp_client.llm_providers.base import ContextLengthExceededError

        self._setup_batch_logging()

        results: Dict[str, Dict[str, Any]] = {}

        # ── skip already-processed DOIs ──────────────────────────────────────
        all_items = list(doi_to_question.items())
        skipped: List[str] = []
        remaining: List[Tuple[str, str]] = []

        for doi, question in all_items:
            if self._is_doi_already_processed(doi):
                skipped.append(doi)
                logger.info("Skipping already processed DOI: %s", doi)
                results[doi] = {
                    "response": "Already processed",
                    "token_usage": None,
                    "error": None,
                    "skipped": True,
                }
            else:
                remaining.append((doi, question))

        if skipped:
            logger.info(
                "Skipped %d already processed DOIs (out of %d total)",
                len(skipped), len(all_items),
            )

        # ── apply max_samples to *remaining* items only ──────────────────────
        items_to_run = remaining
        if max_samples is not None and max_samples > 0:
            items_to_run = remaining[:max_samples]
            logger.info(
                "Limiting to first %d samples (out of %d remaining)",
                max_samples, len(remaining),
            )

        # ── process each DOI ─────────────────────────────────────────────────
        for doi, question in items_to_run:
            if not question:
                logger.warning("Skipping %s: empty question", doi)
                results[doi] = {"response": None, "token_usage": None, "error": "Empty question"}
                continue

            publication_date = (doi_to_date or {}).get(doi)

            logger.info("=" * 60)
            logger.info("Processing: %s", doi)
            preview = question[:100] + "..." if len(question) > 100 else question
            logger.info("Question: %s", preview)
            if publication_date:
                logger.info("Publication Date: %s", publication_date)
            logger.info("=" * 60)

            # Resolve per-DOI Perplexity domain filter (normalization stays here
            # since it requires the full doi_to_domain_filter dict context)
            per_doi_df: Optional[List[str]] = None
            if self.provider_name == "perplexity" and doi_to_domain_filter:
                from sciconharness.mcp_client.llm_providers.perplexity_provider import (
                    _normalize_doi_for_lookup,
                )
                per_doi_df = doi_to_domain_filter.get(_normalize_doi_for_lookup(doi))

            try:
                response, token_usage = await self.query(
                    question,
                    doi=doi,
                    publication_date=publication_date,
                    domain_filter=per_doi_df,
                )
                ok, reason = self._is_well_formatted(response)
                results[doi] = {
                    "response": response,
                    "token_usage": token_usage,
                    "error": None if ok else f"Not well-formatted: {reason}",
                }
            except ContextLengthExceededError:
                logger.warning("Context length exceeded for %s — retrying from scratch", doi)
                try:
                    response, token_usage = await self.query(
                        question,
                        doi=doi,
                        publication_date=publication_date,
                        domain_filter=per_doi_df,
                    )
                    ok, reason = self._is_well_formatted(response)
                    results[doi] = {
                        "response": response,
                        "token_usage": token_usage,
                        "error": None if ok else f"Not well-formatted: {reason}",
                    }
                    if response:
                        logger.info("Successfully completed %s after context-length retry", doi)
                except Exception as retry_err:
                    logger.error(
                        "Context length exceeded for %s even after retry: %s",
                        doi, retry_err, exc_info=True,
                    )
                    results[doi] = {
                        "response": None,
                        "token_usage": None,
                        "error": f"Context length exceeded even after retry: {retry_err}",
                    }
            except Exception as e:
                logger.error("Error processing %s: %s", doi, e, exc_info=True)
                results[doi] = {"response": None, "token_usage": None, "error": str(e)}

        logger.info("=" * 60)
        logger.info("Batch processing complete!")
        logger.info("  Total DOIs: %d", len(results))
        successful = sum(1 for r in results.values() if r.get("response") is not None)
        failed = sum(1 for r in results.values() if r.get("response") is None)
        logger.info("  Successful: %d", successful)
        logger.info("  Failed:     %d", failed)
        logger.info("=" * 60)

        return results

    # ── internal helpers ──────────────────────────────────────────────────────

    async def _query_deep_research(
        self,
        *,
        question: str,
        doi: Optional[str],
        publication_date: Optional[str],
        source_title: Optional[str] = None,
        log_dir: Path,
        data_dir: Optional[Path],
    ) -> Tuple[str, Dict[str, Any]]:
        """Calls OpenAI deep-research via remote MCP servers."""
        from sciconharness.utils.query_utils import (
            configure_remote_mcp_servers,
            build_mcp_tools_for_deep_research,
        )

        mcp_logger = logging.getLogger("mcp_client.harness")
        resolved_title = source_title or self._source_title_for(doi)

        if self.enable_filtering:
            # Warn when both title and date are needed but one is absent
            if not resolved_title or not publication_date:
                mcp_logger.warning("")
                mcp_logger.warning("=" * 80)
                mcp_logger.warning("⚠️  WARNING: MISSING REQUIRED FILTER PARAMETERS ⚠️")
                mcp_logger.warning("=" * 80)
                mcp_logger.warning("DOI: %s", doi)
                mcp_logger.warning("Source Title: %s", "PROVIDED" if resolved_title else "MISSING")
                mcp_logger.warning("Publication Date: %s", "PROVIDED" if publication_date else "MISSING")
                mcp_logger.warning(
                    "The MCP servers require BOTH source_title AND publication_date for "
                    "proper filtering. Without both, filtering will not work."
                )
                mcp_logger.warning("=" * 80)
                mcp_logger.warning("")

            mcp_logger.info("")
            mcp_logger.info("=" * 80)
            mcp_logger.info("CONFIGURING REMOTE MCP SERVERS FOR DEEP RESEARCH (DOI: %s)", doi)
            mcp_logger.info("=" * 80)
            configure_remote_mcp_servers(
                source_title=resolved_title,
                publication_date=publication_date,
                log_dir=log_dir,
            )
        else:
            mcp_logger.info("")
            mcp_logger.info("=" * 80)
            mcp_logger.info("CONFIGURING REMOTE MCP SERVERS FOR LOGGING (FILTERING DISABLED)")
            mcp_logger.info("=" * 80)
            mcp_logger.info("Filtering is disabled. Setting up logging only.")
            mcp_logger.info("Log directory: %s", log_dir)
            try:
                configure_remote_mcp_servers(
                    source_title="",
                    publication_date="",
                    log_dir=log_dir,
                    custom_logger=mcp_logger,
                )
                mcp_logger.info("✓ Remote MCP server configuration completed")
            except Exception as e:
                mcp_logger.error("✗ Error configuring remote MCP servers: %s", e, exc_info=True)
                mcp_logger.warning("Continuing without remote_mcps.log setup...")
            mcp_logger.info("=" * 80)

        mcp_tools = build_mcp_tools_for_deep_research()

        mcp_logger.info("")
        mcp_logger.info("=" * 80)
        mcp_logger.info("CALLING OPENAI DEEP RESEARCH API (DOI: %s)", doi)
        mcp_logger.info("=" * 80)
        mcp_logger.info("Model: %s", self.model)
        mcp_logger.info("Tools configured:")
        for tool in mcp_tools:
            mcp_logger.info(
                "  - %s: %s",
                tool.get("server_label", "Unknown"),
                tool.get("server_url", "Unknown"),
            )
        if self.enable_filtering:
            if resolved_title:
                mcp_logger.info("Source Title: %s", resolved_title)
            if publication_date:
                mcp_logger.info("Publication Date: %s", publication_date)
        mcp_logger.info("")
        mcp_logger.info("Input query:")
        mcp_logger.info(question)
        mcp_logger.info("")

        response_obj, text_content, _tool_calls, _reasoning = (
            await self._llm_provider.call_llm_background(
                query=question,
                mcp_tools=mcp_tools,
                max_tool_calls=self.max_tool_calls,
            )
        )

        response_text = text_content or ""
        token_usage = self._extract_token_usage(response_obj, question, response_text)

        mcp_logger.info("=" * 80)
        mcp_logger.info("OPENAI API RESPONSE RECEIVED (DOI: %s)", doi)
        mcp_logger.info("=" * 80)
        mcp_logger.info("Response:")
        mcp_logger.info(response_text)
        mcp_logger.info("")
        mcp_logger.info("=" * 80)

        # Save only when data_dir is provided. query() always passes data_dir=None to
        # suppress this and instead saves once after all format retries complete.
        if data_dir and self.save_results:
            result_filter = self._make_filter(doi, publication_date, source_title)
            result_file = self._save(
                data_dir=data_dir,
                question=question,
                response=response_text,
                token_usage=token_usage,
                doi=doi,
                publication_date=publication_date,
                result_filter=result_filter,
            )
            if result_file:
                mcp_logger.info("Result saved to: %s", result_file)
                print(f"Result saved to: {result_file}")
            else:
                mcp_logger.warning("Result file verification failed.")

        return response_text, token_usage

    # ── repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        flags = []
        if self.enable_tools:
            flags.append("tools")
        if self.enable_filtering:
            flags.append("filter")
        return (
            f"SciConHarness(provider={self.provider_name!r}, model={self.model!r}"
            + (f", [{', '.join(flags)}]" if flags else "")
            + ")"
        )
