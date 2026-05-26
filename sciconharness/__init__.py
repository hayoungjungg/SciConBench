"""
SciConHarness — LLM query harness for SciConBench.

Provides a modular framework for querying LLM APIs with tool-augmented web search,
designed for evaluating models on scientific conclusion synthesis tasks.

Components:
- mcp_client: MCP stdio client driving the tool loop for standard LLM providers (e.g., OpenAI, Gemini, Claude, Perplexity),
              including clean room filtering protocol, and continuous LLM-tool interactions.
- mcp_server: FastMCP server exposing and executing search/browse tools (Serper, S2, Jina) for LLMs, including caching and rate limiting.
- remote_mcp_servers: HTTP MCP servers for OpenAI deep research models. Provides the same tools as mcp_server, but hosted on a remote server.
              to adhere to OpenAI Deep Research API requirements.
- utils: Provider factory, logging, result persistence, Perplexity filtering, and tool execution.

Entry points:
- python -m sciconharness.cli_scripts.query_single   Single query (interactive or scripted)
- python -m sciconharness.cli_scripts.query_batch    Batch processing over DOI→question dicts
"""

from sciconharness.harness import SciConHarness

__all__ = ["SciConHarness"]

