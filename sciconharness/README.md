# SciConHarness

The evaluation harness for SciConBench. It drives LLM agents through web search and retrieval tasks while enforcing a **clean room evaluation protocol** — filtering out the Cochrane review articles being evaluated, their derivatives, any results published after the review's cut-off date, etc.

---

## Architecture Overview

```
sciconharness/
├── harness.py                  # Top-level API (SciConHarness class)
├── cli_scripts/
│   ├── query_single.py         # CLI interface for the harness: single query
│   └── query_batch.py          # CLI interface for the harness: batch over all DOIs
├── mcp_client/                 # Agentic loop, LLM providers + tools with MCP server, clean room filters
├── mcp_server/                 # Local stdio MCP server (search + browse tools)
├── remote_mcp_servers/         # HTTP MCP servers for OpenAI Deep Research
└── utils/
    ├── perplexity_filtering.py              # Domain filter helpers for Perplexity
    ├── collect_filtered_links_from_logs.py  # Build Perplexity denylist from prior runs
    └── filtered_links_from_logs.json        # Denylist for the N=268 examples in our paper for Perplexity
```

### Subdirectory summaries

Each subdirectory contains their own detailed README.md for users. We summarize them here:

| Directory | What it does |
|-----------|-------------|
| [`mcp_client/`](mcp_client/README.md) | Runs the agentic loop: connects to an MCP server, sends tool-augmented queries to an LLM, applies `CochraneResultFilter` to each tool result, and iterates until the LLM produces a final answer. Also contains the `LLMProvider` abstract interface and all provider implementations (OpenAI, Claude, Gemini, Perplexity). |
| [`mcp_server/`](mcp_server/README.md) | A local [FastMCP](https://github.com/jlowin/fastmcp) server exposing three tools: `serper_google_webpage_search`, `semantic_scholar_snippet_search`, and `jina_fetch_webpage_content`. Spawned automatically by `MCPClient` via stdio; can also run standalone as an HTTP server for testing. |
| [`remote_mcp_servers/`](remote_mcp_servers/README.md) | Two HTTP MCP servers (Serper+Jina on port 8001, Semantic Scholar+Jina on port 8002) that expose the same tools over the network. Required for OpenAI Deep Research models, which must access tools via public HTTP endpoints rather than a local subprocess. Filtering runs server-side because the OpenAI Deep Research client has no access to local Python code. |

---

## Quick Start

### Install

```bash
pip install -e .
```

Set API keys in `.env` at the project root based on which models you'd like to query and
generate conclusions.

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
PERPLEXITY_API_KEY=...
SERPER_API_KEY=...
S2_API_KEY=...
JINA_API_KEY=...
```

### Python API

```python
import asyncio
import os
from datasets import load_dataset
from sciconharness import SciConHarness

# Load the SciConBench dataset from HuggingFace to build clean room filtering variables
# e.g., list of titles, doi-to-title mapping, publication date
ds = load_dataset("hayoungjung/SciConBench", "benchmark", split="test")
all_titles   = list(ds["title"])
doi_to_title = {row["doi"]: row["title"] for row in ds}

harness = SciConHarness(
    provider="openai",
    model="gpt-5.1",
    api_key=os.environ["AZURE_OPENAI_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
    api_version=os.environ.get("OPENAI_API_VERSION"),
    enable_tools=True,
    enable_filtering=True,
    cochrane_titles=all_titles,
    doi_to_title=doi_to_title,
    save_result=True,
)

async def main():
    async with harness:
        for row in questions:
            doi   = row["doi"]
            question = row["question"]
            pub_date = row["publication_date"]
            title    = row["title"]

            print(f"\n  DOI            : {doi}")
            print(f"  Title          : {title}")
            print(f"  Publication date: {pub_date}")
            print(f"  Question       : {textwrap.shorten(question, width=120)}")
            print("  Running…")

            response, usage = await harness.query(
                question,
                doi=doi,
                publication_date=pub_date,
            )

            print_result(label, doi, response, usage)

asyncio.run(main())
```

### CLI

```bash
# Single query
python -m sciconharness.cli_scripts.query_single claude \
    --model claude-sonnet-4-5 \
    --query "What are the benefits and harms of oral antibiotics for otitis media?" \
    --doi "10.1002/14651858.CD015254.pub2" \
    --publication-date "23 October 2023" \
    --cochrane-titles *PATH to JSON file containing the list of titles* \
    --enable-tool-calling --enable-filtering

# Batch over all DOIs
python -m sciconharness.cli_scripts.query_batch openai \
    --model gpt-5.1 \
    --doi-questions data/doi_questions.json \
    --doi-dates data/doi_dates.json \
    --cochrane-titles data/filter_data/cochrane_titles.json \
    --enable-tool-calling --enable-filtering
```

---

## Configuration Reference (`SciConHarness`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | str | `"openai"` | `"openai"` \| `"claude"` \| `"gemini"` \| `"perplexity"` |
| `model` | str | provider default | Model name. Defaults: openai→`gpt-5.1`, claude→`claude-sonnet-4-5`, gemini→`gemini-3-pro-preview`, perplexity→`sonar-reasoning-pro` |
| `enable_tools` | bool | `True` | Enable MCP tool calling (web search + browse). Ignored for models that always use built-in search (see below). |
| `enable_filtering` | bool | `True` | Apply `CochraneResultFilter` to suppress contaminated results. **Requires `cochrane_titles`, `publication_date`, and `doi_to_title` to function correctly.** |
| `cochrane_titles` | `list[str]` or `Path` | `None` | Cochrane review titles for title-based source filtering. Pass a list or a path to a JSON file of title strings. |
| `doi_to_title` | `dict[str, str]` | auto-loaded | DOI → source title mapping. Falls back to the bundled `data/review_articles/data.json`. |
| `temperature` | float | model default | Sampling temperature. `None` = model default. |
| `max_tokens` | int | provider default | Max output tokens. |
| `max_tool_calls` | int | `30` | Hard cap on tool calls per query for OpenAI's deep-research agents. Does not apply for frontier models. |
| `max_format_retries` | int | `3` | Retry attempts if the response is missing the required `[[[...]]]` conclusion marker. |
| `min_conclusion_length` | int | `20` | Minimum character count inside `[[[...]]]` to accept the response as well-formatted. |
| `save_results` | bool | `True` | Save `result.json` to `sciconharness/logs/<model>/<doi>/`. |
| `log_dir` | Path | `None` | Override the base log/output directory. |
| `api_key` | str | env var | Override the provider API key. |
| `base_url` | str | `None` | Azure OpenAI / Foundry endpoint. Activates Azure mode automatically. |

---

## Clean Room Evaluation Protocol

The benchmark uses a **clean room protocol** to prevent data leakage: the agent must not see the Cochrane review article itself, results published after its cut-off date, or any derivative content (news coverage, press releases, etc.).

When `enable_filtering=True`, every tool result is passed through `CochraneResultFilter` before being returned to the LLM. To enable it fully, you must provide:

**1. `cochrane_titles`** — all Cochrane review titles in the benchmark, used for title-based blocking. Generate once from the SciConBench dataset from HuggingFace:

```python
from datasets import load_dataset
import json, pathlib

ds = load_dataset("hayoungjung/SciConBench", "benchmark", split="test")
titles = list(ds["title"])
pathlib.Path("data/filter_data/cochrane_titles.json").write_text(json.dumps(titles, indent=2))
```

**2. `doi_to_title`** — a DOI → review title mapping, used to identify which review is being queried so the filter can block its own source. Build from the same SciConBench dataset:

```python
doi_to_title = {row["doi"]: row["title"] for row in ds}
```

**3. `publication_date`** (per query) — the review's publication date, passed to `harness.query()` or provided as a `--doi-dates` JSON file to the batch CLI. Available from the same SciConBench dataset:

```python
doi_dates = {row["doi"]: row["date"] for row in ds}
```

> **Note for OpenAI's deep research agents:** The remote MCP servers load `cochrane_titles.json` at startup and apply filtering server-side. The `--cochrane-titles` CLI flag is only needed for standard (non-deep-research) models.

---

## Model-Specific Behaviour

### Standard models — Claude, Gemini, OpenAI (non-deep-research, e.g., GPT-5.1)

`MCPClient` manages the agentic loop: spawns the local MCP server as a subprocess, formats tools for the LLM's function-calling API, executes tool calls, and applies `CochraneResultFilter` to each result before passing it back to the LLM.

- `enable_tools` controls whether tool calling is active.
- `enable_filtering` controls whether `CochraneResultFilter` is applied.

### Perplexity (`sonar-pro`, `sonar-reasoning-pro`, etc.)

Perplexity has **built-in web search** and cannot use an external MCP server. As such, tool calling is always disabled automatically when `provider="perplexity"` since it does not use our own MCP servers. Instead, filtering is applied via a **domain denylist** passed to the Perplexity API's `search_domain_filter` parameter.

To obtain most filtered links that leaks the ground-truth artifacts/benchmark, the denylist is built from filtered links collected during prior model runs. URLs that `CochraneResultFilter` suppressed in Claude/Gemini/GPT runs are extracted from their logs, cleaned to domain level, and used to pre-configure Perplexity's search — preventing the same contaminated sources from appearing before Perplexity even runs. This is a key part of maintaining the clean room for models & agents that cannot apply Python-side filtering at inference time.

**Workflow:**

```bash
# 1. After running Claude/Gemini/GPT batches, collect their filtered links:
python sciconharness/utils/collect_filtered_links_from_logs.py
# Reads from sciconharness/logs/<model>/<doi>/mcp_client.log + filtered_links.json
# Writes to sciconharness/utils/filtered_links_from_logs.json

# 2. Pass the file to the Perplexity batch run:
python -m sciconharness.cli_scripts.query_batch perplexity \
    --model sonar-reasoning-pro \
    --doi-questions data/doi_questions.json \
    --doi-dates data/doi_dates.json \
    --filtered-links-json sciconharness/utils/filtered_links_from_logs.json \
    --no-enable-tool-calling --enable-filtering
```

The denylist is applied per DOI — each DOI gets the union of domains filtered across all prior model runs for that DOI, plus Cochrane domains that are always appended automatically.

### OpenAI's Deep Research Agents (`o3-deep-research`, `o4-mini-deep-research`)

- **Tool calling is always on** regardless of `enable_tools`. These models manage their own search; the harness provides MCP servers for them to connect to.
- **`enable_filtering`** is still respected: when `True`, the remote MCP servers are configured with the review's `source_title` and `publication_date` before the query runs, and filtering is applied server-side inside each tool call.
- **Remote MCP servers must be running and publicly accessible** before querying these models. See [`remote_mcp_servers/README.md`](remote_mcp_servers/README.md) for server setup, nginx configuration, and ngrok tunneling.

```bash
# Start remote MCP servers (machine with a public IP or ngrok tunnel)
python -m sciconharness.remote_mcp_servers.serper_jina.main           # port 8001
python -m sciconharness.remote_mcp_servers.semantic_scholar_jina.main  # port 8002

# Run deep research batch
python -m sciconharness.cli_scripts.query_batch openai \
    --model o4-mini-deep-research \
    --doi-questions data/doi_questions.json \
    --doi-dates data/doi_dates.json \
    --enable-filtering
```

---

## Extending the Harness

### Custom filters

SciConHarness enables users to adapt our clean room evaluation protocol. Subclass `BaseResultFilter` from `mcp_client/filters/base.py`. `CochraneResultFilter` in `mcp_client/filters/cochrane.py` is the reference implementation. See [`mcp_client/README.md`](mcp_client/README.md) for the full guide. Current and future benchmark evaluations can see our implementations and adapt accordingly to prevent their benchmark leakage.

```python
from sciconharness.mcp_client.filters.base import BaseResultFilter

class MyFilter(BaseResultFilter):
    def should_filter_tool(self, tool_name: str) -> bool:
        return tool_name in {"serper_google_webpage_search"}

    def filter(self, tool_result, tool_name):
        result = tool_result.copy()
        result["organic"] = [
            item for item in result.get("organic", [])
            if "blocked-domain.com" not in item.get("link", "")
        ]
        return result
```

### New tools

SciConHarness enables users to add new tools. Adding a tool spans `mcp_server/` (API implementation and FastMCP registration), `mcp_client/` (allowlist, tool executor, filter dispatch, logging branch), and optionally `remote_mcp_servers/` if it needs to be accessible to deep research models. The step-by-step checklist is in [`mcp_server/README.md`](mcp_server/README.md). 

### New LLM providers

SciConHarness supports evaluating additional LLM providers beyond default frontier models like gemini-3-pro, claude-sonnet-4.5, and gpt-5.1. Subclass `LLMProvider` from `mcp_client/llm_providers/base.py` and implement `format_tools`, `call_llm`, and `format_tool_response_message`. Register the new provider in `mcp_client/llm_providers/__init__.py` and in `utils/query_utils.py::create_provider()`. See [`mcp_client/README.md`](mcp_client/README.md) for the interface contract.

### Remote MCP servers for new deep research models

Some providers require HTTP-based MCP endpoints rather than a local subprocess. For these cases, you can use the included remote MCP servers or build your own. To add a custom server, follow the guide in [`remote_mcp_servers/README.md`](remote_mcp_servers/README.md):

1. Create `remote_mcp_servers/my_server/main.py` with your tool definitions.
2. Apply `CochraneResultFilter` inside each tool function before returning results.
3. Expose the server publicly via nginx + ngrok or a cloud deployment.
