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
| [`mcp_client/`](mcp_client/README.md) | Runs the agentic loop: connects to an MCP server, sends tool-augmented queries to an LLM, applies `CochraneResultFilter` to each tool result, and iterates until the LLM produces a final answer. Also contains the `LLMProvider` abstract interface and all provider implementations (OpenAI, Claude, Gemini, Perplexity, Azure, OpenRouter). |
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
# Azure OpenAI / Foundry (also used by provider="azure")
AZURE_OPENAI_KEY=...
OPENAI_BASE_URL=https://<resource>.services.ai.azure.com/openai/v1/
# OpenRouter (provider="openrouter")
OPENROUTER_API_KEY=...
SERPER_API_KEY=...
S2_API_KEY=...
JINA_API_KEY=...
```



### Python API

```python
import asyncio
import os
from sciconharness import SciConHarness

# No need to load titles / doi_to_title / doi_to_publication_date yourself —
# when enable_filtering=True and you don't pass them explicitly, the harness
# auto-loads all three from a local cache built from the live SciConBench
# HuggingFace dataset (see "Clean Room Evaluation Protocol" below).
harness = SciConHarness(
    provider="openai",
    model="gpt-5.1",
    api_key=os.environ["AZURE_OPENAI_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
    api_version=os.environ.get("OPENAI_API_VERSION"),
    enable_tools=True,
    enable_filtering=True,
    save_result=True,
)

async def main():
    async with harness:
        for row in questions:
            doi   = row["doi"]
            question = row["question"]
            title    = row["title"]

            print(f"\n  DOI            : {doi}")
            print(f"  Title          : {title}")
            print(f"  Question       : {textwrap.shorten(question, width=120)}")
            print("  Running…")

            # publication_date is optional here too — auto-resolved from the
            # same cache via doi= if omitted.
            response, usage = await harness.query(
                question,
                doi=doi,
            )

            print_result(label, doi, response, usage)

asyncio.run(main())
```



### CLI

```bash
# Single query — --doi is enough; source title, title-list filtering, and
# publication date all auto-resolve from the local HF benchmark cache.
python -m sciconharness.cli_scripts.query_single claude \
    --model claude-sonnet-4-5 \
    --query "What are the benefits and harms of oral antibiotics for otitis media?" \
    --doi "10.1002/14651858.CD015254.pub2" \
    --enable-tool-calling --enable-filtering

# Batch over all DOIs (doi-dates.json / --cochrane-titles are optional
# overrides now — omit them to use the auto-loaded cache)
python -m sciconharness.cli_scripts.query_batch openai \
    --model gpt-5.1 \
    --doi-questions data/doi_questions.json \
    --enable-tool-calling --enable-filtering
```

---



## Configuration Reference (`SciConHarness`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | str | `"openai"` | `"openai"` \| `"claude"` \| `"gemini"` \| `"perplexity"` \| `"azure"` \| `"openrouter"` |
| `model` | str | provider default | Model name. Defaults: openai→`gpt-5.1`, claude→`claude-sonnet-4-5`, gemini→`gemini-3-pro-preview`, perplexity→`sonar-reasoning-pro`, azure→`DeepSeek-V4-Pro`, openrouter→`moonshotai/kimi-k3` |
| `enable_tools` | bool | `True` | Enable MCP tool calling (web search + browse). Ignored for models that always use built-in search (see below). |
| `enable_filtering` | bool | `True` | Apply `CochraneResultFilter` to suppress contaminated results. Fully automatic — `cochrane_titles`, `doi_to_title`, and `doi_to_publication_date` all auto-load from a local HF benchmark cache if not passed explicitly (see below). |
| `cochrane_titles` | `list[str]` or `Path` | auto-loaded | Cochrane review titles for title-based source filtering. Pass a list or a path to a JSON file of title strings to override the auto-loaded cache. |
| `doi_to_title` | `dict[str, str]` | auto-loaded | DOI → source title mapping. Falls back to the local HF benchmark cache (`sciconharness.utils.hf_benchmark_cache`). |
| `doi_to_publication_date` | `dict[str, str]` | auto-loaded | DOI → publication date mapping, used to auto-resolve `query()`'s `publication_date` when omitted. Falls back to the same HF benchmark cache. |
| `temperature` | float | model default | Sampling temperature. `None` = model default. |
| `max_tokens` | int | provider default | Max output tokens. |
| `max_tool_calls` | int | `30` | Hard cap on tool calls per query for OpenAI's deep-research agents. Does not apply for frontier models. |
| `max_format_retries` | int | `3` | Retry attempts if the response is missing the required `[[[...]]]` conclusion marker. |
| `min_conclusion_length` | int | `20` | Minimum character count inside `[[[...]]]` to accept the response as well-formatted. |
| `save_results` | bool | `True` | Save `result.json` to `sciconharness/logs/<model>/<doi>/`. |
| `log_dir` | Path | `None` | Override the base log/output directory. |
| `api_key` | str | env var | Override the provider API key. |
| `base_url` | str | `None` | Azure OpenAI / Foundry endpoint. Required for `provider="azure"`; for openai/claude activates Azure mode automatically. Ignored for `provider="openrouter"` (defaults to `https://openrouter.ai/api/v1`; override via `OPENROUTER_BASE_URL`). |

---



## Clean Room Evaluation Protocol

The benchmark uses a **clean room protocol** to prevent data leakage: the agent must not see the Cochrane review article itself, results published after its cut-off date, or any derivative content (news coverage, press releases, etc.).

When `enable_filtering=True`, every tool result is passed through `CochraneResultFilter` before being returned to the LLM. It needs three pieces of Cochrane-benchmark metadata to work fully:

1. **`cochrane_titles`** — every review title in the benchmark, used for title-based blocking (catches mirrors/reprints of *any* known review, not just the one being queried).
2. **`doi_to_title`** — DOI → review title, used to identify which review is being queried so the filter can also block its own source by exact content match.
3. **`doi_to_publication_date`** — DOI → publication date, used as the results cutoff (anything published after this date is filtered out).

**You don't need to load or pass any of these yourself.** If you don't supply them explicitly, `SciConHarness` auto-loads all three from a local JSON cache built once from the live `hayoungjung/SciConBench` HuggingFace dataset — see `sciconharness/utils/hf_benchmark_cache.py`:

```python
from sciconharness.utils.hf_benchmark_cache import (
    load_cochrane_titles_cached,          # list[str]
    load_doi_to_title_cached,             # {doi: title}
    load_doi_to_publication_date_cached,  # {doi: publication_date}
)
```

The first call that finds no cache on disk (`data_track/hf_benchmark_cache/*.json`, override with `SCICONBENCH_CACHE_DIR`) pulls the full dataset via `datasets.load_dataset(...)` once and writes all three files; every later call — same process or a new one — just reads JSON off disk, no network call needed. This is why `harness.query(question, doi=doi)` alone is now enough to get full clean-room filtering for any DOI in the benchmark: title-list filtering, source-title content matching, *and* the publication-date cutoff all resolve automatically from `doi=`.

To force a fresh pull instead of using whatever cache already exists:

```bash
python -m sciconharness.utils.hf_benchmark_cache --force
```

**Keeping the cache in sync with the dashboard.** `scicon-track`'s monthly HuggingFace uploader (`scicon-track/huggingface/uploader.py`) calls `uploader.refresh_filter_caches()` right after merging new reviews and before pushing to HuggingFace, rebuilding this cache directly from the rows about to be published (no extra network pull — it reuses the already-merged in-memory rows). So the cache is refreshed on every `scicon-track upload` / monthly pipeline run and never drifts more than one dashboard cycle behind the live dataset.

**Overriding the cache.** Pass `cochrane_titles=`, `doi_to_title=`, and/or `doi_to_publication_date=` explicitly to `SciConHarness(...)` (or `--cochrane-titles` / `--doi-dates` to the CLIs) to bypass the cache entirely — useful for ad-hoc DOIs that aren't (yet) part of the tracked benchmark, or for a custom filtering scope.

> **Note for OpenAI's deep research agents:** The remote MCP servers load `cochrane_titles.json` at startup and apply filtering server-side. Pass `--cochrane-titles` explicitly there if you need a scope different from the auto-loaded cache.

---

## Model-Specific Behaviour

### Standard models — Claude, Gemini, OpenAI (non-deep-research, e.g., GPT-5.1)

`MCPClient` manages the agentic loop: spawns the local MCP server as a subprocess, formats tools for the LLM's function-calling API, executes tool calls, and applies `CochraneResultFilter` to each result before passing it back to the LLM.

- `enable_tools` controls whether tool calling is active.
- `enable_filtering` controls whether `CochraneResultFilter` is applied.

#### Reasoning is auto-maxed for every provider, via OpenRouter's model catalog

`ClaudeProvider`, `OpenAIProvider`, and `GeminiProvider` all call straight through to their vendor's own native API — no inference traffic goes through OpenRouter for them. But at construction time, each one calls the shared `reasoning_discovery.py` helper (the same discovery mechanism `OpenRouterProvider` uses — see below) purely as a **reference catalog**: OpenRouter's [`GET /models`](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#discovering-per-model-reasoning-options) aggregates the `supported_efforts` capability data for models across virtually every vendor in one place, so it's a convenient way to look up "what's the highest reasoning effort this exact model supports?" even when you're calling that vendor directly.

The discovered ceiling is translated into whatever mechanism each vendor's *native* API actually accepts:

| Provider | Native mechanism | How the discovered effort is applied |
|----------|-------------------|----------------------------------------|
| `OpenAIProvider` | native `reasoning.effort` string | discovered value used as-is (OpenAI's own concept — OpenRouter just mirrors it) |
| `GeminiProvider` | native `thinking_level` enum | discovered effort mapped through Google's effort→`thinkingLevel` table (Gemini has no tier above `HIGH`) |
| `ClaudeProvider` | native adaptive `output_config.effort` when supported; fixed `thinking.budget_tokens` otherwise | Opus 5 uses adaptive `effort=max`, matching the successful 2026-07 workflow calls; older models can use a discovered fixed token budget |

All three fall back to their previous hardcoded defaults (`"high"` / `HIGH` / a flat 4096-token budget) if OpenRouter has no data for the given model or the lookup fails — this is purely additive and never blocks a run. All three still accept an explicit constructor override (`reasoning_effort=...`, `thinking_level=...`, `thinking_budget_tokens=...`) that skips discovery entirely. For today's default models this doesn't actually change anything — `gpt-5.1`'s and `gemini-3.1-pro-preview`'s discovered ceilings are already `"high"` (their existing hardcoded defaults), and `claude-sonnet-4-5` isn't listed with effort data on OpenRouter, so it keeps its existing flat 4096-token budget — but it means the harness automatically maxes reasoning correctly if you swap in a different/newer model for any of these three providers, without needing a code change.

### Azure Foundry Chat Completions (`provider="azure"`)

Azure Foundry Chat Completions models (**DeepSeek-V4-Pro**, plus control
**DeepSeek-V4-Flash-0731**) use the OpenAI-compatible **Chat Completions**
API, not the OpenAI Responses API. Use `provider="azure"` so the harness
routes through `AzureChatCompletionsProvider`.

The wire format differs from `OpenAIProvider` (nested `function` schema, `role=tool` result messages, an injected `system` message instead of top-level `instructions`), but the functionality and end outcomes are identical: same `RESEARCH_ASSISTANT_PROMPT`, the same one-tool-result-per-message loop (`MCPClient` treats it exactly like OpenAI, since neither defines `format_multiple_tool_response_message`), and the same retry contract (rate-limit backoff, timeout retry, `ContextLengthExceededError` on context-window overflows).

DeepSeek-V4-Pro is deployed on its own dedicated Azure resource, separate from the one used for GPT via Azure OpenAI, so this provider prefers its own credentials (falling back to the shared `AZURE_OPENAI_KEY` / `OPENAI_BASE_URL` if unset):

```
AZURE_OPENAI_KEY=...
OPENAI_BASE_URL=https://<your-resource>.services.ai.azure.com/openai/v1/
OPENAI_API_VERSION=2025-04-01-preview
```

```bash
# Smoke test one DOI — publication date / title filtering auto-resolve from
# --doi via the HF benchmark cache (see "Clean Room Evaluation Protocol").
python -m sciconharness.cli_scripts.query_single azure \
    --model DeepSeek-V4-Pro \
    --query "What are the benefits and harms of oral antibiotics for otitis media?" \
    --doi "10.1002/14651858.CD015254.pub2" \
    --enable-tool-calling --enable-filtering

# Batch — same flags as other providers
python -m sciconharness.cli_scripts.query_batch azure \
    --model DeepSeek-V4-Pro \
    --doi-questions data/doi_questions.json \
    --enable-tool-calling --enable-filtering
```

Pass your Azure **deployment name** exactly as `--model` (`DeepSeek-V4-Pro`
or the control deployment `DeepSeek-V4-Flash-0731`).

DeepSeek defaults match Claude/OpenRouter: completion `max_tokens=8192`,
reasoning maxed via top-level `reasoning_effort="max"`. Azure Foundry does
not expose a separate reasoning token budget (no `thinking` / `extra_body`).

### OpenRouter (`provider="openrouter"`)

[OpenRouter](https://openrouter.ai) exposes the same OpenAI-compatible **Chat Completions** API as `AzureChatCompletionsProvider`, routed to a single OpenRouter API key. Use `provider="openrouter"` so the harness routes through `OpenRouterProvider`. Models onboarded so far:

| Model | `--model` slug | Notes |
|-------|----------------|-------|
| Kimi K3 | `moonshotai/kimi-k3` | |
| GLM-5.2 / 5.3 | `z-ai/glm-5.2`, `z-ai/glm-5.3` | |
| Qwen3.5-9B | `qwen/qwen3.5-9b` | |
| Qwen3.7-max | `qwen/qwen3.7-max` | |
| Qwen3.8-max | `qwen/qwen3.8-max` | |
| Qwen3.8 27B | `qwen/qwen3.8-27b` | **control** (SciConBench-Track) |
| MiniMax M3 | `minimax/minimax-m3` | **control** (SciConBench-Track) |

`OpenRouterProvider` reuses the exact same message-preparation, tool-loop, and retry/error-handling logic as `AzureChatCompletionsProvider` — same `RESEARCH_ASSISTANT_PROMPT`, same one-tool-result-per-message loop, same `ContextLengthExceededError`/rate-limit/timeout contract.

Reasoning is maxed out via OpenRouter's unified `reasoning` object, but instead of hardcoding `effort="max"`, the provider **systematically discovers** each model's actual supported effort levels from [`GET /models`](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#discovering-per-model-reasoning-options) at construction time (cached per model per process, via the shared `reasoning_discovery.py` module — also used by `ClaudeProvider`/`OpenAIProvider`/`GeminiProvider`, see above) and sends the true ceiling explicitly:

| Model | Discovered `supported_efforts` | Reasoning payload sent |
|-------|-------------------------------|------------------------|
| Kimi K3 | `["max", "high", "low"]` | `enabled` + `effort="max"` |
| GLM-5.3 | `["max", "high", "low"]` | `enabled` + `effort="max"` |
| Qwen3.8-max | `["xhigh", "high", "medium", "low", "minimal"]` | `enabled` + `effort="xhigh"` |
| Qwen3.8 27B (control) | `["xhigh", "medium", "low"]` | `enabled` + `effort="xhigh"` |
| MiniMax M3 (control) | *(no effort — not in supported_parameters)* | `enabled` only |

If discovery ever fails (network error, model delisted, etc.) the provider falls back to `effort="max"`, which OpenRouter clamps to whatever the model actually supports.

Reasoning is **always enabled** for these models. Top-level completion `max_tokens` defaults to **8192**. The nested reasoning payload never includes `max_tokens`: effort-capable models use the discovered ceiling (`max` / `xhigh`), while MiniMax sends only `enabled`.

Reasoning is also **preserved** across tool-calling turns per [OpenRouter's best practices](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#preserving-reasoning-blocks): the provider reads back both `message.reasoning_details` (the full structured block) and the plaintext `message.reasoning`/`reasoning_content` alias, and echoes `reasoning_details` verbatim on the next turn's assistant message whenever a model returns it (falling back to the plaintext alias otherwise), so tool-call round trips resume the model's reasoning state exactly rather than dropping it.

#### Prompt caching / sticky routing (`session_id`)

Each tool-calling turn resends the full (growing) message history, so cost grows roughly with the *square* of the conversation length unless the shared prefix (system prompt, tool definitions, earlier turns) hits a prompt cache. OpenRouter routes every request independently by default, so turn *N* of a conversation can land on a different upstream provider than turn *N-1*, defeating that provider's cache. Passing a stable `session_id` opts a conversation into [sticky routing](https://openrouter.ai/docs/features/prompt-caching), so OpenRouter prefers the same upstream provider for every request sharing that id.

`SciConHarness.query()` handles this automatically for `OpenRouterProvider`: it calls `set_session_id()` once per query attempt, *before* that attempt's tool-calling loop starts, with an id of the form `{provider}:{model}:{doi}:{attempt}:{random}` (see `harness.py::_build_session_id`). Every `call_llm()` call inside that one attempt's loop reuses the same id, so its turns route together — while a different query, even the exact same DOI run against a different model/config or rerun later, gets its own fresh id and never shares routing/cache state with an unrelated conversation. Direct `OpenRouterProvider` users (bypassing `SciConHarness`) can call `provider.set_session_id(...)` themselves, or leave it unset to fall back to OpenRouter's default per-request routing.

Actual cache hits are visible per-call in `mcp_client.log` (`OpenRouter usage: ... cached_tokens=...`, from `response.usage.prompt_tokens_details.cached_tokens`) and rolled up into `result.json`'s `token_usage.cached_content_tokens` for the whole query — the same field Gemini populates from its own native cached-token count.

```
OPENROUTER_API_KEY=...
```

```bash
# Smoke test one DOI — publication date / title filtering auto-resolve from
# --doi via the HF benchmark cache (see "Clean Room Evaluation Protocol").
python -m sciconharness.cli_scripts.query_single openrouter \
    --model moonshotai/kimi-k3 \
    --query "What are the benefits and harms of oral antibiotics for otitis media?" \
    --doi "10.1002/14651858.CD015254.pub2" \
    --enable-tool-calling --enable-filtering

# Batch — same flags as other providers; swap --model per model
python -m sciconharness.cli_scripts.query_batch openrouter \
    --model z-ai/glm-5.2 \
    --doi-questions data/doi_questions.json \
    --enable-tool-calling --enable-filtering
```

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
- `enable_filtering` is still respected: when `True`, the remote MCP servers are configured with the review's `source_title` and `publication_date` before the query runs, and filtering is applied server-side inside each tool call.
- **Remote MCP servers must be running and publicly accessible** before querying these models. See `[remote_mcp_servers/README.md](remote_mcp_servers/README.md)` for server setup, nginx configuration, and ngrok tunneling.

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
