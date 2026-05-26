# Sciconharness MCP Server

> **Credit:** The `mcp_server/` directory borrows extensively from the [`dr-agent-lib`](https://github.com/rlresearch/dr-tulu) MCP backend by Rulin Shao, Akari Asai, Shannon Zejiang Shen, Hamish Ivison, and the DR Tulu team at Ai2 ([arXiv:2511.19399](https://arxiv.org/abs/2511.19399)). We thank them for making their infrastructure open source. In turn, we also committed to making our SciConBench evaluation infrastructure, including SciConHarness, open source for future researchers to build better scientific AI agents.

The local MCP server exposes search and browse tools to LLM clients via the [FastMCP](https://github.com/jlowin/fastmcp) framework. It is used automatically by `MCPClient` (stdio transport) and can also be run standalone as an HTTP server.

## Tools

| Tool | Tags | Description |
|------|------|-------------|
| `serper_google_webpage_search` | search | Google web search via Serper.dev |
| `semantic_scholar_snippet_search` | search | Snippet retrieval from Semantic Scholar |
| `jina_fetch_webpage_content` | browse | Full webpage fetch via Jina Reader |

## Required API Keys

All three keys are **required**. The server raises `EnvironmentError` at startup if any are missing, and each API function raises `ValueError` if its key is absent at call time.

Set them in a `.env` file at the project root:

```
SERPER_API_KEY=your_serper_key     # https://serper.dev/
S2_API_KEY=your_s2_key             # https://api.semanticscholar.org/
JINA_API_KEY=your_jina_key         # https://jina.ai/reader/
```
Note you will need to get these API keys from the respective services.
- S2_API_KEY: https://api.semanticscholar.org/
- SERPER_API_KEY: https://serper.dev/
- JINA_API_KEY: https://jina.ai/reader/

## Running the MCP Server

**Automatic (used by MCPClient):** The server is spawned automatically via stdio when `MCPClient.connect_to_server()` is called—no manual startup needed.

**Manual HTTP mode** (e.g., for testing or as a standalone endpoint):

```bash
python -m sciconharness.mcp_server.main --transport http --host 127.0.0.1 --port 8000
```

Options:
- `--transport`: `stdio` | `http` | `sse` | `streamable-http` (default: `http`)
- `--host`: bind address (default: `127.0.0.1`)
- `--port`: port number (default: `8000`)
- `--no-cache`: disable disk caching of API responses

Health check: `curl http://127.0.0.1:8000/health`

## Caching

API responses are cached on disk via `diskcache`. The cache directory is controlled by the `MCP_CACHE_DIR` environment variable. Pass `--no-cache` to disable.

## Code Map when Incorporating New Tools

| Concern | Location |
|---------|----------|
| **Tool definitions** | `@mcp.tool(...)` functions in `mcp_server/main.py`; underlying HTTP calls in `mcp_server/apis/`. These definitions are provided to the model/agent. |
| **Tool allowlist** | `MCPClient.DEFAULT_ALLOWED_TOOLS` in `mcp_client/mcp_client.py` — tools not listed here are dropped at session startup |
| **Tool formatting for the LLM** | `format_tools()` in each `mcp_client/llm_providers/<provider>.py` (converts MCP tool objects to the provider's function-calling schema) |
| **Tool execution** | `ToolExecutor.execute_tool_call()` → `_execute_tool_with_filtering()` in `mcp_client/utils/tool_execution.py`; pagination in `_execute_paginated_search()` |
| **Tool filtering** | `CochraneResultFilter.filter()` in `mcp_client/filters/cochrane.py`; item-level decisions in `_should_filter_item()` / `_should_filter_jina_content()` |
| **Tool result logging** | `log_tool_result()` in `mcp_client/utils/utils.py` — has an explicit branch per tool name; add one for any new tool |
| **Module docstring** | `mcp_client/__init__.py` — lists exposed tools by name; update when adding or removing tools |

---

## Extending with New APIs and Tools

Researchers and practitioners can add entirely new search or browse APIs to the MCP server. The change spans several files across `mcp_server/` and `mcp_client/`.

### Step 1 – Add an API module in `apis/`

Create `apis/my_api.py` implementing the HTTP call(s) to your new service. Follow the conventions of the existing modules:

- Define a Pydantic response model (see `data_model.py` for shared types and `serper_apis.py` for examples).
- Use the shared `cache` decorator from `mcp_server/cache.py` if you want disk caching.
- Raise `ValueError` if the required API key is absent.

### Step 2 – Add the tool name to `MCPClient.DEFAULT_ALLOWED_TOOLS`

In `mcp_client/mcp_client.py`, add the tool name to the `DEFAULT_ALLOWED_TOOLS` set:

```python
DEFAULT_ALLOWED_TOOLS = {
    "serper_google_webpage_search",
    "jina_fetch_webpage_content",
    "semantic_scholar_snippet_search",
    "my_new_tool",   # ← add this
}
```

### Step 3 – Register the tool in `main.py`

Import your new function and expose it as a FastMCP tool:

```python
from .apis.my_api import my_api_function, MyApiResponse

@mcp.tool(tags=["search"])   # or "browse"
async def my_new_tool(query: Annotated[str, "Search query"]) -> MyApiResponse:
    """Short description shown to the LLM."""
    return await my_api_function(query)
```

Add the corresponding API key check to `_check_required_api_keys()`.

### Step 4 – Update the filter in `mcp_client/filters/cochrane.py`

The `CochraneResultFilter` (and any custom filter you write) must be taught about the new tool so it can suppress contaminated results. Three changes are required:

1. **Register the tool name** – add `"my_new_tool"` to the `filtered_tools` set in `CochraneResultFilter.__init__`.

2. **Add field accessors and a dispatch branch** in `filter()`:

   ```python
   elif tool_name == "my_new_tool":
       if "results" in filtered_result and isinstance(filtered_result["results"], list):
           def get_title(item): return item.get("title", "")
           def get_urls(item): return [item.get("url", "")]
           def get_metadata(item): return {"URL": item.get("url", "")}
           def get_date(item): return item.get("published_date")

           filtered_result["results"] = self._filter_list_items(
               filtered_result["results"],
               get_title, get_urls, get_metadata, get_date,
               "my_new_tool"
           )
   ```

   Adjust the result key (`"results"`) and field names to match your API's actual response schema.

3. **Add a tracking entry** – if you want per-tool filtering summaries, add `"my_new_tool": []` to `self.filtered_items` in `__init__` and the corresponding display name in `api_names` inside `log_filtering_summary` / `get_filtering_summary`.

### Step 5 – Update `mcp_client/utils/tool_execution.py`

`ToolExecutor` may need changes depending on the tool's behavior:

- **Non-standard result key** – `_execute_tool_with_filtering` defaults to the `"organic"` key. If your tool uses a different key (e.g., `"results"`), add a branch that reads the correct key and passes it to `result_filter.filter`.
- **Pre-call argument manipulation** – if the new tool accepts a date/year range that should be capped to prevent data leakage, add a conditional block in `execute_tool_call` before the `_execute_tool_with_filtering` call (similar to the existing `semantic_scholar_snippet_search` year-capping logic).
- **Pagination** – if the tool supports page-based pagination (e.g., web_search tool via serper's Google search API, which supports fetching pages of search results from Google) and results need to be fetched until the post-filter count is satisfied, route to `_execute_paginated_search` (or write an analogous method) inside `_execute_tool_with_filtering`.
- **Single-result tools** (like `jina_fetch_webpage_content`) – add an `if tool_name == "my_new_tool"` branch that handles the single-dict result path instead of the list path.

### Step 6 – Add a logging branch in `mcp_client/utils/utils.py`

`log_tool_result()` has an explicit `if/elif` branch for each tool. Add one so the new tool's results appear in the per-call log:

```python
elif tool_name == "my_new_tool":
    logger.info("=" * 80)
    logger.info("MY NEW TOOL RESULTS (my_new_tool)")
    logger.info("=" * 80)
    logger.info("Query: %s", parsed_args.get("query", "N/A"))
    logger.info("Full Results:\n%s", tool_result_str)
    logger.info("=" * 80)
```

### Step 7 – Add the API key to `.env` (if necessary)

```
MY_NEW_API_KEY=your_key_here
```

### Step 8 – Update the module docstring in `mcp_client/__init__.py`

The module docstring lists all exposed tools by name. Keep it in sync:

```python
"""
Modular MCP Client Setup for Multiple LLMs

...exposing four tools:
1. Google Search (serper_google_webpage_search)
2. Browse Web via JINA API (jina_fetch_webpage_content)
3. Semantic Scholar Snippet Search (semantic_scholar_snippet_search)
4. My New Tool (my_new_tool)
"""
```
