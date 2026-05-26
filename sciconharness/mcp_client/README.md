# MCP Client

The `mcp_client/` package drives the agentic loop: it connects to an MCP server, calls LLM APIs with tool-use support, applies result filters, and manages the conversation until the LLM stops requesting tool calls.

## Directory Structure

```
mcp_client/
├── mcp_client.py          # Main MCPClient class (session management, agentic loop)
├── filters/
│   ├── base.py            # BaseResultFilter abstract interface
│   ├── cochrane.py        # CochraneResultFilter – reference implementation
│   └── __init__.py
├── llm_providers/
│   ├── base.py            # LLMProvider abstract interface
│   ├── claude_provider.py # Anthropic Claude implementation
│   ├── gemini_provider.py # Google Gemini implementation
│   ├── openai_provider.py # OpenAI implementation
│   ├── perplexity_provider.py # Perplexity implementation
│   └── __init__.py
├── utils/
│   ├── tool_execution.py  # ToolExecutor – executes tool calls and applies filters
│   ├── message_handlers.py
│   ├── utils.py           # Token counting, URL tracking helpers
│   └── __init__.py
└── prompts/
    ├── research_assistant.py
    └── __init__.py
```

---

## Filters

### `filters/base.py` – `BaseResultFilter`

All filters inherit from `BaseResultFilter`. Subclasses must implement two methods:

```python
class BaseResultFilter:
    def should_filter_tool(self, tool_name: str) -> bool:
        """Return True if this filter should be applied to tool_name."""
        ...

    def filter(self, tool_result: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        """Apply filtering logic and return a (possibly modified) result dict."""
        ...
```

The base class is also callable (`__call__` delegates to `filter`), preserving backward compatibility.

### `filters/cochrane.py` – `CochraneResultFilter`

`CochraneResultFilter` is the reference implementation for SciConBench's benchmark task: preventing contamination from Cochrane review articles, results published after the article, derivative contents (e.g., news coverage), etc. It is the primary example of how to write a domain-specific filter.

**What it filters and how:**

| Tool | Result key | Filter logic |
|------|-----------|--------------|
| `serper_google_webpage_search` | `organic` (list) | URL contains "cochrane", title contains "Cochrane", optional custom title list, or publication date after cutoff |
| `semantic_scholar_snippet_search` | `data` (list) | Same title/URL checks (no date filtering) |
| `jina_fetch_webpage_content` | single dict | Title matches Cochrane titles list, or content contains both "Cochrane" keyword and the source article title |

**Key internal pieces practitioners should understand:**

- **`filtered_tools`** (`set`): controls which tools `should_filter_tool()` returns `True` for. Add a new tool name here to opt it in.
- **`_should_filter_item(title, urls, date)`**: returns `(bool, reason)` for list-based search results. This is where URL checks, title checks, and date cutoff checks live.
- **`_should_filter_jina_content(content, title, url)`**: content-level check for page-fetch tools.
- **`_filter_list_items(...)`**: generic list iterator that calls `_should_filter_item` on each element. It accepts callable accessors (`get_title`, `get_urls`, `get_date`, `get_metadata`) so it works with arbitrarily shaped result dicts.
- **`filter(tool_result, tool_name)`**: the main dispatch method; routes to the appropriate branch per tool name.
- **`filtered_links`** and **`filtered_items`**: session-scoped tracking state, used to propagate previously filtered URLs across subsequent tool calls (e.g., a Jina fetch of a URL already blocked by a search filter).

### Adding or improving filters

**Adapting `CochraneResultFilter` for a new domain or tools:**

1. Keep or extend `_should_filter_item` to add domain-specific title/URL checks.
2. Update `filtered_tools` to include any new tool names that should be filtered.
3. Add a new branch in `filter()` for each new tool, providing `get_title`, `get_urls`, `get_date`, and `get_metadata` accessors that match the tool's actual response schema.

**Creating a new filter from scratch:**

```python
from sciconharness.mcp_client.filters.base import BaseResultFilter

class MyDomainFilter(BaseResultFilter):
    def __init__(self, blocklist: list[str]):
        self.blocklist = set(t.lower() for t in blocklist)
        self.filtered_tools = {"my_search_tool", "my_fetch_tool"}

    def should_filter_tool(self, tool_name: str) -> bool:
        return tool_name in self.filtered_tools

    def filter(self, tool_result, tool_name):
        result = tool_result.copy()
        if tool_name == "my_search_tool":
            result["results"] = [
                item for item in result.get("results", [])
                if item.get("title", "").lower() not in self.blocklist
            ]
        return result
```

Pass your filter instance to `MCPClient` or directly to `ToolExecutor.execute_tool_call(..., result_filter=my_filter)`.

---

## LLM Providers

### `llm_providers/base.py` – `LLMProvider`

All providers inherit from `LLMProvider` and must implement three abstract methods:

```python
class LLMProvider(ABC):
    def format_tools(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """Convert MCP tool objects to the LLM's function-calling schema."""
        ...

    async def call_llm(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Any, Optional[str], List[Dict[str, Any]], Optional[str]]:
        """Call the LLM and return (response, text_content, tool_calls, reasoning_summary)."""
        ...

    def format_tool_response_message(
        self, tool_call_id: str, tool_name: str, tool_result: str
    ) -> Dict[str, Any]:
        """Wrap a tool result in the message format the LLM expects."""
        ...
```

`call_llm` returns a 4-tuple:
- `response` – raw response object from the provider SDK
- `text_content` – plain text from the LLM (may be `None` if only tool calls are returned)
- `tool_calls` – list of dicts with `id`, `name`/`function`, `arguments` keys in the provider's own format
- `reasoning_summary` – chain-of-thought summary (provider-specific; `None` if unsupported)

### Adding a new LLM provider

1. Create `llm_providers/my_provider.py` and subclass `LLMProvider`.
2. Implement `format_tools`, `call_llm`, and `format_tool_response_message` following the conventions in the existing providers (see `claude_provider.py` or `openai_provider.py` for reference).
3. Handle tool-call extraction in `call_llm` so the returned `tool_calls` list uses a consistent dict shape that `ToolExecutor.extract_tool_call_info()` can parse (OpenAI format `{"function": {"name": ..., "arguments": ...}, "id": ...}` or the Gemini flat format `{"name": ..., "arguments": ..., "id": ...}` are both supported).
4. Register the provider in `llm_providers/__init__.py` for easy import.

---

## Tool Executor (`utils/tool_execution.py`)

`ToolExecutor` sits between the agentic loop and the MCP session. It orchestrates:

1. **Argument parsing** – normalizes `arguments` from whatever type the LLM returns (string, dict, `None`).
2. **Pre-call parameter manipulation** – e.g., capping the `year` parameter for `semantic_scholar_snippet_search` to prevent leakage past the publication-date cutoff.
3. **Tool call dispatch** – calls `session.call_tool(tool_name, parsed_args)` and parses the result via `extract_tool_result`.
4. **Filter application** – calls `result_filter.filter(tool_result, tool_name)` for tools where `result_filter.should_filter_tool(tool_name)` is `True`.
5. **Pagination** – for `serper_google_webpage_search`, fetches additional pages until the requested number of post-filter results is satisfied.

### When to update `ToolExecutor` for a new tool

If you add a new tool to the MCP server you must check whether `ToolExecutor` needs changes:

| Change needed | Where |
|---------------|-------|
| New tool uses a non-standard result key (not `organic` or `data`) | `_execute_tool_with_filtering`: add a branch to read the correct key |
| New tool needs pre-call argument manipulation | `execute_tool_call`: add a conditional block before `_execute_tool_with_filtering` |
| New list-search tool should paginate | `_execute_tool_with_filtering`: route to `_execute_paginated_search` |
| New single-result tool (like Jina) needs special post-filter handling | `_execute_tool_with_filtering`: add an `if tool_name == "..."` branch |

The pagination implementation in `_execute_paginated_search` is currently specific to `serper_google_webpage_search`. If a new search tool supports pagination via a `page` parameter, add a similar routing condition in `_execute_tool_with_filtering`.

---

## Prompts (`prompts/`)

`prompts/research_assistant.py` defines the system prompt used by `MCPClient`. If you change the scope of the benchmark or add new tools, update the system prompt so the LLM knows what tools are available and what constraints apply (e.g., do not cite sources published after a given date).
