# Remote MCP Servers for OpenAI Deep Research

Two specialized Remote MCP servers for OpenAI Deep Research, each providing `search` and `fetch` tools.

## Servers

| Server | Port | Search API | Fetch API | Use Case |
|--------|------|------------|-----------|----------|
| **Serper+Jina** | 8001 | Serper (Google Search) | Jina Reader | General web search |
| **Semantic Scholar+Jina** | 8002 | Semantic Scholar Snippet | Jina Reader | Academic paper snippets |

## Quick Start

### Setup

1. **API Keys** - Add to `.env` at the project root:
   ```bash
   SERPER_API_KEY=your_key
   S2_API_KEY=your_key
   JINA_API_KEY=your_key
   MCP_AUTH_TOKEN=your_secret_token  # Optional: for production authentication
   ```

2. **Prepare the Cochrane titles filter file** – the servers load this JSON file at startup and use it to block search/fetch results whose titles match any Cochrane review. Without it the servers will refuse to start (fail-fast). These titles are needed to filter out leakage and facilitate our clean room evaluation protocol.

   The file must be placed at:
   ```
   data/filter_data/cochrane_titles.json
   ```
   relative to the project root (i.e., `SciConBench/data/filter_data/cochrane_titles.json`).

   It must be a JSON array of Cochrane review title strings:
   ```json
   ["Title of review A", "Title of review B", ...]
   ```

   **Generating the file from the HuggingFace dataset:**
   ```python
   from datasets import load_dataset
   import json
   from pathlib import Path

   ds = load_dataset("hayoungjung/SciConBench", "benchmark", split="test")
   titles = list(ds["title"])

   out = Path("data/filter_data/cochrane_titles.json")
   out.parent.mkdir(parents=True, exist_ok=True)
   out.write_text(json.dumps(titles, indent=2))
   print(f"Saved {len(titles)} titles to {out}")
   ```

   Run this once from the project root before starting the servers. It should contain all ~9K titles (as of May 15th).

   > **Note:** The `--cochrane-titles` CLI flag (used by `query_single` / `query_batch` for the local MCP server) is **not** needed for deep research models. The remote MCP servers automatically load the titles file at startup, so no additional flag is required when using OpenAI Deep Research.

3. **Install dependencies**:
   ```bash
   pip install fastmcp python-dotenv requests diskcache pydantic
   ```

### Run Servers

```bash
# Serper+Jina (port 8001)
python -m sciconharness.remote_mcp_servers.serper_jina.main

# Semantic Scholar+Jina (port 8002)
python -m sciconharness.remote_mcp_servers.semantic_scholar_jina.main
```

**Health checks:**
- `curl http://localhost:8001/health`
- `curl http://localhost:8002/health`

## Tools

Both servers implement:

- **`search`**: Find relevant documents/webpages
  - Serper: Web search via Google Search
  - Semantic Scholar: Academic paper snippet search
- **`fetch`**: Retrieve full document content using Jina Reader API

## Exposing Both Servers with Nginx Reverse Proxy

Use nginx as a reverse proxy to route different paths to different servers, then expose through a single ngrok tunnel. This gives you one ngrok URL with two different endpoints. Ideally, host this on some VM with open
ports on Azure or AWS, rather than some server at an academic institution to avoid firewall issues.

### Setup Steps

1. **Install nginx** (if not already installed) [DONE]
   ```bash
   sudo apt-get install nginx
   ```

2. **Start both servers**:
   ```bash
   # Terminal 1: Start Serper+Jina server
   python -m sciconharness.remote_mcp_servers.serper_jina.main --port 8001
   
   # Terminal 2: Start Semantic Scholar+Jina server
   python -m sciconharness.remote_mcp_servers.semantic_scholar_jina.main --port 8002
   ```

3. **Test nginx configuration** (update the path to your clone):
   ```bash
   sudo nginx -t -c /path/to/SciConBench/sciconharness/remote_mcp_servers/nginx.conf
   ```

4. **Start nginx** with the provided configuration:
   ```bash
   sudo nginx -c /path/to/SciConBench/sciconharness/remote_mcp_servers/nginx.conf
   ```

5. **Expose nginx (port 8080) via ngrok**:
   ```bash
   ngrok http 8080 --pooling-enabled
   ```

6. **Use the URLs** with OpenAI Deep Research:
   - **Serper+Jina**: `https://your-ngrok-url.ngrok-free.dev/serper/mcp`
   - **Semantic Scholar+Jina**: `https://your-ngrok-url.ngrok-free.dev/semantic/mcp`

### Example Configuration

```json
{
  "tools": [
    {
      "type": "mcp",
      "server_label": "SerperJinaMCP",
      "server_url": "https://princeton-uwashington-random-url.ngrok-free.dev/serper/mcp",
      "allowed_tools": ["search", "fetch"],
      "require_approval": "never",
      "authorization": "your_MCP_AUTH_TOKEN"
    },
    {
      "type": "mcp",
      "server_label": "SemanticScholarJinaMCP",
      "server_url": "https://princeton-uwashington-random-url.ngrok-free.dev/semantic/mcp",
      "allowed_tools": ["search", "fetch"],
      "require_approval": "never",
      "authorization": "your_MCP_AUTH_TOKEN"
    }
  ]
}
```

### Nginx Configuration

The `nginx.conf` file routes:
- `/serper/*` → `localhost:8001` (Serper+Jina server)
- `/semantic/*` → `localhost:8002` (Semantic Scholar+Jina server)
- `/health` → Health check endpoint

The configuration handles Server-Sent Events (SSE) for MCP protocol and properly forwards headers.

## Integration with OpenAI Deep Research

Once your server is publicly accessible:

1. **Get your public URL** (from ngrok, cloudflare, or your cloud deployment)
2. **Set authentication token** in your `.env`:
   ```bash
   MCP_AUTH_TOKEN=your_secure_random_token_here
   ```
3. **Register with OpenAI Deep Research**:
   ```json
   {
     "server_url": "https://your-public-url.com/mcp",
     "authorization": "your_secure_random_token_here"
   }
   ```

**Example for Semantic Scholar+Jina server:**
```json
{
  "server_url": "https://your-server.com/mcp",
  "authorization": "sk_live_abc123xyz789"
}
```

**Note:** The `/mcp` path is required - it's the MCP protocol endpoint.

See [OpenAI MCP docs](https://platform.openai.com/docs/mcp) for details.

### Disabling Filtering

The remote servers apply the Cochrane title filter by default once the `cochrane_titles.json` file is loaded. If you need to run queries **without** filtering (e.g., for ablation studies without the clean room evaluation protocol), call the `disable_filtering` utility against both running servers:

```bash
python -m sciconharness.remote_mcp_servers.disable_filtering \
    --serper-server-base  "https://your-public-url.com/serper" \
    --semantic-server-base "https://your-public-url.com/semantic" \
    --mcp-auth-token "$MCP_AUTH_TOKEN"
```

The script sends a `/configure` request with empty `source_title` and `publication_date` fields to each server, which clears the active filter configuration. It then calls `/verify-config` on both servers to confirm that filtering is disabled, and prints a pass/fail summary.

Additional options:
- `--no-verify` – skip the post-disable verification step.
- `--restart` – print instructions for restarting the server processes manually (full restart is the most reliable way to reset state).

Server base URLs can also be set via environment variables (`SERPER_SERVER_BASE`, `SEMANTIC_SERVER_BASE`) in the `.env` file instead of passing them as CLI arguments each time.

## Features

- HTTP transport for remote access
- Disk-based caching (disable with `--no-cache`)
- Automatic retry with exponential backoff
- Health check endpoints

## Troubleshooting

- **Port conflicts**: Use `--port` to change ports
- **API errors**: Verify `.env` file location and API keys
- **Rate limits**: Semantic Scholar server auto-retries on 429 errors

---

## Extending with New Tools and APIs

Adding new tools to the remote MCP servers follows the same overall pattern as the local `mcp_server/` (see `mcp_server/README.md` for the API-side steps), but requires additional changes specific to the remote server architecture.

> **Shared code:** The remote servers are thin wrappers — they import and reuse the API functions and cache layer directly from `mcp_server/` and the filter from `mcp_client/`:
> - `mcp_server/apis/serper_apis.py`, `jina_apis.py`, `semantic_scholar_apis.py` — same HTTP call implementations used by both the local and remote servers
> - `mcp_server/cache.py` — same disk cache
> - `mcp_client/filters/cochrane.py` — `CochraneResultFilter` applied server-side inside each remote tool function
>
> This means an API implemented once in `mcp_server/apis/` is immediately available to both the local server (`mcp_server/main.py`) and either remote server just by importing it.

### Step 1 – Add the API call in `mcp_server/apis/`

Implement the new API function in an `apis/` module just as you would for the local server (see `mcp_server/README.md`, Step 1). Because the remote servers import directly from `mcp_server/apis/`, no duplication is needed.

### Step 2 – Expose the tool in the remote server's `main.py`

Each remote server (`serper_jina/main.py`, `semantic_scholar_jina/main.py`) registers its own FastMCP tools. Add your new tool to the appropriate server file, importing the shared API function:

```python
from sciconharness.mcp_server.apis.my_api import my_api_function

@mcp.tool()
async def my_new_tool(query: Annotated[str, "Search query"]) -> dict:
    """Description shown to the LLM (OpenAI Deep Research)."""
    filter_config = get_global_filter_config()
    raw_result = await my_api_function(query)
    # Apply filtering here (see Step 3)
    return filtered_result
```

If the new tool is sufficiently distinct from both existing servers, create a new server module under `remote_mcp_servers/my_server/main.py`, following the structure of `serper_jina/main.py`.

### Step 3 – Apply filtering inside the tool function

Unlike the local server (where `ToolExecutor` applies the filter client-side), the remote servers apply `CochraneResultFilter` **server-side** inside each tool function, because the client is OpenAI Deep Research and has no access to the Python filter code.

The pattern used by existing tools is:

```python
filter_config = get_global_filter_config()
result_filter = CochraneResultFilter(
    title_filter_list=cochrane_titles,
    source_title=filter_config.get("source_title"),
    publication_date=filter_config.get("publication_date"),
)
filtered = result_filter.filter(raw_result_dict, "my_new_tool")
```

For this to work:

1. **Register the tool name** in `CochraneResultFilter.filtered_tools` (in `mcp_client/filters/cochrane.py`) so `should_filter_tool("my_new_tool")` returns `True`.
2. **Add a dispatch branch** in `CochraneResultFilter.filter()` for the new tool, providing the correct result key and field accessor functions (same as Step 3 of `mcp_server/README.md`).
3. **Track filtered URLs** – after calling `result_filter.filter(...)`, call `track_filtered_urls(result_filter.get_filtered_links())` so that URLs blocked in search results are also blocked in subsequent fetch calls.

### Step 4 – Update `mcp_client/utils/tool_execution.py`

Even though filtering runs server-side for remote servers, `ToolExecutor` is still used when the local `MCPClient` connects to any MCP server via stdio or HTTP. If the new tool has:

- A non-standard result key (not `organic` / `data`)
- Tool-specific argument manipulation (e.g., year-capping)
- Pagination support

…then update `ToolExecutor` as described in `mcp_server/README.md` (Step 4) and in `mcp_client/README.md` (the "When to update `ToolExecutor`" table).

### Step 5 – Expose the new server via nginx (optional)

If the new tool runs on a separate port, add a new `location` block in `nginx.conf`:

```nginx
location /my_server/ {
    rewrite ^/my_server(/.*)$ $1 break;
    proxy_pass http://localhost:8003;
    # ... (copy SSE and header settings from existing blocks)
}
```

Then reference it in the OpenAI Deep Research configuration:

```json
{
  "server_url": "https://your-public-url.com/my_server/mcp",
  "authorization": "your_MCP_AUTH_TOKEN"
}
```

### Step 6 – Update `utils.py` if needed

`remote_mcp_servers/utils.py` provides shared state and middleware used by all servers (`_global_filter_config`, `_filtered_urls`, `_allowed_fetch_urls`, `FilterConfigMiddleware`, `ToolSchemaFixMiddleware`, `AuthMiddleware`). If your new server needs a different authentication scheme or additional per-request context, add the necessary helpers here and apply the middleware in your server's `main.py`.

### Step 7 – Add the API key to `.env` (if necessary)

```
MY_NEW_API_KEY=your_key_here
```
