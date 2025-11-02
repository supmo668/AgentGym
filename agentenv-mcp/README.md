# MCP Environment

Simulated Model Context Protocol server with Milvus vector collections for AgentGym.

## Quick Start

```bash
# Install
cd agentenv-mcp && pip install -e .

# Start
mcp --host 127.0.0.1 --port 8000

# Test
curl http://localhost:8000/health
```

## Configuration

### For Agent Environment (mcp.json)

Create `mcp.json` in your agent project:

```json
{
  "mcpServers": {
    "agentgym-mcp": {
      "command": "python",
      "args": ["-m", "agentenv_mcp.mcp_launch"],
      "transport": "sse"
    }
  }
}
```

**Options:**
- Add `"--host", "HOST"` to args (default: 127.0.0.1)
- Add `"--port", "PORT"` to args (default: 8000)
- Set `transport` to `"sse"` or `"rest"`

### Environment Variables

Optional. Copy if needed:
```bash
cp .env.example .env
```

Currently no API keys required (simulated data only).

## Usage

### Simplified Client (Recommended)

```python
from agentenv_mcp.utils import SimpleMCPClient

# Context manager - auto cleanup
with SimpleMCPClient() as client:
    result = client.act("list_collections")
    print(result["observation"])
    
    result = client.act("search_collection", 
                       collection_name="documents", 
                       query="machine learning")
    print(result["observation"])
```

### One-Liners

```python
from agentenv_mcp.utils import list_collections, search_collection

print(list_collections())
print(search_collection("documents", "ML", top_k=3))
```

### AgentGym Native

```python
from agentenv.envs.mcp import MCPEnvClient

client = MCPEnvClient(env_server_base="http://127.0.0.1:8000", data_len=10)
client.reset(data_idx=0)
result = client.step("Action: list_collections with Action Input: {}")
client.close()
```

### Available Tools

| Tool | Description |
|------|-------------|
| `list_collections` | List all collections |
| `get_collection_info` | Get collection schema |
| `query_collection` | Query records |
| `search_collection` | Vector similarity search |
| `get_schema` | Get resource schema |
| `get_prompt` | Get prompt template |
| `format_response` | Format output (JSON/text/markdown) |
| `finish` | Complete task |

## Collections

- **documents** (768-dim): ML articles
- **users** (512-dim): User profiles
- **products** (384-dim): Product catalog

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Server status |
| GET | `/sse` | SSE streaming |
| POST | `/sse/message` | JSON-RPC messages |
| POST | `/create` | Create environment |
| POST | `/step` | Execute action |
| POST | `/reset` | Reset environment |

## Architecture

```
LLM Agent → MCPEnvClient → FastAPI Server → MCPEnv → [Collections, Resources]
```

**Components:**
- `mcp_environment.py` - Core environment (gymnasium.Env)
- `mcp_server.py` - FastAPI server (REST + SSE)
- `mcp_server_wrapper.py` - Multi-instance manager
- `mcp_resources.py` - Simulated collections + resources

**Extending:**
```python
# Add tool
def _tool_my_tool(self, param: str, **kwargs) -> str:
    return result

self.tools["my_tool"] = self._tool_my_tool

# Add collection
self.collections["new"] = {"schema": {...}, "data": [...]}
```

## Examples

See `example_usage.py` for complete demos.

## Integrations

See [INTEGRATIONS.md](./INTEGRATIONS.md) for:
- LangChain MCP Adapter usage
- Direct HTTP integration
- MCP protocol-compatible clients
- Additional simplified patterns

## Contributing

See [CONTRIB.md](../CONTRIB.md) for guidelines.

## License

MIT - See [LICENSE](../LICENSE)
