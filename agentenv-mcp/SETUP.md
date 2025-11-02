# MCP Environment Setup

## For Agent Projects

### 1. Install

```bash
pip install -e agentenv-mcp
```

### 2. Configure mcp.json

Place in your agent project root:

```json
{
  "mcpServers": {
    "agentgym-mcp": {
      "command": "python",
      "args": [
        "-m",
        "agentenv_mcp.mcp_launch"
      ],
      "transport": "sse"
    }
  }
}
```

**Custom host/port:**
```json
{
  "mcpServers": {
    "agentgym-mcp": {
      "command": "python",
      "args": [
        "-m",
        "agentenv_mcp.mcp_launch",
        "--host",
        "0.0.0.0",
        "--port",
        "8080"
      ],
      "transport": "sse"
    }
  }
}
```

### 3. Environment Variables (Optional)

```bash
cp agentenv-mcp/.env.example agentenv-mcp/.env
# Edit .env if needed
```

### 4. Start

Your agent framework will auto-start the server via mcp.json, or start manually:

```bash
mcp --host 127.0.0.1 --port 8000
```

### 5. Verify

```bash
curl http://localhost:8000/health
```

## Transport Modes

- **SSE** (Server-Sent Events): Real-time streaming, recommended for agents
- **REST**: Standard HTTP, good for testing

Set via `"transport"` in mcp.json.
