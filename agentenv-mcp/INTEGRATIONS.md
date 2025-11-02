# Integration with Popular Tools

## LangChain MCP Adapter

Use `langchain-mcp-adapter` to integrate with LangChain:

```bash
pip install langchain-mcp-adapter
```

```python
from langchain_mcp_adapter import MCPClient
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# Connect to AgentGym MCP server
client = MCPClient("http://localhost:8000")

# Use MCP tools with LangChain
tools = client.get_tools()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to Milvus collections."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

# Execute
result = executor.invoke({"input": "List all collections"})
```

## Direct HTTP Integration

For minimal dependencies:

```python
import requests

base_url = "http://localhost:8000"

# Create environment
response = requests.post(f"{base_url}/create", json={"id": 0})
env_id = response.json()

# Execute action
response = requests.post(f"{base_url}/step", json={
    "env_idx": env_id,
    "action": "Action: list_collections with Action Input: {}"
})
result = response.json()
print(result["observation"])
```

## AgentGym Native

Simplest integration with AgentGym:

```python
from agentenv.envs.mcp import MCPEnvClient

client = MCPEnvClient("http://localhost:8000", data_len=10)
client.reset(0)
result = client.step("Action: list_collections with Action Input: {}")
print(result.state)
```

## MCP Protocol Compatible Clients

Any MCP-compatible client can connect via SSE:

```javascript
// Claude Desktop, Cline, or other MCP clients
// Configure in mcp.json:
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

## Environment Variables

Optional `.env` configuration:

```bash
# Copy template
cp .env.example .env

# Edit if needed
MCP_HOST=0.0.0.0
MCP_PORT=8000
LOG_LEVEL=info
```

## Minimal Example

Complete working example in < 10 lines:

```python
from agentenv.envs.mcp import MCPEnvClient

c = MCPEnvClient("http://localhost:8000", data_len=10)
c.reset(0)
r = c.step("Action: list_collections with Action Input: {}")
print(f"Collections: {r.state}")
c.close()
```
