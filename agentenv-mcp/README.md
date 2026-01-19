# AgentEnv-MCP

**Bidirectional MCP (Model Context Protocol) wrapper for AgentGym environments.**

This package provides two complementary wrappers:

1. **AgentEnvToMCP**: Expose any AgentGym environment as an MCP server
2. **MCPToAgentEnv**: Adapt any MCP server into an AgentGym-compatible environment

## Installation

```bash
pip install -e ".[dev,sciworld]"
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AgentEnv-MCP Wrappers                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────┐              ┌─────────────────────┐               │
│  │   AgentEnvToMCP     │              │    MCPToAgentEnv    │               │
│  │   (Export)          │              │    (Import)         │               │
│  ├─────────────────────┤              ├─────────────────────┤               │
│  │ BaseEnvClient ──────┼──► MCP      │ MCP Server ─────────┼──► BaseEnvClient│
│  │                     │    Server   │                     │                │
│  │ • reset()           │    Tools:   │ MCP Tools become:   │                │
│  │ • step()            │    • reset  │ • FUNCTION_DESC     │                │
│  │ • observe()         │    • step   │ • ActionFormat      │                │
│  │                     │    • observe│ • step() mapping    │                │
│  └─────────────────────┘              └─────────────────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Usage

### Export: AgentEnv as MCP Server

```python
from agentenv_mcp import AgentEnvMCPServer
from agentenv.envs.sciworld import SciworldEnvClient

# Create MCP server from any BaseEnvClient
server = AgentEnvMCPServer(
    env_client_cls=SciworldEnvClient,
    client_args={"env_server_base": "http://localhost:8000", "data_len": 100},
)

# Run as MCP server (stdio transport)
server.run()
```

### Import: MCP Server as AgentEnv

```python
from agentenv_mcp import MCPEnvClient, MCPTask

# Connect to any MCP server and use as AgentEnv
client = MCPEnvClient(
    mcp_server_command=["python", "-m", "my_mcp_server"],
    action_format="function_calling",
)

# Use with standard AgentGym evaluation
task = MCPTask(client_args={...})
```

## Examples

See `examples/` for complete demonstrations:

- `sciworld_mcp_server.py` - SciWorld exposed as MCP server
- `mcp_client_demo.py` - Using an MCP server as AgentEnv

## Testing

```bash
pytest tests/ -v
```

## Compatibility

This wrapper is designed to be fully compatible with:

- `BaseEnvClient` interface from `agentenv.controller`
- `BaseTask` for experience generation
- All `ActionFormat` types (REACT, FUNCTION_CALLING, CODE_AS_ACTION)
- `Agent` and `APIAgent` for evaluation
