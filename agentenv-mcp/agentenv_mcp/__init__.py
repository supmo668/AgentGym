"""
AgentEnv-MCP: Bidirectional MCP wrapper for AgentGym environments.

This package provides:
- AgentEnvMCPServer: Expose any AgentEnv as an MCP server
- MCPEnvClient: Adapt any MCP server into an AgentEnv client
- MCPTask: Task wrapper for MCP-based environments
"""

__version__ = "0.1.0"

# Lazy imports to avoid requiring all dependencies at import time
def __getattr__(name):
    if name == "AgentEnvMCPServer":
        from agentenv_mcp.agentenv_to_mcp import AgentEnvMCPServer
        return AgentEnvMCPServer
    elif name == "MCPEnvClient":
        from agentenv_mcp.mcp_to_agentenv import MCPEnvClient
        return MCPEnvClient
    elif name == "MCPAdapter":
        from agentenv_mcp.mcp_to_agentenv import MCPAdapter
        return MCPAdapter
    elif name == "MCPTask":
        from agentenv_mcp.mcp_to_agentenv import MCPTask
        return MCPTask
    elif name == "function_desc_to_mcp_tool":
        from agentenv_mcp.schema_utils import function_desc_to_mcp_tool
        return function_desc_to_mcp_tool
    elif name == "mcp_tool_to_function_desc":
        from agentenv_mcp.schema_utils import mcp_tool_to_function_desc
        return mcp_tool_to_function_desc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentEnvMCPServer",
    "MCPEnvClient",
    "MCPAdapter",
    "MCPTask",
    "function_desc_to_mcp_tool",
    "mcp_tool_to_function_desc",
]
