"""
Schema conversion utilities between AgentEnv function descriptions and MCP tool schemas.
"""

from typing import Any


def function_desc_to_mcp_tool(func_desc: dict[str, Any]) -> dict[str, Any]:
    """
    Convert an AgentEnv function description to MCP tool schema.
    
    AgentEnv format:
    {
        "name": "open",
        "description": "Opens a container.",
        "parameters": {
            "type": "object",
            "properties": {
                "obj": {"type": "string", "description": "The container to open."}
            },
            "required": ["obj"]
        }
    }
    
    MCP format:
    {
        "name": "open",
        "description": "Opens a container.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "obj": {"type": "string", "description": "The container to open."}
            },
            "required": ["obj"]
        }
    }
    """
    return {
        "name": func_desc["name"],
        "description": func_desc.get("description", ""),
        "inputSchema": func_desc.get("parameters", {"type": "object", "properties": {}}),
    }


def mcp_tool_to_function_desc(mcp_tool: dict[str, Any]) -> dict[str, Any]:
    """
    Convert an MCP tool schema to AgentEnv function description.
    
    This is the inverse of function_desc_to_mcp_tool.
    """
    input_schema = mcp_tool.get("inputSchema", {"type": "object", "properties": {}})
    
    return {
        "name": mcp_tool["name"],
        "description": mcp_tool.get("description", ""),
        "parameters": input_schema,
    }


def generate_function_descriptions_from_mcp_tools(
    mcp_tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Convert a list of MCP tools to AgentEnv function descriptions.
    """
    return [mcp_tool_to_function_desc(tool) for tool in mcp_tools]


def generate_mcp_tools_from_function_descriptions(
    func_descs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Convert a list of AgentEnv function descriptions to MCP tools.
    """
    return [function_desc_to_mcp_tool(desc) for desc in func_descs]
