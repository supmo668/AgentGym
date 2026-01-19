"""
AgentEnvToMCP: Wrapper to expose any AgentGym BaseEnvClient as an MCP server.

This allows external MCP-compatible agents to interact with AgentGym environments
using the standard MCP protocol.
"""

import asyncio
import json
from typing import Any, Callable, Type

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    TextContent,
    Tool,
)

from .schema_utils import function_desc_to_mcp_tool


class AgentEnvMCPServer:
    """
    Wraps an AgentGym BaseEnvClient as an MCP server.
    
    This server exposes the environment's actions as MCP tools, plus
    management tools for reset/observe operations.
    
    MCP Tools provided:
    - env_reset(task_idx: int) -> Reset environment to a specific task
    - env_step(action: str) -> Execute an action (raw format)
    - env_observe() -> Get current observation
    - Plus all environment-specific action tools from FUNCTION_DESCRIPTION
    
    Example:
        >>> from agentenv.envs.sciworld import SciworldEnvClient, SCIWORLD_FUNCTION_DESCRIPTION
        >>> server = AgentEnvMCPServer(
        ...     env_client_cls=SciworldEnvClient,
        ...     client_args={"env_server_base": "http://localhost:8000", "data_len": 100},
        ...     function_descriptions=SCIWORLD_FUNCTION_DESCRIPTION,
        ... )
        >>> server.run()
    """
    
    def __init__(
        self,
        env_client_cls: Type,
        client_args: dict[str, Any],
        function_descriptions: list[dict[str, Any]] | None = None,
        env_name: str = "agentenv",
        action_format: str = "function_calling",
    ):
        """
        Initialize the MCP server wrapper.
        
        Args:
            env_client_cls: The BaseEnvClient class to wrap
            client_args: Arguments to pass to the client constructor
            function_descriptions: Optional list of function descriptions for action tools.
                                   If the client has an adapter_cls with these, they'll be used.
            env_name: Name for the environment (used in server identification)
            action_format: Action format to use ("react", "function_calling", "code_as_action")
        """
        self.env_client_cls = env_client_cls
        self.client_args = client_args
        self.env_name = env_name
        self.action_format = action_format
        
        # Try to get function descriptions from the adapter class if not provided
        self.function_descriptions = function_descriptions or []
        
        # Lazily created client
        self._client = None
        self._server = Server(f"agentenv-{env_name}")
        
        self._setup_handlers()
    
    @property
    def client(self):
        """Lazily create the environment client."""
        if self._client is None:
            self._client = self.env_client_cls(
                **self.client_args,
                action_format=self.action_format,
            )
        return self._client
    
    def _setup_handlers(self):
        """Set up MCP tool handlers."""
        
        @self._server.list_tools()
        async def list_tools() -> list[Tool]:
            """List all available tools."""
            tools = [
                # Core environment management tools
                Tool(
                    name="env_reset",
                    description="Reset the environment to a specific task index.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task_idx": {
                                "type": "integer",
                                "description": "Index of the task to reset to (0 to env_size-1)",
                            }
                        },
                        "required": ["task_idx"],
                    },
                ),
                Tool(
                    name="env_step",
                    description="Execute a raw action string in the environment.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": "The action to execute",
                            }
                        },
                        "required": ["action"],
                    },
                ),
                Tool(
                    name="env_observe",
                    description="Get the current observation from the environment.",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
                Tool(
                    name="env_info",
                    description="Get environment information (size, current state).",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
            ]
            
            # Add environment-specific action tools
            for func_desc in self.function_descriptions:
                mcp_tool = function_desc_to_mcp_tool(func_desc)
                tools.append(Tool(
                    name=f"action_{mcp_tool['name']}",
                    description=mcp_tool["description"],
                    inputSchema=mcp_tool["inputSchema"],
                ))
            
            return tools
        
        @self._server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[CallToolResult]:
            """Handle tool calls."""
            try:
                if name == "env_reset":
                    return await self._handle_reset(arguments)
                elif name == "env_step":
                    return await self._handle_step(arguments)
                elif name == "env_observe":
                    return await self._handle_observe()
                elif name == "env_info":
                    return await self._handle_info()
                elif name.startswith("action_"):
                    return await self._handle_action(name[7:], arguments)
                else:
                    return [CallToolResult(
                        content=[TextContent(type="text", text=f"Unknown tool: {name}")],
                        isError=True,
                    )]
            except Exception as e:
                return [CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {str(e)}")],
                    isError=True,
                )]
    
    async def _handle_reset(self, arguments: dict[str, Any]) -> list[CallToolResult]:
        """Handle env_reset tool call."""
        task_idx = arguments.get("task_idx", 0)
        result = self.client.reset(task_idx)
        observation = self.client.observe()
        
        return [CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({
                    "status": "reset",
                    "task_idx": task_idx,
                    "observation": observation,
                    "reset_info": result if isinstance(result, dict) else {},
                }, indent=2)
            )],
            isError=False,
        )]
    
    async def _handle_step(self, arguments: dict[str, Any]) -> list[CallToolResult]:
        """Handle env_step tool call."""
        action = arguments.get("action", "")
        step_output = self.client.step(action)
        
        return [CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({
                    "observation": step_output.state,
                    "reward": step_output.reward,
                    "done": step_output.done,
                }, indent=2)
            )],
            isError=False,
        )]
    
    async def _handle_observe(self) -> list[CallToolResult]:
        """Handle env_observe tool call."""
        observation = self.client.observe()
        
        return [CallToolResult(
            content=[TextContent(type="text", text=observation)],
            isError=False,
        )]
    
    async def _handle_info(self) -> list[CallToolResult]:
        """Handle env_info tool call."""
        return [CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({
                    "env_name": self.env_name,
                    "env_size": len(self.client),
                    "action_format": self.action_format,
                    "available_actions": len(self.function_descriptions),
                }, indent=2)
            )],
            isError=False,
        )]
    
    async def _handle_action(
        self, action_name: str, arguments: dict[str, Any]
    ) -> list[CallToolResult]:
        """Handle environment-specific action tool calls."""
        # Find the function description for this action
        func_desc = None
        for fd in self.function_descriptions:
            if fd["name"] == action_name:
                func_desc = fd
                break
        
        if func_desc is None:
            return [CallToolResult(
                content=[TextContent(type="text", text=f"Unknown action: {action_name}")],
                isError=True,
            )]
        
        # Format the action based on action_format
        if self.action_format == "function_calling":
            # Format as JSON function call
            action_str = json.dumps({
                "thought": arguments.get("thought", "Executing action"),
                "function_name": action_name,
                "arguments": {k: v for k, v in arguments.items() if k != "thought"},
            })
        else:
            # For react format, try to construct the action string
            # This is environment-specific and may need customization
            action_str = self._format_react_action(action_name, arguments)
        
        step_output = self.client.step(action_str)
        
        return [CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({
                    "action": action_name,
                    "observation": step_output.state,
                    "reward": step_output.reward,
                    "done": step_output.done,
                }, indent=2)
            )],
            isError=False,
        )]
    
    def _format_react_action(self, action_name: str, arguments: dict[str, Any]) -> str:
        """Format an action in ReAct style. Override for environment-specific formatting."""
        args_str = " ".join(str(v) for v in arguments.values() if v)
        thought = arguments.get("thought", "")
        action = f"{action_name} {args_str}".strip() if args_str else action_name
        return f"Thought:\n{thought}\n\nAction:\n{action}"
    
    def run(self):
        """Run the MCP server using stdio transport."""
        async def main():
            async with stdio_server() as (read_stream, write_stream):
                await self._server.run(
                    read_stream,
                    write_stream,
                    self._server.create_initialization_options(),
                )
        
        asyncio.run(main())
    
    async def run_async(self):
        """Run the MCP server asynchronously."""
        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(
                read_stream,
                write_stream,
                self._server.create_initialization_options(),
            )
