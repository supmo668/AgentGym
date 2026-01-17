"""LangChain tool creation from AgentEnv wrappers.

This module converts AgentEnv tools into LangChain-compatible tools
that can be used with LangChain's agent framework.
"""

from typing import Any, Callable, Optional, Type
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool, StructuredTool

from ..base import BaseToolWrapper
from ..models import ActionTool, ToolResult


class AgentEnvTool(BaseTool):
    """LangChain tool wrapper for a single AgentEnv action.
    
    This tool wraps a single action from an AgentEnv environment,
    allowing it to be used in LangChain agent chains.
    
    Attributes:
        name: Tool name (from the action)
        description: Tool description
        wrapper: Reference to the parent tool wrapper
        action_name: Original action name in the environment
    """
    
    name: str
    description: str
    wrapper: BaseToolWrapper
    action_name: str
    
    class Config:
        arbitrary_types_allowed = True
    
    def _run(self, *args, **kwargs) -> str:
        """Execute the tool synchronously.
        
        Returns:
            String output from the environment
        """
        result = self.wrapper.execute_tool(self.action_name, **kwargs)
        
        if result.success:
            output = result.output
            if result.done:
                output += "\n\n[Task completed!]"
            return output
        else:
            return f"Error: {result.error}"
    
    async def _arun(self, *args, **kwargs) -> str:
        """Execute the tool asynchronously (delegates to sync)."""
        return self._run(*args, **kwargs)


class AgentEnvStructuredTool(BaseTool):
    """LangChain tool with structured input for parameterized actions.
    
    Use this for actions that take parameters (e.g., "go to <object>").
    """
    
    name: str
    description: str
    wrapper: BaseToolWrapper
    action_template: str
    args_schema: Optional[Type[BaseModel]] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def _run(self, **kwargs) -> str:
        """Execute the tool with parameters."""
        # Format action name from template and parameters
        action_name = self.action_template.format(**kwargs)
        result = self.wrapper.execute_tool(action_name, **kwargs)
        
        if result.success:
            output = result.output
            if result.done:
                output += "\n\n[Task completed!]"
            return output
        else:
            return f"Error: {result.error}"
    
    async def _arun(self, **kwargs) -> str:
        """Execute the tool asynchronously."""
        return self._run(**kwargs)


def create_langchain_tools(
    wrapper: BaseToolWrapper,
    include_check_action: bool = True,
) -> list[BaseTool]:
    """Create LangChain tools from an AgentEnv wrapper.
    
    This function converts all available actions in the wrapper
    into LangChain Tool objects that can be used with agents.
    
    Args:
        wrapper: The AgentEnv tool wrapper
        include_check_action: Whether to include the "check available actions" tool
        
    Returns:
        List of LangChain Tool objects
        
    Example:
        >>> from agentenv_wrapper import BabyAIToolWrapper
        >>> from agentenv_wrapper.langchain import create_langchain_tools
        >>> 
        >>> wrapper = BabyAIToolWrapper("http://localhost:8080")
        >>> wrapper.reset(task_id=1)
        >>> tools = create_langchain_tools(wrapper)
        >>> print([t.name for t in tools])
        ['turn_left', 'turn_right', 'move_forward', ...]
    """
    tools = []
    
    for action_tool in wrapper.get_available_tools():
        # Skip check action if not wanted
        if not include_check_action and "check" in action_tool.name.lower():
            continue
        
        # Convert action name to valid tool name (replace spaces with underscores)
        tool_name = action_tool.name.replace(" ", "_").replace("-", "_")
        
        tool = AgentEnvTool(
            name=tool_name,
            description=action_tool.description,
            wrapper=wrapper,
            action_name=action_tool.name,
        )
        tools.append(tool)
    
    return tools


def create_dynamic_tools_callback(
    wrapper: BaseToolWrapper,
) -> Callable[[], list[BaseTool]]:
    """Create a callback that returns current available tools.
    
    This is useful for environments with dynamic action spaces,
    where available actions change based on state.
    
    Args:
        wrapper: The AgentEnv tool wrapper
        
    Returns:
        Callback function that returns current tools
        
    Example:
        >>> get_tools = create_dynamic_tools_callback(wrapper)
        >>> # After each action, get updated tools:
        >>> current_tools = get_tools()
    """
    def get_current_tools() -> list[BaseTool]:
        return create_langchain_tools(wrapper)
    
    return get_current_tools
