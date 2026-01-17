"""AgentEnv Wrapper - Generic tool-based wrapper for AgentGym environments.

This package provides a unified interface to adapt AgentGym environments into
tool-based environments suitable for agent frameworks like LangChain.

Key Components:
    - BaseToolWrapper: Base class for environment-to-tool adapters
    - ActionTool: Represents an environment action as a callable tool
    - ToolEnvironmentWrapper: Wraps any BaseEnvClient as a tool-based environment
    - BabyAIToolWrapper: Example wrapper for BabyAI environment

Example Usage:
    from agentenv_wrapper import BabyAIToolWrapper
    from langchain.agents import initialize_agent
    
    # Create wrapper
    wrapper = BabyAIToolWrapper(env_server_base="http://localhost:8000")
    
    # Get LangChain tools
    tools = wrapper.get_langchain_tools()
    
    # Use with LangChain agent
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description")
"""

from .models import ActionTool, ToolDefinition, ToolResult, EnvState
from .base import BaseToolWrapper, ToolEnvironmentWrapper
from .adapters.babyai import BabyAIToolWrapper

__all__ = [
    # Models
    "ActionTool",
    "ToolDefinition", 
    "ToolResult",
    "EnvState",
    # Wrappers
    "BaseToolWrapper",
    "ToolEnvironmentWrapper",
    # Adapters
    "BabyAIToolWrapper",
]

__version__ = "0.1.0"
