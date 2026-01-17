"""Base wrapper classes for adapting AgentEnv to tool-based interfaces.

This module provides the foundation for converting AgentGym environments
into tool-based interfaces that can be used with agent frameworks.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from .models import ActionTool, EnvState, ToolDefinition, ToolResult


class BaseToolWrapper(ABC):
    """Abstract base class for environment-to-tool wrappers.
    
    Subclasses must implement methods to:
    - Get current environment state
    - List available tools/actions
    - Execute tools/actions
    
    This design allows any AgentGym environment to be adapted into a
    tool-based interface suitable for LLM agents.
    """
    
    def __init__(self, env_name: str = "AgentEnv"):
        """Initialize the wrapper.
        
        Args:
            env_name: Human-readable name for this environment
        """
        self.env_name = env_name
        self._tools: dict[str, ActionTool] = {}
        self._state: Optional[EnvState] = None
    
    @abstractmethod
    def reset(self, task_id: int = 0) -> EnvState:
        """Reset the environment to initial state.
        
        Args:
            task_id: Optional task/scenario identifier
            
        Returns:
            Initial environment state
        """
        pass
    
    @abstractmethod
    def get_state(self) -> EnvState:
        """Get the current environment state.
        
        Returns:
            Current state including observation and available actions
        """
        pass
    
    @abstractmethod
    def get_available_tools(self) -> list[ActionTool]:
        """Get list of currently available tools/actions.
        
        Returns:
            List of ActionTool objects representing valid actions
        """
        pass
    
    @abstractmethod
    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool/action in the environment.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool-specific parameters
            
        Returns:
            Result of tool execution including new observation
        """
        pass
    
    def get_tool_definitions(self) -> list[ToolDefinition]:
        """Get tool definitions for all available tools.
        
        Returns:
            List of ToolDefinition objects
        """
        return [tool.to_definition() for tool in self.get_available_tools()]
    
    def get_system_prompt(self) -> str:
        """Get a system prompt describing the environment and available tools.
        
        Returns:
            System prompt string
        """
        state = self.get_state()
        tools = self.get_available_tools()
        
        tool_descriptions = "\n".join(
            f"- {tool.name}: {tool.description}"
            for tool in tools
        )
        
        return f"""You are an agent interacting with the {self.env_name} environment.

Current Goal: {state.goal}

Available Actions:
{tool_descriptions}

Current Observation:
{state.observation}

Select an action to take based on your observations and goal."""


class ToolEnvironmentWrapper(BaseToolWrapper):
    """Generic wrapper that adapts a BaseEnvClient to tool-based interface.
    
    This wrapper works with the standard AgentGym BaseEnvClient interface
    and automatically converts the action space to tools.
    """
    
    def __init__(
        self,
        env_server_base: str,
        env_name: str = "AgentEnv",
        timeout: int = 300,
    ):
        """Initialize the wrapper with an environment server.
        
        Args:
            env_server_base: Base URL of the environment server
            env_name: Human-readable name for this environment
            timeout: Request timeout in seconds
        """
        super().__init__(env_name=env_name)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self._env_id: Optional[int] = None
        self._current_observation: str = ""
        self._available_actions: list[str] = []
        self._goal: str = ""
        self._reward: float = 0.0
        self._done: bool = False
        self._step_count: int = 0
    
    def _create_action_tool(self, action_name: str) -> ActionTool:
        """Create an ActionTool for a given action name.
        
        Args:
            action_name: Name of the action
            
        Returns:
            ActionTool that executes this action
        """
        def handler(**kwargs) -> ToolResult:
            return self.execute_tool(action_name, **kwargs)
        
        return ActionTool(
            name=action_name,
            description=f"Execute the '{action_name}' action in the environment",
            handler=handler,
            parameters={},
        )
    
    def _parse_available_actions(self, observation: str) -> list[str]:
        """Parse available actions from observation text.
        
        Override this method for environment-specific parsing.
        
        Args:
            observation: Raw observation text
            
        Returns:
            List of available action names
        """
        # Default implementation: look for "Available actions: [...]" pattern
        import re
        match = re.search(r'Available actions:\s*\[(.*?)\]', observation, re.DOTALL)
        if match:
            actions_str = match.group(1)
            # Parse action names from the list
            actions = re.findall(r'"([^"]+)"', actions_str)
            return actions
        return []
    
    def get_state(self) -> EnvState:
        """Get the current environment state."""
        return EnvState(
            observation=self._current_observation,
            available_actions=self._available_actions,
            goal=self._goal,
            reward=self._reward,
            done=self._done,
            step_count=self._step_count,
        )
    
    def get_available_tools(self) -> list[ActionTool]:
        """Get list of currently available tools/actions."""
        return [
            self._create_action_tool(action)
            for action in self._available_actions
        ]
    
    def reset(self, task_id: int = 0) -> EnvState:
        """Reset the environment to initial state.
        
        This is a generic implementation - subclasses should override
        to handle environment-specific reset logic.
        """
        import requests
        
        # Create environment if needed
        if self._env_id is None:
            response = requests.post(
                f"{self.env_server_base}/create",
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            self._env_id = data.get("id", 0)
        
        # Reset environment
        response = requests.post(
            f"{self.env_server_base}/reset",
            json={"id": self._env_id, "data_idx": task_id},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        
        self._current_observation = data.get("observation", "")
        self._reward = data.get("reward", 0.0)
        self._done = data.get("done", False)
        self._step_count = 0
        
        # Parse available actions from observation
        self._available_actions = self._parse_available_actions(self._current_observation)
        
        # Try to extract goal from observation
        if "Your goal:" in self._current_observation:
            goal_end = self._current_observation.find("\n", self._current_observation.find("Your goal:"))
            if goal_end > 0:
                self._goal = self._current_observation[
                    self._current_observation.find("Your goal:") + 10:goal_end
                ].strip()
        
        return self.get_state()
    
    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool/action in the environment."""
        import requests
        
        if self._env_id is None:
            return ToolResult(
                success=False,
                output="",
                error="Environment not initialized. Call reset() first.",
            )
        
        if tool_name not in self._available_actions:
            return ToolResult(
                success=False,
                output=f"Invalid action '{tool_name}'. Available: {self._available_actions}",
                error=f"Action '{tool_name}' is not currently available.",
            )
        
        try:
            response = requests.post(
                f"{self.env_server_base}/step",
                json={"id": self._env_id, "action": tool_name},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            
            self._current_observation = data.get("observation", "")
            self._reward = data.get("score", data.get("reward", 0.0))
            self._done = data.get("done", False)
            self._step_count += 1
            
            # Update available actions
            self._available_actions = self._parse_available_actions(self._current_observation)
            
            return ToolResult(
                success=True,
                output=self._current_observation,
                reward=self._reward,
                done=self._done,
                metadata={"step": self._step_count},
            )
            
        except requests.RequestException as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Request failed: {str(e)}",
            )
    
    def close(self) -> None:
        """Close the environment and release resources."""
        import requests
        
        if self._env_id is not None:
            try:
                requests.post(
                    f"{self.env_server_base}/close",
                    json={"id": self._env_id},
                    timeout=self.timeout,
                )
            except requests.RequestException:
                pass
            self._env_id = None
