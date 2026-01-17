"""BabyAI-specific tool wrapper implementation.

This module provides a specialized wrapper for the BabyAI environment
that converts its action space into discrete tools.
"""

from typing import Optional
import requests
from requests.exceptions import RequestException

from ..base import BaseToolWrapper
from ..models import ActionTool, EnvState, ToolResult


class BabyAIToolWrapper(BaseToolWrapper):
    """Tool wrapper specifically designed for BabyAI environments.
    
    BabyAI is a gridworld navigation environment where agents must
    follow instructions to complete tasks like "go to the red ball"
    or "pick up the blue key".
    
    This wrapper converts BabyAI's dynamic action space (which depends
    on visible objects) into callable tools.
    
    Example:
        >>> wrapper = BabyAIToolWrapper("http://localhost:8080")
        >>> state = wrapper.reset(task_id=1)
        >>> print(state.goal)
        "go to the red ball"
        >>> tools = wrapper.get_available_tools()
        >>> result = wrapper.execute_tool("move forward")
    """
    
    # Action descriptions for common BabyAI actions
    ACTION_DESCRIPTIONS = {
        "turn left": "Turn 90 degrees to the left",
        "turn right": "Turn 90 degrees to the right", 
        "move forward": "Move one step forward in the current direction",
        "toggle": "Toggle the door/object directly in front of you",
        "drop": "Drop the currently held object",
        "check available actions": "List all currently available actions",
    }
    
    def __init__(
        self,
        env_server_base: str,
        timeout: int = 300,
    ):
        """Initialize the BabyAI tool wrapper.
        
        Args:
            env_server_base: Base URL of the BabyAI environment server
            timeout: Request timeout in seconds
        """
        super().__init__(env_name="BabyAI")
        self.env_server_base = env_server_base.rstrip("/")
        self.timeout = timeout
        
        self._env_id: Optional[int] = None
        self._current_observation: str = ""
        self._available_actions: list[str] = []
        self._goal: str = ""
        self._reward: float = 0.0
        self._score: float = 0.0
        self._done: bool = False
        self._step_count: int = 0
    
    def _post(self, path: str, data: dict) -> dict:
        """Make a POST request to the environment server."""
        data["id"] = self._env_id
        response = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
    
    def _parse_observation(self, raw_observation: str) -> tuple[str, list[str], str]:
        """Parse BabyAI observation to extract components.
        
        Args:
            raw_observation: Raw observation string from server
            
        Returns:
            Tuple of (clean_observation, available_actions, goal)
        """
        import re
        
        observation = raw_observation
        goal = ""
        actions = []
        
        # Extract goal if present
        if "Your goal:" in observation:
            goal_match = re.search(r"Your goal:\s*(.+?)(?:\n|$)", observation)
            if goal_match:
                goal = goal_match.group(1).strip()
                # Remove goal line from observation for cleaner text
                observation = observation.replace(goal_match.group(0), "").strip()
        
        # Extract available actions
        action_match = re.search(
            r'Available actions:\s*\[(.*?)\]', 
            observation, 
            re.DOTALL
        )
        if action_match:
            actions_str = action_match.group(1)
            actions = [a.strip().strip('"') for a in actions_str.split(",") if a.strip()]
            # Remove action list from observation for cleaner text
            observation = observation[:action_match.start()].strip()
        
        return observation, actions, goal
    
    def _get_action_description(self, action: str) -> str:
        """Get a human-readable description for an action.
        
        Args:
            action: Action name
            
        Returns:
            Description of what the action does
        """
        # Check known actions first
        if action in self.ACTION_DESCRIPTIONS:
            return self.ACTION_DESCRIPTIONS[action]
        
        # Generate descriptions for dynamic actions
        if action.startswith("pickup "):
            obj = action[7:]  # Remove "pickup " prefix
            return f"Pick up the {obj}"
        elif action.startswith("go to "):
            target = action[6:]  # Remove "go to " prefix
            return f"Navigate to the {target}"
        elif action.startswith("go through "):
            door = action[11:]  # Remove "go through " prefix
            return f"Walk through the {door}"
        elif action.startswith("toggle and go through "):
            door = action[22:]  # Remove prefix
            return f"Toggle the {door} and walk through it"
        else:
            return f"Execute the '{action}' action"
    
    def _create_action_tool(self, action_name: str) -> ActionTool:
        """Create an ActionTool for a BabyAI action."""
        description = self._get_action_description(action_name)
        
        def handler(**kwargs) -> ToolResult:
            return self.execute_tool(action_name, **kwargs)
        
        return ActionTool(
            name=action_name,
            description=description,
            handler=handler,
            parameters={},
        )
    
    def reset(self, task_id: int = 0) -> EnvState:
        """Reset the BabyAI environment.
        
        Args:
            task_id: Task index (maps to different BabyAI levels/seeds)
            
        Returns:
            Initial environment state
        """
        # Create environment if needed
        if self._env_id is None:
            response = requests.post(
                f"{self.env_server_base}/create",
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            self._env_id = data.get("id", 0)
        
        # Reset to specific task
        data = self._post("reset", {"data_idx": task_id})
        
        raw_obs = data.get("observation", "")
        self._current_observation, self._available_actions, self._goal = \
            self._parse_observation(raw_obs)
        
        self._reward = data.get("reward", 0.0)
        self._score = data.get("score", 0.0)
        self._done = data.get("done", False)
        self._step_count = 0
        
        return self.get_state()
    
    def get_state(self) -> EnvState:
        """Get the current BabyAI environment state."""
        return EnvState(
            observation=self._current_observation,
            available_actions=self._available_actions,
            goal=self._goal,
            reward=self._score,
            done=self._done,
            step_count=self._step_count,
            metadata={
                "step_reward": self._reward,
                "env_id": self._env_id,
            },
        )
    
    def get_available_tools(self) -> list[ActionTool]:
        """Get list of currently available BabyAI tools."""
        return [
            self._create_action_tool(action)
            for action in self._available_actions
        ]
    
    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a BabyAI action as a tool.
        
        Args:
            tool_name: Name of the action to execute
            **kwargs: Additional parameters (unused for BabyAI)
            
        Returns:
            Result including new observation and reward
        """
        if self._env_id is None:
            return ToolResult(
                success=False,
                output="",
                error="Environment not initialized. Call reset() first.",
            )
        
        if tool_name not in self._available_actions:
            available = ", ".join(f"'{a}'" for a in self._available_actions)
            return ToolResult(
                success=False,
                output=f"Action '{tool_name}' is not available. Available actions: {available}",
                error=f"Invalid action: {tool_name}",
            )
        
        try:
            data = self._post("step", {"action": tool_name})
            
            raw_obs = data.get("observation", "")
            self._current_observation, self._available_actions, new_goal = \
                self._parse_observation(raw_obs)
            
            # Update goal only if a new one was provided
            if new_goal:
                self._goal = new_goal
            
            self._reward = data.get("reward", 0.0)
            self._score = data.get("score", 0.0)
            self._done = data.get("done", False)
            self._step_count += 1
            
            return ToolResult(
                success=True,
                output=self._current_observation,
                reward=self._score,
                done=self._done,
                metadata={
                    "step": self._step_count,
                    "step_reward": self._reward,
                    "action": tool_name,
                },
            )
            
        except RequestException as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Request to environment server failed: {str(e)}",
            )
    
    def close(self) -> None:
        """Close the BabyAI environment."""
        if self._env_id is not None:
            try:
                self._post("close", {})
            except RequestException:
                pass
            self._env_id = None
    
    def get_system_prompt(self) -> str:
        """Get a BabyAI-specific system prompt."""
        state = self.get_state()
        tools = self.get_available_tools()
        
        tool_lines = "\n".join(
            f"  - {tool.name}: {tool.description}"
            for tool in tools
        )
        
        return f"""You are an agent navigating a BabyAI gridworld environment.

## Your Goal
{state.goal}

## Current Observation
{state.observation}

## Available Actions
{tool_lines}

## Instructions
- Choose ONE action from the available actions list
- Actions like "go to <object>" will navigate you automatically
- Use "check available actions" if you need to see what's possible
- The task is complete when done=True is returned

Select the best action to accomplish your goal."""
    
    def render(self) -> Optional[str]:
        """Get a rendered image of the current state (if available).
        
        Returns:
            Base64-encoded PNG image, or None if rendering failed
        """
        if self._env_id is None:
            return None
        
        try:
            data = self._post("render", {})
            if isinstance(data, str) and data.startswith("data:image"):
                return data
            return data.get("image") if isinstance(data, dict) else None
        except RequestException:
            return None
