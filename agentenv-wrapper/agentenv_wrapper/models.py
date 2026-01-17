"""Data models for the AgentEnv tool wrapper.

This module defines the core data structures used to represent tools,
tool results, and environment state in a framework-agnostic way.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from pydantic import BaseModel


class ToolDefinition(BaseModel):
    """Definition of a tool that can be called by an agent.
    
    Attributes:
        name: Unique identifier for the tool
        description: Human-readable description of what the tool does
        parameters: JSON schema for the tool's input parameters
        required_params: List of required parameter names
    """
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    required_params: list[str] = field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True


class ToolResult(BaseModel):
    """Result returned from executing a tool.
    
    Attributes:
        success: Whether the tool execution was successful
        output: The output/observation from the tool
        reward: Reward signal from the environment (if applicable)
        done: Whether the episode is complete
        error: Error message if execution failed
        metadata: Additional information about the execution
    """
    success: bool
    output: str
    reward: float = 0.0
    done: bool = False
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class EnvState(BaseModel):
    """Current state of the environment.
    
    Attributes:
        observation: Current text observation
        available_actions: List of valid action names
        goal: Current task/goal description
        reward: Cumulative reward
        done: Whether the episode is complete
        step_count: Number of steps taken
        metadata: Additional state information
    """
    observation: str
    available_actions: list[str] = field(default_factory=list)
    goal: str = ""
    reward: float = 0.0
    done: bool = False
    step_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionTool:
    """Represents a single action as a tool.
    
    This is a lightweight dataclass that bridges environment actions
    to tool-based agent interfaces.
    
    Attributes:
        name: Action/tool name
        description: Description of what this action does
        handler: Callable that executes this action
        parameters: JSON schema for parameters (empty for simple actions)
        returns_observation: Whether this action returns a new observation
    """
    name: str
    description: str
    handler: Callable[..., ToolResult]
    parameters: dict[str, Any] = field(default_factory=dict)
    returns_observation: bool = True
    
    def __call__(self, **kwargs) -> ToolResult:
        """Execute this action/tool."""
        return self.handler(**kwargs)
    
    def to_definition(self) -> ToolDefinition:
        """Convert to a ToolDefinition."""
        required = list(self.parameters.get("required", []))
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            required_params=required,
        )


# Default dataclass factory workaround for Pydantic v2
def field_factory(default_factory):
    """Helper to create default factory for dataclass-style fields in Pydantic."""
    return default_factory()
