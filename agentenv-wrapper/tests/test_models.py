"""Tests for AgentEnv Wrapper models."""

import pytest
from agentenv_wrapper.models import (
    ActionTool,
    ToolDefinition,
    ToolResult,
    EnvState,
)


class TestToolDefinition:
    """Tests for ToolDefinition model."""
    
    def test_basic_creation(self):
        """Test creating a basic tool definition."""
        tool_def = ToolDefinition(
            name="test_tool",
            description="A test tool",
        )
        assert tool_def.name == "test_tool"
        assert tool_def.description == "A test tool"
        assert tool_def.parameters == {}
        assert tool_def.required_params == []
    
    def test_with_parameters(self):
        """Test creating a tool definition with parameters."""
        tool_def = ToolDefinition(
            name="parameterized_tool",
            description="A tool with params",
            parameters={
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                },
            },
            required_params=["target"],
        )
        assert "properties" in tool_def.parameters
        assert "target" in tool_def.required_params


class TestToolResult:
    """Tests for ToolResult model."""
    
    def test_success_result(self):
        """Test creating a successful result."""
        result = ToolResult(
            success=True,
            output="Action completed successfully",
            reward=1.0,
            done=True,
        )
        assert result.success is True
        assert result.error is None
        assert result.done is True
    
    def test_error_result(self):
        """Test creating an error result."""
        result = ToolResult(
            success=False,
            output="",
            error="Invalid action",
        )
        assert result.success is False
        assert result.error == "Invalid action"


class TestEnvState:
    """Tests for EnvState model."""
    
    def test_basic_state(self):
        """Test creating a basic state."""
        state = EnvState(
            observation="You are in a room",
            available_actions=["move forward", "turn left"],
            goal="Go to the red ball",
        )
        assert "room" in state.observation
        assert len(state.available_actions) == 2
        assert state.done is False
    
    def test_state_with_metadata(self):
        """Test state with metadata."""
        state = EnvState(
            observation="Test",
            metadata={"env_id": 42, "level": "easy"},
        )
        assert state.metadata["env_id"] == 42


class TestActionTool:
    """Tests for ActionTool dataclass."""
    
    def test_action_tool_creation(self):
        """Test creating an action tool."""
        def handler(**kwargs):
            return ToolResult(success=True, output="done")
        
        tool = ActionTool(
            name="test_action",
            description="A test action",
            handler=handler,
        )
        assert tool.name == "test_action"
        assert callable(tool.handler)
    
    def test_action_tool_callable(self):
        """Test that action tool is callable."""
        def handler(**kwargs):
            return ToolResult(success=True, output="executed")
        
        tool = ActionTool(
            name="callable_action",
            description="Callable",
            handler=handler,
        )
        
        result = tool()
        assert result.success is True
        assert result.output == "executed"
    
    def test_to_definition(self):
        """Test converting to ToolDefinition."""
        def handler(**kwargs):
            return ToolResult(success=True, output="")
        
        tool = ActionTool(
            name="convert_test",
            description="Test conversion",
            handler=handler,
            parameters={
                "required": ["param1"],
            },
        )
        
        definition = tool.to_definition()
        assert isinstance(definition, ToolDefinition)
        assert definition.name == "convert_test"
        assert "param1" in definition.required_params
