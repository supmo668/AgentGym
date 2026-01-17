"""Tests for base wrapper functionality."""

import pytest
from unittest.mock import Mock, patch
from agentenv_wrapper.base import BaseToolWrapper, ToolEnvironmentWrapper
from agentenv_wrapper.models import ActionTool, EnvState, ToolResult


class MockToolWrapper(BaseToolWrapper):
    """Mock implementation for testing."""
    
    def __init__(self):
        super().__init__(env_name="MockEnv")
        self._actions = ["action1", "action2"]
        self._observation = "Test observation"
        self._goal = "Test goal"
        self._done = False
        self._reward = 0.0
    
    def reset(self, task_id: int = 0) -> EnvState:
        self._done = False
        self._reward = 0.0
        return self.get_state()
    
    def get_state(self) -> EnvState:
        return EnvState(
            observation=self._observation,
            available_actions=self._actions,
            goal=self._goal,
            reward=self._reward,
            done=self._done,
        )
    
    def get_available_tools(self) -> list[ActionTool]:
        return [
            ActionTool(
                name=action,
                description=f"Execute {action}",
                handler=lambda a=action: self.execute_tool(a),
            )
            for action in self._actions
        ]
    
    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        if tool_name not in self._actions:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown action: {tool_name}",
            )
        
        self._observation = f"Executed {tool_name}"
        return ToolResult(
            success=True,
            output=self._observation,
            reward=0.5,
        )


class TestBaseToolWrapper:
    """Tests for BaseToolWrapper."""
    
    def test_reset(self):
        """Test environment reset."""
        wrapper = MockToolWrapper()
        state = wrapper.reset(task_id=1)
        
        assert isinstance(state, EnvState)
        assert state.done is False
    
    def test_get_state(self):
        """Test getting current state."""
        wrapper = MockToolWrapper()
        state = wrapper.get_state()
        
        assert state.observation == "Test observation"
        assert state.goal == "Test goal"
        assert len(state.available_actions) == 2
    
    def test_get_available_tools(self):
        """Test getting available tools."""
        wrapper = MockToolWrapper()
        tools = wrapper.get_available_tools()
        
        assert len(tools) == 2
        assert all(isinstance(t, ActionTool) for t in tools)
        assert tools[0].name == "action1"
    
    def test_execute_tool_success(self):
        """Test successful tool execution."""
        wrapper = MockToolWrapper()
        result = wrapper.execute_tool("action1")
        
        assert result.success is True
        assert "Executed action1" in result.output
    
    def test_execute_tool_invalid(self):
        """Test invalid tool execution."""
        wrapper = MockToolWrapper()
        result = wrapper.execute_tool("invalid_action")
        
        assert result.success is False
        assert result.error is not None
    
    def test_get_tool_definitions(self):
        """Test getting tool definitions."""
        wrapper = MockToolWrapper()
        definitions = wrapper.get_tool_definitions()
        
        assert len(definitions) == 2
        assert definitions[0].name == "action1"
    
    def test_get_system_prompt(self):
        """Test system prompt generation."""
        wrapper = MockToolWrapper()
        prompt = wrapper.get_system_prompt()
        
        assert "MockEnv" in prompt
        assert "Test goal" in prompt
        assert "action1" in prompt


class TestToolEnvironmentWrapper:
    """Tests for ToolEnvironmentWrapper with mocked HTTP."""
    
    @patch("requests.post")
    def test_reset(self, mock_post):
        """Test reset with mocked HTTP."""
        # Mock create response
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.side_effect = [
            {"id": 1},  # create
            {  # reset
                "observation": 'Your goal: test goal\nObs\nAvailable actions: ["move", "turn"]',
                "reward": 0.0,
                "done": False,
            },
        ]
        
        wrapper = ToolEnvironmentWrapper(
            env_server_base="http://test:8080",
            env_name="TestEnv",
        )
        state = wrapper.reset(task_id=0)
        
        assert wrapper._env_id == 1
        assert len(state.available_actions) == 2
    
    @patch("requests.post")
    def test_execute_tool(self, mock_post):
        """Test tool execution with mocked HTTP."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.side_effect = [
            {"id": 1},
            {"observation": 'Obs\nAvailable actions: ["move"]', "reward": 0.0, "done": False},
            {"observation": "New obs\nAvailable actions: []", "score": 1.0, "done": True},
        ]
        
        wrapper = ToolEnvironmentWrapper("http://test:8080")
        wrapper.reset(task_id=0)
        
        result = wrapper.execute_tool("move")
        
        assert result.success is True
        assert result.done is True
    
    def test_execute_without_init(self):
        """Test executing tool without initialization."""
        wrapper = ToolEnvironmentWrapper("http://test:8080")
        result = wrapper.execute_tool("action")
        
        assert result.success is False
        assert "not initialized" in result.error
    
    @patch("requests.post")
    def test_parse_actions(self, mock_post):
        """Test action parsing from observation."""
        wrapper = ToolEnvironmentWrapper("http://test:8080")
        
        obs = 'Some text\nAvailable actions: ["turn left", "move forward", "pick up ball"]'
        actions = wrapper._parse_available_actions(obs)
        
        assert len(actions) == 3
        assert "turn left" in actions
        assert "pick up ball" in actions
