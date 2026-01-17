"""
Tests for agentenv-langchain tools.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from agentenv_langchain.wrapper import AgentEnvToolWrapper, EnvAction, EnvObservation
from agentenv_langchain.tools import (
    AgentEnvStepTool,
    AgentEnvObserveTool,
    AgentEnvResetTool,
    create_agentenv_tools,
    _create_action_input_model,
)


class TestInputModelCreation:
    """Tests for dynamic input model creation."""
    
    def test_empty_params(self):
        action = EnvAction(name="test", description="Test")
        model = _create_action_input_model(action)
        assert model.__name__ == "TestInput"
    
    def test_string_params(self):
        action = EnvAction(
            name="move",
            description="Move",
            parameters={"direction": "string"},
            required_params=["direction"],
        )
        model = _create_action_input_model(action)
        
        # Check that the model can be instantiated
        instance = model(direction="north")
        assert instance.direction == "north"
    
    def test_int_params(self):
        action = EnvAction(
            name="goto",
            description="Go to",
            parameters={"x": "integer", "y": "int"},
            required_params=["x", "y"],
        )
        model = _create_action_input_model(action)
        
        instance = model(x=10, y=20)
        assert instance.x == 10
        assert instance.y == 20


class TestAgentEnvStepTool:
    """Tests for step tool."""
    
    @patch('agentenv_langchain.wrapper.requests')
    def test_step_tool_run(self, mock_requests):
        """Test step tool execution."""
        # Mock responses
        create_response = Mock()
        create_response.status_code = 200
        create_response.json.return_value = {"id": 1}
        
        step_response = Mock()
        step_response.status_code = 200
        step_response.json.return_value = {
            "observation": "Moved forward",
            "reward": 0.1,
            "done": False,
        }
        
        mock_requests.post.side_effect = [create_response, step_response]
        
        wrapper = AgentEnvToolWrapper(
            env_server_base="http://localhost:8000",
            env_name="Test",
            actions=[EnvAction(name="move", description="Move")],
        )
        
        tool = AgentEnvStepTool(wrapper, action_name="move")
        result = tool._run()
        
        assert "Moved forward" in result
        assert "Reward: 0.1" in result


class TestAgentEnvObserveTool:
    """Tests for observe tool."""
    
    @patch('agentenv_langchain.wrapper.requests')
    def test_observe_tool(self, mock_requests):
        """Test observe tool."""
        create_response = Mock()
        create_response.status_code = 200
        create_response.json.return_value = {"id": 1}
        
        mock_requests.post.return_value = create_response
        
        wrapper = AgentEnvToolWrapper(
            env_server_base="http://localhost:8000",
            env_name="Test",
            actions=[EnvAction(name="move", description="Move")],
        )
        
        # Set a current observation
        wrapper._current_observation = EnvObservation(
            state="Test state",
            reward=0.0,
            done=False,
            available_actions=["move"],
        )
        
        tool = AgentEnvObserveTool(wrapper)
        result = tool._run()
        
        assert "Test state" in result
        assert "move" in result


class TestAgentEnvResetTool:
    """Tests for reset tool."""
    
    @patch('agentenv_langchain.wrapper.requests')
    def test_reset_tool(self, mock_requests):
        """Test reset tool."""
        create_response = Mock()
        create_response.status_code = 200
        create_response.json.return_value = {"id": 1}
        
        reset_response = Mock()
        reset_response.status_code = 200
        reset_response.json.return_value = {
            "observation": "Initial state",
            "reward": 0.0,
            "done": False,
        }
        
        mock_requests.post.side_effect = [create_response, reset_response]
        
        wrapper = AgentEnvToolWrapper(
            env_server_base="http://localhost:8000",
            env_name="Test",
            actions=[],
        )
        
        tool = AgentEnvResetTool(wrapper)
        result = tool._run(data_idx=0)
        
        assert "reset to task 0" in result.lower() or "Initial state" in result


class TestCreateAgentenvTools:
    """Tests for tool creation function."""
    
    @patch('agentenv_langchain.wrapper.requests')
    def test_creates_tools(self, mock_requests):
        """Test that tools are created correctly."""
        create_response = Mock()
        create_response.status_code = 200
        create_response.json.return_value = {"id": 1}
        mock_requests.post.return_value = create_response
        
        actions = [
            EnvAction(name="move", description="Move forward"),
            EnvAction(name="turn", description="Turn around"),
        ]
        
        wrapper = AgentEnvToolWrapper(
            env_server_base="http://localhost:8000",
            env_name="Test",
            actions=actions,
        )
        
        tools = create_agentenv_tools(wrapper)
        
        # Should have observe + reset + 2 action tools = 4 tools
        assert len(tools) >= 4
        
        tool_names = [t.name for t in tools]
        assert "env_observe" in tool_names
        assert "env_reset" in tool_names
        assert "env_move" in tool_names
        assert "env_turn" in tool_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
