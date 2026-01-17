"""
Tests for agentenv-langchain wrapper.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from agentenv_langchain.wrapper import (
    AgentEnvToolWrapper,
    EnvAction,
    EnvObservation,
    BABYAI_ACTIONS,
    ALFWORLD_ACTIONS,
    create_wrapper_for_env,
)


class TestEnvAction:
    """Tests for EnvAction dataclass."""
    
    def test_basic_action(self):
        action = EnvAction(name="test", description="Test action")
        assert action.name == "test"
        assert action.description == "Test action"
        assert action.parameters == {}
        assert action.required_params == []
    
    def test_action_with_params(self):
        action = EnvAction(
            name="move",
            description="Move to location",
            parameters={"x": "int", "y": "int"},
            required_params=["x", "y"],
        )
        assert action.parameters == {"x": "int", "y": "int"}
        assert action.required_params == ["x", "y"]


class TestEnvObservation:
    """Tests for EnvObservation dataclass."""
    
    def test_basic_observation(self):
        obs = EnvObservation(state="test state", reward=0.5, done=False)
        assert obs.state == "test state"
        assert obs.reward == 0.5
        assert obs.done is False
        assert obs.info == {}
        assert obs.available_actions == []
    
    def test_observation_with_extras(self):
        obs = EnvObservation(
            state="test",
            reward=1.0,
            done=True,
            info={"steps": 10},
            available_actions=["move", "look"],
        )
        assert obs.info == {"steps": 10}
        assert obs.available_actions == ["move", "look"]


class TestPredefinedActions:
    """Tests for pre-defined action sets."""
    
    def test_babyai_actions_exist(self):
        assert len(BABYAI_ACTIONS) > 0
        action_names = [a.name for a in BABYAI_ACTIONS]
        assert "turn_left" in action_names
        assert "turn_right" in action_names
        assert "move_forward" in action_names
    
    def test_alfworld_actions_exist(self):
        assert len(ALFWORLD_ACTIONS) > 0
        action_names = [a.name for a in ALFWORLD_ACTIONS]
        assert "goto" in action_names
        assert "take" in action_names
        assert "put" in action_names


class TestAgentEnvToolWrapper:
    """Tests for AgentEnvToolWrapper."""
    
    @patch('agentenv_langchain.wrapper.requests')
    def test_create(self, mock_requests):
        """Test environment creation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 42}
        mock_requests.post.return_value = mock_response
        
        wrapper = AgentEnvToolWrapper(
            env_server_base="http://localhost:8000",
            env_name="Test",
            actions=[EnvAction(name="test", description="Test")],
            auto_create=True,
        )
        
        assert wrapper.env_id == 42
        mock_requests.post.assert_called()
    
    @patch('agentenv_langchain.wrapper.requests')
    def test_reset(self, mock_requests):
        """Test environment reset."""
        # Mock create
        create_response = Mock()
        create_response.status_code = 200
        create_response.json.return_value = {"id": 1}
        
        # Mock reset
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
        
        obs = wrapper.reset(0)
        assert obs.state == "Initial state"
        assert obs.reward == 0.0
        assert obs.done is False
    
    @patch('agentenv_langchain.wrapper.requests')
    def test_step(self, mock_requests):
        """Test environment step."""
        # Mock create
        create_response = Mock()
        create_response.status_code = 200
        create_response.json.return_value = {"id": 1}
        
        # Mock step
        step_response = Mock()
        step_response.status_code = 200
        step_response.json.return_value = {
            "observation": "New state",
            "reward": 0.5,
            "done": False,
        }
        
        mock_requests.post.side_effect = [create_response, step_response]
        
        wrapper = AgentEnvToolWrapper(
            env_server_base="http://localhost:8000",
            env_name="Test",
            actions=[EnvAction(name="move", description="Move")],
        )
        
        obs = wrapper.step("move")
        assert obs.state == "New state"
        assert obs.reward == 0.5
        assert wrapper.step_count == 1
        assert wrapper.total_reward == 0.5
    
    def test_get_action_by_name(self):
        """Test action lookup."""
        with patch('agentenv_langchain.wrapper.requests') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"id": 1}
            mock_requests.post.return_value = mock_response
            
            actions = [
                EnvAction(name="move", description="Move"),
                EnvAction(name="look", description="Look"),
            ]
            
            wrapper = AgentEnvToolWrapper(
                env_server_base="http://localhost:8000",
                env_name="Test",
                actions=actions,
            )
            
            assert wrapper.get_action_by_name("move").description == "Move"
            assert wrapper.get_action_by_name("look").description == "Look"
            assert wrapper.get_action_by_name("nonexistent") is None


class TestCreateWrapperForEnv:
    """Tests for factory function."""
    
    @patch('agentenv_langchain.wrapper.requests')
    def test_create_babyai_wrapper(self, mock_requests):
        """Test BabyAI wrapper creation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1}
        mock_requests.post.return_value = mock_response
        
        wrapper = create_wrapper_for_env("BabyAI", "http://localhost:8000")
        
        assert wrapper.env_name == "BabyAI"
        assert len(wrapper.actions) == len(BABYAI_ACTIONS)
    
    @patch('agentenv_langchain.wrapper.requests')
    def test_create_alfworld_wrapper(self, mock_requests):
        """Test ALFWorld wrapper creation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1}
        mock_requests.post.return_value = mock_response
        
        wrapper = create_wrapper_for_env("ALFWorld", "http://localhost:8001")
        
        assert wrapper.env_name == "ALFWorld"
        assert len(wrapper.actions) == len(ALFWORLD_ACTIONS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
