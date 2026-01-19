"""
Tests for AgentEnvToMCP wrapper.

These tests verify that the AgentEnvMCPServer correctly wraps
BaseEnvClient instances as MCP servers.
"""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from dataclasses import dataclass

# Check if MCP is available
try:
    import mcp
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not MCP_AVAILABLE,
    reason="MCP library not installed. Install with: pip install mcp"
)

# Mock the MCP imports for testing without full MCP installation
@dataclass
class MockStepOutput:
    state: str
    reward: float
    done: bool


class MockBaseEnvClient:
    """Mock BaseEnvClient for testing."""
    
    def __init__(self, env_server_base: str, data_len: int, action_format: str = "function_calling"):
        self.env_server_base = env_server_base
        self.data_len = data_len
        self.action_format = action_format
        self._current_obs = "Initial observation"
        self._task_idx = 0
    
    def __len__(self):
        return self.data_len
    
    def reset(self, idx: int):
        self._task_idx = idx
        self._current_obs = f"Task {idx} observation"
        return {"task_idx": idx, "status": "reset"}
    
    def observe(self):
        return self._current_obs
    
    def step(self, action: str):
        self._current_obs = f"After action: {action}"
        return MockStepOutput(
            state=self._current_obs,
            reward=0.5,
            done=False,
        )


SAMPLE_FUNCTION_DESCRIPTIONS = [
    {
        "name": "open",
        "description": "Opens a container.",
        "parameters": {
            "type": "object",
            "properties": {
                "obj": {"type": "string", "description": "Container to open"}
            },
            "required": ["obj"]
        }
    },
    {
        "name": "lookaround",
        "description": "Look around.",
        "parameters": {"type": "object", "properties": {}}
    },
]


class TestAgentEnvMCPServerInit:
    """Tests for AgentEnvMCPServer initialization."""
    
    def test_init_basic(self):
        """Test basic initialization."""
        from agentenv_mcp.agentenv_to_mcp import AgentEnvMCPServer
        
        server = AgentEnvMCPServer(
            env_client_cls=MockBaseEnvClient,
            client_args={"env_server_base": "http://test", "data_len": 10},
            function_descriptions=SAMPLE_FUNCTION_DESCRIPTIONS,
            env_name="test",
        )
        
        assert server.env_name == "test"
        assert server.action_format == "function_calling"
        assert len(server.function_descriptions) == 2
    
    def test_lazy_client_creation(self):
        """Test that client is created lazily."""
        from agentenv_mcp.agentenv_to_mcp import AgentEnvMCPServer
        
        server = AgentEnvMCPServer(
            env_client_cls=MockBaseEnvClient,
            client_args={"env_server_base": "http://test", "data_len": 10},
        )
        
        # Client should not be created yet
        assert server._client is None
        
        # Accessing client property creates it
        client = server.client
        assert client is not None
        assert server._client is client


class TestAgentEnvMCPServerHandlers:
    """Tests for MCP tool handlers."""
    
    @pytest.fixture
    def server(self):
        """Create a test server."""
        from agentenv_mcp.agentenv_to_mcp import AgentEnvMCPServer
        
        return AgentEnvMCPServer(
            env_client_cls=MockBaseEnvClient,
            client_args={"env_server_base": "http://test", "data_len": 100},
            function_descriptions=SAMPLE_FUNCTION_DESCRIPTIONS,
            env_name="test",
        )
    
    @pytest.mark.asyncio
    async def test_handle_reset(self, server):
        """Test reset handler."""
        result = await server._handle_reset({"task_idx": 5})
        
        assert len(result) == 1
        content = json.loads(result[0].content[0].text)
        assert content["status"] == "reset"
        assert content["task_idx"] == 5
    
    @pytest.mark.asyncio
    async def test_handle_observe(self, server):
        """Test observe handler."""
        # First reset to set up state
        await server._handle_reset({"task_idx": 0})
        
        result = await server._handle_observe()
        
        assert len(result) == 1
        assert "Task 0 observation" in result[0].content[0].text
    
    @pytest.mark.asyncio
    async def test_handle_step(self, server):
        """Test step handler."""
        await server._handle_reset({"task_idx": 0})
        
        result = await server._handle_step({"action": "test action"})
        
        assert len(result) == 1
        content = json.loads(result[0].content[0].text)
        assert "observation" in content
        assert content["reward"] == 0.5
        assert content["done"] is False
    
    @pytest.mark.asyncio
    async def test_handle_info(self, server):
        """Test info handler."""
        result = await server._handle_info()
        
        assert len(result) == 1
        content = json.loads(result[0].content[0].text)
        assert content["env_name"] == "test"
        assert content["env_size"] == 100
        assert content["available_actions"] == 2
    
    @pytest.mark.asyncio
    async def test_handle_action(self, server):
        """Test environment-specific action handler."""
        await server._handle_reset({"task_idx": 0})
        
        result = await server._handle_action("open", {"obj": "door"})
        
        assert len(result) == 1
        content = json.loads(result[0].content[0].text)
        assert content["action"] == "open"
        assert "observation" in content
    
    @pytest.mark.asyncio
    async def test_handle_unknown_action(self, server):
        """Test handling of unknown action."""
        result = await server._handle_action("unknown_action", {})
        
        assert len(result) == 1
        assert result[0].isError is True


class TestReactFormatting:
    """Tests for ReAct format action formatting."""
    
    def test_format_react_action(self):
        """Test React action formatting."""
        from agentenv_mcp.agentenv_to_mcp import AgentEnvMCPServer
        
        server = AgentEnvMCPServer(
            env_client_cls=MockBaseEnvClient,
            client_args={"env_server_base": "http://test", "data_len": 10},
            action_format="react",
        )
        
        result = server._format_react_action("open", {"obj": "door", "thought": "Opening the door"})
        
        assert "Thought:" in result
        assert "Action:" in result
        assert "open" in result
