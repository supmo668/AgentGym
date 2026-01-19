"""
Integration tests for SciWorld MCP wrapper.

These tests verify the complete integration between:
1. SciWorld environment (agentenv-sciworld)
2. AgentEnv client (agentenv/envs/sciworld.py)
3. MCP wrapper (agentenv-mcp)

NOTE: These tests require:
- The SciWorld environment server running at http://localhost:8000
- The agentenv package installed

To run integration tests:
    # Start SciWorld server first
    cd ../agentenv-sciworld && uvicorn agentenv_sciworld.server:app --port 8000
    
    # Run tests
    pytest tests/test_sciworld_integration.py -v
"""

import pytest
import json
import sys
import os

# Skip all tests if dependencies not available
pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_INTEGRATION_TESTS", "1") == "1",
    reason="Integration tests disabled. Set SKIP_INTEGRATION_TESTS=0 to run."
)


@pytest.fixture
def sciworld_env_server():
    """Check if SciWorld server is available."""
    import requests
    try:
        response = requests.get("http://localhost:8000/", timeout=2)
        if response.status_code == 200:
            return "http://localhost:8000"
    except:
        pass
    pytest.skip("SciWorld server not available at http://localhost:8000")


class TestSciWorldMCPServerIntegration:
    """Integration tests for SciWorld as MCP server."""
    
    def test_create_mcp_server(self, sciworld_env_server):
        """Test creating MCP server from SciWorld client."""
        from agentenv_mcp.examples.sciworld_mcp_server import create_sciworld_mcp_server
        
        server = create_sciworld_mcp_server(
            env_server_base=sciworld_env_server,
            data_len=10,
        )
        
        assert server is not None
        assert server.env_name == "sciworld"
        assert len(server.function_descriptions) > 0
    
    @pytest.mark.asyncio
    async def test_reset_and_observe(self, sciworld_env_server):
        """Test reset and observe through MCP server."""
        from agentenv_mcp.examples.sciworld_mcp_server import create_sciworld_mcp_server
        
        server = create_sciworld_mcp_server(
            env_server_base=sciworld_env_server,
            data_len=10,
        )
        
        # Reset
        reset_result = await server._handle_reset({"task_idx": 0})
        assert len(reset_result) == 1
        reset_data = json.loads(reset_result[0].content[0].text)
        assert "observation" in reset_data
        
        # Observe
        observe_result = await server._handle_observe()
        assert len(observe_result) == 1
        assert len(observe_result[0].content[0].text) > 0
    
    @pytest.mark.asyncio
    async def test_step_action(self, sciworld_env_server):
        """Test stepping with an action."""
        from agentenv_mcp.examples.sciworld_mcp_server import create_sciworld_mcp_server
        
        server = create_sciworld_mcp_server(
            env_server_base=sciworld_env_server,
            data_len=10,
        )
        
        # Reset first
        await server._handle_reset({"task_idx": 0})
        
        # Step with lookaround
        step_result = await server._handle_action("lookaround", {})
        assert len(step_result) == 1
        step_data = json.loads(step_result[0].content[0].text)
        assert "observation" in step_data
        assert "reward" in step_data
        assert "done" in step_data
    
    @pytest.mark.asyncio
    async def test_multiple_actions(self, sciworld_env_server):
        """Test executing multiple actions in sequence."""
        from agentenv_mcp.examples.sciworld_mcp_server import create_sciworld_mcp_server
        
        server = create_sciworld_mcp_server(
            env_server_base=sciworld_env_server,
            data_len=10,
        )
        
        # Reset
        await server._handle_reset({"task_idx": 0})
        
        # Execute a sequence of actions
        actions = [
            ("lookaround", {}),
            ("inventory", {}),
            ("task", {}),
        ]
        
        for action_name, args in actions:
            result = await server._handle_action(action_name, args)
            assert len(result) == 1
            assert result[0].isError is False


class TestSciWorldClientCompatibility:
    """Test that wrapped SciWorld client is compatible with AgentEnv."""
    
    def test_client_interface(self, sciworld_env_server):
        """Test that SciWorld client has correct interface."""
        from agentenv.envs.sciworld import SciworldEnvClient
        
        client = SciworldEnvClient(
            env_server_base=sciworld_env_server,
            data_len=10,
            action_format="function_calling",
        )
        
        # Test interface
        assert hasattr(client, 'reset')
        assert hasattr(client, 'step')
        assert hasattr(client, 'observe')
        assert hasattr(client, '__len__')
        assert hasattr(client, 'conversation_start')
        
        # Test length
        assert len(client) == 10
        
        # Test conversation start
        assert len(client.conversation_start) == 2
    
    def test_reset_and_step(self, sciworld_env_server):
        """Test basic reset and step operations."""
        from agentenv.envs.sciworld import SciworldEnvClient
        
        client = SciworldEnvClient(
            env_server_base=sciworld_env_server,
            data_len=10,
            action_format="function_calling",
        )
        
        # Reset
        result = client.reset(0)
        assert "observation" in result or client.observe() != ""
        
        # Observe
        obs = client.observe()
        assert isinstance(obs, str)
        assert len(obs) > 0
        
        # Step with function calling format
        action = json.dumps({
            "thought": "Looking around",
            "function_name": "lookaround",
            "arguments": {}
        })
        step_output = client.step(action)
        
        assert hasattr(step_output, 'state')
        assert hasattr(step_output, 'reward')
        assert hasattr(step_output, 'done')


class TestFunctionDescriptionAlignment:
    """Test that function descriptions are properly aligned."""
    
    def test_sciworld_functions_in_wrapper(self):
        """Test that SciWorld functions are correctly represented in wrapper."""
        from agentenv_mcp.examples.sciworld_mcp_server import SCIWORLD_FUNCTION_DESCRIPTION
        from agentenv_mcp.schema_utils import function_desc_to_mcp_tool
        
        # Check some key functions exist
        function_names = {f["name"] for f in SCIWORLD_FUNCTION_DESCRIPTION}
        
        required_functions = {"open", "close", "lookaround", "goto", "pickup", "inventory"}
        assert required_functions.issubset(function_names)
        
        # Check conversion to MCP format
        for func in SCIWORLD_FUNCTION_DESCRIPTION:
            mcp_tool = function_desc_to_mcp_tool(func)
            assert "name" in mcp_tool
            assert "inputSchema" in mcp_tool
            assert mcp_tool["name"] == func["name"]
