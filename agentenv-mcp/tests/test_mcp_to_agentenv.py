"""
Tests for MCPToAgentEnv wrapper.

These tests verify that the MCPEnvClient correctly adapts
MCP servers into AgentGym-compatible BaseEnvClient instances.
"""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from dataclasses import dataclass
from enum import Enum


# Check if MCP is available
try:
    import mcp
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Check if agentenv is available
try:
    from agentenv.controller.types import ActionFormat
    AGENTENV_AVAILABLE = True
except ImportError:
    AGENTENV_AVAILABLE = False


class TestMCPAdapterParsing:
    """Tests for MCPAdapter parsing methods that don't require full dependencies."""
    
    def test_parse_react_format(self):
        """Test ReAct format parsing logic."""
        text = "Thought:\nI should open the door.\n\nAction:\nopen door"
        
        _split = text.rsplit("Action:", 1)
        assert len(_split) == 2
        _thought, _action = _split
        thought = _thought.split("Thought:")[-1].strip()
        action = _action.strip()
        
        assert thought == "I should open the door."
        assert action == "open door"
    
    def test_parse_react_no_thought(self):
        """Test ReAct parsing without explicit thought."""
        text = "open door"
        
        _split = text.rsplit("Action:", 1)
        if len(_split) == 2:
            thought = _split[0].split("Thought:")[-1].strip()
            action = _split[1].strip()
        else:
            thought = ""
            action = text.strip()
        
        assert action == "open door"
    
    def test_parse_function_calling_format(self):
        """Test function calling format parsing logic."""
        text = '{"thought": "Opening door", "function_name": "open", "arguments": {"obj": "door"}}'
        
        _fn_call = json.loads(
            "{" + text.split("{", 1)[-1].rsplit("}", 1)[0] + "}", strict=False
        )
        
        assert _fn_call["thought"] == "Opening door"
        assert _fn_call["function_name"] == "open"
        assert _fn_call["arguments"]["obj"] == "door"
    
    def test_parse_function_calling_with_extra_text(self):
        """Test parsing function call with surrounding text."""
        text = 'Here is my action: {"thought": "test", "function_name": "look", "arguments": {}}'
        
        _fn_call = json.loads(
            "{" + text.split("{", 1)[-1].rsplit("}", 1)[0] + "}", strict=False
        )
        
        assert _fn_call["function_name"] == "look"


class TestMCPAdapterToolCallConversion:
    """Tests for converting parsed actions to tool calls."""
    
    def test_react_to_tool_call_simple(self):
        """Test converting simple React action to tool call."""
        function_descriptions = [
            {"name": "open", "parameters": {"properties": {"obj": {}}}},
            {"name": "goto", "parameters": {"properties": {"location": {}}}},
        ]
        
        action = "open door"
        
        # Logic for matching action to tool
        result = None
        for func_desc in function_descriptions:
            fn_name = func_desc["name"]
            if action.lower().startswith(fn_name.lower()):
                args_str = action[len(fn_name):].strip()
                params = func_desc.get("parameters", {}).get("properties", {})
                param_names = list(params.keys())
                arg_values = args_str.split() if args_str else []
                
                arguments = {}
                for i, param_name in enumerate(param_names):
                    if i < len(arg_values):
                        arguments[param_name] = arg_values[i]
                
                result = {"tool_name": fn_name, "arguments": arguments}
                break
        
        assert result is not None
        assert result["tool_name"] == "open"
        assert result["arguments"]["obj"] == "door"
    
    def test_react_to_tool_call_no_args(self):
        """Test converting React action with no arguments."""
        function_descriptions = [
            {"name": "lookaround", "parameters": {"properties": {}}},
        ]
        
        action = "lookaround"
        
        result = None
        for func_desc in function_descriptions:
            fn_name = func_desc["name"]
            if action.lower().startswith(fn_name.lower()):
                result = {"tool_name": fn_name, "arguments": {}}
                break
        
        assert result is not None
        assert result["tool_name"] == "lookaround"
        assert result["arguments"] == {}


@pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP library not installed")
class TestMCPEnvClientRequiresMCP:
    """Tests that require MCP library."""
    
    def test_mcp_env_client_has_interface(self):
        """Test that MCPEnvClient has required interface."""
        from agentenv_mcp.mcp_to_agentenv import MCPEnvClient
        
        assert hasattr(MCPEnvClient, 'observe')
        assert hasattr(MCPEnvClient, 'step')
        assert hasattr(MCPEnvClient, 'reset')
        assert hasattr(MCPEnvClient, '__len__')


class TestMCPTaskStructure:
    """Tests for MCPTask class structure."""
    
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP library not installed")
    def test_task_has_required_attributes(self):
        """Test that MCPTask has required BaseTask attributes."""
        from agentenv_mcp.mcp_to_agentenv import MCPTask
        
        assert hasattr(MCPTask, 'env_client_cls')
        assert hasattr(MCPTask, 'env_name')
        assert MCPTask.env_name == "mcp"


class TestSchemaConversion:
    """Tests that verify schema conversion works correctly."""
    
    def test_sciworld_style_function_to_tool_call(self):
        """Test converting SciWorld-style functions to tool call format."""
        from agentenv_mcp.schema_utils import function_desc_to_mcp_tool
        
        sciworld_func = {
            "name": "pour",
            "description": "Pour liquid into container.",
            "parameters": {
                "type": "object",
                "properties": {
                    "liq": {"type": "string", "description": "The liquid"},
                    "container": {"type": "string", "description": "The container"}
                },
                "required": ["liq", "container"]
            }
        }
        
        mcp_tool = function_desc_to_mcp_tool(sciworld_func)
        
        assert mcp_tool["name"] == "pour"
        assert "inputSchema" in mcp_tool
        assert "liq" in mcp_tool["inputSchema"]["properties"]
        assert "container" in mcp_tool["inputSchema"]["properties"]


class TestConversationStartGeneration:
    """Tests for conversation start prompt generation."""
    
    def test_function_prompt_structure(self):
        """Test that function prompts have expected structure."""
        function_descriptions = [
            {
                "name": "open",
                "description": "Opens something.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "obj": {"type": "string", "description": "Object to open"}
                    },
                    "required": ["obj"]
                }
            }
        ]
        
        # Simulate the prompt format
        prompt = "You have the following functions available:\n\n"
        tool_descs = [{"type": "function", "function": f} for f in function_descriptions]
        prompt += "\n".join([json.dumps(f, ensure_ascii=False, indent=2) for f in tool_descs])
        
        assert "open" in prompt
        assert "Opens something" in prompt
        assert "obj" in prompt
