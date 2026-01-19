"""
Tests for schema conversion utilities.
"""

import pytest
from agentenv_mcp.schema_utils import (
    function_desc_to_mcp_tool,
    mcp_tool_to_function_desc,
    generate_function_descriptions_from_mcp_tools,
    generate_mcp_tools_from_function_descriptions,
)


class TestFunctionDescToMCPTool:
    """Tests for function_desc_to_mcp_tool conversion."""
    
    def test_basic_conversion(self):
        """Test basic conversion from AgentEnv to MCP format."""
        func_desc = {
            "name": "open",
            "description": "Opens a container.",
            "parameters": {
                "type": "object",
                "properties": {
                    "obj": {
                        "type": "string",
                        "description": "The container to open."
                    }
                },
                "required": ["obj"]
            }
        }
        
        mcp_tool = function_desc_to_mcp_tool(func_desc)
        
        assert mcp_tool["name"] == "open"
        assert mcp_tool["description"] == "Opens a container."
        assert "inputSchema" in mcp_tool
        assert mcp_tool["inputSchema"]["type"] == "object"
        assert "obj" in mcp_tool["inputSchema"]["properties"]
    
    def test_empty_parameters(self):
        """Test conversion with empty parameters."""
        func_desc = {
            "name": "lookaround",
            "description": "Look around the room.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
        
        mcp_tool = function_desc_to_mcp_tool(func_desc)
        
        assert mcp_tool["name"] == "lookaround"
        assert mcp_tool["inputSchema"]["properties"] == {}
    
    def test_missing_description(self):
        """Test conversion with missing description."""
        func_desc = {
            "name": "test",
            "parameters": {"type": "object", "properties": {}}
        }
        
        mcp_tool = function_desc_to_mcp_tool(func_desc)
        
        assert mcp_tool["description"] == ""
    
    def test_multiple_parameters(self):
        """Test conversion with multiple parameters."""
        func_desc = {
            "name": "pour",
            "description": "Pour liquid into container.",
            "parameters": {
                "type": "object",
                "properties": {
                    "liquid": {"type": "string", "description": "The liquid"},
                    "container": {"type": "string", "description": "The container"}
                },
                "required": ["liquid", "container"]
            }
        }
        
        mcp_tool = function_desc_to_mcp_tool(func_desc)
        
        assert len(mcp_tool["inputSchema"]["properties"]) == 2
        assert "liquid" in mcp_tool["inputSchema"]["properties"]
        assert "container" in mcp_tool["inputSchema"]["properties"]


class TestMCPToolToFunctionDesc:
    """Tests for mcp_tool_to_function_desc conversion."""
    
    def test_basic_conversion(self):
        """Test basic conversion from MCP to AgentEnv format."""
        mcp_tool = {
            "name": "open",
            "description": "Opens a container.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "obj": {
                        "type": "string",
                        "description": "The container to open."
                    }
                },
                "required": ["obj"]
            }
        }
        
        func_desc = mcp_tool_to_function_desc(mcp_tool)
        
        assert func_desc["name"] == "open"
        assert func_desc["description"] == "Opens a container."
        assert "parameters" in func_desc
        assert func_desc["parameters"]["type"] == "object"
    
    def test_missing_input_schema(self):
        """Test conversion with missing inputSchema."""
        mcp_tool = {
            "name": "test",
            "description": "A test tool."
        }
        
        func_desc = mcp_tool_to_function_desc(mcp_tool)
        
        assert func_desc["parameters"]["type"] == "object"
        assert func_desc["parameters"]["properties"] == {}


class TestRoundTrip:
    """Test round-trip conversions."""
    
    def test_agentenv_to_mcp_to_agentenv(self):
        """Test AgentEnv -> MCP -> AgentEnv preserves data."""
        original = {
            "name": "goto",
            "description": "Move to a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Target location"
                    }
                },
                "required": ["location"]
            }
        }
        
        mcp_tool = function_desc_to_mcp_tool(original)
        restored = mcp_tool_to_function_desc(mcp_tool)
        
        assert restored["name"] == original["name"]
        assert restored["description"] == original["description"]
        assert restored["parameters"] == original["parameters"]
    
    def test_batch_conversion(self):
        """Test batch conversion of multiple tools."""
        func_descs = [
            {"name": "open", "description": "Open", "parameters": {"type": "object", "properties": {}}},
            {"name": "close", "description": "Close", "parameters": {"type": "object", "properties": {}}},
        ]
        
        mcp_tools = generate_mcp_tools_from_function_descriptions(func_descs)
        restored = generate_function_descriptions_from_mcp_tools(mcp_tools)
        
        assert len(restored) == 2
        assert restored[0]["name"] == "open"
        assert restored[1]["name"] == "close"


class TestSciWorldFunctionDescriptions:
    """Test with real SciWorld function descriptions."""
    
    SAMPLE_SCIWORLD_FUNCTIONS = [
        {
            "name": "open", 
            "description": "Opens a container.",
            "parameters": {
                "type": "object",
                "properties": {
                    "obj": {"type": "string", "description": "The container to open."}
                },
                "required": ["obj"]
            }
        },
        {
            "name": "lookaround",
            "description": "Describe the current room.",
            "parameters": {"type": "object", "properties": {}}
        },
        {
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
        },
    ]
    
    def test_sciworld_conversion(self):
        """Test conversion of SciWorld-style function descriptions."""
        mcp_tools = generate_mcp_tools_from_function_descriptions(
            self.SAMPLE_SCIWORLD_FUNCTIONS
        )
        
        assert len(mcp_tools) == 3
        
        # Check open tool
        open_tool = next(t for t in mcp_tools if t["name"] == "open")
        assert "inputSchema" in open_tool
        assert "obj" in open_tool["inputSchema"]["properties"]
        
        # Check lookaround tool (no args)
        look_tool = next(t for t in mcp_tools if t["name"] == "lookaround")
        assert look_tool["inputSchema"]["properties"] == {}
        
        # Check pour tool (multiple args)
        pour_tool = next(t for t in mcp_tools if t["name"] == "pour")
        assert len(pour_tool["inputSchema"]["properties"]) == 2
