"""
SciWorld MCP Server - Exposes SciWorld environment as an MCP server.

This demonstrates how to wrap an existing AgentGym environment (SciWorld)
as an MCP server that can be accessed by any MCP-compatible agent.

Usage:
    # Start the SciWorld environment server first (agentenv-sciworld)
    uvicorn agentenv_sciworld.server:app --host 0.0.0.0 --port 8000
    
    # Then run this MCP server
    python -m agentenv_mcp.examples.sciworld_mcp_server
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentenv_mcp import AgentEnvMCPServer

# Import SciWorld-specific components
# These are copied from agentenv/envs/sciworld.py for standalone operation
SCIWORLD_FUNCTION_DESCRIPTION = [
    {
        "name": "open", 
        "description": "Opens a container. You may have to give the specific location of the container if necessary(eg.door to kitchen, door to living room).",
        "parameters":{
            "type": "object",
            "properties":{
                "obj":{
                    "type": "string",
                    "description":"The container you want to open."
                }
            },
            "required": ["obj"]
        }
    },
    {
        "name": "close", 
        "description": "Closes a container.",
        "parameters":{
            "type": "object",
            "properties":{
                "obj":{
                    "type": "string",
                    "description":"The container you want to close."
                },
            },
            "required": ["obj"]
        }
    },
    {
        "name": "activate",
        "description": "Activate a device (e.g., turn on a stove to heat something).",
        "parameters":{
            "type": "object",
            "properties":{
                "obj":{
                    "type": "string",
                    "description":"The device you want to activate."
                },
            },
            "required": ["obj"]
        },
    },
    {
        "name": "deactivate",
        "description": "Deactivate a device (e.g., turn off a sink).",
        "parameters":{
            "type": "object",
            "properties":{
                "obj":{
                    "type": "string",
                    "description":"The device you want to deactivate."
                },
            },
            "required": ["obj"]
        },
    },
    {
        "name": "lookaround",
        "description": "Describe the current room.",
        "parameters": {
            "type": "object",
            "properties": {},
        }
    },
    {
        "name": "lookat",
        "description": "Describe an object in detail.",
        "parameters":{
            "type": "object",
            "properties":{
                "obj":{
                    "type": "string",
                    "description":"The object you want to examine."
                }
            },
            "required": ["obj"]
        }
    },
    {
        "name": "pickup",
        "description": "Move an object to your inventory.",
        "parameters":{
            "type": "object",
            "properties":{
                "obj":{
                    "type": "string",
                    "description": "The object to pick up."
                }
            },
            "required": ["obj"]
        },
    },
    {
        "name": "drop",
        "description": "Drop an object from your inventory.",
        "parameters":{
            "type": "object",
            "properties":{
                "obj":{
                    "type": "string",
                    "description":"The object to drop."
                },
            },
            "required": ["obj"]
        },
    },
    {
        "name": "goto",
        "description": "Move to a new location.",
        "parameters":{
            "type": "object",
            "properties":{
                "loc":{
                    "type": "string",
                    "description": "The location to go to."
                }
            },
            "required": ["loc"]
        },
    },
    {
        "name": "use",
        "description": "Use a tool on an object.",
        "parameters":{
            "type": "object",
            "properties":{
                "tool":{
                    "type": "string",
                    "description":"The tool to use."
                },
                "obj":{
                    "type": "string",
                    "description": "The object to use the tool on (optional)."
                }
            },
            "required": ["tool"]
        }
    },
    {
        "name": "pour",
        "description": "Pour a liquid into a container.",
        "parameters":{
            "type": "object",
            "properties":{
                "liq":{
                    "type": "string",
                    "description": "The liquid to pour."
                },
                "container":{
                    "type": "string",
                    "description": "The container to pour into."
                }
            },
            "required": ["liq", "container"]
        },
    },
    {
        "name": "mix",
        "description": "Chemically mix the contents of a container.",
        "parameters":{
            "type": "object",
            "properties":{
                "container":{
                    "type": "string",
                    "description": "The container to mix."
                }
            },
            "required": ["container"]
        },
    },
    {
        "name": "inventory",
        "description": "List items in your inventory.",
        "parameters": {
            "type": "object",
            "properties": {},
        }
    },
    {
        "name": "task",
        "description": "Describe the current task.",
        "parameters": {
            "type": "object",
            "properties": {},
        }
    },
    {
        "name": "wait",
        "description": "Wait for some time.",
        "parameters":{
            "type": "object",
            "properties":{
                "duration":{
                    "type": "integer",
                    "description": "Number of time steps to wait."
                }
            },
            "required": ["duration"]
        },
    },
]


def create_sciworld_mcp_server(
    env_server_base: str = "http://localhost:8000",
    data_len: int = 100,
) -> AgentEnvMCPServer:
    """
    Create an MCP server that wraps the SciWorld environment.
    
    Args:
        env_server_base: Base URL of the SciWorld environment server
        data_len: Number of tasks available
        
    Returns:
        AgentEnvMCPServer instance
    """
    # Import the SciWorld client
    try:
        from agentenv.envs.sciworld import SciworldEnvClient
    except ImportError:
        raise ImportError(
            "agentenv package not found. Please install it first:\n"
            "  pip install -e ../agentenv"
        )
    
    return AgentEnvMCPServer(
        env_client_cls=SciworldEnvClient,
        client_args={
            "env_server_base": env_server_base,
            "data_len": data_len,
        },
        function_descriptions=SCIWORLD_FUNCTION_DESCRIPTION,
        env_name="sciworld",
        action_format="function_calling",
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run SciWorld as an MCP server"
    )
    parser.add_argument(
        "--env-server",
        default="http://localhost:8000",
        help="Base URL of the SciWorld environment server",
    )
    parser.add_argument(
        "--data-len",
        type=int,
        default=100,
        help="Number of tasks available",
    )
    
    args = parser.parse_args()
    
    print(f"Starting SciWorld MCP server...", file=sys.stderr)
    print(f"  Environment server: {args.env_server}", file=sys.stderr)
    print(f"  Data length: {args.data_len}", file=sys.stderr)
    
    server = create_sciworld_mcp_server(
        env_server_base=args.env_server,
        data_len=args.data_len,
    )
    server.run()
