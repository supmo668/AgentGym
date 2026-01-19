"""
Demo: Using an MCP Server as an AgentEnv client.

This demonstrates how to connect to any MCP server and use it as an
AgentGym-compatible environment for agent evaluation.

Usage:
    # First, start the SciWorld MCP server
    python -m agentenv_mcp.examples.sciworld_mcp_server
    
    # Then run this demo (in a different terminal)
    python -m agentenv_mcp.examples.mcp_client_demo
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentenv_mcp import MCPEnvClient


def demo_mcp_client():
    """
    Demonstrate using an MCP server as an AgentEnv client.
    """
    print("Creating MCP Environment Client...")
    print("=" * 60)
    
    # Create client that connects to the SciWorld MCP server
    client = MCPEnvClient(
        server_command=[
            sys.executable, "-m", 
            "agentenv_mcp.examples.sciworld_mcp_server",
            "--env-server", "http://localhost:8000",
        ],
        action_format="function_calling",
        data_len=100,
    )
    
    print(f"Environment size: {len(client)}")
    print(f"Action format: {client.action_format}")
    print()
    
    # Show conversation start
    print("Conversation Start:")
    print("-" * 40)
    for msg in client.conversation_start:
        role = msg["from"]
        value = msg["value"][:200] + "..." if len(msg["value"]) > 200 else msg["value"]
        print(f"[{role}]: {value}")
    print()
    
    # Reset to first task
    print("Resetting to task 0...")
    print("-" * 40)
    reset_result = client.reset(0)
    print(f"Reset result: {reset_result}")
    print()
    
    # Get observation
    print("Current observation:")
    print("-" * 40)
    obs = client.observe()
    print(obs[:500] + "..." if len(obs) > 500 else obs)
    print()
    
    # Try an action (function calling format)
    print("Executing action: lookaround")
    print("-" * 40)
    action = '{"thought": "Let me look around", "function_name": "lookaround", "arguments": {}}'
    result = client.step(action)
    print(f"State: {result.state[:300]}...")
    print(f"Reward: {result.reward}")
    print(f"Done: {result.done}")
    print()
    
    # Try another action
    print("Executing action: inventory")
    print("-" * 40)
    action = '{"thought": "Check my inventory", "function_name": "inventory", "arguments": {}}'
    result = client.step(action)
    print(f"State: {result.state[:300]}...")
    print(f"Reward: {result.reward}")
    print(f"Done: {result.done}")
    
    # Clean up
    client.close()
    print()
    print("Demo complete!")


if __name__ == "__main__":
    demo_mcp_client()
