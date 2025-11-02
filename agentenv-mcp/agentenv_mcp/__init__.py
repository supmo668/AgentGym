"""
AgentEnv MCP - Simulated Model Context Protocol server environment for AgentGym.

This environment simulates an MCP server with tools for fetching Milvus collections,
managing resources including schemas, prompts, and prompt formatters.
"""

__version__ = "0.1.0"

# Export simplified utilities
from .utils import SimpleMCPClient, list_collections, search_collection

__all__ = [
    "SimpleMCPClient",
    "list_collections", 
    "search_collection",
]
