"""
Simplified utilities for common MCP patterns.
"""

from typing import Dict, Any, Optional
import requests


class SimpleMCPClient:
    """Minimal MCP client with simplified interface."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize client."""
        self.base_url = base_url
        self.env_id = None
        
    def start(self, task_id: int = 0) -> str:
        """Create environment and return initial observation."""
        r = requests.post(f"{self.base_url}/create", json={"id": task_id})
        self.env_id = r.json()
        
        r = requests.post(f"{self.base_url}/reset", json={"env_idx": self.env_id, "id": task_id})
        return r.text
    
    def act(self, tool: str, **params) -> Dict[str, Any]:
        """Execute tool with simplified syntax."""
        import json
        action = f"Action: {tool} with Action Input: {json.dumps(params)}"
        
        r = requests.post(f"{self.base_url}/step", json={
            "env_idx": self.env_id,
            "action": action
        })
        return r.json()
    
    def stop(self) -> None:
        """Close environment."""
        if self.env_id is not None:
            requests.post(f"{self.base_url}/close", json={"env_idx": self.env_id})
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, *args):
        """Context manager exit."""
        self.stop()


# Convenience functions
def list_collections(base_url: str = "http://localhost:8000") -> str:
    """Quick helper to list collections."""
    with SimpleMCPClient(base_url) as client:
        result = client.act("list_collections")
        return result["observation"]


def search_collection(collection: str, query: str, top_k: int = 5, 
                     base_url: str = "http://localhost:8000") -> str:
    """Quick helper to search a collection."""
    with SimpleMCPClient(base_url) as client:
        result = client.act("search_collection", 
                          collection_name=collection, 
                          query=query, 
                          top_k=top_k)
        return result["observation"]


# Example usage:
"""
# Minimal usage
from agentenv_mcp.utils import SimpleMCPClient

with SimpleMCPClient() as client:
    result = client.act("list_collections")
    print(result["observation"])
    
    result = client.act("search_collection", 
                       collection_name="documents", 
                       query="machine learning", 
                       top_k=3)
    print(result["observation"])

# Or even simpler
from agentenv_mcp.utils import list_collections, search_collection

print(list_collections())
print(search_collection("documents", "ML", top_k=3))
"""
