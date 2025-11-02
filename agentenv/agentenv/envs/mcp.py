"""
MCP Environment Client - Client-side interface for agents.
"""

import requests
from typing import Any, Mapping, Dict
from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import ConversationMessage, StepOutput


class MCPEnvClient(BaseEnvClient):
    """
    MCP Environment Client for agent interaction.
    
    Communicates with MCP environment server via HTTP and provides
    a clean interface for LLM agents to interact with the MCP server.
    """
    
    conversation_start = (
        ConversationMessage({
            "from": "human",
            "loss": None,
            "value": """You are an AI assistant with access to a Model Context Protocol (MCP) server.
The MCP server provides tools for interacting with a Milvus vector database.

Available tools:
1. list_collections - Get list of all available collections
2. get_collection_info - Get detailed schema and info about a collection
   - Parameters: collection_name (string)
3. query_collection - Query a collection and retrieve records
   - Parameters: collection_name (string), limit (int, default 10), filter_expr (string, optional)
4. search_collection - Perform vector similarity search
   - Parameters: collection_name (string), query (string), top_k (int, default 5)
5. get_schema - Get a resource schema definition
   - Parameters: schema_name (string)
6. get_prompt - Get a prompt template
   - Parameters: prompt_name (string), and any template variables
7. format_response - Format data using a formatter
   - Parameters: data_str (string), formatter (string: 'json', 'text', or 'markdown')
8. finish - Complete the task with your final answer
   - Parameters: answer (string)

Response Format:
You must respond in this format:

Thought: [Your reasoning about what to do next]

Action: tool_name with Action Input: {"param1": "value1", "param2": "value2"}

Important:
- Always start by understanding what collections are available
- Use get_collection_info to understand collection structure before querying
- When searching, be specific about what you're looking for
- Call finish when you have completed the task with your final answer

Example:
Thought: I need to first see what collections are available in the system.

Action: list_collections with Action Input: {}
"""
        }),
        ConversationMessage({
            "from": "gpt",
            "loss": False,
            "value": "OK. I understand. I will use the available tools to interact with the MCP server and complete the assigned tasks."
        }),
    )
    
    def __init__(
        self,
        env_server_base: str,
        data_len: int,
        *args,
        timeout: int = 300,
        **kwargs
    ):
        """
        Initialize MCP environment client.
        
        Args:
            env_server_base: Base URL of environment server (e.g., http://127.0.0.1:8000)
            data_len: Number of tasks in dataset
            timeout: Request timeout in seconds
        """
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len
        self.info = {}
        
        # Create environment on server
        data = {"id": 0}
        response = requests.post(
            f"{self.env_server_base}/create",
            json=data,
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise requests.exceptions.RequestException(
                f"Failed to create environment: {response.text}"
            )
        
        self.env_id = response.json()
        print(f"MCP Environment created with ID: {self.env_id}")
    
    def __len__(self):
        """Return number of tasks."""
        return self.data_len
    
    def _post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make POST request to server.
        
        Args:
            path: API endpoint path
            data: Request data
            
        Returns:
            Response JSON
        """
        data["env_idx"] = self.env_id
        response = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout
        )
        assert response.status_code == 200, f"Request failed: {response.text}"
        return response.json()
    
    def _get(self, path: str) -> Dict[str, Any]:
        """
        Make GET request to server.
        
        Args:
            path: API endpoint path
            
        Returns:
            Response JSON
        """
        response = requests.get(
            f"{self.env_server_base}/{path}?env_idx={self.env_id}",
            timeout=self.timeout
        )
        assert response.status_code == 200, f"Request failed: {response.text}"
        return response.json()
    
    def observe(self) -> str:
        """
        Get current observation from environment.
        
        Returns:
            Current observation string
        """
        if self.info and "observation" in self.info:
            return self.info["observation"]
        
        response = self._get("observation")
        return response if isinstance(response, str) else response.get("observation", "")
    
    def step(self, action: str) -> StepOutput:
        """
        Execute an action in the environment.
        
        Args:
            action: Action string from LLM (should include "Action:" and "Action Input:")
            
        Returns:
            StepOutput with state, reward, and done status
        """
        # Send action to server (server will parse it)
        response = self._post("step", {"action": action})
        
        # Cache response
        self.info = {
            "observation": response["observation"],
            "reward": response["reward"],
            "done": response["done"]
        }
        
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"]
        )
    
    def reset(self, data_idx: int = 0) -> Dict[str, Any]:
        """
        Reset environment to a specific task.
        
        Args:
            data_idx: Task index
            
        Returns:
            Reset response with initial observation
        """
        response = self._post("reset", {"id": data_idx})
        
        # Handle different response formats
        if isinstance(response, str):
            observation = response
        else:
            observation = response.get("observation", str(response))
        
        self.info = {
            "observation": observation,
            "reward": 0.0,
            "done": False
        }
        
        return self.info
    
    def close(self) -> Dict[str, Any]:
        """
        Close the environment and cleanup resources.
        
        Returns:
            Close response
        """
        try:
            response = self._post("close", {})
            return response
        except Exception as e:
            return {"closed": False, "error": str(e)}


class MCPTask(BaseTask):
    """
    MCP Task configuration.
    
    Binds the MCP environment client to task management.
    """
    
    env_client_cls = MCPEnvClient
    env_name = "MCP"
    
    def __init__(
        self,
        client_args: Mapping[str, Any],
        n_clients: int,
        *args,
        **kwargs
    ):
        """
        Initialize MCP task.
        
        Args:
            client_args: Arguments for environment client initialization
            n_clients: Number of parallel clients
        """
        super().__init__(client_args, n_clients, *args, **kwargs)
