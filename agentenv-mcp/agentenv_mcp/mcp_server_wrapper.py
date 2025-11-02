"""
MCP Environment Server - Server-side environment manager.
"""

import threading
from typing import Dict, Optional
from .mcp_environment import MCPEnv


class MCPEnvServer:
    """
    MCP Environment Server manages multiple environment instances.
    
    Responsibilities:
    - Allocate and manage environment instances
    - Provide thread-safe operations
    - Handle environment lifecycle (create, step, reset, close)
    """
    
    def __init__(self):
        """Initialize the environment server."""
        self._max_id: int = 0
        self.env: Dict[int, MCPEnv] = {}
        self.info: Dict[int, dict] = {}
        self.ls: list[int] = []
        self._lock = threading.Lock()
        
        # Simulated task dataset
        self.tasks = [
            {
                "goal": "List all available collections and get information about the 'documents' collection."
            },
            {
                "goal": "Search the 'documents' collection for information about 'machine learning'."
            },
            {
                "goal": "Query the 'users' collection and return all user profiles."
            },
            {
                "goal": "Get the schema for 'tool_schema' and format it as JSON."
            },
            {
                "goal": "Search the 'products' collection for electronics and list the results."
            },
            {
                "goal": "Retrieve information about all collections and summarize their purposes."
            },
            {
                "goal": "Find the top 3 most similar documents to 'deep learning' in the documents collection."
            },
            {
                "goal": "Get collection info for 'users' and query all records with a limit of 5."
            },
            {
                "goal": "Use the search_prompt template to create a query for the products collection."
            },
            {
                "goal": "List all available tools and demonstrate using at least 3 different tools."
            }
        ]
    
    def create(self, task_id: int = 0) -> int:
        """
        Create a new environment instance.
        
        Args:
            task_id: Optional task ID for specific task configuration
            
        Returns:
            Environment ID
        """
        with self._lock:
            env_id = self._max_id
            self._max_id += 1
        
        # Get task data
        task_data = self.tasks[task_id % len(self.tasks)]
        
        # Create environment
        new_env = MCPEnv(task_data=task_data)
        observation = new_env.reset(idx=task_id)
        
        # Register environment
        self.ls.append(env_id)
        self.env[env_id] = new_env
        self.info[env_id] = {
            "observation": observation,
            "reward": 0.0,
            "done": False
        }
        
        print(f"-------MCP Env {env_id} created (task {task_id})--------")
        return env_id
    
    def step(self, env_idx: int, action: str) -> Dict[str, any]:
        """
        Execute an action in the specified environment.
        
        Args:
            env_idx: Environment ID
            action: Action string from agent
            
        Returns:
            Dict with observation, reward, and done status
        """
        if env_idx not in self.env:
            return {
                "observation": f"Error: Environment {env_idx} not found",
                "reward": 0.0,
                "done": True
            }
        
        env = self.env[env_idx]
        observation, reward, terminated, info = env.step(action)
        
        # Update cached info
        self.info[env_idx] = {
            "observation": observation,
            "reward": reward,
            "done": terminated,
            **info
        }
        
        return {
            "observation": observation,
            "reward": reward,
            "done": terminated
        }
    
    def reset(self, env_idx: int, task_id: Optional[int] = None) -> Dict[str, any]:
        """
        Reset the specified environment.
        
        Args:
            env_idx: Environment ID
            task_id: Optional task ID for new task
            
        Returns:
            Dict with initial observation
        """
        if env_idx not in self.env:
            # Create new environment if not found
            return self.create(task_id or 0)
        
        # Get new task data if task_id provided
        if task_id is not None:
            task_data = self.tasks[task_id % len(self.tasks)]
            self.env[env_idx] = MCPEnv(task_data=task_data)
        
        # Reset environment
        observation = self.env[env_idx].reset(idx=task_id or 0)
        
        self.info[env_idx] = {
            "observation": observation,
            "reward": 0.0,
            "done": False
        }
        
        return self.info[env_idx]
    
    def observation(self, env_idx: int) -> Dict[str, str]:
        """
        Get cached observation for environment.
        
        Args:
            env_idx: Environment ID
            
        Returns:
            Dict with observation
        """
        if env_idx in self.info:
            return {"observation": self.info[env_idx]["observation"]}
        return {"error": "Environment not initialized"}
    
    def observe(self, env_idx: int) -> Dict[str, str]:
        """
        Get current observation directly from environment.
        
        Args:
            env_idx: Environment ID
            
        Returns:
            Dict with observation
        """
        if env_idx not in self.env:
            return {"error": f"Environment {env_idx} not found"}
        
        env = self.env[env_idx]
        return {"observation": env.observation()}
    
    def close(self, env_id: int) -> Dict[str, any]:
        """
        Close and cleanup environment instance.
        
        Args:
            env_id: Environment ID to close
            
        Returns:
            Dict with closed status
        """
        try:
            if env_id in self.ls:
                self.ls.remove(env_id)
            
            if env_id in self.env:
                env = self.env.pop(env_id)
                env.close()
            
            if env_id in self.info:
                self.info.pop(env_id)
            
            print(f"-------MCP Env {env_id} closed--------")
            return {"closed": True}
        
        except KeyError:
            return {"closed": False, "error": "Environment does not exist"}
        except Exception as e:
            return {"closed": False, "error": str(e)}
    
    def __del__(self):
        """Cleanup all environments on deletion."""
        for idx in list(self.ls):
            try:
                self.close(idx)
            except Exception:
                pass


# Global server instance
mcp_env_server = MCPEnvServer()
