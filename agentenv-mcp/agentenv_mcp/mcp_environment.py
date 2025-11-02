"""
MCP Environment - Core environment logic for MCP server simulation.
"""

import gymnasium as gym
from typing import Any, Dict, Tuple, List, Optional
import json
import re
from .mcp_resources import SimulatedMilvusCollections, MCPResources


class MCPEnv(gym.Env[str, str]):
    """
    Model Context Protocol environment with simulated Milvus collections.
    
    This environment provides tools for:
    - Listing and querying Milvus collections
    - Vector similarity search
    - Accessing collection schemas
    - Using prompt templates and formatters
    """
    
    def __init__(self, task_data: Optional[Dict[str, Any]] = None):
        """
        Initialize MCP environment.
        
        Args:
            task_data: Optional task configuration with 'goal' and other params
        """
        super().__init__()
        
        # Initialize resources
        self.milvus = SimulatedMilvusCollections()
        self.resources = MCPResources()
        
        # Environment state
        self.task_data = task_data or {}
        self.goal = self.task_data.get("goal", "Explore the available collections")
        self.observation = None
        self.current_step = 0
        self.max_steps = 20
        self.action_history = []
        
        # Available tools/actions
        self.tools = {
            "list_collections": self._tool_list_collections,
            "get_collection_info": self._tool_get_collection_info,
            "query_collection": self._tool_query_collection,
            "search_collection": self._tool_search_collection,
            "get_schema": self._tool_get_schema,
            "get_prompt": self._tool_get_prompt,
            "format_response": self._tool_format_response,
            "finish": self._tool_finish
        }
        
        self.task_completed = False
        self.result = None
    
    def _tool_list_collections(self, **kwargs) -> str:
        """List all available Milvus collections."""
        collections = self.milvus.list_collections()
        return f"Available collections: {', '.join(collections)}"
    
    def _tool_get_collection_info(self, collection_name: str, **kwargs) -> str:
        """Get detailed information about a specific collection."""
        info = self.milvus.get_collection_info(collection_name)
        if "error" in info:
            return info["error"]
        
        schema = info["schema"]
        fields_info = []
        for field in schema["fields"]:
            field_str = f"  - {field['name']} ({field['type']})"
            if field.get("is_primary"):
                field_str += " [PRIMARY]"
            if "max_length" in field:
                field_str += f" max_length={field['max_length']}"
            if "dim" in field:
                field_str += f" dim={field['dim']}"
            fields_info.append(field_str)
        
        return f"""Collection: {info['name']}
Description: {schema['description']}
Total records: {info['count']}
Fields:
{chr(10).join(fields_info)}"""
    
    def _tool_query_collection(
        self,
        collection_name: str,
        limit: int = 10,
        filter_expr: str = None,
        **kwargs
    ) -> str:
        """Query a collection and return results."""
        result = self.milvus.query_collection(collection_name, limit, filter_expr)
        
        if "error" in result:
            return result["error"]
        
        output = [f"Query results from '{collection_name}' (showing {result['count']} records):"]
        
        for i, record in enumerate(result["results"], 1):
            output.append(f"\nRecord {i}:")
            for key, value in record.items():
                if key == "embedding" or key.endswith("_embedding"):
                    output.append(f"  {key}: [vector dim={len(value)}]")
                else:
                    output.append(f"  {key}: {value}")
        
        return "\n".join(output)
    
    def _tool_search_collection(
        self,
        collection_name: str,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> str:
        """Perform vector similarity search on a collection."""
        # Simulate query vector (in real scenario, this would be computed)
        query_vector = [0.4] * 768  # Mock vector
        
        result = self.milvus.search_collection(collection_name, query_vector, top_k)
        
        if "error" in result:
            return result["error"]
        
        output = [f"Search results for query '{query}' in '{collection_name}':"]
        
        for i, record in enumerate(result["results"], 1):
            score = record.get("score", 0.0)
            output.append(f"\nResult {i} (similarity: {score:.3f}):")
            for key, value in record.items():
                if key == "score":
                    continue
                if key == "embedding" or key.endswith("_embedding"):
                    output.append(f"  {key}: [vector]")
                else:
                    output.append(f"  {key}: {value}")
        
        return "\n".join(output)
    
    def _tool_get_schema(self, schema_name: str, **kwargs) -> str:
        """Get a specific schema definition."""
        schema = self.resources.get_schema(schema_name)
        if not schema:
            return f"Schema '{schema_name}' not found. Available schemas: {', '.join(self.resources.schemas.keys())}"
        return json.dumps(schema, indent=2)
    
    def _tool_get_prompt(self, prompt_name: str, **kwargs) -> str:
        """Get a prompt template with optional formatting."""
        prompt = self.resources.get_prompt(prompt_name, **kwargs)
        if not prompt:
            return f"Prompt '{prompt_name}' not found. Available prompts: {', '.join(self.resources.prompts.keys())}"
        return prompt
    
    def _tool_format_response(self, data_str: str, formatter: str = "json", **kwargs) -> str:
        """Format data using specified formatter."""
        try:
            # Try to parse as JSON first
            data = json.loads(data_str)
        except:
            data = data_str
        
        return self.resources.format_response(data, formatter)
    
    def _tool_finish(self, answer: str, **kwargs) -> str:
        """Mark task as complete with final answer."""
        self.task_completed = True
        self.result = answer
        return f"Task completed. Answer: {answer}"
    
    def _parse_action(self, action: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Parse action string to extract tool name and parameters.
        
        Expected format: "Action: tool_name with Action Input: {params}"
        
        Returns:
            Tuple of (tool_name, parameters_dict) or (None, None) if parsing fails
        """
        # Try to extract action and input
        action_pattern = r"Action:\s*(\w+)\s*(?:with Action Input:\s*(.+))?"
        match = re.search(action_pattern, action, re.IGNORECASE | re.DOTALL)
        
        if not match:
            return None, None
        
        tool_name = match.group(1).strip()
        input_str = match.group(2)
        
        # Parse parameters
        params = {}
        if input_str:
            input_str = input_str.strip()
            try:
                # Try to parse as JSON
                params = json.loads(input_str)
            except json.JSONDecodeError:
                # Try to parse as key=value pairs
                param_pattern = r'(\w+)\s*=\s*["\']?([^,"\'\}]+)["\']?'
                matches = re.findall(param_pattern, input_str)
                params = {k: v.strip() for k, v in matches}
        
        return tool_name, params
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Execute an action in the MCP environment.
        
        Args:
            action: Action string from agent
            
        Returns:
            observation: Result of action
            reward: Reward for this step
            terminated: Whether episode is done
            info: Additional information
        """
        self.current_step += 1
        reward = 0.0
        terminated = False
        info = {"step": self.current_step}
        
        # Parse action
        tool_name, params = self._parse_action(action)
        
        if tool_name is None:
            self.observation = (
                "Error: Could not parse action. Please use format: "
                "'Action: tool_name with Action Input: {parameters}'"
            )
            info["error"] = "parse_error"
            return self.observation, reward, terminated, info
        
        # Check if tool exists
        if tool_name not in self.tools:
            self.observation = (
                f"Error: Unknown tool '{tool_name}'. "
                f"Available tools: {', '.join(self.tools.keys())}"
            )
            info["error"] = "unknown_tool"
            return self.observation, reward, terminated, info
        
        # Execute tool
        try:
            result = self.tools[tool_name](**(params or {}))
            self.observation = f"Observation: {result}"
            self.action_history.append({
                "step": self.current_step,
                "tool": tool_name,
                "params": params,
                "result": result
            })
            
            # Reward for successful action
            reward = 0.1
            
            # Check if task is completed
            if self.task_completed:
                reward = 1.0
                terminated = True
                info["result"] = self.result
            
        except Exception as e:
            self.observation = f"Error executing tool '{tool_name}': {str(e)}"
            info["error"] = str(e)
        
        # Check max steps
        if self.current_step >= self.max_steps:
            terminated = True
            info["reason"] = "max_steps_reached"
        
        return self.observation, reward, terminated, info
    
    def reset(self, idx: int = 0) -> str:
        """
        Reset environment to initial state.
        
        Args:
            idx: Task index (can be used for different tasks)
            
        Returns:
            Initial observation
        """
        # Reset state
        self.current_step = 0
        self.action_history = []
        self.task_completed = False
        self.result = None
        
        # Initialize observation with task description
        collections = self.milvus.list_collections()
        tools_list = ", ".join(self.tools.keys())
        
        self.observation = f"""New MCP task started.

Goal: {self.goal}

You have access to a simulated MCP server with Milvus vector database collections.

Available collections: {', '.join(collections)}

Available tools:
{tools_list}

To use a tool, respond in this format:
Thought: [your reasoning]

Action: tool_name with Action Input: {{"param1": "value1", "param2": "value2"}}

You can start by listing collections or getting collection info."""
        
        return self.observation
    
    def observation(self) -> str:
        """Return current observation."""
        return self.observation or "Environment not initialized. Call reset() first."
    
    def close(self):
        """Clean up resources."""
        super().close()
        self.action_history.clear()
