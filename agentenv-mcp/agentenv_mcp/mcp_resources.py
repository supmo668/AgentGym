"""
MCP resources - schemas, prompts, and formatters.

This module provides:
- Collection schemas
- Prompt templates (loaded from prompts.yaml)
- Response formatters
"""

from typing import Dict, List, Any, Optional
import json
import yaml
from pathlib import Path


def load_prompts() -> Dict[str, str]:
    """Load prompts from YAML file."""
    prompts_file = Path(__file__).parent.parent / "prompts.yaml"
    if prompts_file.exists():
        with open(prompts_file) as f:
            return yaml.safe_load(f)
    return {}


class MilvusConnection:
    """Milvus database connection handler."""
    
    def __init__(self, host: str = "localhost", port: int = 19530, 
                 user: str = "", password: str = ""):
        """Initialize Milvus connection parameters."""
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self._connected = False
    
    def connect(self):
        """Connect to Milvus (placeholder for actual connection)."""
        # Actual Milvus connection would go here
        # from pymilvus import connections
        # connections.connect(host=self.host, port=self.port, user=self.user, password=self.password)
        self._connected = True
    
    def list_collections(self) -> List[str]:
        """List all collections in Milvus."""
        if not self._connected:
            self.connect()
        # Actual implementation: from pymilvus import utility
        # return utility.list_collections()
        return []
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection schema and info."""
        if not self._connected:
            self.connect()
        # Actual implementation: from pymilvus import Collection
        # collection = Collection(collection_name)
        # return {"schema": collection.schema, "count": collection.num_entities}
        return {"error": "Not connected to Milvus"}
    
    def query_collection(self, collection_name: str, limit: int = 10, 
                        filter_expr: Optional[str] = None) -> Dict[str, Any]:
        """Query a collection."""
        if not self._connected:
            self.connect()
        # Actual implementation with pymilvus
        return {"results": [], "count": 0}
    
    def search_collection(self, collection_name: str, query_vector: List[float], 
                         top_k: int = 5) -> Dict[str, Any]:
        """Perform vector similarity search."""
        if not self._connected:
            self.connect()
        # Actual implementation with pymilvus
        return {"results": [], "count": 0}


class MCPResources:
    """MCP resources including schemas, prompts, and formatters."""
    
    def __init__(self):
        """Initialize MCP resources."""
        self.schemas = self._init_schemas()
        self.prompts = load_prompts()
        self.formatters = self._init_formatters()
    
    def _init_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Initialize resource schemas."""
        return {
            "tool_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            "response_schema": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "data": {"type": "object"},
                    "error": {"type": "string"}
                }
            }
        }
    
    def _init_formatters(self) -> Dict[str, Any]:
        """Initialize response formatters."""
        return {
            "json_formatter": {
                "name": "JSON Response Formatter",
                "format": lambda data: json.dumps(data, indent=2)
            },
            "text_formatter": {
                "name": "Text Response Formatter",
                "format": lambda data: str(data)
            },
            "markdown_formatter": {
                "name": "Markdown Table Formatter",
                "format": self._format_as_markdown
            }
        }
    
    def _format_as_markdown(self, data: Any) -> str:
        """Format data as markdown table."""
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                keys = list(data[0].keys())
                header = "| " + " | ".join(keys) + " |"
                separator = "| " + " | ".join(["---"] * len(keys)) + " |"
                rows = []
                for item in data:
                    row = "| " + " | ".join(str(item.get(k, "")) for k in keys) + " |"
                    rows.append(row)
                return "\n".join([header, separator] + rows)
        return str(data)
    
    def get_schema(self, schema_name: str) -> Dict[str, Any]:
        """Get a specific schema."""
        return self.schemas.get(schema_name, {})
    
    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        """Get and format a prompt template."""
        template = self.prompts.get(prompt_name, "")
        try:
            return template.format(**kwargs)
        except KeyError:
            return template
    
    def format_response(self, data: Any, formatter: str = "json") -> str:
        """Format response using specified formatter."""
        formatter_name = f"{formatter}_formatter"
        if formatter_name in self.formatters:
            return self.formatters[formatter_name]["format"](data)
        return str(data)

    """Simulated Milvus collections with vector data."""
    
    def __init__(self):
        """Initialize simulated collections."""
        self.collections = {
            "documents": {
                "schema": {
                    "name": "documents",
                    "description": "Document embeddings collection",
                    "fields": [
                        {"name": "id", "type": "INT64", "is_primary": True},
                        {"name": "title", "type": "VARCHAR", "max_length": 512},
                        {"name": "content", "type": "VARCHAR", "max_length": 2048},
                        {"name": "embedding", "type": "FLOAT_VECTOR", "dim": 768},
                        {"name": "timestamp", "type": "INT64"}
                    ]
                },
                "data": [
                    {
                        "id": 1,
                        "title": "Introduction to Machine Learning",
                        "content": "Machine learning is a subset of artificial intelligence...",
                        "embedding": [0.1] * 768,
                        "timestamp": 1704067200
                    },
                    {
                        "id": 2,
                        "title": "Deep Learning Fundamentals",
                        "content": "Deep learning uses neural networks with multiple layers...",
                        "embedding": [0.2] * 768,
                        "timestamp": 1704153600
                    },
                    {
                        "id": 3,
                        "title": "Natural Language Processing",
                        "content": "NLP enables computers to understand and process human language...",
                        "embedding": [0.3] * 768,
                        "timestamp": 1704240000
                    }
                ],
                "count": 3
            },
            "users": {
                "schema": {
                    "name": "users",
                    "description": "User profile embeddings",
                    "fields": [
                        {"name": "id", "type": "INT64", "is_primary": True},
                        {"name": "username", "type": "VARCHAR", "max_length": 128},
                        {"name": "profile_embedding", "type": "FLOAT_VECTOR", "dim": 512},
                        {"name": "created_at", "type": "INT64"}
                    ]
                },
                "data": [
                    {
                        "id": 101,
                        "username": "alice_researcher",
                        "profile_embedding": [0.5] * 512,
                        "created_at": 1703980800
                    },
                    {
                        "id": 102,
                        "username": "bob_engineer",
                        "profile_embedding": [0.6] * 512,
                        "created_at": 1704067200
                    }
                ],
                "count": 2
            },
            "products": {
                "schema": {
                    "name": "products",
                    "description": "Product catalog embeddings",
                    "fields": [
                        {"name": "id", "type": "INT64", "is_primary": True},
                        {"name": "name", "type": "VARCHAR", "max_length": 256},
                        {"name": "category", "type": "VARCHAR", "max_length": 128},
                        {"name": "description_embedding", "type": "FLOAT_VECTOR", "dim": 384},
                        {"name": "price", "type": "FLOAT"}
                    ]
                },
                "data": [
                    {
                        "id": 1001,
                        "name": "Wireless Headphones",
                        "category": "Electronics",
                        "description_embedding": [0.7] * 384,
                        "price": 99.99
                    },
                    {
                        "id": 1002,
                        "name": "Smart Watch",
                        "category": "Electronics",
                        "description_embedding": [0.8] * 384,
                        "price": 249.99
                    },
                    {
                        "id": 1003,
                        "name": "Laptop Stand",
                        "category": "Accessories",
                        "description_embedding": [0.9] * 384,
                        "price": 39.99
                    }
                ],
                "count": 3
            }
        }
    
    def list_collections(self) -> List[str]:
        """Get list of all collection names."""
        return list(self.collections.keys())
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get metadata about a collection."""
        if collection_name not in self.collections:
            return {"error": f"Collection '{collection_name}' not found"}
        
        collection = self.collections[collection_name]
        return {
            "name": collection_name,
            "schema": collection["schema"],
            "count": collection["count"]
        }
    
    def query_collection(
        self,
        collection_name: str,
        limit: int = 10,
        filter_expr: str = None
    ) -> Dict[str, Any]:
        """Query a collection and return results."""
        if collection_name not in self.collections:
            return {"error": f"Collection '{collection_name}' not found"}
        
        data = self.collections[collection_name]["data"]
        
        # Simple filter simulation (just return all for now)
        results = data[:limit] if limit else data
        
        return {
            "collection": collection_name,
            "count": len(results),
            "results": results
        }
    
    def search_collection(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Simulate vector similarity search."""
        if collection_name not in self.collections:
            return {"error": f"Collection '{collection_name}' not found"}
        
        # Simulate search results (just return top items)
        data = self.collections[collection_name]["data"]
        results = data[:min(top_k, len(data))]
        
        # Add simulated similarity scores
        for i, result in enumerate(results):
            result["score"] = 0.95 - (i * 0.05)
        
        return {
            "collection": collection_name,
            "top_k": top_k,
            "results": results
        }


class MCPResources:
    """MCP resources including schemas, prompts, and formatters."""
    
    def __init__(self):
        """Initialize MCP resources."""
        self.schemas = self._init_schemas()
        self.prompts = self._init_prompts()
        self.formatters = self._init_formatters()
    
    def _init_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Initialize collection schemas."""
        return {
            "tool_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            "response_schema": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "data": {"type": "object"},
                    "error": {"type": "string"}
                }
            }
        }
    
    def _init_prompts(self) -> Dict[str, str]:
        """Initialize prompt templates."""
        return {
            "search_prompt": """You are an AI assistant with access to a Milvus vector database.
Use the available tools to search and retrieve information from collections.

Available collections: {collections}

Task: {task}

Think step by step and use the appropriate tools to accomplish the task.""",
            
            "query_prompt": """Query the {collection_name} collection to find relevant information.
Filter: {filter_expr}
Limit: {limit}

Provide a summary of the results.""",
            
            "vector_search_prompt": """Perform a vector similarity search in the {collection_name} collection.
Query: {query}
Top K: {top_k}

Return the most similar items."""
        }
    
    def _init_formatters(self) -> Dict[str, Any]:
        """Initialize prompt formatters."""
        return {
            "json_formatter": {
                "name": "JSON Response Formatter",
                "format": lambda data: json.dumps(data, indent=2)
            },
            "text_formatter": {
                "name": "Text Response Formatter",
                "format": lambda data: str(data)
            },
            "markdown_formatter": {
                "name": "Markdown Table Formatter",
                "format": self._format_as_markdown
            }
        }
    
    def _format_as_markdown(self, data: Any) -> str:
        """Format data as markdown table."""
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                # Create markdown table
                keys = list(data[0].keys())
                header = "| " + " | ".join(keys) + " |"
                separator = "| " + " | ".join(["---"] * len(keys)) + " |"
                rows = []
                for item in data:
                    row = "| " + " | ".join(str(item.get(k, "")) for k in keys) + " |"
                    rows.append(row)
                return "\n".join([header, separator] + rows)
        return str(data)
    
    def get_schema(self, schema_name: str) -> Dict[str, Any]:
        """Get a specific schema."""
        return self.schemas.get(schema_name, {})
    
    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        """Get and format a prompt template."""
        template = self.prompts.get(prompt_name, "")
        try:
            return template.format(**kwargs)
        except KeyError:
            return template
    
    def format_response(self, data: Any, formatter: str = "json") -> str:
        """Format response using specified formatter."""
        formatter_name = f"{formatter}_formatter"
        if formatter_name in self.formatters:
            return self.formatters[formatter_name]["format"](data)
        return str(data)
