# MCP Environment - Model Context Protocol Server Simulation

This environment provides a simulated Model Context Protocol (MCP) server with tools for interacting with Milvus vector database collections. It demonstrates how to integrate complex tool-based environments into AgentGym.

## Overview

The MCP environment simulates a server that provides:

1. **Milvus Collections**: Simulated vector database collections with embeddings
2. **Collection Tools**: Tools to list, query, and search collections
3. **Resources**: Schemas, prompt templates, and response formatters
4. **Integration**: Tools exposed as environment actions for agent interaction

## Features

### Simulated Milvus Collections

The environment includes three pre-populated collections:

1. **documents** - Document embeddings (768-dimensional vectors)
   - Contains articles about ML, deep learning, and NLP
   - Fields: id, title, content, embedding, timestamp

2. **users** - User profile embeddings (512-dimensional vectors)
   - Contains user profiles with embeddings
   - Fields: id, username, profile_embedding, created_at

3. **products** - Product catalog embeddings (384-dimensional vectors)
   - Contains product information
   - Fields: id, name, category, description_embedding, price

### Available Tools

The environment provides the following tools as actions:

1. **list_collections** - Get all available collection names
2. **get_collection_info** - Get detailed schema and metadata for a collection
3. **query_collection** - Query a collection and retrieve records
4. **search_collection** - Perform vector similarity search
5. **get_schema** - Access resource schema definitions
6. **get_prompt** - Retrieve and format prompt templates
7. **format_response** - Format data using JSON, text, or markdown formatters
8. **finish** - Complete task with final answer

### Resources

The environment includes:

- **Schemas**: JSON schemas for tools and responses
- **Prompt Templates**: Pre-defined prompts for search, query, and vector search
- **Formatters**: JSON, text, and markdown table formatters

## Setup

### Installation

```bash
cd agentenv-mcp
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

### Dependencies

- fastapi >= 0.104.0
- uvicorn >= 0.24.0
- pydantic >= 2.0.0
- click >= 8.1.0
- gymnasium >= 0.29.0
- requests >= 2.31.0

## Usage

### Starting the Server

Launch the MCP environment server:

```bash
# Using installed script
mcp --host 0.0.0.0 --port 8000

# Or directly with Python
python -m agentenv_mcp.mcp_launch --host 0.0.0.0 --port 8000

# With auto-reload for development
mcp --host 127.0.0.1 --port 8000 --reload
```

The server will start on the specified host and port. You can verify it's running by visiting:
- `http://localhost:8000/` - Basic connectivity check
- `http://localhost:8000/health` - Health status endpoint
- `http://localhost:8000/docs` - Interactive API documentation (Swagger UI)

### Using with AgentGym

#### Client Setup

```python
from agentenv.envs.mcp import MCPEnvClient, MCPTask

# Create client
client = MCPEnvClient(
    env_server_base="http://127.0.0.1:8000",
    data_len=10,  # Number of tasks
    timeout=300
)

# Reset to start a task
response = client.reset(data_idx=0)
print(response["observation"])

# Execute actions
action = """Thought: I need to see what collections are available.

Action: list_collections with Action Input: {}"""

result = client.step(action)
print(result.state)
print(f"Reward: {result.reward}, Done: {result.done}")

# Close when done
client.close()
```

#### Task Configuration

```python
from agentenv.envs.mcp import MCPTask

# Configure task
task = MCPTask(
    client_args={
        "env_server_base": "http://127.0.0.1:8000",
        "data_len": 10,
        "timeout": 300
    },
    n_clients=1
)
```

### Example Interaction

```python
# Step 1: List collections
action1 = "Action: list_collections with Action Input: {}"
# Response: "Available collections: documents, users, products"

# Step 2: Get collection info
action2 = 'Action: get_collection_info with Action Input: {"collection_name": "documents"}'
# Response: Detailed schema with fields and types

# Step 3: Query collection
action3 = '''Action: query_collection with Action Input: {
    "collection_name": "documents",
    "limit": 3
}'''
# Response: First 3 documents from the collection

# Step 4: Search collection
action4 = '''Action: search_collection with Action Input: {
    "collection_name": "documents",
    "query": "machine learning",
    "top_k": 5
}'''
# Response: Top 5 most similar documents

# Step 5: Finish task
action5 = '''Action: finish with Action Input: {
    "answer": "Found 3 documents about ML topics"
}'''
# Response: Task completed
```

## API Endpoints

### Server Endpoints

- `GET /` - Connectivity check
- `GET /health` - Health status
- `GET /list_envs` - List active environments
- `POST /create` - Create new environment
- `POST /step` - Execute action
- `POST /reset` - Reset environment
- `GET /observation` - Get current observation
- `POST /close` - Close environment

### Request/Response Format

#### Create Environment
```json
POST /create
Body: {"id": 0}  // Optional task ID
Response: 123  // Environment ID
```

#### Step
```json
POST /step
Body: {
    "env_idx": 123,
    "action": "Action: list_collections with Action Input: {}"
}
Response: {
    "observation": "Available collections: ...",
    "reward": 0.1,
    "done": false
}
```

#### Reset
```json
POST /reset
Body: {
    "env_idx": 123,
    "id": 0  // Task index
}
Response: "New MCP task started..."
```

## Task Dataset

The environment includes 10 pre-defined tasks:

1. List all collections and get info about 'documents'
2. Search 'documents' collection for 'machine learning'
3. Query 'users' collection for all profiles
4. Get 'tool_schema' and format as JSON
5. Search 'products' for electronics
6. Summarize all collection purposes
7. Find top 3 documents similar to 'deep learning'
8. Get 'users' info and query with limit 5
9. Use search_prompt template for products
10. Demonstrate using 3+ different tools

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Agent (LLM)                         │
└────────────────────┬────────────────────────────────┘
                     │ HTTP
                     ▼
┌─────────────────────────────────────────────────────┐
│           MCPEnvClient (agentenv/envs/mcp.py)       │
│  - Parses LLM output                                │
│  - Makes HTTP requests                              │
│  - Manages local state                              │
└────────────────────┬────────────────────────────────┘
                     │ HTTP (FastAPI)
                     ▼
┌─────────────────────────────────────────────────────┐
│      MCP Server (agentenv_mcp/mcp_server.py)        │
│  - FastAPI endpoints                                │
│  - Request validation                               │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│    MCPEnvServer (mcp_server_wrapper.py)             │
│  - Manages environment instances                    │
│  - Thread-safe operations                           │
│  - Task dataset                                     │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│         MCPEnv (mcp_environment.py)                 │
│  - Core environment logic                           │
│  - Tool execution                                   │
│  - Action parsing                                   │
│  - Reward calculation                               │
└────────────────────┬────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌──────────────────┐  ┌──────────────────────┐
│ Simulated Milvus │  │   MCP Resources      │
│   Collections    │  │  - Schemas           │
│  - documents     │  │  - Prompts           │
│  - users         │  │  - Formatters        │
│  - products      │  │                      │
└──────────────────┘  └──────────────────────┘
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest agentenv-mcp/tests/

# Run with coverage
pytest --cov=agentenv_mcp agentenv-mcp/tests/
```

### Code Structure

```
agentenv-mcp/
├── agentenv_mcp/
│   ├── __init__.py              # Package initialization
│   ├── mcp_environment.py       # Core environment logic
│   ├── mcp_server_wrapper.py   # Server-side manager
│   ├── mcp_server.py            # FastAPI server
│   ├── mcp_launch.py            # Launch script
│   ├── mcp_model.py             # Pydantic models
│   └── mcp_resources.py         # Simulated data and resources
├── pyproject.toml               # Package configuration
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

### Adding New Collections

To add a new simulated collection, edit `mcp_resources.py`:

```python
self.collections["my_collection"] = {
    "schema": {
        "name": "my_collection",
        "description": "Description",
        "fields": [
            {"name": "id", "type": "INT64", "is_primary": True},
            # ... more fields
        ]
    },
    "data": [
        # ... your data
    ],
    "count": len(data)
}
```

### Adding New Tools

To add a new tool, add a method to `MCPEnv` in `mcp_environment.py`:

```python
def _tool_my_tool(self, param1: str, **kwargs) -> str:
    """Tool description."""
    # Implementation
    return result

# Register in __init__
self.tools["my_tool"] = self._tool_my_tool
```

## Integration as Environment Actions

The MCP environment demonstrates key concepts for tool integration:

1. **Tool Definition**: Each tool is a method in the environment
2. **Action Parsing**: Actions are parsed from LLM output (ReAct format)
3. **Parameter Extraction**: JSON parameters are extracted and validated
4. **Execution**: Tools are executed with parsed parameters
5. **Response Formatting**: Results are formatted as observations
6. **Reward Signaling**: Successful actions receive rewards

This pattern can be extended to integrate any set of tools or APIs as environment actions.

## Example Use Cases

### 1. Information Retrieval
```
Goal: Find all documents about machine learning
Tools: list_collections → get_collection_info → search_collection → finish
```

### 2. Data Exploration
```
Goal: Explore available collections and their schemas
Tools: list_collections → get_collection_info (multiple) → finish
```

### 3. Semantic Search
```
Goal: Find similar products to a query
Tools: get_collection_info → search_collection → format_response → finish
```

### 4. Prompt Engineering
```
Goal: Use a prompt template for a task
Tools: get_prompt → search_collection → finish
```

## Contributing

See [CONTRIB.md](../CONTRIB.md) for general contribution guidelines.

For MCP-specific contributions:
1. Keep simulated data realistic but lightweight
2. Ensure all tools have clear docstrings
3. Add tests for new tools or collections
4. Update this README with new features

## License

MIT License - See [LICENSE](../LICENSE) for details.

## References

- [AgentGym Paper](https://arxiv.org/abs/2406.04151)
- [Milvus Vector Database](https://milvus.io/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
