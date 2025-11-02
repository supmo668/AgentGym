# MCP Environment Architecture

This document describes the architecture of the MCP (Model Context Protocol) environment.

## Overview

The MCP environment demonstrates how to integrate complex tool-based systems into AgentGym. It simulates a Model Context Protocol server with Milvus vector database collections.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         LLM Agent                               │
│  - Generates thoughts and actions                               │
│  - Uses ReAct format                                            │
└────────────────────────┬────────────────────────────────────────┘
                         │ Natural Language Actions
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              MCPEnvClient (agentenv/envs/mcp.py)                │
│  - Manages HTTP communication with server                       │
│  - Maintains conversation history                               │
│  - Caches state locally                                         │
│  - Implements BaseEnvClient interface                           │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP POST/GET
                         │ JSON payloads
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│           FastAPI Server (mcp_server.py)                        │
│  Endpoints:                                                     │
│  - POST /create      → Create environment                       │
│  - POST /step        → Execute action                           │
│  - POST /reset       → Reset to initial state                   │
│  - GET  /observation → Get current state                        │
│  - POST /close       → Cleanup environment                      │
│  - GET  /health      → Server health check                      │
└────────────────────────┬────────────────────────────────────────┘
                         │ Validated Requests
                         │ (Pydantic models)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│         MCPEnvServer (mcp_server_wrapper.py)                    │
│  - Manages multiple environment instances                       │
│  - Thread-safe ID allocation                                    │
│  - Environment lifecycle management                             │
│  - Task dataset management                                      │
│                                                                 │
│  State:                                                         │
│  - env: Dict[int, MCPEnv]          # ID → Environment          │
│  - info: Dict[int, dict]           # Cached observations       │
│  - tasks: List[dict]               # Task configurations       │
└────────────────────────┬────────────────────────────────────────┘
                         │ Method Calls
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              MCPEnv (mcp_environment.py)                        │
│  Core Environment (inherits gym.Env)                            │
│                                                                 │
│  State:                                                         │
│  - goal: str                       # Current task goal         │
│  - observation: str                # Current state             │
│  - current_step: int               # Step counter              │
│  - action_history: List[dict]      # Action log                │
│                                                                 │
│  Methods:                                                       │
│  - step(action)    → Execute action and return obs/reward/done │
│  - reset(idx)      → Initialize to task idx                    │
│  - observation()   → Get current state                         │
│  - close()         → Cleanup resources                         │
└──────────────┬──────────────────────────────┬───────────────────┘
               │                              │
               │ Uses                         │ Uses
               ▼                              ▼
┌──────────────────────────┐   ┌─────────────────────────────────┐
│  SimulatedMilvusCollections│  │     MCPResources               │
│  (mcp_resources.py)        │  │     (mcp_resources.py)         │
│                           │  │                                 │
│  Collections:             │  │  Resources:                     │
│  - documents (768-dim)    │  │  - Schemas                      │
│    * 3 ML articles        │  │    * tool_schema                │
│  - users (512-dim)        │  │    * response_schema            │
│    * 2 user profiles      │  │  - Prompts                      │
│  - products (384-dim)     │  │    * search_prompt              │
│    * 3 products           │  │    * query_prompt               │
│                           │  │    * vector_search_prompt       │
│  Methods:                 │  │  - Formatters                   │
│  - list_collections()     │  │    * json_formatter             │
│  - get_collection_info()  │  │    * text_formatter             │
│  - query_collection()     │  │    * markdown_formatter         │
│  - search_collection()    │  │                                 │
└───────────────────────────┘  └─────────────────────────────────┘
```

## Component Descriptions

### 1. LLM Agent (User Code)
- **Purpose**: Generate actions to accomplish tasks
- **Format**: ReAct (Thought + Action + Action Input)
- **Interface**: Uses MCPEnvClient or MCPTask

### 2. MCPEnvClient
- **Location**: `agentenv/agentenv/envs/mcp.py`
- **Purpose**: Client-side interface for agent interaction
- **Key Features**:
  - HTTP communication with server
  - State caching
  - Conversation management
  - Implements AgentGym's BaseEnvClient

### 3. FastAPI Server
- **Location**: `agentenv-mcp/agentenv_mcp/mcp_server.py`
- **Purpose**: HTTP API layer
- **Key Features**:
  - RESTful endpoints
  - Request validation (Pydantic)
  - Automatic documentation (/docs)
  - Error handling

### 4. MCPEnvServer
- **Location**: `agentenv-mcp/agentenv_mcp/mcp_server_wrapper.py`
- **Purpose**: Server-side environment manager
- **Key Features**:
  - Multi-instance management
  - Thread-safe operations
  - Task dataset
  - Lifecycle management

### 5. MCPEnv
- **Location**: `agentenv-mcp/agentenv_mcp/mcp_environment.py`
- **Purpose**: Core environment logic
- **Key Features**:
  - Tool execution
  - Action parsing
  - Reward calculation
  - State management
  - Inherits gymnasium.Env

### 6. SimulatedMilvusCollections
- **Location**: `agentenv-mcp/agentenv_mcp/mcp_resources.py`
- **Purpose**: Simulated vector database
- **Key Features**:
  - 3 pre-populated collections
  - Vector similarity search simulation
  - Query and filtering

### 7. MCPResources
- **Location**: `agentenv-mcp/agentenv_mcp/mcp_resources.py`
- **Purpose**: Resource management
- **Key Features**:
  - Schema definitions
  - Prompt templates
  - Response formatters

## Data Flow

### Action Execution Flow

```
1. Agent generates action:
   "Thought: I need to list collections
    Action: list_collections with Action Input: {}"

2. Client sends HTTP POST to /step:
   {
     "env_idx": 0,
     "action": "Action: list_collections with Action Input: {}"
   }

3. Server validates and routes to MCPEnvServer.step()

4. MCPEnvServer calls MCPEnv.step()

5. MCPEnv:
   a. Parses action string
   b. Extracts tool name: "list_collections"
   c. Extracts parameters: {}
   d. Executes tool method: _tool_list_collections()
   e. Calculates reward
   f. Returns (observation, reward, done, info)

6. MCPEnvServer caches result and returns to server

7. FastAPI server formats response:
   {
     "observation": "Available collections: documents, users, products",
     "reward": 0.1,
     "done": false
   }

8. Client receives response and returns StepOutput

9. Agent receives observation and generates next action
```

## Tools as Environment Actions

The MCP environment demonstrates tool integration:

```python
# Tool Definition (in MCPEnv.__init__)
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

# Tool Execution (in MCPEnv.step)
tool_name, params = self._parse_action(action)
result = self.tools[tool_name](**params)
```

Each tool:
1. Has a clear interface (parameters and return type)
2. Performs a specific operation
3. Returns a string observation
4. Can access shared resources (Milvus collections, schemas, etc.)

## Thread Safety

- **Global Lock**: Used only for ID allocation
- **Environment Isolation**: Each env_id has independent state
- **No Shared State**: Tools operate on local environment state
- **Concurrent Requests**: Different environments can run in parallel

## Testing Strategy

### 1. Unit Tests
- Test individual components in isolation
- Test environment methods directly
- Test resource classes

### 2. Integration Tests
- Test client-server communication
- Test full action execution flow
- Test multiple environments

### 3. Example Tests
- Demonstrate real-world usage
- Validate end-to-end functionality
- Serve as documentation

## Extension Points

### Adding New Tools

```python
# 1. Add tool method to MCPEnv
def _tool_my_new_tool(self, param1: str, **kwargs) -> str:
    """Tool description."""
    # Implementation
    return result

# 2. Register in __init__
self.tools["my_new_tool"] = self._tool_my_new_tool

# 3. Update client conversation_start with tool description
```

### Adding New Collections

```python
# In SimulatedMilvusCollections.__init__
self.collections["new_collection"] = {
    "schema": {...},
    "data": [...],
    "count": len(data)
}
```

### Adding New Resources

```python
# In MCPResources
def _init_prompts(self):
    return {
        # existing prompts
        "new_prompt": "Template: {param1}, {param2}",
    }
```

## Performance Considerations

1. **Caching**: Server caches last observation to reduce compute
2. **Lazy Loading**: Resources loaded once, reused across instances
3. **Lightweight Data**: Simulated data kept small for fast operations
4. **Async Server**: FastAPI handles concurrent requests efficiently

## Security Considerations

1. **Input Validation**: Pydantic models validate all inputs
2. **Action Parsing**: Safe parsing with JSON/regex
3. **No Eval**: No dynamic code execution
4. **Resource Limits**: Max steps prevent infinite loops
5. **Error Handling**: Graceful degradation on errors

## Future Enhancements

Possible improvements:
- [ ] Add more collections with diverse schemas
- [ ] Implement actual vector similarity computation
- [ ] Add collection filtering and advanced queries
- [ ] Support for custom embeddings
- [ ] Add authentication/authorization
- [ ] Metrics and monitoring endpoints
- [ ] Batch operation support
- [ ] Async tool execution

## References

- [AgentGym Paper](https://arxiv.org/abs/2406.04151)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Milvus Documentation](https://milvus.io/)
