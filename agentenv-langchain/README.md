# AgentEnv LangChain

A general wrapper for adapting any AgentGym environment into a tool-based environment suitable for LangChain's `create_react_agent` (v1 syntax).

## Overview

This package provides a bridge between AgentGym environments and LangChain agents. Instead of creating environment-specific MCP servers, this wrapper allows you to adapt **any** existing AgentGym environment into LangChain-compatible tools.

### Architecture

```text
┌─────────────────────┐         ┌─────────────────────┐
│   LangChain Agent   │         │  AgentGym Server    │
│  (create_react_agent)│        │  (BabyAI, ALFWorld) │
│                     │         │                     │
│  ┌───────────────┐  │  HTTP   │  POST /step         │
│  │ env_turn_left │──┼─────────┼─►POST /reset        │
│  │ env_move_fwd  │  │         │  GET /observation   │
│  │ env_observe   │  │         │                     │
│  │ env_reset     │  │         │                     │
│  └───────────────┘  │         │                     │
└─────────────────────┘         └─────────────────────┘
        │                               │
        └──────── AgentEnvToolWrapper ──┘
```

## Installation

```bash
cd agentenv-langchain
pip install -e .

# With OpenAI support
pip install -e ".[openai]"

# With all LLM providers
pip install -e ".[all]"
```

## Quick Start

### 1. Using Pre-defined Environment Actions

```python
from agentenv_langchain import create_wrapper_for_env, create_agentenv_tools
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# Create wrapper for BabyAI (has pre-defined actions)
wrapper = create_wrapper_for_env("BabyAI", "http://localhost:8000")

# Convert to LangChain tools
tools = create_agentenv_tools(wrapper)

# Create the agent
llm = ChatOpenAI(model="gpt-4")
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run an episode
wrapper.reset(0)  # Reset to task 0
result = executor.invoke({"input": "Complete the navigation task"})
```

### 2. Custom Environment Actions

```python
from agentenv_langchain import AgentEnvToolWrapper, EnvAction, create_agentenv_tools

# Define your environment's action space
actions = [
    EnvAction(
        name="search",
        description="Search for information",
        parameters={"query": "string"},
        required_params=["query"],
    ),
    EnvAction(
        name="click",
        description="Click on an element",
        parameters={"element_id": "string"},
        required_params=["element_id"],
    ),
    EnvAction(
        name="scroll",
        description="Scroll the page",
    ),
]

# Create wrapper
wrapper = AgentEnvToolWrapper(
    env_server_base="http://localhost:8001",
    env_name="WebArena",
    actions=actions,
)

# Get LangChain tools
tools = create_agentenv_tools(wrapper)
```

### 3. Using the Convenience Function

```python
from agentenv_langchain.tools import create_react_agent_for_env
from agentenv_langchain import create_wrapper_for_env
from langchain_openai import ChatOpenAI

wrapper = create_wrapper_for_env("BabyAI", "http://localhost:8000")
llm = ChatOpenAI(model="gpt-4")

# One-liner to create the full agent
executor = create_react_agent_for_env(wrapper, llm)

wrapper.reset(0)
result = executor.invoke({"input": "Navigate to the goal"})
```

## Supported Environments

The wrapper includes pre-defined action sets for:

| Environment | Actions |
|-------------|---------|
| BabyAI | turn_left, turn_right, move_forward, go_to, pick_up, go_through, toggle_and_go_through, toggle |
| ALFWorld | goto, take, put, open, close, toggle, clean, heat, cool, examine, inventory, look |

For other environments, define your own `EnvAction` list.

## API Reference

### AgentEnvToolWrapper

```python
class AgentEnvToolWrapper:
    def __init__(
        self,
        env_server_base: str,      # Base URL of AgentGym server
        env_name: str,             # Environment name
        actions: List[EnvAction],  # Available actions
        action_parser: Callable,   # Optional custom action parser
        observation_parser: Callable,  # Optional custom observation parser
        timeout: int = 300,        # Request timeout
        auto_create: bool = True,  # Auto-create env on init
    ): ...
    
    def reset(self, data_idx: int = 0) -> EnvObservation: ...
    def step(self, action: str, **kwargs) -> EnvObservation: ...
    def observe(self) -> EnvObservation: ...
    def close(self) -> bool: ...
    def as_langchain_tools(self) -> List[Tool]: ...
```

### EnvAction

```python
@dataclass
class EnvAction:
    name: str                          # Action name
    description: str                   # Description for LLM
    parameters: Dict[str, Any] = {}    # Parameter types
    required_params: List[str] = []    # Required parameters
```

### EnvObservation

```python
@dataclass
class EnvObservation:
    state: str                         # Current state description
    reward: float                      # Reward received
    done: bool                         # Episode done flag
    info: Dict[str, Any] = {}          # Additional info
    available_actions: List[str] = []  # Available action names
```

## Running Examples

### BabyAI Example

1. Start the BabyAI server:
```bash
cd ../agentenv-babyai
uvicorn agentenv_babyai.server:app --port 8000
```

2. Run the example:
```bash
export OPENAI_API_KEY=your-key
python examples/babyai_react_agent.py
```

### CLI

```bash
# List available actions for an environment
agentenv-langchain list-actions --env babyai

# Run demo (requires server running)
agentenv-langchain demo --server-url http://localhost:8000
```

## Comparison with agentenv-mcp

| Feature | agentenv-mcp | agentenv-langchain |
|---------|-------------|-------------------|
| Protocol | MCP (Model Context Protocol) | HTTP/REST |
| Agent Framework | Custom MCP client | LangChain |
| Environment Support | Requires MCP server per env | Works with any AgentGym server |
| Tool Definition | MCP tool decorators | EnvAction dataclass |
| Use Case | MCP-native systems | LangChain-based agents |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Run tests: `pytest`
5. Submit a pull request

## License

Apache 2.0
