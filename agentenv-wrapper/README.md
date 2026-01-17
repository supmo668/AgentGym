# AgentEnv Wrapper

Generic tool-based wrapper for AgentGym environments, enabling integration with agent frameworks like LangChain.

## Overview

This package bridges AgentGym's environment-action paradigm with modern LLM agent frameworks that use a tool-calling interface. It provides:

- **Base Wrapper Classes**: Convert any AgentEnv into a tool-based interface
- **Environment Adapters**: Specialized wrappers for specific environments (BabyAI, etc.)
- **LangChain Integration**: Ready-to-use LangChain tools and agents

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Framework                          │
│              (LangChain, Custom Agent, etc.)               │
└─────────────────────────┬───────────────────────────────────┘
                          │ Tool Calls
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  AgentEnv Wrapper                           │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ BaseToolWrapper │  │   LangChain     │                  │
│  │ (Abstract Base) │  │   Integration   │                  │
│  └────────┬────────┘  └─────────────────┘                  │
│           │                                                 │
│  ┌────────▼────────────────────────────────────┐           │
│  │         Environment Adapters                 │           │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │           │
│  │  │ BabyAI   │  │ WebShop  │  │  Custom  │  │           │
│  │  │ Adapter  │  │ Adapter  │  │ Adapter  │  │           │
│  │  └──────────┘  └──────────┘  └──────────┘  │           │
│  └─────────────────────────────────────────────┘           │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP Requests
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              AgentGym Environment Servers                   │
│         (BabyAI Server, WebShop Server, etc.)              │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# From the AgentGym repository root
pip install -e ./agentenv-wrapper

# With LangChain support
pip install -e "./agentenv-wrapper[langchain]"
```

## Quick Start

### Basic Usage

```python
from agentenv_wrapper import BabyAIToolWrapper

# Create wrapper connected to environment server
wrapper = BabyAIToolWrapper("http://localhost:8080")

# Reset to a specific task
state = wrapper.reset(task_id=1)
print(f"Goal: {state.goal}")
print(f"Observation: {state.observation}")

# Get available tools
tools = wrapper.get_available_tools()
for tool in tools:
    print(f"  - {tool.name}: {tool.description}")

# Execute an action
result = wrapper.execute_tool("move forward")
print(f"New observation: {result.output}")
print(f"Done: {result.done}")
```

### With LangChain

```python
from langchain_openai import ChatOpenAI
from agentenv_wrapper import BabyAIToolWrapper
from agentenv_wrapper.langchain import create_agent_for_env

# Setup
wrapper = BabyAIToolWrapper("http://localhost:8080")
wrapper.reset(task_id=1)

llm = ChatOpenAI(model="gpt-4", temperature=0)

# Create agent using LangChain v1 syntax (initialize_agent)
agent = create_agent_for_env(wrapper, llm)

# Run the agent
result = agent.invoke({"input": wrapper.get_system_prompt()})
print(result["output"])
```

### Using AgentEnvRunner

```python
from langchain_openai import ChatOpenAI
from agentenv_wrapper import BabyAIToolWrapper
from agentenv_wrapper.langchain import AgentEnvRunner

# Setup
wrapper = BabyAIToolWrapper("http://localhost:8080")
llm = ChatOpenAI(model="gpt-4")

# Create runner
runner = AgentEnvRunner(wrapper, llm, max_iterations=30)

# Run a single episode
result = runner.run_episode(task_id=1)
print(f"Success: {result['success']}")
print(f"Steps: {result['steps']}")
print(f"Reward: {result['reward']}")

# Run multiple episodes
task_ids = list(range(1, 11))  # Tasks 1-10
results = runner.run_batch(task_ids)
metrics = runner.compute_metrics(results)
print(f"Success rate: {metrics['success_rate']:.2%}")
```

## Creating Custom Adapters

To wrap a new AgentGym environment, extend `BaseToolWrapper`:

```python
from agentenv_wrapper import BaseToolWrapper, ActionTool, EnvState, ToolResult

class MyEnvToolWrapper(BaseToolWrapper):
    def __init__(self, env_server_base: str):
        super().__init__(env_name="MyEnv")
        self.env_server_base = env_server_base
        # ... initialize connection
    
    def reset(self, task_id: int = 0) -> EnvState:
        # Reset environment and return initial state
        ...
    
    def get_state(self) -> EnvState:
        # Return current environment state
        return EnvState(
            observation=self._observation,
            available_actions=self._actions,
            goal=self._goal,
            reward=self._reward,
            done=self._done,
        )
    
    def get_available_tools(self) -> list[ActionTool]:
        # Convert actions to tools
        return [
            ActionTool(
                name=action,
                description=f"Execute {action}",
                handler=lambda a=action: self.execute_tool(a),
            )
            for action in self._actions
        ]
    
    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        # Execute action and return result
        ...
```

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `BaseToolWrapper` | Abstract base for environment wrappers |
| `ToolEnvironmentWrapper` | Generic wrapper for any AgentEnv server |
| `BabyAIToolWrapper` | Specialized wrapper for BabyAI |

### Data Models

| Model | Description |
|-------|-------------|
| `ActionTool` | Represents an action as a callable tool |
| `ToolDefinition` | Schema definition for a tool |
| `ToolResult` | Result from executing a tool |
| `EnvState` | Current environment state |

### LangChain Integration

| Function/Class | Description |
|----------------|-------------|
| `create_langchain_tools()` | Convert wrapper tools to LangChain tools |
| `create_agent_for_env()` | Create a LangChain agent for an environment |
| `AgentEnvRunner` | High-level runner for episodes |

## Supported Environments

Currently supported with specialized adapters:

- ✅ BabyAI - Gridworld navigation with instructions

Planned:
- 🔄 WebShop - E-commerce navigation
- 🔄 TextCraft - Crafting game
- 🔄 SciWorld - Science experiments

Any environment following the AgentGym server protocol can be used with `ToolEnvironmentWrapper`.

## Contributing

See the main AgentGym repository for contribution guidelines.

## License

MIT License - see the repository root for details.
