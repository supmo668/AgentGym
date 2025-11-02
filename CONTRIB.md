# Contributing to AgentGym

Welcome to AgentGym! This guide will help you get started with contributing to the project, whether you're adding new environments, fixing bugs, or improving documentation.

## Table of Contents

1. [Quick Start (5 Minutes)](#quick-start-5-minutes)
2. [Getting Started](#getting-started)
3. [Development Environment Setup](#development-environment-setup)
4. [Architecture Overview](#architecture-overview)
5. [Creating a New Environment](#creating-a-new-environment)
6. [Code Standards and Conventions](#code-standards-and-conventions)
7. [Testing Guidelines](#testing-guidelines)
8. [Contribution Workflow](#contribution-workflow)
9. [Resources](#resources)

## Quick Start (5 Minutes)

### Installation

```bash
# 1. Clone the repository
git clone --recursive https://github.com/WooooDyy/AgentGym
cd AgentGym

# 2. Install core package
cd agentenv
pip install -e .
cd ..

# 3. (Optional) Install a specific environment
cd agentenv-mcp  # Example: MCP environment
pip install -e .
cd ..
```

### Try the MCP Environment

The MCP (Model Context Protocol) environment is a great example to get started:

```bash
# Terminal 1: Start MCP server
mcp --host 127.0.0.1 --port 8000

# Terminal 2: Run example
cd agentenv-mcp
python example_usage.py
```

Or use in code:

```python
from agentenv.envs.mcp import MCPEnvClient

client = MCPEnvClient(
    env_server_base="http://127.0.0.1:8000",
    data_len=10
)
response = client.reset(data_idx=0)
result = client.step("Action: list_collections with Action Input: {}")
client.close()
```

### Key Concepts

- **Environment**: Core logic (inherits `gym.Env`)
- **Server**: FastAPI wrapper exposing HTTP endpoints
- **Client**: Agent-side interface for HTTP communication
- **Task**: Binds client to task configuration

### Common Tasks

- **Add New Environment**: Copy template from `agentenv-mcp/` or `agentenv-textcraft/`
- **Run Tests**: `pytest agentenv-mcp/tests/`
- **Format Code**: `black agentenv-yourenv/`
- **Check Style**: `flake8 agentenv-yourenv/`

## Getting Started

AgentGym is a framework for evolving Large Language Model-based agents across diverse environments. The project supports multiple interactive environments with a unified ReAct format interface.

### Prerequisites

- Python >= 3.10
- Git
- Conda (recommended) or virtualenv
- Basic understanding of:
  - FastAPI for server development
  - Gymnasium for environment interfaces
  - LLM agent architectures

## Development Environment Setup

### 1. Clone the Repository

```bash
git clone --recursive https://github.com/WooooDyy/AgentGym
cd AgentGym
```

### 2. Set Up the Core Package

```bash
cd agentenv
pip install -e .
```

### 3. Set Up Specific Environments

Each environment is in its own `agentenv-*` directory. To work with a specific environment:

```bash
cd agentenv-<environment-name>
# Follow the environment-specific README.md for setup instructions
```

### 4. Install Development Dependencies

```bash
pip install pytest pytest-cov black flake8 mypy
```

## Architecture Overview

AgentGym uses a client-server architecture to decouple environments from agents:

```
┌─────────────────┐         ┌──────────────────┐
│  Agent/         │  HTTP   │  Environment     │
│  Controller     │◄──────►│  Server          │
│  (Client)       │         │  (FastAPI)       │
└─────────────────┘         └──────────────────┘
                                    │
                            ┌───────┴────────┐
                            │  Environment   │
                            │  Instance      │
                            │  (Gym.Env)     │
                            └────────────────┘
```

### Key Components

1. **Environment Server** (`agentenv-*/`)
   - Implements the core environment logic (inherits `gym.Env`)
   - Wraps environment instances with a server manager
   - Exposes HTTP endpoints via FastAPI

2. **Environment Client** (`agentenv/agentenv/envs/`)
   - Communicates with environment server via HTTP
   - Parses LLM outputs and extracts actions
   - Maintains local state cache

3. **Task Class** (`agentenv/agentenv/envs/`)
   - Binds environment client to task configuration
   - Manages multiple parallel clients

4. **Controller** (`agentenv/agentenv/controller/`)
   - Orchestrates agent-environment interaction
   - Handles evaluation, data collection, and training

## Creating a New Environment

Follow these steps to add a custom environment. We recommend using `agentenv-textcraft` or `agentenv-tool` as reference implementations.

### Directory Structure

```
AgentGym/
└── agentenv-<your-env>/
    ├── pyproject.toml          # Package configuration
    ├── README.md               # Environment-specific documentation
    ├── requirements.txt        # Environment dependencies
    ├── setup.sh               # (Optional) Setup script
    └── agentenv_<your_env>/
        ├── __init__.py
        ├── environment.py      # Core environment logic
        ├── server.py          # FastAPI server
        ├── launch.py          # Server launcher
        ├── model.py           # Pydantic models
        └── utils.py           # (Optional) Helper functions
```

### Step 1: Implement the Environment Class

Create `environment.py` with a class inheriting from `gym.Env`:

```python
import gym
from typing import Any, Dict, Tuple

class YourEnv(gym.Env[str, str]):
    def __init__(self, **config):
        """Initialize environment with configuration."""
        self.observation = None
        # Initialize your environment state
        
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        Execute an action and return results.
        
        Args:
            action: Action string from the agent
            
        Returns:
            observation: New state observation (str)
            reward: Immediate reward (float)
            terminated: Whether episode is complete (bool)
            info: Additional information (dict)
        """
        # Parse and validate action
        # Execute action logic
        # Update environment state
        # Calculate reward
        
        return self.observation, reward, terminated, info
    
    def reset(self, idx: int = 0) -> str:
        """
        Reset environment to initial state.
        
        Args:
            idx: Task/dataset index
            
        Returns:
            Initial observation
        """
        # Reset environment state
        # Load task data if needed
        return self.observation
    
    def observation(self) -> str:
        """Return current observation."""
        return self.observation
    
    def close(self):
        """Clean up resources."""
        super().close()
```

### Step 2: Implement the Environment Server

Create a server wrapper in `environment.py`:

```python
import threading
from typing import Dict

class YourEnvServer:
    """Server-side environment manager."""
    
    def __init__(self):
        self._max_id: int = 0
        self.env: Dict[int, YourEnv] = {}
        self.info: Dict[int, dict] = {}
        self.ls: list[int] = []
        self._lock = threading.Lock()
    
    def create(self) -> dict:
        """Create a new environment instance."""
        with self._lock:
            env_id = self._max_id
            self._max_id += 1
        
        new_env = YourEnv()
        new_env.reset(idx=0)
        
        self.ls.append(env_id)
        self.env[env_id] = new_env
        
        return {"id": env_id}
    
    def step(self, env_idx: int, action: str) -> dict:
        """Execute action in environment."""
        env = self.env[env_idx]
        ob, reward, terminated, info = env.step(action)
        
        self.info[env_idx] = {
            "observation": ob,
            "reward": reward,
            "done": terminated,
            **info
        }
        
        return {
            "observation": ob,
            "reward": reward,
            "done": terminated,
        }
    
    def reset(self, env_idx: int, data_idx: int) -> dict:
        """Reset environment to specific task."""
        ob = self.env[env_idx].reset(idx=data_idx)
        self.info[env_idx] = {
            "observation": ob,
            "reward": 0.0,
            "done": False
        }
        return self.info[env_idx]
    
    def observation(self, env_idx: int) -> dict:
        """Get current observation."""
        if env_idx in self.info:
            return {"observation": self.info[env_idx]["observation"]}
        return {"error": "Env not initialized"}
    
    def close(self, env_id: int) -> dict:
        """Close and cleanup environment."""
        try:
            if env_id in self.ls:
                self.ls.remove(env_id)
            env = self.env.pop(env_id)
            env.close()
            self.info.pop(env_id, None)
            return {"closed": True}
        except Exception as e:
            return {"closed": False, "error": str(e)}
```

### Step 3: Implement FastAPI Server

Create `server.py`:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from .environment import YourEnvServer

app = FastAPI()
server = YourEnvServer()

class StepRequestBody(BaseModel):
    id: int
    action: str

class ResetRequestBody(BaseModel):
    id: int
    data_idx: int = 0

class CloseRequestBody(BaseModel):
    id: int

@app.get("/")
def hello():
    return "YourEnv environment server"

@app.post("/create")
def create():
    return server.create()

@app.post("/step")
def step(body: StepRequestBody):
    return server.step(body.id, body.action)

@app.post("/reset")
def reset(body: ResetRequestBody):
    return server.reset(body.id, body.data_idx)

@app.get("/observation")
def get_observation(id: int):
    return server.observation(id)

@app.post("/close")
def close(body: CloseRequestBody):
    return server.close(body.id)
```

### Step 4: Implement Environment Client

Create the client in `agentenv/agentenv/envs/yourenv.py`:

```python
import requests
from typing import Any, Mapping, Dict
from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import ConversationMessage, StepOutput

class YourEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage({
            "from": "human",
            "loss": None,
            "value": "Your task instructions here..."
        }),
        ConversationMessage({
            "from": "gpt",
            "loss": False,
            "value": "OK. I'll follow your instructions."
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
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len
        
        # Create environment on server
        ok = requests.post(
            f"{self.env_server_base}/create",
            timeout=self.timeout
        )
        assert ok.status_code == 200
        self.env_id = ok.json()["id"]
    
    def __len__(self):
        return self.data_len
    
    def _post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        data["id"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout
        )
        assert res.status_code == 200
        return res.json()
    
    def _get(self, path: str) -> Dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?id={self.env_id}",
            timeout=self.timeout
        )
        assert res.status_code == 200
        return res.json()
    
    def observe(self) -> str:
        """Get current observation."""
        if self.info:
            return self.info["observation"]
        response = self._get("observation")
        return response["observation"]
    
    def step(self, action: str) -> StepOutput:
        """Execute action."""
        # Parse action from LLM output if needed
        response = self._post("step", {"action": action})
        self.info = {
            "observation": response["observation"],
            "reward": response["reward"],
            "done": response["done"],
        }
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"]
        )
    
    def reset(self, data_idx: int = 0) -> Dict[str, Any]:
        """Reset environment."""
        response = self._post("reset", {"data_idx": data_idx})
        self.info = {
            "observation": response["observation"],
            "reward": response["reward"],
            "done": response["done"],
        }
        return response
    
    def close(self) -> Dict[str, Any]:
        """Close environment."""
        return self._post("close", {})

class YourEnvTask(BaseTask):
    env_client_cls = YourEnvClient
    env_name = "YourEnv"
    
    def __init__(
        self,
        client_args: Mapping[str, Any],
        n_clients: int,
        *args,
        **kwargs
    ):
        super().__init__(client_args, n_clients, *args, **kwargs)
```

### Step 5: Create Launch Script

Create `launch.py`:

```python
import uvicorn
import click

@click.command()
@click.option("--host", default="127.0.0.1", help="Host to bind")
@click.option("--port", default=8000, help="Port to bind")
def main(host: str, port: int):
    """Launch YourEnv environment server."""
    uvicorn.run(
        "agentenv_yourenv.server:app",
        host=host,
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    main()
```

### Step 6: Add Package Configuration

Create `pyproject.toml`:

```toml
[project]
name = "agentenv-yourenv"
version = "0.1.0"
description = "YourEnv environment for AgentGym"
dependencies = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "click",
    "gymnasium",
]
requires-python = ">=3.10"

[project.scripts]
yourenv = "agentenv_yourenv.launch:main"
```

## Code Standards and Conventions

### Python Style

- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Maximum line length: 100 characters
- Use meaningful variable names

### Formatting

```bash
# Format code with black
black agentenv-yourenv/

# Check with flake8
flake8 agentenv-yourenv/
```

### Documentation

- Add docstrings to all classes and functions
- Use Google-style docstrings
- Include type information in docstrings
- Document any non-obvious behavior

Example:
```python
def step(self, action: str) -> Tuple[str, float, bool, Dict]:
    """
    Execute an action in the environment.
    
    Args:
        action: The action string to execute. Format depends on environment.
    
    Returns:
        A tuple containing:
        - observation (str): The new state observation
        - reward (float): The immediate reward
        - terminated (bool): Whether the episode has ended
        - info (dict): Additional diagnostic information
        
    Raises:
        ValueError: If action format is invalid
    """
```

### API Conventions

#### Required HTTP Endpoints

All environment servers must implement:

1. `POST /create` - Create environment instance
   - Returns: `{"id": int}`

2. `POST /step` - Execute action
   - Body: `{"id": int, "action": str}`
   - Returns: `{"observation": str, "reward": float, "done": bool}`

3. `POST /reset` - Reset environment
   - Body: `{"id": int, "data_idx": int}`
   - Returns: `{"observation": str, "reward": float, "done": bool}`

4. `GET /observation` - Get current observation
   - Query: `?id=int`
   - Returns: `{"observation": str}`

5. `POST /close` - Close environment (optional but recommended)
   - Body: `{"id": int}`
   - Returns: `{"closed": bool}`

#### Data Format

- Use ReAct format for agent interactions:
  ```
  Thought: [reasoning]
  
  Action: [action_name] with Action Input: [action_input]
  ```

- Observations should be descriptive strings
- Rewards should be numeric (float)
- Use `done=True` to signal episode termination

### Thread Safety

- Use locks when allocating environment IDs
- Ensure concurrent requests to different environment instances don't interfere
- Consider per-environment locks for step/reset operations

## Testing Guidelines

### Unit Tests

Create tests in a `tests/` directory:

```python
import pytest
from agentenv_yourenv.environment import YourEnv

def test_environment_creation():
    """Test environment can be created."""
    env = YourEnv()
    assert env is not None

def test_reset():
    """Test environment reset."""
    env = YourEnv()
    obs = env.reset(idx=0)
    assert isinstance(obs, str)
    assert len(obs) > 0

def test_step():
    """Test environment step."""
    env = YourEnv()
    env.reset(idx=0)
    obs, reward, done, info = env.step("test action")
    assert isinstance(obs, str)
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(info, dict)
```

### Integration Tests

Test the full client-server interaction:

```python
import requests
import pytest
from multiprocessing import Process
import time
import uvicorn

@pytest.fixture
def server():
    """Start test server."""
    def run_server():
        uvicorn.run(
            "agentenv_yourenv.server:app",
            host="127.0.0.1",
            port=8888
        )
    
    proc = Process(target=run_server)
    proc.start()
    time.sleep(2)  # Wait for server to start
    yield "http://127.0.0.1:8888"
    proc.terminate()
    proc.join()

def test_create_endpoint(server):
    """Test create endpoint."""
    response = requests.post(f"{server}/create")
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
```

### Running Tests

```bash
# Run all tests
pytest agentenv-yourenv/tests/

# Run with coverage
pytest --cov=agentenv_yourenv agentenv-yourenv/tests/

# Run specific test
pytest agentenv-yourenv/tests/test_environment.py::test_reset
```

## Contribution Workflow

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR-USERNAME/AgentGym
cd AgentGym
git remote add upstream https://github.com/WooooDyy/AgentGym
```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 3. Make Changes

- Write clear, focused commits
- Follow code standards
- Add tests for new functionality
- Update documentation

### 4. Test Your Changes

```bash
# Run tests
pytest

# Check code style
black --check .
flake8 .

# Type check (if applicable)
mypy agentenv-yourenv/
```

### 5. Commit and Push

```bash
git add .
git commit -m "feat: add YourEnv environment

- Implement environment class
- Add FastAPI server
- Create client and task classes
- Add tests and documentation"

git push origin feature/your-feature-name
```

### 6. Create Pull Request

1. Go to GitHub and create a Pull Request
2. Fill in the PR template
3. Link related issues
4. Wait for review

### Commit Message Convention

Follow conventional commits:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Adding or updating tests
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

## Resources

### Documentation

- [AgentGym Paper](https://arxiv.org/abs/2406.04151)
- [Project Website](https://agentgym.github.io/)
- [Tutorials](./docs/tutorials/)
- [Environment Development Guide](./docs/tutorials/en/05-2nd-Development.md)

### Example Environments

- Simple: `agentenv-textcraft`, `agentenv-babyai`
- Tool-based: `agentenv-tool` (weather, movie, etc.)
- Complex: `agentenv-webarena`, `agentenv-alfworld`

### Getting Help

- Open an issue on GitHub
- Check existing issues and discussions
- Review the tutorials and documentation
- Contact maintainers: zhxi22@m.fudan.edu.cn

### Key Dependencies

- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [Gymnasium](https://gymnasium.farama.org/) - Environment interface
- [Pydantic](https://docs.pydantic.dev/) - Data validation
- [Uvicorn](https://www.uvicorn.org/) - ASGI server

## Thank You!

Thank you for contributing to AgentGym! Your contributions help build better tools for the AI agent research community.
