# Contributing to AgentGym

Welcome to AgentGym! 🎉 We're excited that you're interested in contributing to our project. AgentGym is a framework for building and evaluating generally-capable LLM-based agents across diverse environments.

This guide will help you get started with setting up your development environment, understanding the project structure, and making your first contribution.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Setting Up Development Environment](#setting-up-development-environment)
- [Working with Environments](#working-with-environments)
- [Creating a Custom Environment](#creating-a-custom-environment)
- [Running Tests and Evaluations](#running-tests-and-evaluations)
- [Development Workflow](#development-workflow)
- [Code Style Guidelines](#code-style-guidelines)
- [Troubleshooting](#troubleshooting)
- [Getting Help](#getting-help)

## Prerequisites

Before you begin, ensure you have the following installed on your system:

### Required Software

- **Python**: 3.10 or higher
- **Git**: For version control
- **Conda** or **venv**: For managing virtual environments
- **pip** or **uv**: For package management (uv is recommended for faster installs)

### Recommended Tools

- **CUDA**: If you plan to use GPU acceleration (CUDA 11.8+ recommended)
- **Docker**: For containerized environment setup (optional)
- **Java**: Required for some environments (e.g., WebShop)

### System Requirements

- **RAM**: Minimum 16GB (32GB+ recommended for large models)
- **Disk Space**: At least 50GB free space for datasets and models
- **GPU**: Optional but recommended for model training and evaluation

## Getting Started

### 1. Clone the Repository

```bash
git clone --recursive https://github.com/WooooDyy/AgentGym.git
cd AgentGym
```

> **Note**: The `--recursive` flag is important as it initializes all git submodules.

### 2. Set Up the Core Package

The `agentenv` package is the core component of AgentGym. Install it first:

#### Option A: Install from PyPI (Stable)

```bash
pip install agentenv
```

#### Option B: Install from Source (Development)

```bash
cd agentenv
pip install -e .
```

The `-e` flag installs the package in editable mode, allowing you to make changes without reinstalling.

#### Optional Dependencies

For vLLM support (faster inference):
```bash
pip install agentenv[vllm]
```

For Ascend NPU support:
```bash
VLLM_TARGET_DEVICE=npu pip install agentenv[ascend]
```

## Project Structure

```
AgentGym/
├── agentenv/                    # Core package
│   ├── controller.py           # Agent controller and evaluator
│   ├── envs/                   # Environment client implementations
│   ├── trainer/                # Training modules (BC, DPO, AgentEvol)
│   ├── examples/               # Usage examples
│   └── pyproject.toml          # Package configuration
├── agentenv-*/                 # Individual environment servers
│   ├── agentenv-webshop/      # E-commerce navigation
│   ├── agentenv-alfworld/     # Household tasks
│   ├── agentenv-babyai/       # Grid world navigation
│   ├── agentenv-tool/         # Tool usage environments
│   └── ...                     # More environments
├── docs/                       # Documentation and tutorials
│   └── tutorials/              # Step-by-step guides
├── assets/                     # Images and resources
├── README.md                   # Project overview
└── LICENSE                     # MIT License
```

## Setting Up Development Environment

### Creating a Virtual Environment

We recommend using separate conda environments for different parts of the project:

#### For Core Development

```bash
conda create -n agentgym-dev python=3.10
conda activate agentgym-dev
cd agentenv
pip install -e .
```

#### For Specific Environments

Each environment may have different dependencies. For example, to work with WebShop:

```bash
conda env create -f agentenv-webshop/environment.yml -n agentenv-webshop
conda activate agentenv-webshop
cd agentenv-webshop
bash ./setup.sh
```

> **Tip**: Use `uv` for faster package installation:
> ```bash
> curl -LsSf https://astral.sh/uv/install.sh | sh
> uv venv .uv
> uv pip install -e .
> ```

## Working with Environments

AgentGym includes 14 different environments across various domains:

| Environment | Domain | Port (Default) | Setup Complexity |
|------------|--------|----------------|------------------|
| WebShop | E-commerce | 36001 | Medium |
| WebArena | Web Navigation | 36002 | High |
| ALFWorld | Household Tasks | 36003 | Low |
| BabyAI | Grid Navigation | 36004 | Low |
| SciWorld | Science Tasks | 36005 | Medium |
| TextCraft | Crafting Game | 36006 | Low |
| Tool (Weather, Movie, etc.) | Tool Usage | 36007 | Low |
| BIRD (SQL) | Database Queries | 36008 | Medium |

### Setting Up an Environment

#### General Steps

1. **Navigate to the environment directory**:
   ```bash
   cd agentenv-<environment-name>
   ```

2. **Create and activate the environment**:
   ```bash
   conda env create -f environment.yml -n agentenv-<name>
   conda activate agentenv-<name>
   ```
   
   Or with uv:
   ```bash
   uv venv .uv
   uv pip install -e ../agentenv
   uv pip install -e .
   ```

3. **Run the setup script**:
   ```bash
   bash ./setup.sh
   ```
   This downloads datasets and installs dependencies.

4. **Launch the server**:
   ```bash
   # Using the installed console script
   <environment-name> --host 0.0.0.0 --port 36001
   
   # Or using Python module
   python -m agentenv_<name>.launch --host 0.0.0.0 --port 36001
   ```

#### Example: Setting Up WebShop

```bash
cd agentenv-webshop
conda env create -f environment.yml -n webshop
conda activate webshop
bash ./setup.sh
webshop --port 36001
```

#### Example: Setting Up SearchQA (with uv)

```bash
cd agentenv-searchqa
uv venv .uv
uv pip install -e ../agentenv
uv pip install -e .
bash ./setup.sh
uv run searchqa --host 0.0.0.0 --port 36001
```

### Verifying Environment Setup

Once a server is running, verify it's working:

```bash
curl http://localhost:36001/
```

You should see a response like: `"This is environment WebShop."`

## Creating a Custom Environment

Want to add a new environment to AgentGym? Follow our comprehensive guide!

### Overview

Creating a custom environment involves:
1. Implementing an Environment Class (Gym-compatible)
2. Creating an Environment Server Wrapper
3. Implementing an Environment Client
4. Creating a Task Class
5. Setting up HTTP API endpoints (FastAPI)

### Quick Start Template

Use `agentenv-textcraft` or `agentenv-babyai` as a template:

```bash
cp -r agentenv-textcraft agentenv-myenv
cd agentenv-myenv
# Edit files to implement your environment
```

### Detailed Steps

#### 1. Environment Class (`environment.py`)

Implement a Gym-compatible environment:

```python
import gym
from typing import Any, Tuple

class MyCustomEnv(gym.Env[str, str]):
    def __init__(self, **kwargs):
        """Initialize your environment with necessary data."""
        self.observation = None
        # Initialize your environment state
    
    def step(self, action: str) -> Tuple[str, float, bool, dict]:
        """
        Execute an action and return results.
        
        Returns:
            observation: New state description
            reward: Immediate reward (float)
            terminated: Whether episode ended (bool)
            info: Additional information (dict)
        """
        # Parse and execute action
        # Update environment state
        # Calculate reward
        
        terminated = False  # Update based on your logic
        reward = 0.0  # Calculate based on your logic
        info = {}  # Add diagnostic information
        
        return self.observation, reward, terminated, info
    
    def reset(self, idx: int = None) -> str:
        """
        Reset environment to initial state.
        
        Args:
            idx: Optional task/level index
            
        Returns:
            Initial observation
        """
        # Reset environment state
        return self.observation
    
    def close(self):
        """Clean up resources."""
        super().close()
```

#### 2. Server Wrapper (`server.py`)

Manage multiple environment instances:

```python
import threading
from typing import Dict

class MyEnv_Wrapper:
    def __init__(self):
        self._max_id = 0
        self.env: Dict[int, MyCustomEnv] = {}
        self.info: Dict[int, dict] = {}
        self.ls = []
        self._lock = threading.Lock()
    
    def create(self) -> dict:
        """Create new environment instance."""
        with self._lock:
            env_id = self._max_id
            self._max_id += 1
        
        new_env = MyCustomEnv()
        new_env.reset()
        self.ls.append(env_id)
        self.env[env_id] = new_env
        
        return {"id": env_id}
    
    def step(self, env_idx: int, action: str) -> dict:
        """Execute action in specified environment."""
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
        """Reset specified environment."""
        ob = self.env[env_idx].reset(idx=data_idx)
        self.info[env_idx] = {
            "observation": ob,
            "reward": 0.0,
            "done": False
        }
        return self.info[env_idx]
    
    def observation(self, env_idx: int) -> dict:
        """Get current observation."""
        return {"observation": self.info[env_idx]["observation"]}
    
    def close(self, env_id: int) -> dict:
        """Close environment and free resources."""
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

#### 3. HTTP API (`launch.py`)

Expose server via FastAPI:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class StepRequestBody(BaseModel):
    id: int
    action: str

class ResetRequestBody(BaseModel):
    id: int
    data_idx: int = 0

class CloseRequestBody(BaseModel):
    id: int

server = MyEnv_Wrapper()

@app.get("/")
def hello():
    return "This is environment MyCustomEnv."

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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=36001)
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)
```

#### 4. Environment Client (`agentenv/agentenv/envs/myenv.py`)

Implement client-side interface:

```python
import re
import requests
from agentenv.envs import BaseEnvClient, ConversationMessage, StepOutput

class MyEnvClient(BaseEnvClient):
    # Define initial prompt for LLM
    conversation_start = (
        ConversationMessage({
            "from": "human",
            "loss": None,
            "value": "You are an agent that needs to...",
        }),
        ConversationMessage({
            "from": "gpt",
            "loss": False,
            "value": "OK. I'll follow your instructions.",
        }),
    )
    
    def __init__(self, env_server_base: str, data_len: int, 
                 timeout: int = 300, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len
        
        # Create environment on server
        response = requests.post(
            f"{self.env_server_base}/create",
            timeout=self.timeout
        )
        assert response.status_code == 200
        self.env_id = response.json()["id"]
    
    def __len__(self):
        return self.data_len
    
    def observe(self) -> str:
        """Get current observation."""
        if self.info:
            return self.info["observation"]
        response = requests.get(
            f"{self.env_server_base}/observation?id={self.env_id}",
            timeout=self.timeout
        )
        return response.json()["observation"]
    
    def step(self, action: str) -> StepOutput:
        """Execute action."""
        # Parse action from LLM output
        action_matches = re.findall(
            r"Action:\s*(.*?)(?=\n|$)", action, re.DOTALL
        )
        if action_matches:
            action = action_matches[-1].strip()
        
        # Send to server
        response = requests.post(
            f"{self.env_server_base}/step",
            json={"id": self.env_id, "action": action},
            timeout=self.timeout
        )
        result = response.json()
        
        self.info = {
            "observation": result["observation"],
            "reward": result["reward"],
            "done": result["done"],
        }
        
        return StepOutput(
            state=result["observation"],
            reward=result["reward"],
            done=result["done"],
        )
    
    def reset(self, data_idx: int = 0) -> dict:
        """Reset environment."""
        response = requests.post(
            f"{self.env_server_base}/reset",
            json={"id": self.env_id, "data_idx": data_idx},
            timeout=self.timeout
        )
        result = response.json()
        self.info = result
        return result
    
    def close(self) -> dict:
        """Close environment."""
        response = requests.post(
            f"{self.env_server_base}/close",
            json={"id": self.env_id},
            timeout=self.timeout
        )
        return response.json()
```

#### 5. Task Class (`agentenv/agentenv/envs/myenv.py`)

```python
from agentenv.envs import BaseTask
from typing import Mapping, Any

class MyEnvTask(BaseTask):
    env_client_cls = MyEnvClient
    env_name = "MyCustomEnv"
    
    def __init__(self, client_args: Mapping[str, Any], 
                 n_clients: int = 1, *args, **kwargs):
        super().__init__(client_args, n_clients, *args, **kwargs)
```

#### 6. Package Configuration (`pyproject.toml`)

```toml
[project]
name = "agentenv-myenv"
version = "0.1.0"
dependencies = [
    "fastapi",
    "uvicorn",
    "requests",
    "gymnasium",
]

[project.scripts]
myenv = "agentenv_myenv.launch:main"
```

### Testing Your Custom Environment

1. **Launch the server**:
   ```bash
   python -m agentenv_myenv.launch --port 36001
   ```

2. **Test with curl**:
   ```bash
   # Create environment
   curl -X POST http://localhost:36001/create
   
   # Get observation
   curl "http://localhost:36001/observation?id=0"
   
   # Execute action
   curl -X POST http://localhost:36001/step \
     -H "Content-Type: application/json" \
     -d '{"id": 0, "action": "your_action"}'
   ```

3. **Run evaluation script**:
   ```python
   from agentenv.controller import Evaluator, Agent
   from agentenv.envs import MyEnvTask
   
   evaluator = Evaluator(
       agent,
       [MyEnvTask(
           client_args={
               "env_server_base": "http://127.0.0.1:36001",
               "data_len": 100,
           },
           n_clients=1,
       )]
   )
   ```

For more details, see the [2nd Development Tutorial](docs/tutorials/en/05-2nd-Development.md).

## Running Tests and Evaluations

### Basic Evaluation

1. **Start the environment server** (in one terminal):
   ```bash
   conda activate agentenv-webshop
   webshop --port 36001
   ```

2. **Run evaluation** (in another terminal):
   ```bash
   conda activate agentgym-dev
   python agentenv/examples/basic/base_eval.py
   ```

### Example Evaluation Script

See `agentenv/examples/basic/base_eval.py`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from agentenv.controller import Agent, Evaluator
from agentenv.envs import WebshopTask

MODEL_PATH = "THUDM/agentlm-7b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", trust_remote_code=True
).eval()

evaluator = Evaluator(
    Agent(model, tokenizer),
    [WebshopTask(
        client_args={
            "env_server_base": "http://127.0.0.1:36001",
            "data_len": 200,
            "timeout": 300,
        },
        n_clients=1,
    )]
)

exps = evaluator.eval(
    generation_config=GenerationConfig(
        max_length=4096,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    ),
    max_rounds=7,
    idxs=list(range(10)),  # Test on first 10 samples
)

print(f"Score: {exps.score}")
print(f"Success Rate: {exps.success}")
```

### Training

#### Behavioral Cloning

```bash
python agentenv/examples/behavioral_cloning/train_behavioral_clone.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --data_path /path/to/AgentTraj-L \
    --output_dir ./output/bc_model
```

#### AgentEvol

```bash
python agentenv/examples/agentevol/train_agentevol.py \
    --base_model meta-llama/Llama-2-7b-hf \
    --data_path /path/to/trajectory_data \
    --output_dir ./output/agentevol_model
```

See the [tutorials](docs/tutorials/en/) for more training examples.

## Development Workflow

### Making Changes

1. **Create a new branch**:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes** and test them thoroughly

3. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: Brief description of changes"
   ```

4. **Push to your fork**:
   ```bash
   git push origin feature/my-new-feature
   ```

5. **Create a Pull Request** on GitHub

### Commit Message Guidelines

Use clear, descriptive commit messages:

- `Add: New feature or functionality`
- `Fix: Bug fix`
- `Update: Improvements to existing code`
- `Docs: Documentation changes`
- `Refactor: Code restructuring without functionality changes`
- `Test: Adding or updating tests`

Example:
```
Add: Support for custom action spaces in BabyAI environment

- Implemented configurable action vocabulary
- Added validation for custom actions
- Updated documentation with examples
```

### Pull Request Process

1. Ensure your code follows the style guidelines
2. Update documentation if needed
3. Add tests for new functionality
4. Ensure all tests pass
5. Request review from maintainers
6. Address review feedback

## Code Style Guidelines

### Python Code Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use meaningful variable and function names
- Add docstrings to all classes and functions
- Keep functions focused and concise (< 50 lines when possible)

### Docstring Format

Use Google-style docstrings:

```python
def step(self, action: str) -> Tuple[str, float, bool, dict]:
    """Execute an action in the environment.
    
    Args:
        action: The action string to execute
        
    Returns:
        observation: The new observation
        reward: The immediate reward
        terminated: Whether the episode has ended
        info: Additional diagnostic information
        
    Raises:
        ValueError: If action is invalid
    """
    pass
```

### Type Hints

Use type hints for function arguments and return values:

```python
from typing import Dict, List, Optional, Tuple

def process_trajectory(
    trajectory: List[Dict[str, str]], 
    max_length: Optional[int] = None
) -> Tuple[str, float]:
    """Process a trajectory and return summary."""
    pass
```

### Import Organization

Organize imports in this order:
1. Standard library imports
2. Third-party imports
3. Local imports

```python
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModel

from agentenv.envs import BaseEnvClient
from agentenv.controller import Evaluator
```

## Troubleshooting

### Common Issues

#### Issue: Environment Server Won't Start

**Symptoms**: Server fails to start or crashes immediately

**Solutions**:
- Check if the port is already in use: `lsof -i :36001`
- Verify all dependencies are installed: `pip list | grep fastapi`
- Check Python version: `python --version` (should be 3.10+)
- Review setup script output for errors

#### Issue: Client Can't Connect to Server

**Symptoms**: Connection timeout or refused errors

**Solutions**:
- Verify server is running: `curl http://localhost:36001/`
- Check firewall settings
- Ensure correct host/port in client configuration
- Try using `127.0.0.1` instead of `localhost`

#### Issue: CUDA Out of Memory

**Symptoms**: RuntimeError: CUDA out of memory

**Solutions**:
- Reduce batch size
- Use model quantization: `load_in_8bit=True` or `load_in_4bit=True`
- Enable gradient checkpointing
- Use a smaller model
- Clear GPU cache: `torch.cuda.empty_cache()`

#### Issue: Import Errors

**Symptoms**: `ModuleNotFoundError` when importing agentenv

**Solutions**:
- Verify installation: `pip show agentenv`
- Reinstall in editable mode: `pip install -e .`
- Check Python path: `echo $PYTHONPATH`
- Activate the correct conda environment

#### Issue: Dataset Download Fails

**Symptoms**: Setup script fails during dataset download

**Solutions**:
- Check internet connection
- Verify Google Drive access (some datasets use GDrive)
- Try manual download and place in expected location
- Check available disk space

#### Issue: WebShop Java Error

**Symptoms**: Java-related errors when starting WebShop

**Solutions**:
- Install Java 11+: `sudo apt-get install openjdk-11-jdk`
- Set JAVA_HOME: `export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64`
- Verify Java version: `java -version`

### Getting Debug Information

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

For server debugging, check FastAPI logs:

```bash
python -m agentenv_webshop.launch --port 36001 --log-level debug
```

### Performance Issues

If you experience slow performance:

1. **Use vLLM for inference**:
   ```bash
   pip install agentenv[vllm]
   ```

2. **Enable model quantization**:
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       MODEL_PATH,
       load_in_8bit=True,
       device_map="auto"
   )
   ```

3. **Use multiple workers**:
   ```python
   evaluator = Evaluator(agent, tasks, n_workers=4)
   ```

4. **Batch processing**:
   Set appropriate batch sizes in evaluation configs

## Getting Help

### Resources

- **Documentation**: [GitHub Wiki](https://github.com/WooooDyy/AgentGym)
- **Tutorials**: [docs/tutorials/](docs/tutorials/en/)
- **Examples**: [agentenv/examples/](agentenv/examples/)
- **Paper**: [AgentGym on arXiv](https://arxiv.org/abs/2406.04151)
- **Website**: [Project Page](https://agentgym.github.io/)

### Community

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: zhxi22@m.fudan.edu.cn for direct contact

### Reporting Bugs

When reporting bugs, please include:

1. **Environment information**:
   ```bash
   python --version
   pip list | grep agentenv
   uname -a
   ```

2. **Steps to reproduce**: Minimal example that triggers the bug

3. **Expected behavior**: What should happen

4. **Actual behavior**: What actually happens

5. **Error messages**: Full stack trace if applicable

6. **Additional context**: Screenshots, logs, or config files

### Feature Requests

We welcome feature requests! Please:

1. Check if the feature already exists or is requested
2. Describe the use case and motivation
3. Provide examples of how it would be used
4. Consider contributing the implementation yourself!

## Contributing Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guidelines
- [ ] All tests pass locally
- [ ] New features have tests
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] No unnecessary files are included (check `.gitignore`)
- [ ] Changes are minimal and focused
- [ ] PR description explains what and why

## License

By contributing to AgentGym, you agree that your contributions will be licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

Thank you for contributing to AgentGym! Your efforts help build better tools for the AI agent community. 🚀

---

**Questions?** Don't hesitate to reach out through GitHub Issues or Discussions. We're here to help!
