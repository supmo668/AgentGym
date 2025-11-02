# Quick Start Guide for New Contributors

Welcome to AgentGym! This quick start guide will help you get up and running with contributing to the project. For detailed information, see [CONTRIB.md](./CONTRIB.md).

## 🚀 5-Minute Setup

### Prerequisites
- Python >= 3.10
- Git

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

## 🎯 Try the MCP Environment

The MCP (Model Context Protocol) environment is a great example to get started:

### Start the Server

```bash
# Terminal 1: Start MCP server
mcp --host 127.0.0.1 --port 8000
```

### Run Example

```bash
# Terminal 2: Run example
cd agentenv-mcp
python example_usage.py
```

### Use in Code

```python
from agentenv.envs.mcp import MCPEnvClient

# Create client
client = MCPEnvClient(
    env_server_base="http://127.0.0.1:8000",
    data_len=10
)

# Reset to start
response = client.reset(data_idx=0)

# Execute action
result = client.step("Action: list_collections with Action Input: {}")
print(result.state)

# Clean up
client.close()
```

## 📚 Key Concepts

### Environment Architecture

```
Agent/LLM → Client (HTTP) → Server (FastAPI) → Environment (Gym)
```

1. **Environment**: Core logic (inherits `gym.Env`)
2. **Server**: FastAPI wrapper exposing HTTP endpoints
3. **Client**: Agent-side interface for HTTP communication
4. **Task**: Binds client to task configuration

### Required Files for New Environment

```
agentenv-yourenv/
├── agentenv_yourenv/
│   ├── __init__.py
│   ├── environment.py    # Core environment logic
│   ├── server.py         # FastAPI server
│   ├── launch.py         # Launch script
│   └── model.py          # Pydantic models
├── pyproject.toml
└── README.md
```

Plus client in `agentenv/agentenv/envs/yourenv.py`

## 🛠️ Common Tasks

### Add New Environment

1. Copy template from `agentenv-mcp/` or `agentenv-textcraft/`
2. Implement your environment class (inherit `gym.Env`)
3. Create server wrapper and FastAPI endpoints
4. Implement client in `agentenv/agentenv/envs/`
5. Test locally
6. Submit PR

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests for specific environment
pytest agentenv-mcp/tests/

# Run with coverage
pytest --cov=agentenv_mcp
```

### Code Style

```bash
# Format code
black agentenv-yourenv/

# Check style
flake8 agentenv-yourenv/
```

## 📖 Learn by Example

### Example 1: MCP Environment
- **Location**: `agentenv-mcp/`
- **Features**: Tool-based environment with simulated Milvus collections
- **Best for**: Understanding tool integration and resource management

### Example 2: Weather Environment
- **Location**: `agentenv-tool/agentenv_weather/`
- **Features**: Real API integration
- **Best for**: Learning external service integration

### Example 3: TextCraft Environment
- **Location**: `agentenv-textcraft/`
- **Features**: Game-like environment with state management
- **Best for**: Understanding complex state handling

## 🤝 Contributing Workflow

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/my-feature`
3. **Make** your changes
4. **Test** thoroughly
5. **Commit** with clear messages: `git commit -m "feat: add new feature"`
6. **Push** to your fork: `git push origin feature/my-feature`
7. **Create** a Pull Request

### Commit Message Format

```
<type>: <description>

[optional body]
[optional footer]
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`

## 🔍 Resources

- **Full Guide**: [CONTRIB.md](./CONTRIB.md)
- **API Docs**: Start server and visit `/docs` endpoint
- **Paper**: https://arxiv.org/abs/2406.04151
- **Project Page**: https://agentgym.github.io/
- **Tutorials**: [docs/tutorials/](./docs/tutorials/)

## ❓ Getting Help

- **GitHub Issues**: https://github.com/WooooDyy/AgentGym/issues
- **Email**: zhxi22@m.fudan.edu.cn
- **Documentation**: Check tutorials and existing environments

## ✅ Quick Checklist

Before submitting a PR:

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Added tests for new features
- [ ] Updated documentation
- [ ] Commit messages are clear
- [ ] No sensitive data or credentials in code

## 🎉 What to Contribute

We welcome contributions in many areas:

- **New Environments**: Add new task environments
- **Bug Fixes**: Fix issues in existing code
- **Documentation**: Improve guides and examples
- **Tests**: Add more test coverage
- **Examples**: Create tutorials and demos
- **Performance**: Optimize existing code
- **Features**: Enhance platform capabilities

## 📊 Project Statistics

- **14+ Environments**: Diverse task types
- **10K+ Trajectories**: Training data available
- **Multiple Modalities**: Text, vision, embodied tasks
- **Active Community**: Regular updates and improvements

## 🌟 Success Story: MCP Environment

The MCP environment demonstrates a complete implementation:

- ✅ **8 Tools**: All integrated as environment actions
- ✅ **3 Collections**: Simulated vector database
- ✅ **Resources**: Schemas, prompts, formatters
- ✅ **Full Testing**: Unit, integration, and example tests
- ✅ **Documentation**: Comprehensive README and examples
- ✅ **Working Code**: Tested and verified

Study `agentenv-mcp/` to learn best practices!

---

**Ready to contribute?** Jump in and start exploring! 🚀

For questions or support, don't hesitate to reach out to the community.
