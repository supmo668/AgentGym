"""
MCPToAgentEnv: Wrapper to adapt any MCP server into an AgentGym-compatible BaseEnvClient.

This allows AgentGym agents to train and evaluate against external MCP services
using the standard AgentEnv interface.
"""

import asyncio
import json
import re
from abc import ABCMeta
from typing import Any, Mapping, Optional, Sequence, TYPE_CHECKING

from .schema_utils import mcp_tool_to_function_desc, generate_function_descriptions_from_mcp_tools

# Lazy MCP imports to allow module import without MCP installed
if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters
    from mcp.types import CallToolResult, TextContent


def _get_mcp_types():
    """Lazily import MCP types."""
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters
    from mcp.types import CallToolResult, TextContent
    return {
        "ClientSession": ClientSession,
        "stdio_client": stdio_client,
        "StdioServerParameters": StdioServerParameters,
        "CallToolResult": CallToolResult,
        "TextContent": TextContent,
    }


# Import AgentEnv types - these are imported at runtime to avoid hard dependency
def _get_agentenv_types():
    """Lazily import AgentEnv types."""
    from agentenv.controller import BaseEnvClient, BaseTask
    from agentenv.controller.types import (
        ActionFormat,
        ActionWithTought,
        ConversationMessage,
        StepOutput,
    )
    from agentenv.controller.utils import (
        BaseAdapter,
        format_function_call_prompt,
        format_code_as_action_prompt,
    )
    return {
        "BaseEnvClient": BaseEnvClient,
        "BaseTask": BaseTask,
        "ActionFormat": ActionFormat,
        "ActionWithTought": ActionWithTought,
        "ConversationMessage": ConversationMessage,
        "StepOutput": StepOutput,
        "BaseAdapter": BaseAdapter,
        "format_function_call_prompt": format_function_call_prompt,
        "format_code_as_action_prompt": format_code_as_action_prompt,
    }


class MCPAdapter:
    """
    Adapter for parsing actions in various formats for MCP-based environments.
    
    This adapter handles conversion between AgentEnv action formats and MCP tool calls.
    """
    
    INVOKING_FUNCTION_PROMPT = """

If you want to invoke a provided function or tool, please reply in the following *JSON* format:
```json
{
    "thought": "I think ...",
    "function_name": "function_name",
    "arguments": <valid json object of args>
}
```
Only reply the *JSON* object, no other text should be present.
"""

    def __init__(self, function_descriptions: list[dict[str, Any]]):
        """
        Initialize the adapter with function descriptions.
        
        Args:
            function_descriptions: List of function descriptions in AgentEnv format
        """
        self.function_descriptions = function_descriptions
        self._build_conversation_starts()
    
    def _build_conversation_starts(self):
        """Build conversation start prompts for each action format."""
        types = _get_agentenv_types()
        ConversationMessage = types["ConversationMessage"]
        ActionFormat = types["ActionFormat"]
        format_function_call_prompt = types["format_function_call_prompt"]
        format_code_as_action_prompt = types["format_code_as_action_prompt"]
        
        base_instruction = (
            "You are an agent interacting with an environment through tools.\n"
            "Each turn you will receive an observation and must respond with an action.\n"
        )
        
        self.conversation_start_dict = {
            ActionFormat.REACT: (
                ConversationMessage({
                    "from": "human",
                    "loss": None,
                    "value": (
                        f"{base_instruction}"
                        "Your response should use the following format:\n\n"
                        "Thought:\nI think ... \n\nAction:\naction_name arg1 arg2"
                    ),
                }),
                ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
            ),
            ActionFormat.FUNCTION_CALLING: (
                ConversationMessage({
                    "from": "human",
                    "loss": None,
                    "value": (
                        f"{base_instruction}"
                        f"{format_function_call_prompt(self.function_descriptions)}"
                    ),
                }),
                ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
            ),
            ActionFormat.CODE_AS_ACTION: (
                ConversationMessage({
                    "from": "human",
                    "loss": None,
                    "value": (
                        f"{base_instruction}"
                        f"{format_code_as_action_prompt(self.function_descriptions)}"
                    ),
                }),
                ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
            ),
        }
    
    @staticmethod
    def parse_react(text: str):
        """Parse ReAct format action."""
        types = _get_agentenv_types()
        ActionWithTought = types["ActionWithTought"]
        
        _split = text.rsplit("Action:", 1)
        if len(_split) == 2:
            _thought, _action = _split
            thought = _thought.split("Thought:")[-1].strip()
            action = _action.strip()
        else:
            thought = ""
            action = text.strip()
        
        return ActionWithTought(thought, action)
    
    @staticmethod
    def parse_function_calling(text: str):
        """Parse function calling format action."""
        types = _get_agentenv_types()
        ActionWithTought = types["ActionWithTought"]
        
        # Try to extract JSON from the text
        try:
            _fn_call = json.loads(
                "{" + text.split("{", 1)[-1].rsplit("}", 1)[0] + "}", strict=False
            )
            thought = _fn_call.get("thought", "")
            fn_name = _fn_call.get("function_name", "")
            args = _fn_call.get("arguments", {})
            
            # Return as structured action that can be converted to MCP tool call
            action = json.dumps({"function_name": fn_name, "arguments": args})
            return ActionWithTought(thought, action)
        except json.JSONDecodeError:
            return ActionWithTought("", text)
    
    def action_parser(self, action: str, action_format) -> dict[str, Any]:
        """
        Parse action text and return MCP tool call parameters.
        
        Returns:
            dict with "tool_name" and "arguments" keys
        """
        types = _get_agentenv_types()
        ActionFormat = types["ActionFormat"]
        
        if action_format == ActionFormat.REACT:
            parsed = self.parse_react(action)
            # For React format, try to parse the action as "tool_name arg1 arg2"
            return self._react_to_tool_call(parsed.action)
        elif action_format == ActionFormat.FUNCTION_CALLING:
            parsed = self.parse_function_calling(action)
            try:
                call_data = json.loads(parsed.action)
                return {
                    "tool_name": call_data.get("function_name", ""),
                    "arguments": call_data.get("arguments", {}),
                }
            except json.JSONDecodeError:
                return {"tool_name": "", "arguments": {}}
        else:
            # CODE_AS_ACTION - extract and parse
            return {"tool_name": "", "arguments": {"code": action}}
    
    def _react_to_tool_call(self, action: str) -> dict[str, Any]:
        """Convert a React-style action string to tool call parameters."""
        # Try to match against known function names
        for func_desc in self.function_descriptions:
            fn_name = func_desc["name"]
            if action.lower().startswith(fn_name.lower()):
                args_str = action[len(fn_name):].strip()
                # Parse simple space-separated arguments
                params = func_desc.get("parameters", {}).get("properties", {})
                param_names = list(params.keys())
                arg_values = args_str.split() if args_str else []
                
                arguments = {}
                for i, param_name in enumerate(param_names):
                    if i < len(arg_values):
                        arguments[param_name] = arg_values[i]
                
                return {"tool_name": fn_name, "arguments": arguments}
        
        # If no match, return the raw action
        return {"tool_name": action.split()[0] if action else "", "arguments": {}}


class MCPEnvClient:
    """
    AgentEnv-compatible client that wraps an MCP server.
    
    This allows any MCP server to be used as an AgentGym environment.
    The MCP server must provide:
    - env_reset tool (or similar reset mechanism)
    - env_step tool (or action tools)
    - env_observe tool (or observation mechanism)
    
    Example:
        >>> client = MCPEnvClient(
        ...     server_command=["python", "-m", "my_mcp_server"],
        ...     action_format="function_calling",
        ... )
        >>> client.reset(0)
        >>> obs = client.observe()
        >>> result = client.step("some action")
    """
    
    def __init__(
        self,
        server_command: list[str],
        server_args: list[str] | None = None,
        server_env: dict[str, str] | None = None,
        action_format: str = "function_calling",
        data_len: int = 1,
        timeout: float = 30.0,
        reset_tool: str = "env_reset",
        step_tool: str = "env_step",
        observe_tool: str = "env_observe",
    ):
        """
        Initialize the MCP environment client.
        
        Args:
            server_command: Command to start the MCP server
            server_args: Additional arguments for the server
            server_env: Environment variables for the server
            action_format: Action format to use
            data_len: Number of tasks available in the environment
            timeout: Timeout for MCP operations
            reset_tool: Name of the reset tool on the MCP server
            step_tool: Name of the step tool on the MCP server  
            observe_tool: Name of the observe tool on the MCP server
        """
        mcp_types = _get_mcp_types()
        StdioServerParameters = mcp_types["StdioServerParameters"]
        
        types = _get_agentenv_types()
        self.ActionFormat = types["ActionFormat"]
        self.StepOutput = types["StepOutput"]
        
        self.server_params = StdioServerParameters(
            command=server_command[0],
            args=server_command[1:] + (server_args or []),
            env=server_env,
        )
        self.action_format = self.ActionFormat(action_format)
        self.data_len = data_len
        self.timeout = timeout
        
        self.reset_tool = reset_tool
        self.step_tool = step_tool
        self.observe_tool = observe_tool
        
        # State
        self._session = None
        self._tools: list[dict[str, Any]] = []
        self._function_descriptions: list[dict[str, Any]] = []
        self._adapter: Optional[MCPAdapter] = None
        self._current_observation: str = ""
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Initialize connection and discover tools
        self._initialize()
    
    def _initialize(self):
        """Initialize the MCP connection and discover tools."""
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._async_initialize())
    
    async def _async_initialize(self):
        """Async initialization."""
        mcp_types = _get_mcp_types()
        stdio_client = mcp_types["stdio_client"]
        ClientSession = mcp_types["ClientSession"]
        
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Discover tools
                tools_result = await session.list_tools()
                self._tools = [
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "inputSchema": tool.inputSchema or {},
                    }
                    for tool in tools_result.tools
                ]
                
                # Convert to function descriptions (excluding management tools)
                management_tools = {self.reset_tool, self.step_tool, self.observe_tool, "env_info"}
                action_tools = [t for t in self._tools if t["name"] not in management_tools]
                
                # Strip "action_" prefix if present
                for tool in action_tools:
                    if tool["name"].startswith("action_"):
                        tool["name"] = tool["name"][7:]
                
                self._function_descriptions = generate_function_descriptions_from_mcp_tools(
                    action_tools
                )
                
                # Create adapter
                self._adapter = MCPAdapter(self._function_descriptions)
    
    @property
    def conversation_start(self):
        """Get conversation start messages for the current action format."""
        if self._adapter is None:
            raise RuntimeError("Client not initialized")
        return self._adapter.conversation_start_dict[self.action_format]
    
    @property
    def adapter_cls(self):
        """Return the adapter for compatibility with existing code."""
        return MCPAdapter
    
    def __len__(self) -> int:
        """Return the number of tasks available."""
        return self.data_len
    
    def _run_async(self, coro):
        """Run an async coroutine synchronously."""
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)
    
    async def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call an MCP tool and return the result text."""
        mcp_types = _get_mcp_types()
        stdio_client = mcp_types["stdio_client"]
        ClientSession = mcp_types["ClientSession"]
        
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                result = await session.call_tool(tool_name, arguments)
                
                # Extract text from result
                if result.content:
                    texts = []
                    for content in result.content:
                        if hasattr(content, "text"):
                            texts.append(content.text)
                    return "\n".join(texts)
                return ""
    
    def observe(self) -> str:
        """Get the current observation."""
        return self._current_observation
    
    def step(self, action: str) -> "StepOutput":
        """
        Execute an action in the environment.
        
        Args:
            action: The action string in the configured action format
            
        Returns:
            StepOutput with state, reward, and done flag
        """
        # Clean up action string
        if action.endswith("</s>"):
            action = action[:-4]
        
        try:
            # Parse the action based on format
            tool_call = self._adapter.action_parser(action, self.action_format)
            tool_name = tool_call["tool_name"]
            arguments = tool_call["arguments"]
            
            # Check if this is a known action tool
            action_tool_name = f"action_{tool_name}"
            available_tool_names = [t["name"] for t in self._tools]
            
            if action_tool_name in available_tool_names:
                # Call the specific action tool
                result_text = self._run_async(
                    self._call_tool(action_tool_name, arguments)
                )
            elif self.step_tool in available_tool_names:
                # Fall back to generic step tool
                result_text = self._run_async(
                    self._call_tool(self.step_tool, {"action": action})
                )
            else:
                return self.StepOutput(
                    state=f"Error: No suitable tool found for action: {action}",
                    reward=0.0,
                    done=False,
                )
            
            # Parse the result
            try:
                result_data = json.loads(result_text)
                self._current_observation = result_data.get("observation", result_text)
                reward = float(result_data.get("reward", 0.0))
                done = bool(result_data.get("done", False))
            except json.JSONDecodeError:
                self._current_observation = result_text
                reward = 0.0
                done = False
            
            return self.StepOutput(
                state=self._current_observation,
                reward=reward,
                done=done,
            )
            
        except Exception as e:
            return self.StepOutput(
                state=f"Error executing action: {str(e)}\n\n{self._current_observation}",
                reward=0.0,
                done=False,
            )
    
    def reset(self, idx: int = 0) -> dict[str, Any]:
        """
        Reset the environment to a specific task.
        
        Args:
            idx: Task index to reset to
            
        Returns:
            Reset information dictionary
        """
        result_text = self._run_async(
            self._call_tool(self.reset_tool, {"task_idx": idx})
        )
        
        try:
            result_data = json.loads(result_text)
            self._current_observation = result_data.get("observation", result_text)
            return result_data
        except json.JSONDecodeError:
            self._current_observation = result_text
            return {"observation": result_text}
    
    def close(self):
        """Close the MCP connection."""
        if self._loop is not None:
            self._loop.close()
            self._loop = None


class MCPTask:
    """
    Task wrapper for MCP-based environments.
    
    This provides compatibility with AgentGym's BaseTask interface for
    experience generation and evaluation.
    """
    
    env_client_cls = MCPEnvClient
    env_name = "mcp"
    
    def __init__(
        self,
        client_args: Mapping[str, Any],
        n_clients: int = 1,
    ):
        """
        Initialize the MCP task.
        
        Args:
            client_args: Arguments to pass to MCPEnvClient
            n_clients: Number of parallel clients (for batch generation)
        """
        self.clients = [self.env_client_cls(**client_args) for _ in range(n_clients)]
        self.len = len(self.clients[0])
    
    def generate_experience(
        self,
        agent,
        idxs: Sequence[int],
        generation_config=None,
        max_rounds: Optional[int] = None,
    ):
        """
        Generate experience by running the agent through the environment.
        
        This method follows the same pattern as BaseTask._generate_experience.
        """
        # Import here to avoid circular dependency
        types = _get_agentenv_types()
        BaseTask = types["BaseTask"]
        
        # Use the standard experience generation from BaseTask
        # This works because MCPEnvClient implements the BaseEnvClient interface
        experiences = []
        for idx in idxs:
            exp = self._generate_experience_one(
                agent,
                self.clients[0],
                idx,
                generation_config,
                max_rounds,
            )
            experiences.append(exp)
        return experiences
    
    def _generate_experience_one(
        self,
        agent,
        client: MCPEnvClient,
        idx: int,
        generation_config=None,
        max_rounds: Optional[int] = None,
    ):
        """Generate experience for a single task."""
        # This follows the same pattern as BaseTask._generate_experience_one
        # Importing the actual implementation to reuse it
        types = _get_agentenv_types()
        ConversationMessage = types["ConversationMessage"]
        StepOutput = types["StepOutput"]
        
        from agentenv.controller.types import ExperienceOutput, APIExperienceOutput
        from agentenv.controller.agent import Agent, APIAgent
        
        client.reset(idx)
        reward = 0.0
        done = False
        state = client.observe()
        
        if isinstance(agent, Agent):
            tokenizer = agent.tokenizer
            conversation = list(client.conversation_start)
            conversation.append(
                ConversationMessage({"from": "human", "loss": None, "value": state})
            )
            conversation_tokenized = agent.chat_template.tokenize_conversation(
                conversation, tokenizer, add_generation_prompt=True
            )
        elif isinstance(agent, APIAgent):
            from agentenv.controller.types import APIConversationMessage
            conversation = [
                APIConversationMessage({"role": "user", "content": client.conversation_start[0]["value"], "reasoning_content": None}),
                APIConversationMessage({"role": "assistant", "content": client.conversation_start[1]["value"], "reasoning_content": None}),
                APIConversationMessage({"role": "user", "content": state, "reasoning_content": None})
            ]
        else:
            raise NotImplementedError
        
        rounds = 0
        
        while not done:
            if isinstance(agent, Agent):
                input_length = len(conversation_tokenized["input_ids"])
                if input_length >= (generation_config.max_length if generation_config else 4096):
                    break
                try:
                    generated_tokens = agent.generate(
                        [conversation_tokenized["input_ids"]], generation_config
                    )[0]
                except Exception as e:
                    print(e)
                    break
                
                if generated_tokens[-1] != tokenizer.eos_token_id:
                    generated_tokens += [tokenizer.eos_token_id]
                
                generated_text = tokenizer.decode(generated_tokens)
                conversation_tokenized["text"] += f" {generated_text}"
                conversation_tokenized["input_ids"] += generated_tokens
                conversation_tokenized["action_mask"] += [1] * len(generated_tokens)
                
                generated_text = generated_text[:-len(tokenizer.eos_token)]
                conversation.append(
                    ConversationMessage({"from": "gpt", "loss": True, "value": generated_text})
                )
            elif isinstance(agent, APIAgent):
                generated_text, generated_reasoning_text = agent.generate(conversation)
                from agentenv.controller.types import APIConversationMessage
                conversation.append(
                    APIConversationMessage({"role": "assistant", "content": generated_text, "reasoning_content": generated_reasoning_text})
                )
            
            step_output = client.step(generated_text)
            state, reward, done = step_output.state, step_output.reward, step_output.done
            
            if isinstance(agent, Agent):
                env_message = ConversationMessage({"from": "human", "loss": None, "value": state})
                env_message_tokenized = agent.chat_template.tokenize_conversation_one(
                    env_message, tokenizer, add_generation_prompt=True
                )
                conversation.append(env_message)
                conversation_tokenized["text"] += env_message_tokenized["text"]
                conversation_tokenized["input_ids"] += env_message_tokenized["input_ids"]
                conversation_tokenized["action_mask"] += env_message_tokenized["action_mask"]
            elif isinstance(agent, APIAgent):
                from agentenv.controller.types import APIConversationMessage
                conversation.append(
                    APIConversationMessage({"role": "user", "content": state, "reasoning_content": None})
                )
            
            rounds += 1
            if max_rounds is not None and rounds >= max_rounds:
                break
        
        if isinstance(agent, Agent):
            return ExperienceOutput(
                conversation=conversation,
                reward=reward,
                text=conversation_tokenized["text"],
                seq_ids=conversation_tokenized["input_ids"],
                attention_mask=[1] * len(conversation_tokenized["input_ids"]),
                action_mask=conversation_tokenized["action_mask"],
            )
        elif isinstance(agent, APIAgent):
            return APIExperienceOutput(
                conversation=conversation,
                reward=reward,
            )
