"""
LangChain Tool Adapters for AgentEnv.

This module provides tools that wrap AgentEnv actions for use with
LangChain's create_react_agent (v1 syntax).
"""

from typing import Any, Dict, List, Optional, Type, Union, TYPE_CHECKING

from pydantic import BaseModel, Field, create_model

if TYPE_CHECKING:
    from agentenv_langchain.wrapper import AgentEnvToolWrapper, EnvAction


class StepInput(BaseModel):
    """Input schema for the step tool."""
    action: str = Field(description="The action to execute in the environment")


class ResetInput(BaseModel):
    """Input schema for the reset tool."""
    data_idx: int = Field(default=0, description="The task index to reset to")


class ObserveInput(BaseModel):
    """Input schema for the observe tool (empty - no inputs needed)."""
    pass


def _create_action_input_model(action: "EnvAction") -> Type[BaseModel]:
    """Dynamically create a Pydantic model for an action's parameters."""
    if not action.parameters:
        return create_model(f"{action.name.title()}Input")
    
    fields = {}
    for param_name, param_type in action.parameters.items():
        python_type = str
        if param_type in ("integer", "int"):
            python_type = int
        elif param_type in ("number", "float"):
            python_type = float
        elif param_type in ("boolean", "bool"):
            python_type = bool
        
        is_required = param_name in action.required_params
        default = ... if is_required else None
        
        fields[param_name] = (
            Optional[python_type] if not is_required else python_type,
            Field(default=default, description=f"Parameter: {param_name}")
        )
    
    return create_model(f"{action.name.title()}Input", **fields)


class AgentEnvStepTool:
    """
    LangChain-compatible tool for executing actions in an AgentEnv.
    
    This follows LangChain v1 tool syntax for use with create_react_agent.
    """
    
    def __init__(
        self,
        wrapper: "AgentEnvToolWrapper",
        action_name: Optional[str] = None,
        action: Optional["EnvAction"] = None,
    ):
        """
        Initialize the step tool.
        
        Args:
            wrapper: The AgentEnvToolWrapper instance.
            action_name: Optional specific action name (for action-specific tools).
            action: Optional EnvAction object with full action details.
        """
        self.wrapper = wrapper
        self.action_name = action_name
        self.action = action
        
        if action_name and not action:
            self.action = wrapper.get_action_by_name(action_name)
        
        # Tool metadata for LangChain
        if self.action:
            self.name = f"env_{self.action.name}"
            self.description = self.action.description
            self.args_schema = _create_action_input_model(self.action)
        else:
            self.name = "env_step"
            self.description = f"Execute an action in the {wrapper.env_name} environment. Available actions: {', '.join(a.name for a in wrapper.actions)}"
            self.args_schema = StepInput
    
    def _run(self, **kwargs) -> str:
        """Execute the action (sync)."""
        if self.action_name:
            # Action-specific tool
            action_str = self.action_name
            if kwargs:
                # Format parameters into action string
                params = " ".join(f"{v}" for v in kwargs.values() if v is not None)
                if params:
                    action_str = f"{action_str} {params}"
        else:
            # General step tool
            action_str = kwargs.get("action", "")
        
        try:
            observation = self.wrapper.step(action_str)
            result = f"Observation: {observation.state}\nReward: {observation.reward}\nDone: {observation.done}"
            if observation.done:
                result += f"\n\nEpisode complete! Total reward: {self.wrapper.total_reward}"
            return result
        except Exception as e:
            return f"Error executing action: {str(e)}"
    
    async def _arun(self, **kwargs) -> str:
        """Execute the action (async) - delegates to sync for now."""
        return self._run(**kwargs)
    
    def __call__(self, **kwargs) -> str:
        """Make the tool callable."""
        return self._run(**kwargs)


class AgentEnvObserveTool:
    """
    LangChain-compatible tool for observing the current state.
    """
    
    def __init__(self, wrapper: "AgentEnvToolWrapper"):
        self.wrapper = wrapper
        self.name = "env_observe"
        self.description = f"Get the current observation from the {wrapper.env_name} environment."
        self.args_schema = ObserveInput
    
    def _run(self) -> str:
        """Get current observation (sync)."""
        try:
            observation = self.wrapper.observe()
            return f"Current State: {observation.state}\nAvailable Actions: {', '.join(observation.available_actions)}"
        except Exception as e:
            return f"Error getting observation: {str(e)}"
    
    async def _arun(self) -> str:
        """Get current observation (async)."""
        return self._run()
    
    def __call__(self) -> str:
        """Make the tool callable."""
        return self._run()


class AgentEnvResetTool:
    """
    LangChain-compatible tool for resetting the environment.
    """
    
    def __init__(self, wrapper: "AgentEnvToolWrapper"):
        self.wrapper = wrapper
        self.name = "env_reset"
        self.description = f"Reset the {wrapper.env_name} environment to start a new episode."
        self.args_schema = ResetInput
    
    def _run(self, data_idx: int = 0) -> str:
        """Reset environment (sync)."""
        try:
            observation = self.wrapper.reset(data_idx)
            return f"Environment reset to task {data_idx}.\nInitial State: {observation.state}"
        except Exception as e:
            return f"Error resetting environment: {str(e)}"
    
    async def _arun(self, data_idx: int = 0) -> str:
        """Reset environment (async)."""
        return self._run(data_idx)
    
    def __call__(self, data_idx: int = 0) -> str:
        """Make the tool callable."""
        return self._run(data_idx)


def create_agentenv_tools(
    wrapper: "AgentEnvToolWrapper",
    include_step: bool = True,
    include_observe: bool = True,
    include_reset: bool = True,
    individual_action_tools: bool = True,
) -> List[Any]:
    """
    Create LangChain tools from an AgentEnvToolWrapper.
    
    This function creates tools compatible with LangChain's create_react_agent
    using v1 syntax (Tool class with name, description, func).
    
    Args:
        wrapper: The AgentEnvToolWrapper instance.
        include_step: Include a general step tool.
        include_observe: Include an observe tool.
        include_reset: Include a reset tool.
        individual_action_tools: Create individual tools for each action.
    
    Returns:
        List of LangChain-compatible tools.
    
    Example:
        >>> from langchain.agents import create_react_agent, AgentExecutor
        >>> from langchain_openai import ChatOpenAI
        >>> 
        >>> wrapper = AgentEnvToolWrapper(...)
        >>> tools = create_agentenv_tools(wrapper)
        >>> 
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> agent = create_react_agent(llm, tools, prompt)
        >>> executor = AgentExecutor(agent=agent, tools=tools)
    """
    try:
        from langchain.tools import Tool, StructuredTool
    except ImportError:
        raise ImportError(
            "langchain is required for tool creation. "
            "Install it with: pip install langchain"
        )
    
    tools = []
    
    if include_observe:
        observe_tool = AgentEnvObserveTool(wrapper)
        tools.append(Tool(
            name=observe_tool.name,
            description=observe_tool.description,
            func=observe_tool._run,
        ))
    
    if include_reset:
        reset_tool = AgentEnvResetTool(wrapper)
        tools.append(StructuredTool(
            name=reset_tool.name,
            description=reset_tool.description,
            func=reset_tool._run,
            args_schema=reset_tool.args_schema,
        ))
    
    if individual_action_tools and wrapper.actions:
        # Create individual tools for each action
        for action in wrapper.actions:
            action_tool = AgentEnvStepTool(wrapper, action_name=action.name, action=action)
            
            if action.parameters:
                tools.append(StructuredTool(
                    name=action_tool.name,
                    description=action_tool.description,
                    func=action_tool._run,
                    args_schema=action_tool.args_schema,
                ))
            else:
                tools.append(Tool(
                    name=action_tool.name,
                    description=action_tool.description,
                    func=lambda t=action_tool: t._run(),
                ))
    elif include_step:
        # Create a general step tool
        step_tool = AgentEnvStepTool(wrapper)
        tools.append(StructuredTool(
            name=step_tool.name,
            description=step_tool.description,
            func=step_tool._run,
            args_schema=step_tool.args_schema,
        ))
    
    return tools


def create_react_agent_for_env(
    wrapper: "AgentEnvToolWrapper",
    llm: Any,
    prompt: Optional[Any] = None,
    **executor_kwargs,
):
    """
    Convenience function to create a complete ReAct agent for an AgentEnv.
    
    Args:
        wrapper: The AgentEnvToolWrapper instance.
        llm: A LangChain LLM instance (e.g., ChatOpenAI).
        prompt: Optional custom prompt. If None, uses default ReAct prompt.
        **executor_kwargs: Additional kwargs for AgentExecutor.
    
    Returns:
        AgentExecutor ready to run.
    
    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> 
        >>> wrapper = create_wrapper_for_env("BabyAI", "http://localhost:8000")
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> executor = create_react_agent_for_env(wrapper, llm)
        >>> 
        >>> result = executor.invoke({"input": "Complete the task"})
    """
    try:
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain import hub
    except ImportError:
        raise ImportError(
            "langchain is required. Install with: pip install langchain langchain-hub"
        )
    
    tools = create_agentenv_tools(wrapper)
    
    if prompt is None:
        # Use the standard ReAct prompt
        prompt = hub.pull("hwchase17/react")
    
    agent = create_react_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=executor_kwargs.pop("verbose", True),
        handle_parsing_errors=executor_kwargs.pop("handle_parsing_errors", True),
        max_iterations=executor_kwargs.pop("max_iterations", 50),
        **executor_kwargs,
    )
