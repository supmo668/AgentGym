"""LangChain agent creation for AgentEnv environments.

This module provides utilities to create LangChain agents that can
interact with AgentGym environments through the tool wrapper.

Note: Uses LangChain v1 syntax with initialize_agent, not create_react_agent.
"""

from typing import Any, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain.agents import AgentExecutor, initialize_agent, AgentType

from ..base import BaseToolWrapper
from .tools import create_langchain_tools


def create_agent_for_env(
    wrapper: BaseToolWrapper,
    llm: BaseLanguageModel,
    agent_type: AgentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    max_iterations: int = 50,
    verbose: bool = True,
    handle_parsing_errors: bool = True,
    **agent_kwargs,
) -> AgentExecutor:
    """Create a LangChain agent for an AgentEnv environment.
    
    This creates an agent using LangChain's v1 initialize_agent syntax
    (not the newer create_react_agent), configured for the environment.
    
    Args:
        wrapper: The AgentEnv tool wrapper
        llm: Language model to use for the agent
        agent_type: Type of LangChain agent to create
        max_iterations: Maximum steps before stopping
        verbose: Whether to print agent's reasoning
        handle_parsing_errors: Whether to handle LLM output parsing errors
        **agent_kwargs: Additional arguments for initialize_agent
        
    Returns:
        Configured AgentExecutor
        
    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from agentenv_wrapper import BabyAIToolWrapper
        >>> from agentenv_wrapper.langchain import create_agent_for_env
        >>> 
        >>> wrapper = BabyAIToolWrapper("http://localhost:8080")
        >>> wrapper.reset(task_id=1)
        >>> 
        >>> llm = ChatOpenAI(model="gpt-4", temperature=0)
        >>> agent = create_agent_for_env(wrapper, llm)
        >>> 
        >>> # Run the agent
        >>> result = agent.invoke({"input": wrapper.get_system_prompt()})
    """
    tools = create_langchain_tools(wrapper)
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=agent_type,
        verbose=verbose,
        max_iterations=max_iterations,
        handle_parsing_errors=handle_parsing_errors,
        **agent_kwargs,
    )
    
    return agent


class AgentEnvRunner:
    """Runner class for executing LangChain agents on AgentEnv tasks.
    
    This class provides a convenient interface for:
    - Setting up environments and agents
    - Running episodes with automatic tool updates
    - Collecting trajectories and metrics
    
    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from agentenv_wrapper import BabyAIToolWrapper
        >>> from agentenv_wrapper.langchain import AgentEnvRunner
        >>> 
        >>> wrapper = BabyAIToolWrapper("http://localhost:8080")
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> 
        >>> runner = AgentEnvRunner(wrapper, llm)
        >>> result = runner.run_episode(task_id=1)
        >>> print(f"Success: {result['success']}, Steps: {result['steps']}")
    """
    
    def __init__(
        self,
        wrapper: BaseToolWrapper,
        llm: BaseLanguageModel,
        agent_type: AgentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations: int = 50,
        verbose: bool = True,
    ):
        """Initialize the runner.
        
        Args:
            wrapper: The AgentEnv tool wrapper
            llm: Language model for the agent
            agent_type: Type of agent to use
            max_iterations: Maximum iterations per episode
            verbose: Whether to print agent reasoning
        """
        self.wrapper = wrapper
        self.llm = llm
        self.agent_type = agent_type
        self.max_iterations = max_iterations
        self.verbose = verbose
        self._agent: Optional[AgentExecutor] = None
    
    def _create_agent(self) -> AgentExecutor:
        """Create or recreate the agent with current tools."""
        return create_agent_for_env(
            wrapper=self.wrapper,
            llm=self.llm,
            agent_type=self.agent_type,
            max_iterations=self.max_iterations,
            verbose=self.verbose,
        )
    
    def run_episode(
        self,
        task_id: int = 0,
        custom_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run a single episode on the specified task.
        
        Args:
            task_id: Task/level identifier
            custom_prompt: Optional custom prompt (uses default if None)
            
        Returns:
            Dictionary with episode results:
                - success: Whether task was completed
                - reward: Final reward
                - steps: Number of steps taken
                - trajectory: List of (action, observation) pairs
                - final_state: Final environment state
        """
        # Reset environment
        state = self.wrapper.reset(task_id=task_id)
        
        # Create agent with current tools
        self._agent = self._create_agent()
        
        # Prepare prompt
        prompt = custom_prompt or self.wrapper.get_system_prompt()
        
        # Collect trajectory
        trajectory = []
        
        try:
            # Run agent
            result = self._agent.invoke({"input": prompt})
            
            # Get final state
            final_state = self.wrapper.get_state()
            
            return {
                "success": final_state.done,
                "reward": final_state.reward,
                "steps": final_state.step_count,
                "output": result.get("output", ""),
                "trajectory": trajectory,
                "final_state": final_state,
            }
            
        except Exception as e:
            final_state = self.wrapper.get_state()
            return {
                "success": False,
                "reward": final_state.reward,
                "steps": final_state.step_count,
                "error": str(e),
                "trajectory": trajectory,
                "final_state": final_state,
            }
    
    def run_batch(
        self,
        task_ids: list[int],
        stop_on_failure: bool = False,
    ) -> list[dict[str, Any]]:
        """Run multiple episodes.
        
        Args:
            task_ids: List of task identifiers to run
            stop_on_failure: Whether to stop on first failure
            
        Returns:
            List of episode results
        """
        results = []
        
        for task_id in task_ids:
            result = self.run_episode(task_id=task_id)
            results.append(result)
            
            if stop_on_failure and not result["success"]:
                break
        
        return results
    
    def compute_metrics(self, results: list[dict[str, Any]]) -> dict[str, float]:
        """Compute aggregate metrics from episode results.
        
        Args:
            results: List of episode results from run_batch
            
        Returns:
            Dictionary with aggregate metrics
        """
        if not results:
            return {}
        
        successes = sum(1 for r in results if r.get("success", False))
        total_reward = sum(r.get("reward", 0) for r in results)
        total_steps = sum(r.get("steps", 0) for r in results)
        
        return {
            "success_rate": successes / len(results),
            "avg_reward": total_reward / len(results),
            "avg_steps": total_steps / len(results),
            "total_episodes": len(results),
            "successful_episodes": successes,
        }
