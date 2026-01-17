"""
Example: Using BabyAI environment with LangChain ReAct agent.

This example demonstrates how to wrap a BabyAI AgentGym environment
and use it with LangChain's create_react_agent.

Prerequisites:
1. Start the BabyAI server:
   cd agentenv-babyai && uvicorn agentenv_babyai.server:app --port 8000

2. Set your OpenAI API key:
   export OPENAI_API_KEY=your-api-key

3. Run this example:
   python examples/babyai_react_agent.py
"""

import os
from typing import Optional

# Check for API key
if not os.environ.get("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not set. Set it to run with a real LLM.")


def create_babyai_agent(
    server_url: str = "http://localhost:8000",
    model: str = "gpt-4",
    verbose: bool = True,
):
    """
    Create a ReAct agent for the BabyAI environment.
    
    Args:
        server_url: URL of the BabyAI environment server.
        model: LLM model to use.
        verbose: Whether to print agent steps.
    
    Returns:
        tuple: (wrapper, executor)
    """
    from langchain_openai import ChatOpenAI
    from agentenv_langchain.wrapper import create_wrapper_for_env
    from agentenv_langchain.tools import create_react_agent_for_env
    
    # Create the wrapper for BabyAI
    wrapper = create_wrapper_for_env("BabyAI", server_url)
    
    # Create the LLM
    llm = ChatOpenAI(model=model, temperature=0)
    
    # Create the agent executor
    executor = create_react_agent_for_env(
        wrapper,
        llm,
        verbose=verbose,
        max_iterations=30,
    )
    
    return wrapper, executor


def run_episode(
    wrapper,
    executor,
    task_idx: int = 0,
    max_steps: int = 30,
):
    """
    Run a single episode on the BabyAI environment.
    
    Args:
        wrapper: AgentEnvToolWrapper instance.
        executor: AgentExecutor instance.
        task_idx: Task index to run.
        max_steps: Maximum steps before stopping.
    
    Returns:
        dict: Episode results.
    """
    # Reset the environment
    initial_obs = wrapper.reset(task_idx)
    print(f"\n{'='*60}")
    print(f"Task {task_idx} - Initial Observation:")
    print(f"{'='*60}")
    print(initial_obs.state)
    print(f"{'='*60}\n")
    
    # Create the input prompt
    input_prompt = f"""You are playing a BabyAI navigation game. Your goal is to complete the task.

Current observation:
{initial_obs.state}

Use the available tools to navigate and interact with the environment.
Think step by step and take actions to achieve the goal.
When the task is complete (done=True), stop taking actions."""

    # Run the agent
    try:
        result = executor.invoke({"input": input_prompt})
        
        print(f"\n{'='*60}")
        print("Episode Complete!")
        print(f"{'='*60}")
        print(f"Total Steps: {wrapper.step_count}")
        print(f"Total Reward: {wrapper.total_reward}")
        print(f"Success: {wrapper.is_done and wrapper.total_reward > 0}")
        
        return {
            "task_idx": task_idx,
            "steps": wrapper.step_count,
            "reward": wrapper.total_reward,
            "success": wrapper.is_done and wrapper.total_reward > 0,
            "output": result.get("output", ""),
        }
        
    except Exception as e:
        print(f"Error during episode: {e}")
        return {
            "task_idx": task_idx,
            "steps": wrapper.step_count,
            "reward": wrapper.total_reward,
            "success": False,
            "error": str(e),
        }


def main():
    """Main function to run the BabyAI agent demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BabyAI ReAct Agent Demo")
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="BabyAI server URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="LLM model to use",
    )
    parser.add_argument(
        "--task-idx",
        type=int,
        default=0,
        help="Task index to run",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Number of episodes to run",
    )
    
    args = parser.parse_args()
    
    print("Creating BabyAI agent...")
    wrapper, executor = create_babyai_agent(
        server_url=args.server_url,
        model=args.model,
    )
    
    results = []
    for i in range(args.num_episodes):
        task_idx = args.task_idx + i
        print(f"\n\nRunning Episode {i+1}/{args.num_episodes} (Task {task_idx})")
        result = run_episode(wrapper, executor, task_idx)
        results.append(result)
    
    # Print summary
    print(f"\n\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    successes = sum(1 for r in results if r.get("success", False))
    total_reward = sum(r.get("reward", 0) for r in results)
    total_steps = sum(r.get("steps", 0) for r in results)
    
    print(f"Episodes: {len(results)}")
    print(f"Successes: {successes}/{len(results)}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Total Steps: {total_steps}")
    
    # Cleanup
    wrapper.close()


if __name__ == "__main__":
    main()
