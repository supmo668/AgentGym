"""Example: Using BabyAI with LangChain agent.

This example demonstrates how to:
1. Create a BabyAI tool wrapper
2. Connect to a running BabyAI environment server
3. Create a LangChain agent
4. Run episodes and collect results

Prerequisites:
    - BabyAI environment server running (default: http://localhost:8080)
    - OpenAI API key set in environment
    
Usage:
    # Start BabyAI server first (from agentenv-babyai):
    uvicorn agentenv_babyai.server:app --host 0.0.0.0 --port 8080
    
    # Then run this example:
    python babyai_langchain_example.py
"""

import os
from typing import Optional

# Check for required environment variables
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not set. Set it to use this example.")


def run_babyai_agent(
    server_url: str = "http://localhost:8080",
    task_id: int = 1,
    model: str = "gpt-4",
    max_iterations: int = 30,
    verbose: bool = True,
) -> dict:
    """Run a LangChain agent on a BabyAI task.
    
    Args:
        server_url: URL of the BabyAI environment server
        task_id: Task identifier (1-40 for different levels)
        model: OpenAI model to use
        max_iterations: Maximum agent steps
        verbose: Print agent reasoning
        
    Returns:
        Episode results dictionary
    """
    from langchain_openai import ChatOpenAI
    from agentenv_wrapper import BabyAIToolWrapper
    from agentenv_wrapper.langchain import AgentEnvRunner
    
    # Create wrapper
    wrapper = BabyAIToolWrapper(server_url)
    
    # Create LLM
    llm = ChatOpenAI(model=model, temperature=0)
    
    # Create runner
    runner = AgentEnvRunner(
        wrapper=wrapper,
        llm=llm,
        max_iterations=max_iterations,
        verbose=verbose,
    )
    
    # Run episode
    result = runner.run_episode(task_id=task_id)
    
    # Cleanup
    wrapper.close()
    
    return result


def run_batch_evaluation(
    server_url: str = "http://localhost:8080",
    task_ids: Optional[list[int]] = None,
    model: str = "gpt-4",
) -> dict:
    """Run batch evaluation on multiple tasks.
    
    Args:
        server_url: URL of the BabyAI environment server
        task_ids: List of task IDs (default: 1-10)
        model: OpenAI model to use
        
    Returns:
        Aggregate metrics
    """
    from langchain_openai import ChatOpenAI
    from agentenv_wrapper import BabyAIToolWrapper
    from agentenv_wrapper.langchain import AgentEnvRunner
    
    if task_ids is None:
        task_ids = list(range(1, 11))
    
    wrapper = BabyAIToolWrapper(server_url)
    llm = ChatOpenAI(model=model, temperature=0)
    runner = AgentEnvRunner(wrapper=wrapper, llm=llm, verbose=False)
    
    results = runner.run_batch(task_ids)
    metrics = runner.compute_metrics(results)
    
    wrapper.close()
    
    return metrics


def manual_interaction_example(server_url: str = "http://localhost:8080"):
    """Example of manual tool interaction without LangChain agent.
    
    This shows how to use the wrapper directly for custom agents.
    """
    from agentenv_wrapper import BabyAIToolWrapper
    
    # Create wrapper
    wrapper = BabyAIToolWrapper(server_url)
    
    # Reset to task
    state = wrapper.reset(task_id=1)
    
    print(f"\n{'='*60}")
    print(f"Goal: {state.goal}")
    print(f"{'='*60}")
    print(f"\nObservation:\n{state.observation}")
    print(f"\nAvailable actions: {state.available_actions}")
    
    # Execute some actions manually
    actions_to_try = ["move forward", "turn right", "check available actions"]
    
    for action in actions_to_try:
        if action in state.available_actions:
            print(f"\n--- Executing: {action} ---")
            result = wrapper.execute_tool(action)
            print(f"Output: {result.output[:200]}...")
            print(f"Reward: {result.reward}, Done: {result.done}")
            
            if result.done:
                print("\n🎉 Task completed!")
                break
    
    wrapper.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BabyAI LangChain Example")
    parser.add_argument("--server", default="http://localhost:8080", help="BabyAI server URL")
    parser.add_argument("--task", type=int, default=1, help="Task ID (1-40)")
    parser.add_argument("--model", default="gpt-4", help="OpenAI model")
    parser.add_argument("--mode", choices=["single", "batch", "manual"], default="manual",
                        help="Execution mode")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        print(f"Running single episode on task {args.task}...")
        result = run_babyai_agent(
            server_url=args.server,
            task_id=args.task,
            model=args.model,
        )
        print(f"\nResult: Success={result['success']}, Steps={result['steps']}, Reward={result['reward']}")
        
    elif args.mode == "batch":
        print("Running batch evaluation on tasks 1-10...")
        metrics = run_batch_evaluation(
            server_url=args.server,
            model=args.model,
        )
        print(f"\nMetrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
            
    else:  # manual
        print("Running manual interaction example...")
        manual_interaction_example(args.server)
