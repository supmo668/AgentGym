"""
CLI for agentenv-langchain.
"""

import argparse
import sys


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="AgentEnv LangChain - Tool wrapper for AgentGym environments"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run a demo with a BabyAI environment")
    demo_parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="URL of the BabyAI environment server",
    )
    demo_parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="LLM model to use",
    )
    demo_parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum number of steps",
    )
    
    # List actions command
    list_parser = subparsers.add_parser("list-actions", help="List available actions for an environment")
    list_parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=["babyai", "alfworld"],
        help="Environment name",
    )
    
    args = parser.parse_args()
    
    if args.command == "demo":
        run_demo(args)
    elif args.command == "list-actions":
        list_actions(args)
    else:
        parser.print_help()
        sys.exit(1)


def run_demo(args):
    """Run a demo with a BabyAI environment."""
    print(f"Connecting to BabyAI server at {args.server_url}...")
    
    try:
        from agentenv_langchain.wrapper import create_wrapper_for_env
        from agentenv_langchain.tools import create_agentenv_tools
    except ImportError as e:
        print(f"Import error: {e}")
        sys.exit(1)
    
    try:
        wrapper = create_wrapper_for_env("BabyAI", args.server_url)
        tools = create_agentenv_tools(wrapper)
        
        print(f"\nCreated {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:50]}...")
        
        print("\nTo run with an LLM agent, use the Python API:")
        print("""
from langchain_openai import ChatOpenAI
from agentenv_langchain import create_wrapper_for_env
from agentenv_langchain.tools import create_react_agent_for_env

wrapper = create_wrapper_for_env("BabyAI", "http://localhost:8000")
llm = ChatOpenAI(model="gpt-4")
executor = create_react_agent_for_env(wrapper, llm)

wrapper.reset(0)  # Reset to task 0
result = executor.invoke({"input": "Complete the task based on the current observation"})
""")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure the BabyAI server is running:")
        print("  cd agentenv-babyai && uvicorn agentenv_babyai.server:app --port 8000")
        sys.exit(1)


def list_actions(args):
    """List available actions for an environment."""
    from agentenv_langchain.wrapper import BABYAI_ACTIONS, ALFWORLD_ACTIONS
    
    action_sets = {
        "babyai": BABYAI_ACTIONS,
        "alfworld": ALFWORLD_ACTIONS,
    }
    
    actions = action_sets.get(args.env, [])
    
    print(f"\nAvailable actions for {args.env.upper()}:")
    print("-" * 50)
    
    for action in actions:
        print(f"\n{action.name}")
        print(f"  Description: {action.description}")
        if action.parameters:
            print(f"  Parameters: {action.parameters}")
            print(f"  Required: {action.required_params}")


if __name__ == "__main__":
    main()
