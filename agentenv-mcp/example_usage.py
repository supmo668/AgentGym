"""
Example: Using the MCP Environment

This script demonstrates how to interact with the MCP environment
both directly and through the client-server architecture.
"""

import time
from agentenv_mcp.mcp_environment import MCPEnv


def example_direct_interaction():
    """Example of direct interaction with MCP environment."""
    print("=" * 70)
    print("EXAMPLE 1: Direct Interaction with MCP Environment")
    print("=" * 70)
    
    # Create environment
    env = MCPEnv(task_data={"goal": "List all collections and describe the documents collection"})
    
    # Reset to start
    observation = env.reset(idx=0)
    print(f"\n{observation}\n")
    
    # Step 1: List collections
    print("\n" + "-" * 70)
    print("Step 1: List available collections")
    print("-" * 70)
    action = "Action: list_collections with Action Input: {}"
    print(f"Action: {action}")
    observation, reward, done, info = env.step(action)
    print(f"\n{observation}")
    print(f"Reward: {reward}, Done: {done}")
    
    # Step 2: Get collection info
    print("\n" + "-" * 70)
    print("Step 2: Get information about 'documents' collection")
    print("-" * 70)
    action = 'Action: get_collection_info with Action Input: {"collection_name": "documents"}'
    print(f"Action: {action}")
    observation, reward, done, info = env.step(action)
    print(f"\n{observation}")
    print(f"Reward: {reward}, Done: {done}")
    
    # Step 3: Query collection
    print("\n" + "-" * 70)
    print("Step 3: Query the documents collection")
    print("-" * 70)
    action = 'Action: query_collection with Action Input: {"collection_name": "documents", "limit": 2}'
    print(f"Action: {action}")
    observation, reward, done, info = env.step(action)
    print(f"\n{observation}")
    print(f"Reward: {reward}, Done: {done}")
    
    # Step 4: Search collection
    print("\n" + "-" * 70)
    print("Step 4: Search for 'machine learning' in documents")
    print("-" * 70)
    action = 'Action: search_collection with Action Input: {"collection_name": "documents", "query": "machine learning", "top_k": 2}'
    print(f"Action: {action}")
    observation, reward, done, info = env.step(action)
    print(f"\n{observation}")
    print(f"Reward: {reward}, Done: {done}")
    
    # Step 5: Finish
    print("\n" + "-" * 70)
    print("Step 5: Finish the task")
    print("-" * 70)
    action = 'Action: finish with Action Input: {"answer": "Found 3 documents: ML intro, Deep Learning, and NLP"}'
    print(f"Action: {action}")
    observation, reward, done, info = env.step(action)
    print(f"\n{observation}")
    print(f"Reward: {reward}, Done: {done}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Task Summary:")
    print(f"  Total steps: {env.current_step}")
    print(f"  Task completed: {env.task_completed}")
    print(f"  Final result: {env.result}")
    print("=" * 70)
    
    env.close()


def example_with_resources():
    """Example using MCP resources (schemas, prompts, formatters)."""
    print("\n\n")
    print("=" * 70)
    print("EXAMPLE 2: Using MCP Resources")
    print("=" * 70)
    
    from agentenv_mcp.mcp_resources import MCPResources
    
    resources = MCPResources()
    
    # Example 1: Using schemas
    print("\n" + "-" * 70)
    print("Resource Schemas:")
    print("-" * 70)
    tool_schema = resources.get_schema("tool_schema")
    print(f"Tool Schema: {tool_schema}")
    
    # Example 2: Using prompt templates
    print("\n" + "-" * 70)
    print("Prompt Templates:")
    print("-" * 70)
    prompt = resources.get_prompt(
        "search_prompt",
        collections="documents, users, products",
        task="Find information about machine learning"
    )
    print(f"Search Prompt:\n{prompt}")
    
    # Example 3: Using formatters
    print("\n" + "-" * 70)
    print("Response Formatters:")
    print("-" * 70)
    data = [
        {"id": 1, "name": "Item 1", "value": 100},
        {"id": 2, "name": "Item 2", "value": 200}
    ]
    
    print("\nJSON Format:")
    print(resources.format_response(data, "json"))
    
    print("\nMarkdown Format:")
    print(resources.format_response(data, "markdown"))


def example_multiple_tools():
    """Example demonstrating multiple tool usage."""
    print("\n\n")
    print("=" * 70)
    print("EXAMPLE 3: Multiple Tool Demonstration")
    print("=" * 70)
    
    env = MCPEnv(task_data={"goal": "Demonstrate all available tools"})
    env.reset(idx=0)
    
    tools_to_test = [
        ("list_collections", {}),
        ("get_collection_info", {"collection_name": "users"}),
        ("query_collection", {"collection_name": "products", "limit": 2}),
        ("search_collection", {"collection_name": "documents", "query": "NLP", "top_k": 1}),
        ("get_schema", {"schema_name": "response_schema"}),
        ("get_prompt", {"prompt_name": "query_prompt", "collection_name": "users", "filter_expr": "id > 100", "limit": "5"}),
    ]
    
    for i, (tool_name, params) in enumerate(tools_to_test, 1):
        print(f"\n{'-' * 70}")
        print(f"Tool {i}/{len(tools_to_test)}: {tool_name}")
        print(f"Parameters: {params}")
        print("-" * 70)
        
        import json
        params_str = json.dumps(params) if params else "{}"
        action = f"Action: {tool_name} with Action Input: {params_str}"
        
        observation, reward, done, info = env.step(action)
        print(f"Result: {observation[:200]}...")
        print(f"Reward: {reward}")
    
    print("\n" + "=" * 70)
    print(f"Demonstrated {len(tools_to_test)} tools successfully!")
    print("=" * 70)
    
    env.close()


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "MCP ENVIRONMENT EXAMPLES" + " " * 29 + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Run examples
    example_direct_interaction()
    example_with_resources()
    example_multiple_tools()
    
    print("\n\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "ALL EXAMPLES COMPLETED" + " " * 26 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\nTo use with AgentGym client-server architecture:")
    print("  1. Start server: mcp --host 0.0.0.0 --port 8000")
    print("  2. Use MCPEnvClient from agentenv.envs.mcp")
    print()
