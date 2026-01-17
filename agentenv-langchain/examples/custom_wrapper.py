"""
Example: Custom environment wrapper.

This example shows how to create a custom wrapper for any AgentGym environment
that doesn't have pre-defined actions.
"""

from agentenv_langchain.wrapper import AgentEnvToolWrapper, EnvAction, EnvObservation
from agentenv_langchain.tools import create_agentenv_tools


def create_custom_wrapper(
    server_url: str,
    env_name: str,
):
    """
    Example of creating a custom wrapper for a hypothetical environment.
    
    This demonstrates how to:
    1. Define custom actions
    2. Create custom observation parsers
    3. Create custom action parsers
    """
    
    # Define custom actions for your environment
    custom_actions = [
        EnvAction(
            name="search",
            description="Search for information on a topic",
            parameters={"query": "string"},
            required_params=["query"],
        ),
        EnvAction(
            name="click",
            description="Click on an element on the page",
            parameters={"element_id": "string"},
            required_params=["element_id"],
        ),
        EnvAction(
            name="type_text",
            description="Type text into an input field",
            parameters={"element_id": "string", "text": "string"},
            required_params=["element_id", "text"],
        ),
        EnvAction(
            name="scroll",
            description="Scroll the page",
            parameters={"direction": "string", "amount": "integer"},
            required_params=["direction"],
        ),
        EnvAction(
            name="back",
            description="Go back to the previous page",
        ),
    ]
    
    # Custom observation parser
    def parse_observation(response: dict) -> EnvObservation:
        """Custom parser for WebArena-style responses."""
        return EnvObservation(
            state=response.get("html_content", response.get("observation", "")),
            reward=response.get("reward", 0.0),
            done=response.get("done", False),
            info={
                "url": response.get("url", ""),
                "page_title": response.get("title", ""),
            },
            available_actions=[a.name for a in custom_actions],
        )
    
    # Custom action parser
    def parse_action(action: str) -> str:
        """Custom parser to transform actions."""
        # Example: normalize action names
        action = action.strip().lower()
        action_mapping = {
            "go back": "back",
            "navigate back": "back",
            "search for": "search",
        }
        for pattern, replacement in action_mapping.items():
            if action.startswith(pattern):
                action = action.replace(pattern, replacement, 1)
        return action
    
    # Create the wrapper
    wrapper = AgentEnvToolWrapper(
        env_server_base=server_url,
        env_name=env_name,
        actions=custom_actions,
        observation_parser=parse_observation,
        action_parser=parse_action,
        auto_create=False,  # Don't auto-create, useful for testing
    )
    
    return wrapper


def demonstrate_tool_creation():
    """Demonstrate how tools are created from the wrapper."""
    
    # Create a mock wrapper (won't connect to server)
    from agentenv_langchain.wrapper import AgentEnvToolWrapper, EnvAction
    
    # Define minimal actions
    actions = [
        EnvAction(name="move", description="Move in a direction", 
                  parameters={"direction": "string"}, required_params=["direction"]),
        EnvAction(name="look", description="Look around"),
        EnvAction(name="interact", description="Interact with an object",
                  parameters={"object": "string"}, required_params=["object"]),
    ]
    
    print("Action definitions:")
    print("-" * 40)
    for action in actions:
        print(f"  {action.name}: {action.description}")
        if action.parameters:
            print(f"    Parameters: {action.parameters}")
    
    print("\n\nTo create LangChain tools:")
    print("-" * 40)
    print("""
from langchain.tools import Tool, StructuredTool
from agentenv_langchain import AgentEnvToolWrapper, create_agentenv_tools

# Create wrapper with actions
wrapper = AgentEnvToolWrapper(
    env_server_base="http://localhost:8000",
    env_name="MyEnv",
    actions=actions,
)

# Convert to LangChain tools
tools = create_agentenv_tools(wrapper)

# Use with create_react_agent
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain import hub

llm = ChatOpenAI(model="gpt-4")
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

# Run the agent
result = executor.invoke({"input": "Complete the task"})
""")


if __name__ == "__main__":
    demonstrate_tool_creation()
