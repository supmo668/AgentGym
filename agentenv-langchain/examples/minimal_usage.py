"""
Example: Minimal usage of agentenv-langchain.

This shows the simplest way to use the wrapper.
"""

from agentenv_langchain import (
    AgentEnvToolWrapper,
    EnvAction,
    create_agentenv_tools,
)


def main():
    """Minimal example."""
    
    # 1. Define your environment's actions
    actions = [
        EnvAction(name="turn_left", description="Turn left"),
        EnvAction(name="turn_right", description="Turn right"),
        EnvAction(name="move_forward", description="Move forward"),
    ]
    
    # 2. Create the wrapper
    wrapper = AgentEnvToolWrapper(
        env_server_base="http://localhost:8000",  # Your AgentGym server
        env_name="BabyAI",
        actions=actions,
        auto_create=True,  # Automatically create env instance
    )
    
    # 3. Get LangChain tools
    tools = create_agentenv_tools(wrapper)
    
    print(f"Created {len(tools)} tools:")
    for tool in tools:
        print(f"  - {tool.name}")
    
    # 4. Use with LangChain (pseudo-code)
    """
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain import hub
    
    llm = ChatOpenAI(model="gpt-4")
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    
    # Reset and run
    wrapper.reset(0)
    result = executor.invoke({"input": "Navigate to the red ball"})
    """
    
    # 5. Cleanup
    wrapper.close()


if __name__ == "__main__":
    main()
