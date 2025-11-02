"""
FastAPI Server for MCP Environment.
"""

from typing import List
from fastapi import FastAPI
from .mcp_server_wrapper import mcp_env_server
from .mcp_model import (
    CreateQuery,
    StepQuery,
    ResetQuery,
    CloseQuery,
    StepResponse
)

app = FastAPI(
    title="MCP Environment Server",
    description="Simulated Model Context Protocol server with Milvus collections",
    version="0.1.0"
)


@app.get("/", response_model=str)
def hello():
    """Test connectivity."""
    return "MCP Environment Server - Ready"


@app.get("/list_envs", response_model=List[int])
def list_envs():
    """List all active environment IDs."""
    return list(mcp_env_server.env.keys())


@app.post("/create", response_model=int)
def create(create_query: CreateQuery):
    """
    Create a new MCP environment instance.
    
    Args:
        create_query: Query with optional task ID
        
    Returns:
        Environment ID
    """
    env_id = mcp_env_server.create(create_query.id)
    return env_id


@app.post("/step", response_model=StepResponse)
def step(step_query: StepQuery):
    """
    Execute an action in the environment.
    
    Args:
        step_query: Query with environment ID and action
        
    Returns:
        Step response with observation, reward, and done status
    """
    result = mcp_env_server.step(step_query.env_idx, step_query.action)
    return StepResponse(
        observation=result["observation"],
        reward=result["reward"],
        done=result["done"]
    )


@app.get("/observation", response_model=str)
def observation(env_idx: int):
    """
    Get cached observation for environment.
    
    Args:
        env_idx: Environment ID
        
    Returns:
        Current observation string
    """
    result = mcp_env_server.observation(env_idx)
    if "error" in result:
        return result["error"]
    return result["observation"]


@app.post("/reset", response_model=str)
def reset(reset_query: ResetQuery):
    """
    Reset environment to initial state.
    
    Args:
        reset_query: Query with environment ID and optional task ID
        
    Returns:
        Initial observation
    """
    result = mcp_env_server.reset(reset_query.env_idx, reset_query.id)
    if isinstance(result, int):
        # New environment created
        return mcp_env_server.observation(result)["observation"]
    return result["observation"]


@app.post("/close")
def close(close_query: CloseQuery):
    """
    Close and cleanup environment.
    
    Args:
        close_query: Query with environment ID
        
    Returns:
        Close status
    """
    return mcp_env_server.close(close_query.env_idx)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_environments": len(mcp_env_server.env),
        "version": "0.1.0"
    }
