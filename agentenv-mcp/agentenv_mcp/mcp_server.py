"""
FastAPI Server for MCP Environment.
Supports both REST and SSE (Server-Sent Events) transports.
"""

from typing import List, AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json
import asyncio
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
        "version": "0.1.0",
        "transport": ["rest", "sse"]
    }


@app.get("/sse")
async def sse_endpoint(request: Request):
    """
    Server-Sent Events endpoint for MCP protocol.
    
    This endpoint provides SSE transport for the MCP server,
    allowing real-time streaming of environment state updates.
    """
    
    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events."""
        try:
            # Send initial connection message
            yield f"data: {json.dumps({'type': 'connected', 'version': '0.1.0'})}\n\n"
            
            # Keep connection alive with periodic heartbeats
            while True:
                if await request.is_disconnected():
                    break
                
                # Send heartbeat every 30 seconds
                yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': asyncio.get_event_loop().time()})}\n\n"
                await asyncio.sleep(30)
                
        except asyncio.CancelledError:
            # Client disconnected
            pass
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/sse/message")
async def sse_message(request: Request):
    """
    Handle SSE messages from client.
    
    This endpoint receives JSON-RPC style messages and routes them
    to the appropriate handlers.
    """
    try:
        data = await request.json()
        method = data.get("method")
        params = data.get("params", {})
        request_id = data.get("id")
        
        # Route to appropriate handler
        if method == "create":
            env_id = mcp_env_server.create(params.get("id", 0))
            result = {"env_id": env_id}
        elif method == "step":
            result = mcp_env_server.step(params["env_idx"], params["action"])
        elif method == "reset":
            result = mcp_env_server.reset(params["env_idx"], params.get("id"))
        elif method == "observation":
            result = mcp_env_server.observation(params["env_idx"])
        elif method == "close":
            result = mcp_env_server.close(params["env_idx"])
        else:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": f"Method not found: {method}"},
                "id": request_id
            }
        
        return {
            "jsonrpc": "2.0",
            "result": result,
            "id": request_id
        }
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32603, "message": str(e)},
            "id": request_id if 'request_id' in locals() else None
        }
