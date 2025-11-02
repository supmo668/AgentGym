"""
Launch script for MCP Environment Server.
"""

import uvicorn
import click


@click.command()
@click.option("--host", default="127.0.0.1", help="Host address to bind")
@click.option("--port", default=8000, type=int, help="Port to bind")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def main(host: str, port: int, reload: bool):
    """
    Launch the MCP Environment Server.
    
    This starts a FastAPI server that provides HTTP endpoints for
    interacting with the simulated MCP environment.
    """
    print(f"Starting MCP Environment Server on {host}:{port}")
    print("Press CTRL+C to stop the server")
    
    uvicorn.run(
        "agentenv_mcp.mcp_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
