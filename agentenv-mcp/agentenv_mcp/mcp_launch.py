"""
Launch script for MCP Environment Server.
"""

import os
import uvicorn
import click
from pathlib import Path


def load_env():
    """Load environment variables from .env file if present."""
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


@click.command()
@click.option("--host", default=lambda: os.getenv("MCP_HOST", "127.0.0.1"), help="Host address")
@click.option("--port", default=lambda: int(os.getenv("MCP_PORT", "8000")), type=int, help="Port")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def main(host: str, port: int, reload: bool):
    """Launch the MCP Environment Server."""
    load_env()
    print(f"Starting MCP Environment Server on {host}:{port}")
    
    uvicorn.run(
        "agentenv_mcp.mcp_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level=os.getenv("LOG_LEVEL", "info")
    )


if __name__ == "__main__":
    main()
