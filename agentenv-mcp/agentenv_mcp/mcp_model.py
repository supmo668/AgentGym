"""
Pydantic models for MCP environment API requests and responses.
"""

from pydantic import BaseModel
from typing import Optional


class CreateQuery(BaseModel):
    """Request model for creating a new environment."""
    id: Optional[int] = 0


class StepQuery(BaseModel):
    """Request model for executing an action."""
    env_idx: int
    action: str


class ResetQuery(BaseModel):
    """Request model for resetting an environment."""
    env_idx: int
    id: int = 0


class CloseQuery(BaseModel):
    """Request model for closing an environment."""
    env_idx: int


class StepResponse(BaseModel):
    """Response model for step action."""
    observation: str
    reward: float
    done: bool
