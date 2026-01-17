"""
AgentEnv Tool Wrapper.

Provides a general wrapper for adapting any AgentGym environment
into a tool-based interface compatible with LangChain agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Union
import requests
from requests.exceptions import RequestException


@dataclass
class EnvAction:
    """Represents an action in the environment."""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_params: List[str] = field(default_factory=list)


@dataclass
class EnvObservation:
    """Represents an observation from the environment."""
    state: str
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)
    available_actions: List[str] = field(default_factory=list)


class AgentEnvProtocol(Protocol):
    """Protocol defining the interface for AgentGym environments."""
    
    def step(self, action: str) -> Dict[str, Any]:
        """Execute an action in the environment."""
        ...
    
    def reset(self, data_idx: int = 0) -> Dict[str, Any]:
        """Reset the environment."""
        ...
    
    def observe(self) -> str:
        """Get the current observation."""
        ...


class AgentEnvToolWrapper:
    """
    General wrapper for adapting any AgentGym environment into a tool-based interface.
    
    This wrapper connects to AgentGym environment servers and exposes their
    actions as callable tools compatible with LangChain's create_react_agent.
    
    Example usage:
        >>> wrapper = AgentEnvToolWrapper(
        ...     env_server_base="http://localhost:8000",
        ...     env_name="BabyAI",
        ...     actions=[
        ...         EnvAction(name="turn_left", description="Turn the agent left"),
        ...         EnvAction(name="turn_right", description="Turn the agent right"),
        ...         EnvAction(name="move_forward", description="Move forward one step"),
        ...     ]
        ... )
        >>> tools = wrapper.as_langchain_tools()
    """
    
    def __init__(
        self,
        env_server_base: str,
        env_name: str,
        actions: Optional[List[EnvAction]] = None,
        action_parser: Optional[Callable[[str], str]] = None,
        observation_parser: Optional[Callable[[Dict[str, Any]], EnvObservation]] = None,
        timeout: int = 300,
        auto_create: bool = True,
    ):
        """
        Initialize the AgentEnv tool wrapper.
        
        Args:
            env_server_base: Base URL of the AgentGym environment server.
            env_name: Name of the environment (e.g., "BabyAI", "ALFWorld").
            actions: List of available actions. If None, will try to discover from server.
            action_parser: Custom parser to transform action inputs before sending to server.
            observation_parser: Custom parser to transform server responses into EnvObservation.
            timeout: Request timeout in seconds.
            auto_create: Whether to automatically create an environment on init.
        """
        self.env_server_base = env_server_base.rstrip("/")
        self.env_name = env_name
        self.timeout = timeout
        self.env_id: Optional[int] = None
        
        self._actions = actions or []
        self._action_parser = action_parser or self._default_action_parser
        self._observation_parser = observation_parser or self._default_observation_parser
        
        self._current_observation: Optional[EnvObservation] = None
        self._step_count = 0
        self._total_reward = 0.0
        
        if auto_create:
            self.create()
    
    def _default_action_parser(self, action: str) -> str:
        """Default action parser - returns action as-is."""
        return action.strip()
    
    def _default_observation_parser(self, response: Dict[str, Any]) -> EnvObservation:
        """Default observation parser for standard AgentGym responses."""
        return EnvObservation(
            state=response.get("observation", str(response)),
            reward=float(response.get("reward", 0.0)),
            done=bool(response.get("done", False)),
            info={k: v for k, v in response.items() if k not in ("observation", "reward", "done")},
            available_actions=[a.name for a in self._actions],
        )
    
    def _post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send POST request to the environment server."""
        if self.env_id is not None:
            data["id"] = self.env_id
        
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        
        if res.status_code != 200:
            raise RequestException(f"Request failed with status {res.status_code}: {res.text}")
        
        return res.json()
    
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send GET request to the environment server."""
        params = params or {}
        if self.env_id is not None:
            params["id"] = self.env_id
        
        res = requests.get(
            f"{self.env_server_base}/{path}",
            params=params,
            timeout=self.timeout,
        )
        
        if res.status_code != 200:
            raise RequestException(f"Request failed with status {res.status_code}: {res.text}")
        
        return res.json()
    
    def create(self) -> int:
        """Create a new environment instance on the server."""
        response = self._post("create", {})
        self.env_id = response.get("id", response)
        return self.env_id
    
    def reset(self, data_idx: int = 0) -> EnvObservation:
        """Reset the environment to a specific task index."""
        response = self._post("reset", {"data_idx": data_idx})
        self._step_count = 0
        self._total_reward = 0.0
        self._current_observation = self._observation_parser(response)
        return self._current_observation
    
    def step(self, action: str, **kwargs) -> EnvObservation:
        """
        Execute an action in the environment.
        
        Args:
            action: The action to execute.
            **kwargs: Additional parameters for parameterized actions.
        
        Returns:
            EnvObservation with the result of the action.
        """
        # Parse action with any additional parameters
        parsed_action = self._action_parser(action)
        
        # Format action with parameters if provided
        if kwargs:
            parsed_action = self._format_action_with_params(parsed_action, kwargs)
        
        response = self._post("step", {"action": parsed_action})
        
        self._step_count += 1
        observation = self._observation_parser(response)
        self._total_reward += observation.reward
        self._current_observation = observation
        
        return observation
    
    def observe(self) -> EnvObservation:
        """Get the current observation from the environment."""
        if self._current_observation is not None:
            return self._current_observation
        
        response = self._get("observation")
        if isinstance(response, str):
            return EnvObservation(
                state=response,
                reward=0.0,
                done=False,
                available_actions=[a.name for a in self._actions],
            )
        return self._observation_parser(response)
    
    def close(self) -> bool:
        """Close the environment instance."""
        if self.env_id is None:
            return False
        
        try:
            self._post("close", {})
            self.env_id = None
            return True
        except RequestException:
            return False
    
    def _format_action_with_params(self, action: str, params: Dict[str, Any]) -> str:
        """Format an action with its parameters."""
        param_str = " ".join(f"{k}={v}" for k, v in params.items())
        return f"{action} {param_str}".strip()
    
    @property
    def actions(self) -> List[EnvAction]:
        """Get the list of available actions."""
        return self._actions
    
    @actions.setter
    def actions(self, actions: List[EnvAction]) -> None:
        """Set the list of available actions."""
        self._actions = actions
    
    @property
    def step_count(self) -> int:
        """Get the current step count."""
        return self._step_count
    
    @property
    def total_reward(self) -> float:
        """Get the total accumulated reward."""
        return self._total_reward
    
    @property
    def is_done(self) -> bool:
        """Check if the episode is done."""
        return self._current_observation is not None and self._current_observation.done
    
    def get_action_by_name(self, name: str) -> Optional[EnvAction]:
        """Get an action by its name."""
        for action in self._actions:
            if action.name == name:
                return action
        return None
    
    def as_langchain_tools(self):
        """
        Convert the environment actions to LangChain tools.
        
        Returns:
            List of LangChain Tool objects for use with create_react_agent.
        """
        # Import here to avoid dependency issues
        from agentenv_langchain.tools import create_agentenv_tools
        return create_agentenv_tools(self)


# Pre-defined action sets for common environments
BABYAI_ACTIONS = [
    EnvAction(
        name="turn_left",
        description="Turn the agent to the left.",
    ),
    EnvAction(
        name="turn_right",
        description="Turn the agent to the right.",
    ),
    EnvAction(
        name="move_forward",
        description="Move the agent one step forward.",
    ),
    EnvAction(
        name="go_to",
        description="Navigate to a specific object.",
        parameters={"obj": "string", "id": "integer"},
        required_params=["obj", "id"],
    ),
    EnvAction(
        name="pick_up",
        description="Pick up an object.",
        parameters={"obj": "string", "id": "integer"},
        required_params=["obj", "id"],
    ),
    EnvAction(
        name="go_through",
        description="Go through an open door.",
        parameters={"door": "string", "id": "integer"},
        required_params=["door", "id"],
    ),
    EnvAction(
        name="toggle_and_go_through",
        description="Toggle a door (open/unlock) and go through it.",
        parameters={"door": "string", "id": "integer"},
        required_params=["door", "id"],
    ),
    EnvAction(
        name="toggle",
        description="Toggle the door right in front of you.",
    ),
]


ALFWORLD_ACTIONS = [
    EnvAction(
        name="goto",
        description="Move towards a specific receptacle or location.",
        parameters={"recep": "string"},
        required_params=["recep"],
    ),
    EnvAction(
        name="take",
        description="Pick up an object from a receptacle.",
        parameters={"obj": "string", "recep": "string"},
        required_params=["obj", "recep"],
    ),
    EnvAction(
        name="put",
        description="Put an object on/in a receptacle.",
        parameters={"obj": "string", "recep": "string"},
        required_params=["obj", "recep"],
    ),
    EnvAction(
        name="open",
        description="Open a receptacle.",
        parameters={"recep": "string"},
        required_params=["recep"],
    ),
    EnvAction(
        name="close",
        description="Close a receptacle.",
        parameters={"recep": "string"},
        required_params=["recep"],
    ),
    EnvAction(
        name="toggle",
        description="Toggle a device on or off.",
        parameters={"obj": "string"},
        required_params=["obj"],
    ),
    EnvAction(
        name="clean",
        description="Clean an object with a receptacle.",
        parameters={"obj": "string", "recep": "string"},
        required_params=["obj", "recep"],
    ),
    EnvAction(
        name="heat",
        description="Heat an object with a receptacle.",
        parameters={"obj": "string", "recep": "string"},
        required_params=["obj", "recep"],
    ),
    EnvAction(
        name="cool",
        description="Cool an object with a receptacle.",
        parameters={"obj": "string", "recep": "string"},
        required_params=["obj", "recep"],
    ),
    EnvAction(
        name="examine",
        description="Examine an object or receptacle.",
        parameters={"obj": "string"},
        required_params=["obj"],
    ),
    EnvAction(
        name="inventory",
        description="Check your current inventory.",
    ),
    EnvAction(
        name="look",
        description="Look around to see your surroundings.",
    ),
]


def create_wrapper_for_env(
    env_name: str,
    env_server_base: str,
    **kwargs,
) -> AgentEnvToolWrapper:
    """
    Factory function to create a wrapper for a known environment type.
    
    Args:
        env_name: Name of the environment (e.g., "BabyAI", "ALFWorld").
        env_server_base: Base URL of the environment server.
        **kwargs: Additional arguments for AgentEnvToolWrapper.
    
    Returns:
        Configured AgentEnvToolWrapper instance.
    """
    action_sets = {
        "babyai": BABYAI_ACTIONS,
        "alfworld": ALFWORLD_ACTIONS,
    }
    
    env_lower = env_name.lower()
    actions = action_sets.get(env_lower, [])
    
    return AgentEnvToolWrapper(
        env_server_base=env_server_base,
        env_name=env_name,
        actions=actions,
        **kwargs,
    )
