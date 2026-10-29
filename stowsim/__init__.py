"""StowSim: a physics-based simulator for online and offline 3-D baggage stowage.

The package implements a Gymnasium-style environment (``stowsim.env.GroundHandlingEnv``) in which an
agent loads cuboid bags into side-opening containers. Placements are validated with PyBullet
(inclusion, insertion path, settling), and the loading is evaluated by volumetric fill.

Public entry points:
    * ``Agent``                         abstract base class every agent can derive from
    * ``stowsim.env.GroundHandlingEnv`` the environment
    * ``stowsim.app.EvaluationApp``     runs agents in an isolated process with time limits
"""
from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    """
    Abstract base class for bag-placement agents.
    To write your own agent, subclass this class and implement each method.
    """
    def __init__(self, module_path: str):
        self.module_path = module_path

    @abstractmethod
    def get_init_states(self, init_states: dict[str, Any]) -> None:
        """
        Receive the initial state of the environment.

        Args:
            init_states: initial settings such as the container information and lookahead_k
        """
        pass

    @abstractmethod
    def optimize(self, item_list: list[dict[str, Any]]) -> list[int]:
        """
        [Offline optimisation] Receive the information of all bags in advance and compute the best loading order.

        Args:
            item_list: list with the information of every bag
        Returns:
            list[int]: the bag indices in the new order
        """
        pass

    @abstractmethod
    def policy(self, observation: dict[str, Any]) -> dict[str, Any]:
        """
        [Online optimisation / sequential execution] Decide the next action from the current observation.

        Args:
            observation: the bags currently in the pool and the state of the containers
        Returns:
            dict: action dictionary with the chosen bag, container, position and orientation
        """
        pass
