"""Control agent for the pool-size study: the reference agent with a settable number of rollout-evaluated first moves.

``POOL_ROLL_N`` (environment variable, default 9 = reference) sets how many first moves are evaluated by rollouts
(``W.roll_n``). 0 disables rollouts: the agent then scores every distinct bag in the pool once and places the best
(selection without look-ahead). Used to separate the gain from choosing among the pool from the gain from rollouts.
"""
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE_PATH = os.path.join(os.path.dirname(_HERE), 'extreme_point', 'agent.py')
_spec = importlib.util.spec_from_file_location('ep_control_base', _BASE_PATH)
base = importlib.util.module_from_spec(_spec)
sys.modules['ep_control_base'] = base
_spec.loader.exec_module(base)


class Agent(base.Agent):
    """Reference agent; only ``W.roll_n`` differs, set from the environment at the start of every episode."""

    def get_init_states(self, init_states):
        base.W.roll_n = int(os.environ.get('POOL_ROLL_N', 9))
        return super().get_init_states(init_states)
