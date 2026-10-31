"""Agent family for the static-against-physics study (E1): the reference agent with named rule and weight variants.

The variant is chosen per episode from the environment variable ``E1_VARIANT`` (default ``ref``). Each variant changes a few
fields of the reference packer's class ``W`` (support threshold, slenderness limit, soft-stacking limit, cost weights,
bag selection, rollouts) and nothing else; every other field is reset to the reference value at the start of the episode.
The agent also counts decisions and the decisions where it found no placement (``diag``), because the environment then
receives a dummy action, which is rejected like a blocked path.
"""
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE_PATH = os.path.join(os.path.dirname(_HERE), 'extreme_point', 'agent.py')
_spec = importlib.util.spec_from_file_location('ep_e1_base', _BASE_PATH)
base = importlib.util.module_from_spec(_spec)
sys.modules['ep_e1_base'] = base
_spec.loader.exec_module(base)

CM, Boxes = base.CM, base.Boxes   # tools.fast_sim reads these from the agent module

VARIANTS = {
    'ref': {},
    'norollout': dict(roll_n=0),
    'support50': dict(min_support=0.5),
    'support30': dict(min_support=0.3),
    'support90': dict(min_support=0.9),
    'slender3': dict(max_slender=3.0),
    'softstack4': dict(max_soft_stack=4),
    'heavy_first': dict(selm=2.0, selv=0.0),
    'nosoftpen': dict(softsup=0.0, head=0.0),
    'ignore_support_cost': dict(frac=0.0, z=0.0),
    'class_off': dict(class_rule=0),
    'softstack1': dict(max_soft_stack=1),
    'class_soft90': dict(class_rule=2, restricted_support=0.9),   # standard bag on a soft or priority bag allowed with support share >= 0.9
    'class_soft75': dict(class_rule=2, restricted_support=0.75),
}
_TOUCHED = sorted({k for v in VARIANTS.values() for k in v})
_DEFAULTS = {k: getattr(base.W, k) for k in _TOUCHED}


class Agent(base.Agent):
    """Reference agent with the variant named by ``E1_VARIANT``."""

    def get_init_states(self, init_states):
        for k, v in _DEFAULTS.items():
            setattr(base.W, k, v)
        for name in os.environ.get('E1_VARIANT', 'ref').split('+'):   # 'a+b' applies the variants a and b together
            for k, v in VARIANTS[name].items():
                setattr(base.W, k, v)
        self.diag = dict(n_decisions=0, n_no_placement=0)
        return super().get_init_states(init_states)

    def decide(self, state, pool, t0):
        res = super().decide(state, pool, t0)
        self.diag['n_decisions'] += 1
        if res is None:
            self.diag['n_no_placement'] += 1
        return res
