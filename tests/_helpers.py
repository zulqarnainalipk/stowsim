"""Shared helpers for the tests."""
import copy
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)


def small_config(n_items=6, look_ahead=3, optimize=False, task='001'):
    """A tiny task derived from the sample configuration (fast to simulate)."""
    with open(os.path.join(ROOT, 'configs', 'sample_config.json')) as f:
        cfg = json.load(f)[task]
    cfg = copy.deepcopy(cfg)
    cfg['item_stream']['item_list'] = cfg['item_stream']['item_list'][:n_items]
    cfg['item_stream']['look_ahead'] = look_ahead
    cfg['agent']['optimize'] = optimize
    return cfg
