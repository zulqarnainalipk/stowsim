"""Tests of the optional behaviours: insertion-path step, engine settings, continue after a rejected action,
limited stacking on soft bags and the iteration budget of the offline search. Every option must leave the default unchanged."""
import contextlib
import copy
import io
import os
import unittest

import numpy as np

from tests._helpers import small_config
from stowsim.env import GroundHandlingEnv
from tools.run_agent import run


def _env(cfg):
    env = GroundHandlingEnv(config=cfg, verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):
        env.reset_settings()
    env.reset_item_stream()
    env.reset(seed=0)
    return env


def _dummy_action():
    return dict(item_idx=0, container_idx=0, place_pos=np.zeros(3, dtype=np.float32), orientation=0)


def _run(cfg, agent='agents/extreme_point', **env_vars):
    old = {k: os.environ.get(k) for k in env_vars}
    os.environ.update({k: str(v) for k, v in env_vars.items()})
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return run(cfg, agent, shake=False)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class EpisodeRuleTests(unittest.TestCase):
    def test_default_episode_ends_on_a_rejected_action(self):
        env = _env(small_config())
        try:
            _, _, terminated, _, _ = env.step(_dummy_action())
            self.assertTrue(terminated)
            self.assertEqual(env.n_discarded, 0)
        finally:
            env.close()

    def test_continue_after_reject_discards_the_bag_and_goes_on(self):
        cfg = small_config()
        cfg['episode'] = {'continue_after_reject': True}
        env = _env(cfg)
        try:
            before = len(env.stream_manager.all_items)
            _, _, terminated, _, info = env.step(_dummy_action())
            self.assertFalse(terminated)
            self.assertEqual(env.n_discarded, 1)
            self.assertEqual(info['status']['is_included'], False)
            self.assertEqual(len(env.stream_manager.all_items), before)   # the stream length is unchanged; the bag left the pool
        finally:
            env.close()


class ValidatorOptionTests(unittest.TestCase):
    def test_path_step_option_runs_and_default_is_unchanged(self):
        base = _run(small_config(n_items=4))
        cfg = small_config(n_items=4)
        cfg['validator']['path_step'] = 0.01          # the default value written out
        same = _run(cfg)
        self.assertEqual(base['fill'], same['fill'])
        cfg = small_config(n_items=4)
        cfg['validator']['path_step'] = 0.04
        coarse = _run(cfg)
        self.assertIn('fill', coarse)

    def test_engine_settings_are_applied(self):
        cfg = small_config(n_items=3)
        cfg['physics'] = {'time_step': 1 / 480, 'solver_iterations': 100}
        env = _env(cfg)
        try:
            params = env.client.getPhysicsEngineParameters()
            self.assertAlmostEqual(params['fixedTimeStep'], 1 / 480, places=6)
            self.assertEqual(params['numSolverIterations'], 100)
        finally:
            env.close()


class PlannerOptionTests(unittest.TestCase):
    def test_limited_stacking_variants_run(self):
        for variant in ('class_soft90', 'class_soft75'):
            result = _run(small_config(n_items=5), agent='agents/e1agent', E1_VARIANT=variant)
            self.assertIn('fill', result, variant)

    def test_iteration_budget_is_deterministic(self):
        cfg = small_config(n_items=6, optimize=True)
        a = _run(copy.deepcopy(cfg), EP_OPT_ITERS=3)
        b = _run(copy.deepcopy(cfg), EP_OPT_ITERS=3)
        self.assertEqual(a['fill'], b['fill'])
        self.assertEqual(a['opt_iterations'], b['opt_iterations'])
        self.assertLessEqual(a['opt_iterations'], 3)


if __name__ == '__main__':
    unittest.main()
