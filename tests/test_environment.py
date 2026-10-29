"""Smoke tests: the environment runs end to end with the base agent and the reference agent."""
import contextlib
import io
import unittest

import numpy as np

from tests._helpers import small_config
from stowsim.env import GroundHandlingEnv
from tools.run_agent import load_agent, run


def _make_env(cfg):
    env = GroundHandlingEnv(config=cfg, verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):
        env.reset_settings()
    return env


class EnvironmentTests(unittest.TestCase):
    def test_reset_exposes_pool_and_containers(self):
        env = _make_env(small_config())
        try:
            init = env.get_init_states()
            self.assertEqual(init['lookahead_k'], 3)
            self.assertEqual(len(init['container_list']), 1)
            env.reset_item_stream()
            obs, _ = env.reset(seed=0)
            self.assertEqual(len(obs['pool_list']), 3)
            self.assertEqual(obs['shm_shape'][0], 1)
        finally:
            env.close()

    def test_invalid_action_ends_episode(self):
        env = _make_env(small_config())
        try:
            env.reset_item_stream()
            env.reset(seed=0)
            bad = dict(item_idx=99, container_idx=0, place_pos=np.zeros(3, dtype=np.float32), orientation=0)
            _, _, terminated, truncated, _ = env.step(bad)
            self.assertTrue(terminated or truncated)
        finally:
            env.close()


class AgentTests(unittest.TestCase):
    def test_base_agent_template_runs(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = run(small_config(n_items=3), 'agents/base', shake=False)
        self.assertIn('fill', result)

    def test_reference_agent_places_bags(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = run(small_config(n_items=6), 'agents/extreme_point', shake=False)
        self.assertGreaterEqual(result['placed_frac'], 0.5)
        self.assertGreater(result['fill'], 0.0)
        self.assertLess(result['policy_max_s'], 8.0)

    def test_reference_agent_action_format(self):
        cfg = small_config()
        env = _make_env(cfg)
        try:
            agent = load_agent('agents/extreme_point')
            agent.get_init_states(env.get_init_states())
            env.reset_item_stream()
            obs, _ = env.reset(seed=0)
            obs = dict(obs)
            for key in ('shm_name', 'shm_shape', 'shm_dtype'):
                obs.pop(key, None)
            obs['depth_map'] = env.shm_depth_map.copy()
            action = agent.policy(obs)
            self.assertEqual(set(action), {'item_idx', 'container_idx', 'place_pos', 'orientation'})
            self.assertEqual(action['place_pos'].dtype, np.float32)
            self.assertIn(action['orientation'], range(6))
            self.assertLess(action['item_idx'], len(obs['pool_list']))
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
