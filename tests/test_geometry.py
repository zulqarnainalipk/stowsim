"""Unit tests for the geometry helpers of the reference agent (no physics needed)."""
import math
import unittest

import numpy as np

from tests import _helpers  # noqa: F401  (sets up sys.path)
from agents.extreme_point.agent import oriented_dims, quat_to_mat
from stowsim.utils import get_half_ext


class OrientationTests(unittest.TestCase):
    def test_oriented_dims_matches_environment(self):
        l, w, h = 0.65, 0.45, 0.25
        for o in range(6):
            env_dims = [2 * v for v in get_half_ext([l, w, h], o)]
            self.assertEqual(sorted(env_dims), sorted(oriented_dims(l, w, h, o)))
            self.assertEqual(list(oriented_dims(l, w, h, o)), env_dims)

    def test_identity_quaternion(self):
        np.testing.assert_allclose(quat_to_mat((0, 0, 0, 1)), np.eye(3), atol=1e-12)

    def test_quarter_turn_about_z(self):
        s = math.sqrt(0.5)
        r = quat_to_mat((0, 0, s, s))
        np.testing.assert_allclose(r @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
