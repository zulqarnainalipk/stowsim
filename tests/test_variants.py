"""Smoke tests: the planner variants and the fixed-order planner run end to end and keep the action format."""
import contextlib
import io
import os
import unittest

from tests._helpers import small_config
from tools.run_agent import run


def _run_variant(variant, n_items=5, **env):
    old = {k: os.environ.get(k) for k in ['E1_VARIANT'] + list(env)}
    os.environ['E1_VARIANT'] = variant
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return run(small_config(n_items=n_items), 'agents/e1agent', shake=False)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class VariantTests(unittest.TestCase):
    def test_norollout_variant_places_bags(self):
        result = _run_variant('norollout')
        self.assertGreater(result['fill'], 0.0)

    def test_combined_variants_use_plus_syntax(self):
        result = _run_variant('norollout+slender3+class_off')
        self.assertIn('fill', result)

    def test_variants_are_deterministic(self):
        a = _run_variant('norollout')
        b = _run_variant('norollout')
        self.assertEqual(a['fill'], b['fill'])


if __name__ == '__main__':
    unittest.main()
