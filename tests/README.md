# `tests/` – automated tests

```bash
python -m unittest discover -s tests -t . -v
```

| File | Covers |
| --- | --- |
| `test_geometry.py` | Orientation table and quaternion helpers of the reference agent agree with the simulator (no physics) |
| `test_environment.py` | Environment reset and invalid-action handling; the agent template and the reference agent run end to end on a six-bag task; the reference agent returns well-formed actions within the time limit |
| `test_variants.py` | The named planner variants of `agents/e1agent` run end to end, variants combine with `+`, and a variant gives the same result twice (determinism) |
| `test_options.py` | The optional settings are off by default and change nothing then: a rejected action ends the episode, `continue_after_reject` discards the bag and goes on, `path_step` and the `physics` section are applied, the limited-stacking variants (class rule 2) run, and the iteration budget `EP_OPT_ITERS` gives the same plan twice |
| `_helpers.py` | Builds a small task from the sample configuration |

The whole suite (17 tests) takes about a minute. `python -m pytest -q` runs the same tests if pytest is installed.
