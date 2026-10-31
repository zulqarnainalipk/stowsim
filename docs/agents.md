# Agent guide

An agent decides where every bag goes. This page describes the interface, the provided agents and the
algorithm of the reference agent. The full field reference is in the [specification](specification.md).

## Interface

An agent is a class named `Agent` in a file named `agent.py`:

```python
class Agent:
    def __init__(self, module_path: str): ...                  # setup only, no heavy work
    def get_init_states(self, init_states: dict): ...          # containers, lookahead_k, offline flag
    def optimize(self, item_list: list) -> list[int]: ...      # offline mode: new order of all bag indices
    def policy(self, observation: dict) -> dict: ...           # one action per step
```

`policy` returns `{'item_idx', 'container_idx', 'place_pos', 'orientation'}`:

| Key | Meaning |
| --- | --- |
| `item_idx` | position of the chosen bag in `pool_list` (the pool is shorter than `lookahead_k` near the end) |
| `container_idx` | index into `container_list` |
| `place_pos` | `np.float32[3]`, centre of the bag relative to the container origin `(offset_x, 0, 0)` |
| `orientation` | 0-5, see the orientation table in the specification |

Rules of the game worth remembering when designing an agent:

- A rejected action ends the episode; the scene is scored as it is.
- The bag is pushed in from the opening, so anything in front of the target (at the height of the
  lifted bag) blocks it. Fill back to front.
- Soft bags should not carry other bags; priority bags belong in priority containers.
- Use the pool: the best bag to place now is often not the first one.
- Respect the time limits (defaults: 10 s init, 8 s per policy call, 180 s optimize).

## Provided agents

| Directory | Description |
| --- | --- |
| [`agents/base`](../agents/base) | Template with the required methods and a trivial (mostly invalid) placement; the starting point for new agents |
| [`agents/extreme_point`](../agents/extreme_point) | Reference agent: geometric extreme-point packer with rollouts and an offline order search |
| [`agents/experimental/physics_verified`](../agents/experimental/physics_verified) | The reference agent plus a local PyBullet settling check before every placement (slower, experimental) |
| [`agents/e1agent`](../agents/e1agent) | The reference agent with named variants of its rules and weights (`E1_VARIANT`, combine with `+`, e.g. `norollout+slender3`); also the known-items planner (`EP_FIXED_ORDER`, see below) |
| [`agents/poolcontrol`](../agents/poolcontrol) | Reference agent with a settable number of rollout-evaluated first moves (`POOL_ROLL_N`; 0 = choose among the pool without rollouts) |
| [`agents/anytime`](../agents/anytime) | Reference agent that spends a larger per-decision time budget on more rollouts |
| [`agents/learned`](../agents/learned) | Learned bag selector on top of the reference packer (numpy inference; weights from `LEARNED_WEIGHTS_FILE`) |
| [`agents/rawpose`](../agents/rawpose) | Learned policy that chooses bag, orientation and grid cell itself, restricted by an exact feasibility mask (needs PyTorch; weights from `LEARNED_WEIGHTS_FILE`) |

### Variants and switches used in the paper experiments

| Switch | Effect |
| --- | --- |
| `E1_VARIANT` | `ref`, `norollout`, `support50`, `support30`, `support90`, `slender3`, `softstack4`, `softstack1`, `heavy_first`, `nosoftpen`, `ignore_support_cost`, `class_off`, `class_soft90`, `class_soft75`; combine with `+` |
| class rule (`W.class_rule`, `W.restricted_support`) | `1` (default): a standard bag may not rest on a soft or priority bag in the strict tier; `0`: allowed (variant `class_off`); `2`: limited stacking, allowed when the standard bag has a support share of at least `restricted_support` (0.9 in `class_soft90`, 0.75 in `class_soft75`) |
| `EP_OPT_ITERS` | Offline order search: stop after this many search iterations instead of after the time budget `EP_OPT_BUDGET`. The result then does not depend on the machine speed (unset or 0: time budget). The number of iterations used is reported as `opt_iterations` by `tools.run_agent` |
| `EP_FIXED_ORDER`, `EP_FIXED_HORIZON`, `EP_FIXED_OPTIONS` | Known-items regime: the whole list is known and the arrival order is kept; each bag compares its best placement and a few alternatives by a greedy continuation over the next `EP_FIXED_HORIZON` known bags |
| `EP_CONSTANTS` | JSON of module constants; the insertion-path constants `START_Z`, `ENV_SAFETY`, `PATH_M`, `SIDE` must be changed together with the validator's `start_z` and `safety_margin` (the experiment runner does this) |
| `LEARNED_MODE` | `greedy` or `sample` (sampling records a trajectory for training) |

## Reference agent (`agents/extreme_point`)

**Scene model.** The agent ignores the depth map. It rebuilds each container from the observation: the
inner planes (for the exact inclusion test), the shelves, and every packed bag as an axis-aligned box
(the box of the rotated cuboid, so slightly tilted bags are handled conservatively).

**Candidate generation.** For each pool bag and each distinct orientation, candidate positions are
*extreme points*: the bag is pushed flush against walls and neighbours (with small safety gaps) in x and
y, and dropped to the highest surface below its footprint. Two stacking levels are considered: the
container floor and the shelf plate. Candidates must satisfy

- the exact inclusion test of the environment,
- a minimum support fraction of the footprint,
- soft/priority rules (no standard bag on a soft or priority bag, limited stacks of soft bags),
- the ceiling clearance rule, and
- a re-implementation of the insertion-path test (lift height, ceiling clipping, X-then-Y sweep against
  all bags and shelves with the safety margin).

**Ranking.** Valid candidates are ranked by a weighted cost: far from the opening first, then low, then
left, with a reward for contact with walls and neighbours and penalties for weak support, tall
orientations, resting on soft bags, and standard bags in priority containers (all weights in class `W`).

**Online mode.** The agent evaluates the best placement of each distinct pool bag (and a few
alternatives) by a short greedy rollout over the rest of the pool, and plays the first move of the best
rollout. If no candidate passes the strict rules, relaxed rules and finally last-resort tiers are tried so
that a bag is placed whenever any valid position exists. A deadline keeps every call well below the time
limit.

**Offline mode.** `optimize` runs a randomised local search over the loading order. Each order is
evaluated by sequentially packing the bags with the same packer (physics-free); bags that do not fit are
moved to the end, which is exactly what the environment would see. Prefix snapshots make each evaluation
re-simulate only the changed suffix. The objective is placed volume plus a small bonus per placed bag.
The best plan (order and positions) is replayed by `policy`, which re-validates every planned position
against the actual poses and falls back to the greedy packer if needed.

**Tuning.** Weights and constants can be overridden without editing the file through the environment
variables `EP_WEIGHTS` (JSON for class `W`), `EP_CONSTANTS` (JSON for module constants),
`EP_OPT_BUDGET` (seconds for `optimize`) and `EP_CPU_CLOCK` (use CPU time for deadlines when
benchmarking on a loaded machine).

## Evaluating an agent

```bash
python -m scripts.run_test --module-path agents/<name>/     # official-style run with process isolation
python -m tools.run_agent --agent agents/<name> --task 001  # in-process run with quality metrics
python -m tools.fast_sim --agent agents/<name> --n 16       # fast physics-free benchmark (extreme-point agents)
python -m tools.batch_eval --agent agents/<name> --n 8      # many random tasks in real physics
```

See the [tools guide](tools.md) for details.
