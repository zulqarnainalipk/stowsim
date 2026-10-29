# `agents/extreme_point` – reference agent

A geometric packer that rebuilds the scene from the observation, enumerates extreme-point placements,
filters them with a replica of the environment's checks, and chooses bags with short rollouts. In offline
mode it first searches for a good loading order. The algorithm is described in
[../../docs/agents.md](../../docs/agents.md); the module docstring of `agent.py` lists the tuning variables.

## Files

| File | Contents |
| --- | --- |
| `agent.py` | `CM` (container model), `Boxes` (placed bags), `W` (weights), `Packer` (candidate generation, path test, rollouts), `Agent` |

## Main building blocks in `agent.py`

| Name | Role |
| --- | --- |
| `CM` | Container model: inner planes, shelves, exact inclusion test |
| `Boxes` / `boxes_from_container` | Axis-aligned boxes of the packed bags (with soft/priority flags, shelf level, soft-stack depth) |
| `Packer.candidates` | Vectorised extreme-point candidates for one bag and orientation, with support, class and ceiling rules and the path test |
| `Packer.path_ok` / `lift_and_z` | Re-implementation of the environment's insertion-path test |
| `Packer.best_for_bag` | Best candidate over orientations and containers |
| `Packer.choose_rollout` / `choose2` | Bag selection by greedy rollouts / by cost, with relaxed tiers |
| `Agent.optimize` | Randomised local search over the loading order with prefix snapshots |
| `Agent.policy` | Online decision; in offline mode replays and re-validates the plan |

## Usage

```bash
python -m scripts.run_test --module-path agents/extreme_point/
EP_OPT_BUDGET=30 python -m tools.run_agent --agent agents/extreme_point --task 000
```

On Windows PowerShell set variables with `$env:EP_OPT_BUDGET = 30`. The offline search uses 140 s of the
180 s limit by default.

## Notes

- No network access and no hard-coded container or bag sizes; the agent generalises to other containers,
  bag sets and arrival orders.
- Typical decision time is 1-3 s per call (limit 8 s); the agent has internal deadlines.
