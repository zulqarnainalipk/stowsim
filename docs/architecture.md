# Architecture

This page explains how the modules of the `stowsim` package fit together and what happens during one
episode. For configuration keys and the agent interface see the [specification](specification.md).

## Components

```
scripts/run_test.py
        │  (module path, config)
        ▼
stowsim.app.EvaluationApp ──────────────►  results/evaluation_results.json
        │ creates per task
        ├──► stowsim.env.GroundHandlingEnv ── owns ──┬─ containers.MultiContainerManager / Container
        │            ▲                                ├─ items.ItemStreamManager / Item
        │            │ observation / action           ├─ validator.PlacementValidator
        │            │                                ├─ camera.Camera (depth map, optional RGB)
        │            │                                └─ evaluator.Evaluator
        └──► stowsim.runner.TimedAgentRunner ─ spawns ─► agent process (agents/<name>/agent.py)
                         ▲  pipe: get_init_states / optimize / policy (with time limits)
                         └─ shared memory: depth maps
```

| Module | Responsibility |
| --- | --- |
| `app.py` | `EvaluationApp`: loops over the tasks of a configuration, runs the episode, collects results |
| `env.py` | `GroundHandlingEnv`: Gymnasium-style environment (`reset`, `step`, `evaluate`), builds the world |
| `containers.py` | `Container` / `MultiContainerManager`: container geometry (cup mesh, lid, shelves), pre-loaded bags |
| `items.py` | `Item` / `ItemStreamManager`: bag physics properties, arrival order, visible pool and refilling |
| `validator.py` | `PlacementValidator`: action format, inclusion, insertion path, settling test |
| `camera.py` | `Camera`: orthographic depth map of each container (and RGB snapshots for debugging) |
| `evaluator.py` | `Evaluator`: fill score and placed fraction |
| `runner.py` | `TimedAgentRunner`: agent in a separate process, time limits, memory limit, fallbacks |
| `agent_factory.py` | Imports and instantiates the agent class inside the worker process |
| `utils.py` | Orientation table, mesh generation for the container, small geometry helpers |

## Episode flow

1. `EvaluationApp` builds the environment from the task configuration and starts the agent process.
2. `get_init_states` passes the container list, the pool size `lookahead_k` and the offline flag.
3. **Offline mode only:** `optimize(item_list)` returns a permutation of all bag indices; the stream is
   reordered accordingly. A timeout keeps the original order.
4. The stream fills the pool with the first `lookahead_k` bags and the first observation is produced.
5. Every step the agent receives an observation (`depth_map`, `pool_list`, `container_list`) and returns
   an action (`item_idx`, `container_idx`, `place_pos`, `orientation`). A `policy` timeout replaces the
   action with a random one.
6. `GroundHandlingEnv.step` validates the action:

   | Stage | Check | On failure |
   | --- | --- | --- |
   | format | keys, ranges, indices | episode ends |
   | inclusion | all bag corners inside the container planes (with `inclusion_margin`) | episode ends |
   | insertion path | the bag moves from the opening to the target (first along Y, then along X, `start_z` above the target); no packed bag or shelf within `safety_margin` | episode ends |
   | settling | the bag is placed at the target and the simulation runs `settle_wait_step` steps; displacement and rotation must stay below the thresholds | episode ends |

7. On success the bag is registered in its container, removed from the pool, and the pool is refilled.
8. The episode ends when the pool is empty or a check fails. `evaluate()` computes the fill score and
   the placed fraction of the final scene.

## Coordinates and geometry

All lengths are in metres; Z is up. Each container has its own origin `(offset_x, 0, 0)` with
`offset_x = i * spacing`; `place_pos` is relative to that origin and addresses the **centre** of the
bag. The opening is on the negative-Y side; a small shelf above the cut-out corner is always present and
an optional large shelf splits the rear half of the container.

## Observations

* `depth_map` – array of shape `(containers, H, W)` written into shared memory by `Camera`; each pixel is
  the distance from the opening plane to the first surface (small = bag close to the opening).
* `pool_list` – the bags the agent may choose from (dimensions, mass, flags, friction parameters).
* `container_list` – container geometry plus `packed_items` with their exact poses (`pos`, `orn`).

Agents that need exact geometry usually ignore the depth map and rebuild the scene from
`packed_items` (the reference agent does this).

## Process isolation

`TimedAgentRunner` runs the agent in a `spawn`ed process, so agent crashes, long computations and
memory growth cannot affect the environment. Calls are made over a pipe with a timeout. On Linux the
address space of the worker is limited to `max_mem` GB. Time limits default to 10 s (init), 8 s (policy)
and 180 s (optimize) and can be changed in the `agent` section of the configuration.
