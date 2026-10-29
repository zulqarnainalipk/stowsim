# `configs/` – task configurations

| File | Contents |
| --- | --- |
| `sample_config.json` | Two example tasks: `"000"` offline task (pool of 1 bag, container without shelf, the agent reorders the sequence) and `"001"` online task (pool of 10 bags, container with shelf) |
| `item_params.xlsx` | Table of the seven standard bag types (suitcases, duffel, cardboard box, backpacks) with dimensions, mass and physics parameters |

A configuration file maps task IDs to task settings; see the *Specification* for every key
([../docs/specification.md](../docs/specification.md)). Main sections of a task:

`containers`, `item_stream`, `validator`, `camera`, `action`, `agent`, `visualizer`.

## Creating your own task

1. Copy a task from `sample_config.json` under a new ID.
2. Edit the containers (`length`, `width`, `height`, `cut_x`, `cut_y`, `require_shelf`, `is_prioritized`).
3. Provide the `item_list` (use the bag parameters from `item_params.xlsx`) and the pool settings
   `look_ahead` / `max_space`.
4. Set `agent.optimize` to `true` for offline mode (the agent may reorder the sequence).
5. Run it: `python -m scripts.run_test --config-path my_config.json --module-path agents/extreme_point/`.

`tools.fast_sim.make_cfg` shows how to generate random tasks programmatically.
