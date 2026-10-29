# `stowsim/` – the simulator package

The core of StowSim. Import it from the repository root (`from stowsim.env import GroundHandlingEnv`).
The component diagram and the episode flow are in [../docs/architecture.md](../docs/architecture.md); the
configuration keys are in [../docs/specification.md](../docs/specification.md).

| File | Main classes / functions | Role |
| --- | --- | --- |
| `__init__.py` | `Agent` | Abstract base class for agents (`get_init_states`, `optimize`, `policy`) |
| `env.py` | `GroundHandlingEnv` | Gymnasium-style environment: `reset_settings`, `reset_item_stream`, `reset`, `step`, `evaluate`, `get_init_states`, `get_info_for_optimization`, `set_item_order` |
| `containers.py` | `Container`, `MultiContainerManager` | Builds the container meshes, lid and shelves in PyBullet, keeps the packed bags, exports the container information for observations |
| `items.py` | `Item`, `ItemStreamManager` | Bag properties and PyBullet bodies; arrival order, visible pool (`look_ahead`), refilling (`max_space`) |
| `validator.py` | `BaseValidator`, `PlacementValidator` | `check_action`, `check_inclusion`, `check_transport_path`, `place_item` (settling test) |
| `camera.py` | `Camera` | `get_heightmap` (orthographic depth map), `get_rgb_image` (debug snapshots) |
| `evaluator.py` | `Evaluator` | `calculate_fill_rate`, `evaluate` (fill score and placed fraction) |
| `runner.py` | `TimedAgentRunner` | Agent worker process, `call` with timeout and fallback, memory limit |
| `agent_factory.py` | `AgentFactory` | Imports the agent module inside the worker |
| `app.py` | `EvaluationApp` | Runs all tasks of a configuration and writes the result file |
| `utils.py` | `ORNS`, `get_half_ext`, mesh writers | Orientation table and container mesh generation |

## Design notes

- The environment is deterministic given the configuration, the agent's actions and the reset seed.
- Agents run in a separate process; observations carry the depth maps through shared memory
  (`shm_name`, `shm_shape`, `shm_dtype` are replaced by `depth_map` in the worker, see `runner.py`).
- Validation and evaluation parameters live in the `validator` section of the task configuration, so
  experiments can change margins without touching code.
