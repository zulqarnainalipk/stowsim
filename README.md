# StowSim

**A physics-based simulation benchmark for online and offline 3-D baggage stowage.**

StowSim simulates the loading of cuboid bags (suitcases, backpacks, cardboard boxes, ...) into
side-opening air-cargo containers. Every placement is checked and executed in a rigid-body physics
engine (PyBullet): the bag must lie inside the container, must be pushed in along a collision-free
path from the opening, and must settle without collapsing. The loading is scored by volumetric fill
(the other quality terms are measured by the tools in this repository).

The simulator exposes a Gymnasium-style environment, a small agent interface, and a set of tools for
developing, benchmarking and visualising stowage policies.

## Research questions

StowSim is designed to study questions such as:

1. **Online stowage with a limited look-ahead.** How much does a buffer of `k` visible bags help when
   bags must be committed one by one and cannot be rearranged?
2. **Offline sequencing.** When the whole bag sequence is known in advance, how much can an optimised
   loading order improve the packing density compared with the arrival order?
3. **Geometry under access constraints.** Bags enter through a single opening, so the insertion path
   (not only the final position) must be feasible. How does this change classical 3-D bin packing?
4. **Stability and attribute constraints.** Soft bags must not be crushed, priority bags belong in
   priority containers, and the load must survive handling. What is the price of these constraints
   in volumetric efficiency?

## Features

- Gymnasium-compatible environment with depth-map observations and exact bag poses.
- Physics-validated actions: inclusion test, insertion-path test (including shelves and ceiling
  clearance), and a settling test after placement.
- Online mode (pool of `k` visible bags) and offline mode (the agent reorders the whole sequence).
- Multiple containers, optional shelf, priority containers, soft and priority bags, pre-loaded bags.
- Process isolation with time and memory limits for fair agent evaluation.
- A strong reference agent (`agents/extreme_point`) and an agent template (`agents/base`).
- Tools: in-process runner with quality metrics, a fast physics-free simulator for tuning, a real-physics
  batch evaluator, and an interactive browser visualizer for manual stowing.

## Installation

Python 3.10 - 3.12 is required.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

PyBullet ships no pre-built wheel for every platform. On Windows a C++ compiler (MSVC Build Tools)
is needed, or use the Docker image: `docker compose -f docker/docker-compose.yml up -d`
(see [docker/README.md](docker/README.md)).

## Quick start

```bash
# 1. run the agent template on the sample tasks (results/evaluation_results.json)
python -m scripts.run_test --module-path agents/base/

# 2. run the reference agent
python -m scripts.run_test --module-path agents/extreme_point/ --verbose 1

# 3. run one task in-process and print quality metrics (CoG, rule violations, shaking test)
python -m tools.run_agent --agent agents/extreme_point --task 001

# 4. place bags yourself in the browser
python -m tools.visualizer.server --task 001          # then open http://localhost:8765

# 5. run the tests
python -m unittest discover -s tests -t . -v
```

## Writing an agent

An agent is a class `Agent` in `agent.py` with three methods: `get_init_states`, `optimize` (offline
mode only) and `policy`. `policy` returns which pool bag to place, in which container, at which
position and in which of the six axis-aligned orientations. Start from [agents/base](agents/base) and
read the [agent guide](docs/agents.md) and the [specification](docs/specification.md).

## Repository layout

| Path | Purpose |
| --- | --- |
| [`stowsim/`](stowsim) | The simulator package: environment, validator, camera, evaluator, process runner |
| [`agents/`](agents) | Agent implementations: template, reference agent, experimental agents |
| [`configs/`](configs) | Task configurations and the bag-type parameter table |
| [`scripts/`](scripts) | Command-line entry point that evaluates an agent on a configuration |
| [`tools/`](tools) | Development tools: runner with metrics, fast simulation, batch evaluation, visualizer |
| [`tests/`](tests) | Unit and smoke tests |
| [`docker/`](docker) | Dockerfile and compose file of the runtime environment |
| [`docs/`](docs) | Specification, architecture, agent guide and tools guide |

## Documentation

- [Specification](docs/specification.md): configuration, coordinate system, modules, agent interface, metrics
- [Architecture](docs/architecture.md): how the modules fit together and what happens in one step
- [Agent guide](docs/agents.md): the agent interface, the template, and the reference agent's algorithm
- [Tools guide](docs/tools.md): runner, fast simulator, batch evaluator
- [Visualizer](docs/visualizer.md): the interactive manual-stowing tool
- [Development](docs/development.md): tests, conventions, how to extend the simulator

## Evaluation

Each task ends when all bags are loaded or when a placement is rejected by the validator. The
evaluator reports the volumetric **fill score** (volume of bags lying completely inside the usable
container volume) and the fraction of bags placed. `tools/run_agent.py` additionally computes the
mass-weighted centre-of-gravity height, soft/priority rule violations and a shaking test.

## Author

Zulqarnain Ali, Yeungnam University, Daegu, Republic of Korea. ORCID [0009-0009-0281-882X](https://orcid.org/0009-0009-0281-882X), zulqar445ali@gmail.com. The experiments and results that use this simulator are in the companion repository `stowsim-benchmark-study`.

## Citing

If you use StowSim in your research, please cite it. GitHub shows a "Cite this repository" button
generated from [`CITATION.cff`](CITATION.cff); the BibTeX entry is

```bibtex
@software{stowsim,
  title   = {{StowSim}: A Physics-Based Simulation Benchmark for Online and Offline 3-D Baggage Stowage},
  author  = {Ali, Zulqarnain},
  year    = {2026},
  version = {0.1.0},
  note    = {Software}
}
```

## License

StowSim is released under the [MIT License](LICENSE). The license requires that the copyright notice and
the permission notice are kept in all copies and substantial portions of the software, which preserves
attribution. A software license cannot legally require academic citation, so the citation request above
is a matter of good scholarly practice rather than a license condition; please follow it.
