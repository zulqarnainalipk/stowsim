# `tools/` – development tools

Run every tool as a module from the repository root. Details and example output are in
[../docs/tools.md](../docs/tools.md).

| Module | Description |
| --- | --- |
| `run_agent.py` | Run an agent in-process on one task; prints fill, placed fraction, CoG height, rule violations, shaking test |
| `fast_sim.py` | Physics-free benchmark of the extreme-point agent on many random or shuffled scenarios; also contains the random scenario generator `make_cfg` |
| `batch_eval.py` | Real-physics evaluation of an agent on many random scenarios in parallel |
| `visualizer/` | Browser tool for manual stowing and agent debugging (`server.py` backend, `index.html` page) |

```bash
python -m tools.run_agent --agent agents/extreme_point --task 001
python -m tools.fast_sim --like 001 --n 10 --seed 3
python -m tools.batch_eval --n 8 --seed 3
python -m tools.visualizer.server --task 001
```
