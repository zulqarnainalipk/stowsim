# `agents/` – agent implementations

Each sub-directory is a self-contained agent: a folder with an `agent.py` that defines the class `Agent`
(plus optional helper files and a `requirements.txt` for extra libraries).

| Directory | Description |
| --- | --- |
| [`base/`](base) | Minimal template: the required methods with placeholder logic. Copy it to start a new agent |
| [`extreme_point/`](extreme_point) | Reference agent: extreme-point packer, rollout-based bag selection, offline order search |
| [`e1agent/`](e1agent) | Reference agent with named rule and weight variants (ablations) and the known-items planner |
| [`poolcontrol/`](poolcontrol) | Reference agent with a settable number of rollouts (separates choosing from searching) |
| [`anytime/`](anytime) | Reference agent that uses a larger time budget for more rollouts |
| [`learned/`](learned) | Learned bag selector on the reference packer (numpy inference) |
| [`rawpose/`](rawpose) | Learned policy over bag, orientation and grid cell with an exact feasibility mask (needs PyTorch) |
| [`experimental/`](experimental) | Variants that did not become the default (kept for reference) |

The agents from `e1agent` to `rawpose` are the ones used in the accompanying paper; they load the reference agent's module
and change its settings through environment variables, described in [../docs/agents.md](../docs/agents.md).
`rawpose` needs `pip install torch` (CPU is enough for inference).

Run an agent with the official-style runner:

```bash
python -m scripts.run_test --module-path agents/extreme_point/
```

or with the development tools (see [../docs/tools.md](../docs/tools.md)). The agent interface is described in
[../docs/agents.md](../docs/agents.md).
