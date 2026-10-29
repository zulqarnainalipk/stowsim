# Tools guide

The `tools/` package contains development utilities. All commands are run from the repository root
with `python -m tools.<name>`.

| Tool | Purpose | Speed |
| --- | --- | --- |
| `tools.run_agent` | Run one agent in-process on one or all tasks of a configuration; prints fill, placed fraction, centre-of-gravity height, rule violations and a shaking test | seconds to minutes |
| `tools.fast_sim` | Physics-free simulation of the extreme-point agent on many random or shuffled scenarios; for tuning and A/B comparison | very fast |
| `tools.batch_eval` | Real-physics evaluation of an agent on many random scenarios in parallel; reports early terminations | slow |
| `tools.visualizer.server` | Browser-based manual stowing and agent debugging (see [visualizer.md](visualizer.md)) | interactive |

## `tools.run_agent`

```bash
python -m tools.run_agent --agent agents/extreme_point --config configs/sample_config.json --task 001 [--noshake] [--verbose]
```

Output (JSON): `fill`, `placed_frac`, `status` of the last step, `policy_max_s`, `optimize_s`,
`metrics` (`cog_mean_height_frac`, `soft_covered`, `prio_covered`, `prio_in_wrong_container`, `n_items`),
`shake` (displacement statistics) and `scores` (fill, cog, stability, placement on a 0-100 scale).
`fill` is computed with a looser inclusion margin than the validator uses, so bags resting on the floor
or touching a wall are counted.

Python API: `run(task_cfg, agent_path, shake=True, verbose=False)` and `load_agent(path)`.

## `tools.fast_sim`

```bash
python -m tools.fast_sim --n 16 --seed 5 --online --k 10      # online, pool of 10, single container
python -m tools.fast_sim --n 8  --seed 5 --offline            # offline: lookahead 1 + order search
python -m tools.fast_sim --like 001 --n 10 --seed 3           # shuffled copies of sample task 001
```

Every placement is assumed to settle exactly where it was planned. The last line summarises the mean fill,
placed fraction (`frac`), CoG height (`cog`), rule violations (`viol`), priority bags in the wrong
container (`pw`), the number of scenarios below 60 % / 65 % placed (`lt60`, `lt65`) and the slowest decision
(`maxdt`). `OPT_BUDGET` (seconds) controls the offline search. The tool depends on the internals of
`agents/extreme_point`.

## `tools.batch_eval`

```bash
python -m tools.batch_eval --n 8 --seed 3 --procs 4 [--shake]
```

Runs random scenarios (from `tools.fast_sim.make_cfg`) plus the sample tasks through the real
environment and prints one line per scenario. Lines starting with `log:` show collision or
unsafe-placement messages, which indicate an early termination.

## Scenario generator

`tools.fast_sim.make_cfg(rng, ncont=None)` creates a random task: 1-2 containers with random sizes, optional
shelf, a priority container when there are two, 42-57 bags per container drawn from the seven bag types
of `configs/item_params.xlsx`, and a random pool size (1, 5 or 10). Use it to build test sets that do not
depend on the sample configuration.
