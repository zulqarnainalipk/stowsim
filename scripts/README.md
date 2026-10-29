# `scripts/` – command-line entry points

| Script | Description |
| --- | --- |
| `run_test.py` | Evaluates an agent on every task of a configuration with process isolation and time limits; writes `results/evaluation_results.json` |

```bash
python -m scripts.run_test [--config-path configs/sample_config.json] [--module-path agents/base/]
                           [--result-dir results/] [--result-fname evaluation_results.json]
                           [--render-mode human] [--verbose 1]
```

Run it as a module from the repository root. `--module-path` must end with `/` and point to a directory
that contains `agent.py`. The result file contains, per task, the evaluation (`fill_score`,
`num_placed_items`), the validation state of the last step (`place_states`), the maximum call times
(`time_results`) and a `status` message.
