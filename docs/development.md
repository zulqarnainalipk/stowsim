# Development

## Setup

```bash
pip install -r requirements-dev.txt
```

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

The suite contains geometry unit tests for the reference agent and smoke tests that run the environment
end to end on a six-bag task (about 10 seconds). Run it after every change to `stowsim/` or an agent.

## Conventions

- Code, comments, docstrings and documentation are written in English.
- Public functions and classes carry docstrings; keep inline comments for the *why*.
- Units are SI (metres, kilograms, seconds); coordinates are Z-up.
- Tools are run as modules from the repository root (`python -m tools.run_agent ...`); the root is the
  working directory that makes `stowsim`, `agents` and `tools` importable.
- Generated output (`results/`, `images/`, logs, caches) is git-ignored and must not be committed.

## Extending the simulator

- **New bag types** – add them to the `item_list` of a configuration; see `configs/item_params.xlsx` for the
  parameters used by the sample tasks.
- **New metrics** – extend `stowsim/evaluator.py` (`Evaluator.evaluate`) and, for development-time metrics,
  `tools/run_agent.py`.
- **New validation rules** – `stowsim/validator.py`; remember that agents which re-implement the checks
  (such as `agents/extreme_point`) need the same change.
- **New agents** – copy `agents/base`, implement the three methods and run
  `python -m scripts.run_test --module-path agents/<name>/`.

## Troubleshooting

- *`pip install pybullet` fails on Windows:* install the MSVC Build Tools or use the Docker image.
- *Run is slow:* disable `"visualizer": {"vis": true}`; image saving dominates the run time.
- *`ModuleNotFoundError: stowsim`:* run commands from the repository root.
