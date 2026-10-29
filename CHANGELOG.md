# Changelog

## 0.1.0 (2026-10-30)

First public release.

- Simulator: `GroundHandlingEnv` with physics-validated placements (inclusion margin, insertion path with start height and clearance, settling test), online pool of `k` visible bags, offline ordering, soft and priority bags, shelf, several containers.
- Reference agent `agents/extreme_point` (extreme-point packer, rollouts, offline order search) and the agent template `agents/base`.
- Research agents used in the accompanying paper: `e1agent` (named rule and weight variants, known-items planner), `poolcontrol`, `anytime`, `learned` (bag selector, numpy inference), `rawpose` (policy over bag, orientation and grid cell with an exact feasibility mask, needs PyTorch).
- Switches for the experiments: `E1_VARIANT` (variants combine with `+`), `EP_FIXED_ORDER`/`EP_FIXED_HORIZON`/`EP_FIXED_OPTIONS`, `EP_CONSTANTS`, `LEARNED_MODE`, `LEARNED_WEIGHTS_FILE`.
- Tools: in-process runner with metrics and shaking test, physics-free fast simulator, batch evaluator, browser visualiser.
- Tests: 11 smoke and unit tests (`python -m unittest discover -s tests -t .`).
