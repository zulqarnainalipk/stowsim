# `agents/base` – agent template

`agent.py` implements the three required methods with placeholder logic:

| Method | Behaviour in the template |
| --- | --- |
| `__init__(module_path)` | Does nothing; place model loading here if needed |
| `get_init_states(init_states)` | Stores the number of containers |
| `optimize(item_list)` | Returns the bag indices sorted by volume (largest first) |
| `policy(observation)` | Returns a fixed action; replace it with your decision logic |

The placeholder action is usually rejected by the validator, so running the template ends the episode
after the first step. It exists to show the interface and the required return types
(`place_pos` must be a `numpy.float32` array of length 3).
