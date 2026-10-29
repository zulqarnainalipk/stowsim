# Visualizer

`tools/visualizer` is a small web application for placing bags by hand against the real simulator,
inspecting layouts, and debugging agents. It needs only the Python standard library plus the simulator
dependencies; the page is a single HTML file without external assets.

```bash
python -m tools.visualizer.server --config configs/sample_config.json --task 001 --port 8765
# open http://localhost:8765
```

Options: `--agent` (agent used for suggestions and agent steps, default `agents/extreme_point`),
`--agent-order` (for offline tasks run the agent's order search instead of the natural order).

## Using the page

1. **Pick a bag** from the pool (click, or `[` / `]`).
2. **Choose how it lies and where.** Select one of the six orientations, then click in the *top view* to
   set x and y (the height snaps to the surface below when *auto-drop* is on), or use the sliders and
   arrow keys. A green outline means the agent's checker accepts the position; red means it is blocked and
   the reason is shown (outside the container, overlap, insertion path blocked, ...).
3. **Place it.** The action is sent to the real `GroundHandlingEnv.step`; the validator verdict is shown.

Other buttons: **Suggest** (numbered markers show the agent's best positions for the selected bag),
**Agent step** / **Auto-run agent**, **Undo** (replays the history on a fresh environment), **Restart**,
and **Copy moves** (JSON of the actions so far).

Keyboard: arrows move 1 cm (`Shift` 5 cm), `PgUp`/`PgDn` change the height, `R` rotates, `Enter` places,
`S` suggests, `A` agent step, `U` undo.

## Views

A 3-D isometric view and, per container, a top view (x right, y towards the back), a front view and a side
view. Red lines mark the cut-out corner, the small shelf and the shelf plate. Soft bags are blue,
priority bags green, other bags tan; the last placed bag is highlighted.

## HTTP API

The page talks to a small JSON API (useful for scripting):

| Endpoint | Description |
| --- | --- |
| `GET /api/state` | containers, packed bags, pool, last verdict, statistics, move history |
| `POST /api/step` | body `{item_idx, container_idx, place_pos, orientation}`; executes the action |
| `GET /api/check?item=&cont=&ori=&x=&y=[&z=]` | analytic validity check; without `z` the resting height is returned |
| `GET /api/suggest?item=` | agent's best candidate positions for a pool bag |
| `GET /api/agent_step` | let the agent move once |
| `GET /api/undo`, `GET /api/reset`, `GET /api/history` | history management |
