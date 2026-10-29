# `agents/experimental/physics_verified`

The [reference agent](../../extreme_point) plus a **local physics check**: every placement chosen by the
packer is rehearsed in a private PyBullet world (same container meshes, shelves, friction and soft-contact
parameters). If the bag would move more than 12 cm, rotate more than 20 degrees, or push a neighbour more
than 6 cm, the placement is banned and the next best is chosen.

## Why it exists

Relaxed packing rules (low support, standing bags) place more bags but occasionally collapse a stack,
which ends the episode. The local check removes those collapses.

## Trade-off

Each rehearsal simulates up to 300 steps with the neighbouring bags, which costs 0.2-2.5 s. With several
candidates per decision the policy time approaches the 8 s limit on slow machines, so the decision quality
suffers. Use it as a research baseline, not as a default.

## Files

| File | Contents |
| --- | --- |
| `agent.py` | The reference agent code with the embedded `PhysWorld` class (container rebuild, `sync`, `test`) and the verification loop in `Agent._policy` |

Switches: class `W` fields `phys` (enable), `phys_tries` (maximum bans per decision) and `phys_limit`
(seconds after which verification stops).
