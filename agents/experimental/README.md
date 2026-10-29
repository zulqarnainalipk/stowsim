# `agents/experimental` – experimental agents

Variants of the reference agent that explore an idea but are not the default. They are kept for
reference and comparison.

| Directory | Idea | Status |
| --- | --- | --- |
| [`physics_verified/`](physics_verified) | Before committing a placement, replay the environment's settling test in a local PyBullet world and reject placements that would collapse | Works, but the extra simulation makes decisions slow (several seconds per call) |
