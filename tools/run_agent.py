"""Run one agent in-process on a configuration and report fill, placement and stability metrics.

This is a development convenience: it drives `GroundHandlingEnv` directly (no subprocess, no
timeouts), so it is easy to debug and profile an agent. In addition to the evaluator's fill score
it reports a few extra metrics:

* ``cog_mean_height_frac``  mass-weighted mean height of the load divided by the container height
* ``soft_covered`` / ``prio_covered``  bags that have a standard bag directly on top of them
* ``prio_in_wrong_container``  priority bags in a non-priority container
* ``shake``  displacement of the bags under a crude shaking test

Usage (from the repository root)::

    python -m tools.run_agent --agent agents/extreme_point --config configs/sample_config.json [--task 001]
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())
from stowsim.env import GroundHandlingEnv  # noqa: E402


def load_agent(path):
    """Import ``<path>/agent.py`` as module ``agent_mod`` and return an ``Agent`` instance."""
    spec = importlib.util.spec_from_file_location('agent_mod', os.path.join(path, 'agent.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['agent_mod'] = mod
    sys.path.insert(0, path)
    spec.loader.exec_module(mod)
    return mod.Agent(path)


def quality_metrics(env):
    """Compute centre-of-gravity and placement-rule metrics from the final scene.

    Returns ``(metrics_dict, items)`` where ``items`` is a list of ``(container, item, pos, orn)``.
    """
    containers = env.container_manager.containers
    out = {}
    total_mass = 0.0
    z_weighted = 0.0
    n = 0
    covered_soft = 0
    covered_prio = 0
    prio_wrong = 0
    items = []
    for c in containers:
        for it in c.packed_items:
            pos, orn = it.get_pose(env.client)
            items.append((c, it, pos, orn))
    for c, it, pos, orn in items:
        total_mass += it.mass
        z_weighted += it.mass * (pos[2] - c.buffer) / c.height
        n += 1
        if it.is_prioritized and any(cc.is_prioritized for cc in containers) and not c.is_prioritized:
            prio_wrong += 1

    def aabb(it, pos, orn):
        rot = np.abs(np.array(env.client.getMatrixFromQuaternion(orn)).reshape(3, 3))
        half = rot @ np.array([it.length / 2, it.width / 2, it.height / 2])
        return np.array(pos) - half, np.array(pos) + half

    boxes = [aabb(it, pos, orn) for c, it, pos, orn in items]
    for a, (c, it, pos, orn) in enumerate(items):
        lo_a, hi_a = boxes[a]
        soft_hit = prio_hit = False
        for b, (c2, it2, _, _) in enumerate(items):
            if a == b or c2 is not c:
                continue
            lo_b, hi_b = boxes[b]
            ox = min(hi_a[0], hi_b[0]) - max(lo_a[0], lo_b[0])
            oy = min(hi_a[1], hi_b[1]) - max(lo_a[1], lo_b[1])
            if ox > 0.005 and oy > 0.005 and lo_b[2] >= hi_a[2] - 0.02 and lo_b[2] - hi_a[2] < 0.03:
                soft_hit |= it.is_soft and not it2.is_soft
                prio_hit |= it.is_prioritized and not it2.is_prioritized
        covered_soft += soft_hit
        covered_prio += prio_hit
    out['cog_mean_height_frac'] = (z_weighted / total_mass) if total_mass else None
    out['soft_covered'] = covered_soft
    out['prio_covered'] = covered_prio
    out['prio_in_wrong_container'] = prio_wrong
    out['n_items'] = n
    return out, items


def shake_test(env, items, amp=0.15, cycles=2, steps=240):
    """Crude stability test: oscillate the horizontal gravity components and measure bag displacement."""
    if not items:
        return None
    start = [np.array(pos) for _, _, pos, _ in items]
    client = env.client
    g = 9.8
    for _ in range(cycles):
        for axis in (0, 1):
            for s in range(steps):
                gv = [0, 0, -g]
                gv[axis] = amp * g * math.sin(2 * math.pi * s / steps)
                client.setGravity(*gv)
                client.stepSimulation()
    client.setGravity(0, 0, -g)
    for _ in range(120):
        client.stepSimulation()
    disp = np.array([np.linalg.norm(np.array(it.get_pose(client)[0]) - q) for (c, it, pos, orn), q in zip(items, start)])
    return dict(mean_disp=float(disp.mean()), max_disp=float(disp.max()), n_moved_gt_5cm=int((disp > 0.05).sum()))


SUITE_PROTOCOLS = [('tilt15', 'tilt', 0.15), ('pulse20', 'pulse', 0.20), ('pulse30', 'pulse', 0.30), ('pulse45', 'pulse', 0.45)]
SUITE_GRID_DISP = (0.03, 0.05, 0.10)
SUITE_GRID_ANGLE = (5, 10, 20)


def _apply_protocol(client, kind, amp, g=9.8):
    """Shake the loaded scene: 'tilt' = oscillating horizontal gravity; 'pulse' = constant horizontal acceleration
    of ``amp`` g for 0.3 s along +x, -x, +y, -y in turn, each followed by 0.25 s of rest."""
    if kind == 'tilt':
        for _ in range(2):
            for axis in (0, 1):
                for s in range(240):
                    gv = [0, 0, -g]
                    gv[axis] = amp * g * math.sin(2 * math.pi * s / 240)
                    client.setGravity(*gv)
                    client.stepSimulation()
    else:
        for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1)):
            gv = [0, 0, -g]
            gv[axis] = sign * amp * g
            client.setGravity(*gv)
            for _ in range(72):
                client.stepSimulation()
            client.setGravity(0, 0, -g)
            for _ in range(60):
                client.stepSimulation()
    client.setGravity(0, 0, -g)
    for _ in range(120):
        client.stepSimulation()


def shake_suite(env, items):
    """Run every protocol of ``SUITE_PROTOCOLS`` on the final scene (each from the same saved state) and report, per
    protocol, the share of bags that stay put (displacement and rotation grid), the volume share, and the unstable
    counts by level (floor, shelf, stacked) at the default 0.05 m / 10 degrees."""
    if not items:
        return None
    client = env.client
    boxes = []
    for c, it, pos, orn in items:
        rot = np.abs(np.array(client.getMatrixFromQuaternion(orn)).reshape(3, 3))
        half = rot @ np.array([it.length / 2, it.width / 2, it.height / 2])
        bottom = pos[2] - half[2]
        floor_z = c.thickness + c.buffer
        shelf_z = c.height / 2.0 + c.thickness + c.buffer
        level = 'floor' if bottom - floor_z < 0.03 else ('shelf' if abs(bottom - shelf_z) < 0.03 else 'stacked')
        boxes.append((level, it.length * it.width * it.height))
    vol_total = sum(v for _, v in boxes)
    out = {}
    for name, kind, amp in SUITE_PROTOCOLS:
        state = client.saveState()
        _apply_protocol(client, kind, amp)
        disp, ang = [], []
        for (c, it, pos, orn) in items:
            p2, o2 = it.get_pose(client)
            disp.append(float(np.linalg.norm(np.array(p2) - np.array(pos))))
            dot = min(1.0, abs(sum(a * b for a, b in zip(orn, o2))))
            ang.append(math.degrees(2 * math.acos(dot)))
        client.restoreState(stateId=state)
        client.removeState(state)
        client.setGravity(0, 0, -9.8)
        n = len(items)
        for d in SUITE_GRID_DISP:
            for a in SUITE_GRID_ANGLE:
                out['%s_stable_d%g_a%g' % (name, d, a)] = sum(x <= d and y <= a for x, y in zip(disp, ang)) / n
        ok = [x <= 0.05 and y <= 10 for x, y in zip(disp, ang)]
        out[name + '_stable_vol_frac'] = sum(v for (lv, v), s in zip(boxes, ok) if s) / vol_total
        out[name + '_max_disp'] = max(disp)
        out[name + '_mean_disp'] = sum(disp) / n
        for lv in ('floor', 'shelf', 'stacked'):
            out['%s_n_%s' % (name, lv)] = sum(1 for (l, _) in boxes if l == lv)
            out['%s_unstable_%s' % (name, lv)] = sum(1 for (l, _), s in zip(boxes, ok) if l == lv and not s)
    return out


def run(task_cfg, agent_path, shake=True, verbose=False, policy_limit=None, suite=False):
    """Run one task with the given agent directory and return a result dictionary.

    ``policy_limit`` (seconds) overrides the agent's per-decision soft time limit
    (``Agent.POLICY_SOFT_LIMIT``) when the agent has one; used to study decision-time budgets.
    """
    t0 = time.time()
    env = GroundHandlingEnv(config=task_cfg, verbose=False, render_mode=None)
    env.reset_settings()
    env.evaluator.inclusion_margin = 0.02  # evaluate the fill with a looser inclusion margin than the validator
    agent = load_agent(agent_path)
    if policy_limit is not None and hasattr(agent, 'POLICY_SOFT_LIMIT'):
        agent.POLICY_SOFT_LIMIT = float(policy_limit)
    agent.get_init_states(env.get_init_states())
    opt_time = 0.0
    if env.optimize:
        t = time.time()
        order = agent.optimize(env.get_info_for_optimization())
        opt_time = time.time() - t
        assert env.set_item_order(order), 'optimize() returned an invalid order'
    env.reset_item_stream()
    obs, _ = env.reset(seed=42)
    terminated = truncated = False
    policy_max = 0.0
    steps = 0
    info = {}
    while not terminated and not truncated:
        obs = dict(obs)
        for key in ('shm_name', 'shm_shape', 'shm_dtype'):
            obs.pop(key, None)
        obs['depth_map'] = env.shm_depth_map.copy()
        t = time.time()
        action = agent.policy(obs)
        dt = time.time() - t
        policy_max = max(policy_max, dt)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        if verbose:
            print(steps, action['item_idx'], action['container_idx'], np.round(action['place_pos'], 3),
                  action['orientation'], f'{dt:.2f}s', info)
    report = env.evaluate()
    metrics, items = quality_metrics(env)
    suite_result = shake_suite(env, items) if suite else None   # restores the scene after every protocol; run before shake_test
    shaken = shake_test(env, items) if shake else None
    n = max(1, metrics['n_items'])
    stability = 100.0 * (1 - (shaken['n_moved_gt_5cm'] / n)) if shaken else 0.0
    cog = 100.0 * (1 - (metrics['cog_mean_height_frac'] or 1))
    placement = 100.0 * max(0.0, 1 - (metrics['soft_covered'] + metrics['prio_covered'] + metrics['prio_in_wrong_container']) / n)
    result = dict(fill=report['fill_score'], placed_frac=report['num_placed_items'], status=info.get('status'), steps=steps,
                  policy_max_s=policy_max, optimize_s=opt_time, metrics=metrics, shake=shaken, wall_s=time.time() - t0,
                  scores=dict(fill=report['fill_score'], cog=cog, stability=stability, placement=placement),
                  diag=getattr(agent, 'diag', None), suite=suite_result)
    try:
        env.close()
    except Exception:
        pass
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--agent', default='agents/extreme_point', help='agent directory (contains agent.py)')
    ap.add_argument('--config', default='configs/sample_config.json')
    ap.add_argument('--task', default=None, help='run only this task id')
    ap.add_argument('--noshake', action='store_true', help='skip the shaking test')
    ap.add_argument('--verbose', action='store_true', help='print every action')
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    for task_id, task_cfg in cfg.items():
        if a.task and task_id != a.task:
            continue
        print('=== task', task_id)
        print(json.dumps(run(task_cfg, a.agent, shake=not a.noshake, verbose=a.verbose), indent=1, default=str))


if __name__ == '__main__':
    main()
