"""Physics-free simulation of the extreme-point agent on many random scenarios.

The agent's own geometric model is used as the "world": every chosen placement is assumed to
settle exactly where it was planned. This is roughly two orders of magnitude faster than the
real environment and is meant for tuning and A/B comparisons. Always confirm important results
with `tools.batch_eval` (real physics).

The tool relies on internals of `agents/extreme_point` (``CM``, ``Boxes``, ``Agent.decide``,
``Packer.apply``), so it only works for agents built on that code base.

Usage (from the repository root)::

    python -m tools.fast_sim --n 8 --seed 0 [--agent agents/extreme_point] [--procs 4]
    python -m tools.fast_sim --like 001 --n 10 --seed 3      # shuffled copies of a sample task (online)
    python -m tools.fast_sim --offline --n 8 --seed 5        # offline task: lookahead 1 + order search
    python -m tools.fast_sim --online --k 5 --n 16 --seed 5  # online task with a pool of 5 bags

Environment variables: ``OPT_BUDGET`` (seconds for the offline search, default 100).
"""
import argparse
import copy
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.getcwd())
import numpy as np

# (length, width, height, mass, is_soft) of the seven bag types listed in configs/item_params.xlsx
BAG_TYPES = [
    (0.75, 0.56, 0.27, 18, False), (0.65, 0.45, 0.25, 13, False), (0.55, 0.40, 0.24, 8, False),
    (0.60, 0.30, 0.25, 7, True), (0.50, 0.40, 0.40, 10, True), (0.65, 0.35, 0.23, 12, True),
    (0.45, 0.30, 0.20, 5, True)]
BAG_TYPE_PROBABILITIES = [.05, .30, .30, .05, .06, .14, .10]


def make_items(rng, n, p_prio=0.06):
    """Draw ``n`` random bags from the seven standard bag types."""
    items = []
    for i in range(n):
        l, w, h, m, soft = BAG_TYPES[rng.choice(range(len(BAG_TYPES)), p=BAG_TYPE_PROBABILITIES)]
        item = dict(index=i, length=l, width=w, height=h, mass=m, is_prioritized=bool(rng.random() < p_prio),
                    is_soft=soft, lateralFriction=0.8 if soft else 0.4, rollingFriction=0.01, spinningFriction=0.01,
                    restitution=0.2 if not soft else 0.0)
        if soft:
            item.update(contactStiffness=3000, contactDamping=800, linearDamping=0.8)
        items.append(item)
    return items


def make_cfg(rng, ncont=None):
    """Create a random task configuration (1-2 containers, random sizes, shelf, bags, lookahead)."""
    base = json.load(open('configs/sample_config.json'))['001']
    cfg = copy.deepcopy(base)
    ncont = ncont or int(rng.choice([1, 1, 2]))
    containers = []
    for i in range(ncont):
        c = copy.deepcopy(base['containers']['container_list'][0])
        c['index'] = i
        c['length'] = float(round(rng.uniform(1.85, 2.05), 2))
        c['width'] = float(round(rng.uniform(1.4, 1.55), 2))
        c['height'] = float(round(rng.uniform(1.55, 1.65), 2))
        c['thickness'] = float(rng.choice([0.04, 0.05]))
        c['cut_x'] = float(round(rng.uniform(0.40, 0.46), 2))
        c['cut_y'] = float(round(rng.uniform(0.36, 0.42), 2))
        c['require_shelf'] = bool(rng.random() < 0.5)
        c['is_prioritized'] = bool(ncont > 1 and i == 1)
        c['packed_items'] = []
        containers.append(c)
    cfg['containers']['container_list'] = containers
    cfg['containers']['spacing'] = 2.5
    n = int(rng.integers(42, 58)) * ncont
    cfg['item_stream']['item_list'] = make_items(rng, n)
    cfg['item_stream']['look_ahead'] = int(rng.choice([1, 5, 10, 10]))
    cfg['item_stream']['max_space'] = 1
    cfg['camera']['num_containers'] = ncont
    cfg['agent']['optimize'] = bool(cfg['item_stream']['look_ahead'] == 1)
    return cfg


def simulate(args):
    """Simulate one scenario; returns a dict of fill, placed fraction, CoG and rule violations."""
    cfg, agent_path, label = args
    from tools.run_agent import load_agent
    from stowsim.env import GroundHandlingEnv
    env = GroundHandlingEnv(config=cfg, verbose=False)
    env.reset_settings()
    init = env.get_init_states()
    containers_info = env.container_manager.get_item_info_in_containers()
    try:
        env.close()
    except Exception:
        pass
    agent = load_agent(agent_path)
    agent.get_init_states(init)
    agent.OPT_BUDGET = float(os.environ.get('OPT_BUDGET', 100))
    mod = sys.modules['agent_mod']
    items = [dict(i, belongs_to=None, pos=None, orn=None) for i in cfg['item_stream']['item_list']]
    t0 = time.time()
    if init['optimize']:
        order = agent.optimize(items)
        rank = {v: k for k, v in enumerate(order)}
        items.sort(key=lambda i: rank[i['index']])
    k = init['lookahead_k']
    pool = items[:k]
    nxt = k
    cms = [mod.CM(c) for c in containers_info]
    agent.packer.cms = cms
    state = {cm.idx: mod.Boxes() for cm in cms}
    placed_vol = 0.0
    prio_wrong = 0
    placed = 0
    max_dt = 0.0
    while pool:
        t = time.time()
        res = agent.decide(state, pool, time.time())
        max_dt = max(max_dt, time.time() - t)
        if res is None:
            break
        _, i, (_, cidx, o, cd, dims) = res
        bag = pool.pop(i)
        if bag.get('is_prioritized') and any(c_.prio for c_ in cms) and not cms[cidx].prio:
            prio_wrong += 1
        agent.packer.apply(state, cidx, cd, dims, bag)
        placed += 1
        placed_vol += bag['length'] * bag['width'] * bag['height']
        if nxt < len(items):
            pool.append(items[nxt])
            nxt += 1
    volume = sum(cm.volume for cm in cms)
    mass_sum = z_sum = 0.0
    violations = 0
    for cm in cms:
        B = state[cm.idx]
        for a in range(len(B)):
            mass_sum += B.mass[a]
            z_sum += B.mass[a] * ((B.lo[a, 2] + B.hi[a, 2]) / 2 - cm.floor) / cm.H
            for b in range(len(B)):
                if a == b:
                    continue
                ox = min(B.hi[a, 0], B.hi[b, 0]) - max(B.lo[a, 0], B.lo[b, 0])
                oy = min(B.hi[a, 1], B.hi[b, 1]) - max(B.lo[a, 1], B.lo[b, 1])
                if ox > 0.005 and oy > 0.005 and B.hi[a, 2] - 0.01 <= B.lo[b, 2] <= B.hi[a, 2] + 0.03:
                    if (B.soft[a] and not B.soft[b]) or (B.prio[a] and not B.prio[b]):
                        violations += 1
                        break
    return dict(label=label, n=len(items), placed=placed, frac=placed / len(items), fill=100 * placed_vol / volume,
                cog=z_sum / mass_sum if mass_sum else 0, viol=violations, prio_wrong=prio_wrong, maxdt=max_dt,
                secs=time.time() - t0, k=k, ncont=len(cms), traj=getattr(agent, 'traj', None))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=8, help='number of random scenarios')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--agent', default='agents/extreme_point')
    ap.add_argument('--procs', type=int, default=4, help='worker processes')
    ap.add_argument('--k', type=int, default=10, help='pool size for --online')
    ap.add_argument('--like', default=None, help='sample task id: shuffled copies of that task')
    ap.add_argument('--offline', action='store_true', help='single container, lookahead 1, order search')
    ap.add_argument('--online', action='store_true', help='single container, pool of --k bags, no sample tasks')
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    jobs = []
    if a.like:
        base = json.load(open('configs/sample_config.json'))[a.like]
        for i in range(a.n):
            cfg = copy.deepcopy(base)
            items = cfg['item_stream']['item_list']
            shuffled = []
            for j, p in enumerate(rng.permutation(len(items))):
                item = dict(items[p])
                item['index'] = j
                shuffled.append(item)
            cfg['item_stream']['item_list'] = shuffled
            jobs.append((cfg, a.agent, f'like{a.like}_{i}'))
        a.n = 0
    for i in range(a.n):
        cfg = make_cfg(rng, ncont=1 if (a.online or a.offline) else None)
        if a.offline:
            cfg['item_stream']['look_ahead'] = 1
            cfg['agent']['optimize'] = True
        if a.online:
            cfg['item_stream']['look_ahead'] = a.k
            cfg['agent']['optimize'] = False
        jobs.append((cfg, a.agent, f's{a.seed}_{i}'))
    if not a.online and not a.like:
        for task_id, task_cfg in json.load(open('configs/sample_config.json')).items():
            jobs.append((task_cfg, a.agent, 'sample' + task_id))
    with Pool(a.procs) as pool:
        results = pool.map(simulate, jobs, chunksize=1)
    for r in results:
        print({key: (round(v, 3) if isinstance(v, float) else v) for key, v in r.items()})
    print('MEAN fill %.2f frac %.3f cog %.3f viol %d pw %d lt60 %d lt65 %d maxdt %.2f' % (
        np.mean([r['fill'] for r in results]), np.mean([r['frac'] for r in results]), np.mean([r['cog'] for r in results]),
        sum(r['viol'] for r in results), sum(r['prio_wrong'] for r in results), sum(r['frac'] < 0.6 for r in results),
        sum(r['frac'] < 0.65 for r in results), max(r['maxdt'] for r in results)))


if __name__ == '__main__':
    main()
