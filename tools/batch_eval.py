"""Evaluate an agent on many random scenarios with the REAL physics environment, in parallel.

Each scenario is run in-process with `tools.run_agent.run`; the scenarios come from
`tools.fast_sim.make_cfg` plus the sample tasks. Slower than `tools.fast_sim`, but it uses the
actual validator, so early terminations (a rejected or collapsing placement) show up here.

Usage (from the repository root)::

    python -m tools.batch_eval --n 8 --seed 3 [--agent agents/extreme_point] [--procs 4] [--shake]
"""
import argparse
import contextlib
import io
import json
import os
import sys
import traceback
from multiprocessing import Pool

sys.path.insert(0, os.getcwd())
import numpy as np


def job(args):
    """Run one scenario and capture the log lines that indicate collisions or unsafe placements."""
    cfg, agent, label, shake = args
    from tools.run_agent import run
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            result = run(cfg, agent, shake=shake)
        except Exception:
            result = dict(error=traceback.format_exc())
    result['label'] = label
    lines = buf.getvalue().splitlines()
    result['log_tail'] = lines[-6:]
    result['bad_lines'] = [l for l in lines if 'collision' in l or 'Not safe' in l or 'not included' in l][-3:]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=8, help='number of random scenarios (sample tasks are added)')
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--agent', default='agents/extreme_point')
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--shake', action='store_true', help='also run the shaking test')
    a = ap.parse_args()
    from tools.fast_sim import make_cfg
    rng = np.random.default_rng(a.seed)
    jobs = [(make_cfg(rng), a.agent, f'r{a.seed}_{i}', a.shake) for i in range(a.n)]
    for task_id, task_cfg in json.load(open('configs/sample_config.json')).items():
        jobs.append((task_cfg, a.agent, 'sample' + task_id, a.shake))
    with Pool(a.procs) as pool:
        results = pool.map(job, jobs, chunksize=1)
    for r in results:
        if 'error' in r:
            print(r['label'], 'ERROR', r['error'][-600:])
            continue
        print(r['label'], 'fill %.1f placed %.2f steps %d status %s pol_max %.2fs cog %.3f soft/prio covered %s/%s shake %s' % (
            r['fill'], r['placed_frac'], r['steps'], r['status'], r['policy_max_s'],
            r['metrics']['cog_mean_height_frac'], r['metrics']['soft_covered'], r['metrics']['prio_covered'], r['shake']))
        if r['bad_lines']:
            print('   log:', r['bad_lines'])
    ok = [r for r in results if 'fill' in r]
    print('MEAN fill %.2f placed %.3f' % (np.mean([r['fill'] for r in ok]), np.mean([r['placed_frac'] for r in ok])))


if __name__ == '__main__':
    main()
