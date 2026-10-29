"""Anytime variant of the extreme-point reference agent.

The reference agent evaluates every first move once, with one deterministic rollout over the visible pool, and then
stops, so a larger time budget changes nothing. This variant keeps that first pass and then uses the remaining
per-decision time (``Agent.POLICY_SOFT_LIMIT``) for more rollouts: each round re-runs a rollout for every candidate
first move with a random order of the rest of the pool, and the value of a move is the mean over all its rollouts.
More time therefore means more rollouts per move and a less noisy choice.

Optionally (environment variable ``ANYTIME_FUTURE``, default 0) a rollout also appends that many hypothetical future
bags, sampled from the bags seen so far in this episode, so that a move which leaves room for typical bags is preferred.
Hypothetical bags count with weight ``FUTURE_WEIGHT`` in the value. Nothing is read from the future of the real stream.

Environment variables: ANYTIME_FUTURE (int, default 0), ANYTIME_MAX_ROUNDS (int, default 200), ANYTIME_SEED (int, default 0),
plus those of the reference agent (EP_WEIGHTS, EP_CONSTANTS, ...).
"""
import importlib.util
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE_PATH = os.path.join(os.path.dirname(_HERE), 'extreme_point', 'agent.py')
_spec = importlib.util.spec_from_file_location('ep_reference_agent', _BASE_PATH)
base = importlib.util.module_from_spec(_spec)
sys.modules['ep_reference_agent'] = base
_spec.loader.exec_module(base)

FUTURE_WEIGHT = 0.5


class AnytimePacker(base.Packer):
    """Packer whose ``choose_rollout`` keeps improving its estimate until the deadline."""

    def __init__(self, cms, lookahead):
        super().__init__(cms, lookahead)
        self.history = []
        self.rng = random.Random(int(os.environ.get('ANYTIME_SEED', 0)))
        self.n_future = int(os.environ.get('ANYTIME_FUTURE', 0))
        self.max_rounds = int(os.environ.get('ANYTIME_MAX_ROUNDS', 200))
        self.last_rounds = 0

    def _options(self, state, pool):
        """Candidate first moves (score, pool index, placement), as in the reference rollout."""
        W = base.W
        opts, seen = [], set()
        for i, bag in enumerate(pool):
            key = (round(bag['length'], 4), round(bag['width'], 4), round(bag['height'], 4),
                   bool(bag.get('is_soft')), bool(bag.get('is_prioritized')))
            if key in seen:
                continue
            seen.add(key)
            self.collect = [] if W.roll_alt else None
            r = self.best_for_bag(state, bag, True)
            alts, self.collect = (self.collect or []), None
            pen = W.restr if (bag.get('is_soft') or bag.get('is_prioritized')) else 0
            if r is not None:
                opts.append((r[0] + pen, i, r))
                if W.roll_alt:
                    taken = [r]
                    for a in sorted(alts, key=lambda t: t[0]):
                        if all(a[1] != t[1] or abs(a[3]['xc'] - t[3]['xc']) + abs(a[3]['yc'] - t[3]['yc'])
                               + abs(a[3]['zc'] - t[3]['zc']) > 0.04 for t in taken):
                            taken.append(a)
                            opts.append((a[0] + pen + 0.3, i, a))
                            if len(taken) > W.roll_alt:
                                break
        opts.sort(key=lambda o: o[0])
        return opts

    def _rollout(self, state, pool, i, r, order_key, future, deadline):
        """Place move ``r`` of bag ``i`` and then the rest of the pool (and hypothetical bags) greedily; return the volume."""
        W = base.W
        bag = pool[i]
        st = {k_: B.copy() for k_, B in state.items()}
        self.apply(st, r[1], r[3], r[4], bag)
        vol = bag['length'] * bag['width'] * bag['height']
        rest = pool[:i] + pool[i + 1:]
        rest.sort(key=order_key)
        for b2 in rest:
            rr = self.best_for_bag(st, b2, True, deadline)
            if rr is None:
                continue
            self.apply(st, rr[1], rr[3], rr[4], b2)
            vol += b2['length'] * b2['width'] * b2['height'] + W.roll_cnt
        for b2 in future:
            rr = self.best_for_bag(st, b2, True, deadline)
            if rr is None:
                continue
            self.apply(st, rr[1], rr[3], rr[4], b2)
            vol += FUTURE_WEIGHT * (b2['length'] * b2['width'] * b2['height'] + W.roll_cnt)
        return vol

    def _sample_future(self, pool):
        """Hypothetical future bags drawn from the bags seen so far (not currently in the pool)."""
        if not self.n_future or not self.history:
            return []
        pool_ids = {b['index'] for b in pool}
        candidates = [b for b in self.history if b['index'] not in pool_ids] or self.history
        return [dict(self.rng.choice(candidates)) for _ in range(self.n_future)]

    def choose_rollout(self, state, pool, deadline=None):
        W = base.W
        now = base._now
        if len(pool) < 2:
            return self.choose2(state, pool, deadline=deadline)
        opts = self._options(state, pool)[:W.roll_n]
        if not opts:
            return self.choose2(state, pool, deadline=deadline)
        stats = {}        # option position -> [sum of volumes, number of rollouts]
        self.fast = True
        try:
            for pos, (sc, i, r) in enumerate(opts):          # pass 0: the reference's deterministic rollout
                if deadline is not None and now() > deadline and stats:
                    break
                vol = self._rollout(state, pool, i, r, base._tier_key, [], deadline)
                stats[pos] = [vol, 1]
            rounds = 0
            while rounds < self.max_rounds and (deadline is None or now() < deadline):
                rounds += 1
                for pos in list(stats):
                    if deadline is not None and now() > deadline:
                        break
                    sc, i, r = opts[pos]
                    keys = {id(b): self.rng.random() for b in pool}
                    vol = self._rollout(state, pool, i, r, lambda it, k=keys: (base._tier_key(it)[0], k[id(it)]),
                                        self._sample_future(pool), deadline)
                    stats[pos][0] += vol
                    stats[pos][1] += 1
            self.last_rounds = rounds
        finally:
            self.fast = False
        best = None
        for pos, (total, n) in stats.items():
            sc, i, r = opts[pos]
            val = total / n - 0.02 * sc
            if best is None or val > best[0]:
                best = (val, i, r)
        return (0.0, best[1], best[2])


class Agent(base.Agent):
    """Reference agent with the anytime packer; budget = ``POLICY_SOFT_LIMIT`` seconds per decision."""

    def get_init_states(self, init_states):
        ok = super().get_init_states(init_states)
        old = self.packer
        self.packer = AnytimePacker(self.cms, self.k)
        self.packer.__dict__.update({k: v for k, v in old.__dict__.items() if k not in ('history', 'rng', 'n_future', 'max_rounds', 'last_rounds')})
        return ok

    def _policy(self, observation):
        for bag in observation.get('pool_list', []):
            if not any(h['index'] == bag['index'] for h in self.packer.history):
                self.packer.history.append(dict(bag))
        return super()._policy(observation)
