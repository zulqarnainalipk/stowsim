"""Learned bag selector on top of the reference packer (E5).

For every distinct bag in the pool the packer proposes its best placement (the same proposal step as the reference agent,
``Packer.best_for_bag``; no rollouts). A small neural network scores these (bag, placement) options and the agent places
the highest-scoring one (``LEARNED_MODE=greedy``) or samples from the softmax (``sample``, used for training).

The score is ``skip * (-heuristic) + MLP(features, set context)``. The heuristic is the reference selection score (placement
cost minus a preference for heavy and large bags). With the MLP output at zero and ``skip`` > 0 the agent selects exactly like
the reference heuristic, so training starts from the heuristic policy and learns a correction.

Environment variables: LEARNED_WEIGHTS_B64 or LEARNED_WEIGHTS_FILE (zlib + base64 JSON of the weight arrays, as text or as the
path of a text file; default: initial weights),
LEARNED_MODE (greedy | sample, default greedy), LEARNED_SEED (int, sampling).
The inference runs in numpy only; torch is used by the training script.
"""
import base64
import importlib.util
import json
import os
import sys
import zlib

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE_PATH = os.path.join(os.path.dirname(_HERE), 'extreme_point', 'agent.py')
_spec = importlib.util.spec_from_file_location('ep_learned_base', _BASE_PATH)
base = importlib.util.module_from_spec(_spec)
sys.modules['ep_learned_base'] = base
_spec.loader.exec_module(base)
CM, Boxes = base.CM, base.Boxes   # tools.fast_sim reads these from the agent module

N_FEATURES = 24
HIDDEN = 48


def initial_weights(seed=0):
    """Weights of the starting policy: MLP output zero, skip = 2 (softmax temperature 0.5 on the heuristic)."""
    rng = np.random.default_rng(seed)
    return dict(W1=rng.normal(0, 0.3, (N_FEATURES, HIDDEN)), b1=np.zeros(HIDDEN), W2=rng.normal(0, 0.2, (2 * HIDDEN, HIDDEN)),
                b2=np.zeros(HIDDEN), w3=np.zeros(HIDDEN), b3=np.zeros(1), skip=np.array([2.0]))


def encode_weights(w):
    raw = json.dumps({k: np.round(np.asarray(v, dtype=float), 6).tolist() for k, v in w.items()}).encode('utf-8')
    return base64.b64encode(zlib.compress(raw, 9)).decode('ascii')


def decode_weights(text):
    return {k: np.array(v, dtype=float) for k, v in json.loads(zlib.decompress(base64.b64decode(text)).decode('utf-8')).items()}


def scores_numpy(w, feats, hs):
    """Option scores from the weights; feats (n, N_FEATURES), hs (n,) heuristic costs (lower is better)."""
    h = np.maximum(feats @ w['W1'] + w['b1'], 0.0)
    ctx = np.broadcast_to(h.mean(0, keepdims=True), h.shape)
    z = np.maximum(np.concatenate([h, ctx], 1) @ w['W2'] + w['b2'], 0.0)
    return z @ w['w3'] + w['b3'][0] - w['skip'][0] * hs


class Agent(base.Agent):
    """Reference agent whose online bag choice is made by the learned scorer."""

    def get_init_states(self, init_states):
        text = os.environ.get('LEARNED_WEIGHTS_B64')
        if not text and os.environ.get('LEARNED_WEIGHTS_FILE'):
            with open(os.environ['LEARNED_WEIGHTS_FILE'], encoding='utf-8') as fh:
                text = fh.read().strip()
        self.weights = decode_weights(text) if text else initial_weights()
        self.mode = os.environ.get('LEARNED_MODE', 'greedy')
        self.rng = np.random.default_rng(int(os.environ.get('LEARNED_SEED', 0)))
        self.traj = []
        self.diag = dict(n_decisions=0, n_fallback=0)
        return super().get_init_states(init_states)

    def _features(self, state, pool, opts):
        W = base.W
        cms = {cm.idx: cm for cm in self.packer.cms}
        masses = np.array([float(pool[i].get('mass', 1.0)) for i, _ in opts])
        vols = np.array([pool[i]['length'] * pool[i]['width'] * pool[i]['height'] for i, _ in opts])
        # normalisers over the whole pool, as in the reference ``choose``
        mmax = max(max(float(b.get('mass', 1.0)) for b in pool), 1e-6)
        vmax = max(max(b['length'] * b['width'] * b['height'] for b in pool), 1e-6)
        hs = np.zeros(len(opts))
        costs = np.array([r[0] for _, r in opts])
        for j, (i, r) in enumerate(opts):
            restricted = bool(pool[i].get('is_soft', False)) or bool(pool[i].get('is_prioritized', False))
            hs[j] = r[0] - W.selm * masses[j] / mmax - W.selv * vols[j] / vmax + (W.restr if restricted else 0.0)
        total_vol = sum(cm.L * cm.W * cm.H for cm in cms.values())
        placed_vol = sum(float(np.prod(B.hi - B.lo, axis=1).sum()) for B in state.values() if len(B))
        n_placed = sum(len(B) for B in state.values())
        order = np.argsort(np.argsort(hs))
        feats = np.zeros((len(opts), N_FEATURES))
        for j, (i, r) in enumerate(opts):
            bag = pool[i]
            c, cidx, o, cd, dims = r
            cm = cms[cidx]
            feats[j] = [
                bag['length'] / 0.8, bag['width'] / 0.6, bag['height'] / 0.4, masses[j] / 20.0, float(bool(bag.get('is_soft', False))),
                float(bool(bag.get('is_prioritized', False))), vols[j] / 0.1,
                c / 10.0, (c - costs.min()) / 5.0, (hs[j] - hs.min()) / 5.0, order[j] / max(len(opts) - 1, 1),
                (cd['xc'] - cm.x_lo) / (cm.x_hi - cm.x_lo), (cd['yc'] - cm.y_lo) / (cm.y_hi - cm.y_lo), (cd['zc'] - cm.floor) / cm.H,
                cd['frac'], float(cd['level']), (cd['top'] - cm.floor) / cm.H, o / 5.0,
                dims[2] / max(min(dims[0], dims[1]), 1e-6) / 2.0,
                placed_vol / total_vol, n_placed / 50.0, len(opts) / 10.0, len(pool) / 10.0, float(len(cms) > 1)]
        return feats, hs

    def decide(self, state, pool, t0):
        packer = self.packer
        cache, opts = {}, []
        for i, bag in enumerate(pool):
            key = (round(bag['length'], 4), round(bag['width'], 4), round(bag['height'], 4),
                   bool(bag.get('is_soft', False)), bool(bag.get('is_prioritized', False)))
            if key not in cache:
                cache[key] = packer.best_for_bag(state, bag, True)
            if cache[key] is not None:
                opts.append((i, cache[key]))
        self.diag['n_decisions'] += 1
        if not opts:
            self.diag['n_fallback'] += 1
            return super().decide(state, pool, t0)
        feats, hs = self._features(state, pool, opts)
        s = scores_numpy(self.weights, feats, hs)
        if self.mode == 'sample':
            p = np.exp(s - s.max())
            p /= p.sum()
            pos = int(self.rng.choice(len(opts), p=p))
        else:
            pos = int(np.argmax(s))
        self.traj.append((feats, hs, pos))
        i, r = opts[pos]
        return (0.0, i, r)
