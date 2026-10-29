"""Raw-pose learned agent (E11, hypothesis H5a): a policy that chooses a bag, an orientation and a floor cell directly.

The agent does not use the extreme-point candidate generator to propose placements. A convolutional network reads a five-channel
height map of the container (floor-level stack height, height on the shelf plate, shelf plate, restricted top surface, container
interior) and the pool, picks a (bag, orientation) option, then a cell of a 5 cm grid for the lower back corner of the bag. The
height at the chosen pose follows from dropping the bag (as in the reference planner's geometry). Feasibility is given as an exact
mask from the packer's rule replica (inclusion, insertion path, support, class and soft-stack rules): the policy chooses only among
feasible grid poses and then slides the bag towards the walls (at most 4 cm in x and y) to close gaps.

Environment variables: LEARNED_WEIGHTS_FILE (text file with the zlib+base64 torch state dict), LEARNED_MODE (greedy | sample),
LEARNED_SEED (int, sampling). Without weights the agent uses random initial weights (a random policy over feasible poses).
The module is also used for training (see sessions/gpu in the paper repository): with LEARNED_MODE=sample the agent records ``traj``, one entry per decision.
"""
import base64
import importlib.util
import io
import os
import sys
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE_PATH = os.path.join(os.path.dirname(_HERE), 'extreme_point', 'agent.py')
_spec = importlib.util.spec_from_file_location('ep_rawpose_base', _BASE_PATH)
base = importlib.util.module_from_spec(_spec)
sys.modules['ep_rawpose_base'] = base
_spec.loader.exec_module(base)
CM, Boxes = base.CM, base.Boxes   # tools.fast_sim reads these from the agent module

G = 0.05          # grid step (m) of the lower-back corner of the bag
NX, NY = 48, 36   # grid size (covers every container of the benchmark)
N_OPT = 48        # maximum number of (bag, orientation) options
N_POOL = 10
OPT_DIM = 7
POOL_DIM = N_POOL * 8 + 2
SNAP = [(-0.01 * a, 0.01 * b) for a in range(5) for b in range(5)]   # shifts towards the wall and the back wall, metres


class Net(nn.Module):
    """Height-map policy: option logits, cell logits conditioned on the chosen option (FiLM) and a value."""

    def __init__(self, ch=32, hidden=128):
        super().__init__()
        self.c1 = nn.Conv2d(5, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=2, dilation=2)
        self.c3 = nn.Conv2d(ch, ch, 3, padding=4, dilation=4)
        self.pool_mlp = nn.Sequential(nn.Linear(POOL_DIM, 64), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(2 * ch + 64, hidden), nn.ReLU())
        self.opt_mlp = nn.Sequential(nn.Linear(hidden + OPT_DIM, 64), nn.ReLU(), nn.Linear(64, 1))
        self.film = nn.Sequential(nn.Linear(OPT_DIM + hidden, 64), nn.ReLU(), nn.Linear(64, 2 * ch))
        self.c4 = nn.Conv2d(ch, ch, 1)
        self.c5 = nn.Conv2d(ch, 1, 1)
        self.value = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
        self.ch = ch

    def encode(self, obs, pool):
        f = F.relu(self.c1(obs))
        f = F.relu(self.c2(f))
        f = F.relu(self.c3(f))
        g = torch.cat([f.mean((2, 3)), f.amax((2, 3)), self.pool_mlp(pool)], 1)
        return f, self.trunk(g)

    def option_logits(self, h, opt_feats):
        """h (B, hidden), opt_feats (B, N_OPT, OPT_DIM) -> logits (B, N_OPT)."""
        z = torch.cat([h[:, None, :].expand(-1, opt_feats.shape[1], -1), opt_feats], -1)
        return self.opt_mlp(z).squeeze(-1)

    def cell_logits(self, f, h, chosen_feats):
        """f (B, ch, NX, NY), h (B, hidden), chosen_feats (B, OPT_DIM) -> logits (B, NX*NY)."""
        gb = self.film(torch.cat([chosen_feats, h], 1))
        gamma, beta = gb[:, :self.ch, None, None], gb[:, self.ch:, None, None]
        x = F.relu(f * (1 + gamma) + beta)
        x = F.relu(self.c4(x))
        return self.c5(x).flatten(1)

    def forward_value(self, h):
        return self.value(h).squeeze(-1)


def encode_weights(net):
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu().to(torch.float16) for k, v in net.state_dict().items()}, buf)
    return base64.b64encode(zlib.compress(buf.getvalue(), 9)).decode('ascii')


def decode_weights(text, net=None):
    net = net or Net()
    state = torch.load(io.BytesIO(zlib.decompress(base64.b64decode(text))), map_location='cpu')
    net.load_state_dict({k: v.float() for k, v in state.items()})
    return net


def grid_axes(cm):
    """Anchor coordinates: x of the lower edge and y of the upper edge of the bag footprint, one per grid cell."""
    xs = cm.x_lo + base.WALL + np.arange(NX) * G
    ys = cm.y_hi - base.WALL - np.arange(NY) * G
    return xs, ys


def height_maps(cm, B):
    """Five channels on the grid (see module docstring), normalised; cell (i, j) covers [xs_i, xs_i + G] x [ys_j - G, ys_j]."""
    xs, ys = grid_axes(cm)
    out = np.zeros((5, NX, NY), np.float32)
    inside_x = (xs + G <= cm.x_hi - base.WALL + 1e-6)
    inside_y = (ys - G >= cm.y_lo - 1e-6)
    out[4] = (inside_x[:, None] & inside_y[None, :]).astype(np.float32)
    span = cm.ceil - cm.floor
    a_top = np.full((NX, NY), cm.floor, np.float32)
    b_top = np.full((NX, NY), cm.sz1, np.float32)
    on_plate = np.zeros((NX, NY), bool)
    cxs, cys = xs + G / 2, ys - G / 2
    for k in range(len(cm.shelf_lo)):
        on_plate |= ((cxs[:, None] >= cm.shelf_lo[k, 0]) & (cxs[:, None] <= cm.shelf_hi[k, 0]) &
                     (cys[None, :] >= cm.shelf_lo[k, 1]) & (cys[None, :] <= cm.shelf_hi[k, 1]))
    out[2] = on_plate
    restricted = np.zeros((NX, NY), np.float32)
    if len(B):
        ox = (np.minimum(xs[:, None] + G, B.hi[None, :, 0]) - np.maximum(xs[:, None], B.lo[None, :, 0])) > 1e-3      # (NX, Nb)
        oy = (np.minimum(ys[:, None], B.hi[None, :, 1]) - np.maximum(ys[:, None] - G, B.lo[None, :, 1])) > 1e-3      # (NY, Nb)
        ov = ox[:, None, :] & oy[None, :, :]                                                                          # (NX, NY, Nb)
        top = B.hi[:, 2]
        for lvl, tgt in ((False, a_top), (True, b_top)):
            use = ov & (B.lvl == lvl)[None, None, :]
            t = np.where(use, top[None, None, :], -1e9).max(2)
            tgt[:] = np.maximum(tgt, t)
        who = np.where(ov & (~B.lvl)[None, None, :], top[None, None, :], -1e9).argmax(2)
        has = (ov & (~B.lvl)[None, None, :]).any(2)
        restricted = np.where(has, (B.soft | B.prio)[who], False).astype(np.float32)
    out[0] = np.clip((a_top - cm.floor) / span, 0, 1.2)
    out[1] = np.where(on_plate, np.clip((b_top - cm.sz1) / span, 0, 1.2), 0.0)
    out[3] = restricted
    return out


def pool_vector(pool, n_placed, cm):
    v = np.zeros(POOL_DIM, np.float32)
    for s, b in enumerate(pool[:N_POOL]):
        v[s * 8:(s + 1) * 8] = [b['length'] / 0.8, b['width'] / 0.6, b['height'] / 0.4, float(b.get('mass', 1.0)) / 20.0,
                                float(bool(b.get('is_soft', False))), float(bool(b.get('is_prioritized', False))), 1.0, 0.0]
    v[-2] = n_placed / 50.0
    v[-1] = len(pool) / N_POOL
    return v


class Agent(base.Agent):
    """Raw-pose agent: option and grid cell chosen by the network, exact feasibility mask from the packer."""

    def get_init_states(self, init_states):
        text = os.environ.get('LEARNED_WEIGHTS_B64')
        if not text and os.environ.get('LEARNED_WEIGHTS_FILE'):
            with open(os.environ['LEARNED_WEIGHTS_FILE'], encoding='utf-8') as fh:
                text = fh.read().strip()
        torch.set_num_threads(1)
        self.net = decode_weights(text) if text else Net()
        self.net.eval()
        self.mode = os.environ.get('LEARNED_MODE', 'greedy')
        self.rng = np.random.default_rng(int(os.environ.get('LEARNED_SEED', 0)))
        self.traj = []
        self.diag = dict(n_decisions=0, n_fallback=0, n_option_tries=0)
        return super().get_init_states(init_states)

    # ------------------------------------------------------------------
    def _options(self, pool, cm):
        """Distinct (bag, orientation) options as dicts, with a cheap fit test against the container size."""
        opts, seen = [], set()
        for i, bag in enumerate(pool[:N_POOL]):
            soft, prio = bool(bag.get('is_soft', False)), bool(bag.get('is_prioritized', False))
            for o in range(6):
                dims = base.oriented_dims(bag['length'], bag['width'], bag['height'], o)
                key = tuple(round(d, 4) for d in dims) + (soft, prio)
                if key in seen:
                    continue
                seen.add(key)
                count = sum(1 for b2 in pool if (round(b2['length'], 4), round(b2['width'], 4), round(b2['height'], 4),
                                                 bool(b2.get('is_soft', False)), bool(b2.get('is_prioritized', False))) ==
                            (round(bag['length'], 4), round(bag['width'], 4), round(bag['height'], 4), soft, prio))
                fits = (dims[0] <= cm.x_hi - cm.x_lo - 2 * base.WALL and dims[1] <= cm.y_hi - cm.y_lo - base.WALL and
                        dims[2] <= cm.ceil - cm.floor)
                opts.append(dict(slot=i, o=o, dims=dims, key=key, bag=bag, feat=[dims[0] / 0.8, dims[1] / 0.6, dims[2] / 0.4,
                                                                                float(bag.get('mass', 1.0)) / 20.0, float(soft), float(prio), count / N_POOL],
                                 fits=fits))
        return opts[:N_OPT]

    def _cell_mask(self, cm, B, dims, bag):
        """Exact feasibility on the grid: bool (NX, NY) and a dict cell -> candidate (cost, xc, yc, zc, ...)."""
        xs, ys = grid_axes(cm)
        cands = self.packer.candidates(cm, B, dims, bool(bag.get('is_soft', False)), bool(bag.get('is_prioritized', False)),
                                       float(bag.get('mass', 1.0)), True, maxcheck=10 ** 9, want=10 ** 9, fixed=(xs, ys))
        mask = np.zeros((NX, NY), bool)
        by_cell = {}
        for cd in cands:
            ix = int(round((cd['xc'] - dims[0] / 2 - xs[0]) / G))
            iy = int(round((ys[0] - (cd['yc'] + dims[1] / 2)) / G))
            if 0 <= ix < NX and 0 <= iy < NY:
                mask[ix, iy] = True
                by_cell[(ix, iy)] = cd
        return mask, by_cell

    def _snap(self, cm, B, dims, bag, cd):
        """Slide the chosen pose towards the wall and the back wall (small steps), keeping the best valid pose."""
        x0, y0 = cd['xc'] - dims[0] / 2, cd['yc'] + dims[1] / 2
        xs = x0 - 0.01 * np.arange(5)
        ys = y0 + 0.01 * np.arange(5)
        found = self.packer.candidates(cm, B, dims, bool(bag.get('is_soft', False)), bool(bag.get('is_prioritized', False)),
                                       float(bag.get('mass', 1.0)), True, maxcheck=10 ** 9, want=10 ** 9, fixed=(xs, ys))
        best, shift = cd, -1.0
        for c2 in found:
            s = (x0 - (c2['xc'] - dims[0] / 2)) + ((c2['yc'] + dims[1] / 2) - y0)   # how far the pose moved towards the walls
            if s > shift:
                best, shift = c2, s
        return best

    def decide(self, state, pool, t0):
        self.diag['n_decisions'] += 1
        cm = self.packer.cms[0]
        B = state[cm.idx]
        opts = self._options(pool, cm)
        if not opts:
            self.diag['n_fallback'] += 1
            return None
        obs = torch.from_numpy(height_maps(cm, B))[None]
        pv = torch.from_numpy(pool_vector(pool, len(B), cm))[None]
        opt_feats = np.zeros((1, N_OPT, OPT_DIM), np.float32)
        for j, op in enumerate(opts):
            opt_feats[0, j] = op['feat']
        opt_t = torch.from_numpy(opt_feats)
        with torch.no_grad():
            f, h = self.net.encode(obs, pv)
            logits = self.net.option_logits(h, opt_t)[0].numpy()
        allowed = np.zeros(N_OPT, bool)
        for j, op in enumerate(opts):
            allowed[j] = op['fits']
        masks = {}
        while allowed.any():
            lg = np.where(allowed, logits, -1e9)
            if self.mode == 'sample':
                p = np.exp(lg - lg.max())
                p = p / p.sum()
                j = int(self.rng.choice(N_OPT, p=p))
            else:
                j = int(np.argmax(lg))
            op = opts[j]
            self.diag['n_option_tries'] += 1
            if op['key'] not in masks:
                masks[op['key']] = self._cell_mask(cm, B, op['dims'], op['bag'])
            mask, by_cell = masks[op['key']]
            if not mask.any():
                for j2, o2 in enumerate(opts):
                    if o2['key'] == op['key']:
                        allowed[j2] = False
                continue
            chosen = torch.tensor([op['feat']], dtype=torch.float32)
            with torch.no_grad():
                cl = self.net.cell_logits(f, h, chosen)[0].numpy()
            flat = np.where(mask.ravel(), cl, -1e9)
            if self.mode == 'sample':
                p = np.exp(flat - flat.max())
                p = p / p.sum()
                c = int(self.rng.choice(flat.size, p=p))
            else:
                c = int(np.argmax(flat))
            cd = self._snap(cm, B, op['dims'], op['bag'], by_cell[(c // NY, c % NY)])
            if self.mode == 'sample':
                self.traj.append(dict(obs=obs[0].numpy().astype(np.float16), pool=pv[0].numpy(), opt=opt_feats[0].copy(),
                                      opt_mask=allowed.copy(), opt_idx=j, cell_mask=mask.ravel().copy(), cell_idx=c,
                                      reward=100.0 * op['dims'][0] * op['dims'][1] * op['dims'][2] / (cm.L * cm.W * cm.H)))
            return (0.0, op['slot'], (cd['cost'], cm.idx, op['o'], cd, op['dims']))
        self.diag['n_fallback'] += 1
        return None
