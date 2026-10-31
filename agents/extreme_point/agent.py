"""Extreme-point packer agent for the StowSim baggage-loading simulator.

Overview
--------
The agent rebuilds the scene from the observation (placed bags -> axis-aligned boxes),
enumerates flush "extreme point" candidate positions for every (bag, orientation, container)
combination and keeps only the candidates that pass a faithful re-implementation of the
environment's checks (inclusion test, insertion-path test, support / stability rules).

* Online mode (`policy`): candidates are ranked by a weighted cost (back-to-front, low,
  left-to-right, support, side contact). The bag to place is chosen by short greedy rollouts over
  the visible pool.  Quality tiers (strict -> relaxed -> last resort) guarantee that a bag is placed
  whenever any valid position exists.
* Offline mode (`optimize`): a randomised local search over the loading order, evaluated with the
  same packer in a physics-free simulation. The best plan (order + positions) is replayed by `policy`.

Tuning knobs for experiments (environment variables, all optional):
    EP_WEIGHTS      JSON dict overriding fields of class `W`, e.g. '{"contact": 4.0}'
    EP_CONSTANTS    JSON dict overriding module constants, e.g. '{"SIDE": 0.02}'
    EP_OPT_BUDGET   seconds spent in `optimize` (default: Agent.OPT_BUDGET)
    EP_CPU_CLOCK    if set, time budgets use CPU time instead of wall time (benchmarking under load)

The agent uses no network access and no hard-coded container or bag sizes.
"""
import math
import time

_now = time.process_time if __import__('os').environ.get('EP_CPU_CLOCK') else time.time

import numpy as np

# ---- environment constants (mirrors validator.py / evaluation parameters) ----
START_Z = 0.08
CEIL_MARGIN = 0.018
START_MARGIN = 0.01
ENV_SAFETY = 0.015

# ---- our own safety slack on top of the environment ones ----
PATH_M = 0.0165      # collision distance used in our path check
SIDE = 0.0185        # lateral spacing between neighbouring bags
WALL = 0.0075        # spacing to container walls (inclusion margin is 5 mm)
GAP_ITEM = 0.002     # drop height above a supporting bag
GAP_FLOOR = 0.007
GAP_SHELF = 0.02
CEIL_SLACK = 0.012
INCL_MARGIN = -0.0065

_ORN_IDX = [(0, 1, 2), (0, 2, 1), (2, 1, 0), (1, 0, 2), (1, 2, 0), (2, 0, 1)]


def oriented_dims(l, w, h, o):
    d = (l, w, h)
    i = _ORN_IDX[o]
    return d[i[0]], d[i[1]], d[i[2]]


def quat_to_mat(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class CM:
    """Container model (local coordinates: origin at (offset_x, 0, 0))."""

    def __init__(self, c):
        self.idx = int(c['index'])
        self.L = float(c['length'])
        self.W = float(c['width'])
        self.H = float(c['height'])
        self.t = float(c['thickness'])
        self.cx = float(c['cut_x'])
        self.cy = float(c['cut_y'])
        ctr = c['center']
        self.off = float(ctr[0])
        self.buf = float(ctr[2]) - self.H / 2
        self.shelf = bool(c.get('shelf', False))
        self.prio = bool(c.get('is_prioritized', False))
        nv = c.get('n_vecs', c.get('n_vec'))
        self.nv = np.array(nv, float)
        self.absnv = np.abs(self.nv)
        self.pt = np.array(c['points'], float) - np.array([self.off, 0.0, 0.0])
        L, W, H, t, buf = self.L, self.W, self.H, self.t, self.buf
        self.x_lo = -L / 2 + t
        self.x_hi = L / 2 - t
        self.y_lo = -W / 2
        self.y_hi = W / 2 - t
        self.floor = buf + t
        self.ceil = buf + H - t
        self.sz0 = H / 2 + buf
        self.sz1 = self.sz0 + t
        sl, sh = [], []
        # small shelf (always present)
        sl.append([-L / 2 + t, -(W / 2 - t), self.sz0])
        sh.append([-L / 2 + t + self.cx, (W / 2 - t), self.sz1])
        self.big_shelf = None
        if self.shelf:
            sl.append([-(L - t) / 2, t, self.sz0])
            sh.append([(L - t) / 2, W / 2 - t, self.sz1])
        self.shelf_lo = np.array(sl, float)
        self.shelf_hi = np.array(sh, float)
        self.volume = float(c.get('volume', 1.0))

    # exact inclusion test of env.validator.check_inclusion (axis aligned bag)
    def included(self, xc, yc, zc, hx, hy, hz, margin=INCL_MARGIN):
        pos = np.stack([xc, yc, zc], -1)
        dots = ((pos[:, None, :] - self.pt[None]) * self.nv[None]).sum(-1)
        dots = dots + (self.absnv @ np.array([hx, hy, hz]))[None]
        return np.all(dots <= margin, axis=1)

    def in_shelf_region(self, xlo, xhi, ylo, yhi):
        """True where the whole footprint lies on a shelf plate (union)."""
        a = np.array([0.0, 0.5, 1.0])
        px = xlo[:, None, None] + a[None, :, None] * (xhi - xlo)[:, None, None]      # (P,3,1)
        py = ylo[:, None, None] + a[None, None, :] * (yhi - ylo)[:, None, None]      # (P,1,3)
        inside = np.zeros((len(xlo), 3, 3), bool)
        for k in range(len(self.shelf_lo)):
            inside |= ((px >= self.shelf_lo[k, 0] - 1e-6) & (px <= self.shelf_hi[k, 0] + 1e-6) &
                       (py >= self.shelf_lo[k, 1] - 1e-6) & (py <= self.shelf_hi[k, 1] + 1e-6))
        return inside.reshape(len(xlo), 9).all(1)


class Boxes:
    """Placed bags of one container as AABBs (local coords)."""

    def __init__(self):
        self.lo = np.zeros((0, 3))
        self.hi = np.zeros((0, 3))
        self.soft = np.zeros(0, bool)
        self.prio = np.zeros(0, bool)
        self.mass = np.zeros(0)
        self.lvl = np.zeros(0, bool)
        self.sd = np.zeros(0, int)

    def recompute_sd(self):
        n = len(self.lo)
        sd = np.zeros(n, int)
        for a in np.argsort(self.lo[:, 2]):
            if not self.soft[a]:
                continue
            best = 0
            for b in range(n):
                if b == a or not self.soft[b]:
                    continue
                if self.lo[b, 2] < self.lo[a, 2] and abs(self.hi[b, 2] - self.lo[a, 2]) < 0.08:
                    ox = min(self.hi[a, 0], self.hi[b, 0]) - max(self.lo[a, 0], self.lo[b, 0])
                    oy = min(self.hi[a, 1], self.hi[b, 1]) - max(self.lo[a, 1], self.lo[b, 1])
                    if ox > 0.01 and oy > 0.01:
                        best = max(best, sd[b])
            sd[a] = best + 1
        self.sd = sd

    def add(self, lo, hi, soft, prio, mass, cm):
        self.lo = np.vstack([self.lo, lo])
        self.hi = np.vstack([self.hi, hi])
        self.soft = np.append(self.soft, soft)
        self.prio = np.append(self.prio, prio)
        self.mass = np.append(self.mass, mass)
        above = bool(cm.shelf is not None and (lo[2] >= cm.sz1 - 0.012) and
                     bool(cm.in_shelf_region(np.array([(lo[0] + hi[0]) / 2]), np.array([(lo[0] + hi[0]) / 2]),
                                             np.array([(lo[1] + hi[1]) / 2]), np.array([(lo[1] + hi[1]) / 2]))[0]))
        self.lvl = np.append(self.lvl, above)
        self.recompute_sd()

    def copy(self):
        b = Boxes()
        b.lo, b.hi, b.soft, b.prio, b.mass, b.lvl = (self.lo.copy(), self.hi.copy(), self.soft.copy(),
                                                       self.prio.copy(), self.mass.copy(), self.lvl.copy())
        b.sd = self.sd.copy()
        return b

    def __len__(self):
        return len(self.lo)


def boxes_from_container(cm, c):
    B = Boxes()
    for it in c.get('packed_items', []):
        pos = it.get('pos')
        orn = it.get('orn')
        if pos is None or orn is None:
            continue
        R = np.abs(quat_to_mat(orn))
        half = R @ np.array([it['length'] / 2, it['width'] / 2, it['height'] / 2])
        ctr = np.array([pos[0] - cm.off, pos[1], pos[2]])
        B.add(ctr - half, ctr + half, bool(it.get('is_soft', False)), bool(it.get('is_prioritized', False)),
              float(it.get('mass', 1.0)), cm)
    return B


# ----------------------------------------------------------------------------
# Weights
# ----------------------------------------------------------------------------
class W:
    """Tunable weights and switches of the packer (see module docstring for the override mechanism)."""
    # --- placement cost (lower is better) ---
    y = 2.0              # per metre of distance from the back wall (fill back to front)
    z = 1.0              # per metre of bag-bottom height (fill low first)
    x = 0.5              # per metre of distance from the left wall
    frac = 1.5           # penalty for a lack of support below the bag
    tall = 0.6           # penalty for tall / thin orientations
    head = 1.2           # headroom lost above restricted (soft / priority) bags, per metre
    prio_cont = 3.0      # penalty for a standard bag in a priority container
    contact = 6.0        # reward for side contact with walls and neighbours
    softsup = 0.6        # penalty for resting on soft bags (compliant, wobbly)
    hw = 0.0             # wasted-headroom penalty per metre (disabled)
    softfront = 0.0      # pull restricted bags towards the opening (disabled)
    # --- feasibility rules ---
    min_support = 0.72   # minimum supported fraction of the footprint in the strict tier
    max_soft_stack = 2   # maximum number of soft bags stacked on each other
    max_slender = 1.3    # maximum height / smallest-base-side ratio in the strict tier
    align = 1            # also generate candidates aligned with the edges of placed bags
    class_rule = 1       # forbid standard bags on soft / priority bags in the strict tier (0 = ablation, 2 = allowed with support >= restricted_support)
    restricted_support = 0.9   # support share a standard bag needs to rest on a soft or priority bag when class_rule == 2
    # --- bag selection (online) ---
    selm = 0.6           # preference for heavy bags
    selv = 0.6           # preference for large bags
    restr = 0.0          # penalty for placing restricted bags early
    roll_n = 9           # number of first moves evaluated by rollouts (0 disables rollouts)
    roll_orders = 1      # rest-of-pool orderings used per rollout
    roll_cnt = 0.05      # reward per bag placed in a rollout (volume units)
    roll_alt = 2         # alternative positions tried for the best bags
    roll_small = 0       # small-bags-first rollouts (disabled)
    # --- last-resort tiers ---
    desperate = 2        # number of last-resort tiers tried when nothing fits
    tight_last = 1       # use tight tolerances in the final tier
    # --- offline order search ---
    opt_cnt = 0.3        # objective bonus per placed bag (volume + opt_cnt per bag)
    opt_fast = 0         # use the fast candidate mode inside the search (disabled)
    smart_mut = 2        # mutation strategy of the local search
    memo = 1             # skip bags whose type already failed in this rollout
    abort = 1            # exact early abort of hopeless rollouts
    sa_seeds = 1         # add the extra seed orderings
    sa_t0 = 0.0          # simulated-annealing start temperature (0 = plain hill climbing)
    sa_end = 0.8         # fraction of the budget after which annealing ends


_TOL_DEF = {}
_TOL_TIGHT = dict(PATH_M=0.0153, SIDE=0.0156, WALL=0.0055, GAP_ITEM=0.0005, GAP_FLOOR=0.0056, GAP_SHELF=0.0156,
                  CEIL_SLACK=0.0056)


def _set_tol(tight):
    g = globals()
    if not _TOL_DEF:
        for k_ in _TOL_TIGHT:
            _TOL_DEF[k_] = g[k_]
    for k_, v_ in (_TOL_TIGHT if tight else _TOL_DEF).items():
        g[k_] = v_

import os as _os
if _os.environ.get('EP_CONSTANTS'):
    import json as _json
    globals().update(_json.loads(_os.environ['EP_CONSTANTS']))
if _os.environ.get('EP_WEIGHTS'):
    import json as _json
    for _k, _v in _json.loads(_os.environ['EP_WEIGHTS']).items():
        setattr(W, _k, _v)


def _dist2_boxes(slo, shi, blo, bhi):
    """Squared distance between sweep boxes (P,3) and boxes (N,3) -> (P,N)."""
    g = np.maximum(np.maximum(slo[:, None, :] - bhi[None, :, :], blo[None, :, :] - shi[:, None, :]), 0.0)
    return (g * g).sum(-1)


class Packer:
    def __init__(self, cms, lookahead):
        self.cms = cms
        self.k = lookahead
        self.hmin = 0.27
        self.fast = False
        self.desp = False
        self.dbg = None
        self.collect = None

    # ------------------------------------------------------------------
    def lift_and_z(self, cm, zc, hz):
        bottom = zc - hz
        resting = np.zeros(len(zc), bool)
        for r in (cm.t, cm.H / 2.0 + cm.t + cm.buf):
            d = bottom - r
            resting |= (d >= 0) & (d <= 0.05)
        lift = np.where(resting, 0.0, START_Z)
        top = zc + hz
        done = np.zeros(len(zc), bool)
        lift0 = lift.copy()
        for cz in (cm.H / 2.0 + cm.buf, cm.H + cm.buf - cm.t):
            cl = cz - top
            m = (lift0 > 0) & (~done) & (cl >= 0) & (cl < lift0 + CEIL_MARGIN)
            lift = np.where(m, np.maximum(0.0, cl - CEIL_MARGIN - 0.0005), lift)
            done |= m
        zs = np.minimum(cm.H + cm.buf - cm.t - hz - START_MARGIN, zc + lift)
        return zs

    def path_ok(self, cm, B, xc, yc, zc, hx, hy, hz):
        P = len(xc)
        zs = self.lift_and_z(cm, zc, hz)
        x_min = -cm.L / 2 + cm.t + cm.cx + hx + START_MARGIN
        x_max = cm.L / 2 - cm.t - hx - START_MARGIN
        rx = np.minimum(np.maximum(xc, x_min), x_max)
        blo = np.vstack([B.lo, cm.shelf_lo]) if len(B) else cm.shelf_lo
        bhi = np.vstack([B.hi, cm.shelf_hi]) if len(B) else cm.shelf_hi
        # sweep 1: along y at x = rx
        s1lo = np.stack([rx - hx, np.full(P, -cm.W / 2 - hy), zs - hz], -1)
        s1hi = np.stack([rx + hx, yc + hy, zs + hz], -1)
        ok = _dist2_boxes(s1lo, s1hi, blo, bhi).min(1) >= PATH_M ** 2
        s2lo = np.stack([np.minimum(rx, xc) - hx, yc - hy, zs - hz], -1)
        s2hi = np.stack([np.maximum(rx, xc) + hx, yc + hy, zs + hz], -1)
        ok &= _dist2_boxes(s2lo, s2hi, blo, bhi).min(1) >= PATH_M ** 2
        # final pose must not intersect shelves / bags (physical overlap)
        flo = np.stack([xc - hx, yc - hy, zc - hz], -1)
        fhi = np.stack([xc + hx, yc + hy, zc + hz], -1)
        ok &= _dist2_boxes(flo, fhi, cm.shelf_lo, cm.shelf_hi).min(1) >= 0.003 ** 2
        return ok

    def contact(self, cm, B, xlo, xhi, ylo, yhi, zlo, zhi, dims):
        dx, dy, dz = dims
        tot = 2 * dy * dz + 2 * dx * dz
        wl = (xlo <= cm.x_lo + WALL + 0.006) * (dy * dz)
        wr = (xhi >= cm.x_hi - WALL - 0.006) * (dy * dz)
        wb = (yhi >= cm.y_hi - WALL - 0.006) * (dx * dz)
        c = wl + wr + wb
        if len(B):
            def ov1(alo, ahi, blo, bhi):
                return np.maximum(0.0, np.minimum(ahi[:, None], bhi[None]) - np.maximum(alo[:, None], blo[None]))
            oz = ov1(zlo, zhi, B.lo[:, 2], B.hi[:, 2])
            oy = ov1(ylo, yhi, B.lo[:, 1], B.hi[:, 1])
            ox = ov1(xlo, xhi, B.lo[:, 0], B.hi[:, 0])
            t = 0.006
            left = np.abs(xlo[:, None] - (B.hi[:, 0] + SIDE)[None]) < t
            right = np.abs(xhi[:, None] - (B.lo[:, 0] - SIDE)[None]) < t
            back = np.abs(yhi[:, None] - (B.lo[:, 1] - SIDE)[None]) < t
            front = np.abs(ylo[:, None] - (B.hi[:, 1] + SIDE)[None]) < t
            c = c + ((left | right) * oy * oz).sum(1) + ((back | front) * ox * oz).sum(1)
        return np.minimum(c / tot, 1.0)

    # ------------------------------------------------------------------
    def candidates(self, cm, B, dims, soft, prio, mass, strict=True, maxcheck=400, want=6, fixed=None):
        if self.fast:
            maxcheck, want = 120, 1
        dx, dy, dz = dims
        hx, hy, hz = dx / 2, dy / 2, dz / 2
        # ---- candidate coordinates (x_min, y_max edge) ----
        xs = [cm.x_lo + WALL, cm.x_hi - WALL - dx, cm.shelf_hi[0, 0] + SIDE]
        ys = [cm.y_hi - WALL]
        if len(B):
            xs += list(B.hi[:, 0] + SIDE) + list(B.lo[:, 0] - SIDE - dx)
            ys += list(B.lo[:, 1] - SIDE)
            if W.align:
                xs += list(B.lo[:, 0]) + list(B.hi[:, 0] - dx)
                ys += list(B.hi[:, 1]) + list(B.lo[:, 1] + dy)
        if fixed is not None:
            # a pose (x_min, y_max edge) or a pair of arrays: every combination is checked (used by agents/rawpose)
            xs = np.atleast_1d(np.asarray(fixed[0], float))
            ys = np.atleast_1d(np.asarray(fixed[1], float))
        else:
            xs = np.unique(np.round(np.array(xs), 3))
            ys = np.unique(np.round(np.array(ys), 3))
            xs = xs[(xs >= cm.x_lo + WALL - 1e-3) & (xs + dx <= cm.x_hi - WALL + 1e-3)]
            ys = ys[(ys - dy >= cm.y_lo + 0.001) & (ys <= cm.y_hi - WALL + 1e-3)]
        if len(xs) == 0 or len(ys) == 0:
            return []
        XX, YY = np.meshgrid(xs, ys, indexing='ij')
        xlo = XX.ravel()
        yhi = YY.ravel()
        ylo = yhi - dy
        xhi = xlo + dx
        xc = xlo + hx
        yc = ylo + hy
        P = len(xc)

        N = len(B)
        out = []
        thr = W.min_support if strict else ((0.4 if self.desp == 1 else 0.15) if self.desp else 0.55)
        ceil_ok = cm.ceil - CEIL_SLACK
        for level in (0, 1):
            if level == 0:
                base = cm.floor + GAP_FLOOR
                sub = np.arange(P)
            else:
                base = cm.sz1 + GAP_SHELF
                sub = np.nonzero(cm.in_shelf_region(xlo, xhi, ylo, yhi))[0]
                if len(sub) == 0:
                    continue
            if base + dz > ceil_ok:
                continue
            sxlo, sxhi, sylo, syhi = xlo[sub], xhi[sub], ylo[sub], yhi[sub]
            Ps = len(sub)
            valid = np.ones(Ps, bool)
            if N:
                ox = np.minimum(sxhi[:, None], B.hi[None, :, 0]) - np.maximum(sxlo[:, None], B.lo[None, :, 0])
                oy = np.minimum(syhi[:, None], B.hi[None, :, 1]) - np.maximum(sylo[:, None], B.lo[None, :, 1])
                ov = (ox > 1e-3) & (oy > 1e-3)
                lvlmask = (~B.lvl) if level == 0 else B.lvl
                use = ov & lvlmask[None, :]
                btop = B.hi[:, 2] + GAP_ITEM
                zb = np.maximum(base, np.where(use, btop[None, :], -1e9).max(1))
                # class conflicts: bag would end up above a restricted bag
                bad = ov & ((B.soft[None, :] & (not soft)) | (B.prio[None, :] & (not prio)))
                if not self.desp and W.class_rule == 1:
                    valid &= ~bad.any(1)
                sup_mask = use & (np.abs(btop[None, :] - zb[:, None]) < 0.0045)
                sup = np.where(sup_mask, ox * oy, 0.0).sum(1)
                on_base = zb <= base + 1e-6
                frac = np.where(on_base, 1.0, sup / (dx * dy))
                supsoft = np.where(sup_mask & B.soft[None, :], ox * oy, 0.0).sum(1) / (dx * dy)
                supsoft = np.where(on_base, 0.0, supsoft)
                if not self.desp and W.class_rule == 2:   # limited stacking: a standard bag may rest on a soft or priority bag if it is well supported
                    valid &= (~bad.any(1)) | (frac >= W.restricted_support)
                if soft:
                    sdm = np.where(sup_mask & B.soft[None, :], B.sd[None, :], 0).max(1)
                    valid &= (sdm + 1) <= (W.max_soft_stack if strict else W.max_soft_stack + (3 if self.desp else 1))
            else:
                zb = np.full(Ps, base)
                frac = np.ones(Ps)
                supsoft = np.zeros(Ps)
            top_z = zb + dz
            if self.dbg is not None:
                d = self.dbg
                d['pairs'] = d.get('pairs', 0) + Ps
                d['classok'] = d.get('classok', 0) + int(valid.sum())
                d['ceilfail'] = d.get('ceilfail', 0) + int((valid & (top_z > ceil_ok)).sum())
                d['supfail'] = d.get('supfail', 0) + int((valid & (top_z <= ceil_ok) & (frac < thr)).sum())
            valid &= (top_z <= ceil_ok) & (frac >= thr)
            if not valid.any():
                continue
            v = np.nonzero(valid)[0]
            idx = sub[v]
            zbv, frv, topv, ssv = zb[v], frac[v], top_z[v], supsoft[v]
            incl = cm.included(xc[idx], yc[idx], zbv + hz, hx, hy, hz)
            if not incl.any():
                continue
            idx, zbv, frv, topv, ssv = idx[incl], zbv[incl], frv[incl], topv[incl], ssv[incl]
            cost = (W.y * (cm.y_hi - yhi[idx]) + W.z * zbv + W.x * (xlo[idx] - cm.x_lo) + W.frac * (1.0 - frv) + W.softsup * ssv)
            if W.hw:
                ceilc = np.full(len(idx), cm.ceil)
                if level == 0:
                    us = np.zeros(len(idx), bool)
                    for k in range(len(cm.shelf_lo)):
                        us |= ((xhi[idx] > cm.shelf_lo[k, 0]) & (xlo[idx] < cm.shelf_hi[k, 0]) &
                               (yhi[idx] > cm.shelf_lo[k, 1]) & (ylo[idx] < cm.shelf_hi[k, 1]))
                    ceilc = np.where(us, cm.sz0, ceilc)
                hrem = ceilc - topv
                cost = cost + W.hw * np.where(hrem < self.hmin, hrem, 0.0)
            out.append((idx, level, zbv, frv, cost))
        if not out:
            return []
        # merge levels
        a_idx = np.concatenate([o_[0] for o_ in out])
        a_lv = np.concatenate([np.full(len(o_[0]), o_[1]) for o_ in out])
        a_zb = np.concatenate([o_[2] for o_ in out])
        a_fr = np.concatenate([o_[3] for o_ in out])
        a_c = np.concatenate([o_[4] for o_ in out])
        order = np.argsort(a_c, kind='stable')[:maxcheck]
        sel_i, sel_lv, sel_zb, sel_fr, sel_c = a_idx[order], a_lv[order], a_zb[order], a_fr[order], a_c[order]
        if W.contact:
            ct = self.contact(cm, B, xlo[sel_i], xlo[sel_i] + dx, ylo[sel_i], ylo[sel_i] + dy, sel_zb, sel_zb + dz, dims)
            sel_c = sel_c - W.contact * ct
            o2 = np.argsort(sel_c, kind='stable')
            sel_i, sel_lv, sel_zb, sel_fr, sel_c = sel_i[o2], sel_lv[o2], sel_zb[o2], sel_fr[o2], sel_c[o2]
        zc = sel_zb + hz
        nrows = len(sel_i)
        ok = np.zeros(nrows, bool)
        # check in chunks, best first
        chunk = 60
        found = 0
        s = 0
        for s in range(0, nrows, chunk):
            sl = slice(s, s + chunk)
            ok[sl] = self.path_ok(cm, B, xc[sel_i[sl]], yc[sel_i[sl]], zc[sl], hx, hy, hz)
            found = ok.sum()
            if found >= want:
                break
        res = []
        for a in np.nonzero(ok)[0][:want]:
            res.append(dict(cost=float(sel_c[a]), xc=float(xc[sel_i[a]]), yc=float(yc[sel_i[a]]),
                            zc=float(zc[a]), zb=float(sel_zb[a]), frac=float(sel_fr[a]), level=int(sel_lv[a]),
                            top=float(sel_zb[a] + dz)))
        return res

    # ------------------------------------------------------------------
    def best_for_bag(self, state, bag, strict=True, deadline=None):
        """state: dict cidx -> Boxes. returns best (cost, cidx, orient, cand) or None"""
        l, w, h = bag['length'], bag['width'], bag['height']
        soft = bool(bag.get('is_soft', False))
        prio = bool(bag.get('is_prioritized', False))
        mass = float(bag.get('mass', 1.0))
        restricted = soft or prio
        has_prio_cont = any(cm.prio for cm in self.cms)
        seen = set()
        best = None
        for o in range(6):
            dims = oriented_dims(l, w, h, o)
            key = tuple(round(d, 4) for d in dims)
            if key in seen:
                continue
            seen.add(key)
            dx, dy, dz = dims
            tall = max(0.0, dz / max(min(dx, dy), 1e-6) - 0.8)
            if strict and dz > W.max_slender * min(dx, dy):
                continue
            for cm in self.cms:
                if deadline is not None and best is not None and _now() > deadline:
                    return best
                B = state[cm.idx]
                cands = self.candidates(cm, B, dims, soft, prio, mass, strict=strict)
                for cd in cands:
                    c = cd['cost'] + W.tall * tall
                    if restricted:
                        c += W.head * max(0.0, cm.ceil - cd['top'])
                        if W.softfront:
                            c -= W.softfront * (cm.y_hi - (cd['yc'] + dy / 2))
                    if has_prio_cont:
                        if prio and not cm.prio:
                            c += 50.0
                        if (not prio) and cm.prio:
                            c += W.prio_cont
                    if self.collect is not None:
                        self.collect.append((c, cm.idx, o, cd, dims))
                    if best is None or c < best[0]:
                        best = (c, cm.idx, o, cd, dims)
        return best

    def check_planned(self, state, bag, plan):
        cidx, o, xc, yc = plan
        cm = next((c for c in self.cms if c.idx == cidx), None)
        if cm is None:
            return None
        dims = oriented_dims(bag['length'], bag['width'], bag['height'], o)
        cands = self.candidates(cm, state[cidx], dims, bool(bag.get('is_soft', False)),
                                bool(bag.get('is_prioritized', False)), float(bag.get('mass', 1.0)),
                                True, fixed=(xc - dims[0] / 2, yc + dims[1] / 2), want=1)
        if not cands:
            return None
        return (cands[0]['cost'], cidx, o, cands[0], dims)

    def apply(self, state, cidx, cd, dims, bag):
        cm = next(c for c in self.cms if c.idx == cidx)
        hx, hy, hz = dims[0] / 2, dims[1] / 2, dims[2] / 2
        ctr = np.array([cd['xc'], cd['yc'], cd['zc']])
        h = np.array([hx, hy, hz])
        state[cidx].add(ctr - h, ctr + h, bool(bag.get('is_soft', False)), bool(bag.get('is_prioritized', False)),
                        float(bag.get('mass', 1.0)), cm)

    def choose(self, state, pool, strict=True, rng=None, noise=0.0, deadline=None):
        """choose best (pool index, container, orient, cand). pool: list of bag dicts"""
        if not pool:
            return None
        cache = {}
        masses = [float(b.get('mass', 1.0)) for b in pool]
        vols = [b['length'] * b['width'] * b['height'] for b in pool]
        mmax = max(masses) or 1.0
        vmax = max(vols) or 1.0
        best = None
        for i, bag in enumerate(pool):
            if deadline is not None and best is not None and _now() > deadline:
                break
            key = (round(bag['length'], 4), round(bag['width'], 4), round(bag['height'], 4),
                   bool(bag.get('is_soft', False)), bool(bag.get('is_prioritized', False)))
            if key not in cache:
                cache[key] = self.best_for_bag(state, bag, strict, deadline)
            r = cache[key]
            if r is None:
                continue
            restricted = bool(bag.get('is_soft', False)) or bool(bag.get('is_prioritized', False))
            score = r[0] - W.selm * (masses[i] / mmax) - W.selv * (vols[i] / vmax) + (W.restr if restricted else 0.0)
            if noise and rng is not None:
                score += rng.random() * noise
            if best is None or score < best[0]:
                best = (score, i, r)
        return best


    def choose_rollout(self, state, pool, deadline=None):
        first = None
        if len(pool) < 2:
            return self.choose2(state, pool, deadline=deadline)
        opts, seen = [], set()
        for i, bag in enumerate(pool):
            key = (round(bag['length'], 4), round(bag['width'], 4), round(bag['height'], 4), bool(bag.get('is_soft')), bool(bag.get('is_prioritized')))
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
                        if all(a[1] != t[1] or abs(a[3]['xc'] - t[3]['xc']) + abs(a[3]['yc'] - t[3]['yc']) + abs(a[3]['zc'] - t[3]['zc']) > 0.04 for t in taken):
                            taken.append(a)
                            opts.append((a[0] + pen + 0.3, i, a))
                            if len(taken) > W.roll_alt:
                                break
        opts.sort(key=lambda o: o[0])
        best = None
        for sc, i, r in opts[:W.roll_n]:
            if deadline is not None and _now() > deadline and best is not None:
                break
            bag = pool[i]
            vol_tot = 0.0
            orders = [_tier_key, lambda it: (_tier_key(it)[0], -it['length'] * it['width'], -it['height'])][:W.roll_orders]
            if W.roll_small:
                orders = [lambda it: (_tier_key(it)[0], it['length'] * it['width'] * it['height'])]
            self.fast = True
            for okey in orders:
                st = {k_: B.copy() for k_, B in state.items()}
                self.apply(st, r[1], r[3], r[4], bag)
                rest = pool[:i] + pool[i + 1:]
                vol = bag['length'] * bag['width'] * bag['height']
                rest.sort(key=okey)
                for b2 in rest:
                    rr = self.best_for_bag(st, b2, True, deadline)
                    if rr is None:
                        continue
                    self.apply(st, rr[1], rr[3], rr[4], b2)
                    vol += b2['length'] * b2['width'] * b2['height'] + W.roll_cnt
                vol_tot += vol
            self.fast = False
            vol = vol_tot / len(orders)
            val = vol - 0.02 * sc
            if best is None or val > best[0]:
                best = (val, i, r)
        if best:
            return (0.0, best[1], best[2])
        return self.choose2(state, pool, deadline=deadline)

    def choose2(self, state, pool, **kw):
        _ts = time.time()
        res = self.choose(state, pool, strict=True, **kw)
        if res is None:
            self.fast = True
            try:
                res = self.choose(state, pool, strict=False, **kw)
            finally:
                self.fast = False
        for lvl in range(1, 1 + W.desperate):
            if res is not None or time.time() - _ts > 3.5:
                break
            self.desp = lvl
            self.fast = True
            if W.tight_last and lvl == W.desperate:
                _set_tol(True)
            try:
                res = self.choose(state, pool, strict=False, **kw)
            finally:
                self.desp = False
                self.fast = False
                _set_tol(False)
        return res

    def run(self, state0, items, k, rng=None, noise=0.0, deadline=None):
        """Physics-free greedy rollout. Unplaceable items are moved to the end (as the env would end on them).
        returns (placed_volume, order_of_indices)"""
        state = {i: B.copy() for i, B in state0.items()}
        pool = list(items[:k])
        nxt = len(pool)
        seq, failed, vol = [], [], 0.0
        while pool:
            if deadline is not None and _now() > deadline:
                # out of time: append everything left unchanged
                failed += [b['index'] for b in pool] + [b['index'] for b in items[nxt:]]
                break
            res = self.choose2(state, pool, rng=rng, noise=noise)
            if res is None:
                bag = pool.pop(0)
                failed.append(bag['index'])
            else:
                _, i, (c, cidx, o, cd, dims) = res
                bag = pool.pop(i)
                self.apply(state, cidx, cd, dims, bag)
                seq.append(bag['index'])
                vol += bag['length'] * bag['width'] * bag['height']
            if nxt < len(items):
                pool.append(items[nxt])
                nxt += 1
        return vol, seq + failed


def _tkey(it):
    return (round(it['length'], 3), round(it['width'], 3), round(it['height'], 3), bool(it.get('is_soft')), bool(it.get('is_prioritized')))


def _tier_key(it):
    soft = bool(it.get('is_soft', False))
    prio = bool(it.get('is_prioritized', False))
    vol = it['length'] * it['width'] * it['height']
    tier = 0
    if prio and not soft:
        tier = 1
    elif soft and not prio:
        tier = 2
    elif soft and prio:
        tier = 3
    return (tier, -it.get('mass', 1.0), -vol)


class Agent:
    OPT_BUDGET = 140.0   # seconds spent in optimize (limit is 180; final plan run reserved inside, hard cap 172)
    POLICY_SOFT_LIMIT = 2.2

    def __init__(self, module_path: str):
        self.packer = None
        self.cms = []
        self.k = 1
        self.init_containers = []
        self.plan = {}
        if _os.environ.get('EP_OPT_BUDGET'):
            self.OPT_BUDGET = float(_os.environ['EP_OPT_BUDGET'])

    def get_init_states(self, init_states: dict):
        self.k = int(init_states.get('lookahead_k', 1))
        self.init_containers = init_states['container_list']
        self.cms = [CM(c) for c in self.init_containers]
        self.packer = Packer(self.cms, self.k)
        self.offline = bool(init_states.get('optimize', False))
        return True

    # ---------------------------------------------------------------
    def _place_one(self, state, bag):
        r = self.packer.best_for_bag(state, bag, True)
        if r is None:
            r = self.packer.best_for_bag(state, bag, False)
        for lvl in range(1, 1 + W.desperate):
            if r is not None:
                break
            self.packer.desp = lvl
            if W.tight_last and lvl == W.desperate:
                _set_tol(True)
            try:
                r = self.packer.best_for_bag(state, bag, False)
            finally:
                self.packer.desp = False
                _set_tol(False)
        return r

    def _seq_run(self, state, bags, snaps=None, deadline=None, plan=None, fail_cap=None):
        """Sequential skip-on-failure rollout. Returns (succ_bags, fail_bags, vol, state)."""
        succ, fail, vol = [], [], 0.0
        failed_types = set()
        fvol = 0.0
        for b in bags:
            if deadline is not None and _now() > deadline:
                fail.append(b)
                continue
            fk = _tkey(b)
            if W.memo and fk in failed_types:
                fail.append(b)
                fvol += b['length'] * b['width'] * b['height'] + W.opt_cnt
                if fail_cap is not None and fvol > fail_cap + 1e-9:
                    return None, fail, -1.0, state
                continue
            r = self._place_one(state, b)
            if r is None:
                fail.append(b)
                failed_types.add(fk)
                fvol += b['length'] * b['width'] * b['height'] + W.opt_cnt
                if fail_cap is not None and fvol > fail_cap + 1e-9:
                    return None, fail, -1.0, state
                continue
            _, cidx, o, cd, dims = r
            self.packer.apply(state, cidx, cd, dims, b)
            if plan is not None:
                plan[b['index']] = (cidx, o, cd['xc'], cd['yc'])
            succ.append(b)
            vol += b['length'] * b['width'] * b['height'] + W.opt_cnt
            if snaps is not None:
                snaps.append(({i: B.copy() for i, B in state.items()}, vol))
        return succ, fail, vol, state

    def _optimize_fixed(self, item_list: list):
        """Known-items regime: the whole bag list is known but the arrival order is fixed.

        Each bag, in arrival order, takes the placement (best one and up to EP_FIXED_OPTIONS - 1 distinct alternatives) whose greedy
        continuation over the next EP_FIXED_HORIZON known bags places the most volume; after the time budget it places greedily.
        """
        t0 = _now()
        deadline = t0 + self.OPT_BUDGET
        n_opt = int(_os.environ.get('EP_FIXED_OPTIONS', 3))
        horizon = int(_os.environ.get('EP_FIXED_HORIZON', 6))
        items = [dict(it) for it in item_list]
        cms = [CM(c) for c in self.init_containers]
        self.packer.cms = cms
        state = {cm.idx: boxes_from_container(cm, c) for cm, c in zip(cms, self.init_containers)}
        plan = {}
        for n, bag in enumerate(items):
            self.packer.fast = False
            self.packer.collect = []
            r = self._place_one(state, bag)
            alts, self.packer.collect = (self.packer.collect or []), None
            if r is None:
                break   # the episode ends at the first bag that cannot be placed
            best = r
            if n_opt > 1 and horizon > 0 and _now() < deadline:
                options = [r]
                for a in sorted(alts, key=lambda t: t[0]):
                    if len(options) >= n_opt:
                        break
                    if all(a[1] != t[1] or abs(a[3]['xc'] - t[3]['xc']) + abs(a[3]['yc'] - t[3]['yc']) + abs(a[3]['zc'] - t[3]['zc']) > 0.04 for t in options):
                        options.append(a)
                if len(options) > 1:
                    best_val = None
                    self.packer.fast = True
                    for opt in options:
                        if _now() > deadline and best_val is not None:
                            break
                        st = {k_: B.copy() for k_, B in state.items()}
                        self.packer.apply(st, opt[1], opt[3], opt[4], bag)
                        vol = 0.0
                        for b2 in items[n + 1:n + 1 + horizon]:
                            rr = self.packer.best_for_bag(st, b2, True, deadline)
                            if rr is None:
                                break
                            self.packer.apply(st, rr[1], rr[3], rr[4], b2)
                            vol += b2['length'] * b2['width'] * b2['height'] + W.roll_cnt
                        val = vol - 0.02 * opt[0]
                        if best_val is None or val > best_val:
                            best_val, best = val, opt
                    self.packer.fast = False
            _, cidx, o, cd, dims = best
            self.packer.apply(state, cidx, cd, dims, bag)
            plan[bag['index']] = (cidx, o, cd['xc'], cd['yc'])
        self.plan = plan
        return [b['index'] for b in items]

    def optimize(self, item_list: list):
        if _os.environ.get('EP_FIXED_ORDER'):
            return self._optimize_fixed(item_list)
        import random
        t0 = _now()
        iters = int(_os.environ.get('EP_OPT_ITERS', '0') or 0)   # > 0: stop after this many search iterations, not after a time budget (machine independent)
        deadline = t0 + (1e9 if iters > 0 else self.OPT_BUDGET)
        self._hard_dl = t0 + (1e9 if iters > 0 else min(self.OPT_BUDGET + 12.0, 172.0))
        items = [dict(it) for it in item_list]
        cms = [CM(c) for c in self.init_containers]
        self.packer.cms = cms
        state0 = {cm.idx: boxes_from_container(cm, c) for cm, c in zip(cms, self.init_containers)}
        rng = random.Random(12345)
        self.packer.fast = bool(W.opt_fast)

        def full(order_bags, dl=None):
            snaps = [({i: B.copy() for i, B in state0.items()}, 0.0)]
            st = {i: B.copy() for i, B in state0.items()}
            succ, fail, vol, _ = self._seq_run(st, order_bags, snaps, dl)
            return succ, fail, vol, snaps

        def area(it):
            return it['length'] * it['width']

        def vol_of(it):
            return it['length'] * it['width'] * it['height']

        tk = _tier_key
        seeds = [sorted(items, key=tk),
                 sorted(items, key=lambda it: (tk(it)[0], -vol_of(it), -it.get('mass', 1.0))),
                 sorted(items, key=lambda it: (tk(it)[0], -area(it), -it['height']))]
        if W.opt_cnt:
            seeds += [sorted(items, key=lambda it: (tk(it)[0], vol_of(it))), sorted(items, key=lambda it: (tk(it)[0], area(it), it['height']))]
        if W.sa_seeds:
            seeds += [sorted(items, key=lambda it: (tk(it)[0], -max(it['length'], it['width'], it['height']), -vol_of(it))),
                      sorted(items, key=lambda it: (tk(it)[0], -min(it['length'], it['width'], it['height']), -vol_of(it)))]
        best = None
        run_times = []
        for s in seeds:
            ts = _now()
            succ, fail, vol, snaps = full(s, deadline)
            run_times.append(_now() - ts)
            if best is None or vol > best[2]:
                best = (succ, fail, vol, snaps)
            if _now() + 2.0 * max(run_times) > deadline:
                return self._finish(state0, best[0] + best[1])
        succ, fail, bvol, snaps = best
        # reserve time for the final plan-building run
        search_dl = deadline - 1.3 * max(run_times) - 0.5
        best_order = succ + fail
        best_vol = bvol
        cvol = bvol
        vols = sorted(vol_of(it) for it in items)
        T0 = W.sa_t0 * vols[len(vols) // 2]
        t_start = _now()
        span = max(1e-6, search_dl - t_start)
        import math as _m
        reset_done = False
        avg_eval = max(run_times) * 0.3
        nit = 0
        nabort = 0
        bvol0 = bvol
        while (nit < iters) if iters > 0 else (_now() + avg_eval < search_dl):
            nit += 1
            self.last_nit = nit   # number of search iterations done so far (reported with the result)
            frac_t = (nit / iters) if iters > 0 else (_now() - t_start) / span
            if W.sa_t0 > 0:
                if frac_t > W.sa_end and not reset_done:
                    # final phase: greedy from the best order found so far
                    reset_done = True
                    L0 = best_order
                    st = {k_: B.copy() for k_, B in state0.items()}
                    snaps = [({k_: B.copy() for k_, B in state0.items()}, 0.0)]
                    succ, fail, cvol, _ = self._seq_run(st, L0, snaps, None)
                    if cvol < best_vol - 1e-9:  # should not happen (deterministic)
                        cvol = best_vol
                T = 0.0 if reset_done else T0 * max(0.0, 1.0 - frac_t / W.sa_end)
            else:
                T = 0.0
            L = succ + fail
            n = len(L)
            cand = list(L)
            r = rng.random()
            i = rng.randrange(n)
            j = rng.randrange(n)
            if W.smart_mut:
                for _ in range(6):
                    if _tkey(L[i]) != _tkey(L[j]):
                        break
                    j = rng.randrange(n)
                if W.smart_mut > 1 and fail and rng.random() < 0.4:
                    i = len(succ) + rng.randrange(len(fail))
                    j = rng.randrange(max(1, len(succ)))
            if r < 0.35:
                cand[i], cand[j] = cand[j], cand[i]
            elif r < 0.8:
                v = cand.pop(i)
                cand.insert(j, v)
            else:
                a, b = sorted((i, j))
                seg = cand[a:b + 1]
                rng.shuffle(seg)
                cand[a:b + 1] = seg
            p = 0
            while p < n and cand[p] is L[p]:
                p += 1
            if p >= n:
                continue
            p = min(p, len(succ))
            ts = _now()
            fcap = sum(b_['length'] * b_['width'] * b_['height'] + W.opt_cnt for b_ in fail)
            st = {k_: B.copy() for k_, B in snaps[p][0].items()}
            new_snaps = []
            s2, f2, v2, _ = self._seq_run(st, cand[p:], new_snaps, search_dl, None, fcap if (T == 0 and W.abort) else None)
            if s2 is None:
                nabort += 1
                continue
            if _now() > search_dl:
                break
            avg_eval = 0.9 * avg_eval + 0.1 * (_now() - ts)
            vol = snaps[p][1] + v2
            if vol >= cvol - 1e-9 or (T > 0 and rng.random() < _m.exp((vol - cvol) / T)):
                succ = cand[:p] + s2
                fail = f2
                snaps = snaps[:p + 1] + [(s_, snaps[p][1] + v_) for (s_, v_) in new_snaps]
                cvol = vol
                if cvol > best_vol + 1e-9:
                    best_vol = cvol
                    best_order = succ + fail
        return self._finish(state0, best_order)

    def _finish(self, state0, order_bags):
        plan = {}
        self.packer.fast = False
        st = {i: B.copy() for i, B in state0.items()}
        self._seq_run(st, order_bags, None, getattr(self, '_hard_dl', None), plan)
        self.plan = plan
        return [b['index'] for b in order_bags]

    # ---------------------------------------------------------------
    def decide(self, state, pool, t0):
        res = None
        if self.offline and pool:
            r = None
            pl = self.plan.get(pool[0]['index'])
            if pl is not None:
                r = self.packer.check_planned(state, pool[0], pl)
            if r is None:
                r = self._place_one(state, pool[0])
            if r is not None:
                res = (0.0, 0, r)
        if res is None:
            if W.roll_n:
                res = self.packer.choose_rollout(state, pool, deadline=t0 + self.POLICY_SOFT_LIMIT)
            else:
                res = self.packer.choose2(state, pool, deadline=t0 + self.POLICY_SOFT_LIMIT)
        return res

    def policy(self, observation: dict):
        try:
            return self._policy(observation)
        except Exception:
            n = len(observation.get('pool_list', [])) or 1
            return {'item_idx': 0, 'container_idx': 0,
                    'place_pos': np.array([0.0, 0.0, 0.5], dtype=np.float32), 'orientation': 0}

    def _policy(self, observation: dict):
        t0 = _now()
        cl = observation['container_list']
        pool = observation['pool_list']
        cms = [CM(c) for c in cl]
        self.packer.cms = cms
        state = {cm.idx: boxes_from_container(cm, c) for cm, c in zip(cms, cl)}
        res = self.decide(state, pool, t0)
        if res is None:
            return {'item_idx': 0, 'container_idx': 0,
                    'place_pos': np.array([0.0, 0.0, 0.5], dtype=np.float32), 'orientation': 0}
        _, i, (c, cidx, o, cd, dims) = res
        return {'item_idx': int(i), 'container_idx': int(cidx),
                'place_pos': np.array([cd['xc'], cd['yc'], cd['zc']], dtype=np.float32),
                'orientation': int(o)}