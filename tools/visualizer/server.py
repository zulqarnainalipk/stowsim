"""Interactive visualizer / manual-placement tool for the ground-handling simulator.

  python -m tools.visualizer.server --config configs/sample_config.json --task 001 --port 8765
  then open http://localhost:8765 in a browser (run from the repository root)

Features
  * three 2-D views (top x-y, front x-z, side y-z) of every container, bags coloured by type
  * pick a bag from the pool, orientation, container, x/y/z, press Place -> the REAL env.step runs (validator verdict shown)
  * "Suggest" asks our agent (agents/extreme_point) for its best candidate positions for the selected bag; click one to load it
  * "Agent step" lets the agent move; "Undo" / "Reset" replay the action history on a fresh env
  * live stats: placed count / fraction, fill score (looser margin), mass-weighted CoG, class violations
"""
import argparse, contextlib, io, json, os, sys, traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.getcwd())
import numpy as np
from stowsim.env import GroundHandlingEnv

ARGS = None
CFG = None
STATE = {}


def quat_aabb(orn, dims):
    x, y, z, w = orn
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return (np.abs(R) @ (np.array(dims) / 2)).tolist()


def new_env():
    env = GroundHandlingEnv(config=json.loads(json.dumps(CFG)), verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):
        env.reset_settings()
    env.evaluator.inclusion_margin = 0.02
    return env


def start(history=None):
    if STATE.get('env') is not None:
        try:
            STATE['env'].close()
        except Exception:
            pass
    env = new_env()
    from tools.run_agent import load_agent
    ag = load_agent(ARGS.agent)
    ag.get_init_states(env.get_init_states())
    if env.optimize:
        env.set_item_order(ag.optimize(env.get_info_for_optimization()) if ARGS.agent_order else
                           sorted(env.stream_manager.all_indices))
    env.reset_item_stream()
    obs, _ = env.reset(seed=42)
    STATE.update(env=env, agent=ag, obs=obs, history=[], last={'msg': 'ready'}, done=False)
    for a in (history or []):
        do_step(a, record=True)


def do_step(a, record=True):
    env = STATE['env']
    pl = STATE['obs'].get('pool_list', [])
    bag_index = pl[int(a['item_idx'])]['index'] if 0 <= int(a['item_idx']) < len(pl) else -1
    action = dict(item_idx=int(a['item_idx']), container_idx=int(a['container_idx']),
                  place_pos=np.array(a['place_pos'], dtype=np.float32), orientation=int(a['orientation']))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        obs, r, term, trunc, info = env.step(action)
    if obs:
        STATE['obs'] = obs
    st = info.get('status', {})
    ok = bool(st.get('is_included') and st.get('is_valid') and st.get('is_placed_safe'))
    msgs = [l for l in buf.getvalue().splitlines() if 'collision' in l or 'Not safe' in l or 'not included' in l or 'not inside' in l]
    STATE['last'] = dict(ok=ok, status=st, msgs=msgs[:4], terminated=bool(term or trunc))
    STATE['done'] = bool(term or trunc)
    if ok and record:
        STATE['history'].append(dict(a, place_pos=[float(v) for v in a['place_pos']], bag=bag_index))
    return ok


def view():
    env = STATE['env']
    obs = STATE['obs']
    containers = []
    cl = obs['container_list'] if 'container_list' in obs else []
    for c in cl:
        off = c['center'][0]
        bags = []
        for it in c['packed_items']:
            if it['pos'] is None:
                continue
            half = quat_aabb(it['orn'], [it['length'], it['width'], it['height']])
            bags.append(dict(index=it['index'], c=[it['pos'][0] - off, it['pos'][1], it['pos'][2]], h=half, soft=it['is_soft'],
                             prio=it['is_prioritized'], mass=it['mass'], dims=[it['length'], it['width'], it['height']]))
        t = c['thickness']
        buf = c['center'][2] - c['height'] / 2
        containers.append(dict(index=c['index'], L=c['length'], W=c['width'], H=c['height'], t=t, cut_x=c['cut_x'], cut_y=c['cut_y'],
                               buf=buf, shelf=c['shelf'], prio=c['is_prioritized'], bags=bags, offset=off))
    pool = [dict(i=i, index=p['index'], dims=[p['length'], p['width'], p['height']], mass=p['mass'], soft=p['is_soft'], prio=p['is_prioritized'])
            for i, p in enumerate(obs.get('pool_list', []))]
    rep = env.evaluate()
    n_total = env.num_total_items
    allb = [b for c in containers for b in c['bags']]
    m = sum(b['mass'] for b in allb)
    cog = sum(b['mass'] * b['c'][2] for b in allb) / m if m else 0
    return dict(containers=containers, pool=pool, last=STATE['last'], done=STATE['done'],
                stats=dict(placed=len(allb), total=n_total, fill=rep['fill_score'], cog_z=cog, steps=len(STATE['history'])),
                history_view=[dict(bag=h.get('bag', h['item_idx']), container_idx=h['container_idx'], orientation=h['orientation'], place_pos=h['place_pos']) for h in STATE['history']],
                orients=['0: [l,w,h]', '1: [l,h,w]', '2: [h,w,l]', '3: [w,l,h]', '4: [w,h,l]', '5: [h,l,w]'])


def suggest(item_idx, topn=10):
    ag = STATE['agent']
    mod = sys.modules['agent_mod']
    obs = STATE['obs']
    pool = obs['pool_list']
    if not pool:
        return []
    bag = pool[item_idx]
    cms = [mod.CM(c) for c in obs['container_list']]
    st = {cm.idx: mod.boxes_from_container(cm, c) for cm, c in zip(cms, obs['container_list'])}
    ag.packer.cms = cms
    out = []
    for strict in (True, False):
        ag.packer.collect = []
        try:
            ag.packer.best_for_bag(st, bag, strict)
            alts = sorted(ag.packer.collect, key=lambda t: t[0])
        finally:
            ag.packer.collect = None
        seen = []
        for c, cidx, o, cd, dims in alts:
            key = (cidx, o, round(cd['xc'], 2), round(cd['yc'], 2), round(cd['zc'], 2))
            if key in seen:
                continue
            seen.append(key)
            out.append(dict(cost=round(float(c), 3), container=int(cidx), orient=int(o), x=cd['xc'], y=cd['yc'], z=cd['zc'],
                            frac=cd.get('frac'), strict=strict))
            if len(out) >= topn:
                break
        if out:
            break
    return out


def check(item_idx, cidx, orient, x, y, z=None):
    """Analytic check of a manual placement with the agent's validator replica. z=None -> drop to resting height."""
    ag = STATE['agent']
    mod = sys.modules['agent_mod']
    obs = STATE['obs']
    pool = obs['pool_list']
    if not pool:
        return dict(valid=False, reason='pool empty')
    bag = pool[min(item_idx, len(pool) - 1)]
    cms = [mod.CM(c) for c in obs['container_list']]
    cm = cms[cidx]
    st = {m.idx: mod.boxes_from_container(m, c) for m, c in zip(cms, obs['container_list'])}
    ag.packer.cms = cms
    dims = mod.oriented_dims(bag['length'], bag['width'], bag['height'], orient)
    hx, hy, hz = [d / 2 for d in dims]
    out = dict(valid=False)
    res = ag.packer.check_planned(st, bag, (cidx, orient, x, y))
    if res is not None:
        cd = res[3]
        out.update(valid=True, z=cd['zc'], frac=cd['frac'], level=cd['level'])
        return out
    # explain why not (at the dropped height if z is None)
    B = st[cm.idx]
    zc = z if z is not None else cm.floor + 0.007 + hz
    if z is None and len(B):
        ov = ((x + hx > B.lo[:, 0]) & (x - hx < B.hi[:, 0]) & (y + hy > B.lo[:, 1]) & (y - hy < B.hi[:, 1]))
        if ov.any():
            zc = max(zc, float(B.hi[ov, 2].max()) + 0.002 + hz)
    reasons = []
    incl = cm.included(np.array([x]), np.array([y]), np.array([zc]), hx, hy, hz)[0]
    if not incl:
        reasons.append('outside container (walls / ceiling / cut corner)')
    if len(B):
        lo = np.array([x - hx, y - hy, zc - hz]); hi = np.array([x + hx, y + hy, zc + hz])
        if (((lo[None] < B.hi - 1e-3) & (hi[None] > B.lo + 1e-3)).all(1)).any():
            reasons.append('overlaps another bag')
    if incl:
        pok = ag.packer.path_ok(cm, B, np.array([x]), np.array([y]), np.array([zc]), hx, hy, hz)[0]
        if not pok:
            reasons.append('insertion path blocked (need 1.5 cm clearance from the opening to the spot)')
    out.update(z=float(zc), reason='; '.join(reasons) or 'support too small / too close to ceiling / soft-hard rule')
    return out


def agent_step():
    ag = STATE['agent']
    obs = dict(STATE['obs'])
    for k in ('shm_name', 'shm_shape', 'shm_dtype'):
        obs.pop(k, None)
    a = ag.policy(obs)
    return do_step(dict(item_idx=a['item_idx'], container_idx=a['container_idx'], place_pos=[float(v) for v in a['place_pos']],
                        orientation=a['orientation']))


class H(BaseHTTPRequestHandler):
    def _send(self, obj, code=200, ctype='application/json'):
        data = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == '/':
                html = open(os.path.join(os.path.dirname(__file__), 'index.html'), encoding='utf-8').read()
                self._send(html.encode('utf-8'), ctype='text/html; charset=utf-8')
            elif u.path == '/api/state':
                self._send(view())
            elif u.path == '/api/suggest':
                self._send(suggest(int(q.get('item', ['0'])[0])))
            elif u.path == '/api/check':
                zz = q.get('z', [None])[0]
                self._send(check(int(q['item'][0]), int(q['cont'][0]), int(q['ori'][0]), float(q['x'][0]), float(q['y'][0]),
                                 None if zz in (None, '', 'null') else float(zz)))
            elif u.path == '/api/agent_step':
                agent_step()
                self._send(view())
            elif u.path == '/api/reset':
                start()
                self._send(view())
            elif u.path == '/api/undo':
                h = STATE['history'][:-1]
                start(h)
                self._send(view())
            elif u.path == '/api/history':
                self._send(STATE['history'])
            else:
                self._send({'error': 'not found'}, 404)
        except Exception:
            self._send({'error': traceback.format_exc()}, 500)

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(n) or b'{}')
        try:
            if self.path == '/api/step':
                do_step(body)
                self._send(view())
            else:
                self._send({'error': 'not found'}, 404)
        except Exception:
            self._send({'error': traceback.format_exc()}, 500)


def main():
    global ARGS, CFG
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/sample_config.json')
    ap.add_argument('--task', default='001')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--agent', default='agents/extreme_point')
    ap.add_argument('--agent-order', action='store_true', help='for offline tasks use the agent order search (slow); default: natural order')
    ARGS = ap.parse_args()
    CFG = json.load(open(ARGS.config))[ARGS.task]
    start()
    print(f'Visualizer running on http://localhost:{ARGS.port}  (task {ARGS.task})', flush=True)
    HTTPServer(('127.0.0.1', ARGS.port), H).serve_forever()


if __name__ == '__main__':
    main()
