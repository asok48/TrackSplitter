import sys, os, numpy as np
from gamecheck_replay import *

def heard_variant(r, a, lo_ms=40, hi_ms=200, pre_ms=20, clear=0.1, only_unclear=False, only_hard=True):
    sr = r['sample_rate']; F = r['frame_fields']; ic, it, ir = F.index('ctx'), F.index('t'), F.index('raw')
    end_of = lambda f: int(round(f[ic]*sr)) - r['first_frame']
    ft = np.array([f[it] for f in r['frames']]); fe = np.array([end_of(f) for f in r['frames']])
    def end_at(t):
        i = int(np.clip(np.searchsorted(ft, t), 1, len(ft) - 1)); i = i if abs(ft[i] - t) < abs(ft[i-1] - t) else i - 1
        return int(fe[i]) if abs(ft[i] - t) < 0.015 else None
    attacks = sorted(set(x for x in (end_at(p['at']) for p in (e['p'] for e in r['events'] if e['e']=='pluck') if not only_hard or not p.get('soft')) if x is not None))
    out = []; j = 0; extra = 0
    for f in r['frames']:
        e = end_of(f); out.append((f[it], f[ir]))
        while j + 1 < len(attacks) and attacks[j+1] <= e: j += 1
        if not attacks or attacks[j] > e: continue
        since = (e - attacks[j]) / sr * 1000
        if not (lo_ms <= since <= hi_ms): continue
        if only_unclear and f[ir] is not None: continue
        s0 = attacks[j] - int(pre_ms/1000*sr)
        if s0 < 0 or e > len(a): continue
        m = raw_of(yin(a[s0:e], sr, clear=clear))
        if m is not None and m != f[ir]:
            out.append((f[it], m)); extra += 1
    run.extra = getattr(run, 'extra', 0) + extra
    return out, extra

def run(variant, names, shifts=(0, 2, -1, 1, -2)):
    tot = {s: 0 for s in shifts}; base = {s: 0 for s in shifts}
    for name in names:
        d = os.path.join(LIB, name); r = load(d); a = load_audio(d, r['sample_rate'])
        hv, extra = heard_variant(r, a, **variant); hr = recorded_heard(r)
        row = []
        for s in shifts:
            b = judge_run(r, hr, shift=s)[0]; v = judge_run(r, hv, shift=s)[0]
            base[s] += b; tot[s] += v; row.append(f'{b}->{v}')
        print(f"  {name[:16]} {r['title'][:16]:16} judged {r['judged']:4}  " + '  '.join(f'{s:+d}:{x}' for s, x in zip(shifts, row)))
    print('extra readings', getattr(run, 'extra', 0))
    print('TOTAL', '  '.join(f'{s:+d}: {base[s]}->{tot[s]} ({tot[s]-base[s]:+d})' for s in shifts))

if __name__ == '__main__':
    names = [n for n in sorted(os.listdir(LIB)) if load(os.path.join(LIB, n))['judged'] >= 100 and 'Outside' not in n]
    import ast
    variant = ast.literal_eval(sys.argv[1]) if len(sys.argv) > 1 else {}
    print(variant); run(variant, names)
