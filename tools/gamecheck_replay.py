"""Replay Encore's judging on recorded game checks (run.json + input.flac).

judge() is a Python copy of index.html's (single notes, bass). Plucks come from the recorded
pluck events; heard pitches from the recorded frames, or from re-reading input.flac with a
pitch reader variant. `shift` moves every tab note by that many semitones (the wrong-notes
control: notes the player didn't play).

Run from this folder with the app's venv: ../.venv/bin/python gamecheck_replay.py (reproduces each
game's hits within 0-3; older games recorded under earlier rules differ more). Variants of the pitch
reader: gamecheck_experiment.py "{'lo_ms': 40}" (see NOTES.md, Game screen, 2026-10-31).
"""
import json, os, subprocess, sys
import numpy as np

LIB = os.path.expanduser('~/Music/Track Splitter/.game-checks')
FFMPEG = None


def ffmpeg():
    global FFMPEG
    if not FFMPEG:
        import imageio_ffmpeg
        FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    return FFMPEG


def load_audio(d, sr):
    cache = os.path.join(os.path.expanduser('~/.cache/track-splitter-replay'), os.path.basename(d) + '.npy')
    if os.path.exists(cache):
        return np.load(cache)
    out = subprocess.run([ffmpeg(), '-v', 'quiet', '-i', os.path.join(d, 'input.flac'), '-f', 'f32le', '-ac', '1', '-ar', str(sr), '-'],
                         capture_output=True, check=True).stdout
    a = np.frombuffer(out, dtype=np.float32).copy()
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, a)
    return a


def yin(buf, sr, lo=28, hi=420, clear=0.1):
    q = max(1, round(sr / 11025))
    n = len(buf) // q
    x = buf[:n * q].reshape(n, q).mean(axis=1).astype(np.float64)
    rate = sr / q
    minT = int(rate // hi)
    maxT = min(int(np.ceil(rate / lo)), n // 2)
    w = n - maxT
    if maxT <= minT + 2 or w <= 0:
        return 0.0
    # difference function for t = 1..maxT over the first w samples
    xw = x[:w]
    d = np.empty(maxT + 1)
    d[0] = 0
    for t in range(1, maxT + 1):
        v = xw - x[t:t + w]
        d[t] = np.dot(v, v)
    cs = np.cumsum(d[1:])
    c = np.ones(maxT + 2)
    ts = np.arange(1, maxT + 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        c[1:maxT + 1] = np.where(cs > 0, d[1:] * ts / cs, 1)
    t = minT
    while t < maxT:
        if c[t] < 0.15:
            while t + 1 < maxT and c[t + 1] < c[t]:
                t += 1
            if c[t] > clear:
                return 0.0
            a, b, e = c[t - 1], c[t], c[t + 1]
            den = 2 * (a - 2 * b + e)
            bend = (a - e) / den if den else 0
            bend = max(-1, min(1, bend)) if np.isfinite(bend) else 0
            return rate / (t + bend)
        t += 1
    return 0.0


def raw_of(hz):
    if not hz:
        return None
    s = 12 * np.log2(hz / 440) + 69
    return int(round(s)) if abs(s - round(s)) <= 0.4 else None


def load(d):
    return json.load(open(os.path.join(d, 'run.json')))


def judge_run(r, heard_frames, shift=0, plucks_extra=(), dup=0.0, quieter=None, settle=0.12):
    """heard_frames: list of (now, raw) in frame order. Returns (hits, judged)."""
    early, late, perfect = r['windows']
    fix = r['input_calibration']
    sh = r.get('shift', 0)
    op = r['open']
    notes = r['notes']
    pitch = [op[n['string']] + n['fret'] + shift if n['fret'] != 'x' else None for n in notes]
    judged_set = [n.get('result') in ('hit', 'miss') for n in notes]
    pl_ev = [e['p'] for e in r['events'] if e['e'] == 'pluck'] + list(plucks_extra)
    pl_ev.sort(key=lambda p: p['at'])
    plucks, heard, levels = [], [], []
    result = [None] * len(notes)
    nxt = 0
    while nxt < len(notes) and not judged_set[nxt]:
        nxt += 1
    pi = 0
    PITCH_WAIT, MISS_AFTER = r['pitch_wait'], r['miss_after']
    for now, raw in heard_frames:
        while pi < len(pl_ev) and pl_ev[pi]['at'] <= now + 1e-9:
            p = dict(pl_ev[pi]); p['used'] = None; p['fin'] = bool(p.get('soft'))
            plucks.append(p); pi += 1
        for p in plucks:
            if not p['fin'] and now - p['at'] >= 0.06:
                p['fin'] = True; levels = (levels + [p['db']])[-40:]
        if raw is not None:
            heard.append((now - 0.05 - fix, raw - sh, now))
        quiet = sorted(levels)[len(levels) >> 1] - 15 if len(levels) >= 8 else -1e9
        k = nxt
        while k < len(notes) and notes[k]['start'] - early <= now:
            if result[k] or not judged_set[k]:
                k += 1; continue
            ns, pt = notes[k]['start'], pitch[k]

            def own(p):
                if pt is None:
                    return True
                for (_, m, at) in heard:
                    if m == pt and at >= p['at'] + (0 if p.get('soft') else 0.015) and at <= p['at'] + PITCH_WAIT + 0.02:
                        if not any(q is not p and q['at'] - p['at'] >= max(dup, 1e-9) and q['at'] < at - (0 if q.get('soft') else 0.015)
                                   and (quieter is None or q['at'] - p['at'] >= settle or q['db'] >= p['db'] - quieter) for q in plucks):
                            return True
                return False
            cands = [p for p in plucks if ns - early <= p['t'] <= ns + late and (p['used'] is None or abs(p['used'] - ns) < 0.03)
                     and p['db'] >= quiet and own(p)]
            match = None if pt is None else next((h for h in heard if h[1] == pt and ns - early <= h[0] <= ns + late + 0.15), None)
            if cands and (pt is None or match):
                pk = ([p for p in cands if p['t'] <= match[0] + 0.03] or [None])[-1] if match else None
                pk = pk or cands[0]
                pk['used'] = ns
                result[k] = 'hit'
            k += 1

        def waiting(k):
            return any(p['used'] is None and p['db'] >= quiet and now - p['at'] < PITCH_WAIT and notes[k]['start'] - early <= p['t'] <= notes[k]['start'] + late for p in plucks)
        while nxt < len(notes) and (result[nxt] or not judged_set[nxt] or (notes[nxt]['start'] + late + MISS_AFTER < now and not waiting(nxt))):
            if judged_set[nxt] and not result[nxt]:
                result[nxt] = 'miss'
            nxt += 1
        plucks = [p for p in plucks if p['t'] > now - 2]
        heard = [h for h in heard if h[0] > now - 2]
    hits = sum(1 for x in result if x == 'hit')
    judged = sum(1 for x in result if x in ('hit', 'miss'))
    return hits, judged, result


def recorded_heard(r):
    F = r['frame_fields']; it, ir = F.index('t'), F.index('raw')
    return [(f[it], f[ir]) for f in r['frames']]


if __name__ == '__main__':
    for name in sorted(os.listdir(LIB)):
        r = load(os.path.join(LIB, name))
        if r['judged'] < 100:
            continue
        h, j, _ = judge_run(r, recorded_heard(r))
        print(f"{name[:16]} {r['title'][:18]:18} recorded {r['hits']}/{r['judged']}  replay {h}/{j}")
