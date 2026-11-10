"""Benchmark how recorded takes become notes (app.take_notes) against what was played.

Two kinds of right answers:
  labelled   the user's string tests in <library>/.bass-recordings (every pluck labelled
             with its time, string and fret: string-test-onsets.json): exact.
  takes      the user's real takes (<song>/.my-bass/take-*.flac, placed by the draft's
             "takes") over songs with an imported tab: the tab's notes in the take's
             stretch. Approximate: the user may play something else than the tab.

Scores as tools/tab_bench.py: found / real (same pitch, onset within 50 ms, one to one),
any octave, onset only, timing, and why notes were missed or are wrong.

  python tools/take_bench.py [--save NAME] [--tol 0.05]
  python tools/take_bench.py compare A B

Reads the library (or --library); writes only to --cache.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tab_bench as B   # noqa: E402  (pairs, classify, load_app)


def labelled(app, tab, lib):
    """[(name, take audio, truth [(start, end, pitch)])] from the string tests."""
    rec = lib / ".bass-recordings"
    plucks = json.loads((rec / "string-test-onsets.json").read_text())
    out = []
    for name in sorted({p["take"] for p in plucks}):
        flac = rec / name.replace(".wav", ".flac")
        if not flac.exists():
            continue
        mine = [p for p in plucks if p["take"] == name]
        audio = app.decode(flac, tab.SAMPLERATE, channels=2)[mine[0]["ch"]]
        times = sorted((p["t"], p["midi"]) for p in mine)
        truth = [(t, times[k + 1][0] if k + 1 < len(times) else t + 0.5, m) for k, (t, m) in enumerate(times)]
        out.append((f"labelled {mine[0]['tag']}", audio, truth, None))
    return out


def real_takes(app, tab, lib):
    """[(name, take audio, truth from the imported tab, the take)] for takes with audio over imported tabs."""
    out = []
    for f in sorted(lib.glob(f"*/*/{app.MINE_FILE}")):
        song_dir = f.parent
        mine, imported = app.read_mine(song_dir), app.read_import(song_dir)
        opens = imported and app.import_opens(imported)
        if not mine or not opens:
            continue
        notes = sorted((n[0], n[1], opens[n[2]] + n[3]) for n in app.unified_notes(imported) if not n[4])
        for k, t in enumerate(mine.get("takes", [])):
            path = song_dir / app.MY_BASS_DIR / t.get("audio", "")
            if not t.get("audio") or not path.exists():
                continue
            audio = app.decode(path, tab.SAMPLERATE, channels=1)[0]
            a, b = t["start"], t["start"] + len(audio) / tab.SAMPLERATE
            # the take's own times; pitches as played (a raised song is played a half step up)
            truth = [(s - a, e - a, p + t.get("shift", 0)) for s, e, p in notes if a <= s < b]
            out.append((f"{song_dir.name[:24]} take {k + 1}", audio, truth, t))
    return out


def score(name, truth, det, played_span=None):
    """tab_bench-style counts for one take. played_span: only score where the take has
    notes (a take recorded over a longer stretch than the user played)."""
    truth = [(s, e, p) for s, e, p in truth]
    det = [(s, e, p) for s, e, p, *_ in det]
    if played_span and det:
        lo, hi = det[0][0] - 0.5, det[-1][0] + 0.5
        truth = [x for x in truth if lo <= x[0] <= hi]
    exact = B.pairs(truth, det, lambda a, b: a == b)
    octave = B.pairs(truth, det, lambda a, b: (a - b) % 12 == 0)
    onset = B.pairs(truth, det, lambda a, b: True)
    dts = [det[j][0] - truth[i][0] for i, j in onset]
    miss, fake = B.classify(truth, det, exact)
    return {"name": name, "truth": len(truth), "det": len(det), "hit": len(exact), "hit_octave": len(octave),
            "hit_onset": len(onset), "dt": float(np.median(dts)) if dts else None,
            "abs_dt": float(np.median(np.abs(np.array(dts) - np.median(dts)))) if dts else None,
            "miss": dict(miss), "fake": dict(fake)}


def show(rows, label):
    print(f"\n{label}")
    print(f"{'take':36} {'played':>6} {'shown':>6} {'found':>6} {'real':>6} {'F1':>5} {'8va F1':>6} {'onset F1':>8} {'lag ms':>7}")
    for r in rows:
        f = lambda h: 2 * h / max(1, r["truth"] + r["det"])
        print(f"{r['name'][:36]:36} {r['truth']:6} {r['det']:6} {r['hit'] / max(1, r['truth']):6.1%} {r['hit'] / max(1, r['det']):6.1%}"
              f" {f(r['hit']):5.3f} {f(r['hit_octave']):6.3f} {f(r['hit_onset']):8.3f} {1000 * (r['dt'] or 0):7.0f}")
    t, d, h = (sum(r[k] for r in rows) for k in ("truth", "det", "hit"))
    print(f"{'all':36} {t:6} {d:6} {h / max(1, t):6.1%} {h / max(1, d):6.1%} {2 * h / max(1, t + d):5.3f}")
    miss = sum((Counter(r["miss"]) for r in rows), Counter())
    fake = sum((Counter(r["fake"]) for r in rows), Counter())
    print("  missed:", ", ".join(f"{k} {v}" for k, v in miss.most_common()))
    print("  wrong: ", ", ".join(f"{k} {v}" for k, v in fake.most_common()))
    return {"truth": t, "det": d, "hit": h, "f1": 2 * h / max(1, t + d)}


def run(args):
    app, tab = B.load_app()
    lib = args.library.expanduser()
    out = {}
    for label, items in (("labelled string tests", labelled(app, tab, lib)), ("real takes vs imported tabs", real_takes(app, tab, lib))):
        rows = []
        for name, audio, truth, take in items:
            det = app.take_notes(audio)
            if take is not None and args.lag:   # the take's notes later than the tab overall: line them up first
                pre = score(name, truth, det, True)
                det = [(s - (pre["dt"] or 0), e - (pre["dt"] or 0), p, *r) for s, e, p, *r in det]
            rows.append(score(name, truth, det, played_span=take is not None))
        out[label] = {"summary": show(rows, label), "rows": rows}
    if args.save:
        Path(args.cache).mkdir(parents=True, exist_ok=True)
        (Path(args.cache) / f"take-{args.save}.json").write_text(json.dumps(out))


def compare(args):
    a, b = (json.loads((Path(args.cache) / f"take-{n}.json").read_text()) for n in (args.a, args.b))
    for label in a:
        print(label)
        rb = {r["name"]: r for r in b[label]["rows"]}
        f = lambda r: 2 * r["hit"] / max(1, r["truth"] + r["det"])
        for r in a[label]["rows"]:
            if r["name"] in rb:
                print(f"  {r['name'][:36]:36} {f(r):.3f} -> {f(rb[r['name']]):.3f}")
        print(f"  {'all':36} {a[label]['summary']['f1']:.3f} -> {b[label]['summary']['f1']:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", nargs="?", default="run", choices=["run", "compare"])
    ap.add_argument("a", nargs="?")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--library", type=Path, default=B.LIBRARY)
    ap.add_argument("--cache", type=Path, default=B.CACHE)
    ap.add_argument("--tol", type=float, default=0.05, help="onset tolerance in seconds")
    ap.add_argument("--lag", action="store_true", default=True, help="real takes: remove the take's overall lag vs the tab first")
    ap.add_argument("--save")
    args = ap.parse_args()
    B.TOL = args.tol
    B.CACHE = args.cache.expanduser()
    {"run": run, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
