"""Benchmark the automatic bass tab against the user's imported tabs.

Every imported tab (.bass-tab-imported.json, already synced to the recording) is taken
as the right answer for its song. The automatic tab is made the way the app makes it
(app.auto_notes, app.fit_learning) from each song's "(bass only)" track, and scored:

  found / real      recall and precision: same pitch, onset within 50 ms (one to one)
  any octave        the same, with the pitch allowed to be off by octaves
  onset             the same, any pitch (is a note there at all?)
  timing            median |onset difference| of matched notes, and the median signed one
  fingering         matched notes on the same string and fret (fingered in the tab's tuning)
  fingering (tab)   the tab's own pitches fingered by the app vs the tab (fingering alone)

Only where the imported tab has bars (gaps in it are skipped) and the song's bass stem
isn't silent (app.without_silence). Learning (note settings, note models, fingering
costs) is fitted on the other fold of songs, as learn_from_imports does with all of them,
so every song is scored held out.

  python tools/tab_bench.py prepare            decode songs, run Basic Pitch once (cache)
  python tools/tab_bench.py learn              fit learning per fold (cache)
  python tools/tab_bench.py run [--save NAME]  score every song held out
  python tools/tab_bench.py compare A B        two saved runs side by side

Reads ~/Music/Track Splitter (or --library); writes only to --cache.
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
LIBRARY = Path.home() / "Music" / "Track Splitter"
CACHE = Path(tempfile.gettempdir()) / "track-splitter-bench"
FOLDS = 2
TOL = 0.05


def load_app():
    """app and tab, with app's library pointed away from the real one (it is only read here)."""
    os.environ.setdefault("TRACK_SPLITTER_THREADS", "4")
    import app
    import tab
    app.OUTPUT_DIR = CACHE / "library-unused"
    app.LEARNED_FILE = CACHE / "library-unused" / ".learned-from-tabs.json"
    return app, tab


def song_key(song_dir):
    return hashlib.sha1(str(song_dir.relative_to(LIBRARY)).encode()).hexdigest()[:12]


# --- prepare -----------------------------------------------------------------------

def covered_spans(lines):
    """Where the imported tab has bars of a normal length (gaps it doesn't cover are one long bar)."""
    starts = sorted({b for _, _, bars in lines for b in bars} | {lines[-1][1]}) if lines else []
    if len(starts) < 3:
        return []
    lengths = np.diff(starts)
    usual = float(np.median(lengths))
    return [[a, b] for a, b, n in zip(starts, starts[1:], lengths) if n < 1.8 * usual]


def prepare(args):
    app, tab = load_app()
    CACHE.mkdir(parents=True, exist_ok=True)
    index = []
    for f in sorted(LIBRARY.glob(f"*/*/{app.IMPORT_FILE}")):
        song_dir = f.parent
        imported = app.read_import(song_dir)
        track = app.bass_track(song_dir)
        opens = imported and app.import_opens(imported)
        if not imported or not track or not opens:
            print("skip", song_dir.name)
            continue
        key = song_key(song_dir)
        index.append({"key": key, "song": str(song_dir.relative_to(LIBRARY)), "name": song_dir.name})
        meta_path = CACHE / f"{key}.json"
        if meta_path.exists() and not args.force:
            continue
        t0 = time.time()
        bass, song = app.tab_audio(song_dir, track)
        frames, onsets = tab._model_output(bass, lambda fraction: None)
        played = [n for n in app.unified_notes(imported) if not n[4]]
        heard = app.without_silence(played, bass)
        n = len(bass) // (tab.SAMPLERATE // 20)
        level = tab._levels(bass, n)
        quiet = (level < np.percentile(level, 95) - app.SILENT_DB).tolist()
        meta = {"song": str(song_dir.relative_to(LIBRARY)), "name": song_dir.name, "tuning": imported["tuning"],
                "opens": opens, "notes": [[x[0], x[1], opens[x[2]] + x[3], x[2], x[3], x[5] if len(x) > 5 else ""] for x in heard],
                "silent_dropped": len(played) - len(heard), "spans": covered_spans(imported.get("lines") or []),
                "quiet": quiet, "track": track.name, "bpm": imported.get("bpm")}
        np.savez(CACHE / f"{key}.npz", bass=bass.astype(np.float32), song=None if song is None else song.astype(np.float16),
                 frames=frames.astype(np.float16), onsets=onsets.astype(np.float16))
        meta_path.write_text(json.dumps(meta))
        print(f"{song_dir.name}: {len(heard)} notes ({len(played) - len(heard)} in silence), {time.time() - t0:.1f} s")
    (CACHE / "index.json").write_text(json.dumps(index, indent=1))
    print(len(index), "songs")


# --- songs in memory -----------------------------------------------------------------

def load_songs(names=None):
    index = json.loads((CACHE / "index.json").read_text())
    songs = []
    for k, item in enumerate(index):
        if names and not any(n.lower() in item["name"].lower() for n in names):
            continue
        meta = json.loads((CACHE / f"{item['key']}.json").read_text())
        z = np.load(CACHE / f"{item['key']}.npz", allow_pickle=True)
        song = z["song"]
        meta.update(key=item["key"], fold=k % FOLDS, bass=z["bass"], song=None if song.dtype == object else song.astype(np.float32),
                    frames=z["frames"].astype(np.float32), onsets=z["onsets"].astype(np.float32))
        songs.append(meta)
    return songs


def patch_model(tab, songs):
    """tab._model_output answers from the cache (Basic Pitch already ran on each bass track)."""
    by_id = {id(s["bass"]): (s["frames"], s["onsets"]) for s in songs}
    real = tab._model_output

    def cached(audio, on_progress):
        return by_id[id(audio)] if id(audio) in by_id else real(audio, on_progress)
    tab._model_output = cached


def fingers_of(s):
    """(open strings, [(time, pitch, string, fret)]) as learn_from_imports reads a tab."""
    return s["opens"], [(n[0], n[2], n[3], n[4]) for n in s["notes"]]


# --- learning per fold -----------------------------------------------------------------

def learn(args):
    app, tab = load_app()
    songs = load_songs()
    patch_model(tab, songs)
    for fold in range(FOLDS):
        train = [s for s in songs if s["fold"] != fold]
        t0 = time.time()
        examples = [{**tab.note_example(s["bass"], s["song"], [(n[0], n[2]) for n in s["notes"]]), "name": s["name"]}
                    for s in train]
        learned = app.fit_learning([fingers_of(s) for s in train], examples,
                                   lambda fraction=None, step=None: step and print(f"  fold {fold}: {step}", flush=True))
        learned["train"] = [s["name"] for s in train]
        (CACHE / f"learned-fold{fold}{args.tag}.json").write_text(json.dumps(learned))
        print(f"fold {fold}: {time.time() - t0:.0f} s", {k: {kk: vv for kk, vv in (v or {}).items() if kk not in ("models", "style", "settings")}
                                                        for k, v in learned.items() if k != "train"})


# --- scoring -----------------------------------------------------------------------------

def pairs(truth, det, ok, tol=None):
    """One-to-one matches [(truth index, det index)], closest onsets first, where ok(tp, dp)."""
    tol = TOL if tol is None else tol
    tt = np.array([t[0] for t in truth])
    cand = []
    for j, d in enumerate(det):
        lo, hi = np.searchsorted(tt, d[0] - tol), np.searchsorted(tt, d[0] + tol, "right")
        cand += [(abs(tt[i] - d[0]), i, j) for i in range(lo, hi) if ok(truth[i][2], d[2])]
    used_t, used_d, out = set(), set(), []
    for _, i, j in sorted(cand):
        if i not in used_t and j not in used_d:
            used_t.add(i)
            used_d.add(j)
            out.append((i, j))
    return out


def inside(spans, t):
    return any(a <= t < b for a, b in spans)


def classify(truth, det, matched):
    """Why each unmatched tab note was missed and each unmatched note shown is wrong."""
    mt, md = {i for i, _ in matched}, {j for _, j in matched}
    tt = np.array([t[0] for t in truth])
    ds = np.array([d[0] for d in det]) if det else np.zeros(0)
    miss, fake = Counter(), Counter()
    for i, (t, e, p) in enumerate((x[0], x[1], x[2]) for x in truth):
        if i in mt:
            continue
        near = [det[j] for j in np.nonzero(np.abs(ds - t) < TOL)[0]] if len(ds) else []
        holding = [d for d in det if d[0] < t - TOL and d[1] > t - 0.02 and d[2] == p]
        late = [det[j] for j in np.nonzero(np.abs(ds - t) < 0.15)[0] if det[j][2] == p] if len(ds) else []
        if any((d[2] - p) % 12 == 0 for d in near):
            miss["octave"] += 1
        elif holding:
            miss["re-pluck (held over)"] += 1
        elif late:
            miss["timing 50-150 ms"] += 1
        elif near:
            miss["other pitch"] += 1
        else:
            miss["nothing heard"] += 1
    for j, (s, e, p) in enumerate((d[0], d[1], d[2]) for d in det):
        if j in md:
            continue
        near = [truth[i] for i in np.nonzero(np.abs(tt - s) < TOL)[0]]
        sounding = [x for x in truth if x[0] < s - TOL and x[1] > s + 0.02]
        before = [x for x in truth if x[0] < s - TOL and x[0] > s - 1.0]
        late = [truth[i] for i in np.nonzero(np.abs(tt - s) < 0.15)[0] if truth[i][2] == p]
        if any((x[2] - p) % 12 == 0 for x in near):
            fake["octave"] += 1
        elif late and all(i in mt for i in np.nonzero(np.abs(tt - s) < 0.15)[0] if truth[i][2] == p):
            fake["double (note already found)"] += 1
        elif late:
            fake["timing 50-150 ms"] += 1
        elif near:
            fake["other pitch"] += 1
        elif sounding and sounding[-1][2] == p or (not sounding and before and before[-1][2] == p):
            fake["extra re-pluck"] += 1
        elif sounding and any((x[2] - p) % 12 == 0 or abs(x[2] - p) in (7, 19) for x in sounding):
            fake["overtone of held note"] += 1
        elif sounding:
            fake["leakage / other pitch"] += 1
        else:
            fake["ghost (tab rests)"] += 1
    return miss, fake


def finger_map(tab, notes, tuning, opens, style):
    """{note index: (open pitch of its string, fret)} as the app fingers notes in a tuning."""
    key = tuning if tuning in tab.TUNINGS else "_bench"
    if key == "_bench":
        tab.TUNINGS[key] = ("bench", list(opens), [str(o) for o in opens])
    strings = tab.TUNINGS[key][1]
    return {i: (strings[s], f) for i, s, f in tab.fingering(notes, key, style)}


def score_song(app, tab, s, learned, split=True):
    settings = dict(((learned or {}).get("notes") or {}).get("settings") or tab.NOTES)
    models = ((learned or {}).get("note_models") or {}).get("models")
    style = ((learned or {}).get("fingering") or {}).get("style")
    t0 = time.time()
    notes, notes_split, grid = app.auto_notes(s["bass"], s["song"], settings, models)
    return {**evaluate(tab, s, notes_split if split else notes, style), "bpm": grid.get("bpm"), "seconds": round(time.time() - t0, 1)}


def evaluate(tab, s, det_all, style=None, fingers=True):
    """The scores of notes [(start, end, pitch, amplitude)] shown for song s (fingering too if `fingers`)."""
    spans = s["spans"]
    quiet = s["quiet"]
    loud = lambda t: not quiet[min(len(quiet) - 1, max(0, int(t * 20)))]
    truth = [n for n in s["notes"] if inside(spans, n[0])]
    keep = [k for k, d in enumerate(det_all) if inside(spans, d[0]) and loud(d[0])]
    det = [det_all[k] for k in keep]
    exact = pairs(truth, det, lambda a, b: a == b)
    octave = pairs(truth, det, lambda a, b: (a - b) % 12 == 0)
    onset = pairs(truth, det, lambda a, b: True)
    dts = [det[j][0] - truth[i][0] for i, j in exact]
    same = tsame = []
    if fingers:
        fmap = finger_map(tab, det_all, s["tuning"], s["opens"], style)
        same = [fmap.get(keep[j]) == (s["opens"][truth[i][3]], truth[i][4]) for i, j in exact]
        tnotes = [(n[0], n[1], n[2], 1.0) for n in s["notes"]]
        tmap = finger_map(tab, tnotes, s["tuning"], s["opens"], style)
        tsame = [tmap.get(k) == (s["opens"][n[3]], n[4]) for k, n in enumerate(s["notes"])]
    miss, fake = classify(truth, det, exact)
    return {"name": s["name"], "fold": s["fold"], "truth": len(truth), "det": len(det), "hit": len(exact),
            "hit_octave": len(octave), "hit_onset": len(onset),
            "abs_dt": float(np.median(np.abs(dts))) if dts else None, "dt": float(np.median(dts)) if dts else None,
            "finger_same": int(sum(same)), "finger_n": len(same), "tab_finger_same": int(sum(tsame)), "tab_finger_n": len(tsame),
            "miss": dict(miss), "fake": dict(fake)}


def summary(rows):
    tot = lambda k: sum(r[k] for r in rows)
    t, d = tot("truth"), tot("det")
    dts = [r["abs_dt"] for r in rows if r["abs_dt"] is not None]
    f1 = lambda h: 2 * h / max(1, t + d)
    return {"songs": len(rows), "truth": t, "det": d,
            "recall": tot("hit") / max(1, t), "precision": tot("hit") / max(1, d), "f1": f1(tot("hit")),
            "recall_octave": tot("hit_octave") / max(1, t), "precision_octave": tot("hit_octave") / max(1, d),
            "recall_onset": tot("hit_onset") / max(1, t), "precision_onset": tot("hit_onset") / max(1, d),
            "f1_song_median": float(np.median([2 * r["hit"] / max(1, r["truth"] + r["det"]) for r in rows])),
            "abs_dt_ms": 1000 * float(np.median(dts)) if dts else None,
            "fingering": tot("finger_same") / max(1, tot("finger_n")),
            "tab_fingering": tot("tab_finger_same") / max(1, tot("tab_finger_n")),
            "miss": dict(sum((Counter(r["miss"]) for r in rows), Counter())),
            "fake": dict(sum((Counter(r["fake"]) for r in rows), Counter()))}


def print_rows(rows):
    print(f"{'song':44} {'tab':>5} {'shown':>5} {'found':>6} {'real':>6} {'F1':>5} {'8va F1':>6} {'dt ms':>6} {'finger':>6}")
    for r in sorted(rows, key=lambda r: 2 * r["hit"] / max(1, r["truth"] + r["det"])):
        f1 = 2 * r["hit"] / max(1, r["truth"] + r["det"])
        f1o = 2 * r["hit_octave"] / max(1, r["truth"] + r["det"])
        print(f"{r['name'][:44]:44} {r['truth']:5} {r['det']:5} {r['hit'] / max(1, r['truth']):6.1%} {r['hit'] / max(1, r['det']):6.1%}"
              f" {f1:5.3f} {f1o:6.3f} {1000 * (r['abs_dt'] or 0):6.0f} {r['finger_same'] / max(1, r['finger_n']):6.1%}")


def print_summary(sm, label=""):
    print(f"\n{label} {sm['songs']} songs, {sm['truth']} tab notes, {sm['det']} shown")
    print(f"  found {sm['recall']:.1%}  real {sm['precision']:.1%}  F1 {sm['f1']:.3f} (median song {sm['f1_song_median']:.3f})")
    print(f"  any octave: found {sm['recall_octave']:.1%} real {sm['precision_octave']:.1%};"
          f"  onset only: found {sm['recall_onset']:.1%} real {sm['precision_onset']:.1%}")
    print(f"  timing {sm['abs_dt_ms']:.0f} ms;  fingering {sm['fingering']:.1%} of found notes, {sm['tab_fingering']:.1%} of the tab's own")
    print("  missed:", ", ".join(f"{k} {v}" for k, v in sorted(sm["miss"].items(), key=lambda kv: -kv[1])))
    print("  wrong: ", ", ".join(f"{k} {v}" for k, v in sorted(sm["fake"].items(), key=lambda kv: -kv[1])))


def run(args):
    app, tab = load_app()
    songs = load_songs(args.songs)
    patch_model(tab, songs)
    learned = {}
    for fold in range(FOLDS):
        path = CACHE / f"learned-fold{fold}{args.tag}.json"
        learned[fold] = None if args.builtin or not path.exists() else json.loads(path.read_text())
        if not args.builtin and learned[fold] is None:
            print(f"(no learning for fold {fold}: run learn first; using built-in settings)")
    rows = []
    for s in songs:
        rows.append(score_song(app, tab, s, learned[s["fold"]], split=not args.unsplit))
        if args.verbose:
            print(rows[-1]["name"], rows[-1]["hit"], rows[-1]["truth"], rows[-1]["det"], flush=True)
    print_rows(rows)
    sm = summary(rows)
    print_summary(sm, args.save or "")
    if args.save:
        (CACHE / f"run-{args.save}.json").write_text(json.dumps({"summary": sm, "rows": rows}))


def compare(args):
    a, b = (json.loads((CACHE / f"run-{n}.json").read_text()) for n in (args.a, args.b))
    sa, sb = a["summary"], b["summary"]
    for k in ("recall", "precision", "f1", "f1_song_median", "recall_octave", "precision_octave", "recall_onset",
              "precision_onset", "abs_dt_ms", "fingering", "tab_fingering"):
        print(f"{k:18} {sa[k]:8.3f} {sb[k]:8.3f}  {sb[k] - sa[k]:+.3f}")
    rb = {r["name"]: r for r in b["rows"]}
    f1 = lambda r: 2 * r["hit"] / max(1, r["truth"] + r["det"])
    print()
    for r in sorted(a["rows"], key=lambda r: f1(rb[r["name"]]) - f1(r) if r["name"] in rb else 0):
        if r["name"] in rb:
            print(f"{r['name'][:44]:44} {f1(r):.3f} -> {f1(rb[r['name']]):.3f}")


def main():
    global CACHE, LIBRARY
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["prepare", "learn", "run", "compare"])
    ap.add_argument("a", nargs="?")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--library", type=Path, default=LIBRARY)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--force", action="store_true", help="prepare: redo songs already cached")
    ap.add_argument("--tag", default="", help="learn/run: a name for this learning (several kept side by side)")
    ap.add_argument("--builtin", action="store_true", help="run: tab.NOTES, no learning")
    ap.add_argument("--unsplit", action="store_true", help="run: notes as heard (the Split repeated notes switch off)")
    ap.add_argument("--save", help="run: keep the result under this name")
    ap.add_argument("--songs", nargs="*", help="run: only songs whose names contain these")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    LIBRARY, CACHE = args.library.expanduser(), args.cache.expanduser()
    {"prepare": prepare, "learn": learn, "run": run, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
