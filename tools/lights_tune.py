"""Tune lights.py's big moments and stabs against the user's ratings from the light check page.

Ratings are <song>/.light-ratings.json in the library (static/lights-check.html writes them). For each rated song the
whole song is read once (cached), then every combination of settings in GRID is run and scored:
- big moments: wanted = detected big moments rated yes, stabs rated "bigger", and spots marked missed; unwanted = big
  moments rated no. A detection within 0.6 s of a wanted spot finds it.
- stabs: wanted = stabs rated yes; unwanted = stabs rated no (within 0.15 s).
Score = found wanted - unwanted detections - 0.5 x new unrated detections (each one flashes on a song the user has
heard); the current settings are shown first. Run from this folder with the app's venv:
    ../.venv/bin/python lights_tune.py [library folder]
"""
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app      # noqa: E402  (decode, the library layout)
import lights   # noqa: E402

LIB = Path(sys.argv[1]) if len(sys.argv) > 1 else app.OUTPUT_DIR
CACHE = Path(os.path.expanduser("~/.cache/track-splitter-lights"))
GRID = {"jump": [0.12, 0.15, 0.2], "loud_pct": [0.5, 0.6, 0.7], "gap": [10.0, 15.0, 20.0], "per_s": [20.0, 30.0, 45.0],
        "stop_db": [10.0, 12.0, 15.0], "stab_fall": [5.0, 6.0, 8.0], "stab_rise": [4.0, 5.0, 7.0]}


def rated_songs():
    for f in sorted(LIB.glob("*/*/" + app.LIGHT_RATINGS)):
        marks = json.loads(f.read_text(encoding="utf-8")).get("marks", [])
        if marks:
            yield f.parent, marks


def mono(song_dir):
    CACHE.mkdir(parents=True, exist_ok=True)
    c = CACHE / (song_dir.parent.name + " - " + song_dir.name + ".npy")
    if c.exists():
        return np.load(c)
    mix = app.whole_song(song_dir)
    x = app.to_light_rate(mix)
    np.save(c, x)
    return x


def near(t, times, tol):
    return any(abs(t - x) <= tol for x in times)


def score(cues, marks):
    big, stab = cues["big"], [a for a in cues["accent"] if not near(a, cues["big"], 0.15)]
    want_big = [m["t"] for m in marks if (m["kind"] == "big" and m["verdict"] == "yes") or m["verdict"] in ("bigger", "missed")]
    no_big = [m["t"] for m in marks if m["kind"] == "big" and m["verdict"] == "no"]
    want_stab = [m["t"] for m in marks if m["kind"] == "stab" and m["verdict"] == "yes"]
    no_stab = [m["t"] for m in marks if m["kind"] == "stab" and m["verdict"] == "no"]
    rated = [m["t"] for m in marks]
    r = {"big_found": sum(near(t, big, 0.6) for t in want_big), "big_want": len(want_big),
         "big_bad": sum(near(t, no_big, 0.6) for t in big), "big_new": sum(not near(t, rated, 0.6) for t in big),
         "stab_found": sum(near(t, stab, 0.15) for t in want_stab), "stab_want": len(want_stab),
         "stab_bad": sum(near(t, no_stab, 0.15) for t in stab), "stab_new": sum(not near(t, rated, 0.15) for t in stab)}
    r["score"] = r["big_found"] - r["big_bad"] - 0.5 * r["big_new"] + 0.5 * (r["stab_found"] - r["stab_bad"]) - 0.1 * r["stab_new"]
    return r


def main():
    songs = [(d, m, lights.features(mono(d))) for d, m in rated_songs()]
    if not songs:
        print(f"No ratings yet in {LIB}: rate some songs on the Light check page (Encore › Settings › Light check).")
        return
    print(f"{len(songs)} rated songs: " + ", ".join(d.name for d, _, _ in songs))
    keys = list(GRID)
    combos = [dict(lights.BIG)] + [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    results = []
    for k, settings in enumerate(combos):
        tot = {}
        for _, marks, x in songs:
            for name, v in score(lights.decide(x, settings), marks).items():
                tot[name] = tot.get(name, 0) + v
        results.append((tot["score"], settings, tot))
        if k == 0:
            print("current:", settings, tot)
    results.sort(key=lambda r: -r[0])
    print("best 5:")
    for sc, settings, tot in results[:5]:
        print(f"  {sc:6.1f}", settings, {k: v for k, v in tot.items() if k != "score"})


if __name__ == "__main__":
    main()
