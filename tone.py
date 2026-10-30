"""Match the song's bass tone: an EQ curve that makes the user's live bass (a DI
through the interface) sound like the bass in a song, for the page's "Hear myself"
monitor.

Method, and the measurements behind each choice (the user's library, Oct 2026: 14 songs
with an HQ "bass only" track, 8 DI recordings of the user's bass):
- Tone = long-term spectrum in 1/3-octave bands (FFT 8192, Hann, 75% overlap), over the
  frames within 20 dB of the loud parts (rests don't count). A song's curve is stable:
  computed from each half of a song against the same DI, the two differ by 1.8 dB rms
  (median over 40 Hz-8 kHz). Comparing the same notes only (each frame's pitch by a
  harmonic sum; song and DI spectra per note, then the ratio over shared notes) was
  barely better (1.8 / 5.7 dB vs 1.8 / 6.1), so the plain spectrum is used.
- The DI is what varies: two of the user's takes were ~20 dB brighter at 8 kHz than the
  rest (a pick or the tone knob, probably), which moves the curve 6 dB rms. So the page
  can capture the tone as it is now ("Capture my tone", 45 s of playing); without one,
  every DI recording in the library is pooled (each brought to the same level first).
  How long: windows of the user's recordings against all of them, 90th percentile
  10 s 7.4 dB rms, 20 s 6.9, 30 s 6.2, 45 s 4.3, 60 s 4.1 (median ~2.9 from 30 s on).
  A song's own 20 s stretches differ from the whole song by 3-4 dB, so the song side
  uses the whole "bass only" track. In the page, a capture of a stretch of a song's
  bass came out within ~0.3 dB of the same stretch analysed here.
- Curve = song - DI, clamped to +-12 dB, held flat below 40 Hz (and at most +3 dB there)
  and above 8 kHz (a separated stem is mostly leakage up there; the DI is noise). Fitted
  with the browser's own filters (BiquadFilterNode: RBJ cookbook, shelves with S=1), so
  the page builds exactly the response fitted here; IIR filters add no buffering delay.
  Loudness is kept the same as heard (`trim_db`, A-weighted), so switching it on doesn't
  change the level. It first kept the raw energy the same, which is mostly the low end;
  the EQ cuts mids and highs, where the ear is most sensitive, so it sounded ~3 dB
  quieter (median of the 14 songs, up to 5.7; user: "hard to hear").
- Dynamics are not matched. Song basses look squashed harder than a DI (spread of 10 ms
  levels while playing, 90th-10th percentile: songs median ~8.6 dB, DI ~13; crest ~5.1
  vs ~6.5 dB), but a compressor with no look-ahead (DynamicsCompressorNode has a fixed
  6 ms one, so it would have been an AudioWorklet) didn't close that gap on the pooled
  DI at any threshold (0 to -15 dB under the playing level) or ratio (1.5-4:1): spread
  13.7 -> 13.2-15.0 dB (a slow release widens it), crest 6.4 -> 6.2. The gap is mostly
  playing and arrangement (note lengths, gaps), not something to copy on the monitor.
- Distortion is not matched either: a separated stem's grit can't be told apart from
  leakage.
"""

import numpy as np

SR = 44100
N = 8192
HOP = N // 4
BANDS = 1000 * 2 ** (np.arange(-15, 14) / 3)       # 31 Hz .. 16 kHz, 1/3 octave
FIT_LO, FIT_HI = 40, 8000                           # where the curve is measured
GUITAR_LO = 70                                      # a guitar's lowest note is E2, 82 Hz
LIMIT = 12                                          # dB either way
_f2 = BANDS ** 2
A_WEIGHT = 20 * np.log10(12194 ** 2 * _f2 ** 2 / ((_f2 + 20.6 ** 2) * np.sqrt((_f2 + 107.7 ** 2) * (_f2 + 737.9 ** 2)) * (_f2 + 12194 ** 2))) + 2.0
_freqs = np.fft.rfftfreq(N, 1 / SR)
_band_bins = [(_freqs >= c * 2 ** (-1 / 6)) & (_freqs < c * 2 ** (1 / 6)) for c in BANDS]


def _frames(x):
    n = max(0, (len(x) - N) // HOP)
    idx = np.arange(N)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx] * np.hanning(N)


def _band_power(x):
    """Band power per frame and frame power."""
    parts, levels = [], []
    step = HOP * 256
    for a in range(0, max(1, len(x) - N + 1), step):
        f = _frames(x[a:a + step + N - HOP])
        if not len(f):
            continue
        p = np.abs(np.fft.rfft(f)) ** 2
        parts.append(np.stack([p[:, m].mean(1) for m in _band_bins], 1))
        levels.append((f ** 2).mean(1))
    if not parts:
        return np.zeros((0, len(BANDS))), np.zeros(0)
    return np.concatenate(parts), np.concatenate(levels)


def profile(x):
    """A recording's tone: band levels (dB, playing frames only) and how much playing
    it has."""
    bp, lv = _band_power(x)
    if not len(lv):
        return None
    act = lv > np.percentile(lv, 90) * 10 ** -2.0
    return {"bands": [round(float(b), 2) for b in BANDS],
            "db": [round(float(v), 2) for v in 10 * np.log10(bp[act].mean(0) + 1e-20)],
            "seconds": round(float(act.sum() * HOP / SR), 1)}


def pool(profiles):
    """Several recordings as one: each brought to the same level (mean over 40 Hz-8 kHz),
    then averaged in dB, weighted by playing time."""
    sel = (BANDS >= FIT_LO) & (BANDS <= FIT_HI)
    w = np.array([p["seconds"] for p in profiles], float)
    db = np.array([np.array(p["db"]) - np.mean(np.array(p["db"])[sel]) for p in profiles])
    return {"bands": profiles[0]["bands"], "db": [round(float(v), 2) for v in (db * w[:, None]).sum(0) / w.sum()],
            "seconds": round(float(w.sum()), 1)}


def biquad(kind, f0, gain, q, sr, freqs):
    """Complex response of a WebAudio BiquadFilterNode (spec formulas)."""
    w0 = 2 * np.pi * f0 / sr
    A = 10 ** (gain / 40)
    cw, sw = np.cos(w0), np.sin(w0)
    if kind == "peaking":
        alpha = sw / (2 * q)
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind in ("lowpass", "highpass"):          # q in dB, as WebAudio takes it for these
        alpha = sw / (2 * 10 ** (q / 20))
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2] if kind == "lowpass" else [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    else:
        alpha = sw / 2 * np.sqrt(2)                    # S = 1
        k = 2 * np.sqrt(A) * alpha
        if kind == "lowshelf":
            b = [A * ((A + 1) - (A - 1) * cw + k), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - k)]
            a = [(A + 1) + (A - 1) * cw + k, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - k]
        else:
            b = [A * ((A + 1) + (A - 1) * cw + k), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - k)]
            a = [(A + 1) - (A - 1) * cw + k, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - k]
    z = np.exp(-1j * 2 * np.pi * np.asarray(freqs) / sr)
    return (b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z)


def response_db(filters, freqs, sr=48000):
    h = np.ones(len(freqs), complex)
    for f in filters:
        h *= biquad(f["type"], f["freq"], f["gain"], f.get("q", 1.0), sr, freqs)
    return 20 * np.log10(np.abs(h) + 1e-12)


def target(song_db, di_db, lo_hz=FIT_LO):
    """song - DI per band: level-matched, clamped, held flat outside lo_hz-8 kHz."""
    t = np.array(song_db) - np.array(di_db)
    sel = (BANDS >= lo_hz) & (BANDS <= FIT_HI)
    di_heard = 10 ** ((np.array(di_db) + A_WEIGHT) / 10) * sel
    t -= 10 * np.log10(np.sum(di_heard * 10 ** (t / 10)) / np.sum(di_heard))   # same loudness on the DI
    t = np.clip(t, -LIMIT, LIMIT)
    lo, hi = np.argmax(sel), len(sel) - 1 - np.argmax(sel[::-1])
    t[:lo] = min(t[lo], 3.0)
    t[hi + 1:] = t[hi]
    return t


# Peaking filters every half octave (Q 1.41, one octave wide), plus shelves at the ends
# so the curve holds outside them.
FILTER_FREQS = 50 * 2 ** (np.arange(0, 15) / 2)    # 50 Hz .. 6.4 kHz


def fit(t, sr=48000):
    """Filter gains whose combined response best matches target `t` (per band), by
    Gauss-Newton on a 1/12-octave grid."""
    grid = 30 * 2 ** (np.arange(0, 10 * 12 + 1) / 12)
    grid = grid[grid < 15000]
    want = np.interp(np.log2(grid), np.log2(BANDS), t)
    filters = [{"type": "lowshelf", "freq": 40.0, "gain": 0.0}]
    filters += [{"type": "peaking", "freq": float(f), "q": 1.41, "gain": 0.0} for f in FILTER_FREQS]
    filters += [{"type": "highshelf", "freq": 8000.0, "gain": 0.0}]
    for _ in range(8):
        now = response_db(filters, grid, sr)
        jac = np.stack([response_db([dict(f, gain=1.0)], grid, sr) for f in filters], 1)
        step = np.linalg.lstsq(np.vstack([jac, 0.05 * np.eye(len(filters))]),
                               np.concatenate([want - now, np.zeros(len(filters))]), rcond=None)[0]
        for f, s in zip(filters, step):
            f["gain"] = float(np.clip(f["gain"] + s, -18, 18))
    for f in filters:
        f["gain"] = round(f["gain"], 2)
    got = response_db(filters, grid, sr)
    return filters, grid, want, got


def match(song, di, sr=48000, lo_hz=FIT_LO):
    """The tone match for one song: target curve and filters."""
    t = target(song["db"], di["db"], lo_hz)
    filters, grid, want, got = fit(t, sr)
    # Same loudness on the DI as without, as heard (A-weighted; the fit isn't exact and
    # the shelves act outside the measured range).
    band_got = np.interp(np.log2(BANDS), np.log2(grid), got)
    di_heard = 10 ** ((np.array(di["db"]) + A_WEIGHT) / 10)
    trim = -10 * np.log10(np.sum(di_heard * 10 ** (band_got / 10)) / np.sum(di_heard))
    sel = (grid >= lo_hz) & (grid <= FIT_HI)
    return {"target": [round(float(v), 2) for v in t], "filters": filters, "trim_db": round(float(trim), 2),
            "fit_error_db": round(float(np.sqrt(np.mean((got - want)[sel] ** 2))), 2),
            "fit": [round(float(v), 2) for v in band_got]}


# --- The song's tone over time, and distortion (guitar) -------------------------------
# User: "So Far So Fake starts with a clean tone and then changes to a distorted tone. I
# want my guitar to sound pretty close to the album." One EQ for the whole song averaged
# the clean and distorted parts, and an EQ can't make a clean guitar distorted.
#
# Sections: the guitar track in 1 s windows, each described by how noise-like its
# spectrum is (spectral flatness, 150 Hz-4 kHz: a distorted guitar's dense harmonics and
# intermodulation fill the gaps between notes' partials) and its spectral shape. An
# optimal segmentation (dynamic programming, a cost per section, sections >= 4 s) cuts
# the song where that changes; sections that sound alike are then grouped, so every
# chorus shares one tone measured from all of them. Measured on So Far So Fake's guitar
# track: flatness ~0.005 in the clean parts, ~0.2 in the distorted ones.
#
# Distortion: the page puts a drive stage in front of the EQ (`CHAIN`: a high-pass and a
# mid hump before the clipper, as overdrive pedals do, tanh clipping, a low-pass for the
# fizz). Each captured guitar is run through it offline at a range of drives
# (`drive_table`), giving the flatness and spectrum each drive makes of that guitar as
# the user plays it; a section gets the drive whose flatness is closest to its own, and
# its EQ is fitted against the driven guitar. Clean sections get no drive at all.

FLAT_LO, FLAT_HI = 150, 4000
_flat_bins = (_freqs >= FLAT_LO) & (_freqs < FLAT_HI)
WIN = 1.0                       # seconds per window
MIN_SECTION = 4                 # windows
SECTION_COST = 6.0              # what a new section must explain (in the features' units)
GROUP_DIST = 1.0                # sections closer than this share a tone
DRIVES = [0] + [round(float(v), 2) for v in 2 ** np.arange(0, 11.5, 0.5)]   # 1 .. 2048x the raw input
CLEAN_FLAT = 0.015              # a section below this (and not much above the guitar) stays clean
ROOM_MAX = 0.25                 # most side/mid power the room may make (-6 dB)
CHAIN = {                       # the page builds exactly this (BiquadFilterNode, WaveShaper)
    "pre": [{"type": "highpass", "freq": 120.0, "q": -3.01, "gain": 0.0},
            {"type": "peaking", "freq": 800.0, "q": 0.7, "gain": 6.0}],
    "post": [{"type": "lowpass", "freq": 6500.0, "q": -3.01, "gain": 0.0}],
}


def _stats(x):
    """Per frame: band power, power, spectral flatness."""
    bands, levels, flats = [], [], []
    step = HOP * 256
    for a in range(0, max(1, len(x) - N + 1), step):
        f = _frames(x[a:a + step + N - HOP])
        if not len(f):
            continue
        p = np.abs(np.fft.rfft(f)) ** 2
        bands.append(np.stack([p[:, m].mean(1) for m in _band_bins], 1))
        levels.append((f ** 2).mean(1))
        q = p[:, _flat_bins] + 1e-20
        flats.append(np.exp(np.log(q).mean(1)) / q.mean(1))
    if not bands:
        return np.zeros((0, len(BANDS))), np.zeros(0), np.zeros(0)
    return np.concatenate(bands), np.concatenate(levels), np.concatenate(flats)


def _features(bp, flat, sel):
    """One window's description: log flatness, and its shape (dB, 70 Hz-8 kHz, level
    removed) scaled so ~6 dB counts like a 10x change in flatness."""
    db = 10 * np.log10(bp[sel].mean(0) + 1e-20)
    use = (BANDS >= 70) & (BANDS <= FIT_HI)
    shape = (db[use] - db[use].mean()) / 6
    return np.concatenate([[2 * np.log10(np.median(flat[sel]) + 1e-6)], shape / np.sqrt(use.sum())])


def sections(x, side=None):
    """The song's tone over time: {"sections": [[start, end, group], ...] (seconds),
    "groups": [{"db", "flat", "seconds", "side_db"}, ...]}; None when there's too little
    playing. `side` is the track's (L - R) / 2 when it's stereo (x being (L + R) / 2):
    how wide each tone is (side_db, side vs mid power while playing)."""
    bp, lv, flat = _stats(x)
    side_lv = None if side is None else np.concatenate(
        [(_frames(side[a:a + HOP * 256 + N - HOP]) ** 2).mean(1)
         for a in range(0, max(1, len(side) - N + 1), HOP * 256)])
    if len(lv) < 10:
        return None
    act = lv > np.percentile(lv, 90) * 10 ** -2.0
    per = max(1, int(round(WIN * SR / HOP)))
    nw = len(lv) // per
    feats, playing = [], []
    for w in range(nw):
        sel = np.zeros(len(lv), bool)
        sel[w * per:(w + 1) * per] = act[w * per:(w + 1) * per]
        playing.append(sel.sum() >= per * 0.3)
        feats.append(_features(bp, flat, sel) if playing[-1] else None)
    idx = [w for w in range(nw) if playing[w]]
    if len(idx) < MIN_SECTION:
        return None
    F = np.array([feats[w] for w in idx])
    # Optimal segmentation of the playing windows: within-section squared deviation +
    # SECTION_COST per section, each at least MIN_SECTION windows.
    m = len(F)
    c1 = np.vstack([np.zeros(F.shape[1]), np.cumsum(F, 0)])
    c2 = np.concatenate([[0], np.cumsum((F ** 2).sum(1))])
    def cost(i, j):                                 # windows i..j-1
        s = c1[j] - c1[i]
        return c2[j] - c2[i] - (s @ s) / (j - i)
    best = np.full(m + 1, np.inf); best[0] = 0
    back = np.zeros(m + 1, int)
    for j in range(MIN_SECTION, m + 1):
        for i in range(0, j - MIN_SECTION + 1):
            if best[i] == np.inf:
                continue
            v = best[i] + cost(i, j) + SECTION_COST
            if v < best[j]:
                best[j], back[j] = v, i
    cuts, j = [], m
    while j > 0:
        cuts.append((back[j], j)); j = back[j]
    cuts.reverse()
    # Group sections that sound alike (closest pair first, by their mean features).
    segs = [{"wins": idx[i:j], "mean": F[i:j].mean(0), "n": j - i} for i, j in cuts]
    groups = [[k] for k in range(len(segs))]
    def gmean(g):
        n = sum(segs[k]["n"] for k in g)
        return sum(segs[k]["mean"] * segs[k]["n"] for k in g) / n
    while len(groups) > 1:
        d, a, b = min((np.linalg.norm(gmean(groups[a]) - gmean(groups[b])), a, b)
                      for a in range(len(groups)) for b in range(a + 1, len(groups)))
        if d > GROUP_DIST:
            break
        groups[a] += groups.pop(b)
    label = {}
    for gi, g in enumerate(sorted(groups, key=lambda g: min(g))):
        for k in g:
            label[k] = gi
    # Section bounds in time: a change sits between the last playing window of one
    # section and the first of the next; within that, at the frame where flatness
    # changes most (so a clean intro hands over right where the distortion comes in).
    lf = np.log10(flat + 1e-6)
    out = []
    for k, sg in enumerate(segs):
        start = 0.0 if k == 0 else None
        out.append([start, None, label[k]])
    for k in range(1, len(segs)):
        a = segs[k - 1]["wins"][-1] * per          # first frame of the window before
        b = (segs[k]["wins"][0] + 1) * per         # end of the first window after
        if label[k] != label[k - 1] and b - a > 2:
            jumps = [abs(np.median(lf[max(a, t - per // 2):t]) - np.median(lf[t:min(b, t + per // 2)]))
                     for t in range(a + 1, b - 1)]
            t = a + 1 + int(np.argmax(jumps))
        else:
            t = segs[k]["wins"][0] * per
        out[k][0] = out[k - 1][1] = round(t * HOP / SR, 2)
    out[-1][1] = round(len(x) / SR, 2)
    # Merge neighbours that ended up in the same group.
    merged = []
    for sct in out:
        if merged and merged[-1][2] == sct[2]:
            merged[-1][1] = sct[1]
        else:
            merged.append(sct)
    gs = []
    for gi in range(len(groups)):
        sel = np.zeros(len(lv), bool)
        for k, sg in enumerate(segs):
            if label[k] == gi:
                for w in sg["wins"]:
                    sel[w * per:(w + 1) * per] = act[w * per:(w + 1) * per]
        gs.append({"db": [round(float(v), 2) for v in 10 * np.log10(bp[sel].mean(0) + 1e-20)],
                   "flat": round(float(np.median(flat[sel])), 4), "seconds": round(float(sel.sum() * HOP / SR), 1)})
        if side_lv is not None and len(side_lv) == len(lv):
            gs[-1]["side_db"] = round(float(10 * np.log10(side_lv[sel].sum() / (lv[sel].sum() + 1e-20) + 1e-6)), 1)
    return {"sections": merged, "groups": gs}


def _filter(x, filters, sr=SR):
    """x through BiquadFilterNodes (by their magnitude and phase, via the FFT)."""
    n = 1 << int(np.ceil(np.log2(len(x) + sr)))
    f = np.fft.rfftfreq(n, 1 / sr)
    h = np.ones(len(f), complex)
    for flt in filters:
        h *= biquad(flt["type"], flt["freq"], flt["gain"], flt.get("q", 1.0), sr, f)
    return np.fft.irfft(np.fft.rfft(x, n) * h, n)[:len(x)]


def loudness(db):
    """A-weighted level of a band spectrum (dB)."""
    return 10 * np.log10(np.sum(10 ** ((np.array(db) + A_WEIGHT) / 10)))


def drive_table(x):
    """What each drive in DRIVES does to this recording of the user's guitar (the raw
    input, as the monitor gets it): flatness and band spectrum (dB) while playing.
    Drive 0 is the clean guitar itself."""
    pre = _filter(x, CHAIN["pre"])
    rows = []
    for d in DRIVES:
        y = x if d == 0 else _filter(np.tanh(d * pre), CHAIN["post"])
        bp, lv, flat = _stats(y)
        act = lv > np.percentile(lv, 90) * 10 ** -2.0
        db = 10 * np.log10(bp[act].mean(0) + 1e-20)
        rows.append({"drive": d, "flat": round(float(np.median(flat[act])), 4),
                     "db": [round(float(v), 2) for v in db]})
    # How loud the guitar comes in while playing (dBFS, RMS of the playing frames), for
    # the page's noise gate.
    bp, lv, flat = _stats(x)
    rows[0]["play_db"] = round(float(10 * np.log10(np.median(lv[lv > np.percentile(lv, 90) * 0.01]) / np.mean(np.hanning(N) ** 2) + 1e-20)), 1)
    return rows


def pick_drive(table, flat):
    """The table row for a section of the song with this flatness: clean when the song
    is about as clean as the guitar, else the drive whose flatness is closest (in log)."""
    clean = table[0]
    if flat < max(CLEAN_FLAT, clean["flat"] * 2):
        return clean
    # A song's distorted guitar is usually several tracks of chords, noisier than one
    # guitar can get (So Far So Fake 0.21; the clean intro, as a stand-in DI, topped out
    # at 0.13 with drive 2048x, 0.10 at 512x): past 90% of what this guitar reaches, more
    # drive mostly brings up noise, so stop there.
    flat = min(flat, 0.9 * max(r["flat"] for r in table[1:]))
    return min(table[1:], key=lambda r: abs(np.log(r["flat"] + 1e-6) - np.log(flat + 1e-6)))


def match_sections(song, table, sr=48000, lo_hz=GUITAR_LO):
    """Per group of the song's sections: the drive, a make-up gain (the driven guitar as
    loud as the clean one, A-weighted) and the EQ (fitted against the driven guitar)."""
    clean_loud = loudness(table[0]["db"])
    out = []
    for g in song["groups"]:
        row = pick_drive(table, g["flat"])
        makeup = clean_loud - loudness(row["db"])
        m = match(g, {"db": list(np.array(row["db"]) + makeup)}, sr, lo_hz)
        out.append({"drive": row["drive"], "makeup_db": round(float(makeup), 2), "flat": g["flat"],
                    "guitar_flat": row["flat"], "seconds": g["seconds"], "room": room(g, row), **m})
    return out


# Room (user: "I want my clean sound to be as close to the original guitar in the song as
# possible"): with the EQ matched, what a clean guitar on a record still has and a DI
# hasn't is space. So Far So Fake's clean intro: side (L - R) only 3.6 dB under mid
# (L/R correlation 0.43; a DI is mono), the side 6 dB under in loud moments and 3.8 in
# quiet ones (tails filling the gaps), not a delayed copy of the mid (no echo). The page
# adds a stereo room (a decaying noise impulse response, different left and right) in
# parallel, as much as gives the song's side/mid ratio: with the room's output as loud as
# its input and uncorrelated left / right, side / mid = (W/2) / (D + W/2) for dry power D
# and wet power W. Capped at ROOM_MAX (some of a separated track's side is other parts
# leaking in), and only on clean parts: the distorted tone is liked as it is.
def room(group, row):
    """{"dry", "wet"} gains (amplitude, together as loud as the dry alone) for a group."""
    if row["drive"] or group.get("side_db") is None:
        return {"dry": 1.0, "wet": 0.0}
    r = min(10 ** (group["side_db"] / 10), ROOM_MAX)
    w = 2 * r / (1 - r)                             # wet / dry power
    return {"dry": round(float(1 / np.sqrt(1 + w)), 3), "wet": round(float(np.sqrt(w / (1 + w))), 3)}
