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


def target(song_db, di_db):
    """song - DI per band: level-matched, clamped, held flat outside 40 Hz-8 kHz."""
    t = np.array(song_db) - np.array(di_db)
    sel = (BANDS >= FIT_LO) & (BANDS <= FIT_HI)
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


def match(song, di, sr=48000):
    """The tone match for one song: target curve and filters."""
    t = target(song["db"], di["db"])
    filters, grid, want, got = fit(t, sr)
    # Same loudness on the DI as without, as heard (A-weighted; the fit isn't exact and
    # the shelves act outside the measured range).
    band_got = np.interp(np.log2(BANDS), np.log2(grid), got)
    di_heard = 10 ** ((np.array(di["db"]) + A_WEIGHT) / 10)
    trim = -10 * np.log10(np.sum(di_heard * 10 ** (band_got / 10)) / np.sum(di_heard))
    sel = (grid >= FIT_LO) & (grid <= FIT_HI)
    return {"target": [round(float(v), 2) for v in t], "filters": filters, "trim_db": round(float(trim), 2),
            "fit_error_db": round(float(np.sqrt(np.mean((got - want)[sel] ** 2))), 2),
            "fit": [round(float(v), 2) for v in band_got]}
