"""Light cues for Encore's stage show, from the whole song (user: "use the songs audio track to queue flashing
lights", "really, we should just take the whole songs track and look at that"), made when a song is added (or a
version made) and otherwise just before it's first played; saved in <song>/.light-cues.json.

All on 10 ms steps of the mono mix (22.05 kHz):
- kicks: the low end (one-pole low-pass at 150 Hz) jumping over the loudest of the last 50 ms by at least the song's
  90th-percentile jump (6 dB minimum), at a local peak, 120 ms apart; snares / cymbals: the same on the sample-to-
  sample change (the bright end), 100 ms apart.
- level: loudness 0-1 between the song's quiet 10% and loud 2%, smoothed (~0.2 s), sent at 20 a second.
- big moments: the level jumping 0.15+ from the 2 s before to the 1 s after, into a part louder than 60% of the song
  (a chorus or drop coming in), strongest first, 15 s apart, one per 30 s of song at most, on the kick that starts it;
  plus full stops: a loud hit with near silence 0.6-2 s later (12 dB+ down: Flower Power Madness's last hit at 3:12.3).
- accents: a kick and a crash together, both among the song's strongest; and stabs: a loud hit surging 5 dB+ over
  the 0.4 s before and falling 6 dB+ in the 0.15-0.4 s after, on 150 ms averages (the band hitting together and
  stopping: Flower Power Madness's ending 3:03-3:05).
These were first worked out in the page on the no-bass version (see NOTES.md, Game screen, 2026-10-31); the whole
song has the bass too, so kicks and drops come through better.

Held vocal notes (user: "sync it up with lyrics where they go oh ah"), from a vocals stem (a by-product of separating
the song, or a one-time pass for songs added before): pitch by YIN on 46 ms windows; a held note is 0.5 s+ where
the voice is pitched (YIN dip < 0.35: a separated rock vocal has bleed, a stricter 0.15 gave a pitch to 27% of the
sung frames, this 56%), loud (above the vocal's 40th percentile) and stays within a semitone of where it is (vibrato
went past 0.6), unclear for at most 60 ms at a time; phrases start where the voice comes in after 0.35 s+ of quiet.
Flower Power Madness has one: the high G at 2:24.4, 1.9 s (no other steady pitch there lasts over 0.41 s).
"""
import numpy as np

RATE = 22050
TOL, GAPS = 1.0, 6                                 # held note: within a semitone of where it is, unclear for up to 60 ms
HOP = RATE // 100                                  # 10 ms


def _db(power):
    return 10 * np.log10(power + 1e-12)


def _frames_power(x):
    n = len(x) // HOP
    return (x[:n * HOP].reshape(n, HOP) ** 2).mean(axis=1), n


def _lowpass(x, hz):
    """One-pole low-pass, the same as the page's (a = 1 - exp(-2 pi f / rate))."""
    import torch
    import torchaudio.functional as F
    a = 1 - np.exp(-2 * np.pi * hz / RATE)
    y = F.lfilter(torch.from_numpy(x[None].astype(np.float64)), torch.tensor([[1.0, -(1 - a)]], dtype=torch.float64),
                  torch.tensor([[a, 0.0]], dtype=torch.float64), clamp=False)
    return y[0].numpy()


def _hits(e, gap):
    """Times (s) and strengths of jumps in a 10 ms dB envelope: over the loudest of the 50 ms before."""
    n = len(e)
    prev = np.full(n, -200.0)
    for k in range(1, 6):
        prev[k:] = np.maximum(prev[k:], e[:-k])
    rise = np.maximum(0, e - prev)
    rise[:5] = 0
    pos = np.sort(rise[rise > 0])
    bar = max(6.0, float(pos[int(len(pos) * 0.9)]) if len(pos) else 6.0)
    peak = (rise >= bar) & (rise >= np.roll(rise, 1)) & (rise >= np.roll(rise, -1))
    times, strength = [], []
    for i in np.flatnonzero(peak):
        if 0 < i < n - 1 and (not times or i * 0.01 - times[-1] >= gap):
            times.append(i * 0.01)
            strength.append(float(rise[i]))
    return times, strength


# The detection's settings (tools/lights_tune.py tries others against the user's ratings from the light check page).
BIG = {"jump": 0.15, "loud_pct": 0.6, "gap": 15.0, "per_s": 30.0,   # big moments: loudness jump, into the top 40%, 15 s apart, 1 per 30 s
       "stop_db": 12.0, "stab_fall": 6.0, "stab_rise": 5.0}         # full stops; stabs: fall after / surge over before (dB)


def song_cues(mix, settings=None):
    """Kicks, snares, level, big moments and accents from the mono mix at RATE (see the module docstring)."""
    return decide(features(mix), settings)


def features(mix):
    """The slow part, the same whatever the settings: envelopes, hits and loudness (tools/lights_tune.py keeps it)."""
    x = np.asarray(mix, dtype=np.float64)
    power, n = _frames_power(x)
    if n < 300:
        return None
    low, _ = _frames_power(_lowpass(x, 150))
    high, _ = _frames_power(np.diff(x, prepend=0.0))
    low, high, alldb = _db(low), _db(high), _db(power)
    kick, kick_s = _hits(low, 0.12)
    snare, snare_s = _hits(high, 0.1)

    srt = np.sort(alldb)
    lo, hi = srt[int(n * 0.1)], srt[int(n * 0.98)]
    raw = np.clip((alldb - lo) / max(1.0, hi - lo), 0, 1)
    level = np.empty(n)
    sm = 0.0
    for i in range(n):
        sm += (raw[i] - sm) * 0.05
        level[i] = sm
    return {"n": n, "power": power, "alldb": alldb, "level": level, "kick": kick, "kick_s": kick_s, "snare": snare, "snare_s": snare_s}


def decide(f, settings=None):
    """Big moments and accents from `features` with these settings (BIG, overridden)."""
    if f is None:
        return {"kick": [], "snare": [], "level": [], "big": [], "accent": []}
    P = {**BIG, **(settings or {})}
    n, power, alldb, level = f["n"], f["power"], f["alldb"], f["level"]
    kick, kick_s, snare, snare_s = f["kick"], f["kick_s"], f["snare"], f["snare_s"]
    srt = np.sort(alldb)

    # Big moments: into a loud part.
    lsum = np.concatenate([[0.0], np.cumsum(level)])
    mean = lambda a, b: (lsum[b] - lsum[a]) / (b - a)
    p60 = np.sort(level)[int(n * P["loud_pct"])]
    jumps = []
    for i in range(200, n - 100, 5):
        j = mean(i, i + 100) - mean(i - 200, i - 20)
        if j > P["jump"] and mean(i, i + 100) > p60:
            jumps.append((j, i))
    jumps.sort(reverse=True)
    karr = np.array(kick)
    big = []
    for _, i in jumps:
        at = i * 0.01
        if len(karr):
            k = karr[np.argmin(np.abs(karr - at))]
            if abs(k - at) < 0.3:
                at = float(k)
        if len(big) < int(np.ceil(n / (P["per_s"] * 100))) and all(abs(b - at) >= P["gap"] for b in big):
            big.append(at)

    # Accents: kick + crash among the strongest; stabs; full stops (big).
    def top(s, q):
        return np.sort(s)[int(len(s) * q)] if len(s) else np.inf
    ktop, stop_ = top(np.array(kick_s), 0.97), top(np.array(snare_s), 0.9)
    sarr, sstr = np.array(snare), np.array(snare_s)
    accent = [k for k, s in zip(kick, kick_s) if s >= ktop and len(sarr) and np.any((np.abs(sarr - k) <= 0.03) & (sstr >= stop_))]
    psum = np.concatenate([[0.0], np.cumsum(power)])

    def win(a, b):
        a, b = max(0, a), min(n, b)
        return _db((psum[b] - psum[a]) / (b - a)) if b > a else -120.0
    loud_bar = srt[int(n * 0.6)]
    stops = []
    for x_ in sorted(kick + snare):
        i = int(round(x_ * 100))
        peak = win(i, i + 10)
        if peak < loud_bar:
            continue
        if peak - win(i + 60, i + 200) >= P["stop_db"]:
            if not stops or x_ - stops[-1] > 2:
                stops.append(x_)
        elif peak - win(i + 15, i + 40) >= P["stab_fall"] and peak - win(i - 40, i - 5) >= P["stab_rise"] and all(abs(x_ - a) > 0.3 for a in accent[-3:]):
            accent.append(x_)
    big += [s for s in stops if all(abs(b - s) >= 4 for b in big)]
    r = lambda v: [round(float(t), 3) for t in sorted(v)]
    return {"kick": r(kick), "snare": r(snare), "level": [round(float(v), 3) for v in level[::5]],   # 20 a second
            "big": r(big), "accent": r(accent)}


def _yin(frames, fmin=70, fmax=1000, clear=0.35):
    """Pitch (Hz, 0 = unclear) of each row of `frames` (YIN, cumulative mean normalised difference)."""
    w = frames.shape[1] // 2
    tmin, tmax = int(RATE / fmax), min(w - 1, int(RATE / fmin))
    out = np.zeros(len(frames))
    for c in range(0, len(frames), 1500):
        f = frames[c:c + 1500]
        # d(t) = sum (x_i - x_{i+t})^2 over the first w samples, via cumulative energy and correlation by FFT
        size = 1 << int(np.ceil(np.log2(frames.shape[1] + w)))
        fx = np.fft.rfft(f, size)
        fw = np.fft.rfft(f[:, :w], size)
        corr = np.fft.irfft(fx * np.conj(fw), size)[:, :w]
        energy = np.cumsum(f ** 2, axis=1)
        e0 = energy[:, w - 1][:, None]
        et = np.stack([energy[:, t + w - 1] - (energy[:, t - 1] if t else 0) for t in range(w)], axis=1)
        d = e0 + et - 2 * corr
        d[:, 0] = 0
        cum = np.cumsum(d[:, 1:], axis=1)
        cmnd = np.ones_like(d)
        cmnd[:, 1:] = d[:, 1:] * np.arange(1, w) / np.maximum(cum, 1e-12)
        for j, row in enumerate(cmnd):
            below = np.flatnonzero(row[tmin:tmax] < clear)
            if not len(below):
                continue
            t = tmin + below[0]
            while t + 1 < tmax and row[t + 1] < row[t]:
                t += 1
            a, b, e = row[t - 1], row[t], row[t + 1]
            den = 2 * (a - 2 * b + e)
            bend = (a - e) / den if den else 0.0
            out[c + j] = RATE / (t + max(-1.0, min(1.0, bend)))
    return out


def vocal_cues(vocals):
    """Held notes [[start, end, midi, loudness 0-1]] and phrase starts from a mono vocals stem at RATE."""
    x = np.asarray(vocals, dtype=np.float64)
    power, n = _frames_power(x)
    if n < 100:
        return {"held": [], "phrases": []}
    db = _db(power)
    win = 1024
    starts = np.arange(n) * HOP
    pad = np.concatenate([x, np.zeros(win)])
    frames = np.stack([pad[s:s + win] for s in starts])
    hz = _yin(frames)
    midi = np.where(hz > 0, 69 + 12 * np.log2(np.maximum(hz, 1) / 440), np.nan)
    voiced = db > max(np.percentile(db, 40), db.max() - 45)
    srt = np.sort(db[voiced]) if voiced.any() else np.array([-60.0])
    lo, hi = srt[int(len(srt) * 0.1)], srt[int(len(srt) * 0.98)]
    held, i = [], 0
    while i < n:
        if not (voiced[i] and np.isfinite(midi[i])):
            i += 1
            continue
        j, ref, gaps = i, midi[i], 0
        while j + 1 < n and voiced[j + 1] and (np.isfinite(midi[j + 1]) and abs(midi[j + 1] - ref) <= TOL or (not np.isfinite(midi[j + 1]) and gaps < GAPS)):
            j += 1
            gaps = gaps + 1 if not np.isfinite(midi[j]) else 0
            if np.isfinite(midi[j]):
                ref = 0.9 * ref + 0.1 * midi[j]          # follows a slow drift (vibrato stays inside 0.6)
        if (j - i + 1) * 0.01 >= 0.5:
            loud = float(np.clip((np.mean(db[i:j + 1]) - lo) / max(1.0, hi - lo), 0, 1))
            held.append([round(i * 0.01, 2), round((j + 1) * 0.01, 2), round(float(np.nanmedian(midi[i:j + 1])), 1), round(loud, 2)])
        i = j + 1
    phrases, quiet = [], 0
    for k in range(n):
        if voiced[k]:
            if quiet >= 35:
                phrases.append(round(k * 0.01, 2))
            quiet = 0
        else:
            quiet += 1
    return {"held": held, "phrases": phrases}
