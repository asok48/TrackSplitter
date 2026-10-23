"""Bass tab: bass audio -> notes -> bars -> strings and frets.

1. Notes: Spotify's Basic Pitch (basicpitch/, Apache 2.0) run with onnxruntime. Its
   package needs TensorFlow, which has no Python 3.13 build for Mac, so the model file
   and the few functions that turn its output into notes are reproduced here with
   numpy only (from basic_pitch/inference.py and note_creation.py, v0.4.0). Then one
   note at a time, since a bass part is monophonic, and nothing where the bass track is
   near silent (leakage). On a known bass line with real guitar and drums leaking in,
   96% of notes shown are real and 96% of real notes are found (onset within 50 ms).
2. Bars: tempo and beats from the song's onsets (autocorrelation + Ellis' dynamic
   programming beat tracker), the downbeat from where the bass notes land. Used only
   when the bass notes actually sit on the beat grid; otherwise lines are fixed-length.
3. Frets: a hand-position model (one finger per fret over 4 frets, plus open strings;
   moving the hand is expensive), each pitch then pulled toward its usual place so
   riffs are fingered the same each time, tried around several hand positions, and the
   easiest to play wins. On six songs: a third fewer hand shifts than note-by-note
   fingering, repeated riffs fingered differently 1% of the time instead of 9%.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).parent / "basicpitch" / "nmp.onnx"
SAMPLERATE = 22050          # what Basic Pitch expects

TUNINGS = {   # key: (label, open strings low to high as MIDI notes, string names)
    "standard": ("Standard (E A D G)", [28, 33, 38, 43], ["E", "A", "D", "G"]),
    "drop-d": ("Drop D (D A D G)", [26, 33, 38, 43], ["D", "A", "D", "G"]),
    "half-down": ("Half step down (E♭ A♭ D♭ G♭)", [27, 32, 37, 42], ["E♭", "A♭", "D♭", "G♭"]),
    "five-string": ("5-string (B E A D G)", [23, 28, 33, 38, 43], ["B", "E", "A", "D", "G"]),
}
MAX_FRET = 21


# --- 1. Notes (Basic Pitch) -------------------------------------------------------

FFT_HOP = 256
AUDIO_N_SAMPLES = SAMPLERATE * 2 - FFT_HOP
FPS = SAMPLERATE // FFT_HOP
ANNOT_N_FRAMES = FPS * 2
OVERLAP_FRAMES = 30
MIDI_OFFSET, MAX_FREQ_IDX = 21, 87
_session = None


def _model_output(audio, on_progress):
    """Basic Pitch's (note, onset) activations, each (n_frames, 88), for mono 22.05 kHz audio."""
    global _session
    if _session is None:
        import onnxruntime
        _session = onnxruntime.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    overlap = OVERLAP_FRAMES * FFT_HOP
    hop = AUDIO_N_SAMPLES - overlap
    x = np.concatenate([np.zeros(overlap // 2, np.float32), audio.astype(np.float32)])
    starts = list(range(0, len(x), hop))
    notes, onsets = [], []
    for b in range(0, len(starts), 16):
        batch = np.stack([np.pad(x[s:s + AUDIO_N_SAMPLES], (0, max(0, s + AUDIO_N_SAMPLES - len(x))))
                          for s in starts[b:b + 16]])[..., None]
        note, onset = _session.run(["StatefulPartitionedCall:1", "StatefulPartitionedCall:2"],
                                   {"serving_default_input_2:0": batch})
        notes.append(note)
        onsets.append(onset)
        on_progress(min(1.0, (b + 16) / len(starts)))
    n_keep = int(np.floor(len(audio) * (FPS / SAMPLERATE)))
    trim = OVERLAP_FRAMES // 2
    unwrap = lambda parts: np.concatenate(parts)[:, trim:-trim, :].reshape(-1, 88)[:n_keep]
    return unwrap(notes), unwrap(onsets)


def _frames_to_time(n):
    window_offset = (FFT_HOP / SAMPLERATE) * (ANNOT_N_FRAMES - AUDIO_N_SAMPLES / FFT_HOP) + 0.0018
    return np.arange(n) * FFT_HOP / SAMPLERATE - window_offset * np.floor(np.arange(n) / ANNOT_N_FRAMES)


def _hz_to_midi(hz):
    return 12 * (np.log2(hz) - np.log2(440.0)) + 69


def _note_events(frames, onsets, onset_thresh=0.6, frame_thresh=0.3, min_note_len=7,
                 min_freq=27.5, max_freq=420.0, energy_tol=11, infer=True, melodia=False):
    """Basic Pitch's output_to_notes_polyphonic: note events as (start frame, end frame,
    MIDI pitch, amplitude). Its "melodia trick" (notes grown from leftover energy, without
    a clear onset) is off by default: with real guitar and drums leaking into a known bass
    line it added notes never played (147 found for 114 real; precision 75% -> 96% off,
    recall unchanged at 96%)."""
    n_frames = frames.shape[0]
    frames, onsets = frames.copy(), onsets.copy()
    hi = int(np.round(_hz_to_midi(max_freq) - MIDI_OFFSET))
    lo = int(np.round(_hz_to_midi(min_freq) - MIDI_OFFSET))
    onsets[:, hi:] = frames[:, hi:] = 0
    onsets[:, :lo] = frames[:, :lo] = 0
    if infer:   # extra onsets where the frame activations jump
        diffs = [np.concatenate([np.zeros((n, 88)), frames])[n:] - np.concatenate([np.zeros((n, 88)), frames])[:-n]
                 for n in (1, 2)]
        frame_diff = np.clip(np.min(diffs, axis=0), 0, None)
        frame_diff[:2] = 0
        if frame_diff.max() > 0:
            frame_diff = onsets.max() * frame_diff / frame_diff.max()
        onsets = np.maximum(onsets, frame_diff)
    # Onset peaks in time (scipy.signal.argrelmax, mode "clip").
    prev = np.vstack([onsets[:1], onsets[:-1]])
    nxt = np.vstack([onsets[1:], onsets[-1:]])
    peaks = (onsets > prev) & (onsets > nxt) & (onsets >= onset_thresh)
    t_idx, f_idx = np.nonzero(peaks)
    order = np.argsort(t_idx, kind="stable")[::-1]
    remaining = frames.copy()
    events = []
    for start, f in zip(t_idx[order], f_idx[order]):
        if start >= n_frames - 1:
            continue
        i, k = start + 1, 0
        while i < n_frames - 1 and k < energy_tol:
            k = k + 1 if remaining[i, f] < frame_thresh else 0
            i += 1
        i -= k
        if i - start <= min_note_len:
            continue
        remaining[start:i, max(0, f - 1):min(MAX_FREQ_IDX, f + 1) + 1] = 0
        events.append((start, i, f + MIDI_OFFSET, float(np.mean(frames[start:i, f])), False))
    # Melodia trick: notes without a clear onset, grown from the strongest leftover energy.
    # Cells only ever get cleared, so walking them strongest first (ties in array order,
    # like argmax) picks the same cells as searching for the maximum each time, but fast.
    if melodia:
        flat = remaining.ravel()
        above = np.nonzero(flat > frame_thresh)[0]
        order = above[np.argsort(-flat[above], kind="stable")]
    for cell in (order if melodia else ()):
        if flat[cell] <= frame_thresh:
            continue   # cleared since
        mid, f = divmod(int(cell), 88)
        remaining[mid, f] = 0
        band = slice(max(0, f - 1), min(MAX_FREQ_IDX, f + 1) + 1)
        i, k = mid + 1, 0
        while i < n_frames - 1 and k < energy_tol:
            k = k + 1 if remaining[i, f] < frame_thresh else 0
            remaining[i, band] = 0
            i += 1
        end = i - 1 - k
        i, k = mid - 1, 0
        while i > 0 and k < energy_tol:
            k = k + 1 if remaining[i, f] < frame_thresh else 0
            remaining[i, band] = 0
            i -= 1
        start = i + 1 + k
        if end - start > min_note_len:
            events.append((start, end, f + MIDI_OFFSET, float(np.mean(frames[start:end, f])), True))
    return events


def _one_at_a_time(notes):
    """Where notes overlap, keep the stronger one, but prefer a lower note when the higher
    one is its octave or twelfth (an overtone of the bass, not a second note)."""
    out = []
    for n in notes:
        if out and n[0] < out[-1][1] - 0.03:
            prev = out[-1]
            together = n[0] - prev[0] < 0.08
            overtone = together and (n[2] - prev[2]) in (12, 19, 24)
            undertone = together and (prev[2] - n[2]) in (12, 19, 24)
            if overtone or (together and not undertone and n[3] < prev[3]):
                continue
            if undertone or together:
                out[-1] = n
                continue
            out[-1] = (prev[0], n[0], prev[2], prev[3])   # the new note cuts the previous short
        out.append(n)
    return out


def _levels(x, n):
    """Power per 50 ms frame, averaged over ~1 s, in dB (n frames)."""
    hop = SAMPLERATE // 20
    x = np.pad(x[:n * hop], (0, max(0, n * hop - len(x))))
    power = np.mean(x.reshape(n, hop) ** 2, axis=1)
    return 10 * np.log10(np.convolve(power, np.ones(20) / 20, mode="same") + 1e-12)


def _audible(notes, bass, mix=None, quieter_by=30, below_song=25):
    """Drop notes where the separated "bass" is really other instruments leaking in.
    Measured on five songs: where bass plays, the bass track is 0-16 dB below the whole
    song; in intros and breakdowns without bass it is 32-67 dB below (a quiet section of
    American Idiot sat at 22-24 dB, so the cut is at 25). Without the song to compare,
    fall back to the bass track's own loud parts (no-bass stretches were 55-73 dB below)."""
    n = len(bass) // (SAMPLERATE // 20)
    level = _levels(bass, n)
    ok = level >= np.percentile(level, 95) - quieter_by
    if mix is not None:
        ok &= level - _levels(mix, n) >= -below_song
    keep = []
    for note in notes:
        a, b = int(note[0] * 20), max(int(note[0] * 20) + 1, int(note[1] * 20))
        if a < n and ok[a:b].mean() >= 0.5:
            keep.append(note)
    return keep


def _gap_fillers(events):
    """Keep "melodia" notes (no clear onset) only where they fill a gap between notes
    with a clear onset. Overlapping one, they were leakage or overtones (the fake notes);
    in a gap they're usually re-plucked repeated notes the onsets missed (fast 8ths)."""
    plain = [e for e in events if not e[4]]
    keep = list(plain)
    for e in events:
        if e[4] and not any(p[0] < e[1] - 2 and p[1] > e[0] + 2 for p in plain):   # 2 frames ~ 23 ms
            keep.append(e)
    return keep


def _split_replucks(events, onsets, thresh, min_gap=6):
    """Split a held note where the model's onset activation for its pitch peaks again:
    a re-plucked repeated note whose attack didn't reach the onset threshold (driving
    8ths on one note came out as a few long notes). min_gap frames (~70 ms) apart."""
    out = []
    for s, e, p, a, m in events:
        col = onsets[:, p - MIDI_OFFSET]
        cuts = [s]
        for i in range(s + min_gap, e - min_gap):
            if col[i] >= thresh and col[i] >= col[i - 1] and col[i] > col[i + 1] and i - cuts[-1] >= min_gap:
                cuts.append(i)
        cuts.append(e)
        out += [(c0, c1, p, a, m) for c0, c1 in zip(cuts, cuts[1:])]
    return out


def _fix_octaves(notes, bass, level=0.25, lowest=23, half_level=None, memo=None):
    """Move notes down an octave when the bass track says the real note is lower.

    The transcriber reports low notes (low E, F, G: fundamentals ~41-49 Hz) an octave or
    two too high (on a known line, most of its pitch mistakes). A note really at f has
    nothing at f/2 or 3f/2; a note at f/2 has overtones exactly there. So if the track
    has clear energy at both, compared with f, the note is an octave lower. memo: a dict
    that keeps each note's answer (learning tries many settings on the same notes)."""
    out = []
    for s, e, p, a in notes:
        key = (s, e, p, level, half_level)
        if memo is not None and key in memo:
            out.append((s, e, memo[key], a))
            continue
        seg = bass[int(s * SAMPLERATE):int(min(e, s + 0.4) * SAMPLERATE)]
        while len(seg) >= SAMPLERATE // 10 and p - 12 >= lowest:
            spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 1 << 15))
            hz = 440 * 2 ** ((p - 69) / 12)

            def at(f):   # strongest magnitude within ~3% of f
                i = f * (1 << 15) / SAMPLERATE
                return spec[int(i * 0.97):int(i * 1.03) + 2].max()

            # The lower note's fundamental (f/2) is often faint in a separated track (~41-46 Hz);
            # its 3rd harmonic (3f/2) is the stronger evidence. half_level: how much f/2 is
            # needed (0.08: Electra's stray octave-up F#/G# 8 -> 1, known lines unchanged).
            if at(hz * 1.5) < level * at(hz) or at(hz / 2) < (level if half_level is None else half_level) * at(hz):
                break
            p -= 12
        if memo is not None:
            memo[key] = p
        out.append((s, e, p, a))
    return out


def transcribe(bass, on_progress=lambda fraction: None, mix=None, min_amp=0.0, melodia="gaps", split=None,
               keep=None, octaves=0.25, half_level=0.08, **options):
    """Notes (start s, end s, MIDI pitch, amplitude) of a mono 22.05 kHz bass track
    (mix: the whole song at the same rate, if available, to spot leakage). keep: a dict
    that receives the model's onset activations and frame times, for split_on_grid."""
    frames, onsets = _model_output(bass, on_progress)
    times = _frames_to_time(frames.shape[0])
    if keep is not None:
        keep.update(onsets=onsets, times=times, frames=frames)
    events = _note_events(frames, onsets, melodia=bool(melodia), **options)
    if melodia == "gaps":
        events = _gap_fillers(events)
    if split:
        events = _split_replucks(events, onsets, split)
    notes = sorted((float(times[s]), float(times[e]), int(p), a) for s, e, p, a, _ in events
                   if 23 <= p <= 67 and a >= min_amp)
    notes = _one_at_a_time(_audible(notes, bass, mix))
    return _fix_octaves(notes, bass, octaves, half_level=half_level) if octaves else notes


# --- 2. Beats and bars ------------------------------------------------------------

def _onset_envelope(mix, hop=512, n_fft=2048):
    """Spectral flux of the whole song: where things start, ~43 values per second.
    Frames are centered (value i is about time i * hop), or beats come out ~60 ms early."""
    mix = np.pad(mix, n_fft // 2)
    n = 1 + (len(mix) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    mag = np.abs(np.fft.rfft(mix[idx] * np.hanning(n_fft), axis=1))
    logmag = np.log1p(100 * mag)
    flux = np.clip(np.diff(logmag, axis=0), 0, None).sum(axis=1)
    k = int(SAMPLERATE / hop / 2)                     # remove the slow trend (~0.5 s)
    flux = np.clip(flux - np.convolve(flux, np.ones(k) / k, mode="same"), 0, None)
    return np.concatenate([[0], flux / (flux.std() or 1)]), SAMPLERATE / hop


def _tempo_period(env, fps):
    """Beat period in frames, by autocorrelation with a prior around 120 bpm."""
    n = len(env)
    ac = np.fft.irfft(np.abs(np.fft.rfft(env, 2 * n)) ** 2)[:n]
    lags = np.arange(int(fps * 60 / 200), int(fps * 60 / 60) + 1)
    bpm = 60 * fps / lags
    weighted = ac[lags] * np.exp(-0.5 * (np.log2(bpm / 120) / 1.0) ** 2)
    i = int(np.argmax(weighted))
    period = float(lags[i])
    if 0 < i < len(lags) - 1:                          # refine between lags
        a, b, c = weighted[i - 1:i + 2]
        period += 0.5 * (a - c) / (a - 2 * b + c or 1)
    return period


def _beats(env, fps, period):
    """Ellis' dynamic-programming beat tracker for a given beat period (frames)."""
    n = len(env)
    width = max(1, int(period / 32))
    g = np.exp(-0.5 * (np.arange(-4 * width, 4 * width + 1) / width) ** 2)
    local = np.convolve(env, g / g.sum(), mode="same")
    score, back = local.copy(), np.full(n, -1)
    lo, hi = int(round(period / 2)), int(round(2 * period))
    offsets = np.arange(lo, hi + 1)
    penalty = 100 * np.log(offsets / period) ** 2
    for t in range(lo, n):
        cand = t - offsets
        ok = cand >= 0
        vals = score[cand[ok]] - penalty[ok]
        j = int(np.argmax(vals))
        if vals[j] > 0:
            score[t] += vals[j]
            back[t] = cand[ok][j]
    tail = n - 1 - int(np.argmax(score[::-1][:int(period)]))
    beats = [tail]
    while back[beats[-1]] >= 0:
        beats.append(back[beats[-1]])
    return np.array(beats[::-1]) / fps


def lines_for(mix, notes, duration):
    """Split the song into lines: about 2 bars each when the bass sits on a steady beat
    grid, otherwise 4 s each. Returns ({"bpm", "lines": [[start, end, [bar starts]]]})."""
    starts = np.array([s for s, _, _, _ in notes])
    # The tracker can lock onto half or double the real tempo, and swung or triplet parts
    # fall between straight 8ths. So try those, keep the grid the bass notes fit best,
    # scored above what chance alone would put on a grid that fine.
    best = (0.0, np.array([]), None, 0.0, 2)       # (score, beats, bpm, fraction on grid, division)
    try:
        env, fps = _onset_envelope(mix)
        base = _tempo_period(env, fps)
    except (ValueError, IndexError):
        env, base = None, None
    for factor in ((0.5, 1.0, 2.0) if base and len(starts) else ()):
        period = base * factor
        bpm = 60 * fps / period
        if not 50 <= bpm <= 240:
            continue
        beats = _beats(env, fps, period)
        if len(beats) < 8:
            continue
        step = float(np.median(np.diff(beats)))
        for division in (2, 3):                    # straight 8ths, or triplets
            grid = np.sort(np.concatenate([beats + k * step / division for k in range(division)]))
            gaps = starts - grid[np.abs(starts[:, None] - grid[None, :]).argmin(axis=1)]
            close = np.abs(gaps) < 0.06
            shift = float(np.median(gaps[close])) if close.any() else 0.0
            hit = float(np.mean(np.abs(gaps - shift) < 0.04))
            chance = min(0.95, 2 * 0.04 * division / step)
            score = (hit - chance) / (1 - chance)
            if score > best[0]:
                best = (score, beats + shift, bpm, hit, division)   # grid lined up with the bass itself
    score, beats, bpm, on_grid, division = best
    if score < 0.4:   # no steady grid: fixed-length lines without bar lines
        edges = np.arange(0, duration + 4, 4.0)
        return {"bpm": None, "on_grid": round(on_grid, 2), "grid": [],
                "lines": [[float(a), float(min(b, duration)), []] for a, b in zip(edges, edges[1:]) if a < duration]}
    # Downbeat: the beat (of each 4) where bass notes land most and hit hardest.
    weight = np.zeros(4)
    for s, e, _, a in notes:
        j = int(np.argmin(np.abs(beats - s)))
        if abs(beats[j] - s) < 0.06:
            weight[j % 4] += a * min(e - s, 0.5)
    bars = beats[int(np.argmax(weight))::4]
    bar_len = float(np.median(np.diff(bars)))
    per_line = int(np.clip(round(5.0 / bar_len), 1, 4))
    edges = list(bars[::per_line])
    while edges[0] > bar_len * per_line * 0.5:          # lines before the first downbeat
        edges.insert(0, max(0.0, edges[0] - bar_len * per_line))
    if edges[0] > 0:
        edges[0] = 0.0
    while edges[-1] < duration:
        edges.append(edges[-1] + bar_len * per_line)
    lines = []
    for a, b in zip(edges, edges[1:]):
        inside = [float(x) for x in bars if a <= x < b]
        lines.append([float(a), float(min(b, duration)), inside])
    step = float(np.median(np.diff(beats)))
    grid = np.sort(np.concatenate([beats + k * step / division for k in range(division)]))
    return {"bpm": round(float(bpm), 1), "on_grid": round(on_grid, 2), "grid": grid.tolist(),
            "lines": [l for l in lines if l[0] < duration]}


def _attack_strength(bass, hop=256, n_fft=1024):
    """How sharply the bass track's sound starts at each moment (log spectral flux),
    ~86 values per second."""
    x = np.pad(bass.astype(np.float32), n_fft // 2)
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
    mag = np.abs(np.fft.rfft(frames * np.hanning(n_fft).astype(np.float32), axis=1))
    flux = np.clip(np.diff(np.log1p(100 * mag), axis=0), 0, None).sum(axis=1)
    return np.concatenate([[0.0], flux]), SAMPLERATE / hop


def _overtone_jump(bass, t, pitch, n=1024, nfft=8192):
    """(rise of the note's own overtones, rise between them) in dB, just after t vs just
    before. A re-plucked string re-excites exactly its overtones (2nd-8th harmonics); a
    kick drum leaking into the bass track raises everything, between them too."""
    f0 = 440 * 2 ** ((pitch - 69) / 12)
    a, b = int((t - 0.06) * SAMPLERATE), int((t + 0.005) * SAMPLERATE)
    if a < 0 or b + n > len(bass):
        return 0.0, 0.0
    win = np.hanning(n)
    spec = [np.abs(np.fft.rfft(bass[i:i + n] * win, nfft)) ** 2 for i in (a, b)]
    hz = np.fft.rfftfreq(nfft, 1 / SAMPLERATE)

    def band(sp, f):
        m = (hz > f * 0.97) & (hz < f * 1.03)
        return sp[m].max() if m.any() else 0.0

    harm = [sum(band(sp, k * f0) for k in range(2, 9)) + 1e-12 for sp in spec]
    off = [sum(band(sp, (k + 0.5) * f0) for k in range(2, 8)) + 1e-12 for sp in spec]
    return 10 * np.log10(harm[1] / harm[0]), 10 * np.log10(off[1] / off[0])


def _low_bump(bass, t, n=880):
    """Bass-range loudness (below ~400 Hz) in the 40 ms after t minus the 40 ms before, dB.
    A re-pluck makes the note louder again; a held note only decays."""
    a, b = int((t - 0.045) * SAMPLERATE), int((t + 0.005) * SAMPLERATE)
    if a < 0 or b + n > len(bass):
        return 0.0
    hz = np.fft.rfftfreq(n, 1 / SAMPLERATE)
    low = hz < 400
    e = [np.sum(np.abs(np.fft.rfft(bass[i:i + n] * np.hanning(n)))[low] ** 2) + 1e-12 for i in (a, b)]
    return 10 * np.log10(e[1] / e[0])


def split_on_grid(notes, grid, bass, ratio=2.0, activations=None, onset_level=None, overtones=None, bump=None):
    """Split long notes where the bass is re-plucked: only at points of the song's 8th
    (or triplet) grid, and only where the bass track has a real attack there, several
    times sharper than the rest of that note. The transcriber hears repeated plucks of
    one pitch as a single long note; driving 8ths came out as a few long notes."""
    if not grid:
        return notes
    flux, fps = _attack_strength(bass)
    grid = np.asarray(grid)
    out = []
    for s, e, p, a in notes:
        inside = grid[(grid > s + 0.06) & (grid < e - 0.06)]
        if not len(inside):
            out.append((s, e, p, a))
            continue
        i0, i1 = int(s * fps), max(int(s * fps) + 1, int(e * fps))
        base = float(np.median(flux[i0:i1])) + 1e-9
        cuts = [s]
        for g in inside:
            j = int(g * fps)
            lo = max(0, j - 3)
            w = flux[lo:j + 4]                       # within ~35 ms of the grid point
            if bump is not None:
                if _low_bump(bass, g) >= bump:
                    cuts.append(float(g))
                continue
            if overtones is not None:                # (rise dB, margin over in-between dB)
                rise, between = _overtone_jump(bass, g, p)
                if rise >= overtones[0] and rise - between >= overtones[1]:
                    cuts.append(float(g))
                continue
            if len(w) and w.max() >= ratio * base:
                cuts.append(float((lo + int(np.argmax(w))) / fps))
            elif activations is not None and onset_level is not None:
                # Or the transcriber's own onset signal for this pitch bumps up here.
                t, act = activations["times"], activations["onsets"][:, p - MIDI_OFFSET]
                k = np.searchsorted(t, g)
                win = act[max(0, k - 3):k + 4]
                if len(win) and win.max() >= onset_level:
                    cuts.append(float(t[max(0, k - 3) + int(np.argmax(win))]))
        cuts.append(e)
        out += [(c0, c1, p, a) for c0, c1 in zip(cuts, cuts[1:]) if c1 - c0 > 0.04]
    return out


# --- 3. Strings and frets ---------------------------------------------------------

def suggest_tuning(notes):
    """The lowest notes (ignoring a few stray ones) decide: standard, drop D or 5-string."""
    if not notes:
        return "standard"
    low = np.percentile([p for _, _, p, _ in notes], 2)
    return "standard" if low >= 28 else "drop-d" if low >= 26 else "five-string"


def maybe_half_down(notes):
    """Might the band be tuned half a step down? Shown as a hint, never chosen for you.

    Half step down can't be heard (it's just lower notes), but bands tune so riffs fall
    on open strings: if half step down turns many more notes into open strings (20+
    points more, at least double) without making the part harder, it's likely. Right
    for American Idiot (2% -> 47% open; recorded half step down), but Electra looks the
    same (5% -> 37%) and is in standard tuning (the user's Songsterr tab), so this was
    demoted from choosing the tuning to a hint. Difficulty alone favoured half step down
    even on a standard-tuned Strokes song.
    """
    if not notes or suggest_tuning(notes) != "standard":
        return False
    if np.percentile([p for _, _, p, _ in notes], 2) < TUNINGS["half-down"][1][0]:
        return False

    def could_be_open(tuning):   # share of notes that are an open string's pitch
        strings = set(TUNINGS[tuning][1])
        return sum(1 for _, _, p, _ in notes if p in strings) / len(notes)

    open_std, open_half = could_be_open("standard"), could_be_open("half-down")
    return bool(open_half >= open_std + 0.2 and open_half >= 2 * open_std)


# How a bassist tends to finger, as costs. These defaults were set by hand; the app
# refits them to the user's imported tabs (`learn_style`) and passes those instead.
STYLE = {
    "open": 0.2,        # an open string (vs "fretted" for a note under the hand)
    "fretted": 0.05,
    "stretch": 0.4,     # a note one fret outside the hand's 4
    "string": 0.2,      # per string up from the lowest (leaning to the thicker strings)
    "low": 0.02,        # per fret the hand sits up the neck
    "shift": 3.0,       # moving the hand, plus "shift_fret" per fret moved
    "shift_fret": 0.3,
    "cross": 0.15,      # per string crossed between notes
}


def _hand_position(pitches, strings, usual=None, weight=0.0, target=None, pull=0.0, style=STYLE, forced=None,
                   with_hand=False):
    """Viterbi over (hand position, string, fret). The hand covers 4 frets (one finger
    per fret, a 1-fret stretch allowed at a cost) plus open strings; moving it is costly.
    Like a bassist (and the user's Songsterr tab of Electra, all on E and A at frets 2-6),
    it leans to the thicker strings and to a fretted note under the hand over an open
    string (STYLE["string"] per string up, STYLE["open"] vs STYLE["fretted"]).
    `forced` (a (string, fret) per note) only finds the hand positions for a given
    fingering; `with_hand` returns (hand, string, fret)."""
    anchors = range(1, MAX_FRET - 2)
    home = (lambda a: style["low"] * a) if target is None else (lambda a: pull * abs(a - target))

    def place(p, a, only=None):
        opts = []
        for s, o in enumerate(strings):
            f = p - o
            if only is not None and only != (s, f):
                continue
            if f == 0:
                c = style["open"]
            elif a <= f <= a + 3:
                c = style["fretted"]
            elif f in (a + 4, a - 1):
                c = style["stretch"]
            else:
                continue
            if usual and usual.get(p, (s, f)) != (s, f):
                c += weight
            opts.append((s, f, c + style["string"] * s))
        return opts

    only = forced or [None] * len(pitches)
    hist = [{(a, s, f): (c + home(a), None) for a in anchors for s, f, c in place(pitches[0], a, only[0])}]
    for i, p in enumerate(pitches[1:], 1):
        prev = hist[-1]
        keys = list(prev)
        pa = np.array([k[0] for k in keys])
        ps = np.array([k[1] for k in keys])
        pc = np.array([prev[k][0] for k in keys])
        cur = {}
        for a in anchors:
            move = np.where(pa == a, 0.0, style["shift"] + style["shift_fret"] * np.abs(pa - a))
            for s, f, c in place(p, a, only[i]):
                total = pc + move + style["cross"] * np.abs(ps - s)
                j = int(np.argmin(total))
                cur[(a, s, f)] = (float(total[j]) + c + home(a), keys[j])
        hist.append(cur)
    state = min(hist[-1], key=lambda k: hist[-1][k][0])
    path = [state]
    for i in range(len(hist) - 1, 0, -1):
        state = hist[i][state][1]
        path.append(state)
    return path[::-1] if with_hand else [(s, f) for _, s, f in path[::-1]]


def _consistent(pitches, strings, target=None, pull=0.0, rounds=3, style=STYLE):
    """Hand position, then again with a cost for playing a pitch away from its usual
    place (from the round before), so each note settles into one home."""
    pos = _hand_position(pitches, strings, target=target, pull=pull, style=style)
    for _ in range(rounds):
        counts = defaultdict(Counter)
        for p, sf in zip(pitches, pos):
            counts[p][sf] += 1
        usual = {p: c.most_common(1)[0][0] for p, c in counts.items()}
        pos = _hand_position(pitches, strings, usual=usual, weight=1.0, target=target, pull=pull, style=style)
    return pos


def _difficulty(times, positions):
    """How hard a fingering is (per note, lower is easier): hand shifts (more for long or
    quick ones), string crossings (more when skipping a string), fretted notes."""
    total, lo, last_t, last_s = 0.0, None, None, None
    for t, (s, f) in zip(times, positions):
        hurry = 1 + 0.15 / (1.0 if last_t is None else max(0.05, t - last_t))
        if f:
            total += 0.1
            if lo is None:
                lo = f
            elif not lo <= f <= lo + 3:
                total += (2 + 0.5 * ((lo - f) if f < lo else (f - lo - 3))) * hurry
                lo = f if f < lo else f - 3
        if last_s is not None and s != last_s:
            total += (0.2 + (0.3 if abs(s - last_s) > 1 else 0)) * hurry
        last_t, last_s = t, s
    return total / max(1, len(positions))


def fingering(notes, tuning, learned=None):
    """[(note index, string, fret)] for the notes playable in this tuning. With costs
    learned from the user's imported tabs (`learned`), the fingering those costs find;
    otherwise the easiest of fingerings centered on different hand positions."""
    strings = TUNINGS[tuning][1]
    playable = [i for i, (_, _, p, _) in enumerate(notes) if strings[0] <= p <= strings[-1] + MAX_FRET]
    if not playable:
        return []
    pitches = [notes[i][2] for i in playable]
    times = [notes[i][0] for i in playable]
    if learned:
        best = _consistent(pitches, strings, style={**STYLE, **learned})
    else:
        options = [_consistent(pitches, strings)] + [_consistent(pitches, strings, target=t, pull=0.1) for t in (1, 5, 9)]
        best = min(options, key=lambda pos: _difficulty(times, pos))
    return [(i, s, f) for i, (s, f) in zip(playable, best)]


# Learning the user's fingering from imported tabs. The user found the automatic
# fingering unlike tabs they trust (Cry Baby Cry's riff written A7 D5 D7 came out D2 G2
# D5: 77% of notes on the same string and fret). Each imported tab is the right answer
# for its notes, so the hand-position costs are fitted to reproduce it: a structured
# perceptron (Collins 2002): find the hand positions that best explain the tab's own
# fingering (`forced`), find the fingering the current costs choose, and move every
# cost toward what the tab does more often; the averaged costs are kept. Measured on
# Cry Baby Cry, fitted on one half and checked on the other: 84% -> 97% and 73% -> 100%
# (100 rounds, ~4 s per 500 notes; 300 rounds were no better).
# Picking the easiest of several tries is skipped with learned costs (that is the
# hand-made taste being replaced). Other songs stayed playable (frets 2-7, more notes
# on E and A, like the user's Electra tab).
def _path_features(path):
    """Per-note totals of what each STYLE cost charges for, along (hand, string, fret)."""
    f = dict.fromkeys(STYLE, 0.0)
    for k, (a, s, fret) in enumerate(path):
        f["open" if fret == 0 else "fretted" if a <= fret <= a + 3 else "stretch"] += 1
        f["string"] += s
        f["low"] += a
        if k:
            pa, ps = path[k - 1][0], path[k - 1][1]
            if pa != a:
                f["shift"] += 1
                f["shift_fret"] += abs(pa - a)
            f["cross"] += abs(ps - s)
    return np.array([f[k] / len(path) for k in STYLE])


def _usable(examples):
    """Examples [(open strings, [(time, pitch, string, fret)])] as (strings, times, pitches,
    positions), keeping the notes a 4-fret hand model can follow (frets up to MAX_FRET)."""
    out = []
    for strings, notes in examples:
        notes = [n for n in notes if 0 <= n[2] < len(strings) and 0 <= n[3] <= MAX_FRET]
        if len(notes) >= 8:
            out.append((strings, [n[0] for n in notes], [n[1] for n in notes], [(n[2], n[3]) for n in notes]))
    return out


def learn_style(examples, epochs=100, rate=2.0, on_progress=lambda fraction: None):
    """Costs (a STYLE dict) fitted to examples [(open strings, [(time, pitch, string, fret)])]."""
    examples = _usable(examples)
    w = np.array(list(STYLE.values()))
    total = np.zeros_like(w)
    for e in range(epochs):
        for strings, _, p, pos in examples:
            style = dict(zip(STYLE, w))
            gold = _hand_position(p, strings, style=style, forced=pos, with_hand=True)
            guess = _hand_position(p, strings, style=style, with_hand=True)
            w = np.maximum(0.0, w - rate * (_path_features(gold) - _path_features(guess)))
        total += w
        on_progress((e + 1) / epochs)
    return {k: round(float(v), 4) for k, v in zip(STYLE, total / epochs)}


def style_match(examples, learned=None):
    """Share of the examples' notes fingered as written, by the default way or `learned`."""
    same = count = 0
    for strings, times, p, pos in _usable(examples):
        if learned:
            got = _consistent(p, strings, style={**STYLE, **learned})
        else:
            options = [_consistent(p, strings)] + [_consistent(p, strings, target=t, pull=0.1) for t in (1, 5, 9)]
            got = min(options, key=lambda o: _difficulty(times, o))
        same += sum(g == w for g, w in zip(got, pos))
        count += len(p)
    return same / max(1, count)


# Learning note settings from imported tabs. The same tabs say which notes are really
# played, so the detection settings are fitted to them (F1: notes found vs notes that
# aren't in the tab, matched by pitch within 70 ms). On Cry Baby Cry the hand-set values
# were already close to its best (F1 0.87; each setting moved alone gained <= 0.005);
# most misses were re-plucks inside one long note (37 of 76), which "split repeated
# notes" catches at some cost in extra notes. More tabs from other mixes are where this
# should pay off. Decoding is cheap once the model has run (~1 s per song), so learning
# reruns it on every imported song that has a "bass only" track.
NOTES = {
    "onset_thresh": 0.6, "frame_thresh": 0.3, "min_note_len": 7,   # Basic Pitch's decoding
    "melodia": "gaps",           # notes without a clear onset: only in gaps, or never
    "octaves": 0.25, "half_level": 0.08,   # _fix_octaves (0: off)
    "grid_split": None,          # [ratio, onset_level] to split repeated notes by default
}
NOTE_CHOICES = {
    "onset_thresh": [0.45, 0.5, 0.55, 0.6, 0.65, 0.7], "frame_thresh": [0.2, 0.25, 0.3, 0.35, 0.4],
    "min_note_len": [5, 6, 7, 8, 10], "melodia": ["gaps", False], "octaves": [0, 0.15, 0.25, 0.4],
    "half_level": [0.04, 0.08, 0.15], "grid_split": [None, [1.35, 0.2], [2.0, 0.4], [3.0, 0.6]],
}


def note_example(bass, mix, truth):
    """What learning needs from one song: the model's output (run once), the beat grid,
    and the tab's notes [(time, pitch)]."""
    frames, onsets = _model_output(bass, lambda fraction: None)
    times = _frames_to_time(frames.shape[0])
    ex = {"bass": bass, "mix": mix, "frames": frames, "onsets": onsets, "times": times,
          "truth": sorted(truth), "cache": {}, "octaves": {}}
    ex["grid"] = lines_for(mix if mix is not None else bass, detect(ex, NOTES), len(bass) / SAMPLERATE)["grid"]
    return ex


def detect(ex, settings):
    """The notes `transcribe` + `split_on_grid` would give for this example's song."""
    key = json.dumps(settings, sort_keys=True)
    if key not in ex["cache"]:
        s = settings
        events = _note_events(ex["frames"], ex["onsets"], onset_thresh=s["onset_thresh"], frame_thresh=s["frame_thresh"],
                              min_note_len=s["min_note_len"], melodia=bool(s["melodia"]))
        if s["melodia"] == "gaps":
            events = _gap_fillers(events)
        times = ex["times"]
        notes = sorted((float(times[a]), float(times[b]), int(p), amp) for a, b, p, amp, _ in events if 23 <= p <= 67)
        notes = _one_at_a_time(_audible(notes, ex["bass"], ex["mix"]))
        if s["octaves"]:
            notes = _fix_octaves(notes, ex["bass"], s["octaves"], half_level=s["half_level"], memo=ex["octaves"])
        if s["grid_split"] and ex.get("grid") is not None:
            ratio, level = s["grid_split"]
            notes = split_on_grid(notes, ex["grid"], ex["bass"], ratio=ratio, onset_level=level,
                                  activations={"onsets": ex["onsets"], "times": times})
        ex["cache"][key] = notes
    return ex["cache"][key]


def note_counts(examples, settings, part=None):
    """(matched, detected, in the tabs), over each song's first (part 0) or second
    (part 1) half or all of it."""
    hit = found = wanted = 0
    for ex in examples:
        truth = ex["truth"]
        mid = (truth[0][0] + truth[-1][0]) / 2
        inside = (lambda t: True) if part is None else (lambda t: (t < mid) == (part == 0))
        tt = np.array([t for t, p in truth if inside(t)])
        tp = np.array([p for t, p in truth if inside(t)])
        used = np.zeros(len(tt), bool)
        notes = [n for n in detect(ex, settings) if inside(n[0]) and truth[0][0] - 1 <= n[0] <= truth[-1][0] + 1]
        for s, e, p, a in notes:
            k = np.nonzero((np.abs(tt - s) < 0.07) & (tp == p) & ~used)[0]
            if len(k):
                used[k[0]] = True
                hit += 1
        found += len(notes)
        wanted += len(tt)
    return hit, found, wanted


def f1(counts):
    hit, found, wanted = counts
    return 2 * hit / max(1, found + wanted)


def learn_notes(examples, part=None, on_progress=lambda fraction: None):
    """Note settings fitted to the examples (one setting at a time, best F1, 2 rounds)."""
    best = dict(NOTES)
    score = f1(note_counts(examples, best, part))
    steps = 2 * sum(len(v) for v in NOTE_CHOICES.values())
    done = 0
    for _ in range(2):
        for key, values in NOTE_CHOICES.items():
            for v in values:
                trial = {**best, key: v}
                s = f1(note_counts(examples, trial, part))
                if s > score + 1e-6:
                    best, score = trial, s
                done += 1
                on_progress(done / steps)
    return best


# Note models learned from imported tabs. The setting search above can't tell a real
# re-pluck from a ringing or leaking sound, so the user still saw fake notes (a held
# F#2 in You Only Live Once came out as three, and a stray B3 shown on G16) and missed
# 8ths. Three small logistic models judge each case from measurements the app already
# makes, labelled by the imported tabs:
#   octave: move a note down or up an octave (wrong-octave notes were 99 of Wow! Wow!
#           Wow!'s fake notes);
#   pluck:  split a long note at a beat-grid point (re-plucks heard as one note were
#           the biggest group of missed notes, 280 in 4 songs);
#   keep:   drop a note that's likely not played.
# Each song tested with models trained on the other three (4 tabs): see CLAUDE.md.
OCTAVE_CUT, PLUCK_CUT, KEEP_CUT = 0.5, 0.4, 0.3


def _fit(X, y, l2=1.0, steps=3000, rate=0.1):
    """Logistic regression (standardized features, L2) by gradient descent."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Z.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-Z @ w))
        w -= rate * (Z.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0] / len(y))
    return {"mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist()}


def _prob(model, X):
    X = np.asarray(X, float).reshape(len(X), -1)
    Z = np.hstack([(X - np.array(model["mu"])) / np.array(model["sd"]), np.ones((len(X), 1))])
    return 1 / (1 + np.exp(-Z @ np.array(model["w"])))


def _keep_features(ctx, notes):
    """Per note: overtone rise, model confidence, onset strength, length, attack
    sharpness, loudness rise, whether it just continues the same pitch, pitch, levels."""
    flux, fps = ctx["flux"]
    times, on, fr, bass = ctx["times"], ctx["onsets"], ctx["frames"], ctx["bass"]
    lv = _levels(bass, len(bass) // (SAMPLERATE // 20))
    mixlv = _levels(ctx["mix"], len(lv)) if ctx.get("mix") is not None else lv
    loud = np.percentile(lv, 95)
    mid = np.median([n[2] for n in notes])
    X = []
    for k, (s, e, p, a) in enumerate(notes):
        i, j, f = int(np.searchsorted(times, s)), int(s * fps), min(int(s * 20), len(lv) - 1)
        prev = notes[k - 1] if k else None
        rise, between = _overtone_jump(bass, s, p)   # a ringing note's "re-start" barely rises (You Only Live Once)
        X.append([
            rise, rise - between,
            a, on[max(0, i - 3):i + 4, p - MIDI_OFFSET].max(), np.log(max(e - s, 0.02)),
            np.log1p(flux[max(0, j - 3):j + 4].max() / (np.median(flux[max(0, j - 40):j + 40]) + 1e-9)),
            _low_bump(bass, s), float(prev is not None and prev[2] == p and s - prev[1] < 0.03),
            (p - mid) / 12, float(p >= 52), (lv[f] - loud) / 10, (lv[f] - mixlv[f]) / 10,
            fr[i:i + 6, p - MIDI_OFFSET].mean() - (fr[i:i + 6, p - 12 - MIDI_OFFSET].mean() if p - 12 >= MIDI_OFFSET else 0),
        ])
    return X


def _octave_features(ctx, notes):
    """Per note: energy at f/4, f/2, 3f/2, 2f, 3f vs f in the bass track, and the
    model's activation an octave below/above."""
    bass, fr, times = ctx["bass"], ctx["frames"], ctx["times"]
    X = []
    for s, e, p, a in notes:
        a0 = int(s * SAMPLERATE)
        seg = bass[a0:max(int(min(e, s + 0.4) * SAMPLERATE), a0 + SAMPLERATE // 20)]
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 1 << 15)) if len(seg) > 8 else np.zeros((1 << 14) + 1)
        hz = 440 * 2 ** ((p - 69) / 12)

        def at(f):
            i = f * (1 << 15) / SAMPLERATE
            return spec[int(i * 0.97):int(i * 1.03) + 2].max() + 1e-9 if i * 1.03 + 2 < len(spec) else 1e-9

        ref = at(hz)
        i = int(np.searchsorted(times, s))
        act = lambda q: fr[i:i + 6, q - MIDI_OFFSET].mean() if 0 <= q - MIDI_OFFSET < 88 else 0.0
        X.append([np.log(at(hz / 2) / ref), np.log(at(hz * 1.5) / ref), np.log(at(hz * 2) / ref),
                  np.log(at(hz * 3) / ref), np.log(at(hz / 4) / ref) if hz / 4 > 25 else -5.0,
                  act(p) - act(p - 12), act(p) - act(p + 12), act(p - 24), (p - 40) / 12, a])
    return X


def _join_repeats(notes):
    """Same-pitch notes that touch, joined into one: (notes, [times they were split at]).
    Detection also re-starts a ringing note (a held F#2 in You Only Live Once came out as
    three notes, without the loudness/overtone rise of a pluck), so whether such a point
    is a re-pluck is left to the pluck model too."""
    joined, at = [], []
    for n in notes:
        if joined and n[2] == joined[-1][2] and n[0] - joined[-1][1] < 0.03:
            s, e, p, a = joined[-1]
            joined[-1] = (s, n[1], p, a)
            at[-1].append(n[0])
        else:
            joined.append(n)
            at.append([])
    return joined, at


def _pluck_points(ctx, notes, at, grid=True):
    """Where a long note might be re-plucked: where detection split it (`at`), and (with
    `grid`) beat-grid points inside it away from those: (note index, time)."""
    beats = np.asarray(ctx.get("grid") or []) if grid else np.zeros(0)
    out = []
    for k, (s, e, p, a) in enumerate(notes):
        here = [float(t) for t in at[k]]
        here += [float(g) for g in beats[(beats > s + 0.06) & (beats < e - 0.06)] if all(abs(g - t) > 0.06 for t in at[k])]
        out += [(k, t) for t in sorted(here)]
    return out


def _pluck_features(ctx, notes, points):
    """Per point: attack sharpness vs the note, onset strength for the pitch, loudness
    and overtone rise, activation rise, time into/left of the note in grid steps."""
    flux, fps = ctx["flux"]
    times, on, fr, bass = ctx["times"], ctx["onsets"], ctx["frames"], ctx["bass"]
    grid = np.asarray(ctx["grid"])
    step = float(np.median(np.diff(grid))) if len(grid) > 1 else 0.25
    X = []
    for k, g in points:
        s, e, p, a = notes[k]
        base = np.median(flux[int(s * fps):max(int(s * fps) + 1, int(e * fps))]) + 1e-9
        j, i = int(g * fps), int(np.searchsorted(times, g))
        rise, between = _overtone_jump(bass, g, p)
        X.append([np.log1p(flux[max(0, j - 3):j + 4].max() / base), on[max(0, i - 3):i + 4, p - MIDI_OFFSET].max(),
                  _low_bump(bass, g), rise, rise - between,
                  fr[i:i + 4, p - MIDI_OFFSET].mean() - fr[max(0, i - 6):max(1, i - 2), p - MIDI_OFFSET].mean(),
                  np.log((g - s) / step), np.log((e - g) / step), a])
    return X


def note_context(bass, mix, activations, grid):
    """What the note models measure, for one song."""
    return {"bass": bass, "mix": mix, "frames": activations["frames"], "onsets": activations["onsets"],
            "times": activations["times"], "grid": list(grid or []), "flux": _attack_strength(bass)}


def _octave_step(models, ctx, notes):
    if not notes:
        return notes
    X = _octave_features(ctx, notes)
    down, up = _prob(models["down"], X), _prob(models["up"], X)
    return [(s, e, p - 12 if d >= OCTAVE_CUT and d >= u else p + 12 if u >= OCTAVE_CUT else p, a)
            for (s, e, p, a), d, u in zip(notes, down, up)]


def _pluck_step(model, ctx, notes, grid=True):
    notes, at = _join_repeats(notes)
    points = _pluck_points(ctx, notes, at, grid)
    if not points:
        return notes
    cuts = defaultdict(list)
    for (k, g), yes in zip(points, _prob(model, _pluck_features(ctx, notes, points)) >= PLUCK_CUT):
        if yes:
            cuts[k].append(g)
    out = []
    for k, (s, e, p, a) in enumerate(notes):
        edges = [s, *cuts[k], e]
        out += [(a0, a1, p, a) for a0, a1 in zip(edges, edges[1:])]
    return out


def _keep_step(model, ctx, notes):
    if not notes:
        return notes
    return [n for n, ok in zip(notes, _prob(model, _keep_features(ctx, notes)) >= KEEP_CUT) if ok]


def apply_note_models(models, ctx, notes, split=True):
    """Notes after the learned octave fix, re-pluck decisions (at the beat grid too if
    `split`) and fake-note filter."""
    notes = _octave_step(models["octave"], ctx, notes)
    if models.get("pluck"):
        notes = _pluck_step(models["pluck"], ctx, notes, grid=split)
    return _keep_step(models["keep"], ctx, notes)


def _match(truth, notes):
    """For each note: the tab's pitch at that moment (within 70 ms) relative to it:
    0 same, -12/-24/+12 octave off, None nothing there."""
    tt = np.array([t for t, p in truth])
    tp = np.array([p for t, p in truth])
    out = []
    for s, e, p, a in notes:
        near = tp[np.abs(tt - s) < 0.07]
        out.append(0 if p in near else -12 if (p - 12 in near or p - 24 in near) else 12 if p + 12 in near else None)
    return out


def learn_note_models(examples):
    """Fit the three models to examples [(context, notes as detected, tab [(time, pitch)])]."""
    Xo, yo = [], []
    for ctx, notes, truth in examples:
        for x, m in zip(_octave_features(ctx, notes), _match(truth, notes)):
            if m is not None:
                Xo.append(x)
                yo.append(m)
    yo = np.array(yo)
    octave = {"down": _fit(Xo, yo < 0), "up": _fit(Xo, yo > 0)}
    stage = [(ctx, _octave_step(octave, ctx, notes), truth) for ctx, notes, truth in examples]
    Xp, yp = [], []
    for ctx, notes, truth in stage:
        notes, at = _join_repeats(notes)
        points = _pluck_points(ctx, notes, at)
        if points:
            Xp += _pluck_features(ctx, notes, points)
            tt, tp = np.array([t for t, p in truth]), np.array([p for t, p in truth])
            yp += [bool(np.any((np.abs(tt - g) < 0.07) & (tp == notes[k][2]))) for k, g in points]
    pluck = _fit(Xp, yp) if len(set(yp)) == 2 else None
    if pluck:
        stage = [(ctx, _pluck_step(pluck, ctx, notes), truth) for ctx, notes, truth in stage]
    Xk, yk = [], []
    for ctx, notes, truth in stage:
        Xk += _keep_features(ctx, notes)
        yk += [m == 0 for m in _match(truth, notes)]
    return {"octave": octave, "pluck": pluck, "keep": _fit(Xk, yk)}


def model_counts(truth, notes):
    """(matched, detected, in the tab) over the tab's span."""
    lo, hi = truth[0][0] - 1, truth[-1][0] + 1
    notes = [n for n in notes if lo <= n[0] <= hi]
    tt, tp = np.array([t for t, p in truth]), np.array([p for t, p in truth])
    used = np.zeros(len(tt), bool)
    hit = 0
    for s, e, p, a in notes:
        k = np.nonzero((np.abs(tt - s) < 0.07) & (tp == p) & ~used)[0]
        if len(k):
            used[k[0]] = True
            hit += 1
    return hit, len(notes), len(tt)
