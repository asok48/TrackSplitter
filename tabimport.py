"""Bring in bass tabs people trust (Guitar Pro, MusicXML, or text pasted from a tab
site) and line them up with the song's recording.

The user plays from tabs found online (Songsterr, Ultimate Guitar...) and found the
automatic tab (tab.py) not trustworthy enough. Files and pasted text are brought in by
the user; the app doesn't fetch from those sites (their terms don't allow automated
downloading, and Songsterr's Guitar Pro download is a paid feature).

Every format is read into the same shape, a TabTrack: the tuning and the bars as
written (time signature, tempo changes, repeat signs, notes with string and fret). Then
`timeline()` writes out the repeats and gives each note a time in seconds from the
tab's own tempo, and `sync()` maps those times onto the recording.
"""

import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


@dataclass
class TabNote:
    pos: float          # quarter notes from the start of the bar
    dur: float          # quarter notes
    string: int         # 0 = lowest string
    fret: int
    tie: bool = False   # continues the previous note on this string
    dead: bool = False  # muted ("x")


@dataclass
class TabBar:
    beats: float                                   # bar length in quarter notes
    tempos: list = field(default_factory=list)     # [(pos, bpm)] changes in this bar
    repeat_open: bool = False
    repeat_close: int = 0                          # total plays when the repeat closes here
    endings: set = field(default_factory=set)      # alternative endings this bar belongs to
    notes: list = field(default_factory=list)


@dataclass
class TabTrack:
    name: str
    strings: list       # open-string MIDI notes, lowest first
    bars: list
    tempo: float = 120.0
    timed: bool = True  # False for text tabs: positions are only the order of the notes


def is_bass(track):
    return bool(track.strings) and min(track.strings) <= 35 or "bass" in track.name.lower()


# --- Guitar Pro 3-5 (PyGuitarPro) -------------------------------------------------

def read_gp5(path):
    import guitarpro
    song = guitarpro.parse(str(path))
    tracks = []
    for t in song.tracks:
        if t.isPercussionTrack or not t.strings:
            continue
        n = len(t.strings)
        strings = [s.value for s in sorted(t.strings, key=lambda s: -s.number)]   # number 1 = highest
        bars = []
        for measure in t.measures:
            h = measure.header
            bar = TabBar(beats=h.timeSignature.numerator * 4 / h.timeSignature.denominator.value,
                         repeat_open=h.isRepeatOpen, repeat_close=max(0, h.repeatClose),
                         endings={i + 1 for i in range(8) if h.repeatAlternative >> i & 1})
            for beat in measure.voices[0].beats:
                pos = (beat.start - measure.start) / 960
                mix = beat.effect.mixTableChange if beat.effect else None
                if mix is not None and mix.tempo is not None and mix.tempo.value > 0:
                    bar.tempos.append((pos, float(mix.tempo.value)))
                for note in beat.notes:
                    if note.type.name == "rest":
                        continue
                    bar.notes.append(TabNote(pos, beat.duration.time / 960, n - note.string, note.value,
                                             tie=note.type.name == "tie", dead=note.type.name == "dead"))
            bars.append(bar)
        tracks.append(TabTrack(t.name, strings, bars, float(song.tempo)))
    return tracks


# --- Guitar Pro 7/8 (.gp: a zip holding Content/score.gpif) ------------------------
# Written from the format's known layout; not yet checked against a real file.

VALUES = {"Whole": 4.0, "Half": 2.0, "Quarter": 1.0, "Eighth": 0.5, "16th": 0.25, "32nd": 0.125,
          "64th": 0.0625, "128th": 0.03125}


def read_gp(path):
    with zipfile.ZipFile(path) as z:
        name = next((n for n in z.namelist() if n.endswith("score.gpif")), None)
        if not name:
            raise ValueError("This .gp file has no score inside.")
        root = ElementTree.fromstring(z.read(name))
    by_id = lambda tag: {el.get("id"): el for el in root.iter(tag) if el.get("id") is not None}
    bars, voices, beats, notes, rhythms = (by_id(t) for t in ("Bar", "Voice", "Beat", "Note", "Rhythm"))
    tempo_changes = {}   # master bar -> [(fraction of the bar, bpm)]
    for a in root.iter("Automation"):
        if a.findtext("Type") == "Tempo":
            bar = int(a.findtext("Bar") or 0)
            tempo_changes.setdefault(bar, []).append((float(a.findtext("Position") or 0),
                                                      float((a.findtext("Value") or "120").split()[0])))
    masters = list(root.iter("MasterBar"))

    def length(rhythm_id):
        r = rhythms.get(rhythm_id)
        if r is None:
            return 1.0
        d = VALUES.get(r.findtext("NoteValue") or "Quarter", 1.0)
        dots = r.find("AugmentationDot")
        if dots is not None:
            d *= {1: 1.5, 2: 1.75}.get(int(dots.get("count") or 1), 1.5)
        t = r.find("PrimaryTuplet")
        if t is not None:
            d *= int(t.get("den")) / int(t.get("num"))
        return d

    tracks = []
    for ti_, track in enumerate(root.iter("Track")):
        pitches = None
        for prop in track.iter("Property"):
            if prop.get("name") == "Tuning" and prop.findtext("Pitches"):
                pitches = [int(x) for x in prop.findtext("Pitches").split()]   # lowest string first
                break
        if not pitches:
            continue
        tbars, first_tempo = [], None
        for mi, mb in enumerate(masters):
            num, den = (int(x) for x in (mb.findtext("Time") or "4/4").split("/"))
            rep = mb.find("Repeat")
            bar = TabBar(beats=num * 4 / den,
                         repeat_open=rep is not None and rep.get("start") == "true",
                         repeat_close=int(rep.get("count") or 2) if rep is not None and rep.get("end") == "true" else 0,
                         endings={int(x) for x in (mb.findtext("AlternateEndings") or "").split()})
            for frac, bpm in tempo_changes.get(mi, []):
                bar.tempos.append((frac * bar.beats, bpm))
                first_tempo = first_tempo or bpm
            bar_ids = (mb.findtext("Bars") or "").split()
            b = bars.get(bar_ids[ti_]) if ti_ < len(bar_ids) else None
            voice_ids = (b.findtext("Voices") or "").split() if b is not None else []
            v = voices.get(voice_ids[0]) if voice_ids and voice_ids[0] != "-1" else None
            pos = 0.0
            for beat_id in ((v.findtext("Beats") or "").split() if v is not None else []):
                beat = beats.get(beat_id)
                if beat is None:
                    continue
                r = beat.find("Rhythm")
                dur = length(r.get("ref") if r is not None else None)
                for note_id in (beat.findtext("Notes") or "").split():
                    note = notes.get(note_id)
                    if note is None:
                        continue
                    props = {p_.get("name"): p_ for p_ in note.iter("Property")}
                    string = props.get("String")
                    fret = props.get("Fret")
                    if string is None or fret is None:
                        continue
                    tie = note.find("Tie")
                    bar.notes.append(TabNote(pos, dur, int(string.findtext("String")), int(fret.findtext("Fret")),
                                             tie=tie is not None and tie.get("destination") == "true",
                                             dead="Muted" in props))
                pos += dur
            tbars.append(bar)
        name = track.findtext("Name") or ""
        tracks.append(TabTrack(name, pitches, tbars, first_tempo or 120.0))
    return tracks


# --- MusicXML (.musicxml / .xml / .mxl) -----------------------------------------

def _pitch(el):
    p = el.find("pitch")
    if p is None:
        return None
    return 12 * (int(p.findtext("octave")) + 1) + STEP[p.findtext("step")] + int(float(p.findtext("alter") or 0))


def read_musicxml(path):
    data = Path(path).read_bytes()
    if data[:2] == b"PK":   # .mxl: zipped, the container names the score file
        with zipfile.ZipFile(path) as z:
            container = ElementTree.fromstring(z.read("META-INF/container.xml"))
            root_file = next(el.get("full-path") for el in container.iter() if el.tag.endswith("rootfile"))
            data = z.read(root_file)
    root = ElementTree.fromstring(data)
    names = {sp.get("id"): (sp.findtext("part-name") or "") for sp in root.iter("score-part")}
    tracks = []
    for part in root.iter("part"):
        divisions, beats, strings, tempo = 1, 4.0, [], None
        bars, pitched = [], []
        for measure in part.iter("measure"):
            bar = TabBar(beats=beats)
            cursor, last_start = 0.0, 0.0
            for el in measure:
                if el.tag == "attributes":
                    divisions = int(el.findtext("divisions") or divisions)
                    if el.find("time") is not None:
                        t = el.find("time")
                        beats = int(t.findtext("beats")) * 4 / int(t.findtext("beat-type"))
                        bar.beats = beats
                    tuning = el.findall("staff-details/staff-tuning")
                    if tuning:
                        lines = sorted(tuning, key=lambda s: int(s.get("line")))   # line 1 = lowest
                        strings = [12 * (int(s.findtext("tuning-octave")) + 1) + STEP[s.findtext("tuning-step")]
                                   + int(float(s.findtext("tuning-alter") or 0)) for s in lines]
                elif el.tag == "direction" or el.tag == "sound":
                    sound = el if el.tag == "sound" else el.find("sound")
                    if sound is not None and sound.get("tempo"):
                        bar.tempos.append((cursor / divisions, float(sound.get("tempo"))))
                        tempo = tempo or float(sound.get("tempo"))
                elif el.tag == "barline":
                    rep, ending = el.find("repeat"), el.find("ending")
                    if rep is not None and rep.get("direction") == "forward":
                        bar.repeat_open = True
                    if rep is not None and rep.get("direction") == "backward":
                        bar.repeat_close = int(rep.get("times") or 2)
                    if ending is not None and ending.get("type") == "start":
                        bar.endings |= {int(x) for x in re.findall(r"\d+", ending.get("number") or "")}
                elif el.tag == "backup":
                    cursor -= int(el.findtext("duration"))
                elif el.tag == "forward":
                    cursor += int(el.findtext("duration"))
                elif el.tag == "note":
                    dur = int(el.findtext("duration") or 0)
                    start = last_start if el.find("chord") is not None else cursor
                    if el.find("chord") is None:
                        cursor += dur
                        last_start = start
                    if el.find("rest") is not None or (el.findtext("voice") or "1") != "1":
                        continue
                    tie = any(t.get("type") == "stop" for t in el.findall("tie"))
                    string, fret = el.findtext("notations/technical/string"), el.findtext("notations/technical/fret")
                    if string is not None and fret is not None:
                        bar.notes.append(TabNote(start / divisions, dur / divisions, int(string), int(fret), tie=tie))
                    else:   # notation only, no tab: keep the pitch, fingered later
                        p = _pitch(el)
                        if p is not None:
                            pitched.append((len(bars), TabNote(start / divisions, dur / divisions, -1, p, tie=tie)))
            bars.append(bar)
        if not strings:
            strings = [28, 33, 38, 43]
        n = len(strings)
        for bar in bars:   # MusicXML strings count from the highest (1)
            for note in bar.notes:
                note.string = n - note.string
        for i, note in pitched:   # pitches only: place on the lowest string that has them
            s = max((k for k, o in enumerate(strings) if 0 <= note.fret - o <= 24), default=None)
            if s is not None:
                note.string, note.fret = s, note.fret - strings[s]
                bars[i].notes.append(note)
        tracks.append(TabTrack(names.get(part.get("id"), ""), strings, bars, tempo or 120.0))
    return tracks


# --- Text tabs (pasted from a tab site) -----------------------------------------

NAMES = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5, "F#": 6, "GB": 6,
         "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11}
LINE = re.compile(r"^\s*([A-Ga-g][#b]?)\s*[|:\-]")


def _open_strings(labels):
    """String names as written in a text tab (highest string first) -> open-string MIDI
    notes, lowest first: the lowest string sits between A0 and F#1 (E -> 28, D -> 26,
    B -> 23), each string above is the next note up with its name (E A D G)."""
    classes = [NAMES[l.upper()] for l in reversed(labels)]
    pitches = [24 + classes[0] if classes[0] <= 6 else 12 + classes[0]]
    for c in classes[1:]:
        pitches.append(pitches[-1] + ((c - pitches[-1]) % 12 or 12))
    return pitches


def read_text(text):
    """Systems of 4-6 lines like 'G|--5--7--|' (highest string on top). Columns give the
    order and rough spacing of notes; bar lines '|' give bars. No rhythm or tempo."""
    lines = text.replace("\t", "    ").splitlines()
    systems, block = [], []
    for line in lines + [""]:
        m = LINE.match(line)
        if m and "-" in line:
            block.append((m.group(1), line[m.end() - 1:]))
        else:
            if 4 <= len(block) <= 6:
                systems.append(block)
            block = []
    if not systems:
        return []
    labels = [l for l, _ in systems[0]]
    strings = _open_strings(labels)
    n = len(strings)
    bars, column = [], 0
    for block in systems:
        width = max(len(body) for _, body in block)
        rows = [body.ljust(width, "-") for _, body in block]
        cuts = [i for i in range(width) if all(r[i] == "|" for r in rows if i < len(r))] or [0, width]
        if cuts[0] != 0:
            cuts.insert(0, 0)
        if cuts[-1] != width:
            cuts.append(width)
        for a, b in zip(cuts, cuts[1:]):
            if b - a <= 1:
                continue
            bar = TabBar(beats=(b - a) / 4)
            for k, row in enumerate(rows):
                string = n - 1 - k        # top line = highest string
                for m in re.finditer(r"\d+|x", row[a:b]):
                    bar.notes.append(TabNote(m.start() / 4, 0.25, string, 0 if m.group() == "x" else int(m.group()),
                                             dead=m.group() == "x"))
            bars.append(bar)
        column += width
    return [TabTrack("Pasted tab", strings, bars, 120.0, timed=False)]


# --- Any file ----------------------------------------------------------------------

def read_file(path):
    """Tracks of a tab file, bass-looking ones first."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".gp3", ".gp4", ".gp5"):
        tracks = read_gp5(path)
    elif ext == ".gp":
        tracks = read_gp(path)
    elif ext == ".gpx":
        raise ValueError("Guitar Pro 6 (.gpx) files can't be read. In Guitar Pro, save it as .gp or .gp5, or export MusicXML.")
    elif ext in (".musicxml", ".xml", ".mxl"):
        tracks = read_musicxml(path)
    elif ext == ".txt":
        tracks = read_text(path.read_text(encoding="utf-8", errors="replace"))
    else:
        raise ValueError(f"Can't read {ext} files. Use Guitar Pro (.gp, .gp3, .gp4, .gp5), MusicXML (.musicxml, .mxl), or paste a text tab.")
    tracks = [t for t in tracks if any(b.notes for b in t.bars)]
    return sorted(tracks, key=lambda t: not is_bass(t))


# --- Playing order and times ------------------------------------------------------

def play_order(bars):
    """Bar indexes in playing order: repeats written out, alternative endings taken
    on the matching pass."""
    order, i, start, passes = [], 0, 0, {}
    guard = 0
    while i < len(bars) and guard < 20000:
        guard += 1
        bar = bars[i]
        if bar.repeat_open:
            start = i
        p = passes.get(start, 1)
        if bar.endings and p not in bar.endings:
            i += 1
            continue
        order.append(i)
        if bar.repeat_close and p < bar.repeat_close:
            passes[start] = p + 1
            i = start
            continue
        if bar.repeat_close:
            passes.pop(start, None)
        i += 1
    return order


def timeline(track):
    """[(tab time s, end s, string, fret, dead, bar number)] in playing order, ties
    merged into the note they continue, plus the bar start times."""
    tempo, t, notes, bar_starts = track.tempo, 0.0, [], []
    open_note = {}
    for k, idx in enumerate(play_order(track.bars)):
        bar = track.bars[idx]
        changes = sorted(bar.tempos)
        bar_starts.append(t)

        def at(pos):   # seconds from the bar start to pos, following tempo changes
            secs, last, bpm = 0.0, 0.0, tempo
            for cpos, cbpm in changes:
                if cpos >= pos:
                    break
                secs += (cpos - last) * 60 / bpm
                last, bpm = cpos, cbpm
            return secs + (pos - last) * 60 / bpm

        for note in sorted(bar.notes, key=lambda n: (n.pos, n.string)):
            start, end = t + at(note.pos), t + at(note.pos + note.dur)
            if note.tie and note.string in open_note:
                open_note[note.string][1] = end          # extend the note it continues
                continue
            row = [start, end, note.string, note.fret, note.dead, k]
            notes.append(row)
            open_note[note.string] = row
        t += at(bar.beats)
        if changes:
            tempo = changes[-1][1]
    return [tuple(n) for n in notes], bar_starts


# --- Lining the tab up with the recording -----------------------------------------
# `heard`: notes transcribed from the song's bass track (tab.transcribe), as
# (start s, end s, MIDI pitch, amplitude). The tab says what is played and (for files)
# in what rhythm; the recording says when.

def _hits(times, pitches, heard_t, heard_p, tol=0.05):
    """How many tab notes land on a heard note of the same pitch (octave slips count half)."""
    j = np.searchsorted(heard_t, times)
    score = 0.0
    for t, p, k in zip(times, pitches, j):
        for m in (k - 1, k):
            if 0 <= m < len(heard_t) and abs(heard_t[m] - t) <= tol:
                if heard_p[m] == p:
                    score += 1
                    break
                if (heard_p[m] - p) % 12 == 0:
                    score += 0.5
                    break
    return score


def _sync_timed(tab_t, tab_p, bar_starts, bar_of, heard):
    """Tab with rhythm: one overall start + tempo ratio (searched), then a small
    correction per bar (+-0.4 s, from the notes of nearby bars, median-smoothed), so
    tempo drift is followed but a single odd bar can't drag the rest."""
    heard_t = np.array([h[0] for h in heard])
    heard_p = np.array([h[2] for h in heard])
    tab_t, tab_p = np.asarray(tab_t), np.asarray(tab_p)
    # Overall: for each tempo ratio, the start that lines up most notes (onset histogram).
    best = (-1, 1.0, 0.0)
    for scale in np.arange(0.85, 1.15001, 0.0025):
        diffs = (heard_t[None, :] - tab_t[::max(1, len(tab_t) // 400), None] * scale).ravel()
        diffs = diffs[(diffs > -30) & (diffs < 600)]
        if not len(diffs):
            continue
        hist, edges = np.histogram(diffs, bins=np.arange(diffs.min(), diffs.max() + 0.05, 0.02))
        k = int(np.argmax(np.convolve(hist, np.ones(3), mode="same")))
        offset = edges[k] + 0.01
        score = _hits(tab_t * scale + offset, tab_p, heard_t, heard_p)
        if score > best[0]:
            best = (score, scale, offset)
    _, scale, offset = best
    # Per bar: best small shift for the notes in this bar and its neighbours.
    n_bars = len(bar_starts)
    shifts = np.zeros(n_bars)
    for b in range(n_bars):
        near = (bar_of >= b - 2) & (bar_of <= b + 2)
        if near.sum() < 3:
            shifts[b] = np.nan
            continue
        cands = np.arange(-0.4, 0.4001, 0.01)
        scores = [_hits(tab_t[near] * scale + offset + d, tab_p[near], heard_t, heard_p) for d in cands]
        top = max(scores)
        shifts[b] = cands[int(np.argmax(scores))] if top > 0 else np.nan
    # Fill bars with no evidence from their neighbours, then smooth.
    idx = np.arange(n_bars)
    ok = ~np.isnan(shifts)
    shifts = np.interp(idx, idx[ok], shifts[ok]) if ok.any() else np.zeros(n_bars)
    smooth = np.array([np.median(shifts[max(0, b - 2):b + 3]) for b in range(n_bars)])
    bar_audio = np.asarray(bar_starts) * scale + offset + smooth
    bar_audio = np.maximum.accumulate(bar_audio)   # never runs backwards
    return bar_audio, best[0] / max(1, len(tab_t)), scale


def _sync_order(tab_p, heard, expected=None, unit=None):
    """Tab without rhythm (pasted text): align the sequence of tab pitches with the
    sequence of heard pitches (Needleman-Wunsch: missed and extra notes allowed on both
    sides; heard notes before and after the tab are free). With expected times (from a
    text tab's spacing), matches far from where the note should be score lower, which
    settles which of several repeated notes is which. Returns, per tab note, the index
    of the heard note it matched, or -1."""
    a, b = list(tab_p), [h[2] for h in heard]
    n, m = len(a), len(b)
    MATCH, OCTAVE, MISS, SKIP_TAB, SKIP_HEARD = 2.0, 1.0, -1.5, -1.0, -0.6
    score = np.zeros((n + 1, m + 1))
    move = np.zeros((n + 1, m + 1), np.int8)       # 1 diagonal, 2 skip tab note, 3 skip heard note
    score[1:, 0] = SKIP_TAB * np.arange(1, n + 1)
    move[1:, 0] = 2
    move[0, 1:] = 3                                 # heard notes before the tab: free
    for i in range(1, n + 1):
        p = a[i - 1]
        q = np.array(b)
        sub = np.where(q == p, MATCH, np.where((q - p) % 12 == 0, OCTAVE, MISS))
        if expected is not None:   # off by a whole beat costs a full match
            sub = sub - MATCH * np.minimum(1.0, np.abs(np.array([h[0] for h in heard]) - expected[i - 1]) / unit)
        diag = score[i - 1, :-1] + sub
        up = score[i - 1, 1:] + SKIP_TAB
        row = np.empty(m + 1)
        row[0] = score[i, 0]
        mv = np.empty(m + 1, np.int8)
        mv[0] = 2
        for j in range(1, m + 1):                   # left moves depend on the row itself
            best, how = (diag[j - 1], 1) if diag[j - 1] >= up[j - 1] else (up[j - 1], 2)
            left = row[j - 1] + SKIP_HEARD
            if left > best:
                best, how = left, 3
            row[j], mv[j] = best, how
        score[i], move[i] = row, mv
    j = int(np.argmax(score[n]))                    # heard notes after the tab: free
    match = [-1] * n
    i = n
    while i > 0 and j >= 0:
        how = move[i, j]
        if how == 1:
            if a[i - 1] % 12 == b[j - 1] % 12:
                match[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif how == 2:
            i -= 1
        else:
            j -= 1
    return match


def sync(track, heard, beat=None):
    """The tab's notes and bars on the recording's clock (beat: the song's beat length in
    seconds, from the beat tracker; text tabs need it, they have no tempo of their own).

    Returns {"notes": [[start, end, string, fret, dead]], "bars": [start s],
    "quality": share of tab notes that matched a heard note}."""
    notes, bar_starts = timeline(track)
    playable = [k for k, n in enumerate(notes) if not n[4]]
    tab_t = [notes[k][0] for k in playable]
    tab_p = [track.strings[notes[k][2]] + notes[k][3] for k in playable]
    if not heard or not playable:
        return {"notes": [list(n[:5]) for n in notes], "bars": bar_starts, "quality": 0.0}
    if track.timed:
        bar_of = np.array([notes[k][5] for k in playable])
        bar_audio, quality, scale = _sync_timed(tab_t, tab_p, bar_starts, bar_of, heard)
        ends = list(bar_starts[1:]) + [bar_starts[-1] + (bar_starts[-1] - bar_starts[-2] if len(bar_starts) > 1 else 2.0)]

        # Bars from the overall fit are a little rough on a noisy separated track (77% of
        # notes within 50 ms end to end on a test song); placing notes inside each bar like
        # text tabs do (matched in order, guided by the exact rhythm) fixes that.
        out, matched = _place_in_bars(track, notes, playable, heard, list(bar_audio), bar_starts, ends,
                                      lambda b: (ends[b] - bar_starts[b]) * scale, keep_lengths=True)
        return {"notes": [[round(n[0], 3), round(n[1], 3), *n[2:]] for n in out],
                "bars": [round(float(x), 3) for x in bar_audio], "quality": round(matched / max(1, len(playable)), 2)}
    # Order only (text tab): spacing inside a bar isn't reliable, and matching the whole
    # pitch sequence slips by a riff repetition (60% right on a test line). So bars are
    # placed by their notes' content (which pitches, not where), at the song's own bar
    # length, then notes are matched in order within each bar.
    return _sync_text(track, notes, playable, tab_p, heard, beat)


def _content(bar_pcs, heard_t, heard_pc, a, b):
    """How many of a bar's pitch classes (a multiset) the recording has between a and b."""
    have = {}
    for pc in heard_pc[(heard_t >= a) & (heard_t < b)]:
        have[pc] = have.get(pc, 0) + 1
    return sum(min(c, have.get(pc, 0)) for pc, c in bar_pcs.items())


def _sync_text(track, notes, playable, tab_p, heard, beat):
    _, bar_starts_t = timeline(track)
    bar_ends = bar_starts_t[1:] + [bar_starts_t[-1] + (bar_starts_t[-1] - bar_starts_t[-2] if len(bar_starts_t) > 1 else 2.0)]
    heard_t = np.array([h[0] for h in heard])
    heard_pc = np.array([h[2] % 12 for h in heard])
    n_bars = max(n[5] for n in notes) + 1
    pcs = [{} for _ in range(n_bars)]
    for k, p in zip(playable, tab_p):
        d = pcs[notes[k][5]]
        d[p % 12] = d.get(p % 12, 0) + 1
    best = (-1, 2.0, 0.0)
    for length in [4 * beat * f for f in (0.5, 1.0, 2.0)] if beat else np.arange(1.2, 4.01, 0.05):
        # Overall start: every (bar, heard note of one of its pitch classes) pair votes for
        # the starts that would put that note inside that bar.
        votes = np.zeros(int((heard_t[-1] + 5) / 0.05) + 1)
        for b, d in enumerate(pcs):
            for pc in d:
                for t in heard_t[heard_pc == pc]:
                    if t - b * length < 0:      # this note can't be in this bar for any start >= 0
                        continue
                    lo, hi = int(max(0, t - (b + 1) * length) / 0.05), int((t - b * length) / 0.05)
                    votes[lo:hi + 1] += 1
        start = float(np.argmax(votes) * 0.05)
        score = sum(_content(d, heard_t, heard_pc, start + b * length, start + (b + 1) * length) for b, d in enumerate(pcs))
        if score > best[0]:
            best = (score, length, start)
    _, length, start = best
    # Then follow the bars in order like counting along: each bar is expected one bar
    # after the previous one and may move +-0.3 s from there, judged by how well its
    # notes match in order (a riff repeated in look-alike bars can't make it jump).
    def expect(ks, a0):   # where the text's spacing puts each note in a bar starting at a0
        b = notes[ks[0]][5]
        span = max(1e-6, bar_ends[b] - bar_starts_t[b])
        return [a0 + (notes[k][0] - bar_starts_t[b]) / span * length for k in ks]

    def ordered(ks, a0):
        inside = [h for h in heard if a0 - 0.08 <= h[0] < a0 + length - 0.08]
        if not inside:
            return 0.0
        seq = [track.strings[notes[k][2]] + notes[k][3] for k in ks]
        m = _sync_order(seq, inside, expect(ks, a0), length / 4)
        return sum(1.0 if inside[j][2] == p else 0.5 for j, p in zip(m, seq) if j >= 0)

    bar_audio, prev = np.zeros(n_bars), None
    for b in range(n_bars):
        expected = start + b * length if prev is None else prev + length
        ks = [k for k in playable if notes[k][5] == b]
        if ks:
            cands = np.arange(-0.3, 0.3001, 0.02)
            scores = [ordered(ks, expected + c) for c in cands]
            top = max(scores)
            if top > 0:   # ties: the smallest move
                expected += min((c for c, sc in zip(cands, scores) if sc == top), key=abs)
        bar_audio[b] = prev = expected
    out, matched = _place_in_bars(track, notes, playable, heard, bar_audio, bar_starts_t, bar_ends,
                                  lambda b: length, keep_lengths=False)
    return {"notes": [[round(n[0], 3), round(n[1], 3), *n[2:]] for n in out],
            "bars": [round(float(x), 3) for x in bar_audio], "quality": round(matched / max(1, len(playable)), 2)}


def _place_in_bars(track, notes, playable, heard, bar_audio, bar_starts_t, bar_ends_t, audio_len, keep_lengths):
    """Each bar's notes on the recording: matched in order to the heard notes in that bar
    (guided by where the tab puts them), unmatched ones placed by the tab's own timing
    between their matched neighbours. keep_lengths: notes keep the tab's lengths (files
    with rhythm); otherwise each lasts until the next (text tabs)."""
    out = [[float(n[0]), float(n[1]), n[2], n[3], n[4]] for n in notes]
    matched = 0
    for b in range(len(bar_audio)):
        a0 = bar_audio[b]
        a1 = bar_audio[b + 1] if b + 1 < len(bar_audio) else a0 + audio_len(b)
        span = max(1e-6, bar_ends_t[b] - bar_starts_t[b])
        rate = (a1 - a0) / span
        ks = [k for k in range(len(notes)) if notes[k][5] == b]
        if not ks:
            continue
        expected = [a0 + (notes[k][0] - bar_starts_t[b]) * rate for k in ks]
        play = [i for i, k in enumerate(ks) if k in playable]
        inside = [h for h in heard if a0 - 0.08 <= h[0] < a1 - 0.08]
        match = (_sync_order([track.strings[notes[ks[i]][2]] + notes[ks[i]][3] for i in play], inside,
                             [expected[i] for i in play], (a1 - a0) / 4) if inside and play else [-1] * len(play))
        # Where a note sits in the bar: its exact time for files with rhythm; for text tabs
        # just its place in the order (hand-typed spacing is too sloppy: 72% vs 82% on a test).
        where = (lambda i: notes[ks[i]][0]) if keep_lengths else (lambda i: i + 1)
        anchor_x = [bar_starts_t[b] if keep_lengths else 0]
        anchor_t = [a0]
        for i, j in zip(play, match):
            if j >= 0:
                anchor_x.append(where(i))
                anchor_t.append(inside[j][0])
                matched += 1
        anchor_x.append(bar_ends_t[b] if keep_lengths else len(ks) + 1)
        anchor_t.append(a1)
        order = np.argsort(anchor_x, kind="stable")
        ax, at = np.array(anchor_x)[order], np.maximum.accumulate(np.array(anchor_t)[order])
        for i, k in enumerate(ks):
            t = float(np.interp(where(i), ax, at))
            length = (notes[k][1] - notes[k][0]) * rate
            out[k][0], out[k][1] = t, t + length
    if not keep_lengths:
        for k in range(len(out)):   # each note lasts until the next one starts
            nxt = next((out[j][0] for j in range(k + 1, len(out)) if out[j][0] > out[k][0]), out[k][0] + 0.3)
            out[k][1] = min(nxt, out[k][0] + 1.0)
    return out, matched
