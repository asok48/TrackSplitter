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
from dataclasses import dataclass, field, replace
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
    # Slides and hammer-ons / pull-offs, as letters (shown in the tab, and the game doesn't
    # want a pluck for a note slid or hammered into):
    legato_in: str = ""    # "b" / "a" slid in from below / above, "h" hammered or pulled into
    legato_out: str = ""   # "s" slides / "h" hammers or pulls to the next note on this string, "d" / "u" slides off down / up


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
    timed: bool = True  # False: positions are only the order of the notes (sync's fallback for spaced text tabs)
    capo: int = 0       # frets are written from the capo: a note sounds strings[s] + capo + fret
    tempo_known: bool = True   # False for text tabs with counts: rhythm but no tempo (sync guesses it)
    spaced: bool = False       # text tab without counts: rhythm read from the spacing (see read_text)

    def pitch(self, string, fret):
        return self.strings[string] + self.capo + fret


def is_bass(track):
    return bool(track.strings) and min(track.strings) <= 35 or "bass" in track.name.lower()


def is_guitar(track):
    """6 or more strings tuned like a guitar (Guitar Pro also gives vocal and keyboard
    tracks a guitar's strings; their names usually say so)."""
    name = track.name.lower()
    return (len(track.strings) >= 6 and min(track.strings) >= 33 and not is_bass(track)
            and not any(w in name for w in ("vocal", "voice", "sax", "piano", "keys", "synth", "organ", "string ens")))


def note_count(track):
    return sum(1 for b in track.bars for n in b.notes if not n.tie)


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
                    slides = {s_.name for s_ in (note.effect.slides or [])}
                    bar.notes.append(TabNote(pos, beat.duration.time / 960, n - note.string, note.value,
                                             tie=note.type.name == "tie", dead=note.type.name == "dead",
                                             legato_in="b" if "intoFromBelow" in slides else "a" if "intoFromAbove" in slides else "",
                                             legato_out="s" if slides & {"shiftSlideTo", "legatoSlideTo"} else "h" if note.effect.hammer
                                             else "d" if "outDownwards" in slides else "u" if "outUpwards" in slides else ""))
            bars.append(bar)
        tracks.append(TabTrack(t.name, strings, bars, float(song.tempo), capo=max(0, t.offset or 0)))
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
        pitches, capo = None, 0
        for prop in track.iter("Property"):
            if prop.get("name") == "Tuning" and prop.findtext("Pitches") and pitches is None:
                pitches = [int(x) for x in prop.findtext("Pitches").split()]   # lowest string first
            elif prop.get("name") == "CapoFret":
                capo = int(prop.findtext("Fret") or 0)
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
                    # Slide flags: 1 shift slide / 2 legato slide to the next note, 4 / 8 slide out
                    # down / up, 16 / 32 slide in from below / above; hammer-ons and pull-offs:
                    # HopoOrigin/Destination.
                    slide = int(props["Slide"].findtext("Flags") or 0) if "Slide" in props else 0
                    bar.notes.append(TabNote(pos, dur, int(string.findtext("String")), int(fret.findtext("Fret")),
                                             tie=tie is not None and tie.get("destination") == "true",
                                             dead="Muted" in props,
                                             legato_in="b" if slide & 16 else "a" if slide & 32 else "h" if "HopoDestination" in props else "",
                                             legato_out="s" if slide & 3 else "h" if "HopoOrigin" in props
                                             else "d" if slide & 4 else "u" if slide & 8 else ""))
                pos += dur
            tbars.append(bar)
        name = track.findtext("Name") or ""
        tracks.append(TabTrack(name, pitches, tbars, first_tempo or 120.0, capo=max(0, capo)))
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
        name = names.get(part.get("id"), "")
        if not strings:   # notation only: a guitar's strings if it's named or pitched like one
            heard = sorted(n.fret for _, n in pitched)
            guitar = "guitar" in name.lower() or (heard and heard[len(heard) // 2] >= 52 and "bass" not in name.lower())
            strings = [40, 45, 50, 55, 59, 64] if guitar else [28, 33, 38, 43]
        n = len(strings)
        for bar in bars:   # MusicXML strings count from the highest (1)
            for note in bar.notes:
                note.string = n - note.string
        if len(strings) >= 6 and pitched:   # a guitar's chords: fingered as hand shapes
            import tab
            order = sorted(range(len(pitched)), key=lambda k: (pitched[k][0], pitched[k][1].pos))
            fake = [(pitched[k][0] * 100 + pitched[k][1].pos, 0, pitched[k][1].fret, 1.0) for k in order]
            key = next((k for k, v in tab.GUITAR_TUNINGS.items() if v[1] == strings), None)
            if key:
                for j, s, f in tab.guitar_fingering(fake, key):
                    i, note = pitched[order[j]]
                    note.string, note.fret = s, f
                    bars[i].notes.append(note)
                pitched = []
        for i, note in pitched:   # pitches only: place on the lowest string that has them
            s = max((k for k, o in enumerate(strings) if 0 <= note.fret - o <= 24), default=None)
            if s is not None:
                note.string, note.fret = s, note.fret - strings[s]
                bars[i].notes.append(note)
        tracks.append(TabTrack(name, strings, bars, tempo or 120.0))
    return tracks


# --- Text tabs (pasted from a tab site) -----------------------------------------

NAMES = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5, "F#": 6, "GB": 6,
         "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11}
LINE = re.compile(r"^\s*([A-Ga-g][#b]?)\s*[|:\-]")


def _open_strings(labels):
    """String names as written in a text tab (highest string first) -> open-string MIDI
    notes, lowest first: the lowest string sits between A0 and F#1 (E -> 28, D -> 26,
    B -> 23) on a bass (4-5 lines), G1-F#2 on a guitar (6 lines: E -> 40, D -> 38), each
    string above is the next note up with its name (E A D G, E A D G B E)."""
    classes = [NAMES[l.upper()] for l in reversed(labels)]
    octave = 12 if len(labels) >= 6 else 0
    pitches = [octave + (24 + classes[0] if classes[0] <= 6 else 12 + classes[0])]
    for c in classes[1:]:
        pitches.append(pitches[-1] + ((c - pitches[-1]) % 12 or 12))
    return pitches


COUNT = re.compile(r"^\s*1(\s+[1-9+&ea])+\s*$")   # a count line under a system: "1 + 2 + 3 + 4 + 1 + ..."
REPEATS = re.compile(r"[x×]\s*(\d+)|(\d+)\s*[x×]")


def _row_notes(row, a, b):
    """The notes in row[a:b] as (column, fret or None for "x", legato in, legato out): 5h7 /
    5p4 hammer or pull to the next note, 5/7 5\\3 slide to it, 7\\- 7/- slide off down / up,
    -/5 -\\5 slide in from below / above."""
    out = []
    for m in re.finditer(r"\d+|x", row[a:b]):
        i, j = a + m.start(), a + m.end()
        before, after = row[i - 1] if i > a else "", row[j] if j < b else ""
        nxt = row[j + 1] if j + 1 < b else ""
        if after == "-" and nxt in ("/", "\\") and not row[j + 2:j + 3].isdigit():   # "7-\\-": slides off
            after, nxt = nxt, ""
        out_mark = ("h" if after in "hp" and nxt.isdigit() else "s" if after in "/\\" and nxt.isdigit()
                    else "u" if after == "/" else "d" if after == "\\" else "") if after else ""
        prev = row[i - 2] if i - 2 >= a else ""
        in_mark = ("b" if before == "/" else "a" if before == "\\" else "") if before and not prev.isdigit() else ""
        out.append((i, None if m.group() == "x" else int(m.group()), in_mark, out_mark))
    return out


def read_text(text):
    """Systems of 4-6 lines like 'G|--5--7--|' (highest string on top; 6 lines: a guitar).
    Columns give the order and rough spacing of notes, notes in one column are played
    together; bar lines '|' give bars. With a count line under
    every system ("1 + 2 + 3 + 4 + ...", as Ultimate Guitar bass tabs have; user: "I also
    want to be able to use ultimate guitar tabs, even if it is just pasting") the counts
    give the bars and the rhythm, so the tab is synced like a file (its tempo is guessed
    from the song's, see sync). Also read: repeat signs (|o ... o| or |: ... :|) with a
    play count after the system ("x4", default 2), hammer-ons / pull-offs (h, p) and
    slides (/ \\). Text after a system's last bar line isn't read as notes ("x4" was a
    dead note and a 4). Without count lines the rhythm comes from the spacing
    (`_rhythm_from_spacing`)."""
    lines = text.replace("\t", "    ").splitlines()
    systems, block = [], []
    for line in lines + [""]:
        m = LINE.match(line)
        if m and "-" in line:
            block.append((m.group(1), line, m.end() - 1))
            continue
        if 4 <= len(block) <= 6:   # a count line comes right under its system
            systems.append({"rows": block, "count": line if COUNT.match(line) else None})
        block = []
    if not systems:
        return []
    labels = [l for l, _, _ in systems[0]["rows"]]
    strings = _open_strings(labels)
    n = len(strings)
    timed = all(sy["count"] for sy in systems)
    bars = []
    for sy in systems:
        off = sy["rows"][0][2]
        rows, tail = [], ""
        for _, line, o in sy["rows"]:
            body = line[o:]
            last = body.rfind("|")
            if last > 0 and body[last + 1:].strip():
                tail += " " + body[last + 1:]
                body = body[:last + 1]
            rows.append(" " * (o - off) + body)
        width = max(len(r) for r in rows)
        rows = [r.ljust(width, "-") for r in rows]
        # Repeat signs, then blanked so they aren't read as anything else.
        opens = [i for i in range(width - 1) if any(r[i] == "|" and r[i + 1] in "o:" for r in rows)]
        closes = [i for i in range(1, width) if any(r[i] == "|" and r[i - 1] in "o:" for r in rows)]
        rows = [re.sub(r"(?<=\|)[o:]|[o:](?=\|)", "-", r) for r in rows]
        times = REPEATS.search(tail)
        plays = int(times.group(1) or times.group(2)) if times else 2
        cuts = [i for i in range(width) if all(r[i] == "|" for r in rows if i < len(r))]
        per_beat, edges = 4.0, sorted(set([0, width] + cuts))
        if timed:
            count = sy["count"][off:] if len(sy["count"]) > off else ""
            ones = [m.start() for m in re.finditer(r"(?<!\S)1(?!\S)", count)]
            twos = [m.start() for m in re.finditer(r"(?<!\S)2(?!\S)", count)]
            if ones and twos and twos[0] > ones[0]:
                per_beat = float(twos[0] - ones[0])
            if ones:
                measure = (ones[1] - ones[0]) if len(ones) > 1 else 4 * per_beat
                end = ones[-1] + measure
                edges = sorted(set(ones + [min(end, width)] + [c for c in cuts if c > end] + [width]))
        notes = [(k, note) for k, r in enumerate(rows) for note in _row_notes(r, 0, width)]
        starts = sorted({note[0] for _, note in notes})
        made = []   # (first column, last column, bar) of this system
        for a, b in zip(edges, edges[1:]):
            inside = [(k, note) for k, note in notes if a <= note[0] < b or (a == edges[0] and note[0] < a)]
            if b - a <= 1 or (timed and not inside and b - a < per_beat):
                continue
            # Without counts a bar's time starts after its bar line and the "-" padding
            # Ultimate Guitar puts after it ("|-3---3---...": the first 8th is at 0, not
            # a quarter beat in, which left a gap at every bar line).
            lead = 0 if timed else sum(1 for c in (a, a + 1) if c < b - 1 and all(r[c] in "|-" for r in rows))
            a += lead
            bar = TabBar(beats=(b - a) / per_beat)
            for k, (col, fret, into, out) in inside:
                if timed:   # until the next note on any string, within the bar
                    nxt = next((c for c in starts if c > col), b)
                    dur = max(0.25, (min(nxt, b) - col) / per_beat)
                else:
                    dur = 0.25
                bar.notes.append(TabNote(max(0, col - a) / per_beat, dur, n - 1 - k, fret or 0,
                                         dead=fret is None, legato_in=into, legato_out=out))
            made.append((a - lead, b, bar))
        # A repeat opens on the first bar from its sign on, and closes on the last bar
        # that ends by its sign (the space before a count's first 1 and after its last
        # measure isn't a bar).
        for c in opens:
            hit = next((m for m in made if m[0] >= c - 2), None)
            if hit:
                hit[2].repeat_open = True
        for c in closes:
            hit = next((m for m in reversed(made) if m[1] <= c + 2), None)
            if hit:
                hit[2].repeat_close = plays
        bars += [m[2] for m in made]
    if not timed:
        _rhythm_from_spacing(bars)
    return [TabTrack("Pasted tab", strings, bars, 120.0, timed=True, tempo_known=False, spaced=not timed)]


def _rhythm_from_spacing(bars):
    """A text tab without count lines: take the rhythm from the spacing. Ultimate Guitar
    lays tabs out roughly in proportion to time (a steady 8th every 4 columns, the same bar
    29-33 columns wide), so a bar is 4 beats unless it's much shorter or longer than the
    typical bar (then its width against that, in whole beats: "|--------|" is 1), and notes
    sit where their column puts them, each lasting until the next. Read as order only, the
    strummed chorus of So Far So Fake came out with its steady 8ths 0.03-0.8 s apart (user:
    "some of it comes in ok, but other parts are just flat out wrong"): each note was pulled
    onto whatever the guitar transcription heard nearby."""
    widths = [b.beats for b in bars if b.notes]
    if not widths:
        return
    typical = float(np.median(widths))
    for bar in bars:
        ratio = bar.beats / typical
        beats = 4.0 if 0.7 <= ratio <= 1.3 else max(1.0, float(round(4 * ratio)))
        k = beats / bar.beats
        starts = sorted({n.pos * k for n in bar.notes})
        for n in bar.notes:
            n.pos *= k
            nxt = next((s for s in starts if s > n.pos + 1e-9), beats)
            n.dur = max(0.125, nxt - n.pos)
        bar.beats = beats


# --- Any file ----------------------------------------------------------------------

def read_tracks(path):
    """Every track of a tab file that has notes, in the file's order."""
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
    return [t for t in tracks if any(b.notes for b in t.bars)]


def read_file(path, instrument="bass"):
    """Tracks of a tab file, the ones for this instrument first (guitar: tracks named
    guitar first, then the busiest)."""
    return rank_tracks(read_tracks(path), instrument)


def rank_tracks(tracks, instrument="bass"):
    if instrument == "guitar":
        return sorted(tracks, key=lambda t: (not is_guitar(t), "guitar" not in t.name.lower(), -note_count(t)))
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
    """[(tab time s, end s, string, fret, dead, bar number, legato)] in playing order,
    ties merged into the note they continue, plus the bar start times. legato: letters for
    how the note is reached and left: "s" slid from the note before on its string, "h"
    hammered or pulled from it, "b" / "a" slid in from below / above (each: starts without a
    pluck of its own), then "d" / "u" when it slides off down / up; "" for none."""
    tempo, t, notes, bar_starts = track.tempo, 0.0, [], []
    open_note, leads_on = {}, {}
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
            came = leads_on.get(note.string, "")
            into = came if came in ("s", "h") else note.legato_in
            off = note.legato_out if note.legato_out in ("d", "u") else ""
            row = [start, end, note.string, note.fret, note.dead, k, into + off]
            notes.append(row)
            open_note[note.string] = row
            leads_on[note.string] = note.legato_out
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


def _sounding(times, pitches, heard_t, heard_e, heard_p):
    """How many tab notes fall inside a heard note of the same pitch class (heard sorted by start)."""
    k = np.searchsorted(heard_t, np.asarray(times) + 0.05, side="right")
    count = 0
    for t, p, j in zip(times, pitches, k):
        for m in range(max(0, j - 8), j):
            if heard_t[m] - 0.05 <= t <= heard_e[m] and (heard_p[m] - p) % 12 == 0:
                count += 1
                break
    return count


SYNC_GRID = 0.05                                 # s: where a bar may start
SYNC_DRIFT, SYNC_JUMP, SYNC_SKIP = 0.6, 8.0, 3.0  # costs, in matched notes (see _sync_timed)


def _tempo_fit(tab_t, tab_p, heard_t, heard_p):
    """The tab's overall tempo ratio and start: for each ratio the start that lines up most
    notes (onset histogram), 0.6-1.7 in 1% steps, then 0.25% around the best. It searched
    only 0.85-1.15, which missed tabs written at another tempo."""
    def at(scale):
        diffs = (heard_t[None, :] - tab_t[::max(1, len(tab_t) // 400), None] * scale).ravel()
        diffs = diffs[(diffs > -60) & (diffs < 600)]
        if not len(diffs):
            return (-1, scale, 0.0)
        hist, edges = np.histogram(diffs, bins=np.arange(diffs.min(), diffs.max() + 0.05, 0.02))
        k = int(np.argmax(np.convolve(hist, np.ones(3), mode="same")))
        offset = edges[k] + 0.01
        return (_hits(tab_t * scale + offset, tab_p, heard_t, heard_p), scale, offset)
    coarse = max(at(x) for x in np.arange(0.6, 1.7001, 0.01))
    return max(at(x) for x in np.arange(coarse[1] - 0.012, coarse[1] + 0.0121, 0.0025))


def _bar_scores(offs, pitches, grid, by_pitch, by_pc):
    """For one bar's notes (offsets from its start): on every grid start, notes landing on a
    heard note of the same pitch (an octave off: half), plus half for each lying inside a
    heard note of that pitch class (separated bass often comes out as long merged notes)."""
    def near(times, t):        # within 50 ms of one of `times` (sorted)
        k = np.searchsorted(times, t)
        a, b = times[np.clip(k - 1, 0, len(times) - 1)], times[np.clip(k, 0, len(times) - 1)]
        return np.minimum(np.abs(a - t), np.abs(b - t)) <= 0.05
    total = np.zeros(len(grid))
    for o, p in zip(offs, pitches):
        t = grid + o
        same = by_pitch.get(p)
        hit = near(same, t) if same is not None else np.zeros(len(t), bool)
        octave, inside = np.zeros(len(t), bool), np.zeros(len(t), bool)
        if p % 12 in by_pc:
            starts, ends = by_pc[p % 12]
            octave = near(starts, t)
            j = np.searchsorted(starts, t + 0.05, side="right") - 1
            inside = (j >= 0) & (t <= ends[np.clip(j, 0, len(ends) - 1)])
        total += np.where(hit, 1.0, np.where(octave, 0.5, 0.0)) + 0.5 * inside
    return total


def _sync_timed(tab_t, tab_p, bar_starts, bar_ends, bar_of, heard):
    """Tab with rhythm: the overall tempo (`_tempo_fit`), then every bar placed on the
    recording in one pass (Viterbi over start times on a 50 ms grid): a bar starts where
    the one before ended (give or take 100 ms, SYNC_DRIFT a step), or after a jump forward
    (SYNC_JUMP: the recording has something the tab doesn't), or is left out (SYNC_SKIP a
    bar: the tab has something the recording doesn't, e.g. a part repeated more often).
    Scored by `_bar_scores`. Returns per bar its start and end on the recording (None when
    left out), the overall match and the tempo ratio.

    It was one overall start + tempo, then up to +-2.5 s per bar along a smooth path, which
    couldn't follow tabs and recordings of different shapes (user: "Baby's tab starts way
    too early and Look What Happened starts way too late"): Baby's tab is ~22 s longer
    than the recording's bass part, so its opening sat in the bassless intro (now 10 of 89
    bars left out, first note at the bass's entry, notes on a heard note 57 -> 76%); Look
    What Happened's recording drifts in tempo at the start and its tab's ghost-note E3s are
    never heard, so the old fit put the first note at 0:29 (now 0:00.6). The 11 other
    imported tabs came out the same within a point or two."""
    heard = sorted(heard)
    heard_t = np.array([h[0] for h in heard])
    heard_p = np.array([h[2] for h in heard])
    tab_t, tab_p = np.asarray(tab_t), np.asarray(tab_p)
    best, scale, _ = _tempo_fit(tab_t, tab_p, heard_t, heard_p)
    by_pitch = {}
    for t, p in zip(heard_t, heard_p):
        by_pitch.setdefault(int(p), []).append(t)
    by_pitch = {p: np.array(v) for p, v in by_pitch.items()}
    by_pc = {}
    for h in heard:
        by_pc.setdefault(int(h[2]) % 12, []).append((h[0], h[1]))
    by_pc = {c: (np.array([a for a, _ in v]), np.maximum.accumulate(np.array([b for _, b in v]))) for c, v in by_pc.items()}
    grid = np.arange(min(-5.0, heard_t[0] - 5), heard[-1][1] + 5, SYNC_GRID)
    N, B = len(grid), len(bar_starts)
    length = [int(round((bar_ends[b] - bar_starts[b]) * scale / SYNC_GRID)) for b in range(B)]
    score = np.zeros((B, N))
    for b in range(B):
        mine = bar_of == b
        if mine.any():
            score[b] = _bar_scores((tab_t[mine] - bar_starts[b]) * scale, tab_p[mine], grid, by_pitch, by_pc)
    NEG = -1e18
    placed, waiting = np.full((B, N), NEG), np.full((B, N), NEG)   # bar b starts at j / bar b left out, the next due at j
    p_from, p_kind = np.zeros((B, N), np.int32), np.zeros((B, N), np.int8)   # kind: 0 follows, 1 jumps (from placed), 2/3 (from waiting), 4 first
    w_from, w_kind = np.zeros((B, N), np.int32), np.zeros((B, N), np.int8)   # kind: 0 after a placed bar, 1 still waiting, 4 first
    placed[0], waiting[0] = score[0], -SYNC_SKIP
    p_kind[0] = w_kind[0] = 4
    cols = np.arange(N)

    def follow(prev, shift):   # prev[j - shift + d] - SYNC_DRIFT * |d|, d in -2..2
        best, src = np.full(N, NEG), np.zeros(N, np.int32)
        for d in range(-2, 3):
            idx = cols - shift + d
            ok = (idx >= 0) & (idx < N)
            cand = np.full(N, NEG)
            cand[ok] = prev[idx[ok]] - SYNC_DRIFT * abs(d)
            better = cand > best
            best[better], src[better] = cand[better], idx[better]
        return best, src

    def jump(prev, shift):     # the best prev[i] with i + shift < j - 2, minus SYNC_JUMP
        top = np.maximum.accumulate(prev)
        arg = np.zeros(N, np.int32)
        cur = 0
        for i in range(N):
            if prev[i] >= prev[cur]:
                cur = i
            arg[i] = cur
        best, src = np.full(N, NEG), np.zeros(N, np.int32)
        idx = cols - shift - 3
        ok = idx >= 0
        best[ok], src[ok] = top[idx[ok]] - SYNC_JUMP, arg[idx[ok]]
        return best, src

    for b in range(B - 1):
        k = length[b]
        options = [follow(placed[b], k), jump(placed[b], k), follow(waiting[b], 0), jump(waiting[b], 0)]
        vals, srcs = np.stack([o[0] for o in options]), np.stack([o[1] for o in options])
        which = np.argmax(vals, axis=0)
        placed[b + 1] = vals[which, cols] + score[b + 1]
        p_kind[b + 1], p_from[b + 1] = which, srcs[which, cols]
        after = np.full(N, NEG)
        after[k:] = placed[b][:N - k] if k < N else after[k:]
        use = after >= waiting[b]
        waiting[b + 1] = np.where(use, after, waiting[b]) - SYNC_SKIP
        w_kind[b + 1], w_from[b + 1] = np.where(use, 0, 1), np.where(use, cols - k, cols)
    starts = [None] * B
    on = placed[B - 1].max() >= waiting[B - 1].max()
    j = int(np.argmax(placed[B - 1] if on else waiting[B - 1]))
    for b in range(B - 1, -1, -1):
        kind = p_kind[b, j] if on else w_kind[b, j]
        if on:
            starts[b] = float(grid[j])
        if kind == 4:
            break
        j, on = (int(p_from[b, j]), kind in (0, 1)) if on else (int(w_from[b, j]), kind == 0)
    ends = [None if a is None else a + (bar_ends[b] - bar_starts[b]) * scale for b, a in enumerate(starts)]
    return starts, ends, best / max(1, len(tab_t)), scale


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


SPACING_TRUST = 0.8   # a spaced text tab keeps its spacing unless order-only placing finds 25% more notes


def sync(track, heard, beat=None):
    """The tab's notes and bars on the recording's clock (beat: the song's beat length in
    seconds, from the beat tracker; text tabs need it, they have no tempo of their own).

    Returns {"notes": [[start, end, string, fret, dead]], "bars": [start s],
    "quality": share of tab notes that matched a heard note}."""
    if track.spaced:
        # Rhythm read from a pasted tab's spacing: usually close (Ultimate Guitar lays tabs
        # out in proportion to time), but a hand-typed tab can be spaced anyhow, so if
        # placing notes in order finds far more of them, that wins.
        by_spacing = sync(replace(track, spaced=False), heard, beat)
        by_order = sync(replace(track, spaced=False, timed=False), heard, beat)
        return by_spacing if by_spacing["found"] >= SPACING_TRUST * by_order["found"] else by_order
    if track.timed and not track.tempo_known:
        # A pasted tab with counts has rhythm but no tempo: try the song's (the beat
        # tracker's, which can be off by 2x either way) and keep the best fit; the tempo
        # search in _tempo_fit then fine-tunes it (0.6-1.7x).
        bpm = 60 / beat if beat else 120.0
        guesses = [b for b in (bpm, bpm / 2, bpm * 2) if 50 <= b <= 260] or [120.0]
        tries = [sync(replace(track, tempo=b, tempo_known=True), heard, beat) for b in guesses]
        # Judged on all the tab's notes: "quality" leaves out bars the sync left out, and
        # at half or double speed it left out half the tab and still looked best.
        return max(tries, key=lambda r: r["found"])
    notes, bar_starts = timeline(track)
    playable = [k for k, n in enumerate(notes) if not n[4]]
    tab_t = [notes[k][0] for k in playable]
    tab_p = [track.pitch(notes[k][2], notes[k][3]) for k in playable]
    if not heard or not playable:
        return {"notes": [list(n[:5]) for n in notes], "bars": bar_starts, "quality": 0.0, "found": 0.0}
    if track.timed:
        bar_of = np.array([notes[k][5] for k in playable])
        ends = list(bar_starts[1:]) + [bar_starts[-1] + (bar_starts[-1] - bar_starts[-2] if len(bar_starts) > 1 else 2.0)]
        bar_audio, bar_audio_end, quality, scale = _sync_timed(tab_t, tab_p, bar_starts, ends, bar_of, heard)

        # Bars from the overall fit are a little rough on a noisy separated track (77% of
        # notes within 50 ms end to end on a test song); placing notes inside each bar like
        # text tabs do (matched in order, guided by the exact rhythm) fixes that.
        out, matched = _place_in_bars(track, notes, playable, heard, list(bar_audio), bar_starts, ends,
                                      lambda b: (ends[b] - bar_starts[b]) * scale, keep_lengths=True, bar_audio_end=bar_audio_end)
        # Bars left out (not in this recording) drop their notes; the tab file keeps them.
        return {"notes": [[round(n[0], 3), round(n[1], 3), *n[2:5], notes[k][6] or ""] for k, n in enumerate(out)
                          if bar_audio[notes[k][5]] is not None],
                "bars": [round(float(x), 3) for x in bar_audio if x is not None],
                "quality": round(matched / max(1, sum(1 for k in playable if bar_audio[notes[k][5]] is not None)), 2),
                "found": matched / len(playable)}
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
        seq = [track.pitch(notes[k][2], notes[k][3]) for k in ks]
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
            "bars": [round(float(x), 3) for x in bar_audio], "quality": round(matched / max(1, len(playable)), 2),
            "found": matched / max(1, len(playable))}


STEADY_MAX = 0.08   # how far (s) a bar may be shifted to fit what's heard (see _place_in_bars)


def _bar_fit(pairs):
    """A bar's shift to fit its heard notes, as (shift at mid, change per second, mid):
    medians (one wrong match can't pull it), at most 80 ms and 8% faster/slower."""
    if not pairs:
        return None
    xs, ds = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    if len(pairs) == 1:
        return float(np.clip(ds[0], -0.05, 0.05)), 0.0, float(xs[0])
    slopes = [(ds[j] - ds[i]) / (xs[j] - xs[i]) for i in range(len(xs)) for j in range(i + 1, len(xs)) if xs[j] - xs[i] > 0.05]
    return (float(np.clip(np.median(ds), -STEADY_MAX, STEADY_MAX)),
            float(np.clip(np.median(slopes), -0.08, 0.08)) if slopes else 0.0, float(np.median(xs)))


def _place_in_bars(track, notes, playable, heard, bar_audio, bar_starts_t, bar_ends_t, audio_len, keep_lengths, bar_audio_end=None):
    """Each bar's notes on the recording: matched in order to the heard notes in that bar
    (guided by where the tab puts them).

    Files with rhythm (keep_lengths): the tab's rhythm as written, the notes keeping their
    lengths. Each bar is shifted and stretched as a whole to fit its matched notes, and
    neighbouring bars meet at the bar line (one shift there, the mean of what both want),
    so steady 8ths and 16ths stay even. Heard notes used to nudge each note (up to 100 ms
    or 35% of the gap): a quarter of steady note pairs came out > 25% uneven, up to 47%
    across bar lines on Sha-La-La-Lady (user: "straight eighth/16th notes ... spaced weird so
    it makes it hard to hit them"). On the user's 5 file tabs now 0%; notes within 50 ms of
    a heard one: written rhythm alone 55.8-86.4%, this 56.5-88.9%, the nudging 62.9-94.2%
    (much of that the nudge itself moving notes onto noisy heard times).

    Text tabs: unmatched notes placed in order between their matched neighbours, each lasting
    until the next (hand-typed spacing is too sloppy to trust: 72% vs 82% on a test)."""
    out = [[float(n[0]), float(n[1]), n[2], n[3], n[4]] for n in notes]
    matched, bars = 0, []
    for b in range(len(bar_audio)):
        a0 = bar_audio[b]
        if a0 is None:   # left out (bar_audio_end given): not in this recording
            bars.append(None)
            continue
        a1 = (bar_audio_end[b] if bar_audio_end is not None
              else bar_audio[b + 1] if b + 1 < len(bar_audio) else a0 + audio_len(b))
        span = max(1e-6, bar_ends_t[b] - bar_starts_t[b])
        rate = (a1 - a0) / span
        ks = [k for k in range(len(notes)) if notes[k][5] == b]
        expected = [a0 + (notes[k][0] - bar_starts_t[b]) * rate for k in ks]
        play = [i for i, k in enumerate(ks) if k in playable]
        inside = [h for h in heard if a0 - 0.08 <= h[0] < a1 - 0.08]
        match = (_sync_order([track.pitch(notes[ks[i]][2], notes[ks[i]][3]) for i in play], inside,
                             [expected[i] for i in play], (a1 - a0) / 4) if inside and play else [-1] * len(play))
        matched += sum(1 for j in match if j >= 0)   # heard (the match quality counts it either way)
        if keep_lengths:
            pairs = [(expected[i], inside[j][0] - expected[i]) for i, j in zip(play, match)
                     if j >= 0 and abs(inside[j][0] - expected[i]) <= 0.1]   # further off: likely a wrong match
            bars.append((a0, a1, rate, ks, expected, _bar_fit(pairs)))
            continue
        if not ks:
            continue
        # Notes written in one column (a chord) share a place, so they stay together.
        columns = {x: r + 1 for r, x in enumerate(sorted({notes[k][0] for k in ks}))}
        place = lambda i: columns[notes[ks[i]][0]]
        anchor_x, anchor_t = [0], [a0]
        for i, j in zip(play, match):
            if j >= 0:
                anchor_x.append(place(i))
                anchor_t.append(inside[j][0])
        anchor_x.append(len(columns) + 1)
        anchor_t.append(a1)
        order = np.argsort(anchor_x, kind="stable")
        ax, at = np.array(anchor_x)[order], np.maximum.accumulate(np.array(anchor_t)[order])
        for i, k in enumerate(ks):
            out[k][0] = float(np.interp(place(i), ax, at))
    if keep_lengths and any(bars):
        def shift(b, x):
            fit = bars[b][5] if 0 <= b < len(bars) and bars[b] else None
            return None if fit is None else float(np.clip(fit[0] + fit[1] * (x - fit[2]), -STEADY_MAX, STEADY_MAX))
        # A bar's shift at its start and at its end; where two placed bars meet (one starts
        # where the other ends) both use the mean of what they want, so the line is unbroken.
        meets = lambda b: b + 1 < len(bars) and bars[b] and bars[b + 1] and abs(bars[b][1] - bars[b + 1][0]) < 0.15
        start, end = [None] * len(bars), [None] * len(bars)
        for b, bar in enumerate(bars):
            if not bar:
                continue
            want = [shift(b, bar[0])] + ([shift(b - 1, bar[0])] if b and meets(b - 1) else [])
            want = [v for v in want if v is not None]
            start[b] = float(np.mean(want)) if want else None
            want = [shift(b, bar[1])] + ([shift(b + 1, bar[1])] if meets(b) else [])
            want = [v for v in want if v is not None]
            end[b] = float(np.mean(want)) if want else None
        seq = [v for b in range(len(bars)) for v in (start[b], end[b])]
        known = [i for i, v in enumerate(seq) if v is not None]
        seq = [v if v is not None else (float(np.interp(i, known, [seq[k] for k in known])) if known else 0.0) for i, v in enumerate(seq)]
        for b, bar in enumerate(bars):
            if not bar:
                continue
            a0, a1, rate, ks, expected, _ = bar
            for i, k in enumerate(ks):
                t = expected[i] + float(np.interp(expected[i], [a0, a1], [seq[2 * b], seq[2 * b + 1]]))
                out[k][0], out[k][1] = t, t + (notes[k][1] - notes[k][0]) * rate
    if not keep_lengths:
        for k in range(len(out)):   # each note lasts until the next one starts
            nxt = next((out[j][0] for j in range(k + 1, len(out)) if out[j][0] > out[k][0]), out[k][0] + 0.3)
            out[k][1] = min(nxt, out[k][0] + 1.0)
    return out, matched
