# Track Splitter

A small local app that takes instruments out of songs, so a musician can play along
(the original use case: remove the bass to practice bass). Upload songs or paste a
link, pick what to remove (bass, drums, vocals, guitar, piano), and it produces a
version without those parts plus the removed part on its own. Everything runs on the
user's own computer; no audio leaves the machine.

Primary user runs it on a MacBook Pro (M4); it was developed and tested on Windows
(CPU only). It has also been run successfully on a second, fresh Mac from git.

## Files

- `app.py`: everything server-side: the separation engine, the library on disk, the
  processing queue, and a FastAPI web API. Also serves the page and creates the app
  shortcut. Run with `python app.py` (port 7860, opens the browser).
- `static/index.html`: the whole UI in one file (HTML + CSS + vanilla JS, no build
  step, no external downloads so it works offline).
- `start.sh` (Mac/Linux) / `start.bat` (Windows): create `.venv` on first run, reinstall
  packages whenever `requirements.txt` is newer than `.venv/.installed`, then start.
- `roformer/`: BS-RoFormer model code copied from the `BS-RoFormer` package v0.4.1
  (lucidrains, MIT, see `roformer/LICENSE`), only the import made relative. That
  version matches the BS-Roformer-SW checkpoint exactly (newer ones changed the
  architecture), and copying it avoids the package's `librosa` dependency.
- `requirements.txt`: torch, torchaudio, demucs, fastapi, uvicorn, python-multipart,
  imageio-ffmpeg, numpy, send2trash, yt-dlp, truststore, einops, beartype,
  rotary-embedding-torch (the last three for `roformer/`), onnxruntime (bass tab),
  PyGuitarPro (LGPL-3.0, reading Guitar Pro 3-5 files).
- `tab.py`: bass tab: transcription (Basic Pitch), beats/bars, fingering. Module
  docstring has the method and the measurements behind each choice.
- `tabimport.py`: imported tabs (Guitar Pro / MusicXML / pasted text): reading, repeats,
  syncing to the recording. See "Imported tabs".
- `basicpitch/`: Spotify's Basic Pitch model (`nmp.onnx`, 230 KB) + its LICENSE/NOTICE
  (Apache 2.0). The pip package needs TensorFlow (no Python 3.13 build on Mac), so
  `tab.py` runs the ONNX model with onnxruntime and reimplements its note decoding in
  numpy (no librosa/scipy).
- `track-splitter.zip`: an email-sized copy of the app (git-ignored). Email filters
  block `.bat`, so inside the zip it is `start.bat.txt`; never commit that name.

## How separation works

- Demucs v4 (Meta, open source). Standard quality = `htdemucs`; High quality =
  `htdemucs_ft`. Guitar and piano: Standard = `htdemucs_6s` (fast, less precise);
  High quality = BS-Roformer-SW by jarredou (`run_roformer()`), a 6-stem RoFormer
  that separates guitar much better but runs at about real time on an M4 (~2 min for
  a 2 min song). Its ~700 MB checkpoint downloads once into torch's hub cache
  (sha256-checked) from the python-audio-separator GitHub release. Inference follows
  the model's own config (588800-sample chunks, 2 overlapping passes, sin² crossfade).
  The copied model code gives bit-identical output to audio-separator's on the same
  chunk; full songs differ slightly (~-18 dB) only because audio-separator uses its own
  chunking (409600 samples, 8 passes, Hamming). The user asked to keep both models
  available, hence the HQ switch.
- RoFormer speed (benchmarked on 60 s, M4, best of 2): fp16 autocast on GPU = 1.1x,
  output within -61 dB, so always on (not on CPU). "Faster" option (`fast`): chunks
  overlap by 1 s instead of half, so each moment is separated once: 2.0x (2.3x with
  fp16), output ~-20 dB different, possibly slightly rougher. Batching 2 chunks: no
  gain. Fast and normal share the "[HQ]" name; `.song.json` records which, for Redo.
- Lead / rhythm guitar (`center_of()`): no model separates them locally (MVSep and
  Moises do, but in the cloud). The guitar stem is split by stereo position: the
  power both channels share (Re of L·R*, averaged over 3 STFT frames, so separate
  takes cancel out) is the center's; each slice keeps that fraction (^1.5) as center
  (lead), the rest is sides (rhythm). The first version compared loudness only and
  counted equally loud doubled rhythm takes as lead; the user heard the bleed. On a
  test mix with known parts (center vocal + two different guitar takes panned L/R)
  bleed went -7.3 -> -10.0 dB; ~-10 dB is about the ceiling for this trick (window
  sizes and smoothing were tuned). "Getting Along": 15% of the guitar counted as lead
  (34% before). Poor on live/mono mixes. Picking both parts = All guitar.
- `htdemucs_ft` is a bag of 4 per-stem specialists. `stem_model()` runs only the
  specialist(s) needed: identical output to the full bag, ~4x faster. (Outputs differ
  slightly run to run only because of Demucs' random time shift, `shifts=1`.)
- The "without" track is `original - removed_parts * (1 - level)`, so everything not
  removed is untouched. "Only" is the removed parts. Both together equal the original
  (up to MP3 loss), which Original mode falls back to when the real original can't be used.
- Device: CUDA if present, else Apple MPS, else CPU. MPS lacks some ops:
  `PYTORCH_ENABLE_MPS_FALLBACK=1` is set before torch loads, and `run_model()` retries
  a song on the CPU if MPS still raises.
- Audio decode/encode goes through the ffmpeg binary from `imageio-ffmpeg` (no system
  ffmpeg needed). Output is 320 kbps MP3 at 44.1 kHz (was 192 until Oct 2026; most
  originals are YouTube Opus ~125 kbps, so versions are a 2nd lossy generation and the
  user heard a "wavy" sound; older versions improve when redone).
- Progress comes from Demucs' `apply_model(callback=...)` (segment offsets).

## Library on disk

Lives in `~/Music/Track Splitter/`, outside the project, so app updates never touch it.

```
<Artist>/<Song>/
  original.<ext>               the uploaded/downloaded file (lets new versions be made without re-uploading)
  <Song> (no bass).mp3         one pair of files per version
  <Song> (bass only).mp3
  <Song> [HQ] (no drums & vocals).mp3 / ... (drums & vocals only).mp3
  .bass-tab.json               bass tab: notes, lines (bars), bpm, suggested tuning, fingering per tuning
  (library root) .learned-from-tabs.json   note settings + fingering costs learned from imported tabs
  .song.json                   {"sha1", "source", "added", "versions"}: sha1 of the original spots
                               repeat uploads; versions = {" [HQ] (no bass)": {"level": 30, "fast": false}}, keyed by
                               name minus the song title (survives renames), for one-click Redo
```

- Ids: artist = `Artist`, song = `Artist/Song`, version = `Artist/Song/<Song> (no X)`.
  `library_path(id, depth)` validates every id (no `..`, must stay inside the library).
- Songs without an artist go to `Unknown artist`.
- The filesystem is the database (no separate index); `library()` scans it.
- Startup migrations (`migrate_old_library`): moves the old `~/Music/Bass Remover` and
  `./output` libraries in, then turns the earlier flat layout (`Artist/Song (no bass).mp3`)
  into song folders. Those older songs have no original; the UI asks for the file once
  ("+ Version" -> attach original -> make version).
- Deletes go to the Trash (send2trash), never permanent.
- Renames handle case-only changes (Mac/Windows filesystems ignore case) via a temp name.

## Adding songs (two steps)

1. Stage: `POST /api/stage` (files) or `POST /api/stage-link` (URL via yt-dlp). Reads
   title/artist from tags, or guesses from the file/video name ("03 - Artist - Song
   (Official Video)" -> Artist / Song), and flags duplicates by sha1.
2. Process: `POST /api/process` with staged ids (or existing song ids for a new
   version), instruments, level, hq. One background worker thread processes jobs in
   order; the page polls `GET /api/jobs/{id}`.
   - Progress: the engine calls `on_progress(fraction, step)`; `step` is the text the
     page shows under the bar ("Reading the song", "Downloading the high-quality model
     · 210 of 667 MB", "Loading…", "Separating guitar (high quality)", "Splitting lead
     and rhythm guitar", "Saving", or the Apple GPU → CPU fallback notice).
     The page appends "about 1 min 30 s left" (`timeLeft()`), from the bar's rate since it
     started moving (so downloads/loading don't skew it); hidden until 4 s and 3% in.
   - Stop: `POST /api/jobs/{id}/stop` sets a flag that the job's `report()` checks on
     every progress call (every few seconds) and raises `Stopped` (not a RuntimeError,
     so `run_model` doesn't retry it). Nothing is written before separation finishes,
     so a stopped song leaves nothing behind; unprocessed uploads stay staged, and
     `job.finished` (item indexes) tells the page which dialog items to clear.

yt-dlp is used the standard way, with no cookies or bot-check workarounds. This is
deliberate: YouTube's "confirm you're not a bot" check blocked it on the user's work
network, and getting around bot detection was declined. The app just explains that
error (and certificate errors) in plain language.

`truststore.inject_into_ssl()` makes Python trust the OS certificate store (Mac
Keychain), which fixed a certificate error on the Mac; certificate checking stays on.

## Bass tab

The user plays bass and asked for tab of songs, shown like a lyrics view. Per song:
`POST /api/tab` queues a "tab" job (same worker, progress, Stop); `make_tab()` reuses a
"(bass only)" track (HQ preferred) or isolates the bass with htdemucs_ft, transcribes,
finds bars, and saves `.bass-tab.json`. `GET /api/tab?song&tuning` returns lines and
notes as [start, end, string, fret]; other tunings are fingered on first request.

- Notes: Basic Pitch, then one note at a time. Leakage (other instruments in the
  separated bass) made fake notes, which the user noticed. Fixes, each measured: its
  "melodia" notes (no clear onset) are kept only when they fill a gap between onset
  notes (they're re-plucked fast 8ths there; on top of another note they were leakage);
  notes are dropped where the bass track is > 25 dB below the whole song (no-bass
  intros measured 32-67 dB below, real bass 0-16) or > 30 dB below its own loud parts.
  Known bass line + real guitar/drum leakage: ~95% of shown notes real, ~96% found.
  Octaves (`_fix_octaves`): low notes (low E-G) came out 1-2 octaves high (most pitch
  errors on a known driving-8ths line); a note at f has nothing at f/2 and 3f/2, one at
  f/2 has overtones there, so if the bass track has both it moves down (that line:
  found 76% -> 86%, real 80% -> 91%; real songs: 0-11 notes moved).
  Re-plucks (`split_on_grid`, in make_tab after the grid is known): repeated plucks of
  one pitch came out as long notes (user: You Only Live Once "should be 8ths"; only 27%
  of 8th slots had a note). Long notes are split only at grid points, where the bass
  track has a sharp attack (log spectral flux >= 1.35x the note's median) or Basic
  Pitch's onset activation for that pitch >= 0.2. Splitting anywhere on that signal
  alone doubled notes on the known line (48% real). 8th-slot coverage now: You Only
  Live Once 27 -> 58%, Getting Along 36 -> 56%, American Idiot 20 -> 37%.
  `NOTES_RULES` in app.py: bump when notes would come out differently; older tabs then
  show "can be made better" and a Make it again button.
- MuScriptor (Mirelo x Kyutai, open weights) was considered as a better transcriber:
  weights CC BY-NC 4.0 and gated on Hugging Face (user login), Python <= 3.12. Set up
  in a scratch env but not run (the user chose to improve Basic Pitch instead).
  YourMT3+ ruled out: its own RWC-pop bass result is ~31% onset F1.
- Bars: tempo by onset autocorrelation, Ellis DP beat tracking at 1/2x/2x that tempo,
  8th or triplet grid; the grid the bass notes fit best (above chance) wins, shifted
  to the notes; downbeat from where bass notes land. If < 0.4 fit score: 4 s lines.
  Found 186/186/121 bpm on American Idiot, Getting Along, You Only Live Once;
  American Confidence has no steady grid.
- Fingering: hand-position Viterbi (4-fret span + open strings, shifts cost), then
  each pitch pulled to its usual place (so riffs repeat identically; the user's
  complaint about AI tabs), tried around frets 1/5/9 and unbiased; lowest playing
  difficulty (shifts, quick jumps, string skips) wins. Tunings: standard, drop D,
  half step down, 5-string; suggested from the 2nd-percentile lowest note, and half
  step down when it turns many more notes into open strings (American Idiot: 2% -> 47%;
  the user found its standard-tuning tab "off" vs tab videos). Bump `TUNING_RULES` in
  app.py when the suggestion changes; saved tabs then get re-suggested on open.
- Learning from imported tabs (the user wants tab making to get smarter over time):
  every imported tab is a right answer for its song, so after each import job (and in
  a background "learn" job at startup or after Remove import, when the set of imported
  tabs changed) `learn_from_imports()` fits two things to all of them and saves them in
  `<library>/.learned-from-tabs.json` with held-out before/after scores. Each is used
  only if it beats the built-in settings on the half of each tab it wasn't fitted on.
  - Fingering: the hand-position costs (`tab.STYLE`) via a structured perceptron
    (`tab.learn_style`: hand positions that explain the tab's own fingering vs the ones
    the current costs pick; 100 rounds, ~4 s per 500 notes). Cry Baby Cry was 78% same
    string/fret (its riff A7 D5 D7 came out D2 G2 D5); fitted on one half and checked
    on the other: 84 -> 97% and 73 -> 100%. With learned costs the "easiest of 4 tries"
    step is skipped. Saved fingerings carry `fingering_style` and are redone when it
    changes. With one tab it may overfit that song (other songs moved to more E/A
    string playing, frets 2-7); more tabs balance it.
  - Notes: the detection settings (`tab.NOTES`: Basic Pitch thresholds, min length,
    melodia, octave fix, and `grid_split` = split repeated notes on by default with its
    strength) by trying each value in `tab.NOTE_CHOICES` (2 rounds, F1 by pitch within
    70 ms; reruns the model on each imported song with a "bass only" track, ~1 s each).
    Must beat the built-in settings by 0.5 points held out. On Cry Baby Cry alone the
    hand-set values were already near best (F1 0.866; held-out 0.859, so not used);
    most misses were re-plucks inside one long note. Automatic tabs store
    `note_settings`; when the learned ones differ, the tab shows "can be made better".
    `split_default` sets the split switch until the user flips it for that song.
  - The tab view's subtitle says what's in use ("Fingering learned from your imported
    tab (matches it 99%, was 79%)", "Notes tuned to your imported tabs (...)").
- Page: Tab button in the player opens the view over the song list: previous line
  dimmed, current large with playhead and the sounding note highlighted, next two
  below; slides up on each new line; click a line to seek; tuning picker (choice kept
  per song in localStorage); lines shrink to fit (5 strings, small windows). The
  playhead subtracts the audio output delay (`ctx.outputLatency + baseLatency`, ~87 ms
  on this Mac, more on Bluetooth) so it follows what you hear; it ran ahead before.
  Note times themselves were within ~16 ms of the bass attacks (American Idiot).
- Raise ½ step (user plays a standard-tuned bass): with the tab in "Half step down",
  a switch plays the song a semitone higher at the same tempo and labels the strings
  E A D G, so the easy half-step-down shapes fit a standard bass. Per song in
  localStorage (`track-splitter-raise`); switching tuning away turns it off. Server:
  `/api/audio?...&shift=1` -> `shifted()`: ffmpeg aresample/asetrate/atempo (no
  rubberband in imageio-ffmpeg; measured +98 cents, same length), cached as float WAV
  in a temp dir cleared at startup. atempo works per file, so shifted parts don't add up
  to the shifted original sample-for-sample (correlation 0.07, offset < 1 ms):
  `matchOriginal(..., shifted)` then trusts the original by length and matches loudness.
- Line timing (user: the line lagged): Earlier/Later in 20 ms steps (+-600 ms), on top
  of the output-delay correction, kept per computer (`track-splitter-line-sync`).
- Crowded notes: each fret number shrinks by its own room to its neighbours (down to
  55%), so fast passages don't overlap while the rest stay full size.

## Imported tabs (tabimport.py)

The user found automatic tabs not trustworthy and plays from tabs online, so the tab
view can import one: a file (Guitar Pro .gp3/.gp4/.gp5 via PyGuitarPro, .gp 7/8 via its
score.gpif XML, MusicXML .musicxml/.xml/.mxl) or a text tab pasted from a tab site. The
app does not fetch from Songsterr/Ultimate Guitar links (their terms forbid automated
downloading; Songsterr's GP download is a paid feature); the user brings the file/text.
- Every format -> TabTrack (tuning + bars as written: time signature, tempo changes,
  repeat signs, notes with string/fret/tie/dead); `timeline()` writes out repeats
  (`play_order`) and times notes from the tab's tempo. Round-trip tests (GP5, MusicXML,
  .mxl, text) read back every note with the right pitch and time, repeats included.
  The .gp (GP7/8) reader follows the documented layout but is untested on a real file;
  .gpx (GP6) is refused with a "save as .gp/.gp5 or MusicXML" message.
- Sync to the recording uses the automatic tab's heard notes (made first if missing):
  files with rhythm: overall start + tempo ratio search, then per-bar shifts; text tabs:
  bar length from the song's tempo, bars placed by note content and then followed in
  order (+-0.3 s, ordered matching) so look-alike riff repeats can't make it jump. Both
  then place notes inside each bar by matching them in order to heard notes
  (`_place_in_bars`): files use their exact rhythm between matches, text tabs plain
  order. Test song (line played 3.2 s late, 3% slower, +-0.15 s drift, real band
  leakage, through the full app): MusicXML 93% of notes within 50 ms, pasted text 97%.
- Stored next to the song: the file as given (`imported-tab - <name>.<ext>`, text as
  `imported-tab.txt`; a replaced import goes to the Trash after the new one reads OK) and
  `.bass-tab-imported.json` (synced notes, lines, tuning, match quality).
  `POST /api/tab/import` (multipart: song, file or text) queues an "import" job;
  `GET /api/tab?source=imported`; `POST /api/tab/import/remove` (to the Trash).
- Page: "Import tab…" (file or paste box; also "Import bass tab…" in each song's ⋯ menu,
  which loads the song and opens its tab view behind the box: the user couldn't find
  the button inside the tab view), an Imported/Automatic switch (per song,
  imported by default), the tab's own tuning (raise ½ step still works if it's half
  step down), and a warning when < 35% of its notes matched (other version/tuning?).

## UI (static/index.html)

- Sidebar: All songs, artists (hover ... -> rename/merge, delete; + New artist; drop a
  song row on an artist to move it), device label, Open library folder, Create app
  shortcut, Quit.
- Main: searchable song list; each row shows version chips (click to play) and
  "+ Version"; row ... menu: New version, Edit song & versions (rename, change artist,
  delete versions), Delete song. Hovering a chip widens it and shows ↻ Redo (remakes
  it right away with its own instruments/HQ from the name and level from .song.json,
  via `POST /api/version/redo`; versions made before levels were recorded redo at 0%)
  and 🗑 (to the Trash, no confirmation). Right-click a chip: Play, Redo, Redo with
  different settings… (the dialog prefilled from the version), Delete.
- Player bar: Web Audio, not `<audio>` elements. Both files (without + only) are
  fetched and decoded into memory, then started on the same `AudioContext` clock so
  they stay sample-exact; the Without / Original / Only switch fades gain nodes.
  (Two `<audio>` elements drifted apart, which made Original sound hollow/muted.)
  Original mode plays the song's real original (`/api/audio?part=original`: decoded by
  ffmpeg to 44.1 kHz float WAV, so peaks above full scale survive), started on the same
  clock. `matchOriginal()` uses it only if it correlates > 0.97 with without + only
  (checked: lag 0, r 0.999 in Chromium) and scales it to the versions' anti-clip
  turn-down; otherwise (no original, other browser offsets) Original = without + only.
  `headroom()` turns each song down just enough that no mode clips.
- Add songs dialog: drop files or paste a link, edit title/artist per item, pick
  instruments (guitar is All guitar / Lead guitar / Rhythm guitar), level, HQ, and
  "Faster guitar/piano" (shown only with HQ + guitar/piano). Upload is disabled until the required fields are filled. On
  success the songs clear and the dialog closes; settings persist (localStorage).
- Dark theme, violet accent, system font; fits one screen down to ~1024x768 and
  collapses for narrow windows.

## Launching without a terminal

- The app asks before creating a shortcut (welcome card, "Don't ask me again" stored in
  localStorage). Mac: `~/Applications/Track Splitter.app` whose launcher runs `start.sh`
  in the background (log: `~/Library/Logs/Track Splitter.log`) or reopens the page if
  already running. Windows: Desktop `.lnk` running `pythonw.exe app.py` (does not
  reinstall packages; untested on a real desktop).
- Starting a second copy just opens the browser. Quit in the sidebar stops the server
  (`POST /api/quit`).
- The static files are mounted last in `app.py`; any new `/api/...` route must be
  defined before `api.mount("/", ...)` or it will 404.

## Testing approach used so far

No test suite in the repo. Changes were verified by running the server on a spare port
(`app.main(port=7866, open_browser=False)`), driving the API with small scripts
(upload, poll, move, delete, bad ids), and clicking through the page in a browser. A
synthetic 15 s clip (sine bass line + chords + clicks) was used, measuring how much
40-100 Hz energy remains to confirm bass removal.

## Open ideas / to do

- Windows polish: accept any recent Python (start.bat hard-codes `py -3.13`), make the
  desktop shortcut install package updates like the Mac one, use an NVIDIA GPU (needs
  the CUDA build of torch; the default Windows build is CPU-only).
- Playlists / practice sets that pull songs from any artist.
- The user once asked for "filters": interpreted as folders; filtering the song list
  (by instrument removed, HQ, etc.) may still be wanted.
- Mac-only paths (app shortcut, MPS fallback, certificate fix) were written on Windows;
  they have been reported working on Mac but have no automated checks.
