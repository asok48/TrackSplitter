# Track Splitter

A small local app that takes instruments out of songs, so a musician can play along
(the original use case: remove the bass to practice bass). Upload songs or paste a
link, pick what to remove (bass, drums, vocals, guitar, piano), and it produces a
version without those parts plus the removed part on its own. Everything runs on the
user's own computer; no audio leaves the machine.

Primary user runs it on a MacBook Pro (M4); it was developed and tested on Windows
(CPU only). It has also been run successfully on a second, fresh Mac from git.

## Working on this project

The user plays bass (a Focusrite Scarlett interface on the Mac) and uses the app to
practise; requests come from playing with it, so features are judged by how they feel
in use. Ground rules agreed so far (kept here so every computer's session has them):
- Don't commit or push unless the user asks; they often commit themselves.
- yt-dlp stays standard: no browser cookies or ways around YouTube's "confirm you're not
  a bot" check (considered and declined); explain the error or suggest uploading the
  file. Certificate checking stays on (truststore only).
- Never fetch tabs from Songsterr / Ultimate Guitar (their terms forbid it) or wire in
  third-party downloaders for them; the user brings the file or text.
- Deletes go to the Trash (send2trash), never permanently.
- Test on a copy of the library (a spare port and `OUTPUT_DIR` pointed at a scratch
  copy), never the user's real `~/Music/Track Splitter`; reading it for analysis is fine.
- After every change, load the page and check for errors and that the main screens
  (song list, player bar, tab view, game) still appear (`drawHighway` / `drawGame`
  return early when their canvas is under 60 px: at ~0 height while the view opened,
  sizes went negative and `arc` threw, which broke the page when opening a tab): a stray closing tag once pushed
  the player bar off screen, and removing the Guitar Hero look took the 3D/Flat
  setting's code with it.
- Mac-only paths (app shortcut, MPS fallback, certificate fix) were written on Windows;
  say when something is untested on a Mac.

## Files

- `app.py`: everything server-side: the separation engine, the library on disk, the
  processing queue, and a FastAPI web API. Also serves the page and creates the app
  shortcut. Run with `python app.py` (port 7860, opens the browser).
- `static/index.html`: the whole UI in one file (HTML + CSS + vanilla JS, no build
  step, no external downloads so it works offline).
- `start.sh` (Mac/Linux) / `start.bat` (Windows): create `.venv` on first run, reinstall
  packages whenever `requirements.txt` is newer than `.venv/.installed`, then start.
  Both pick Python 3.10-3.13 (newest first; start.sh also looks in python.org's and
  Homebrew's folders) and remake a `.venv` built with anything else. The user's
  brother got "No matching distribution found for truststore": a Mac's built-in
  python3 is 3.9 and truststore needs 3.10. With none found they say to install 3.13
  from python.org. start.bat once lost its backslashes, slashes and `||` (it went
  through an .rtf in git history; restored from the first commit's start.bat.txt).
  It's untested on a real Windows machine since; keep it CRLF.
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
- `tone.py`: "Match the song's tone" (an EQ for Hear myself). Module docstring has the
  method and measurements; see "Match the song's tone" under My tab.
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
  (library root) .calibration.json         string + timing calibrations (copies of the page's; "previous" keeps replaced ones)
  (library root) .game-checks/<date time> <song>/   recorded games for checking: run.json + input.flac
  (library root) .instrument-tones.json    the user's instruments for tone matching: name, bass/guitar, captured spectrum; the one picked per type; pooled bass recordings; "removed"
  .tone.json                   the song's bass / guitar spectra (from its "bass only" / "guitar only" tracks) for tone matching
  (library root) .bass-recordings/         the user's labelled test recordings (FLAC) + recorder pages and analysis scripts; README.txt
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
  every imported tab is a right answer for its song, so `learn_from_imports()` fits two
  things to all of them and saves them in `<library>/.learned-from-tabs.json` (written
  to a .tmp and swapped in) with held-out before/after scores. It runs on its own
  `learner` thread, not the job queue: after an import, after Remove import, and at
  startup, whenever the set of imported tabs changed (`queue_learning()` sets an
  Event; tabs imported meanwhile cause one more round). On the queue it made tabs sit
  at "waiting to start" for minutes, which the user took for a hang. 4 tabs: ~2 min. Each is used
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
    Speed: the melodia step in `_note_events` walks cells strongest first instead of
    an argmax over the whole grid per note (identical notes, 10-15x faster), and
    `_fix_octaves(memo=...)` keeps each note's answer across settings: ~5 s per song
    for all three fits (was ~45 s).
    Must beat the built-in settings by 0.5 points held out. On Cry Baby Cry alone the
    hand-set values were already near best (F1 0.866; held-out 0.859, so not used);
    most misses were re-plucks inside one long note. With 4 tabs (Cry Baby Cry,
    Electra, Wow! Wow! Wow!, You Only Live Once) it was used: 0.748 -> 0.776; fingering
    0.622 -> 0.784. Automatic tabs store `note_settings`; when the learned ones differ,
    the tab is still shown, with "Redo tab to use what was learned from your imported
    tabs" (hiding it behind "can be made better" looked like Redo did nothing).
    `split_default` sets the split switch until the user flips it for that song.
  - Note models (`tab.learn_note_models`, needs 2+ tabbed songs with a "bass only"
    track): the setting search couldn't tell real re-plucks from ringing/leakage (user:
    fake notes, e.g. a held F#2 in You Only Live Once came out as three A9s and a stray
    B3 as G16; and missing 8ths). On 4 songs the misses were mostly re-plucks heard as
    one note (280), wrong octaves (136) and nothing heard (186); fakes were mostly wrong
    octaves (Wow! Wow! Wow!: 99). Three logistic regressions (numpy, saved as JSON in
    the learned file) labelled by the tabs, applied after detection: octave (move a note
    down/up, energy at f/4..3f vs f), pluck (split a long note at a beat-grid point:
    attack, onset, loudness/overtone rise), keep (drop a likely-fake note). Cuts
    0.5/0.4/0.3. Leave-one-song-out: found 2068 -> 2213 of 2802 tab notes, fake 417 ->
    431, F1 per song +0.02-0.04 (Wow! 0.676 -> 0.712); the old split switch added
    hundreds of fakes. Used only if leave-one-out F1 improves by 0.5 points (4 tabs:
    0.782 -> 0.813). With models, "Split repeated notes" = the pluck model, on by default.
    Not perfect: that F#2 still has one extra note. ~6 s to fit.
    Same-pitch notes that touch are joined first (`_join_repeats`) and the pluck model
    also decides at those joins (with the split switch off, only there): found 2238,
    fake 436. You Only Live Once ~0:54-1:08 (user: "still doesn't look right"): the
    8ths at 54-58 s now mostly come out, but held D4/D2 notes at 60-65 s still get
    extra re-plucks. Measured: real plucks there rise +5-16 dB / overtones +8-20 dB,
    the extras <= +4 dB; but the muted picked 8ths just before don't rise either, so
    per moment they look alike. Tried and dropped: overtone rise in the keep model (no
    change); attack strength at the 4 beats each side as context (no fix there, Wow!
    521 -> 462 found: drums leak into the bass track). In the same passage the learned
    fingering plays A7/A9 where the tab moves down to D2/D4/D6 (the costs prefer the
    song's usual 7th position, 90% right for the song; not the usual-place rounds).
  - Runs in its own process (`learner` spawns `_learn_in_child`: nice 15, the note model
    on 2 threads, torch 2), 20 s after startup (`LEARN_DELAY`), reporting progress over a
    queue (user: "when the app loads and does the learning, it slows everything down a
    ton"). Measured on the test library with a forced re-learn: the server answered tab
    requests in 4-13 ms either way, so the slowdown was the whole Mac (learning at normal
    priority on every core); now the server sits at 0% CPU and the child at nice 19.
    Re-syncing an import keeps its "made" time: the learner compares those, and a new
    time made it re-learn everything at the next start.
  - `LEARN_RULES` in app.py: bump when learning changes; saved learning is redone at the
    next start (learning otherwise reruns only when the imported tabs change, so the
    note models never got learned in the user's library after the update).
  - Status (user asked to see what it's doing): the learner fills `learning` (step,
    fraction, tabs, error), `GET /api/learning` returns that plus a summary of the
    saved learning; the sidebar under the device label shows "Learning from your 4
    imported tabs…" with the step and a bar (polled every 1.5 s while running, 15 s
    otherwise, and right after an import), then "Learned from 4 imported tabs · 1:55 PM
    / Notes 78% → 82% · Fingering 62% → 78%" (hover explains the numbers). Timing on an
    idle M4 with 4 tabs: reading 7 s, notes 38 s, fingering ~100 s (the slow part); a
    run alongside other heavy work took 22 min.
  - The tab view's subtitle says what's in use ("Fingering learned from your imported
    tab (matches it 99%, was 79%)", "Notes tuned to your imported tabs (...)").
- Page: Tab button in the player opens the view over the song list. Lines view: two
  lines at full size, the one being played on top (on the card, with the playhead and
  the sounding note highlighted), the next below; on each new line the line below moves
  up and the one after fades in underneath, nothing slides (`buildSlots`, `makeSlot`;
  user: "show the current and next line", "not have to keep shifting my eyes", "the
  previous line should change to the current line so I can see the next"; it was
  previous / current / next / next-but-one sliding up every line). Click a line to seek; tuning picker (choice kept
  per song in localStorage); lines shrink to fit (5 strings, small windows). The
  playhead follows `audioNow()`, the audio clock smoothed between its updates (with the
  'playback' latency setting `ctx.currentTime` moves in ~23 ms steps; at 120 Hz the
  Highway and game scrolled in jumps, user: "a little choppier"): carried on by
  performance.now(), eased a tenth of the way toward each update, never backwards
  (jumps when > 0.1 s off). Playhead test: still in ~2 of 3 frames before, every frame
  moving after, steps within ~15% of each other. It also subtracts the audio output delay (`ctx.outputLatency + baseLatency`, ~87 ms
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
- Hold bars (user: "show how long to hold notes, like Guitar Hero"): each note has a
  soft accent bar along its string from its start to its end (or the next note on that
  string, or the line's end), behind the fret number; notes still ringing from the line
  before continue on the next line without a number. While a note sounds its bar fills
  up to the playhead; finished bars go back to dim. Imported tabs: lengths from the
  tab's rhythm (ties merged); automatic: as heard.
- Highway (user: "a Guitar Hero-like mode", chose view only, no scoring; and "a
  separate mode, not replace what we have"): a Lines | Highway switch in the tab view
  (Lines is the default; choice kept per computer, `track-splitter-tab-mode`). Drawn
  on a canvas each frame (`drawHighway`). Sideways like tab (the user found that
  better than the game's falling notes): one lane per string, highest on top, in the
  game's colours; gems with the fret number travel right to left to a hit line on the
  left, tails as long as the note rings; a held note sits on the line and glows; bar
  lines drift left; the far end recedes a little (CSS rotateY). Speed: 3 s ahead, but
  at least fast enough that the song's quick notes (20th percentile gap) don't overlap,
  so narrow windows show less time. Clicking the highway plays/pauses.
  User: "sort of hard to read" (too fast, too small): lanes now fill the height (up to
  120 px), gems up to 52 px with numbers ~0.62 of that, string names 13-22 px; a
  Slow-Fast slider (0.25-1.6 x 3 s ahead, 120-420 px/s, nothing overrides it since the
  repeat ticks (see Game looks); `track-splitter-highway-speed`)
  replaced the speed-from-the-quickest-notes rule above. (Crowded gems used to get
  narrower; now every gem in a song is one size, see "Gem size".)
- Toolbar cleanup (user: "it's getting really crowded and hard to use and gets messy
  when things are selected causing everything to move around"; "I don't even want the
  subtitle"): the Tab section's row is fixed: source, view, tuning (210 px), View ▾,
  Bass input, More ▾. View ▾ (`#tab-viewopts`) holds what depends on the view: speed,
  Scroll, Flat, Split repeated notes, raise ½ step, Show my timing + Calibrate (one menu
  open at a time). What comes and goes floats over the tab (`.tab-stage` / `.tab-float`):
  the timing average top right, the draft's Delete note / Undo top left. The subtitle is
  hidden; its text is the song name's tooltip. Checked: no toolbar control moves when
  switching views, sources or toggling any View setting.
- Header (user: "really crowded", "put the game in a different section", "make the input
  a bubble popup"): title + subtitle, and top right [Tab | Game] (`#tab-section`,
  `tabView.section`, `#tab-view[data-section]`; `.only-tab` / `.only-game` elements) and
  close. One row of tools: which tab, Lines/Highway (Tab), speed (Highway or Game),
  tuning, raise ½ step (when it applies), Delete note/Undo (when a Mine note is
  selected), "Your best" (Game), Bass input, More ▾ (`#tab-more`: Timing, Split
  repeated notes, Import tab…, Redo tab, Remove import/mine; an action closes it),
  Start (Game). The Game section always draws the Highway without changing the Tab
  section's Lines/Highway choice.
  - Bass input is a bubble (`#input-bar`, absolutely positioned under its button) so it
    can't push the tab around (as an inline bar its text changed width and wrapped, and
    the highway jumped). Clicking elsewhere hides it but keeps listening; the button
    then shows a dot and the note (green on a match, red otherwise), "Listening"
    between notes, "Rec 0:12" while recording; × in the bubble turns the input off.
    The note readout and Record button have fixed widths.
  - Highway tilt: rotateY(12deg) from the left edge, so the far end recedes (-12deg
    brought it closer and cut the score off); 6deg in a game, to use more width.
- Scroll (the Lines view's switch, `track-splitter-tab-scroll`; user: "still hard to read
  the tab, maybe a horizontal scroll toggle so lines stop shifting"): `drawTabScroll`, one
  continuous tab staff on a canvas gliding right to left past a fixed playhead a fifth
  of the way in: thin strings up to 76 px apart, plain numbers up to 34 px on a gap in
  the string (crowded ones shrink to their room, down to 60%), the sounding note
  highlighted, bar and faint beat lines (`d.barTimes`), soft bars for held notes,
  Show my timing marks above notes, string names in a fixed column. Speed: the slider
  (shown in this mode), never so slow that ordinary notes touch (`typicalLaneGap`).
  Click plays/pauses.
- Flat (Tab section Highway, `track-splitter-highway-flat`; user: "a flat highway in the
  tab view, make it a toggle"): a switch next to the speed slider takes the lean off.
  Spacing (user: "more space on the highway"): the slider rules, but never so slow that
  the song's ordinary notes on a string overlap (`typicalLaneGap`, the median same-string
  gap; Wow! Wow! Wow!'s 8ths piled up at the slowest setting); lanes up to 140 px.
- Highway look (user: "make it look better too"): dark rounded backdrop, each lane
  tinted in its string's colour and its string drawn through it, bar lines plus
  fainter beat lines (from bpm), a glowing hit line with a target ring per string
  (lit for 0.15 s by a hit), shaded gems with a dark edge; in a game "Perfect!"
  (pluck within 70 ms) or "Good" floats up, a score panel (score, streak, accuracy, a
  multiplier badge grey/blue/green/gold), a progress bar on top, the countdown in a
  circle. Full screen in a game (user: "doesn't take up the whole screen"): the
  player bar hides too, title and Stop share one slim row, the grid row is
  minmax(0, 1fr) (with plain 1fr the canvas grew past the window), lanes up to 170 px
  below a 100 px strip for the score.
- Game, leaving and pausing (the user closed the tab mid-game: the game never ended,
  the page stayed in game layout and looked frozen): closing the tab, turning the
  input off or loading another song ends a game quietly (`endGame(false)`, no results).
  The × is hidden in a game; the button reads Pause; Pause/Space opens a pause menu
  (`#pause-dlg`: points and position, Resume / Restart / Quit). Resume counts 3-2-1
  and plays on from the same spot (no seeking); Escape = Resume; Restart starts over;
  Quit ends with the results. The song's end (its onended resets the position to 0)
  still ends the game.
- Timing: Relaxed (default) / Normal / Tight (`TIMINGS`, `track-splitter-game-timing`),
  hit windows early/late/Perfect 0.12/0.14/0.05, 0.09/0.10/0.04, 0.06/0.07/0.03 s (user:
  "we need to bring the range for a hit way down"; they were 0.25/0.3/0.1, 0.15/0.2/0.07,
  0.1/0.12/0.05 when the user wanted wiggle room and plucks were read ~120 ms late).
  Best scores per song + tab + timing. Simulated on-time player: Relaxed 29/29, Normal
  28/29, Tight 27/29; 0.2 s late on Relaxed: 2/28 (it used to score). Calibrate timing
  matters more now.
- Hit animation (user: "when I hit a note, make animation like guitar hero", "and when
  I hold"): on its own clock (performance.now()), drawn with 'lighter' blending: per
  hit a radial flash at the target and 18 sparks in the string's colour (+8 gold on a
  Perfect) fanning upwards and falling a little (`game.bursts` -> `game.sparks`, at
  most 600); the target ring swells for 160 ms; while a hit long note is held at its
  pitch (the hold bonus sets `game.sustain[string]`) sparks stream and the target
  glows and pulses; the multiplier badge pulses when it goes up; every 50 in a row
  "<n> note streak!" in gold for 1.4 s.
- Gem size (user: "make all the notes the same size, some are very small"): one size
  per song and speed (no more narrowing crowded ones): width from the song's quick
  same-string gaps (10th percentile) x speed, 0.7-1.25 x the gem height; the number
  sized for two digits.
- Game look (user: "more professional", "like an actual game"): game mode has its own
  renderer, `drawGame` (the Tab section's Highway, `drawHighway`, stays plain and has
  no game code). A full-window stage (no app chrome; only a pill-shaped Pause over it):
  dark gradient, two slowly sweeping lights, drifting dust, vignette; the fretboard in
  perspective, sideways: z = seconds ahead (speed slider: 2.6 s / speed), depth(z) =
  dd/(z+dd) shrinks gems, tails and lanes toward the right, lanes converge, glowing
  rails, bar/beat lines across; strike buttons per string (swell and fill on a hit);
  beveled gems with a shadow and white outlined numbers, one size per song; sparks,
  flashes, PERFECT!/GOOD, hold glow; HUD: song + timing top left, glowing score, a
  multiplier ring filling toward the next level, "n NOTE STREAK · %", a GROOVE meter
  (last 24 notes) and a progress line; countdown 3-2-1-GO!; milestones. Pause and
  results are `dialog.game-screen`s (purple glow, pill buttons); results show 1-5
  stars (accuracy 40/60/75/88/95%), "Song complete" or "Results", the score counting
  up.
- Game looks and HUD (user: "hard to see what's coming up", "like Rock Band"; a
  vertical Rock Band track was tried and dropped: "vertical is too confusing, keep it
  horizontal"). A Guitar Hero look (GH3-style LCD panel, ROCK meter, round gems,
  flames) was added and later removed (user: "get rid of the guitar hero view, it sort
  of sucks compared to Rock Band"); Rock Band is the only look.
  - Both: sideways perspective with far notes at ~half size (dd = Z), one gem size per
    song and speed. Speed is the slider alone (Note speed 0.25-1.6; 2.6 s ahead /
    speed). There used to be a floor so two notes on a string never touched
    (`minLaneGap`), which on busy songs overrode the slider (Wow! Wow! Wow! ~520 px/s in
    the Tab Highway against Slow's ~150; user: "still too hard to read the highway",
    "too fast, even on slow"). Now gems narrow to fit two digits (0.7x their height) and
    a quick repeat of the same fret on a string, closer than a gem, is drawn as a slim
    tick without a number, so a run of 16ths reads "7 ||| 5 ||| 3" instead of a pile of
    overlapping numbers (both views). Whether a note is a tick is decided once, from
    its spacing at the strike line, not per frame: judged at each depth (and dropped
    once the note before was hit) notes turned from tick to gem on the way, and the
    user saw them "pop in and out". Far notes fade in over the last 15% of the board.
  - Board (Game settings: 3D / Flat, `track-splitter-game-board`; user: "more legible
    from a not perspective"): Flat has no depth (every note the same size, straight
    lanes) and half the seconds ahead, so notes cross the strike line at the 3D board's
    speed.
  - Rock Band (from the user's screenshot): rounded fret buttons; slab gems (lit top,
    darker front edge, gloss, white number); top right a black score box with a white
    border and spaced digits, five star medallions under it (silver when earned, gold
    at five, the next one's ring filling: `starsFor`, score vs a perfect run of the whole
    song (`perfectScore`: every note, x4 from the 30th) at 12/25/45/65/85%; it was vs
    100 a note at 0.25-2.4x, and the user "got gold star way too fast, like halfway
    through"); the multiplier badge (blue ring, "4x") at the base of the strike line
    with a yellow bar up the line toward the next level.
    The hit flash is a flare in the button's shape (the user disliked a round one).
  - Groove on the board (user: "make the groove meter look better, incorporate it into
    the tab", "in Guitar Hero, when you're doing well, it does something similar"):
    the board's rails and strike line glow by the groove (last 24 notes): green, then
    yellow, then red and pulsing below 40%; at x4 (max) the rails turn electric blue
    like the full meter (user: "the outline of the highway should turn blue"), thicken,
    and a blue wash lights the board. The separate groove bar is gone.
  - Rock Band multiplier meter (from the user's Rock Band picture; replaced the blue
    badge at the base): a chrome-framed bar along the strike line just behind the fret
    buttons, a silver "4X" disc in its middle, yellow filling outward from the disc
    toward the next multiplier; at x4 it's full, glows electric blue and throws off blue
    sparks (`maxSparks`). No streak count or percentage (the user didn't want them).
  - The board uses most of the height (lanes up to 170 px, 66% of the window for
    the strings; user: "still cramped").
  - Play on: Board / Tab (Game settings, `track-splitter-game-view`; user: "an option
    to play the game mode in the tab", then "still add animations and make it look like
    Rock Band's UI"). Tab plays the game on the Lines view filling the window
    (`tabGameFrame`; `body.game-tab`): notes turn green when hit, red when missed,
    amber when messy (`.note.hit/.miss/.messy`, line notes carry their index), and a
    canvas over the lines (`drawTabGameFx`) has the board's score box and star
    medallions (`rbScore`, shared), a chrome multiplier meter bottom left (yellow, at
    x4 blue and sparking, and the playing line's card glows blue: `body.max`), a flash
    in the string's colour and sparks from each note you hit (`noteBox` finds it on
    screen), PERFECT!/GOOD rising from it, MISS/MESSY from missed/messy notes, streak
    milestones and STREAK LOST across the top, the countdown, and "You: A5" bottom
    right. Lines are re-laid out for the game's (difficulty's) notes and window when it
    starts and for the whole tab when it ends. Same listening and scoring as the board.
  - Testing: localStorage `track-splitter-game-start` (seconds, set in the console)
    starts the game later in the song; Electra's first ~20 s are nearly empty, so tests
    use 30.
  - Results: finished songs get stars from `starsFor`; stopped early, from accuracy.
    Header (user: "instead of 'Song complete' with an X: song name big, band name
    under smaller, and difficulty"): title, artist, a pill "Expert · Relaxed timing"
    (+ "stopped early"); no ×, Close and Play again in the footer.
  - Difficulty (user: Fireman & Dancer, Wow! Wow! Wow! "too hard to play", "difficult
    to see the notes"): Easy / Medium / Hard / Expert in Game settings
    (`track-splitter-game-difficulty`, default Expert). `gameNotesFor` keeps at most
    one note per beat / half / quarter beat (tab bpm, 120 if none; double stops stay
    together except on Easy); startGame swaps `tabView.data` for that copy (caches
    `minLaneGap`/`quickLaneGap`/`lanes` cleared, so fewer notes also scroll slower) and
    endGame puts the whole tab back. Best scores per difficulty (`bestKey`; Expert keeps
    the old keys).
  - The game plays the "no bass" version (user: "when you go to game mode, always do no
    bass"): startGame loads the song's no-bass version (HQ first) if the loaded one
    removes something else, and switches to Without; the mode before comes back after.
  - What you're playing (user: "show the string you are hitting even if there's no
    notes"): the string's strike button gets a white ring (green on a match, amber when
    it's the right note on another string) and the fret number, between notes too
    (`input.lane`, see String detection).
  - Messy notes (user: "I was off note and slid on and it still counted"): still scores
    but ends the combo; amber "MESSY · combo lost" instead of PERFECT/GOOD. A note takes
    the pluck closest before its pitch is first heard (an earlier one belongs to whatever
    was played before; taking it made clean notes look messy). Messy when the note
    started without a pluck of its own (a pitch change, `soft`: slid or hammered) and the
    real pluck within 0.35 s before was on a wrong note actually held (3+ readings in a
    row of one pitch that isn't this one, an octave, or one of the tab's last 3 notes;
    pitches a slide passes through don't count), or it was plucked but such a held wrong
    note came before its pitch. Never for notes the imported tab marks as slid or
    hammered into (`legato`, the 5th value of a note; Guitar Pro slide-into / slide-to-
    next / hammer-on and pull-off marks, read by tabimport; `IMPORT_RULES` 3 re-syncs old
    imports; MCR has 32 such notes, You Only Live Once 28, Electra none). The user felt
    combos broke while on time: the first version fired on clean playing (e.g. a note
    the tab doesn't have, plucked just before). Simulated Electra: 0 false MESSY in 40 s;
    a held wrong note slid into the right one: MESSY.
  - Pause is a pill at the top centre (the HUD has the corners).
- Game settings (user: "a settings page in the pause menu, like changing the look"):
  `#settings-dlg`, opened from the pause menu's Settings (Done returns to it) or
  "Game settings" in the Game section: Look, Note speed (drives `#hw-speed`), Timing,
  Mistake sound, Hear myself + My bass volume (drive the input bubble's), Song volume
  (drives the player's). Timing and Mistake sound moved here from the Game section.
  Also Difficulty and Timing calibration (opens the calibration screen).
- Miss timing (user: "the miss indicator, it's really delayed"; it looked like ghost
  notes): a note is missed `MISS_AFTER` (0.06 s, was 0.15) after its late window closes,
  in the game and Show my timing: on Relaxed ~0.36 s after the note instead of ~0.45 s
  (shorter on Normal/Tight). Simulated on-time player: 38/38, streak unbroken.
- Mistakes (user: "feedback when a mistake is made"): a miss flashes that string's
  button red with "MISS" rising, and a lost streak of 10+ shows "STREAK LOST · n"; a
  wrong note (`checkWrong`: a new pitch the tab doesn't have while it expects another
  note within the window; not over rests) shows "WRONG" + "F2 · tab E♭2". A short thunk
  (filtered noise + a low knock on the player's context) plays on both unless Mistake
  sound is off. Labels on one string stack. Show my timing also says "Missed" (only
  while you're playing) and "Wrong: G1". Wrong notes don't change the score.
- Responsiveness and finger lifts (user: "it says I missed notes right after I hit
  them, like when I lift my finger off"; "make it more responsive"): the input is read
  every 20 ms (was 40). Hits are judged from a pluck plus one clearly pitched reading
  (`raw`); the steady note (live readout, wrong-note checks) needs 5 readings in a row
  (~100 ms). An attack is the level rising 3 dB over the reading before last. A wrong
  note needs a real attack within 0.3 s: lifting a finger lets an open string ring
  (a new pitch with no attack), which used to flag a mistake. Same in Show my timing.
  Simulated perfect run: 52 streak, 100%, no stray WRONG/MISS.
  Plucks (user: "I can't keep a combo even though I'm on", "lag between when I hit a
  note and it shows"): the 85 ms buffer's peak often didn't rise when a note was
  plucked while the last one rang, so the pluck came from the steady pitch (5
  readings) ~120 ms late, or fell out of the window. Now an attack is also the newest
  20 ms jumping 4 dB (`plucked`, as for string detection), and a new pitch from 2
  matching readings adds a `soft` pluck; every sign is kept (de-duplicating cost notes:
  22 vs 12 missed). Simulated Electra (40 s from 30 s): hits judged a median 119 -> ~58
  ms after the note's time (p90 274 -> ~255); misses cluster at 56-69 s in both old and
  new (that passage of the separated track doesn't match the tab).
- Lines spacing (user: "make sure all the notes are easy to read and spaced enough"):
  `layoutLines` re-cuts the tab's bars into lines, recomputed on resize: one bar count
  per line for the whole song (8, 4, 2 or 1, the most that keeps 80% of lines with two
  notes on a string >= 28 px apart, within ~6.5 s), lines every N bars from the top.
  Readability (user: Wow! Wow! Wow! "so hard to follow along"): faint beat lines inside
  each bar (beats from the tab's bpm, else 4) so steady 8ths show their pulse, and hold
  bars only for notes held >= 0.45 s (every 8th's bar ran into the next and smeared the
  strings). At 650 px of tab Wow! is 1 bar a line (1.9 s); ~1100 px gives 2.
  It used to pack each line greedily, so lines ran 1-4 bars (Fireman & Dancer 1.6-6.4 s)
  and the playhead sped up and slowed down, and odd-length lines split repeated riffs
  differently (user: "standardize the speed"). Now Fireman 2 bars (3.2 s) every line,
  Electra 2, Amsterdam 2, Save Yourself 1 (busy 16ths) at 1280 px wide; numbers only shrink for a close
  same-string neighbour (other strings are other rows). The Tab section's Highway
  has the same speed floor and minimum gem.
- App finish: a refinement layer at the end of the stylesheet (deeper surfaces,
  34/30 px controls, 10 px radius, gradient primary buttons and active segments,
  focus rings, frosted player bar, the playing tab line on a card with a glowing
  playhead, softer dialogs).
- Show my timing (user: "a toggle for the normal highway that lets you see the timing
  on the hit"): a switch in the Tab section when the Highway is shown (opens the bass
  input if needed; `track-splitter-show-timing`). `practiceListen` uses the game's
  pluck + pitch matching and windows (the Timing choice) without score or locks; each
  hit rises from its target as "On time" (<= 20 ms) or "Early/Late n ms" (green within
  the Perfect window, else blue/amber), and top right "Avg: n ms late" over the last
  16. Jumping elsewhere resets it. Simulated player 60 ms late: average read ~39 ms
  late (detection timing assumptions; Timing under More shifts it). With the switch on
  but no bass input, the Highway says so.
- Switching songs with the tab open (user: "I got the wrong tab when switching songs"):
  the title changed at once but the last song's tab stayed until the new one arrived
  (seconds while an import re-syncs). `loadTab` now clears it and shows "Loading the tab
  for …" as soon as the song changes.
- Whole tab (user: "would be nice to see the whole tab"): a third view next to Lines /
  Highway (`tabMode` 'whole', `wholeFrame`): every line at the small size in a scrolling
  column, the playing one on the card with a playhead and note highlights, kept centred
  unless you scrolled in the last 4 s; click a line to jump.
  Hand scrolling is told apart by wheel/touch/keys/scrollbar, not 'scroll' events: the
  page's own smooth follow-scroll counted as the user's and following stopped for good
  (user: "whole tab should still scroll"; at 0:30 the playing line was below the view).
- String detection (user: "a 5 on the E is an A but shows as an open A"): the same pitch
  on another string sounds different (how sharp the overtones run, inharmonicity B; the
  levels of harmonics 2-6). Measured on the user's two test takes (scratch analysis;
  recorder page, slow + fast 8ths at 130 bpm): from 0.1 s of a note, fast notes right
  84-93%, 94-98% of the 70% it's surest about; B alone fails at speed (10-20x noisier).
  - In the page: `stringFeatures` (4096 samples, Hann, x4 zero-padded FFT
    (`fftMagnitude`), 12 harmonics, a line fit of (f_n/n)^2 vs n^2; null unless within
    half a semitone of the expected pitch), `fitStrings` (per string: open notes' mean,
    fretted notes a line in the fret; one spread per feature), `stringScores` (every
    place the pitch can be played on the bass as tuned now, `shownTuning()`; a
    5-string's B is uncalibrated), `checkStrings` (fit on every other note of each spot,
    test the rest). Checked against the Python analysis on the same takes: same numbers.
  - Live: plucks for this come from the newest 20 ms of the analyser (>4 dB over the
    last 3 readings, 90 ms apart): the 85 ms buffer's peak barely rises when a ringing
    note is re-plucked (fast repeats were missed). 115 ms after a pluck the buffer holds
    ~0.03-0.12 s of the note (`readString`; the pitch is the reading then, or the last
    one: low notes often have no clear reading yet). Shown only when the best place
    beats the next by `margin` (max(2, the self-check's 95% margin): on positions left
    out entirely that was right 95% with 73% of notes shown). Readout: "on A5 · D0";
    the Highway ring and the game's button go on that string; amber + "same note · tab:
    A5" when the tab has it on another string.
  - Calibration (`#strcal-dlg`: More → "Calibrate strings…", "Strings…" in the Bass
    input bubble): standard tuning, each string at frets 0 2 3 5 7 8 10 12, 8 notes a
    spot (the expected pitch only; full spots move on by themselves), a fretboard
    showing progress, Back / Skip, Save once every string has its open note and two
    fretted spots. Saved per computer (`track-splitter-string-calibration`: samples,
    margin, self-check). More → "Reset string calibration".
    Both calibrations (strings, timing) are also kept in the library folder (user: "save
    our string detection data so we don't ever need to do it again"): `keepCalibration`
    saves to localStorage and `POST /api/calibration` ({key, value}, null forgets;
    `<library>/.calibration.json`, a lock since the page saves both at once, a replaced or
    removed value kept under "previous"); at start the page takes the library's copy, or
    uploads one only the browser has. Tested: an existing calibration was copied on the
    first load; with the browser's storage cleared it came back (96 samples, 42 ms).
    The labelled recordings behind string detection and the hand-touch rule are in
    `<library>/.bass-recordings/` (FLAC, identical to the float WAVs; hidden folders
    aren't listed as artists). End to end with the user's
    fast take as a simulated Scarlett: 155 notes saved, self-check 92%; their slow take
    then read live: right 87% of the notes it named, half named (that test calibration
    had big gaps on the A string). Not yet tried on the real bass.
- Show my timing in every Tab view (user: "why can't I view the timing anymore", then
  "show my timing in the Lines and Whole tab views too"): the switch, Calibrate and
  listening work in Lines / Highway / Whole tab. Lines and Whole tab mark each note
  played (`practice.marks`, `markTiming`): ✓ within 20 ms, −n / +n ms early / late, ✗
  missed, "G?" a wrong note, in the Highway's colours (green within the Perfect window,
  blue early, amber late), set as style (the tab numbers' CSS fill beat the attribute);
  Whole tab only decorates the lines around the playing one. The average of the last 16
  sits next to the switch (`#timing-avg`; the Highway draws its own). Marks clear on a
  jump, toggling, Calibrate and Reset.
- Timing calibration (user: "it always says I'm early": the Mac/browser's reported
  output/input delays don't match the real ones; then "a calibrate page where it does
  it for you with a made-up set of notes"): `#calib-dlg`, opened from Calibrate (next
  to Show my timing), More -> "Calibrate timing…" and Game settings (Close returns
  there). Clicks at 96 bpm on the player's context (4 count-in, then 16 quarter notes
  of a random line: open strings / frets 2-5, never the same string twice) on a small
  4-lane highway; each pluck shows Early/Late n ms. Plucks use the game's attack rule
  and nudge on the same clock (audio clock - output delay + Line timing), without the
  current calibration; the median of >= 8 heard notes is offered as the new
  `inputCalibration` (`track-splitter-input-calibration`, seconds subtracted from
  every pluck and heard-pitch time in Show my timing and the game). While it runs,
  `listen` feeds only `calibListen`. More -> "Reset timing calibration". Simulated
  plucks: shifting them 100 ms later moved the reading by 105 ms, spread ±5-9 ms.
- Hand thuds (user: "my hand hits the string and it picks it up as a note"): live
  detection now needs a clearly pitched reading (YIN dip < 0.1) and 3 readings in a
  row (~120 ms) to agree (was 2 of 3). Test: 16 damped bass-pitched thuds and noise
  bursts gave no note; a perfect game run still scored 83/84.
- Hand touches (user: "my hand hitting the string after I pick is causing it to
  register"): a pluck counts (game `judge`, Show my timing) only with the note's pitch
  read after it (a reading taken >= 15 ms after the attack was detected; plucks and
  readings carry `at`, the moment read; a pitch-change pluck is its own reading). Before,
  any reading in the window counted, so a hand muting a ringing note just ahead of a
  repeat of that note scored the repeat. Measured on a labelled recording of the user's
  bass (scratch recorder page `touch.html`: pluck and let ring / stop with picking hand /
  fretting hand / touch only / normal 8ths / moving around): touches never gave a clear
  pitch within 100 ms (0 of 21) and die away (median -15 dB at 150 ms vs -2.6 for
  plucks), but look like attacks (the touch-only step alone made 21). Replaying it, tab
  notes at touches on a ringing note: old rule scored 17 of 17, new 2 of 17; real plucks
  35 of 35 both. Electra simulated: 31/31 as before. On the earlier string-test takes the
  attack rule also fires ~1 extra per 2 plucks (touches and a ringing low note's 20 ms
  peaks wobbling); level, high-frequency and sustain rules couldn't separate those from
  fast re-plucks, so attacks stay as they are and the pitch rule does the work.
  Waiting for the pitch (user: "timing in game is a little too strict now"): a pluck's
  own pitch reads clearly 20-200 ms after it on the user's bass (median ~90 ms; the attack
  blurs the first readings), so a note with an unused pluck in its window isn't a miss
  until `PITCH_WAIT` (0.22 s) after that pluck, and the pitch may come up to then (game
  and Show my timing). The miss timer alone (window end + `MISS_AFTER`) cut slightly late
  plucks off even before: replaying the touch take on Normal 50 ms late, pluck-then-stop
  notes counted 20/35 (old), 19/35 (pitch after the pluck), 35/35 with the wait; touches
  still 4/17. Electra simulated: Relaxed 30/31, Normal 30/31, Normal 60 ms late 29/31.
- Game check (user: "inconsistencies with the game and picking up random notes, can we
  make a test page to figure out what is going on"; "do Addictive as the test"): Game
  settings → "Record games for checking" (`track-splitter-game-check`) records each game:
  the input as the analyser hears it (`startCheck`, the recorder worklet), every 20 ms
  reading from `listen` (ctx time, tab time, song position, level, newest-20-ms level,
  gate, Hz, raw/steady/shown note, pluck signs), every pluck (with what set it off), hit,
  miss (with why: no pluck in the window / only too-soft plucks / its pluck went to
  another note / pitch never heard / pitch only before the pluck) and wrong note, and the
  notes with results. `finishCheck` (from endGame) posts it to `POST /api/game-checks`
  -> `<library>/.game-checks/<date time> <song>/` (run.json, input.flac);
  `GET /api/game-checks[/{id}[/audio]]`. `static/check.html` (Game settings → Game checks
  → Open…): pick a run; a summary (misses by reason, wrong notes, plucks that scored
  nothing), a zoomable timeline (tab lanes coloured by result with their hit windows,
  pitch readings green when the tab has that note then / red when not, plucks white /
  violet (new pitch) / green (scored) / orange (nowhere near a note), wrong notes, level vs
  gate), hover details, and a list of every miss / wrong / messy / stray pluck that jumps
  there. Click plays your input from there with the song as you heard it (song position
  minus the output delay). A test start (`track-splitter-game-start`) now skips the notes
  before it. First finding (simulated, Addictive's own bass track, 0:23-0:25): repeated
  E♭2 8ths read the right pitch throughout but the level barely moves, so no pluck is
  detected and all miss; the user's real normal-8ths take also lost 5 of 24 re-plucks.
  The user's own game check (Addictive, whole song, Relaxed, real bass; "my playing
  sounded right", "on 8th and 16th notes it loses a bit of accuracy"): 602/726; 105 of
  124 misses "no pluck in the window", 101 of them a note repeating the one before (8ths
  ~0.22 s apart): re-plucks of a ringing note raised the newest 20 ms only 0.6-3.5 dB (hit
  ones 4-10), under the attack rule. Both show the pitch readings blinking out for 1-4
  readings at the attack, then the same note back. Re-pluck rule (`rePluck` in `listen`,
  a soft pluck in gameListen / practiceListen timed at the blink, unless an attack or new
  pitch already counted): replaying that game's frames through the judging in Python
  (scratch `replay/sim.py`, reproduces 602/726 note for note) gives 680/726 (+78; blink
  <= 2 readings: +18, <= 3: +54). Ringing notes blinked unplucked once in 91 s (slow
  string-test take); after a hand touch the note came back 1 time in 25. The
  separated-track simulation gains less (Addictive 0:20-0:34: 45 -> 48 of 56): its repeated
  E♭2s neither jump nor blink. Also seen: the user hits a median 49 ms early with the tab
  on the recording (imported notes vs heard notes: median 0 ms): Calibrate timing again.
  Second game with the rule (user: "it's more the missed notes that I actually hit that I
  care about"): 434/461 (94%), 147 re-plucks found by it. Of its 27 misses 11 were repeats
  whose pitch blinked exactly 5 readings, so the rule takes 1-5 (replays: first game 696/726,
  second 439/461; ringing notes still blinked unplucked once in 91 s, touches 2 of 25).
  The pitch must also be read before the next pluck (`ownPitch`): within PITCH_WAIT it
  could be the next note's, and a control (the game replayed with every tab note moved 2
  semitones, notes not played) scored 97 such hits; now 34, real hits unchanged.
  Still missed (~9 a song): a note on another string while the last one rings (G1 on E then
  B♭1 on A; F1 then C2): YIN finds no clear pitch for ~300 ms. Tried and dropped: the chord
  reader (`heardNotes` with a bass range) for the wanted note (+13 real, +62-133 false in
  the moved-tab controls), and "the wanted note's own energy rises" (any pluck raises
  every frequency: hundreds false). Repeats with no blink and no level rise (~4) remain.
  Pick vs fingers (user: detection "way better using my fingers than a pick"): a labelled
  recording (recorder page `pick.html`, kept in `.bass-recordings` with `tools/live.py`, a
  Python copy of the live detection) showed no difference at one note per click (pick
  timing tighter). Two Fireman & Dancer game checks did: fingers 221/233, pick 202/224
  with 13 "no pluck" misses (fingers 2), mostly repeated notes: re-picking a ringing note
  kept the pitch reading unbroken (no blink) and raised the level only 1.4-3.8 dB. A pick's
  click is bright, so a pluck is also the newest 20 ms's high frequencies (the input's 2nd
  difference) jumping >= 9 dB over the two readings before (`clicked`, passed to
  gameListen / practiceListen with `plucked`; the readout and string detection keep
  `plucked`). Replays of every recorded game (scratch `gamesim.py` on run.json frames + the
  saved input, reproducing the game's counts): pick 202 -> 210, fingers 221 (same),
  Addictive 696 -> 708 and 438 -> 445, the rest +0-5; the 2-semitone wrong-notes control
  within +-2 everywhere. Game checks now also record `hf_db` and `clicked` per frame.
- Too-quiet plucks (user: a note "too quiet to actually hear" still counted): in a game a
  pluck counts only if it's no more than 15 dB below the median of the player's last 40
  pluck levels (once there are 8). Each pluck's level is the newest 20 ms (`freshDb`),
  its max over the 60 ms after the attack; hammer-ons/slides use the level then. A miss
  with only quiet plucks in its window says "MISS · too soft". Simulated Scarlett on
  Electra from 30 s: 31/31; a 4 s stretch 26 dB down: its 10 notes missed, the rest hit;
  10 dB down: 31/31. (A test start via `track-splitter-game-start` counts the notes before
  it as misses: "31 of 66" there means all 31 after it.)
- Game (user: "a guitar hero style game that keeps score; once you start, you can
  only stop it, you can't move around"): Start in the Game section (any tab:
  imported, automatic, mine) opens the bass input if needed, switches to the Highway,
  jumps to 0:00 and counts 3-2-1, then plays. `gameListen` (from `listen`, every 40
  ms): plucks = the input jumping up (from below -50 dB, or by 3 dB) or a new pitch
  (hammer-ons); heard pitches minus the player's shift. `judge`: a note is hit when a
  pluck is within -0.15/+0.2 s of it and its pitch is heard in that window (+0.15 s
  for detection delay); dead notes need only the pluck; a pluck serves notes within
  30 ms of each other (double stops). The pluck rule stops a held note scoring a run
  of 8ths. 100 x multiplier (x2/x3/x4 from streaks of 10/20/30); +2 x multiplier per
  40 ms while a hit long note (> 0.4 s) is held at its pitch; a miss ends the streak.
  Highway in a game: hit gems burst and go, missed ones grey, score / "x4 · streak 54
  · 100%" top right, the countdown in the middle. Locked: `seekTo` refuses while
  `game.on`; body.game-on greys out seek and ±10 s and hides the sidebar, subtitle,
  section switch, input bubble and every tool but Stop, so the highway gets the window
  (the user couldn't see all of it; its size now stays fixed through a game); the ×
  (close) is hidden too. Pause/Space
  or "Stop game" ends it (so does the song's end); results dialog: score, new best or
  best (per song + tab source, `track-splitter-game-best`), accuracy, notes hit of
  judged, best streak, "Stopped at m:ss" when not finished; Play again. Simulated
  Scarlett on Electra: in time 74/74 (100%, 26,498 at 0:46); 3 s off 23/35 (66%:
  the riff repeats).
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
  files with rhythm: overall start + tempo ratio search, then per-bar shifts along one
  smooth path (Viterbi over -2.5..+2.5 s in 50 ms steps; each bar scored on its and its
  neighbours' notes landing on heard starts plus 0.5 x lying inside a heard note of that
  pitch, since separated bass often comes out as long merged notes; 0.6 per 50 ms of
  change, 0.4/s for leaving the overall fit). It was +-0.4 s median-smoothed, which
  left Save Yourself, I'll Hold Them Back (MCR; user: "tab is all off, starts too early,
  way too fast") 1.3-1.7 s early for its first two minutes. In bars, a heard note may
  only nudge a written one (`NUDGE_MAX` 0.1 s or `NUDGE_SHARE` 35% of the gap to its
  neighbours): unlimited, wrong matches squeezed notes together (MCR 15% of gaps < 0.6x
  their written length; the user saw tabs "speeding up", and the game's speed floor
  went up to match). Re-synced, MCR's notes inside a heard note 67 -> 89%, squeezed
  0.4-1.3% on all 8 imported tabs, its quickest same-string gap 0.056 -> 0.141 s; the
  other songs' scores moved by a few points either way.
  Steady rhythm (user: "straight eighth/16th notes but they are spaced weird so it makes
  it hard to hit them"): even those nudges left 10-35% of steady 8th/16th pairs > 25%
  uneven inside bars and 16-47% across bar lines. Now no note is nudged on its own: each
  bar keeps the written rhythm and is shifted/stretched as a whole by a robust fit to its
  matched notes (`_bar_fit`: medians, <= 80 ms (`STEADY_MAX`), <= 8% stretch), and
  neighbouring bars meet at the bar line (one shift there, the mean of both). On the 5
  file tabs in the test library: 0% uneven inside and across bars; notes within 50 ms of
  a heard note 56.5-88.9% (written rhythm alone 55.8-86.4%, nudged 62.9-94.2%, partly
  the nudge moving notes onto noisy heard times). IMPORT_RULES 4.
  Shape (user: "Baby's tab starts way too early and Look What Happened starts way too
  late"): the tempo search was 0.85-1.15 and bars could only move +-2.5 s from one overall
  fit, so a tab and recording of different shapes went wrong at one end. Baby's tab is
  ~22 s longer than the recording's bass part (a part repeated more often) and its opening
  sat in the bassless intro; Look What Happened's recording drifts in tempo at the start
  and a third of its tab (E3 ghost notes) is never heard, and the fit put its first note at
  0:29. Now `_tempo_fit` searches 0.6-1.7 (1%, then 0.25%) and `_sync_timed` places every
  bar in one Viterbi pass over start times (50 ms grid): it follows the bar before (+-100
  ms, SYNC_DRIFT 0.6 a step), jumps forward (SYNC_JUMP 8: the recording has something the
  tab doesn't) or is left out (SYNC_SKIP 3 a bar: its notes are dropped from the synced
  tab, the file keeps them), scored by `_bar_scores` (hits + octave halves + half for
  inside a heard note). `_place_in_bars` takes each bar's own end (`bar_audio_end`) and
  joins shifts only where placed bars meet. On the user's 13 imported file tabs: Baby first
  note 20.6 s (bass enters 20.5), 10 of 89 bars left out, notes on a heard note 57 -> 77%;
  Look What Happened 0:29 -> 0:00.6 (bass from 0.6), 30 -> 34%; the other 11 the same
  within a point (Amsterdam drops one 2-note bar); notes always in order; ~1-1.6 s a tab.
  IMPORT_RULES 5 (opening an import re-syncs it once, ~5 s). `IMPORT_RULES` (app.py): bump
  when syncing changes; `get_tab` then re-syncs an old import from its saved file
  (`resync_import`, ~1-2 s) when the automatic tab is current. Text tabs:
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
- Import dialog (`#import-dlg.fancy-dlg`, same look as Add songs; user: "how about the
  tab uploader"): song in the subtitle, a drop zone for the file (click or drop; the
  hidden `#import-file` holds it; once chosen it shows as a card with × instead), "or
  paste a text tab", a monospace paste box. A file wins over pasted text.
- Page: "Import tab…" (file or paste box; also "Import bass tab…" in each song's ⋯ menu,
  which loads the song and opens its tab view behind the box: the user couldn't find
  the button inside the tab view), an Imported/Automatic switch (per song,
  imported by default), the tab's own tuning (raise ½ step still works if it's half
  step down), and a warning when < 35% of its notes matched (other version/tuning?).
- Other tunings (user: "can you put an imported tab in a different tuning", "if there's
  a better way to play in another tuning, do that"): the tuning picker works for
  imported tabs. The tab's own tuning is shown as written ("(as written)"); another
  one re-fingers the tab's pitches with the learned fingering (dead notes keep their
  string; notes below the lowest string are counted). `easiest_tunings()` compares
  every tuning (notes it can't reach, then 4-string before 5-string, then
  `tab._difficulty` of its fingering) and marks the easiest "(easiest)" + "easiest to
  play in …" in the subtitle; the tab's own tuning wins unless another is >= 25%
  easier (retuning is a bother). Cached in the import (`tuning_report`, per learned
  style). User's tabs: Cry Baby Cry and Electra drop D (~1/3 easier: the low riff on
  the dropped D and open A), YOLO and Wow! Wow! Wow! standard. Tuning choices are
  kept per song and per tab source (`song|imported`), so one tab's doesn't re-finger
  another's.
- Make repeats match (user: "find repeated parts in tabs that are the same but played
  differently and allow me to choose ... to all match to one"): More ▾ → "Make repeats
  match…" (imported tab, in its written tuning).
  - Finding (`repeat_groups()`): bars (from `lines`, 3+ played notes, dead ones left out)
    whose pitches mostly match (>= 75% in order, `_similar`: LCS; a pickup note or a
    changed last note still counts) join the first such bar's group (key = its pitches).
    Within a group, bars that put a shared pitch on another string are another way; a
    way = {pitch: (string, fret)} (`_bar_map`). Exact matches only missed Sha-La-La-Lady's
    A7 A7 A9 vs D2 D4 D4 endings (the user's screenshot); now 11 groups there.
  - Choosing: "Use everywhere" (`POST /api/tab/repeats` {song, key, map}) plays each
    pitch the way plays where it does, in every bar of the group; "Easiest in each place"
    ({key, easiest: true}; user: "the easier way to play based off the previous bar and
    following bar") gives each bar the way with the lowest `tab._difficulty` over that bar
    and its two neighbours (`fit_costs`, `best_here`: ties keep what the tab has),
    with the user's other choices applied to the neighbours (scored as written, a bar kept
    D2/D4 while the next, chosen A7/A9, jumped back). "Easiest" marks the way that's
    easiest used everywhere. Saved in the import's `unify` ({key: {pitch: [s, f]}} or
    "easiest"); older choices (a fingering per note of an exact pitch key) move to the group
    with a bar of that key (`unify_rules`) and are rewritten on the next save.
    `map: null` = back as written, no key = all back.
  - Dialog: each way drawn as a line of tab (`repeatLine`, SVG; user: "show the lines to
    compare"), notes where they fall in the bar (`times` 0-1 of the way's first bar),
    notes placed differently from the most common way lit; "Way n · bars · times"
    (clickable).
  - `unified_notes()` applies the choices (maps first, then "easiest" groups) wherever the
    import's notes are shown or copied (get_tab, drafts); the file and synced notes stay
    as they were, and `resync_import` carries `unify` over. Learning from imported tabs
    uses the choices too (user: "can we make it use my choices?"): `learn_from_imports`
    fingers from `unified_notes`, `learn_sources()` records a tab's `unify` with its time,
    and `set_repeats` queues learning (~2-3 min on the test library); a restart with
    nothing changed doesn't re-learn.
- Dead notes ("x") are left out of imported tabs when shown (user: "remove Xs in
  tabs"; Amsterdam's .gp had 223 of 868), so they're not in the views or the game;
  the saved import still has them (`notes[i][4]`).

## My tab: recording your own playing

The user plays along through a Focusrite Scarlett (bass in an instrument input, Inst on,
Direct Monitor to hear themselves) and gets their own tab, and can hear their playing
with the band.
- Page: "Bass input" in the tab view opens `#input-bar` (it replaced a record dialog;
  the user also wanted to see what's coming through without recording, and to hear
  themselves): input (an input named Scarlett/Focusrite is picked first; input,
  channel and monitor choices kept in localStorage), channel (Input 1/2 when stereo),
  level meter (tooltip: too loud / coming in / nothing), the live note, "Hear myself",
  Record. Closing the tab view closes the bar.
  - Layout (user: "bass input goes off screen, can we clean up the settings", then
    "make it some sort of drawer that pulls up on the bottom"): `#input-bar` is a drawer
    fixed above the player bar, centred over the main area (`--side-w`, 0 when the
    sidebar collapses), sliding up on open; columns Input (device/channel, level, the
    note), Listen (Hear myself + volume), Tools (Tuner, Calibrate strings…), Record into
    your draft (Add notes / Replace, Record); two columns under 1000 px; the tuner opens
    full width below; scrolls if the window is short. The Bass input button has one
    fixed width (9.2em): its text changing ("● A1", "Listening") used to wrap the tools
    row and move the button. Game settings likewise grouped: Game (Play on; Board only when
    playing on the board; Difficulty, Note speed, Timing), Sound, Setup (Calibrate
    timing…, Calibrate strings…), scrolling on short windows.
  - Tuner (user: "add a tuner in the sound popup"): Tuner in the Bass input bubble opens
    a chromatic tuner from the same readings (YIN, median of the last 5): note, Hz, a
    needle -50..+50 cents (green within 5), "n cents flat: tune up", and the strings of
    the bass as tuned now with the matching one lit. Simulated A 10 cents flat read
    "10 cents flat", then "In tune".
  - Ghost notes (user: "a ton of ghost notes, even with my hands off the bass"): mains hum
    (60 Hz falls between A#1 and B1) and noise were read as notes. Now: a noise gate 10 dB
    above the input's own quiet level (`input.floorDb`: 20th percentile of the last ~10 s
    of readings with no pluck in the 1.5 s before; at least -50 dBFS); a reading must be
    within 40 cents of a note (hum is ~50 off); the readout only takes a new note after a
    pluck in the 0.7 s before (`input.shownMidi`). The tuner still shows out-of-tune
    pitches. Simulated 60 Hz hum (+ harmonics, ~-30 dBFS) and hiss: 0 notes in 9 s;
    Electra's bass over them in the game: 29/29 after the start, streak unbroken.
  - Dropouts (user: "bass randomly cut off, I had to change my input and back"): when
    the input track ends, or stays muted for 1.5 s, the same input is opened again
    (`reopenInput`, with a toast), and the player's audio clock is resumed if it gets
    suspended while the input is on.
  - Inputs: Both inputs (default, `track-splitter-record-channel`) / Input 1 / Input 2,
    always shown; Both sums the interface's two inputs (`connectChannel`, the recorder
    worklet too), so an instrument is heard whichever input it's in. The user's brother
    (guitar) got nothing: the choice only appeared when the browser reported 2 channels,
    else Input 1 was used. When a browser reports one channel the drawer says only Input
    1 can be heard (use Chrome or Input 1).
  - Bass / Guitar (drawer's Input row, `track-splitter-instrument`; user: "add a Bass /
    Guitar choice" for his brother): `PITCH_RANGE` bass 28-420 Hz, guitar 70-1400 Hz
    (bass mode read a guitar's C5/E5/B5 an octave low); the readout and tuner use the
    guitar's strings (`GUITAR`, E A D G B E, frets to 24); `samePitch` lets a note in
    another octave count in the game, Show my timing and wrong-note checks (a guitar
    plays a bass part an octave or two up); the board/Highway ring only shows on a
    matching tab note (`input.lane`; `input.laneFret`, `input.placeLabel` for the fret
    shown); string detection is off (its calibration is the bass). Labels say "Guitar
    input". Simulated guitar tones E2-B5 all read right; the guitar tuner read B 10
    cents sharp.
  - Opening (user hit "Couldn't open the audio input"): the remembered device id can go
    stale (browsers re-issue ids after restarts, replugging, cleared site data); then
    the default input is opened and the Scarlett found by name. Errors are explained
    per kind (macOS microphone permission, nothing plugged in, input busy, gone) and
    logged to the console.
  - Live note: every 40 ms the analyser's last ~85 ms (4096 samples) go through YIN
    (`pitchOf`, signal averaged down to ~11 kHz, 28-420 Hz, threshold 0.15); a note
    shows once 2 of the last 3 readings agree. Shown with where it can be fretted (in
    the shown tuning; standard strings when the song is raised) and against the tab's
    notes sounding now (+-120 ms, `tabNotesNow`, pitches from each tuning's "open"
    MIDI notes in /api/tab): "✓ matches the tab" (green), "right note, other octave",
    or "tab: A7" (red). The Highway draws a ring on the hit line in that note's lane
    (the tab note's string if it matches), green on a match. Simulated Scarlett
    (Electra's bass track played along): 59 of 61 readings with a note matched; the 2
    others were at a note change.
  - Hear myself: the input to the speakers on a separate AudioContext with latencyHint
    0 (the smallest buffer; was 'interactive') at the input's sample rate, and the input
    opened with `latency: 0` (user: "lag from when I play and hear my bass"); the
    player's 'playback' context is ~90 ms. The drawer shows the delay ("~13 ms · Direct
    Monitor: none" in the test browser); the interface's Direct Monitor has none.
    Volume slider 0 to +40 dB (default +6, kept per computer): the user found it too
    quiet unless they turned the interface gain up, which they don't want to do (+24 was
    still "hard to hear"; their DI peaks around -25 dBFS). After the volume, a soft
    clipper (WaveShaper, no oversampling, so no delay; 1/4 in front so it takes peaks to
    4x full scale): linear to 0.7, rounding off to 1, so loud notes at a big boost don't
    crackle. Tested: 0.1 in -> 0.1 out on the same sample, overs capped at 1.
  - Match the song's tone (user: "automatically pick up the sound of the bass in a song
    and make that as an effect so you have the same tone"): a switch under Hear myself
    in the drawer and in Game settings (Sound), on per song (`track-splitter-tone`, a
    map of song ids; switching it on turns Hear myself on). `GET /api/tone?song&rate&kind`
    compares the song's "bass only" track (HQ preferred; for guitar "guitar only", else
    lead, else rhythm: `part_track`) with the user's instrument and returns
    filters fitted to BiquadFilterNode's own formulas at the monitor's sample rate;
    `applyTone`/`wireTone` put them (and a loudness trim) between the monitor's input
    and its volume. Method (tone.py docstring): long-term 1/3-octave spectrum of the
    playing frames, curve = song - DI, +-12 dB, flat outside 40 Hz-8 kHz, a low shelf, 15
    half-octave peaking filters and a high shelf (17 biquads), same loudness as heard
    (A-weighted: balancing raw energy, mostly lows, left it ~3 dB quieter by ear, up to
    5.7, since the EQ cuts mids; now 0.0 dB on every song). On the 14 library songs vs
    the user's 8 DI recordings: shape off from the song 6.2 dB rms (median; 4.8-9.5)
    before, 1.1 with it (the worst where the ±12 dB limit bites); fit within 0.2-1.0 dB of the curve; 0.3-0.5 s to measure a song the first time,
    then cached. In the page the built chain matched the fitted response within 0.1 dB,
    an impulse came out at sample 0 (no added delay; IIR filters have only the usual
    minimum-phase lag at low frequencies) and the drawer's latency readout was the same
    on and off (~13 ms in the test browser). The DI side: a capture is 45 s of playing
    (recorder worklet, `POST /api/tone/capture`, needs 15 s of playing; 20 s varied up
    to ~7 dB, 45 s ~4), since the user's DI changes (two takes were ~20 dB brighter at
    8 kHz: pick or tone knob).
    Instruments (user: "I have 3 different basses that all sound different, would be nice
    to pick which I am playing", "a settings menu where you can manage this", "it should
    work for any instrument"): each capture is saved with a name and a type (bass /
    guitar) in `<library>/.instrument-tones.json`, one picked per type; the Bass / Guitar
    switch decides which type applies (and which part of the song it's matched with).
    Picker + "Instruments…" in the drawer, "Instrument" + "Manage…" in Game settings; the
    Your instruments dialog (`#inst-dlg`, `renderInstDialog`) lists them (Use, Capture
    again, Rename inline, Remove: kept in the file under "removed") and adds new ones
    (name, Bass/Guitar, Capture 45 s; the countdown and Stop on that button). For bass,
    "My recordings" (every DI recording in the library pooled: `.bass-recordings`, every
    `.my-bass` take) is listed too and used until a bass is captured. The first version's
    single capture (`.bass-di-tone.json`) is brought in as "My bass". Tested: a guitar
    capture fed from The Eternal's lead guitar track matched that track within ±1 dB. A capture in
    the page of a song's bass stretch matched that stretch analysed offline within
    ~0.3 dB. Not matched: compression (a no-look-ahead compressor didn't bring the DI's
    level spread toward the songs' at any setting; the gap is playing, not processing)
    and distortion (can't be told from leakage in a stem). Before/after spectra were
    plotted for The Eternal, American Idiot, You Only Live Once and Electra. Tested with a
    simulated Scarlett only; not yet heard on the real bass. `TONE_RULES` in app.py: bump
    when `tone.profile` changes.
  - Quiet input: the meter is in dB (-60 to 0 dBFS; tooltip gives the level) and pitch
    detection listens from -50 dBFS (YIN doesn't depend on loudness). Simulated input
    at -34 dB (~-44 dB peaks while playing): 44 of 50 readings found, all matched.
  - Record: the song jumps back 2 s and plays; the drawer tucks away so the tab stays in
    view (user: "when you hit record, the modal blocks the tab") and the Bass input button
    reads "Rec 0:12" (pause, Space or reopening the drawer's Stop ends it). The input
    runs on the player's own AudioContext; an AudioWorklet (inline, Blob URL) posts
    each block with its frame number, so every sample has a time on the song's clock.
    The take ends on Stop, pause, a jump elsewhere, or the song's end; it's sent as a
    mono float WAV with `song_start` = song position at its first sample minus the
    output delay and the input's reported latency, and the player's `shift`. The bar
    stays open afterwards.
- Server: `POST /api/tab/record` queues a "record" job -> `record_tab()`: transcribe the
  take as is (tab.NOTES: a clean bass has no leakage, so no learned settings/models),
  `align_take()` matches its attacks (log spectral flux, z-scored) to the bass track
  (or the song) within +-0.4 s to fix what the page can't measure; pitches minus
  `shift`; the take replaces notes in its stretch, so sections can be redone. Saved in
  `.bass-tab-mine.json` (notes [start, end, pitch], takes with start/end/shift/audio).
  A quiet take is brought up (99.9th percentile to 0.5, at most x1000) before
  transcribing: a -52 dB take still gave all 31 tab notes of its stretch
  and each take's audio in `.my-bass/take-<ms>.flac`. `GET /api/tab?source=mine`
  fingers it in any tuning (learned fingering) with the automatic tab's lines; a
  missing source falls back imported -> mine -> auto. `POST /api/tab/mine/remove`
  trashes both.
- Drafts (user: "drafts of tabs, then play my bass over an imported tab and add notes
  to it"): My tab is shown as "Draft". More ▾ → "Start a draft from this tab"
  (`POST /api/tab/draft` {song, source imported|auto, split}) copies that tab's notes
  into `.bass-tab-mine.json` as `base` (+ the imported tab's own `lines` and `tuning`,
  `from`, `from_name`); an existing draft and its takes go to the Trash (asked first).
  Takes have a `mode` (Bass input bubble: Add notes (default, `track-splitter-
  record-mode`) / Replace): `apply_take()` "add" keeps what's there and only fills
  gaps: notes within 0.25 s of an existing one (any pitch) are left out and added ones
  snap to their bar's 16th grid (within 60 ms; `bar_starts` from the draft's lines).
  It used to add anything not within 70 ms of a same-pitch note; a take slightly off the
  tab put notes between the tab's (Amsterdam 61 -> 103 notes in 18 s; user: "it speeds
  it way up, keep the same speed and structure"); MINE_RULES 3 rebuilt drafts.
  That then blocked real additions (user: "trying to add more 6s in Amsterdam at 45
  seconds": their extra E♭2 8ths, A6 in standard, between the tab's notes were all
  dropped; 3 takes added 5 notes). Now (`apply_take`, MINE_RULES 4): a played note goes
  into its bar's 16th slot when the slot is empty, unless the same pitch is in the next
  slot (a copy played a little off), and only if the transcription is sure of it
  (amplitude >= `ADD_MIN_AMP` 0.55: a near-silent take's notes, even the tab's, were
  0.33-0.53; the added E♭2s 0.67-0.80). Those takes now add 39, the E♭2s at 0:47-0:50 in;
  take notes carry their amplitude for this. The user's two newest takes were ~14 dB
  quieter than the first (peaks -36/-32 dBFS, mostly ~-51): the recording is the raw
  input, so check the interface gain and the chosen input.
  Recording in Add notes with no draft starts one from the imported tab (else the
  automatic one), as "Start a draft from this tab" would (`new_draft`, shared with
  `start_draft`): the user's draft had been deleted (More → Delete draft, no question
  asked) and their next Add take became the whole draft (225 notes, no tab: "it added them
  but got rid of a bunch of stuff too"). Delete draft now asks first, with the number of
  takes. Their old draft (base + 3 takes) was put back from a test copy with the new take
  added: 720 notes, all 645 of the tab's kept. Removed
  notes played again come back,
  "replace" takes over the stretch. `redo_mine` rebuilds from `base` then each take in
  its mode. Test (simulated Scarlett, Electra): draft 1010 notes, 3 deleted -> 1007,
  an Add take over that stretch -> 1012 (the 3 back, nothing doubled, 2 extra).
- Review before saving (user: "a way to redo a section and then save my changes instead
  of auto saving, and maybe display what was added and an option to redo sections that
  were added"): a finished take waits in the draft as `pending` (`record_tab`; another take
  meanwhile replaces it, its audio to the Trash) until `POST /api/tab/mine/take` {keep:
  true/false} (`keep_take`). `GET /api/tab?source=mine` shows the draft as it would be:
  each note's 6th value says where it came from ("tab", "take:<n>", "new" = the pending
  take adds it), the notes it would replace come apart in `gone`; `pending` has the
  stretch, mode, +added / -replaced, and a check: how much of what was heard matches the
  draft there (same note within 0.1 s or an octave) and the peak level. Page: a review bar
  over the tab (`#take-review`: New take / Redo of take n, 0:32-1:12, Add notes, +15
  notes; Play it, Redo, Discard, Keep), new notes green, replaced ones red and struck above
  their string; a warning when < 20% matches ("is your bass on the input chosen ... gain
  up?"; quiet alone isn't flagged: a good take peaked at -36 dB and matched 88%).
  Where notes came from is kept as a 4th value in the draft's notes (the take's "made";
  0 = the tab), written by `apply_take(tag=...)` and `redo_mine` (MINE_RULES 5 rebuilt
  drafts to get it). View → "Show what I added" (`track-splitter-show-added`, on) colours
  kept takes' notes blue. More → "Your takes…" (`#takes-dlg`, `take_list`: n, stretch,
  mode, notes in the draft from it): Go there, Redo (`redoSection`: Bass input opened if
  needed, the take's mode, records from 2 s before and stops itself 0.3 s after its end;
  sent with `replaces`, so on Keep the old take and its audio go and the draft is made
  again), Remove (asks; `remove_take`: audio to the Trash, draft made again). Tested on a
  copy of the user's Amsterdam draft: preview, Discard, Keep, Replace (+132 / -189), Redo
  of take 2 (replaced, old audio trashed), Remove; audio files always matched the takes.
  The user's two newest takes (22:28 and before) came in at ~-33 dBFS and matched the tab
  3-5% (mostly a steady E4): not their bass on the chosen input; ADD_MIN_AMP 0.55 left
  them out, a cut-off relative to each take let their noise in, so it stays fixed.
- Ghost notes (user: recorded tabs are more accurate than automatic ones, but have
  ghost notes). The user's Cry Baby Cry take vs their imported tab: 289 of 335 notes in
  the tab; of the 46 others, 20 were same-note restarts with real attacks (stronger
  than real re-plucks: probably played, or the tab simplifies; not filtered) and most
  of the rest E1/E3 around a played E2 with no attack: the open low E ringing along
  and overtones. `tab.drop_ringing_octaves` (in `take_notes`): drop a note an octave or
  two from one played in the last 0.5 s when the bass gets < 1 dB louder there: 47 ->
  36 notes not in the tab, 288 -> 287 real. `MINE_RULES` in app.py: when take_notes
  changes, recorded tabs are redone from their takes' audio on the next open
  (`redo_mine`, ~1 s; keeps the lining up and the user's removals).
- Removing notes: in the Mine tab's Lines view a note is clickable (red when
  selected); "Delete <string><fret> at m:ss" or the Delete/Backspace key removes it,
  Escape deselects, Undo puts the last one back. `POST /api/tab/mine/note` ({song,
  start, pitch}; `?restore=true` with end to put back) keeps a `removed` list
  ([start, pitch], matched within 60 ms) that redone takes respect; a new take over a
  stretch clears removals there.
- Playback: the player's 4th mode "My bass" (shown when `song.my_bass`) = without +
  `/api/audio?part=mine` (`my_bass()`: all takes on the song timeline as one float WAV,
  later takes win, pitch-moved if recorded at another `shift`; each take scaled to the
  record's own bass there by `playing_level()` (90th percentile of 50 ms RMS, so rests
  don't count; 0.1 without a bass track), so a quiet take plays about as loud as the
  band's bass (-18 vs -15 dB measured); cached in SHIFT_DIR).
  The player's mode buttons are `#hear-mode` (they used to be any `.segmented`, so the
  tab view's switches changed the player's mode).
- Tested with a simulated Scarlett in the browser (Electra's bass-only track as the
  input; each getUserMedia call needs its own MediaStreamDestination, since cloned
  tracks of one ended together): 23/23 notes of the imported tab, timing within 23 ms;
  a stand-in take sent 0.25 s off was realigned to 6 ms. The user tried the real
  Scarlett and said it seems to work.

## Guitar chords (input drawer)

The user asked for chord recognition for guitar input. It uses the same Bass | Guitar
switch as the guitar mode above (`instrument`, `track-splitter-instrument`; the chords
branch had its own switch, `track-splitter-input-instrument`, merged into that one).
Guitar mode also: the note readout shows the chord ("Am7" / "A C E G", also on the input
button), or a single note with where it is on a standard-tuned 6-string ("A3 · G2 · D7 ·
A12"); the tuner shows E A D G B e; Record and Calibrate strings are hidden (bass only).
The game and Show my timing still use the guitar mode's octave matching (`samePitch`).
- `heardNotes` (a second analyser, 8192 samples, 16384 above 50 kHz): iterative
  harmonic multi-pitch. FFT peaks within 45 dB of the loudest; each peak 75-1200 Hz is a
  possible fundamental scored by the peaks at its first 10 harmonics (2.5%, ^0.6
  amplitudes, 1/h^0.25); the best becomes a note and its peaks are used up; repeat until
  < 18% of the first (max 6). Octave doublings vanish (harmless: only pitch classes and
  the lowest note matter). A note-template NMF version was tried and was worse on real
  guitar notes (50% vs 70% single readings).
- `chordOf`: pitch classes from 0.2 of the strongest, scored against 16 shapes (maj,
  m, 5, 7, maj7, m7, sus4, sus2, dim, aug, 6, m6, add9, m7♭5, dim7, 7sus4): +weight per
  chord tone, -0.7 per missing one (fifth -0.15), -weight per extra note, +0.35 when the
  lowest note is the root, a small cost per shape so simpler names win; added notes
  (6/7/9) need 0.3. Lowest note a chord tone other than the root -> slash chord.
- `chordTracker`: strums found like plucks (newest 20 ms 4 dB over the readings
  before, from -60 dB; guitar is quieter than bass), a reading 0.14 s after a strum then
  every 0.1 s, using only the sound since the strum (85-170 ms) and averaging each
  note's salience since. Shown when 2 readings agree (3 for 4-note chords or the chord
  shown plus a note). A new chord needs 2 notes louder than just before the strum, or
  the same chord as the strum before (stops a note over ringing strings becoming a
  chord). The chord shown stays through re-strums, upstrokes and dying away; a slash
  chord change needs a lower bass note, or the old one stopped. Cleared below -66 dB.
- Tested (scratch, not in the repo) on chords built from recorded guitar notes
  (tonejs-instruments acoustic + electric samples, shifted by resampling; random strum
  speed/direction, detune, levels; 44.1/48 kHz): 50 chord shapes, one strum: right 95%
  of 0.1 s frames from 0.3 s on; 4-chord progressions 96% (median 0.37 s to show);
  strummed down-up 8ths 83%; one-note melodies never called chords (let-ring melodies
  over held strings sometimes are, arguably right); same at -44 dB peaks, 91% at -54.
  In the page (headless Chromium, simulated Scarlett playing a rendered take): every
  chord named, no page errors. Not yet tried on the user's real guitar.
- Weak spots: synthesized (Karplus-Strong) strings were misleading, so tests use real
  samples; sustained strumming sometimes flashes an add9/6 from the attack; acoustic
  body resonance can put a false low bass note in (Dsus2 heard as Asus4).

## UI (static/index.html)

- Sidebar: All songs, artists (hover ... -> rename/merge, delete; + New artist; drop a
  song row on an artist to move it; clicking one also closes the tab view so the list
  shows, user: "show it no matter where I am"; not during a game), device label, Open library folder, Create app
  shortcut, Quit.
- Main: searchable song list; each row shows version chips (click loads it paused, user: "not autoplay"; click again or Space plays; right-click → Play starts it) and
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
- Skip silence (user: "detect if there's dead space or silence at the beginning or end of
  a song and shorten it", "look at Baby"): measured on the library, ends had 1-4 s, Baby
  ~18, American Confidence 9, American Idiot 7; starts 0-1 s (You Only Live Once 2.4). The
  file isn't cut (tabs, imports, drafts and game checks are timed from its start): the
  player finds the sound (`soundBounds`: 50 ms RMS of without + only above -60 dBFS and
  within 50 dB of the song's 90th percentile, 0.5 s kept either side) when a song loads;
  `play()` starts at the first sound when it would start before it and stops the sources
  after the last, which runs the song-end path (back to 0:00, a game ends with results).
  The volume popover has "Skip silence at the start and end" (`track-splitter-skip-silence`,
  on) and says what this song skips ("17.5 s at the end skipped"). Tested on Baby: playing
  from 3:00 stopped at ~3:03 instead of 3:20; switched off, it played on.
  Baby's first 11 s aren't silent, they're quiet (30-40 dB under the song, a faint tone
  ~1.4 kHz), so 50 dB keeps them; 30 dB would find Baby's 0:11.8 but also cut real quiet
  intros (Fireman & Dancer 0:08, Save Yourself 0:05.5, Amsterdam 0:12 and 28 s of its end).
  So the automatic skip stays at 50 dB and the user can trim by hand (user: "a way for me to
  trim the songs myself"; asked whether as a cut version or kept with the song, they chose
  the song): `POST /api/song/trim` {song, start, end} (none = whole song) -> `.song.json`
  "trim", sent in the library as `song.trim`; `playRange()` uses it before the automatic
  skip, so every version and the game play between those points and tabs stay aligned.
  Trim dialog (`#trim-dlg`, `openTrim`; the song's ⋯ menu "Trim start and end…", marked
  "(trimmed)", "Trim this song…" in the volume popover, and "Trim song…" in the Bass input
  drawer's recording column with the trim shown under it; user: "make it also available in
  the recording section"): the whole song's waveform
  (without + only, min/max per pixel) with two draggable handles, close-ups 6 s around each
  (drag there for fine moves), times typed as m:ss.s or nudged 0.1 s, "Play from start" /
  "Play up to end" (4 s through the player's gains), "Use the whole song", Save. It opens on
  the saved trim, else the automatic points. Tested on Baby: start 0:11.5 saved, play from
  the top began there, the popover said "Trimmed: plays 0:11.5 to 3:03.4"; whole song
  cleared it.
- Volumes (user: "change that to have 3, a master volume, a song volume, and an instrument
  volume"): the player's slider is the master; its speaker opens a popover (`#mixer`) with
  Master, Song and Instrument. Master scales the song (`state.volume * state.songVolume *
  headroom`) and the Hear myself monitor (`monitorGain()`, its own AudioContext); Instrument
  is the Hear myself boost (`#input-volume`, 0 to +40 dB). Master and song are kept per
  computer (`track-splitter-master-volume`, `-song-volume`); Game settings has all three.
  Under 1100 px only the speaker shows.
- Instrument picker in the player (user: "select what instrument you are playing and then
  a dropdown updates with your saved instruments"): Bass | Guitar (`#player-kind`, the same
  `setInstrument` as the drawer's switch) and that type's saved instruments (`#player-inst`,
  "Default" when there are none, "Manage instruments…" opens Your instruments).
  Guitar is disabled there and in Your instruments' type choice for now (user: "my
  brother is handling guitar, so leave the option disabled"); the server side and the
  drawer's Bass / Guitar input switch (the guitar input mode he uses) are unchanged.
- Add songs dialog: drop files or paste a link, edit title/artist per item, pick
  instruments (guitar is All guitar / Lead guitar / Rhythm guitar), level, HQ, and
  "Faster guitar/piano" (shown only with HQ + guitar/piano). Upload is disabled until the required fields are filled. On
  success the songs clear and the dialog closes; settings persist (localStorage).
  Styled to match the newer screens (user: "update the file uploader to match the rest
  of the UI"; CSS for `.fancy-dlg` at the end of the stylesheet, shared with Import a
  tab): violet glow,
  title with a gradient badge and subtitle, glowing drop zone, link field with an icon,
  item cards with a note/link icon, small uppercase section labels, selected
  instruments as gradient pills, level + HQ in one panel, pill buttons. Its switches
  are laid out as rows there (`.field > label` made them blocks).
- Dark theme, violet accent, system font; fits one screen down to ~1024x768 and
  collapses for narrow windows.

## Launching without a terminal

- The app asks before creating a shortcut (welcome card, "Don't ask me again" stored in
  localStorage). Mac: `~/Applications/Track Splitter.app` whose launcher runs `start.sh`
  in the background (log: `~/Library/Logs/Track Splitter.log`) or reopens the page if
  already running. Windows: Desktop `.lnk` running `pythonw.exe app.py` (does not
  reinstall packages; untested on a real desktop).
- Updates while running: the server is the code it started with, so a page reload
  doesn't pick up changes to app.py/tab.py/tabimport.py (the user's tuning picker
  "didn't switch" because the running app predated it). `/api/info` returns
  `restart_needed` when those files changed since start (`CODE_AT_START`), and the
  sidebar then shows "The app was updated: click Quit below, then open it again"
  (checked every 30 s). Page-only changes (index.html) just need a reload.
- Starting a second copy just opens the browser. Quit in the sidebar stops the server
  (`POST /api/quit`).
- The static files are mounted last in `app.py`; any new `/api/...` route must be
  defined before `api.mount("/", ...)` or it will 404.

## Testing approach used so far

No test suite in the repo. Changes were verified by running the server on a spare port
(`app.main(port=7866, open_browser=False)`, with `app.OUTPUT_DIR` set to a copy of the
library; when several sessions test at once each needs its own port: two shared 7866 and
one's page tests ran against the other's server and library), driving the API with small scripts
(upload, poll, move, delete, bad ids), and clicking through the page in a browser. A
synthetic 15 s clip (sine bass line + chords + clicks) was used, measuring how much
40-100 Hz energy remains to confirm bass removal.
The bass input was tested with a simulated Scarlett: in the page, `getUserMedia` is
replaced by a MediaStreamDestination (a fresh one per call) playing a song's "bass only"
track, or the user's own recordings; `track-splitter-game-start` starts a game later in
the song. String detection was measured on two recordings of the user's bass (slow and
fast takes, each position labelled), made with a small local recorder page.

## Open ideas / to do

- Windows polish: make the
  desktop shortcut install package updates like the Mac one, use an NVIDIA GPU (needs
  the CUDA build of torch; the default Windows build is CPU-only).
- Playlists / practice sets that pull songs from any artist.
- The user once asked for "filters": interpreted as folders; filtering the song list
  (by instrument removed, HQ, etc.) may still be wanted.
- Mac-only paths (app shortcut, MPS fallback, certificate fix) were written on Windows;
  they have been reported working on Mac but have no automated checks.
