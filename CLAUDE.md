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
  Slow-Fast slider (0.4-1.6 x 3 s ahead, 120-420 px/s; `track-splitter-highway-speed`)
  replaced the speed-from-the-quickest-notes rule above. (Crowded gems used to get
  narrower; now every gem in a song is one size, see "Gem size".)
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
- Timing (user: "wiggle room"): Relaxed (default) / Normal / Tight in the Game section,
  hit windows early/late/Perfect 0.25/0.3/0.1, 0.15/0.2/0.07, 0.1/0.12/0.05 s
  (`TIMINGS`, `track-splitter-game-timing`); best scores per song + tab + timing. A
  player 0.2 s late: Relaxed 7/7, Tight 1/7.
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
- Hand thuds (user: "my hand hits the string and it picks it up as a note"): live
  detection now needs a clearly pitched reading (YIN dip < 0.1) and 3 readings in a
  row (~120 ms) to agree (was 2 of 3). Test: 16 damped bass-pitched thuds and noise
  bursts gave no note; a perfect game run still scored 83/84.
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
    'interactive' (~16 ms in Chromium here; the player's 'playback' context is ~90 ms).
    The switch's tooltip gives the delay and points to the interface's Direct Monitor.
    Volume slider 0 to +24 dB (default +6, kept per computer): the user found it too
    quiet unless they turned the interface gain up, which they don't want to do.
  - Quiet input: the meter is in dB (-60 to 0 dBFS; tooltip gives the level) and pitch
    detection listens from -50 dBFS (YIN doesn't depend on loudness). Simulated input
    at -34 dB (~-44 dB peaks while playing): 44 of 50 readings found, all matched.
  - Record: the song jumps back 2 s and plays; the button reads "Stop 0:12". The input
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
  record-mode`) / Replace): `apply_take()` "add" keeps what's there and adds notes not
  already in it (same pitch within 70 ms; removed notes played again come back),
  "replace" takes over the stretch. `redo_mine` rebuilds from `base` then each take in
  its mode. Test (simulated Scarlett, Electra): draft 1010 notes, 3 deleted -> 1007,
  an Add take over that stretch -> 1012 (the 3 back, nothing doubled, 2 extra).
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
(`app.main(port=7866, open_browser=False)`), driving the API with small scripts
(upload, poll, move, delete, bad ids), and clicking through the page in a browser. A
synthetic 15 s clip (sine bass line + chords + clicks) was used, measuring how much
40-100 Hz energy remains to confirm bass removal.

## Open ideas / to do

- Windows polish: make the
  desktop shortcut install package updates like the Mac one, use an NVIDIA GPU (needs
  the CUDA build of torch; the default Windows build is CPU-only).
- Playlists / practice sets that pull songs from any artist.
- The user once asked for "filters": interpreted as folders; filtering the song list
  (by instrument removed, HQ, etc.) may still be wanted.
- Mac-only paths (app shortcut, MPS fallback, certificate fix) were written on Windows;
  they have been reported working on Mac but have no automated checks.
