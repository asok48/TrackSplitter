# Track Splitter

A small local app that takes instruments out of songs, so a musician can play along
(the original use case: remove the bass to practice bass). Upload songs or paste a
link, pick what to remove (bass, drums, vocals, guitar, piano), and it produces a
version without those parts plus the removed part on its own. Everything runs on the
user's own computer; no audio leaves the machine.

Primary user runs it on a MacBook Pro (M4); it was developed and tested on Windows
(CPU only). It has also been run successfully on a second, fresh Mac from git.

**Before changing an area, read its section in `NOTES.md`** (same headings): the
history, measurements, user requests and specifics behind each feature live there.
This file is the short version loaded every session; add new detail to `NOTES.md` and
only a line or two here.

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
- After a change, load the page and check for console errors and that the main screens
  (song list, player bar, tab view, game) appear, using text/DOM checks (read_page,
  get_page_text, JS) rather than screenshots. Take screenshots only for visual/layout
  changes or when the user asks; for page-only tweaks a quick error check is enough.
  Why: a stray closing tag once pushed the player bar off screen, and removing the
  Guitar Hero look took the 3D/Flat setting's code with it. `drawHighway` / `drawGame`
  return early when their canvas is under 60 px (at ~0 height while the view opened,
  sizes went negative and `arc` threw, which broke the page when opening a tab).
- Mac-only paths (app shortcut, MPS fallback, certificate fix) were written on Windows;
  say when something is untested on a Mac.
- Keep this file slim (every session loads it; it was 121 KB before the split): when
  documenting a change, put the history, measurements, user quotes and test results in
  that area's `NOTES.md` section, and add or edit at most a line or two here (a new code
  name, setting, `*_RULES` constant, route or ground rule). Keep it under ~25 KB
  (`wc -c CLAUDE.md`); if it grows past that, move detail to `NOTES.md` word for word.

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

- Demucs v4. Standard = `htdemucs`, High quality = `htdemucs_ft` (`stem_model()` runs
  only the needed specialist(s) of its bag of 4: same output, ~4x faster). Guitar and
  piano: Standard = `htdemucs_6s`; High quality = BS-Roformer-SW (`run_roformer()`,
  ~700 MB checkpoint downloaded once, sha256-checked; ~real time on an M4; fp16 on GPU;
  "Faster" (`fast`) = 1 s chunk overlap, ~2x). The user wants both models kept.
- Lead / rhythm guitar: `center_of()` splits the guitar stem by stereo position
  (center = lead, sides = rhythm); ~-10 dB bleed is about the ceiling; poor on mono.
- "Without" = `original - removed_parts * (1 - level)`; "Only" = the removed parts;
  together they equal the original (Original mode's fallback).
- Device: CUDA, else MPS (`PYTORCH_ENABLE_MPS_FALLBACK=1`; `run_model()` retries on CPU),
  else CPU. Audio through imageio-ffmpeg's binary; output 320 kbps MP3, 44.1 kHz.
  Progress from `apply_model(callback=...)`.

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

Automatic tab per song: `POST /api/tab` queues a "tab" job; `make_tab()` reuses a
"(bass only)" track (HQ first) or isolates bass, transcribes (Basic Pitch via ONNX,
leakage/octave/re-pluck fixes), finds bars (tempo, beat grid, downbeat; "Beat 1 ◀ ▶"
shifts them, `POST /api/tab/bars`, `bar_shift`), fingers (hand-position Viterbi; tunings
standard, drop D, half step down, 5-string, with a suggestion) and saves
`.bass-tab.json`. `GET /api/tab?song&tuning` returns lines and notes.
- Version constants in app.py: bump `NOTES_RULES` (notes), `BARS_RULES`, `TUNING_RULES`,
  `LEARN_RULES`, `IMPORT_RULES`, `MINE_RULES`, `TONE_RULES` when that output changes, so
  saved data is redone or flagged.
- Learning from imported tabs (`learn_from_imports()`, `<library>/.learned-from-tabs.json`):
  fingering costs (`tab.learn_style`), note settings (`tab.NOTES` / `tab.NOTE_CHOICES`)
  and note models (`tab.learn_note_models`), each used only if it beats the built-in on
  held-out data. Runs in its own low-priority process (`learner`, `_learn_in_child`,
  `LEARN_DELAY` 20 s after start), not the job queue; status via `GET /api/learning`.
- Page views: Lines (current + next line, `buildSlots`; Bars per line, `#tab-bpl`,
  per song `track-splitter-bars-per-line`, Auto = `layoutLines`), Scroll (`drawTabScroll`),
  Highway (`drawHighway`, Flat switch), Whole tab (`wholeFrame`). Playhead follows
  `audioNow()` minus output latency; Line timing (`track-splitter-line-sync`); Raise ½
  step (`/api/audio?shift=1`, `track-splitter-raise`). Show my timing
  (`practiceListen`, `track-splitter-show-timing`) works in every view.
- Game section: `startGame` / `endGame`, `drawGame` (Rock Band look, 3D/Flat board
  `track-splitter-game-board`, play on Board/Tab `track-splitter-game-view`), difficulty
  (`gameNotesFor`), timing windows (`TIMINGS`, `MISS_AFTER`, `PITCH_WAIT`), plays the
  no-bass version, pause/settings dialogs (`#pause-dlg`, `#settings-dlg`), best scores
  (`bestKey`). Judging (`judge`, `gameListen`): a pluck (attack, `plucked`, `clicked`,
  `rePluck`, soft pitch-change) plus the note's own pitch read after it; messy notes,
  wrong notes (`checkWrong`), too-soft plucks, tuning check (`tuneCheck`). Game checks
  record runs to `<library>/.game-checks/` for `static/check.html`.
- Bass input listening (`listen`, YIN `pitchOf`, noise gate `input.floorDb`), string
  detection (`stringFeatures`, `#strcal-dlg`) and timing calibration (`#calib-dlg`,
  `inputCalibration`); both calibrations also saved to `<library>/.calibration.json`.
- Test start: localStorage `track-splitter-game-start` (seconds); Electra tests use 30.

## Imported tabs (tabimport.py)

The user brings a file (Guitar Pro .gp3/4/5 via PyGuitarPro, .gp 7/8, MusicXML) or
pasted text (incl. Ultimate Guitar count lines); never fetched from tab sites. Every
format -> TabTrack -> `timeline()` (repeats written out), then synced to the automatic
tab's heard notes (`_tempo_fit`, `_sync_timed` Viterbi placing/skipping bars,
`_place_in_bars`; text via `_sync_text`; text without count lines: rhythm from the
spacing, `spaced`, unless order-only finds more, `SPACING_TRUST`). Stored as `imported-tab - <name>.<ext>` /
`imported-tab.txt` + `.bass-tab-imported.json`; `POST /api/tab/import`,
`GET /api/tab?source=imported`, `POST /api/tab/import/remove`. Bump `IMPORT_RULES` when
syncing changes (`get_tab` re-syncs via `resync_import`).
- Other tunings: re-fingered with the learned fingering, `easiest_tunings()`.
- Make repeats match / Other ways to play (`repeat_groups`, `other_ways`,
  `POST /api/tab/repeats`), saved in the import's `unify` / `unify_tunings`, applied by
  `unified_notes()`; the tuning picker then offers "· original" / "· mine".
- Slides and hammer-ons/pull-offs: a code per note (6th value; /api/tab's 5th),
  `legatoMarks` draws them; the game's `legatoIn` / `onSlide` handle them. Dead notes
  are hidden when shown.

## Guitar tabs (imported)

The user's brother works on guitar (a Bass | Guitar switch in the tab view header,
`#tab-instrument`, per song `track-splitter-tab-instrument`; guitar tuning choices kept as
`song|guitar`). Guitar shows imported tabs only; the tab views, tunings, raise ½ step and
Make repeats match work with 6 strings. Files `imported-guitar-tab - <name>.<ext>` /
`.guitar-tab-imported.json` (+ `instrument`, `capo`, `track_index`); `instrument=guitar`
on the tab APIs; Track picker (`POST /api/tab/tracks`, `tabimport.rank_tracks`); capo;
6-line text tabs are guitar. Syncing uses `guitar_heard()` (`.guitar-heard.json`,
`GUITAR_HEARD_RULES`); fingering `tab.guitar_fingering` (`GUITAR_STYLE`). Learning from
imported tabs stays bass only.

## Game screen (/game)

The game as its own screen (`/game`, `arcadeOn`, `#arcade`): title, song wall with covers
(`/api/cover`, `.cover.<ext>`) and previews, song screen (instrument, difficulty, bests,
input) -> `arcadePlay` -> `startGame`. Mac icon "Track Splitter Game" (`install_shortcut(game=True)`,
`app.py --game`). Stars kept per best key (`STARS_KEY`). Menus: concert stage, song select
carousel (`renderCarousel`). In-game look "clean and premium" (mockups in `design/game-mockups/`):
`drawStage` mood background, `drawMultiplier` + `drawRingFire` by the score (`rbScore`); test
streak `track-splitter-game-streak`. Phases 2-3 (scoring depth, practice mode) to come.

## Guitar game (step 2)

The game works for guitar tabs: plays a version without the guitar and switches the input
to Guitar; chords (notes within 30 ms) are judged and scored as one note (`strumByChange`,
`readChord`, `chordPluck`, `CHORD_WAIT`); chord cards and bars on the board
(`drawChordCards`, `drawChordBars`). Show my timing doesn't judge chords yet.

## Guitar calibration

Timing and string calibration follow the input's Bass / Guitar switch: guitar has its own
keys (+ `-guitar`, `calKey`, `CALIBRATION_KEYS` in app.py), loaded by `loadCalibrations()`.
Strings dialog: Open strings · 1 min (default) | Full (`strcalMode`, `strcalPlan()`).

## Guitar game feel

`tab.strum_times` / `tab.snap_to_strums` move heard guitar notes onto strums
(`GUITAR_HEARD_RULES` 2); looser strum and chord rules in the game. Not yet tried on a
real guitar.

## My tab: recording your own playing

Bass input drawer (`#input-bar`): input/channel (Both inputs default,
`track-splitter-record-channel`), Bass/Guitar (`track-splitter-instrument`,
`PITCH_RANGE`), level, live note, Hear myself (low-latency monitor, soft clipper),
Tuner, Record. Stays on between songs; reopens on dropouts (`reopenInput`).
- Record: `POST /api/tab/record` -> `record_tab()` (transcribe, `align_take()`,
  Click switch (`#record-click`, `startClickTrack`, `track-splitter-record-click`);
  `apply_take()` add/replace, both snapped to the take's `grid` (the page's `editGrid`);
  add: `ADD_MIN_AMP` / `ADD_LOOSE_AMP`, `bar_starts` fills gap bars), saved in `.bass-tab-mine.json` + `.my-bass/take-<ms>.flac`;
  `redo_mine` rebuilds (bump `MINE_RULES`). A take waits as `pending` for review
  (`#take-review`, `POST /api/tab/mine/take`); Your takes (`#takes-dlg`).
- Drafts: `POST /api/tab/draft` (`new_draft` / `start_draft`), editing by delete
  (`/api/tab/mine/note`), drag (`/api/tab/mine/move`, `moved`), add / Fill
  (`/api/tab/mine/add`, `hand`), grid picker (`#tab-grid`, `track-splitter-edit-grid`),
  Undo per song; imported-tab drafts keep the tab's fingering (`import_places`).
- Playback: the player's "My bass" mode (`my_bass()`, `/api/audio?part=mine`).
- Match the song's tone (`tone.py`, `GET /api/tone`, `applyTone`): EQ from the song's
  stem vs the user's instrument captures (`<library>/.instrument-tones.json`,
  `#inst-dlg`); per song `track-splitter-tone`. Guitar: follows the song's clean /
  distorted parts (`tone.sections`, drive table, `CHAIN`, noise gate `GATE`) and adds a
  stereo room on clean parts (`tone.room`, `ROOM_MAX`).

## Guitar chords (input drawer)

Guitar mode of the input (same `instrument` switch): `heardNotes` (harmonic multi-pitch
on a second analyser), `chordOf` (16 shapes, slash chords), `chordTracker` (strum
detection, agreement before showing). Tested on sampled guitar chords; not yet on the
user's real guitar.

## UI (static/index.html)

- Sidebar (artists, rename/merge, drag a song to an artist), searchable song list with
  version chips (click loads paused; hover ↻ Redo via `POST /api/version/redo`, 🗑).
- Player bar: Web Audio, without + only decoded and started on one clock; Without /
  Original / Only / My bass (`#hear-mode`); `matchOriginal()`, `headroom()`. Skip silence
  (`soundBounds`, `track-splitter-skip-silence`) and hand trim (`#trim-dlg`,
  `POST /api/song/trim`, `playRange()`). Volumes: master / song / instrument (`#mixer`).
  Instrument picker (`#player-kind`, `#player-inst`; Bass | Guitar).
- Add songs dialog and Import tab share the `.fancy-dlg` look. Dark theme, violet
  accent; fits down to ~1024x768.

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
