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
  rotary-embedding-torch (the last three for `roformer/`).
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
