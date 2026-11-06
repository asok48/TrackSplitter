# Track Splitter: detailed notes

The detail behind `CLAUDE.md`: feature history, measurements, user quotes and
per-feature specifics, moved here word for word so `CLAUDE.md` (loaded every session)
stays short. Read the section for an area before changing it. Same headings as
`CLAUDE.md`.

## Working on this project

The page-check rule as it read before it was lightened (see `CLAUDE.md`):

- After every change, load the page and check for errors and that the main screens
  (song list, player bar, tab view, game) still appear (`drawHighway` / `drawGame`
  return early when their canvas is under 60 px: at ~0 height while the view opened,
  sizes went negative and `arc` threw, which broke the page when opening a tab): a stray closing tag once pushed
  the player bar off screen, and removing the Guitar Hero look took the 3D/Flat
  setting's code with it.

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
  to the notes; downbeat from where bass notes land. If < 0.4 fit score: 4 s lines,
  unless the fit is >= 0.15 and the song is steady (`_pulse`: the mix's onsets on the
  beats >= 1.5x at random spots; beat-length jitter < 6%). The bass fit alone left 8 of
  24 library songs without bars (Brianstorm, Amsterdam, Save Yourself, Wow! Wow! Wow!,
  Lazerlove, LoveCop, American Confidence: 0.21-0.44 at the right tempo; busy fast
  notes wobble, and an 8th grid at 166 bpm catches 43% of notes by chance), so the
  draft's 16th/32nd grid had nothing to snap to. Tempos were within 2% of the imported
  tabs; pulse 1.8-3.0 on every song. Now all 24 have bars, the 16 before unchanged; the
  beat grid hits 89-100% of the imported tabs' bar lines. The downbeat (which beat is
  "1") is still often a beat off (Save Yourself 3%, Number of the Beast 7% of the tab's
  bar lines on ours). Tabs saved without bars before `BARS_RULES` show "can be made
  better" (bar-less only; others aren't asked).
  Beat 1 (user: "fix the downbeat so bar lines start on beat 1"): measured on the 19
  imported tabs, the downbeat rule (where bass notes land, by loudness x length) is right
  on 14 of the 17 with a clear answer (Metalheads and Fireman & Dancer half a bar off, Save
  Yourself a beat). Tried adding where the bass changes note, where the harmony changes
  (bass pitch classes over 2 beats vs the 2 before), and kick / mid / high / bass-track
  onsets: the best mix fit 16/17, but chosen leaving each song out it got 14/17, the same
  as now, so the rule stayed. Instead "Beat 1 ◀ ▶" in the toolbar right after View ▾ (the
  user moved it there from More ▾, then from over the tab; shown for the automatic tab
  and a draft started from it; readout "0" / "+1" / "+2" / "−1", the words in its tooltip).
  To fit one row at 1440 px wide the tab toolbar's gaps went 8 -> 6 px, its segment
  buttons' padding 12 -> 9 px and the tuning picker 210 -> 165 px (measured: imported 964,
  automatic and draft 1109 of 1128 px); narrower windows wrap, as before, moves the bar lines a beat at a time (`POST /api/tab/bars`
  {song, beats}; `bar_shift` 0-3 in .bass-tab.json; `auto_lines()` -> `tab.shift_bars`,
  each beat a quarter of its bar, lines re-cut with `_lines_from_bars`). Fireman & Dancer:
  0% of its tab's bar lines on ours -> 93% two beats later. Making the tab again finds the
  downbeat afresh (the shift isn't kept).
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
  Slow-Fast slider (0.25-2.5 x 3 s ahead, 120-420 px/s, nothing overrides it since the
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
- Gaps in an imported tab (user, Look What Happened: "looks totally wrong on the import",
  "looks right in songsterr"): the file (Songsterr downloader export, 133 bars, no repeat
  or D.S. marks, 200 bpm) covers ~178 s of a 210 s recording; the sync left 26-51 s and
  62-69 s without bars (the bass-only track plays a B/E/F# riff and an A-G#-F#-E-D# run
  there that aren't in the file; bars 21+ match from 51 s, notes within ~10 ms). Only bar
  starts are kept, so bar 20 became one 25 s bar and its Lines line ~30 s with the notes
  crammed into the start. `layoutLines` now fills a bar >= 1.8x the median with empty
  bars of about the median length (Whole tab: 8 blank lines there instead of one squashed
  one). Scroll / Highway are by time and were fine. The file itself was read right
  (intro double stops with a ringing E3 on the G string, ties merged).
- Bars per line (View ▾ in Lines and Whole tab, `#tab-bpl`, per song in
  `track-splitter-bars-per-line`; Auto shows what it picked, "Auto (4)"): on You Know What
  They Do to Guys Like Us in Prison (186 bpm, ~5 notes/s) the user found the Lines view
  "move too fast" and asked: "ideally we could fit more bars onto a single line". The
  imported tab was within ~10 ms of the heard bass notes throughout, so it wasn't sync:
  Auto (`layoutLines`: 8/4/2/1 bars, <= 6.5 s, notes >= 28 px apart) gives 4 bars (5.2 s)
  there, and only 2 in a 1024 px window. A picked count (2, 3, 4, 6, 8) overrides Auto;
  crowded numbers shrink as before (down to 55%). 8 bars at 1024 px still read fine.
  Also used by the game played on the tab.
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
    song and speed. Speed is the slider alone (Note speed 0.25-2.5, was 1.6 (see below); 2.6 s ahead /
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
    Faster top speed (user, 2026-10-30: "in game mode, notes are still too close"):
    Note speed / the Highway's Slow-Fast now go to 2.5 (were 1.6; shared `hwSpeed`).
    On the 3D board spacing at the far end is 1/4 of the strike line's (dd = Z) while
    gems only halve, so on You Know What They Do... (186 bpm) far 16ths overlapped even at
    1.6: ~18 px apart vs ~30 px gems on a 1400 px window; at 2.5 ~45 px. Flat spaces
    evenly (Z / 2, ~910 px/s at speed 1). The Tab Highway stays capped at 600 px/s and
    Scroll at 700. Checked: Game section draws at 2.5, no console errors.
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
  - "Your best" in the Game section is set on every render (no tab: empty); it only changed
    when a song had a tab, so a new song kept the last song's best (user noticed).
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
  Editing in Whole tab (user: "it keeps scrolling back to the top when editing"): 4 s after
  the last hand scroll it snapped back to the playing line even while paused (the top, when
  stopped at the start). Now after a hand scroll it follows again only once the playing line
  changes (never while paused); clicking a line (which seeks there, e.g. a double-click to
  add a note) doesn't scroll it; and a redraw of the same song (every edit reloads the tab)
  keeps the scroll position. Tested: scrolled to 2000 px, added a note, undid it: 1986 px
  after 8 s and 6 s; playing, it followed the playing line again (in view at 1:12).
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
  Fast notes (user, 2026-10-30: "16/32nd notes are still inconsistently hit"; no game check
  of the song they meant, MCR's You Know What They Do..., so the 10-26 ones were used). A
  Python replay of `judge` from a check's recorded plucks + readings matches the games note
  for note (Brianstorm 205/205, Addictive 460/461); re-running `pitchOf` on input.flac at
  each reading matches the recorded `raw` 99.4%. 16ths (< 0.2 s apart) score 90-95% on most
  songs; Hysteria 289/642: its riff alternates a low D2/A1 with higher notes, the low ones
  are plucked (plucks found) but come in 5-15 dB quieter and the 85 ms buffer holds the
  previous note too, so YIN gives no clear reading. Offline Basic Pitch on the input also
  finds almost none of the missed notes (12 of 353). Tried and dropped: a second YIN on the
  newest 2048 samples (>= 50 Hz) when the long one is unclear: +31 real hits over 8 games,
  +26 in the wrong-notes control (tab moved +2 / -1). Also no gain: letting plucks < 30-70 ms
  after a pluck (the same attack counted twice) not take its pitch. 32nds: 8 in all the
  checks, too few to say. Next step: a game check of a song where it feels off.
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
- Pasted Ultimate Guitar tabs (user: "I also want to be able to use ultimate guitar tabs,
  even if it is just pasting"; still never fetched from the site): `read_text` uses the
  count line under each system ("1 + 2 + 3 + 4 + ...", `COUNT`) for bars (each "1") and
  rhythm (columns per beat from "1" to "2"), so such a paste is `timed` and synced like a
  file; with no tempo of its own (`tempo_known` False) `sync` tries the beat tracker's
  bpm, half and double, and keeps the best fit. Also read: repeat signs `|o`/`o|` (or
  `|:`/`:|`) with "x4" after the system (default 2 plays), h/p and / \ (`_row_notes`:
  legato like Guitar Pro's). Text after a system's last bar line is skipped: " x4" used
  to be read as a dead note and a 4. Hysteria
  (the user's paste): one 16-beat bar per system and untimed before, first note at 0:41,
  24% matched; now 4/4 bars of 8ths, intro x4, first note 0:00.2, 75%. IMPORT_RULES 7.
  The tempo guesses are judged by notes found over the whole tab (`found`), not
  `quality`, which leaves out bars the sync skipped (at 2x off it skipped half the tab and
  still won).
  Amsterdam (user: "doesn't look right"; no count lines, "(x4)" after a system's row or
  "(x2)" on the line under it, section times like "0:35" that aren't reliable: Intro and
  Verse both say 0:10, so they're not used): "(xN)" with no repeat signs plays the whole
  system N times (`REPEAT_LINE`). Its tab also doesn't follow the recording: a separate
  intro the song doesn't have and half the post-chorus left out, and text sync follows
  bars in order. So an untimed paste is also synced as evenly spaced bars (each one bar
  of the length `_sync_text` found, notes spaced as written) through the file sync, which
  can leave bars out and jump; the better quality wins; `_sync_text` itself may jump
  ahead (up to 60 s) where a bar fits badly and it and the next fit clearly further on.
  Amsterdam 38% (repeats alone 34%, tab ending at 2:45 of 4:00) -> 49%, choruses at the
  recording's (0:48, 1:30, 3:18); its 2nd verse (1:06-1:24) is played differently from
  the tab's riff. Tried and dropped: letting a bar between bar lines be 1-4 bars long
  (some tabs write 4 bars of 8ths without bar lines): longer windows matched repeated
  notes anyway (20%). Hysteria without its count lines: 60%. IMPORT_RULES 8.
  Then (user: "still off", "too fast and notes aren't lining up"): the evenly spaced bars
  kept the text's spacing as rhythm (`keep_lengths`), so notes were squeezed to the start
  of each bar and only 28% sat within 50 ms of a heard note. Now `TabTrack.rhythm` False
  for them: matched notes go where they're heard, the rest evenly between (as text sync
  does); a segment with more than 1.5x a typical one's notes counts as round(n / typical)
  bars (Amsterdam's 32 post-chorus 8ths: 3). Amsterdam 49% -> 55% (533 of 585 notes kept,
  bars not in the recording left out), Hysteria unchanged (75%). Still loose: ~40 places
  where two notes come < 80 ms apart. A Guitar Pro file syncs better. IMPORT_RULES 9.
- Pasted tabs without count lines (user, guitar paste of So Far So Fake, Ultimate
  Guitar's "Page 1/4" layout: "some of it comes in ok, but other parts are just flat out
  wrong"): the notes were read right, but placing them by order only pulled each onto
  whatever the guitar transcription heard nearby, so the strummed chorus's steady 8ths
  came out 0.03-0.8 s apart (clean single-note riffs were fine). Ultimate Guitar lays tabs
  out in proportion to time (an 8th every 4 columns, one "-" after each bar line), so
  `_rhythm_from_spacing` reads the rhythm from the columns (a bar is 4 beats unless far
  from the typical width; the bar line and its "-" aren't time) and the paste syncs like a
  counted one (`spaced`: timed, tempo unknown; bars placed by the Viterbi with jumps and
  skips, then kept steady). If order-only placing finds 25% more notes (`SPACING_TRUST`,
  a hand-typed tab spaced anyhow) that's used instead. So Far So Fake: chorus 8ths now
  0.22-0.25 s apart, matched notes a median 1 ms off (p25-75 -32/+25 ms), found 54% vs
  54% order-only (power chords in a distorted mix are only partly transcribed); the same
  tab re-spaced at random like a hand-typed one: 48% vs 45%. ~20 s to sync it (both ways).
  IMPORT_RULES 8. (In the code it became IMPORT_RULES 10 when this was merged with the
  Amsterdam changes, which had used 8 and 9.)
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
- Other ways to play (user: "similar to the pattern matching, also find common patterns
  and give you other options on how to play it", "I play Hysteria lower on the neck and
  want to move to that"): More ▾ → "Other ways to play…" (imported tab, any tuning; it was
  hidden outside the written one and the user couldn't find it on Amsterdam's .gp in E♭ B♭
  E♭ A♭, then: "I want to do other ways to play for standard tuning") opens the Make repeats match dialog in a patterns mode: every part played in 2+ bars
  (`repeat_groups(every=True)`; Make repeats match keeps only parts with 2+ fingerings),
  as written, then up to 5 ways the tab doesn't use (`other_ways`): per hand position (4
  frets + a 1-fret stretch, plus open strings) each pitch on a string under the hand,
  once with open strings anywhere and once open only where the tab plays open (the
  difficulty measure counts open strings as free, so without the second Hysteria's riff
  never came out on the D string at 5-7 over the open A, only on open D and G); ranked
  by `fit_costs` over all the part's bars with their neighbours, kept if 1.5 frets apart
  (`_neck_pos`: mean fretted fret; open strings left out, the open A pedal made every way
  look low), always with the lowest and the easiest that's 1.5+ frets lower ("lower").
  Each shows its frets, "n% easier/harder than as written" and Lower/Higher on the neck;
  Use everywhere saves it in `unify` like a chosen way (set_repeats accepts any map
  playing each of the part's pitches). "Lower on the neck everywhere" (`lower: true`)
  sets each part's "lower" way, keeping other choices. Learning uses these too.
  Other tunings (both dialogs, `tuning` in GET/POST /api/tab/repeats): `import_in_tuning`
  gives the import as shown there (its pitches fingered again with the learned fingering,
  in the import's shape, notes out of reach left out) with that tuning's own choices,
  `unify_tunings[tuning]` (carried over on re-sync; not learned from), so the same
  functions run on it; get_tab applies them. The dialog then says "as shown" for the
  app's fingering and names the tuning. Amsterdam in standard: 12 parts, 61 of 643 notes
  below the low E.
  Two versions per tuning (user: "each tuning would have 2 versions, the original and if a
  new one is made from this, that one you can pick"): once a tuning has choices (`versions`
  from get_tab), the tuning picker lists it twice, "… · original" and "… · mine" (values
  `key~original` / `key~mine`, kept like any tuning choice; `loadTab` splits them and sends
  `version`); without `version` the user's version is shown. Making a choice in the dialog
  switches to "mine". Both versions keep the choices; the game plays whichever is shown.
- Slides and hammer-ons / pull-offs (user: "get slides and hammer-ons/pull-offs working";
  chose: show them in the tab, count them in the game). The readers keep the kind
  (`TabNote.legato_in` "b"/"a" slid in from below/above, "h" hammered into; `legato_out`
  "s" slide / "h" hammer or pull to the next note on the string, "d"/"u" slide off; GP3-5 via
  PyGuitarPro slides + hammer, GP7/8 Slide flags 1|2/4/8/16/32 + HopoOrigin/Destination;
  MusicXML and text tabs carry none). `timeline` gives each note a code: how it's reached
  ("s", "h", "b", "a") then "d"/"u" if it slides off; it's the imported note's 6th value
  and /api/tab's 5th (IMPORT_RULES 6 re-synced old imports, which had true/false). In the
  user's tabs: Baby 35 slides, 10 hammer-ons/pull-offs, 5 slides off; YOLO 28 slides; Save
  Yourself 32; Sha-La-La-Lady 11. Page: `legatoMarks(d, k)` (into "/" "\" "h" "p", the note
  it comes from, out mark); Lines / Whole tab draw a slide as a sloped line from the note
  before (5/7, 7\5) or "/" "\" beside the number when that note is on another line, a
  hammer-on / pull-off as an arc with "h" / "p", slides off after the number; Scroll the
  same on its canvas; the Highway a line between the gems and the mark above the gem; the
  game board the mark above the slab. Game: `legatoIn` keeps such notes from being messy;
  a pitch-change pluck counts for them however quiet (a hammer-on / pull-off is quieter
  than a pluck); `onSlide` stops WRONG for pitches a slide passes through (between a
  slide's two notes, up to 5 semitones under / over a slide in, up to 7 past a slide off,
  for 0.6 s). Only 10 slid / hammered notes in the user's game checks so far (7 hit), so the
  game side was untested on real slides. The user's first game with them (Baby, 08:26;
  "seems inconsistent"): hammer-ons / pull-offs 2/2, slides 3/7, other notes 172/194. The 4
  missed slides all landed on G1 (E3) and read 45-50 cents sharp (A♭1-47), and the E8 they
  came from +30-40: the E string was out of tune (median +21-30 cents in the last three
  games against +2-13 on the others; +7 the day before), past the 40 cents a reading may
  be off; "heard A♭1, tab G1" WRONGs and Cry Baby Cry's G1/A♭1 misses had the same cause.
  Tuning check (`tuneCheck` / `tuningVerdict`): in a game, a pitch within 75 cents of a tab
  note sounding 0.08-0.4 s in counts toward its string's offset (`input.semis`, the exact
  reading); the results name a string off by >= 15 cents (median of >= 30 readings over >= 5
  notes): "Your E string 26 cents sharp: notes on it can miss. Retune with the Tuner (Bass
  input)." Simulated bass 30 cents sharp: "A string 43 cents sharp" (the track itself
  reads ~+13).
- Dead notes ("x") are left out of imported tabs when shown (user: "remove Xs in
  tabs"; Amsterdam's .gp had 223 of 868), so they're not in the views or the game;
  the saved import still has them (`notes[i][4]`).
- Sha-La-La-Lady gave a 500 when played (user: "when I play sha lala lady I get a 500 error"): /api/tab raised
  KeyError '40,40,33,33,35' in `unified_notes`. A saved "make repeats match" choice (new style, keyed by group) whose
  group no longer exists (the bars grouped differently after a re-sync) was passed on as is. `unify_rules` now sends such
  a choice to the group with a bar playing exactly those pitches, like the older choices, or drops it. Checked on a copy
  of the song: /api/tab (imported, standard) and /api/tab/repeats return 200, and the game plays it.

## Guitar tabs (imported)

The user asked for guitar tabs "similar to how we have tabs working for bass" (design:
imported first, then playing along, then automatic). Step 1, imported guitar tabs:
- A Bass | Guitar switch in the tab view header (`#tab-instrument`, per song in
  `track-splitter-tab-instrument`); tuning choices for guitar are kept as `song|guitar`.
  Guitar shows imported tabs only (no Automatic/Draft, no Redo);
  Lines, Highway, Whole tab, Scroll, tunings, raise ½ step and Make repeats match work
  with 6 strings (`LANE_COLORS` got a 6th, purple).
- Files: `imported-guitar-tab - <name>.<ext>` / `imported-guitar-tab.txt` and
  `.guitar-tab-imported.json` (same shape as the bass import plus `instrument`, `capo`,
  `track_index`), so a song has both. API: `instrument=guitar` on `GET /api/tab`,
  `POST /api/tab/import` (+ `track`), `/api/tab/import/remove`, `/api/tab/repeats`.
  Learning from imported tabs stays bass only.
- Import dialog: Tab for Bass | Guitar, and a Track picker when a file has several
  (`POST /api/tab/tracks`; `tabimport.rank_tracks`: guitar = 6+ strings, not named
  vocal/sax/keys..., named guitar first, then the busiest). The user's American Idiot .gp
  (Guitar Pro 8; the first real .gp the GP7/8 reader saw: read fine) has a "Vocals -
  Baritone Sax" track with guitar strings, and Rhythm (picked), Lead, Lead 2, Overdub.
- Capo (GP3-5 `track.offset`, GP7/8 `CapoFret`): the tab's own tuning is "custom", open
  strings + capo, frets as written. 6-line text tabs are guitar (lowest string E2 = 40);
  notes in one column are a chord and stay together when synced. Notation-only MusicXML
  guitar parts are fingered as chords.
- Syncing needs notes heard in the recording: `guitar_heard()` transcribes the guitar
  track ("(guitar only)" version, else htdemucs_6s separation, ~a minute) with Basic Pitch
  polyphonic, 70-1400 Hz (`tab.transcribe_guitar`), cached in `.guitar-heard.json`
  (`GUITAR_HEARD_RULES`). Text tabs need a beat: the guitar's, else the bass tab's.
- Fingering (`tab.guitar_fingering`): notes starting within 30 ms are one shape (a repeated
  pitch starts the next one: two strums squeezed together), each note on its own string,
  fretted ones within the hand's 4 frets (5 with a stretch); Viterbi over (hand position,
  shape) with `GUITAR_STYLE` (an open C beats it barred at the 8th; unplayed strings inside
  a chord and >4 fingers without a barre cost), then rounds pulling each chord to its usual
  shape. On the American Idiot file it matches the tab's own fingering 87% (lead) / 90%
  (rhythm); the overdub's B9 riff comes out on e4. `guitar_difficulty` counts a chord as one
  move. Tunings: standard, drop D, half step down, whole step down (D standard).
- Tested on a synthetic recording rendered from that file (played 2.3 s in, 2% slower,
  +-0.12 s drift): rhythm guitar 100% of notes within 50 ms (99% before the steady-bar
  fitting), no chord split apart; the same
  bars pasted as a text tab 84% (100 s, no drift); the bass import of the same song still
  99%. Not yet tried on a real recording. (The beat tracker found no tempo on the drifting
  synthetic song, bass or guitar, so text tabs there have no beat to go on.)

## Guitar game (step 2)

User: "Can we try adding it to game mode?" The Game section works for guitar tabs too.
- `startGame` for a guitar tab: plays a version without the guitar (all guitar first, then
  lead/rhythm, HQ first; none: Original with a toast), switches the input to Guitar
  (`setInstrument`) and back at the end (`game.instrumentBefore`).
- Chords: notes starting within 30 ms are one chord (`game.lead`, `game.members`), judged,
  scored and counted (streak, accuracy, stars via `game.units`) as one note; PERFECT/GOOD
  once over it; sparks on every string. Single notes use the bass judging (YIN, octaves
  count in guitar mode). `checkWrong` skips chords. On Easy chords stay chords (fewer of
  them); bass Easy still keeps one note of a double stop.
- Strums (`strumByChange`): distorted guitar is compressed and a chord change over ringing
  strings doesn't raise the level, so the attack rule found 21 of 114 strums on the test.
  Spectral flux on a 2048-sample analyser (`input.fluxAn`, bins 80 Hz-5 kHz within 60 dB
  of the loudest): a pluck at a peak >= 1.5 dB and >= 2x the median of the last 2 s.
- `readChord`: after each attack, `heardNotes` every 40 ms from 90 ms to 0.45 s on the sound
  since it (above -60 dB). `chordPluck`: a pluck in the window whose readings (before the
  next pluck, within `CHORD_WAIT` 0.4 s) hold enough of the chord's pitch classes (both of
  2, 2 of 3, 3 of 4+), with one reading mostly the chord (extras <= max(1, half)) and its
  lowest note the chord's lowest (pitch class).
- Tested (scratch `gtest.js` + `replay.py`): the synthetic American Idiot's "guitar only"
  track as a simulated input on its imported Rhythm tab, 40 s from 0:20: 121-125 of 146
  chords (Easy 23/27, Medium 34/40); controls: its bass track played in 0 of 145 (33 before
  the lowest-note rule), the tab moved 2/5/7 semitones 0-5 of 145; the bass game on the
  same song still 49/49. Headless Chromium at full window size gets `listen` only every
  ~40 ms (drawing starves it) and hits drop to ~60%; at 640x480 it runs every 20 ms. Not
  tried on a real guitar or a real recording. Show my timing doesn't judge chords yet.

- Chord cards (user: "above the highway, show the name of the chord with the fingering
  below it; for chords repeated more than once, just some indicator that it's the same
  chord"): on the board (`drawChordCards`, `chordRuns`, `chordCard`), each run of one chord
  shape (strums in a row; single notes between don't break it, 2.5 s without it does; a
  strum of fewer strings of the shape counts as the same) gets a card above the board's top
  rail at its first strum: the name (`chordOf` on the shape's pitches; raised songs named
  as played) and a chord box (low E left, × / ○, dots in the lane colours, "5fr" past the
  4th fret). It waits above the strike line, glowing on each repeat strum, until the run
  ends or the next card pushes it off. Only the first strum of a run has gems with fret
  numbers; repeat strums are one rounded bar across the strings they play, in those
  strings' colours, with the chord name above it (`drawChordBars`; user: "just show the
  chord name and have the fingerings be a solid bar across all the frets"; it was a "%"
  pill over the repeat's gems). Hit, the bar goes like a gem; missed, it greys. Cards
  too close to the next shrink to the name; on short windows (< 64 px above the board) and
  under the score box only the name. Board only, not "Play on: Tab". Tested on a synthetic
  tab (G C D Em F A5 B5, 8th strums) in 3D and Flat at 1440x900 and 1280x720.

## Guitar calibration

User: "add timing and string calibration for guitar similar to how bass works". Both follow
the input's Bass / Guitar switch: guitar has its own calibrations (keys + `-guitar`, `calKey`;
in localStorage and the library's `.calibration.json`, `CALIBRATION_KEYS` in app.py), loaded
by `loadCalibrations()` whenever the switch changes (`input.calibReady` guards the first
call, which runs before the `let`s exist). Timing: the same screen with 6 lanes (E A D G B e),
the result used by the guitar game and Show my timing. Strings (`openStringCal`): 6 strings at
the same 8 frets, 6 plucks a spot (288, ~6 min); `fitStrings(samples, strings)`,
`checkStrings(samples, opens)`, `stringScores` skips uncalibrated low strings by
`opens.length - m.strings.length`. Live: `readString` runs in guitar mode too (opens: a
guitar tab's tuning, else standard); the single-note readout says "on B5" first when sure.
Tested with synthetic plucks (per-string overtone stretch and brightness, made distinct on
purpose, so it checks the plumbing, not real accuracy): 258 samples, self-check 98%, 8 of 8
test notes on the right string; timing screen read 16 notes; bass calibration untouched
when guitar's is saved. Not yet tried on a real guitar.
Open strings only (user: "can you make the calibration simpler? Maybe just open strings?"):
the strings dialog has Open strings · 1 min (default, 8 plucks per open string) | Full
(`strcalMode`, `strcalPlan()`). A string with no fretted spots gets a worked-out line
(`d.guessed`): log B + ln2/6 a fret (B ~ 1/length^2), harmonic levels as the open string's;
spreads x1.5 when any line is guessed. The self-check then tests open notes against those
worked-out fretted places. Synthetic test: guitar 48 plucks in 28 s, 8 of 8 right; bass 3
of 6 shown, all right. Not measured on real recordings: the user's labelled bass takes
(`.bass-recordings`) would show how much open-only loses against Full.

## Guitar game feel

User: "the timing of the notes on the highway seems a little off and it's pretty difficult to
play and get a consistent streak". Measured on the user's real American Idiot (its guitar
track as heard, vs strum onsets found by hand-tuned flux):
- Tab timing: Basic Pitch hears distorted guitar ~33 ms late, so the synced tab sat ~38 ms
  behind the strums (41% of chords within 50 ms). `tab.strum_times` (log spectral flux,
  peaks > 1.8x the median of +-0.3 s) and `tab.snap_to_strums` (a heard start moves to a
  strum within -80/+30 ms) now run in `transcribe_guitar`: 57% within 50 ms, median -14 ms
  (rigid bars cap it at ~58%). `GUITAR_HEARD_RULES` 2: saved heard notes are snapped on
  the next open (~2 s), and imports carry `heard_rules`, so `get_guitar_tab` re-syncs them.
- Judging: on that stem the game scored ~0 chords (few strums detected; the lowest-note
  and "mostly the chord" rules failed on several distorted guitars at once). Now a strum
  is a flux peak >= 1.3x the median (was 2x); `chordPluck` takes readings' notes with
  salience >= 0.25 and needs 2 of the chord's pitch classes (fewer if the chord has
  fewer) with >= 40% of the salience on chord tones; no lowest-note rule. Offline replay
  (scratch `chordsim.js`) 20-60 s: real stem 1% -> 53%, synthetic 88 -> 91%; tab moved
  +2 semitones 7%, +5 27% (power chords share notes). In the page (640x480): real stem
  92/158 from 0:30, synthetic 146/163; the bass track played into the guitar game 22/126
  (a bass root's 3rd harmonic is a power chord's fifth). Not yet tried on a real guitar.

## My tab: recording your own playing

The user plays along through a Focusrite Scarlett (bass in an instrument input, Inst on,
Direct Monitor to hear themselves) and gets their own tab, and can hear their playing
with the band.
- Live note with fast notes (2026-10-30, user: "if I play fast, it doesn't register all
  the notes I play on the note output, it skips some"): the readout needed `steady` (5
  agreeing readings, ~100 ms), and the attack blurs the first few, so 8ths at 180 bpm
  (167 ms) often ended first. Within 250 ms of a pluck or pick click (`input.attackAt`) 3
  agreeing readings now show it (`quick`); otherwise still 5 (thud / hum guard). Readout
  only: `steady` for the game, Show my timing and wrong notes is unchanged. Replaying the
  user's Look What Happened take (page YIN on its audio): 106 shown of 154 notes, now 154
  (2 readings: 165, more than played); Fireman & Dancer take 168 -> 188. In the page with
  a simulated input (E1/B1 8ths at 180 bpm, then 16 repeated E1s): 40 of 40 shown, a
  repeat as a brief "–" then the note again.
- Page: "Bass input" in the tab view opens `#input-bar` (it replaced a record dialog;
  the user also wanted to see what's coming through without recording, and to hear
  themselves): input (an input named Scarlett/Focusrite is picked first; input,
  channel and monitor choices kept in localStorage), channel (Input 1/2 when stereo),
  level meter (tooltip: too loud / coming in / nothing), the live note, "Hear myself",
  Record. Closing the tab view hides the bar but keeps listening (see below).
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
  - Stays on between songs (user: "keep the bass running between switching songs, I have
    to click listen every time"): closing the tab view (to pick another song) only hides
    the drawer and stops a recording; the player bar then shows "Input on" (a green dot
    under 1400 px, `#player-input`), which turns it off. A game still ends on closing.
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
    Recorder leak (fixed): stopping a capture, take or game check disconnected only the
    recorder's output, so the input kept feeding it and its handler pushed blocks into the
    next one too; the user's PRS "Capture again" came out 90 s, two copies interleaved, so
    its spectrum and drive table were wrong. `dropRecorder` now stops the worklet and drops
    its handler; tested: cancel a capture then capture again, 4.5 s of audio for 4.5 s (was
    9.0). Direct Monitor on the interface plays the dry guitar over the matched tone, so the
    drawer note says to turn it off.
    Clean parts (user: "I want my clean sound to be as close to the original guitar in the
    song as possible"): measured So Far So Fake's clean parts against the PRS capture: the
    EQ already fits within 0.6-1.1 dB; what's missing is space. The song's clean intro has
    side (L-R) only 3.6 dB under mid (L/R correlation 0.43), more in quiet moments (tails),
    no echo; a DI is mono and dry. No sign of compression to copy (the song's level drops
    as fast after attacks as the DI's; the level spread gap is the playing). So clean parts
    (no drive) get a stereo room (`tone.room`, page `roomImpulse`: independent decaying
    noise L/R, 12 ms pre-delay, 1.2 s, highs dying faster, no lows under 150 Hz, scaled to
    unit gain over the song's guitar spectrum) in parallel, wet so the output's side/mid
    matches the song's, capped at -6 dB (`ROOM_MAX`); `sections()` now takes the side
    signal and stores `side_db` per group (TONE_RULES 3). Distorted parts stay dry (the user
    liked them). Measured in the page with the clean intro as the guitar: side -6.0 dB as
    meant, flatness 0.0073 (song 0.0062; dry it was 0.0018). The quiet bridge (song 0.0016,
    very tonal, side +3 dB from some stereo effect) comes out wetter than the record.
    `CLEAN_FLAT` 0.03 -> 0.015: a "clean" part noticeably noisier than the guitar gets a
    light drive (Crunch) matched by flatness, so edge-of-breakup parts aren't left clean.
    A guitar song with one tone now also gets `groups` (so the room applies). Not yet heard
    on the real guitar.
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
    simulated Scarlett only; not yet heard on the real bass.
    Guitar: the curve is measured from 70 Hz (`tone.GUITAR_LO`; bass 40 Hz) and held flat
    below: a guitar's lowest note is 82 Hz, and what a separated guitar stem has under it
    is bass leakage, which a guitar DI lacks (a synthetic stem with 55 Hz leakage asked for
    +10 dB at 50 Hz on the guitar; now -2, flat). Only EQ: a clean guitar DI against a
    distorted song guitar gets a big treble lift (+11 dB in the test), not distortion.
    Tested on a scratch library (synthetic distorted guitar stem, clean DI capture) through
    the API and the page (fake microphone); not yet heard on a real guitar.
    Following the song (user: "So Far So Fake starts with a clean tone and then changes to a
    distorted tone. I want my guitar to sound pretty close to the album"): for guitar,
    `tone.sections` splits the song's guitar track into parts (1 s windows: spectral
    flatness 150 Hz-4 kHz + spectral shape; optimal segmentation, `SECTION_COST` per part,
    >= 4 s; alike parts grouped, `GROUP_DIST`), stored in `.tone.json` (TONE_RULES 2).
    So Far So Fake: clean to 0:37 (flatness ~0.006), distorted 0:37-0:53 / 1:25-2:00 /
    2:35-3:23 (~0.21), a mostly clean verse, a quieter clean bridge, a crunchy outro (0.08);
    American Idiot distorted throughout (0.25) bar a lighter stretch. A guitar capture now
    keeps its audio (`<library>/.instrument-tones/<id>.flac`, decoded mono: ffmpeg's
    mono-to-stereo is 3 dB down, which put the drive's make-up 3 dB off) and its
    `drives` table (`tone.drive_table`, ~8 s at capture): the capture through `CHAIN`
    (high-pass 120 Hz, +6 dB at 800 Hz, gain, tanh, low-pass 6.5 kHz) at drives 1-2048x,
    flatness and spectrum each. `/api/tone` (guitar) adds `sections`, `groups` (per tone:
    drive whose flatness is nearest the song's, capped at 90% of what the guitar reaches,
    since a mix's stacked distorted chords are noisier than one guitar: the clean intro as a
    stand-in DI topped out at 0.13; make-up to the clean guitar's A-weighted loudness; EQ
    fitted against the driven guitar), `chain`, `play_db`, and `needs_capture` for a guitar
    captured before (EQs only; the note asks for a new capture). Page (`wireTone`,
    `followTone` every 50 ms on the song position minus the output delay): clean and
    driven paths summed into one EQ, gains ramped on each change of part; a noise gate
    worklet (`GATE`) before the drive, opening 12 dB over the input's quiet level
    (`floorDb`) but never above `play_db` - 25 (with constant playing floorDb was the
    playing level and the gate stayed shut); WaveShaper oversample 2x (+2.7 ms on the
    driven path; 4x +4 ms). The drawer note says "Now: Distorted (drive 60 dB) · Low ..."
    with the parts in its tooltip. Tested with the clean intro (-17 dB) as the guitar via a
    simulated input: the page's output flatness clean 0.004 / distorted 0.21 / outro 0.13
    (song 0.006 / 0.21 / 0.08), loudness by the server's model within 0.2 dB across parts,
    in the page ~3 dB quieter when distorted (the gate between notes). Not yet heard on a
    real guitar; the user's first guitar capture ("PRS guitar") predates drives. `TONE_RULES` in app.py: bump
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
  Octaves (user: "trying to add my 16th note 6s around 50 seconds", then "picking up more
  now but not all of them"; MINE_RULES 6): a played note an octave (or two) from the
  note in its 16th slot replaces it, on that note's timing, also from the slot either side
  when its own is empty. Amsterdam's draft (from the .gp in E♭ B♭ E♭ A♭) has E♭1s that
  standard tuning can't reach, so they're hidden there; the user played E♭2 (A6) in those
  gaps and the hidden notes kept 7 of 22 heard notes out. That take now: 13 in, 7 E♭1s
  replaced; one more was too quiet (0.52 < ADD_MIN_AMP) and 3 quick repeats weren't
  transcribed. An earlier take that day was silent for its first 13 s (0:42-0:55) and then
  only rumble (~52 Hz, no overtones): nothing reached Input 2 then; the same input heard
  the bass at -34 dB in a game. Wow! Wow! Wow!'s draft (one quiet Oct 23 take, still at
  MINE_RULES 3) comes out empty when rebuilt: its notes are under ADD_MIN_AMP.
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
- Moving notes (user: "drag notes on the tab to other spots"; chose: in time only, the
  draft only): in the draft's Lines / Whole tab views a note dragged left/right follows the
  pointer, snapped to its bar's 16ths within its line (`dragNote`); a click without moving
  still selects it. `POST /api/tab/mine/move` {song, start, pitch, to} records
  [from, pitch, to] in `moved`, applied in order after a rebuild (`with_moves`, before
  removals; 60 ms like them); a Replace take over the stretch drops moves in it. Undo
  moves it back. Tested: a drag of 60 px snapped and saved, kept through `redo_mine`.
  A take waiting for review: its green notes aren't in the draft yet (user got "that note
  isn't in the draft any more"), so moves of notes only in the preview go in
  `pending.moved`, shown in the preview, added to `moved` on Keep, dropped on Discard.
  No scrolling while dragging (user: "the page scrolls easily when dragging"): pointerdown
  stops text selection, the tab body's overflow is hidden, wheel events are cancelled, and
  `tabFrame` skips frames while `tabView.dragging` (Whole tab's follow-scroll, line changes).
- Adding notes by hand (user: "an easier way to match the notes in the song ... more of the
  6s", "what I really need is just 16th notes", "trouble getting it to match the song"):
  in the draft, double-click a string where there's no note (the nearest fret on that
  string in the line, snapped to 16ths), Option-drag a note to put a copy there, or select
  a note and Fill 16ths / F (it again on every 16th of its bar's grid up to the next note,
  at most a bar). `POST /api/tab/mine/add` ({start, end, pitch} or {notes: [...]}) keeps
  them in `hand`, added after moves when the draft is made again (`with_hand_added`, not
  within 30 ms of one of that pitch); deleting one just takes it out of `hand`; dragging
  one moves its `hand` entry. Shown like a kept take's notes ("hand", blue with Show what I
  added). Undo takes the last add / fill out.
  Editing grid and getting there (user: imported tabs are "very close" but changing them is
  "a pain", Brianstorm needs 16ths/32nds): a grid picker over the draft (`#tab-grid`,
  `editGrid`: 8ths / 16ths / 32nds / 8th or 16th triplets, steps per bar taken as 4 beats,
  `track-splitter-edit-grid`) sets where double-click, Option-drag, drags and Fill snap
  (the button reads "Fill 32nds"); Undo / ⌘Z goes back step by step (`tabView.undo`, per
  song; it kept only the last change). "Edit" (was "Edit this tab", floating over the tab; the user:
  "it covers the tab") sits in the toolbar after View ▾ and Beat 1 on an imported or
  automatic tab, the grid picker in its place on the draft (both 72 px, so nothing moves)
  (`editThisTab`): starts a draft from it, or with a draft already offers to open it or
  start again; double-clicking such a tab asks the same.
  A draft from an imported tab keeps the tab's strings and frets (user: "I want to edit an
  imported tab"; re-fingered from its pitches, 222 of Brianstorm's 536 notes moved, its A4s
  showing as E9): in the tab's written tuning `import_places` pins each draft note the tab
  has (same pitch within 30 ms, the tab as shown, "other ways to play" choices included) and
  `tab.fingering(pinned=...)` places only the rest (added, moved, recorded notes), each
  pitch leaning to where the tab plays it. Brianstorm: 536 of 536 kept. Other tunings, and
  drafts from the automatic tab, are fingered as before.
  Recording: "add" skipped a played note when the same pitch sat in the 16th slot beside
  it (a written note played a little off), which cut runs of 16ths of one note in half;
  now only when the take doesn't also play that neighbour's own slot.
  Tried and dropped: splitting a take's long notes at sharp level jumps that hold (re-
  plucks blur together when transcribed): on the labelled recordings fast 8ths found 246
  -> 261 of 306, but more fake notes there and from hand touches (pluck-then-stop steps 21
  -> 25 notes for 10 plucks).
  Add over a gap (2026-10-30, Look What Happened, user: "I'm playing 8th notes but it gives
  a single note and it's wrong"): the draft started from an import whose sync left 26-51 s
  without bars, so `bar_starts` had a 25 s bar, a 16th of it 1.56 s, and the take's 8ths
  went in one per 1.56 s (19 added). `bar_starts` now cuts bars >= 1.8x the median into
  bars of about the median length (as the page's `layoutLines`): the same pending take
  adds 65, on 8ths. "Wrong": its E2s replaced the tab's open-E 8ths by the octave rule,
  but the audio there has no 41 Hz (E1) and no 124 Hz, only 82 Hz: real E2s, not a
  misread (where it read E1, 41 Hz is strong). Still open: 22-27 s, ~14 level jumps came
  out as 4 notes; Basic Pitch's re-pluck split (`split=` 0.15-0.4) cut notes 70-150 ms
  apart where no pluck was, so not used. MINE_RULES not bumped (kept takes keep their notes).
  Next take (user: "still not what I played"): pitches now matched the audio (YIN runs vs
  notes found), but 43 of 154 notes were under ADD_MIN_AMP (0.30-0.55: re-plucked 8ths of
  a ringing B/A), so e.g. 12 B 8ths gave 1 note. Saved takes: clear ones have 18-48% under
  0.55; two junk ones (Cry Baby Cry 23 notes, Wow! Wow! Wow! 20) have all of them under,
  and peak level didn't tell them apart (-8 to -13 dB). So a take whose median note is
  >= 0.55 keeps notes down to ADD_LOOSE_AMP 0.3, else stays at 0.55: that take 104 -> 131
  added. Also by design: "add" keeps a tab note of another pitch in its slot (B played
  over the tab's open-E 8ths at 24.8-27.3 s isn't added); Replace takes the stretch over.
  Open-E 8ths (next take): re-plucks of E1 keep one steady pitch (YIN 106 readings in a
  row), so ~15 plucks came out as 4 and 7 notes; E2/B/A runs were right. `take_notes` now
  splits re-plucks (`_split_replucks`, `TAKE_SPLIT` 0.2, at least 8 frames ~93 ms apart)
  of notes up to G1 (31) only: those runs 12 and 15 of ~15, the take's adds 121 -> 137.
  Splitting every note (min gap 6) doubled notes 80 ms apart and on Fireman & Dancer's take
  added 24 notes with 2 on the tab; low notes only: Fireman +2, Cry Baby Cry's long take
  +41 (8 on its tab: unknown whether real). MINE_RULES not bumped (kept takes unchanged).
  Then up to E2 (40), at least 10 frames (~116 ms) apart: the 9:02 take's E2 8ths also
  came out as quarters or gaps (22.6-24.4 s: 5 notes, now 11 even 8ths; 27.4-30 s: 9-10
  of ~15). Up to E2 with an 8-frame gap split held notes in two 0.1 s apart (Fireman &
  Dancer: +17 notes, 2 on its tab, inside the tab's quarter notes); with 10: +9, 1 on, and
  fewer notes < 0.11 s apart on every take (Cry Baby Cry 13 -> 5). Snapping to the take's
  grid merges what's left of those into one step.
  Takes on the grid (user: "a mode where it just overwrites with what I played" = Replace,
  which existed; then "notes take my exact timing when they should be equally spaced"):
  the page sends the draft's grid picker (`editGrid`: 8, 16, 32, 12, 24 steps a bar) with
  each take (`grid`, kept in the pending take and the kept take). Replace now snaps each
  played note to its bar's nearest step (two in one step: the surer stays); "add" uses the
  grid instead of fixed 16ths. Takes without `grid` (older) keep their timing / 16ths. A
  note rounding onto the bar line is the next bar's first step (before, step 16 of one bar
  and step 0 of the next were two slots, so a note could double up). On the latest Look
  What Happened take, Replace: median gap 0.174 s with 50 different gap sizes -> 0.165 s
  (16ths: 28 sizes, 8ths: 21, from the bars' small length differences), none < 40 ms.
  Delete / Replace bugs (user: "deleting a note from my draft doesn't work and when I did
  replace ... only recording part was left, the rest of the song was gone"): (1) the note
  was the waiting take's (green); `edit_mine` put it in the draft's `removed`, which the
  take's notes never pass through. Now a note not in the draft goes to the pending take's
  `removed` (undo takes it out), applied in the preview and on Keep (and copied into the
  draft's `removed`, so `redo_mine` keeps it out). (2) The draft had been deleted before the
  take (no log to say how; the import and auto tab were also rewritten at 9:07-9:08);
  `record_tab` started a missing draft from the tab only for "add", so Replace made a draft
  of just the take. Now for both modes. Checked on a copy: delete 176 -> 175 new notes,
  undo 176; Replace with no draft -> draft from the import (794), on Keep 58 notes before
  the take and 654 after it kept. Then "still deleted everything": Discard kept the empty
  draft that bug had left, and an empty draft wasn't "no draft". Now a draft with no notes,
  takes or base also starts from the tab (copy: 70 before / 661 after kept).
  Replace only where played (user: "when I don't play anything, it removes what was there
  when I actually want to keep it"): it cleared the whole stretch, the 2 s run-in included.
  `played_spans` = the bars (of the draft's `bar_starts`) the take has notes in (a note
  within 1/32 bar of the next bar line counts for that bar, as it snaps there); only those
  are replaced, in the preview, on Keep (and the draft's `removed` / `moved` are cleared
  only there). No bars: 1 s after each note. The 9:18 take (15.0-56.6 s): 79 draft notes
  in the stretch, 0 kept before, 38 now (the intro bars 15-21.9 s, before the first note).
  Click track (user: "can we add a click track when recording?"): a Click switch above
  Record (`#record-click`, `track-splitter-record-click`). While a take records,
  `startClickTrack` schedules `metronomeClick` (the timing calibration's tick) ~0.3 s ahead
  on the audio clock for every beat of the tab's bars as shown (`clickBeats`: the page's
  lines, so gap-filled bars too; beats a bar from the bpm like the faint beat lines, beat 1
  higher), re-finding its place when playback restarts; the 2 s run-in is the count-in.
  The recorder only takes the input. Checked in the page (simulated input, Look What
  Happened): 20 clicks in 6.5 s, 0.325-0.338 s apart, an accent every 4; none after Stop.
  Review warning: `match` is now taken only over take notes within 0.5 s of a draft note
  (>= 10 of them, else none), and a take whose median note amplitude (`sure`) is >= 0.55
  isn't flagged: the take over the import's gap said "Only 11% matches" (105 of 141 notes
  where the draft had none; 44% where it had notes) at -15.6 dB from the Scarlett's
  instrument input (Input 2).
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
  Guitar was disabled there and in Your instruments' type choice for a while (user: "my
  brother is handling guitar"); turned on again when the user asked for guitar tone
  matching "similar to what exists for bass". Your instruments' new-instrument type
  starts on what you're playing now (placeholder "e.g. Strat" for guitar).
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


## Game screen (/game)

User (2026-10-30): "the game mode is a lot of fun, I would like to expand on it and make it
its own standalone game that feels more legit". Asked what standalone means, they picked
"its own screen in this app" (not a separate project) and, for feel, menus & song select,
scoring depth and practice mode. Done in phases; phase 1 is the screen and menus.

Phase 1 (screen, song select, Mac icon):
- `/game` (`game_page()` in app.py) serves the same index.html; the page sees the path
  (`arcadeOn`) and shows `#arcade` (fixed, z-index 45, hidden while `body.game-on`) over the
  app instead of the shortcut card (`startArcade`). Everything else (library, tabs, input,
  calibration, startGame / endGame) is the app's own code, so nothing is duplicated.
- Screens (`setScreen`): title (Play, Game settings, Calibrate timing/strings, "Add game to
  Applications", Back to Track Splitter), song wall (`renderWall`, `wallCard`; songs with at
  least one version; search, sort Recent / A–Z / Artist kept in
  `track-splitter-arcade-sort`), song screen (`renderSong`: instrument, the four
  difficulties with stars and best at the current timing (`arcadeBest`), input, Play;
  "Make the bass tab" runs the tab view's `makeTab` job when there's none).
- Play (`arcadePlay`): connects the input if needed, loads the song's no-bass version
  (`gameVersion`), opens the tab for the picked instrument, waits for it, switches to the
  Game section and calls `startGame`. Results have a "Songs" button (`#game-songs`) in
  /game; closing them shows the song screen with the new best. Going back to the wall or
  title closes the hidden tab view so the highway stops drawing.
- Stars: `endGame` now keeps the most stars per best key (`STARS_KEY`
  `track-splitter-game-stars`). `bestKey` adds "-guitar" after the source for guitar tabs
  (bass and guitar imports were both "imported" and shared bests; any guitar bests from
  before are no longer shown).
- Covers: `GET /api/cover?song` serves `.cover.<ext>` from the song folder, else a picture
  inside the original (ffmpeg, cached in the temp `track-splitter-covers`), else 204 (not 404,
  so the console stays clean) and the page shows a drawn cover (`coverEl`: two hues and a
  pattern from a hash of the song id). The user's library is all YouTube .webm with no art,
  and `.song.json` keeps the video title, not the link, so existing songs get drawn covers.
  New links save the video's thumbnail (`fetch_thumbnail`, copied in as `.cover.<ext>` with
  the original). Dropping a picture on the song screen's cover saves it
  (`POST /api/cover`; the old one goes to the Trash). `song_info` has `tabs` {bass, guitar}
  and `cover` (mtime, cache key).
- Previews (`startPreview`): the game's version from 35% in, fading in over 0.8 s, 20 s.
- Keyboard: title ↑/↓; wall arrows by grid column (`moveInGrid`), typing searches, ↓/Enter
  from search into the grid, Esc back; song ↑/↓ difficulty, ←/→ instrument, Enter plays.
- Input on the song screen mirrors the drawer's device list, level and note (the drawer is
  in the hidden tab view).
- Mac icon: `install_shortcut(game=True)` makes `~/Applications/Track Splitter Game.app`
  that opens /game (or runs `start.sh --game`; start.sh passes its arguments on, `app.py
  --game` opens /game). `/api/shortcut` has `game_exists`; POST takes `{"game": true}`.
  Windows: a second .lnk running `app.py --game` (untested).
- Tested on a copy of the library (port 7871): wall 28 songs, arrows/search/Enter, song
  screen, simulated bass input, Play → full-screen game, pause → quit → results → Close /
  Play again / Songs; normal app at / unchanged; no console errors. Shortcut made in a temp
  home folder only. Thumbnail saving untested against a real link.

Phase 1 redesign (same day):
- Menus: the user picked a "concert stage" look (of stage / neon arcade / console clean):
  title screen with a lighting truss, sweeping beams, haze and a lit floor; the song wall and
  song screen merged into one song select: a 3D carousel of covers (`renderCarousel`, cards
  kept by song id so moves animate, ±6 around the picked one; ←/→, wheel, click a side card),
  under it the picked song (title, artist, Bass | Guitar, Make the bass tab) and the four
  difficulties (skewed rows, colour per difficulty `DIFF_COLORS`) with Play. Input chip top
  right. Keys: ←/→ song, ↑/↓ difficulty, Tab instrument, Enter play, typing searches, Esc back.
  Display font Avenir Next Condensed (`--arc-font`; user: "just worry about mac").
- In-game screen: a first restyle (stage truss, beams, crowd silhouette, chrome rails,
  condensed italic HUD) was rejected ("looks like a flash game"). The user then chose a
  "clean and premium" direction and iterated on mockups (kept in `design/game-mockups/`):
  look A "Studio" (of Studio / Glass / Mono), then:
  - Only the score at the top (no song title, artist or progress); stars with progress under
    the score ("I still want the star system too being shown with the progress"); the
    multiplier moved next to the score as a prominent ring badge (×1 grey, ×2 yellow, ×3
    orange, ×4 electric blue), ring filling toward the next multiplier, streak under it.
  - At ×4 the badge is on fire: particle fire (soft sprites rising and cooling white → cyan →
    blue, drawn over the badge). Electric bolts were tried first; the user preferred flames;
    a thinner, bluer fire drawn behind the ring "doesn't look as good" — keep the brighter one
    (`board-hud-fire.html`, user: "that's sick").
  - Dynamic background that follows how you're doing: deep violet glow when steady, bright
    blue/violet at ×4, dim cold grey with a faint red tinge when struggling, easing between.
  - Hits Rock Band-like but "more subtle and clean": button pops slightly and fills, quick soft
    flare, thin ring, ~6 small sparks (~0.2 s); ×4 = the same in blue with a soft strike-line
    glow, edges glowing blue, notes with a faint blue rim. No flames on hits, no lightning.
  - Legibility: milder perspective (far notes ≥ ~75% size), faint per-string bands tinted the
    string's colour and coloured string names, bigger bolder fret digits, open strings hollow
    (outlined), bar lines stronger with bar numbers and fainter beat lines, a dashed purple
    line before a move up the neck. Nothing under the board ("don't show the text at the
    bottom": no position label, no multiplier there).
  - System font (SF), weights 500-800, no italics.
- Built (same day), replacing the rejected restyle: `drawStage` (mood glows, `MOODS`, eased via
  `game.mood`), `drawGame` (milder perspective `dd = Z * 3`, notes `size(z) >= 0.75`, string bands,
  bar numbers, `handMoves` dashed lines, flat notes / hollow opens, subtle hits: flare + ring + 6-8
  sparks, Perfect / Good / Miss in sentence case), `rbScore(g, right, notesLen, cy)` (score + stars,
  also used by Play on Tab), `drawMultiplier` (`MULT_COLORS`, ring fill = streak % 10) and
  `drawRingFire` (particle sprites `FIRE_TINTS`, state `ringFire`). The score and ring sit lower
  (`ringY` 70-96) so the x4 fire isn't cut off at the top. Pause / results / settings dialogs:
  system font, plain #16161c panels, one purple button. `GAME_FONT` is the system font now (the
  menus keep Avenir Next Condensed via CSS). Play on Tab keeps its older effects apart from the
  score and stars. Removed: the song card, truss / beams / crowd, chrome rails, the multiplier
  meter on the strike line.
- Testing: localStorage `track-splitter-game-streak` starts a game at that streak (35 = x4) to
  see the x2-x4 looks; with `track-splitter-game-start`. A simulated input must be a fresh
  MediaStreamDestination per getUserMedia call (the app opens the input twice and stops the
  first stream), kept live (a silent ConstantSource) until the bass track starts.
- Tested on a library copy (port 7871): /game song select → Hysteria, steady, struggling (red
  mood, amber edges), hits with calls, x4 via the test streak (blue board, fire, streak under the
  ring), pause, results; the normal app's Game section and Play on Tab; no console errors. Not
  yet seen with real playing at x4 for long.
- "Still looks flash gamey" after the clean look was built: the diagnosis given was that it's
  drawn (flat 2D shapes on a slanted canvas), not rendered. The user agreed to prototype a real 3D
  board (WebGL, three.js). Notes coming toward you was offered and picked, then the user recalled it
  had been hard to read: a vertical track was already tried and dropped ("vertical is too
  confusing, keep it horizontal"). Any 3D board stays sideways (right to left, low E at the bottom).

### 3D board (in progress, saved 2026-10-30)
- Built: `static/game3d.js` (three.js 0.160 saved in `static/vendor/three/`, MIT, offline;
  import map in index.html). `drawGame` uses it when Board = 3D (`use3D()`; Flat and guitar chords
  stay 2D). Sideways board leaning back 45°, maple or dark fretboard (Game settings → Fretboard,
  `track-splitter-fretboard`), fret wires as bar lines (dot inlays removed: "remove the dot"),
  one-size notes with the number printed on top (white on colour), quick repeats as ticks, notes
  fade in from the distance, pads dip and light on hits, mood-driven stage lights, x4 = blue pads,
  blue highway wash, fire on the multiplier. HUD (`drawRockHud`): angled metal score plate with
  stars, metal multiplier disc, the latest call (PERFECT / MISS ...) beside the score ("the accuracy
  alert gets in the way" → "put the accuracy next to the score": user said "this is good").
- Prototype: `design/game-mockups/board-3d-prototype.html`.
- "Make the game look more finished" (same day, user picked all four):
  - Menus in the rock style: violet swapped for amber / red / warm white (beams, lamps, buttons,
    logo, carousel glow), dark metal plates for secondary buttons, grit over the menus.
  - Pause / results / settings (`dialog.game-screen`): dark metal plates with a bright top edge,
    heavy condensed type, amber main button, gold stars popping in, metal stat tiles.
  - 3D board: bar numbers above the top string (sprites, `bar<i>` textures), dashed amber lines
    before a hand move (`moves: handMoves(d)`).
  - Loading screen `#arc-loading` (cover, title, artist, a tip from `LOAD_TIPS`, animated bar) and
    fades (`fadeTo`, `#fader`): song select → loading → game, results → song select.
- Not done yet: guitar chords in 3D; not tried with real playing.
- The game is called **Encore** (user: a name "general for all instruments"; picked from Encore /
  Headliner / Live Wire / Jam Session): title screen, window title, and the Mac app
  `~/Applications/Encore.app` (`install_shortcut(game=True)`). The main app's sidebar has a big amber
  "Encore · Play along with your songs" card (`#encore-btn`, a link to /game; user: "a prominent
  button on the side bar") and, on a Mac without the Encore app yet, "Create Encore shortcut"
  (`#encore-shortcut-btn`) in the sidebar footer. A "Track Splitter Game.app" made before the rename
  isn't removed.
- Song select tidied ("clean up the select song screen"): tools in one group at the top right (search
  with the song count as its placeholder, sort, Covers | List), the input picker moved by Play, the
  "Relaxed timing" line only shows warnings now, shorter key hints (`#arc-hint`), and the stage's floor
  grid removed (it looked like highway bars). List view ("just see a list view", `arcade.view`,
  `track-splitter-arcade-view`, `renderList` / `markList`): rows with a small cover, title, artist,
  tabs and stars for the chosen difficulty, details + Play on the right; in List ↑/↓ is the song and
  ←/→ the difficulty (Covers: the other way round); double-click a row plays.
- Album covers (user: "search google for album covers?" → MusicBrainz / Cover Art Archive instead, free
  and meant for apps): `album_candidates` (MusicBrainz recording search by artist + title; live versions
  and live albums skipped, Brianstorm had come back as a Royal Albert Hall live album; the earliest
  official studio album first) and `save_album_cover` (Cover Art Archive front-500). One file per album
  in `<library>/.covers/<release group id>.jpg`, songs point at it via `.song.json` "cover_lookup":
  {"album"} ("I don't want to store duplicate covers"); no match is remembered too. `cover_file` order:
  dropped picture (.cover.*), album cover, link thumbnail (.thumb.*, now only a fallback), art inside
  the original. Runs in the background (`cover_worker`, `queue_covers`): when /game opens
  (`POST /api/covers/fetch`, `fetchCovers` polls `GET /api/covers/status` and re-renders as covers
  arrive) and when a new song is added ("so they load after a song is uploaded"). Only the MusicBrainz
  search is paced (~1/s); downloads run 4 at once ("it's pretty slow to get them": ~10 s → ~1 s a song).
  Test copy: 27 songs in ~30 s, 21 found, 16 files. Only the artist and title are sent.
- Covers: rate-limited lookups are retried (back in the queue after 2 / 4 / 8 s, up to 4 tries;
  searches 1.2 s apart) instead of waiting for the next visit ("songs that are skipped are not tried
  again until I go back"); a refresh on /game returns to song select with the same song
  (`encore-place` in sessionStorage).
- Encore as an app (user: "make this a standalone app", "like a mac app"; told a native WebKit
  wrapper wouldn't run faster and risks the Scarlett's second channel and the timing calibration, so
  picked "an installable Chrome app"): `static/encore-manifest.json` (name Encore, start /game, scope
  /game, standalone) linked only on /game, icons `static/icons/encore-*.png` (a guitar pick with an E,
  from encore.svg via qlmanage), an offline copy `static/encore-sw.js` (network first, never /api).
  Install from Encore's title screen ("Install Encore app", Chrome's install prompt) or the sidebar link
  (opens /game). If Track Splitter isn't running when Encore opens, Encore shows "Starting Track
  Splitter…" (`#arc-starting`, `startTrackSplitter`), opens `encore-start://`, and reloads once it's up;
  the link is handled by `~/Applications/Encore Helper.app` (`install_encore_helper`: no Dock icon,
  registered with lsregister, runs `start.sh --no-browser`; `app.py --no-browser`). The helper is made
  when installing. Untested: the browser pane used for testing refuses service workers, so the offline
  start screen and the install prompt need trying in real Chrome.
- Title screen polished ("the menu screen looks sort of weak ... polished"): a drifting, dimmed wall of
  the library's covers (`#arc-wall`), the brand (pick icon + ENCORE) and a console-style menu on the left
  (big text items, an amber bar sliding to the focused one, `moveMenuBar`; Play, Settings, the two
  calibrations), "Jump back in" on the right (the last song played, `encore-last-song`, with its best),
  Install / Back to Track Splitter as small footer links (`renderTitle`). Not yet looked at in a
  browser (the user stopped browser checks).
- "Make Encore only available from that button": the main app's tab view has no Tab | Game switch any
  more (`#tab-section` hidden outside /game, `setSection` forces 'tab'); the game is reached through the
  sidebar's Encore button (and the installed Encore app).
- "I always see the app was updated text": it was accurate (the running server predated edits to app.py
  made while it ran). The note now has a Restart now button (sidebar and Encore's title): `POST
  /api/restart` stops the server and `main()` re-runs the same program with `--no-browser`
  (`os.execv`); refused while a song is being processed; the page polls /api/info and reloads.
- "Make the rest of the game ui look more modern" (all four picked), in the title screen's style:
  - Pause, results and settings are full screens (`dialog.game-screen.full`, `.dlg-cover` with the song's
    cover blurred behind via `setDlgCover`): big gradient titles and text-menu buttons (Resume large and
    amber, Quit soft red), arrow keys move through them; results show the cover beside the title
    (`#game-art`), a 96px score and the stats.
  - Song select and loading: the picked song's cover blurred across the background (`#arc-select-bg`,
    `#arc-load-bg`, crossfading), the stage beams faded to 25%, difficulties as chips, light glass
    buttons instead of metal plates.
  - HUD: a flat frosted score panel and a flat multiplier ring (x4 fire kept); the Pause pill is glass.
  Not looked at in a browser yet (syntax checked only).
- No Pause pill in a game ("do we want a pill?"): Space, Esc or a click on the board pauses. The pause
  menu shows the frozen game blurred and dimmed behind it ("just a blurred background for the pause
  menu"); results and settings keep the blurred cover.
- Installed Encore "keeps waiting for Track Splitter to start": no Encore Helper existed (the running server
  was older and had made the earlier Encore.app launcher instead), and once made, the helper's shell
  launcher ran under Rosetta, so the venv's arm64 numpy failed to load ("have arm64, need x86_64"). Fixed:
  the helper's Info.plist prefers arm64 / requires native, its launcher runs `arch -arm64 /bin/sh start.sh`,
  the waiting screen shows a "Start Track Splitter" button after 4 s (Chrome may block the encore-start://
  link without a click), and the installed app re-makes its helper on every start. Checked: `open
  encore-start://open` started Track Splitter natively.
- Stop when closed ("can we stop tracksplitter when I close it?"): the helper starts it with
  `--auto-quit`; pages POST /api/ping every 5 s and a goodbye beacon on pagehide; `auto_quit_watch` stops the
  server 15 s after the last check-in (45 s grace at start), never while a song is processing. Track
  Splitter started any other way keeps running. The quit itself not yet seen happening.
- `~/Applications/Encore.app` (the older launcher made before the helper) is left in place; the installed
  Encore lives in `~/Applications/Chrome Apps/`.
- No pills on song select ("I don't like the pill look"): sort / view / instrument as text tabs with an amber
  underline, an underlined search, a bare back chevron, 10px-corner buttons.
- Settings flattened ("just put settings at the top and then put the settings without a box around it"):
  heading at the top, label column + plain choices (active underlined amber), amber text links for actions, no boxes.
- Settings: transparent and blurred over what's behind (no cover), the whole width in two columns, scrolls on a
  small window ("make it a transparent background and the whole width"; the list had been cut off, unable to scroll).
- Pause works during a count-in too ("I don't want to wait for the countdown to pause again"): pauseGame cancels it
  (`game.countToken`, checked by the start and resume count-in loops); Resume counts in again.
- In-game, "more modern, keep 3D": tried the song's blurred cover behind the board (user: "I don't like the cover in
  the back"); a concert with a crowd was declined ("no crowd, just lights in the back"); cone-shaped light beams
  looked "cheap ... like a flash game". Now: soft out-of-focus glows (bokeh, `buildShow` / `updateShow` in
  game3d.js) and a faint glow far behind the board, drifting slowly, brightening on the beat (`beatAt` from the
  bars and `bpm`), coloured by the mood (dim red / amber / blue at x4). Also: strings glow faintly in their colour,
  the string names sit on glass chips, the frame fades to dark at the edges, the call pops in sliding up.
- The light show made livelier ("just more dynamic"; real concert footage was offered and not wanted): three rows
  of soft lights like rigs at different depths, swaying; a wave chases across them at each bar; rows flash in turn
  on the beats; each hit flares a front light in the string's colour; brighter with the multiplier (`mult`); x4 blue
  with a white flash on beat 1. Streak banners moved left (W * 0.36) so they don't overlap the call by the score.
- Holding a note ("make the effect of holding a note look better"): the tail burns bright and pulses with a soft
  additive glow (`tailGlow`), the note glows on the pad, the string vibrates and glows (`lanes.strings`), small
  sparks stream off the pad, a burst when the hold reaches its end, the pad pulses. Not yet seen with a real hold.
- Repeats: the first note has its number; a repeat (the very next note played, same string and fret, within 2.5 s;
  notes on other strings in between make it a new note) is the thin bar ("I wanted the first note to have a number and
  any repeating to have the thinner bar"; "there's not repeating notes that are not showing").
- Board surface: dark glass by default ("not huge on the fretboard look" → "dark glass highway"): satin (a mirror
  finish caught the lights: "too shiny"), faint lane washes, glowing thin strings, light edges, soft light bar
  lines; Settings › Fretboard: Glass | Maple | Dark, stored as `encore-fretboard` (new key, so everyone starts on glass).
- A comment added inside a one-line `if { ... }` once commented out its closing brace and broke the whole page
  ("Loading…" forever): re-check every inline script after each index.html change.
- String names drawn on the HUD (`Game3D.nameScreen`), crisp: just the letter in the string's colour with a soft
  shadow (circles round them were disliked) (as textures in the scene they looked "sloppy"); the 3D name sprites stay hidden, only for the camera fit.
- "Higher res": 4x MSAA on the glow pass's render targets (EffectComposer had none: jagged edges), pixel ratio up to
  the screen's (3), note / bar number textures 256 px.
- Settings was stuck 500 px wide (an older `#settings-dlg { width }` rule outranked the full-screen class: "too narrow to
  work"); `#settings-dlg.full` sets the full window size.
  Also: the list grows to its height and the screen scrolls (a height-limited multi-column list spilled sideways off
  screen), two columns from ~1000 px wide, one below; sliders / switches amber (`--accent` set on #settings-dlg).
- Tuner in Settings ("can we add the tuner to the settings?"): Setup → Tuner → Open tuner moves the input drawer's tuner
  (#tuner) into Settings (connecting the input if needed) and switches it on; closing puts it back and switches it off
  if Settings switched it on (`settingsTuner`).
- "Match the song's tone" defaults to on in Encore (/game) for songs never set; switching it off is kept per song (0).
- Cleanup batch (2026-10-31; user's list, easiest first):
  - Song select slid up and off the right edge (list view "right side is getting cut off"): `.arcade` had
    `overflow: hidden`, which still scrolls on focus / scrollIntoView (the blurred background reaches 80 px past the
    edges), so it sat scrolled 68 px right, 72 down. Now `overflow: clip`.
  - List view "too much empty space": the picked song's big cover tops the right panel (`#arc-now-art`, list mode
    only) and rows have a best-score column (`.arc-row-best`).
  - Track Splitter into Encore (2026-11-06; user: "tired of moving back and forth between my encore and track
    splitter", wants Add songs and tabs in Encore plus a Guitar Pro download, then "I would just like to move track
    splitter into encore. Using the same styling"). A tab-page URL -> .gp converter was asked for and declined
    (Songsterr / UG terms); pasted text or a file the user brings -> Import tab -> .gp5 instead. Plan: 1 mockup,
    2 Library + player bar + Add songs in Encore, 3 tab view + recording, 4 remaining dialogs and retiring the old
    screen (old screen keeps working until then). Mockup: `design/game-mockups/library.html` (title menu Play ·
    Library · Add songs · Settings; Library = song list with versions / tab tags, Artist sort as headings in place
    of the sidebar, picked-song panel with versions, Trim, tab actions incl. .gp5, Play in Encore; glass player bar;
    job strip; Add songs as a full screen). Real song list baked in; covers load from the running app (`?port=`,
    default 7860). Not wired to anything.
    Practice in the game added to the mockup (user: "it would be nice if we could have a practice mode in the
    actual game and move around the song in there too"; the long-planned phase-3 practice mode): Practice on the
    title menu and the Library's Practice button open the board with no score or fail; a timeline of parts (bars
    that repeat get the same name; placeholder names in the mockup) with note density, a draggable playhead and loop
    ends, click a part to loop it; ←/→ bar, ↑/↓ part, [ ] loop start/end, Space; speed 50/75/90/100% (+5% after a
    loop with >= 90% clean, "Speed up when clean"), Wait for me (the song holds at each note until it's played),
    Click (beat + a counted-in bar each loop); sides show loop number, clean % per loop as dots, trouble spots;
    "Play it for score" goes to the normal game. The mockup's board is a flat canvas stand-in for game3d.js. To build:
    slowed audio needs the server (ffmpeg atempo, cached like `shift`), part detection for automatic tabs is new.
  - Practice built in the real game (user: "I want the same look at the real game though", "Yes, build it"). Entry:
    Practice on the title (`#arc-practice`) or Play | Practice by the Play button (`#arc-mode`, `arcade.mode`); startGame
    sets `game.pr` (newPractice) and hands off to `startPractice` (no 3-2-1: `practiceGo` counts in a bar of clicks when
    Click is on, `game.countdown` shows the beats). No score, multiplier or record (`multiplier()` is 1, the HUDs skip the
    score, endGame returns before results / bests / game checks); the calls (Perfect/Good/Miss) stay. Strip `#pr-hud`
    (fixed, 112 px; `body.game-on.practice .tab-body` stops above it): canvas timeline (note density, bar numbers, loop,
    playhead; click = move, drag across bars = loop, drag a loop end), transport, status / loop dots (green >= 90%,
    amber >= 70%) / most-missed bars, speed 50/75/90/100, Speed up when clean (+5% after a >= 90% loop), Wait for me,
    Click; kept in `encore-practice`. Moving (`practiceSeek`) stops, clears results from there (`practiceReset`) and
    counts in again. Wait for me stops the audio on the note's time and `practiceHeld` (the listen loop's not-playing
    branch) takes a pluck plus that pitch (any pluck for a dead note, any chord note for a chord), then plays on.
    Speed: `/api/audio?speed=` (50-100) through `shifted()` (atempo, chained below 0.5x for raised songs; cached in
    SHIFT_DIR); the page swaps in slowed buffers and `player.rate` (song s per audio s) scales `currentTime`, `duration`,
    `play`'s offsets, `tabTime`'s output delay, and gameListen's calibration / reading delays (`sc`). Light cues fall
    back to none when slowed. Parts (Verse/Chorus) not done: ↑/↓ move 4 bars.
    Tested on a library copy (port 7871, Hysteria, simulated input: an oscillator stream): practice starts on both
    boards, keys / loop / speed 75% (107 MB WAV, ~0.5 s server) / looping with loop count and dots, Wait for me waited
    on each note and went on after the right pitch (a dead note after any pluck). Not tried on a real bass yet.
  - Stage 2, the Library in Encore (user: "Yes, start the Library"; mockup design/game-mockups/library.html). Title menu:
    Play · Practice · Library · Add songs · Settings (Add songs opens the Library, then Track Splitter's upload dialog).
    `#arc-library`: every song (with or without versions), Recent / A–Z / Artist (artists as headings with ⋯ =
    artistMenu, in place of the sidebar; `encore-library-sort`), search like Track Splitter's (instruments too); a
    row: cover, title (▶ / ❚❚ on the loaded one), versions, tab tags, added. ↑/↓ pick, Enter listens, double-click
    listens, Esc back; a refresh comes back to it (`encore-place`). Panel: cover, title, ⋯ = songMenu (edit, trim,
    import, delete); Versions (click listens / play-pause, Redo, Trash, Trim, + New version = openUpload(song)); Tab
    with Bass | Guitar: Make the bass tab (`libMakeTab`, like song select's), Import tab (openImport), .gp5;
    Play in Encore / Practice open song select with the song picked and the mode set. Job strip from state.job
    (`renderJob` calls `renderLibJob`; Stop is a `.stop-job`). Player bar mirrors Track Splitter's (togglePlay, seekTo,
    state.mode + applyMode, the master volume through `#volume`); hooks in renderPlayer / refreshLibrary re-render
    it (guarded by `state.libReady`, set once the module has run). Add songs / Import tab dialogs over Encore:
    `body.arcade dialog.fancy-dlg` swaps the violet for white and Encore's type (their full redesign is stage 4).
  - .gp5 download: `GET /api/tab/gp5?song&tuning&source&instrument` takes the tab as /api/tab gives it (the page sends
    the tab view's saved source and tuning) through `tabimport.write_gp5`: the tab's bars, beats per bar from the
    tempo (n/4), notes on the nearest 16th, held notes tied over, rests in gaps, dead notes dead, "h"/"s" legato as a
    hammer / slide on the note before; one tempo for the song. Read back with read_gp5: a test bar and Hysteria's
    imported tab (1314 notes -> 173 bars, 1315 incl. one tie). Not opened in Guitar Pro itself yet.
  - Tested on a library copy (port 7871): Library at 1512x900 and 1024x768, search / sort / artist headings,
    listening (player bar time, mode, play / pause), Add songs and Import tab dialogs open over it, Play / Practice
    to song select, the title menu, Track Splitter's page. Not tried: a real upload / split or Make the bass tab from
    the Library (the job strip has only been seen in code), and the stage 3 bits (Open tab, Beat 1).
  - Stage 3, the tab view in Encore (user: "Yes, start stage 3"). Not rebuilt: Library › Open tab (`libOpenTab`) loads
    the song (the no-bass / no-guitar version, `gameVersion`), sets the tab instrument, opens Track Splitter's tab view
    and adds `body.tab-encore`, which hides #arcade, the sidebar, the song list's header, the tab view's Game section
    and its ✕, shows a ‹ back button (`#tab-back-lib` -> `closeEncoreTab`: closes the tab, back to the Library) and
    restyles everything through Track Splitter's own CSS variables (dark, white for violet, Encore's type), the song's
    cover blurred behind (`--tab-cover`), the player as a glass bar, pill buttons, the Bass input drawer centred on
    the window (`--side-w: 0`). Encore's keys stand aside while it's open. Everything in it is the tab view as it was:
    Imported / Automatic / Draft, Lines / Highway / Whole tab (+ Scroll), tuning, View, Beat 1 ◀ ▶, Edit, Bass input
    (Hear myself, Tuner, Record), More (takes, import, redo, repeats). Colours drawn in canvases (a few violet
    accents, e.g. the line playhead) are unchanged.
    Tested on the copy (port 7871): Open tab from the Library, Lines / Highway / Whole tab, the Automatic source with
    Beat 1, the Bass input drawer with a simulated input, Back to the Library; no console errors. Not tried: a
    recorded take or editing notes inside Encore (same code as Track Splitter's).
  - Stage 4 (user: "Yes, start stage 4"): the dialogs and retiring the old screen.
    - Track Splitter's colour variables are swapped for all of Encore on `:root:has(> body.arcade)` (on :root because
      canvases read them from document.documentElement): dark panels, white text, and Encore's warm amber `#c08457`
      as the accent instead of violet (white wasn't usable: the playing note in Lines is the accent with white text).
      Colours written as violet instead of the accent are overridden under body.arcade (Lines playhead, playing
      note glow, slide / hammer marks, segmented buttons, instrument rows, repeat choices, the chosen import file)
      and in a few canvases (Trim's waveform, Scroll's held-note tails and legato marks, the tab game's progress).
    - Every Track Splitter dialog in Encore (`body.arcade dialog:not(.game-screen)`: Add songs, Import tab, Trim,
      Your takes, Make repeats match, Edit song & versions, prompts) is a glass panel over a blurred screen like
      Settings, light 24 px headings, spaced capitals for labels, pill buttons with the main one white; menus and
      toasts in the same glass. Markup unchanged.
    - Retired: `GET /` redirects to `/game` (so the old Track Splitter.app shortcut and bookmarks open Encore);
      `/index.html` still serves the old screen for checking. Encore's title link "Back to Track Splitter" is now
      Quit (asks, then POST /api/quit). The Library got "Open folder" (the sidebar's). Encore's texts that said "in
      Track Splitter" now point to the Library. The old screen's code stays (the tab view, dialogs and player are
      shared); the Mac shortcuts are unchanged (both now land in Encore; untested on the Mac).
    - Tested on the copy (port 7871): / -> /game (307), Trim, Edit song, the song menu, the Quit prompt and the tab
      view in the new colours (playhead amber), no console errors.
  - Tab view in Encore, readable and tidier (user: "its hard to see the tab view with the current background. Also, can
    we clean it up?"; Hysteria's grey cover washed it out): the cover is only a faint tint (brightness .16, a dark
    vignette over it), the tab sits on one solid panel (#0c0c11, `.tab-stage`), notes' boxes are the panel's colour so
    they cut the strings, strings / bars / numbers brighter, the current line a faint band, the next one dimmer.
    Toolbar: segmented groups as Encore's text tabs (spaced capitals, white underline), ghost pills for the buttons
    (Edit no longer white), the tuning as an underlined select, Bass input wide enough (`#tab-input` 13.5em). Selectors
    are `body.tab-encore.arcade` where `body.arcade .segmented button.active` (white fill) would otherwise win.
  - Getting around (user: "maybe a way to get there [the tab view] from the song select screen?", then "I don't want to
    have to go back to the main menu to get to the library screen every time"): song select has View tab (`#arc-tab`,
    or T; `selectOpenTab`) and a Library button (`#arc-to-library`, or L); the Library has Play (`#lib-to-select`, or P;
    `libToSelect`), each keeping the picked song. The tab view's ‹ goes back where it was opened from (`lib.tabFrom`).
    Back / Esc on song select and the Library go to the screen you came from (user: "when I click back from the library
    when I came from song select, it brought me back to the main menu"): `arcade.backTo` per screen, `screenBack`;
    going back clears the other's way back, so a second back reaches the title instead of bouncing. Checked:
    title > select > library > (back) select > (back) title, and title > library > select > (Esc) library > (Esc) title.
  - Keyboard focus lines (user: "I get the white lines when I use keyboard"): Chrome outlined the button or row last
    clicked once a key was pressed. No focus outlines anywhere in Encore now (`.arc-screen`, the tab view in Encore,
    the practice strip), and after a mouse click on a button the keys go back to the screen (`#lib-list` in the
    Library, the song list on song select, the page in the tab view; not when a dialog opened), also when the panel
    was drawn again under the click. Space no longer re-presses the clicked button. Checked with real clicks and keys.
  - Only the Arcade board (user, mid-build: "get rid of the old game highways and just use arcade"): removed
    drawGame3D, drawRockHud, the Board / Fretboard / Hand guide / Big moments settings (3D-only), the game3d.js script
    tag and its CSS; `drawGame` sends every non-chord game to `drawArcade`. Guitar games with chords still use the 2D
    board (`flat = false` path) because Arcade can't draw chords yet. game3d.js, board-lab.html and the hand-move code
    (`handMoves`, used by the 2D board) are left in place. The old `track-splitter-game-board` pref is ignored.
  - List view "all jank" (2026-11-06): the list-mode grid's middle row was a plain `1fr`, which grew to the whole
    list's height, so Play and the key hints sat below the window (page ~80 px taller than the screen). Now
    `minmax(0, 1fr)`; the right panel scrolls if it still can't fit, hides the empty song note, and the cover is 24vh
    (16vh under 820 px tall). Rows: fixed tag (128 px) and score (116 px, `.arc-row-score` = best + record) columns so
    played and unplayed rows line up; 44 px cover without the inner padding.
  - Long names "cut off in the cover menu": `fitText(node, min)` shrinks the song title (and Jump back in's) until no
    line is hidden (2-line clamp kept below the minimum); drawn covers use a smaller font past 24 characters.
  - Logo: the sidebar card is a dark card with the pick mark (inline SVG, `.encore-mark`) and the gold wordmark (was
    a play button on a flat amber slab); the title screen uses the same mark without the app-icon tile.
  - Note speed slider: value on the right ("1.25×", `#gs-speed-val`) like the other sliders, no Slow / Fast words;
    double-click = 1×.
  - In game, floating text ("no pills, transparent background with floating text"): score and stars without the
    frosted panel, the multiplier as a bare ring (the call sits left of the score, measured), no box behind Resume /
    Play again (an older `.game-screen .btn.primary` shadow), results stats without tiles.
  - Bigger board tried (camera fit so the board spanned ~70% of the height, note speed scaled to what's on screen):
    user: "I liked the board when it wasn't so huge", reverted.
  - Picking, from the 41 game checks (no code change yet): in the recent games most misses are "plucked, but its
    pitch was never heard" (820 of 1701 overall); of 462 since 10-26, 205 heard only the neighbouring notes (274 of
    the 462 within 0.2 s of the next note), 206 other notes, 30 nothing. Tuning isn't it (readings median +3 to +27
    cents). Look What Happened's swing (+88 ms early part, -81 ms later) is not the tab: its imported notes sit
    within ±10 ms of the heard notes all song (only ~55% of them match a heard note at all, sync quality 0.37).
  - Purple on hover in Encore ("why is there a purple background when i hover on things?"): the app's
    `.btn.primary:hover:not(:disabled)` purple outranked the game screens' button styles (pause, results, settings,
    calibration). Hover rules for `.game-screen` / `.arc-screen` buttons now stay amber / dark, and the full screens'
    text buttons stay bare. Checked by copying every :hover rule as a class rule (the browser pane doesn't hover).
  - Results: title, artist and run type on one left edge (the run type kept its old pill padding).
  - Difficulty chips: stars and best moved a little lower; the max-height 820 rule no longer squeezes chips to 38 px.
- Fast-note misses, tried and dropped (2026-10-31; user: "only keep it if the wrong-notes control stays clean"). Replay
  kept in the repo this time: `tools/gamecheck_replay.py` (judge in Python on run.json + input.flac; reproduces every
  game within 0-3 hits) and `tools/gamecheck_experiment.py` (pitch reader variants; control = tab moved +2/-1/+1/-2).
  The frames' audio lines up at sample round(ctx * rate) - first_frame (49/50 readings identical). 24 games (8253 hits):
  - A second reading from only the sound since the last attack (40-200 ms after it, from 20 ms before it, YIN clear
    0.1), added beside the normal one: +29 real, control +2/+7/+4/+5. Stricter (60-200 ms) +8 vs +1/+4/+1/+1; looser
    (clear 0.13) +47 vs +6/+15/+9/+10. Never inside the ±2 bar.
  - A pick sets off several pluck signs per stroke (click, newest 20 ms, level), 20-70 ms apart, and `ownPitch` treats
    them as "the next pluck", blocking the stroke's own pitch (the new Thank You for the Venom pick game, 511/549:
    19 misses "pitch only heard before the pluck"). Not blocking within 30 ms: +27 real, control +26/+10/+11/+11;
    70 ms: +48 vs +59/+34/+26/+28. Dropped.
- Flower Power Madness "hard to read" ("this was more an issue in game mode"), 2026-10-31:
  - Repeats: once a run's numbered note reached the pads, the rest were bare ticks with no fret on the board. Now the
    tick right behind it carries the number (`lead` in game3d.js), so the coming fret is always shown.
  - Bar lines (imported tabs, IMPORT_RULES 11): `_place_in_bars` shifted each bar's notes to fit what's heard but
    left the bar line on the 50 ms sync grid, so a bar's first note sat up to ~100 ms before its line (drawn at the
    end of the bar before; the same riff bar came out as 8 notes or 9). Bar lines now move with their notes. Flower
    Power Madness: downbeat notes before their line 43 -> 1 of 83; every imported tab on the test copy 22-89 -> 0
    (Number of the Beast 89 -> 33, Amsterdam 37 -> 4), median offset ~17 ms -> 0, notes unchanged.
- Game performance ("can we also improve performance?"): the scene update takes ~1.5 ms; the frame rate was limited
  by pixels drawn (browser pane, 1440x900 at 2x: 21 fps; a quarter of the pixels: 44). The glow pass now runs at a
  quarter of the pixels, and the sharpness follows the frame rate (`keepPace`: steps of 0.25 down when frames average
  slower than ~45 fps over 2 s, back up while at the screen's rate but never to a ratio that was too slow; first 3 s
  ignored). Browser pane: settles in ~5 s at 46-50 fps. Not yet measured in Chrome on the M4.
- Holds ("can we improve how held notes look? Also, I always lose it at the end"), 2026-10-31:
  - Game checks: 431 held notes hit; the pitch was last read a median 62 ms before the end, a quarter 0.2 s+ early;
    in the last 150 ms the non-matching readings were mostly unclear (1036) vs another note (297) / under the gate
    (253). Now (`judge`, `game.hold`): a hold survives unclear readings up to `HOLD_GAP` 0.25 s, is lost on another
    note or silence, and held into the last `HOLD_GRACE` 0.2 s (or quarter of the note) counts to the end. Replayed on
    the frames: held to the end 261 -> 349 of 431.
  - Look (game3d.js): the tail burns only while really held (`v.hold[k].lost` greys the rest), wider on long notes
    (wider still while held), and an end cap (`cap`) marks where to let go, lighting up over the last 0.4 s. Not seen
    with a real hold: a simulated input (the song's bass-only track fed in through a hooked AudioBufferSourceNode.start)
    didn't register as playing in the browser pane.
- Song select ("improve the look of the spotlight over the cover and change the play button so it matches the rest of
  the UI"): the light triangle above the picked cover and its white + amber double ring became a fine edge, a warm glow
  behind it and a pool of light under it (`.arc-carousel::before/::after`, so they stay put while covers slide). Play
  (song select and Jump back in) is floating amber text with an arrow, like the pause menu.
- Smoothness, 120 fps, the dark triangle, lights from the song (2026-10-31):
  - "Skips for a tiny second": the shown clock (`audioNow`) never went back, so whenever the audio clock said it was a
    little ahead it stood still; now it eases (up to 10% slower / faster) and only jumps when > 0.1 s off. game3d.js
    also made textures mid-song (each bar number as it came into view; a note's grey number at the moment of a miss),
    grew its note pool mid-song (6 new materials a note) and made a new mesh + material per spark; now `prewarm`
    makes every bar / fret texture (also the grey ones) and 60 notes + 40 sparks before the first frame and compiles
    the shaders with everything shown once; sparks are pooled; no per-frame colours.
  - 120 fps ("can we make the game run at 120fps?"): `keepPace` aims for the screen's own refresh, measured once while
    the menus are idle (`measureScreen`; during a game a 120 Hz screen showing frames a refresh late looks like 60 Hz);
    steps sharpness down by 0.25 while frames run 15%+ slower than the screen, measured against where the steps began,
    and goes back if the bottom gave no gain. Browser pane at 1440x900: settles in ~5 s at a steady 120 fps (frames
    8.3 ms, slowest 9 ms) at pixel ratio 1.0 there; at 800x600 120 fps at full 2x. Not measured in Chrome on the M4.
  - Dark triangle ("a weird dark static triangle going across the screen"): the far glow behind the board (z -80, ~96
    from the camera) crossed the camera's far plane (100) and was cut along a straight diagonal; far plane now 300.
  - Lights from the song ("could we use the songs audio track to queue flashing lights?"): `lightCues` reads the
    version the game plays once (3 s of audio per slice) into 10 ms steps: kick hits (< ~150 Hz jumping over the last
    50 ms, above the song's 90th-percentile jump, 6 dB min), snare / cymbal hits (the same on the sample-to-sample
    change) and loudness; `cuesAt` per frame. game3d.js: kicks pulse the far glow and flash the front rig, snare hits
    flash the other two rigs in turn, loud parts brighten the show; the tab-beat flashes are the fallback while the
    cues are being read. Flower Power Madness: 332 kicks a beat apart (0.485 s at 123.7 bpm), 288 snare / hat hits on
    beats and offbeats. Not yet seen while playing well (the pane's test games miss every note, so the show is dim).
- Lights, more dynamic ("instead of going left to right everytime and make big moments look bigger"; "when things go
  quite, dim the lights"), 2026-10-31:
  - `PATTERNS` in game3d.js, one per bar (`pickPattern`, never the same twice running, weighted by loudness): sweeps
    left / right, from / to the centre, front rig to back, sparkle (random lights on each hit), alternate (odd / even on
    each hit), strobe (everything on the kicks; loud parts only). Rigs sway more when it's loud.
  - Big moments: `lightCues` finds the loudness jumping 0.15+ (of the song's range, 2 s before vs 1 s after) into a
    part louder than 60% of the song, strongest first, 15 s apart, one per 30 s at most, on the kick that starts it
    (Hysteria from its first chorus at 0:59.6; a 0.2 bar into the top 45% found none in Flower Power Madness). Accents:
    a kick and a crash together, both among the song's strongest. The game's own: the multiplier going up, streak
    milestones (`moment`). On one, every light flares white and grows, the far glow floods and grows, the bloom swells
    (`mats.bloom`), settling over ~1.5 s; accents are a smaller version.
  - Quiet parts: the show is scaled 0.25 + level^1.2 (a quarter in near silence, half in a quiet verse, ~1.25 at the
    loudest); the board's own lights unchanged.
  - Checked: no errors, 119 fps in the browser pane. Not seen with real playing (the test games miss every note, so the
    show sits in its dim "struggling" mood).
- Pop and big notes ("louder moments to have some real pop"; "the end of flower power madness, those should be
  considered big notes"; "I don't like the screen shake"), 2026-10-31:
  - Loud parts (from ~0.55 of the song's loudness): each kick punches: a quick zoom (`pop`, fov -2.5%), lights flare
    harder, the far glow and the bloom jump. Big moments add a 5% zoom punch, a warm white flash over everything (a
    plane in front of the camera, ~0.15 s) and pyro (`pyro` / `updateEmbers`: four fountains of sparks behind the
    board, ~1.6 s). A screen shake was tried and removed (user didn't like it).
  - Big notes: stabs (a loud hit surging 5 dB+ over the 0.4 s before and falling 6 dB+ within 0.15-0.4 s, on 150 ms
    averages: 10 ms steps called every drum hit a stab) count as accents: lights, bloom and zoom, not the white flash
    (on every stab it washed the board grey). Flower Power Madness: its ending (3:03-3:05), intro and 1:37-1:51, 112 in
    all. Full stops (a loud hit with near silence 0.6-2 s later, 12 dB+ down) are big moments: its last hit at 3:12.3
    and a break at 1:02.6; none in Hysteria or American Idiot.
  - Vocals ("sync it up with lyrics where they go oh ah"): not done; the game only has the no-bass mix, where singing
    can't be told from guitars. Would need a vocals separation per song (see the session's question).
- Light cues on the server (user: "separate the vocals once per song ... and sync the lights to the long sung notes";
  "can we do all this during upload?"; "really, we should just take the whole songs track and look at that"),
  2026-10-31: `lights.py` (module docstring has the method), saved per song as `.light-cues.json` (`LIGHTS_RULES`,
  `VOCALS_RULES`). Made in `make_version` from the mix and the vocals the separation produced anyway
  (`separate(keep=)`: any Demucs model's or RoFormer's vocals); otherwise `GET /api/lights` makes the song part on
  the spot (~1 s; whole song = the original, else a version's two halves added up) and queues the vocals for
  `vocals_worker` (htdemucs, ~1-2 min a song, one at a time; `gpu_lock` in `run_model` keeps it from running beside a
  song being processed). The page asks when a song is picked in song select (`songLights`) and uses its own reading
  of the no-bass version only until they arrive. Whole song vs no-bass version (Outside): kicks 1.0 vs 1.5 a second.
  - Held vocal notes: YIN on the vocals (dip < 0.35; 0.15 gave a pitch to 27% of Flower Power Madness's sung frames,
    0.35 56%), within a semitone (vibrato broke 0.6), 0.5 s+. Flower Power Madness has one, the high G at 2:24.4
    (1.9 s); a pitch-free test (steady loud voice 0.6 s+) found the same and nothing else. Look: the back two rigs fill
    with a gold wash spreading from the middle over the note, the far glow turns gold, a 0.4 s fade after; a vocal
    phrase starting brings a soft glow from the middle. Upload path checked on a 30 s clip (vocals kept, the held note
    found at the same place). The gold wash itself not yet seen (the test games sit in the dim mood).
  - Calmer ("outside for example flashes way too much"): kick and snare flashes softer, a loud-part punch at most every
    0.9 s, strobe only in the loudest parts and rarely, x4 white flashes on accents only.
  - The board never moves ("I don't want the board to move at all during big moment, just the lighting"): the zoom
    punch is gone; big moments keep the lighting, the white flash (softer, 0.2) and the pyro.
- Quiet parts calm ("outside still flashes a ton the begining when it is a quiter part"), 2026-10-31: the cues were
  calm there (Outside 25-65 s: loudness ~0.4, 1-8 hits per 5 s, no accents or big moments); the show wasn't. The
  sparkle pattern (a flash on every hit) was the likeliest pick in quiet parts, and each of the player's notes flared a
  front light at full strength. Now `drive` (0 below ~0.35 of the song's loudness, 1 from ~0.65) scales every hit
  reaction (kick / snare flashes, the hit-driven patterns, the player's flares down to a quarter); quiet parts only get
  slow, dim sweeps (sparkle / alternate / strobe have no weight there).
- The vocals pass shared the GPU with the game: the board fell to 16 fps while Outside's vocals were separated. Encore's
  5 s check-in now says when a game is on (`/api/ping?game=1`, also sent at the game's start), and `vocals_worker`
  waits until no game for 12 s, pausing mid-song through the model's progress callback (`game_on`). Checked: during a
  game the server sat at 0% CPU and the board at 81 fps (rising as it settled); after it, Outside's vocals finished
  (9 held notes, e.g. 1:07.3 and 2:06.9 for 1.7 s, 5:02.2 for 2.4 s).
- Big moments off for now ("can we just remove the big moments and come back to it later. makes it harder to play
  right now"), 2026-10-31: `BIG_MOMENTS = false` in game3d.js turns off the blasts (song big moments, multiplier /
  milestone moments), the white flash, the pyro and the stab accents; lights.py still saves `big` / `accent`, so
  setting it back to true brings them back. The loud-part lighting punch, patterns, quiet dimming and vocal wash stay.
- Light check page and a Big moments setting (user: "build the big-moments check page so I can rate them, and use my
  ratings to tune the detection"; "make it a setting in the menu"), 2026-10-31:
  - Settings › Big moments: On / Off (`#game-big`, `encore-big-moments`, off by default), passed to game3d.js as
    `v.bigMoments` (replaces the `BIG_MOMENTS` constant). Settings › Light check › Open… opens the page.
  - `static/lights-check.html`: pick a song; a timeline of its loudness with the detected big moments (gold), stabs
    (violet), held sung notes (gold, along the bottom) and the user's missed marks (cyan); a list with Yes / No /
    Bigger (stabs that should be big moments). Click a mark to hear 2.5 s before to 3 s after (the version's two halves
    played together: the whole song without a 60 MB WAV); click elsewhere to play from there. Keys: Space, ← →, Y, N, B,
    M (a missed one at the playhead), Delete. Saved as you go: `POST /api/lights/ratings` -> `<song>/.light-ratings.json`
    (`GET` reads it back).
  - Tuning: lights.py's thresholds are `lights.BIG` (jump, loud_pct, gap, per_s, stop_db, stab_fall, stab_rise);
    `song_cues` = `features` (slow, ~0.1 s a song) + `decide` (4 ms). `tools/lights_tune.py` reads every rated song,
    runs a grid of 2,187 settings and ranks them: wanted big moments found (yes / bigger / missed) minus unwanted
    (no) minus half of each new unrated one, plus half the stab score. Checked with made-up ratings on one song (11 s);
    no real ratings yet. Once there are some: run it, pick the best settings into `lights.BIG`, bump `LIGHTS_RULES`.
- Cooler board, mockups (user: "can we make the board look cooler?", picked glass reflections, living strings and new
  strike pads; "lets do mockups"), 2026-11-01: built straight into game3d.js behind `v.look` {reflect, strings, pads}
  (all off in the game for now) and shown on `static/board-lab.html` (not linked from the app): the real board
  auto-playing a made-up bass line at 120 bpm, switches for each look, Old / All, x4 and 5 strings; `?look=...&fire=1`.
  - reflect: a coloured streak on the glass under each note (`refl`, fading away from it) and a pool of the pad's
    light in front of each pad (`pool`), brighter on hits and as notes come in. Sprites, not a mirror pass (that draws
    the scene twice).
  - strings: in the strings' shader (`livingString`, 400 segments): a ripple travelling out from the pad when a note
    lands (6 units/s, fading over ~1 s), and a pulse of light running down every string to reach the pads on each beat
    (beat 1 brighter), blue at x4.
  - pads: a glowing ring + soft core that fills with the string's colour over the last 0.6 s before a note (`coming`),
    a ring of light bursting outward on a hit (`burst`), red on a miss, white for what you're playing; the old lens and
    rim hidden.
  - Browser pane, everything on vs the old look: 77 vs 79 fps. A first try declared `pulse` twice in `frame` and the
    whole module failed to load (the game would have lost its 3D board): renamed `beatPulse`.
- After the mockups ("I like the old pads, the strings are pretty distracting though. Can we make them look like real
  strings?"), 2026-11-01: the new pads and the living strings (ripples running along, beat pulses) are removed;
  `look.strings` is now real strings (`realString`): graded 0.075 -> 0.04 thick (low string ~2x the high one), nickel
  (#9c968c, metalness 1, roughness 0.42, reflections at 0.55) with a round-wound bump map (`windTex`, 700 turns along
  the string), the string's colour at 12% of its glow, and a short shimmer near the pad when a note is played (~0.4 s;
  a bright #d8d2c8 finish set off the glow pass and read as white neon). The lab page now has Reflections / Real
  strings. Not yet in the game (`v.look` isn't passed there).
- Minimal look (user: "the strings look a little weird. Would a minimalist design work better?"), 2026-11-01:
  `look.minimal` in game3d.js: strings as thin flat matte lines (`lanes.lines`, 0.04 -> 0.028 wide, the string's
  colour at 80%; the tubes hidden), a moment brighter when played and while held, blue at x4; a plain matte board
  (#0c0c10, no texture, no light edges, lane bands at 1-3%); flat matte notes (roughness 0.9, no clearcoat, the
  string's colour at 80%). Old pads kept. Lab page: Reflections / Real strings / Minimal (one string style at a time).
- Reading along (user: "I'm still struggling with reading along while playing"; picked fret numbers, not enough
  warning, fast runs / shifts), 2026-11-01:
  - Fret numbers drawn upright on the HUD (`v.hudNumbers`, `Game3D.labels()`, drawn in `drawRockHud`): each note's
    number projected to the screen, outlined white (open strings in the string's colour, misses grey), 17-34 px by
    the note's on-screen width; the flat numbers on the notes are hidden (squashed by the board's lean, tiny far off).
  - Look-ahead 3.4 s at 1x note speed (was 2.6: `Z` in drawGame3D); notes reach full opacity within 1.8 units of
    coming on (FADE was 3.5: far notes stayed faint).
  - Hand moves: `handMoves` also keeps `d.handTargets` (the fret starting each move, up or down); the HUD shows
    "↑ 6" / "↓ 3" in amber above the dashed line before each move.
  - Checked in a game (Flower Power Madness): 11 numbers on screen, all readable to the far end; "↑ 6" over the line
    before the move to the 6th fret. Not yet tried with real playing.
- Hand guide (user: "mock up the hand-position guide (a small neck by the pads showing where my hand should be and
  sliding before a shift)"), 2026-11-01: `drawHandGuide` in the HUD, bottom left under the board: a neck to one fret
  past the song's highest (16-24), frets narrowing like a real neck, inlays at 3 5 7 9 12 15..., low string at the
  bottom; an amber box over the four frets the hand covers (`d.handPos` from `handMoves`: [time, first fret]); the next
  1.5 s of notes as dots in their string's colour (fading with time, the next one ringed white); within 1.5 s of a
  shift a dashed box at the new place with an arrow, the box sliding over in the last 0.25 s. Settings › Hand guide
  On / Off (`encore-hand-guide`, on). Checked in a game: box and dots follow (frets 3-6, then 10-13 for the high
  part); the slide itself not caught on screen. Not yet tried with real playing.
- Hand guide off by default ("I don't like it, sort of distracting"; then "make these all settings" / "no, the hand
  position"): Settings › Hand guide On / Off (`#game-hand`, `encore-hand-guide`, off), next to Big moments. A short-lived
  Settings › Board group (strings style, reflections, fret numbers, move labels) was undone: upright numbers and move
  labels stay always on, the board looks stay on the lab page.
- Choppiness again + pick (user: "getting some choppyness again, can we optimize? Also, using a pick is still
  inconsistent. fingers work much better"), 2026-11-01:
  - Measured in a game: the HUD's upright numbers cost 0.09 ms a frame (21 text calls), the 3D frame ~1 ms; the
    hitches were the sharpness steps at the start of every song (each resizes the drawing buffers, ~40-70 ms, up to four
    of them) and the warm-up (`prewarm`, ~350 ms) landing on the count-in's "3". Now the settled pixel ratio is kept
    (`encore-pixel-ratio`) and the next game starts there, and `startGame` draws the board's first frame before the
    count-in. Checked: the warm-up now lands 0.4 s after the game screen appears, before "3".
  - Pick, tried on the replay (all 25 games, wrong-notes control +2/-1/+1/-2), both dropped:
    - a cool-down on click plucks (like `plucked`'s 90 ms): real hits 9124 -> 9119 / 9105 / 9084 at 60 / 90 / 120 ms,
      Venom (pick) 508 -> 505; the clicks are often the real pluck.
    - a quieter sign within 0.12 s of a pluck not counting as "the next pluck" (a pick's rattle): Venom 509 -> 510,
      Fireman (pick) unchanged, control up as much as real hits (1 dB: +32 real, +59/+34/+21/+29 control).
    In the pick games a stroke makes 4-5 pluck signs over ~100 ms and the pitch reads clearly only after; nothing in the
    recorded frames tells that rattle from a real next note. Next step: a pick game and a fingers game of the same song,
    recorded for checking, to compare the input itself. `tools/gamecheck_replay.py` gained `quieter` / `settle`.
- Pick vs fingers on the same song (user played Outside, pick first: game checks 2026-11-01 10:49 (8 s restart),
  10:50 pick, 10:51 fingers), 2026-11-01. Over the 143 notes both judged (first 30 s): pick 137 hits, fingers 140; pluck
  signs per note 3.6 vs 3.4; own pitch read a median 80 vs 78 ms after the first sign; pick plucks 3 dB quieter (median
  -12.0 vs -8.9 dB) with the same spread (6 dB); pick timing tighter (middle half 47 vs 57 ms wide, Perfect 108 vs 102
  of the hits). The 6 notes only the pick missed: two its own timing (one 176 ms late), the rest scattered. The 10:49
  restart's "too soft" misses were signs at ~-58 dB at the start, not strokes. Thank You for the Venom (pick, 10-31):
  its 21 "no pluck in the window" misses have almost no attack in the recorded input (biggest 2 ms level jump in the
  window ~1-2 dB, hits ~10 dB), so not strokes the detector missed. No code change; next comparison would be Venom with
  fingers (fast repeated notes, where the pick games lose most).
- Pick vs fingers, Thank You for the Venom (game checks 2026-11-01 10:55 pick, 10:56 fingers): whole games 468/479 vs
  472/492; over the 479 notes both judged, pick 468, fingers 462; repeated notes 396 vs 389 of 401, fast notes
  (< 0.2 s apart) 418 vs 414 of 429. The pick's own trace: 5.0 pluck signs a note vs 4.2, and its pitch read later
  after the first sign (median 58 vs 21 ms), behind its 7 "pitch only heard before the pluck" misses (fingers 3).
  Misses only one way: pick 10 (5 of those, and 4 at the very end where the game stopped), fingers 16 (a run of 10 at
  notes 458-467, pitch never heard). Timing the same (median -25 vs -23 ms, calibration -18 ms in use); Perfect 252 vs
  271. With Outside (pick 137 vs fingers 140 of 143): no sign the detection favours fingers. No code change.
- Faster hit response and a fingering fix (user: "it would be nice if the response on the game notes being hit was
  faster and more responsive. Also, in places like thank you for the venom, it makes no sense to switch around playing
  2 on the A and 7 on the E"), 2026-11-01:
  - Pluck response: in `gameListen`, an attack while a note is due (inside its timing window; a chord's notes within
    30 ms together) sets `game.laneTapMs[string]`; game3d.js dips and lights that pad at once (0.85 of a hit's dip,
    0.6 of its light, fading over 160 ms). The hit itself (burst, score) still waits for the note's pitch: ~20 ms after
    the pluck with fingers, ~60 ms with a pick (the Venom comparison). Not seen with real plucks (test input is silent).
  - Venom's B1s: the tab file keeps them on A2, but a "make repeats match" choice ({35 x 8} -> E7) moved only its
    group's bars, so runs of B1 switched between E7 and A2 at group edges (31.45 -> 31.58 s). `same_note_same_place`
    (end of `unified_notes`): the very next note played, the same pitch, 0.03-0.6 s later, stays where the run began.
    Venom: back-to-back switches 1 -> 0 as the game gets it; B1 now E7 319, A2 11 (isolated ones).
  - The pluck response lit the note before as well ("Like a E7 triggers the A"): that note was often still waiting for
    its pitch, so still "due". Now only the note nearest the pluck in time lights, with any struck within 30 ms of it.
  - "it thinks I'm playing the A string when I'm not" (playing E7, the same B as A2): the board's "what you're
    playing" string came from string detection first (overtones, the calibration), which took E7 for A2. In a game
    (`showNote`, `trustTab`) a note matching one due now is shown on the tab's string (the game scores by pitch), and
    the "same note, other string" amber flag is off then; outside a game the readout still names the string it heard.
- Repeats written differently by mistake (user: "look at fireman and dancer, there's parts towards the end that are
  different when they shouldn't be"), 2026-11-01, IMPORT_RULES 12: the differences were in the Guitar Pro file (read
  straight from it: the same 106 bars and late variations), not the sync. Against the song's bass track: bars 83 and
  87 (2:18, 2:24) are written G1 G1 A1 B1 D2 G2 G2 E2 D#2 (E3 E3 E5 E7 A5 ...), the song plays bar 28's G1 G1 A#1 D2
  ... (an A#1 heard at 0.46 s where bar 28 has it); bars 78-79 (the fill at 2:11) really differ and the tab's notes
  fit. `tabimport._repair_repeats` (after placing a timed tab): a bar that's a near-copy of an earlier one (same length,
  up to 4 notes each way, both with notes of their own) takes the copy's notes when, among bars written the same way,
  the tab's own differing notes are never heard and in one of them at least half the copy's are (one at least) with the
  shared notes heard 70%+ (tolerance 0.1 s). Saved as "repaired" [{bar, from, at}] in the import. Tried first and
  tightened: copying a repair to every identical bar (Outside: 37 bars, many where the tab's notes were heard) and
  repairs that only add notes; then requiring half the copy's notes heard (Wow! Wow! Wow!'s bar 52, 1 of 3, left as
  written). Across the 24 imported tabs only Fireman & Dancer's bars 83 and 87 change (679 notes, was 681).
- Virtual drop D (user: "keep the E tuning, but it would come out as a D in the app"; "build the virtual drop D
  experiment (shift my E-string notes down a whole step in Hear myself and score them as D)"), 2026-11-01: Settings ›
  Virtual drop D On / Off (`#game-vdrop`, `encore-virtual-dropd`, off: it would break scoring for someone really tuned
  down). Applies when the tab's tuning is drop D with the other strings standard (`isDropTuning`: open[0] = open[1] - 7,
  open[1..] = 33 38 43; Amsterdam's [27, 34, 39, 44] isn't covered) and the tab is bass.
  - Scoring: `playOpens(d)` / `playNames` (judge, chordPluck, checkWrong, tuneCheck, tabNotesNow, the board's string
    names, the game check's run.json): in a game the low string counts as a standard E (open = open[1] - 5), so the
    tab's frets played on a standard E string score (fret 4 -> G#1 32 instead of F#1 30). run.json keeps it
    (`open`, `strings`, `virtual_drop`). The tab view still shows the real tuning.
  - Hear myself: a pitch shifter (AudioWorklet from a Blob, `SHIFT_WORKLET`, `input.monShift`, before the monitor's
    input gain) lowers the signal by 2 semitones (`VDROP_RATIO`) on low-string notes: in a game switched from the tab
    (`updateMonShift`: the latest note starting within the next 50 ms, its pitch on a standard E gives the period),
    outside a game from the pitch heard (below A1). TD-PSOLA: slices two periods long laid down 1/ratio periods apart;
    delay while shifting about two periods (~48 ms on a low E, ~32 ms on a B), dry otherwise; ~10 ms fade in and out.
    A delay-line shifter was tried first and smeared bass (E1 41.2 Hz in, 35 Hz peak, the wanted 36.7 Hz 18-24 dB
    down). Measured offline (sawtooth through a 900 Hz low-pass): E1 41.2 -> 36.72 Hz (D1 36.71), G1 49 -> 43.65, B1
    61.74 -> 54.99, E2 82.41 -> 73.40, the original 33-42 dB down; a period 1% off still lands right.
  - Checked on a library copy (port 7891): American Idiot (drop D, 118 of 365 notes on the low string) started with the
    setting on and Hear myself on: the shifter loaded in the live monitor, the game note "Virtual drop D: play the low
    string's frets on your E string (Hear myself lowers them to D)", run.json open [28, 33, 38, 43] / E A D G; the
    setting off by default; no console errors; the main app loads. Not heard with the user's real bass: how the shift
    sounds and feels (the extra ~30-50 ms on those notes) needs trying. The monitor's latency note only mentions the
    extra delay if Hear myself was switched on after a drop D tab loaded.
    User's first try (2026-11-01): "it doesn't work very well", "it seems to only work for the open D and there's lag".
    The lag is the shifter's (two periods); they then asked for a tuner on song select instead.
- Song select tuner (user: "can we just add the tuner into the song select screen and it shows up what the tuning is
  and if each string is in tune?"), 2026-11-01: a Tuning row under the input on song select (`#arc-tune`, chips
  `.arc-tune-s` per string). `songTuning(song, instrument)` asks `/api/tab` with the song's saved tuning pick and
  source (bass: the picked source, else imported; guitar: imported) and shows its label and strings; Standard when
  there's no tab ("no tab yet"), a half-step-down song that's raised ("song raised ½ step"), or a drop D tab with
  Virtual drop D on. Cached per song + instrument.
  - Reading: `listen()` sets `input.tunerSemis` on song select from the fixed gate (`GATE_DB`, like Settings' tuner),
    since the adaptive gate learns a ringing open string as background and then reads nothing. `readTuner()` (from
    `mirrorInput`, wrapped so an error can't stop the loop) takes the pitch every 20 ms; 8 in a row within 0.15
    semitone, within 2.5 semitones of an open string or an octave either side (closest wins), mark that string:
    ✓ green within 5 cents, amber within 20, red beyond with ± cents; the ringing string shows live (`.live`).
    Marks stay while the tuning stays the same.
  - Checked by feeding the page's `readTuner` the pitch of the user's open A, D, G (from
    .bass-recordings/take-20261023-232440.flac, Input 2; YIN every 20 ms): A +7, D +4, G +4 cents, each pluck starting
    ~+15-19 sharp and settling, ending ✓ ✓ ✓, E left blank. Before the octave-down check, the D string read an octave
    low for 0.3 s marked E −198. The live test in the browser pane couldn't run (the pane was off screen, ~2 frames a
    second); not yet tried with the real bass live, and the low E string isn't in that recording.
- Full glass look, mockup (user: "I think we could improve the game UI by going with a full glass look", "please mockup
  first"), 2026-11-01: `design/game-mockups/glass.html`, nothing in the game changed. Song select, in-game HUD, pause and
  results with every panel, button and chip as frosted glass (backdrop blur 28 px + saturate, a bright top edge, a soft
  sheen, deep shadow; amber kept as the accent); sliders for frost and tint; on the in-game screen a "Cheap glass"
  switch (the same panels with no real blur, which the HUD canvas could draw every frame). The notes and fret numbers stay
  off glass. Asked whether it would help performance: no; real blur over the moving 3D board costs GPU every frame
  (the frame rate is limited by pixels drawn, see Game performance), so in-game it would be the cheap kind; the menus,
  pause and results can use real blur (nothing moves behind them). Note this goes against earlier picks ("no pills,
  transparent background with floating text", "I don't like the pill look", settings "without a box around it").
- Highway looks (user: "I like the menus as they are now. can we improve the look of the highway?"), 2026-11-01:
  three more switches on `/board-lab.html` (keys 4 5 6, "All three"), `v.look` {depth, strike, gems} in game3d.js, all
  off in the game:
  - depth: the glass lit near the pads and falling into the dark (`depthTex`, a gradient along the board), each lane's
    colour brightest at the pads (`bandTex`; 0.15 read as coloured stripes, now 0.055-0.075), a groove with a faint lit
    near edge between lanes (`lanes.grooves`).
  - strike: a strike line through the pads with a soft glow (brighter on hits, blue at x4; `lanes.strike`,
    `strikeGlow`), faint beat lines between bar lines (four to a bar, `lanes.beats`), bar and beat lines lighting up
    as they cross it (bar lines now have their own material each).
  - gems: notes in the string's full colour (0.9), clearcoat 1, a soft glow of their own (0.22) and a lit underside
    (the skirt in the string's colour, glowing) instead of the dark skirt.
  - Checked in the lab: no console errors, each look on / off and at x4. The browser pane kept serving the old
    game3d.js from its cache (the strike line seemed missing): test the lab from a server that sends no-store.
- Fast notes hard to read (user, 2026-11-01: "the gems are too hard to reach when it is fast. Sometimes they are on top of
  eacother, other times it's just hard to track along", "maybe a different way to display notes"; the highway looks are
  parked: "lets circle back on this later"):
  - Why: the board's speed (game3d.js `speed`) spaces notes for the 10th-percentile gap between different notes on a
    string (`d.distinctLaneGap`, index.html) and is capped at 1.6x, so the fastest tenth can touch; the HUD fret numbers
    are at least 17 px, so far off they can run into each other. Measured on the library's 54 tabs (auto + imported):
    14 have any overlapping pairs, 0.5-8% of their note pairs (Baby 8%, Cry Baby Cry 4%, Helena 4%), mostly under
    100 ms apart. Wow! Wow! Wow!'s chromatic walk A0-1-2 is 0.12 s a note: 0.63 board units against a 0.56-wide note.
  - Mockup `design/game-mockups/note-styles.html` (+ `note-styles-data.js`: 24 s of Wow! Wow! Wow!, Flower Power Madness
    and Baby from the imported tabs), animated at real speed on a board drawn like the game's: 1 Now (box + number
    above), 2 Number tiles (the number is the note, repeats as dots), 3 Runs (notes within 0.36 s on a string joined by a
    bar, a number only where the fret changes, repeats as notches), 4 Tab lines (big numbers on bright string lines, a
    timing tick under each); S Spread fast runs (the board's speed follows the notes: stretched so distinct notes on a
    string are NOTE_W + 0.6 apart, held over ±0.4 s and eased over ±0.4 s, up to 2.2x; notes still land on time).
    Tiles sized to the note width read small up close. Serve it from a server (the pane's file snapshot can't load the data file).
  - User picked 4 Tab lines ("some version of 4 would be the best. The numbers are pretty big"). Mockup: a Number size
    slider (0.6-1.8x, `?size=`) for tiles / runs / tab lines; the string gaps behind tab numbers are drawn before any
    number (one number's gap had cut into its neighbour in "1 0"). Not in the game yet.
- "I would like the tab to be slow enough to read easily but can still be understood. It doesn't have to be the same style
  ... open to other ideas": moving numbers are the hard part, so the mockup got a Motion switch (keys 7 8 9, `?motion=`):
  Scroll (as now); Bar steps (the board holds at the bar's start while an amber cursor sweeps the bar, then slides one
  bar on in 0.22 s; played notes dim; the next bar always in view); Pages (two flat lines of tab, 2 bars each, the
  playing line on top with a cursor, the next note glowing amber, the next line sliding up in 0.3 s; Tab lines numbers).
  Play on Tab (Game settings › Play on: Tab, `tabGameFrame`) already plays the game on the still Lines view, in the
  older look; Pages is that idea in Encore's style.
- "Some version of the static tab could work, but it's boring": Pages got a live layer, each part switchable (Rings,
  Hits, Strings, Lights, x4, Plain; `live` in the mockup): rings closing in on the next two notes over 0.7 s (on every
  coming note they cluttered fast runs); hits that flare, ring, throw 7 sparks and stay lit in the string's colour; the
  string vibrating out from the note played (~0.5 s); stage lights drifting behind and pulsing on the beat (beat 1
  most), bar lines flashing as the cursor crosses, a glowing trail behind the cursor; x4 turns it blue. The numbers
  never move. Hits are simulated (every note counts as hit).
- "It's very arcady now. I still think it's boring how the tabs just move down a level ... otherwise I probably wont come
  back. I don't know how to make it more exciting": Pages line changes (`change` in the mockup, Slide / Fly / Cards /
  Reel, `?change=`): Fly (coming lines wait smaller in the distance, 3 ahead; the camera flies forward, the finished line
  sweeping past and fading), Cards (the finished line flips off to the left, the next snaps up with a small overshoot),
  Reel (lines on a drum, rolling with an overshoot). Each starts 0.28 s early (`LEAD`) so the new line lands on the
  downbeat, with Punch (3% zoom + an edge flash on the downbeat) and Cascade (the line coming into view: numbers pop in
  30 ms apart). The playing line (cursor, hits) is the one containing now (`rPlay`), the moved-to line `r`.
- "I think the fly is cool, I think one hard thing about it is that you need to read the end of the tab and then shift
  back over constantly": Strip line change (`?change=strip`, `stripV`): one continuous line showing 0.3 bar before the
  playing bar to 2.2 bars after; at each bar (0.28 s early, LEAD) it slides one bar left in 0.35 s with a small
  overshoot and the downbeat punch, so the next bar arrives where the eyes already are; the far right fades out over its
  last 20%. Bar lines are now drawn for every bar in a line's time span (rows and strip).
- "That has the same problem with jumping back and forth. I think that's the main problem to solve": anything that holds
  still makes the eyes follow a cursor and come back (a line in Pages, a bar in Strip), so the only way to keep them
  still is a still cursor with the tab moving. Suspected reason the game's board is hard to follow: its 45-degree lean
  makes notes grow and speed up as they come in. Flow (`?change=flow`): the flat tab lines glide under a fixed cursor
  20% in at one steady speed (0.25 board lengths behind, 1 ahead, via `warp`, so Note speed and Spread apply), a punch
  on every bar, the ends clipped. With Spread the Wow! Wow! Wow! 0-1-2 run spreads out.
- "I like this idea [Flow] more with a slower speed. I would like to make the notes different, something easier to
  understand at a glance": note looks in Pages / Flow (`noteLook`, `?look=`, plus `?speed=`): Numbers (as before), Pills
  (a capsule in the string's colour with a dark number, as long as the note rings, stopping 6 px before the next note on
  that string, `nextOn`; repeats are short pills without a number; open strings hollow), Discs (a round badge with the
  number; repeats small dots), Pills + shifts (an amber outline and ▲/▼ on notes where a 4-fret hand has to move along
  the neck, `shifts`, a simple guess, not the tab's `handMoves`).
- "I'm not big on any of these to be honest. I think maybe the numbers, but I would like to make it more exciting
  somehow": energy switches for the numbers (`energy`, all on in the mockup): Neon (a white core with a glow in the
  string's colour, brightening from 55% as the note comes within ~1.7 s of the cursor; 30% was too dim to read ahead),
  Bounce (numbers ahead and strings pulse on each beat, beat 1 strongest), Heat (a simulated streak meter filling over
  18 s: light streaks rush past behind the flowing tab, played notes throw embers, full = ON FIRE x4 blue), Shatter (a
  played number breaks into four pieces with +100 rising and is gone, its string gap too). Wow! Wow! Wow!'s excerpt has a
  rest from ~13 s to ~20 s (73 notes in 24 s), so Flower Power Madness shows heat best.
- "Or how about, looks flash gamey" (read as: the 2D Flow mockup looks flash-gamey, as the drawn 2D board once did): Flow on
  the real 3D board, `look.flow` in game3d.js + Flow / Slow on `/board-lab.html` (`?look=flow&slow=1`; Slow = Z 4.5 s
  ahead instead of 2.6): the board turned to face the camera (`world.rotation.x` 1.42 instead of 45 degrees, no camera
  yaw), so notes glide at one steady speed to the pads; notes shown as their fret number (1.7x, the body hidden; quick
  repeats keep their slim tick), no rise-in from the distance; the glass darker (#040508, facing the stage lights it
  washed out grey) and the x4 wash at 40%. Not in the game (`v.look` isn't passed there). No console errors in the lab.
- Arcade mode, premium (user: "the shatter and score popup are distracting and repeated notes are hard to read", "make it
  more premium", "the energy around the game could use improving"; on joining repeats with a line: "the line to me means
  hold the note"): a separate focused mockup `design/game-mockups/arcade.html` (Flow only, the older note-styles page left
  as it was). Numbers in SF (700, no outline) on dark keycaps with a rim in the string's colour (brightening as the note
  comes due, open strings rimmed full colour with a coloured 0); a repeat is the number again at 78% size and ~60%
  opacity, never a line (lines only for notes ringing 0.3 s+); played notes fade to ~30%; a hit fills the keycap with the
  string colour for 0.2 s with a soft glow, no sparks or popups. Softer string colours, thin strings brightening when
  played, the track a dark glass panel (played side darker), hairline bar lines and beat ticks along the bottom, a slim
  cursor with a soft column of light. Around the track, each switchable: Light pools (drifting, brighter with the streak,
  how busy the music is and the beat), Edge glow (top / bottom in the mood colour, swelling on beat 1), Dust (motes
  drifting left behind the panel), Milestone sweep (one soft band of light along the track when the multiplier goes up),
  Score (light 40 px score, small multiplier ring, streak). Mood colour per multiplier (warm white, amber, orange-rose,
  blue) from a simulated streak (every note hit). Spread makes room for the keycap's width (up to 3.5x), and a keycap
  still too close to a neighbour on its string shrinks (not below 70%). Speed 0.7x by default.
- "It's hard to read when it speeds up super fast" (Spread up to 3.5x): Spread capped at 1.35x and held / eased over
  ±0.8 s; keycaps can shrink to 60%. Result at 0.7x: no surges, but 16ths at ~0.12 s are ~15 px apart, so fast runs
  (Wow! Wow! Wow!'s 2-2-2-1-0, its open-E 0s) shrink and crowd. The limit: slow speed, fast notes and full-size numbers
  can't all fit; one has to give (a faster base speed, shorter marks for quick repeats, or gentle spread + smaller).
- "In flower power madness, numbers are on top of each other" (keycaps shrank against one neighbour at a time, min
  60%): sizes now set per string from both neighbours at once (`lay`): each pair of keycaps scaled so their half-widths
  fit the gap less 4 px (not below 50%); a quick repeat (within 0.25 s) is a thin timing mark on the string, not a keycap
  (slower repeats keep the smaller number). Flower Power Madness at 4, 10, 17 s: nothing overlaps.
- "It says notes are repeats when they are not": the mockup compared each note only with the previous note on its
  string, so A5 E0 A5 counted. Now the game's rule: the very next note played (notes struck together skipped), same
  string and fret, within 2.5 s. Flower Power Madness's excerpt: 10 repeats of 97 notes.
- Still overlapping ("baby has overlapping notes", "the small 10 and 12s next to each other") and "I also liked the more
  arcade feel it had before when playing": a left-to-right pass per string guarantees room (a keycap that still runs into
  the one before becomes just the number, then a mark, then is merged; Baby's tab has notes 34 ms apart); keycaps at least
  8 px apart (Baby's repeated 10s at 0.27 s had 3 px and read as jammed); a repeat that would shrink below 85% becomes a
  timing mark (the first stays big). Checked every 0.25 s through all three passages (`window.__lay`): no keycaps or
  numbers touching; Baby hides 35 of its doubled notes. Arcade feel back as switches: Rings (closing in on the next two
  notes), Hit sparks (a ring + 6 sparks), String wave (the string vibrates out from the cursor when played).
- "I'm liking this. I feel like the repeat notes could look cooler": Repeats switch in arcade.html (`repStyle`, `?rep=`):
  Pips (default; glowing diamonds with a facet of light), Echoes (small outlined keycaps, no number, faintly filled),
  Chevrons (‹ pointing the way the tab moves), Bars (the thin marks before). Each brightens and grows a little over the
  last 0.6 s and flashes white for 0.18 s when played; layout room follows the style's width (`markW`). Slower repeats
  keep the smaller number.
- "Echos is cool, what other looks could we do with that?": Echo looks under Repeats › Echoes (`echo`, `?echo=`; Echoes
  now the default): Outline (as before), Mini number (the fret tiny inside the echo), Fading (each repeat in a row
  smaller and fainter, `repN`, to 55%), Ripple (a fainter outer outline widening as the note comes due), Glass (frosted,
  a light top edge). "Can we give it an effect on hit? Maybe more subtle": echoes no longer flash white or grow 1.5x; on a
  hit the echo fills gently with its string's colour, swells 12% and a thin outline ripples out and fades over 0.35 s.
  - User picked Mini number ("I like the mini number"): now the default (echoes + mini).
  - "Actually number could get crowded, give me other options": more echo looks: Simile (the % repeat sign inside the
    echo, "play the same again"), Dot (one bright dot), Auto (the mini number only when the echo has room: neighbours on
    its string more than 2.4 echo-widths away, `it.room`; else plain). Mini number stays the default until the user picks.
  - "What about like a diamond?": Diamond echo look (the echo turned 45 degrees at 92%, a faint fill, a small facet of
    light; its hit ripple is a diamond too).
- "I want the repeats to be smaller to create less clutter": Repeat size slider (`repSize`, default 0.7) scales the
  echo marks, and every repeat is now an echo (before, slower repeats showed a smaller number; shrinking those let more
  of them fit as numbers, more clutter). Only fret changes carry a number; same-fret notes with another string played
  between keep theirs (not repeats by the game's rule).
- "Can we make zero like the other numbers?": open strings in arcade.html now drawn like any fret (white digit, the same
  rim that brightens as the note comes due); they had a full-colour rim and a coloured 0.
- "Can we make an effect even on the repeat notes?": echoes get the numbered notes' effects at their size (following the
  Rings / Hit sparks switches): a ring closing in when the echo is one of the next two notes, and on a hit a soft glow
  with 4 small sparks (0.3 s), on top of the subtle fill-and-ripple pulse.
- "It's easier to read when the numbers aren't huge": Size defaults to 0.75x (slider 0.5-1.5); Spread's room follows it.
- "How could we make the background better?" then "I want it to look premium": Background switch in arcade.html
  (`bgStyle`, `?bg=`): Studio (default), Stage, Aurora, Landscape, Pools; the old "Light pools" switch is now
  "Background" (on/off). Studio: colour falling softly from above, 14 large lens-blur lights (flat inside with a faint rim,
  barely drifting, `BOKEH`), a haze band behind the track, the track's light as a faint reflection on a polished floor.
  Stage: three rows of soft out-of-focus lights flashing in turn on the beats, a wave across each row per bar (its hard
  bright cores removed: they looked cheap). Aurora: slow ribbons of light (additive). Landscape: a silhouette of how busy
  the song is (`dens`, smoothed note density), scrolling with the tab. All styles now get a soft vignette and fine film
  grain (`GRAIN`, 3.5%). A first build failed to load (`-x**2` is a syntax error in JS).
- "Still looks cheap" / "I want it to be dynamic with the song": Mesh background (default now; `MESH`, a 48x30 canvas of 4
  huge radial colour clouds stretched over the screen, so no edge shows, then a 28% dark overlay): brighter on every
  beat (beat 1 most), brighter and drifting faster with how busy the music is (`busy`, notes in the last second), the
  colours trading places every 4 bars (eased over the phrase's first bar), a swell when it suddenly gets busier than its
  running average (`busyAvg`), colours from the multiplier. In the game these should come from the song's light cues
  (`/api/lights`: kicks, snares, loudness, big moments) rather than the tab. Dust and Edge glow now off by default (the
  clutter read as cheap).
  - "A bit too flashy": Mesh calmer: clouds 0.16 + 0.16 busy + 0.05 beat + 0.14 swell (were 0.22 / 0.3 / 0.18 / 0.35),
    drawn at 70% under a 42% dark overlay (85% / 28%), drifting slower (0.2 + 0.6 busy).
- x4 and sizes (user: "can we improve the 4x look?", "the changing note sizes is tough", "I liked when the board turned
  blue before", "not sure how I feel about the line going around" / "we could make an outline, just cooler", "I still
  want to see the stars", "make stars similar to other theme"), arcade.html:
  - x4 (eased in over ~0.5 s, `x4k`; x4 preview switch `fx.x4`, `?x4=1`): the board turns blue (a wash strongest at the
    cursor, like the 3D board's), strings and keycap rims tinted blue, glowing caps on the cursor, soft blue flame
    rising off the multiplier ring (`FLAMES`, white -> cyan -> indigo, 0.7 s). x4 outline (on): the board's whole edge
    glowing, colour shimmering slowly along it (cyan / blue / violet), breathing on the beat, a faint inner line. The
    travelling glint is its own switch, off.
  - One note size per song (`songFit`: 80% of the gaps between numbered notes on a string fit a keycap + 8 px, not
    below 80%; at 90% / 60% Flower Power Madness came out tiny); no per-note shrinking. When two numbered notes are too
    close both drop their keycaps (bare numbers, same size); a run still too tight is staggered alternately above /
    below the string (`dy`, 0.22 of the string spacing); an echo right before a numbered note merges if needed. Only
    repeats are ever merged. Checked every 0.2 s on all three passages: no fret change hidden (before: Baby 13 of 58,
    Flower Power Madness 15 of 87, Wow! Wow! Wow! 6 of 34), nothing touching.
  - Stars under the score as in the game's HUD (`drawRockHud`): 17 px, #f5b70a, warm empty stars, a glow when full;
    filling with the score toward a perfect run's.
  - "Remove fire effect on the multiplier": the flame off the ring at x4 removed (the ring keeps its glow).
- Arcade board in Encore (user: "I'm good with this for now, implement"), 2026-11-02: Game settings › Board: 3D | Flat |
  Arcade (`track-splitter-game-board` = `arcade`; the Fretboard row only shows for 3D). `drawGame` calls `drawArcade`
  (index.html, before `drawGame`); guitar chord charts (`game.members`) fall back to the 2D board. A port of
  design/game-mockups/arcade.html with its defaults, on the game's own state: `game.result` (hit / miss: misses grey),
  `game.hitAt` (hit glow / ring / sparks), `game.laneHitMs` (string brighten + wave), `game.score`, `game.streak`,
  `multiplier()`, `starsFor`, the latest call (Perfect / Good / Miss / Wrong + what was played) right of the streak, the
  count-in. Speed = 0.7 x the Note speed setting (`hwSpeed`); room for fast runs up to 1.35x. `arcadePrep` caches per
  song + window size + speed in `d.arcadeLayout`: the position warp, repeats (the game's rule), neighbours, the song's
  note size (`fit`). The background mesh follows the song's light cues (`cuesAt`: kick / snare pulses, loudness as how
  busy, `big` moments as swells), falling back to note density. State in `arcadeBoard` (`arcade` is already the Encore
  screens' state: a first build declared `const arcade` again and the whole page failed to load). Echo look: mini number.
  - Tested on a library copy (Baby + Flower Power Madness, port 7873, silent simulated input): song select -> game,
    notes gliding, misses grey with the Miss call, held-note lines, staggered pairs; x4 via `track-splitter-game-streak`
    35 (blue board, outline, cursor caps, 4x ring) during the count-in; pause, Settings (Board switch, Fretboard row
    hidden for Arcade / shown for 3D, saved), restart, quit; the normal app at / (song list, player bar) clean. Not yet
    played with a real bass: hits, sparks and the string wave were only checked in the mockup.
- "I like the idea of putting the numbers on different sides of the line, I just feel like we could make it easier to
  read": Stagger switch in arcade.html (`stagger`, `?stagger=`; mockup only, the game still floats them): Stems (default;
  a dot on the string where the note falls, a thin stem in the string's colour to the number, which keeps a small
  keycap), Flags (a keycap beside the line with a small pointer touching it), Float (as in the game). Stems / Flags sit a
  little further off the line (0.3 of the string spacing, was 0.22). The dots also show which of a staggered pair is
  first.
  - User picked Stems; ported into the game's `drawArcade` (staggered runs 0.3 of the string spacing off the line, a dot
    on the string + a stem to a small keycap). Tested in a game on the library copy (Baby, 2:08 in): staggered 10 / 12
    and 12 / 8 pairs show their dots and stems, no console errors.
- "I feel like there's a lot of wasted space on the screen" (Arcade): the track now up to half the window (400 px max,
  was 30% / 250 px), centred at 56% of the height; note sizes, held-note lines and the string wave keep their old sizes
  (`sizeSp`), the strings spread apart instead. Bare numbers closer than half a number apart now stagger (Baby's "12 8"
  side by side read as "128"; the rule had only staggered numbers that touched). A first try put a `//` comment inside a
  one-line loop and the page stuck on "Loading…" (the known trap: comments in one-line blocks). Tested in a game on the
  library copy: no console errors.
- "Arcade mode is laggy and the tab isn't matching up": measured in a game at 1440x900 on a 2x screen: the board's code
  took 0.3 ms a frame but the game ran at 20 fps (51 ms frames): the cost was drawing pixels (full-screen layers: the
  dark fill, the stretched mesh, a dark overlay, the vignette, moving grain; shadowBlur on the panel (40 px), every string,
  the cursor and the x4 outline). A frame that late shows the tab behind the music, the "not matching up" (the board gets
  the same time as the 3D board, `tabTime()` with the output delay). Now: the mesh canvas has the dark overlay (0.594,
  the same result) and the glow behind the cursor baked in and is stretched once (no separate fill); the vignette, grain
  (now still), the glass panel with its shadow, border and darker played side are drawn once per window size into
  `arcadeBoard.static`; strings, cursor and x4 outline glow with a wider faint stroke instead of shadowBlur; echoes blur
  only when near or hit. Result: 119 fps (115 at x4), frames 8.4 ms, no console errors.
- "Did we say we would do the diamonds for the repeat notes?" (we hadn't settled it: Mini number was the last confirmed
  pick) -> "sure": the game's echoes are now small diamonds (the echo turned 45 degrees at 92%, a faint fill, a facet of
  light, no number); the mockup's default too. Checked in a game on the library copy (Baby's E-string 10s), no errors.
- "Can we improve how held notes look?": held notes in `drawArcade` (and the mockup): a ribbon in the string's colour
  fading toward its end, a light top edge, an end marker where to let go; while held bright with a pulsing glow round
  it, streaks of light flowing along it into the cursor, a glow where it meets the cursor, the end marker brightening
  and glowing in the last 0.4 s; the finished part fades behind the cursor; let go early (`game.hold[k].lost`) or missed
  = grey. Feeding the bass-only track in as the input (a second AudioContext mirroring the player's `start`) gave no
  hits in the browser pane, so the user said "just use the mockup and I'll tell you what looks good": checked there
  (Baby 11.85 s, Wow! Wow! Wow! 5.75 s). The game's version is in place but its holding state is untested.
- Score section (user: "can we make the score section look better? I don't like how much wasted space we have"): the
  score (52 px) with the stars beside it over the start of the panel; the streak (a big number, a small STREAK label) at
  the panel's right edge with the latest call left of it. No song title / progress (not wanted on the game screen).
- "I would like to incorporate the multiplier into the board": the cursor is the multiplier meter (fills from the bottom
  in the multiplier's colour with the streak toward the next level, full and glowing at x4) with a 1x-4x badge on top
  that pops when it goes up; the ring by the score removed. The page's scripts can be syntax-checked without a browser
  with macOS's jsc (`new Function(src)`), which catches the one-line-comment kind of break. Checked in a game on the
  library copy: at a 25 streak the cursor half full in orange with a 3x badge, "25 STREAK" right, score + stars left;
  after the misses an empty grey cursor, 1x, "Miss" beside the streak; no console errors.
- Score layouts in the mockup (user: "I don't like how above the highway looks"): Score switch (`hudStyle`, `?hud=`):
  Row (score + award left, streak right; the award under the score, beside it ran into the cursor's badge), Centred
  (one group over the middle), Below (under the track), Off; the cursor-as-multiplier copied in from the game.
- "How about instead of doing stars, we do records? bronze/silver/gold/platinum/diamond?": Records in the mockup (`award2`,
  `?award=`, Stars | Records): the 5 star levels as Bronze / Silver / Gold / Platinum / Diamond; a small vinyl (grooves,
  a highlight) with the tier's colour on its label, a thin ring filling in the next tier's colour, "Gold record" beside
  it ("No record yet" before Bronze); a pop and a shine across it when a tier is reached; Diamond: an icy label, a
  shimmering ring and two sparkles. Mockup only; stars are also on song select, results and bests in the game.
- "I don't want to show the streak, but show something at 50/100/150/etc. I also want to refine the look of the record,
  make it look more premium" / "I'd like the whole record to be the color and then the progress fills with the new
  color" (mockup): no streak number; every 50 notes a milestone over the middle of the track (a large light number, NOTE
  STREAK in spaced capitals, thin lines drawing out to each side, ~1.6 s; `mileN` / `mileAt`). The record: the whole disc
  in the tier's metal (bronze / silver / gold / platinum / diamond gradients; the first palette read almost silver, now
  richer), the next tier's metal sweeping round it like a clock hand as you get closer (a bright leading edge), grooves,
  a reflection turning as if it spins, a bright edge, a deeper label, a spindle hole, sparkles on Diamond, a pop and a
  shine on reaching a tier; the tier in spaced capitals. Bugs found: jumping back in time (the mockup's loop) left the
  pop timers in the future and the record swelled huge (clamped, also the cursor badge); a very narrow window made the
  x4 wash's colour stop negative and the frame threw (clamped in the mockup and in `drawArcade`).
- "I think the score and record could look better": Plaque score look in the mockup (`scoreLook`, Plaque | Classic, with
  Records): one unit, the record as its emblem (26 px, no name beside it), the score beside it in 200-weight 48 px type
  with a white-to-silver gradient, under the score a short rule and the tier in its metal's colour ("GOLD RECORD", spaced
  capitals); works in Row (left), Centred (one group over the middle) and Below.
  - "And the next color should be easy to see and get color as you get more": the incoming tier's wedge starts faint
    (35%) and grows solid, tinted more and more with a vivid accent per tier (orange, bright silver, rich gold, icy
    platinum, cyan diamond; gold -> platinum had looked alike), a leading edge in the accent, and a glowing accent arc
    round the rim tracing the progress.
- In the game ("Yes, put them in"): `drawArcade` now has the Plaque centred over the track (`ARCADE_TIERS` metals,
  `ARCADE_VIVID` accents: the record as the emblem with the next tier sweeping round it, faint to solid, a glowing rim arc;
  a pop and a shine on a new tier, `arcadeBoard.tier` / `tierAt`; the score in light silver type; the tier under it), no
  streak number, the latest call at the right edge, and a moment every 50 notes over the middle of the track
  (`arcadeBoard.mileN` / `mileAt`, 64 px). The game keeps its own Classic HUD for 3D / Flat. Checked in a game on the
  library copy: plaque, call, cursor badge, x4, no console errors; the milestone (started at a 50 streak) had faded
  during the count-in before the screenshot, so it was only seen in the mockup. Song select / results still show stars.

### Encore in the Arcade look; records instead of stars (2026-11-02)
User: "How much work would it be to make the rest of the UI match this arcade mode?", then "can we replace stars with
records before we do anything? should be everywhere", "do the mockup first, that will include this change", and after
the mockup (`design/game-mockups/ui.html`: Title, Song select, Loading, Results, Pause, Settings) "Yes, build it".
Mockup feedback built in: no record on the album covers on song select ("don't show the record on the album cover when
selecting"); the record type between the record and the score on the difficulty cards; and progress as a ring round the
outside instead of the next tier's metal sweeping over the disc ("instead of changing entire record color when
progessing the new one, just do a ring on the outside").
- Records: one drawing, `drawRecord(g, x, y, r, val, t, popAt)` (val = `starsFor`'s 0..5; tier = floor, the rest is the
  ring), used by the Arcade plaque, the 3D / Flat HUDs (`hudRecordAt`, where the five stars were, tier name beside it),
  results (190 px, filling as the score counts up over 1.6 s, popping at each tier; a finished run shows "N to Platinum"
  in `#game-next`), pause (a plaque: record, score, tier, time), song select (difficulty cards: record, tier, best; list
  rows: record), title (Jump back in: record, score, "GOLD RECORD · HARD"). Page records are small canvases
  (`recordEl`, `paintRecord`). Saved data unchanged: `STARS_KEY` keeps the whole star count (0 = no record … 5 = Diamond),
  so saved records show no ring.
- Look: the CSS block "Encore in the Arcade board's look" near the end of the styles overrides the rock-stage rules
  (kept above it): SF Pro, light weights for big text, small spaced capitals for labels, silver gradient numbers, white
  pill buttons, hairline rows, segmented controls as pills, the cursor line as the focus mark on menus. It sits after
  the older hover / Play-as-text rules on purpose (they outranked it before it moved). The stage (beams, truss) and the
  cover wall are hidden; `#arc-mesh` / `drawArcMesh` draws the board's colour mesh behind the menus (~30 fps, stops while
  a game runs or the window is hidden; `setInterval(startArcMesh)` restarts it) and on the title a faint tab gliding
  under a still cursor. Song select / loading / results still blur the song's cover over it.
- Settings: one centred panel with Game / Sound / Setup tabs (built at start from the `.set-group` headings, each group's
  rows moved into a `.set-sec`) instead of two crowded columns (user: "can we improve the settings on the main page");
  a Settings gear on song select (`#arc-select-settings`; "also have a settings button on the song select page").
- Fixed while checking: Resume showed a white box on hover (the white-pill hover rule caught the pause menu; user).
- Checked on the library copy (port 7873): title, song select (covers and list), settings (all tabs, from the title and
  song select), an Arcade game, pause, quit → results; no console errors from the page (the only one is the existing
  service worker fetch in the browser pane).
- Follow-ups: the "white box" was the keyboard focus outline round Play on the title (user sent a screenshot; Play has
  focus when Encore opens); menus, covers, list rows and difficulty cards no longer draw it (the cursor line or the
  highlight shows where you are), and the pause words stay bare in every state. Opening splash (user: "can we also
  update the loading splash when you first open encore"): `.encore-splash` (brand + a light sweeping a thin line) covers
  the page from the first paint until `startArcade` adds `body.arcade` (`html.encore-boot`, set in the head script on
  /game; before, the Track Splitter page showed for a moment); "Starting Track Splitter…" uses the same brand and line;
  the installed app's colours (`encore-manifest.json`, theme-color) match the new background.
- Arcade board: every note but a repeat keeps its keycap (user: "not all notes have the box around them. Any none
  repeating note and new note needs it. Thank you for the venom for example"). Two keycaps too close to sit side by side
  used to drop to bare numbers; now they go above / below the string on stems (with their keycaps), and a repeat's
  diamond gives way to a keycap it would touch. Checked in a game on Thank You for the Venom (copied into the test
  library; every non-repeat note boxed, none needed staggering) and Flower Power Madness (fast runs staggered on stems,
  boxed).
- Staggered notes "can be hard to read" (user): the Arcade mockup got Stagger options (Stems, Lift: the next close one
  just above on a short stem; Joined: a run of close notes shares one keycap on the string, numbers in order with thin
  dividers; Slim: narrower keycaps packed closer). User picked Joined; in the game a run of keycaps too close to sit side
  by side (`grp`) is one keycap centred on the run, each number at its own spot inside it, lighting / flashing / dimming
  on its own; rings and hit sparks sit on the number. Stems are gone from the game. Checked in a game on Flower Power
  Madness ("3 | 5" runs on the A string, no console errors).
- Arcade: the score group no longer shifts as the score changes (user: "the score at the top is shifting left to
  right"; it was centred on the score's current width): it's laid out for the widest score the song can reach and the
  longest tier name, the score drawn from its left edge. Held notes lost their end bar (user: "held notes have a bar at
  the end, can we make it not have that"): the ribbon just fades to its rounded end.
- Hold line "more premium like the rest of the app" (user): the Arcade mockup got Hold looks (Ribbon, the game's; Glass;
  Filament, a fine glowing line that mostly blended into the string; Satin, a brushed-metal ribbon with a sheen). User
  picked Glass: a dark glass tube with a fine rim in the string's colour like the keycaps, a top reflection and a thin core
  of light; while held the core brightens and one soft glint travels along it (no more flowing dashes). Checked in a game
  on Flower Power Madness (tubes on the D and A strings, no console errors); the held state, which the silent test input
  can't reach, was checked in the mockup.
- A quit song isn't saved and shows no record (user: "if I quit a song, don't save the score or show me a record"; it
  saved the score as a best and stars from its accuracy). Results then say "Not saved: you stopped early" (with your
  best, if any); a finished song is unchanged. Checked on the library copy: quit from pause → no record, the saved
  bests and records untouched.
- "How can we improve the game view? I feel like it's still pretty basic" (2026-11-06). Of six ideas offered (timing
  feedback, results breakdown, board reacting to the music, Encore phrases, intro / ending moments, crowd sound) the user
  picked "3/4/5". Mockup first: `design/game-mockups/moments.html` (served by `.claude/launch.json` "mockups", port 7881),
  built on arcade.html with the game's picks. Options: Beat (Off / Glow: light rising in the glass on kicks, a flick from
  the top on snares / Edge: the board's edge flashes), Big moments (rim lit, haze, fuller mesh, a flash and band on the
  way in), Landings (a ring of light from the cursor and every string lit on a note after a 0.8 s gap); Encore phrases
  (2-bar gold phrases as Gold notes / Lane / Both, each fills half a meter Under board or Under score; full, Encore runs
  4 bars on its own: gold board and outline, 8× badge, ENCORE banner, "+N ENCORE BONUS" after); Intro (Card where the
  score goes, or Big title before the board; Board opens from the cursor on the count-in); Ending (Song complete /
  Full combo over the dimmed board, score counting up, See results). Kicks / snares / big moments are faked from the
  beat grid and note density; in the game they'd come from `lights.py`'s cues. Waiting on the user's picks.
  User: "I like the bar under the score, we may need to clean it up though". It had crowded the record and sat on the
  board's edge, with ENCORE / READY labels out to the sides and a width that moved with the score. Now it's the plaque's
  last row: two slim segments (one per phrase) under the score's column, the plaque laid out for the widest score (raised
  12 px to fit); no side labels: the tier line reads ENCORE READY (pulsing) when full and ENCORE · DOUBLE POINTS during
  Encore, then goes back to the record tier. Under score is the mockup's default now.
- Picks for the game (2026-11-06): Beat Glow, Gold notes, Big moments, Landings, Board opens, the ending moment, the
  meter under the score. Intro: the user liked Card "but I also like how the board appeared on the big title comes in
  without the countdown. One of my complaints before was that it just started from the first note and was hard to time.
  Maybe show the song info where the score is and transition into the score when the song is about to start". So in the
  mockup: the card sits where the score goes, the board opens on the count-in's first beat and the first notes glide in
  during the count, the big 3-2-1 over the board is gone (the cursor's badge counts 4, 3, 2, 1 on the beats, a ring
  going out and the line flashing), and on the last beat the cover flies to the record's spot and rounds into it while
  the words lift away and the score fades in.
  The card-to-score hand-over had to change (user picked "Card to score" as the tweak). Mockup options under "Card to
  score": Sleeve (default: the record slides half out of the cover like vinyl from its sleeve, then shrinks into the
  plaque's record while the sleeve fades; 2 beats), Flip (the cover shrinks to the record's place and turns over, the
  record on its back), Morph (each line goes to its place: cover to record, title to the score, artist to the tier line),
  Shrink (the first version, 1 beat). The intro draws the plaque's own record (`drawRec`, `plaquePos`).
  User picked Flip ("Flip"); it's the mockup's default now. Asked how often Encore triggers: the mockup's phrases are
  placed by hand (once per 24 s passage); proposed for the game: a gold phrase about every 8 bars on busier stretches with
  a clear start and end, two to fill the meter, a missed note voids the phrase -> about every 16-20 bars when playing
  well, ~2-4 times a song (not yet confirmed).
  "Can we make the score show a little sooner?": with Flip the score now fades in over 30-65% of the hand-over (was the
  last 20%), alongside the cover as it reaches the record's place; the plaque's own record stays hidden until the flipped
  cover lands (92%), so there's never two records.
- "Can we remove the double points text and then build" (2026-11-06): the mockup lost DOUBLE POINTS (the banner has no
  line under ENCORE; the tier line shows the record during Encore), then it went into the game (drawArcade):
  - Encore phrases: `encorePlan(d)` per song (cached on the tab data): in each 8 bars from the 3rd, the 2 bars with the
    most notes (>= 4) are a gold phrase; Baby (184 s, 79 bars) gets 10. `game.enc` (Arcade board only: not the tab view,
    guitar chords' board or practice): phrase states (done / broken by a miss or a messy note / skip: before a later start
    or overlapping Encore), the meter (half a phrase each), `on` {t0, t1}: Encore from the bar after the meter fills, 4
    bars. `encoreTick(now)` after each `judge`; every score goes through `addScore(pts, now)` (x2 in Encore, its extra
    kept as the bonus). Note: records come from `perfectScore`, which doesn't count Encore, so Encore makes them a bit
    easier to reach.
  - Look as in the mockup: gold keycaps (warm fill, gold rim and glow, a glint), the board and its outline gold in Encore
    (x4's blue gives way), the badge doubled (1× reads 2×), ENCORE over the score (no sub line), "+N ENCORE BONUS" after,
    a gold flare and motes into the meter when a phrase is done; the meter is the plaque's last row (plaque 12 px higher),
    ENCORE READY on its tier line while full.
  - Reacting to the music: big moments (a big cue in the last 12 s while the level stays > 0.45: rim lit, haze, fuller
    mesh; a flash and band as it comes in) and landings (`plan.land`: a note after 0.8 s with nothing ringing; when hit, a
    ring from the cursor and every string lit). Kicks and snares lighting the glass were built, then taken out (user: "the
    board flashing to the song is distracting"; off in the mockup too).
  - Intro (`startGame`, `game.intro`, `introCard()`): the card (the song's picture from /api/cover, else a drawn cover in
    coverEl's hues; title, artist, difficulty, best) for 1.8 s with `game.countdown` = 9 (nothing judged), then 4 beats at
    the song's tempo (median bar / 4, 0.4-0.75 s; timed from the start, not chained timeouts) while the board opens from
    the cursor and the tab glides in (drawArcade draws an earlier time), the badge counting 4-3-2-1 (also 3-2-1 after a
    pause; the big numbers over the board are gone), the last two beats flipping the cover into the record. The tab view
    and guitar chords keep the old 3-2-1.
  - Ending: once every note's judged and the last has ended (+0.5 s), or the song stops then (`game.songEnded`: the board
    holds its last frame; Baby's audio ends 2 s after its last note, which used to end the game first), SONG COMPLETE /
    FULL COMBO (all hits, nothing messy) over the dimmed board, the score counting up with its record, See results; a
    click or Space (pauseGame) or 6 s goes to the results.
  - Tested on a library copy (Baby and Flower Power Madness, port 7883) with a simulated Scarlett playing Baby's bass-only
    track: intro card and opening, gold phrases (broken ones turning plain), hits and holds scoring, Encore forced by
    filling the meter (gold board, banner, 2× badge, meter draining, +3,200 bonus), the ending from a late start (moment,
    then the results by themselves, finished), practice unchanged; no console errors but the service worker's. Not yet
    played with a real bass; a phrase completing on its own wasn't caught (the simulated input slips now and then).
- "Can we improve the visuals of the game? For instance, long notes look a bit weird, especially when there are multiple.
  Also, can we handle playing multiple notes at once and hammer-on/pull-offs?" (2026-11-06). Looked first on a library copy
  (All My Best Friends Are Metalheads ~191 s, held double stops; NARCISISTA; No You Girls, hammer-ons at 13 s and double
  stops at 79 s; Amsterdam, a held power chord at 148.5 s):
  - Held notes: every note at least 0.3 s before the next on its string got a tube, which ran right up to the next keycap,
    so a steady line on long notes read as one wire through the keycaps (three of them for the Metalheads chords). Now a
    tube only for a note held longer than a beat (the bars' beat, `beatLen`: the tab's bpm is double time on some songs,
    Metalheads 185 for ~93; at least 0.4 s, from which a hold scores, unchanged) and it stops a keycap short of the next note
    (`tails`). A held chord gets a faint sheet of glass between its outer tubes, brighter while held.
  - Notes struck together (within 30 ms, different strings; `chords` / `chordOf` in drawArcade): one glass frame behind the
    keycaps with a line joining them, glowing when it's next, and one ring closing in round the frame (`ringed` is now the
    next two strikes, a chord's notes one; `isNext` within 30 ms).
  - Slides and hammer-ons / pull-offs (imported tabs' marks, `legatoMarks`) weren't drawn on the Arcade board at all: now a
    slur over the two keycaps with h / p on it, a slide a slanted line between them (rising = up the neck), short slanted
    lines for slides in / off; a slide between notes sharing a joined keycap is a slur with / or \. Their judging was
    already in (no pluck of their own needed, never messy).
  - Bass double stops judged as one (`bassChordPluck`): the pitch reading hears one note of the two, so each judged on its
    own missed the other (and counted as two notes). Now, as guitar chords, `lead` / `members` / `units` for bass too when
    the tab has double stops (none: unchanged); hit by one pluck in the window with either note or an octave of one read
    after it, or the chord reading (`readChord`, only near a double stop on bass: `chordNear`) finding its notes as 40% of
    what it hears; 100 x multiplier once. Held: one hold per chord (its longest note), kept by either note. `game.strums`
    (guitar) keeps guitar's strum detection, `chordPluck` and the 2D chord board; bass chords stay on the Arcade board.
    Practice's Wait for me takes either note or an octave of one.
  - Bug found: imported tabs can list a bar's notes string by string, out of time order (No You Girls: 21 notes after later
    ones), which the game's judging (`game.next`) and grouping assume; `gameNotesFor` now sorts the game's copy (29 double
    stops found there instead of 21). The tab views still get the file's order.
  - Tested on a library copy (port 7893): screens above at 1440x900 incl. a held power chord forced to "hit and held" (both
    tubes lit, the sheet brighter, hold points); plucks + pitch readings fed into a running game at No You Girls' double
    stops: lower note -> hit, upper note -> hit, octave of the upper -> hit, a wrong note -> miss, 100 points each;
    120 fps, no console errors. Not yet played with a real bass (double stops on the Scarlett), and the chord-reading path
    (`heardNotes` starts at 75 Hz, tuned for guitar) wasn't exercised; the pitch path carries it.
- Hammer-ons / pull-offs animated (user: "can we make the hammer on and pull off have an animation?"): between the first
  note and the second a spark with a short trail runs along the slur (it rides the cursor, reaching the second note when
  you should hammer), the slur glowing meanwhile; when the second note is hit the slur flashes and its h / p pops. Slides
  between two notes do the same along their line. The spark runs even if the first note was missed (a timing cue).
  Checked on a frozen frame mid-hammer (No You Girls 15.07 s); no console errors.
- Song select's List view cut off on the right after playing a song (user, with a screenshot: the difficulty cards, input
  row and Play past the window's edge). Once the input is connected the row shows the device picker, and a select is as wide
  as its longest option ("Scarlett Solo USB (1235:8211)"); the list-mode panel's `1fr` column couldn't shrink below that.
  Now `minmax(0, 1fr)` and the picker `min-width: 0` with an ellipsis. Checked with a fake device of that name at 1280x800
  and in a ~770 px pane: nothing past the edge (only the top bar's buttons at 770, below the ~1024 the app is made for).
  Then "can we just make the right side wider?": the list view's song side is 400-580 px, at most 42% of the window (was 320-440; uncapped, at 1024 wide it outgrew the list), so the device name
  fits again at the user's window size.
- "The album cover doesn't show up before the song": the intro card always drew the made-up cover. `game.intro` was a copy
  of `introCard()`'s object (`{ ...introCard() }`) and the picture, loading a moment later, was set on the original. Now the
  card object itself (`Object.assign(introCard(), ...)`). Checked: No You Girls' thumbnail on the card.
- "Could we do something where the loading screen transitions into it?" (it faded through black): `flyCover` in arcadePlay
  (games with the intro card: bass, Play, the board). A copy of the loading screen's cover is lifted out (measured before
  startGame hides the screen) and the loading screen stays over the game (`body.arc-handoff`); once the intro card is drawn it
  fades out (0.6 s) while the cover flies to the card's cover spot (`arcadeBoard.introCover`, 0.75 s), and the board draws its
  own cover from the landing (`arcadeBoard.flyUntil`). Guitar, practice and Play on Tab keep the fade. Checked mid-flight on
  the library copy; the classes and the copy are gone afterwards, no console errors. The landing itself wasn't caught on a
  screenshot (it lines up by construction).
- "I feel like we could improve the look of held notes" (2026-11-06): moments.html got Hold looks (Glass, the game's; Ringing
  string: the string lit thicker over the note, vibrating while held; Light trail: a comet streak tapering to a point, sparks
  while held; Capsule: the keycap stretched into one long keycap with a channel of light) and three passages with long notes
  from the user's tabs (`holds-data.js`: NARCISISTA, Amsterdam's held power chords, Dust to Dust); the mockup also follows the
  game's newer tail rules. User picked Capsule, then "can we match the style of the normal notes too? ... I don't want it to
  be off style wise": in drawArcade the tail is the note's own keycap stretched (from the keycap's left edge, its width
  `cw` measured with the tails), with the keycaps' fill and rim rules (lit when next / near, glow, gold in Encore phrases,
  x4's blue rim, grey when missed or let go, 45% once passed); the keycap draws only its number over it (`tailOf`, no seam)
  until it's held, when the capsule starts at the cursor and the light in its channel runs into it. Held chords keep their
  glass sheet. Checked on a frozen frame on the library copy (NARCISISTA at 2.9 s, and at 3.7 s with the A 0 forced held);
  Encore gold and x4 not looked at in the game (same rules as the keycaps).
- "I think the slides could look better" (were a thin slanted line, 1.8 px): a slide is now a slim slanted keycap in the keycaps'
  style (dark fill tinted with the string's colour, a rim, a channel of light), drawn before the keycaps. First it sat
  between the notes with a small gap (tucked under their centres, its end stuck out round a repeat's small diamond); user: "a
  slide should be connected to the note", so it runs edge to edge, 3 px under each keycap, rising 0.22 of a keycap's height
  each side for a slide up the neck (falling down it). Slides in / off: a short one (1.1 keycaps) into or off the note, its
  far end at 55%, stopping before the next note. The spark rides it as before (lit, flashing on the hit). Slides between
  notes too close for a bar stay a slur with / or \.
- "I don't like the h and p, I know if it is a hammeron or pulloff based off the notes": the slur has no letter now (its
  spark and flash stay).
- Checked on I Love My Friends (copied into the test library: 43 slides, 17 slides in, 36 off, 49 hammer-ons) on frozen frames
  blown up on the page (a canvas crop scaled into an <img>: the pane's screenshots are too small to judge these marks).
- "Still not right, a slid note should be fully attached, right now it has rounded edges on the note", and "is there a cooler
  way we could display hammer on and pulloffs?": slides and hammer-ons / pull-offs are one band now (`band` in drawArcade),
  stroked along a path from the first note's centre, out of its edge, to the next note's edge and into its centre (round caps
  and joins under the keycaps, so nothing round shows at a note; a diamond's end tilts 40% as much, it being smaller): slanted
  for a slide, level for a hammer-on / pull-off (the slur and its h / p are gone; a slide too close for a band keeps a slur with
  / or \). The second note's keycap is tinted like the band (`tied`); hit, a hammered note dips and a pulled one flicks up
  (0.2 s, 16% of the note size). The spark runs along the visible part. Checked blown up on I Love My Friends (hammer 5-7 on
  the A string, a slide off a repeat, a slide up into 12); the dip wasn't caught on a frame.
- Making the game "more fun, less sanitized" but "polished and professional" (2026-11-06). Tried in mockups and turned down:
  fun.html (Lively: colour-filled bobbing notes, gliding slide streaks, hopping hammer-on arcs, bursts on hits; Neon; Chunky):
  "flash gamey", "the explosion is distracting"; effect.html (a Tetris Effect-like field of light reacting to hits): "super
  distracting, I want the background to look like a show"; show.html (a drawn concert: beams, LED wall, crowd; then rendered out
  of focus with bokeh and grain) and webgl.html (a three.js stage: volumetric beams, spots on a glossy floor, haze, bloom): both
  "flash gamey". Offered real footage (the song's music video, needing a video download) or the cover as an Apple Music-like
  backdrop; the user chose "I think we can improve what we have now" and picked Song colours, Hit feel and Streak build-up:
  - Song colours: `songPalette(song)` (at game start, `arcadeBoard.pal`): the cover drawn 32 px, hue bins weighted by colour,
    the strongest hue and the next one at least 45 degrees away (a grey or missing cover: the drawn cover's two hues);
    `songMood(pal, mult)` gives the mood's light and deep colours for x1-x3, richer as the multiplier goes up; x4 stays blue
    and Encore gold (ARCADE_MOOD). I Love My Friends: pink-red and orange.
  - Hit feel, no explosion: the ring and six sparks are gone; a soft glow, the keycap pops (18%, 0.22 s), its string rings
    brighter and longer (0.25 -> 0.45 s, vibrating 0.9 s, less far along it), and the cursor pulses in the string's colour where it
    crosses it (0.35 s).
  - Streak build-up (`build`, toward x4; `warm` = build outside x4 / Encore): the track warms in the song's deep colour toward
    the cursor, the strings take on its light colour and brighten, the cursor's glow and the background mesh lift (heat 0.06 ->
    0.14), before x4's blue.
  - Checked on the library copy (I Love My Friends with a 25 streak, a hit forced at the cursor): 120 fps, no console errors.
- "I liked the dynamic stage lighting" / "I want it to feel epic": soft beams from a rig above behind the Arcade board (drawn at 0.3
  scale, blurred small and laid in; the song's colours, its loudness from a few dim beams to the whole rig, big moments fanning
  them with a sweep, kicks / snares pumping, accents snapping them, the streak adding fixtures; 121 fps). User: "Looks like a
  flash game still..."; taken out again. Drawn or rendered lights haven't worked in five tries (show.html, webgl.html, this);
  what's left untried is real footage (filmed stage lighting behind the board, as Guitar Hero Live used filmed video).
  Footage: the user's reference was a photo from inside a festival crowd (warm golden haze, lights through smoke, silhouettes
  with hands up); a shortlist was found on Pexels (free licence; e.g. 7722221, 16501141, 19679435, 720p files 1-15 MB), then
  "abandon this idea". Nothing was downloaded. The game keeps the song colours, hit feel and streak build-up.

## Open ideas / to do

- Windows polish: make the
  desktop shortcut install package updates like the Mac one, use an NVIDIA GPU (needs
  the CUDA build of torch; the default Windows build is CPU-only).
- Playlists / practice sets that pull songs from any artist.
- The user once asked for "filters": interpreted as folders; filtering the song list
  (by instrument removed, HQ, etc.) may still be wanted.
- Mac-only paths (app shortcut, MPS fallback, certificate fix) were written on Windows;
  they have been reported working on Mac but have no automated checks.
