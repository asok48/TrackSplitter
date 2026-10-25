"""Track Splitter: a small local app that takes instruments out of songs.

Upload songs, pick what to remove (bass, drums, vocals, guitar, piano), and
Demucs separates it out, so you get each song without it plus the removed part
on its own. The library keeps each song's original, so new versions can be
made later without uploading again.

This file is the engine plus a small local web server; the page itself is
static/index.html.
"""

import bisect
import hashlib
import json
import multiprocessing
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
import webbrowser
from pathlib import Path
from typing import Optional

import truststore

# Verify secure connections (link downloads, model downloads) against the certificates
# the computer itself trusts (the Keychain on a Mac), rather than Python's own list.
# That covers Pythons installed without certificates and work networks that use their
# own certificate.
truststore.inject_into_ssl()

# Let PyTorch run operations the Apple GPU (MPS) lacks on the CPU instead of failing.
# Must be set before torch is imported; run_model() covers anything that still fails.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import imageio_ffmpeg
import numpy as np
import torch
import uvicorn
import yt_dlp
from demucs.apply import apply_model
from demucs.pretrained import get_model
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from send2trash import send2trash

import tab
import tabimport

HOST, PORT = "127.0.0.1", 7860
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
STATIC_DIR = Path(__file__).parent / "static"
# Kept in the user's Music folder so the library survives re-copying the app.
OUTPUT_DIR = Path.home() / "Music" / "Track Splitter"
# Where earlier versions kept the library, back when the app was called Bass Remover.
OLD_OUTPUT_DIRS = [Path.home() / "Music" / "Bass Remover", Path(__file__).parent / "output"]
STAGE_DIR = Path(tempfile.gettempdir()) / "track-splitter-uploads"
SHIFT_DIR = Path(tempfile.gettempdir()) / "track-splitter-shifted"   # pitch-shifted audio cache
SAMPLERATE = 44100  # all Demucs v4 models use this
INSTRUMENTS = ["bass", "drums", "vocals", "guitar", "lead guitar", "rhythm guitar", "piano"]
GUITAR_PARTS = {"lead guitar", "rhythm guitar"}  # split out of the guitar stem by stereo position
SIX_STEM_ONLY = {"guitar", "piano"}  # htdemucs_6s (standard) or BS-RoFormer SW (HQ)
HQ_TAG = " [HQ]"
# "Song [HQ] (no bass & drums)" -> removed part is "bass & drums"
NO_RE = re.compile(r" \(no (.+)\)$")
UNKNOWN_ARTIST = "Unknown artist"
SONG_META = ".song.json"
if torch.cuda.is_available():
    DEVICE, DEVICE_NAME = "cuda", "NVIDIA GPU"
elif torch.backends.mps.is_available():
    DEVICE, DEVICE_NAME = "mps", "Apple GPU"
else:
    DEVICE, DEVICE_NAME = "cpu", "CPU"


# --- Separation engine --------------------------------------------------------

_bags = {}


def stem_model(stem, high_quality):
    """The model to run for one instrument.

    htdemucs_ft is a bag of 4 fine-tuned specialists, one per instrument, so we run
    only the specialist we need: same result, ~4x faster. The other models are a
    single network that separates everything at once.
    """
    if stem in SIX_STEM_ONLY:
        name = "htdemucs_6s"
    else:
        name = "htdemucs_ft" if high_quality else "htdemucs"
    if name not in _bags:
        _bags[name] = get_model(name)
        _bags[name].eval()
    bag = _bags[name]
    k = bag.sources.index(stem)
    return max(zip(bag.models, bag.weights), key=lambda mw: mw[1][k])[0]


# High quality guitar/piano: BS-RoFormer SW (by jarredou), a newer 6-stem model that
# separates those far better than htdemucs_6s, at roughly real-time speed on an M4.
# Downloaded once (~700 MB) into the same cache as the Demucs models.
ROFORMER_URL = ("https://github.com/nomadkaraoke/python-audio-separator/releases/download/"
                "model-configs/BS-Roformer-SW.ckpt")
ROFORMER_SHA256 = "24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e"
ROFORMER_SOURCES = ["bass", "drums", "other", "vocals", "guitar", "piano"]
ROFORMER_CHUNK = 588800  # samples per pass (~13 s), the length it was trained on
_roformer = None


def roformer_model(on_progress):
    global _roformer
    if _roformer is None:
        from roformer import BSRoformer
        weights = Path(torch.hub.get_dir()) / "checkpoints" / "BS-Roformer-SW.ckpt"
        if not weights.exists():
            download_model(ROFORMER_URL, weights, ROFORMER_SHA256, on_progress)
        on_progress(None, "Loading the high-quality model")
        model = BSRoformer(dim=256, depth=12, stereo=True, num_stems=6,
                           time_transformer_depth=1, freq_transformer_depth=1,
                           freqs_per_bands=(2,) * 24 + (4,) * 12 + (12,) * 8 + (24,) * 8 + (48,) * 8 + (128, 129))
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        _roformer = model.eval()
    return _roformer


def download_model(url, dest, sha256, on_progress):
    """Download a model file once, showing MB done, and check it arrived intact."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response, open(part, "wb") as f:
        total_mb = int(response.headers.get("Content-Length", 0)) >> 20
        done = 0
        while chunk := response.read(1 << 20):
            f.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            of = f" of {total_mb}" if total_mb else ""
            on_progress(None, f"Downloading the high-quality model (first time only) · {done >> 20}{of} MB")
    if digest.hexdigest() != sha256:
        part.unlink(missing_ok=True)
        raise RuntimeError("the high-quality model download was damaged; please try again")
    part.replace(dest)


def run_roformer(mix, on_progress, device=DEVICE, fast=False):
    """All six RoFormer stems for mix, shaped (6, 2, samples).

    The song goes through in overlapping chunks, crossfaded with sin² ramps that add
    up to 1, so there are no seams. Normally chunks overlap by half: every moment is
    separated twice and blended, as the model's own config does. fast overlaps them
    by just 1 s: half the work (~2x faster on an M4), output ~-20 dB different.
    On a GPU the model runs in half precision: ~10% faster, output within -61 dB.
    """
    model = roformer_model(on_progress).to(device)
    length = mix.shape[1]
    fade = SAMPLERATE if fast else ROFORMER_CHUNK // 2
    step = ROFORMER_CHUNK - fade
    # Pad so every real sample is fully covered and the last chunk is full.
    total = length + 2 * fade
    total += -(total - ROFORMER_CHUNK) % step
    x = np.pad(mix, ((0, 0), (fade, total - length - fade)))
    ramp = np.sin(np.pi / 2 * (np.arange(fade) + 0.5) / fade) ** 2
    window = np.ones(ROFORMER_CHUNK, np.float32)
    window[:fade], window[ROFORMER_CHUNK - fade:] = ramp, ramp[::-1]
    out = np.zeros((len(ROFORMER_SOURCES), 2, total), np.float32)
    weight = np.zeros(total, np.float32)
    starts = range(0, total - ROFORMER_CHUNK + 1, step)
    for i, s in enumerate(starts):
        chunk = torch.from_numpy(x[:, s:s + ROFORMER_CHUNK].copy())[None].to(device)
        with torch.no_grad(), torch.autocast(device, dtype=torch.float16, enabled=device != "cpu"):
            y = model(chunk)[0].float().cpu().numpy()
        n = min(y.shape[-1], ROFORMER_CHUNK)   # the inverse STFT can come back a little short
        out[..., s:s + n] += y[..., :n] * window[:n]
        weight[s:s + n] += window[:n]
        on_progress((i + 1) / len(starts))
    return out[..., fade:fade + length] / np.maximum(weight[fade:fade + length], 1e-6)


def decode(path, samplerate):
    """Decode an audio file to a float32 stereo array shaped (2, samples)."""
    cmd = [FFMPEG, "-v", "error", "-i", str(path),
           "-f", "f32le", "-ac", "2", "-ar", str(samplerate), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()


def encode_mp3(audio, samplerate, dest):
    cmd = [FFMPEG, "-v", "error", "-y", "-f", "f32le", "-ac", "2", "-ar", str(samplerate),
           "-i", "-", "-b:a", "320k", str(dest)]   # most originals are already lossy; keep the 2nd generation clean
    pcm = np.clip(audio, -1.0, 1.0).T.astype(np.float32).tobytes()
    subprocess.run(cmd, input=pcm, check=True)


def separate(mix, stems, high_quality, on_progress=lambda fraction=None, step=None: None, fast=False):
    """Return the chosen instruments summed together, same shape as mix.

    on_progress(fraction, step): how far along (0-1, or None) and what it's doing now.
    """
    wav = torch.from_numpy(mix)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std()
    x = ((wav - mean) / std)[None]

    # Lead and rhythm guitar are both cut out of the guitar stem.
    bases = list(dict.fromkeys("guitar" if s in GUITAR_PARTS else s for s in stems))
    # Group instruments by model so each model runs once per song. None = RoFormer.
    on_progress(None, "Loading the separation model")
    runs = {}
    for stem in bases:
        model = None if high_quality and stem in SIX_STEM_ONLY else stem_model(stem, high_quality)
        runs.setdefault(id(model), (model, []))[1].append(stem)

    parts = {}
    length = mix.shape[1]
    for r, (model, wanted) in enumerate(runs.values()):
        def progress(fraction=None, step=None, r=r):
            on_progress(None if fraction is None else (r + min(fraction, 1.0)) / len(runs), step)

        if model is None:
            roformer_model(progress)   # download/load outside run_model, so a failure there isn't retried
        quality = (" (high quality, faster)" if fast and model is None else " (high quality)") if high_quality else ""
        progress(0.0, f"Separating {stems_label(wanted)}{quality}")
        if model is None:
            out = run_model(lambda device: run_roformer(mix, progress, device, fast), progress)
            for stem in wanted:
                parts[stem] = out[ROFORMER_SOURCES.index(stem)]
            continue

        def callback(d):
            if d.get("state") == "end":
                progress(d["segment_offset"] / length)

        def demucs(device, model=model, callback=callback):
            with torch.no_grad():
                return apply_model(model, x, device=device, split=True, overlap=0.25,
                                   callback=callback)[0]
        out = run_model(demucs, progress)
        for stem in wanted:
            parts[stem] = (out[model.sources.index(stem)] * std).numpy()

    if GUITAR_PARTS & set(stems):
        on_progress(1.0, "Splitting lead and rhythm guitar")
        parts["lead guitar"] = center_of(parts["guitar"])
        parts["rhythm guitar"] = parts["guitar"] - parts["lead guitar"]
    return sum(parts[s] for s in stems)


def center_of(stereo):
    """The part of a stereo track that sits in the middle.

    There is no model for lead vs rhythm guitar, but mixes usually double-track the
    rhythm guitars and pan them hard left/right, with the lead in the middle. Two
    separate takes don't move in step, so averaged over a few frames left x right
    leaves only what both channels share: the center's power. Each time-frequency
    slice keeps that fraction of itself as center (lead); the rest is sides (rhythm).
    Comparing only loudness (the first version) counted doubled rhythm takes as center
    whenever they were equally loud; this roughly halves that bleed (measured on a mix
    with known parts: -7.3 -> -10.0 dB). Still only as good as the mix allows.
    """
    x = torch.from_numpy(np.ascontiguousarray(stereo))
    window = torch.hann_window(4096)
    spec = torch.stft(x, 4096, 1024, window=window, return_complex=True)
    left, right = spec

    def smooth(z):   # average over 3 neighboring frames
        return torch.nn.functional.avg_pool1d(z[:, None], 3, stride=1, padding=1,
                                              count_include_pad=False)[:, 0]

    shared = smooth((left * right.conj()).real).clamp(min=0)
    masks = [(shared / (smooth(ch.abs() ** 2) + 1e-12)).clamp(0, 1) ** 1.5 for ch in (left, right)]
    center = torch.stack([left * masks[0], right * masks[1]])
    return torch.istft(center, 4096, 1024, window=window, length=x.shape[1]).numpy()


def run_model(fn, on_progress=lambda fraction=None, step=None: None):
    """fn(device) on the GPU, retried on the CPU if the Apple GPU lacks an operation."""
    try:
        return fn(DEVICE)
    except (RuntimeError, NotImplementedError):
        if DEVICE != "mps":
            raise
        print("The Apple GPU couldn't run this model; using the CPU instead (slower).", flush=True)
        on_progress(0.0, "The Apple GPU couldn't run this; continuing on the CPU (slower)")
        return fn("cpu")


def stems_label(stems):
    """["bass", "drums", "vocals"] -> "bass, drums & vocals"."""
    return stems[0] if len(stems) == 1 else f"{', '.join(stems[:-1])} & {stems[-1]}"


def read_tags(path):
    """Title/artist tags from an audio file, or None if ffmpeg can't read it as audio."""
    result = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-f", "ffmetadata", "-"],
                            capture_output=True)
    if result.returncode != 0:
        return None
    tags = {}
    for line in result.stdout.decode("utf-8", "replace").splitlines():
        key, sep, value = line.partition("=")
        if sep and not line.startswith(";"):
            tags.setdefault(key.strip().lower(), re.sub(r"\\(.)", r"\1", value).strip())
    return tags


# "(Official Video)", "[Lyrics]", "(HD)"... as found in video and file names.
JUNK_RE = re.compile(r"\s*[(\[][^)\]]*\b(official|video|audio|lyrics?|visuali[sz]er|hd|4k|hq|remaster(ed)?)\b[^)\]]*[)\]]",
                     re.IGNORECASE)


def guess_title_artist(tags, name):
    """Best guess at a clean title and artist from tags, falling back to a file or video name."""
    title = tags.get("title", "")
    artist = tags.get("artist") or tags.get("album_artist") or ""
    if not title or (" - " in title and not artist):
        # "03 - Artist - Song_Name (Official Video)" -> artist "Artist", title "Song Name"
        stem = title or Path(name).stem
        stem = re.sub(r"^\s*\d{1,3}\s*[-._)]\s*", "", stem).replace("_", " ").strip()
        if " - " in stem and not artist:
            artist, title = stem.split(" - ", 1)
        else:
            title = stem
    title = JUNK_RE.sub("", title).strip()
    return clean_name(title) or "Untitled", clean_name(artist) or UNKNOWN_ARTIST


# --- Library -----------------------------------------------------------------
# On disk the library is  <Artist>/<Song>/  holding:
#   original.<ext>                the uploaded file, so new versions need no re-upload
#   <Song> (no bass).mp3          one pair of files per version
#   <Song> (bass only).mp3
#   .song.json                    the original's checksum, to spot repeat uploads
# Songs are identified by "Artist/Song", versions by "Artist/Song/<Song> (no bass)".

def clean_name(name):
    """Make typed text safe as a file or folder name ("" if nothing usable is left)."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "", name or "").strip().strip(".")


def rel(path):
    return path.relative_to(OUTPUT_DIR).as_posix()


def visible_dirs(parent):
    return sorted((p for p in parent.iterdir() if p.is_dir() and not p.name.startswith(".")),
                  key=lambda p: p.name.lower())


def library_path(id_, depth):
    """Resolve an artist (1), song (2) or version (3) id, refusing anything outside the library."""
    parts = (id_ or "").split("/")
    if len(parts) != depth or any(p in ("", ".", "..") or p != clean_name(p) for p in parts):
        raise HTTPException(404, "That item doesn't exist.")
    path = OUTPUT_DIR.joinpath(*parts)
    if not path.resolve().is_relative_to(OUTPUT_DIR.resolve()):
        raise HTTPException(404, "That item doesn't exist.")
    return path


def version_files(version):
    """(without, only) MP3 paths for a version id."""
    only = NO_RE.sub(r" (\1 only)", version)
    return OUTPUT_DIR / f"{version}.mp3", OUTPUT_DIR / f"{only}.mp3"


def read_meta(song_dir):
    try:
        return json.loads((song_dir / SONG_META).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_meta(song_dir, meta):
    (song_dir / SONG_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def version_key(song_dir, version):
    """A version's settings key in .song.json: its name minus the song title, e.g.
    " [HQ] (no bass)", so it survives renaming the song."""
    name = version.rsplit("/", 1)[-1]
    return name[len(song_dir.name):] if name.startswith(song_dir.name) else name


def original_file(song_dir):
    return next(iter(song_dir.glob("original.*")), None)


def song_info(song_dir):
    meta = read_meta(song_dir)
    settings = meta.get("versions", {})
    versions = []
    for p in song_dir.glob("* (no *).mp3"):
        m = NO_RE.search(p.stem)
        if m:
            made = settings.get(version_key(song_dir, p.stem), {})
            versions.append({"id": rel(p.with_suffix("")), "removed": m.group(1),
                             "level": made.get("level", 0), "fast": made.get("fast", False),
                             "hq": p.stem[: m.start()].endswith(HQ_TAG), "added": p.stat().st_mtime})
    versions.sort(key=lambda v: v["added"], reverse=True)
    added = meta.get("added") or max((v["added"] for v in versions), default=song_dir.stat().st_mtime)
    return {"id": rel(song_dir), "title": song_dir.name, "artist": song_dir.parent.name,
            "added": added, "has_original": original_file(song_dir) is not None,
            "my_bass": any(t.get("audio") for t in (read_mine(song_dir) or {}).get("takes", [])),
            "versions": versions}


def library():
    artists = visible_dirs(OUTPUT_DIR)
    songs = [song_info(d) for a in artists for d in visible_dirs(a)]
    songs.sort(key=lambda s: max([s["added"], *(v["added"] for v in s["versions"])]), reverse=True)
    return {"artists": [a.name for a in artists], "songs": songs}


def find_song_by_hash(sha1):
    for artist in visible_dirs(OUTPUT_DIR):
        for song_dir in visible_dirs(artist):
            if read_meta(song_dir).get("sha1") == sha1:
                return song_dir
    return None


def free_song_dir(artist, title):
    """A song folder path that isn't taken yet: "Title", then "Title (2)", ..."""
    path, n = OUTPUT_DIR / artist / title, 2
    while path.exists():
        path, n = OUTPUT_DIR / artist / f"{title} ({n})", n + 1
    return path


def rename_dir(src, dest):
    """Rename a folder, also when only the letter case changes (Mac/Windows ignore case)."""
    if src.exists() and dest.exists() and src.samefile(dest) and src.name != dest.name:
        tmp = src.with_name(f".renaming-{uuid.uuid4().hex}")
        src.rename(tmp)
        src = tmp
    src.rename(dest)


def rename_song_files(song_dir, old_title, new_title):
    for p in song_dir.glob("*.mp3"):
        if p.name.startswith(old_title):
            p.rename(song_dir / (new_title + p.name[len(old_title):]))


def restructure_flat_library():
    """Turn the earlier flat layout ("Artist/Song (no bass).mp3") into song folders."""
    if not OUTPUT_DIR.is_dir():
        return
    flat = [*OUTPUT_DIR.glob("* (no *).mp3"), *OUTPUT_DIR.glob("*/* (no *).mp3")]
    for without in flat:
        m = NO_RE.search(without.stem)
        if not m or not without.exists():
            continue
        title = without.stem[: m.start()]
        if title.endswith(HQ_TAG):
            title = title[: -len(HQ_TAG)]
        artist = without.parent.name if without.parent != OUTPUT_DIR else UNKNOWN_ARTIST
        song_dir = OUTPUT_DIR / artist / (clean_name(title) or "Untitled")
        song_dir.mkdir(parents=True, exist_ok=True)
        only = without.with_name(NO_RE.sub(r" (\1 only)", without.stem) + ".mp3")
        for f in (without, only):
            if f.exists() and not (song_dir / f.name).exists():
                f.replace(song_dir / f.name)
        if not (song_dir / SONG_META).exists():
            write_meta(song_dir, {"added": song_dir.stat().st_mtime})


def migrate_old_library():
    """Move tracks and artist folders from earlier library locations into this one."""
    for old in OLD_OUTPUT_DIRS:
        if not old.is_dir():
            continue
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for entry in old.iterdir():
            dest = OUTPUT_DIR / entry.name
            if entry.is_dir() and dest.is_dir():
                # Artist folder exists in both: merge its tracks in.
                for f in entry.glob("*.mp3"):
                    if not (dest / f.name).exists():
                        f.replace(dest / f.name)
            elif not dest.exists() and (entry.is_dir() or entry.suffix == ".mp3"):
                entry.replace(dest)
        # Remove the old folders once everything has moved out.
        for d in sorted((p for p in old.rglob("*") if p.is_dir()), reverse=True):
            if not any(d.iterdir()):
                d.rmdir()
        if not any(old.iterdir()):
            old.rmdir()
    restructure_flat_library()


# --- Uploads and processing jobs ------------------------------------------------
# Adding songs is two steps: files are staged first (so the page can show the
# title/artist read from their tags and flag songs already in the library), then
# processed. Processing runs one job at a time on a background thread; the page
# polls the job for progress.

staged = {}  # stage id -> {"path", "filename", "sha1", "created"}
jobs = {}
job_queue = queue.Queue()


def make_version(source, song_dir, stems, level, high_quality, on_progress, fast=False):
    on_progress(0.0, "Reading the song")
    mix = decode(source, SAMPLERATE)
    removed = separate(mix, stems, high_quality, on_progress, fast)
    on_progress(1.0, "Saving")
    # Subtracting the separated parts from the original keeps everything else untouched.
    without = mix - removed * (1.0 - level / 100.0)
    # Loud masters often go past full scale once a part is taken out; clipping that
    # crackles. Turn both files down by the same amount instead, so they still add up.
    peak = max(np.abs(without).max(), np.abs(removed).max(), np.abs(without + removed).max())
    if peak > 0.98:
        without, removed = without * (0.98 / peak), removed * (0.98 / peak)

    # The tag keeps standard and high-quality versions side by side.
    version = f"{rel(song_dir)}/{song_dir.name}{HQ_TAG if high_quality else ''} (no {stems_label(stems)})"
    song_dir.mkdir(parents=True, exist_ok=True)
    without_path, only_path = version_files(version)
    encode_mp3(without, SAMPLERATE, without_path)
    encode_mp3(removed, SAMPLERATE, only_path)
    # The name records instruments and HQ; keep the rest too, so Redo can repeat it exactly.
    meta = read_meta(song_dir)
    meta.setdefault("versions", {})[version_key(song_dir, version)] = {"level": level, "fast": fast}
    write_meta(song_dir, meta)
    return version


# --- Bass tab ---------------------------------------------------------------------
# Per song, in <song>/.bass-tab.json: the bass notes, the lines (bars), and fingerings
# per tuning (made on first request, they take under a second). See tab.py.

TAB_FILE = ".bass-tab.json"
TUNING_RULES = 3   # bump when tab.suggest_tuning changes, so saved tabs get a new suggestion
NOTES_RULES = 4    # bump when the notes would come out differently; older tabs ask to be remade


def bass_track(song_dir):
    """An existing "bass only" file for exactly bass (HQ preferred), or None."""
    tracks = [p for p in song_dir.glob("* (bass only).mp3")]
    return min(tracks, key=lambda p: HQ_TAG not in p.name, default=None)


def read_tab(song_dir):
    try:
        return json.loads((song_dir / TAB_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def half_rate(x):
    """44.1 kHz stereo -> 22.05 kHz mono (what tab.py works at)."""
    return (x[:, 0::2][:, :x.shape[1] // 2] + x[:, 1::2][:, :x.shape[1] // 2]).mean(0) / 2


def tab_audio(song_dir, track):
    """(bass, whole song or None) at tab.SAMPLERATE, from a "bass only" track."""
    source = original_file(song_dir)
    song = half_rate(decode(source, SAMPLERATE)) if source else None
    bass = decode(track, tab.SAMPLERATE).mean(0)
    partner = track.with_name(track.name.replace("(bass only)", "(no bass)"))
    if song is None and partner.exists():
        # No saved original: its "no bass" version plus the bass add up to the song.
        rest = decode(partner, tab.SAMPLERATE).mean(0)
        n = min(len(rest), len(bass))
        song = rest[:n] + bass[:n]
    return bass, song


def make_tab(song_dir, report):
    """Isolate the bass (or reuse a bass track), transcribe it, find the bars, save."""
    report(0.0, "Reading the song")
    track = bass_track(song_dir)
    if track:
        bass, song = tab_audio(song_dir, track)
    else:
        mix = decode(original_file(song_dir), SAMPLERATE)
        song = half_rate(mix)

        def separating(fraction=None, step=None):
            report(None if fraction is None else 0.7 * fraction, step and step.replace("Separating", "Isolating"))
        bass = half_rate(separate(mix, ["bass"], True, separating))
    report(0.7, "Writing out the notes")
    settings, models = note_settings(), note_models()
    activations = {}
    notes = tab.transcribe(bass, lambda f: report(0.7 + 0.2 * f), mix=song, keep=activations,
                           **{k: v for k, v in settings.items() if k not in ("grid_split", "models")})
    report(0.9, "Finding the beat")
    grid = tab.lines_for(song if song is not None else bass, notes, len(bass) / tab.SAMPLERATE)
    # Repeated plucks of one note can come out as one long note. Splitting them on the
    # beat grid helps driving 8ths (Getting Along, You Only Live Once) but chops held
    # notes where drums leak into the bass track (Electra, checked against the user's
    # Songsterr tab); no audio test told the two apart, so both are kept and the page
    # has a per-song "Split repeated notes" switch.
    beat_grid = grid.pop("grid")
    if models:   # learned from imported tabs: octaves, re-plucks (the split version) and fake notes
        ctx = tab.note_context(bass, song, activations, beat_grid)
        split = tab.apply_note_models(models, ctx, notes, split=True)
        notes = tab.apply_note_models(models, ctx, notes, split=False)
    else:
        ratio, level = settings["grid_split"] or (1.35, 0.2)   # learned from imported tabs: also on by default
        split = tab.split_on_grid(notes, beat_grid, bass, ratio=ratio, activations=activations, onset_level=level)
    report(0.95, "Working out the fingering")
    suggested = tab.suggest_tuning(notes)
    rows = lambda ns: [[round(s, 3), round(e, 3), p, round(a, 2)] for s, e, p, a in ns]
    data = {"source": "bass track" if track else "separated", "made": time.time(),
            "notes": rows(notes), "notes_split": rows(split),
            **grid, "suggested": suggested, "maybe_half_down": tab.maybe_half_down(notes),
            "tuning_rules": TUNING_RULES, "notes_rules": NOTES_RULES, "note_settings": settings,
            "split_default": bool(settings["grid_split"] or (models and models.get("pluck"))),
            "fingering": {suggested: tab.fingering(notes, suggested, learned_style()[0])},
            "fingering_style": learned_style()[1]}
    (song_dir / TAB_FILE).write_text(json.dumps(data), encoding="utf-8")


# --- Imported tabs ------------------------------------------------------------------
# A tab the user trusts (Guitar Pro / MusicXML file, or text pasted from a tab site),
# synced to the recording (tabimport.py). Kept next to the song: the file as given
# (imported-tab.<ext>) and the synced result (.bass-tab-imported.json).

IMPORT_FILE = ".bass-tab-imported.json"
# Bump when syncing changes; older imports are synced again from their file on open.
# 2: bars follow tempo drift up to +-2.5 s on one smooth path, and heard notes only nudge
#    written ones (Save Yourself, I'll Hold Them Back was 1.3-1.7 s early for two minutes
#    and 15% of its notes were squeezed together).
# 3: notes keep whether they're slid or hammered into (Guitar Pro marks), for the game.
IMPORT_RULES = 3
TAB_EXTENSIONS = {".gp", ".gp3", ".gp4", ".gp5", ".gpx", ".musicxml", ".xml", ".mxl", ".txt"}


def note_name(p):
    return ["C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B"][p % 12]


def import_tab(song_dir, source, report):
    report(0.0, "Reading the tab")
    tracks = tabimport.read_file(source)
    if not tracks:
        raise ValueError("no tab notes found in it")
    track = tracks[0]                               # bass-looking tracks come first
    data = read_tab(song_dir)
    if data is None or data.get("notes_rules") != NOTES_RULES:
        # The sync needs the notes heard in the recording: make the automatic tab first.
        make_tab(song_dir, lambda f=None, step=None: report(None if f is None else 0.8 * f, step))
        data = read_tab(song_dir)
    report(0.85, "Lining the tab up with the recording")
    synced = tabimport.sync(track, [tuple(n) for n in data["notes"]], 60 / data["bpm"] if data.get("bpm") else None)
    bars = synced["bars"]
    end = max((n[1] for n in synced["notes"]), default=bars[-1] if bars else 0.0)
    bar_len = float(np.median(np.diff(bars))) if len(bars) > 1 else 2.0
    per_line = int(np.clip(round(5.0 / max(bar_len, 0.1)), 1, 4))
    edges = bars[::per_line] + [max(end, bars[-1] + bar_len)]
    lines = [[float(a), float(b), [float(x) for x in bars if a <= x < b]] for a, b in zip(edges, edges[1:]) if b > a]
    key = next((k for k, (_, strings, _) in tab.TUNINGS.items() if strings == track.strings), None)
    names = tab.TUNINGS[key][2] if key else [note_name(p) for p in track.strings]
    out = {"name": "Pasted tab" if source.name == "imported-tab.txt" else source.name.replace("imported-tab - ", ""),
           "track": track.name, "made": time.time(), "quality": synced["quality"], "tuning": key or "custom",
           "label": tab.TUNINGS[key][0] if key else "Tab's tuning (" + " ".join(names) + ")",
           "strings": names, "open": track.strings, "notes": synced["notes"], "lines": lines, "bpm": data.get("bpm"),
           "rules": IMPORT_RULES}
    (song_dir / IMPORT_FILE).write_text(json.dumps(out), encoding="utf-8")


# --- My tab (recorded) ---------------------------------------------------------------
# The user plays along through an audio interface (a Focusrite Scarlett, instrument
# input): the page records the clean bass and uploads each take with the song time it
# started at. A clean bass signal has none of the leakage of a separated track, so the
# take is transcribed as is (tab.NOTES; the learned settings and models are for
# separated tracks). A take replaces the notes in the stretch it covers, so a section
# can be redone. Saved as .bass-tab-mine.json: notes [start, end, MIDI pitch], fingered
# on request in any tuning (learned fingering), with the automatic tab's lines.

MINE_FILE = ".bass-tab-mine.json"
MY_BASS_DIR = ".my-bass"   # each take's audio (FLAC), for playing your bass with the band


def read_mine(song_dir):
    try:
        return json.loads((song_dir / MINE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def align_take(take, song_dir, song_start, search=0.4):
    """Seconds to add to the take's times so its attacks line up with the song's bass
    (the bass track, or the whole song), searching +-`search` s around where the page
    says it started; 0 if nothing lines up clearly. Covers what the page can't measure:
    the interface's input delay and how late the song reached the player's ears."""
    track = bass_track(song_dir)
    ref = decode(track, tab.SAMPLERATE).mean(0) if track else half_rate(decode(original_file(song_dir), SAMPLERATE))
    a = int(max(0.0, song_start - search - 0.1) * tab.SAMPLERATE)
    seg = ref[a:a + len(take) + int((2 * search + 0.2) * tab.SAMPLERATE)]
    (fa, fps), (fb, _) = tab._attack_strength(take), tab._attack_strength(seg)
    if len(fa) < fps or len(fb) < fps:
        return 0.0
    z = lambda f: (np.log1p(f) - np.log1p(f).mean()) / (np.log1p(f).std() + 1e-9)
    fa, fb = z(fa), z(fb)
    base = int(round((song_start - a / tab.SAMPLERATE) * fps))   # take frame 0 <-> segment frame `base`
    lags = range(-int(search * fps), int(search * fps) + 1)
    score = []
    for lag in lags:
        j = base + lag
        n = min(len(fa), len(fb) - j) if j >= 0 else 0
        score.append(float(np.dot(fa[:n], fb[j:j + n]) / n) if n > fps else -1.0)
    score = np.array(score)
    best = int(np.argmax(score))
    clear = score[best] > 0.1 and score[best] > np.median(score) + 3 * score.std() / 2
    return lags[best] / fps if clear else 0.0


def playing_level(x, rate):
    """How loud someone is while playing: the 90th percentile of 50 ms RMS (so rests
    and pauses don't count)."""
    hop = rate // 20
    frames = x[:len(x) // hop * hop].reshape(-1, hop)
    return float(np.percentile(np.sqrt((frames ** 2).mean(axis=1)), 90)) if len(frames) else 0.0


def my_bass(song_dir, shift):
    """All takes as one float WAV on the song's timeline (a later take wins where takes
    overlap), pitch-moved to match the player's `shift`; cached until a new take."""
    mine = read_mine(song_dir) or {}
    takes = [t for t in mine.get("takes", []) if t.get("audio") and (song_dir / MY_BASS_DIR / t["audio"]).exists()]
    if not takes:
        raise HTTPException(404, "No recorded bass for this song.")
    key = hashlib.sha1(f"{song_dir}|{mine['made']}|{shift}".encode()).hexdigest()
    dest = SHIFT_DIR / f"{key}-mine.wav"
    if dest.exists():
        return dest
    out = np.zeros(int(max(t["end"] for t in takes) * SAMPLERATE) + SAMPLERATE, np.float32)
    track = bass_track(song_dir)
    band_bass = decode(track, SAMPLERATE).mean(0) if track else None
    for t in takes:
        src = song_dir / MY_BASS_DIR / t["audio"]
        move = shift - t.get("shift", 0)          # recorded raised, played back as is (or the other way)
        x = decode(shifted(src, move) if move else src, SAMPLERATE).mean(0)
        # As loud as the record's own bass there (the user keeps the interface gain low,
        # so takes come in quiet), or a sensible level without a bass track.
        a0, a1 = int(t["start"] * SAMPLERATE), int(t["end"] * SAMPLERATE)
        target = playing_level(band_bass[max(0, a0):a1], SAMPLERATE) if band_bass is not None else 0.0
        level = playing_level(x, SAMPLERATE)
        if level > 1e-6:
            x = x * float(np.clip((target or 0.1) / level, 0.1, 300))
        a = int(round(t["start"] * SAMPLERATE))
        x = x[max(0, -a):][:len(out) - max(a, 0)]
        out[max(a, 0):max(a, 0) + len(x)] = x
    SHIFT_DIR.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "f32le", "-ar", str(SAMPLERATE), "-ac", "1", "-i", "-",
                    "-ac", "2", "-c:a", "pcm_f32le", "-f", "wav", str(part)], input=out.tobytes(), check=True)
    part.replace(dest)
    return dest


MINE_RULES = 3   # bump when take_notes or apply_take changes: recorded tabs are redone from their takes' audio


def take_notes(take, on_progress=lambda fraction: None):
    """Notes of a clean take (times from its start)."""
    loud = float(np.percentile(np.abs(take), 99.9))
    if loud > 1e-5:   # a quiet take (low interface gain) brought up to a normal level first
        take = take * min(0.5 / loud, 1000.0)
    return tab.drop_ringing_octaves(tab.transcribe(take, on_progress), take)


def without_removed(notes, removed):
    """Notes minus those the user removed ([start, pitch]; redone takes may move a note
    by a few ms, so within 60 ms)."""
    return [n for n in notes if not any(abs(n[0] - r[0]) < 0.06 and n[2] == r[1] for r in removed)]


def apply_take(notes, new, start, end, mode="replace", bars=None):
    """A take's notes into the tab: "replace" takes over the stretch it covers; "add"
    keeps what's there and fills only the gaps: a note played within 0.25 s of one the
    draft already has (any pitch) is left out, and added notes snap to the 16th-note grid
    of their bar (within 60 ms) when the draft has bars. It used to add every note not
    within 70 ms of a same-pitch one; a take a little off the tab's timing then put
    notes between the tab's (Amsterdam: 61 -> 103 notes in 18 s), which the user saw as
    the tab speeding up: "keep the same speed and structure"."""
    if mode == "add":
        have = sorted(o[0] for o in notes)
        bars = sorted(bars or [])

        def near(t):
            k = bisect.bisect_left(have, t)
            return any(abs(have[m] - t) < 0.25 for m in (k - 1, k) if 0 <= m < len(have))

        def snap(t):
            k = bisect.bisect_right(bars, t) - 1
            if k < 0 or k + 1 >= len(bars):
                return t
            step = (bars[k + 1] - bars[k]) / 16
            grid = bars[k] + round((t - bars[k]) / step) * step
            return grid if abs(grid - t) <= 0.06 else t

        added = []
        for n in new:
            if near(n[0]):
                continue
            t = round(snap(n[0]), 3)
            added.append([t, round(n[1] + t - n[0], 3), n[2]])
        return sorted(notes + added)
    return sorted([n for n in notes if not start <= n[0] < end] + new)


def bar_starts(mine):
    """A draft's bar times (from the tab it started from), for snapping added notes."""
    return sorted({t for a, b, inner in mine.get("lines") or [] for t in (a, b, *inner)})


def redo_mine(song_dir, mine):
    """A recorded tab (draft) made again from the tab it started from and its takes'
    saved audio, in recording order (the lining up is kept), when the way takes become
    notes has improved."""
    notes = [list(n) for n in mine.get("base", [])]
    for t in mine["takes"]:
        path = song_dir / MY_BASS_DIR / t.get("audio", "")
        if not t.get("audio") or not path.exists():   # no audio: keep what it gave
            notes = apply_take(notes, [n for n in mine["notes"] if t["start"] <= n[0] < t["end"]], t["start"], t["end"])
            continue
        new = [[round(t["start"] + s, 3), round(t["start"] + e, 3), p - t.get("shift", 0)]
               for s, e, p, a in take_notes(decode(path, tab.SAMPLERATE).mean(0))]
        notes = apply_take(notes, new, t["start"], t["end"], t.get("mode", "replace"), bar_starts(mine))
    mine.update(notes=sorted(without_removed(notes, mine.get("removed", []))), rules=MINE_RULES)
    (song_dir / MINE_FILE).write_text(json.dumps(mine), encoding="utf-8")
    return mine


def record_tab(song_dir, take_path, song_start, shift, report, mode="replace"):
    report(0.0, "Reading your recording")
    take = decode(take_path, tab.SAMPLERATE).mean(0)
    data = read_tab(song_dir)
    if data is None or data.get("notes_rules") != NOTES_RULES:
        # The lines (bars) come from the automatic tab: make it first.
        make_tab(song_dir, lambda f=None, step=None: report(None if f is None else 0.5 * f, step))
    report(0.55, "Writing out your notes")
    notes = take_notes(take, lambda f: report(0.55 + 0.3 * f))
    report(0.88, "Lining your take up with the song")
    start = song_start + align_take(take, song_dir, song_start)
    # Played on a standard bass with the song raised a half step: the tab is for the song as is.
    new = [[round(start + s, 3), round(start + e, 3), p - shift] for s, e, p, a in notes]
    end = start + len(take) / tab.SAMPLERATE
    # Keep the take's audio too, to play it with the band ("With my bass").
    made = time.time()
    audio = f"take-{int(made * 1000)}.flac"
    (song_dir / MY_BASS_DIR).mkdir(exist_ok=True)
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(take_path), "-ac", "1", "-ar", str(SAMPLERATE),
                    "-c:a", "flac", str(song_dir / MY_BASS_DIR / audio)], check=True)
    mine = read_mine(song_dir) or {"notes": [], "takes": []}
    if mode == "add":   # removed notes played again come back
        mine["removed"] = [r for r in mine.get("removed", []) if not any(abs(n[0] - r[0]) < 0.06 and n[2] == r[1] for n in new)]
    else:               # new playing over the whole stretch
        mine["removed"] = [r for r in mine.get("removed", []) if not start <= r[0] < end]
    mine.update(made=made, notes=apply_take(mine["notes"], new, start, end, mode, bar_starts(mine)), rules=MINE_RULES,
                takes=mine["takes"] + [{"start": round(start, 3), "end": round(end, 3), "shifted_by": round(start - song_start, 3),
                                        "notes": len(new), "made": made, "audio": audio, "shift": shift, "mode": mode}])
    (song_dir / MINE_FILE).write_text(json.dumps(mine), encoding="utf-8")


# --- Learning from imported tabs ---------------------------------------------------
# Every imported tab is a right answer for its song: which notes are played (tuning the
# note detection, tab.learn_notes) and where (the fingering costs, tab.learn_style).
# Both are fitted to all imported tabs and kept in the library. Each is used only when
# it beats the built-in settings on the half of every tab it wasn't fitted on. Saved
# fingerings remember which learned costs made them and are redone when those change;
# automatic tabs made with other note settings ask to be made again.

LEARNED_FILE = OUTPUT_DIR / ".learned-from-tabs.json"
LEARN_RULES = 2   # bump when learning changes, so saved learning is redone at the next start


def read_learned():
    try:
        return json.loads(LEARNED_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def imported_tabs():
    """{song folder: its imported tab} for every song with one."""
    out = {}
    for f in OUTPUT_DIR.glob(f"*/*/{IMPORT_FILE}"):
        data = read_import(f.parent)
        if data:
            out[f.parent] = data
    return out


def learn_sources():
    """What learning was fitted to: each imported tab's time, with the user's "make repeats
    match" choices when it has some (changing them re-learns the fingering)."""
    return {rel(song_dir): [data["made"], data["unify"]] if data.get("unify") else data["made"]
            for song_dir, data in imported_tabs().items()}


def learn_from_imports(report=lambda fraction=None, step=None: None):
    """Fit the note settings and the fingering costs to all imported tabs and save them."""
    report(0.0, "Reading your imported tabs")
    sources = learn_sources()   # as they are now: changes made meanwhile cause another round
    fingers, songs = [], []
    for song_dir, data in imported_tabs().items():
        report(None, f"Reading {song_dir.name}")
        strings = data.get("open") or (tab.TUNINGS[data["tuning"]][1] if data["tuning"] in tab.TUNINGS else None)
        if not strings:
            continue
        # The tab as the user plays it: their "make repeats match" choices are right answers too.
        played = [(n[0], strings[n[2]] + n[3], n[2], n[3]) for n in unified_notes(data) if not n[4]]
        fingers.append((strings, played))
        track = bass_track(song_dir)
        if track and played:   # note learning reruns detection, so it needs the bass on its own
            bass, song = tab_audio(song_dir, track)
            songs.append({**tab.note_example(bass, song, [(t, p) for t, p, _, _ in played]), "name": song_dir.name})
        report(0.2 * len(fingers) / max(1, len(imported_tabs())))

    notes = None
    if songs:
        report(0.2, "Learning which notes your tabs play")
        a = tab.learn_notes(songs, 0, lambda f: report(0.2 + 0.15 * f))
        b = tab.learn_notes(songs, 1, lambda f: report(0.35 + 0.15 * f))
        # Each half scored with the settings fitted on the other one, vs the built-in ones.
        before = (tab.f1(tab.note_counts(songs, tab.NOTES, 1)) + tab.f1(tab.note_counts(songs, tab.NOTES, 0))) / 2
        after = (tab.f1(tab.note_counts(songs, a, 1)) + tab.f1(tab.note_counts(songs, b, 0))) / 2
        better = after > before + 0.005   # one song can't show a smaller gain is real
        notes = {"settings": tab.learn_notes(songs, None, lambda f: report(0.5 + 0.1 * f)) if better else None,
                 "tabs": len(songs), "before": round(before, 3), "after": round(after, 3)}

    models = None
    if len(songs) >= 2:   # each song is checked with models trained on the other songs
        report(0.6, "Learning which sounds are real notes")
        used = {**((notes or {}).get("settings") or tab.NOTES), "grid_split": None}
        items = [(tab.note_context(ex["bass"], ex["mix"], ex, ex["grid"]), tab.detect(ex, used), ex["truth"]) for ex in songs]
        before, after = np.zeros(3), np.zeros(3)
        for k, (ctx, found, truth) in enumerate(items):
            report(0.6 + 0.05 * k / len(items), f"Checking what it learned on {songs[k]['name']}")
            trial = tab.learn_note_models(items[:k] + items[k + 1:])
            before += tab.model_counts(truth, found)
            after += tab.model_counts(truth, tab.apply_note_models(trial, ctx, found))
        before, after = tab.f1(before), tab.f1(after)
        models = {"models": tab.learn_note_models(items) if after > before + 0.005 else None,
                  "tabs": len(items), "before": round(before, 3), "after": round(after, 3)}

    report(0.65, "Learning how your tabs finger the notes")
    halves = [[(s, ns[:len(ns) // 2]) for s, ns in fingers], [(s, ns[len(ns) // 2:]) for s, ns in fingers]]
    first = tab.learn_style(halves[0], on_progress=lambda f: report(0.65 + 0.1 * f))
    second = tab.learn_style(halves[1], on_progress=lambda f: report(0.75 + 0.1 * f))
    n0, n1 = (max(1, sum(len(ns) for _, ns in h)) for h in halves)
    before = (tab.style_match(halves[1]) * n1 + tab.style_match(halves[0]) * n0) / (n0 + n1)
    after = (tab.style_match(halves[1], first) * n1 + tab.style_match(halves[0], second) * n0) / (n0 + n1)
    style = tab.learn_style(fingers, on_progress=lambda f: report(0.85 + 0.15 * f)) if after > before else None
    temp = LEARNED_FILE.with_suffix(".tmp")   # written whole, then swapped in: tabs read it meanwhile
    temp.write_text(json.dumps({
        "made": time.time(), "sources": sources, "rules": LEARN_RULES, "notes": notes, "note_models": models,
        "fingering": {"style": style, "tabs": len(fingers), "before": round(before, 3), "after": round(after, 3)}}),
        encoding="utf-8")
    temp.replace(LEARNED_FILE)


def learned_style():
    """(learned fingering costs or None, their id) for fingering the automatic tabs."""
    saved = read_learned()
    style = saved and saved.get("fingering", {}).get("style")
    return (style, saved["made"]) if style else (None, None)


def note_settings():
    """The note settings automatic tabs are made with: learned, or tab.NOTES; plus which
    learned note models ("models": when they were made), if any."""
    saved = read_learned() or {}
    settings = dict((saved.get("notes") or {}).get("settings") or tab.NOTES)
    if note_models():
        settings["models"] = saved["made"]
    return settings


def note_models():
    """The learned note models (tab.apply_note_models), or None."""
    return ((read_learned() or {}).get("note_models") or {}).get("models")


class Stopped(Exception):
    """The user pressed Stop. Deliberately not a RuntimeError, so run_model doesn't retry."""


def worker():
    while True:
        job_id, settings = job_queue.get()
        job = jobs[job_id]
        if job.get("stop"):  # stopped while still waiting its turn
            job.update(status="done", stopped=True)
            continue
        job["status"] = "running"

        def report(fraction=None, step=None):
            """Engine progress: how far along (0-1) and/or what it's doing now.

            The engine calls this every few seconds, so it's also where Stop takes effect.
            """
            if job.get("stop"):
                raise Stopped
            if fraction is not None:
                job["fraction"] = fraction
            if step:
                job["step"] = step

        for i, item in enumerate(settings["items"]):
            job.update(index=i, song=item["title"], fraction=0.0, step="Starting")
            try:
                song_dir = item["song_dir"]
                if settings.get("kind") == "tab":
                    make_tab(song_dir, report)
                    job["done"].append(rel(song_dir))
                    continue
                if settings.get("kind") == "record":
                    try:
                        record_tab(song_dir, item["path"], item["song_start"], item["shift"], report, item.get("mode", "replace"))
                    finally:
                        item["path"].unlink(missing_ok=True)
                    job["done"].append(rel(song_dir))
                    continue
                if settings.get("kind") == "import":
                    import_tab(song_dir, item["path"], report)
                    job["done"].append(rel(song_dir))
                    queue_learning()
                    continue
                if item.get("stage"):
                    # New song: separate first, so a failed file leaves no empty song behind.
                    stage = item["stage"]
                    version = make_version(stage["path"], song_dir, settings["stems"],
                                           settings["level"], settings["hq"],
                                           on_progress=report, fast=settings["fast"])
                    if not original_file(song_dir):
                        shutil.copy2(stage["path"], song_dir / f"original{stage['path'].suffix.lower()}")
                        write_meta(song_dir, {**read_meta(song_dir),   # keeps the version's level
                                              "sha1": stage["sha1"], "source": stage["filename"],
                                              "added": time.time()})
                else:
                    version = make_version(original_file(song_dir), song_dir, settings["stems"],
                                           settings["level"], settings["hq"],
                                           on_progress=report, fast=settings["fast"])
                job["done"].append(version)
            except Stopped:
                # Nothing is saved before separation finishes, so there's nothing to undo.
                # Uploads not yet processed stay staged, so the page can start them again.
                job["stopped"] = True
                break
            except subprocess.CalledProcessError:
                job["failed"].append(f"{item['title']} (couldn't read the file)")
            except Exception as e:  # keep going with the rest of the batch
                job["failed"].append(f"{item['title']} ({e})")
            finally:
                if not job.get("stopped"):
                    job["finished"].append(i)   # item index, so the page knows which to clear
                    if item.get("stage"):
                        item["stage"]["path"].unlink(missing_ok=True)
                        staged.pop(item["stage_id"], None)
        job.update(status="done", fraction=1.0)


# --- Web API -------------------------------------------------------------------

api = FastAPI(title="Track Splitter")


# The app's own code as it was when it started: when an update changes it, the running
# app is still the old code (a page reload isn't enough), and the page says to restart.
CODE_FILES = [Path(__file__).resolve().parent / f for f in ("app.py", "tab.py", "tabimport.py")]
CODE_AT_START = {f: f.stat().st_mtime for f in CODE_FILES if f.exists()}


@api.get("/api/info")
def info():
    updated = any(f.exists() and f.stat().st_mtime > m for f, m in CODE_AT_START.items())
    return {"device": DEVICE_NAME, "library": str(OUTPUT_DIR), "instruments": INSTRUMENTS,
            "unknown_artist": UNKNOWN_ARTIST, "restart_needed": updated}


@api.get("/api/library")
def get_library():
    return library()


def shifted(source, semitones):
    """source raised/lowered by whole semitones at the same tempo, as a cached float WAV.

    Played faster by the pitch ratio (resample), then stretched back to the original
    length (ffmpeg's atempo; this ffmpeg has no rubberband). For playing along on a bass
    tuned differently from the record, e.g. a half-step-down song on a standard bass.
    The stretch works on each file separately, so shifted parts no longer add up to the
    shifted original sample for sample (they stay within ~1 ms); the player knows."""
    stat = source.stat()
    key = hashlib.sha1(f"{source}|{stat.st_mtime_ns}|{semitones}".encode()).hexdigest()
    dest = SHIFT_DIR / f"{key}.wav"
    if not dest.exists():
        SHIFT_DIR.mkdir(parents=True, exist_ok=True)
        r = 2 ** (semitones / 12)
        f = f"aresample={SAMPLERATE},asetrate={SAMPLERATE * r:.4f},aresample={SAMPLERATE},atempo={1 / r:.6f}"
        part = dest.with_suffix(".part")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(source), "-ac", "2", "-af", f,
                        "-c:a", "pcm_f32le", "-f", "wav", str(part)], check=True)
        part.replace(dest)
    return dest


@api.get("/api/audio")
def audio(version: str, part: str, shift: int = 0):
    """A version's without/only file or the song's original; shift: semitones (-2..2)."""
    path = library_path(version, 3)
    if not -2 <= shift <= 2:
        raise HTTPException(400, "The pitch can be moved by up to 2 semitones.")
    if part == "mine":   # the user's recorded takes ("With my bass")
        return FileResponse(my_bass(path.parent, shift), media_type="audio/wav")
    if part == "original":
        # The song's own original, for the player's Original mode, decoded exactly as
        # separation does (44.1 kHz) so it lines up with the versions. Float WAV keeps
        # peaks above full scale (common in lossy files) that 16-bit would clip, and
        # every browser can decode it.
        source = original_file(path.parent)
        if not source:
            raise HTTPException(404, "This song has no saved original.")
        if shift:
            return FileResponse(shifted(source, shift), media_type="audio/wav")
        cmd = [FFMPEG, "-v", "error", "-i", str(source), "-ac", "2", "-ar", str(SAMPLERATE),
               "-c:a", "pcm_f32le", "-f", "wav", "-"]
        return Response(subprocess.run(cmd, capture_output=True, check=True).stdout, media_type="audio/wav")
    without, only = version_files(rel(path))
    path = without if part == "without" else only
    if not path.exists():
        raise HTTPException(404, "That version doesn't exist.")
    if shift:
        return FileResponse(shifted(path, shift), media_type="audio/wav")
    return FileResponse(path, media_type="audio/mpeg")


def forget_old_uploads():
    """Drop uploads that were staged but never processed after a few hours."""
    for sid, s in list(staged.items()):
        if time.time() - s["created"] > 6 * 3600:
            s["path"].unlink(missing_ok=True)
            staged.pop(sid, None)


def register_staged(sid, path, filename, sha1, guess_tags=None):
    """Record a file waiting to be processed and describe it for the page."""
    tags = read_tags(path)
    if tags is None:
        path.unlink(missing_ok=True)
        return {"filename": filename, "error": "This isn't an audio file I can read."}
    title, artist = guess_title_artist(guess_tags or tags, filename)
    staged[sid] = {"path": path, "filename": filename, "sha1": sha1, "created": time.time()}
    existing = find_song_by_hash(sha1)
    return {"stage_id": sid, "filename": filename, "title": title, "artist": artist,
            "duplicate": {"song": rel(existing), "title": existing.name,
                          "artist": existing.parent.name} if existing else None}


def file_sha1(path):
    h = hashlib.sha1()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


@api.post("/api/stage")
async def stage(files: list[UploadFile] = File(...)):
    forget_old_uploads()
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for f in files:
        sid = uuid.uuid4().hex
        filename = f.filename or "song"
        path = STAGE_DIR / f"{sid}{Path(filename).suffix}"
        sha1 = hashlib.sha1()
        with path.open("wb") as out:
            while chunk := await f.read(1 << 20):
                sha1.update(chunk)
                out.write(chunk)
        results.append(register_staged(sid, path, filename, sha1.hexdigest()))
    return results


class LinkRequest(BaseModel):
    url: str


@api.post("/api/stage-link")
def stage_link(req: LinkRequest):
    """Download a song's audio from a link (YouTube, SoundCloud, Bandcamp...) with yt-dlp."""
    url = req.url.strip()
    if not re.match(r"https?://", url):
        raise HTTPException(400, "Paste a full link starting with http:// or https://.")
    forget_old_uploads()
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    sid = uuid.uuid4().hex
    opts = {"format": "bestaudio/best", "outtmpl": str(STAGE_DIR / f"{sid}.%(ext)s"),
            "noplaylist": True, "quiet": True, "no_warnings": True, "ffmpeg_location": FFMPEG}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info))
    except yt_dlp.utils.DownloadError as e:
        message = re.sub(r"\x1b\[[0-9;]*m|^ERROR:\s*", "", str(e)).strip()
        if "CERTIFICATE_VERIFY_FAILED" in message or "certificate verify failed" in message.lower():
            raise HTTPException(400, "The secure connection couldn't be verified (certificate "
                                     "error). This usually means a network that inspects traffic "
                                     "with a certificate this computer doesn't trust; try another "
                                     "network, or add the song as a file.")
        if "confirm you" in message and "bot" in message:
            raise HTTPException(400, "YouTube blocked the download with its “confirm you’re not a "
                                     "bot” check. This depends on your network, so it may work "
                                     "elsewhere; otherwise add the song as a file.")
        raise HTTPException(400, f"Couldn't download that link: {message[:300]}")
    if not path.exists():
        raise HTTPException(400, "The download finished but no audio file was saved.")
    # Prefer the music metadata sites provide; fall back to the video title.
    uploader = re.sub(r"\s*-\s*Topic$", "", info.get("uploader") or "")
    guess = {"title": info.get("track") or info.get("title") or "",
             "artist": info.get("artist") or info.get("creator") or ""}
    result = register_staged(sid, path, guess["title"] or path.name, file_sha1(path), guess)
    if result.get("artist") == UNKNOWN_ARTIST and uploader:
        result["artist"] = clean_name(uploader) or UNKNOWN_ARTIST
    result["filename"] = info.get("title") or url
    return result


class ProcessItem(BaseModel):
    stage_id: Optional[str] = None
    song: Optional[str] = None
    title: str = ""
    artist: str = ""


class ProcessRequest(BaseModel):
    items: list[ProcessItem]
    instruments: list[str]
    level: int = 0
    hq: bool = False
    fast: bool = False  # HQ guitar/piano: 1 s crossfade instead of 2x overlap (see run_roformer)


@api.post("/api/process")
def process(req: ProcessRequest):
    picked = set(req.instruments)
    # All guitar covers both parts, and both parts together are all guitar.
    if "guitar" in picked or GUITAR_PARTS <= picked:
        picked = (picked - GUITAR_PARTS) | {"guitar"}
    stems = [s for s in INSTRUMENTS if s in picked]
    if not stems:
        raise HTTPException(400, "Pick at least one instrument to remove.")
    if not req.items:
        raise HTTPException(400, "Add at least one song.")

    items, claimed = [], set()
    for it in req.items:
        if it.song:  # new version of a song already in the library
            song_dir = library_path(it.song, 2)
            if not original_file(song_dir):
                raise HTTPException(400, f"“{song_dir.name}” has no saved original. Upload it again "
                                         "once and it will be kept from then on.")
            items.append({"song_dir": song_dir, "title": song_dir.name})
            continue
        stage = staged.get(it.stage_id or "")
        if not stage:
            raise HTTPException(400, "That upload expired. Please add the file again.")
        existing = find_song_by_hash(stage["sha1"])
        if existing:  # same file as a song we already have: add a version to it
            song_dir = existing
        else:
            title = clean_name(it.title) or "Untitled"
            artist = clean_name(it.artist) or UNKNOWN_ARTIST
            song_dir = free_song_dir(artist, title)
            # Two new songs with the same name in one batch each get their own folder.
            while song_dir in claimed:
                song_dir = free_song_dir(artist, f"{title} ({len(claimed) + 1})")
            claimed.add(song_dir)
        items.append({"song_dir": song_dir, "title": song_dir.name, "stage": stage,
                      "stage_id": it.stage_id})

    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "queued", "total": len(items), "index": 0, "song": items[0]["title"],
                    "fraction": 0.0, "done": [], "failed": [], "finished": []}
    job_queue.put((job_id, {"items": items, "stems": stems, "hq": req.hq, "fast": req.fast,
                            "level": max(0, min(req.level, 100))}))
    return {"job": job_id}


@api.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Unknown job.")
    return jobs[job_id]


@api.post("/api/jobs/{job_id}/stop")
def stop_job(job_id: str):
    """Stop a job: the song being separated stops within a few seconds, the rest are skipped."""
    if job_id not in jobs:
        raise HTTPException(404, "Unknown job.")
    jobs[job_id]["stop"] = True
    return {"ok": True}


class SongEdit(BaseModel):
    song: str
    title: str
    artist: str


@api.post("/api/song/edit")
def edit_song(req: SongEdit):
    """Rename a song and/or move it to another artist."""
    song_dir = library_path(req.song, 2)
    if not song_dir.is_dir():
        raise HTTPException(404, "That song doesn't exist.")
    title = clean_name(req.title)
    if not title:
        raise HTTPException(400, "Give the song a name.")
    artist = clean_name(req.artist) or UNKNOWN_ARTIST
    dest = OUTPUT_DIR / artist / title
    if dest.exists() and not dest.samefile(song_dir):
        raise HTTPException(400, f"{artist} already has a song called “{title}”.")
    rename_song_files(song_dir, song_dir.name, title)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rename_dir(song_dir, dest)
    return {"song": rel(dest)}


class IdRequest(BaseModel):
    id: str


@api.post("/api/song/delete")
def delete_song(req: IdRequest):
    song_dir = library_path(req.id, 2)
    if song_dir.is_dir():
        send2trash(str(song_dir))
    return {"ok": True}


@api.post("/api/version/delete")
def delete_version(req: IdRequest):
    for f in version_files(rel(library_path(req.id, 3))):
        if f.exists():
            send2trash(str(f))
    return {"ok": True}


def read_import(song_dir):
    try:
        return json.loads((song_dir / IMPORT_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def resync_import(song_dir, imported):
    """An import made with older sync rules, synced again from its saved file (~1 s), when
    the automatic tab it lines up with is current (making that would take minutes)."""
    source = next((f for f in song_dir.glob("imported-tab*") if f.suffix.lower() in TAB_EXTENSIONS), None)
    auto = read_tab(song_dir)
    if not source or not auto or auto.get("notes_rules") != NOTES_RULES:
        return imported
    try:
        import_tab(song_dir, source, lambda fraction=None, step=None: None)
    except Exception as err:   # keep the old one rather than fail the page
        print(f"Couldn't sync {source.name} again: {err}")
        return imported
    fresh = read_import(song_dir)
    if not fresh:
        return imported
    # Same tab, synced again: it keeps its time, which is what tells the learner whether
    # the imported tabs changed (a new time made it re-learn everything at the next start).
    fresh["made"] = imported.get("made", fresh["made"])
    if imported.get("unify"):   # the user's "make repeats match" choices
        fresh["unify"] = imported["unify"]
    (song_dir / IMPORT_FILE).write_text(json.dumps(fresh), encoding="utf-8")
    return fresh


def import_opens(imported):
    """An imported tab's open-string MIDI notes (its own tuning), or None."""
    w = imported["tuning"]
    return imported.get("open") or (tab.TUNINGS[w][1] if w in tab.TUNINGS else None)


def _bar_map(notes, idx, opens):
    """Where a bar plays each pitch: {pitch: (string, fret)}, the usual place if it varies."""
    seen = {}
    for i in idx:
        n = notes[i]
        seen.setdefault(opens[n[2]] + n[3], []).append((n[2], n[3]))
    return {p: max(set(v), key=v.count) for p, v in seen.items()}


def _similar(a, b):
    """Share of two pitch sequences that lines up in order (longest common subsequence)."""
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for k, y in enumerate(b):
            cur.append(prev[k] + 1 if x == y else max(prev[k + 1], cur[k]))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def repeat_groups(imported):
    """Parts the imported tab repeats but plays in different places: bars (3+ played notes)
    whose pitches mostly match (>= 75% in order: a pickup note or a changed last note still
    counts; user: an A7 A7 A9 ending against a D2 D4 D4 one wasn't caught) and that put a
    pitch they share on different strings. {key: [ways]}, key = the first such bar's
    pitches; each way = {"map": {pitch: (string, fret)}, "bars": [(start, note indexes,
    pitches)]}, the most played way first; only groups with more than one way."""
    opens = import_opens(imported)
    starts = sorted({b for line in imported["lines"] for b in line[2]})
    if not opens or not starts:
        return {}
    notes, by_bar = imported["notes"], {}
    for i, n in enumerate(notes):
        if not n[4]:
            by_bar.setdefault(max(0, bisect.bisect_right(starts, n[0] + 0.02) - 1), []).append(i)
    clusters = []   # [first bar's pitches, [bars]]
    for b in sorted(by_bar):
        idx = by_bar[b]
        if len(idx) < 3:
            continue
        pitches = [opens[notes[i][2]] + notes[i][3] for i in idx]
        bar = (starts[b], idx, pitches)
        home = next((c for c in clusters if _similar(c[0], pitches) >= 0.75), None)
        if home:
            home[1].append(bar)
        else:
            clusters.append([pitches, [bar]])
    groups = {}
    for first, bars in clusters:
        ways = []   # a bar joins the first way that plays every pitch they share in the same place
        for bar in bars:
            fmap = _bar_map(notes, bar[1], opens)
            way = next((w for w in ways if all(w["map"][p] == f for p, f in fmap.items() if p in w["map"])), None)
            if way:
                way["map"].update({p: f for p, f in fmap.items() if p not in way["map"]})
                way["bars"].append(bar)
            else:
                ways.append({"map": dict(fmap), "bars": [bar]})
        if len(ways) > 1:
            ways.sort(key=lambda w: (-len(w["bars"]), w["bars"][0][0]))
            groups[",".join(map(str, first))] = ways
    return groups


def _rule_map(key, rule):
    """A saved "make repeats match" choice as {pitch: (string, fret)}. Older ones were a
    fingering per note of the key's pitches."""
    if isinstance(rule, dict):
        return {int(p): tuple(f) for p, f in rule.items()}
    return {int(p): tuple(f) for p, f in zip(key.split(","), rule)}


def fit_costs(imported, ways, notes=None):
    """How hard each way is to play in each of the group's bars, with the bar before and
    the bar after as the tab writes them (user: "the easier way to play based off the
    previous bar and following bar"): tab._difficulty (hand shifts, quick jumps, string
    skips) over those three bars, the neighbours as `notes` has them (the user's other
    choices applied) or as written. {bar start: [cost per way]}."""
    notes, opens = notes or imported["notes"], import_opens(imported)
    starts = sorted({b for line in imported["lines"] for b in line[2]})
    played = [i for i, n in enumerate(notes) if not n[4]]
    times = [notes[i][0] for i in played]
    out = {}
    for way in ways:
        for start, idx, pitches in way["bars"]:
            k = bisect.bisect_left(starts, start)
            lo = starts[k - 1] if k > 0 else start
            hi = starts[k + 2] if k + 2 < len(starts) else float("inf")
            window = played[bisect.bisect_left(times, lo - 0.02):bisect.bisect_left(times, hi - 0.02)]
            mine = set(idx)
            costs = []
            for other in ways:
                pos = [tuple(other["map"].get(opens[notes[i][2]] + notes[i][3], (notes[i][2], notes[i][3])))
                       if i in mine else (notes[i][2], notes[i][3]) for i in window]
                costs.append(tab._difficulty([notes[i][0] for i in window], pos) * len(window))
            out[start] = costs
    return out


def best_here(costs, own):
    """The way that fits a bar best; on a tie the one the tab already plays there."""
    return own if costs[own] <= min(costs) + 1e-9 else costs.index(min(costs))


def unify_rules(imported, groups):
    """The import's saved choices as {group key: {pitch: (string, fret)}}. Choices saved
    before near repeats were grouped are keyed by one exact pitch sequence: each goes to the
    group with a bar playing exactly that (later ones win where they overlap)."""
    out = {}
    for key, rule in (imported.get("unify") or {}).items():
        if rule == "easiest":   # each bar the way that fits it best
            if key in groups:
                out[key] = rule
            continue
        if isinstance(rule, dict):
            out.setdefault(key, {}).update(_rule_map(key, rule))
            continue
        pitches = [int(p) for p in key.split(",")]
        home = next((k for k, ways in groups.items() if any(b[2] == pitches for w in ways for b in w["bars"])), None)
        if home:
            out.setdefault(home, {}).update(_rule_map(key, rule))
    return out


def unified_notes(imported, skip=None):
    """The imported tab's notes with the user's "make repeats match" choices applied (all
    but `skip`): in every bar of a group with a chosen way, each pitch that way plays goes
    where it plays it; then the groups set to "easiest", each bar the way that fits it best
    between its neighbours as they now are."""
    notes = [list(n) for n in imported["notes"]]
    if not imported.get("unify") or not import_opens(imported):
        return notes
    groups = repeat_groups(imported)
    rules = {k: r for k, r in unify_rules(imported, groups).items() if k != skip}
    for key in sorted(rules, key=lambda k: rules[k] == "easiest"):
        costs = fit_costs(imported, groups[key], notes) if rules[key] == "easiest" else None
        for own, way in enumerate(groups[key]):
            for start, idx, pitches in way["bars"]:
                fmap = groups[key][best_here(costs[start], own)]["map"] if costs else rules[key]
                for i, p in zip(idx, pitches):
                    if p in fmap:
                        notes[i][2], notes[i][3] = fmap[p]
    return notes


def easiest_tunings(song_dir, imported, opens):
    """For an imported tab: per tuning, how many notes it can't reach and how hard the
    fingering is (tab._difficulty: hand shifts, quick jumps, string skips), with the
    easiest one; saved in the import, recomputed when the learned fingering changes."""
    learned, style_id = learned_style()
    saved = imported.get("tuning_report")
    if saved and saved.get("style") == style_id and saved.get("rules") == 2:
        return saved
    played = [n for n in imported["notes"] if not n[4]]
    notes = [(n[0], n[1], opens[n[2]] + n[3], 1.0) for n in played]
    report = {"style": style_id, "rules": 2, "tunings": {}}
    for key in tab.TUNINGS:
        fingered = tab.fingering(notes, key, learned)
        report["tunings"][key] = {"dropped": len(notes) - len(fingered),
                                  "difficulty": round(tab._difficulty([notes[i][0] for i, _, _ in fingered],
                                                                       [(s, f) for _, s, f in fingered]), 3)}
    # Easiest: every note reachable first, then a 4-string tuning, then the easiest
    # fingering; retuning is a bother, so the tab's own tuning wins unless another is at
    # least 25% easier (Cry Baby Cry, Electra: drop D ~1/3 easier; YOLO's half step
    # down only ~1/5, so it stays standard).
    written = imported["tuning"]
    # A 5-string only when the tab is written for one or nothing else reaches every note.
    def cost(key):
        r = report["tunings"][key]
        return (r["dropped"], key == "five-string" and written != "five-string", r["difficulty"] * (0.75 if key == written else 1.0))
    report["easiest"] = min(report["tunings"], key=cost)
    imported["tuning_report"] = report
    (song_dir / IMPORT_FILE).write_text(json.dumps(imported), encoding="utf-8")
    return report


@api.get("/api/tab")
def get_tab(song: str, tuning: Optional[str] = None, split: Optional[bool] = None, source: str = "auto"):
    """A song's bass tab: the automatic one in a tuning (default: the suggested one), with
    repeated notes split on the beat or as heard; or the imported one (source=imported);
    or the one recorded from the user's playing (source=mine); or, without a tab, what's
    possible. split: None = as learned from imported tabs. A source the song doesn't
    have falls back to the imported tab, then the recorded one, then the automatic one."""
    song_dir = library_path(song, 2)
    imported, mine = read_import(song_dir), read_mine(song_dir)
    if imported and imported.get("rules") != IMPORT_RULES:
        imported = resync_import(song_dir, imported)
    if mine and mine.get("rules") != MINE_RULES:   # made before takes became notes better: redo (a few s)
        mine = redo_mine(song_dir, mine)
    has_mine = bool(mine and mine["notes"])
    about = {"has_import": imported is not None, "import_name": imported and imported["name"], "has_mine": has_mine,
             "can_make": bool(original_file(song_dir) or bass_track(song_dir))}
    if source == "imported" and not imported and has_mine:
        source = "mine"
    if source == "mine" and not has_mine:
        source = "imported"
    # "open": each string's MIDI note, so the page can place what the user plays live.
    tunings = [{"key": k, "label": label, "strings": names, "open": opens} for k, (label, opens, names) in tab.TUNINGS.items()]
    auto = read_tab(song_dir)
    if source == "mine" and auto:   # lines (bars): the draft's own (from an imported tab) or the automatic tab's
        tuning = tuning if tuning in tab.TUNINGS else (mine.get("tuning") or auto["suggested"])
        notes = [(s, e, p, 1.0) for s, e, p in mine["notes"]]
        fingered = tab.fingering(notes, tuning, learned_style()[0])
        return {**about, "exists": True, "source": "mine", "tuning": tuning, "suggested": auto["suggested"],
                "bpm": auto.get("bpm"), "split": False, "maybe_half_down": auto.get("maybe_half_down", False),
                "lines": mine.get("lines") or auto["lines"], "dropped": len(notes) - len(fingered), "takes": len(mine["takes"]),
                "draft_from": mine.get("from"), "draft_from_name": mine.get("from_name"),
                "tunings": tunings, "notes": [[notes[i][0], notes[i][1], s, f] for i, s, f in fingered]}
    if source == "imported" and imported:
        written = imported["tuning"]
        if written == "custom":
            tunings.append({"key": "custom", "label": imported["label"], "strings": imported["strings"], "open": imported.get("open")})
        opens = imported.get("open") or (tab.TUNINGS[written][1] if written in tab.TUNINGS else None)
        report = easiest_tunings(song_dir, imported, opens) if opens else None
        base = {**about, "exists": True, "source": "imported", "suggested": written, "written": written,
                "easiest": report and report["easiest"], "tuning_report": report and report["tunings"],
                "bpm": imported.get("bpm"), "split": False, "maybe_half_down": False, "lines": imported["lines"],
                "quality": imported["quality"], "track": imported.get("track"), "tunings": tunings}
        # Dead (muted) notes, written "x", are left out: the user asked for no x's. A 5th
        # value says the note is slid or hammered into (the game doesn't call it messy).
        played = [n for n in unified_notes(imported) if not n[4]]
        legato = lambda n: bool(n[5]) if len(n) > 5 else False
        if tuning in (None, written) or tuning not in tab.TUNINGS or not opens:
            return {**base, "tuning": written, "dropped": 0, "notes": [[n[0], n[1], n[2], n[3], legato(n)] for n in played]}
        # In another tuning: the tab's pitches fingered again (learned fingering), the
        # way the automatic and recorded tabs are.
        notes = [(n[0], n[1], opens[n[2]] + n[3], 1.0) for n in played]
        fingered = tab.fingering(notes, tuning, learned_style()[0])
        return {**base, "tuning": tuning, "dropped": len(notes) - len(fingered),
                "notes": [[notes[i][0], notes[i][1], s, f, legato(played[i])] for i, s, f in fingered]}
    data = read_tab(song_dir)
    if data is None or data.get("notes_rules") != NOTES_RULES:   # none yet, or made the old way
        return {**about, "exists": False, "outdated": data is not None}
    # Made before (or with other) note settings learned from imported tabs: still shown,
    # with a suggestion to redo it.
    relearned = data.get("note_settings", tab.NOTES) != note_settings()
    if split is None:
        split = data.get("split_default", False)
    notes = [tuple(n) for n in data["notes_split" if split else "notes"]]
    changed = False
    if data.get("tuning_rules") != TUNING_RULES:   # saved before the suggestion improved
        held = [tuple(n) for n in data["notes"]]
        data.update(suggested=tab.suggest_tuning(held), maybe_half_down=tab.maybe_half_down(held),
                    tuning_rules=TUNING_RULES)
        changed = True
    learned, style_id = learned_style()
    if data.get("fingering_style") != style_id:   # fingered before the learned costs changed
        data.update(fingering={}, fingering_style=style_id)
        changed = True
    tuning = tuning if tuning in tab.TUNINGS else data["suggested"]
    key = tuning + ("|split" if split else "")
    if key not in data["fingering"]:   # first time in this tuning/version: work it out and keep it
        data["fingering"][key] = tab.fingering(notes, tuning, learned)
        changed = True
    if changed:
        (song_dir / TAB_FILE).write_text(json.dumps(data), encoding="utf-8")
    saved = read_learned()
    style = saved and {**saved["fingering"], "used": bool(learned), "style": None}
    tuned = saved and (saved.get("note_models") if note_models() else saved.get("notes"))
    tuned = tuned and {"used": bool(tuned.get("models") or tuned.get("settings")), "tabs": tuned["tabs"],
                       "before": tuned["before"], "after": tuned["after"]}
    return {**about, "exists": True, "source": "auto", "style": style, "tuned": tuned,
            "relearned": relearned, "learning": learning["running"], "tuning": tuning, "suggested": data["suggested"], "bpm": data["bpm"],
            "split": split, "maybe_half_down": data.get("maybe_half_down", False),
            "lines": data["lines"], "dropped": len(notes) - len(data["fingering"][key]),
            "tunings": tunings,
            "notes": [[notes[i][0], notes[i][1], s, f] for i, s, f in data["fingering"][key]]}


@api.post("/api/tab/import")
async def start_import(song: str = Form(...), file: Optional[UploadFile] = File(None), text: str = Form("")):
    """Bring in a tab file or pasted text tab and queue syncing it to the recording."""
    song_dir = library_path(song, 2)
    if not song_dir.is_dir():
        raise HTTPException(404, "That song doesn't exist.")
    if not (original_file(song_dir) or bass_track(song_dir)):
        raise HTTPException(400, "This song has no saved original or bass track to line the tab up with.")
    if file is not None and file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in TAB_EXTENSIONS:
            raise HTTPException(400, "That isn't a tab file. Use Guitar Pro (.gp, .gp3-.gp5), MusicXML (.musicxml, .mxl), or paste a text tab.")
        content = await file.read()
        name = file.filename
    elif text.strip():
        ext, content, name = ".txt", text.encode("utf-8"), "imported-tab.txt"
    else:
        raise HTTPException(400, "Choose a tab file or paste a tab.")
    final = f"imported-tab{ext}" if ext == ".txt" else f"imported-tab - {clean_name(Path(name).stem) or 'tab'}{ext}"
    trial = STAGE_DIR / f"{uuid.uuid4().hex}-{final}"
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    trial.write_bytes(content)
    try:   # read it first: a file the app can't read is reported at once, the old import kept
        if not tabimport.read_file(trial):
            raise ValueError("No tab notes found in it.")
    except Exception as e:
        trial.unlink(missing_ok=True)
        raise HTTPException(400, f"Couldn't read that tab: {e}")
    for old in song_dir.glob("imported-tab*"):   # replacing an earlier import: to the Trash
        send2trash(str(old))
    path = song_dir / final
    shutil.move(str(trial), str(path))
    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "queued", "total": 1, "index": 0, "song": song_dir.name,
                    "fraction": 0.0, "done": [], "failed": [], "finished": []}
    job_queue.put((job_id, {"kind": "import", "items": [{"song_dir": song_dir, "title": song_dir.name, "path": path}]}))
    return {"job": job_id}


@api.get("/api/learning")
def learning_status():
    """What learning from imported tabs is doing now, and what it learned last time."""
    saved = read_learned() or {}
    summary = lambda part: part and {"used": bool(part.get("models") or part.get("settings") or part.get("style")),
                                     "before": part["before"], "after": part["after"], "tabs": part["tabs"]}
    return {**{k: v for k, v in learning.items()},
            "last": {"made": saved["made"], "tabs": len(saved.get("sources", {})),
                     "notes": summary(saved.get("note_models") if note_models() else saved.get("notes")),
                     "fingering": summary(saved.get("fingering"))} if saved else None}


@api.post("/api/tab/record")
async def start_record(song: str = Form(...), song_start: float = Form(...), shift: int = Form(0),
                       mode: str = Form("replace"), file: UploadFile = File(...)):
    """A take recorded along with the song (WAV): queue turning it into My tab."""
    song_dir = library_path(song, 2)
    if not song_dir.is_dir():
        raise HTTPException(404, "That song doesn't exist.")
    if not (original_file(song_dir) or bass_track(song_dir)):
        raise HTTPException(400, "This song has no saved original or bass track to line your playing up with.")
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = STAGE_DIR / f"{uuid.uuid4().hex}-take.wav"
    path.write_bytes(await file.read())
    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "queued", "total": 1, "index": 0, "song": song_dir.name,
                    "fraction": 0.0, "done": [], "failed": [], "finished": []}
    job_queue.put((job_id, {"kind": "record", "items": [{"song_dir": song_dir, "title": song_dir.name, "path": path,
                                                         "song_start": max(0.0, song_start), "shift": shift,
                                                         "mode": "add" if mode == "add" else "replace"}]}))
    return {"job": job_id}


class MineEdit(BaseModel):
    song: str
    start: float
    pitch: int
    end: Optional[float] = None   # given when putting a removed note back (undo)


@api.post("/api/tab/mine/note")
def edit_mine(req: MineEdit, restore: bool = False):
    """Remove a note from the user's recorded tab (a ghost note), or put it back."""
    song_dir = library_path(req.song, 2)
    mine = read_mine(song_dir)
    if not mine:
        raise HTTPException(404, "This song has no recorded tab.")
    removed = [r for r in mine.get("removed", []) if not (abs(r[0] - req.start) < 0.06 and r[1] == req.pitch)]
    if restore:
        mine["notes"] = sorted(mine["notes"] + [[req.start, req.end if req.end is not None else req.start + 0.2, req.pitch]])
    else:
        removed.append([req.start, req.pitch])
        mine["notes"] = without_removed(mine["notes"], [[req.start, req.pitch]])
    mine["removed"] = removed
    (song_dir / MINE_FILE).write_text(json.dumps(mine), encoding="utf-8")
    return {"ok": True}


class DraftRequest(BaseModel):
    song: str
    source: str            # "imported" or "auto": the tab to start from
    split: bool = False    # the automatic tab with repeated notes split


@api.post("/api/tab/draft")
def start_draft(req: DraftRequest):
    """Start the user's draft (My tab) from the imported or automatic tab: its notes,
    and the imported tab's own bars and tuning. A draft already there goes to the
    Trash with its takes."""
    song_dir = library_path(req.song, 2)
    draft = {"takes": [], "made": time.time(), "rules": MINE_RULES, "from": req.source}
    if req.source == "imported":
        imported = read_import(song_dir)
        if not imported:
            raise HTTPException(404, "This song has no imported tab.")
        opens = imported.get("open") or tab.TUNINGS.get(imported["tuning"], (None, None))[1]
        if not opens:
            raise HTTPException(400, "Couldn't read that tab's tuning.")
        draft.update(base=[[n[0], n[1], opens[n[2]] + n[3]] for n in unified_notes(imported) if not n[4]],
                     lines=imported["lines"], tuning=imported["tuning"] if imported["tuning"] in tab.TUNINGS else None,
                     from_name=imported["name"])
    else:
        auto = read_tab(song_dir)
        if not auto:
            raise HTTPException(404, "This song has no automatic tab yet.")
        draft.update(base=[[n[0], n[1], n[2]] for n in auto["notes_split" if req.split else "notes"]])
    draft["notes"] = sorted(draft["base"])
    for f in (song_dir / MINE_FILE, song_dir / MY_BASS_DIR):
        if f.exists():
            send2trash(str(f))
    (song_dir / MINE_FILE).write_text(json.dumps(draft), encoding="utf-8")
    return {"ok": True, "notes": len(draft["notes"])}


@api.post("/api/tab/mine/remove")
def remove_mine(req: IdRequest):
    """Move a song's recorded tab and its takes' audio to the Trash."""
    song_dir = library_path(req.id, 2)
    for f in (song_dir / MINE_FILE, song_dir / MY_BASS_DIR):
        if f.exists():
            send2trash(str(f))
    return {"ok": True}


@api.post("/api/tab/import/remove")
def remove_import(req: IdRequest):
    """Move a song's imported tab (file and synced copy) to the Trash."""
    song_dir = library_path(req.id, 2)
    for f in [song_dir / IMPORT_FILE, *song_dir.glob("imported-tab*")]:
        if f.exists():
            send2trash(str(f))
    queue_learning()
    return {"ok": True}


@api.get("/api/tab/repeats")
def tab_repeats(song: str):
    """Repeated parts of the imported tab played in different places: per group each way
    (where it plays each pitch, its bars, and its first bar's notes to draw) and the one
    chosen, if any."""
    imported = read_import(library_path(song, 2))
    if not imported:
        raise HTTPException(404, "This song has no imported tab.")
    groups = repeat_groups(imported)
    names, rules, out = imported.get("strings") or [], unify_rules(imported, groups), []
    starts = sorted({b for line in imported["lines"] for b in line[2]})
    notes = imported["notes"]
    as_json = lambda fmap: {str(p): list(fmap[p]) for p in sorted(fmap)}
    for key, ways in groups.items():
        costs, variants = fit_costs(imported, ways, unified_notes(imported, skip=key)), []
        # Used everywhere, how hard each way is over all the group's bars (with their
        # neighbours); and how many bars each way fits best (Easiest in each place).
        totals = [sum(costs[b[0]][k] for w in ways for b in w["bars"]) for k in range(len(ways))]
        fits = [0] * len(ways)
        for own, w in enumerate(ways):
            for b in w["bars"]:
                fits[best_here(costs[b[0]], own)] += 1
        for way in ways:
            start, idx, pitches = way["bars"][0]
            # Where each note sits in its bar (0-1), so the ways draw as tab lines in step.
            k = bisect.bisect_left(starts, start)
            length = (starts[k + 1] if k + 1 < len(starts) else notes[idx[-1]][1]) - start
            variants.append({"map": as_json(way["map"]), "bars": [b[0] for b in way["bars"]], "pitches": pitches,
                             "fingering": [[notes[i][2], notes[i][3]] for i in idx],
                             "times": [round(max(0.0, notes[i][0] - start) / max(length, 0.1), 4) for i in idx],
                             "easiest": totals[len(variants)] <= min(totals) + 1e-9 and len(set(round(t, 6) for t in totals)) > 1,
                             "fits": fits[len(variants)]})
        out.append({"key": key, "notes": [note_name(int(p)) for p in key.split(",")],
                    "bars": sum(len(w["bars"]) for w in ways),
                    "chosen": rules.get(key) if rules.get(key) == "easiest" else as_json(rules[key]) if key in rules else None,
                    "variants": variants})
    out.sort(key=lambda g: (-g["bars"], min(v["bars"][0] for v in g["variants"])))
    return {"groups": out, "strings": names}


class UnifyRequest(BaseModel):
    song: str
    key: Optional[str] = None     # a group's key; none (with no map) = put all back
    map: Optional[dict] = None    # {pitch: [string, fret]}: play these pitches there in all its bars; none = as written
    easiest: bool = False         # instead: each bar the way that fits it best between its neighbours


@api.post("/api/tab/repeats")
def set_repeats(req: UnifyRequest):
    """Make every bar of a repeated part play its pitches in one place (or put it back as
    written). Kept in the import as "unify", apart from its notes, so it survives re-syncs."""
    song_dir = library_path(req.song, 2)
    imported = read_import(song_dir)
    if not imported:
        raise HTTPException(404, "This song has no imported tab.")
    groups = repeat_groups(imported)
    rules = unify_rules(imported, groups)   # older saved choices move to the new groups here
    if req.key is None:
        rules = {}
    elif req.easiest:
        if req.key not in groups:
            raise HTTPException(400, "That part isn't in the tab.")
        rules[req.key] = "easiest"
    elif req.map is None:
        rules.pop(req.key, None)
    else:
        chosen = {int(p): tuple(f) for p, f in req.map.items()}
        if not any(w["map"] == chosen for w in groups.get(req.key) or []):
            raise HTTPException(400, "That way of playing it isn't in the tab.")
        rules[req.key] = chosen
    imported["unify"] = {k: m if m == "easiest" else {str(p): list(m[p]) for p in sorted(m)} for k, m in rules.items()}
    (song_dir / IMPORT_FILE).write_text(json.dumps(imported), encoding="utf-8")
    queue_learning()   # the choices are learned from too
    return {"ok": True, "chosen": len(rules)}


def queue_learning():
    """Refit what's learned from imported tabs when they changed, on the learner thread
    (not the job queue: with several tabs it takes a minute or more, and tabs and
    versions shouldn't wait for it)."""
    learn_wanted.set()


learn_wanted = threading.Event()
LEARN_DELAY = 20   # seconds after startup before learning may start
learning = {"running": False}   # what the learner is doing, for GET /api/learning


def _learn_in_child(out_dir, progress):
    """Learning, in its own process (started by `learner`): lowest CPU priority and few
    threads. In the server's own process it held Python's lock for minutes, so the page's
    requests (tabs, audio) waited behind it and the app crawled at startup (the user:
    "when the app loads and does the learning, it slows everything down a ton")."""
    global OUTPUT_DIR, LEARNED_FILE
    try:
        os.nice(15)
    except (AttributeError, OSError):   # Windows has no nice
        pass
    os.environ["TRACK_SPLITTER_THREADS"] = "2"
    torch.set_num_threads(2)
    OUTPUT_DIR = Path(out_dir)
    LEARNED_FILE = OUTPUT_DIR / ".learned-from-tabs.json"
    try:
        learn_from_imports(lambda fraction=None, step=None: progress.put(("progress", fraction, step)))
        progress.put(("done", None, None))
    except Exception as e:
        progress.put(("error", str(e), None))


def learner():
    time.sleep(LEARN_DELAY)   # let the app and the page start first
    while True:
        learn_wanted.wait()
        learn_wanted.clear()   # tabs imported while learning set it again: one more round
        saved = read_learned() or {}
        if saved.get("sources", {}) == learn_sources() and saved.get("rules") == LEARN_RULES:
            continue
        learning.clear()
        learning.update(running=True, fraction=0.0, step="Starting", started=time.time(), tabs=len(learn_sources()))
        try:
            ctx = multiprocessing.get_context("spawn")
            progress = ctx.Queue()
            child = ctx.Process(target=_learn_in_child, args=(str(OUTPUT_DIR), progress), daemon=True)
            child.start()
            while True:
                try:
                    kind, a, b = progress.get(timeout=1)
                except queue.Empty:
                    if not child.is_alive():
                        raise RuntimeError(f"the learning process stopped (exit code {child.exitcode})")
                    continue
                if kind == "progress":
                    if a is not None:
                        learning["fraction"] = a
                    if b:
                        learning["step"] = b
                elif kind == "error":
                    raise RuntimeError(a)
                else:
                    break
            child.join(timeout=10)
        except Exception as e:   # tried again at the next start or import
            print(f"Couldn't learn from the imported tabs: {e}")
            learning["error"] = str(e)
        finally:
            learning["running"] = False


@api.post("/api/tab")
def start_tab(req: IdRequest):
    """Queue making a song's bass tab (replacing an existing one)."""
    song_dir = library_path(req.id, 2)
    if not song_dir.is_dir():
        raise HTTPException(404, "That song doesn't exist.")
    if not (original_file(song_dir) or bass_track(song_dir)):
        raise HTTPException(400, "This song has no saved original or bass track to work from.")
    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "queued", "total": 1, "index": 0, "song": song_dir.name,
                    "fraction": 0.0, "done": [], "failed": [], "finished": []}
    job_queue.put((job_id, {"kind": "tab", "items": [{"song_dir": song_dir, "title": song_dir.name}]}))
    return {"job": job_id}


@api.post("/api/version/redo")
def redo_version(req: IdRequest):
    """Make a version again with its own settings, replacing its files.

    Instruments and HQ come from its name; level and fast from .song.json (versions
    made before those were recorded count as 0% and not fast, as nearly all were).
    """
    path = library_path(req.id, 3)
    song_dir, m = path.parent, NO_RE.search(path.name)
    if not m or not version_files(req.id)[0].exists():
        raise HTTPException(404, "That version doesn't exist.")
    made = read_meta(song_dir).get("versions", {}).get(version_key(song_dir, req.id), {})
    return process(ProcessRequest(items=[ProcessItem(song=rel(song_dir))],
                                  instruments=re.split(r", | & ", m.group(1)), level=made.get("level", 0),
                                  hq=path.name[:m.start()].endswith(HQ_TAG), fast=made.get("fast", False)))


class ArtistEdit(BaseModel):
    artist: str
    name: str


@api.post("/api/artist/create")
def create_artist(req: IdRequest):
    name = clean_name(req.id)
    if not name:
        raise HTTPException(400, "Give the artist a name.")
    (OUTPUT_DIR / name).mkdir(parents=True, exist_ok=True)
    return {"artist": name}


@api.post("/api/artist/rename")
def rename_artist(req: ArtistEdit):
    """Rename an artist; renaming onto an existing artist merges the two."""
    src = library_path(req.artist, 1)
    name = clean_name(req.name)
    if not name:
        raise HTTPException(400, "Give the artist a name.")
    dest = OUTPUT_DIR / name
    if not dest.exists() or dest.samefile(src):
        rename_dir(src, dest)
    else:
        for song_dir in visible_dirs(src):
            target = free_song_dir(name, song_dir.name)
            if target.name != song_dir.name:
                rename_song_files(song_dir, song_dir.name, target.name)
            song_dir.rename(target)
        shutil.rmtree(src, ignore_errors=True)  # only hidden leftovers like .DS_Store remain
    return {"artist": name}


@api.post("/api/artist/delete")
def delete_artist(req: IdRequest):
    artist_dir = library_path(req.id, 1)
    if artist_dir.is_dir():
        send2trash(str(artist_dir))
    return {"ok": True}


@api.post("/api/open-folder")
def open_folder():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(OUTPUT_DIR)
    else:
        subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", str(OUTPUT_DIR)])
    return {"ok": True}


_server = []  # the running uvicorn server, so the page's Quit button can stop it


@api.post("/api/quit")
def quit_app():
    # Reply first, then stop shortly after so the page gets the answer.
    threading.Timer(0.5, lambda: setattr(_server[0], "should_exit", True)).start()
    return {"ok": True}


# --- Launching without a terminal ----------------------------------------------

APP_ROOT = Path(__file__).resolve().parent


_desktop = []


def shortcut_path():
    """Where the app shortcut lives on this computer (None if not supported here)."""
    if sys.platform == "darwin":
        return Path.home() / "Applications" / "Track Splitter.app"
    if sys.platform == "win32":
        if not _desktop:  # the real Desktop may be redirected, e.g. into OneDrive
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "[Environment]::GetFolderPath('Desktop')"],
                                 capture_output=True, text=True).stdout.strip()
            _desktop.append(Path(out) if out else Path.home() / "Desktop")
        return _desktop[0] / "Track Splitter.lnk"
    return None


def install_shortcut():
    """Create an app shortcut (Mac: ~/Applications, Windows: Desktop) that starts
    Track Splitter without a terminal window. Only runs when the user asks for it."""
    path = shortcut_path()
    if sys.platform == "darwin":
        macos = path / "Contents" / "MacOS"
        macos.mkdir(parents=True, exist_ok=True)
        (path / "Contents" / "Info.plist").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
            '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
            '<plist version="1.0"><dict>\n'
            '  <key>CFBundleName</key><string>Track Splitter</string>\n'
            '  <key>CFBundleIdentifier</key><string>local.track-splitter</string>\n'
            '  <key>CFBundleExecutable</key><string>launch</string>\n'
            '  <key>CFBundlePackageType</key><string>APPL</string>\n'
            '</dict></plist>\n')
        # Runs start.sh in the background, so package updates still get installed;
        # if Track Splitter is already running it just opens the page again.
        launcher = macos / "launch"
        launcher.write_text(
            "#!/bin/sh\n"
            f"cd {shlex.quote(str(APP_ROOT))} || exit 1\n"
            f"if curl -s -o /dev/null --max-time 1 http://{HOST}:{PORT}/api/info; then\n"
            f"    open http://{HOST}:{PORT}\n"
            "else\n"
            '    nohup sh start.sh >> "$HOME/Library/Logs/Track Splitter.log" 2>&1 &\n'
            "fi\n")
        launcher.chmod(0o755)
    elif sys.platform == "win32":
        pythonw = Path(sys.executable).with_name("pythonw.exe")
        script = (f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{path}'); "
                  f"$s.TargetPath = '{pythonw}'; $s.Arguments = 'app.py'; "
                  f"$s.WorkingDirectory = '{APP_ROOT}'; $s.Save()")
        subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, check=True)
    else:
        raise HTTPException(400, "Shortcuts are only supported on Mac and Windows.")
    return path


@api.get("/api/shortcut")
def shortcut_status():
    path = shortcut_path()
    return {"supported": path is not None, "exists": bool(path and path.exists()),
            "where": "in your Applications folder" if sys.platform == "darwin" else "on your Desktop"}


@api.post("/api/shortcut")
def create_shortcut():
    try:
        path = install_shortcut()
    except (OSError, subprocess.CalledProcessError) as e:
        raise HTTPException(500, f"Couldn't create the shortcut: {e}")
    return {"path": str(path)}


class FreshStaticFiles(StaticFiles):
    """Makes the browser check for a newer page every time, so app updates show up
    without a hard reload (it still gets a quick "not modified" when nothing changed)."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


# Serves the page itself; mounted last because it answers every path not matched above.
api.mount("/", FreshStaticFiles(directory=STATIC_DIR, html=True), name="static")


def already_running(port):
    try:
        urllib.request.urlopen(f"http://{HOST}:{port}/api/info", timeout=1)
        return True
    except OSError:
        return False


def main(port=PORT, open_browser=True):
    url = f"http://{HOST}:{port}"
    if already_running(port):
        if open_browser:
            webbrowser.open(url)
        print(f"Track Splitter is already running at {url}")
        return
    # Started from a shortcut there's no console to print to; log to a file instead.
    if sys.stdout is None or sys.stderr is None:
        sys.stdout = sys.stderr = open(APP_ROOT / "track-splitter.log", "a", buffering=1)

    migrate_old_library()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(STAGE_DIR, ignore_errors=True)  # uploads from a previous run can't be used anymore
    shutil.rmtree(SHIFT_DIR, ignore_errors=True)
    if open_browser:
        threading.Timer(1.5, webbrowser.open, [url]).start()
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=learner, daemon=True).start()
    queue_learning()   # tabs imported before the app learned fingering, or removed by hand
    print(f"Track Splitter is running at {url} (use Quit in the app, or press Ctrl+C)")
    server = uvicorn.Server(uvicorn.Config(api, host=HOST, port=port, log_level="warning"))
    _server.append(server)
    server.run()


if __name__ == "__main__":
    main()
