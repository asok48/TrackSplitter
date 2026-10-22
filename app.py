"""Track Splitter: a small local app that takes instruments out of songs.

Upload songs, pick what to remove (bass, drums, vocals, guitar, piano), and
Demucs separates it out, so you get each song without it plus the removed part
on its own. The library keeps each song's original, so new versions can be
made later without uploading again.

This file is the engine plus a small local web server; the page itself is
static/index.html.
"""

import hashlib
import json
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

import imageio_ffmpeg
import numpy as np
import torch
import uvicorn
import yt_dlp
from demucs.apply import apply_model
from demucs.pretrained import get_model
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from send2trash import send2trash

HOST, PORT = "127.0.0.1", 7860
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
STATIC_DIR = Path(__file__).parent / "static"
# Kept in the user's Music folder so the library survives re-copying the app.
OUTPUT_DIR = Path.home() / "Music" / "Track Splitter"
# Where earlier versions kept the library, back when the app was called Bass Remover.
OLD_LIBRARY_DIRS = [Path.home() / "Music" / "Bass Remover", Path(__file__).parent / "output"]
STAGE_DIR = Path(tempfile.gettempdir()) / "track-splitter-uploads"
SAMPLERATE = 44100  # all Demucs v4 models use this
INSTRUMENTS = ["bass", "drums", "vocals", "guitar", "piano"]
SIX_STEM_ONLY = {"guitar", "piano"}  # only htdemucs_6s separates these
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


def decode(path, samplerate):
    """Decode an audio file to a float32 stereo array shaped (2, samples)."""
    cmd = [FFMPEG, "-v", "error", "-i", str(path),
           "-f", "f32le", "-ac", "2", "-ar", str(samplerate), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()


def encode_mp3(audio, samplerate, dest):
    cmd = [FFMPEG, "-v", "error", "-y", "-f", "f32le", "-ac", "2", "-ar", str(samplerate),
           "-i", "-", "-b:a", "192k", str(dest)]
    pcm = np.clip(audio, -1.0, 1.0).T.astype(np.float32).tobytes()
    subprocess.run(cmd, input=pcm, check=True)


def separate(mix, stems, high_quality, on_progress=lambda fraction: None):
    """Return the chosen instruments summed together, same shape as mix."""
    wav = torch.from_numpy(mix)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std()
    x = ((wav - mean) / std)[None]

    # Group instruments by model so each model runs once per song.
    runs = {}
    for stem in stems:
        model = stem_model(stem, high_quality)
        runs.setdefault(id(model), (model, []))[1].append(stem)

    removed = np.zeros_like(mix)
    length = mix.shape[1]
    for r, (model, wanted) in enumerate(runs.values()):
        def callback(d, r=r):
            if d.get("state") == "end":
                on_progress((r + min(d["segment_offset"] / length, 1.0)) / len(runs))
        with torch.no_grad():
            out = apply_model(model, x, device=DEVICE, split=True, overlap=0.25,
                              callback=callback)[0]
        for stem in wanted:
            removed += (out[model.sources.index(stem)] * std).numpy()
    return removed


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


def original_file(song_dir):
    return next(iter(song_dir.glob("original.*")), None)


def song_info(song_dir):
    versions = []
    for p in song_dir.glob("* (no *).mp3"):
        m = NO_RE.search(p.stem)
        if m:
            versions.append({"id": rel(p.with_suffix("")), "removed": m.group(1),
                             "hq": p.stem[: m.start()].endswith(HQ_TAG), "added": p.stat().st_mtime})
    versions.sort(key=lambda v: v["added"], reverse=True)
    meta = read_meta(song_dir)
    added = meta.get("added") or max((v["added"] for v in versions), default=song_dir.stat().st_mtime)
    return {"id": rel(song_dir), "title": song_dir.name, "artist": song_dir.parent.name,
            "added": added, "has_original": original_file(song_dir) is not None,
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
    for old in OLD_LIBRARY_DIRS:
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


def make_version(source, song_dir, stems, level, high_quality, on_progress):
    mix = decode(source, SAMPLERATE)
    removed = separate(mix, stems, high_quality, on_progress)
    # Subtracting the separated parts from the original keeps everything else untouched.
    without = mix - removed * (1.0 - level / 100.0)

    # The tag keeps standard and high-quality versions side by side. High quality
    # only changes bass, drums and vocals, so guitar/piano alone aren't tagged.
    hq = high_quality and any(s not in SIX_STEM_ONLY for s in stems)
    version = f"{rel(song_dir)}/{song_dir.name}{HQ_TAG if hq else ''} (no {stems_label(stems)})"
    song_dir.mkdir(parents=True, exist_ok=True)
    without_path, only_path = version_files(version)
    encode_mp3(without, SAMPLERATE, without_path)
    encode_mp3(removed, SAMPLERATE, only_path)
    return version


def worker():
    while True:
        job_id, settings = job_queue.get()
        job = jobs[job_id]
        job["status"] = "running"
        for i, item in enumerate(settings["items"]):
            job.update(index=i, song=item["title"], fraction=0.0)
            try:
                song_dir = item["song_dir"]
                if item.get("stage"):
                    # New song: separate first, so a failed file leaves no empty song behind.
                    stage = item["stage"]
                    version = make_version(stage["path"], song_dir, settings["stems"],
                                           settings["level"], settings["hq"],
                                           on_progress=lambda f: job.update(fraction=f))
                    if not original_file(song_dir):
                        shutil.copy2(stage["path"], song_dir / f"original{stage['path'].suffix.lower()}")
                        write_meta(song_dir, {"sha1": stage["sha1"], "source": stage["filename"],
                                              "added": time.time()})
                else:
                    version = make_version(original_file(song_dir), song_dir, settings["stems"],
                                           settings["level"], settings["hq"],
                                           on_progress=lambda f: job.update(fraction=f))
                job["done"].append(version)
            except subprocess.CalledProcessError:
                job["failed"].append(f"{item['title']} (couldn't read the file)")
            except Exception as e:  # keep going with the rest of the batch
                job["failed"].append(f"{item['title']} ({e})")
            finally:
                if item.get("stage"):
                    item["stage"]["path"].unlink(missing_ok=True)
                    staged.pop(item["stage_id"], None)
        job.update(status="done", fraction=1.0)


# --- Web API -------------------------------------------------------------------

api = FastAPI(title="Track Splitter")


@api.get("/api/info")
def info():
    return {"device": DEVICE_NAME, "library": str(OUTPUT_DIR), "instruments": INSTRUMENTS,
            "unknown_artist": UNKNOWN_ARTIST}


@api.get("/api/library")
def get_library():
    return library()


@api.get("/api/audio")
def audio(version: str, part: str):
    without, only = version_files(rel(library_path(version, 3)))
    path = without if part == "without" else only
    if not path.exists():
        raise HTTPException(404, "That version doesn't exist.")
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


@api.post("/api/process")
def process(req: ProcessRequest):
    stems = [s for s in INSTRUMENTS if s in req.instruments]
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
                    "fraction": 0.0, "done": [], "failed": []}
    job_queue.put((job_id, {"items": items, "stems": stems, "hq": req.hq,
                            "level": max(0, min(req.level, 100))}))
    return {"job": job_id}


@api.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Unknown job.")
    return jobs[job_id]


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


# Serves the page itself; mounted last because it answers every path not matched above.
api.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


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
    if open_browser:
        threading.Timer(1.5, webbrowser.open, [url]).start()
    threading.Thread(target=worker, daemon=True).start()
    print(f"Track Splitter is running at {url} (use Quit in the app, or press Ctrl+C)")
    server = uvicorn.Server(uvicorn.Config(api, host=HOST, port=port, log_level="warning"))
    _server.append(server)
    server.run()


if __name__ == "__main__":
    main()
