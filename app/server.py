#!/usr/bin/env python3
"""LIMINAL ENGINE — local web front end for latent_dream.py (LATENT DREAM), app/bank.py (CUT BANK)
and app/granular.py (GRANULAR, 11-23).
The old Stable Diffusion mode was retired 10-18 (its code is in _cleanup/retired-2026-10-18/).
Standard library only. One job at a time (the GPU is the bottleneck).
Run via "LIMINAL ENGINE.command" or: python3 app/server.py
"""
import ast, json, math, os, re, shutil, signal, subprocess, sys, threading, time, urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import models_lib  # noqa: E402  (model shelf: catalog, picks, T7 downloads)
import updater     # noqa: E402  (shared copies only: checks GitHub for a newer version)

# A GUI-launched .app inherits a minimal PATH with no homebrew; repair it before
# anything shells out to ffmpeg/ffprobe.
for _p in ("/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin"):
    if os.path.isdir(_p) and _p not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")
FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

ROOT = Path(__file__).resolve().parent.parent          # ~/Movies/python
APP = ROOT / "app"
INBOX = APP / "inbox"
OUT = APP / "out"
DELIV = Path(os.environ["LE_RENDERS"]).expanduser() if os.environ.get("LE_RENDERS") else Path.home() / "Movies" / "Liminal Engine"   # finished renders (LE_RENDERS: a test folder)
# 11-01: that folder is now a shortcut to T7 (/Volumes/T7/_Video Projects/_Liminal Engine) so Premiere links keep working.
# If T7 isn't plugged in when the engine starts, renders go to a local offline folder instead (move them to T7 later).
if not DELIV.exists():
    DELIV = Path.home() / "Movies" / "Liminal Engine (offline)"
    DELIV.mkdir(parents=True, exist_ok=True)
ARM_DIR = {"latent": "dream", "dream": "dream", "bank": "cut bank", "crt": "crt", "granular": "granular"}
PY = ROOT / ".venv" / "bin" / "python"
LATENT = ROOT / "latent_dream.py"          # LATENT DREAM: StyleGAN walk (Lucid Sonic Dreams method) + bend/decay
MODELS = ROOT / "models" / "stylegan"       # drop StyleGAN2 .pkl files here
PORT = int(os.environ.get("PORT", "8391"))
INBOX.mkdir(parents=True, exist_ok=True)   # a fresh copy ships without these;
OUT.mkdir(parents=True, exist_ok=True)     # list_media/list_outputs iterdir them on the first request
for _a in set(ARM_DIR.values()): (DELIV / _a).mkdir(parents=True, exist_ok=True)
OWNER = (ROOT / ".git").exists()   # 11-25: Eric's own working copy (a git checkout); friends' downloads aren't
CRT_PREFIX = "amfm-crt-" if OWNER else "crt-"   # AM.FM is Eric's station: friends' exports are plain crt-...
def ai_ready():
    """The AI tier is optional: the CUT BANK only needs ffmpeg + librosa."""
    try: return any((PY.parent.parent / "lib").glob("python*/site-packages/torch"))
    except Exception: return False
def granular_ready():
    """The Granular arm needs OpenCV (opencv-python-headless, base tier since 11-23)."""
    try: return any((PY.parent.parent / "lib").glob("python*/site-packages/cv2"))
    except Exception: return False
MAX_UPLOAD = 6 * 1024 * 1024 * 1024        # uploads are read into memory; refuse absurd ones
EXECUTABLE_EXT = {".app", ".command", ".sh", ".terminal", ".webloc", ".dmg", ".pkg", ".scpt", ".workflow", ".shortcut"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}
AUD_EXT = {".m4a", ".mp3", ".wav", ".aif", ".aiff", ".flac", ".ogg", ".mp4", ".mov"}
VID_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".mts"}
BANK = APP / "bank.py"
GRANULAR = APP / "granular.py"   # GRANULAR (11-23): one clip through the Takagi looks (granular / paint / outline)
DROPPED = INBOX / "dropped"   # clips dragged onto the cut bank
# footage folders the cut bank / collage can point at (only listed if present). Kept in app/lanes.json, which is
# local to this Mac and never shipped: a JSON list of folder paths ("~" allowed).
# Each entry is a path, or {"path": ..., "name": ...} to show a nicer name.
LANES, LANE_NAMES = [], {}
try:
    for _l in json.loads((APP / "lanes.json").read_text()):
        _p = os.path.expanduser(_l["path"] if isinstance(_l, dict) else _l)
        LANES.append(_p)
        if isinstance(_l, dict) and _l.get("name"): LANE_NAMES[_p] = _l["name"]
except (OSError, ValueError, KeyError, TypeError):
    pass

job = None          # current/last job dict
lock = threading.Lock()


_dur_cache = {}


def has_sound(path):
    """True when the file has at least one audio stream (a silent .mp4 can be offered as a song)."""
    if not FFPROBE: return True
    try:
        r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, timeout=20)
        return bool(r.stdout.strip())
    except Exception: return True


def ffprobe_duration(path):
    """Duration in seconds, cached on (path, mtime, size) — /api/state asks constantly."""
    if not FFPROBE:
        return 0.0
    try:
        st = os.stat(path)
        key = (str(path), st.st_mtime, st.st_size)
    except OSError:
        return 0.0
    if key in _dur_cache:
        return _dur_cache[key]
    try:
        r = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, timeout=20)
        val = float(r.stdout.strip() or 0)
    except Exception:
        val = 0.0
    if len(_dur_cache) > 2000:
        _dur_cache.clear()
    _dur_cache[key] = val
    return val


_vid_cache = {}


def ffprobe_video(path):
    """{"width", "height", "fps"} of a clip's picture as displayed (rotation and non-square pixels applied, the way
    granular.py's probe() sees it), cached like ffprobe_duration. The Granular tab's time estimate works out the
    output size from it. {} when it can't be read."""
    if not FFPROBE: return {}
    try:
        st = os.stat(path); key = (str(path), st.st_mtime, st.st_size)
    except OSError: return {}
    if key in _vid_cache: return _vid_cache[key]
    val = {}
    try:
        r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height,avg_frame_rate,r_frame_rate,sample_aspect_ratio:stream_tags=rotate:stream_side_data=rotation",
                            "-of", "json", str(path)], capture_output=True, text=True, timeout=20)
        s = (json.loads(r.stdout or "{}").get("streams") or [{}])[0]
        def rate(x):
            a, _, b = str(x or "").partition("/")
            try: v = float(a) / float(b or 1)
            except (ValueError, ZeroDivisionError): return 0.0
            return v if v > 0 and math.isfinite(v) else 0.0
        if s.get("width") and s.get("height"):
            rot = 0
            try: rot = int(float((s.get("tags") or {}).get("rotate", 0)))
            except ValueError: pass
            for sd in s.get("side_data_list") or []:
                if "rotation" in sd:
                    try: rot = int(float(sd["rotation"]))
                    except ValueError: pass
            dw, dh = int(s["width"]) * (rate((s.get("sample_aspect_ratio") or "1:1").replace(":", "/")) or 1.0), int(s["height"])
            if abs(rot) % 180 == 90: dw, dh = dh, dw
            val = {"width": round(dw, 3), "height": dh, "fps": round(rate(s.get("avg_frame_rate")) or rate(s.get("r_frame_rate")), 4)}
    except Exception:
        val = {}
    if len(_vid_cache) > 2000: _vid_cache.clear()
    _vid_cache[key] = val
    return val


def list_media():
    """Songs for the Recents lists: the inbox first, then loose audio in the project folder."""
    seen, auds = set(), []
    for d in (INBOX, ROOT):
        for p in sorted(d.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not p.is_file() or p.name.startswith(".") or p.name in seen: continue
            seen.add(p.name)
            e = p.suffix.lower()
            if e in AUD_EXT and not p.name.startswith("_seg") and not (e in (".mp4", ".mov") and d is ROOT):
                auds.append({"name": p.name, "path": str(p), "duration": ffprobe_duration(p), "inbox": d is INBOX})
    return auds[:40]


def list_inbox():
    """Everything sitting in the inbox (uploads land here), biggest first, so it can be cleared out."""
    out = []
    for p in INBOX.iterdir():
        if not p.is_file() or p.name.startswith("."): continue
        e = p.suffix.lower()
        kind = "audio" if e in AUD_EXT - VID_EXT else "video" if e in VID_EXT else "image" if e in IMG_EXT else "file"
        try: out.append({"name": p.name, "path": str(p), "size": p.stat().st_size, "kind": kind, **({"duration": ffprobe_duration(p), **ffprobe_video(p)} if kind == "video" else {})})
        except OSError: pass
    out.sort(key=lambda f: f["size"], reverse=True)
    return out


# ---- removing things: always to the macOS Trash (11-18), never deleted, never into the project folder ----
def trash(p):
    """Move p to the Trash. Uses Finder's own call so "Put Back" works; on another volume it lands in that volume's
    .Trashes. Falls back to a plain move into ~/.Trash (no Put Back) if pyobjc isn't there. Returns where it went."""
    p = Path(p)
    try:
        from Foundation import NSFileManager, NSURL
        ok, res, err = NSFileManager.defaultManager().trashItemAtURL_resultingItemURL_error_(NSURL.fileURLWithPath_(str(p)), None, None)
        if ok and res is not None: return Path(str(res.path()))
    except Exception:
        pass
    t = Path.home() / ".Trash"; t.mkdir(exist_ok=True)
    dst = t / p.name; n = 2
    while dst.exists() or dst.is_symlink(): dst = t / f"{p.stem} {n}{p.suffix}"; n += 1
    shutil.move(str(p), str(dst))
    return dst


models_lib.TRASH = trash


def trash_item(p):
    """trash() + the record /api/restore needs to put it back."""
    dst = trash(p); lib_dirty()
    return {"orig": str(p), "trashed": str(dst)}


def in_trash(p):
    """Only things inside a Trash can be restored from."""
    try: r = Path(p).resolve()
    except Exception: return False
    homes = [(Path.home() / ".Trash").resolve()]
    homes += [v / ".Trashes" for v in Path("/Volumes").glob("*")]
    return any(h == r.parent or h in r.parents for h in homes)


def deliver(path, kind):
    """A finished render → ~/Movies/Liminal Engine/<arm>/. Hard link (same volume, zero bytes) else copy. Returns the delivered path."""
    src = Path(path)
    if not src.is_file(): return None
    d = DELIV / ARM_DIR.get(kind, "dream")
    try: d.mkdir(parents=True, exist_ok=True)
    except OSError:   # the T7 went away mid-session (the shortcut dangles): keep the render reachable locally
        d = Path.home() / "Movies" / "Liminal Engine (offline)" / ARM_DIR.get(kind, "dream"); d.mkdir(parents=True, exist_ok=True)
    dst = d / src.name; i = 2
    while dst.exists() and not (dst.stat().st_ino == src.stat().st_ino): dst = d / f"{src.stem}-{i}{src.suffix}"; i += 1
    if not dst.exists():
        try: os.link(src, dst)
        except OSError: shutil.copy2(src, dst)
    return dst

# ---- 11-25 render files (Eric picked B + C): projects and per-clip stacks, one copy, settings kept for Remake.
# New renders land in <renders>/<project>/<clip>/<clip>_<look>_<date-time>.mp4 (moved, not copied, from app/out when
# they finish). Older renders stay where they are (Premiere links to them) and show as "Earlier renders". Stars and
# the page settings each render was made with live in <renders>/.liminal-library.json.
ARM_FOLDERS = {"dream", "cut bank", "granular", "crt", "collage"}   # the old per-tab folders: never a project's name
PROJECT_DEFAULT = "Tests"
LIB_META = DELIV / ".liminal-library.json"
_meta_lock = threading.Lock()


def fold_name(x, n=48):
    """a folder name from anything the user typed or a clip's name: no slashes / colons / control chars, no leading dot."""
    x = re.sub(r"[\\/:\x00-\x1f]+", " ", str(x or "")); x = re.sub(r"\s+", " ", x).strip(" .")[:n].strip(" .")
    return x or "untitled"


def project_name(x):
    x = fold_name(x or PROJECT_DEFAULT, 40)
    return PROJECT_DEFAULT if x.lower() in ARM_FOLDERS or x.startswith("_") else x


def list_projects():
    try: ps = [d.name for d in DELIV.iterdir() if d.is_dir() and d.name not in ARM_FOLDERS and not d.name.startswith((".", "_"))]
    except OSError: ps = []
    return sorted(set(ps) | {PROJECT_DEFAULT}, key=str.lower)


def meta_load():
    try: m = json.loads(LIB_META.read_text())
    except (OSError, ValueError): m = {}
    return m if isinstance(m, dict) else {}


# 11-25: your own Granular presets, per look, kept beside the renders (survives app updates). Removed ones are kept
# under "_removed" (nothing is thrown away).
PRESETS = DELIV / ".liminal-presets.json"
_pre_lock = threading.Lock()
PRE_LOOKS = ("granular", "paint", "outline", "light")
def presets_load():
    try: m = json.loads(PRESETS.read_text())
    except (OSError, ValueError): m = {}
    return m if isinstance(m, dict) else {}
def presets_edit(fn):
    with _pre_lock:
        m = presets_load(); fn(m)
        DELIV.mkdir(parents=True, exist_ok=True)
        tmp = PRESETS.with_name(PRESETS.name + ".tmp"); tmp.write_text(json.dumps(m, indent=1)); os.replace(tmp, PRESETS)
        return {k: m.get(k, []) for k in PRE_LOOKS}

def meta_update(rel, **kw):
    with _meta_lock:
        m = meta_load(); e = dict(m.get(rel) or {}); e.update({k: v for k, v in kw.items() if v is not None})
        for k in [k for k, v in kw.items() if v is None]: e.pop(k, None)
        m[rel] = e
        try:
            tmp = LIB_META.with_name(LIB_META.name + ".tmp"); tmp.write_text(json.dumps(m, separators=(",", ":"))); os.replace(tmp, LIB_META)
        except OSError: pass


def place(j, req, source, tag):
    """where a render will land and what it'll be called: project / clip stack / clip_tag_date-time."""
    j["project"] = project_name(req.get("project"))
    j["stack"] = fold_name(Path(source).stem if source else tag)
    if not (req.get("name") or "").strip():
        slug = re.sub(r"[^a-z0-9]+", "-", Path(source).stem.lower()).strip("-")[:32] if source else ""
        j["name"] = unique_out(f"{slug + '_' if slug else ''}{tag}_" + time.strftime("%m%d-%H%M")); j["out"] = str(OUT / f"{j['name']}.mp4")
    j["work"] = j["out"]
    ui = req.get("ui")
    if isinstance(ui, dict) and len(json.dumps(ui)) < 60000: j["ui"] = ui   # the page's own settings, for Remake


def deliver_job(j):
    """a finished render -> its project / clip folder (moved: one copy). Old-style jobs fall back to deliver()."""
    src = Path(j["out"])
    if not j.get("project") or not src.is_file(): return deliver(j["out"], j.get("kind"))
    d = DELIV / j["project"] / j["stack"]
    try: d.mkdir(parents=True, exist_ok=True)
    except OSError:   # the T7 went away mid-session: keep it reachable locally
        d = Path.home() / "Movies" / "Liminal Engine (offline)" / j["project"] / j["stack"]; d.mkdir(parents=True, exist_ok=True)
    dst = d / src.name; i = 2
    while dst.exists(): dst = d / f"{src.stem}-{i}{src.suffix}"; i += 1
    shutil.move(str(src), str(dst))
    j["out"] = str(dst)
    try: meta_update(str(dst.relative_to(DELIV)), kind=j.get("kind"), look=j.get("look"), ui=j.get("ui"), made=time.time())
    except ValueError: pass
    return dst


CRT_OUT = OUT / "_work" / "crt"      # browser recordings, frame jobs, raw stills: working files, not deliverables
CRT_DELIV = DELIV / "crt"
_SAFE_NAME = re.compile(r"[^\w .()\-]+")
CRT_BUSY = [0]   # CRT saves / renders in flight: an update waits for them
def crt_deliv():
    """Liminal Engine/crt, or the local offline folder when the T7 went away mid-session (as deliver() does)."""
    try: CRT_DELIV.mkdir(parents=True, exist_ok=True); return CRT_DELIV
    except OSError:
        d = Path.home() / "Movies" / "Liminal Engine (offline)" / "crt"; d.mkdir(parents=True, exist_ok=True); return d
def crt_stem():   # second-stamped name + a short random tail: two windows saving in the same second never collide
    return CRT_PREFIX + time.strftime("%Y%m%d-%H%M%S") + "-" + os.urandom(2).hex()
def need_ffmpeg():
    if not FFMPEG: raise RuntimeError("ffmpeg isn't installed (brew install ffmpeg); only WebM recordings, PNG stills and ASCII text can be saved without it")
def ff_to(cmd, dst, timeout=1800):
    """Run ffmpeg into a hidden .part file next to dst and rename it on success: a failed or timed-out encode never
    leaves a half-written video in the deliveries folder. cmd ends where the output path goes."""
    need_ffmpeg(); tmp = dst.with_name("." + dst.stem + ".part" + dst.suffix)
    try:
        r = subprocess.run([*cmd, str(tmp)], capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0: raise RuntimeError((r.stderr.strip().splitlines() or ["ffmpeg failed"])[-1])
        os.replace(tmp, dst); return dst
    except subprocess.TimeoutExpired: raise RuntimeError("ffmpeg took longer than %d min" % (timeout // 60))
    finally: Path(tmp).unlink(missing_ok=True)
def crt_export(body, kind, fmt, name=None, folder=None):
    """CRT arm: the page records/grabs in the browser (webm / png); turn it into the chosen file type.
    Stills may carry a name and a folder (batch stills, 11-01) so a fast run never overwrites itself."""
    CRT_OUT.mkdir(parents=True, exist_ok=True)
    stem = CRT_PREFIX + time.strftime("%Y%m%d-%H%M%S")
    raw = CRT_OUT / "_raw"; raw.mkdir(exist_ok=True)
    if kind == "video":   # the take goes to disk first: whatever fails after this, the recording survives
        head = body.open("rb").read(8) if isinstance(body, Path) else body[:8]
        is_mp4 = head[4:8] == b"ftyp"; stem = crt_stem()
        src = raw / (stem + (".mp4" if is_mp4 else ".webm"))
        if isinstance(body, Path): os.replace(body, src)
        else: src.write_bytes(body)
        try:
            dd = crt_deliv()
            if fmt == "webm" and not is_mp4:
                dst = dd / (stem + ".webm"); shutil.copyfile(src, dst); src.unlink(missing_ok=True); return dst
            dst = crt_convert(src, fmt, stem, dd)
        except ValueError: raise
        except Exception as e:
            raise RuntimeError(str(e)[-300:] + " · the recording is kept at " + str(src))
        src.unlink(missing_ok=True)   # converted: the raw recording was only a working file
        return dst
    if isinstance(body, Path): b = body.read_bytes(); body.unlink(missing_ok=True); body = b
    CRT_DELIV_ = crt_deliv()
    if kind == "text":   # ASCII saved as characters (11-16): the page builds the file; only these three types
        ext = {"txt": ".txt", "html": ".html", "svg": ".svg"}.get(fmt)
        if not ext: raise ValueError("unknown text format")
        base = stem + "-%03d-ascii" % (int(time.time() * 1000) % 1000); dst = CRT_DELIV_ / (base + ext); n = 2
        while dst.exists(): dst = CRT_DELIV_ / (base + "-" + str(n) + ext); n += 1   # two saves in one moment never overwrite
        with open(dst, "xb") as f: f.write(body)
        return dst
    if kind == "still":
        out_dir = CRT_DELIV_
        if folder:
            out_dir = CRT_DELIV_ / (_SAFE_NAME.sub("_", folder).strip(" .")[:80] or "batch"); out_dir.mkdir(exist_ok=True)
        if name: stem = _SAFE_NAME.sub("_", name).strip(" .")[:120] or stem
        ext = {"png": ".png", "jpg": ".jpg", "tif": ".tif", "tiff": ".tif"}.get(fmt)
        if not ext: raise ValueError("unknown still format")
        if not name: stem += "-%03d" % (int(time.time() * 1000) % 1000)
        dst = out_dir / (stem + ext); n = 2
        while dst.exists(): dst = out_dir / (stem + "-" + str(n) + ext); n += 1   # same second, or IMG_1.jpg + IMG_1.png in a batch: never overwrite
        if fmt == "png":
            with open(dst, "xb") as f: f.write(body)
            return dst
        need_ffmpeg(); src = raw / (stem + "-" + os.urandom(3).hex() + ".png"); src.write_bytes(body)
        args = ["-q:v", "2"] if fmt == "jpg" else ["-compression_algo", "lzw"]
        try: return ff_to([FFMPEG, "-v", "error", "-y", "-i", str(src), *args], dst, 120)
        finally: src.unlink(missing_ok=True)   # the PNG was only a working copy for ffmpeg
    raise ValueError("unknown kind")


def crt_convert(src, fmt, stem, dd):
    """A finished CRT video (browser recording, or frames already stitched) → the chosen file type."""
    cfr = ["-r", "30000/1001"]     # browser recordings have uneven timing; editors want constant frame rate
    if fmt == "webm":
        dst = dd / (stem + ".webm")
        return ff_to([FFMPEG, "-v", "error", "-y", "-i", str(src), "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "18", "-row-mt", "1",
               "-c:a", "libopus", "-b:a", "192k"], dst)
    if fmt == "mp4":
        dst = dd / (stem + ".mp4")
        cmd = [FFMPEG, "-v", "error", "-y", "-i", str(src), *cfr, "-c:v", "libx264", "-crf", "14", "-preset", "slow", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart"]
    elif fmt == "prores":
        dst = dd / (stem + ".mov")
        cmd = [FFMPEG, "-v", "error", "-y", "-i", str(src), *cfr, "-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le",
               "-c:a", "pcm_s16le"]
    elif fmt == "gif":
        dst = dd / (stem + ".gif")
        cmd = [FFMPEG, "-v", "error", "-y", "-i", str(src), "-vf",
               "fps=15,scale=960:-2:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"]
    elif fmt == "png":
        need_ffmpeg(); dst = dd / (stem + "_png"); dst.mkdir(exist_ok=False)
        r = subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(src), *cfr, str(dst / "frame_%04d.png")], capture_output=True, text=True, timeout=1800)
        if r.returncode != 0: raise RuntimeError((r.stderr.strip().splitlines() or ["ffmpeg failed"])[-1])
        return dst
    else:
        raise ValueError("unknown format " + fmt)
    return ff_to(cmd, dst)


# EXPORT WHOLE VIDEO without playing it through (10-22, Eric: "just have it download instead of having to play
# through"): the page steps the video frame by frame and draws each through the CRT. Where the browser has a hardware
# H.264 encoder it sends one encoded stream (part "stream"); otherwise every frame as a JPEG. It also sends the
# original video (part "source") for its sound. Here they are stitched, then converted like a recording.
_JOB_OK = re.compile(r"^[a-z0-9]{4,24}$")
def crt_job_dir(job):
    if not _JOB_OK.fullmatch(job or ""): raise ValueError("bad job")
    d = CRT_OUT / "_raw" / ("job-" + job); d.mkdir(parents=True, exist_ok=True); return d

def crt_frame(job, i, body):
    if not body or body[:2] != b"\xff\xd8": raise ValueError("not a JPEG frame")
    if str(i) == "0": need_ffmpeg()   # say so at the first frame, not after stepping the whole video
    (crt_job_dir(job) / ("%06d.jpg" % int(i))).write_bytes(body)

def crt_part(job, name, body):
    if name not in ("stream", "source", "sound"): raise ValueError("bad part")
    if name == "stream": need_ffmpeg()
    dst = crt_job_dir(job) / {"stream": "stream.h264", "source": "source.bin", "sound": "sound.wav"}[name]
    if isinstance(body, Path):
        if body.stat().st_size == 0: body.unlink(missing_ok=True); raise ValueError("empty part")
        os.replace(body, dst); return
    if not body: raise ValueError("empty part")
    dst.write_bytes(body)

def crt_audio(job):
    """The uploaded video's sound as 48 kHz stereo WAV for the page's sound chain, or None when it has none."""
    d = crt_job_dir(job); src = d / "source.bin"; dst = d / "source.wav"
    if not src.exists(): return None
    need_ffmpeg()
    r = subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(src), "-map", "0:a:0?", "-vn", "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", "-f", "wav", str(dst)],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not dst.exists() or dst.stat().st_size <= 44: return None
    return dst

def crt_render(job, fmt, fps):
    d = crt_job_dir(job); need_ffmpeg()
    stream = d / "stream.h264"
    if stream.exists():
        # a raw H.264 stream carries no timestamps: wrap it first (no re-encode), or the sound will not line up with it
        wrapped = d / "video.mp4"
        r = subprocess.run([FFMPEG, "-v", "error", "-y", "-r", fps, "-f", "h264", "-i", str(stream), "-c:v", "copy", str(wrapped)],
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0: raise RuntimeError(r.stderr.strip() or "ffmpeg failed")
        frames_in = ["-i", str(wrapped)]
    elif any(d.glob("*.jpg")): frames_in = ["-framerate", fps, "-i", str(d / "%06d.jpg")]
    else: raise ValueError("no frames arrived")
    stem = crt_stem(); dd = crt_deliv()
    src = d / "source.bin"
    snd = d / "sound.wav"     # the page's CRT sound render (11-05) wins over the video's own sound
    if snd.exists(): src = snd
    audio_in, amap = (["-i", str(src)], ["-map", "1:a:0?"]) if src.exists() else ([], [])
    try:
        if fmt == "png":   # frames straight out, no sound
            dst = dd / (stem + "_png"); dst.mkdir(exist_ok=False)
            cmd = [FFMPEG, "-v", "error", "-y", *frames_in, str(dst / "frame_%04d.png")]
        elif fmt == "gif":
            dst = dd / (stem + ".gif")
            cmd = [FFMPEG, "-v", "error", "-y", *frames_in, "-vf",
                   "fps=15,scale=960:-2:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"]
        else:
            # one pass: the frames + the video's own sound (padded with silence so the picture is never cut short).
            # MP4 from the Mac's encoder is already H.264 — it is only wrapped, not encoded again.
            pad = ["-af", "apad", "-shortest"] if amap else []
            if fmt == "mp4":
                dst = dd / (stem + ".mp4")
                v = ["-c:v", "copy"] if stream.exists() else ["-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p"]
                tail = [*v, "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart"]
            elif fmt == "prores":
                dst = dd / (stem + ".mov")
                tail = ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-c:a", "pcm_s16le"]
            elif fmt == "webm":
                dst = dd / (stem + ".webm")
                tail = ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "18", "-row-mt", "1", "-deadline", "good", "-cpu-used", "4", "-c:a", "libopus", "-b:a", "192k"]
            else:
                raise ValueError("unknown format " + fmt)
            cmd = [FFMPEG, "-v", "error", "-y", *frames_in, *audio_in, "-map", "0:v", *amap, *tail, *pad]
        if fmt == "png":
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if r.returncode != 0: raise RuntimeError((r.stderr.strip().splitlines() or ["ffmpeg failed"])[-1])
        else: ff_to(cmd, dst)
        shutil.rmtree(d, ignore_errors=True)   # done: the job folder was only a staging area for this one file
        return dst
    except ValueError: raise
    except Exception as e:
        raise RuntimeError(str(e)[-300:] + f" · the frames are kept in {d}")


def list_models():
    MODELS.mkdir(parents=True, exist_ok=True)
    try:
        return models_lib.describe(models_lib.model_paths())
    except Exception:
        return [{"name": p.stem, "path": str(p), "size": p.stat().st_size, "group": "other"} for p in sorted(models_lib.model_paths())]


_LIB = {"key": None, "t": 0.0, "v": None}


def lib_dirty():
    _LIB["v"] = None


def _lib_key():
    ks = []
    for d in (OUT, ROOT, DELIV, *[DELIV / a for a in sorted(set(ARM_DIR.values()))]):
        try: ks.append(d.stat().st_mtime_ns)
        except OSError: ks.append(0)
    return tuple(ks)


def list_outputs():
    """Every render, newest first. Cached: rescanning the T7 renders folder on every 2 s poll was slow. The cache
    is dropped when a top folder changes (mtime), when a render finishes or something is removed, and after 30 s."""
    key = _lib_key()
    if _LIB["v"] is not None and _LIB["key"] == key and time.time() - _LIB["t"] < 30:
        return _LIB["v"]
    outs, seen, work = [], set(), {}
    meta = meta_load(); KIND_ARM = {"latent": "dream", "bank": "cut bank", "granular": "granular"}
    for d in (OUT, ROOT):
        for p in d.glob("*.mp4"):
            try:
                fr = p.parent / (p.stem + "_frames"); st = p.stat(); seen.add(st.st_ino); work[p.name] = p
                live = job and (job.get("work") or job.get("out")) == str(p)   # rendering now: it'll move to its project when done
                arm = next((a for a in ("dream", "cut bank", "granular") if (DELIV / a / p.name).exists() and same_render(DELIV / a / p.name, p)), None) \
                      or (KIND_ARM.get(job.get("kind")) if live else None) \
                      or ("cut bank" if p.name.startswith("bank_") else "granular" if p.name.startswith("granular_") else "dream")
                dl = DELIV / arm / p.name
                outs.append({"name": p.name, "path": str(p), "deliv": str(dl) if dl.exists() else "", "arm": arm, "size": st.st_size, "mtime": st.st_mtime,
                             "frames": len(preview_files(fr)), "project": job.get("project") if live and job.get("project") else "Earlier renders",
                             "stack": job.get("stack") if live and job.get("stack") else arm})
            except OSError: continue
    if DELIV.exists():
        for p in DELIV.rglob("*"):
            try:
                if not p.is_file() or p.name.startswith(".") or p.suffix.lower() not in (".mp4", ".mov", ".webm", ".gif"): continue
                st = p.stat()
                if st.st_ino in seen or (p.name in work and same_render(p, work[p.name])): continue   # the delivered copy of a listed render
                parts = p.relative_to(DELIV).parts; rel = str(p.relative_to(DELIV)); m = meta.get(rel) or {}
                if parts[0] in ARM_FOLDERS or len(parts) < 2:   # the old per-tab folders
                    arm = parts[0] if len(parts) > 1 else "other"
                    proj, stack = ("CRT", "crt") if arm == "crt" else ("Earlier renders", arm)   # CRT still saves there
                else:
                    proj, stack = parts[0], parts[1] if len(parts) > 2 else ""
                    arm = KIND_ARM.get(m.get("kind")) or ("crt" if p.name.startswith(("crt-", "amfm-crt")) else "")
                outs.append({"name": p.name, "path": str(p), "deliv": str(p), "arm": arm, "size": st.st_size, "mtime": st.st_mtime, "frames": 0,
                             "project": proj, "stack": stack, "star": bool(m.get("star")), "remake": "ui" in m, "look": m.get("look") or ""})
            except (OSError, ValueError): continue
    outs.sort(key=lambda o: o["mtime"], reverse=True)
    _LIB.update(key=key, t=time.time(), v=outs)
    return outs


def frames_dir_for(out_path):
    p = Path(out_path)
    return p.parent / (p.stem + "_frames")


_pv = {}


def preview_files(fd):
    """the preview pictures in a <name>_frames folder, in order: f_000123.png (latent, collage) or .jpg (granular, 11-24:
    a 320 px PNG per picture was ~170 KB). Re-listed only when the folder changes, and at least once a second (an exFAT
    folder's time is coarse): a long render's folder holds thousands, and the page asks for one several times a second."""
    try: m = fd.stat().st_mtime_ns
    except OSError: return []
    c = _pv.get(str(fd))
    if c and c[0] == m and time.time() - c[1] < 1.0: return c[2]
    fs = sorted([*fd.glob("f_*.png"), *fd.glob("f_*.jpg")], key=lambda x: x.name)
    if len(_pv) > 64: _pv.clear()
    _pv[str(fd)] = (m, time.time(), fs)
    return fs


def count_frames(out_path):
    return len(preview_files(frames_dir_for(out_path)))


def latest_frame(out_path):
    fs = preview_files(frames_dir_for(out_path))
    return fs[-1] if fs else None


def run_job(j):
    """Spawn the engine (or the bank), follow its stdout, keep j updated."""
    if j.get("kind") == "latent":
        cmd = [str(PY), "-u", str(LATENT), "--pkl", j["model"], "--audio", j["audio"], "--out", j["out"],
               "--start", str(j["start"]), "--fps", str(j["fps"]), "--size", str(j["size"]), "--mode", j["engine_mode"],
               "--bend", str(j["bend"]), "--spin", str(j.get("spin", 0)), "--decay", str(j["decay"]), "--decay_shape", str(j["decay_shape"]),
               "--speed_fpm", str(j["speed_fpm"]), "--pulse_react", str(j["pulse_react"]),
               "--motion_react", str(j["motion_react"]), "--psi", str(j["psi"]), "--seed", str(j["seed"])]
        if j["duration"] > 0: cmd += ["--duration", str(j["duration"])]
        if j.get("draft"): cmd.append("--draft")
    elif j.get("kind") == "bank":
        cmd = [str(PY), "-u", str(BANK), "--audio", j["audio"], "--out", j["out"], "--start", str(j["start"]),
               "--duration", str(j["duration"]), "--width", str(j["width"]), "--height", str(j["height"]), "--fps", str(j["fps"]),
               "--min_shot", str(j["min_shot"]), "--max_shot", str(j["max_shot"]), "--moves", j["moves"],
               "--move_chance", str(j["move_chance"]), "--tape", j["tape"], "--gen", str(j["gen"]),
               "--damage", str(j["damage"]), "--faults", j["faults"], "--react", "1" if j["react"] else "0",
               "--max_clips", str(j["max_clips"]), "--seed", str(j["seed"])]
        for c in j["clips"]: cmd += ["--clips", c]
    elif j.get("kind") == "granular":
        cmd = [str(PY), "-u", str(GRANULAR), "--source", j["source"], "--out", j["out"], "--look", j["look"],
               "--start", str(j["start"]), "--duration", str(j["duration"]), "--width", str(j["width"]), "--seed", str(j["seed"])]
        if j.get("audio"): cmd += ["--audio", j["audio"], "--song_start", str(j["song_start"])]
        if j.get("settings"): cmd += ["--set", *[f"{k}={v}" for k, v in j["settings"].items()]]
    else:
        j.update(stage="error", error="unknown kind of render", running=False, ended=time.time()); return
    j["cmd"] = " ".join(cmd)
    j["stage"] = "starting"
    j["log"] = []
    try:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1, start_new_session=True)
    except Exception as e:
        j.update(stage="error", error=str(e), running=False, ended=time.time()); return
    j["pid"] = proc.pid
    try:
        _follow(j, proc)
    except Exception as e:
        j["error"] = f"lost track of the render: {e}"
        try: proc.wait(timeout=5)
        except Exception: pass
    finally:
        if j.get("running"):
            j["running"] = False; j["ended"] = time.time()
            if j.get("stage") not in ("done", "error", "stopped"): j["stage"] = "error"
        if j.get("stage") == "done":
            try: j["deliv"] = str(deliver_job(j) or "")
            except Exception as e: j["log"].append(f"deliver: {e}")
            tidy_after(j)
        lib_dirty()


def tidy_after(j):
    """A render finished cleanly: its working leftovers go to the Trash (recoverable), not left to pile up.
    Cut bank: the <name>_shots folder. Latent dream: <name>_frames-old-* folders it parked on a re-run.
    Granular: the <name>_frames preview folder."""
    out = Path(j.get("work") or j["out"])   # side folders sit next to the working copy in app/out
    side = []
    if j.get("kind") == "bank": side.append(out.parent / (out.stem + "_shots"))
    if j.get("kind") == "latent": side += sorted(out.parent.glob(out.stem + "_frames-old-*"))
    if j.get("kind") == "granular": side.append(out.parent / (out.stem + "_frames"))   # its live-preview frames
    for d in side:
        if d.is_dir():
            try: j.setdefault("trashed", []).append(trash_item(d)); j["log"].append(f"[tidy] {d.name} → Trash")
            except Exception as e: j["log"].append(f"[tidy] could not move {d.name} to the Trash: {e}")


def _follow(j, proc):
    for line in proc.stdout:
        line = line.rstrip()
        if not line: continue
        j["log"].append(line); j["log"] = j["log"][-80:]
        m = re.search(r"\[audio\] (\d+) frames", line) or re.search(r"\[cuts\] (\d+) shots", line) or re.search(r"frames=(\d+)", line)
        if m and not (j.get("kind") == "collage" and "[audio]" in line): j["frames_total"] = int(m.group(1))   # collage counts pictures, not frames
        m = re.search(r"^\s*shot (\d+)/(\d+)", line)
        if m: j["shots_done"] = int(m.group(1)); j["frames_total"] = int(m.group(2))
        if j.get("kind") == "granular":   # granular's previews aren't one per frame, so progress comes from its lines
            m = re.search(r"\[preview\] ([\d.]+) per s", line)   # one preview per new picture: they play back at this rate
            if m and float(m.group(1)) > 0: j["pfps"] = float(m.group(1))
            m = re.search(r"\[clip\] (\d+) frames \([\d.]+ s @ ([\d.]+) fps", line)
            if m: j["frames_total"], j["fps"] = int(m.group(1)), float(m.group(2))
            m = re.search(r"^\s*frame (\d+)/(\d+)", line)
            if m: j["gdone"], j["frames_total"] = int(m.group(1)), int(m.group(2))
        m = re.search(r"\[tape\] (\d+)/(\d+) frames", line)
        if m: j["tape_done"], j["tape_total"] = int(m.group(1)), int(m.group(2))
        if "[model]" in line: j["stage"] = "loading model"
        elif "[audio]" in line: j["stage"] = "reading the song"
        elif "[cuts]" in line or "[clips]" in line: j["stage"] = "cutting"
        elif "[clip]" in line: j["stage"] = "reading the clip"
        elif "[sources]" in line: j["stage"] = "gathering pictures"
        elif "[sort]" in line: j["stage"] = "sorting by colour"
        elif "[render]" in line or line.strip().startswith("frame ") or line.strip().startswith("shot "): j["stage"] = "rendering"
        elif "[encode]" in line: j["stage"] = "encoding"
        elif "[tape]" in line: j["stage"] = "recording to tape"
        elif "[done]" in line: j["stage"] = "done"
        elif re.search(r"error|traceback|\[warn\]", line, re.I):
            if re.search(r"error|traceback", line, re.I): j["stage"] = "error"
    rc = proc.wait()
    j["rc"] = rc; j["running"] = False; j["ended"] = time.time()
    if rc != 0:
        j["stage"] = "stopped" if j.get("stopping") else "error"
        if not j.get("error"):
            tail = [l for l in j.get("log", []) if l.strip()][-6:]
            j["error"] = "\n".join(tail) or f"the render exited with code {rc}"
    elif j["stage"] != "error":
        j["stage"] = "done"
    else:
        # exited 0 but printed an error: keep the error, say so plainly
        if not j.get("error"):
            j["error"] = "\n".join([l for l in j.get("log", []) if l.strip()][-6:])


def start_bank(req):
    global job
    with lock:
        if job and job.get("running"): return {"error": "a render is already running"}
        clips = [c for c in (req.get("clips") or []) if c and Path(c).exists()]   # folders, or single dropped clips
        if not clips: return {"error": "pick at least one footage folder"}
        audio = req.get("audio")
        if not audio or not Path(audio).is_file(): return {"error": "pick a song"}
        name = unique_out(re.sub(r"[^a-z0-9]+", "-", (req.get("name") or "").lower()).strip("-") or time.strftime("bank_%m%d_%H%M%S"))
        def f(k, d):
            try: return float(req[k]) if req.get(k) not in (None, "") else d
            except Exception: return d
        j = {"name": name, "kind": "bank", "clips": clips, "audio": audio, "out": str(OUT / f"{name}.mp4"),
             "start": f("start", 0), "duration": f("duration", 20), "song_duration": ffprobe_duration(audio), "length": "bank",
             "width": int(f("width", 1280)), "height": int(f("height", 720)), "fps": int(f("fps", 24)),
             "min_shot": f("min_shot", 0.125), "max_shot": f("max_shot", 0.8),
             "moves": ",".join(m for m in str(req.get("moves", "fast,slow,reverse")).split(",") if m in ("fast", "slow", "reverse")),
             "move_chance": max(0.0, min(1.0, f("move_chance", 0.3))),
             "tape": req.get("tape") if req.get("tape") in ("off", "vhs-sp", "vhs-ep", "umatic", "betacam-sp") else "off",
             "gen": int(max(1, min(5, f("gen", 1)))), "damage": max(0.0, min(1.0, f("damage", 0.3))),
             "faults": ",".join(x for x in str(req.get("faults", "dropout,tracking,clog,skew,wobble,ckill")).split(",")
                                if x in ("dropout", "tracking", "clog", "skew", "wobble", "ckill")),
             "react": bool(req.get("react", True)),
             "max_clips": int(f("max_clips", 600)), "seed": int(f("seed", 0)),
             "running": True, "started": time.time(), "frames_total": 0, "stage": "queued", "log": []}
        place(j, req, audio, "cuts")
        job = j
        threading.Thread(target=run_job, args=(j,), daemon=True).start()
        return {"ok": True, "job": public(j)}


def same_render(a, b):
    """Two files are one render (hard link or the delivered copy) when they share an inode, or match in size and time."""
    try:
        sa, sb = a.stat(), b.stat()
        return sa.st_ino == sb.st_ino or (sa.st_size == sb.st_size and abs(sa.st_mtime - sb.st_mtime) < 3)
    except OSError: return False

def unique_out(name):
    """name -> a name whose .mp4 and side folders don't exist yet (adds -2, -3 …)."""
    base, i = name, 2
    while any((OUT / f"{name}{suf}").exists() for suf in (".mp4", "_frames", "_shots", "_src")) or any((DELIV / a / f"{name}.mp4").exists() for a in set(ARM_DIR.values())):
        name = f"{base}-{i}"; i += 1
    return name


def start_latent(req):
    global job
    with lock:
        if job and job.get("running"): return {"error": "a render is already running"}
        model, audio = req.get("model"), req.get("audio")
        if not model or not Path(model).is_file() or Path(model).suffix.lower() != ".pkl": return {"error": "pick a model"}
        if not audio or not Path(audio).is_file(): return {"error": "pick a song"}
        def f(k, d):
            try: return float(req[k]) if req.get(k) not in (None, "") else d
            except Exception: return d
        song = ffprobe_duration(audio)
        full = req.get("length") == "full"
        start, duration = (0.0, 0.0) if full else (max(0.0, f("start", 0)), f("duration", 15))
        if not full and song and start >= song - 1:
            return {"error": f"the song is only {int(song // 60)}:{int(song % 60):02d} long — start earlier than that"}
        name = unique_out(re.sub(r"[^a-z0-9]+", "-", (req.get("name") or "").lower()).strip("-") or time.strftime("latent_%m%d_%H%M%S"))
        j = {"name": name, "kind": "latent", "model": model, "audio": audio, "out": str(OUT / f"{name}.mp4"),
             "start": start, "duration": duration, "song_duration": song, "length": "full" if full else "test",
             "fps": max(6, min(60, int(f("fps", 24)))), "size": max(64, min(2048, int(f("size", 512)))), "engine_mode": "lsd" if req.get("classic") else "plus",
             "bend": max(0.0, min(1.0, f("bend", 0))), "spin": max(0.0, min(1.0, f("spin", 0))), "decay": max(0.0, min(1.0, f("decay", 0))), "decay_shape": f("decay_shape", 1.6),
             "speed_fpm": f("speed_fpm", 12), "pulse_react": f("pulse_react", 0.5),
             "motion_react": f("motion_react", 0.5), "psi": f("psi", 0.8), "seed": int(f("seed", 1)), "draft": bool(req.get("draft")),
             "running": True, "started": time.time(), "frames_total": int((duration or song) * int(f("fps", 24)) / (2 if req.get("draft") else 1)),
             "stage": "queued", "log": []}
        place(j, req, audio, "dream")
        job = j
        threading.Thread(target=run_job, args=(j,), daemon=True).start()
        return {"ok": True, "job": public(j)}


# GRANULAR (11-23): which look settings the page may send, and their ranges (anything else is ignored; out of range is
# clamped). Integers are whole-number settings in the engine. freeze_sec is the engine's shortcut for freeze_at.
GRAN_SET = {
    "granular": {"l_bands": (1, 12), "s_bands": (1, 10), "rebuild": (0, 1), "paper": (0.05, 1), "density": (0, 20000), "size_dark": (0.5, 16),
                 "size_light": (0.3, 10), "envelope": (0, 1), "ink": (0, 1), "sat": (0, 4), "spray": (0, 20), "time_spray": (0, 23),
                 "decay": (1, 30), "follow": (0, 2), "freeze_sec": (-1, 36000), "freeze_len": (1, 24), "dry": (0, 1), "step": (1, 6), "strict": (0, 1),
                 "exposure": (0, 0.9), "mod_density": (0, 1), "mod_size": (0, 2), "mod_spray": (0, 2), "echo": (0, 8), "echo_decay": (0, 1), "echo_op": (0, 4), "ps_filter": (0, 4), "ps_p1": (0, 20), "ps_p2": (0, 20), "ps_p3": (0, 20)},
    "paint": {"exposure": (0, 0.9), "ink_amount": (0, 2), "ink_start": (0, 0.9), "brush": (0, 12), "strict": (0, 1), "step": (1, 6), "echo": (0, 8), "echo_decay": (0, 1), "echo_op": (0, 4), "ps_filter": (0, 4), "ps_p1": (0, 20), "ps_p2": (0, 20), "ps_p3": (0, 20)},
    "outline": {"edge_gain": (0, 10), "wheel": (0, 1), "palette": (-1, 11), "strict": (0, 1),   # palette 9-11: strict's measured maps only (the engine clamps)
                "colorama": (0, 32), "ae_lines": (0, 2), "step": (1, 6), "echo": (0, 8), "echo_decay": (0, 1), "echo_op": (0, 4), "ps_filter": (0, 4), "ps_p1": (0, 20), "ps_p2": (0, 20), "ps_p3": (0, 20)},   # 11-24: AE Colorama preset (0 off) and AE-exact Find Edges
    "light": {"colorama": (0, 33), "body": (0.1, 1.5), "shade": (0, 1), "ground": (0.01, 0.4), "contrast": (0.5, 3), "streaks": (0, 1),
              "smear_len": (0, 200), "glow": (0, 0.9), "reach": (0, 0.8), "sparks": (0, 2), "cutout": (0, 2), "steady": (0, 0.9), "step": (1, 3), "echo": (0, 8), "echo_decay": (0, 1), "echo_op": (0, 4), "ps_filter": (0, 4), "ps_p1": (0, 20), "ps_p2": (0, 20), "ps_p3": (0, 20)},   # 11-25: formed Light (no strict)
}
GRAN_INT = {"l_bands", "s_bands", "density", "decay", "freeze_len", "brush", "palette", "strict", "colorama", "ae_lines", "cutout", "echo", "echo_op", "ps_filter", "ps_p1", "ps_p2", "ps_p3"}   # step can be 2.5 (held 2,3,2,3)


def short_time(s):
    """0.4 -> "0.4 s", 75 -> "1:15" (a clip under a minute reads better in seconds than as 0:00)."""
    return f"{s:.1f} s" if s < 60 else f"{int(s // 60)}:{int(s % 60):02d}"


_strict = {"key": None, "val": {}}


def granular_strict():
    """granular.py's STRICT table ({look: {setting: value}}) for the page's "Takagi strict" toggle, so the page moves its
    sliders to the engine's own strict values and leaves them out of the job (sent values beat STRICT in the engine).
    Read from granular.py's source, not imported: this server runs on the system python3 (no numpy / OpenCV) and
    never runs the engine in-process. Re-read whenever the file changes, so new STRICT keys reach the page by
    themselves. A value that isn't a plain literal falls back to asking the engine's own Python for the table."""
    try:
        st = GRANULAR.stat(); key = (st.st_mtime_ns, st.st_size)
    except OSError: return {}
    if _strict["key"] == key: return _strict["val"]
    val, whole = {}, True
    try:
        tree = ast.parse(GRANULAR.read_text(encoding="utf-8"))
        def table(node):   # {look: dict(k=v, ...) or {"k": v}} -> plain dicts; whole=False when something isn't a literal
            nonlocal whole
            out = {}
            if not isinstance(node, ast.Dict): whole = False; return out
            for kn, vn in zip(node.keys, node.values):
                if not (isinstance(kn, ast.Constant) and isinstance(kn.value, str)): whole = False; continue
                pairs = []
                if isinstance(vn, ast.Call) and isinstance(vn.func, ast.Name) and vn.func.id == "dict" and not vn.args:
                    pairs = [(kw.arg, kw.value) for kw in vn.keywords]
                elif isinstance(vn, ast.Dict):
                    pairs = [(k.value if isinstance(k, ast.Constant) else None, v) for k, v in zip(vn.keys, vn.values)]
                else: whole = False
                d = {}
                for k, v in pairs:
                    try:
                        x = ast.literal_eval(v)
                        if k is None or not isinstance(x, (bool, int, float, str)): raise ValueError
                        d[k] = x
                    except Exception: whole = False
                out[kn.value] = d
            return out
        for node in tree.body:   # the last top-level STRICT = {...} wins, as it would on import
            tg = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) and node.value else []
            if any(isinstance(t, ast.Name) and t.id == "STRICT" for t in tg): val = table(node.value)
            elif any(isinstance(n, ast.Name) and n.id == "STRICT" for t in tg for n in ast.walk(t)): whole = False   # STRICT[...] = ... later on
            elif isinstance(node, ast.Expr) and any(isinstance(n, ast.Name) and n.id == "STRICT" for n in ast.walk(node)): whole = False   # STRICT[...].update(...)
        if not val: whole = False
    except (OSError, SyntaxError, ValueError):
        whole = False
    if not whole and PY.exists() and granular_ready():   # import it in the engine's Python (main() doesn't run on import)
        try:
            r = subprocess.run([str(PY), "-c", "import json, sys, importlib.util as u; s = u.spec_from_file_location('granular', sys.argv[1]); "
                                "g = u.module_from_spec(s); s.loader.exec_module(g); "
                                "print(json.dumps({l: {k: v for k, v in d.items() if isinstance(v, (bool, int, float, str))} for l, d in g.STRICT.items()}))",
                                str(GRANULAR)], capture_output=True, text=True, timeout=60, cwd=str(GRANULAR.parent))
            got = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else None
            if isinstance(got, dict) and got: val = got
        except Exception: pass
    if not val: val = _strict["val"]   # half-saved or broken file: keep the last table that read cleanly
    _strict.update(key=key, val=val)
    return val


def granular_settings(req, look):
    """the look settings a request may send ({k: number}), clamped to GRAN_SET; (settings, None) or (None, error)."""
    settings = {}
    for k, v in (req.get("settings") or {}).items() if isinstance(req.get("settings"), dict) else []:
        if k not in GRAN_SET[look]: continue
        try: x = float(v)
        except (TypeError, ValueError): return None, f"{k} needs a number"
        if x != x or not math.isfinite(x): continue   # NaN / Infinity
        lo, hi = GRAN_SET[look][k]; x = max(lo, min(hi, x))
        settings[k] = int(round(x)) if k in GRAN_INT else round(x, 4)
    return settings, None


# 11-25: one picture of the look for the Granular tab's preview (layout C). Runs granular.py --still beside any render;
# a newer request stops the one before it. Pictures go to app/out/.stills (git-ignored), the last few kept.
STILLS = OUT / ".stills"
_still = {"n": 0, "proc": None}
_still_lock = threading.Lock()


def granular_still(req):
    if not (PY.exists() and granular_ready()): return {"error": "Granular needs OpenCV, which isn't installed in this copy"}
    src = safe_path(req.get("source") or "") if req.get("source") else None
    if not src or not src.is_file() or src.suffix.lower() not in VID_EXT: return {"error": "pick a clip"}
    look = req.get("look") if req.get("look") in GRAN_SET else "granular"
    settings, err = granular_settings(req, look)
    if err: return {"error": err}
    try: at = float(req.get("at") or 0); at = at if math.isfinite(at) else 0.0
    except (TypeError, ValueError): at = 0.0
    try: width = int(max(160, min(640, float(req.get("width") or 640))))   # small: it's a preview, and sizes scale with width
    except (TypeError, ValueError): width = 640
    try: seed = max(0, int(float(req.get("seed") or 3)))
    except (TypeError, ValueError): seed = 3
    with _still_lock:
        _still["n"] += 1; n = _still["n"]; old = _still["proc"]
    if old and old.poll() is None:
        try: os.killpg(old.pid, signal.SIGTERM)
        except Exception: pass
    STILLS.mkdir(parents=True, exist_ok=True)
    out = STILLS / f"still_{n}.jpg"
    cmd = [str(PY), "-u", str(GRANULAR), "--source", str(src), "--out", str(out), "--look", look, "--still", f"{max(0.0, at):.3f}",
           "--width", str(width), "--seed", str(seed), "--preview", "0"]
    if settings: cmd += ["--set", *[f"{k}={v}" for k, v in settings.items()]]
    t0 = time.time()
    try:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", start_new_session=True)
    except Exception as e: return {"error": str(e)}
    with _still_lock: _still["proc"] = proc
    try: text, _ = proc.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        try: os.killpg(proc.pid, signal.SIGKILL)
        except Exception: pass
        return {"error": "the preview took too long"}
    if _still["n"] != n: return {"stale": True}   # a newer preview was asked for meanwhile
    if proc.returncode != 0 or not out.exists():
        msg = next((l[8:] for l in reversed(text.splitlines()) if l.startswith("[error] ")), "") or (text.strip().splitlines() or ["the preview failed"])[-1]
        return {"error": msg}
    for old_f in sorted(STILLS.glob("still_*.jpg"), key=lambda f: f.stat().st_mtime)[:-8]:   # keep the last few (4 pairs)
        try: old_f.unlink()
        except OSError: pass
    return {"ok": True, "after": str(out), "before": str(out.with_name(out.stem + "_src.jpg")), "at": at, "secs": round(time.time() - t0, 1)}


def start_granular(req):
    global job
    with lock:
        if job and job.get("running"): return {"error": "a render is already running"}
        src = safe_path(req.get("source") or "") if req.get("source") else None
        if not req.get("source"): return {"error": "pick a clip"}
        if not src: return {"error": "that clip is outside the folders this app can use (drop it in instead)"}
        if not src.is_file(): return {"error": "that clip isn't there any more"}
        if src.suffix.lower() not in VID_EXT: return {"error": "that isn't a video clip"}
        clip = ffprobe_duration(src)
        if clip <= 0: return {"error": "couldn't read that clip (is it a video ffmpeg can open?)"}
        if clip <= 0.2: return {"error": "that clip is too short — pick a longer one"}
        audio = ""
        if req.get("audio"):
            a = safe_path(req["audio"])
            if not a or not a.is_file() or a.suffix.lower() not in AUD_EXT: return {"error": "that song can't be used — pick another, or use the clip's own sound"}
            if not has_sound(a): return {"error": "that file has no sound — pick another, or use the clip's own sound"}
            audio = str(a)
        def f(k, d):   # NaN / Infinity (json.loads accepts them) fall back to the default
            try:
                x = float(req[k]) if req.get(k) not in (None, "") else d
                return x if math.isfinite(x) else d
            except Exception: return d
        full = req.get("length") == "full"
        start, duration = (0.0, 0.0) if full else (max(0.0, f("start", 0)), max(1.0, min(3600.0, f("duration", 15))))   # 11-25: the Length slider is freeform (1 s .. the clip)
        if start >= clip - 0.2:   # the engine needs a few frames past the start (it says the same)
            return {"error": f"the clip is only {short_time(clip)} long — start earlier than that"}
        song = ffprobe_duration(audio) if audio else 0.0
        song_start = max(0.0, f("song_start", 0)) if audio else 0.0
        if audio and song and song_start >= song - 1:
            return {"error": f"the song is only {int(song // 60)}:{int(song % 60):02d} long — start it earlier than that"}
        look = req.get("look") if req.get("look") in GRAN_SET else "granular"
        settings, err = granular_settings(req, look)
        if err: return {"error": err}
        name = unique_out(re.sub(r"[^a-z0-9]+", "-", (req.get("name") or "").lower()).strip("-") or time.strftime(f"granular_{look}_%m%d_%H%M%S"))
        length = (duration and min(duration, clip - start)) or (clip - start)
        cfps = ffprobe_video(src).get("fps") or 29.97
        fps = 30000 / 1001 if cfps > 31 else cfps   # the engine conforms clips above 31 fps to 29.97, keeps the rest
        j = {"name": name, "kind": "granular", "source": str(src), "audio": audio, "out": str(OUT / f"{name}.mp4"), "look": look,
             "start": start, "duration": duration, "song_start": song_start, "clip_duration": clip, "song_duration": song,
             "length": "full" if full else "test", "width": int(max(160, min(1920, f("width", 960)))), "seed": max(0, int(f("seed", 3))),
             "settings": settings, "fps": round(fps, 4),
             "running": True, "started": time.time(), "frames_total": max(1, int(length * fps)), "stage": "queued", "log": []}
        place(j, req, str(src), look)
        job = j
        threading.Thread(target=run_job, args=(j,), daemon=True).start()
        return {"ok": True, "job": public(j)}


def public(j):
    if not j: return None
    if j.get("kind") == "bank":
        sd = Path(j["out"]).parent / (Path(j["out"]).stem + "_shots")
        done = len(list(sd.glob("s_*.mp4"))) if sd.exists() else 0
    elif j.get("kind") == "collage":
        done = j.get("shots_done", 0) if j.get("running") or j.get("stage") != "done" else j.get("frames_total", 0)
    elif j.get("kind") == "granular":
        done = j.get("gdone", 0)
    else:
        done = count_frames(j.get("work") or j["out"])
    total = j.get("frames_total") or 0
    if j.get("kind") in ("latent", "bank", "granular") and not j.get("running") and j.get("stage") == "done" and Path(j["out"]).exists():
        done = total                      # latent_dream clears its frame folder once the mp4 is written; the bank's shots go to the Trash
    elapsed = (j.get("ended") or time.time()) - j["started"]
    rate = None
    if done > j.get("frames_at_start", 0) and j.get("render_t0"):
        rate = (time.time() - j["render_t0"]) / max(1, done - j["frames_at_start"])
    if j.get("stage") == "rendering" and not j.get("render_t0"):
        j["render_t0"] = time.time(); j["frames_at_start"] = done
    eta = (total - done) * rate if (rate and total) else None
    lf = latest_frame(j.get("work") or j["out"])
    # preview frames on disk (latent: one per frame; collage: one per picture shown) and the rate to play them back at
    pfps = j.get("fps", 24) / ((2 if str(j.get("hold")) == "auto" else max(1, int(j.get("hold") or 2))) if j.get("kind") == "collage" else (2 if j.get("draft") else 1))
    if j.get("kind") == "granular": pfps = j.get("pfps") or j.get("fps", 29.97) / 2   # the engine's "[preview] X per s" (until it says: step 2's rate)
    return {k: j[k] for k in ("name", "kind", "clips", "audio", "out", "start", "duration", "song_duration", "length", "steps", "fps", "seed", "running",
                              "started", "stage", "log", "width", "height", "min_shot", "max_shot", "moves", "move_chance", "tape", "gen", "damage", "faults", "react", "tape_done", "tape_total",
                              "model", "bend", "spin", "decay", "technique", "hold", "lens", "label", "decay_shape", "speed_fpm", "pulse_react",
                              "motion_react", "psi", "size", "engine_mode", "deliv", "draft", "trashed",
                              "source", "look", "settings", "song_start", "clip_duration") if k in j} | {
        "frames_done": done, "frames_total": total, "elapsed": elapsed, "eta": eta, "sec_per_frame": rate,
        "preview_frames": count_frames(j.get("work") or j["out"]) if lf else 0, "preview_fps": pfps,
        "latest_frame": str(lf) if lf else None, "latest_index": int(lf.stem.split("_")[1]) if lf else None,
        "error": j.get("error"), "rc": j.get("rc"), "ended": j.get("ended"),
        "output_ready": Path(j["out"]).exists() and not j.get("running") and Path(j["out"]).stat().st_mtime >= j["started"] - 1,
    }


def safe_path(q):
    """Confine reads to the project folder and the media roots the app offers.
    A bare prefix test also matched sibling folders (…/python-evil), so compare
    resolved parents instead."""
    try:
        p = Path(q).expanduser().resolve()
    except Exception:
        return None
    roots = [ROOT, DELIV] + [Path(l) for l in LANES] + [Path("/Volumes")]
    for r in roots:
        try:
            if p == r or r in p.parents:
                return p
        except Exception:
            continue
    return None


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send_json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)

    def send_file(self, path, ctype=None, sandbox=False):
        p = Path(path)
        if not p.is_file(): self.send_error(404); return
        ctype = ctype or {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".tif": "image/tiff", ".tiff": "image/tiff",
                          ".webp": "image/webp", ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm", ".gif": "image/gif", ".m4a": "audio/mp4", ".mp3": "audio/mpeg", ".wav": "audio/wav",
                          ".html": "text/html; charset=utf-8"}.get(p.suffix.lower(), "application/octet-stream")
        size = p.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        ok_range = False
        if rng and rng.startswith("bytes="):
            try:
                a, _, b = rng[6:].split(",")[0].partition("-")
                start = int(a) if a else max(0, size - int(b)); end = int(b) if (b and a) else size - 1
                end = min(end, size - 1); ok_range = 0 <= start <= end
            except ValueError:
                ok_range = False
            if not ok_range: start, end = 0, size - 1
        if ok_range:
            self.send_response(206); self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Accept-Ranges", "bytes"); self.send_header("Content-Type", ctype); self.send_header("X-Content-Type-Options", "nosniff")
        if sandbox: self.send_header("Content-Security-Policy", "sandbox")   # an .html from disk can be viewed, never run scripts on this origin
        self.send_header("Content-Length", str(end - start + 1)); self.send_header("Cache-Control", "no-store"); self.end_headers()
        with open(p, "rb") as f:
            f.seek(start); left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk: break
                try: self.wfile.write(chunk)
                except BrokenPipeError: return
                left -= len(chunk)

    def same_origin(self):
        """Reject cross-site requests. The browser is inside localhost, so binding
        to 127.0.0.1 is not on its own a defence: any page you visit can reach us."""
        o = self.headers.get("Origin") or self.headers.get("Referer") or ""
        if not o:
            return True                      # curl / the native window send neither
        host = urllib.parse.urlparse(o).netloc
        return host in (f"127.0.0.1:{PORT}", f"localhost:{PORT}", f"[::1]:{PORT}")

    def bad_host(self):
        """DNS-rebinding guard: a hostile page can point its own domain at 127.0.0.1; the Host header gives it away."""
        h = (self.headers.get("Host") or "").lower()
        if h in (f"127.0.0.1:{PORT}", f"localhost:{PORT}", f"[::1]:{PORT}"): return False
        self.send_error(403); return True

    def deny_cross_site(self):
        if self.same_origin():
            return False
        self.send_json({"error": "cross-site request refused"}, code=403)
        return True

    def safe_path(self, q):
        return safe_path(q)   # module-level since 11-23 (start_granular checks the clip with it too)

    def do_GET(self):
        if self.bad_host(): return
        if self.path.startswith("/api/") and (self.headers.get("Sec-Fetch-Site") or "") == "cross-site":
            return self.send_error(403)   # another website's <img>/<video> probing local files
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        if u.path in ("/", "/index.html"): return self.send_file(APP / "index.html")
        if u.path == "/favicon.ico": self.send_response(204); self.end_headers(); return
        if u.path == "/models": return self.send_file(APP / "models.html")
        if u.path == "/crt":   # friends' copies: no AM.FM button, logos or names (the page reads OWNER)
            if OWNER: return self.send_file(APP / "crt.html")
            b = (APP / "crt.html").read_text(encoding="utf-8").replace("const OWNER=/*server*/true;", "const OWNER=/*server*/false;", 1).encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b); return
        if u.path.startswith("/crt/logos/"):
            return self.send_file(APP / "crt_logos" / Path(urllib.parse.unquote(u.path)).name)
        if u.path == "/api/presets": m = presets_load(); return self.send_json({k: m.get(k, []) for k in PRE_LOOKS})
        if u.path == "/api/models/state": return self.send_json(models_lib.state())
        if u.path == "/update.js": return self.send_file(APP / "update.js", "text/javascript; charset=utf-8")
        if u.path == "/api/update": return self.send_json(updater.state())
        if u.path == "/api/state":
            auds = list_media()
            def lane_name(l):
                if l in LANE_NAMES: return LANE_NAMES[l]
                pp = Path(l)
                if pp.name == "FOOTAGE": return "FOOTAGE (all)"
                if pp.name == "clips": return pp.parent.name + " clips"
                return pp.name
            lanes = [{"name": lane_name(l), "path": l} for l in LANES if Path(l).is_dir()]
            DROPPED.mkdir(parents=True, exist_ok=True)
            dropped = [{"name": f.name, "path": str(f), "kind": "video" if f.suffix.lower() in VID_EXT else "image", "size": f.stat().st_size,
                        **({"duration": ffprobe_duration(f), **ffprobe_video(f)} if f.suffix.lower() in VID_EXT else {})}
                       for f in sorted(DROPPED.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True) if f.is_file() and not f.name.startswith(".")]
            return self.send_json({"job": public(job), "ai": ai_ready(), "audio": auds, "inbox": list_inbox(), "lanes": lanes, "models": list_models(), "models_dir": str(MODELS), "dropped": dropped, "dropped_dir": str(DROPPED), "outputs": list_outputs(), "projects": list_projects(), "photoshop": Path("/Applications/Adobe Photoshop 2026/Adobe Photoshop 2026.app").exists(), "granular": granular_ready(),
                                   "strict": granular_strict(), "root": str(ROOT), "engine_ok": LATENT.exists() and PY.exists()})
        if u.path == "/api/file":
            p = self.safe_path(q.get("p", [""])[0])
            return self.send_file(p, sandbox=True) if p else self.send_error(403)
        if u.path == "/api/frame":
            # latest frame of the current job, or frame n of an output
            if job and q.get("latest"):
                lf = latest_frame(job.get("work") or job["out"]); return self.send_file(lf) if lf else self.send_error(404)
            out = self.safe_path(q.get("out", [""])[0])
            try: n = max(0, int(q.get("n", ["0"])[0]))
            except ValueError: n = 0
            # finished renders live in DELIV (Movies/Liminal Engine/<arm>/, which may be a shortcut to the T7), working
            # ones in OUT/ROOT; anything else is refused. Thumbnails of finished renders are cached in OUT/_thumbs so
            # nothing gets written next to the user's videos.
            in_work = bool(out) and out.parent in (OUT, ROOT)
            try: in_deliv = bool(out) and DELIV.resolve() in out.parents
            except Exception: in_deliv = False
            if not out or out.suffix.lower() not in (".mp4", ".mov", ".webm") or not (in_work or in_deliv): return self.send_error(403)
            if in_work:
                fs = preview_files(frames_dir_for(out))
                if fs: return self.send_file(fs[min(n, len(fs) - 1)])
            # no frame folder (a cut bank, a delivered render, or an old one): pull a midpoint frame from the video once
            if in_work: th = out.parent / (out.stem + "_thumb.jpg")
            else:
                td = OUT / "_thumbs"; td.mkdir(parents=True, exist_ok=True)
                th = td / (out.parent.name + "--" + out.stem + "_thumb.jpg")
            if not th.exists() and out.exists():
                d = ffprobe_duration(out)
                subprocess.run([FFMPEG or "ffmpeg", "-y", "-v", "error", "-ss", f"{d/2:.2f}", "-i", str(out), "-frames:v", "1", "-vf", "scale=480:-2", str(th)], capture_output=True, timeout=30)
            return self.send_file(th) if th.exists() else self.send_error(404)
        if u.path in ("/api/reveal", "/api/open"):
            # These hand a path to macOS `open`, which will execute an app or an
            # installer. GET made them reachable from any page via an <img> tag.
            return self.send_json({"error": "use POST"}, code=405)
        self.send_error(404)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        if self.bad_host() or self.deny_cross_site(): return
        try: n = int(self.headers.get("Content-Length") or 0)
        except ValueError: n = -1
        if n < 0: return self.send_json({"error": "bad Content-Length"}, code=400)
        if u.path == "/api/upload" and n > MAX_UPLOAD:
            return self.send_json({"error": f"file is larger than {MAX_UPLOAD // (1024*1024)} MB"}, code=413)
        streamed = u.path == "/api/crt_part" or (u.path == "/api/crt_export" and q.get("kind", ["video"])[0] == "video")
        if streamed and n:   # recordings and source videos can be GBs: straight to disk in 1 MB pieces, never whole in memory
            CRT_OUT.mkdir(parents=True, exist_ok=True); (CRT_OUT / "_raw").mkdir(exist_ok=True)
            body = CRT_OUT / "_raw" / ("incoming-" + os.urandom(6).hex()); got = 0
            try:
                with open(body, "wb") as f:
                    while got < n:
                        chunk = self.rfile.read(min(1 << 20, n - got))
                        if not chunk: break
                        f.write(chunk); got += len(chunk)
            except BaseException:
                body.unlink(missing_ok=True); raise   # a dropped connection or a full disk leaves no partial file
            if got != n: body.unlink(missing_ok=True); return self.send_json({"error": "the upload was cut short"}, 400)
        else:
            body = self.rfile.read(n) if n else b""
            if len(body) != n: return self.send_json({"error": "the upload was cut short"}, 400)
        try: return self._post(u, q, body)
        finally:
            if isinstance(body, Path): body.unlink(missing_ok=True)   # a streamed body the handler didn't move into place

    def _post(self, u, q, body):
        global job
        if u.path == "/api/renders_folder":
            subprocess.Popen(["open", str(DELIV)]); return self.send_json({"ok": True, "path": str(DELIV)})
        if u.path == "/api/open_browser":   # the CRT in the default browser (11-16: the app window may not get the camera)
            subprocess.Popen(["open", f"http://127.0.0.1:{PORT}/crt"]); return self.send_json({"ok": True})
        if u.path == "/api/update/check":
            return self.send_json(updater.check(force=True))
        if u.path == "/api/update/apply":
            def busy():
                if job and job.get("running"): return "a render is running; update when it's done"
                if CRT_BUSY[0]: return "a CRT export is saving; update when it's done"
                models_lib.clear_stale()
                if any(d.get("state") in ("queued", "downloading") for d in models_lib.DL.values()):
                    return "a model is downloading; update when it's done"
            ok, why = updater.apply(busy)
            return self.send_json({"ok": ok, "error": why, **updater.state()})
        if u.path in ("/api/reveal", "/api/open"):
            try: req = json.loads(body or b"{}")
            except Exception: req = {}
            p = self.safe_path(req.get("p", ""))
            if not p or not p.exists():
                return self.send_json({"error": "nothing at that path"})
            media = IMG_EXT | AUD_EXT | VID_EXT
            if u.path == "/api/open" and not (p.suffix.lower() in media and p.is_file()):
                return self.send_json({"error": "refusing to open anything but a picture, song or video"}, code=403)
            if p.suffix.lower() in EXECUTABLE_EXT or p.is_dir() and p.suffix.lower() == ".app":
                return self.send_json({"error": "refusing to open an executable"}, code=403)
            subprocess.Popen(["open", *(["-R"] if u.path == "/api/reveal" else []), str(p)])
            return self.send_json({"ok": True})
        if u.path == "/api/upload":
            name = Path(urllib.parse.unquote(q.get("name", ["file"])[0])).name
            name = re.sub(r"[^A-Za-z0-9._ ()-]+", "_", name)
            folder = DROPPED if q.get("kind", [""])[0] == "dropped" else INBOX
            folder.mkdir(parents=True, exist_ok=True)
            dest = folder / name; i = 1
            while dest.exists() and dest.stat().st_size != len(body):
                dest = folder / f"{Path(name).stem}-{i}{Path(name).suffix}"; i += 1
            dest.write_bytes(body)
            return self.send_json({"ok": True, "path": str(dest), "name": dest.name})
        if u.path == "/api/models/remove":
            try: req = json.loads(body or b"{}")
            except Exception: req = {}
            pth = self.safe_path(req.get("path", ""))
            if not pth or pth.suffix.lower() != ".pkl" or not pth.is_file() or pth.parent not in [d.resolve() for d in models_lib.model_dirs()]:
                return self.send_json({"error": "that is not an installed model"}, 400)
            if job and job.get("running") and job.get("model") and Path(job["model"]).resolve() in [pth, *models_lib.copies_of(pth)]:
                return self.send_json({"error": "that model is rendering right now"}, 409)
            try: items = [trash_item(c) for c in [pth, *models_lib.copies_of(pth)]]   # the same model on this Mac and the T7: both go
            except Exception as e: return self.send_json({"error": f"could not move it to the Trash: {e}"}, 500)
            return self.send_json({"ok": True, "trashed": items})
        if u.path in ("/api/models/picks", "/api/models/download"):
            try: req = json.loads(body or b"{}")
            except Exception: req = {}
            names = [n for n in (req.get("names") or []) if isinstance(n, str)]
            if u.path.endswith("picks"):
                models_lib.save_picks(names); return self.send_json({"ok": True})
            return self.send_json(models_lib.queue(names))
        if u.path == "/api/crt_frame":
            try: crt_frame(q.get("job", [""])[0], q.get("i", ["0"])[0], body)
            except Exception as e: return self.send_json({"error": str(e)[-200:]}, 400)
            return self.send_json({"ok": True})
        if u.path == "/api/crt_part":
            try: crt_part(q.get("job", [""])[0], q.get("name", [""])[0], body)
            except Exception as e: return self.send_json({"error": str(e)[-200:]}, 400)
            return self.send_json({"ok": True})
        if u.path == "/api/crt_audio":
            try: wav = crt_audio(q.get("job", [""])[0])
            except Exception as e: return self.send_json({"error": str(e)[-200:]}, 400)
            if not wav: self.send_response(204); self.end_headers(); return
            return self.send_file(wav, "audio/wav")
        if u.path == "/api/crt_render":
            fps = q.get("fps", ["30000/1001"])[0]
            if not re.fullmatch(r"[1-9]\d{0,5}(/[1-9]\d{0,4})?", fps): return self.send_json({"error": "bad fps"}, 400)
            CRT_BUSY[0] += 1
            try: out = crt_render(q.get("job", [""])[0], q.get("fmt", ["mp4"])[0], fps)
            except ValueError as e: return self.send_json({"error": str(e)[-400:]}, 400)
            except Exception as e: return self.send_json({"error": str(e)[-400:]}, 500)
            finally: CRT_BUSY[0] -= 1
            return self.send_json({"ok": True, "path": str(out)})
        if u.path == "/api/crt_export":
            kind = q.get("kind", ["video"])[0]; fmt = q.get("fmt", ["mp4"])[0]
            if not body or (isinstance(body, Path) and body.stat().st_size == 0): return self.send_json({"error": "nothing was recorded"}, 400)
            CRT_BUSY[0] += 1
            try: out = crt_export(body, kind, fmt, q.get("name", [None])[0], q.get("folder", [None])[0])
            except ValueError as e: return self.send_json({"error": str(e)[-400:]}, 400)
            except Exception as e: return self.send_json({"error": str(e)[-400:]}, 500)
            finally: CRT_BUSY[0] -= 1
            return self.send_json({"ok": True, "path": str(out)})
        if u.path == "/api/latent":
            try: req = json.loads(body or b"{}")
            except Exception: return self.send_json({"error": "bad request"}, 400)
            return self.send_json(start_latent(req))
        if u.path == "/api/bank":
            try: req = json.loads(body or b"{}")
            except Exception: return self.send_json({"error": "bad request"}, 400)
            return self.send_json(start_bank(req))
        if u.path == "/api/granular":
            try: req = json.loads(body or b"{}")
            except Exception: return self.send_json({"error": "bad request"}, 400)
            if not isinstance(req, dict): return self.send_json({"error": "bad request"}, 400)
            return self.send_json(start_granular(req))
        if u.path in ("/api/presets/save", "/api/presets/remove"):
            try: req = json.loads(body or b"{}")
            except Exception: return self.send_json({"error": "bad request"}, 400)
            look, name = req.get("look"), " ".join(str(req.get("name") or "").split())[:40]
            if look not in PRE_LOOKS or not name: return self.send_json({"error": "a preset needs a look and a name"}, 400)
            if u.path == "/api/presets/save":
                v = req.get("v")
                if not isinstance(v, dict) or len(json.dumps(v)) > 20000: return self.send_json({"error": "bad settings"}, 400)
                v = {str(k)[:40]: (x if isinstance(x, (bool, int, float)) else str(x)[:40]) for k, x in v.items()}
                def save(m):
                    lst = [x for x in m.get(look, []) if x.get("name") != name]; lst.append({"name": name, "v": v, "made": time.time()}); m[look] = lst
                return self.send_json({"ok": True, "presets": presets_edit(save)})
            def remove(m):
                keep, gone = [], []
                for x in m.get(look, []): (gone if x.get("name") == name else keep).append(x)
                m[look] = keep; m.setdefault("_removed", []).extend(dict(x, look=look, removed=time.time()) for x in gone)
            return self.send_json({"ok": True, "presets": presets_edit(remove)})
        if u.path == "/api/granular/still":
            try: req = json.loads(body or b"{}")
            except Exception: return self.send_json({"error": "bad request"}, 400)
            if not isinstance(req, dict): return self.send_json({"error": "bad request"}, 400)
            return self.send_json(granular_still(req))
        if u.path in ("/api/star", "/api/recipe", "/api/project"):   # 11-25 render files: keep, Remake, new project
            try: req = json.loads(body or b"{}")
            except Exception: return self.send_json({"error": "bad request"}, 400)
            if u.path == "/api/project":
                name = project_name(req.get("name"))
                try: (DELIV / name).mkdir(parents=True, exist_ok=True)
                except OSError as e: return self.send_json({"error": f"couldn't make that folder: {e}"})
                lib_dirty(); return self.send_json({"ok": True, "project": name, "projects": list_projects()})
            p = self.safe_path(req.get("path") or "")
            try: rel = str(p.relative_to(DELIV)) if p else None
            except ValueError: rel = None
            if not rel or not p.is_file(): return self.send_json({"error": "that render isn't in the renders folder"})
            if u.path == "/api/star":
                meta_update(rel, star=True if req.get("on") else None); lib_dirty(); return self.send_json({"ok": True})
            m = meta_load().get(rel) or {}
            if "ui" not in m: return self.send_json({"error": "this render was made before settings were kept, so it can't be remade"})
            return self.send_json({"ok": True, "ui": m["ui"]})
        if u.path == "/api/remove_outputs":
            try: req = json.loads(body or b"{}")
            except Exception: req = {}
            items, moved, errs = [], [], []
            for raw in req.get("paths") or []:
                out = self.safe_path(raw)
                if not out or out.suffix.lower() not in (".mp4", ".mov", ".webm", ".gif") or not out.exists(): continue
                if job and job.get("running") and Path(job.get("out", "")).resolve() == out: errs.append(f"{out.name} is rendering"); continue   # never pull a render mid-flight
                # a render can live twice: the engine's copy (app/out or ROOT) and its hard link / copy in
                # ~/Movies/Liminal Engine/<arm>/. Removing one must take the twin too, or it reappears.
                twins = [out]
                for cand in [OUT / out.name, ROOT / out.name, *DELIV.glob("*/" + out.name)]:
                    if cand.exists() and cand.resolve() not in [t.resolve() for t in twins] and same_render(cand, out): twins.append(cand)
                try:
                    for t in twins:
                        items.append(trash_item(t))
                        for suf in ("_frames", "_src", "_shots", "_thumb.jpg"):
                            side = t.parent / (t.stem + suf)
                            if side.exists(): items.append(trash_item(side))
                    moved.append(out.name)
                except Exception as e: errs.append(f"{out.name}: {e}")
            return self.send_json({"ok": True, "moved": moved, "trashed": items, "error": "; ".join(errs) or None})
        if u.path == "/api/restore":
            # Undo: put things that were just moved to the Trash back where they were
            try: req = json.loads(body or b"{}")
            except Exception: req = {}
            back, errs = [], []
            for it in req.get("items") or []:
                src, dst = Path(str(it.get("trashed", ""))), Path(str(it.get("orig", "")))
                if not in_trash(src) or not src.exists() and not src.is_symlink(): errs.append(f"{dst.name}: no longer in the Trash"); continue
                if not self.safe_path(str(dst)) or not dst.is_absolute(): errs.append(f"{dst.name}: not a place this app writes to"); continue
                if dst.exists(): errs.append(f"{dst.name}: something new is already there"); continue
                try: dst.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(src), str(dst)); back.append(str(dst))
                except Exception as e: errs.append(f"{dst.name}: {e}")
            lib_dirty()
            return self.send_json({"ok": not errs, "restored": back, "error": "; ".join(errs) or None})
        if u.path in ("/api/dropped_remove", "/api/inbox_remove"):
            try: req = json.loads(body or b"{}")
            except Exception: req = {}
            pth = self.safe_path(req.get("path", ""))
            home = DROPPED if u.path == "/api/dropped_remove" else INBOX
            if not pth or pth.parent != home.resolve() or not pth.is_file():
                return self.send_json({"error": "that file is not in the " + ("dropped clips" if home is DROPPED else "inbox")}, 400)
            if job and job.get("running") and str(pth) in (job.get("audio"), job.get("model"), job.get("source")):
                return self.send_json({"error": "that file is in the render that's running"}, 409)
            try: return self.send_json({"ok": True, "trashed": [trash_item(pth)]})
            except Exception as e: return self.send_json({"error": f"could not move it to the Trash: {e}"}, 500)
        if u.path == "/api/check":
            try: req = json.loads(body or b"{}")
            except Exception: req = {}
            pth = self.safe_path(req.get("path", ""))
            if pth is None:
                return self.send_json({"error": "that path is outside the folders this app can use"})
            if pth.is_file():
                e = pth.suffix.lower(); kind = "model" if e == ".pkl" else "video" if e in VID_EXT else "image" if e in IMG_EXT else "audio" if e in AUD_EXT else "file"
                return self.send_json({"ok": True, "kind": kind, "path": str(pth), "name": pth.name, "duration": ffprobe_duration(pth) if kind != "image" else 0,
                                      **(ffprobe_video(pth) if kind == "video" else {})})
            if pth.is_dir(): return self.send_json({"ok": True, "kind": "folder", "path": str(pth), "name": pth.name})
            return self.send_json({"error": "nothing at that path"})
        if u.path == "/api/stop":
            if job and job.get("running") and job.get("pid"):
                job["stopping"] = True
                try: os.killpg(os.getpgid(job["pid"]), 15)   # ffmpeg children too
                except Exception:
                    try: os.kill(job["pid"], 15)
                    except Exception: pass
            return self.send_json({"ok": True})
        if u.path in ("/api/clear_frames", "/api/rerender"):
            # remove a finished render (video, its delivered twin, working folders) to the Trash so the name is free.
            # /api/rerender then starts the same render again under the same name.
            try: req = json.loads(body or b"{}")
            except Exception: req = {}
            out = self.safe_path(req.get("out", ""))
            mine = bool(out) and bool(job) and Path(job.get("out", "")).resolve() == out   # the last render, wherever it was moved to
            if not out or out.suffix.lower() != ".mp4" or (out.parent not in (OUT.resolve(), ROOT.resolve()) and not mine):
                return self.send_json({"error": "that is not one of this app's renders"}, 400)
            if job and job.get("running"):
                return self.send_json({"error": "a render is running"}, 409)
            items = []
            try:
                for suf in ("_frames", "_src", "_shots", "_thumb.jpg"):
                    fd = out.parent / (out.stem + suf)
                    if fd.exists(): items.append(trash_item(fd))
                if out.exists():
                    for tw in DELIV.glob("*/" + out.name):
                        if same_render(tw, out): items.append(trash_item(tw))
                    items.append(trash_item(out))
            except Exception as e:
                return self.send_json({"error": f"could not move it to the Trash: {e}", "trashed": items}, 500)
            if u.path == "/api/rerender":
                if not job or Path(job.get("out", "")).resolve() != out:
                    return self.send_json({"error": "only the last render can be rendered again", "trashed": items}, 400)
                with lock:
                    keep = {k: v for k, v in job.items() if k not in ("running", "started", "ended", "stage", "log", "pid", "rc", "error", "stopping",
                                                                       "deliv", "trashed", "shots_done", "tape_done", "tape_total", "render_t0", "frames_at_start", "cmd", "gdone")}
                    job = {**keep, "out": keep.get("work") or keep["out"], "running": True, "started": time.time(), "stage": "queued", "log": []}   # renders from app/out again, then moves
                    threading.Thread(target=run_job, args=(job,), daemon=True).start()
                return self.send_json({"ok": True, "trashed": items, "job": public(job)})
            return self.send_json({"ok": True, "trashed": items})
        self.send_error(404)


updater.check_soon()   # no-op in Eric's copy (no app/channel.json)

if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    print(f"LIMINAL ENGINE  →  http://localhost:{PORT}   (leave this window open; close it to stop)")
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
