#!/usr/bin/env python3
"""LIMINAL ENGINE — local web front end for latent_dream.py (LATENT DREAM) and app/bank.py (CUT BANK).
The old Stable Diffusion HALLUCINATE mode was retired 10-18 (hallucinate.py → _cleanup/retired-2026-10-18/).
Standard library only. One job at a time (the GPU is the bottleneck).
Run via "LIMINAL ENGINE.command" or: python3 app/server.py
"""
import json, os, re, shutil, subprocess, sys, threading, time, urllib.parse
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
DELIV = Path.home() / "Movies" / "Liminal Engine"      # finished renders, by arm
# 11-01: that folder is now a shortcut to T7 (/Volumes/T7/_Video Projects/_Liminal Engine) so Premiere links keep working.
# If T7 isn't plugged in when the engine starts, renders go to a local offline folder instead (move them to T7 later).
if not DELIV.exists():
    DELIV = Path.home() / "Movies" / "Liminal Engine (offline)"
    DELIV.mkdir(parents=True, exist_ok=True)
ARM_DIR = {"latent": "dream", "dream": "dream", "collage": "collage", "bank": "cut bank", "crt": "crt"}
PY = ROOT / ".venv" / "bin" / "python"
ENGINE = ROOT / "hallucinate.py"
LATENT = ROOT / "latent_dream.py"          # LATENT DREAM: StyleGAN walk (Lucid Sonic Dreams method) + bend/decay
MODELS = ROOT / "models" / "stylegan"       # drop StyleGAN2 .pkl files here
PORT = int(os.environ.get("PORT", "8391"))
INBOX.mkdir(parents=True, exist_ok=True)   # a fresh copy ships without these;
OUT.mkdir(parents=True, exist_ok=True)     # list_media/list_outputs iterdir them on the first request
for _a in set(ARM_DIR.values()): (DELIV / _a).mkdir(parents=True, exist_ok=True)
HUB = Path.home() / ".cache/huggingface/hub"
MODEL_CACHED = (HUB / "models--stable-diffusion-v1-5--stable-diffusion-v1-5").exists()
def ai_ready():
    """The AI tier is optional: the CUT BANK only needs ffmpeg + librosa."""
    try: return any((PY.parent.parent / "lib").glob("python*/site-packages/torch"))
    except Exception: return False
MAX_UPLOAD = 6 * 1024 * 1024 * 1024        # uploads are read into memory; refuse absurd ones
EXECUTABLE_EXT = {".app", ".command", ".sh", ".terminal", ".webloc", ".dmg", ".pkg", ".scpt", ".workflow", ".shortcut"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}
AUD_EXT = {".m4a", ".mp3", ".wav", ".aif", ".aiff", ".flac", ".ogg", ".mp4", ".mov"}
VID_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".mts"}
BANK = APP / "bank.py"
COLLAGE = APP / "collage.py"
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

COHERENCE = {1: (4.5, 0.06, 0.34), 2: (5.5, 0.10, 0.30), 3: (7.0, 0.12, 0.26), 4: (8.5, 0.16, 0.22), 5: (10.0, 0.20, 0.18)}
STYLE_TAIL = ", grainy film photograph, muted tones, soft focus"
NEG = "text, watermark, neon, oversaturated, garish"

job = None          # current/last job dict
lock = threading.Lock()


_dur_cache = {}


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


def list_media():
    seen, imgs, auds, vids = set(), [], [], []
    for d in (INBOX, ROOT):
        for p in sorted(d.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not p.is_file() or p.name.startswith(".") or p.name in seen: continue
            seen.add(p.name)
            e = p.suffix.lower()
            if e in IMG_EXT: imgs.append({"name": p.name, "path": str(p)})
            elif e in VID_EXT and d is INBOX: vids.append({"name": p.name, "path": str(p), "duration": ffprobe_duration(p)})
            elif e in AUD_EXT and not p.name.startswith("_seg") and not (e in (".mp4", ".mov") and d is ROOT): auds.append({"name": p.name, "path": str(p), "duration": ffprobe_duration(p)})
    return imgs[:40], auds[:40], vids[:40]


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

CRT_OUT = OUT / "_work" / "crt"      # browser recordings, frame jobs, raw stills: working files, not deliverables
CRT_DELIV = DELIV / "crt"
_SAFE_NAME = re.compile(r"[^\w .()\-]+")
def crt_export(body, kind, fmt, name=None, folder=None):
    """CRT arm: the page records/grabs in the browser (webm / png); turn it into the chosen file type.
    Stills may carry a name and a folder (batch stills, 11-01) so a fast run never overwrites itself."""
    CRT_OUT.mkdir(parents=True, exist_ok=True); CRT_DELIV.mkdir(parents=True, exist_ok=True)
    stem = "amfm-crt-" + time.strftime("%Y%m%d-%H%M%S")
    raw = CRT_OUT / "_raw"; raw.mkdir(exist_ok=True)
    if kind == "still":
        out_dir = CRT_DELIV
        if folder:
            out_dir = CRT_DELIV / (_SAFE_NAME.sub("_", folder).strip(" .")[:80] or "batch"); out_dir.mkdir(exist_ok=True)
        if name: stem = _SAFE_NAME.sub("_", name).strip(" .")[:120] or stem
        src = raw / (stem + "-" + os.urandom(3).hex() + ".png"); src.write_bytes(body)
        if fmt == "png":
            dst = out_dir / (stem + ".png"); shutil.copyfile(src, dst); return dst
        dst = out_dir / (stem + (".jpg" if fmt == "jpg" else ".tif"))
        args = ["-q:v", "2"] if fmt == "jpg" else ["-compression_algo", "lzw"]
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(src), *args, str(dst)], check=True, capture_output=True, timeout=60)
        return dst
    # Chrome records WebM; Safari's engine (the native app) may only record MP4 — name the file by what it really is
    is_mp4 = body[4:8] == b"ftyp"
    src = raw / (stem + (".mp4" if is_mp4 else ".webm")); src.write_bytes(body)
    if fmt == "webm" and not is_mp4:
        dst = CRT_DELIV / (stem + ".webm"); shutil.copyfile(src, dst); src.unlink(missing_ok=True); return dst
    dst = crt_convert(src, fmt, stem)
    src.unlink(missing_ok=True)   # converted: the raw recording was only a working file
    return dst


def crt_convert(src, fmt, stem):
    """A finished CRT video (browser recording, or frames already stitched) → the chosen file type."""
    cfr = ["-r", "30000/1001"]     # browser recordings have uneven timing; editors want constant frame rate
    if fmt == "webm":
        dst = CRT_DELIV / (stem + ".webm")
        cmd = [FFMPEG, "-v", "error", "-y", "-i", str(src), "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "18", "-row-mt", "1",
               "-c:a", "libopus", "-b:a", "192k", str(dst)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0: raise RuntimeError(r.stderr.strip() or "ffmpeg failed")
        return dst
    if fmt == "mp4":
        dst = CRT_DELIV / (stem + ".mp4")
        cmd = [FFMPEG, "-v", "error", "-y", "-i", str(src), *cfr, "-c:v", "libx264", "-crf", "14", "-preset", "slow", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", str(dst)]
    elif fmt == "prores":
        dst = CRT_DELIV / (stem + ".mov")
        cmd = [FFMPEG, "-v", "error", "-y", "-i", str(src), *cfr, "-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le",
               "-c:a", "pcm_s16le", str(dst)]
    elif fmt == "gif":
        dst = CRT_DELIV / (stem + ".gif")
        cmd = [FFMPEG, "-v", "error", "-y", "-i", str(src), "-vf",
               "fps=15,scale=960:-2:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4", str(dst)]
    elif fmt == "png":
        dst = CRT_DELIV / (stem + "_png"); dst.mkdir(exist_ok=True)
        cmd = [FFMPEG, "-v", "error", "-y", "-i", str(src), *cfr, str(dst / "frame_%04d.png")]
    else:
        raise ValueError("unknown format " + fmt)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0: raise RuntimeError(r.stderr.strip() or "ffmpeg failed")
    return dst


# EXPORT WHOLE VIDEO without playing it through (10-22, Eric: "just have it download instead of having to play
# through"): the page steps the video frame by frame and draws each through the CRT. Where the browser has a hardware
# H.264 encoder it sends one encoded stream (part "stream"); otherwise every frame as a JPEG. It also sends the
# original video (part "source") for its sound. Here they are stitched, then converted like a recording.
_JOB_OK = re.compile(r"^[a-z0-9]{4,24}$")
def crt_job_dir(job):
    if not _JOB_OK.match(job or ""): raise ValueError("bad job")
    d = CRT_OUT / "_raw" / ("job-" + job); d.mkdir(parents=True, exist_ok=True); return d

def crt_frame(job, i, body):
    if not body or body[:2] != b"\xff\xd8": raise ValueError("not a JPEG frame")
    (crt_job_dir(job) / ("%06d.jpg" % int(i))).write_bytes(body)

def crt_part(job, name, body):
    if name not in ("stream", "source", "sound"): raise ValueError("bad part")
    if not body: raise ValueError("empty part")
    (crt_job_dir(job) / {"stream": "stream.h264", "source": "source.bin", "sound": "sound.wav"}[name]).write_bytes(body)

def crt_audio(job):
    """The uploaded video's sound as 48 kHz stereo WAV for the page's sound chain, or None when it has none."""
    d = crt_job_dir(job); src = d / "source.bin"; dst = d / "source.wav"
    if not src.exists(): return None
    r = subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(src), "-map", "0:a:0?", "-vn", "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", "-f", "wav", str(dst)],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not dst.exists() or dst.stat().st_size <= 44: return None
    return dst

def crt_render(job, fmt, fps):
    d = crt_job_dir(job)
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
    stem = "amfm-crt-" + time.strftime("%Y%m%d-%H%M%S")
    src = d / "source.bin"
    snd = d / "sound.wav"     # the page's CRT sound render (11-05) wins over the video's own sound
    if snd.exists(): src = snd
    audio_in, amap = (["-i", str(src)], ["-map", "1:a:0?"]) if src.exists() else ([], [])
    try:
        if fmt == "png":   # frames straight out, no sound
            dst = CRT_DELIV / (stem + "_png"); dst.mkdir(exist_ok=True)
            cmd = [FFMPEG, "-v", "error", "-y", *frames_in, str(dst / "frame_%04d.png")]
        elif fmt == "gif":
            dst = CRT_DELIV / (stem + ".gif")
            cmd = [FFMPEG, "-v", "error", "-y", *frames_in, "-vf",
                   "fps=15,scale=960:-2:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4", str(dst)]
        else:
            # one pass: the frames + the video's own sound (padded with silence so the picture is never cut short).
            # MP4 from the Mac's encoder is already H.264 — it is only wrapped, not encoded again.
            pad = ["-af", "apad", "-shortest"] if amap else []
            if fmt == "mp4":
                dst = CRT_DELIV / (stem + ".mp4")
                v = ["-c:v", "copy"] if stream.exists() else ["-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p"]
                tail = [*v, "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart"]
            elif fmt == "prores":
                dst = CRT_DELIV / (stem + ".mov")
                tail = ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-c:a", "pcm_s16le"]
            elif fmt == "webm":
                dst = CRT_DELIV / (stem + ".webm")
                tail = ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "18", "-row-mt", "1", "-deadline", "good", "-cpu-used", "4", "-c:a", "libopus", "-b:a", "192k"]
            else:
                raise ValueError("unknown format " + fmt)
            cmd = [FFMPEG, "-v", "error", "-y", *frames_in, *audio_in, "-map", "0:v", *amap, *tail, *pad, str(dst)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0: raise RuntimeError((r.stderr.strip() or "ffmpeg failed") + f" · the frames are kept in {d}")
        shutil.rmtree(d, ignore_errors=True)   # done: the job folder was only a staging area for this one file
        return dst
    except Exception:
        raise   # on failure the upload stays, so a retry doesn't mean stepping the whole video again


def list_models():
    MODELS.mkdir(parents=True, exist_ok=True)
    out = []
    paths = [p for d in models_lib.model_dirs() for p in d.glob("*.pkl")
             if not p.name.startswith("._") and p.stat().st_size > 3_000_000]   # ._ = exFAT sidecars on the T7
    try:
        return models_lib.describe(paths)
    except Exception:
        return [{"name": p.stem, "path": str(p), "size": p.stat().st_size, "group": "other"} for p in sorted(paths)]


def list_outputs():
    outs, seen = [], set()
    for d in (OUT, ROOT):
        for p in d.glob("*.mp4"):
            fr = p.parent / (p.stem + "_frames"); seen.add(p.stat().st_ino)
            arm = next((a for a in ("dream", "collage", "cut bank") if (DELIV / a / p.name).exists() and same_render(DELIV / a / p.name, p)), None) \
                  or ("cut bank" if p.name.startswith("bank_") else "collage" if p.name.startswith("collage_") else "dream")
            dl = DELIV / arm / p.name
            outs.append({"name": p.name, "path": str(p), "deliv": str(dl) if dl.exists() else "", "arm": arm, "size": p.stat().st_size, "mtime": p.stat().st_mtime,
                         "frames": len(list(fr.glob("f_*.png"))) if fr.exists() else 0})
    if DELIV.exists():
        for p in DELIV.rglob("*"):
            if not p.is_file() or p.name.startswith(".") or p.suffix.lower() not in (".mp4", ".mov", ".webm", ".gif") or p.stat().st_ino in seen: continue
            outs.append({"name": p.name, "path": str(p), "deliv": str(p), "arm": p.parent.name, "size": p.stat().st_size, "mtime": p.stat().st_mtime, "frames": 0})
    outs.sort(key=lambda o: o["mtime"], reverse=True)
    return outs[:60]


def frames_dir_for(out_path):
    p = Path(out_path)
    return p.parent / (p.stem + "_frames")


def count_frames(out_path):
    fd = frames_dir_for(out_path)
    return len(list(fd.glob("f_*.png"))) if fd.exists() else 0


def latest_frame(out_path):
    fd = frames_dir_for(out_path)
    if not fd.exists(): return None
    fs = sorted(fd.glob("f_*.png"))
    return fs[-1] if fs else None


def run_job(j):
    """Spawn the engine (or the bank), follow its stdout, keep j updated."""
    if j.get("kind") == "latent":
        cmd = [str(PY), "-u", str(LATENT), "--pkl", j["model"], "--audio", j["audio"], "--out", j["out"],
               "--start", str(j["start"]), "--fps", str(j["fps"]), "--size", str(j["size"]), "--mode", j["engine_mode"],
               "--bend", str(j["bend"]), "--spin", str(j.get("spin", 0)), "--decay", str(j["decay"]), "--decay_shape", str(j["decay_shape"]),
               "--flash", str(j["flash"]), "--speed_fpm", str(j["speed_fpm"]), "--pulse_react", str(j["pulse_react"]),
               "--motion_react", str(j["motion_react"]), "--psi", str(j["psi"]), "--seed", str(j["seed"])]
        if j["duration"] > 0: cmd += ["--duration", str(j["duration"])]
        if j.get("draft"): cmd.append("--draft")
    elif j.get("kind") == "collage":
        cmd = [str(PY), "-u", str(COLLAGE), "--audio", j["audio"], "--out", j["out"], "--technique", j["technique"],
               "--start", str(j["start"]), "--width", str(j["width"]), "--height", str(j["height"]), "--fps", str(j["fps"]),
               "--hold", str(j["hold"]), "--steps", str(j["steps"]), "--lens", str(j["lens"]), "--seed", str(j["seed"])]
        if j["duration"] > 0: cmd += ["--duration", str(j["duration"])]
        if j.get("printscan"): cmd.append("--print")
        if j.get("label"): cmd += ["--label", j["label"]]
        for c in j["clips"]: cmd += ["--sources", c]
    elif j.get("kind") == "bank":
        cmd = [str(PY), "-u", str(BANK), "--audio", j["audio"], "--out", j["out"], "--start", str(j["start"]),
               "--duration", str(j["duration"]), "--width", str(j["width"]), "--height", str(j["height"]), "--fps", str(j["fps"]),
               "--min_shot", str(j["min_shot"]), "--max_shot", str(j["max_shot"]), "--effects", j["effects"],
               "--effects_per_shot", j["effects_per_shot"], "--plain_chance", str(j["plain_chance"]), "--strobe_hz", str(j["strobe_hz"]),
               "--max_clips", str(j["max_clips"]), "--seed", str(j["seed"])]
        for c in j["clips"]: cmd += ["--clips", c]
    else:
        cmd = [str(PY), "-u", str(ENGINE), "--out", j["out"],
           "--start", str(j["start"]), "--duration", str(j["duration"]), "--fps", str(j["fps"]),
           "--interp_fps", str(j["interp_fps"]), "--steps", str(j["steps"]), "--prompt", j["prompt"],
           "--negative", j["negative"], "--cfg", str(j["cfg"]), "--anchor", str(j["anchor"]),
           "--base_strength", str(j["base_strength"]), "--sat", str(j["sat"]), "--seed", str(j["seed"]),
           "--loud_react", str(j["loud_react"]), "--pulse_react", str(j["pulse_react"]),
           "--zoom_base", str(j["zoom_base"]), "--zoom_react", str(j["zoom_react"]),
           "--width", str(j["width"]), "--height", str(j["height"])]
        if j.get("video"): cmd += ["--video", j["video"], "--footage_mix", str(j["footage_mix"])]
        if j.get("image"): cmd += ["--image", j["image"]]
        if j.get("audio"): cmd += ["--audio", j["audio"]]
        if j["fp16"]: cmd.append("--fp16")
    j["cmd"] = " ".join(cmd)
    j["stage"] = "starting"
    j["log"] = []
    try:
        env = dict(os.environ)
        if (HUB / "models--stable-diffusion-v1-5--stable-diffusion-v1-5").exists():
            env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DIFFUSERS_OFFLINE="1")  # cached → never phone home
        else:
            for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "DIFFUSERS_OFFLINE"):
                env.pop(k, None)   # the .app launcher exports these; a missing model must be allowed to download once
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1, env=env, start_new_session=True)
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
            try: j["deliv"] = str(deliver(j["out"], j.get("kind")) or "")
            except Exception as e: j["log"].append(f"deliver: {e}")


def _follow(j, proc):
    for line in proc.stdout:
        line = line.rstrip()
        if not line: continue
        j["log"].append(line); j["log"] = j["log"][-80:]
        m = re.search(r"\[audio\] (\d+) frames", line) or re.search(r"\[cuts\] (\d+) shots", line) or re.search(r"frames=(\d+)", line)
        if m: j["frames_total"] = int(m.group(1))
        m = re.search(r"^\s*shot (\d+)/(\d+)", line)
        if m: j["shots_done"] = int(m.group(1)); j["frames_total"] = int(m.group(2))
        if "[model]" in line: j["stage"] = "loading model"
        elif "[audio]" in line: j["stage"] = "reading the song"
        elif "[resume]" in line: j["stage"] = "resuming"
        elif "[video]" in line: j["stage"] = "reading the footage"
        elif "[cuts]" in line or "[clips]" in line: j["stage"] = "cutting"
        elif "[sources]" in line: j["stage"] = "gathering pictures"
        elif "[sort]" in line: j["stage"] = "sorting by colour"
        elif "[render]" in line or line.strip().startswith("frame ") or line.strip().startswith("shot "): j["stage"] = "rendering"
        elif "[encode]" in line: j["stage"] = "encoding"
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


MAX_PIXELS = 900_000       # measured: 1024x768 peaks ~11.6 GB on an 18 GB M3 Pro


def _dim(v):
    """Round a requested dimension to a multiple of 8, with a floor."""
    try: n = int(float(v))
    except Exception: n = 0
    return max(64, (n // 8) * 8)


def start_job(req):
    global job
    with lock:
        if job and job.get("running"): return {"error": "a render is already running"}
        image, audio, video = req.get("image"), req.get("audio"), req.get("video")
        if video and not Path(video).is_file(): return {"error": "can't find that footage"}
        if not video and (not image or not Path(image).is_file()): return {"error": "pick an image or footage"}
        if audio and not Path(audio).is_file(): return {"error": "can't find that song"}
        if not video and not audio: return {"error": "pick a song"}
        adv0 = req.get("advanced") or {}
        w0, h0 = _dim(adv0.get("width", 512)), _dim(adv0.get("height", 384))
        if w0 * h0 > MAX_PIXELS:
            import math as _m
            k = _m.sqrt(MAX_PIXELS / (w0 * h0))
            return {"error": f"{w0}x{h0} is too big for this machine — try "
                             f"{max(64, int(w0*k)//8*8)}x{max(64, int(h0*k)//8*8)} or smaller."}
        if video: image = None
        c = int(req.get("coherence", 3)); cfg, anc, stre = COHERENCE.get(c, COHERENCE[3])
        length = req.get("length", "test"); start = float(req.get("start", 0) or 0)
        song = ffprobe_duration(audio or video)
        if length == "full": duration, start = 0, 0
        else: duration = float(req.get("duration", 15) or 15)
        subject = (req.get("subject") or "").strip()
        prompt = (subject + STYLE_TAIL) if subject else "dreamlike abstract morphing forms, painterly, film grain, muted tones"
        name = re.sub(r"[^a-z0-9]+", "-", (req.get("name") or "").lower()).strip("-") or time.strftime("dream_%m%d_%H%M%S")
        if (OUT / f"{name}.mp4").exists(): name = unique_out(name)      # never overwrite a finished render
        out = str(OUT / f"{name}.mp4")
        adv = req.get("advanced") or {}
        def f(k, d): 
            try: return float(adv[k]) if adv.get(k) not in (None, "") else d
            except Exception: return d
        j = {
            "name": name, "kind": "dream", "image": image, "video": video, "audio": audio, "out": out, "start": start, "duration": duration,
            "footage_mix": f("footage_mix", 0.55),
            "song_duration": song, "length": length, "coherence": c, "subject": subject, "prompt": prompt, "negative": NEG,
            "cfg": f("cfg", cfg), "anchor": f("anchor", anc), "base_strength": f("base_strength", stre), "sat": f("sat", 0.94),
            "steps": int(f("steps", 10)), "fps": int(f("fps", 6)), "interp_fps": int(f("interp_fps", 24)), "seed": int(f("seed", 0)),
            "loud_react": f("loud_react", 0.18), "pulse_react": f("pulse_react", 0.12), "zoom_base": f("zoom_base", 0.008), "zoom_react": f("zoom_react", 0.020),
            "width": _dim(f("width", 512)), "height": _dim(f("height", 384)), "fp16": bool(adv.get("fp16", True)),
            "running": True, "started": time.time(), "frames_total": 0, "stage": "queued", "log": [],
        }
        est_frames = int((duration if duration > 0 else song) * j["fps"])
        j["frames_total"] = est_frames
        job = j
        threading.Thread(target=run_job, args=(j,), daemon=True).start()
        return {"ok": True, "job": public(j)}


def start_bank(req):
    global job
    with lock:
        if job and job.get("running"): return {"error": "a render is already running"}
        clips = [c for c in (req.get("clips") or []) if c and Path(c).is_dir()]
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
             "min_shot": f("min_shot", 0.125), "max_shot": f("max_shot", 0.8), "effects": req["effects"] if isinstance(req.get("effects"), str) and req.get("effects") is not None else "negate,chroma,smear,trails,twocolor,dither,pixel,scanlines,mirror,contrast,hue,jitter,strobe,fast,posterize",
             "effects_per_shot": req.get("effects_per_shot") or "1-2", "plain_chance": f("plain_chance", 0.15), "strobe_hz": f("strobe_hz", 3.0),
             "max_clips": int(f("max_clips", 600)), "seed": int(f("seed", 0)),
             "running": True, "started": time.time(), "frames_total": 0, "stage": "queued", "log": [], "fp16": False}
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
             "flash": max(0.0, min(1.0, f("flash", 0.2))), "speed_fpm": f("speed_fpm", 12), "pulse_react": f("pulse_react", 0.5),
             "motion_react": f("motion_react", 0.5), "psi": f("psi", 0.8), "seed": int(f("seed", 1)), "draft": bool(req.get("draft")),
             "running": True, "started": time.time(), "frames_total": int((duration or song) * int(f("fps", 24)) / (2 if req.get("draft") else 1)),
             "stage": "queued", "log": [], "fp16": False}
        job = j
        threading.Thread(target=run_job, args=(j,), daemon=True).start()
        return {"ok": True, "job": public(j)}


def start_collage(req):
    global job
    with lock:
        if job and job.get("running"): return {"error": "a render is already running"}
        clips = [c for c in (req.get("clips") or []) if c and Path(c).exists()]
        if not clips: return {"error": "pick at least one folder of pictures or clips"}
        audio = req.get("audio")
        if not audio or not Path(audio).is_file(): return {"error": "pick a song"}
        def f(k, d):
            try: return float(req[k]) if req.get(k) not in (None, "") else d
            except Exception: return d
        song = ffprobe_duration(audio); full = req.get("length") == "full"
        start, duration = (0.0, 0.0) if full else (max(0.0, f("start", 0)), f("duration", 15))
        if not full and song and start >= song - 1:
            return {"error": f"the song is only {int(song // 60)}:{int(song % 60):02d} long — start earlier than that"}
        tech = req.get("technique") if req.get("technique") in ("colour", "stepin", "archive") else "colour"
        hold = req.get("hold") if str(req.get("hold")) in ("1", "2", "3", "4", "auto") else "2"
        w, h = {"16:9": (1280, 720), "16:9 hd": (1920, 1080), "4:3": (960, 720), "square": (1080, 1080), "vertical": (1080, 1920)}.get(req.get("frame"), (1280, 720))
        name = unique_out(re.sub(r"[^a-z0-9]+", "-", (req.get("name") or "").lower()).strip("-") or time.strftime(f"collage_{tech}_%m%d_%H%M%S"))
        fps = max(6, min(60, int(f("fps", 24)))); steps = max(2, min(8, int(f("steps", 4))))
        avg = 2 if hold == "auto" else int(hold)
        j = {"name": name, "kind": "collage", "clips": clips, "audio": audio, "out": str(OUT / f"{name}.mp4"), "technique": tech,
             "start": start, "duration": duration, "song_duration": song, "length": "full" if full else "test",
             "width": w, "height": h, "fps": fps, "hold": hold, "steps": steps, "lens": max(0.0, min(1.0, f("lens", 0.5))),
             "printscan": bool(req.get("printscan")), "label": (req.get("label") or "")[:60], "seed": int(f("seed", 0)),
             "running": True, "started": time.time(), "frames_total": max(1, int((duration or song) * fps / avg / (steps if tech == "stepin" else 1))),
             "stage": "queued", "log": [], "fp16": False}
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
    else:
        done = count_frames(j["out"])
    total = j.get("frames_total") or 0
    if j.get("kind") == "latent" and not j.get("running") and j.get("stage") == "done" and Path(j["out"]).exists():
        done = total                      # latent_dream clears its frame folder once the mp4 is written
    elapsed = (j.get("ended") or time.time()) - j["started"]
    rate = None
    if done > j.get("frames_at_start", 0) and j.get("render_t0"):
        rate = (time.time() - j["render_t0"]) / max(1, done - j["frames_at_start"])
    if j.get("stage") == "rendering" and not j.get("render_t0"):
        j["render_t0"] = time.time(); j["frames_at_start"] = done
    eta = (total - done) * rate if (rate and total) else None
    lf = latest_frame(j["out"])
    return {k: j[k] for k in ("name", "kind", "image", "video", "footage_mix", "clips", "audio", "out", "start", "duration", "song_duration", "length", "coherence", "subject",
                              "prompt", "cfg", "anchor", "base_strength", "sat", "steps", "fps", "interp_fps", "seed", "running",
                              "started", "stage", "log", "fp16", "width", "height", "min_shot", "max_shot", "effects", "effects_per_shot", "plain_chance", "strobe_hz", "model", "bend", "spin", "decay", "technique", "hold", "steps", "lens", "printscan", "label", "decay_shape", "flash", "speed_fpm", "pulse_react", "motion_react", "psi", "size", "engine_mode", "deliv", "draft") if k in j} | {
        "frames_done": done, "frames_total": total, "elapsed": elapsed, "eta": eta, "sec_per_frame": rate,
        "latest_frame": str(lf) if lf else None, "latest_index": int(lf.stem.split("_")[1]) if lf else None,
        "error": j.get("error"), "rc": j.get("rc"), "ended": j.get("ended"),
        "output_ready": Path(j["out"]).exists() and not j.get("running") and Path(j["out"]).stat().st_mtime >= j["started"] - 1,
    }


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send_json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)

    def send_file(self, path, ctype=None):
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
        self.send_header("Accept-Ranges", "bytes"); self.send_header("Content-Type", ctype)
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

    def do_GET(self):
        if self.bad_host(): return
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        if u.path in ("/", "/index.html"): return self.send_file(APP / "index.html")
        if u.path == "/favicon.ico": self.send_response(204); self.end_headers(); return
        if u.path == "/models": return self.send_file(APP / "models.html")
        if u.path == "/crt": return self.send_file(APP / "crt.html")
        if u.path == "/crt/crt_sound.js": return self.send_file(APP / "crt_sound.js", "text/javascript; charset=utf-8")
        if u.path.startswith("/crt/logos/"):
            return self.send_file(APP / "crt_logos" / Path(urllib.parse.unquote(u.path)).name)
        if u.path == "/api/models/state": return self.send_json(models_lib.state())
        if u.path == "/update.js": return self.send_file(APP / "update.js", "text/javascript; charset=utf-8")
        if u.path == "/api/update": return self.send_json(updater.state())
        if u.path == "/api/state":
            imgs, auds, vids = list_media()
            def lane_name(l):
                if l in LANE_NAMES: return LANE_NAMES[l]
                pp = Path(l)
                if pp.name == "FOOTAGE": return "FOOTAGE (all)"
                if pp.name == "clips": return pp.parent.name + " clips"
                return pp.name
            lanes = [{"name": lane_name(l), "path": l} for l in LANES if Path(l).is_dir()]
            DROPPED.mkdir(parents=True, exist_ok=True)
            dropped = [{"name": f.name, "path": str(f), "kind": "video" if f.suffix.lower() in VID_EXT else "image", "size": f.stat().st_size}
                       for f in sorted(DROPPED.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True) if f.is_file() and not f.name.startswith(".")]
            return self.send_json({"job": public(job), "ai": ai_ready(), "model": MODEL_CACHED, "images": imgs, "audio": auds, "videos": vids, "lanes": lanes, "models": list_models(), "models_dir": str(MODELS), "dropped": dropped, "dropped_dir": str(DROPPED), "outputs": list_outputs(),
                                   "root": str(ROOT), "engine_ok": LATENT.exists() and PY.exists()})
        if u.path == "/api/file":
            p = self.safe_path(q.get("p", [""])[0])
            return self.send_file(p) if p else self.send_error(403)
        if u.path == "/api/frame":
            # latest frame of the current job, or frame n of an output
            if job and q.get("latest"):
                lf = latest_frame(job["out"]); return self.send_file(lf) if lf else self.send_error(404)
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
                fs = sorted(frames_dir_for(out).glob("f_*.png"))
                if fs: return self.send_file(fs[min(n, len(fs) - 1)])
            # no frame folder (a cut bank, a delivered render, or an old one): pull a midpoint frame from the video once
            if in_work: th = out.parent / (out.stem + "_thumb.jpg")
            else:
                td = OUT / "_thumbs"; td.mkdir(parents=True, exist_ok=True)
                th = td / (out.parent.name + "--" + out.stem + "_thumb.jpg")
            if not th.exists() and out.exists():
                d = ffprobe_duration(out)
                subprocess.run([FFMPEG or "ffmpeg", "-y", "-v", "error", "-ss", f"{d/2:.2f}", "-i", str(out), "-frames:v", "1", "-vf", "scale=480:-2", str(th)], capture_output=True)
            return self.send_file(th) if th.exists() else self.send_error(404)
        if u.path in ("/api/reveal", "/api/open"):
            # These hand a path to macOS `open`, which will execute an app or an
            # installer. GET made them reachable from any page via an <img> tag.
            return self.send_json({"error": "use POST"}, code=405)
        self.send_error(404)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        if self.bad_host() or self.deny_cross_site(): return
        n = int(self.headers.get("Content-Length") or 0)
        if u.path == "/api/upload" and n > MAX_UPLOAD:
            return self.send_json({"error": f"file is larger than {MAX_UPLOAD // (1024*1024)} MB"}, code=413)
        body = self.rfile.read(n) if n else b""
        if u.path == "/api/renders_folder":
            subprocess.Popen(["open", str(DELIV)]); return self.send_json({"ok": True, "path": str(DELIV)})
        if u.path == "/api/update/check":
            return self.send_json(updater.check(force=True))
        if u.path == "/api/update/apply":
            def busy():
                if job and job.get("running"): return "a render is running; update when it's done"
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
        if u.path == "/api/start":
            return self.send_json({"error": "HALLUCINATE was retired — use LATENT DREAM"}, 410)
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
            if not re.match(r"^\d{1,6}(/\d{1,5})?$", fps): return self.send_json({"error": "bad fps"}, 400)
            try: out = crt_render(q.get("job", [""])[0], q.get("fmt", ["mp4"])[0], fps)
            except Exception as e: return self.send_json({"error": str(e)[-400:]}, 500)
            return self.send_json({"ok": True, "path": str(out)})
        if u.path == "/api/crt_export":
            kind = q.get("kind", ["video"])[0]; fmt = q.get("fmt", ["mp4"])[0]
            if not body: return self.send_json({"error": "nothing was recorded"}, 400)
            try: out = crt_export(body, kind, fmt, q.get("name", [None])[0], q.get("folder", [None])[0])
            except Exception as e: return self.send_json({"error": str(e)[-400:]}, 500)
            return self.send_json({"ok": True, "path": str(out)})
        if u.path == "/api/latent":
            try: req = json.loads(body or b"{}")
            except Exception: return self.send_json({"error": "bad request"}, 400)
            return self.send_json(start_latent(req))
        if u.path == "/api/collage":
            try: req = json.loads(body or b"{}")
            except Exception: return self.send_json({"error": "bad request"}, 400)
            return self.send_json(start_collage(req))
        if u.path == "/api/bank":
            try: req = json.loads(body or b"{}")
            except Exception: return self.send_json({"error": "bad request"}, 400)
            return self.send_json(start_bank(req))
        if u.path == "/api/remove_outputs":
            try: req = json.loads(body or b"{}")
            except Exception: req = {}
            stamp = time.strftime("%Y%m%d-%H%M%S"); cu = APP / "_cleanup" / f"library-{stamp}"; moved = []
            for raw in req.get("paths") or []:
                out = self.safe_path(raw)
                if not out or out.suffix.lower() not in (".mp4", ".mov", ".webm", ".gif") or not out.exists(): continue
                if job and job.get("running") and job.get("out") == str(out): continue   # never pull a render mid-flight
                cu.mkdir(parents=True, exist_ok=True)
                # a render can live twice: the engine's copy (app/out or ROOT) and its hard link in
                # ~/Movies/Liminal Engine/<arm>/. Removing one must take the twin too, or it reappears.
                twins = [out]
                for cand in [OUT / out.name, ROOT / out.name, *DELIV.glob("*/" + out.name)]:
                    if cand.exists() and cand not in twins and same_render(cand, out): twins.append(cand)
                for i, t in enumerate(twins):
                    dst = cu / (t.name if i == 0 else f"{t.stem} ({t.parent.name}){t.suffix}")
                    shutil.move(str(t), str(dst))
                    for suf in ("_frames", "_src", "_shots", "_thumb.jpg"):
                        side = t.parent / (t.stem + suf)
                        if side.exists(): shutil.move(str(side), str(cu / side.name))
                moved.append(out.name)
            return self.send_json({"ok": True, "moved": moved, "to": str(cu) if moved else None})
        if u.path == "/api/dropped_remove":
            try: req = json.loads(body or b"{}")
            except Exception: req = {}
            pth = self.safe_path(req.get("path", ""))
            if pth and pth.parent == DROPPED and pth.exists():
                cu = APP / "_cleanup" / "dropped"; cu.mkdir(parents=True, exist_ok=True)
                shutil.move(str(pth), str(cu / f"{pth.stem}-{int(time.time())}{pth.suffix}"))
            return self.send_json({"ok": True})
        if u.path == "/api/check":
            try: req = json.loads(body or b"{}")
            except Exception: req = {}
            pth = self.safe_path(req.get("path", ""))
            if pth is None:
                return self.send_json({"error": "that path is outside the folders this app can use"})
            if pth.is_file():
                e = pth.suffix.lower(); kind = "model" if e == ".pkl" else "video" if e in VID_EXT else "image" if e in IMG_EXT else "audio" if e in AUD_EXT else "file"
                return self.send_json({"ok": True, "kind": kind, "path": str(pth), "name": pth.name, "duration": ffprobe_duration(pth) if kind != "image" else 0})
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
        if u.path == "/api/clear_frames":
            # start a name over: move its frames to _cleanup (never delete)
            try: req = json.loads(body or b"{}")
            except Exception: req = {}
            out = self.safe_path(req.get("out", ""))
            if out and (out.suffix.lower() != ".mp4" or ROOT not in out.parents):
                return self.send_json({"error": "that is not one of this app's renders"})
            if out:
                cu = APP / "_cleanup"; cu.mkdir(exist_ok=True)
                for suf in ("_frames", "_src", "_shots"):
                    fd = out.parent / (out.stem + suf)
                    if fd.exists(): shutil.move(str(fd), str(cu / f"{fd.name}-{int(time.time())}"))
                if out.exists():
                    for tw in DELIV.glob("*/" + out.name):
                        if same_render(tw, out): shutil.move(str(tw), str(cu / f"{out.stem}-{int(time.time())} ({tw.parent.name}).mp4"))
                    shutil.move(str(out), str(cu / f"{out.stem}-{int(time.time())}.mp4"))
            return self.send_json({"ok": True})
        self.send_error(404)


updater.check_soon()   # no-op in Eric's copy (no app/channel.json)

if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    print(f"LIMINAL ENGINE  →  http://localhost:{PORT}   (leave this window open; close it to stop)")
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
