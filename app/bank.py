#!/usr/bin/env python3
"""FREAKOUT CUT BANK — beat-cut a folder of footage/stills into one abstract sequence.
Reads the song's onsets, cuts on every hit, gives every shot a random abstraction, concatenates,
muxes the song. ffmpeg + librosa only. Progress lines: "shot i/N".
"""
import argparse, json, math, os, random, shutil, subprocess, sys, time
from pathlib import Path

# --- ffmpeg/ffprobe discovery ------------------------------------------------
# A GUI-launched .app inherits a minimal PATH with no homebrew, so a bare
# "ffmpeg" silently fails. Put the usual install locations back, then resolve
# the binaries once and say so plainly if they are genuinely missing.
for _p in ("/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin"):
    if os.path.isdir(_p) and _p not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")


def _need(tool):
    found = shutil.which(tool)
    if not found:
        sys.exit(f"[error] {tool} not found on PATH. Install it with:  brew install ffmpeg")
    return found


FFMPEG = _need("ffmpeg")
FFPROBE = _need("ffprobe")

VID = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".mts", ".flv", ".wmv"}
IMG = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

PALETTES = [("255,0,120", "10,10,10"), ("255,255,255", "0,0,0"), ("255,40,40", "0,0,0"), ("240,230,0", "20,0,60"),
            ("0,255,180", "0,0,0"), ("255,255,255", "170,0,255"), ("255,120,0", "0,20,40")]


def probe(p):
    """Duration in seconds, or 0.0 if this particular file is unreadable."""
    try:
        r = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                           capture_output=True, text=True, timeout=20)
        return float(r.stdout.strip() or 0)
    except subprocess.TimeoutExpired:
        print(f"[warn] ffprobe timed out on {Path(p).name} — skipping", flush=True)
        return 0.0
    except Exception as e:
        print(f"[warn] could not read {Path(p).name}: {e}", flush=True)
        return 0.0


def two_color(pal):
    (r1, g1, b1), (r0, g0, b0) = [tuple(map(int, c.split(","))) for c in pal]
    t = random.randint(90, 170)
    return (f"format=gray,geq=r='if(gt(lum(X,Y),{t}),{r1},{r0})':g='if(gt(lum(X,Y),{t}),{g1},{g0})':b='if(gt(lum(X,Y),{t}),{b1},{b0})',format=yuv420p")


def effect_chain(names, fps, strobe_hz, W, H):
    """Return an ffmpeg -vf fragment (applied AFTER the shot is scaled to W×H)."""
    chain = []
    for n in names:
        if n == "negate": chain.append("negate")
        elif n == "chroma": chain.append(f"rgbashift=rh={random.randint(-24,24)}:bh={random.randint(-24,24)}:gv={random.randint(-8,8)}")
        elif n == "smear": chain.append(f"tmix=frames={random.randint(4,10)}")
        elif n == "trails": chain.append(f"lagfun=decay={random.choice([0.9,0.95,0.97])}")
        elif n == "twocolor": chain.append(two_color(random.choice(PALETTES)))
        elif n == "dither":
            k = random.choice([2, 3, 4])
            chain.append(f"format=gray,scale=iw/{k}:ih/{k},format=monob,scale={W}:{H}:flags=neighbor,format=yuv420p")
        elif n == "pixel": chain.append(f"pixelize=w={random.choice([8,12,16,24])}:h={random.choice([8,12,16,24])}")
        elif n == "scanlines": chain.append(f"drawgrid=w=iw:h={random.choice([2,3,4])}:t=1:c=black@0.7")
        elif n == "mirror": chain.append("crop=iw/2:ih:0:0,split[a][b];[b]hflip[c];[a][c]hstack")
        elif n == "contrast": chain.append(f"eq=contrast={random.choice([2.5,3.5,5])}:saturation={random.choice([0,0,2.5])}")
        elif n == "hue": chain.append(f"hue=h={random.randint(0,359)}:s={random.choice([1,2,3])}")
        elif n == "jitter":
            z = random.choice([0.55, 0.65, 0.8])
            chain.append(f"crop=iw*{z}:ih*{z}:(iw-ow)*random(1):(ih-oh)*random(1),scale={W}:{H}")
        elif n == "strobe":
            k = max(2, math.ceil(fps / strobe_hz))          # period in frames; never faster than strobe_hz
            chain.append(f"negate=enable='lt(mod(n,{k}),{k//2})'")
        elif n == "fast": chain.append(f"setpts={random.choice([0.5,0.33,0.25])}*PTS")
        elif n == "slow": chain.append(f"setpts={random.choice([2,3])}*PTS")
        elif n == "reverse": chain.append("reverse")
        elif n == "posterize": chain.append(f"lutyuv=y='if(gt(val,{random.randint(90,170)}),255,0)'")
    return chain


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clips", action="append", required=True, help="folder(s) of footage and/or stills (recursive)")
    p.add_argument("--audio", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--min_shot", type=float, default=0.125, help="shortest shot, seconds (8 cuts/sec max at 0.125)")
    p.add_argument("--max_shot", type=float, default=0.8)
    p.add_argument("--effects", default="negate,chroma,smear,trails,twocolor,dither,pixel,scanlines,mirror,contrast,hue,jitter,strobe,fast,posterize")
    p.add_argument("--effects_per_shot", default="1-2")
    p.add_argument("--plain_chance", type=float, default=0.15, help="fraction of shots left untouched")
    p.add_argument("--strobe_hz", type=float, default=3.0, help="hard cap on flash rate")
    p.add_argument("--max_clips", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    a.min_shot = max(0.04, a.min_shot); a.max_shot = max(a.min_shot, a.max_shot)   # a 0 s shot would never advance
    random.seed(a.seed or time.time_ns())

    import numpy as np, librosa
    print("[audio] reading onsets...", flush=True)
    y, sr = librosa.load(a.audio, sr=None, mono=True, offset=a.start, duration=a.duration if a.duration > 0 else None)
    total = librosa.get_duration(y=y, sr=sr)
    on = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=False)
    cuts = [0.0]
    for t in on:
        if t - cuts[-1] >= a.min_shot: cuts.append(float(t))
    # break long gaps
    shots = []
    for i, t0 in enumerate(cuts):
        t1 = cuts[i + 1] if i + 1 < len(cuts) else total
        while t1 - t0 > a.max_shot:
            d = random.uniform(a.min_shot, a.max_shot); shots.append((t0, d)); t0 += d
        if t1 - t0 >= 0.03: shots.append((t0, t1 - t0))
    print(f"[cuts] {len(shots)} shots over {total:.1f}s", flush=True)

    print("[clips] scanning...", flush=True)
    files = []
    for d in a.clips:
        for root, _, fs in os.walk(d):
            if "_cleanup" in root or "/." in root: continue
            for f in fs:
                if f.startswith("."): continue          # ._ AppleDouble sidecars on the T7: not media, and ffmpeg hangs on them
                e = Path(f).suffix.lower()
                if e in VID or e in IMG: files.append(Path(root) / f)
    random.shuffle(files); files = files[:a.max_clips]
    if not files: sys.exit("no footage found")
    vids = [f for f in files if f.suffix.lower() in VID]
    durs = {}
    for v in vids:
        durs[v] = probe(v)
    vids = [v for v in vids if durs[v] > 1.0]
    stills = [f for f in files if f.suffix.lower() in IMG]
    print(f"[clips] {len(vids)} clips, {len(stills)} stills", flush=True)
    if not vids and not stills:
        sys.exit(f"[error] none of the {len(files)} file(s) found could be read as video or stills. "
                 f"If they look fine in Finder this is usually a broken ffmpeg install.")

    out = Path(a.out).resolve(); work = out.parent / (out.stem + "_shots"); work.mkdir(parents=True, exist_ok=True)
    for old in work.glob("s_*.mp4"): old.unlink()
    names = [n.strip() for n in a.effects.split(",") if n.strip()]
    try:
        lo, _, hi = a.effects_per_shot.strip().partition("-")
        lo, hi = int(lo), int(hi or lo)
        if lo < 0 or hi < lo: raise ValueError
    except ValueError:
        sys.exit(f"[error] treatments-per-shot must be a number or a range like 1-2, not '{a.effects_per_shot}'")
    W, H, F = a.width, a.height, a.fps
    base = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={F},format=yuv420p"
    listing = []
    for i, (t0, d) in enumerate(shots):
        use_still = stills and (not vids or random.random() < len(stills) / (len(stills) + 4 * len(vids)))
        chain = [] if random.random() < a.plain_chance else effect_chain(random.sample(names, k=min(len(names), random.randint(lo, hi))), F, a.strobe_hz, W, H)
        vf = ",".join([base] + chain)
        dst = work / f"s_{i:05d}.mp4"
        if use_still:
            src = random.choice(stills)
            cmd = [FFMPEG, "-y", "-v", "error", "-loop", "1", "-t", f"{d:.3f}", "-i", str(src), "-vf", vf, "-t", f"{d:.3f}", "-an",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(dst)]
        else:
            src = random.choice(vids); need = d * (4 if "fast" in vf else 1) + 0.2
            inp = random.uniform(0, max(0.0, durs[src] - need))
            cmd = [FFMPEG, "-y", "-v", "error", "-ss", f"{inp:.3f}", "-t", f"{need:.3f}", "-i", str(src), "-vf", vf, "-t", f"{d:.3f}", "-an",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(dst)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            print(f"  [warn] shot {i+1} timed out", flush=True); r = None
        if r is None or r.returncode != 0 or not dst.exists():
            # fall back to a plain shot, then to black, so every cut keeps its length and stays on the beat
            cmd[cmd.index("-vf") + 1] = base
            try: subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            except subprocess.TimeoutExpired: pass
            if not dst.exists():
                subprocess.run([FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", f"color=black:s={W}x{H}:r={F}:d={d:.3f}",
                                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(dst)], capture_output=True, timeout=60)
        if dst.exists(): listing.append(dst)
        if i % 5 == 0 or i == len(shots) - 1:
            print(f"  shot {i+1}/{len(shots)}", flush=True)

    lst = work / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in listing))
    print("[encode] joining + muxing...", flush=True)
    seg = work / "_seg.m4a"
    tflag = ["-t", str(a.duration)] if a.duration > 0 else []
    subprocess.run([FFMPEG, "-y", "-v", "error", "-ss", str(a.start), *tflag, "-i", a.audio, "-vn", "-c:a", "aac", "-b:a", "192k", str(seg)])
    r = subprocess.run([FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(seg),
                        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        print("[error] join failed:", r.stderr.strip()[-400:], flush=True); sys.exit(2)
    json.dump({"shots": len(listing), "effects": names, "clips": [str(v) for v in vids[:50]]}, open(work / "recipe.json", "w"), indent=1)
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
