#!/usr/bin/env python3
"""CUT BANK — beat-cut a folder of footage/stills into one sequence.
Reads the song's onsets, cuts on every hit, picks a random moment of a random clip for every shot, concatenates,
muxes the song. Some shots get an editing move (fast / slow / reverse). Optionally the finished edit is recorded
onto a modelled analogue tape (era_texture.py: VHS SP/EP, U-matic, Betacam SP; generation; damage; faults that the
song's hits trigger), as one continuous recording, the way a tape is.
ffmpeg + librosa (numpy, scipy) only. Progress lines: "shot i/N", then "[tape] i/N frames".
"""
import re, argparse, json, os, random, shutil, subprocess, sys, time
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


MOVES = {   # editing moves, not looks: a shot plays faster, slower or backwards (picture only; the song is untouched)
    "fast": lambda: f"setpts={random.choice([0.5, 0.33, 0.25])}*PTS",      # 2x / 3x / 4x speed
    "slow": lambda: f"setpts={random.choice([2, 3])}*PTS",                  # 1/2 or 1/3 speed
    "reverse": lambda: "reverse",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clips", action="append", required=True, help="folder(s) of footage and/or stills (recursive), or single files")
    p.add_argument("--audio", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--min_shot", type=float, default=0.125, help="shortest shot, seconds (8 cuts/sec max at 0.125)")
    p.add_argument("--max_shot", type=float, default=0.8)
    p.add_argument("--moves", default="fast,slow,reverse", help="editing moves a shot may get: fast, slow, reverse (empty = none)")
    p.add_argument("--move_chance", type=float, default=0.3, help="fraction of shots that get one move")
    p.add_argument("--tape", default="off", help="record the finished edit onto tape: off, vhs-sp, vhs-ep, umatic, betacam-sp")
    p.add_argument("--gen", type=int, default=1, help="tape generation: 1 = original, 2+ = analogue dubs")
    p.add_argument("--damage", type=float, default=0.3, help="tape damage 0..1: how many faults, how big")
    p.add_argument("--faults", default="dropout,tracking,clog,skew,wobble,ckill", help="faults that can happen")
    p.add_argument("--react", type=int, default=1, help="1 = the song's hits trigger the faults, 0 = they fall at random")
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
    # every shot a whole number of frames on one shared frame grid: per-shot rounding otherwise adds up, the edit
    # comes out short of the song and the cuts slide off the beat
    F = a.fps; grid = []
    for t0, d in shots:
        f0, f1 = round(t0 * F), round((t0 + d) * F)
        if f1 > f0: grid.append((f0 / F, (f1 - f0) / F, f1 - f0))
    if grid: grid[-1] = (grid[-1][0], (round(total * F) - round(grid[-1][0] * F)) / F, round(total * F) - round(grid[-1][0] * F))
    shots = [g for g in grid if g[2] > 0]
    print(f"[cuts] {len(shots)} shots over {total:.1f}s", flush=True)

    print("[clips] scanning...", flush=True)
    files = []
    for d in a.clips:
        if os.path.isfile(d):                     # a single clip (dropped clips with some left out)
            if Path(d).suffix.lower() in VID | IMG: files.append(Path(d))
            continue
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
    moves = [n.strip() for n in a.moves.split(",") if n.strip() in MOVES]
    W, H, F = a.width, a.height, a.fps
    base = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={F},format=yuv420p"
    HOLD = "tpad=stop_mode=clone:stop_duration=2"    # a source that runs out early holds its last frame: never a short shot
    listing = []
    for i, (t0, d, nf) in enumerate(shots):
        use_still = stills and (not vids or random.random() < len(stills) / (len(stills) + 4 * len(vids)))
        mv = random.choice(moves) if moves and random.random() < a.move_chance else None
        if use_still and mv: mv = None                                  # a still has no time to speed up or reverse
        f = MOVES[mv]() if mv else ""
        vf = ",".join(([f] if f.startswith("setpts") else []) + [base] + ([f] if f == "reverse" else []) + [HOLD])
        dst = work / f"s_{i:05d}.mp4"
        if use_still:
            src = random.choice(stills)
            cmd = [FFMPEG, "-y", "-v", "error", "-loop", "1", "-t", f"{d + 0.1:.3f}", "-i", str(src), "-vf", vf, "-frames:v", str(nf), "-an",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(dst)]
        else:
            src = random.choice(vids)
            # how much source a shot eats: setpts=k*PTS plays k× slower, so it needs d/k seconds (fast 0.25 → 4×d)
            k = 1.0
            for m in re.findall(r"setpts=([0-9.]+)\*PTS", vf): k *= float(m)
            need = d / k + 0.2
            inp = random.uniform(0, max(0.0, durs[src] - need))
            cmd = [FFMPEG, "-y", "-v", "error", "-ss", f"{inp:.3f}", "-t", f"{need:.3f}", "-i", str(src), "-vf", vf, "-frames:v", str(nf), "-an",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(dst)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            print(f"  [warn] shot {i+1} timed out", flush=True); r = None
        if r is None or r.returncode != 0 or not dst.exists():
            # fall back to a plain shot, then to black, so every cut keeps its length and stays on the beat
            cmd[cmd.index("-vf") + 1] = base + "," + HOLD
            try: subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            except subprocess.TimeoutExpired: pass
            if not dst.exists():
                subprocess.run([FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", f"color=black:s={W}x{H}:r={F}:d={d + 0.1:.3f}", "-frames:v", str(nf),
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
    dst = work / "_edit.mp4" if a.tape != "off" else out
    r = subprocess.run([FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(seg),
                        "-c:v", "libx264", "-preset", "fast", "-crf", "16" if a.tape != "off" else "18", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", str(dst)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not dst.exists():
        print("[error] join failed:", r.stderr.strip()[-400:], flush=True); sys.exit(2)
    if a.tape != "off":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import era_texture
        if a.tape not in era_texture.FORMATS: sys.exit(f"[error] unknown tape format '{a.tape}'")
        print(f"[tape] recording onto {era_texture.FORMATS[a.tape]['name']}, generation {a.gen}, damage {a.damage:.2f}"
              f"{', faults on the hits' if a.react else ''}", flush=True)
        def tick(i, n):
            if i % 24 == 0: print(f"  [tape] {i}/{n} frames", flush=True)
        try:
            _, ev = era_texture.process(str(dst), str(out), fmt=a.tape, gen=a.gen, damage=a.damage, faults=a.faults.split(","),
                                        seed=a.seed or 3, react=a.audio if a.react else None, start=a.start, duration=a.duration,
                                        crf=18, progress=tick)
            print("[tape] faults: " + (", ".join(f"{v} {k}" for k, v in ev.items() if v) or "none"), flush=True)
        except Exception as e:
            print(f"[error] tape pass failed: {e}", flush=True); sys.exit(2)
    json.dump({"shots": len(listing), "moves": moves, "move_chance": a.move_chance, "tape": a.tape, "gen": a.gen, "damage": a.damage, "faults": a.faults, "react": a.react,
               "clips": [str(v) for v in vids[:50]]}, open(work / "recipe.json", "w"), indent=1)
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
