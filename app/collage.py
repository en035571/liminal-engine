#!/usr/bin/env python3
"""COLLAGE — moving-stills video collages for LIMINAL ENGINE (the three techniques from GSTV Screens III).

  colour   COLOUR RUNS   one centred subject per frame, sorted by colour so the cuts flow into each other;
                         soft swirl lens, dark corners. (after cetkhup2, "Garden")
  stepin   STEPPING IN   consecutive frames walk closer to one detail, then jump to the next picture.
                         (after cetkhup2's "SPILL" rescore)
  archive  THE ARCHIVE   each picture kept at its own size and shape on black; compression breaks and
                         dropped frames stay in. (after Truls Krane Meby, "The Weight of Sight")

Pictures come from folders of stills and/or clips (clips are sampled into stills). Each picture is held
for HOLD frames (2 = "on twos"); --hold auto holds longer in quiet parts and shorter in loud ones.

  python app/collage.py --sources DIR [--sources DIR2] --audio song.wav --technique colour --out out.mp4
"""
import argparse, os, random, re, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageDraw, ImageFont

for _p in ("/opt/homebrew/bin", "/usr/local/bin"):
    if os.path.isdir(_p) and _p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")
FF = shutil.which("ffmpeg") or "ffmpeg"
FP = shutil.which("ffprobe") or "ffprobe"
IMG = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
VID = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".mts"}


def gather(sources, want, rng, tmp):
    imgs, vids = [], []
    for s in sources:
        p = Path(s)
        it = [p] if p.is_file() else (Path(r) / f for r, _, fs in os.walk(p) if "_cleanup" not in r for f in fs)
        for f in it:
            if f.name.startswith("."): continue                 # ._ sidecars on the T7
            e = f.suffix.lower()
            if e in IMG: imgs.append(f)
            elif e in VID: vids.append(f)
    rng.shuffle(imgs); rng.shuffle(vids)
    # clips → a few stills each, until there are enough pictures
    need_from_vids = max(0, want - len(imgs))
    if vids and need_from_vids:
        per = max(1, min(6, -(-need_from_vids // len(vids))))
        print(f"[sources] {len(imgs)} stills, sampling {per} frames from up to {len(vids)} clips", flush=True)
        for k, v in enumerate(vids[:400]):
            try:
                d = float(subprocess.run([FP, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(v)],
                                         capture_output=True, text=True, timeout=20).stdout.strip() or 0)
            except Exception:
                d = 0
            if d < 1: continue
            for j in range(per):
                t = rng.uniform(0.2, max(0.3, d - 0.3)); out = tmp / f"v{k:04d}_{j}.jpg"
                subprocess.run([FF, "-v", "error", "-ss", f"{t:.2f}", "-i", str(v), "-frames:v", "1", "-q:v", "3", "-y", str(out)],
                               capture_output=True, timeout=30)
                if out.exists() and out.stat().st_size > 2000: imgs.append(out)
            if len(imgs) >= want * 1.3: break
    print(f"[sources] {len(imgs)} pictures", flush=True)
    return imgs


def load(p, maxside=1600):
    try:
        im = Image.open(p); im.draft("RGB", (maxside, maxside)); im = ImageOps.exif_transpose(im).convert("RGB")
        if max(im.size) > maxside: im.thumbnail((maxside, maxside), Image.LANCZOS)
        return im
    except Exception:
        return None


def cover(im, W, H, cx=0.5, cy=0.5, zoom=1.0):
    """Crop to fill W×H around (cx, cy), zoomed in by `zoom`."""
    iw, ih = im.size; s = max(W / iw, H / ih) * zoom
    cw, ch = W / s, H / s
    x0 = min(max(cx * iw - cw / 2, 0), iw - cw); y0 = min(max(cy * ih - ch / 2, 0), ih - ch)
    return im.resize((W, H), Image.LANCZOS, box=(x0, y0, x0 + cw, y0 + ch))


def lens_maps(W, H, swirl, k):
    """Precomputed sampling grid: barrel distortion k + a gentle swirl that grows towards the edges."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    nx, ny = (x - W / 2) / (W / 2), (y - H / 2) / (H / 2)
    r = np.sqrt(nx ** 2 + ny ** 2)
    f = 1 + k * r ** 2
    a = swirl * r ** 2
    sx = (nx * np.cos(a) - ny * np.sin(a)) * f; sy = (nx * np.sin(a) + ny * np.cos(a)) * f
    return np.clip(sx * W / 2 + W / 2, 0, W - 1), np.clip(sy * H / 2 + H / 2, 0, H - 1)


def apply_lens(arr, maps):
    mx, my = maps
    x0 = mx.astype(np.int32); y0 = my.astype(np.int32)
    x1 = np.minimum(x0 + 1, arr.shape[1] - 1); y1 = np.minimum(y0 + 1, arr.shape[0] - 1)
    fx = (mx - x0)[..., None]; fy = (my - y0)[..., None]
    a = arr.astype(np.float32)
    top = a[y0, x0] * (1 - fx) + a[y0, x1] * fx; bot = a[y1, x0] * (1 - fx) + a[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def vignette(W, H, amt):
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2)
    return np.clip(1 - amt * np.clip(r - 0.35, 0, None) ** 1.6, 0, 1)[..., None]


def detail_point(im, rng):
    """Where to step in: the busiest patch (edge energy), with a little randomness."""
    g = np.asarray(im.convert("L").resize((48, 48)), np.float32)
    e = np.abs(np.diff(g, axis=0))[:, :-1] + np.abs(np.diff(g, axis=1))[:-1, :]
    e[:6] *= .4; e[-6:] *= .4; e[:, :6] *= .4; e[:, -6:] *= .4
    flat = e.ravel(); idx = rng.choice(len(flat), p=(flat + 1e-3) / (flat + 1e-3).sum())
    yy, xx = divmod(idx, e.shape[1])
    return (xx + .5) / e.shape[1], (yy + .5) / e.shape[0]


def colour_key(im):
    import colorsys
    a = np.asarray(im.resize((24, 24)), np.float32) / 255
    r, g, b = a[..., 0].mean(), a[..., 1].mean(), a[..., 2].mean()
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return (round(h * 12) if s > 0.08 else 12, l)          # 12 hue bands, greys last; light→dark inside a band


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", action="append", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--technique", choices=["colour", "stepin", "archive"], default="colour")
    p.add_argument("--start", type=float, default=0); p.add_argument("--duration", type=float, default=0)
    p.add_argument("--width", type=int, default=1280); p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--hold", default="2", help="frames per picture (1-6) or 'auto' (follows the song's loudness)")
    p.add_argument("--steps", type=int, default=4, help="stepping in: steps toward each detail")
    p.add_argument("--lens", type=float, default=0.5, help="swirl lens + dark corners 0..1 (colour runs)")
    p.add_argument("--print", dest="printscan", action="store_true", help="print + scan: paper grain, tilt, scanner light")
    p.add_argument("--label", default="", help="small text burned in top-left (e.g. GARDEN STATE NETWORK)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed or None); prng = random.Random(a.seed or None)
    W, H, F = a.width // 2 * 2, a.height // 2 * 2, max(6, a.fps)

    import librosa
    y, sr = librosa.load(a.audio, sr=22050, mono=True, offset=a.start, duration=a.duration or None)
    dur = len(y) / sr
    if dur < 1: sys.exit(f"error: nothing to render — the song is shorter than the start point ({a.start:.0f}s)")
    n = int(dur * F)
    rms = librosa.feature.rms(y=y, hop_length=int(sr / F))[0][:n]
    rms = (rms - rms.min()) / (np.ptp(rms) + 1e-9) if len(rms) else np.zeros(n)
    print(f"[audio] {n} frames ({dur:.1f}s @ {F}fps, technique={a.technique})", flush=True)

    def hold_at(i):
        if a.hold == "auto":
            v = rms[min(i, len(rms) - 1)] if len(rms) else .5
            return 1 if v > .66 else 2 if v > .33 else 3
        return max(1, min(6, int(a.hold)))

    avg_hold = 2 if a.hold == "auto" else max(1, min(6, int(a.hold)))
    want = int(n / avg_hold / (a.steps if a.technique == "stepin" else 1)) + 8
    tmp = Path(a.out).with_name(Path(a.out).stem + "_src"); tmp.mkdir(parents=True, exist_ok=True)
    pics = gather(a.sources, want, prng, tmp)
    if not pics: sys.exit("error: no pictures found in those folders")

    frames_dir = Path(a.out).with_name(Path(a.out).stem + "_frames"); frames_dir.mkdir(parents=True, exist_ok=True)
    if a.technique == "colour":
        print("[sort] reading colours", flush=True)
        keyed = []
        for k, f in enumerate(pics[:want * 2]):
            im = load(f, 400)
            if im: keyed.append((colour_key(im), f))
        keyed.sort(key=lambda t: t[0]); pics = [f for _, f in keyed]
    elif a.technique == "archive":
        pics.sort(key=lambda f: f.name.lower())           # the archive keeps its own order

    maps = lens_maps(W, H, 0.55 * a.lens, 0.12 * a.lens) if a.technique == "colour" and a.lens > 0 else None
    vig = vignette(W, H, 1.6 * a.lens) if a.technique == "colour" and a.lens > 0 else None
    grain = None
    if a.printscan:
        grain = (rng.normal(0, 1, (H, W, 1)) * 9).astype(np.float32)
        paper = np.array([248, 243, 230], np.float32) / 255
    try: font = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", max(11, H // 52))
    except Exception: font = ImageFont.load_default()

    enc = subprocess.Popen([FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(F), "-i", "-",
                            "-ss", str(a.start), *(["-t", str(a.duration)] if a.duration else []), "-i", a.audio,
                            "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-b:a", "256k", "-shortest", a.out], stdin=subprocess.PIPE)

    def finish(img_arr, label_right):
        arr = img_arr
        if maps is not None: arr = apply_lens(arr, maps) * vig
        if a.printscan:
            arr = arr * (0.82 + 0.18 * paper) + grain + 6 * np.sin(np.linspace(0, 40, H))[:, None, None]
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        if a.label or label_right:
            im = Image.fromarray(arr); d = ImageDraw.Draw(im); m = max(12, H // 36)
            if a.label: d.text((m, m), a.label.upper(), fill=(235, 232, 225), font=font)
            if label_right: d.text((W - m, H - m), label_right, fill=(235, 232, 225), font=font, anchor="rb")
            arr = np.asarray(im)
        return arr

    i, pic_i, shot, t0, last, bad = 0, 0, 0, time.time(), None, 0
    total_shots = max(1, int(n / avg_hold / (a.steps if a.technique == "stepin" else 1)))
    while i < n:
        f = pics[pic_i % len(pics)]; pic_i += 1
        im = load(f)
        if im is None:
            bad += 1
            if bad >= len(pics): sys.exit("error: none of the pictures in this folder could be opened")
            continue
        bad = 0
        shot += 1
        if a.technique == "colour":
            seq = [cover(im, W, H)]
        elif a.technique == "stepin":
            cx, cy = detail_point(im, rng)
            seq = [cover(im, W, H, cx, cy, zoom=1.0 + 2.2 * k / max(1, a.steps - 1)) for k in range(a.steps)]
        else:  # archive: native size on black, wear kept in
            canvas = Image.new("RGB", (W, H), (0, 0, 0)); pic = im.copy()
            if pic.width > W * .92 or pic.height > H * .92: pic.thumbnail((int(W * .92), int(H * .92)), Image.LANCZOS)
            if rng.random() < 0.18:                             # compression break
                from io import BytesIO
                b = BytesIO(); pic.save(b, "JPEG", quality=int(rng.integers(3, 12))); pic = Image.open(BytesIO(b.getvalue())).convert("RGB")
            canvas.paste(pic, ((W - pic.width) // 2, (H - pic.height) // 2)); seq = [canvas]
        for k, fr in enumerate(seq):
            arr = np.asarray(fr, np.float32)
            if a.printscan:
                fr2 = fr.rotate(rng.uniform(-.6, .6), resample=Image.BILINEAR, fillcolor=(20, 18, 16)); arr = np.asarray(fr2, np.float32)
            right = (f"{shot} / {total_shots}" if a.technique == "colour" else
                     f"step {k + 1} of {len(seq)}" if a.technique == "stepin" else f"file {shot:03d}") if a.label else ""
            out = finish(arr, right)
            h = hold_at(i)
            if a.technique == "archive" and rng.random() < 0.06 and last is not None:
                out = last                                      # a dropped frame: the previous picture hangs on
            for _ in range(h):
                if i >= n: break
                enc.stdin.write(out.tobytes()); i += 1
            last = out
        if shot % 3 == 0 or i >= n:
            for old in frames_dir.glob("f_*.png"): old.unlink()
            Image.fromarray(last).resize((W // 2, H // 2)).save(frames_dir / f"f_{shot:06d}.png", compress_level=1)
            el = time.time() - t0
            print(f"  shot {min(shot, total_shots)}/{total_shots}  {i}/{n} frames  {el / max(1, i):.3f}s/frame", flush=True)
    enc.stdin.close(); rc = enc.wait()
    shutil.rmtree(frames_dir, ignore_errors=True); shutil.rmtree(tmp, ignore_errors=True)
    if rc != 0: sys.exit("error: encoding failed")
    print(f"[done] {a.out}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
