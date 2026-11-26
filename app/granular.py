#!/usr/bin/env python3
"""GRANULAR — one handheld clip through a rebuild of Masakatsu Takagi's 2002–03 video looks (LIMINAL ENGINE).

  granular  Takagi's Girls method treated as a granular synth: grains sample colour from a frame buffer
            (filter bank = brightness x saturation bands, density, size, envelope, spray, time spray, decay,
            follow motion, freeze, dry painted bed vs grains on paper; a song's loudness can modulate it).
  paint     Girls (2003): vivid painted base + a dot texture driven by darkness + brush smudge.
  outline   Rama (2002): per-channel Find Edges lines over a Colorama hue wheel / palettes sampled from his frames.

Copied 11-23 from the dev sandbox ~/Projects/takagi-fx (takagi_fx.py: Paint, Rama2 + helpers; granulator.py:
Granulator) so the app ships on its own. The defaults below are Eric's picks; same seed + same clip = same video.

  python app/granular.py --source CLIP --out OUT.mp4 --look granular [--audio SONG --song_start S]
                         [--start S] [--duration D] [--width 960] [--seed 3] [--set key=value ...]

Length: --duration seconds from --start of the clip (0 = to the end of the clip). With --audio the song plays from
--song_start for the same length (padded with silence if it runs out) and its loudness drives mod_*; without it the
clip keeps its own sound (from the same start). Clips above 30 fps are resampled to 29.97 so step/decay/follow mean
what they were tuned for; 24/25p clips keep their own rate and the settings counted in video frames (step, time_spray,
buffer, freeze_len, edge_step) are scaled by fps/29.97 instead. Frames are scaled so the width is at most --width
(aspect kept, even sizes); sizes in pixels are scaled by width/640, so an HD render keeps the 640 look.
The settings used go into the mp4's comment tag.
"""
import argparse, json, math, os, shutil, signal, subprocess, sys, time
from pathlib import Path

# A GUI-launched .app inherits a minimal PATH with no homebrew; repair it before shelling out to ffmpeg/ffprobe.
for _p in ("/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin"):
    if os.path.isdir(_p) and _p not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")
FF = shutil.which("ffmpeg")
FP = shutil.which("ffprobe")

try:
    import cv2
except ImportError:
    print("[error] Granular needs OpenCV, which isn't installed in this copy. Run Set Up again (it adds it), then reopen the app.", flush=True)
    sys.exit(1)
import numpy as np

KS = 1.0   # output width / 640, set by main(): pixel-sized settings and constants were tuned on 640-wide Xacti/DV


# ================================================================ helpers (takagi_fx.py)
def luma(bgr):
    return (0.114 * bgr[..., 0] + 0.587 * bgr[..., 1] + 0.299 * bgr[..., 2])


def smoothstep(a, b, x):
    t = np.clip((x - a) / max(b - a, 1e-6), 0, 1)
    return t * t * (3 - 2 * t)


def _stretch(L, plo, phi, min_range=0.15):
    """stretch brightness between two percentiles; a near-flat frame (white wall, black frame) keeps its own level
    instead of blowing its noise up to the full range (11-23 audit fix: flat frames used to come out dark)."""
    lo, hi = np.percentile(L, plo), np.percentile(L, phi)
    r = hi - lo
    if r < min_range:   # near-flat frame: fade toward no stretch at all, so a white wall stays white and a black
        t = r / min_range  # frame / fade-out stays dark (11-24: the centred window turned flat black into mid-grey)
        lo, r = lo * t, r * t + (1 - t)
    return np.clip((L - lo) / r, 0, 1)


class Cadence:
    """frame holds for a (possibly fractional) step: 2 = every other frame (~15 fps); 2.5 = held 2,3,2,3 (12 fps, Girls)."""
    def __init__(self, step):
        self.step, self.acc = max(1.0, float(step)), 0.0

    def next_hold(self):
        self.acc += self.step
        n = int(self.acc + 1e-6)
        self.acc -= n
        return n - 1


def warp_by_flow(img, flow, scale=1.0):
    """semi-Lagrangian advection: sample img at x - flow (pulls content forward along motion)."""
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx = gx - flow[..., 0] * scale
    my = gy - flow[..., 1] * scale
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def streak(img, flow, n):
    """cheap line-integral smear: average samples backwards along the flow."""
    if n <= 0:
        return img
    acc = img.copy()
    for i in range(1, n + 1):
        acc += warp_by_flow(img, flow, i / n)
    return acc / (n + 1)


class Flow:
    def __init__(self):
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.prev = None

    def __call__(self, bgr8):
        g = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY)
        if self.prev is None:
            self.prev = g
            return np.zeros(g.shape + (2,), np.float32)
        f = self.dis.calc(self.prev, g, None)
        self.prev = g
        return f


def _norm_noise(rng, h, w, sigma):
    n = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), sigma)
    n /= n.std() + 1e-6
    return 0.5 * (1 + np.tanh(n * 0.85))  # ~uniform 0..1


def brush_smudge(img, gray, n, step):
    """line-integral smudge along the local edge direction (structure tensor) -> hair/brush streaks."""
    if n <= 0:
        return img
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    ts = 3 * KS
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), ts); jxy = cv2.GaussianBlur(gx * gy, (0, 0), ts); jyy = cv2.GaussianBlur(gy * gy, (0, 0), ts)
    ang = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2  # tangent = along the grain
    tx, ty = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    h, w = gray.shape
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    acc = img.copy(); wsum = 1.0
    for k in range(1, n + 1):
        wk = 1 - k / (n + 1)
        for sgn in (1, -1):
            acc += wk * cv2.remap(img, X + sgn * k * step * tx, Y + sgn * k * step * ty, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            wsum += wk
    return acc / wsum


# ================================================================ PAINT (Girls, 2003) — takagi_fx.Paint
# Mostly continuous vivid colour (rebuilt per band), with a dot TEXTURE whose size and coverage follow darkness:
# speckle in lights, leopard blobs in mids, big spots in darks. Blobs = thresholded soft noise (screen-locked).
PAINT = dict(
    l_bands=6, s_bands=5,   # ~30 tonal images (brightness x saturation) for the colour rebuild
    rebuild=0.30,           # band colour vs the frame's own colour (1 = flat band fills)
    sat=2.2,                # saturation of the painted base
    exposure=0.30,          # gamma lift on brightness before anything (sunny, high-key)
    bright=0.35,            # lift (gamma) on the base -> brighter, high-key
    paper_lo=0.66, paper_hi=0.90,  # stretched brightness where the base melts to white
    ink_start=0.22,         # darkness where dots begin
    detail=0.65,            # dot texture follows the image's own detail (vs pure noise)
    sprinkle=0.06,          # sparse dark dots scattered over light areas and paper
    cov_max=0.62,           # even the darkest areas keep colour showing between the dots
    ink_gamma=1.25,         # how fast coverage grows with darkness
    ink_amount=1.0,         # overall dot coverage (0 = clean painted image, 1 = full pointillism)
    ink_dark=0.30,          # how dark the dots are vs the base colour
    ink_soft=0.06,          # edge softness of each blob
    mask_brush=1,           # also brush-smudge the dot mask along the grain (0 = crisp dots, near-black cores; strict)
    noise_brush=0,          # smudge the dot noise along the grain before the dots are cut (samples each side): crisp dots
                            # that still line up along edges (strict 2, with mask_brush 0)
    dots=0,                 # round-dot engine (Girls, 11-24 audit E4): the light/mid dots and the paper sprinkle become
                            # Poisson-placed round dots; the noise texture only prints the merged dark mass (strict 1)
    dot_density=9.0,        # dots tried per 1000 px where the dot layer is at full strength
    dot_floor=0.12,         # dot density in the lights and on the paper, relative to full strength
    dot_spread=0.0,         # px: paper dots stay near the image and fade out this far away (0 = the floor everywhere)
    dot_r=1.9, dot_cv=0.5,  # median dot radius (px) and the lognormal spread of radii
    dot_grow=0.8,           # radius grows with darkness
    dot_stretch=0.6,        # dots stretch along clear image edges (0 = perfectly round)
    dot_gauss=0.0,          # >0: each dot is a Gaussian splat (sigma = this x radius) cut by levels: black centre, grey rim
    dot_halo=1.0,           # (dot_gauss 0 only) px: flat grey halo around each hard-edged dot
    dot_rough=0.0,          # ragged dot edges
    dot_grey=0.0,           # share of dots that print grey (his ~0.25-0.35); where dots overlap, the darker one wins
    dot_top=0.92,           # no round dots where the (smoothed) darkness is above this: the merged mass takes over
    merge=0.45,             # with dots: the noise texture only prints where dark coverage is above this
    fine=0.9, mid=1.8, coarse=3.6,  # noise scales (px sigma) for lights / mids / darks
    refresh=0,              # re-roll the noise every N processed frames (0 = screen-locked)
    pre_blur=1.2,
    watercolor=1,           # edge-preserving smoothing on the base
    smear=0.35, advect=1.0, # Echo-like persistence along the motion
    step=2.0,               # stepped ~15 fps (2.5 = held 2,3,2,3 = 12 fps, like Girls)
    brush=5,                # brush-stroke smudge along the image's own grain (samples each side, 0 = off)
    brush_len=1.6,          # px per sample
    bleed=0.6,              # white paper bleeding into the image (0 = off)
    motion_streak=4,        # streak samples along the motion (0 = off)
    soft=0.7,               # final softness (px)
)  # defaults = variant "B" + medium brush, picked by Eric 11-23


class Paint:
    def __init__(self, p, rng):
        self.p, self.rng = p, rng
        self.flow = Flow()
        self.fb = None
        self.held, self.hold = None, 0
        self.cad = Cadence(p["step"])
        self.noise, self.nproc = None, 0
        self.gap = None        # motion summed over the input frames since the last processed frame

    def _dark_mass(self, n, cov, p):
        """the noise texture, kept only where the darks merge into masses (dots=1). Coverage ramps from 0 at merge to
        cov_max at cov_max (cov never goes past it), so merge 0 prints the same dark texture as without dots."""
        c2 = np.clip((cov - p["merge"]) / max(p["cov_max"] - p["merge"], 1e-3), 0, 1) * p["cov_max"]
        return np.clip((n - (1 - c2) + p["ink_soft"]) / (2 * max(p["ink_soft"], 1e-6)), 0, 1) * (c2 > 0.01)   # ink_soft 0 = hard edge

    def _round_dots(self, L, d, p, h, w):
        """dots=1: round dots, Poisson-placed with a density that follows darkness (a floor in the lights and on the
        paper), sizes lognormal and growing with darkness, stretched a little along clear edges. Returns ink coverage."""
        rng = self.rng
        dd = cv2.GaussianBlur(d.astype(np.float32), (0, 0), 4 * KS)
        ramp = smoothstep(p["ink_start"], 0.85, dd)
        if p["dot_spread"] > 0:   # the sprinkle spills off the image onto the paper and fades with distance
            near = cv2.GaussianBlur((dd > 0.3).astype(np.float32), (0, 0), p["dot_spread"])
            lam = np.maximum(ramp, p["dot_floor"] * np.clip(near * 2.5, 0, 1))
        else:
            lam = p["dot_floor"] + (1 - p["dot_floor"]) * ramp
        lam = lam * (dd < p["dot_top"])   # darker than this is left to the merged mass, not dots
        lam = lam * min(1.0, p["ink_amount"])   # Ink 0 = no round dots either, like the noise dots (11-24 review)
        nmax = int(rng.poisson(p["dot_density"] * 1e-3 * h * w / (KS * KS)))   # per 640-scale px: same look at any width
        xs = rng.uniform(0, w, nmax); ys = rng.uniform(0, h, nmax)
        keep = rng.uniform(0, 1, nmax) < lam[ys.astype(int), xs.astype(int)]
        xs, ys = xs[keep], ys[keep]
        r = p["dot_r"] * np.exp(rng.normal(0, p["dot_cv"], len(xs))) * (1 - p["dot_grow"] / 2 + p["dot_grow"] * dd[ys.astype(int), xs.astype(int)])
        r = np.maximum(r, 0.05)   # dot_r 0 (or a big dot_grow) from --set: a speck, not a divide by zero
        gl = cv2.GaussianBlur(L.astype(np.float32), (0, 0), 3 * KS)
        gx = cv2.Sobel(gl, cv2.CV_32F, 1, 0); gy = cv2.Sobel(gl, cv2.CV_32F, 0, 1)
        ts = 8 * KS
        jxx = cv2.GaussianBlur(gx * gx, (0, 0), ts); jxy = cv2.GaussianBlur(gx * gy, (0, 0), ts); jyy = cv2.GaussianBlur(gy * gy, (0, 0), ts)
        ang = np.degrees(0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2)
        coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-9)
        tone = np.where(rng.uniform(0, 1, len(xs)) < p["dot_grey"], rng.uniform(0.3, 0.7, len(xs)), 1.0)
        if p["dot_gauss"] > 0:   # Gaussian splat, then levels: the centre clips black, the rim ramps out to ~1.55 r
            G = np.zeros((h, w), np.float32)
            sg = p["dot_gauss"]
            jit = None
            if p["dot_rough"] > 0:   # ragged rims: fine noise added to each splat before its levels
                jit = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 0.9 * KS)
                jit *= p["dot_rough"] * 0.25 / (jit.std() + 1e-6)
            for x, y, rr, tv in zip(xs, ys, r, tone):
                xi, yi = int(x), int(y)
                e = 1 + p["dot_stretch"] * coh[yi, xi]
                a, b = rr * np.sqrt(e), rr / np.sqrt(e)
                R = int(np.ceil(3 * sg * a)) + 1
                x0, x1, y0, y1 = max(0, xi - R), min(w, xi + R + 1), max(0, yi - R), min(h, yi + R + 1)
                yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
                t = np.radians(ang[yi, xi]); c, s_ = np.cos(t), np.sin(t)
                u = (xx - x) * c + (yy - y) * s_; v = (yy - y) * c - (xx - x) * s_
                g = np.exp(-0.5 * ((u / (sg * a)) ** 2 + (v / (sg * b)) ** 2))
                if jit is not None:
                    g = (g + jit[y0:y1, x0:x1]) * (g > 0.12)
                lv = np.clip((g - 0.30) / (0.78 - 0.30), 0, 1) * tv
                G[y0:y1, x0:x1] = np.maximum(G[y0:y1, x0:x1], lv)   # overlapping dots: the darker one wins
            return G
        if p["dot_grey"] > 0:   # darker wins where dots overlap: draw the lightest first
            o = np.argsort(tone, kind="stable"); xs, ys, r, tone = xs[o], ys[o], r[o], tone[o]
        m = np.zeros((h, w), np.uint8); tm = np.zeros((h, w), np.uint8)
        for x, y, rr, tv in zip(xs, ys, r, tone):   # hard-edged anti-aliased ellipses at 1/16 px
            xi, yi = int(x), int(y)
            e = 1 + p["dot_stretch"] * coh[yi, xi]
            a, b = rr * np.sqrt(e), rr / np.sqrt(e)
            cv2.ellipse(m, (int(x * 16), int(y * 16)), (max(1, int(a * 16)), max(1, int(b * 16))), float(ang[yi, xi]), 0, 360, 255, -1, cv2.LINE_AA, 4)
            if p["dot_grey"] > 0:
                cv2.ellipse(tm, (int(x * 16), int(y * 16)), (max(1, int(a * 16)) + 48, max(1, int(b * 16)) + 48), float(ang[yi, xi]), 0, 360, int(255 * tv), -1, cv2.LINE_8, 4)
        core = m.astype(np.float32) / 255
        if p["dot_rough"] > 0:
            soft = cv2.GaussianBlur(core, (0, 0), 1.2 * KS)
            jit = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 0.9 * KS)
            jit /= jit.std() + 1e-6
            core = np.clip((soft + p["dot_rough"] * 0.25 * jit - 0.45) / 0.12, 0, 1) * (soft > 0.12)
        if p["dot_grey"] > 0:
            core = core * (tm.astype(np.float32) / 255)
        if p["dot_halo"] > 0:
            halo = cv2.GaussianBlur(core, (0, 0), p["dot_halo"] * 1.5)
            core = np.maximum(core, np.clip(halo * 1.4, 0, 0.6))
        return core

    def frame(self, bgr8):
        p = self.p
        flow = None   # motion tracking only when smear-along-motion or the motion streak use it (11-23: skip it otherwise)
        if (p["smear"] > 0 and p["advect"]) or p["motion_streak"] > 0:
            f1 = self.flow(bgr8)
            self.gap = f1 if self.gap is None else self.gap + f1
        if self.held is not None and self.hold > 0:
            self.hold -= 1
            return self.held
        flow, self.gap = self.gap, None   # the real motion since the last processed frame (was one frame's x step)
        h, w = bgr8.shape[:2]
        self.nproc += 1
        if self.noise is None or (p["refresh"] and self.nproc % int(p["refresh"]) == 0):
            self.noise = [_norm_noise(self.rng, h, w, s) for s in (p["fine"], p["mid"], p["coarse"])]

        src = cv2.GaussianBlur(bgr8, (0, 0), p["pre_blur"]) if p["pre_blur"] > 0 else bgr8
        if p["watercolor"]:
            src = cv2.edgePreservingFilter(src, flags=cv2.RECURS_FILTER, sigma_s=min(200, 30 * KS), sigma_r=0.3)
        f = src.astype(np.float32) / 255
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        L = luma(f)
        V = np.power(_stretch(L, 2, 98), 1 - p["exposure"])

        # rebuild colour from ~30 tonal images (mean colour per brightness x saturation band)
        lb = np.clip((V * p["l_bands"]).astype(int), 0, p["l_bands"] - 1)
        sb = np.clip((hsv[..., 1] * p["s_bands"]).astype(int), 0, p["s_bands"] - 1)
        band = (lb * p["s_bands"] + sb).ravel()
        nb = p["l_bands"] * p["s_bands"]
        cnt = np.maximum(np.bincount(band, minlength=nb), 1)
        cols = np.stack([np.bincount(band, weights=f[..., c].ravel(), minlength=nb) / cnt for c in range(3)], 1)
        fill = cols[band].reshape(h, w, 3).astype(np.float32)
        base = fill   # 11-25: entirely from band colours, no original pixels (was: fill * rebuild + f * (1-rebuild))

        # vivid + bright, highlights melt to paper
        bh = cv2.cvtColor(np.clip(base, 0, 1), cv2.COLOR_BGR2HSV)
        bh[..., 1] = np.clip(bh[..., 1] * p["sat"], 0, 1)
        bh[..., 2] = np.power(np.clip(bh[..., 2], 0, 1), (1 - p["bright"]) * (1 - p["exposure"]))
        base = cv2.cvtColor(bh, cv2.COLOR_HSV2BGR)
        paper = smoothstep(p["paper_lo"], p["paper_hi"], V)
        if p["bleed"] > 0:
            dk = max(3, int(round(5 * KS)) | 1)   # odd
            spread = cv2.GaussianBlur(cv2.dilate(paper, np.ones((dk, dk), np.uint8)), (0, 0), 4 * KS)
            paper = np.maximum(paper, spread * p["bleed"])
        paper = paper[..., None]
        base = base * (1 - paper) + paper

        # dot texture: size + coverage follow darkness
        d = 1 - V
        cov = np.clip((d - p["ink_start"]) / (1 - p["ink_start"]), 0, 1) ** p["ink_gamma"] * p["ink_amount"]
        cov = np.clip(np.maximum(cov, p["sprinkle"] * min(1.0, p["ink_amount"])), 0, p["cov_max"])   # ink 0 = no dots at all
        nf, nm, nc = self.noise
        wf = np.clip(1 - d * 2, 0, 1); wc = np.clip(d * 2 - 1, 0, 1); wm = 1 - wf - wc
        n = nf * wf + nm * wm + nc * wc
        if p["detail"] > 0:
            g = luma(bgr8.astype(np.float32) / 255)
            hp = g - cv2.GaussianBlur(g, (0, 0), 2.5 * KS)
            hp = 0.5 * (1 - np.tanh(hp / (hp.std() + 1e-6) * 0.9))  # dark detail -> high
            n = n * (1 - p["detail"]) + hp * p["detail"]
        if p["noise_brush"] > 0:   # same spread afterwards, so coverage still means coverage
            sd, mu = n.std(), n.mean()
            ns = brush_smudge(np.ascontiguousarray(n, np.float32), L, int(p["noise_brush"]), p["brush_len"])
            n = mu + (ns - ns.mean()) * (sd / (ns.std() + 1e-6))
        thr = 1 - cov
        mask = (np.clip((n - thr + p["ink_soft"]) / (2 * max(p["ink_soft"], 1e-6)), 0, 1)[..., None]) * (cov[..., None] > 0.01)
        if p["dots"]:   # round dots + the merged dark mass in place of the noise-cut dots
            mask = np.maximum(self._dark_mass(n, cov, p), self._round_dots(L, d, p, h, w))[..., None]
        # 11-25: one field, not a mask overlay. The dot coverage drives how dark each pixel is in a single image:
        # where there's no dot the paint is bright (the base), where there is a dot the paint is dark (ink_dark of
        # the base's own colour). The image is base * (1 - mask * (1 - ink_dark)), so it goes from base at mask=0
        # to base*ink_dark at mask=1, keeping the base's hue. No separate ink layer is composed on top.
        base = brush_smudge(base.astype(np.float32), L, int(p["brush"]), p["brush_len"])
        if p["motion_streak"] > 0:
            rel = flow - np.median(flow.reshape(-1, 2), axis=0)
            base = streak(base, rel, int(p["motion_streak"]))
        if p["soft"] > 0:
            base = cv2.GaussianBlur(base, (0, 0), p["soft"])
        if p["brush"] > 0 and p["mask_brush"]:
            mask = brush_smudge(np.ascontiguousarray(mask[..., 0], np.float32), L, max(1, int(p["brush"]) // 3), p["brush_len"])[..., None]
        # the dot darkens the base's own colour at each pixel, and saturates it slightly (ink on wet paint)
        dark = mask * (1 - p["ink_dark"])   # 0 where no dot, up to (1-ink_dark) where full dot
        sat_boost = 1 + 0.1 * mask[..., 0] if mask.ndim == 3 else 1 + 0.1 * mask   # slight saturation lift where dots print
        bh2 = cv2.cvtColor(np.clip(base, 0, 1), cv2.COLOR_BGR2HSV)
        bh2[..., 1] = np.clip(bh2[..., 1] * sat_boost, 0, 1)
        img = cv2.cvtColor(bh2, cv2.COLOR_HSV2BGR) * (1 - dark)

        if self.fb is not None and p["smear"] > 0:
            mem = warp_by_flow(self.fb, flow, p["advect"]) if p["advect"] else self.fb
            mem = mem * p["smear"] + (1 - p["smear"])
            img = np.minimum(img, mem)
        img = np.clip(img, 0, 1)
        self.fb = img
        out = (img * 255).astype(np.uint8)
        self.held, self.hold = out, self.cad.next_hold()
        return out


# ================================================================ OUTLINE (Rama, 2002) — takagi_fx.Rama2
# Thin doubled dark lines (AE Find Edges, multiplied) over a gradient map / Colorama wheel, new palette per shot.
RAMA2 = dict(
    palette=-1,          # force one of the 9 sampled Rama palettes (-1 = random per shot)
    pre_blur=0.8,
    edge_gain=3.0,       # Find Edges contrast
    edge_mix=0.85,       # how strongly the edge layer multiplies in
    map_mix=0.85,        # gradient map vs original colour
    wheel=0.6,           # AE Colorama default (intensity -> full hue wheel) mixed over the ramp
    cycles=1.3,          # how many times brightness wraps the hue wheel
    wheel_sat=0.85,
    shot_thresh=0.45,
    cycle=0.0,           # Colorama-style phase drift per frame (0..1 of the palette)
    lift=0.25,
    ghost=0.3,
    grain=0.025,
    step=1.0,
    edge_step=0,         # hold the line layer this many frames (0 = redraw with the colour; Rama holds it 6)
    dv=0,                # DV 4:1:1 chroma: colour in 4-pixel-wide blocks, like Takagi's camera
    tone=0,              # 1 = set brightness from a curve fitted to Rama's frames (keeps the hue; strict stage 1, off
                         # since the measured maps: cmap already lands his brightness)
    tone_sat=0.0,        # with tone, where a colour can't reach his brightness: 0 = go pale to reach it, 1 = keep the
                         # colour (only the brightest tones go pale, as his do)
    col_sat=1.0,         # saturation of the mapped colour (his frames are more vivid than the sampled palettes; was 1.8
                         # in strict stage 1, before the measured maps)
    edge_floor=0.0,      # edge strength below this draws no line (faint texture stays clean; strict 0.35)
    cmap=0,              # 1 = colour from maps measured off his Rama frames (RAMA_MAPS), read by brightness rank, a new
                         # map at each cut; replaces palette + wheel (Palette >= 0 fixes one map; strict)
    line_gain=0.0,       # line strength used instead of edge_gain when > 0 (strict only: the app always sends edge_gain)
    colorama=0,          # 0 = off; n = After Effects Colorama preset COLORAMA[n-1], bit-exact (replaces palette, wheel,
                         # measured maps, col_sat and tone; reads the unblurred frame, as AE would)
    ae_lines=0,          # 0 = the tuned lines (Lines slider); 1 = AE Find Edges bit-exact at render size; 2 = the same at
                         # DV size (720 wide, scaled back up: thicker, softer lines, like his 2002-08 masters)
)
# After Effects Colorama presets, measured off AE 2027 renders (11-24, ~/Projects/takagi-fx/verify/RESULTS.md): 0 error on
# grey levels, 384 random colours and a real frame. Phase = (R+G+B)/3, so ae_colorama.npy holds each preset's colour for
# every R+G+B sum 0..765 (35 x 766 x 3, RGB), indexed by AE's menu position (1-based). Defaults only: Phase Shift 0,
# Cycle Repetitions 1, Interpolate on. Order here = AE's menu: the named looks, then the utility ramps.
COLORAMA = [(n, i) for i, n in enumerate(
    ["None", "Alpha Ramp", "Hue Cycle", "Negative", "Ramp Red", "Ramp Green", "Ramp Blue", "Ramp Grey", "RGB", "Saturation Ramp",
     "Solarize Red", "Solarize Green", "Solarize Blue", "Solarize Grey", "-", "Carribean", "Clay", "Copper", "Deep Ocean",
     "Earthenware", "Fire", "Fire And Smoke", "Golden 1", "Golden 2", "Granite", "Horizon", "Leather", "Moldy", "Mossy",
     "Old Glory", "Rusty", "Sepia 1", "Sepia 2", "Skink", "Sunset"], 1)]
COLORAMA = [c for c in COLORAMA if c[1] >= 16] + [c for c in COLORAMA if 3 <= c[1] <= 14]   # 20 looks + 12 utilities
_COLORAMA_T = None


def colorama(bgr8, n):
    """AE Colorama, preset COLORAMA[n-1], default settings: bit-exact on 8-bit input. Returns BGR uint8."""
    global _COLORAMA_T
    if _COLORAMA_T is None:
        _COLORAMA_T = np.load(Path(__file__).with_name("ae_colorama.npy"))[..., ::-1].copy()   # RGB -> BGR
    return _COLORAMA_T[COLORAMA[int(n) - 1][1] - 1][bgr8.astype(np.int32).sum(2)]


# NIHITI's pale-blue scenes (11-25): not an AE preset (best preset/channel-swap/hue-turn fit: median dE 3.4, worst
# tenth 8-45) but two hand-made Colorama cycles: one for the figures + their halo (bluer), one for the ground (greener),
# median dE 2.1 / 90% 4.6 on 8 of his frames (red with two cycles: no gain, 1.0 - it really is one palette). Fitted in
# ~/Projects/takagi-fx/verify (fit_two_cycles.py, make_blue_luts.py). Colours are his; the tonal curve is Fire's (the
# point on each blue cycle with Fire's luminance at that R+G+B), so the Light build is unchanged. nihiti_blue.npy:
# (2 = figures, ground) x 766 x RGB, indexed by R+G+B like ae_colorama.npy.
LIGHT_BLUE = len(COLORAMA) + 1   # the Light look's Colorama slider: one stop past the AE presets
_BLUE_T = None


def nihiti_blue(bgr8, share):
    """bgr8: the Light composite; share: 0..1 per pixel, how much of its light comes from the figure layer."""
    global _BLUE_T
    if _BLUE_T is None:
        _BLUE_T = np.load(Path(__file__).with_name("nihiti_blue.npy"))[..., ::-1].astype(np.float32)   # RGB -> BGR
    i = bgr8.astype(np.int32).sum(2); w = share[..., None]
    return np.clip(np.floor(_BLUE_T[1][i] * (1 - w) + _BLUE_T[0][i] * w + 0.5), 0, 255).astype(np.uint8)


_SOBEL = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)


def ae_find_edges(bgr8, dv=False):
    """AE Find Edges (Invert off), bit-exact at render size: per channel unnormalised 3x3 Sobel, L2, rounded, clipped,
    255 - that; border pixels repeat (verified 0 error on 24 renders, 11-24). dv: run it at 720 wide and scale back up.
    Returns 0..1 float32 (1 = flat / white)."""
    h, w = bgr8.shape[:2]
    src = cv2.resize(bgr8, (720, max(2, round(h * 720 / w))), interpolation=cv2.INTER_AREA) if dv and w > 720 else bgr8
    s = src.astype(np.float32)
    gx = cv2.filter2D(s, -1, _SOBEL, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.filter2D(s, -1, _SOBEL.T, borderType=cv2.BORDER_REPLICATE)
    e = (255 - np.clip(np.round(np.sqrt(gx * gx + gy * gy)), 0, 255)) / 255
    return cv2.resize(e, (w, h), interpolation=cv2.INTER_LINEAR) if e.shape[:2] != (h, w) else e


def _box3(w):
    """Three passes of a continuous-width box (each pixel integrates the box): the blur inside AE's Directional Blur
    and Glow (measured 11-24). Returns the 1-D kernel."""
    R = int(np.ceil(1.6 * w)) + 2
    x = np.arange(-R, R + 1); k = np.clip(np.minimum(x + .5, w / 2) - np.maximum(x - .5, -w / 2), 0, None); k /= k.sum()
    out = k
    for _ in range(2): out = np.convolve(out, k, mode="same")
    return (out / out.sum()).astype(np.float32)


def ae_directional_blur(img, length):
    """AE Directional Blur, direction 90 (horizontal): <=1 level on every pixel vs AE 2027, L 3..80 (11-24). Box width
    0.7375 L + 1; beyond the frame is black, as AE's transparent edge comes out over black. float32 in, float32 out."""
    if length <= 0: return img
    k = _box3(0.7375 * length + 1)
    return cv2.filter2D(img, -1, k[None], borderType=cv2.BORDER_CONSTANT)


_GLOW_W = [(0, 0.0), (3, 1.861), (4, 3.128), (10, 7.585), (25, 19.151)]   # Glow Radius -> 3-box width (measured)


def ae_glow(img, threshold=153, radius=10, intensity=1.0):
    """AE Glow, other settings default (Color Channels, Add): per channel, a soft threshold from threshold-16 over 32
    levels, 3-box blur sized by Radius, x Intensity, added. ~99% of pixels within 1 level of AE 2027 at intensity 1
    (11-24, not bit-exact). float32 0..255 in, float32 out (unclipped)."""
    if intensity <= 0 or radius <= 0: return img
    w = float(np.interp(radius, *zip(*_GLOW_W))) if radius <= 25 else 0.7661 * radius
    S = np.clip((img - (threshold - 16)) / 32, 0, 1) * 255
    k = _box3(w)
    return img + intensity * cv2.sepFilter2D(S, -1, k, k, borderType=cv2.BORDER_CONSTANT)
# AE Gaussian Blur (Legacy), 11-25. Fast Blur (Legacy) renders pixel-identically in AE 2027 (Repeat Edge Pixels off).
# Not a true Gaussian: 3 passes of a continuous-width box, width 0.7375*b + 1 (Directional Blur's own law); Blurriness
# 1 and 2 are least-squares kernels. <=1 level on 100% of pixels for b = 3..60, b=2 max 1, b=1 max 4 at the frame edge
# (~/Projects/takagi-fx/verify: ls_eblur.py, law_eblur.py, test_eblur.py). Fractional b below 3 interpolates (unverified).
_LEG_SMALL = {0: np.array([1.0]), 1: np.array([.2651, .4698, .2651]), 2: np.array([.0916, .2440, .3288, .2440, .0916])}


def _legacy_kernel(b):
    b = max(0.0, float(b))
    if b >= 3: return _box3(0.7375 * b + 1)
    lo = int(np.floor(b)); t = b - lo
    hi_k = _box3(0.7375 * 3 + 1) if lo + 1 >= 3 else _LEG_SMALL[lo + 1]; lo_k = _LEG_SMALL[lo]; n = max(len(lo_k), len(hi_k))
    pad = lambda k: np.pad(k, ((n - len(k)) // 2,))
    return ((1 - t) * pad(lo_k) + t * pad(hi_k)).astype(np.float32)


def ae_legacy_blur(img, blurriness):
    """AE Gaussian Blur (Legacy), both directions, outside the layer = black. float32 in, float32 out (unrounded)."""
    k = _legacy_kernel(blurriness)
    return cv2.sepFilter2D(img.astype(np.float32), -1, k, k, borderType=cv2.BORDER_CONSTANT)


# brightness (stretched) -> output luma, fitted to Takagi's Rama frames (11-23 audit)
TONE_X = [0, .010, .087, .145, .208, .279, .347, .422, .553, .990, 1]
TONE_Y = [.12, .166, .404, .562, .671, .750, .801, .829, .854, .892, .90]
# Measured colour maps (11-24, audit E3; mkmaps_rank.py): for each of 12 Rama frames, the median colour of its flat
# pixels at 17 brightness-rank stops (stop k = the pixels k/16 of the way up the frame's own brightness order).
# Read by rank, so each map lands his share of each colour (his red-clipped share ~0.51) whatever the clip's exposure.
RAMA_MAPS = [  # 17 stops, darkest -> brightest by brightness RANK (share of the frame), BGR 0..1; one per Rama frame
    [(0.216, 0.055, 0.012), (0.231, 0.082, 0.022), (0.239, 0.114, 0.031), (0.416, 0.118, 0.161), (0.486, 0.102, 0.392), (0.49, 0.122, 0.518), (0.502, 0.161, 0.741), (0.12, 0.851, 0.984), (0.2, 0.929, 0.91), (0.208, 0.933, 0.922), (0.224, 0.937, 0.925), (0.231, 0.937, 0.933), (0.212, 0.941, 0.945), (0.2, 0.941, 0.965), (0.204, 0.941, 0.976), (0.208, 0.945, 0.984), (0.227, 0.949, 0.992)],  # f020.png
    [(0.302, 0.067, 0.2), (0.314, 0.196, 0.235), (0.318, 0.314, 0.349), (0.369, 0.341, 0.49), (0.235, 0.396, 0.69), (0.133, 0.447, 0.91), (0.11, 0.529, 0.969), (0.11, 0.612, 0.976), (0.102, 0.678, 1), (0.165, 0.749, 0.914), (0.231, 0.867, 0.757), (0.271, 0.898, 0.765), (0.318, 0.918, 0.78), (0.361, 0.925, 0.804), (0.408, 0.933, 0.816), (0.475, 0.937, 0.835), (0.549, 0.953, 0.843)],  # f095.png
    [(0.408, 0.22, 0.09), (0.435, 0.325, 0.09), (0.431, 0.404, 0.329), (0.365, 0.365, 0.937), (0.38, 0.455, 1), (0.384, 0.518, 1), (0.4, 0.584, 1), (0.427, 0.643, 1), (0.486, 0.667, 1), (0.514, 0.686, 1), (0.522, 0.733, 1), (0.514, 0.788, 0.973), (0.549, 0.82, 0.949), (0.62, 0.863, 0.906), (0.675, 0.898, 0.863), (0.725, 0.918, 0.843), (0.769, 0.925, 0.851)],  # f140.png
    [(0.043, 0.114, 0.204), (0.29, 0.204, 0.369), (0.388, 0.298, 0.455), (0.396, 0.341, 0.678), (0.455, 0.369, 0.827), (0.41, 0.459, 0.82), (0.349, 0.671, 0.663), (0.345, 0.796, 0.663), (0.4, 0.863, 0.733), (0.408, 0.902, 0.816), (0.396, 0.925, 0.855), (0.427, 0.929, 0.867), (0.494, 0.937, 0.867), (0.549, 0.945, 0.867), (0.584, 0.957, 0.871), (0.659, 0.961, 0.875), (0.753, 0.965, 0.89)],  # f170.png
    [(0.373, 0.129, 0.016), (0.31, 0.22, 0.047), (0.267, 0.298, 0.067), (0.286, 0.369, 0.173), (0.333, 0.184, 1), (0.349, 0.275, 1), (0.337, 0.345, 1), (0.333, 0.408, 1), (0.322, 0.467, 1), (0.302, 0.561, 1), (0.212, 0.729, 1), (0.196, 0.784, 1), (0.192, 0.816, 1), (0.267, 0.831, 1), (0.278, 0.851, 1), (0.325, 0.882, 1), (0.396, 0.925, 1)],  # f305.png
    [(0.267, 0.2, 0.027), (0.259, 0.247, 0.051), (0.247, 0.29, 0.067), (0.251, 0.337, 0.075), (0.235, 0.373, 0.098), (0.212, 0.427, 0.118), (0.2, 0.506, 0.173), (0.357, 0.282, 1), (0.369, 0.349, 1), (0.29, 0.49, 1), (0.286, 0.592, 1), (0.114, 0.78, 1), (0.102, 0.835, 0.992), (0.118, 0.847, 1), (0.067, 0.875, 1), (0.38, 0.839, 1), (0.427, 0.855, 1)],  # f395.png
    [(0.737, 0.161, 0.027), (0.89, 0.294, 0.067), (0.953, 0.38, 0.133), (0.882, 0.392, 0.29), (0.882, 0.431, 0.388), (0.894, 0.49, 0.447), (0.941, 0.573, 0.49), (0.953, 0.635, 0.537), (0.898, 0.655, 0.729), (0.882, 0.675, 0.867), (0.882, 0.702, 0.945), (0.894, 0.722, 0.976), (0.941, 0.745, 0.988), (1, 0.784, 0.992), (1, 0.808, 1), (1, 0.816, 1), (1, 0.827, 1)],  # f425.png
    [(0.592, 0.145, 0.09), (0.612, 0.224, 0.192), (0.745, 0.306, 0.243), (0.388, 0.416, 0.161), (0.467, 0.62, 0.075), (0.682, 0.451, 0.341), (0.79, 0.486, 0.392), (0.863, 0.498, 0.522), (0.859, 0.541, 0.667), (0.843, 0.627, 0.78), (0.318, 0.678, 0.984), (0.886, 0.694, 0.945), (0.945, 0.706, 0.953), (0.969, 0.722, 0.969), (0.984, 0.737, 0.984), (0.992, 0.757, 0.996), (0.992, 0.78, 1)],  # f455.png
    [(0.141, 0.29, 0.118), (0.271, 0.349, 0.263), (0.043, 0.345, 1), (0.255, 0.494, 0.992), (0.918, 0.608, 0.933), (0.969, 0.655, 0.973), (0.988, 0.675, 0.992), (1, 0.69, 1), (1, 0.698, 1), (0.996, 0.714, 1), (1, 0.722, 1), (0.988, 0.737, 1), (0.988, 0.749, 1), (0.992, 0.761, 0.996), (0.992, 0.776, 0.992), (0.992, 0.792, 0.992), (0.996, 0.82, 0.984)],  # f530.png
    [(0.412, 0.329, 0.169), (0.863, 0.376, 0.063), (0.867, 0.427, 0.063), (0.875, 0.463, 0.122), (0.804, 0.49, 0.278), (0.414, 0.424, 0.765), (0.031, 0.447, 1), (0.071, 0.478, 1), (0.255, 0.502, 1), (0.612, 0.494, 1), (0.796, 0.502, 1), (0.843, 0.525, 1), (0.867, 0.541, 1), (0.894, 0.557, 1), (0.918, 0.58, 1), (0.953, 0.608, 1), (0.996, 0.643, 1)],  # f575.png
    [(0.365, 0.129, 0.129), (0.439, 0.184, 0.235), (0.455, 0.271, 0.561), (0.451, 0.278, 0.945), (0.431, 0.447, 0.945), (0.298, 0.616, 0.925), (0.133, 0.714, 0.902), (0.075, 0.741, 0.965), (0.118, 0.773, 0.965), (0.165, 0.796, 0.965), (0.18, 0.816, 0.965), (0.216, 0.827, 0.965), (0.251, 0.847, 0.953), (0.278, 0.859, 0.953), (0.325, 0.875, 0.953), (0.408, 0.886, 0.945), (0.51, 0.914, 0.933)],  # f710.png
    [(0.38, 0.11, 0.102), (0.435, 0.149, 0.157), (0.455, 0.18, 0.322), (0.424, 0.255, 0.596), (0.431, 0.294, 0.863), (0.416, 0.408, 0.914), (0.451, 0.494, 0.949), (0.376, 0.659, 0.855), (0.333, 0.741, 0.875), (0.306, 0.784, 0.902), (0.298, 0.816, 0.925), (0.278, 0.839, 0.945), (0.271, 0.855, 0.961), (0.302, 0.859, 0.965), (0.325, 0.863, 0.969), (0.38, 0.871, 0.969), (0.463, 0.878, 0.969)],  # f740.png
]
PALETTES = [  # dark -> light, BGR 0..1 — sampled from 9 frames of Takagi's Rama (2002)
    [(0.25, 0.25, 0.73), (0.33, 0.36, 0.95), (0.33, 0.51, 0.97), (0.36, 0.60, 0.99), (0.37, 0.67, 1.00)],
    [(0.23, 0.21, 0.56), (0.35, 0.44, 0.94), (0.45, 0.62, 0.98), (0.60, 0.84, 0.93), (0.73, 0.98, 0.80)],
    [(0.35, 0.04, 0.08), (0.41, 0.10, 0.16), (0.46, 0.45, 0.80), (0.53, 0.68, 0.93), (0.38, 0.88, 0.94)],
    [(0.22, 0.40, 0.78), (0.23, 0.75, 0.92), (0.33, 0.90, 0.97), (0.47, 0.95, 0.92), (0.58, 0.98, 0.90)],
    [(0.38, 0.22, 0.16), (0.36, 0.33, 0.40), (0.37, 0.47, 0.93), (0.35, 0.69, 0.98), (0.38, 0.76, 1.00)],
    [(0.65, 0.27, 0.37), (0.82, 0.45, 0.67), (0.95, 0.62, 0.89), (0.99, 0.69, 0.96), (1.00, 0.72, 0.98)],
    [(0.18, 0.27, 0.39), (0.77, 0.46, 0.80), (0.95, 0.69, 0.89), (0.97, 0.75, 0.93), (0.97, 0.81, 0.92)],
    [(0.23, 0.14, 0.44), (0.29, 0.24, 0.69), (0.31, 0.45, 0.70), (0.35, 0.85, 0.44), (0.47, 0.96, 0.69)],
    [(0.31, 0.18, 0.47), (0.43, 0.35, 0.70), (0.62, 0.50, 0.81), (0.51, 0.86, 0.83), (0.53, 0.95, 0.87)],
]


def gradient_map(t, pal, phase=0.0):
    pal = np.array(pal, np.float32)
    t = (t + phase) % 1.0 if phase else np.clip(t, 0, 1)
    x = t * (len(pal) - 1)
    i = np.clip(np.floor(x).astype(int), 0, len(pal) - 2)
    a = (x - i)[..., None]
    return pal[i] * (1 - a) + pal[i + 1] * a


class Rama2:
    def __init__(self, p, rng):
        self.p, self.rng = p, rng
        self.pick = lambda: PALETTES[p["palette"] % len(PALETTES)] if p["palette"] >= 0 else PALETTES[self.rng.integers(len(PALETTES))]   # (9-11 = a measured map: cmap only)
        self.pal = self.pick()
        self.shot_phase = rng.random()
        self.pick_map = lambda: RAMA_MAPS[p["palette"] % len(RAMA_MAPS)] if p["palette"] >= 0 else RAMA_MAPS[self.rng.integers(len(RAMA_MAPS))]
        self.map = self.pick_map() if p["cmap"] else None
        self.prev_hist = None
        self.fb = None
        self.flow = Flow()
        self.phase = 0.0
        self.held, self.hold = None, 0
        self.cad = Cadence(p["step"])
        self.k, self.edge_k, self.edges = 0, 0, None

    def frame(self, bgr8):
        p = self.p
        flow = self.flow(bgr8) if p["ghost"] > 0 else None   # motion tracking only feeds the ghost
        hsv8 = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv8], [0, 1], None, [30, 16], [0, 180, 0, 256]); cv2.normalize(hist, hist)
        if self.prev_hist is not None and cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > p["shot_thresh"]:
            self.pal = self.pick(); self.shot_phase = self.rng.random()
            if p["cmap"]:
                self.map = self.pick_map()
        self.prev_hist = hist
        self.k += 1
        if self.held is not None and self.hold > 0:
            self.hold -= 1
            return self.held
        src = cv2.GaussianBlur(bgr8, (0, 0), p["pre_blur"]) if p["pre_blur"] > 0 else bgr8
        f = src.astype(np.float32) / 255
        L = luma(f)
        Lr = L if p["cmap"] else None
        L = np.power(_stretch(L, 1, 99), 1 - p["lift"])
        self.phase = (self.phase + p["cycle"]) % 1.0
        if p["colorama"]:   # AE Colorama preset, exact, on the unblurred frame (palette, wheel, maps, sat + tone skipped)
            col = colorama(bgr8, p["colorama"]).astype(np.float32) / 255
        elif p["cmap"]:   # his measured map, read by brightness rank (ties share a rank, so flat areas stay flat)
            q = np.clip((Lr * 1023 + 0.5).astype(np.int32), 0, 1023)
            h = np.bincount(q.ravel(), minlength=1024).astype(np.float32); ch = np.cumsum(h)
            t = ((ch - 0.5 * h) / q.size)[q]
            rng_ = (np.searchsorted(ch, 0.99 * q.size) - np.searchsorted(ch, 0.01 * q.size)) / 1023
            if rng_ < 0.15:   # near-flat frame has no brightness order: lean on its real brightness (white -> top of map)
                w = rng_ / 0.15; t = w * t + (1 - w) * _stretch(Lr, 1, 99)
            col = gradient_map(t, self.map).astype(np.float32)
        else:
            col = gradient_map(L, self.pal, self.phase).astype(np.float32)   # 11-25: no original pixels blended back (was: * map_mix + f * (1-map_mix))
        if p["wheel"] > 0 and not p["cmap"] and not p["colorama"]:
            hh = ((L * p["cycles"] + self.shot_phase + self.phase) % 1.0) * 360
            wh = cv2.cvtColor(np.dstack([hh, np.full_like(L, p["wheel_sat"]), 0.35 + 0.65 * L]).astype(np.float32), cv2.COLOR_HSV2BGR)
            col = col * (1 - p["wheel"]) + wh * p["wheel"]
        if p["col_sat"] != 1 and not p["colorama"]:
            ch = cv2.cvtColor(np.clip(col, 0, 1).astype(np.float32), cv2.COLOR_BGR2HSV)
            ch[..., 1] = np.clip(ch[..., 1] * p["col_sat"], 0, 1)
            col = cv2.cvtColor(ch, cv2.COLOR_HSV2BGR)
        if p["tone"] and not p["colorama"]:   # keep the hue, take the brightness from his measured curve
            tg = np.interp(L, TONE_X, TONE_Y).astype(np.float32)
            sc = tg / np.maximum(luma(col), 1e-3)
            col = col * np.minimum(sc, 1 / np.maximum(col.max(2), 1e-3))[..., None]   # as far as hue + saturation allow
            lc = luma(col)   # the rest of the way by going pale (toward white); tone_sat keeps the colour, except the brightest
            a = np.clip((tg - lc) / np.maximum(1 - lc, 1e-3), 0, 1) * (1 - p["tone_sat"] * (1 - smoothstep(0.80, 0.88, tg)))
            col = np.clip(col + (1 - col) * a[..., None], 0, 1)
        # AE Find Edges: per-channel gradient, inverted (white where flat, coloured dark lines on edges)
        if self.edges is None or p["edge_step"] < 1 or self.k - self.edge_k >= p["edge_step"]:
            first = self.edges is None or p["edge_step"] < 1
            if p["ae_lines"]:   # AE Find Edges, exact, on the unblurred frame (Lines, line_gain, edge_floor skipped)
                self.edges = ae_find_edges(bgr8, dv=p["ae_lines"] >= 2)
            else:
                e = np.zeros_like(f)
                for c in range(3):
                    gx = cv2.Sobel(f[..., c], cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(f[..., c], cv2.CV_32F, 0, 1, ksize=3)
                    e[..., c] = np.maximum(np.sqrt(gx * gx + gy * gy) - p["edge_floor"], 0) if p["edge_floor"] > 0 else np.sqrt(gx * gx + gy * gy)
                self.edges = np.clip(1 - e * (p["line_gain"] if p["line_gain"] > 0 else p["edge_gain"]), 0, 1)
            # keep the line clock on the edge_step grid (a fractional step used to make it drift)
            self.edge_k = self.k if first else self.edge_k + p["edge_step"] * ((self.k - self.edge_k) // p["edge_step"])
        edges = self.edges   # Rama: colour moves at 15 fps, the lines only every 6 frames (5 fps)
        img = col * (1 - p["edge_mix"] + p["edge_mix"] * edges)
        if self.fb is not None and p["ghost"] > 0:
            img = img * (1 - p["ghost"]) + warp_by_flow(self.fb, flow, 0.5) * p["ghost"]
        if p["dv"]:   # DV 4:1:1: full-res brightness, colour 4 px wide
            y = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_BGR2YCrCb)
            hh, ww = y.shape[:2]
            c = cv2.resize(y[..., 1:], (max(1, ww // 4), hh), interpolation=cv2.INTER_AREA)
            y[..., 1:] = cv2.resize(c, (ww, hh), interpolation=cv2.INTER_NEAREST)
            img = cv2.cvtColor(y, cv2.COLOR_YCrCb2BGR)
        # 11-25: random grain removed (footage-independent noise; the DV chroma above is the recording medium)
        img = np.clip(img, 0, 1)
        self.fb = img
        out = (img * 255).astype(np.uint8)
        self.held, self.hold = out, self.cad.next_hold()
        return out


# ================================================================ LIGHT BODIES (NIHITI, 2008)
# 11-25 rebuild: FORMED, not composited. The footage is only read for measurements and every output pixel is made new
# (Eric: "flavour baked into the bread, not toppings"; NIHITI's figures look like a thermal camera recording heat,
# not a thermal filter over video). The first version (AE layer stack: Levels'd cut-out over a Levels'd ground, halo
# + Glow added) is in git history (be48f0f..72924f3).
#   scene  : where people are (Apple Vision occupancy, steadied), each body's broad warmth (its own luminance, heavily
#            blurred, relative to that body's mean so clothing colour doesn't set the heat), the landscape's light
#            (linear luminance smeared sideways = AE Directional Blur's core, auto-exposed on the ground)
#   emit   : one emission field: bodies give off body heat, the ground its own dim light, dust in the air lit by both
#   optics : one energy-conserving lens scatter (3-box PSF at two scales + a long faint tail) spreads every source's light
#   sensor : a saturating response gives the R+G+B index Colorama reads; Fire (his red scenes) / NIHITI blue colour it
# Fitted 11-25 (physical bounds) to 8 of his red frames read back through Fire: figures 512 (his 518), glow 40/30/20 px
# out 140/194/314 (his 197/217/308), ground streaks 1.46-1.56 (1.75), sparks 1.7 per 10k px (1.7). The gaps left are
# his footage (many small figures, rocky ground), not his method. Fit: ~/…/scratchpad burn/fit2.py, finger.py.
LIGHT = dict(
    colorama=6,          # COLORAMA position (6 = Fire); LIGHT_BLUE (33) = his pale-blue scenes' two measured cycles;
                         # 0 = no palette: the heat picture in grey
    body=0.6225,         # body heat (emission of a person)
    shade=0.35,          # broad shading inside a body (0 = an even glow; ~0.8 lets faces show)
    ground=0.095,        # the ground's own light, at its median brightness
    contrast=1.3875,     # the ground's contrast (power on its auto-exposed light)
    streaks=1.0,         # 1 = the ground softened sideways only (keeps his streaks), 0 = softened all round
    smear_len=45.0,      # sideways smear of the landscape's light, px at 640 wide (AE Directional Blur's Blur Length)
    glow=0.525,          # share of all light the lens scatters (0.2..0.9; energy is kept)
    reach=0.3,           # share of the scattered light that goes far (the lens's long faint tail)
    sparks=0.15,         # dust specks in the air, lit by whatever light reaches them (count ~ sparks x 120 at 640)
    cutout=2,            # Vision quality: 0 fast, 1 balanced, 2 accurate
    feather=1.5,         # px at 640: softens the occupancy edge (a body's own edge, before the lens)
    steady=0.7,          # 0..0.9: how much of the last frame's cut-out each new one keeps (Vision redraws the edge)
    step=2.0,            # frame hold: 2 = on twos (15 new pictures a second), NIHITI's red section is mostly on twos
    hold_jitter=1,       # 1 = vary each hold by -1/0/0/+1 frame (his 120-200 s mixes 1s, 2s and 3s); 0 = strict twos
    dv=0,                # DV 4:1:1 chroma (the recording medium)
)
LIGHT_FIXED = dict(psf1=3.25, psf2=31.0, psf_mix=0.45, psf3=60.0, sensor=1.6, dust_gain=6.0, dust_base=0.1)
# the page's preset chips (11-25): Eric's picks B / C, each first-fit and honed
LIGHT_PRESETS = {
    "flat": {},
    "shaded": dict(shade=0.8),
    "flat close": dict(body=0.61, ground=0.055, streaks=0.0, glow=0.6, reach=0.0),
    "shaded close": dict(body=0.61, ground=0.055, streaks=0.0, glow=0.6, reach=0.0, shade=0.8),
}
_PM_SRC = Path(__file__).with_name("person_mask.swift")


def person_mask_tool():
    """Path to the compiled Vision helper, built from person_mask.swift on first use (needs Apple's command line tools)."""
    cache = Path.home() / "Library" / "Caches" / "liminal-engine"
    exe = cache / "person_mask"
    if exe.is_file() and exe.stat().st_mtime >= _PM_SRC.stat().st_mtime:
        return exe
    cache.mkdir(parents=True, exist_ok=True)
    swiftc = shutil.which("swiftc")
    if not swiftc:
        sys.exit("[error] Light bodies needs Apple's command line tools to cut out people: run  xcode-select --install")
    print("[look] building the person cut-out helper (first time only)", flush=True)
    r = subprocess.run([swiftc, "-O", str(_PM_SRC), "-o", str(exe)], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("[error] couldn't build the person cut-out helper:\n" + r.stderr[-800:])
    return exe


class PersonMask:
    """Streams frames to the Vision helper (one process per clip, so Vision keeps the mask steady). Masks come back
    0..1 float32 at the frame size. Frames go in at most 640 wide (Vision works below that anyway)."""
    def __init__(self, quality):
        self.q, self.p, self.size = int(quality), None, None

    def __call__(self, bgr8):
        h, w = bgr8.shape[:2]
        if self.p is None:
            sw = min(640, w); sh = max(2, round(h * sw / w)); self.size = (sw, sh)
            self.p = subprocess.Popen([str(person_mask_tool()), str(sw), str(sh), str(self.q)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        sw, sh = self.size
        small = cv2.resize(bgr8, (sw, sh), interpolation=cv2.INTER_AREA) if (sw, sh) != (w, h) else bgr8
        self.p.stdin.write(cv2.cvtColor(small, cv2.COLOR_BGR2RGBA).tobytes()); self.p.stdin.flush()
        hdr = self.p.stdout.read(8)
        if len(hdr) < 8: sys.exit("[error] the person cut-out helper stopped")
        mw, mh = np.frombuffer(hdr, "<i4")
        if mw <= 0: return np.zeros((h, w), np.float32)
        m = np.frombuffer(self.p.stdout.read(int(mw) * int(mh)), np.uint8).reshape(int(mh), int(mw))
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255

    def __del__(self):
        try:
            if self.p: self.p.stdin.close(); self.p.wait(timeout=5)
        except Exception: pass


class LightBodies:
    def __init__(self, p, rng):
        self.p, self.rng = p, rng
        self.mask = PersonMask(p["cutout"])
        self.cad = Cadence(p["step"])
        self.held, self.hold = None, 0
        self.dust = None     # [y, x, frames left]
        self.msm = None      # the steadied cut-out

    def _optics(self, E):
        """the lens: (1 - glow) of the light lands where it was, glow of it is scattered (two 3-box scales + a far tail)."""
        p, f = self.p, LIGHT_FIXED
        k1, k2 = _box3(f["psf1"] * KS), _box3(f["psf2"] * KS)
        Es = f["psf_mix"] * cv2.sepFilter2D(E, -1, k1, k1, borderType=cv2.BORDER_REFLECT) \
            + (1 - f["psf_mix"]) * cv2.sepFilter2D(E, -1, k2, k2, borderType=cv2.BORDER_REFLECT)
        if p["reach"] > 0:   # the long tail, worked out at a quarter size (it's smooth)
            h, w = E.shape; sm = cv2.resize(E, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA); k3 = _box3(f["psf3"] * KS / 4)
            far = cv2.resize(cv2.sepFilter2D(sm, -1, k3, k3, borderType=cv2.BORDER_REFLECT), (w, h), interpolation=cv2.INTER_LINEAR)
            Es = (1 - p["reach"]) * Es + p["reach"] * far
        return (1 - p["glow"]) * E + p["glow"] * Es

    def frame(self, bgr8):
        p, f = self.p, LIGHT_FIXED
        m = self.mask(bgr8)   # every frame, so Vision's sequence handler sees the whole clip
        if p["steady"] > 0 and self.msm is not None and self.msm.shape == m.shape:
            m = self.msm * p["steady"] + m * (1 - p["steady"])
        self.msm = m
        if self.held is not None and self.hold > 0:
            self.hold -= 1
            return self.held
        occ = cv2.GaussianBlur(m, (0, 0), p["feather"] * KS) if p["feather"] > 0 else m
        Y = np.power(bgr8.astype(np.float32) / 255, 2.2) @ np.array([0.0722, 0.7152, 0.2126], np.float32)   # linear luminance
        # the ground: the landscape's light, smeared sideways and softened (sideways only keeps his streaks)
        gl = ae_directional_blur(Y, p["smear_len"] * KS)
        st = float(np.clip(p["streaks"], 0, 1)); sx, sy = (0.25 * st + 1.5 * (1 - st)) * KS, (0.65 * st + 1.5 * (1 - st)) * KS
        gl = cv2.GaussianBlur(gl, (0, 0), sigmaX=sx, sigmaY=sy)
        gm = np.median(gl[occ < 0.2]) if (occ < 0.2).any() else np.median(gl)   # auto exposure on the ground, like a camera
        Eg = p["ground"] * np.power(np.clip(gl / (gm + 1e-6), 0, 6), p["contrast"])
        # the bodies: body heat, warmer where that body's own light is (broad only: no fabric, little face)
        sig = float(np.interp(p["shade"], [0.35, 0.8], [7.0, 3.75])) * KS
        yb = cv2.GaussianBlur(Y * occ, (0, 0), sig) / np.maximum(cv2.GaussianBlur(occ, (0, 0), sig), 1e-4)
        nb, lab = cv2.connectedComponents((occ > 0.3).astype(np.uint8))
        rel = np.ones_like(yb)
        if nb > 1:
            idx = lab.ravel(); mean = np.bincount(idx, (yb * occ).ravel(), nb) / np.maximum(np.bincount(idx, occ.ravel(), nb), 1e-4)
            mean[0] = (yb * occ).sum() / max(occ.sum(), 1e-4)
            near = cv2.dilate(lab.astype(np.uint16 if nb > 255 else np.uint8), np.ones((15, 15), np.uint8))   # soft edge -> nearest body
            rel = yb / np.maximum(mean[near], 1e-4)
        Eb = p["body"] * (1 + p["shade"] * np.clip(rel - 1, -0.6, 0.6)) * occ
        E = Eg * (1 - occ) + Eb
        # dust in the air, lit by whatever light reaches it (a pixel at 640, d x d px wider)
        if p["sparks"] > 0:
            h, w = E.shape; n = int(self.rng.poisson(p["sparks"] * 120 * h * w / (640 * 480 * KS * KS)))
            new = np.c_[self.rng.integers(0, h, n), self.rng.integers(0, w, n), self.rng.integers(2, 7, n)]
            self.dust = new if self.dust is None else np.r_[self.dust[self.dust[:, 2] > 1] - [0, 0, 1], new]
            lit = cv2.GaussianBlur(E, (0, 0), 20 * KS); ys, xs = self.dust[:, 0], self.dust[:, 1]; add = f["dust_gain"] * (f["dust_base"] + lit[ys, xs])
            d = max(1, int(round(KS)))
            for oy in range(d):
                for ox in range(d): np.add.at(E, (np.minimum(ys + oy, h - 1), np.minimum(xs + ox, w - 1)), add)
        Eo = self._optics(E)
        s = np.clip(np.round(765 * (1 - np.exp(-f["sensor"] * Eo))), 0, 765).astype(np.int32)   # sensor -> R+G+B index
        c = int(p["colorama"])
        if c == LIGHT_BLUE:   # how much of each pixel's light came from the bodies (after the lens): picks its cycle
            share = np.clip(self._optics(Eb) / np.maximum(Eo, 1e-6), 0, 1)
            out = nihiti_blue(_sum_img(s), share)
        elif c:
            out = colorama(_sum_img(s), c)
        else:
            out = np.repeat(np.clip(s // 3, 0, 255).astype(np.uint8)[..., None], 3, 2)   # no palette: the heat picture in grey
        if p["dv"]:
            y = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb); hh, ww = y.shape[:2]
            cc = cv2.resize(y[..., 1:], (max(1, ww // 4), hh), interpolation=cv2.INTER_AREA)
            y[..., 1:] = cv2.resize(cc, (ww, hh), interpolation=cv2.INTER_NEAREST); out = cv2.cvtColor(y, cv2.COLOR_YCrCb2BGR)
        h0 = self.cad.next_hold()
        # 11-25: his hold pattern mixes 1s, 2s and 3s (not pure twos). At step 2, the distribution is
        # P(1)≈0.28, P(2)≈0.50, P(3)≈0.22, well modelled by step + choice([-1,0,0,1]) (measured from NIHITI 120-200 s).
        if p.get("hold_jitter", 1) and p["step"] >= 1.5:
            h0 = max(0, h0 + self.rng.choice([-1, 0, 0, 1]))
        self.held, self.hold = out, h0
        return out


def _sum_img(s):
    """an 8-bit BGR image whose R+G+B is exactly s (0..765): what colorama() / nihiti_blue() index by."""
    b = s // 3; r = s - 3 * b   # r in 0..2: spread over G then R
    return np.dstack([b, b + (r > 0), b + (r > 1)]).astype(np.uint8)


# ================================================================ GRANULAR (the headline) — granulator.Granulator
# Audio idea -> video:
#   filter bank (bands)   : split each source frame into brightness x saturation bands (Takagi: ~30 images)
#   grain                 : a soft dot that samples colour from the source at (x + spray, t - time_spray)
#   grain size / density  : dot size per band (darker = bigger) / new grains per frame
#   envelope              : hard disc <-> soft gaussian edge
#   decay                 : how long a grain lingers (reverb tail), fading out
#   follow                : grains ride the motion while they live (dots trail moving things)
#   freeze                : stop recording into the buffer and granulate the held frames (el viento move)
#   dry / wet             : painted bed (the Girls 'Paint' look without dots) vs the grain cloud
#   mod_*                 : audio loudness modulates density / size / spray (like an envelope follower)
GRAIN = dict(
    # filter bank
    l_bands=6, s_bands=5,       # 30 bands, like Takagi
    rebuild=0.5,                # grain colour: band-average colour (1) vs the exact sampled pixel (0)
    paper=0.60,                 # source brightness (stretched) above which no grains spawn -> white paper
    knee_lo=0.12, knee_hi=0.08, # how gradually the bed fades to white paper: from paper-knee_lo to paper+knee_hi
    spawn_dark=1.0,             # relative grain density in the darkest tones (1 = uniform; <1 = darks stay continuous image)
    spawn_gamma=1.0,            # >1 = grains crowd toward the light edge where the image meets the paper
    sprinkle=0.0,               # relative grain density over white paper (0 = none: the hard cut; Girls has a few)
    sprinkle_ink=1.0,           # ink of the grains that land on paper when ink_flat is off (1 = their own colour)
    # grains
    density=2500,               # grains tried per processed picture; ones landing on white paper are dropped (bar the
                                # sprinkle), so about 70-80 % survive
    size_dark=5.5, size_light=1.2,  # grain radius (px at 640 wide: scaled with the width) in the darkest / lightest band
    size_curve=2.0,             # how fast size grows with darkness
    size_jitter=0.35,
    envelope=0.30,              # 0 = hard disc, 1 = soft gaussian
    ink=0.30,                   # how dark the darkest grains print vs their colour (0 = black ink, 1 = true colour);
                                # lighter grains shade back toward their colour
    ink_flat=0,                 # 1 = every grain gets the full ink (Girls: near-black cores even on small light dots)
    elong=0.0,                  # oval grains: stretched along the picture's edges, length/width = 1 + elong * edge
                                # strength (0 = round dots, the old look; Girls: about half the dots are ovals)
    elong_free=0.0,             # share of grains that get the full stretch in a random direction instead
    elong_bins=4,               # oval directions drawn (grains snap to the nearest; more = finer angles, slower)
    sat=2.2,                    # grain colour saturation
    # spray
    spray=1.5,                  # px at 640 wide: grain samples colour this far from where it lands
    time_spray=3.0,             # frames: grains sample from this far back in the buffer (|gaussian|)
    buffer=24,                  # video frames the buffer holds at most; only the newest is kept unless time spray or
                                # freeze is on
    # tail
    decay=3,                    # pictures a grain lasts while it fades (each held for the step: 3 at step 2 = a fifth of a
                                # second; 1 = no tail)
    follow=0.8,                 # how much living grains ride the local motion
    # freeze
    freeze_at=-1,               # input (video) frame index to freeze the buffer at (-1 = never; the app sends freeze_sec)
    freeze_len=3,               # frames held when frozen (granulate a tenth of a second)
    # bed (dry)
    dry=0.8,                    # painted bed under the grains (0 = grains on white paper)
    bed_sat=2.0, bed_bright=0.4, bed_brush=5,
    bed_blur=0.0,               # px: soften the painted bed (0 = off), so its fine detail doesn't repeat picture to picture
    wet=1.0,
    # output
    step=2.0,                   # stepped ~15 fps (2.5 = held 2,3,2,3 = 12 fps, like Girls)
    exposure=0.35,
    # audio modulation (needs --audio)
    mod_density=0.0, mod_size=0.0, mod_spray=0.0,
)


def _kernel(r, env):
    r = max(r, 0.5)
    R = int(np.ceil(r * (1 + env))) + 1
    y, x = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
    d = np.sqrt(x * x + y * y)
    hard = np.clip(r + 0.5 - d, 0, 1)
    soft = np.exp(-(d / max(r * 0.75, 0.4)) ** 2)
    k = hard * (1 - env) + soft * env
    return k / k.max()


def _kernel_aniso(r, env, a, th):
    """oval grain: the round grain's area stretched to length/width a, long axis at angle th (radians)."""
    r = max(r, 0.5); ra, rb = r * np.sqrt(a), r / np.sqrt(a)
    R = int(np.ceil(ra * (1 + env))) + 1
    y, x = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
    u = x * np.cos(th) + y * np.sin(th); v = -x * np.sin(th) + y * np.cos(th)
    d = np.sqrt((u / ra) ** 2 + (v / rb) ** 2) * r          # 'radius-equivalent' distance
    hard = np.clip(r + 0.5 - d, 0, 1)
    soft = np.exp(-(d / max(r * 0.75, 0.4)) ** 2)
    k = hard * (1 - env) + soft * env
    return k / k.max()


class Granulator:
    SIZE_CLASSES = 7

    def __init__(self, p, rng):
        self.p, self.rng = p, rng
        self.flow = Flow()
        self.buf = []          # list of (8-bit frame, V, band, band colours)
        self.last_f = None     # the newest recorded frame as float (for the bed)
        self.g = np.zeros((0, 12), np.float32)  # x y r life life0 b g r band darkness angle aspect
        self.held, self.hold, self.n = None, 0, 0
        self.cad = Cadence(p["step"])
        self.frozen = False
        self.audio = None      # per-input-frame loudness 0..1 (set by main)
        self.in_idx = 0

    def _analyse(self, bgr8):
        f = bgr8.astype(np.float32) / 255
        L = luma(f)
        V = np.power(_stretch(L, 2, 98), 1 - self.p["exposure"])
        S = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[..., 1]
        return f, V, S

    def _bands(self, f, V, S):
        p = self.p
        lb = np.clip((V * p["l_bands"]).astype(int), 0, p["l_bands"] - 1)
        sb = np.clip((S * p["s_bands"]).astype(int), 0, p["s_bands"] - 1)
        band = lb * p["s_bands"] + sb
        nb = p["l_bands"] * p["s_bands"]
        fl = band.ravel(); cnt = np.maximum(np.bincount(fl, minlength=nb), 1)
        cols = np.stack([np.bincount(fl, weights=f[..., c].ravel(), minlength=nb) / cnt for c in range(3)], 1)
        return band.astype(np.int16), cols.astype(np.float32)   # int16: same values, a quarter of the buffer's RAM

    def _bed(self, f, V):
        p = self.p
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * p["bed_sat"], 0, 1)
        hsv[..., 2] = np.power(np.clip(hsv[..., 2], 0, 1), (1 - p["bed_bright"]) * (1 - p["exposure"]))
        bed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        paper = smoothstep(p["paper"] - p["knee_lo"], p["paper"] + p["knee_hi"], V)[..., None]
        bed = bed * (1 - paper) + paper
        if p["bed_brush"] > 0:
            bed = brush_smudge(bed.astype(np.float32), luma(f), int(p["bed_brush"]), 1.6 * KS)
        if p["bed_blur"] > 0:   # after the paper edge too: that edge is much of what repeats (keeping it sharp undid half the gain)
            bed = cv2.GaussianBlur(bed, (0, 0), p["bed_blur"])
        return bed

    def frame(self, bgr8):
        p = self.p
        flow = self.flow(bgr8) if p["follow"] > 0 else None   # motion tracking only feeds follow
        amp = float(self.audio[min(self.in_idx, len(self.audio) - 1)]) if self.audio is not None and len(self.audio) else 0.0
        self.in_idx += 1
        # buffer (record unless frozen). 8-bit frames (exact: /255 on read); with no time spray and no freeze only the
        # newest frame is ever read, so only that one is kept
        if not self.frozen:
            f, V, S = self._analyse(bgr8)
            self.last_f = f
            self.buf.append((bgr8, V, *self._bands(f, V, S)))   # S is never read back
            keep = max(1, int(p["buffer"])) if (p["time_spray"] > 0 or p["freeze_at"] >= 0) else 1
            self.buf = self.buf[-keep:]
            # freeze: freeze_at is an input-frame index; checked after this frame is recorded, with freeze_len frames in hand
            fl = max(1, int(p["freeze_len"]))
            if p["freeze_at"] >= 0 and self.in_idx - 1 >= max(p["freeze_at"], fl - 1):
                self.frozen = True
                self.buf = self.buf[-fl:]
        # living grains ride the motion every input frame
        if len(self.g) and p["follow"] > 0:
            xi = np.clip(self.g[:, 0].astype(int), 0, flow.shape[1] - 1); yi = np.clip(self.g[:, 1].astype(int), 0, flow.shape[0] - 1)
            self.g[:, 0] += flow[yi, xi, 0] * p["follow"]; self.g[:, 1] += flow[yi, xi, 1] * p["follow"]
        if self.held is not None and self.hold > 0:
            self.hold -= 1
            return self.held
        self.n += 1
        h, w = bgr8.shape[:2]
        rng = self.rng

        # ---- spawn grains
        md = p["mod_density"] if self.audio is not None else 0.0   # no song = the Density mod is off, not silence (11-23 fix)
        dens = p["density"] * (1 + md * (amp * 2 - 1))
        n = max(0, int(dens))
        x = rng.uniform(0, w, n).astype(np.float32); y = rng.uniform(0, h, n).astype(np.float32)
        spray = p["spray"] * (1 + p["mod_spray"] * amp * 3)
        sx = np.clip(x + rng.normal(0, spray, n), 0, w - 1).astype(int)
        sy = np.clip(y + rng.normal(0, spray, n), 0, h - 1).astype(int)
        nbuf = len(self.buf)
        if self.frozen:
            ti = rng.integers(0, nbuf, n)  # granulate the held window
        else:
            back = np.abs(rng.normal(0, p["time_spray"], n)); over = back >= nbuf
            if p["time_spray"] > 0 and len(self.buf) >= int(p["buffer"]) and over.any():   # past the buffer: spread back over it
                back[over] = rng.uniform(0, nbuf, over.sum())                                # (was: all piled on the oldest frame)
            ti = nbuf - 1 - np.minimum(back.astype(int), nbuf - 1)
        cols = np.zeros((n, 3), np.float32); dark = np.zeros(n, np.float32); band = np.zeros(n, int)
        for t in np.unique(ti):
            m = ti == t
            b8, V, bnd, bcols = self.buf[t]
            px = b8[sy[m], sx[m]].astype(np.float32) / 255; bi = bnd[sy[m], sx[m]]
            cols[m] = bcols[bi] * p["rebuild"] + px * (1 - p["rebuild"])
            dark[m] = 1 - V[sy[m], sx[m]]
            band[m] = bi
        if p["spawn_dark"] < 1 or p["sprinkle"] > 0:   # weighted spawn: fewer grains in the darks, a few on the paper
            Vs = 1 - dark
            w_in = p["spawn_dark"] + (1 - p["spawn_dark"]) * np.clip(Vs / max(p["paper"], 1e-3), 0, 1) ** p["spawn_gamma"]
            keep = rng.uniform(0, 1, len(Vs)) < np.where(Vs < p["paper"], w_in, p["sprinkle"])
        else:
            keep = dark > (1 - p["paper"])
        x, y, cols, dark = x[keep], y[keep], cols[keep], dark[keep]
        t01 = np.clip((dark - (1 - p["paper"])) / max(p["paper"], 1e-3), 0, 1)
        r = p["size_light"] + (p["size_dark"] - p["size_light"]) * t01 ** (1 / max(p["size_curve"], 0.1))
        r = r * (1 + p["mod_size"] * amp) * rng.uniform(1 - p["size_jitter"], 1 + p["size_jitter"], len(r))
        # grain colour: saturate, then ink density (darker grains for darker bands)
        if len(x):   # a frame can spawn nothing (density mod at full on a silent moment) — just age the old grains
            hsv = cv2.cvtColor(cols[None], cv2.COLOR_BGR2HSV)[0]
            hsv[:, 1] = np.clip(hsv[:, 1] * p["sat"], 0, 1)
            if p["ink_flat"]:
                inkf = p["ink"]
            else:
                inkf = p["ink"] + (1 - p["ink"]) * (1 - t01)
                if p["sprinkle"] > 0:   # grains that landed on the paper: inked like the image's dots
                    inkf = np.where(dark < 1 - p["paper"], p["sprinkle_ink"], inkf)
            hsv[:, 2] = np.clip(hsv[:, 2] * inkf, 0, 1)
            cols = cv2.cvtColor(hsv[None], cv2.COLOR_HSV2BGR)[0]
        life = np.full(len(x), max(1, p["decay"]), np.float32)
        th = np.zeros(len(x), np.float32); asp = np.ones(len(x), np.float32)
        if p["elong"] > 0 and len(x):   # oval grains: long axis along the local edge (structure tensor of the newest frame)
            Lc = cv2.GaussianBlur(luma(self.last_f), (0, 0), 3 * KS)
            gx_ = cv2.Sobel(Lc, cv2.CV_32F, 1, 0); gy_ = cv2.Sobel(Lc, cv2.CV_32F, 0, 1)
            jxx = cv2.GaussianBlur(gx_ * gx_, (0, 0), 10 * KS); jyy = cv2.GaussianBlur(gy_ * gy_, (0, 0), 10 * KS)
            jxy = cv2.GaussianBlur(gx_ * gy_, (0, 0), 10 * KS)
            xi_ = np.clip(x.astype(int), 0, w - 1); yi_ = np.clip(y.astype(int), 0, h - 1)
            a_, b_, c_ = jxx[yi_, xi_], jyy[yi_, xi_], jxy[yi_, xi_]   # only at the grains, not the whole frame
            th = (0.5 * np.arctan2(2 * c_, a_ - b_) + np.pi / 2).astype(np.float32)
            asp = (1 + p["elong"] * np.sqrt((a_ - b_) ** 2 + 4 * c_ ** 2) / (a_ + b_ + 1e-9)).astype(np.float32)
            if p["elong_free"] > 0:   # the rest: full stretch, random direction
                fr = rng.uniform(0, 1, len(x)) < p["elong_free"]
                th = np.where(fr, rng.uniform(0, np.pi, len(x)), th).astype(np.float32)
                asp = np.where(fr, 1 + p["elong"], asp).astype(np.float32)
        new = np.column_stack([x, y, r, life, life, cols, np.zeros(len(x)), t01, th, asp]).astype(np.float32)
        self.g = np.vstack([self.g, new])
        self.g[:, 3] -= 1
        mg = 8 * KS   # off-screen margin
        self.g = self.g[(self.g[:, 3] > -1) & (self.g[:, 0] > -mg) & (self.g[:, 0] < w + mg) & (self.g[:, 1] > -mg) & (self.g[:, 1] < h + mg)]

        # ---- render grains: splat per size class, convolve with the grain envelope
        g = self.g
        env_amp = np.clip((g[:, 3] + 1) / g[:, 4], 0, 1)  # fade over the tail
        weight = env_amp * (1 + 3 * g[:, 9])             # darker grains win overlaps (drawn 'on top')
        rmin, rmax = max(0.5, g[:, 2].min()) if len(g) else 1, g[:, 2].max() if len(g) else 1
        edges = np.geomspace(rmin, rmax + 1e-3, self.SIZE_CLASSES + 1)
        C = np.zeros((h, w, 3), np.float32); A = np.zeros((h, w), np.float32); Wt = np.zeros((h, w), np.float32)
        xi = np.clip(np.round(g[:, 0]).astype(int), 0, w - 1); yi = np.clip(np.round(g[:, 1]).astype(int), 0, h - 1)
        # ovals (elong): one kernel per size class x direction bin, stamped straight at each grain and summed in one pass
        # below (the same sum filter2D gives, without a full-frame convolution per bin); nearly round grains stay discs
        NB = max(1, int(p["elong_bins"]))
        obin = np.where(g[:, 11] < 1.25, -1, (np.mod(g[:, 10], np.pi) / (np.pi / NB)).astype(int) % NB)
        # oval stamps are queued as (pixel, grain, kernel weight) entries and summed into these 5 maps every ~2M entries,
        # so memory stays flat however big the grains or the frame are (11-24 review: keeping every class's entries
        # until the end took 2.3 GB at 960 wide and 7 GB at 1920 with big grains)
        s_pix, s_row, s_ker, s_n, acc = [], [], [], [0], None
        vals = (g[:, 5] * weight, g[:, 6] * weight, g[:, 7] * weight, weight, env_amp)
        CHUNK = 2_000_000
        OVAL_FILTER_AT = 1.0   # stamps per frame pixel above which a bin is convolved instead (measured: faster at 960 and 1920)

        def flush():
            if not s_pix:
                return
            pix, row, kw = np.concatenate(s_pix), np.concatenate(s_row), np.concatenate(s_ker)
            s_pix.clear(); s_row.clear(); s_ker.clear(); s_n[0] = 0
            for j, v in enumerate(vals):
                acc[j] += np.bincount(pix, v[row] * kw, minlength=h * w)
        for k, o in [(k, o) for k in range(self.SIZE_CLASSES) for o in ([-1] + list(range(NB)) if p["elong"] > 0 else [-1])]:
            m = (g[:, 2] >= edges[k]) & (g[:, 2] < edges[k + 1]) if k < self.SIZE_CLASSES - 1 else (g[:, 2] >= edges[k])
            m = m & (obin == o)
            if not m.any():
                continue
            if o >= 0:
                if acc is None:
                    acc = np.zeros((5, h * w), np.float64); yi32, xi32 = yi.astype(np.int32), xi.astype(np.int32)
                ker = _kernel_aniso(float(np.sqrt(edges[k] * edges[k + 1])), p["envelope"], float(np.median(g[m, 11])), (o + 0.5) * np.pi / NB)
                R = ker.shape[0] // 2; oy, ox = np.mgrid[-R:R + 1, -R:R + 1]
                nz = ker.ravel() > 1e-3; kv = ker.ravel()[nz].astype(np.float32)   # the oval's own pixels, not the empty corners
                oyn = oy.ravel()[nz].astype(np.int32); oxn = ox.ravel()[nz].astype(np.int32)
                gi_all = np.nonzero(m)[0].astype(np.int32)
                if len(gi_all) * len(kv) > OVAL_FILTER_AT * h * w:   # lots of big ovals: splat + one convolution per map is faster
                    pix0 = yi32[gi_all] * w + xi32[gi_all]           # (an oval is point-symmetric: filter2D = the stamp sum)
                    for j, v in enumerate(vals):
                        sp = np.bincount(pix0, v[gi_all], minlength=h * w).reshape(h, w).astype(np.float32)
                        acc[j] += cv2.filter2D(sp, -1, ker, borderType=cv2.BORDER_CONSTANT).ravel()
                    continue
                step_g = max(1, CHUNK // len(kv))
                for s0 in range(0, len(gi_all), step_g):   # a slice of the grains at a time: at most ~CHUNK entries
                    gi = gi_all[s0:s0 + step_g]
                    Y = yi32[gi, None] + oyn; X = xi32[gi, None] + oxn   # int32: half the memory of int64
                    ok = (Y >= 0) & (Y < h) & (X >= 0) & (X < w)
                    s_pix.append((Y * w + X)[ok]); s_row.append(np.broadcast_to(gi[:, None], ok.shape)[ok])
                    s_ker.append(np.broadcast_to(kv, ok.shape)[ok])
                    del Y, X, ok
                    s_n[0] += len(s_pix[-1])
                    if s_n[0] >= CHUNK:
                        flush()
                continue
            pc = np.zeros((h, w, 3), np.float32); pa = np.zeros((h, w), np.float32); pw = np.zeros((h, w), np.float32)
            for c in range(3):
                np.add.at(pc[..., c], (yi[m], xi[m]), g[m, 5 + c] * weight[m])
            np.add.at(pw, (yi[m], xi[m]), weight[m])
            np.add.at(pa, (yi[m], xi[m]), env_amp[m])
            ker = _kernel(float(np.sqrt(edges[k] * edges[k + 1])), p["envelope"])
            C += cv2.filter2D(pc, -1, ker, borderType=cv2.BORDER_CONSTANT)
            Wt += cv2.filter2D(pw, -1, ker, borderType=cv2.BORDER_CONSTANT)
            A += cv2.filter2D(pa, -1, ker, borderType=cv2.BORDER_CONSTANT)
        if acc is not None:
            flush()
            for c in range(3):
                C[..., c] += acc[c].reshape(h, w).astype(np.float32)
            Wt += acc[3].reshape(h, w).astype(np.float32); A += acc[4].reshape(h, w).astype(np.float32)
        gcol = C / np.maximum(Wt, 1e-6)[..., None]
        cov = np.clip(A, 0, 1)[..., None] * p["wet"]

        f, V = self.last_f, self.buf[-1][1]
        bed = self._bed(f, V) if p["dry"] > 0 else np.ones_like(f)
        bed = bed * p["dry"] + (1 - p["dry"])
        img = bed * (1 - cov) + gcol * cov
        out = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        self.held, self.hold = out, self.cad.next_hold()
        return out


def audio_envelope(path, fps, n_frames, start=0.0):
    """per-video-frame loudness 0..1 of a song from `start` seconds (ffmpeg decode)."""
    sr = 22050
    raw = subprocess.run([FF, "-v", "error", "-nostdin", "-ss", f"{start:.3f}", "-i", path, "-t", f"{n_frames / fps + 1:.3f}",
                          "-map", "0:a:0", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"], capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    hop = sr / fps
    env = np.zeros(n_frames)
    for i in range(n_frames):
        seg = a[int(i * hop):int((i + 1) * hop)]
        env[i] = np.sqrt(np.mean(seg ** 2) + 1e-12) if len(seg) else 0.0
    env = np.convolve(env, np.ones(3) / 3, mode="same")
    lo, hi = np.percentile(env, 5), np.percentile(env, 98)
    return np.clip((env - lo) / max(hi - lo, 1e-6), 0, 1)




# ================================================================ PHOTOSHOP FILTER GALLERY (11-25)
# New Flat (2003): "Photoshop effects forced into After Effects". Which filters he used is unknown, so four Filter
# Gallery effects are offered, each with its own three Photoshop settings in Photoshop's units (pixels at render size).
# Measured against Photoshop 2026's own renders (~/Projects/takagi-fx/verify/ps): Glowing Edges and Poster Edges are
# maths; Dry Brush and Watercolor are rendered by Photoshop itself when it's installed (exact by definition).
# All run on the look's finished picture, after Echo.
from scipy.ndimage import rank_filter as _rank_filter


def ps_glowing_edges(bgr8, W=2, B=6, S=5):
    """Stylize > Glowing Edges (Edge Width, Edge Brightness, Smoothness). Fitted 11-25 on 70 renders: 0.04 levels mean,
    99.7 % of pixels within 1 level (the rest in the outer border strip when Smoothness > 1). Per channel: S x S median
    (the lower middle value for even S) -> brightest minus darkest in a (W+1) square reaching ceil(W/2) back and
    floor(W/2) forward -> x B/2, rounded."""
    W, B, S = max(1, int(W)), max(0, int(B)), max(1, int(S))
    f = bgr8.astype(np.float32)
    if S > 1: f = _rank_filter(f, (S * S - 1) // 2, size=(S, S, 1), mode="nearest")
    k, a = W + 1, (W + 1) // 2; ker = np.ones((k, k), np.uint8)
    g = cv2.dilate(f, ker, anchor=(a, a), borderType=cv2.BORDER_REPLICATE) - cv2.erode(f, ker, anchor=(a, a), borderType=cv2.BORDER_REPLICATE)
    return np.clip(np.floor(g * B / 2 + 0.5), 0, 255).astype(np.uint8)


_PE = dict(thr=0.600, g0=2.484, g1=67.34)   # line constants (see ps_poster_edges: partly fitted)


def _box_mean(a, k, anc):
    p = cv2.copyMakeBorder(a, anc, k - 1 - anc, anc, k - 1 - anc, cv2.BORDER_REPLICATE)
    c = cv2.integral(p); h, w = a.shape
    return (c[k:k + h, k:k + w] - c[0:h, k:k + w] - c[k:k + h, 0:w] + c[0:h, 0:w]) / (k * k)


def ps_poster_edges(bgr8, T=2, I=1, P=2):
    """Artistic > Poster Edges (Edge Thickness, Edge Intensity, Posterization). PARTLY fitted (11-25): the softening
    (3x3 average, rounded down) and posterising (on V, the brightest channel, step 256 >> (P+1), halves up; colour kept
    by scaling all channels with V'/V) are exact on every grey test; the dark lines are fitted to step edges but not yet
    to real texture (real frames ~7 levels mean)."""
    T, I, P = max(0, int(T)), max(0, int(I)), min(6, max(0, int(P)))
    f = np.floor(cv2.blur(bgr8.astype(np.float64), (3, 3), borderType=cv2.BORDER_REPLICATE) + 1e-9)
    V = f.max(2); st = 256 >> (P + 1); Vq = np.clip(np.floor(V / st + 0.5) * st, 0, 255)
    k = T + 5; line = np.maximum(_box_mean(V, k, k // 2) - V - _PE["thr"], 0) * (_PE["g0"] + _PE["g1"] * I / (T + 1))
    Vo = np.clip(Vq - line, 0, 255)
    return np.clip(np.floor(f * (Vo / np.maximum(V, 1e-9))[..., None] + 0.5), 0, 255).astype(np.uint8)


PS_APP = "Adobe Photoshop 2026"
PS_APP_PATH = Path("/Applications/Adobe Photoshop 2026/Adobe Photoshop 2026.app")


class PhotoshopFilter:
    """A Filter Gallery effect rendered by Photoshop itself (exact): each new picture goes out as a PNG, Photoshop runs the
    filter without its dialog (executeAction "GEfc" with the effect's 4-letter ID) and the result comes back. ~0.55 s a
    picture at 960 wide. A held frame (the same picture again) reuses the last result. Needs Photoshop 2026."""
    KINDS = {"poster_edges": ("PstE", ("EdgT", "EdgI", "Pstr"), "Poster Edges"), "dry_brush": ("DryB", ("BrsS", "BrsD", "Txtr"), "Dry Brush"), "watercolor": ("Wtrc", ("BrsD", "ShdI", "Txtr"), "Watercolor")}
    JSX = r"""#target photoshop
app.displayDialogs = DialogModes.NO;
var d = "%DIR%/", st = new File(d + "status.txt"), msg = "ok", doc = null;
try {
    doc = app.open(new File(d + "in.png"));
    if (doc.mode != DocumentMode.RGB) doc.changeMode(ChangeMode.RGB);
    doc.bitsPerChannel = BitsPerChannelType.EIGHT;
    var a = new ActionDescriptor();
    a.putEnumerated(charIDToTypeID("GEfk"), charIDToTypeID("GEft"), charIDToTypeID("%KIND%"));
    %PARAMS%
    executeAction(charIDToTypeID("GEfc"), a, DialogModes.NO);
    var o = new PNGSaveOptions(); o.compression = 1;
    doc.saveAs(new File(d + "out.png"), o, true, Extension.LOWERCASE);
} catch (e) { msg = "error " + e; }
if (doc) doc.close(SaveOptions.DONOTSAVECHANGES);
st.open("w"); st.write(msg); st.close();
"""

    def __init__(self, name, params):
        import tempfile
        kind, keys, self.label = self.KINDS[name]
        if not PS_APP_PATH.exists():
            sys.exit(f"[error] {self.label} renders through Photoshop 2026, which isn't installed on this Mac")
        self.dir = Path(tempfile.mkdtemp(prefix="liminal-ps-"))
        body = "\n    ".join(f'a.putInteger(charIDToTypeID("{k}"), {int(v)});' for k, v in zip(keys, params))
        (self.dir / "run.jsx").write_text(self.JSX.replace("%DIR%", str(self.dir)).replace("%KIND%", kind).replace("%PARAMS%", body))
        self.last_in, self.last_out = None, None
        if subprocess.run(["pgrep", "-f", "Adobe Photoshop 2026.app/Contents/MacOS/Adobe Photoshop 2026"], capture_output=True).returncode != 0:
            print(f"[filter] opening Photoshop for {self.label}", flush=True)
            subprocess.run(["open", "-a", PS_APP])
        for _ in range(60):   # wait until it answers scripts
            r = subprocess.run(["osascript", "-e", "with timeout of 10 seconds", "-e", f'tell application "{PS_APP}" to do javascript "1"', "-e", "end timeout"],
                               capture_output=True, text=True)
            if r.returncode == 0: break
            time.sleep(2)
        else:
            sys.exit(f"[error] Photoshop didn't answer: {self.label} can't render")

    def __call__(self, bgr8):
        if self.last_in is not None and np.array_equal(bgr8, self.last_in): return self.last_out
        cv2.imwrite(str(self.dir / "in.png"), bgr8, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        (self.dir / "status.txt").write_text("")
        r = subprocess.run(["osascript", "-e", "with timeout of 300 seconds", "-e",
                            f'tell application "{PS_APP}" to do javascript (read POSIX file "{self.dir / "run.jsx"}")', "-e", "end timeout"],
                           capture_output=True, text=True)
        status = (self.dir / "status.txt").read_text().strip() if (self.dir / "status.txt").exists() else ""
        out = cv2.imread(str(self.dir / "out.png")) if status == "ok" else None
        if out is None or out.shape != bgr8.shape:
            hint = " (scratch disk full or missing? check Photoshop > Settings > Scratch Disks)" if "open options" in status or "scratch" in status else ""
            sys.exit(f"[error] Photoshop couldn't render {self.label}: {status or r.stderr.strip()[:200]}{hint}")
        self.last_in, self.last_out = bgr8.copy(), out
        return out

    def __del__(self):
        try: shutil.rmtree(self.dir, ignore_errors=True)   # its own temp folder
        except Exception: pass


# ps_filter: 0 none, 1 Glowing Edges, 2 Poster Edges, 3 Dry Brush (Photoshop), 4 Watercolor (Photoshop); ps_p1..3 = its settings
PS_NAMES = {1: "glowing_edges", 2: "poster_edges", 3: "dry_brush", 4: "watercolor"}
PS_DEFAULTS = {"glowing_edges": (2, 6, 5), "poster_edges": (2, 1, 2), "dry_brush": (2, 8, 1), "watercolor": (14, 1, 1)}
PS_RANGES = {"glowing_edges": ((1, 14), (0, 20), (1, 15)), "poster_edges": ((0, 10), (0, 10), (0, 6)),
             "dry_brush": ((0, 10), (0, 10), (1, 3)), "watercolor": ((1, 14), (0, 10), (1, 3))}


def ps_filter_from(p):
    """the Photoshop filter a render asked for (ps_filter = code or name, ps_p1..3), or None."""
    v = p.get("ps_filter", 0)
    try: name = PS_NAMES.get(int(float(v)))
    except (TypeError, ValueError): name = str(v) if str(v) in PS_DEFAULTS else None
    if not name: return None
    prm = [int(round(float(p.get(f"ps_p{n + 1}", d)))) for n, d in enumerate(PS_DEFAULTS[name])]
    prm = [max(lo, min(hi, x)) for x, (lo, hi) in zip(prm, PS_RANGES[name])]
    print(f"[filter] {name} {prm}", flush=True)
    if name == "glowing_edges": return lambda im: ps_glowing_edges(im, *prm)
    if name == "poster_edges" and not PS_APP_PATH.exists():   # no Photoshop: the partly-fitted maths (lines approximate)
        print("[filter] Poster Edges: Photoshop isn't installed, using the approximate version", flush=True)
        return lambda im: ps_poster_edges(im, *prm)
    return PhotoshopFilter(name, prm)


# ================================================================ ECHO (11-25)
# Trails from blending the last N pictures the look made (not the footage, so it stays "made new"). Note: AE's own
# Echo samples the layer BEFORE its effects; running on the formed output matches a precomposed layer instead.
# Operator maths MEASURED against AE 2027 (11-25, ~/Projects/takagi-fx/verify/ae/echo_model.py: 12 renders, every operator
# within 1 level): the current picture is a layer at opacity = Starting Intensity; echo i (newest first) goes on at
# opacity start*decay^i with Adobe's blend-with-alpha; Add sums brightness and caps at the new opacity each step; AE's
# "Blend" gives the same as Add on solid footage, so the page offers Composite In Back instead. Below 1, the picture is seen over black (AE leaves it see-through).
class Echo:
    def __init__(self, echoes=3, time=0.033, start=1.0, decay=0.5, operator="add"):
        self.n = max(0, int(echoes))
        self.dt = time          # not used (we work in frames, and the look's cadence sets the rate)
        self.start = max(0.0, min(2.0, float(start)))
        self.decay = max(0.0, min(1.0, float(decay)))
        self.op = operator.lower()
        self.buf = []           # the last N+1 formed frames

    def apply(self, img):
        """img: the look's formed output (float32 0..1 or uint8). Returns the echoed picture, same type."""
        was_uint8 = img.dtype == np.uint8
        f = img.astype(np.float32) / 255 if was_uint8 else img.astype(np.float32)
        self.buf.append(f.copy())
        if len(self.buf) > self.n + 1: self.buf = self.buf[-(self.n + 1):]
        if self.n == 0: return img
        frames = list(reversed(self.buf))   # newest first
        ab = min(1.0, self.start); cp = frames[0] * ab          # premultiplied result + its opacity
        for i, e in enumerate(frames[1:], 1):
            a = min(1.0, self.start * self.decay ** i); an = a + ab - a * ab
            if self.op in ("add", "blend"):
                cp = np.minimum(cp + a * e, an)
            else:
                Cb = cp / ab if ab > 0 else np.zeros_like(cp)
                if self.op == "maximum": m = np.maximum(Cb, e)
                elif self.op == "minimum": m = np.minimum(Cb, e)
                elif self.op == "screen": m = Cb + e - Cb * e
                elif self.op == "comp_back": m = e      # Composite In Back: older pictures show through at their opacity
                else: m = Cb                             # Composite In Front: the current picture covers them
                cp = e * a * (1 - ab) + cp * (1 - a) + a * ab * m
            ab = an
        return np.clip(np.floor(cp * 255 + 0.5), 0, 255).astype(np.uint8) if was_uint8 else np.clip(cp, 0, 1)


LOOKS = {"granular": (Granulator, GRAIN), "paint": (Paint, PAINT), "outline": (Rama2, RAMA2), "light": (LightBodies, LIGHT)}

# "Takagi strict" (11-23): what frame-by-frame measurement of his videos showed (~/reports/Takagi method gaps.md).
# Girls: 12 fps held 2,3,2,3; a new dot pattern every processed frame; dots never smear along motion (only along
# edges); round dots with near-black cores and soft grey halos; sizes vary a lot. Rama: colour at 15 fps, lines held
# 6 frames; lines ~35 % of the colour under them (never black); no posterize; DV 4:1:1 chroma.
# Stage 1 of the 11-23 double audit (~/reports/Granular tab double audit.md, section 3): granular gets flat near-black
# ink and a few dots on the paper; paint crisp dot masks, a brighter base and more paper; outline his brightness
# curve (tone), no lines from faint edges (edge_floor) and a weaker hue wheel.
# Fix pass (11-24, all 3 looks rendered together on slide/tree/green vs his frames): paint cuts its crisp dots from noise
# smudged along the grain (noise_brush), so they follow edges again (edge alignment 0.38-0.54, his 0.39-0.45; crisp
# masks alone fell to 0.11-0.19), with finer noise to keep the dot count and lighter cores (0.05-0.06, his 0.04-0.08);
# outline gets more vivid colour that the tone curve keeps (col_sat, tone_sat): pixels above saturation 200 0.22-0.30
# (his 0.25-0.30; tone alone washed it pale pink at 0.01), brightness 0.53-0.60 (his 0.57-0.63).
# Eric's picks stay the defaults; strict goes on first and any setting sent with it still wins.
STRICT = {
    # granular stage 1 (11-24, tuned together on slide/tree/green vs his frames): spawn_dark stays 1 (0.4 left it
    # too light: black 0.06 vs his 0.12), knee 0.20/0.20 (0.25/0.0 gave too much paper), sprinkle 0.35 for his paper dots.
    # Stage 2 E1 (11-24): oval grains along the edges. elong_free 0.3, not the audit's 0.5: on this stage-1 strict 0.5
    # overshot (stretched share 0.62-0.72, edge alignment 0.28-0.36); 0.3 gives 0.48-0.60 / 0.38-0.47 (his 0.55-0.59 / 0.39-0.45)
    # Stage 2 E2 (11-24): a softer painted bed (bed_blur 3 px), so still shots stop carrying the bed's fine detail from
    # picture to picture (his fine-band match between pictures is near zero: the dots carry the image). Slide (still
    # shot), YouTube-sim, seeds 3/7: 0.5-1.5 px 0.038 -> 0.018/0.020, 1-2 px 0.071 -> 0.038/0.042 (his 0.009-0.020 /
    # 0.034-0.049), brightness and colour unchanged (Y 0.50, S 0.45; his 0.506 / 0.446). The audit's dry 0.5 + bed_sat 3
    # (thinning the bed toward white) reached only 0.021/0.022 and washed it out: Y 0.58, S 0.33. Costs: 2-4 px lands at
    # his low edge (0.148-0.154, his 0.153-0.178), a little less white paper (0.18 vs 0.20, his 0.21), and the flat green
    # clip drops further below his fine-band range (0.007-0.010; it was already at his low end).
    "granular": dict(step=2.5, decay=1, follow=0.0, time_spray=0.0, density=3000, envelope=0.5, ink=0.10,
                     size_jitter=0.55, ink_flat=1, sprinkle=0.35, size_light=1.5, knee_lo=0.20, knee_hi=0.20,
                     elong=0.8, elong_free=0.3, bed_blur=3.0),
    # paint stage 2 E4 (11-24): round dots (dots) in the lights and out on the paper, each a Gaussian splat cut by levels
    # (near-black centre, soft grey rim) with ragged edges, 30 % of them grey, the darker dot winning where they overlap;
    # the noise texture still prints the darks. Gated against stage-1 strict, means over seeds 3/7/11 on slide/tree/
    # green: dots out on the paper 59/32/44 -> 97/57/80 a frame (his 92-124), grey share inside the dots 0.30/0.26/0.33
    # -> 0.35/0.30/0.39 (his 0.59-0.70), roundness 0.47/0.47/0.46 -> 0.45/0.46/0.44 (his 0.31-0.49), solidity 0.79/0.80/
    # 0.80 -> 0.78/0.78/0.77 (his 0.67-0.82), clumping 0.368/0.583/0.483 -> 0.367/0.586/0.482 (his 0.51-0.56).
    # dot_top 0.3 keeps dots off the mid-tones and ink_gamma 1.22 (from 1.25) a touch more dark mass: without them the
    # extra isolated dots thinned the clumping (green 0.48 -> 0.42). The audit's variant F + Gaussian halo as tested
    # (dots up to 0.92 darkness, merge 0) gave 135/67/101 paper dots but perfect ellipses (roundness 0.60-0.65) and
    # clumping 0.21-0.38. Cost: black 0.259 -> 0.265 (his 0.18-0.23), brightness down 0.006, strict paint
    # 28 -> 37 ms a frame at 640.
    "paint": dict(step=2.5, refresh=1, motion_streak=0, smear=0.0, detail=0.0, ink_dark=0.12, ink_soft=0.06,
                  mask_brush=0, noise_brush=2, fine=0.7, mid=1.4, coarse=2.8, exposure=0.45, paper_lo=0.50, paper_hi=0.78,
                  dots=1, dot_density=5.0, dot_floor=0.5, dot_spread=20.0, dot_r=1.8, dot_cv=0.6, dot_stretch=2.0,
                  dot_gauss=1.0, dot_rough=1.0, dot_grey=0.3, dot_top=0.3, merge=0.0, ink_gamma=1.22),
    # outline stage 2 E3 (11-24): colour from his measured maps (cmap), read by brightness rank, a new map at each cut.
    # Measured over all 12 maps x slide/tree/green (lines after a YouTube-style re-encode, as his frames are): red-clipped
    # share 0.38 -> 0.49 (his 0.51), hue agreement 0.63 -> 0.96 (his 0.93), brightness 0.58 -> 0.60 (his 0.63), line
    # area 0.037 -> 0.069 / 0.084 -> 0.112 (his 0.062 / 0.112), line depth 0.70 -> 0.47 of the colour (his 0.42).
    # tone and col_sat are off: on top of the maps they pull it away (hue agreement 0.86, red 0.38, saturated 0.74).
    # The audit's pre_blur 1.6 + line strength 1.5 left lines half his area and wider (FWHM 4.75 px vs his 3; 4 here),
    # so lines keep pre_blur 0.8 and strength 3, set by line_gain (the app always sends the Lines slider's edge_gain).
    "outline": dict(step=2.0, edge_step=6, edge_mix=0.65, ghost=0.0, dv=1, cmap=1, line_gain=3.0, edge_floor=0.35,
                    wheel=0.0),
    "light": {},   # nothing measured for strict yet (11-24): the look is already built from AE-measured effects
}


# ================================================================ the clip in, the video out
def _rate(s):
    try:
        a, _, b = str(s).partition("/")
        v = float(a) / float(b or 1)
        return v if v > 0 and np.isfinite(v) else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe(path):
    """size (as displayed: rotation and non-square pixels applied), frame rate, length, and whether it has sound."""
    r = subprocess.run([FP, "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,avg_frame_rate,r_frame_rate,sample_aspect_ratio:stream_tags=rotate:stream_side_data=rotation:format=duration",
                        "-of", "json", path], capture_output=True, text=True, timeout=60)
    try: info = json.loads(r.stdout or "{}")
    except ValueError: info = {}
    st = (info.get("streams") or [{}])[0]
    if not st.get("width"):
        return None
    w, h = int(st["width"]), int(st["height"])
    rot = 0
    try: rot = int(float((st.get("tags") or {}).get("rotate", 0)))
    except ValueError: pass
    for sd in st.get("side_data_list") or []:
        if "rotation" in sd:
            try: rot = int(float(sd["rotation"]))
            except ValueError: pass
    sar = _rate((st.get("sample_aspect_ratio") or "1:1").replace(":", "/")) or 1.0
    dw, dh = w * sar, h
    if abs(rot) % 180 == 90:
        dw, dh = dh, dw
    fr = st.get("avg_frame_rate") if _rate(st.get("avg_frame_rate")) else st.get("r_frame_rate")
    try: dur = float((info.get("format") or {}).get("duration") or 0)
    except ValueError: dur = 0.0
    a = subprocess.run([FP, "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", path],
                       capture_output=True, text=True, timeout=60)
    return {"dw": dw, "dh": dh, "fps_str": fr or "30000/1001", "fps": _rate(fr), "dur": dur, "audio": bool(a.stdout.strip())}


def even(x):
    return max(2, int(round(x / 2)) * 2)


PREVIEW_GAP = 2   # video frames at least between two saved previews: at most ~15 a second (a step-1 look makes 30)


def preview_rate(step, fps, n=3000):
    """previews a second the render loop saves: one per new picture (the look's cadence), no closer than PREVIEW_GAP."""
    cad, i, last, saved = Cadence(step), 0, -PREVIEW_GAP, 0
    while i < n:
        if i - last >= PREVIEW_GAP: saved, last = saved + 1, i
        i += cad.next_hold() + 1
    return fps * saved / n


STILL_WARM = {"granular": 24, "paint": 12, "outline": 8, "light": 10}   # video frames run before a still


def still(a, cls, p, seed, src, info, vf, fps_str, fps, W, H, t0):
    """One picture of the look at a.still seconds, for the page's preview (11-25, Granular layout C). The look runs over
    a short stretch before that moment (STILL_WARM), so tails, frame holds and the steadied cut-out have built up.
    Granular and Paint carry state over many frames, so the picture is close to the render's frame there, not identical.
    No sound: Audio mod is left out. Freeze is left out (it's counted from the render's start)."""
    if "freeze_at" in p: p["freeze_at"] = -1
    dur = info["dur"] or (a.still + 1)
    at = max(0.0, min(a.still, dur - 2 / fps))
    s0 = max(0.0, at - STILL_WARM.get(a.look, 8) / fps)
    n = int(round((at - s0) * fps)) + 1
    r = subprocess.run([FF, "-v", "error", "-nostdin", "-ss", f"{s0:.3f}", "-i", src, "-frames:v", str(n), "-map", "0:v:0", "-an", "-sn",
                        "-vf", ",".join(vf), "-r", fps_str, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, timeout=300)
    size = W * H * 3
    k = len(r.stdout) // size
    if k == 0: sys.exit("[error] no frames could be read from the clip there")
    fx = cls(p, np.random.default_rng(seed))
    echo = Echo(int(p.get("echo", 0)), decay=float(p.get("echo_decay", 0.5)),
                operator=["add", "maximum", "minimum", "screen", "comp_back"][min(4, max(0, int(p.get("echo_op", 0))))]) if p.get("echo", 0) else None
    ps_fn = ps_filter_from(p)
    fr = res = None
    for j in range(k):
        fr = np.frombuffer(r.stdout, np.uint8, size, j * size).reshape(H, W, 3).copy()
        res = fx.frame(fr)
        if echo: res = echo.apply(res)
        if ps_fn: res = ps_fn(res)
    out = Path(a.out)
    for img, path in ((res, out), (fr, out.with_name(out.stem + "_src.jpg"))):   # written aside and renamed in
        tmp = path.with_name(".next_" + path.name)
        if not cv2.imwrite(str(tmp), img, [cv2.IMWRITE_JPEG_QUALITY, 88]): sys.exit("[error] couldn't write the picture")
        os.replace(tmp, path)
    print(f"[still] {out} ({k} frames from {s0:.2f} s, {time.time() - t0:.1f} s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description="GRANULAR: one clip through Takagi-style looks")
    ap.add_argument("--source", required=True, help="the clip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--look", choices=list(LOOKS), default="granular")
    ap.add_argument("--audio", default="", help="a song: replaces the clip's sound and drives mod_* (granular)")
    ap.add_argument("--song_start", type=float, default=0.0, help="seconds into the song")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the clip")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds (0 = to the end of the clip)")
    ap.add_argument("--width", type=int, default=960, help="largest output width")
    ap.add_argument("--seed", type=int, default=3, help="3 = the seed of the 11-23 test renders Eric picked from")
    ap.add_argument("--preview", type=int, default=10, help="0 = no previews; any other number = a small preview JPEG of every new picture (at most ~15 a second)")
    ap.add_argument("--still", type=float, default=-1.0, help="seconds into the clip: write one picture of the look there "
                    "(--out, a .jpg) and the clip's own frame next to it (_src.jpg), no video. -1 = render the video")
    ap.add_argument("--set", action="extend", nargs="+", default=[], metavar="KEY=VALUE",
                    help="look settings; granular also takes freeze_sec=S (freeze the buffer S seconds in, -1 = off)")
    a = ap.parse_args()
    if not FF or not FP:
        sys.exit("[error] ffmpeg isn't installed (brew install ffmpeg)")
    t0 = time.time()

    for k in ("start", "duration", "song_start"):
        if not math.isfinite(getattr(a, k)): sys.exit(f"[error] --{k} needs a finite number")
    seed = max(0, a.seed)   # numpy refuses a negative seed

    cls, defaults = LOOKS[a.look]
    p, freeze_sec = dict(defaults), None
    # --set: every value must be a finite number; strict goes on first, then the values sent win over it
    vals, strict = [], False
    for kv in a.set:
        k, eq, v = kv.partition("=")
        if k == "ps_filter" and not v.replace(".", "", 1).lstrip("-").isdigit(): p["ps_filter"] = v; continue   # a name (or a code number, below)
        try: x = float(v)
        except ValueError: sys.exit(f"[error] {k} needs a number, got '{v}'")
        if not math.isfinite(x): sys.exit(f"[error] {k} needs a finite number, got '{v}'")
        if k == "strict": strict = x != 0; continue
        if k == "freeze_sec" and a.look == "granular": freeze_sec = x; continue
        if k.startswith("echo") or k.startswith("ps_"): vals.append((k, x)); continue   # Echo + PS filters: shared
        if not eq or k not in defaults: sys.exit(f"[error] '{k}' is not a setting of the {a.look} look")
        vals.append((k, x))
    if strict:
        p.update(STRICT[a.look])
    for k, x in vals:
        if k.startswith(("echo", "ps_")): continue   # handled after this loop (not look settings)
        p[k] = int(x) if isinstance(defaults[k], int) else x
    for k in ("l_bands", "s_bands"):
        if k in p: p[k] = max(1, p[k])
    if "palette" in p:   # one of his 9 palettes, or with cmap (strict) one of the 12 measured maps
        p["palette"] = max(-1, min((len(RAMA_MAPS) if p.get("cmap") else len(PALETTES)) - 1, p["palette"]))
    if "colorama" in p:   # AE-exact options (11-24): a preset number, and 0 / 1 / 2 for the lines
        p["colorama"] = max(0, min(LIGHT_BLUE if a.look == "light" else len(COLORAMA), p["colorama"]))
    if "ae_lines" in p: p["ae_lines"] = max(0, min(2, p["ae_lines"]))
    if "cutout" in p: p["cutout"] = max(0, min(2, p["cutout"]))
    if "steady" in p: p["steady"] = max(0.0, min(0.9, float(p["steady"])))   # 1 would freeze the first cut-out forever
    if a.look == "light":   # 11-25 formed Light: physical bounds (a lens can't scatter more light than it gets)
        p["glow"] = max(0.0, min(0.9, float(p["glow"]))); p["reach"] = max(0.0, min(0.8, float(p["reach"])))
        p["streaks"] = max(0.0, min(1.0, float(p["streaks"]))); p["shade"] = max(0.0, min(1.0, float(p["shade"])))
    # Echo is shared (not in any look's defaults): set it on p after the strict pass
    for k, x in vals:
        if k.startswith("echo"): p[k] = int(x) if k in ("echo", "echo_op") else x
        elif k.startswith("ps_"): p[k] = int(round(x))
    sent = {k for k, _ in vals if k in defaults and p[k] != STRICT[a.look].get(k)} if strict else set()   # a sent value beat the strict one

    src = str(Path(a.source).expanduser())
    if not Path(src).is_file(): sys.exit(f"[error] the clip isn't there any more: {src}")
    info = probe(src)
    if not info: sys.exit("[error] that file has no picture ffmpeg can read")
    if info["fps"] < 4: sys.exit("[error] couldn't read the clip's frame rate")
    vf, fps_str, fps = [], info["fps_str"], info["fps"]
    if fps > 31:   # tuned on 29.97 (DV / Xacti): step 2 = ~15 fps, decay/follow in frames
        fps_str, fps = "30000/1001", 30000 / 1001
        vf.append("fps=30000/1001")
    if fps < 29:   # 24/25p keep their own rate; settings counted in video frames were tuned at 29.97, so scale them
        r = fps / (30000 / 1001)   # (decay and refresh count processed pictures: left alone)
        st = p["step"] * r
        if abs(st - round(st)) < 0.02: st = float(round(st))
        p["step"] = max(1.0, st)
        if "time_spray" in p: p["time_spray"] *= r
        for k in ("buffer", "freeze_len", "edge_step"):
            if k in p and p[k] > 0: p[k] = max(1, int(round(p[k] * r)))
        print(f"[note] {fps:.2f} fps clip: frame-counted settings scaled x{r:.3f} (step {p['step']:.3f})", flush=True)
    start = max(0.0, a.start)
    if info["dur"] and info["dur"] <= 0.2:
        sys.exit(f"[error] that clip is only {info['dur']:.2f} s long — too short, pick a longer one")
    if info["dur"] and start >= info["dur"] - 0.2:
        sys.exit(f"[error] the clip is only {info['dur']:.1f} s long — start earlier than that")
    left = (info["dur"] - start) if info["dur"] else (a.duration or 15)
    length = min(left, a.duration) if a.duration > 0 else left
    maxw = max(64, a.width)
    if info["dw"] > maxw: W, H = even(maxw), even(info["dh"] * maxw / info["dw"])
    else: W, H = even(info["dw"]), even(info["dh"])
    vf.append(f"scale={W}:{H}:flags=area,setsar=1")
    global KS
    KS = W / 640   # sizes in px were tuned at 640 wide: an HD render keeps the same look, not finer dots
    if KS != 1.0:
        for k in {"granular": ("size_dark", "size_light", "spray", "bed_blur"),
                  "paint": ("fine", "mid", "coarse", "brush_len", "pre_blur", "soft", "dot_r", "dot_spread", "dot_halo")}.get(a.look, ()):
            p[k] = p[k] * KS
    if strict:   # the values actually used: after a sent value beats the strict one (marked), and after the fps and width scaling
        shown = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)
        print(f"[look] Takagi strict: {', '.join(f'{k}={shown(p[k])}' + (' (sent)' if k in sent else '') for k in STRICT[a.look])}", flush=True)
    N = max(1, int(round(length * fps)))
    print(f"[clip] {N} frames ({length:.1f} s @ {fps:.2f} fps, {W}x{H}, look={a.look}, seed={seed})", flush=True)
    if a.preview > 0:   # the rate the page plays the previews back at (said early, before the song is read)
        print(f"[preview] {preview_rate(p.get('step', 1), fps):.3f} per s", flush=True)

    if freeze_sec is not None:   # counted in video frames, from the start of the render
        p["freeze_at"] = int(round(freeze_sec * fps)) if freeze_sec >= 0 else -1
    ps_fn = ps_filter_from(p)
    if a.still >= 0:
        return still(a, cls, p, seed, src, info, vf, fps_str, fps, W, H, t0)
    fx = cls(p, np.random.default_rng(seed))
    echo = Echo(int(p.get("echo", 0)), decay=float(p.get("echo_decay", 0.5)),
                operator=["add", "maximum", "minimum", "screen", "comp_back"][min(4, max(0, int(p.get("echo_op", 0))))]) if p.get("echo", 0) else None
    song = str(Path(a.audio).expanduser()) if a.audio else ""
    if song:
        if not Path(song).is_file(): sys.exit(f"[error] the song isn't there any more: {song}")
        if not subprocess.run([FP, "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", song],
                              capture_output=True, text=True, timeout=60).stdout.strip():
            sys.exit("[error] that song file has no sound — pick another, or use the clip's own sound")
        if a.look == "granular" and (p["mod_density"] or p["mod_size"] or p["mod_spray"]):
            print("[audio] reading the song's loudness", flush=True)
            try: fx.audio = audio_envelope(song, fps, N, max(0.0, a.song_start))
            except subprocess.CalledProcessError: sys.exit("[error] couldn't read any sound from that song")

    out = Path(a.out)
    part = out.with_name(out.name + ".part")       # never a half-written .mp4 in the renders list
    frames_dir = out.with_name(out.stem + "_frames")
    if a.preview > 0: frames_dir.mkdir(parents=True, exist_ok=True)
    # decode: a constant frame grid (CFR) at the clip's own rate, scaled, as raw BGR; the sound is handled by the encoder
    dvf = vf   # always scale: non-square-pixel clips (DV 720x480 at 8:9) display at W x H but are stored wider
    dec = subprocess.Popen([FF, "-v", "error", "-nostdin", "-ss", f"{start:.3f}", "-i", src, "-t", f"{length:.3f}",
                            "-map", "0:v:0", "-an", "-sn", *(["-vf", ",".join(dvf)] if dvf else []), "-r", fps_str,
                            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    if song:
        ain, amap = ["-ss", f"{max(0.0, a.song_start):.3f}", "-i", song], ["-map", "1:a:0"]
    elif info["audio"]:
        ain, amap = ["-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", src], ["-map", "1:a:0"]
    else:
        ain, amap = [], []
    try:   # the recipe goes into the mp4's comment tag, so a render can be remade or compared later. File names only, no
        # folders: the tag travels with every delivered render (client work too), and a path shows the drive layout
        recipe = {"engine": "granular.py", "look": a.look, "strict": strict, "seed": seed, "source": Path(src).name, "start": round(start, 3),
                  "duration": round(length, 3), "width": W, "height": H, "fps": round(fps, 3), "song": Path(song).name if song else "",
                  "song_start": a.song_start if song else 0.0, "set": a.set,
                  "p": {k: (v.item() if hasattr(v, "item") else v) for k, v in p.items()}}
        meta = ["-metadata", "comment=" + json.dumps(recipe, separators=(",", ":"), default=str)]
    except (TypeError, ValueError):
        meta = []
    enc = subprocess.Popen([FF, "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", fps_str, "-i", "-",
                            *ain, "-map", "0:v", *amap, "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
                            *(["-c:a", "aac", "-b:a", "256k", "-af", "apad", "-shortest"] if amap else []),
                            *meta, "-movflags", "+faststart", "-f", "mp4", str(part)], stdin=subprocess.PIPE)

    def stop(*_):
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, stop)
    ok, i, size = False, 0, W * H * 3
    try:
        pw = min(320, W); ph = even(H * pw / W)
        print("[render] starting", flush=True)
        prev, saved_at = None, -PREVIEW_GAP
        while True:
            buf = dec.stdout.read(size)
            if len(buf) < size: break
            fr = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
            res = fx.frame(fr)
            if echo: res = echo.apply(res)
            if ps_fn: res = ps_fn(res)
            try: enc.stdin.write(res.tobytes())
            except BrokenPipeError: sys.exit("[error] the encoder stopped (see the line above)")
            if a.preview > 0 and res is not prev and i - saved_at >= PREVIEW_GAP:   # a held frame is the same picture: only new ones
                # JPEG, 25-35 KB (a PNG of every picture was ~170 KB: ~3 GB per 10 min of outline; now under 0.3 GB),
                # written aside and renamed in, so the app never shows a half-written one
                tmp = frames_dir / ".next.jpg"
                cv2.imwrite(str(tmp), cv2.resize(res, (pw, ph), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 80])
                os.replace(tmp, frames_dir / f"f_{i:06d}.jpg"); saved_at = i
            prev = res
            i += 1
            if i % 10 == 0:
                print(f"  frame {i}/{max(N, i)}  {(time.time() - t0) / i:.3f}s/frame", flush=True)
        drc = dec.wait()
        if i == 0: sys.exit("[error] no frames could be read from the clip")
        if drc != 0 and i < N - 3:
            sys.exit(f"[error] the clip couldn't be read past frame {i} of {N} (damaged or partly copied file?)")
        if i < N - 3:
            print(f"[warn] the clip ended early: {i} of {N} frames", flush=True)
        print(f"  frame {i}/{i}", flush=True)
        print("[encode] writing the video", flush=True)
        enc.stdin.close()
        if enc.wait() != 0: sys.exit("[error] encoding failed")
        os.replace(part, out); ok = True
        print(f"[done] {out}  ({i} frames, {(time.time() - t0) / 60:.1f} min)", flush=True)
    finally:
        if not ok:
            for pr in (dec, enc):
                try: pr.kill()
                except Exception: pass
            part.unlink(missing_ok=True)   # a working file that never finished, never a deliverable


if __name__ == "__main__":
    main()
