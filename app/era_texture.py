"""TAPE — record a finished video onto a modelled analogue video tape and play it back.

  .venv/bin/python app/era_texture.py --in clip.mp4 --out clip_tape.mp4 [--format vhs-sp] [--gen 1] [--damage 0.3]
                                      [--faults dropout,tracking,clog,skew,wobble,ckill] [--react song.wav --start 0]

A physical model, not a filter. Every number is a property of the format (sources next to each value below), scaled
to the frame by the active line time, so the look is the same at 640 or 1920 px across:

  active line      52.6 us of picture per NTSC line (SMPTE 170M; PAL 52 us)  ->  px per microsecond = W / 52.6
  RECORD           luma and chroma low-passed to the format's bandwidth (Gaussian, -6 dB at the quoted figure);
                   colour-under formats (VHS, U-matic) blend chroma with the next line (the playback comb);
                   a small Y/C delay; the deck's detail enhancer (edge overshoot); luma and chroma noise at the
                   format's signal-to-noise ratio, along the line, each line independent.
  GENERATION       every analogue dub records the playback again: bandwidth losses cascade (Gaussian sigmas add in
                   quadrature -> x sqrt(n)), noise powers add (SNR - 10 log10 n dB), time-base error variance adds.
  TRANSPORT        no time-base corrector on consumer decks: lines and fields wobble by tens of ns; the head-switching
                   point sits ~6.5 lines before vertical sync, so the bottom lines tear sideways (none on Betacam SP,
                   which is played through a broadcast TBC).
  FAULTS           events on the tape, each a known failure: dropouts (oxide missing; the deck's dropout compensator
                   repeats the line above), tracking error (noise bars + torn lines), head clog (one head reads one
                   field, so every other line is lost to snow), skew (tape tension: the top lines hook sideways and the
                   TV catches up), timebase wobble (a jump in the flutter), colour loss (the TV's colour killer drops a
                   weak colour burst). With an --react song the faults are triggered by its onsets — louder hit,
                   bigger fault — so the damage plays the song; without, they fall at random like a worn tape.

A tape is one continuous recording, so this runs over a whole finished edit, frame index continuous. Seeded, so
re-runs match. numpy + scipy only (librosa only for --react), so it works without the AI tier.
Parameters mirror the CRT tool's Tape group (app/crt.html, TAPE / TAPEMECH) where the two overlap.
"""
import argparse, json, math, os, shutil, subprocess
import numpy as np
from scipy.ndimage import gaussian_filter1d

for _p in ("/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin"):
    if os.path.isdir(_p) and _p not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"

ACTIVE_LINE_US = 52.6          # SMPTE 170M (NTSC): 52.6 us of active picture per line
FIELD_LINES = 242.5            # SMPTE 170M: 485 active lines per frame = 242.5 per field
# Per format. luma/chroma: bandwidth in MHz; snr/csnr: dB (100 IRE p-p over rms noise); yc: Y/C delay in us;
# comb: colour-under playback comb (chroma averaged over two lines); tbc: played through a time-base corrector;
# jitter_ns: rms line-to-line time-base error without a TBC; hs_us: head-switching displacement at the bottom line.
FORMATS = {
    # VHS (IEC 60774-1; colour-under at 629 kHz). ~240 TVL -> 240 / 0.75 / 2 cycles over 52.6 us = 3.0 MHz luma;
    # chroma ~0.5 MHz around the colour-under carrier; SNR ~43 dB luma / ~40 dB chroma AM (typical consumer deck
    # spec sheets). EP: a third of the tape speed, a 19 um track instead of 58 um -> ~5 dB worse SNR and a softer
    # picture (~200 TVL -> 2.5 MHz). Head switch ~6.5 lines before vsync (VCR service manuals; the CRT tool uses 4.5
    # lines in its 262.5-line field). Time-base error without TBC: tens of ns (no published single figure; 30 ns rms).
    "vhs-sp":     {"name": "VHS SP", "luma": 3.0, "chroma": 0.5, "snr": 43.0, "csnr": 40.0, "yc": 0.10, "comb": True, "tbc": False, "jitter_ns": 30, "hs_us": 1.0},
    "vhs-ep":     {"name": "VHS EP", "luma": 2.5, "chroma": 0.4, "snr": 38.0, "csnr": 36.0, "yc": 0.15, "comb": True, "tbc": False, "jitter_ns": 40, "hs_us": 1.2},
    # U-matic (low band, IEC 60712; colour-under at 688 kHz). ~250-260 TVL -> ~3.1 MHz; chroma ~0.5 MHz; SNR ~46 dB
    # (Sony VO-5850 class spec sheets). Wider tape, slower drum: a smaller head-switch step and steadier timebase.
    "umatic":     {"name": "U-matic", "luma": 3.1, "chroma": 0.5, "snr": 46.0, "csnr": 43.0, "yc": 0.07, "comb": True, "tbc": False, "jitter_ns": 20, "hs_us": 0.5},
    # Betacam SP (component, no colour-under, no comb). Sony BVW-75 spec: Y 4.5 MHz (-3 dB; the CRT tool uses 4.1),
    # R-Y/B-Y 1.5 MHz, Y SNR 51 dB, C SNR 53 dB. Played through the deck's TBC: no timebase wobble, no switching tear.
    "betacam-sp": {"name": "Betacam SP", "luma": 4.5, "chroma": 1.5, "snr": 51.0, "csnr": 53.0, "yc": 0.0, "comb": False, "tbc": True, "jitter_ns": 0, "hs_us": 0.0},
}
FAULTS = ("dropout", "tracking", "clog", "skew", "wobble", "ckill")
HEAD_SWITCH_LINES = 6.5        # lines above vertical sync where the drum swaps heads (VHS service manuals)
_SIG_6DB = math.sqrt(math.log(2) / (2 * math.pi ** 2))   # Gaussian: |H(f)| = 0.5 at f when sigma_t = 0.1874 / f


def probe(path):
    r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate:format=duration",
                        "-of", "json", path], capture_output=True, text=True, check=True)
    j = json.loads(r.stdout); s = j["streams"][0]
    n, d = s["r_frame_rate"].split("/")
    return s["width"], s["height"], float(n) / float(d), s["r_frame_rate"], float(j.get("format", {}).get("duration") or 0)


def shift_rows(x, off):
    """Shift each row of x (H, W[, C]) right by off[row] pixels (float), edge-replicated, linear interpolation."""
    H, W = x.shape[:2]
    xs = np.clip(np.arange(W, dtype=np.float32)[None, :] - off[:, None].astype(np.float32), 0, W - 1)
    x0 = xs.astype(np.int32); x1 = np.minimum(x0 + 1, W - 1); f = xs - x0
    rows = np.arange(H)[:, None]
    if x.ndim == 3: f = f[..., None]
    return x[rows, x0] * (1 - f) + x[rows, x1] * f


def shift_const(x, d):
    """Shift every row of x (H, W, C) right by the same d >= 0 pixels (edge-replicated)."""
    if d <= 0: return x
    k = int(math.floor(d)); f = np.float32(d - k)
    p = np.concatenate([np.repeat(x[:, :1], k + 1, axis=1), x], axis=1)      # left edge held
    W = x.shape[1]
    return p[:, 1:W + 1] * (1 - f) + p[:, :W] * f


def music_hits(audio, start, duration, fps):
    """Onsets of the song as (frame, strength 0..1), plus a per-frame loudness envelope 0..1 (librosa)."""
    import librosa
    y, sr = librosa.load(audio, sr=22050, mono=True, offset=start, duration=duration if duration and duration > 0 else None)
    hop = 256
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    on = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop, units="frames")
    top = np.percentile(env, 99) + 1e-9
    hits = [(int(round(librosa.frames_to_time(f, sr=sr, hop_length=hop) * fps)), float(min(1.0, env[f] / top))) for f in on]
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    n = int(len(y) / sr * fps) + 1
    loud = np.interp(np.arange(n) / fps, t, rms)
    loud = (loud - np.percentile(loud, 10)) / (np.ptp(loud) + 1e-9)
    return hits, np.clip(loud, 0, 1)


class Tape:
    def __init__(self, W, H, fps, fmt="vhs-sp", gen=1, damage=0.3, faults=FAULTS, seed=3, hits=None, loud=None, n_frames=None):
        F = FORMATS[fmt]; self.F = F
        self.W, self.H, self.fps, self.seed = W, H, fps, seed
        self.gen = max(1, int(gen)); self.dmg = float(np.clip(damage, 0, 1))
        self.faults = set(f for f in faults if f in FAULTS)
        if F["tbc"]: self.faults -= {"wobble", "skew"}            # a TBC re-times every line: no wobble, no skew
        g = math.sqrt(self.gen)                                     # generation: sigmas add in quadrature
        self.px_us = W / ACTIVE_LINE_US                             # pixels per microsecond of line time
        self.s_luma = _SIG_6DB / F["luma"] * self.px_us * g
        self.s_chroma = _SIG_6DB / F["chroma"] * self.px_us * g
        loss_db = 10 * math.log10(self.gen) + 3.0 * self.dmg        # dub noise powers add; worn oxide a few dB more
        self.n_luma = 10 ** (-(F["snr"] - loss_db) / 20)
        self.n_chroma = 10 ** (-(F["csnr"] - loss_db) / 20)
        self.yc = F["yc"] * self.px_us * g
        self.jit = F["jitter_ns"] / 1000 * self.px_us * g * (1 + self.dmg)
        self.hs = max(2, int(round(H * HEAD_SWITCH_LINES / FIELD_LINES))) if F["hs_us"] > 0 else 0
        self.line_px = H / (2 * FIELD_LINES)                        # frame rows per scan line of one field
        self.loud = loud
        # ---- the fault schedule: one list of (start frame, frames, size, place) per fault, planned up front
        r = np.random.default_rng(seed)
        self.ev = {k: [] for k in FAULTS}
        if hits is None:                                            # no song: a worn tape fails at random
            n = n_frames or int(fps * 600)
            hits = [(int(t), float(r.uniform(0.3, 1))) for t in np.cumsum(r.exponential(fps * 4 / max(self.dmg, 0.05), 4096)) if t < n]
        thr = 1 - self.dmg                                          # more damage: softer hits break it too
        dur = {"dropout": (0.04, 0.15), "tracking": (0.3, 1.2), "clog": (0.25, 1.0), "skew": (0.2, 0.5), "wobble": (0.3, 0.8), "ckill": (0.15, 0.6)}
        need = {"dropout": 0.0, "skew": 0.15, "wobble": 0.15, "ckill": 0.3, "tracking": 0.45, "clog": 0.65}   # bigger faults need bigger hits
        on = [k for k in FAULTS if k in self.faults]
        for f0, s in hits:
            if not on or s < thr * 0.9 or r.random() > 0.35 + 0.65 * self.dmg: continue
            can = [k for k in on if s >= need[k] * (1.2 - 0.6 * self.dmg)]
            if not can: continue
            k = can[int(r.integers(len(can)))]
            lo, hi = dur[k]
            self.ev[k].append((f0, max(1, int(fps * (lo + (hi - lo) * s))), s * (0.4 + 0.6 * self.dmg), float(r.random())))

    def _now(self, k, i):
        """Strength of fault k at frame i (0 when none), with where in the frame it sits."""
        a, place = 0.0, 0.5
        for f0, n, s, pl in self.ev[k]:
            if f0 <= i < f0 + n:
                u = (i - f0) / n
                env = 1.0 if k in ("clog", "ckill") else math.sin(math.pi * min(1, u * 1.5 + 0.1)) if k == "tracking" else math.exp(-3 * u)
                if s * env > a: a, place = s * env, pl
        return a, place

    def frame(self, rgb, i):
        """rgb float32 (H, W, 3) in 0..1 -> the same frame played back off tape."""
        H, W, F = self.H, self.W, self.F
        g = np.random.default_rng((self.seed, i))
        r, gr, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        y = 0.299 * r + 0.587 * gr + 0.114 * b                                # BT.601
        cb = 0.564 * (b - y); cr = 0.713 * (r - y)
        # ---- record: bandwidth, detail enhancer, noise
        y = gaussian_filter1d(y, self.s_luma, axis=1, mode="nearest")
        y = y + 0.45 * (y - gaussian_filter1d(y, 2.5 * self.s_luma, axis=1, mode="nearest"))
        n = gaussian_filter1d(g.standard_normal((H, W), dtype=np.float32), 0.6 * self.s_luma, axis=1, mode="wrap")
        y = y + n * (self.n_luma / (n.std() + 1e-9))
        c = gaussian_filter1d(np.stack([cb, cr], -1), self.s_chroma, axis=1, mode="nearest")
        if F["comb"]:                                                         # playback comb: chroma averaged with the next line
            step = max(1, int(round(2 * self.line_px)))
            c = 0.5 * (c + np.concatenate([c[step:], c[-step:]], 0))
        cn = gaussian_filter1d(g.standard_normal((H, W, 2), dtype=np.float32), 0.6 * self.s_chroma, axis=1, mode="wrap")
        c = c + cn * (self.n_chroma / (cn.std() + 1e-9))
        c = shift_const(c, self.yc)
        ck, _ = self._now("ckill", i)
        if ck > 0.25: c = c * 0.0                                             # colour killer: on or off, never half
        x = np.stack([y + 1.403 * c[..., 1], y - 0.344 * c[..., 0] - 0.714 * c[..., 1], y + 1.773 * c[..., 0]], -1)
        # ---- transport: time-base error, wobble jumps, skew, head switching, tracking
        rows = np.arange(H, dtype=np.float32)
        off = np.zeros(H, np.float32)
        if self.jit > 0:
            j = gaussian_filter1d(g.standard_normal(H).astype(np.float32), 2.0 * self.line_px)
            wob, _ = self._now("wobble", i)
            off += j / (j.std() + 1e-9) * self.jit * (1 + 6 * wob)
            off += (np.arange(H) % 2) * g.normal(0, 0.5 * self.jit)          # the two fields don't line up
            if wob > 0: off += wob * 0.6 * self.px_us * np.sin(rows / H * math.pi + i * 0.7)   # a flutter jump bends the frame
        sk, _ = self._now("skew", i)
        if sk > 0: off += sk * 6.0 * self.px_us * np.exp(-rows / (38 * H / 486))   # ~6 us hook at the top, the TV's line sync catches up over ~100 of 486 lines (CRT tool: 90 of 910 samples, tau 38 lines)
        if self.hs:
            hs = self.hs
            off[H - hs:] += np.linspace(0, 1, hs) ** 1.5 * F["hs_us"] * (0.8 + 0.4 * g.random()) * math.sqrt(self.gen) * self.px_us
        trk, place = self._now("tracking", i)
        if self.loud is not None and "tracking" in self.faults:           # loud stretches lose tracking a little
            trk = max(trk, 0.25 * self.dmg * max(0.0, float(self.loud[min(i, len(self.loud) - 1)]) - 0.7) / 0.3)
        bar = np.zeros(H, np.float32)
        if trk > 0:
            for k in range(1 + int(trk > 0.6)):                              # one noise bar, two on a bad stretch
                cy = ((place + 0.37 * k - 0.15 * (i % 97) / 97) % 1.0) * H
                d = np.abs(rows - cy); d = np.minimum(d, H - d)
                bar = np.maximum(bar, np.clip(1 - (d - H * 0.03 * trk) / (H * 0.02), 0, 1) * min(1.0, trk * 1.5))
            off += bar * g.normal(0, 1, H).astype(np.float32) * 0.8 * self.px_us
        x = shift_rows(x, off)
        if self.hs:                                                           # switching noise below the switch point
            sw = gaussian_filter1d(g.standard_normal((self.hs, W), dtype=np.float32), self.s_luma, axis=1, mode="wrap")
            x[H - self.hs:] = x[H - self.hs:] * 0.85 + 0.15 * (0.5 + 0.5 * sw / (sw.std() + 1e-9))[..., None]
        if bar.any():                                                         # inside a bar: streaky snow, colour gone
            snow = gaussian_filter1d(g.random((H, W), dtype=np.float32), 0.6 * self.s_luma, axis=1)
            streak = (g.random((H, W)) > 0.993).astype(np.float32)
            snow = np.clip(snow * 0.5 + gaussian_filter1d(streak, 5 * self.px_us, axis=1) * 8, 0, 1)
            m = bar[:, None, None]
            x = x * (1 - m) + snow[..., None] * m
        # ---- head clog: one head is blind, so its field (every other line) is snow
        cl, pl = self._now("clog", i)
        if cl > 0:
            par = int(pl * 2) % 2
            sn = gaussian_filter1d(g.random((H, W), dtype=np.float32), 0.6 * self.s_luma, axis=1)
            fld = ((np.arange(H) // max(1, int(round(self.line_px)))) % 2 == par).astype(np.float32)[:, None, None] * min(1.0, 0.5 + cl)
            x = x * (1 - fld) + (sn[..., None] * 0.8 + 0.1) * fld
        # ---- dropouts: a few us of a line missing; the dropout compensator repeats the line above
        rate = 0.0
        if "dropout" in self.faults:                                          # a few a second on any tape, a burst on a hit
            do, _ = self._now("dropout", i)
            rate = 0.05 + 0.45 * self.dmg + 40 * do
        lh = max(1, int(round(self.line_px)))
        for _ in range(g.poisson(rate)):
            r0 = int(g.integers(lh, H)); c0 = int(g.integers(0, W)); L = int(g.uniform(2, 25) * self.px_us)
            x[r0:r0 + lh, c0:c0 + L] = x[r0 - lh:r0, c0:c0 + L]
        return np.clip(x, 0, 1).astype(np.float32, copy=False)


def process(src, out, fmt="vhs-sp", gen=1, damage=0.3, faults=FAULTS, seed=3, react=None, start=0.0, duration=0.0,
            audio_from=None, crf=17, progress=None):
    """Play src through the tape model into out. react: a song whose onsets trigger the faults. Sound: copied from
    audio_from (or src) untouched."""
    W, H, fps, rate, dur = probe(src)
    nf = int(round(dur * fps))
    hits, loud = music_hits(react, start, duration, fps) if react else (None, None)
    model = Tape(W, H, fps, fmt, gen, damage, faults, seed, hits, loud, nf)
    aud = audio_from or src
    dec = subprocess.Popen([FFMPEG, "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen([FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", rate, "-i", "-",
                            "-i", aud, "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
                            "-pix_fmt", "yuv420p", "-c:a", "copy", "-t", f"{dur:.3f}", out], stdin=subprocess.PIPE)
    n = W * H * 3; i = 0
    try:
        while True:
            buf = dec.stdout.read(n)
            if len(buf) < n: break
            x = np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(np.float32) / 255
            enc.stdin.write((model.frame(x, i) * 255 + 0.5).astype(np.uint8).tobytes()); i += 1
            if progress: progress(i, nf)
    finally:
        enc.stdin.close(); rc = enc.wait(); dec.wait()
    if rc != 0: raise RuntimeError("tape encode failed")
    return i, {k: len(v) for k, v in model.ev.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="src", required=True); p.add_argument("--out", required=True)
    p.add_argument("--format", choices=sorted(FORMATS), default="vhs-sp")
    p.add_argument("--gen", type=int, default=1, help="generation: 1 = the original recording, 2 = a dub of it, ...")
    p.add_argument("--damage", type=float, default=0.3, help="0 = a fresh tape, 1 = badly damaged (more and bigger faults)")
    p.add_argument("--faults", default=",".join(FAULTS), help="which faults can happen: " + ",".join(FAULTS))
    p.add_argument("--react", default="", help="a song: its onsets trigger the faults (else they fall at random)")
    p.add_argument("--start", type=float, default=0.0); p.add_argument("--duration", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--crf", type=int, default=17)
    a = p.parse_args()
    n, ev = process(a.src, a.out, a.format, a.gen, a.damage, a.faults.split(","), a.seed, a.react or None, a.start, a.duration,
                    crf=a.crf, progress=lambda i, t: i % 48 == 0 and print(f"[tape] {i}/{t} frames", flush=True))
    print(f"[done] {a.out}  ({n} frames; faults {ev})")


if __name__ == "__main__":
    main()
