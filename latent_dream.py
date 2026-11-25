#!/usr/bin/env python3
"""
LATENT DREAM — audio-reactive StyleGAN latent walk (the Lucid Sonic Dreams method, on a Mac).

Unlike the retired Stable Diffusion mode (img2img feedback, where every frame is a re-paint of the
last one), this walks smoothly through a trained GAN's latent space, so every frame is a clean
sample from the model and nothing accumulates or melts.

--mode lsd       faithful port of Lucid Sonic Dreams (mikaelalafriz/lucid-sonic-dreams, 2021):
                 keyframe noise vectors lerped at speed_fpm, harmonic -> cumulative "motion" drift that
                 bounces inside +-2*truncation, percussive -> a smoothed "pulse" added to every z dim.
--bend / --spin / --decay  NETWORK BENDING, SPIN and MEMORY DECAY: hooks inside the generator (see flags)
--mode plus      same core, improved:
                 * keyframes interpolated on the sphere (slerp) — no washed-out midpoints
                 * keyframe changes land on detected beats (phrase-aligned), not on a fixed clock
                 * pulse goes to the FINE layers only (texture/colour twitch on the drum, shape holds),
                   motion to the coarse layers (shape drifts with the harmony)
                 * exact output fps (LSD rounds the hop and ends up at odd rates like 21.5)

  python latent_dream.py --pkl models/stylegan/StyleGAN2_microscopev1.pkl --audio song.wav --out dream.mp4
"""
import argparse, os, sys, shutil, subprocess, time
import numpy as np
import librosa
from scipy.stats import truncnorm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "models", "stylegan2-ada-pytorch"))
import torch  # noqa: E402
import legacy  # noqa: E402  (NVlabs stylegan2-ada-pytorch; loads TF and torch .pkl)

FF = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


def spec_norm(y, sr, hop, n_mels=512):
    """LSD's signal: per-frame max of the mel spectrogram, normalised 0..1. Same call as upstream get_spec_norm
    (lucidsonicdreams/helper_functions.py): n_mels=512, librosa's default fmax (sr/2), checked 11-18."""
    s = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop).max(axis=0)
    return (s - s.min()) / (np.ptp(s) + 1e-9)


def model_z_dim(pkl):
    """Load the generator once up front so a model with z_dim != 512 doesn't crash after the audio analysis."""
    global _G
    print("[model] loading " + os.path.basename(pkl), flush=True)
    with open(pkl, "rb") as f:
        _G = legacy.load_network_pkl(f)["G_ema"].eval().float()
    return int(_G.z_dim)


_G = None


def slerp(a, b, t):
    an, bn = a / np.linalg.norm(a), b / np.linalg.norm(b)
    om = np.arccos(np.clip(np.dot(an, bn), -1, 1))
    if om < 1e-4:
        return (1 - t) * a + t * b
    return (np.sin((1 - t) * om) * a + np.sin(t * om) * b) / np.sin(om)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--out", default="latent_dream.mp4")
    p.add_argument("--mode", choices=["lsd", "plus"], default="plus")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--speed_fpm", type=float, default=12, help="new keyframes per minute")
    p.add_argument("--pulse_react", type=float, default=0.5)
    p.add_argument("--motion_react", type=float, default=0.5)
    p.add_argument("--motion_randomness", type=float, default=0.5)
    p.add_argument("--truncation", type=float, default=1.0)
    p.add_argument("--psi", type=float, default=0.8, help="StyleGAN truncation_psi at synthesis")
    p.add_argument("--fine_from", type=int, default=8, help="plus mode: first W layer that takes the pulse")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--draft", action="store_true", help="render half the frames, blend the in-betweens on encode (~2x faster)")
    p.add_argument("--bend", type=float, default=0.0, help="NETWORK BENDING 0..1: shred texture channels on drums, stutter (echo an old frame's structure) on big hits")
    p.add_argument("--spin", type=float, default=0.0, help="SPIN 0..1: the harmony turns the model's structure layers, drum hits push in (was part of --bend)")
    p.add_argument("--decay", type=float, default=0.0, help="MEMORY DECAY 0..1: over the song the model forgets — detail channels die first, noise creeps into structure, everything drifts to the model's average image")
    p.add_argument("--decay_shape", type=float, default=1.6, help="decay curve exponent (>1 = slow start, steep end)")
    a = p.parse_args()
    out_fps = a.fps
    if a.draft:
        a.fps = max(6, a.fps // 2)            # everything below (audio hop, motion, frames) runs at the draft rate
    rng = np.random.default_rng(a.seed)

    # ---------- audio -> per-frame signals ----------
    y, sr = librosa.load(a.audio, sr=22050, mono=True, offset=a.start, duration=a.duration)
    hop = int(round(sr / a.fps))
    harm, perc = librosa.effects.hpss(y)
    pulse = spec_norm(perc, sr, hop)            # percussive -> pulse (LSD default)
    motion = spec_norm(harm, sr, hop)           # harmonic   -> motion (LSD default)
    n = min(len(pulse), len(motion), int(len(y) / hop))
    pulse, motion = pulse[:n], motion[:n]
    if n < 2:
        sys.exit(f"error: nothing to render — the song is shorter than the start point ({a.start:.0f}s)")
    print(f"[audio] {n} frames  ({len(y)/sr:.1f}s @ {a.fps}fps, mode={a.mode})", flush=True)

    # ---------- keyframes ----------
    Z = model_z_dim(a.pkl)
    tn = lambda: (a.truncation * truncnorm.rvs(-2, 2, size=Z, random_state=rng)).astype(np.float32)
    dur_min = len(y) / sr / 60
    k = max(1, int(round(dur_min * a.speed_fpm)))   # at least one change, never forced faster on short clips
    keys = [tn() for _ in range(k + 1)]
    if a.mode == "plus":
        # snap keyframe times to the nearest beat so changes land on the music
        _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
        times = np.linspace(0, n - 1, k + 1)
        if len(beats) > 4:
            times = np.array([beats[np.argmin(np.abs(beats - t))] for t in times], float)
            times[0], times[-1] = 0, n - 1
            times = np.maximum.accumulate(times)
    else:
        times = np.linspace(0, n - 1, k + 1)
    base = np.zeros((n, Z), np.float32)
    for i in range(n):
        j = min(np.searchsorted(times, i, side="right") - 1, k - 1)
        span = max(times[j + 1] - times[j], 1)
        t = (i - times[j]) / span
        base[i] = slerp(keys[j], keys[j + 1], t) if a.mode == "plus" else (1 - t) * keys[j] + t * keys[j + 1]

    # ---------- LSD pulse + motion ----------
    # LSD's motion step is react*20/fps (≈ react*20 per second at any fps).
    # PLUS keeps the 0.75 smoothing as tuned on 10-18; classic re-bases it to 24 fps to match LSD's feel at 43
    SMOOTH = 0.75 ** (43 / a.fps) if a.mode == "lsd" else 0.75
    m_react = a.motion_react * 20 / a.fps
    signs = rng.choice([1, -1], Z)
    rand = rng.choice([1, 1 - a.motion_randomness], Z)
    p_prev = np.zeros(Z, np.float32); m_prev = np.zeros(Z, np.float32); m_cum = np.zeros(Z, np.float32)
    z_motion = np.zeros((n, Z), np.float32); z_pulse = np.zeros((n, Z), np.float32)
    for i in range(n):
        if i % round(a.fps * 4) == 0:
            rand = rng.choice([1, 1 - a.motion_randomness], Z)
        p_add = np.full(Z, a.pulse_react, np.float32) * pulse[i]
        m_add = np.full(Z, m_react, np.float32) * motion[i] * signs * rand
        if i:
            p_add = p_prev * SMOOTH + p_add * (1 - SMOOTH)
            m_add = m_prev * SMOOTH + m_add * (1 - SMOOTH)
        p_prev, m_prev = p_add, m_add
        m_cum += m_add
        z_motion[i] = base[i] + m_cum
        z_pulse[i] = p_add
        cur = z_motion[i]
        signs = np.where(cur - m_react < -2 * a.truncation, 1, np.where(cur + m_react >= 2 * a.truncation, -1, signs))

    # ---------- model ----------
    G = _G
    # Apple GPU: ~0.11 s/frame at 512 on an M3 Pro vs 0.6–1.7 s on CPU (the 10-18 "no faster" test timed the warm-up)
    dev = os.environ.get("LATENT_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[device] {dev}", flush=True)
    G = G.to(dev)
    c = None
    if G.c_dim:                                    # conditional model: draw one class (picked by the seed) instead of an empty label
        c = torch.zeros(1, G.c_dim, device=dev); c[0, a.seed % G.c_dim] = 1
        print(f"[model] conditional: {G.c_dim} classes, using class {a.seed % G.c_dim} (change the seed for another)", flush=True)
    frames = os.path.splitext(a.out)[0] + "_frames"
    if os.path.isdir(frames) and os.listdir(frames):          # never wipe frames that aren't ours: park them
        os.rename(frames, f"{frames}-old-{int(time.time())}")
    os.makedirs(frames, exist_ok=True)

    from PIL import Image
    import torch.nn.functional as F
    ST = {"i": 0, "angle": 0.0, "zoom": 1.0, "shred": 0.0, "decay": 0.0, "stutter": False}
    trng = torch.Generator().manual_seed(a.seed)
    perm = {}                                        # fixed channel order per block, so decay is monotonic
    echo = {}                                        # stored structure for stutter

    def affine(x, ang, zoom):
        th = torch.tensor([[np.cos(ang) / zoom, -np.sin(ang) / zoom, 0], [np.sin(ang) / zoom, np.cos(ang) / zoom, 0]],
                          dtype=x.dtype, device=x.device)[None].repeat(x.shape[0], 1, 1)
        return F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), padding_mode="reflection", align_corners=False)

    def make_hook(res):
        def hook(mod, inp, out):
            x, img = out
            C = x.shape[1]
            if res not in perm:
                perm[res] = torch.randperm(C, generator=trng).to(x.device)
            if a.spin > 0 and res in (16, 32):            # SPIN: structure turns with harmony, pushes on drums
                x = affine(x, ST["angle"], ST["zoom"])
            if a.bend > 0:
                if res in (16, 32):
                    if res == 32:
                        if ST["stutter"] and "s32" in echo:
                            x = 0.35 * x + 0.65 * echo["s32"]
                        elif ST["i"] % 6 == 0:
                            echo["s32"] = x.clone()
                if res == 64 and ST["shred"] > 0:          # texture: drop channels on the drum
                    k = int(C * ST["shred"])
                    if k:
                        x = x.clone(); x[:, perm[res][:k]] = 0
            d = ST["decay"]
            if d > 0:
                if res in (8, 16, 32):                     # structure goes noisy
                    x = x + torch.randn(x.shape, generator=trng).to(x.device) * x.std() * 0.9 * d
                if res >= max(32, G.img_resolution // 4):  # fine detail dies first, for good (hook edits land one block later)
                    k = int(C * min(0.95, 1.2 * d))
                    if k:
                        x = x.clone(); x[:, perm[res][-k:]] = 0
                if res == 64 and d > 0.15:                 # mid detail smears
                    f = 1 + int(6 * d)
                    x = F.interpolate(F.avg_pool2d(x, f, ceil_mode=True), size=x.shape[-2:], mode="bilinear", align_corners=False)
            return (x, img)
        return hook
    if a.bend > 0 or a.decay > 0 or a.spin > 0:
        for res in G.synthesis.block_resolutions:
            getattr(G.synthesis, f"b{res}").register_forward_hook(make_hook(res))
    onset_thr = np.percentile(pulse, 92)
    m_angle = 0.0
    t0 = time.time()
    with torch.no_grad():
        for i in range(n):
            prog = i / max(n - 1, 1)
            ST["i"] = i
            ST["decay"] = a.decay * prog ** a.decay_shape
            m_angle += (motion[i] - 0.5) * 0.06 * a.spin
            ST["angle"] = m_angle + 0.5 * a.spin * pulse[i] * np.sin(i * 0.37)
            ST["zoom"] = 1 + 0.35 * a.spin * pulse[i]
            ST["shred"] = min(0.7, a.bend * 0.9 * pulse[i] ** 2)
            ST["stutter"] = a.bend > 0 and pulse[i] > onset_thr
            psi_now = a.psi * (1 - 0.92 * ST["decay"])     # forgetting = sliding toward the average memory
            if a.mode == "lsd":
                z = torch.from_numpy((z_motion[i] + z_pulse[i])[None]).to(dev)
                ws = G.mapping(z, c, truncation_psi=psi_now)
            else:
                zm = torch.from_numpy(z_motion[i][None]).to(dev)
                zp = torch.from_numpy((z_motion[i] + z_pulse[i])[None]).to(dev)
                ws = G.mapping(zm, c, truncation_psi=psi_now)
                wp = G.mapping(zp, c, truncation_psi=psi_now)
                ws[:, a.fine_from:] = wp[:, a.fine_from:]
            img = G.synthesis(ws, noise_mode="const", force_fp32=True)
            im = ((img.clamp(-1, 1) + 1) * 127.5).permute(0, 2, 3, 1)[0].byte().cpu().numpy()
            if im.shape[2] == 1:                       # grayscale models (X-ray, MRI…)
                out = Image.fromarray(im[:, :, 0], "L").convert("RGB")
            elif im.shape[2] == 4:
                out = Image.fromarray(im, "RGBA").convert("RGB")
            else:
                out = Image.fromarray(im[:, :, :3])
            if a.size != out.width:
                out = out.resize((a.size, a.size), Image.LANCZOS)
            out.save(f"{frames}/f_{i:06d}.png", compress_level=1)   # temp frames: fast write, not small
            if i % 48 == 0:
                el = time.time() - t0
                print(f"  frame {i+1}/{n}  {el/(i+1):.2f}s/frame  eta {el/(i+1)*(n-i-1)/60:.1f}min", flush=True)

    print("[encode]", flush=True)
    # frames were cut from the audio every `hop` samples, so they run at sr/hop fps, not exactly a.fps — encode at
    # that rate so picture and sound stay together, then resample to the requested rate
    enc = subprocess.run([FF, "-v", "error", "-y", "-framerate", f"{sr}/{hop}", "-i", f"{frames}/f_%06d.png",
                    "-ss", str(a.start), *(["-t", str(a.duration)] if a.duration else []), "-i", a.audio,
                    "-map", "0:v", "-map", "1:a", *(["-vf", f"minterpolate=fps={out_fps}:mi_mode=blend"] if a.draft else ["-r", str(out_fps)]), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
                    "-c:a", "aac", "-b:a", "256k", "-shortest", a.out], capture_output=True, text=True)
    if enc.returncode != 0:   # frames are KEPT on failure: the GPU work isn't lost, the encode can be retried
        msg = (enc.stderr or "").strip().splitlines()[-1:] or ["ffmpeg failed"]
        sys.exit("error: encoding failed — " + msg[0] + (" (does the song file have an audio track?)" if "matches no streams" in (enc.stderr or "") else "") + f" · frames kept in {frames}")
    shutil.rmtree(frames, ignore_errors=True)
    print(f"[done] {a.out}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
