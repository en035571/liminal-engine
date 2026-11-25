"""Model shelf for LATENT DREAM: the awesome-pretrained-stylegan2 catalog, Eric's picks, and downloads to the T7.

Downloads run one at a time in a background thread (separate from the render slot). Each model lands as
<slug>.pkl, so "installed" is just a file check. Hosts: mega.nz (megadl), Google Drive (gdown),
yadi.sk (Yandex public API), everything else plain HTTP.
"""
import json, os, re, shutil, subprocess, threading, time, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / "models" / "stylegan"
GALLERY = LOCAL / "_gallery"
T7 = Path("/Volumes/T7/_Libraries/stylegan-models")
# downloads go to the T7 when it is plugged in, otherwise into the engine's own models folder (any other Mac, or T7 unplugged)
SHELF = T7 if Path("/Volumes/T7").is_dir() else LOCAL   # at import; _download re-checks
CATALOG_URL = "https://raw.githubusercontent.com/justinpinkney/awesome-pretrained-stylegan2/master/models.json"
PICKS = GALLERY / "picks.json"
UA = {"User-Agent": "Mozilla/5.0 (LIMINAL ENGINE model shelf)"}
PY = ROOT / ".venv" / "bin" / "python"

BRIEF = ["microscope images", "wildlife", "modern art", "textures", "Maps", "WikiArt", "Abstract art", "beetles", "trypophobia",
         "flowers", "more abstract art", "figure drawings", "Doors", "floor plans", "church"]
STARTER = ["wildlife", "modern art", "textures", "Maps", "WikiArt", "Abstract art"]

DL = {}            # name -> {"state": queued|downloading|done|failed, "pct", "bytes", "msg", "path"}
_q, _lock = [], threading.Lock()
_worker = None


def gid(url):
    """gdown 6 dropped --fuzzy; hand it the bare Drive file id."""
    m = re.search(r"/d/([A-Za-z0-9_-]{10,})", url) or re.search(r"[?&]id=([A-Za-z0-9_-]{10,})", url)
    return m.group(1) if m else url


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def model_dirs():
    return [d for d in (LOCAL, T7, T7 / "faces") if d.is_dir()]   # faces/ loads too; stylegan3/ other-code/ quarantine/ do not (yet)


def _pkls():
    for d in model_dirs():
        for p in d.glob("*.pkl"):
            try:
                if not p.name.startswith("._") and p.stat().st_size > 3_000_000: yield p   # ._ = exFAT sidecars on the T7
            except OSError: continue


def copies_of(p):
    """The same model file elsewhere on the shelf (same name and size, e.g. this Mac and the T7)."""
    p = Path(p).resolve()
    try: size = p.stat().st_size
    except OSError: return []
    return [q.resolve() for q in _pkls() if q.name == p.name and q.resolve() != p and q.stat().st_size == size]


def model_paths():
    """Every model once: a model that is both on this Mac and on the T7 is listed by its local copy (loads faster,
    works with the T7 unplugged)."""
    seen, out = set(), []
    for p in _pkls():                      # model_dirs() lists LOCAL first
        k = (p.name, p.stat().st_size)
        if k in seen: continue
        seen.add(k); out.append(p)
    return out


GROUPS = [  # first match wins
    ("faces", r"face|ffhq|obama|anime|waifu|portrait|pony|fursona|person|people"),
    ("science", r"brain|chest|x-?ray|mri|lesion|wound|rim|thyroid|lung|breast|brecahad|iris|microscop|embryo|galax|astro|solar|patholog|skin|oasis|tumou?r|bbl"),
    ("art", r"art|paint|figure|graffiti|escher|machineray|abstract|wikiart|drawing|sketch|illustr|ukiyo|omniart|seal"),
    ("places + things", r"map|church|door|car|bicycle|archi|circuit|tech|bedroom|liminal|toilet|cityscape|floor|imagenet|cifar|synth-?eye|fingerprint|front|back|food"),
    ("nature", r"wild|bird|cub|dog|parrot|horse|cat|flower|trypo|texture|lhq|landscape|beetle|fish|insect"),
]


_IDX = {"t": 0, "v": None}


def _index():
    """slug -> catalog/scour entry, so a downloaded file can be shown with its real name and sample.
    Cached for a minute and never touches the network (it runs inside the 2 s /api/state poll)."""
    if _IDX["v"] is not None and time.time() - _IDX["t"] < 60:
        return _IDX["v"]
    idx = {}
    cf = GALLERY / "models.json"
    cat = json.loads(cf.read_text()) if cf.exists() else []
    for x in cat:
        idx[slug(x["name"])] = {"name": x["name"], "res": x.get("resolution"),
                                "sample_local": sample(x["name"], "samples", False) or sample(x["name"], "thumbnail", False)}
    try:
        for r in json.loads((GALLERY / "scour.json").read_text()):
            n = r.get("_n") or r["name"]
            generic = re.search(r"generic|teaser|banner|no [^.]*sample|not [^.]*specific", r.get("notes") or "", re.I)
            idx.setdefault(slug(n)[:80], {"name": n, "res": r.get("resolution"), "sample_url": None if generic else r.get("sample_url")})
    except Exception:
        pass
    idx["stylegan2-microscopev1"] = idx.get("microscope-images", {"name": "microscope images"})
    _IDX.update(t=time.time(), v=idx)
    return idx


def describe(paths):
    idx = _index(); out = []
    for p in paths:
        p = Path(p); s = slug(p.stem)
        meta = idx.get(s) or idx.get(s[:80]) or {}
        name = meta.get("name") or p.stem.replace("-", " ")
        grp = next((g for g, rx in GROUPS if re.search(rx, name, re.I)), "other")
        if p.parent.name == "faces": grp = "faces"
        out.append({"name": name, "path": str(p), "size": p.stat().st_size, "group": grp, "res": meta.get("res"),
                    "sample": meta.get("sample_local"), "sample_url": meta.get("sample_url"),
                    "where": ("this Mac + T7" if copies_of(p) else "T7" if str(p).startswith("/Volumes/") else "this Mac")})
    out.sort(key=lambda m: m["name"].lower())
    return out


def installed():
    """slug -> path for every .pkl we can see (local + T7)."""
    out = {}
    for d in model_dirs():
        for p in d.glob("*.pkl"):
            if p.name.startswith("._"): continue          # exFAT AppleDouble sidecars, not models
            out.setdefault(slug(p.stem), str(p))    # LOCAL is listed first: a model on both is used from this Mac
    # the microscope model came in under its original file name
    for p in LOCAL.glob("*microscope*.pkl"):
        out["microscope-images"] = str(p)
    return out


def catalog():
    GALLERY.mkdir(parents=True, exist_ok=True)
    f = GALLERY / "models.json"
    if not f.exists() or time.time() - f.stat().st_mtime > 7 * 86400:
        try:
            f.write_bytes(urllib.request.urlopen(urllib.request.Request(CATALOG_URL, headers=UA), timeout=20).read())
        except Exception:
            pass
    rows = json.loads(f.read_text()) if f.exists() else []
    for x in rows:
        x.update(FIXES.get(x["name"], {}))
    return rows


# the catalog is from 2020; some hosts have since moved or died (checked 10-20)
FIXES = {
    "faces (FFHQ config-f 512x512)": {   # aydao's mega link is taken down (EBLOCKED) -> NVIDIA's own 512 FFHQ net
        "download_url": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl",
        "author": "NVIDIA", "license": "Nvidia Source Code License",
        "source_url": "https://github.com/NVlabs/stylegan2-ada-pytorch"},
    "CIFAR 10": {"dead": "gone upstream: MIT took the DiffAugment models down (404 on every mirror)"},
    "CIFAR 100": {"dead": "gone upstream: MIT took the DiffAugment models down (404 on every mirror)"},
}


def picks():
    try: return json.loads(PICKS.read_text())
    except Exception: return []


def save_picks(names):
    GALLERY.mkdir(parents=True, exist_ok=True)
    PICKS.write_text(json.dumps(sorted(set(names)), indent=1))


def sample(name, kind, fetch=True):
    p = GALLERY / "img" / f"{name.replace('/', '-')}_{kind}.jpg"
    if not p.exists() and not fetch:
        return None
    if not p.exists():
        try:
            u = ("https://raw.githubusercontent.com/justinpinkney/awesome-pretrained-stylegan2/master/content/"
                 + urllib.parse.quote(name) + f"/{kind}.jpg")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=20).read())
        except Exception:
            return None
    return str(p)


TRASH = None        # server.trash, handed in at start-up (moves to the macOS Trash)
_tidied = [False]


def clear_stale():
    """Downloads that can't still be running: a worker that died mid-file leaves its entry at queued/downloading
    forever (and that blocked app updates). Mark those failed. Once per run, partial .tmp-* folders left by a
    download that was interrupted (app closed, T7 pulled) go to the Trash."""
    with _lock:
        alive = _worker is not None and _worker.is_alive()
        if not alive:
            for st in DL.values():
                if st.get("state") in ("queued", "downloading"):
                    st.update(state="failed", msg="interrupted — pick it and download again")
        if alive or _tidied[0] or TRASH is None: return
        _tidied[0] = True
    for d in {LOCAL, T7}:
        if not d.is_dir() or any(st.get("state") in ("queued", "downloading") for st in DL.values()): continue
        for t in d.glob(".tmp-*"):
            if t.is_dir():
                try: TRASH(t)
                except Exception: pass


def state():
    clear_stale()
    inst = installed()
    rows = []
    for x in catalog():
        s = slug(x["name"])
        rows.append({**x, "slug": s, "brief": x["name"] in BRIEF, "starter": x["name"] in STARTER,
                     "installed": inst.get(s), "dl": DL.get(x["name"]),
                     "sample": sample(x["name"], "samples") or sample(x["name"], "thumbnail")})
    rows.sort(key=lambda r: (not r["brief"], BRIEF.index(r["name"]) if r["brief"] else 0, r["name"].lower()))
    shelf = T7 if Path("/Volumes/T7").is_dir() else LOCAL     # same rule as downloads (checked now, T7 can come and go)
    return {"models": rows, "picks": picks(), "t7": True, "t7_dir": str(shelf), "where": "T7" if shelf == T7 else "this Mac"}


# ---------------------------------------------------------------- downloading
def _http(url, dest, st):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        with open(dest, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b: break
                f.write(b); got += len(b)
                st["bytes"] = got; st["pct"] = round(100 * got / total) if total else None


def _yadisk(url, dest, st):
    api = "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=" + urllib.parse.quote(url, safe="")
    href = json.loads(urllib.request.urlopen(urllib.request.Request(api, headers=UA), timeout=30).read())["href"]
    _http(href, dest, st)


def _tool(cmd, tmp, st):
    """megadl / gdown write into tmp; follow the folder size for progress."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = []
    while p.poll() is None:
        st["bytes"] = sum(f.stat().st_size for f in tmp.glob("**/*") if f.is_file())
        try:
            line = p.stdout.readline()
            if line: tail = (tail + [line.strip()])[-4:]
            m = re.search(r"(\d+(?:\.\d+)?)%", line or "")
            if m: st["pct"] = round(float(m.group(1)))
        except Exception:
            pass
    rest = p.stdout.read() or ""
    tail = (tail + rest.strip().splitlines())[-4:]
    if p.returncode != 0:
        raise RuntimeError(" / ".join(t for t in tail if t)[-300:] or f"exit {p.returncode}")
    files = [f for f in tmp.glob("**/*") if f.is_file() and not f.name.startswith("._")]
    if not files:
        raise RuntimeError("nothing came down (link may be dead or rate-limited)")
    return max(files, key=lambda f: f.stat().st_size)


def _download(x):
    name, url = x["name"], x["download_url"]
    st = DL[name]; st.update(state="downloading", pct=0, bytes=0, msg="")
    SHELF = T7 if Path("/Volumes/T7").is_dir() else LOCAL
    tmp = SHELF / f".tmp-{slug(name)}"
    final = SHELF / f"{slug(name)}.pkl"
    try:
        SHELF.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
        host = urllib.parse.urlparse(url).netloc
        if "mega.nz" in host:
            got = _tool([shutil.which("megadl") or "/opt/homebrew/bin/megadl", "--path", str(tmp), url], tmp, st)
        elif "drive.google.com" in host:
            got = _tool([str(PY), "-m", "gdown", gid(url), "-O", str(tmp) + "/"], tmp, st)
        elif "yadi.sk" in host:
            got = tmp / "model.pkl"; _yadisk(url, got, st)
        else:
            got = tmp / "model.pkl"; _http(url, got, st)
        head = got.read_bytes()[:2] if got.stat().st_size else b""
        if got.stat().st_size < 5_000_000:
            raise RuntimeError(f"only {got.stat().st_size // 1024} KB came down — probably an error page, not a model")
        if got.suffix.lower() in (".zip",) or head == b"PK" and got.suffix.lower() != ".pkl":
            raise RuntimeError(f"came down as {got.name} (an archive) — unzip by hand")
        scan = subprocess.run([str(ROOT / ".venv/bin/picklescan"), "-p", str(got)], capture_output=True, text=True)
        if scan.returncode != 0:          # 1 = dangerous globals, 2 = could not scan: either way it never reaches LATENT DREAM
            q = SHELF / "quarantine"; q.mkdir(parents=True, exist_ok=True)
            shutil.move(str(got), str(q / final.name))
            raise RuntimeError("picklescan " + ("flagged dangerous code" if scan.returncode == 1 else "could not scan it") + " — moved to quarantine/")
        shutil.move(str(got), str(final))
        st.update(state="done", pct=100, path=str(final), msg=f"{final.stat().st_size // 1_000_000} MB · scanned clean")
    except Exception as e:
        st.update(state="failed", msg=str(e)[:300])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run():
    global _worker
    try:
        while True:
            with _lock:
                if not _q:
                    return
                x = _q.pop(0)
            try: _download(x)
            except Exception as e: DL.get(x["name"], {}).update(state="failed", msg=str(e)[:300])
    finally:
        with _lock:
            _worker = None


def queue(names):
    global _worker
    cat = {x["name"]: x for x in catalog()}
    inst = installed()
    added = []
    with _lock:
        for n in names:
            x = cat.get(n)
            if not x or inst.get(slug(n)): continue
            if x.get("dead"):
                DL[n] = {"state": "failed", "pct": 0, "bytes": 0, "msg": x["dead"]}; continue
            if DL.get(n, {}).get("state") in ("queued", "downloading"): continue
            DL[n] = {"state": "queued", "pct": 0, "bytes": 0, "msg": ""}
            _q.append(x); added.append(n)
        if added and _worker is None:
            _worker = threading.Thread(target=_run, daemon=True); _worker.start()
    return {"ok": True, "queued": added}
