"""Updates for shared copies of LIMINAL ENGINE.

A shared copy has app/channel.json ({"repo": "owner/name"}) and VERSION + MANIFEST.json at its root, all written
by dist/release.py. Eric's working copy has no channel.json, so nothing here runs there.

check(): asks GitHub for the latest release (one small request, at most every 30 min).
apply(): downloads that release's update zip, checks its sha256, installs new Python packages first if
requirements.txt changed (a failure there leaves the copy untouched), then swaps in only the files that differ.
Every replaced or removed file is moved to _cleanup/update-<old version>-<stamp>/; if anything fails half way
the old files are put back. The .venv, models, renders and uploads are never touched.
"""
import hashlib, json, os, shutil, ssl, subprocess, sys, threading, time, urllib.request, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHANNEL = ROOT / "app" / "channel.json"
VERSION_F = ROOT / "VERSION"
MANIFEST_F = ROOT / "MANIFEST.json"
WORK = ROOT / ".update"
PY = ROOT / ".venv" / "bin" / "python"
ASSET = "liminal-engine-update.zip"
# never written by an update, whatever a zip says
PROTECTED = (".venv/", "models/stylegan/", "app/out/", "app/inbox/", "_cleanup/", ".update/", "app/lanes.json")
UA = {"User-Agent": "liminal-engine-updater", "Accept": "application/vnd.github+json"}

S = {"enabled": CHANNEL.exists(), "current": None, "latest": None, "notes": "", "available": False,
     "checked": 0, "error": "", "stage": "", "pct": 0, "url": None, "sha_url": None, "page": None}
_lock = threading.Lock()
RELAUNCH = None   # set by native.py: closes the window and reopens the app


def _ver(s):
    s = (s or "").strip().lstrip("vV")
    try: return tuple(int(x) for x in s.split("."))
    except ValueError: return ()


def _ctx():
    try:
        import certifi   # python.org builds ship without system certificates
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _get(url, timeout=15):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout, context=_ctx())


def current():
    try: return VERSION_F.read_text().strip()
    except OSError: return "0"


def state():
    S["current"] = current()
    return {k: v for k, v in S.items() if k not in ("url", "sha_url")}


def check(force=False):
    if not S["enabled"] or S["stage"] in ("downloading", "installing", "packages", "restarting"):
        return state()
    if not force and time.time() - S["checked"] < 1800:
        return state()
    S["checked"] = time.time()
    try:
        repo = json.loads(CHANNEL.read_text())["repo"]
        api = os.environ.get("LIMINAL_UPDATE_API", "https://api.github.com")   # override only for testing
        with _get(f"{api}/repos/{repo}/releases/latest") as r:
            rel = json.loads(r.read().decode())
        assets = {a["name"]: a["browser_download_url"] for a in rel.get("assets", [])}
        S.update(latest=rel.get("tag_name", "").lstrip("v"), notes=(rel.get("body") or "").strip(),
                 url=assets.get(ASSET), sha_url=assets.get(ASSET + ".sha256"), page=rel.get("html_url"), error="")
        S["available"] = bool(S["url"] and S["sha_url"] and _ver(S["latest"]) > _ver(current()))
    except Exception as e:
        S["error"] = f"couldn't check for updates ({type(e).__name__})"
    return state()


def check_soon():
    threading.Thread(target=check, daemon=True).start()


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def _safe(rel):
    return rel and not rel.startswith(("/", "\\")) and ".." not in Path(rel).parts and not rel.startswith(PROTECTED)


def apply(busy=None):
    """busy: callable returning a reason string if a render/download is running."""
    with _lock:
        if not S["available"]: return False, "no update"
        if S["stage"] in ("downloading", "installing", "packages", "restarting"): return False, "already updating"
        why = busy() if busy else None
        if why: return False, why
        S.update(stage="downloading", pct=0, error="")
    threading.Thread(target=_apply, daemon=True).start()
    return True, ""


def _apply():
    old = current()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = ROOT / "_cleanup" / f"update-{old}-{stamp}"
    done = []   # (rel, had_old) in the order written, for rollback
    try:
        if WORK.exists(): shutil.rmtree(WORK)
        WORK.mkdir(parents=True)
        z = WORK / ASSET
        with _get(S["url"], timeout=60) as r, open(z, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0); got = 0
            while True:
                b = r.read(1 << 16)
                if not b: break
                f.write(b); got += len(b)
                if total: S["pct"] = int(got * 100 / total)
        with _get(S["sha_url"]) as r:
            want = r.read().decode().split()[0].strip().lower()
        if _sha(z) != want:
            raise RuntimeError("the download was damaged (checksum mismatch); try again")

        S.update(stage="installing", pct=0)
        new = WORK / "new"
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if n.endswith("/"): continue
                if not _safe(n) and n != "MANIFEST.json": raise RuntimeError(f"unexpected file in update: {n}")
            zf.extractall(new)
        man = json.loads((new / "MANIFEST.json").read_text())
        files = man["files"]   # rel -> {"sha": ..., "mode": int}
        try: old_files = json.loads(MANIFEST_F.read_text()).get("files", {})
        except Exception: old_files = {}

        # packages first: if pip fails, nothing has been changed yet
        req_new, req_old = new / "requirements.txt", ROOT / "requirements.txt"
        if req_new.exists() and (not req_old.exists() or _sha(req_new) != _sha(req_old)) and PY.exists():
            S.update(stage="packages")
            p = subprocess.run([str(PY), "-m", "pip", "install", "-q", "-r", str(req_new)],
                               capture_output=True, text=True, timeout=1800)
            if p.returncode != 0:
                tail = (p.stderr or p.stdout).strip().splitlines()[-1:] or ["pip failed"]
                raise RuntimeError("couldn't install the new packages: " + tail[0][:200])
            S.update(stage="installing")

        todo = [rel for rel, meta in files.items() if _safe(rel) and
                (not (ROOT / rel).is_file() or _sha(ROOT / rel) != meta["sha"])]
        gone = [rel for rel in old_files if rel not in files and _safe(rel) and (ROOT / rel).is_file()]
        for i, rel in enumerate(todo):
            dst, src = ROOT / rel, new / rel
            had = dst.exists()
            if had:
                (backup / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dst, backup / rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".updating")
            shutil.copyfile(src, tmp)
            os.chmod(tmp, files[rel].get("mode", 0o644))
            os.replace(tmp, dst)
            done.append((rel, had))
            S["pct"] = int((i + 1) * 100 / max(1, len(todo)))
        for rel in gone:   # dropped from this version: moved aside, never deleted
            (backup / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(ROOT / rel), str(backup / rel))
        shutil.copyfile(new / "MANIFEST.json", MANIFEST_F)
        VERSION_F.write_text(man["version"] + "\n")
        shutil.rmtree(WORK, ignore_errors=True)
        S.update(available=False, current=man["version"], stage="restarting" if RELAUNCH else "done", pct=100)
        if RELAUNCH:
            threading.Timer(1.2, RELAUNCH).start()
    except Exception as e:
        for rel, had in reversed(done):   # put the old version back
            try:
                if had: shutil.copy2(backup / rel, ROOT / rel)
                else:
                    (backup / "_new-from-failed-update" / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(ROOT / rel), str(backup / "_new-from-failed-update" / rel))
            except Exception:
                pass
        S.update(stage="error", error=str(e) or type(e).__name__, pct=0)
