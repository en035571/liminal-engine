#!/usr/bin/env python
"""LIMINAL ENGINE — native window. Starts the local server in-process, opens a WebKit window on it.
Closing the window stops the server (a running render is stopped too; it starts from the beginning next time)."""
import os, sys, threading, time, urllib.request, subprocess
from pathlib import Path
for _p in ("/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin"):
    if os.path.isdir(_p) and _p not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
os.environ.setdefault("PORT", "8391")
PORT = int(os.environ["PORT"])

import server  # noqa: E402  (the stdlib http server; import binds nothing yet)
import updater  # noqa: E402
import shlex  # noqa: E402

RELAUNCHING = False
def relaunch():
    """After an update: close this window, wait for this process to end, open the app again."""
    global RELAUNCHING
    RELAUNCHING = True
    app = ROOT / "Liminal Engine.app"
    again = f"open {shlex.quote(str(app))}" if app.is_dir() else \
            f"cd {shlex.quote(str(ROOT))} && exec ./.venv/bin/python app/native.py"
    subprocess.Popen(["/bin/sh", "-c", f"while kill -0 {os.getpid()} 2>/dev/null; do sleep 0.2; done; {again}"],
                     start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    w.destroy()
updater.RELAUNCH = relaunch
from http.server import ThreadingHTTPServer

def serve():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), server.H)
    srv.serve_forever()

# Another copy may already be serving this port WITH A RENDER IN FLIGHT.
# Killing it blindly used to orphan that render and report IDLE in the new window.
def port_busy():
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/state", timeout=1.5)
        return True
    except Exception:
        return False


if port_busy():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/state", timeout=2) as r:
            import json as _json
            st = _json.loads(r.read().decode())
        running = bool((st.get("job") or {}).get("running"))
    except Exception:
        running = False
    if running:
        sys.exit("A LIMINAL ENGINE is already running a render on this port.\n"
                 "Use that window, or stop the render there first.")
    subprocess.run(["pkill", "-f", f"app/server.py"], capture_output=True)
    time.sleep(0.4)
threading.Thread(target=serve, daemon=True).start()
for _ in range(50):
    try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/state", timeout=1); break
    except Exception: time.sleep(0.1)

import webview  # noqa: E402
w = webview.create_window("LIMINAL ENGINE", f"http://127.0.0.1:{PORT}/", width=1440, height=920, min_size=(1100, 700), background_color="#0e0e10")
selftest = "--selftest" in sys.argv
def on_start():
    if selftest:
        time.sleep(3); w.destroy()
webview.start(on_start, private_mode=False)
# window closed → stop any running render cleanly (renders do not resume; the next run starts over)
if server.job and server.job.get("running") and server.job.get("pid"):
    server.job["stopping"] = True
    try: os.killpg(os.getpgid(server.job["pid"]), 15)
    except Exception:
        try: os.kill(server.job["pid"], 15)
        except Exception: pass
if RELAUNCHING:
    os._exit(0)   # don't let a leftover thread keep this process (and the relaunch) waiting
