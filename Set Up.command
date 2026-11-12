#!/bin/bash
# Liminal Engine — one-time setup. Double-click (first time: right-click → Open).
# Installs into a .venv folder right here; nothing else on the Mac is changed except Homebrew packages if you say yes.
# Safe to run again: it keeps what's already installed and fills in anything missing.
cd "$(dirname "$0")" || exit 1
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
say(){ printf "\n\033[1m%s\033[0m\n" "$1"; }
pause(){ read -n 1 -s -r -p "Press any key to close this window."; echo; }
clear; echo "  LIMINAL ENGINE · setup  ($(cat VERSION 2>/dev/null))"; echo "  ─────────────────────"

# 1. Python 3.10 – 3.12 (the AI libraries are tested on these). Each candidate is RUN, not just found.
ok(){ "$1" -c 'import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)' >/dev/null 2>&1; }
PY=""
for c in python3.11 python3.12 python3.10 /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.10 \
         /usr/local/bin/python3.11 /usr/local/bin/python3.12 /usr/local/bin/python3.10 \
         /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 python3; do
  p=$(command -v "$c" 2>/dev/null) || continue; if ok "$p"; then PY="$p"; break; fi; done
HAVE_BREW=0; command -v brew >/dev/null 2>&1 && HAVE_BREW=1
if [ -x .venv/bin/python ] && ok .venv/bin/python; then PY=.venv/bin/python; fi
if [ -z "$PY" ]; then
  say "Python 3.11 is needed."
  if [ $HAVE_BREW = 1 ]; then read -r -p "Install it with Homebrew now? [Y/n] " a; if [ "${a:-y}" != "n" ]; then brew install python@3.11 && PY=/opt/homebrew/bin/python3.11; [ -x "$PY" ] || PY=/usr/local/bin/python3.11; fi; fi
  if [ -z "$PY" ] || ! ok "$PY"; then echo "Install Python 3.11 from https://www.python.org/downloads/macos/ , then run Set Up again."; open "https://www.python.org/downloads/macos/"; pause; exit 1; fi
fi
echo "Python: $PY ($("$PY" -V 2>&1))"

# 2. ffmpeg (renders, sound, video in and out)
if ! command -v ffmpeg >/dev/null 2>&1; then
  say "ffmpeg is needed."
  if [ $HAVE_BREW = 1 ]; then read -r -p "Install it with Homebrew now? [Y/n] " a; [ "${a:-y}" != "n" ] && brew install ffmpeg; fi
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Install Homebrew from https://brew.sh (one line in Terminal), then run Set Up again; it will install ffmpeg."
    open "https://brew.sh"; pause; exit 1; fi
fi
echo "ffmpeg: $(command -v ffmpeg)"

# 3. the engine's own Python environment
say "Installing the engine (about 2 minutes)…"
[ -x .venv/bin/python ] || "$PY" -m venv .venv || { echo "Could not make the .venv folder."; pause; exit 1; }
.venv/bin/python -m pip install -q --upgrade pip setuptools wheel
.venv/bin/python -m pip install -q -r requirements.txt || { echo "Install failed (see above). Check the internet connection and run Set Up again."; pause; exit 1; }

# 4. latent dream needs PyTorch (~400 MB) and at least one model
if .venv/bin/python -c "import torch" >/dev/null 2>&1; then
  echo "PyTorch: already installed"; AI=1
else
  say "Latent Dream (the AI picture generator) needs PyTorch, about a 400 MB download."
  read -r -p "Install it now? [Y/n] " a; AI=0
  if [ "${a:-y}" != "n" ]; then .venv/bin/python -m pip install -q torch && AI=1 || echo "PyTorch did not install; everything except Latent Dream still works."; fi
fi
mkdir -p models/stylegan
if [ "$AI" = 1 ] && ! ls models/stylegan/*.pkl >/dev/null 2>&1; then
  say "Getting a first model for Latent Dream (microscope images, about 350 MB)…"
  if ! command -v megadl >/dev/null 2>&1 && [ $HAVE_BREW = 1 ]; then brew install megatools >/dev/null 2>&1; fi
  if command -v megadl >/dev/null 2>&1 && megadl --path models/stylegan "https://mega.nz/#!PbgzWTZT!JbVpqgMU7AOg-sQUoG1BDepuwKtgAsLgjd4YwlTXlpc"; then
    echo "Model: done"
  else
    echo "Couldn't download it now. In the app, open Models and pick any model to download."
  fi
fi

mkdir -p "$HOME/Movies/Liminal Engine"
xattr -dr com.apple.quarantine "Liminal Engine.app" 2>/dev/null
say "Done. Open \"Liminal Engine\" in this folder."
echo "Renders are saved to  Movies › Liminal Engine."
echo "Updates arrive inside the app; you won't need to download it again."
[ -n "$LIMINAL_TEST" ] || open "$(pwd)"; [ -n "$LIMINAL_TEST" ] || pause
