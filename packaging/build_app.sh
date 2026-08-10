#!/usr/bin/env bash
# ===================================================================
#  Build the distributable Seizure Review application on macOS or Linux.
#
#  The POSIX counterpart of packaging/build_app.bat, with the same four
#  gates. PyInstaller does not cross-compile, so run this ON the machine
#  you are building for.
#
#      bash packaging/build_app.sh
#
#  Requires the modern environment (see requirements-modern.txt):
#      conda create -n seizmodern python=3.11
#      conda activate seizmodern
#      pip install -r requirements-modern.txt pyinstaller
#
#  Read docs/portability.md first. The modern stack does not reproduce the
#  Python 3.6 build bit-for-bit — the non-converged ICA makes that impossible
#  on any platform — so caches built here differ in the third decimal from
#  those built on Windows. Regenerate caches on the machine you evaluate on,
#  and do not mix figures from two platforms in one table.
# ===================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PYEXE="${SEIZ_PYTHON:-python}"
if ! command -v "$PYEXE" >/dev/null 2>&1; then
    echo "[X] No python found. Set SEIZ_PYTHON or activate the environment."
    exit 1
fi
echo "using $("$PYEXE" --version 2>&1) at $(command -v "$PYEXE")"

echo
echo "[1/4] Checking the model weights are present and unmodified..."
"$PYEXE" - <<'PY'
import sys
sys.path.insert(0, '.')
import eval_config as c
from gui.io.cache import sha256_file
h = sha256_file(c.WEIGHTS)
if h != c.WEIGHTS_SHA256:
    print('    weights missing or hash mismatch')
    sys.exit(1)
print('    OK')
PY

echo
echo "[2/4] Running the test suite..."
"$PYEXE" -m unittest discover -s tests -q
echo "    OK"

echo
echo "[3/4] Freezing with PyInstaller. This takes several minutes..."
"$PYEXE" -m PyInstaller packaging/SeizureReview.spec --noconfirm \
    --distpath dist --workpath build/pyi

echo
echo "[4/4] Smoke-testing the frozen application..."
"$PYEXE" packaging/smoke_test.py

cat <<EOF

============================================================
 Build complete:  $REPO/dist/SeizureReview

 macOS note: the app is not notarised, so the first launch
 needs right-click -> Open rather than a double-click.
 See docs/DEPLOYMENT.md.
============================================================
EOF
