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
#  Requires the modern environment (see requirements-modern.txt). Use a plain
#  venv, not conda: every DLL failure in docs/known_issues.md 1 came from
#  conda's layout, and the macOS build that does work was made this way.
#      python3 -m venv ~/seizmodern
#      source ~/seizmodern/bin/activate
#      pip install -r requirements-modern.txt pyinstaller
#
#  SEIZ_DIST and SEIZ_WORK move PyInstaller's output off the repository, which
#  matters under WSL: the repo lives on /mnt/c and every one of the ~1.2 GB it
#  writes would cross the 9p filesystem boundary. Point them at the Linux
#  filesystem and copy the result back once, at the end.
#      SEIZ_DIST=~/seizdist SEIZ_WORK=~/seizbuild bash packaging/build_app.sh
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
# One interpreter per module, not one `discover` run over all of them.
#
# On macOS the single-process run segfaults (exit 139) partway through, during
# Qt teardown, once enough PyQt5/pyqtgraph test modules have accumulated in one
# interpreter. It is not a defect in any test or in the application: every
# module passes on its own, module pairs pass repeatedly, and the frozen app
# builds all 157 widgets. Windows and Linux do not hit it.
#
# This runs the SAME tests and suppresses nothing - each module must still exit
# 0, and the loop stops at the first that does not. Isolation is stricter than
# `discover`, since no module can leave state behind for the next one, which
# also makes an ordering-dependent test fail here rather than pass by luck.
_n_mod=0
for _f in "$REPO"/tests/test_*.py; do
    _m="tests.$(basename "$_f" .py)"
    if ! "$PYEXE" -m unittest "$_m" -q >/dev/null 2>&1; then
        echo "    FAILED in $_m - rerun it directly to see why:"
        echo "        $PYEXE -m unittest $_m -v"
        exit 1
    fi
    _n_mod=$((_n_mod + 1))
done
echo "    OK ($_n_mod modules)"

DIST="${SEIZ_DIST:-$REPO/dist}"
WORK="${SEIZ_WORK:-$REPO/build/pyi}"

echo
echo "[3/4] Freezing with PyInstaller. This takes several minutes..."
echo "      dist -> $DIST"
"$PYEXE" -m PyInstaller packaging/SeizureReview.spec --noconfirm \
    --distpath "$DIST" --workpath "$WORK"

echo
echo "[4/4] Smoke-testing the frozen application..."
"$PYEXE" packaging/smoke_test.py --dist "$DIST/SeizureReview"

cat <<EOF

============================================================
 Build complete:  $DIST/SeizureReview

 macOS note: the app is not notarised, so the first launch
 needs right-click -> Open rather than a double-click.
 See docs/DEPLOYMENT.md.
============================================================
EOF
