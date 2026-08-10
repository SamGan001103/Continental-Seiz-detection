# Building the Linux app — via WSL on this Windows PC

You do not need a separate Linux machine. WSL2 runs a real Linux kernel, and a PyInstaller build
made inside it is an ordinary Linux ELF binary that runs on a normal Ubuntu machine.

**Status: built and verified on 2026-08-10.** All four gates pass, including gate 4 on a real
recording. What follows is the procedure that was actually executed, not a plan.

---

## Step 1 — install WSL and Ubuntu 24.04

Open **PowerShell as Administrator** — press Start, type `powershell`, right-click *Windows
PowerShell*, choose **Run as administrator** — then:

```powershell
wsl --install                              # the feature; then REBOOT
wsl --install -d Ubuntu-24.04 --no-launch  # the distribution
```

`wsl --install` on its own installs the *feature* but no distribution, and `wsl --list --verbose`
afterwards reports "has no installed distributions". The second command is not optional.
`--no-launch` skips the interactive username prompt, which lets the rest be scripted; commands
then run as root via `wsl -d Ubuntu-24.04 -u root`.

### Why 24.04 and not 22.04 — and what it costs

An earlier version of this document specified Ubuntu **22.04**, on the reasoning that its older
glibc (2.35) produces a binary that runs on more machines. That reasoning is correct and still
worth understanding, but 22.04 cannot build this application:

* `scipy==1.17.1` requires **Python ≥ 3.11**, so Ubuntu 22.04's default python3.10 cannot install
  the pinned stack at all — `No matching distribution found for scipy==1.17.1`.
* 22.04's `python3.11` package is **`3.11.0~rc1`**, a release candidate. Shipping a clinical
  review tool on an RC interpreter to gain a lower glibc is a bad trade.

24.04 ships stable **Python 3.12.3** in the official archive, matching the macOS build's 3.12,
with no third-party PPA in the provenance.

**The cost, measured rather than estimated.** The bundle's floor is set by whichever bundled
shared object demands the newest symbol — not by the launcher, which reports a misleadingly low
`GLIBC_2.14` because it is a tiny C program. Measured across all 538 bundled objects:

```
GLIBC_2.38   libpython3.12.so.1.0, libstdc++.so.6, libsystemd.so.0,
             libxkbcommon.so.0, libkrb5.so.3, _decimal...so
```

| target | glibc | runs? |
|---|---|---|
| Ubuntu 24.04 | 2.39 | yes |
| Ubuntu 23.10 | 2.38 | yes |
| Debian 12 | 2.36 | **no** |
| Ubuntu 22.04 LTS | 2.35 | **no** |
| RHEL / Rocky / Alma 9 | 2.34 | **no** |

If Linux reach ever matters, build on the oldest host that can supply a stable Python ≥ 3.11:
AlmaLinux 9 (glibc 2.34, python3.12 from AppStream) would cover every row above, and is available
as `wsl --install -d AlmaLinux-9`. That was not done here because Linux is the least likely
platform for this tool and the constraint is documented rather than hidden.

## Step 2 — the build

Everything below was run as root inside `wsl -d Ubuntu-24.04`.

```bash
apt-get update
apt-get install -y python3.12 python3.12-venv python3-pip binutils \
    libgl1 libegl1 libxkbcommon-x11-0 libdbus-1-3 \
    libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
    libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xfixes0

python3.12 -m venv ~/seizmodern
~/seizmodern/bin/python -m pip install --upgrade pip
cd /mnt/c/Users/User/Continental-Seiz-detection
~/seizmodern/bin/python -m pip install -r requirements-modern.txt pyinstaller==6.22.0

SEIZ_PYTHON=~/seizmodern/bin/python \
SEIZ_DIST=~/seizdist SEIZ_WORK=~/seizbuild \
    bash packaging/build_app.sh
```

**`SEIZ_DIST` and `SEIZ_WORK` matter here.** The repository lives on `/mnt/c`, reached through
WSL's 9p filesystem, and PyInstaller writes about 1.8 GB. Sending that across the boundary is
slow; sending it to the Linux filesystem is not. The repository itself stays on `/mnt/c`, which
is what lets gate 4 reach `sample_data/` and score a **real** recording rather than the synthetic
fallback.

`binutils` is required and easy to miss — PyInstaller calls `objdump` to walk shared-library
dependencies.

The `libxcb-*` packages are Qt's X11 dependencies. The bundle carries Qt itself but not the
system libraries it links against, and without them the app fails at startup with *"could not
load the Qt platform plugin xcb"* — the single most common Linux packaging failure. They are
needed on the **target** machine too.

**A venv, not conda, on purpose.** Every DLL problem in the Windows modern-stack build came from
conda's layout (`docs/known_issues.md` §1). A plain venv installs wheels from PyPI with their
shared libraries beside the extension modules, which is where PyInstaller looks. Both builds that
work — macOS and Linux — were made this way.

## Step 3 — what was verified

| gate | result |
|---|---|
| 1. weights hash | OK — `90c046ee80b50499…` |
| 2. test suite | 176 tests, OK (4 skipped) |
| 3. freeze | OK — 1.8 GB, ELF 64-bit x86-64 |
| 4. smoke test | OK — GUI builds 131 widgets; **scored a real 31 s recording**, 4/4 windows |

Model parameter count 384846, matching Windows and macOS.

To actually *see* the GUI, Windows 11 includes WSLg and `~/seizmodern/bin/python -m gui.main`
opens a real window. On Windows 10 you would need a separate X server, which is not worth setting
up just for this — the offscreen self-test proves the widgets construct.

## Library versions

Identical to the macOS build, which was the point of pinning them:

| | macOS (arm64) | Linux (x86-64) |
|---|---|---|
| python | 3.12.10 | 3.12.3 |
| numpy | 1.26.4 | 1.26.4 |
| scipy | 1.17.1 | 1.17.1 |
| scikit-learn | 1.9.0 | 1.9.0 |
| mne | 1.12.1 | 1.12.1 |
| tensorflow | 2.21.0 | 2.21.0 |

## A second source of numerical drift, specific to this platform

TensorFlow on x86-64 enables **oneDNN**, and says so at import:

> oneDNN custom operations are on. You may see slightly different numerical results due to
> floating-point round-off errors from different computation orders.

Until now `docs/portability.md` attributed all cross-platform variation to the non-converged ICA.
That is incomplete: on Linux there is also a TensorFlow-level reordering that macOS arm64 does not
perform. Setting `TF_ENABLE_ONEDNN_OPTS=0` disables it, which is the lever for separating the two
effects if a measurement ever needs to.

## Expected problems

| symptom | fix |
|---|---|
| `has no installed distributions` after `wsl --install` | the feature installed but no distro; run `wsl --install -d Ubuntu-24.04` |
| `No matching distribution found for scipy==1.17.1` | Python is older than 3.11. Check `python3 --version` inside the venv. |
| `could not load the Qt platform plugin "xcb"` | install the `libxcb-*` packages above on the **target** machine too |
| `Permission denied` running the app | `chmod +x SeizureReview` — FAT32/exFAT USB sticks do not carry the executable bit |
| `GLIBC_2.38 not found` on the target | the target is older than Ubuntu 23.10. See the table above; rebuild on AlmaLinux 9. |
| PyInstaller cannot find `objdump` | `apt-get install binutils` |
