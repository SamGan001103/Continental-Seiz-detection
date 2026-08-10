# Building the Linux app — via WSL on this Windows PC

You do not need a separate Linux machine. WSL2 runs a real Linux kernel, and a PyInstaller build
made inside it is an ordinary Linux ELF binary that runs on a normal Ubuntu machine.

---

## Step 1 — install WSL (you must do this; it needs administrator rights)

Open **PowerShell as Administrator** — press Start, type `powershell`, right-click *Windows
PowerShell*, choose **Run as administrator** — then:

```powershell
wsl --install -d Ubuntu-22.04
```

Then **reboot**. On first launch it asks for a UNIX username and password; anything is fine, and
you will rarely need them.

Ubuntu **22.04** is specified deliberately. Its glibc 2.35 is older than 24.04's, and a binary
built against an older glibc runs on newer systems while the reverse is not true. Building on the
oldest glibc you can tolerate is what makes the result portable.

## Step 2 — check it worked

Back in a normal terminal:

```powershell
wsl --list --verbose        # Ubuntu-22.04 should be listed, VERSION 2
wsl -- uname -a             # should print a Linux kernel
```

## Step 3 — everything from here can be automated

Tell me once WSL is installed and I will run the rest. For reference, it is:

```bash
# inside WSL
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3-pip \
    libgl1 libegl1 libxkbcommon-x11-0 libdbus-1-3 \
    libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
    libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xfixes0

cd /mnt/c/Users/User/Continental-Seiz-detection
python3.11 -m venv ~/seizmodern
source ~/seizmodern/bin/activate
pip install -r requirements-modern.txt pyinstaller

python -m unittest discover -s tests -q      # expect 176
bash packaging/build_app.sh
python packaging/make_usb.py --dest /mnt/e/SeizureReview
```

The `libxcb-*` packages are Qt's X11 dependencies. The bundle carries Qt itself but not the
system libraries it links against, and without them the app fails at startup with *"could not
load the Qt platform plugin xcb"* — the single most common Linux packaging failure.

**A venv, not conda, on purpose.** Every DLL problem in the Windows modern-stack build came from
conda's layout (`docs/known_issues.md` §1). A plain venv installs wheels from PyPI with their
shared libraries beside the extension modules, which is where PyInstaller looks.

## Step 4 — testing it

The build's own smoke test runs headless and needs no display, so it works in WSL as-is.

To actually *see* the GUI, Windows 11 includes WSLg and `python -m gui.main` opens a real window.
On Windows 10 you would need a separate X server, which is not worth setting up just for this —
the offscreen self-test proves the widgets construct.

## What the result runs on

A normal x86-64 Linux desktop with glibc ≥ 2.35 — Ubuntu 22.04 and later, Debian 12, RHEL 9 and
equivalents. **Not** Linux on ARM, which would need building on ARM hardware.

## Expected problems

| symptom | fix |
|---|---|
| `could not load the Qt platform plugin "xcb"` | install the `libxcb-*` packages above on the **target** machine too |
| `Permission denied` running the app | `chmod +x SeizureReview` — FAT32/exFAT USB sticks do not carry the executable bit |
| `GLIBC_2.xx not found` on the target | the build machine's glibc is newer than the target's. Build on an older Ubuntu. |
| TensorFlow fails to load its native runtime | the Windows failure in `known_issues.md` §1. If it recurs here, try `tensorflow==2.10`. |
