# Deploying the Seizure Review application

How this software gets onto a hospital PC, why it is built the way it is, and
what to do when it does not start.

---

## 1. Why a desktop application and not a web service

This was a real architectural choice, so it is recorded rather than assumed.

| | **Desktop app (chosen)** | Web/server |
|---|---|---|
| Where EEG lives | Stays on the workstation | Uploaded to a server |
| Ethics / governance | No data transfer to justify | PHI on the wire — needs a data-transfer agreement and an infosec review |
| Hospital IT involvement | Copy a folder. No admin rights, no ports, no install | Server provisioning + security assessment, typically months |
| Network outage | Works | Does not work |
| 200 MB recordings | Read locally | Uploaded per recording |
| Concurrent users | One reviewer per machine | Many — which this project does not have |
| Updates | Re-copy the folder | Central (the one real advantage) |
| GPU | Whatever the workstation has | Possible | 

The deciding factor is governance, not engineering. A hospital will let a
student plug in a USB stick long before it will host a server holding patient
EEG, and every hour spent on that approval is an hour not spent on the thesis.
The desktop build also keeps the strongest privacy claim available: **the
recording never leaves the machine it was opened on.**

If compute later becomes the bottleneck, an *optional* remote scoring backend
can be added without changing how the software is deployed.

---

## 2. What gets built

`packaging/build_app.bat` produces:

```
dist/SeizureReview/
    SeizureReview.exe            <- double-click this
    convlstm_ICA_12_train.h5     <- the reviewed model weights
    utils/params_*.txt           <- electrode montage definitions
    doc/                         <- README, INSTALL, RESULTS, INTENDED_USE
    ... ~1.2 GB of Python, Qt, TensorFlow and MNE runtime ...
```

It is a **one-folder** PyInstaller build. Three choices in
`packaging/SeizureReview.spec` are deliberate and should not be "optimised"
away:

- **One folder, not one file.** A one-file build unpacks the whole 1.2 GB into
  `%TEMP%` on *every* launch. It is slow, and a large self-extracting
  executable is exactly the shape enterprise endpoint protection blocks.
- **UPX compression disabled.** UPX-packed executables are heuristically
  classified as malware by most enterprise antivirus. Saving disk is not worth
  failing the scan on the one machine that matters.
- **No console window, but everything is logged.** There is no terminal on a
  clinical PC, so `gui/main.py` installs an exception hook that writes to
  `%LOCALAPPDATA%\SeizureReview\logs\seizure_review.log` and shows the path in
  a dialog the user can read out over the phone.

### Building

On a development machine with the `seiz36` environment:

```bat
packaging\build_app.bat
```

It refuses to build if the model weights hash does not match
`eval_config.WEIGHTS_SHA256`, or if the test suite fails. Both are fatal rather
than warnings: a build carrying the wrong weights is indistinguishable from a
correct one at run time.

After freezing it runs `packaging/smoke_test.py`, which launches the *frozen
executable* and scores a real recording end to end. This is not optional
belt-and-braces — a PyInstaller build routinely compiles cleanly and then dies
on the first function-level import it never bundled. Nothing ships without
passing it.

---

## 3. Getting 1.2 GB to the hospital

Too big to email. In rough order of preference:

| Method | Notes |
|---|---|
| **USB stick** | Simplest and works when the network does not. Bring one to any clinician session. |
| **Hospital network share** | Good if you already have access. Ask before copying a gigabyte onto a clinical file server. |
| **Zip + institutional file transfer** (CloudStor, OneDrive, USyd's own service) | Zips to roughly half that — measure it, do not quote this from memory. Use the university's service, not a personal Dropbox — a hospital will care about that distinction. |

Zip the folder — not the contents — so the recipient cannot accidentally
extract a headless `.exe`:

```powershell
Compress-Archive -Path dist\SeizureReview -DestinationPath SeizureReview.zip
Get-FileHash SeizureReview.zip -Algorithm SHA256
```

Send the hash alongside it. A truncated download produces an application that
starts and then fails in an unobvious place, and the hash is the only cheap way
to rule that out before spending an hour on it.

> **Do not put this on a public download page.** The bundle contains the model
> weights and the application is not code-signed or registered as a medical
> device. Distribute it deliberately, to named people, with
> `docs/INTENDED_USE.md` attached.

---

## 3b. Preparing the USB stick

```
packaging\build_app.bat            # or: bash packaging/build_app.sh
python packaging/make_usb.py --dest E:/SeizureReview
```

`make_usb.py` is **additive**: run it once per platform, pointing at the same stick, and it adds
that platform's build without disturbing the others. This is not a convenience — **PyInstaller
cannot cross-compile**, so a macOS build has to be made on a Mac and a Linux build on Linux.
There is no way to produce all three from one machine, and the script writes an explicit
`NOT_BUILT.txt` into any slot it cannot fill rather than leaving an empty folder that looks like
a copy failure.

```
SeizureReview/
    START_HERE.txt      plain text, readable on any machine, no viewer needed
    windows/            ~1.2 GB
    macos/              built on the Mac
    linux/              built on Linux
    source/             the repository, so a missing platform can be built from the stick itself
    docs/               DEPLOYMENT, INTENDED_USE, RESULTS, portability, source_verification
    CHECKSUMS.txt       SHA-256 of every executable and of the model weights
```

Use a stick of **32 GB or more** to carry all three builds, and prefer USB 3.0 — copying 1.2 GB
over USB 2.0 takes about ten minutes per platform.

**Check the checksums after copying.** A truncated copy produces an application that starts and
then fails somewhere unobvious; `CHECKSUMS.txt` is the cheap way to rule that out before spending
an hour on it. `certutil -hashfile <file> SHA256` on Windows, `shasum -a 256 <file>` elsewhere.

---

## 3c. Setting up on the target machine, per platform

**The one rule that applies everywhere: copy the folder off the USB first.** Do not run it from
the stick. It is slow, USB drives are often mounted read-only or `noexec`, and the application
needs somewhere to write its cache and logs.

### Windows

1. Copy `windows\SeizureReview` somewhere the logged-in user can write —
   `C:\Users\<name>\SeizureReview` is ideal. **Not** `Program Files`, which needs admin.
2. Double-click `SeizureReview.exe`.
3. SmartScreen shows *"Windows protected your PC"* because the app is unsigned →
   **More info → Run anyway**. Warn the clinician *before* they see this.
4. Optional: right-click → *Send to* → *Desktop (create shortcut)*.

### macOS (Apple Silicon)

1. Copy `macos/SeizureReview` off the stick.
2. **Right-click the app and choose Open — do not double-click.** Then *Open* again in the dialog.
   Gatekeeper is stricter than SmartScreen: double-clicking an un-notarised app gives a dead end
   with no "open anyway" button, whereas right-click → Open offers one. Once only.
3. If macOS still refuses — common for anything that arrived on a USB, which sets the quarantine
   attribute:
   ```
   xattr -dr com.apple.quarantine /path/to/SeizureReview
   ```
4. Notarisation removes both steps and needs a paid Apple Developer account.

### Linux

1. Copy `linux/SeizureReview` off the stick.
2. `chmod +x SeizureReview` — the executable bit is lost on FAT32/exFAT sticks.
3. `./SeizureReview`
4. If Qt reports a missing platform plugin, install the system X11/xcb libraries. The bundle
   carries Qt itself but not the OS-level display libraries.

### Where each platform writes

Nothing is written into the application folder, so it can live on a read-only share.

| | per-user data directory |
|---|---|
| Windows | `%LOCALAPPDATA%\SeizureReview\` |
| macOS | `~/Library/Application Support/SeizureReview/` |
| Linux | `$XDG_DATA_HOME/SeizureReview/`, else `~/.local/share/SeizureReview/` |

Each holds `logs/seizure_review.log` and, when the recording's own folder is not writable,
`cache/`.

### First run, on any platform

- **Bring recordings.** None are on the USB — TUSZ needs its own free registration, and no
  patient data may travel on a stick.
- The first open of a recording scores it, which takes minutes; every open after that is instant,
  because the result is cached beside the recording.
- To make a demo instant, pre-score in advance and copy the `.probs.npz` sidecars alongside the
  EDFs: `python precompute_probs.py <folder>`.
- **Numbers differ slightly between platforms.** The ICA does not converge, so a 10^-15 difference
  in linear algebra changes the decomposition — see `docs/portability.md`. No detection decision
  changed across the 74 windows tested, but do not mix figures from two platforms in one table.

---

## 4. Installing on a hospital PC

**Requirements:** 64-bit Windows 10 or 11, ~2.5 GB free disk, 8 GB RAM.
No Python. No conda. No internet. No administrator rights.

1. Copy the whole `SeizureReview` folder to the PC — USB stick, network share,
   or the user's own Documents folder. Copy the **entire** folder; the `.exe`
   alone will not run.
2. Put it somewhere the logged-in user can write, e.g.
   `C:\Users\<name>\SeizureReview`. `C:\Program Files` needs admin rights and
   is not required.
3. Double-click `SeizureReview.exe`.
4. Optionally right-click it → *Send to* → *Desktop (create shortcut)*.

### First launch

Windows SmartScreen will show *"Windows protected your PC"* because the
executable is not code-signed. Click **More info → Run anyway**. This is
expected for unsigned software and is worth warning the clinician about
*before* they see it, so it does not read as a security incident.

If the hospital's antivirus quarantines the folder, IT will need to allow-list
that path. Code-signing the executable would avoid both prompts; it needs a
certificate the project does not currently have. Flag it as a known gap rather
than working around it.

### Where the application writes

Nothing is written into the application folder, so it can live on a read-only
share. Per-user data goes to `%LOCALAPPDATA%\SeizureReview\`:

- `logs\seizure_review.log` — crash reports
- `cache\` — model probabilities, **only** when the recording's own folder is
  not writable

Model probabilities are normally cached as a `.probs.npz` sidecar next to the
EDF, so they travel with the recording. Clinical recordings usually sit on a
read-only share, so a failed sidecar write falls back to the per-user cache
instead of discarding a multi-minute inference run.

---

## 5. Performance expectations

Inference is CPU-only and is dominated by per-window ICA (~90 % of the loop —
the neural network is only ~7 %). Measured at **0.043× real time** on the
development machine; see `docs/RESULTS.md` §6.

| Recording length | First open (dev machine) | Allow, on slower hardware |
|---|---|---|
| 5 min | ~13 s | under a minute |
| 30 min | ~1.3 min | ~4 min |
| 60 min | ~2.5 min | ~8 min |

The right-hand column assumes a clinical workstation roughly 3× slower than the
machine these were measured on — a guess, not a measurement. **Time one
recording on the actual target PC before promising a clinician anything.**

Re-opening the same recording is instant — the cache is reused. For a review
session with known recordings, pre-scoring them in advance is still worth it:

```bat
precompute_probs.py <folder-of-edfs>
```

The resulting `.probs.npz` files can be copied alongside the EDFs, and the
application will pick them up with no compute at all.

---

## 6. Troubleshooting

**Nothing happens when I double-click it.**
Check `%LOCALAPPDATA%\SeizureReview\logs\seizure_review.log`. If there is no
log at all, the executable was blocked before Python started — antivirus or
an incomplete copy. Re-copy the whole folder.

**"Model weights not found".**
The folder was copied incompletely. `convlstm_ICA_12_train.h5` must sit beside
`SeizureReview.exe`.

**"Could not find the required 19 EEG channels in this EDF".**
The recording does not carry a standard 10–20 montage under a label the
loader recognises. The detector requires all 19; see
`utils/params_common_electrodes.txt` for the accepted names.

**It is very slow.**
Expected — see the table above. Pre-score with `precompute_probs.py`.

---

## 7. Known gaps

Recorded honestly, because a deployment document that only lists successes is
not useful to anyone who has to rely on it.

- **Not code-signed.** SmartScreen and antivirus prompts on first run.
- **No auto-update.** A new version is a new folder copy.
- **CPU-only.** No GPU path is bundled; a GPU build would be a separate spec.
- **Not a medical device.** The application is a research prototype for
  supervised review. It is not registered with the TGA, has not been through
  clinical validation, and must not be used to make a clinical decision
  without a qualified reviewer reading the underlying EEG. See
  `docs/INTENDED_USE.md`.
