"""GUI helpers for optional full-file ZUNA runs.

The GUI runs in the legacy seizure-detector environment, while ZUNA needs a
modern Python stack. This module keeps that boundary explicit: the GUI starts
``utils/zuna_bridge.py`` in a separate interpreter and then consumes the
exported repo-format NPZ.
"""
import hashlib
import os
import signal
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def zuna_available():
    """Whether a full-ZUNA run can be started at all.

    ZUNA is a side-study whose result was negative, and it is a *research*
    capability: it needs a second Python interpreter with a modern stack,
    ``utils/zuna_bridge.py``, and a writable ``artifacts/`` tree. None of those
    exist in the packaged application, which ships one frozen interpreter and
    may sit on a read-only share.

    Without this check the frozen build still showed a "Run full ZUNA" button
    and a ZUNA entry in the AI-source list. A clinician pressing it would get a
    subprocess failure referring to a script they do not have — the worst
    possible moment for an unexplained error, and for a feature that has no
    business being in a clinical deployment in the first place.
    """
    from gui.paths import is_frozen
    if is_frozen():
        return False
    return os.path.exists(os.path.join(REPO, 'utils', 'zuna_bridge.py'))


def _safe_stem(edf_path):
    stem = os.path.splitext(os.path.basename(edf_path))[0]
    seed = os.path.abspath(edf_path)
    try:
        st = os.stat(edf_path)
        seed = '{}|{}|{:.6f}'.format(seed, int(st.st_size), float(st.st_mtime))
    except OSError:
        pass
    digest = hashlib.sha1(seed.encode('utf-8')).hexdigest()
    return '{}_{}'.format(stem, digest[:10])


def zuna_session_paths(edf_path):
    """Return stable artifact paths for one EDF's full-ZUNA GUI session."""
    session = _safe_stem(edf_path)
    root = os.path.join(REPO, 'artifacts', 'gui_zuna', session)
    stem = os.path.splitext(os.path.basename(edf_path))[0]
    fif_dir = os.path.join(root, 'fif')
    work_dir = os.path.join(root, 'full_zuna')
    out_fif = os.path.join(work_dir, '4_fif_output', stem + '.fif')
    return {
        'root': root,
        'stem': stem,
        'fif_dir': fif_dir,
        'fif': os.path.join(fif_dir, stem + '.fif'),
        'work_dir': work_dir,
        'out_fif': out_fif,
        'npz': os.path.join(root, stem + '.zuna_19ch.npz'),
        'probs': os.path.join(root, stem + '.zuna.probs.npz'),
        'log': os.path.join(root, 'zuna_full.log'),
    }


def find_zuna_python():
    """Find the modern Python interpreter used for ZUNA.

    ``ZUNA_PYTHON`` is the explicit override. The default candidates match the
    local setup used during the thesis experiments.
    """
    override = os.environ.get('ZUNA_PYTHON')
    if override:
        return override if os.path.exists(override) else None

    home = os.path.expanduser('~')
    candidates = [
        os.path.join(home, 'miniconda3', 'envs', 'zuna-gpu', 'python.exe'),
        os.path.join(home, 'miniconda3', 'envs', 'zuna-eeg', 'python.exe'),
        os.path.join(home, 'anaconda3', 'envs', 'zuna-gpu', 'python.exe'),
        os.path.join(home, 'anaconda3', 'envs', 'zuna-eeg', 'python.exe'),
        '/mnt/research-data/venvs/zuna-eeg/bin/python',
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _env_int(name, default):
    value = os.environ.get(name, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise RuntimeError('{} must be an integer, got {!r}'.format(
            name, value))


def default_diffusion_steps():
    return _env_int('ZUNA_DIFFUSION_STEPS', '50')


def default_tokens_per_batch():
    value = os.environ.get('ZUNA_TOKENS_PER_BATCH')
    if value:
        return _env_int('ZUNA_TOKENS_PER_BATCH', value)
    # GUI-triggered full ZUNA should prefer safe defaults over throughput.
    return 128


def default_gpu_device():
    return os.environ.get('ZUNA_GPU_DEVICE', '0')


class ZunaFullRunner(object):
    """Sequential full-ZUNA bridge runner used by the GUI worker thread."""

    def __init__(self, edf_path, zuna_python=None, diffusion_steps=None,
                 tokens_per_batch=None, gpu_device=None):
        self.edf_path = os.path.abspath(edf_path)
        self.paths = zuna_session_paths(self.edf_path)
        self.zuna_python = zuna_python or find_zuna_python()
        self.diffusion_steps = (
            default_diffusion_steps()
            if diffusion_steps is None else int(diffusion_steps))
        self.tokens_per_batch = (
            default_tokens_per_batch()
            if tokens_per_batch is None else tokens_per_batch)
        self.gpu_device = default_gpu_device() if gpu_device is None else gpu_device
        self._proc = None
        self._cancelled = False

    def cancel(self):
        self._cancelled = True
        if self._proc is not None and self._proc.poll() is None:
            try:
                if os.name == 'nt':
                    subprocess.call([
                        'taskkill', '/F', '/T', '/PID',
                        str(self._proc.pid)])
                else:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            except Exception:
                pass

    def _bridge_cmd(self, *args):
        script = os.path.join(REPO, 'utils', 'zuna_bridge.py')
        return [self.zuna_python, script] + list(args)

    def _run_cmd(self, label, cmd, log, status_cb=None):
        if self._cancelled:
            raise RuntimeError('ZUNA run cancelled.')
        if status_cb:
            status_cb(label)
        log.write('\n[{}] {}\n'.format(
            time.strftime('%Y-%m-%d %H:%M:%S'), label))
        log.write('command: {}\n'.format(' '.join(cmd)))
        log.flush()
        popen_kwargs = {}
        if os.name != 'nt':
            popen_kwargs['preexec_fn'] = os.setsid
        self._proc = subprocess.Popen(
            cmd,
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            **popen_kwargs
        )
        for line in self._proc.stdout:
            log.write(line)
            log.flush()
        ret = self._proc.wait()
        self._proc = None
        if self._cancelled:
            raise RuntimeError('ZUNA run cancelled.')
        if ret:
            raise RuntimeError('{} failed with exit code {}. See {}'.format(
                label, ret, self.paths['log']))

    def run(self, status_cb=None):
        if not self.zuna_python:
            raise RuntimeError(
                'Could not find a ZUNA Python interpreter. Set ZUNA_PYTHON to '
                'the python.exe inside the zuna-gpu or zuna-eeg conda env.')
        if not os.path.exists(self.zuna_python):
            raise RuntimeError('ZUNA Python does not exist: {}'.format(
                self.zuna_python))

        for key in ('root', 'fif_dir'):
            if not os.path.exists(self.paths[key]):
                os.makedirs(self.paths[key])
        with open(self.paths['log'], 'a') as log:
            log.write('\n=== Full ZUNA GUI run for {} ===\n'.format(
                self.edf_path))
            log.write('python: {}\n'.format(self.zuna_python))
            log.write('diffusion_steps: {}\n'.format(self.diffusion_steps))
            log.write('tokens_per_batch: {}\n'.format(self.tokens_per_batch))
            log.write('gpu_device: {}\n'.format(self.gpu_device))

            if not os.path.exists(self.paths['fif']):
                self._run_cmd(
                    'Preparing EDF for ZUNA',
                    self._bridge_cmd(
                        'prepare',
                        '--edf', self.edf_path,
                        '--out', self.paths['fif'],
                        '--overwrite',
                    ),
                    log,
                    status_cb=status_cb,
                )

            if not os.path.exists(self.paths['out_fif']):
                run_cmd = self._bridge_cmd(
                    'run',
                    '--fif-dir', self.paths['fif_dir'],
                    '--work-dir', self.paths['work_dir'],
                    '--gpu-device', str(self.gpu_device),
                    '--diffusion-steps', str(self.diffusion_steps),
                )
                if self.tokens_per_batch is not None:
                    run_cmd.extend([
                        '--tokens-per-batch', str(self.tokens_per_batch)])
                self._run_cmd(
                    'Running full ZUNA reconstruction',
                    run_cmd,
                    log,
                    status_cb=status_cb,
                )

            if not os.path.exists(self.paths['npz']):
                self._run_cmd(
                    'Exporting ZUNA output to repo NPZ',
                    self._bridge_cmd(
                        'export-npz',
                        '--fif', self.paths['out_fif'],
                        '--out', self.paths['npz'],
                        '--overwrite',
                    ),
                    log,
                    status_cb=status_cb,
                )

        if not os.path.exists(self.paths['npz']):
            raise RuntimeError('ZUNA NPZ was not created: {}'.format(
                self.paths['npz']))
        return self.paths
