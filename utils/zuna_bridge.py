"""Offline bridge between EDF files and the ZUNA EEG reconstruction package.

This module is intentionally not imported by the GUI or ConvLSTM inference
path. The current detector runs in the legacy ``seiz36`` environment
(Python 3.6 / TensorFlow 1.x), while ZUNA requires a newer Python stack.

Typical use from a modern Python environment:

    python utils/zuna_bridge.py prepare --edf input.edf --out work/fif/input.fif
    python utils/zuna_bridge.py run --fif-dir work/fif --work-dir work/zuna
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_CHANNELS_19 = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2',
]

# MNE's standard montages use the modern 10-20 temporal labels.
REPO_TO_MNE_1005 = {
    'Fp1': 'Fp1', 'Fp2': 'Fp2',
    'F7': 'F7', 'F3': 'F3', 'Fz': 'Fz', 'F4': 'F4', 'F8': 'F8',
    'T3': 'T7', 'C3': 'C3', 'Cz': 'Cz', 'C4': 'C4', 'T4': 'T8',
    'T5': 'P7', 'P3': 'P3', 'Pz': 'Pz', 'P4': 'P4', 'T6': 'P8',
    'O1': 'O1', 'O2': 'O2',
}

MNE_TO_REPO_1005 = {v: k for k, v in REPO_TO_MNE_1005.items()}
DEFAULT_TARGET_CHANNELS = [REPO_TO_MNE_1005[ch] for ch in REPO_CHANNELS_19]


def _require_mne():
    try:
        import mne
    except ImportError as ex:
        raise RuntimeError(
            'ZUNA bridge requires mne in a Python >=3.8 environment. '
            'Install with: pip install zuna mne') from ex
    return mne


def _canonical_key(label):
    text = str(label).strip()
    text = re.sub(r'^EEG\s*', '', text, flags=re.IGNORECASE)
    text = text.replace(' ', '')
    text = text.split('-')[0]
    return text.upper()


def _raw_to_mne_name(label):
    key = _canonical_key(label)
    aliases = {
        'FP1': 'Fp1', 'FP2': 'Fp2',
        'F7': 'F7', 'F3': 'F3', 'FZ': 'Fz', 'F4': 'F4', 'F8': 'F8',
        'T3': 'T7', 'T7': 'T7',
        'C3': 'C3', 'CZ': 'Cz', 'C4': 'C4',
        'T4': 'T8', 'T8': 'T8',
        'T5': 'P7', 'P7': 'P7',
        'P3': 'P3', 'PZ': 'Pz', 'P4': 'P4',
        'T6': 'P8', 'P8': 'P8',
        'O1': 'O1', 'O2': 'O2',
    }
    return aliases.get(key)


def prepare_fif_from_edf(edf_path, output_fif, target_channels=None,
                         montage_name='standard_1005', resample_hz=256,
                         overwrite=False):
    """Convert an EDF to an MNE FIF with valid 3D channel positions.

    The output is suitable as input to ZUNA's ``preprocessing`` function.
    Channels are renamed from TUH/legacy labels (for example T3/T4) to the
    modern names used by MNE standard montages (T7/T8).
    """
    mne = _require_mne()
    edf_path = os.path.abspath(edf_path)
    output_fif = os.path.abspath(output_fif)
    os.makedirs(os.path.dirname(output_fif), exist_ok=True)

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='ERROR')
    rename = {}
    for name in raw.ch_names:
        mne_name = _raw_to_mne_name(name)
        if mne_name and mne_name not in rename.values():
            rename[name] = mne_name
    if not rename:
        raise RuntimeError('No recognizable EEG channels found in {}'.format(edf_path))
    raw.rename_channels(rename)

    target_channels = list(target_channels or DEFAULT_TARGET_CHANNELS)
    available = [ch for ch in target_channels if ch in raw.ch_names]
    if len(available) < 2:
        raise RuntimeError(
            'ZUNA needs at least two recognizable EEG channels; found {}'.format(
                available))

    raw.pick_channels(available, ordered=True)
    raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})
    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(montage, on_missing='raise')
    if resample_hz:
        raw.resample(float(resample_hz))
    raw.save(output_fif, overwrite=overwrite)
    return output_fif


def run_zuna_pipeline(fif_dir, work_dir, target_channels=None,
                      bad_channels=None, gpu_device=0,
                      diffusion_sample_steps=50,
                      tokens_per_batch=None,
                      allow_high_memory=False):
    """Run the public ZUNA pipeline over prepared FIF files."""
    try:
        import zuna
        import zuna.pipeline as zuna_pipeline
    except ImportError as ex:
        raise RuntimeError(
            'ZUNA package is not installed. Use a Python >=3.8 environment '
            'and install with: pip install zuna') from ex

    fif_dir = os.path.abspath(fif_dir)
    work_dir = os.path.abspath(work_dir)
    target_channels = list(target_channels or DEFAULT_TARGET_CHANNELS)
    bad_channels = list(bad_channels or [])
    cpu_mode = _zuna_cpu_mode(gpu_device)
    if cpu_mode:
        if tokens_per_batch is None:
            tokens_per_batch = 512
        if int(tokens_per_batch) > 512 and not allow_high_memory:
            raise RuntimeError(
                'Refusing CPU ZUNA run with tokens_per_batch={}. '
                'CPU mode is capped at 512 to avoid OOM. Use a CUDA PyTorch '
                'environment, or pass --allow-high-memory only after checking '
                'available RAM.'.format(tokens_per_batch))
    elif tokens_per_batch is None:
        tokens_per_batch = 100000

    preprocessed_fif_dir = os.path.join(work_dir, '1_fif_filter')
    pt_input_dir = os.path.join(work_dir, '2_pt_input')
    pt_output_dir = os.path.join(work_dir, '3_pt_output')
    fif_output_dir = os.path.join(work_dir, '4_fif_output')
    figures_dir = os.path.join(work_dir, 'FIGURES')
    for path in (preprocessed_fif_dir, pt_input_dir, pt_output_dir,
                 fif_output_dir, figures_dir):
        os.makedirs(path, exist_ok=True)

    if os.name == 'nt':
        shim_dir = os.path.join(work_dir, '_bin')
        os.makedirs(shim_dir, exist_ok=True)
        shim_path = os.path.join(shim_dir, 'python3.cmd')
        with open(shim_path, 'w') as f:
            f.write('@echo off\r\n"{}" %*\r\n'.format(sys.executable))
        os.environ['PATH'] = shim_dir + os.pathsep + os.environ.get('PATH', '')

    zuna.preprocessing(
        input_dir=fif_dir,
        output_dir=pt_input_dir,
        apply_notch_filter=False,
        apply_highpass_filter=True,
        apply_average_reference=True,
        target_channel_count=target_channels,
        bad_channels=bad_channels,
        preprocessed_fif_dir=preprocessed_fif_dir,
    )
    _run_zuna_inference_with_current_python(
        zuna_pipeline=zuna_pipeline,
        input_dir=pt_input_dir,
        output_dir=pt_output_dir,
        gpu_device=gpu_device,
        tokens_per_batch=tokens_per_batch,
        data_norm=10.0,
        diffusion_cfg=1.0,
        diffusion_sample_steps=diffusion_sample_steps,
        plot_eeg_signal_samples=False,
        inference_figures_dir=figures_dir,
    )
    zuna.pt_to_fif(
        input_dir=pt_output_dir,
        output_dir=fif_output_dir,
    )
    return {
        'preprocessed_fif_dir': preprocessed_fif_dir,
        'pt_input_dir': pt_input_dir,
        'pt_output_dir': pt_output_dir,
        'fif_output_dir': fif_output_dir,
        'figures_dir': figures_dir,
    }


def _zuna_cpu_mode(gpu_device):
    if str(gpu_device).strip() == '':
        return True
    try:
        import torch
        return not torch.cuda.is_available()
    except Exception:
        return True


def _run_zuna_inference_with_current_python(zuna_pipeline, input_dir,
                                            output_dir, gpu_device=0,
                                            tokens_per_batch=None,
                                            data_norm=None,
                                            diffusion_cfg=1.0,
                                            diffusion_sample_steps=50,
                                            plot_eeg_signal_samples=False,
                                            inference_figures_dir=None):
    """Run ZUNA inference with this interpreter instead of Windows python3."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    package_dir = Path(zuna_pipeline.__file__).parent
    config_path = (
        package_dir /
        'inference/AY2l/lingua/apps/AY2latent_bci/configs/config_infer.yaml')
    eeg_eval_script = (
        package_dir /
        'inference/AY2l/lingua/apps/AY2latent_bci/eeg_eval.py')

    command_script = eeg_eval_script
    prefix_args = []
    if os.name == 'nt':
        wrapper = output_path.parent / '_bin' / 'zuna_eval_wrapper.py'
        wrapper.parent.mkdir(parents=True, exist_ok=True)
        wrapper.write_text(
            "import os, runpy, signal, sys\n"
            "if not hasattr(signal, 'SIGUSR2'):\n"
            "    signal.SIGUSR2 = getattr(signal, 'SIGBREAK', signal.SIGTERM)\n"
            "for _key in ('LOCAL_RANK', 'RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):\n"
            "    os.environ.pop(_key, None)\n"
            "os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')\n"
            "os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')\n"
            "try:\n"
            "    import torch\n"
            "    def _no_compile(fn=None, *args, **kwargs):\n"
            "        if fn is None:\n"
            "            return lambda real_fn: real_fn\n"
            "        return fn\n"
            "    torch.compile = _no_compile\n"
            "    _init_process_group = torch.distributed.init_process_group\n"
            "    def _windows_init_process_group(*args, **kwargs):\n"
            "        if kwargs.get('backend') == 'nccl':\n"
            "            kwargs['backend'] = 'gloo'\n"
            "        return _init_process_group(*args, **kwargs)\n"
            "    torch.distributed.init_process_group = _windows_init_process_group\n"
            "except Exception:\n"
            "    pass\n"
            "script = sys.argv[1]\n"
            "sys.path.insert(0, os.path.dirname(script))\n"
            "sys.argv = [script] + sys.argv[2:]\n"
            "runpy.run_path(script, run_name='__main__')\n")
        command_script = wrapper
        prefix_args = [str(eeg_eval_script)]

    cmd = [
        sys.executable,
        str(command_script),
    ] + prefix_args + [
        'config={}'.format(config_path),
        'data.data_dir={}'.format(str(Path(input_dir).absolute())),
        'data.export_dir={}'.format(str(output_path.absolute())),
        'diffusion_cfg={}'.format(diffusion_cfg),
        'diffusion_sample_steps={}'.format(diffusion_sample_steps),
        'plot_eeg_signal_samples={}'.format(plot_eeg_signal_samples),
        'inference_figures_dir={}'.format(inference_figures_dir),
    ]
    if tokens_per_batch is not None:
        cmd.append('data.target_packed_seqlen={}'.format(tokens_per_batch))
    if data_norm is not None:
        cmd.append('data.data_norm={}'.format(data_norm))

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_device)
    env.setdefault('TORCH_COMPILE_DISABLE', '1')
    env.setdefault('TORCHDYNAMO_DISABLE', '1')
    subprocess.run(cmd, env=env, check=True)


def export_repo_npz_from_fif(fif_path, output_npz, target_fs=250,
                             overwrite=False):
    """Convert a reconstructed ZUNA FIF back to repo canonical 19-channel NPZ."""
    mne = _require_mne()
    import numpy as np

    fif_path = os.path.abspath(fif_path)
    output_npz = os.path.abspath(output_npz)
    if os.path.exists(output_npz) and not overwrite:
        raise RuntimeError('Output already exists: {}'.format(output_npz))
    os.makedirs(os.path.dirname(output_npz), exist_ok=True)

    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose='ERROR')
    rename = {}
    for name in raw.ch_names:
        if name in MNE_TO_REPO_1005:
            rename[name] = MNE_TO_REPO_1005[name]
    raw.rename_channels(rename)
    missing = [ch for ch in REPO_CHANNELS_19 if ch not in raw.ch_names]
    if missing:
        raise RuntimeError(
            'Reconstructed FIF is missing required channels: {}'.format(
                ', '.join(missing)))
    raw.pick_channels(REPO_CHANNELS_19, ordered=True)
    if target_fs:
        raw.resample(float(target_fs))
    # MNE returns EEG data in volts, while this repo's pyedflib path feeds
    # physical EDF values in microvolts into the legacy detector.
    data = (raw.get_data() * 1e6).astype('float32')
    fs = int(round(float(raw.info['sfreq'])))
    np.savez_compressed(
        output_npz,
        data=data,
        fs=np.array(fs, dtype=np.int32),
        channels=np.array(REPO_CHANNELS_19),
        units=np.array('microvolts'),
        source_fif=np.array(fif_path),
    )
    return output_npz


def _split_csv(value):
    if not value:
        return []
    return [part.strip() for part in value.split(',') if part.strip()]


def main():
    ap = argparse.ArgumentParser(description='Prepare and run ZUNA EEG bridge')
    sub = ap.add_subparsers(dest='cmd')

    p_prepare = sub.add_parser('prepare', help='convert one EDF to ZUNA-ready FIF')
    p_prepare.add_argument('--edf', required=True)
    p_prepare.add_argument('--out', required=True)
    p_prepare.add_argument('--targets', default='',
                           help='comma-separated MNE target channel names')
    p_prepare.add_argument('--overwrite', action='store_true')

    p_run = sub.add_parser('run', help='run ZUNA over a directory of FIF files')
    p_run.add_argument('--fif-dir', required=True)
    p_run.add_argument('--work-dir', required=True)
    p_run.add_argument('--targets', default='',
                       help='comma-separated MNE target channel names')
    p_run.add_argument('--bad-channels', default='',
                       help='comma-separated channel names to zero before ZUNA')
    p_run.add_argument('--gpu-device', default=0)
    p_run.add_argument('--diffusion-steps', type=int, default=50)
    p_run.add_argument('--tokens-per-batch', type=int, default=None)
    p_run.add_argument('--allow-high-memory', action='store_true',
                       help='permit CPU token batches above the safe cap')

    p_export = sub.add_parser(
        'export-npz', help='convert reconstructed FIF back to 19-channel NPZ')
    p_export.add_argument('--fif', required=True)
    p_export.add_argument('--out', required=True)
    p_export.add_argument('--target-fs', type=int, default=250)
    p_export.add_argument('--overwrite', action='store_true')

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help()
        return 2
    if args.cmd == 'prepare':
        out = prepare_fif_from_edf(
            args.edf, args.out,
            target_channels=_split_csv(args.targets) or None,
            overwrite=args.overwrite)
        print(out)
    elif args.cmd == 'run':
        gpu_device = args.gpu_device
        if str(gpu_device).strip() == '':
            gpu_device = ''
        else:
            gpu_device = int(gpu_device)
        outputs = run_zuna_pipeline(
            args.fif_dir, args.work_dir,
            target_channels=_split_csv(args.targets) or None,
            bad_channels=_split_csv(args.bad_channels),
            gpu_device=gpu_device,
            diffusion_sample_steps=args.diffusion_steps,
            tokens_per_batch=args.tokens_per_batch,
            allow_high_memory=args.allow_high_memory)
        for key in sorted(outputs):
            print('{}={}'.format(key, outputs[key]))
    elif args.cmd == 'export-npz':
        out = export_repo_npz_from_fif(
            args.fif, args.out,
            target_fs=args.target_fs,
            overwrite=args.overwrite)
        print(out)


if __name__ == '__main__':
    main()
