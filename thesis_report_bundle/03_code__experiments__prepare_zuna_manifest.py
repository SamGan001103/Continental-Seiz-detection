"""Prepare ZUNA FIF inputs from a manifest CSV.

The manifest is expected to contain at least:
    stem,edf

Outputs are written as ``<stem>.fif`` in the requested output directory and a
JSONL status log is written for audit/restart.
"""
from __future__ import print_function

import argparse
import csv
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from utils.zuna_bridge import prepare_fif_from_edf  # noqa: E402


def build_arg_parser():
    ap = argparse.ArgumentParser(description='Prepare ZUNA FIF files.')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--log', required=True)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--skip-existing', action='store_true')
    return ap


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if not os.path.exists(args.manifest):
        raise SystemExit('manifest not found: {}'.format(args.manifest))
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    log_parent = os.path.dirname(os.path.abspath(args.log))
    if log_parent and not os.path.exists(log_parent):
        os.makedirs(log_parent)

    with open(args.manifest, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit('manifest is empty: {}'.format(args.manifest))

    ok = 0
    failed = []
    skipped = 0
    with open(args.log, 'w') as log:
        for row in rows:
            stem = row.get('stem')
            edf = row.get('edf')
            if not stem or not edf:
                raise SystemExit('manifest row must contain stem and edf')
            output_fif = os.path.join(args.out_dir, stem + '.fif')
            t0 = time.time()
            item = {
                'stem': stem,
                'edf': edf,
                'fif': output_fif,
            }
            try:
                if args.skip_existing and os.path.exists(output_fif):
                    item.update({
                        'status': 'skipped',
                        'seconds': round(time.time() - t0, 3),
                    })
                    skipped += 1
                else:
                    prepare_fif_from_edf(
                        edf, output_fif, overwrite=args.overwrite)
                    item.update({
                        'status': 'ok',
                        'seconds': round(time.time() - t0, 3),
                    })
                    ok += 1
                    print('OK {} -> {}'.format(stem, output_fif))
            except Exception as ex:
                item.update({
                    'status': 'failed',
                    'error': repr(ex),
                    'seconds': round(time.time() - t0, 3),
                })
                failed.append(item)
                print('FAILED {}: {}'.format(stem, ex))
            log.write(json.dumps(item, sort_keys=True) + '\n')
            log.flush()

    print('prepared={} skipped={} failed={}'.format(ok, skipped, len(failed)))
    if failed:
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
