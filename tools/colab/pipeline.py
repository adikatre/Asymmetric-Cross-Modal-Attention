"""Fail-fast remote launcher for the single frozen reviewer-control experiment.

All phases write status independently of the occupied notebook kernel. The
local file-transport monitor mirrors immutable checkpoints outside this VM.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path('/content/acma')
SOURCE = ROOT / 'source'
STATUS = ROOT / 'pipeline_status.json'
OUTPUT = ROOT / 'output'
SESSION = os.environ.get('ACMA_SESSION', 'acma-r1-shared-frozen-l4-20261117')


def write_status(**fields):
    payload = {'session': SESSION, 'heartbeat_at': time.time(),
               'started_at': started, **fields}
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(STATUS)


def run_stage(name, argv, expected_status):
    log_path = ROOT / f'{name}.log'
    print(f'ACMA_STAGE_START {name}', flush=True)
    with log_path.open('a') as log:
        process = subprocess.Popen(argv, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        while process.poll() is None:
            write_status(status='running', phase=name, pid=process.pid,
                         log=str(log_path), stage_started_at=started)
            time.sleep(10)
    if process.returncode != 0:
        raise RuntimeError(f'{name} exited {process.returncode}; see {log_path}')
    path, status_key, required = expected_status
    payload = json.loads(path.read_text())
    if payload.get(status_key) != required:
        raise RuntimeError(f'{name} lacks expected {required} marker: {payload}')
    print(f'ACMA_STAGE_OK {name}', flush=True)


def main():
    ROOT.mkdir(exist_ok=True)
    OUTPUT.mkdir(exist_ok=True)
    lock = ROOT / 'pipeline.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as stream:
        stream.write(str(os.getpid()))
    try:
        wrapper = str(SOURCE / 'tools/colab/reviewer_controls.py')
        common = [sys.executable, wrapper, '--notebook',
                  str(SOURCE / 'notebooks/train_evaluate_visualize_colab_frozen_5_gcp.ipynb'),
                  '--data-dir', str(ROOT / 'data')]
        run_stage('self_test', [*common, '--output-dir', str(ROOT / 'smoke'), '--self-test'],
                  (ROOT / 'smoke/status.json', 'status', 'self_test_complete'))
        run_stage('prepare_data', [sys.executable, str(SOURCE / 'tools/colab/prepare_vqa.py'),
                                  '--data-dir', str(ROOT / 'data'),
                                  '--status', str(ROOT / 'data_status.json')],
                  (ROOT / 'data_status.json', 'phase', 'complete'))
        run_stage('frozen_controls', [*common, '--output-dir', str(OUTPUT), '--variants',
                                     'symmetric_single', 'symmetric_sequential',
                                     'asymmetric', 'symmetric_matched'],
                  (OUTPUT / 'status.json', 'status', 'complete'))
        write_status(status='complete', phase='complete', started_at=started,
                     completion_marker='ACMA_REVIEWER_FROZEN_CONTROLS_COMPLETE')
        print('ACMA_REVIEWER_FROZEN_CONTROLS_COMPLETE', flush=True)
    except BaseException as error:
        write_status(status='failed', phase='failed', error=repr(error),
                     traceback=traceback.format_exc(), started_at=started)
        raise
    finally:
        lock.unlink(missing_ok=True)


started = time.time()
if __name__ == '__main__':
    main()
