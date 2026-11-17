"""Copy the completed frozen cache outside Colab without blocking training.

No runtime allocation, kernel execution, restart, or stop is performed here.
Only a matching full-file hash promotes the streamed partial to a usable cache.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def write_state(path, **fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'updated_at': time.time(), **fields}, indent=2))
    temporary.replace(path)


def verify(path, size, digest):
    if path.stat().st_size != size:
        raise ValueError('Frozen-cache transfer size mismatch')
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != digest:
        raise ValueError('Frozen-cache transfer SHA-256 mismatch')


def profiling_finished(status):
    # Preflight invokes train_one_epoch too. A real epoch and run identity are
    # published only when the fixed-horizon comparison starts after profiling.
    return bool(status.get('run_name') and int(status.get('epoch') or 0) >= 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    parser.add_argument('--local-data-dir', required=True, type=Path)
    parser.add_argument('--state-dir', required=True, type=Path)
    parser.add_argument('--colab', default='colab')
    parser.add_argument('--timeout-hours', type=float, default=12)
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    args.local_data_dir.mkdir(parents=True, exist_ok=True)
    state = args.state_dir / 'status.json'
    provenance = args.state_dir / 'cache_provenance.json'
    deadline = time.monotonic() + args.timeout_hours * 3600
    try:
        while time.monotonic() < deadline:
            write_state(state, status='waiting_for_completed_cache', session=args.session)
            try:
                result = subprocess.run([args.colab, 'download', '-s', args.session,
                                         '/content/acma/output/cache_provenance.json', str(provenance)],
                                        stdin=subprocess.DEVNULL, capture_output=True, timeout=120)
            except (OSError, subprocess.TimeoutExpired) as error:
                write_state(state, status='waiting_after_network_error', error=repr(error))
                time.sleep(60)
                continue
            if result.returncode == 0 and provenance.exists():
                spec = json.loads(provenance.read_text())
                break
            time.sleep(60)
        else:
            raise TimeoutError('No completed frozen-cache provenance before the mirror deadline')
        while time.monotonic() < deadline:
            write_state(state, status='waiting_for_training_after_profiling', session=args.session)
            training_path = args.state_dir / 'training_status.json'
            try:
                result = subprocess.run([args.colab, 'download', '-s', args.session,
                                         '/content/acma/output/status.json', str(training_path)],
                                        stdin=subprocess.DEVNULL, capture_output=True, timeout=120)
                if result.returncode == 0 and training_path.exists():
                    training = json.loads(training_path.read_text())
                    if training.get('status') == 'failed':
                        raise RuntimeError('Training failed before cache backup could start')
                    if profiling_finished(training):
                        break
            except (OSError, subprocess.TimeoutExpired):
                pass
            time.sleep(60)
        else:
            raise TimeoutError('Profiling did not finish before the mirror deadline')
        remote = Path(spec['path'])
        if remote.parent != Path('/content/acma/data') or remote.suffix != '.h5':
            raise ValueError('Cache provenance points outside the expected data directory')
        destination = args.local_data_dir / remote.name
        if destination.exists():
            verify(destination, spec['size_bytes'], spec['sha256'])
        else:
            partial = destination.with_suffix('.h5.mirror-part')
            for attempt in (1, 2):
                write_state(state, status='streaming', session=args.session, attempt=attempt,
                            partial_path=str(partial), expected_bytes=spec['size_bytes'])
                try:
                    with partial.open('wb') as output, (args.state_dir / 'transfer.log').open('ab') as errors:
                        subprocess.run([args.colab, 'ssh', '-s', args.session, '--', 'cat', str(remote)],
                                       stdin=subprocess.DEVNULL, stdout=output, stderr=errors,
                                       timeout=max(1, deadline-time.monotonic()), check=True)
                        output.flush()
                        os.fsync(output.fileno())
                    write_state(state, status='verifying_full_hash', path=str(partial))
                    verify(partial, spec['size_bytes'], spec['sha256'])
                    partial.replace(destination)
                    break
                except Exception:
                    if attempt == 2 or time.monotonic() >= deadline:
                        raise
                    time.sleep(30)
        write_state(state, status='complete', session=args.session, path=str(destination),
                    size_bytes=spec['size_bytes'], sha256=spec['sha256'], verified_at=time.time())
    except BaseException as error:
        write_state(state, status='failed', session=args.session, error=repr(error))
        raise


if __name__ == '__main__':
    main()
