"""Verify the staged source bundle, run transport/data checks, start pipeline."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

root = Path('/content/acma')
source = root / 'source'
source.mkdir(exist_ok=True)
with tarfile.open(root / 'source_bundle.tar.gz') as archive:
    archive.extractall(source, filter='data')
manifest = json.loads((source / 'source_manifest.json').read_text())
for filename, expected in manifest.items():
    actual = hashlib.sha256((source / filename).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f'Staged source checksum mismatch: {filename}')
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tools/colab',
                '-p', 'test_*.py', '-v'], cwd=source, check=True)
print('ACMA_REMOTE_SOURCE_AND_DATA_TESTS_OK', flush=True)
if os.environ.get('ACMA_MODE') == 'l4-preflight':
    subprocess.run([sys.executable, str(source / 'tools/colab/l4_preflight.py'),
                    '--notebook', str(source / 'notebooks/train_evaluate_visualize_colab_frozen_5_gcp.ipynb'),
                    '--controls', str(source / 'tools/colab/reviewer_controls.py'),
                    '--output-dir', str(root / 'l4_preflight')], cwd=source, check=True)
else:
    subprocess.run([sys.executable, str(source / 'tools/colab/pipeline.py')],
                   cwd=source, check=True)
