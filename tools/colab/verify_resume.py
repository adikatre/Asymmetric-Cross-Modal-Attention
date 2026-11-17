"""Run the real frozen-head checkpoint/resume check without dataset access."""
from pathlib import Path
import subprocess
import sys

root = Path('/content/acma')
source = root / 'source'
with (root / 'resume_verification.log').open('w') as log:
    subprocess.run([sys.executable, str(source / 'tools/colab/reviewer_controls.py'),
                    '--notebook', str(source / 'notebooks/train_evaluate_visualize_colab_frozen_5_gcp.ipynb'),
                    '--data-dir', str(root / 'data'), '--output-dir', str(root / 'resume_verification'),
                    '--self-test'], stdout=log, stderr=subprocess.STDOUT, check=True)
print('ACMA_RESUME_VERIFIED', flush=True)
