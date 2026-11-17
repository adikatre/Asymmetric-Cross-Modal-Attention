"""Verify local Pillow preprocessing against the original torchvision path."""
import json
from pathlib import Path
import subprocess
import sys

result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
                         'tools/colab', '-p', 'test_prepare_vqa.py', '-v'],
                        cwd='/content/acma/source', capture_output=True, text=True)
report = {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
Path('/content/acma/preprocessing_verification.json').write_text(json.dumps(report, indent=2))
print(result.stdout)
print(result.stderr)
result.check_returncode()
print('ACMA_PREPROCESSING_PARITY_VERIFIED')
