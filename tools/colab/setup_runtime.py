"""Pin the saved parallel campaign's Torch/Transformers environment."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

packages = ['torch==2.14.0', 'torchvision==0.29.0', 'transformers==5.17.0']
subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                *packages], check=True)
versions = {name: importlib.metadata.version(name)
            for name in ('torch', 'torchvision', 'transformers', 'numpy', 'h5py', 'Pillow')}
Path('/content/acma/setup_versions.json').write_text(json.dumps(versions, indent=2))
print(json.dumps(versions, indent=2))
print('ACMA_SETUP_OK', flush=True)
