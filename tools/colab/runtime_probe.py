"""Read-only remote hardware/environment preflight for the reviewer campaign."""
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess

root = Path('/content/acma')
root.mkdir(exist_ok=True)
packages = {}
for name in ('torch', 'torchvision', 'transformers', 'h5py', 'numpy', 'Pillow', 'requests'):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
disk = shutil.disk_usage('/content')
report = {
    'packages': packages,
    'gpu': subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version',
                           '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout.strip(),
    'disk': dict(zip(('total', 'used', 'free'), disk)),
    'meminfo': {line.split(':')[0]: line.split(':')[1].strip()
                for line in Path('/proc/meminfo').read_text().splitlines()
                if line.split(':')[0] in ('MemTotal', 'MemAvailable')},
}
(root / 'runtime_probe.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
print('ACMA_RUNTIME_PROBE_OK')
