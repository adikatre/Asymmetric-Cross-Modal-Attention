"""Compare value-identical HDF5 batch reading strategies on the real cache."""
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np

path = next(Path('/content/acma/data').glob('vqa_frozen_features_*.h5'))
rows = []
with h5py.File(path, 'r') as cache:
    group = cache['train']
    indices = np.random.default_rng(7).choice(len(group['image_features']), 2048, replace=False)
    order = np.argsort(indices)
    inverse = np.argsort(order)
    names = ('image_features', 'text_features', 'attention_mask', 'answer_target')
    for strategy in ('fancy', 'rows', 'rows', 'fancy'):
        start = time.monotonic()
        digest = hashlib.sha256()
        for name in names:
            dataset = group[name]
            if strategy == 'fancy':
                values = dataset[indices[order]][inverse]
            else:
                values = np.stack([dataset[int(index)] for index in indices])
            digest.update(memoryview(values).cast('B'))
            del values
        rows.append({'strategy': strategy, 'seconds': time.monotonic()-start,
                     'batch_sha256': digest.hexdigest()})
        print(json.dumps(rows[-1]), flush=True)
assert len({row['batch_sha256'] for row in rows}) == 1
Path('/content/acma/cache_read_benchmark.json').write_text(json.dumps(rows, indent=2))
print('ACMA_CACHE_READ_BENCHMARK_OK', flush=True)
