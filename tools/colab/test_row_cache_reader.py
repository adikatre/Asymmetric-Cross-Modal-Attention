"""Compare the repaired reader against the canonical reader's actual values."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

try:
    import h5py
    import numpy as np
    import torch
except ImportError:
    h5py = np = torch = None

HERE = Path(__file__).resolve().parent


@unittest.skipIf(torch is None, 'Requires the training environment')
class RowReaderTests(unittest.TestCase):
    def test_canonical_and_row_reader_batches_are_exact(self):
        spec = importlib.util.spec_from_file_location('controls', HERE / 'reviewer_controls.py')
        controls = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(controls)
        notebook = json.loads((HERE.parents[1] / controls.NOTEBOOK).read_text())
        classes = []
        for source in (controls.cell_source(notebook, 23), controls.adapted_source(notebook, 23)):
            cls = next(node for node in ast.walk(ast.parse(source))
                       if isinstance(node, ast.ClassDef) and node.name == 'PrecomputedVQADataset')
            namespace = {'torch': torch, 'h5py': h5py, 'np': np, 'Dataset': torch.utils.data.Dataset}
            exec(compile(ast.Module(body=[cls], type_ignores=[]), 'reader', 'exec'), namespace)
            classes.append(namespace['PrecomputedVQADataset'])
        self.assertFalse(hasattr(classes[1], '__getitems__'))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'features.h5'
            rng = np.random.default_rng(7)
            with h5py.File(path, 'w') as cache:
                group = cache.create_group('train')
                for name, shape, dtype in [('image_features', (7, 3, 4), 'float16'),
                                            ('text_features', (7, 2, 4), 'float16'),
                                            ('attention_mask', (7, 2), 'int8'),
                                            ('answer_target', (7, 5), 'float32')]:
                    group[name] = rng.integers(0, 2, size=shape).astype(dtype)
            datasets = [cls(path, 'train') for cls in classes]
            loaders = [torch.utils.data.DataLoader(ds, batch_size=3, shuffle=True,
                        generator=torch.Generator().manual_seed(42)) for ds in datasets]
            for old_batch, new_batch in zip(*loaders):
                for original, repaired in zip(old_batch, new_batch):
                    self.assertEqual(original.dtype, repaired.dtype)
                    self.assertTrue(torch.equal(original, repaired))
            self.assertEqual(next(iter(loaders[1]))[2].dtype, torch.int64)
            for dataset in datasets:
                dataset._h5f.close()


if __name__ == '__main__':
    unittest.main()
