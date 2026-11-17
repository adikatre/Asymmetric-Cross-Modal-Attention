"""Small fixtures exercise recovery behavior without downloading the real dataset."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import prepare_vqa as prepare


def zip_bytes(name="fixture.json", content=b'{"ok":true}'):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(name, content)
    return stream.getvalue()


class Response(io.BytesIO):
    def __init__(self, payload, status=200, headers=None):
        super().__init__(payload)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(payload))}


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.events = []

    def tearDown(self):
        self.temporary.cleanup()

    def status(self, phase, **fields):
        self.events.append((phase, fields))

    def test_partial_transfer_uses_range_and_preserves_exact_bytes(self):
        payload = zip_bytes()
        destination = self.root / "fixture.zip"
        destination.with_suffix(".zip.part").write_bytes(payload[:17])
        requests = []

        def respond(request, timeout):
            requests.append(request)
            return Response(payload[17:], 206, {
                "Content-Range": f"bytes 17-{len(payload)-1}/{len(payload)}",
                "Content-Length": str(len(payload) - 17)})

        with patch.object(prepare, "urlopen", respond):
            receipt = prepare.download("https://example.invalid/fixture.zip", destination, self.status)
        self.assertEqual(requests[0].get_header("Range"), "bytes=17-")
        self.assertEqual(destination.read_bytes(), payload)
        self.assertEqual(receipt["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertFalse(destination.with_suffix(".zip.part").exists())

    def test_server_ignoring_range_restarts_instead_of_appending(self):
        payload = zip_bytes()
        destination = self.root / "fixture.zip"
        destination.with_suffix(".zip.part").write_bytes(payload[:17])
        with patch.object(prepare, "urlopen", return_value=Response(payload)):
            prepare.download("https://example.invalid/fixture.zip", destination, self.status)
        self.assertEqual(destination.read_bytes(), payload)

    def test_retry_mirror_and_bad_range_leave_partial_unpublished(self):
        payload = zip_bytes()
        destination = self.root / "fixture.zip"
        destination.with_suffix(".zip.part").write_bytes(payload[:17])
        responses = [OSError("source unavailable"), Response(payload[17:], 206, {
            "Content-Range": f"bytes 18-{len(payload)-1}/{len(payload)}",
            "Content-Length": str(len(payload)-17)})]
        with patch.object(prepare, "urlopen", side_effect=responses) as opening, \
             patch.object(prepare.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "Unexpected Content-Range"):
                prepare.download("https://original.invalid/fixture.zip", destination,
                                 self.status, retries=2, mirror="https://mirror.invalid/fixture.zip")
        self.assertEqual(opening.call_args_list[1].args[0].full_url,
                         "https://mirror.invalid/fixture.zip")
        self.assertFalse(destination.exists())
        self.assertEqual(destination.with_suffix(".zip.part").read_bytes(), payload[:17])

    def test_exact_metadata_hash_and_safe_member_selection(self):
        content = b'{"questions":[]}'
        source = self.root / "metadata.zip"
        source.write_bytes(zip_bytes("fixture.json", content))
        output = self.root / "questions" / "fixture.json"
        expected = hashlib.sha256(content).hexdigest()
        receipt = prepare.extract_json(source, "fixture.json", output, expected)
        self.assertEqual(output.read_bytes(), content)
        self.assertEqual(receipt["sha256"], expected)
        with self.assertRaisesRegex(ValueError, "plain JSON"):
            prepare.extract_json(source, "../fixture.json", output, expected)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            prepare.extract_json(source, "fixture.json", output, "0" * 64)
        self.assertEqual(output.read_bytes(), content)

    def test_images_sorted_and_duplicates_rejected(self):
        source = self.root / "images.zip"
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("train2014/COCO_train2014_000000000002.jpg", b"2")
            archive.writestr("../escape.jpg", b"ignore")
            archive.writestr("train2014/COCO_train2014_000000000001.jpg", b"1")
        with zipfile.ZipFile(source) as archive:
            self.assertEqual([row[1] for row in prepare.image_members(archive, "train2014", 2)], [1, 2])
            with self.assertRaisesRegex(ValueError, "Expected 3"):
                prepare.image_members(archive, "train2014", 3)

    def test_atomic_status_is_parseable_and_phase_counters_reset(self):
        status = prepare.Status(self.root / "status.json")
        status("downloading", bytes=123)
        status("building_image_cache", images_written=1000)
        actual = json.loads((self.root / "status.json").read_text())
        self.assertNotIn("bytes", actual)
        self.assertEqual(actual["images_written"], 1000)

    def test_pillow_geometry_preserves_torchvision_truncation_and_crop_rounding(self):
        # Geometry can be tested without installing any image or tensor package.
        class GeometryImage:
            def __init__(self, shape):
                self.size, self.resized_to, self.crop_box = shape, None, None

            def resize(self, shape, resample):
                self.size = self.resized_to = shape
                return self

            def crop(self, box):
                self.crop_box = box
                return self

        cases = [((224, 225), None, (0, 0, 224, 224)),
                 ((224, 227), None, (0, 2, 224, 226)),
                 ((225, 224), None, (0, 0, 224, 224)),
                 ((227, 224), None, (2, 0, 226, 224)),
                 ((640, 480), (298, 224), (37, 0, 261, 224)),
                 ((13, 101), (224, 1740), (0, 758, 224, 982))]
        fake_pil = SimpleNamespace(Image=SimpleNamespace(Resampling=SimpleNamespace(BICUBIC=3)))
        with patch.dict("sys.modules", {"PIL": fake_pil}):
            for shape, resized, crop_box in cases:
                with self.subTest(shape=shape):
                    image = prepare.pillow_preprocess(GeometryImage(shape))
                    self.assertEqual(image.resized_to, resized)
                    self.assertEqual(image.crop_box, crop_box)


@unittest.skipUnless(all(importlib.util.find_spec(k) is not None for k in
                         ["h5py", "numpy", "PIL"]),
                     "HDF5/image libraries unavailable on this host")
class ImageCacheTests(unittest.TestCase):
    def test_pixels_order_and_partial_resume_match_canonical_transform(self):
        import h5py
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            expected = []
            for split, image_id, shape in [("train2014", 1, (240, 350)), ("val2014", 2, (350, 240))]:
                pixels = np.arange(shape[0] * shape[1] * 3, dtype=np.uint8).reshape(*shape, 3)
                image = Image.fromarray(pixels)
                stream = io.BytesIO()
                image.save(stream, format="JPEG")
                encoded = stream.getvalue()
                with Image.open(io.BytesIO(encoded)) as decoded:
                    expected.append(np.asarray(prepare.pillow_preprocess(decoded.convert("RGB"))))
                path = root / (split + ".zip")
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr(f"{split}/COCO_{split}_{image_id:012d}.jpg", encoded)
                sources.append({"path": path, "split": split, "count": 1, "sha256": prepare.sha256(path)})
            target = root / "images.h5"
            prepare.build_image_cache(sources, target, lambda *args, **kwargs: None,
                                      expected_count=2, image_backend="pillow")
            with h5py.File(target, "a") as h5:
                self.assertTrue(np.array_equal(h5["images"][:], np.stack(expected)))
                self.assertEqual(h5["image_ids"][:].tolist(), [1, 2])
                h5["images"][1] = 0
                h5["image_ids"][1] = 0
                h5.attrs.update(num_written=1, complete=False)
            prepare.build_image_cache(sources, target, lambda *args, **kwargs: None,
                                      expected_count=2, image_backend="pillow")
            with h5py.File(target, "r") as h5:
                self.assertTrue(np.array_equal(h5["images"][:], np.stack(expected)))
                self.assertTrue(h5.attrs["complete"])
                self.assertEqual(h5.attrs["num_written"], 2)


@unittest.skipUnless(all(importlib.util.find_spec(k) is not None for k in
                         ["numpy", "PIL", "torch", "torchvision"]),
                     "torchvision unavailable for independent pixel parity")
class PillowParityTests(unittest.TestCase):
    def assert_parity(self, image):
        import numpy as np
        rgb = image.convert("RGB")
        canonical = prepare.image_preprocessor("torchvision")(rgb)
        lightweight = prepare.pillow_preprocess(rgb)
        self.assertEqual(lightweight.size, (224, 224))
        self.assertTrue(np.array_equal(np.asarray(canonical), np.asarray(lightweight)),
                        f"Pixel mismatch for original shape {image.size}")

    def test_odd_even_extreme_shapes_and_jpeg_decoding(self):
        import numpy as np
        from PIL import Image
        # Include half-pixel crop ties, unchanged-short-side, up/downsampling,
        # square, and very tall/wide inputs. The reference is real torchvision.
        shapes = [(224, 224), (224, 225), (224, 227), (225, 224), (227, 224),
                  (225, 226), (226, 225), (639, 481), (640, 480), (481, 639),
                  (13, 101), (101, 13), (5, 89), (89, 5), (1, 1), (1, 7), (7, 1)]
        rng = np.random.default_rng(2026)
        for width, height in shapes:
            with self.subTest(width=width, height=height):
                image = Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
                self.assert_parity(image)
                encoded = io.BytesIO()
                image.save(encoded, format="JPEG", quality=85)
                with Image.open(io.BytesIO(encoded.getvalue())) as decoded:
                    self.assert_parity(decoded)

    def test_available_real_coco_jpegs(self):
        from PIL import Image
        explicit = os.environ.get("VQA_PARITY_ZIP")
        candidates = [Path(explicit)] if explicit else [
            Path("/content/acma/data/zip/val2014.zip"),
            Path(__file__).resolve().parents[2] / "data/zip/val2014.zip"]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            self.skipTest("Set VQA_PARITY_ZIP to verify real downloaded COCO JPEGs")
        with zipfile.ZipFile(path) as archive:
            members = sorted(name for name in archive.namelist() if name.endswith(".jpg"))[:32]
            self.assertEqual(len(members), 32)
            for name in members:
                with self.subTest(name=name), Image.open(io.BytesIO(archive.read(name))) as image:
                    self.assert_parity(image)


if __name__ == "__main__":
    unittest.main()
