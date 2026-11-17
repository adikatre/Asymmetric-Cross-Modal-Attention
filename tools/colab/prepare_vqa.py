#!/usr/bin/env python3
"""Recover the canonical VQA v2 224px inputs from official public archives.

No JPEG tree is extracted. Images are read directly from COCO ZIPs and transformed
exactly as canonical notebook cell 10: RGB, bicubic Resize(224), CenterCrop(224).
The resumable HDF5 has the canonical images/image_ids schema and completion attrs.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import time
import traceback
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile

IMAGE_COUNT = 123_287
IMAGE_BYTES = IMAGE_COUNT * 224 * 224 * 3 + IMAGE_COUNT * 8
FEATURE_BYTES = 77_632_503_580
EXPECTED_JSON = {
    "v2_mscoco_train2014_annotations.json": "770d2d3980e7838d6186082e0ddeddd804014c390ae45a592e996d6baf841d92",
    "v2_mscoco_val2014_annotations.json": "0564650e712b34c246e4ad142894692bf2e45dee8e9cc6ebb2cedbe85731c4bf",
    "v2_OpenEnded_mscoco_train2014_questions.json": "cb671e77b52a86fc3722b2a0ffc456c8199240391bfd97eaf2d850448404845c",
    "v2_OpenEnded_mscoco_val2014_questions.json": "f34d9c9909dea76700d361baddd63dd47b4494945097459b542d9fe840244811",
}


def official_sources():
    sources = []
    for split, title in (("train", "Train"), ("val", "Val")):
        for kind, folder, filename in (
            ("Annotations", "answers", f"v2_mscoco_{split}2014_annotations.json"),
            ("Questions", "questions", f"v2_OpenEnded_mscoco_{split}2014_questions.json"),
        ):
            archive = f"v2_{kind}_{title}_mscoco.zip"
            sources.append({"name": archive,
                            "url": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/" + archive,
                            "folder": folder, "member": filename})
    for split, count in (("train2014", 82_783), ("val2014", 40_504)):
        sources.append({"name": split + ".zip",
                        "url": f"https://images.cocodataset.org/zips/{split}.zip",
                        # Same public S3 bucket, path-style HTTPS avoids a
                        # certificate mismatch on the dotted COCO bucket name.
                        "mirror": f"https://s3.amazonaws.com/images.cocodataset.org/zips/{split}.zip",
                        "split": split, "count": count})
    return sources


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class Status:
    def __init__(self, path):
        self.path = Path(path)
        self.record = {"phase": "starting", "started_at": time.time(), "pid": os.getpid()}

    def __call__(self, phase, **fields):
        # Keep download counters out of subsequent image-building phases.
        if self.record.get("phase") != phase:
            self.record = {k: v for k, v in self.record.items()
                           if k in {"started_at", "pid"}}
        self.record.update(phase=phase, heartbeat_at=time.time(), **fields)
        atomic_json(self.path, self.record)
        print(json.dumps(self.record), flush=True)


def archive_members(path):
    with zipfile.ZipFile(path) as archive:
        return archive.namelist()


def download(url, destination, status, retries=5, mirror=None):
    """Resume a partial HTTP transfer; never append a 200 response to old bytes."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if destination.exists():
        archive_members(destination)  # central directory must be readable
        status("hashing_download", file=destination.name, bytes=destination.stat().st_size)
        return {"url": url, "bytes": destination.stat().st_size, "sha256": sha256(destination)}
    for attempt in range(1, retries + 1):
        actual_url = mirror if mirror and attempt > 1 else url
        offset = partial.stat().st_size if partial.exists() else 0
        status("downloading", file=destination.name, bytes=offset, attempt=attempt)
        try:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            try:
                response = urlopen(Request(actual_url, headers=headers), timeout=60)
            except HTTPError as error:
                if error.code != 416:
                    raise
                response = error
            with response:
                # A complete .part can survive a crash before the rename.
                if response.status == 416 and offset:
                    total = response.headers.get("Content-Range", "").split("/")[-1]
                    if total.isdigit() and int(total) == offset:
                        archive_members(partial)
                        partial.replace(destination)
                        break
                if response.status not in (200, 206):
                    raise RuntimeError(f"Unsupported download status {response.status}")
                if response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise RuntimeError(f"Unexpected Content-Range: {content_range}")
                else:
                    offset = 0
                expected = response.headers.get("Content-Length")
                total = offset + int(expected) if expected is not None else None
                written = offset
                last_report = time.monotonic()
                with partial.open("ab" if offset else "wb") as stream:
                    for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                        stream.write(block)
                        written += len(block)
                        if time.monotonic() - last_report >= 10:
                            stream.flush()
                            status("downloading", file=destination.name, bytes=written,
                                   total_bytes=total, attempt=attempt, progress_at=time.time())
                            last_report = time.monotonic()
                    stream.flush()
                    os.fsync(stream.fileno())
                if total is not None and written != total:
                    raise IOError(f"Incomplete transfer: {written} of {total} bytes")
                archive_members(partial)
                partial.replace(destination)
                break
        except Exception as error:
            status("download_retry", file=destination.name, attempt=attempt,
                   error=f"{type(error).__name__}: {error}")
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 20))
    status("hashing_download", file=destination.name, bytes=destination.stat().st_size)
    return {"url": url, "download_url": actual_url,
            "bytes": destination.stat().st_size, "sha256": sha256(destination)}


def extract_json(archive_path, member, destination, expected_sha):
    """Extract only the named official basename, rejecting ambiguous ZIP members."""
    if Path(member).name != member:
        raise ValueError("Expected a plain JSON filename")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        matches = [row for row in archive.infolist() if row.filename == member]
        if len(matches) != 1 or matches[0].is_dir():
            raise ValueError(f"Expected exactly one {member} in {archive_path}")
        temporary = destination.with_name(destination.name + ".tmp")
        with archive.open(matches[0]) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        actual = sha256(temporary)
        if actual != expected_sha:
            temporary.unlink()
            raise ValueError(f"Original dataset hash mismatch for {member}: {actual}")
        temporary.replace(destination)
    return {"bytes": destination.stat().st_size, "sha256": actual}


def image_members(archive, split, expected_count):
    pattern = re.compile(rf"^{re.escape(split)}/COCO_{re.escape(split)}_(\d{{12}})\.jpg$")
    members = []
    for row in archive.infolist():
        match = pattern.fullmatch(row.filename)
        if match and not row.is_dir():
            members.append((row.filename, int(match.group(1))))
    members.sort()
    if len(members) != expected_count or len({row[1] for row in members}) != expected_count:
        raise ValueError(f"Expected {expected_count} unique {split} JPEGs, found {len(members)}")
    return members


def pillow_preprocess(image):
    """Pillow equivalent of torchvision Resize(224,BICUBIC), CenterCrop(224).

    Geometry follows the official torchvision PIL implementation: the long side
    truncates after scaling; crop offsets use Python round (ties to even).
    https://github.com/pytorch/vision/blob/main/torchvision/transforms/functional.py
    The caller supplies RGB, just as notebook cell 10 does.
    """
    from PIL import Image

    width, height = image.size
    short, long = sorted((width, height))
    scaled_long = int(224 * long / short)
    output_shape = (224, scaled_long) if width <= height else (scaled_long, 224)
    if output_shape != image.size:
        image = image.resize(output_shape, resample=Image.Resampling.BICUBIC)
    left = int(round((output_shape[0] - 224) / 2.0))
    top = int(round((output_shape[1] - 224) / 2.0))
    return image.crop((left, top, left + 224, top + 224))


def image_preprocessor(backend):
    if backend == "pillow":
        return pillow_preprocess
    if backend != "torchvision":
        raise ValueError(f"Unknown image backend: {backend}")
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
    ])


def build_image_cache(archives, target, status, expected_count=IMAGE_COUNT,
                      image_backend="torchvision"):
    import h5py
    import numpy as np
    import PIL
    from PIL import Image

    preprocess = image_preprocessor(image_backend)
    source_hashes = {source["split"]: source["sha256"] for source in archives}
    source_identity = json.dumps(source_hashes, sort_keys=True)
    with ExitStack() as stack:
        rows = []
        for source in archives:
            archive = stack.enter_context(zipfile.ZipFile(source["path"]))
            rows.extend((archive, filename, image_id)
                        for filename, image_id in image_members(archive, source["split"], source["count"]))
        if len(rows) != expected_count or len({r[2] for r in rows}) != expected_count:
            raise ValueError("Combined COCO image count or unique IDs mismatch")
        with h5py.File(target, "a") as h5:
            if "images" not in h5:
                h5.create_dataset("images", (expected_count, 224, 224, 3), dtype=np.uint8,
                                  chunks=(1, 224, 224, 3))
                h5.create_dataset("image_ids", (expected_count,), dtype=np.int64)
                h5.attrs.update(num_written=0, complete=False, source_sha256=source_identity,
                                preprocessing="torchvision_RGB_Resize224_BICUBIC_CenterCrop224",
                                image_backend=image_backend, pillow_version=PIL.__version__,
                                numpy_version=np.__version__, h5py_version=h5py.__version__)
            if h5["images"].shape != (expected_count, 224, 224, 3) or h5["images"].dtype != np.uint8:
                raise ValueError("Incompatible existing image cache")
            if h5["image_ids"].shape != (expected_count,) or h5["image_ids"].dtype != np.int64:
                raise ValueError("Incompatible existing image IDs")
            if h5.attrs.get("source_sha256") != source_identity:
                raise ValueError("Image cache resume source hashes do not match")
            cursor = int(h5.attrs.get("num_written", 0))
            if not 0 <= cursor <= expected_count:
                raise ValueError("Invalid image-cache resume cursor")
            expected_ids = np.asarray([row[2] for row in rows], dtype=np.int64)
            if not np.array_equal(h5["image_ids"][:cursor], expected_ids[:cursor]):
                raise ValueError("Existing image IDs differ from canonical sorted source order")
            status("building_image_cache", images_written=cursor, total_images=expected_count)
            for index in range(cursor, expected_count):
                archive, filename, image_id = rows[index]
                # Reading each ZIP member verifies its CRC before storing the image.
                with Image.open(io.BytesIO(archive.read(filename))) as image:
                    h5["images"][index] = np.asarray(preprocess(image.convert("RGB")))
                h5["image_ids"][index] = image_id
                if (index + 1) % 1000 == 0:
                    h5.attrs["num_written"] = index + 1
                    h5.flush()
                    status("building_image_cache", images_written=index + 1,
                           total_images=expected_count, progress_at=time.time())
            h5.attrs["num_written"] = expected_count
            h5.attrs["complete"] = True
            h5.flush()
    status("hashing_image_cache", images_written=expected_count, bytes=Path(target).stat().st_size)
    return {"path": str(target), "bytes": Path(target).stat().st_size,
            "sha256": sha256(target), "source_sha256": source_hashes,
            "count": expected_count, "complete": True, "image_backend": image_backend,
            "pillow_version": PIL.__version__, "numpy_version": np.__version__,
            "h5py_version": h5py.__version__}


def complete_cache(path):
    if not path.exists():
        return False
    import h5py
    import numpy as np
    with h5py.File(path, "r") as h5:
        valid = ("images" in h5 and "image_ids" in h5
                 and h5["images"].shape == (IMAGE_COUNT, 224, 224, 3)
                 and h5["images"].dtype == np.uint8
                 and h5["image_ids"].shape == (IMAGE_COUNT,)
                 and int(h5.attrs.get("num_written", 0)) == IMAGE_COUNT
                 and bool(h5.attrs.get("complete", False)))
        if not valid:
            return False
        ids = h5["image_ids"][:]
        return bool(np.all(ids != 0) and len(np.unique(ids)) == IMAGE_COUNT)


def preflight_space(data_dir, need_images, reserve_feature_gb, download_only=False):
    # COCO archives total about 20GB; use 25GB as a conservative download bound.
    archive_reserve = 25_000_000_000 if need_images else 0
    image_reserve = IMAGE_BYTES if need_images and not download_only else 0
    # Existing sparse/partial images consume only their allocated blocks.
    image_path = data_dir / "vqa_images_224.h5"
    if need_images and not download_only and image_path.exists():
        image_reserve = max(0, IMAGE_BYTES - image_path.stat().st_blocks * 512)
    zip_dir = data_dir / "zip"
    existing_downloads = sum(p.stat().st_size for p in zip_dir.glob("*.zip*")) if zip_dir.exists() else 0
    archive_reserve = max(0, archive_reserve - existing_downloads)
    feature_path = data_dir / "vqa_frozen_features_dcfce3a81574.h5"
    feature_reserve = int(reserve_feature_gb * 1e9)
    if feature_path.exists():
        feature_reserve = max(0, feature_reserve - feature_path.stat().st_blocks * 512)
    required = archive_reserve + image_reserve + feature_reserve + 8_000_000_000
    free = shutil.disk_usage(data_dir).free
    if free < required:
        raise RuntimeError(f"Need approximately {required / 1e9:.1f}GB free for downloads, "
                           f"224px cache, downstream features, and reserve; have {free / 1e9:.1f}GB")
    return {"free_bytes": free, "estimated_required_bytes": required}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/content/acma/data"))
    parser.add_argument("--status", type=Path, default=Path("/content/acma/data_status.json"))
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--reserve-feature-gb", type=float, default=FEATURE_BYTES / 1e9)
    parser.add_argument("--download-only", action="store_true",
                        help="Download official archives and verify/extract JSON; no ML/image dependencies")
    parser.add_argument("--image-backend", choices=("torchvision", "pillow"), default="torchvision",
                        help="pillow builds the same resize/crop pixels without installing torch")
    args = parser.parse_args(argv)
    if args.retries < 1 or args.reserve_feature_gb < 0:
        parser.error("retries must be positive and feature reserve nonnegative")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    status = Status(args.status)
    manifest_path = args.data_dir / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "schema_version": 1, "created_at": time.time(), "sources": {}, "files": {}, "events": []}
    try:
        target = args.data_dir / "vqa_images_224.h5"
        cache_ready = False if args.download_only else complete_cache(target)
        space = preflight_space(args.data_dir, not cache_ready,
                                0 if args.download_only else args.reserve_feature_gb,
                                download_only=args.download_only)
        status("disk_preflight", **space)
        archives = []
        for source in official_sources():
            if "member" in source:
                output = args.data_dir / source["folder"] / source["member"]
                expected = EXPECTED_JSON[source["member"]]
                if output.exists() and sha256(output) == expected:
                    manifest["files"][source["member"]] = {
                        "path": str(output), "sha256": expected, "bytes": output.stat().st_size}
                    atomic_json(manifest_path, manifest)
                    continue
            elif cache_ready:
                continue
            archive_path = args.data_dir / "zip" / source["name"]
            receipt = download(source["url"], archive_path, status, retries=args.retries,
                               mirror=source.get("mirror"))
            previous = manifest["sources"].get(source["name"])
            if previous and previous["sha256"] != receipt["sha256"]:
                raise ValueError(f"Downloaded source changed: {source['name']}")
            manifest["sources"][source["name"]] = receipt
            manifest["events"].append({"time": time.time(), "event": "source_verified",
                                       "name": source["name"], **receipt})
            atomic_json(manifest_path, manifest)
            if "member" in source:
                receipt = extract_json(archive_path, source["member"], output, expected)
                manifest["files"][source["member"]] = {"path": str(output), **receipt}
                atomic_json(manifest_path, manifest)
            else:
                archives.append({**source, "path": archive_path, "sha256": receipt["sha256"]})
        if args.download_only:
            manifest["downloads_completed_at"] = time.time()
            manifest["events"].append({"time": time.time(), "event": "downloads_complete"})
            atomic_json(manifest_path, manifest)
            status("downloads_complete", manifest=str(manifest_path),
                   completion_marker="VQA_DOWNLOADS_COMPLETE")
            return
        if not cache_ready:
            manifest["image_cache"] = build_image_cache(archives, target, status,
                                                        image_backend=args.image_backend)
        else:
            status("hashing_image_cache", bytes=target.stat().st_size)
            manifest["image_cache"] = {"path": str(target), "bytes": target.stat().st_size,
                                       "sha256": sha256(target), "count": IMAGE_COUNT,
                                       "complete": True, "reused": True}
        manifest["completed_at"] = time.time()
        manifest["events"].append({"time": time.time(), "event": "preparation_complete"})
        atomic_json(manifest_path, manifest)
        status("complete", image_cache=str(target), manifest=str(manifest_path),
               total_images=IMAGE_COUNT, completion_marker="VQA_DATA_PREPARATION_COMPLETE")
    except BaseException as error:
        status("failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
