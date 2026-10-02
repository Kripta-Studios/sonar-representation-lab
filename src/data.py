"""Clip-preserving CFC manifests and grayscale geometry."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import shutil
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import platform_compat  # noqa: F401
import numpy as np
import psutil
import torch
from PIL import Image

from acquire import ROOT, save_json

SIZE = 448


def clip_name(filename):
    stem, frame = Path(filename).stem.rsplit("_", 1)
    assert frame.isdigit(), filename
    return stem


def ranked_clips(clips):
    return sorted(clips, key=lambda c: hashlib.sha256(("sonar-lab-subsets-v1:" + c).encode()).hexdigest())


def nested_clips(clips, fraction):
    return ranked_clips(clips)[: max(1, math.ceil(len(clips) * fraction))]


def prepare(root=ROOT):
    meta = root / "metadata"
    dest = root / "manifests"
    dest.mkdir(exist_ok=True)
    source = {}
    for split in ("train", "val"):
        path = meta / "coco_annotations_v1.1" / f"kenai-{split}.json"
        d = json.loads(path.read_text())
        assert d["categories"] == [{"supercategory": "", "id": 1, "name": "fish"}]
        assert len({i["id"] for i in d["images"]}) == len(d["images"])
        assert len({i["file_name"] for i in d["images"]}) == len(d["images"])
        lists = {
            Path(s).name for s in (meta / "file_lists_v1.1" / f"kenai-{split}.txt").read_text().splitlines()
        }
        assert lists == {i["file_name"] for i in d["images"]}
        clips = {clip_name(i["file_name"]) for i in d["images"]}
        clip_meta = json.loads((meta / "metadata" / f"kenai-{split}.json").read_text())
        assert clips == {c["clip_name"] for c in clip_meta}
        counts = Counter(clip_name(i["file_name"]) for i in d["images"])
        differences = [
            {
                "clip": c["clip_name"],
                "metadata_frames": c["num_frames"],
                "coco_and_split_frames": counts[c["clip_name"]],
            }
            for c in clip_meta
            if counts[c["clip_name"]] != c["num_frames"]
        ]
        # The current official v1.1 lists contain one fewer frame per clip than the older metadata.
        # Keep actual official membership; do not fabricate an extra frame or drop labels.
        save_json(dest / f"{split}-metadata_discrepancies.json", differences)
        image_ids = {i["id"] for i in d["images"]}
        for a in d["annotations"]:
            assert a["image_id"] in image_ids and a["category_id"] == 1
            assert a["bbox"][2] > 0 and a["bbox"][3] > 0 and a["iscrowd"] == 0
        source[split] = (d, clips)
    assert source["train"][1].isdisjoint(source["val"][1])
    for fraction in (0.01, 0.1, 1.0):
        d, clips = source["train"]
        selected = nested_clips(clips, fraction)
        ims = [i for i in d["images"] if clip_name(i["file_name"]) in set(selected)]
        ids = {i["id"] for i in ims}
        ann = [a for a in d["annotations"] if a["image_id"] in ids]
        used = {a["image_id"] for a in ann}
        manifest = {
            "fraction": fraction,
            "clip_names": selected,
            "images": ims,
            "annotations": ann,
            "categories": d["categories"],
            "counts": {
                "clips": len(selected),
                "frames": len(ims),
                "annotations": len(ann),
                "negative_frames": len(ids - used),
            },
            "source_sha256": hashlib.sha256(
                (meta / "coco_annotations_v1.1/kenai-train.json").read_bytes()
            ).hexdigest(),
        }
        save_json(dest / f"train-{int(fraction * 100):03d}.json", manifest)
        print(fraction, manifest["counts"], flush=True)
    d, _ = source["train"]
    # This is the ONLY input consumed by SSL: no categories, boxes or track IDs.
    save_json(
        dest / "ssl-train.json",
        {"split": "official-kenai-train", "filenames": [i["file_name"] for i in d["images"]]},
    )
    save_json(dest / "val.json", source["val"][0])


def letterbox(image, size=SIZE):
    if image.mode != "L":
        raise ValueError(f"Expected official grayscale, got {image.mode}")
    w, h = image.size
    scale = size / max(w, h)
    rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = image.resize((rw, rh), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(resized, (0, 0))
    x = torch.from_numpy(np.asarray(canvas).copy()).float().div_(255).unsqueeze(0).repeat(3, 1, 1)
    return x, (rw / w, rh / h)


def xywh_to_xyxy(boxes):
    out = boxes.clone()
    out[:, 2:] += out[:, :2]
    return out


def scale_boxes(boxes, scales):
    return boxes * boxes.new_tensor([scales[0], scales[1], scales[0], scales[1]])


def undo_boxes(boxes, scales, width, height):
    out = boxes / boxes.new_tensor([scales[0], scales[1], scales[0], scales[1]])
    out[:, 0::2].clamp_(0, width)
    out[:, 1::2].clamp_(0, height)
    return out


CACHE_FORMAT = "CFC-v1.1-L-PIL-bilinear-448-topleft-uint8-v1"


def prepare_cache(root=ROOT, location="kenai", workers=4):
    """CPU-only, bounded preparation of exactly the existing detector canvases."""
    sources = (
        ["manifests/train-100.json", "manifests/val.json"]
        if location == "kenai"
        else ["metadata/coco_annotations_v1.1/kenai-channel.json"]
    )
    images = []
    source_hashes = {}
    for name in sources:
        raw = (root / name).read_bytes()
        source_hashes[name] = hashlib.sha256(raw).hexdigest()
        images.extend(json.loads(raw)["images"])
    assert len({i["file_name"] for i in images}) == len(images)
    directory = root / "cache"
    directory.mkdir(exist_ok=True)
    target = directory / f"{location}-gray448-u8.bin"
    metadata = directory / f"{location}-gray448.json"
    temporary = target.with_suffix(".partial")
    if target.exists() or metadata.exists() or temporary.exists():
        raise RuntimeError("Cache output already exists; preserve it instead of silently rebuilding")
    required = len(images) * SIZE * SIZE
    if shutil.disk_usage(directory).free < required + 1024**3:
        raise RuntimeError(f"Insufficient cache space: need {required + 1024**3} bytes")
    started = time.time()

    def convert(info):
        with Image.open(root / "images" / location / info["file_name"]) as image:
            if image.mode != "L" or image.size != (info["width"], info["height"]):
                raise ValueError(f"Cache grayscale/dimensions mismatch: {info['file_name']}")
            w, h = image.size
            scale = SIZE / max(w, h)
            rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
            canvas = Image.new("L", (SIZE, SIZE), 0)
            canvas.paste(image.resize((rw, rh), Image.Resampling.BILINEAR), (0, 0))
            raw = canvas.tobytes()
        return raw, {"file_name": info["file_name"], "width": w, "height": h, "crc32": zlib.crc32(raw)}

    rows = []
    digest = hashlib.sha256()
    process = psutil.Process()
    with target.with_suffix(".partial").open("wb") as stream:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for first in range(0, len(images), 128):
                # map only this bounded window, never the complete dataset at once.
                for raw, row in pool.map(convert, images[first : first + 128]):
                    assert len(raw) == SIZE * SIZE
                    stream.write(raw)
                    digest.update(raw)
                    rows.append(row)
                memory = process.memory_info()
                if max(memory.rss, getattr(memory, "peak_wset", memory.rss)) >= 22 * 1024**3:
                    raise RuntimeError("Owned cache preparation RAM ceiling reached")
                if first % 2048 == 0 or len(rows) == len(images):
                    print(
                        json.dumps(
                            {
                                "cached_frames": len(rows),
                                "total": len(images),
                                "cpu_wall_seconds": time.time() - started,
                            }
                        ),
                        flush=True,
                    )
    temporary.replace(target)
    assert target.stat().st_size == required
    save_json(
        metadata,
        {
            "format": CACHE_FORMAT,
            "location": location,
            "size": SIZE,
            "source_sha256": source_hashes,
            "blob_sha256": digest.hexdigest(),
            "blob_bytes": required,
            "rows": rows,
            "cpu_wall_seconds": time.time() - started,
            "owned_native_peak_bytes": getattr(process.memory_info(), "peak_wset", process.memory_info().rss),
            "gpu_used": False,
        },
    )


class DetectionData:
    def __init__(self, manifest, root=ROOT, location="kenai", use_cache=True):
        self.manifest = Path(manifest)
        self.data = json.loads(self.manifest.read_text())
        self.images = self.data["images"]
        self.root = root / "images" / location
        self.annotations = defaultdict(list)
        for a in self.data["annotations"]:
            self.annotations[a["image_id"]].append(a)
        self.cache_identity = None
        self.cache_rows = {}
        self.cache_blob = root / "cache" / f"{location}-gray448-u8.bin"
        metadata = root / "cache" / f"{location}-gray448.json"
        if use_cache and metadata.exists():
            cached = json.loads(metadata.read_text())
            if cached["format"] != CACHE_FORMAT or cached["size"] != SIZE or cached["location"] != location:
                raise ValueError("Detector canvas cache format changed")
            for name, expected in cached["source_sha256"].items():
                if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                    raise ValueError("Detector cache source manifest changed")
            if self.cache_blob.stat().st_size != cached["blob_bytes"]:
                raise ValueError("Detector canvas cache size changed")
            self.cache_rows = {row["file_name"]: (index, row) for index, row in enumerate(cached["rows"])}
            self.cache_identity = {
                "format": CACHE_FORMAT,
                "manifest_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
                "blob_sha256": cached["blob_sha256"],
            }
            for info in self.images:
                row = self.cache_rows[info["file_name"]][1]
                assert (row["width"], row["height"]) == (info["width"], info["height"])

    def __len__(self):
        return len(self.images)

    def get(self, index):
        info = self.images[index]
        if self.cache_identity is None:
            with Image.open(self.root / info["file_name"]) as im:
                assert im.size == (info["width"], info["height"]), (im.size, info)
                x, scales = letterbox(im)
        else:
            index, row = self.cache_rows[info["file_name"]]
            with self.cache_blob.open("rb") as stream:
                stream.seek(index * SIZE * SIZE)
                raw = stream.read(SIZE * SIZE)
            if len(raw) != SIZE * SIZE or zlib.crc32(raw) != row["crc32"]:
                raise ValueError(f"Detector cache row integrity failed: {info['file_name']}")
            canvas = np.frombuffer(raw, dtype=np.uint8).reshape(SIZE, SIZE)
            x = torch.from_numpy(canvas.copy()).float().div_(255).unsqueeze(0).repeat(3, 1, 1)
            w, h = info["width"], info["height"]
            scale = SIZE / max(w, h)
            scales = (max(1, round(w * scale)) / w, max(1, round(h * scale)) / h)
        raw = torch.tensor([a["bbox"] for a in self.annotations[info["id"]]], dtype=torch.float32).reshape(
            -1, 4
        )
        boxes = scale_boxes(xywh_to_xyxy(raw), scales)
        target = {
            "boxes": boxes,
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "image_id": torch.tensor(info["id"]),
        }
        return x, target, scales, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--cache", choices=["kenai", "channel"])
    args = p.parse_args()
    if args.cache:
        prepare_cache(args.root, args.cache)
    else:
        prepare(args.root)


if __name__ == "__main__":
    main()
