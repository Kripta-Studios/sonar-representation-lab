"""Clip-preserving CFC manifests and grayscale geometry."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import platform_compat  # noqa: F401
import numpy as np
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


class DetectionData:
    def __init__(self, manifest, root=ROOT, location="kenai"):
        self.manifest = Path(manifest)
        self.data = json.loads(self.manifest.read_text())
        self.images = self.data["images"]
        self.root = root / "images" / location
        self.annotations = defaultdict(list)
        for a in self.data["annotations"]:
            self.annotations[a["image_id"]].append(a)

    def __len__(self):
        return len(self.images)

    def get(self, index):
        info = self.images[index]
        with Image.open(self.root / info["file_name"]) as im:
            assert im.size == (info["width"], info["height"]), (im.size, info)
            x, scales = letterbox(im)
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
    args = p.parse_args()
    prepare(args.root)


if __name__ == "__main__":
    main()
