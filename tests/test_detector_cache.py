import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import data


def tiny_dataset(root):
    (root / "manifests").mkdir()
    (root / "images/kenai").mkdir(parents=True)
    images = []
    for index, (width, height) in enumerate([(217, 486), (51, 31)]):
        filename = f"known_clip_{index}_0.jpg"
        pixels = np.arange(width * height, dtype=np.uint32).reshape(height, width) % 256
        Image.fromarray(pixels.astype(np.uint8)).save(root / "images/kenai" / filename, quality=97)
        images.append({"id": index + 1, "file_name": filename, "width": width, "height": height})
    truth = {
        "images": images,
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40], "area": 1200, "iscrowd": 0}
        ],
        "categories": [{"id": 1, "name": "fish"}],
    }
    (root / "manifests/train-100.json").write_text(json.dumps(truth))
    (root / "manifests/val.json").write_text(json.dumps({**truth, "images": [], "annotations": []}))
    return root / "manifests/train-100.json"


def test_cache_tensors_targets_geometry_equal_and_corruption_rejected(tmp_path):
    torch.set_num_threads(4)
    manifest = tiny_dataset(tmp_path)
    original = data.DetectionData(manifest, tmp_path)
    expected = [original.get(index) for index in range(len(original))]
    data.prepare_cache(tmp_path)
    cached = data.DetectionData(manifest, tmp_path)
    assert cached.cache_identity is not None
    for index, reference in enumerate(expected):
        # Removing JPEGs proves this actually exercises the cached path.
        (tmp_path / "images/kenai" / reference[3]["file_name"]).unlink()
        actual = cached.get(index)
        assert torch.equal(actual[0], reference[0])
        assert actual[2:] == reference[2:]
        for key in reference[1]:
            assert torch.equal(actual[1][key], reference[1][key])
    assert cached.get(1)[1]["boxes"].shape == (0, 4)
    with cached.cache_blob.open("r+b") as stream:
        first = stream.read(1)
        stream.seek(0)
        stream.write(bytes([first[0] ^ 1]))
    with pytest.raises(ValueError, match="integrity"):
        cached.get(0)


def test_cache_rejects_changed_source_manifest(tmp_path):
    manifest = tiny_dataset(tmp_path)
    data.prepare_cache(tmp_path)
    truth = json.loads(manifest.read_text())
    truth["annotations"] = []
    manifest.write_text(json.dumps(truth))
    with pytest.raises(ValueError, match="source manifest changed"):
        data.DetectionData(manifest, tmp_path)


def test_real_train_cache_bitwise_reload():
    root = Path("D:/sonar-representation-lab-data")
    if not (root / "cache/kenai-gray448.json").exists():
        pytest.skip("Real derived cache has not completed")
    manifest = root / "manifests/train-010.json"
    original = data.DetectionData(manifest, root, use_cache=False)
    cached = data.DetectionData(manifest, root)
    indices = torch.randperm(len(original), generator=torch.Generator().manual_seed(971))[:128]
    for index in indices.tolist():
        a, b = original.get(index), cached.get(index)
        assert torch.equal(a[0], b[0])
        assert a[2:] == b[2:]
        assert all(torch.equal(a[1][key], b[1][key]) for key in a[1])
