import json
import ast
import subprocess
import types

import numpy as np
import pytest
import torch
from PIL import Image

import data
from models import detector


def tiny_dataset(root):
    (root / "manifests").mkdir()
    (root / "images/kenai").mkdir(parents=True)
    pixels = np.arange(217 * 486, dtype=np.uint32).reshape(486, 217) % 256
    filename = "known_clip_0.jpg"
    Image.fromarray(pixels.astype(np.uint8)).save(root / "images/kenai" / filename, quality=97)
    image = {"id": 1, "file_name": filename, "width": 217, "height": 486}
    truth = {
        "images": [image],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 20, 30, 40],
                "area": 1200,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "fish"}],
    }
    (root / "manifests/train-100.json").write_text(json.dumps(truth))
    (root / "manifests/val.json").write_text(json.dumps({**truth, "images": [], "annotations": []}))
    return root / "manifests/train-100.json"


@pytest.mark.parametrize("size", [448, 672])
def test_rounded_coordinate_round_trip_at_supported_resolutions(size):
    image = Image.new("L", (217, 486), 89)
    tensor, scales = data.letterbox(image, size)
    assert tensor.shape == (3, size, size)
    assert scales == (round(217 * size / 486) / 217, size / 486)
    boxes = torch.tensor([[10.0, 20.0, 40.0, 60.0]])
    assert torch.allclose(data.undo_boxes(data.scale_boxes(boxes, scales), scales, 217, 486), boxes)


def test_default_448_letterbox_is_bitwise_equivalent():
    image = Image.fromarray(np.arange(217 * 486, dtype=np.uint8).reshape(486, 217))
    implicit = data.letterbox(image)
    explicit = data.letterbox(image, 448)
    assert torch.equal(implicit[0], explicit[0])
    assert implicit[1] == explicit[1]
    assert data.CACHE_FORMAT == data.cache_format(448)


def test_cache_identity_is_resolution_specific_and_wrong_cache_is_rejected(tmp_path):
    manifest = tiny_dataset(tmp_path)
    data.prepare_cache(tmp_path, size=448)
    cached = data.DetectionData(manifest, tmp_path, size=448)
    assert cached.cache_identity["size"] == 448
    assert cached.get(0)[0].shape == (3, 448, 448)

    cache = tmp_path / "cache"
    (cache / "kenai-gray672-u8.bin").write_bytes((cache / "kenai-gray448-u8.bin").read_bytes())
    (cache / "kenai-gray672.json").write_bytes((cache / "kenai-gray448.json").read_bytes())
    with pytest.raises(ValueError, match="resolution or format"):
        data.DetectionData(manifest, tmp_path, size=672)


def test_672_original_jpeg_path_without_cache(tmp_path):
    manifest = tiny_dataset(tmp_path)
    dataset = data.DetectionData(manifest, tmp_path, size=672)
    tensor, target, scales, _ = dataset.get(0)
    assert dataset.cache_identity is None
    assert tensor.shape == (3, 672, 672)
    original = torch.tensor([[10.0, 20.0, 40.0, 60.0]])
    assert torch.allclose(data.undo_boxes(target["boxes"], scales, 217, 486), original)


def test_resolution_does_not_change_same_seed_head_initialization():
    torch.set_num_threads(4)
    model_448 = detector("random", seed=31, size=448)
    model_672 = detector("random", seed=31, size=672)
    state_448, state_672 = model_448.state_dict(), model_672.state_dict()
    head_keys = [key for key in state_448 if not key.startswith("backbone.encoder.")]
    assert head_keys
    assert all(state_448[key].shape == state_672[key].shape for key in head_keys)
    assert all(torch.equal(state_448[key], state_672[key]) for key in head_keys)
    assert model_448.transform.min_size == (448,)
    assert model_672.transform.min_size == (672,)


@pytest.mark.parametrize("size,grid,last_stride", [(448, 32, 64), (672, 48, 67)])
def test_real_patch_register_pyramid_anchor_and_roi_conventions(size, grid, last_stride):
    from torchvision.models.detection.image_list import ImageList
    from torchvision.ops import roi_align

    torch.set_num_threads(4)
    model = detector("random", size=size).eval()
    x = torch.zeros(1, 3, size, size)
    with torch.no_grad():
        features = model.backbone.encoder.forward_features(x)
        assert features["x_norm_patchtokens"].shape == (1, grid * grid, 384)
        assert features["x_norm_regtokens"].shape == (1, 4, 384)
        assert features["x_prenorm"].shape[1] == grid * grid + 5
        pyramid = model.backbone(x)
        assert [f.shape[-1] for f in pyramid.values()] == [size // s for s in (8, 16, 32, 64)]
        boxes = [torch.tensor([[8.0, 16.0, 40.0, 48.0]])]
        actual = model.roi_heads.box_roi_pool(pyramid, boxes, [(size, size)])
        expected = roi_align(pyramid["0"], boxes, 7, spatial_scale=1 / 8, sampling_ratio=2)
        assert torch.equal(actual, expected)
        assert model.roi_heads.box_roi_pool.scales == [1 / 8, 1 / 16, 1 / 32, 1 / 64]
        anchors = model.rpn.anchor_generator(ImageList(x, [(size, size)]), list(pyramid.values()))[0]
        last = anchors[-((size // 64) ** 2) * 3 :]
        centers = (last[:, :2] + last[:, 2:]) / 2
        assert centers[3, 0] - centers[0, 0] == last_stride
        # At 672 Torchvision uses floor RPN strides and nearest power-of-two ROI scales.
        # Preserve and disclose this rounding; do not silently change only one arm.
        normalized, _ = model.transform([x[0]])
        expected_padding = -torch.tensor(model.transform.image_mean) / torch.tensor(model.transform.image_std)
        assert torch.allclose(normalized.tensors[0, :, -1, -1], expected_padding)


def test_full_high_resolution_cache_is_not_admitted(tmp_path):
    with pytest.raises(ValueError, match="prohibits a full 672 cache"):
        data.prepare_cache(tmp_path, size=672)


def test_448_factory_and_backbone_regress_against_completed_source():
    from pathlib import Path
    import models

    # This is the inspected project commit that produced the completed 448 runs.
    old_source = subprocess.check_output(["git", "show", "f87f7d3:src/models.py"], text=True)
    current_source = Path(models.__file__).read_text()
    old_tree, new_tree = ast.parse(old_source), ast.parse(current_source)
    old_pyramid = next(
        node for node in old_tree.body if isinstance(node, ast.ClassDef) and node.name == "SpatialPyramid"
    )
    new_pyramid = next(
        node for node in new_tree.body if isinstance(node, ast.ClassDef) and node.name == "SpatialPyramid"
    )
    assert ast.dump(old_pyramid) == ast.dump(new_pyramid)
    reference = types.ModuleType("completed_448_reference")
    reference.__file__ = models.__file__
    exec(compile(old_source, "completed_448_reference", "exec"), reference.__dict__)
    torch.set_num_threads(4)
    legacy = reference.detector("random", seed=7).eval()
    current = detector("random", seed=7, size=448).eval()
    assert all(torch.equal(v, current.state_dict()[k]) for k, v in legacy.state_dict().items())
    image = torch.zeros(3, 448, 448)
    image[:, :300, :175] = torch.linspace(0, 1, 300 * 175).reshape(300, 175)
    with torch.no_grad():
        before, after = legacy([image])[0], current([image])[0]
    assert all(torch.equal(before[key], after[key]) for key in before)
