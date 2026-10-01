import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from data import clip_name, letterbox, nested_clips, scale_boxes, undo_boxes, xywh_to_xyxy
from evaluate import score_predictions
from models import PRETRAINED, DinoObjective, detector, ema_update
from runtime import checkpoint


def ground_truth():
    return {
        "images": [{"id": 41, "width": 100, "height": 100}, {"id": 72, "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "fish"}],
        "annotations": [
            {"id": 9, "image_id": 41, "category_id": 1, "bbox": [10, 20, 30, 40], "area": 1200, "iscrowd": 0}
        ],
    }


def test_known_answer_scoring_and_negative_frames(tmp_path):
    gt = ground_truth()
    pred = [{"image_id": 41, "category_id": 1, "bbox": [10, 20, 30, 40], "score": 0.9}]
    perfect = score_predictions(gt, pred, tmp_path)
    assert perfect["AP50"] == pytest.approx(1)
    assert perfect["AP50:95"] == pytest.approx(1)
    assert perfect["precision"] == perfect["recall"] == 1
    assert perfect["negative_frames"] == 1
    assert perfect["negative_false_positives"] == 0
    bad = pred + [{"image_id": 72, "category_id": 1, "bbox": [0, 0, 20, 20], "score": 0.99}]
    result = score_predictions(gt, bad)
    assert result["AP50"] == pytest.approx(0.5)
    assert result["precision"] == 0.5
    assert result["false_positives_per_negative_frame"] == 1
    missed = score_predictions(gt, [])
    assert missed["AP50"] == 0 and missed["recall"] == 0
    assert json.loads((tmp_path / "metrics.json").read_text())["images_evaluated"] == 2
    assert (tmp_path / "coco_eval_images.json").exists()


def test_wrong_identity_rejected():
    with pytest.raises(AssertionError):
        score_predictions(
            ground_truth(), [{"image_id": 42, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 1}]
        )
    with pytest.raises(AssertionError):
        score_predictions(
            ground_truth(), [{"image_id": 41, "category_id": 0, "bbox": [0, 0, 1, 1], "score": 1}]
        )


def test_rounding_geometry_and_grayscale():
    image = Image.new("L", (217, 486), 89)
    x, scales = letterbox(image)
    assert x.shape == (3, 448, 448)
    assert torch.equal(x[0], x[1]) and torch.equal(x[1], x[2])
    assert scales[0] != scales[1]  # exact rounded axes, not an approximate single scale
    raw = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    expected = torch.tensor([[10.0, 20.0, 40.0, 60.0]])
    assert torch.equal(xywh_to_xyxy(raw), expected)
    assert torch.allclose(undo_boxes(scale_boxes(expected, scales), scales, 217, 486), expected)
    with pytest.raises(ValueError):
        letterbox(Image.new("RGB", (50, 50)))


def test_deterministic_nested_clip_policy():
    clips = {f"clip_{i}" for i in range(482)}
    one, ten, full = (nested_clips(clips, f) for f in (0.01, 0.1, 1.0))
    assert (len(one), len(ten), len(full)) == (5, 49, 482)
    assert one == ten[:5] and ten == full[:49]
    assert ten == nested_clips(list(reversed(sorted(clips))), 0.1)
    assert clip_name("clip_with_underscores_138.jpg") == "clip_with_underscores"


def test_teacher_stop_gradient_cross_view_center_and_ema():
    objective = DinoObjective(3)
    students = [torch.tensor([[1.0, 2.0, 0.0]], requires_grad=True) for _ in range(4)]
    teachers = [torch.tensor([[1.0, 0.0, 2.0]], requires_grad=True) for _ in range(2)]
    loss, targets = objective(students, teachers, 0.04)
    expected = -(targets[0] * torch.log_softmax(students[0] / 0.1, -1)).sum()
    assert float(loss.detach()) == pytest.approx(float(expected.detach()))
    assert torch.equal(objective.center, torch.zeros(1, 3))
    loss.backward()
    assert all(t.grad is None for t in teachers)
    assert all(s.grad is not None for s in students)
    objective.update_center([t.detach() for t in teachers])
    assert torch.allclose(objective.center, torch.tensor([[0.1, 0.0, 0.2]]))
    teacher, student = torch.nn.Linear(2, 1, bias=False), torch.nn.Linear(2, 1, bias=False)
    teacher.weight.data.fill_(2)
    student.weight.data.fill_(6)
    ema_update(teacher, student, 0.75)
    assert torch.equal(teacher.weight, torch.tensor([[3.0, 3.0]]))


def test_optimizer_checkpoint_resume(tmp_path):
    # Adam state and RNG continuity, not a synthetic claim of successful research training.
    torch.manual_seed(73)
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)

    def step():
        opt.zero_grad()
        model(torch.randn(3, 2)).square().mean().backward()
        opt.step()

    step()
    checkpoint(
        tmp_path / "resume.pt",
        {"model": model.state_dict(), "optimizer": opt.state_dict(), "rng": torch.get_rng_state()},
    )
    step()
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    state = torch.load(tmp_path / "resume.pt", weights_only=True)
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["rng"])
    step()
    assert all(torch.equal(v, expected[k]) for k, v in model.state_dict().items())


def test_matched_head_initialization_and_spatial_interface():
    if not PRETRAINED.exists():
        pytest.skip('Official checkpoint unavailable')
    torch.set_num_threads(4)
    published = detector('published', seed=7)
    random = detector('random', seed=7)
    a, b = published.state_dict(), random.state_dict()
    head_keys = [k for k in a if not k.startswith('backbone.encoder.')]
    assert all(torch.equal(a[k], b[k]) for k in head_keys)
    assert not torch.equal(a['backbone.encoder.patch_embed.proj.weight'],
                           b['backbone.encoder.patch_embed.proj.weight'])
    assert published.backbone.out_channels == random.backbone.out_channels == 96
    assert published.backbone.encoder.num_register_tokens == 4
    assert all(not p.requires_grad for p in published.backbone.encoder.parameters())


def test_real_manifest_no_split_overlap_and_complete_labels():
    root = Path("D:/sonar-representation-lab-data/manifests")
    if not (root / "train-010.json").exists():
        pytest.skip("metadata preparation not yet finished")
    one, ten, full, val = [
        json.loads((root / name).read_text())
        for name in ["train-001.json", "train-010.json", "train-100.json", "val.json"]
    ]
    assert one["clip_names"] == ten["clip_names"][: len(one["clip_names"])]
    assert ten["clip_names"] == full["clip_names"][: len(ten["clip_names"])]
    assert set(full["clip_names"]).isdisjoint({clip_name(i["file_name"]) for i in val["images"]})
    ids = {i["id"] for i in ten["images"]}
    assert ten["annotations"] == [a for a in full["annotations"] if a["image_id"] in ids]
    assert ten["counts"]["negative_frames"] > 0
    ssl = json.loads((root / "ssl-train.json").read_text())
    assert set(ssl) == {"split", "filenames"}
    assert set(ssl["filenames"]) == {i["file_name"] for i in full["images"]}
