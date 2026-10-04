"""Cached pooled AP must equal explicitly remapped duplicate-clip COCO scoring."""

import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from clip_bootstrap import compact_rows, pooled_ap, remap_clips, stream_rows  # noqa: E402
from evaluate import score_predictions  # noqa: E402


def test_duplicate_clips_ties_and_negative_frames_match_full_coco(tmp_path):
    gt = {
        "images": [
            {"id": 41, "file_name": "positive_0.jpg", "width": 100, "height": 100},
            {"id": 42, "file_name": "positive_1.jpg", "width": 100, "height": 100},
            {"id": 72, "file_name": "negative_0.jpg", "width": 100, "height": 100},
        ],
        "categories": [{"id": 1, "name": "fish"}],
        "annotations": [
            {"id": 9, "image_id": 41, "category_id": 1, "bbox": [10, 20, 30, 40], "area": 1200, "iscrowd": 0}
        ],
    }
    predictions = [
        {"image_id": 41, "category_id": 1, "bbox": [10, 20, 30, 40], "score": 0.9},
        {"image_id": 72, "category_id": 1, "bbox": [0, 0, 20, 20], "score": 0.9},
    ]
    score_predictions(gt, predictions, tmp_path)
    with gzip.open(tmp_path / "coco_eval_images.json.gz", "rt") as stream:
        saved = json.load(stream)
    assert list(stream_rows(tmp_path / "coco_eval_images.json.gz")) == saved
    compact = compact_rows(iter(saved), gt["images"])
    full = score_predictions(gt, predictions)
    original = pooled_ap(compact, ["negative", "positive"], original_order=True)
    assert abs(original[0] - full["AP50"]) < 1e-12
    for draw in (
        ["positive", "negative"],
        ["negative", "positive", "negative"],
        ["positive", "positive", "negative"],
        ["positive"],
        ["negative"],
    ):
        truth, detections = remap_clips(gt, predictions, draw)
        assert len({row["id"] for row in truth["images"]}) == len(truth["images"])
        assert len({row["id"] for row in truth["annotations"]}) == len(truth["annotations"])
        reference = score_predictions(truth, detections)
        result = pooled_ap(compact, draw)
        if reference["AP50"] == -1:
            assert np.isnan(result).all()
        else:
            assert abs(result[0] - reference["AP50"]) < 1e-12
            assert abs(result.mean() - reference["AP50:95"]) < 1e-12
