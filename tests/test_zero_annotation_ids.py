"""Actual publisher zero IDs must not collide with COCO's unmatched sentinel."""

import json

import pytest

from evaluate import score_predictions


def test_annotation_zero_id_known_answer_and_source_preservation(tmp_path):
    truth = {
        "images": [{"id": 0, "width": 100, "height": 100}, {"id": 1, "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "fish"}],
        "annotations": [
            {"id": 0, "image_id": 0, "category_id": 1, "bbox": [10, 20, 30, 40], "area": 1200, "iscrowd": 0}
        ],
    }
    predictions = [{"image_id": 0, "category_id": 1, "bbox": [10, 20, 30, 40], "score": 0.9}]
    result = score_predictions(truth, predictions, tmp_path)
    assert result["AP50"] == pytest.approx(1)
    assert result["AP50:95"] == pytest.approx(1)
    assert truth["annotations"][0]["id"] == 0
    identity = json.loads((tmp_path / "evaluator_identity.json").read_text())
    assert identity["annotation_id_mapping"] == [{"source_id": 0, "evaluator_id": 1}]
    duplicate = predictions * 2
    assert score_predictions(truth, duplicate)["AP50"] == pytest.approx(1)
    assert score_predictions(truth, duplicate)["fp"] == 1
