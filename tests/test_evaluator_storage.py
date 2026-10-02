"""Known-answer reload of every saved per-image COCO field, including negatives."""

import contextlib
import copy
import gzip
import io
import json

import numpy as np
import pytest
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import evaluate
from evaluate import score_predictions


def test_complete_compressed_evaluator_reload(tmp_path):
    truth = {
        "info": {},
        "images": [{"id": 41, "width": 100, "height": 100}, {"id": 72, "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "fish"}],
        "annotations": [
            {"id": 9, "image_id": 41, "category_id": 1, "bbox": [10, 20, 30, 40], "area": 1200, "iscrowd": 0}
        ],
    }
    predictions = [
        {"image_id": 41, "category_id": 1, "bbox": [10, 20, 30, 40], "score": 0.9},
        {"image_id": 72, "category_id": 1, "bbox": [0, 0, 20, 20], "score": 0.99},
    ]
    result = score_predictions(truth, predictions, tmp_path)
    assert result["AP50"] == pytest.approx(0.5)
    assert result["negative_false_positives"] == 1

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = copy.deepcopy(truth)
        coco.createIndex()
        expected = COCOeval(coco, coco.loadRes(copy.deepcopy(predictions)), "bbox")
        expected.params.imgIds = [41, 72]
        expected.evaluate()
        expected.accumulate()
    with gzip.open(tmp_path / "coco_eval_images.json.gz", "rt", encoding="utf8") as stream:
        saved = json.load(stream)
    reference = [
        {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in row.items()}
        if row is not None
        else None
        for row in expected.evalImgs
    ]
    assert saved == reference
    assert {row["image_id"] for row in saved if row is not None} == {41, 72}
    with np.load(tmp_path / "coco_arrays.npz") as arrays:
        for name in ["precision", "recall", "scores"]:
            assert np.array_equal(arrays[name], expected.eval[name])


def test_failed_evaluator_write_does_not_publish_metrics(tmp_path, monkeypatch):
    truth = {
        "info": {},
        "images": [{"id": 41, "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "fish"}],
        "annotations": [],
    }

    def disk_full(*args, **kwargs):
        raise OSError("Simulated full output disk")

    monkeypatch.setattr(evaluate.json, "dump", disk_full)
    with pytest.raises(OSError, match="Simulated full output disk"):
        score_predictions(truth, [], tmp_path)
    assert not (tmp_path / "metrics.json").exists()
    assert not (tmp_path / "coco_eval_images.json.gz").exists()
