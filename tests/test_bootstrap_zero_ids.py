"""Historical cached zero-ID rows must equal correctly remapped full scoring."""

import gzip
import json
import sys
import types
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from clip_bootstrap import compact_rows, pooled_ap, remap_clips  # noqa: E402
from evaluate import score_predictions  # noqa: E402


def test_cached_annotation_zero_recovery_against_explicit_duplicate_scoring(tmp_path):
    with zipfile.ZipFile("artifacts/localization-block/historical_source.zip") as archive:
        source = archive.read("src/evaluate.py").decode("utf8")
    legacy = types.ModuleType("legacy_evaluator")
    exec(compile(source, "legacy_evaluator", "exec"), legacy.__dict__)
    truth = {
        "images": [
            {"id": 0, "file_name": "fish_0.jpg", "width": 100, "height": 100},
            {"id": 1, "file_name": "negative_0.jpg", "width": 100, "height": 100},
        ],
        "categories": [{"id": 1, "name": "fish"}],
        "annotations": [
            {"id": 0, "image_id": 0, "category_id": 1, "bbox": [10, 20, 30, 40], "area": 1200, "iscrowd": 0}
        ],
    }
    predictions = [{"image_id": 0, "category_id": 1, "bbox": [10, 20, 30, 40], "score": 0.9}]
    old = legacy.score_predictions(truth, predictions, tmp_path)
    assert old["AP50"] == 0
    with gzip.open(tmp_path / "coco_eval_images.json.gz", "rt") as stream:
        compact = compact_rows(json.load(stream), truth["images"])
    assert compact["legacy_zero_annotation_image_ids"] == [0]
    for selected in (["fish", "negative"], ["fish", "fish", "negative"]):
        gt, detections = remap_clips(truth, predictions, selected)
        result = score_predictions(gt, detections)
        cached = pooled_ap(compact, selected)
        assert cached[0] == pytest.approx(result["AP50"], abs=1e-12)
        assert cached.mean() == pytest.approx(result["AP50:95"], abs=1e-12)
