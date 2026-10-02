"""COCO AP and fixed-threshold operating-point metrics from saved predictions."""

import argparse
import contextlib
import copy
import gzip
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from acquire import save_json


def iou_xywh(a, b):
    x = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    y = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = x * y
    return inter / max(1e-12, a[2] * a[3] + b[2] * b[3] - inter)


def operating_point(gt, predictions, score=0.5):
    truth, dets = defaultdict(list), defaultdict(list)
    for a in gt["annotations"]:
        truth[a["image_id"]].append(a)
    for p in predictions:
        if p["score"] >= score:
            dets[p["image_id"]].append(p)
    tp = fp = fn = negative = negative_fp = negative_hit = 0
    for image in gt["images"]:
        anns = truth[image["id"]]
        found = set()
        pp = sorted(dets[image["id"]], key=lambda p: -p["score"])
        if not anns:
            negative += 1
            negative_fp += len(pp)
            negative_hit += bool(pp)
        for p in pp:
            matches = [
                (iou_xywh(p["bbox"], a["bbox"]), j)
                for j, a in enumerate(anns)
                if j not in found and p["category_id"] == a["category_id"]
            ]
            ov, j = max(matches, default=(0, -1))
            if ov >= 0.5:
                tp += 1
                found.add(j)
            else:
                fp += 1
        fn += len(anns) - len(found)
    return {
        "score_threshold": score,
        "iou_threshold": 0.5,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "negative_frames": negative,
        "negative_false_positives": negative_fp,
        "false_positives_per_negative_frame": negative_fp / max(negative, 1),
        "fraction_negative_frames_with_fp": negative_hit / max(negative, 1),
    }


def score_predictions(gt, predictions, out=None):
    ids = {i["id"] for i in gt["images"]}
    cats = {c["id"] for c in gt["categories"]}
    for p in predictions:
        assert p["image_id"] in ids and p["category_id"] in cats
        assert len(p["bbox"]) == 4 and p["bbox"][2] > 0 and p["bbox"][3] > 0
        assert np.isfinite(p["bbox"]).all() and 0 <= p["score"] <= 1
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        coco = COCO()
        coco.dataset = copy.deepcopy(gt)
        coco.dataset.setdefault("info", {})
        coco.createIndex()
        if predictions:
            dt = coco.loadRes(copy.deepcopy(predictions))
        else:
            dt = COCO()
            dt.dataset = {"images": gt["images"], "categories": gt["categories"], "annotations": []}
            dt.createIndex()
        ev = COCOeval(coco, dt, "bbox")
        ev.params.imgIds = sorted(ids)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    names = [
        "AP50:95",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR1",
        "AR10",
        "AR100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    result = dict(zip(names, ev.stats.tolist()))
    result.update(operating_point(gt, predictions))
    result.update(
        {
            "images_evaluated": len(ids),
            "detections": len(predictions),
            "status": "author-evaluated exploratory; not independently reviewed",
        }
    )
    if out:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "coco_output.txt").write_text(text.getvalue(), encoding="utf8")
        np.savez_compressed(
            out / "coco_arrays.npz",
            precision=ev.eval["precision"],
            recall=ev.eval["recall"],
            scores=ev.eval["scores"],
        )
        # Serialize one complete row at a time; avoid a second dataset-sized object.
        target = out / "coco_eval_images.json.gz"
        temporary = out / "coco_eval_images.json.gz.tmp"
        with gzip.open(temporary, "wt", encoding="utf8", compresslevel=1) as stream:
            stream.write("[")
            for index, row in enumerate(ev.evalImgs):
                if index:
                    stream.write(",")
                serial = (
                    {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in row.items()}
                    if row is not None
                    else None
                )
                json.dump(serial, stream, default=lambda value: value.tolist())
            stream.write("]")
        temporary.replace(target)
        save_json(
            out / "evaluator_params.json",
            {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in vars(ev.params).items()},
        )
        # Publish the completion marker only after every evaluator artifact is durable.
        save_json(out / "metrics.json", result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    print(
        json.dumps(
            score_predictions(
                json.loads(args.annotations.read_text()), json.loads(args.predictions.read_text()), args.out
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
