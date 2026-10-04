"""Bounded Kenai proposal/ROI diagnosis on a pre-recorded clip-stratified subset."""

import argparse
import gc
import gzip
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import platform_compat  # noqa: E402,F401
import torch  # noqa: E402
from torchvision.ops import box_iou  # noqa: E402
from acquire import ROOT, save_json  # noqa: E402
from data import DetectionData, clip_name  # noqa: E402
from runtime import Resources, file_sha, seed_all  # noqa: E402
from models import detector  # noqa: E402


def select_frames(images):
    clips = defaultdict(list)
    for image in images:
        clips[clip_name(image["file_name"])].append(image)
    chosen = []
    for clip in sorted(clips):
        ordered = sorted(
            clips[clip],
            key=lambda i: hashlib.sha256(("sonar-proposals-v1:" + i["file_name"]).encode()).hexdigest(),
        )
        chosen.extend(ordered[:8])
    if len(chosen) > 512:
        raise ValueError("Diagnostic subset exceeds 512 frames")
    return chosen


def proposal_coverage(proposals, truth, limit, threshold):
    if not len(truth):
        return torch.zeros(0, dtype=torch.bool)
    if not len(proposals):
        return torch.zeros(len(truth), dtype=torch.bool)
    return box_iou(proposals[:limit].float(), truth.float()).max(dim=0).values >= threshold


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/proposal_diagnosis"))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--allocation", type=Path)
    args = parser.parse_args()
    manifest = ROOT / "manifests/val.json"
    data = DetectionData(manifest, ROOT)
    selected = select_frames(data.images)
    index = {i["id"]: j for j, i in enumerate(data.images)}
    selection = {
        "rule": "8 smallest SHA256(sonar-proposals-v1:filename) per official validation clip",
        "images": selected,
        "annotation_sha256": file_sha(manifest),
        "negative_frames": sum(not data.annotations[i["id"]] for i in selected),
        "selection_uses_labels": False,
        "site": "Kenai validation",
    }
    if not selection["negative_frames"]:
        raise ValueError("Selected diagnostic frames contain no negatives")
    subset = args.out / "subset.json"
    if subset.exists():
        if json.loads(subset.read_text()) != selection:
            raise ValueError("Frozen diagnostic selection differs")
    else:
        save_json(subset, selection)
    seed_all(7)
    with Resources("bounded-proposal-diagnosis", args.out, args.allocation) as resources:
        for path in args.checkpoints:
            output = args.out / path.parent.name
            if (output / "metrics.json").exists():
                continue
            state = torch.load(path, map_location="cpu", weights_only=True)
            assert state["step"] == state["config"]["steps"] == 2000
            config = state["config"]
            size = config["detector_size"]
            data = DetectionData(manifest, ROOT, size=size)
            model = (
                detector(
                    "random", frozen=True, seed=config["seed"], size=size, neck=config.get("neck", "none")
                )
                .cuda()
                .eval()
            )
            model.load_state_dict(state["model"], strict=True)
            del state
            capture = {}
            original = model.rpn.filter_proposals

            def observe(*positional, _original=original, _capture=capture, **keywords):
                boxes, scores = _original(*positional, **keywords)
                _capture["boxes"] = [v.float().cpu() for v in boxes]
                _capture["scores"] = [v.float().cpu() for v in scores]
                return boxes, scores

            model.rpn.filter_proposals = observe
            rows, counts = [], defaultdict(int)
            elapsed = 0.0
            for first in range(0, len(selected), args.batch):
                items = [data.get(index[image["id"]]) for image in selected[first : first + args.batch]]
                images = [item[0].cuda() for item in items]
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(images)
                torch.cuda.synchronize()
                elapsed += time.perf_counter() - started
                if first == 0:
                    model.rpn.filter_proposals = original
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        plain = model(images)
                    assert all(
                        torch.equal(a[key], b[key]) for a, b in zip(outputs, plain, strict=True) for key in a
                    ), "Proposal instrumentation changed predictions"
                    model.rpn.filter_proposals = observe
                for local, (prediction, item) in enumerate(zip(outputs, items, strict=True)):
                    _, target, scales, info = item
                    truth = target["boxes"]
                    proposals, scores = capture["boxes"][local], capture["scores"][local]
                    row = {
                        "image_id": info["id"],
                        "proposals_xyxy_canvas": proposals.tolist(),
                        "objectness_scores": scores.tolist(),
                        "rounded_resize_scales": list(scales),
                        "gt_count": len(truth),
                    }
                    counts["ground_truth"] += len(truth)
                    for limit in [100, 200]:
                        for overlap in [0.5, 0.75]:
                            counts[f"covered_at_{limit}_iou_{overlap}"] += int(
                                proposal_coverage(proposals, truth, limit, overlap).sum()
                            )
                    covered = proposal_coverage(proposals, truth, 200, 0.5)
                    final_boxes = prediction["boxes"][prediction["scores"] >= 0.5].float().cpu()
                    final_covered = proposal_coverage(final_boxes, truth, len(final_boxes), 0.5)
                    counts["proposal_covered_but_no_final_coverage"] += int((covered & ~final_covered).sum())
                    counts["no_proposal_coverage"] += int((~covered).sum())
                    counts["final_box_coverage"] += int(final_covered.sum())
                    if not len(truth):
                        counts["negative_frames"] += 1
                        counts["negative_proposals"] += len(proposals)
                        counts["negative_proposals_objectness_ge_0.5"] += int((scores >= 0.5).sum())
                        counts["negative_final_detections_score_ge_0.5"] += len(final_boxes)
                    row["final_boxes_xyxy_canvas_score_ge_0.5"] = final_boxes.tolist()
                    rows.append(row)
                usage = resources.check()
                if first % 64 == 0:
                    print(
                        json.dumps({"model": path.parent.name, "frames": len(rows), **usage}),
                        flush=True,
                    )
            output.mkdir(parents=True, exist_ok=True)
            with gzip.open(output / "proposals.json.gz", "wt", encoding="utf8") as stream:
                json.dump(rows, stream)
            result = {
                "frames": len(rows),
                "counts": dict(counts),
                "proposal_recall": {
                    key: value / max(1, counts["ground_truth"])
                    for key, value in counts.items()
                    if key.startswith("covered_")
                },
                "checkpoint_sha256": file_sha(path),
                "subset_sha256": file_sha(subset),
                "instrumentation_sha256": file_sha(__file__),
                "model_inference_seconds": elapsed,
                "latency_seconds_per_frame": elapsed / len(rows),
                "latency_includes_hooks_and_cpu_proposal_copy": True,
                "first_batch_hook_predictions_bitwise_equal": True,
                "detector_size": size,
                "neck": config.get("neck", "none"),
                "geometry": f"xyxy on {size} canvas; same rounded GT scaling as production",
                "roi_failure_definition": "GT with any RPN IoU>=0.5 proposal, but no final score>=0.5 box covering it; coverage is not one-to-one AP matching",
                "status": "author diagnostic on fixed subset, not full validation AP",
            }
            resources.check()
            save_json(output / "metrics.json", result)
            print(json.dumps({"model": path.parent.name, **result}), flush=True)
            model.rpn.filter_proposals = original
            del model, original, capture, rows, observe
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
