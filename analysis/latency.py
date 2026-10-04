"""Bounded matched inference latency on fixed real TRAIN frames, without fitting."""

import argparse
import gc
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import platform_compat  # noqa: E402,F401
import torch  # noqa: E402
from acquire import ROOT, save_json  # noqa: E402
from data import DetectionData  # noqa: E402
from models import detector  # noqa: E402
from runtime import Resources, file_sha, seed_all  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batches", nargs="+", type=int, choices=(1, 4, 8), default=[1, 8])
    args = parser.parse_args()
    if (args.out / "latency.json").exists():
        raise ValueError("Preserve the recorded latency probe")
    manifest = ROOT / "manifests/train-010.json"
    records = json.loads(manifest.read_text())["images"]
    chosen = sorted(
        records,
        key=lambda item: hashlib.sha256(("sonar-latency-v1:" + item["file_name"]).encode()).hexdigest(),
    )[:8]
    selected = [item["id"] for item in chosen]
    seed_all(7)
    result = {
        "frames": [item["file_name"] for item in chosen],
        "split": "Kenai TRAIN only",
        "selection": "eight lowest filename SHA256 with fixed sonar-latency-v1 prefix",
        "manifest_sha256": file_sha(manifest),
        "script_sha256": file_sha(__file__),
        "timing_scope": "warm model-only wall/stream elapsed time; inputs already on GPU; excludes decode, loading and COCO scoring",
        "models": [],
    }
    with Resources(args.out.name, args.out) as resources:
        for directory in args.runs:
            path = directory / "checkpoint.pt"
            state = torch.load(path, map_location="cpu", weights_only=True)
            config = state["config"]
            assert state["step"] == config["steps"] == 2000
            data = DetectionData(manifest, ROOT, size=config["detector_size"])
            lookup = {item["id"]: index for index, item in enumerate(data.images)}
            images = [data.get(lookup[image_id])[0].cuda() for image_id in selected]
            model = (
                detector(
                    "random",
                    seed=config["seed"],
                    size=config["detector_size"],
                    neck=config.get("neck", "none"),
                )
                .cuda()
                .eval()
            )
            model.load_state_dict(state["model"], strict=True)
            del state
            measurements = []
            for batch_size in args.batches:
                wall, stream, counts = [], [], []
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for iteration in range(14):
                        batch = [
                            images[(iteration * batch_size + offset) % len(images)]
                            for offset in range(batch_size)
                        ]
                        start, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        start.record()
                        predictions = model(batch)
                        end.record()
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter() - started) * 1000
                        resources.check()
                        if iteration >= 4:
                            wall.append(elapsed)
                            stream.append(start.elapsed_time(end))
                            counts.append(sum(len(prediction["scores"]) for prediction in predictions))
                measurements.append(
                    {
                        "batch": batch_size,
                        "warmups": 4,
                        "measured_batches": 10,
                        "wall_ms": wall,
                        "cuda_stream_ms": stream,
                        "detections": counts,
                        "wall_median_ms_per_frame": statistics.median(wall) / batch_size,
                    }
                )
            result["models"].append(
                {
                    "run": directory.name,
                    "checkpoint_sha256": file_sha(path),
                    "size": config["detector_size"],
                    "measurements": measurements,
                }
            )
            resources.check()
            del model, images, predictions, batch, data
            gc.collect()
            torch.cuda.empty_cache()
        save_json(args.out / "latency.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
