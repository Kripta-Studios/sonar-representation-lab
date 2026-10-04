"""Train, profile, resume and predict with the common detector."""

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import platform_compat  # noqa: F401
import torch

from acquire import ROOT, save_json
from data import DEFAULT_SIZE, SUPPORTED_SIZES, DetectionData, undo_boxes
from evaluate import score_predictions
from models import PRETRAINED_SHA256, SOURCE_REVISION, detector
from runtime import (
    Resources,
    append_curve,
    assert_state_equal,
    checkpoint,
    code_identity,
    file_sha,
    restore_rng,
    rng_state,
    seed_all,
)


def schedule(step, total, base):
    if step < 100:
        return base * (step + 1) / 100
    phase = min(1, (step - 100) / max(1, total - 100))
    return base * (0.1 + 0.9 * (1 + math.cos(math.pi * phase)) / 2)


def batch(data, indices):
    items = [data.get(int(i)) for i in indices]
    return [i[0].cuda() for i in items], [{k: v.cuda() for k, v in i[1].items()} for i in items]


def train(args):
    if (args.out / "checkpoint.pt").exists() and args.resume is None:
        raise ValueError("Checkpoint already exists; explicit resume required")
    manifest = args.root / "manifests" / f"train-{args.fraction:03d}.json"
    data = DetectionData(manifest, args.root, size=args.size)
    config = {
        "kind": args.kind,
        "frozen": args.kind != "finetune",
        "seed": args.seed,
        "fraction": args.fraction,
        "steps": args.steps,
        "batch": args.batch,
        "accumulation": args.accumulation,
        "adapted": str(args.adapted) if args.adapted else None,
        "adapted_sha256": file_sha(args.adapted) if args.adapted else None,
        "published_sha256": PRETRAINED_SHA256,
        "manifest_sha256": file_sha(manifest),
        "source_revision": SOURCE_REVISION,
        "detector_size": args.size,
        "neck": getattr(args, "neck", "none"),
        "detector_cache": data.cache_identity,
        "head_lr": 3e-4,
        "backbone_lr": 1e-5,
        "code_identity": code_identity(),
        "protocol_sha256": file_sha("PROTOCOL.md"),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    save_json(args.out / "config.json", config)
    seed_all(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    resource_options = {"allocation": args.allocation} if args.allocation is not None else {}
    with Resources(args.out.name, args.out, **resource_options) as resources:
        model = (
            detector(
                "published" if args.kind == "finetune" else args.kind,
                args.adapted,
                args.kind != "finetune",
                args.seed,
                size=args.size,
                neck=getattr(args, "neck", "none"),
            )
            .cuda()
            .train()
        )
        enc = list(model.backbone.encoder.parameters())
        expected_cuda_rng = torch.Generator(device="cuda").manual_seed(args.seed).get_state()
        assert_state_equal(torch.cuda.get_rng_state(), expected_cuda_rng)
        save_json(
            args.out / "initial_rng_identity.json",
            {
                "cuda_matches_common_head_seed": args.seed,
                "branch_does_not_reset_cuda_rng": True,
                "branch_seed": args.seed + 50000 if getattr(args, "neck", "none") != "none" else None,
            },
        )
        enc_ids = {id(p) for p in enc}
        head = [p for p in model.parameters() if id(p) not in enc_ids]
        save_json(
            args.out / "parameter_summary.json",
            {
                "total": sum(p.numel() for p in model.parameters()),
                "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "encoder": sum(p.numel() for p in enc),
                "detector_and_neck": sum(p.numel() for p in head),
                "new_branch": sum(
                    p.numel()
                    for p in getattr(model.backbone, "localization_branch", torch.nn.Identity()).parameters()
                ),
                "neck": getattr(args, "neck", "none"),
            },
        )
        groups = [{"params": head, "lr": 3e-4, "base_lr": 3e-4}]
        if args.kind == "finetune":
            groups.append({"params": enc, "lr": 1e-5, "base_lr": 1e-5})
        optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
        first, exposures = 0, Counter()
        if args.resume:
            state = torch.load(args.resume, map_location="cpu", weights_only=True)
            assert state["config"] == config, "Resume config mismatch"
            model.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            first = state["step"]
            exposures.update(state["exposures"])
            restore_rng(state["rng"], gen)
            assert_state_equal(model.state_dict(), state["model"])
            assert_state_equal(optimizer.state_dict(), state["optimizer"])
            assert_state_equal(rng_state(gen), state["rng"])
            save_json(
                args.out / f"resume_verification_step{first}.json",
                {
                    "step": first,
                    "model_optimizer_rng_exact_on_reload": True,
                    "presentations": sum(exposures.values()),
                    "checkpoint_sha256": file_sha(args.resume),
                },
            )
            del state

        def save(step):
            checkpoint(
                args.out / "checkpoint.pt",
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": config,
                    "rng": rng_state(gen),
                    "exposures": dict(exposures),
                },
            )

        last_step = first
        recovery_rng = None
        pending_exposures = Counter()
        attempted_step = first
        try:
            for step in range(first, args.steps):
                attempted_step = step
                recovery_rng = rng_state(gen)
                pending_exposures = Counter()
                t = time.monotonic()
                for group in optimizer.param_groups:
                    group["lr"] = schedule(step, args.steps, group["base_lr"])
                optimizer.zero_grad(set_to_none=True)
                losses = Counter()
                for _ in range(args.accumulation):
                    indices = torch.randint(len(data), (args.batch,), generator=gen)
                    images, targets = batch(data, indices)
                    for i in indices.tolist():
                        pending_exposures[data.images[i]["id"]] += 1
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss_parts = model(images, targets)
                        loss = sum(loss_parts.values()) / args.accumulation
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite detector loss")
                    loss.backward()
                    if args.accumulation > 1:
                        resources.check()
                    losses.update({k: float(v.detach()) / args.accumulation for k, v in loss_parts.items()})
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 3
                )
                if not torch.isfinite(grad_norm):
                    raise ValueError("Nonfinite detector gradients")
                optimizer.step()
                exposures.update(pending_exposures)
                last_step = step + 1
                if step % 25 == 0 or last_step == args.steps:
                    torch.cuda.synchronize()
                    row = {
                        "step": last_step,
                        "losses": dict(losses),
                        "head_lr": optimizer.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm),
                        "step_seconds": time.monotonic() - t,
                        **resources.check(),
                    }
                    append_curve(args.out / "curve.jsonl", row)
                    print(json.dumps(row), flush=True)
                if (
                    last_step % 200 == 0
                    or last_step == args.steps
                    or (args.stop_after and last_step >= args.stop_after)
                ):
                    save(last_step)
                if args.stop_after and last_step >= args.stop_after:
                    break
            save_json(
                args.out / "exposure_summary.json",
                {
                    "available": data.data.get("counts"),
                    "unique_images_seen": len(exposures),
                    "frame_presentations": sum(exposures.values()),
                    "successful_updates": last_step,
                },
            )
        except BaseException:
            if recovery_rng is not None and last_step == attempted_step:
                restore_rng(recovery_rng, gen)
                append_curve(
                    args.out / "discarded_updates.jsonl",
                    {
                        "attempted_update": attempted_step + 1,
                        "discarded_presentations": sum(pending_exposures.values()),
                        "rng_restored_to_last_successful_update": True,
                    },
                )
            save(last_step)
            raise


@torch.no_grad()
def predict(args):
    if (args.out / "predictions.json").exists() or (args.out / "metrics.json").exists():
        raise ValueError("Prediction output already exists; preserve it and use a new replay directory")
    seed_all(7)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    if state["step"] != config["steps"]:
        raise ValueError("Incomplete detector checkpoint cannot be final-evaluated")
    checkpoint_size = config.get("detector_size", DEFAULT_SIZE)
    if checkpoint_size not in SUPPORTED_SIZES:
        raise ValueError(f"Unsupported checkpoint detector size: {checkpoint_size}")
    if args.size is not None and args.size != checkpoint_size:
        raise ValueError(
            f"Requested detector size {args.size} does not match checkpoint size {checkpoint_size}"
        )
    size = checkpoint_size
    if args.split == "channel":
        frozen = json.loads(Path("artifacts/freeze.json").read_text())
        roster = {m["checkpoint_sha256"] for m in frozen["models"]}
        if file_sha(args.checkpoint) not in roster or code_identity() != frozen["code_identity"]:
            raise ValueError("Checkpoint or implementation changed after freeze")
        if file_sha("PROTOCOL.md") != frozen["protocol_sha256"]:
            raise ValueError("Protocol changed after freeze")
        if (args.out / "exposure_started.json").exists():
            raise ValueError("This model has already been exposed to Channel; do not silently repeat")
        save_json(
            args.out / "exposure_started.json",
            {
                "checkpoint_sha256": file_sha(args.checkpoint),
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "status": "Channel image inference starting; preserve any defect/exposure",
            },
        )
    path = (
        args.root / "manifests/val.json"
        if args.split == "val"
        else args.root / "metadata/coco_annotations_v1.1/kenai-channel.json"
    )
    data = DetectionData(
        path,
        args.root,
        "kenai" if args.split == "val" else "channel",
        size=size,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    resource_options = {"allocation": args.allocation} if args.allocation is not None else {}
    with Resources(args.out.name, args.out, **resource_options) as resources:
        model = (
            detector("random", frozen=True, seed=config["seed"], size=size, neck=config.get("neck", "none"))
            .cuda()
            .eval()
        )
        model.load_state_dict(state["model"], strict=True)
        del state
        predictions = []
        for first in range(0, len(data), args.batch):
            items = [data.get(i) for i in range(first, min(first + args.batch, len(data)))]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model([i[0].cuda() for i in items])
            for output, (_, _, scales, info) in zip(outputs, items):
                boxes = undo_boxes(output["boxes"].float().cpu(), scales, info["width"], info["height"])
                for box, score, label in zip(
                    boxes.tolist(), output["scores"].float().cpu().tolist(), output["labels"].cpu().tolist()
                ):
                    x, y, x2, y2 = box
                    if x2 > x and y2 > y:
                        assert label == 1
                        predictions.append(
                            {
                                "image_id": info["id"],
                                "category_id": label,
                                "bbox": [x, y, x2 - x, y2 - y],
                                "score": score,
                            }
                        )
            if first % 500 == 0:
                print(
                    json.dumps({"frames": first + len(items), "total": len(data), **resources.check()}),
                    flush=True,
                )
        save_json(args.out / "predictions.json", predictions)
        save_json(
            args.out / "prediction_identity.json",
            {
                "checkpoint_sha256": file_sha(args.checkpoint),
                "annotation_sha256": file_sha(path),
                "images": len(data),
                "split": args.split,
                "preprocessing": (
                    f"{size} aspect-preserving grayscale top-left letterbox; ImageNet normalization"
                ),
                "detector_size": size,
                "score_threshold": 0.001,
                "nms": 0.5,
                "max_detections": 100,
                "detector_cache": data.cache_identity,
            },
        )
        # Evaluation is CPU-only after prediction; resource charge remains conservative.
        result = score_predictions(data.data, predictions, args.out)
        # Capture native CPU evaluator RAM peaks and enforce the same resource limits.
        resources.check()
        print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--kind", choices=["published", "adapted", "random", "finetune"], default="published")
    tr.add_argument("--fraction", type=int, choices=[1, 10, 100], default=10)
    tr.add_argument("--seed", type=int, default=7)
    tr.add_argument("--steps", type=int, default=2000)
    tr.add_argument("--accumulation", type=int, default=1)
    tr.add_argument("--adapted", type=Path)
    tr.add_argument("--resume", type=Path)
    tr.add_argument("--stop-after", type=int)
    tr.add_argument("--size", type=int, choices=SUPPORTED_SIZES, default=DEFAULT_SIZE)
    tr.add_argument("--neck", choices=["none", "image", "capacity"], default="none")
    pr = sub.add_parser("predict")
    pr.add_argument("--checkpoint", type=Path, required=True)
    pr.add_argument("--split", choices=["val", "channel"], default="val")
    pr.add_argument("--size", type=int, choices=SUPPORTED_SIZES)
    for command in (tr, pr):
        command.add_argument("--root", type=Path, default=ROOT)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--batch", type=int, default=8)
        command.add_argument("--allocation", type=Path)
    args = p.parse_args()
    if args.command == "train":
        if args.kind == "adapted" and args.adapted is None:
            p.error("Adapted comparison requires an actually adapted encoder")
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
