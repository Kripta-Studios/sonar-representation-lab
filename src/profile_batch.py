"""Real TRAIN batch profiling and checkpoint reload verification (no stdlib name collision)."""

import argparse
import json
import time
from pathlib import Path

import platform_compat  # noqa: F401
import torch
from PIL import Image, ImageDraw

from acquire import ROOT, save_json
from data import DEFAULT_SIZE, SUPPORTED_SIZES, DetectionData, letterbox, scale_boxes, xywh_to_xyxy
from models import detector
from runtime import Resources, assert_state_equal, checkpoint, seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--size", type=int, choices=SUPPORTED_SIZES, default=DEFAULT_SIZE)
    parser.add_argument("--allocation", type=Path)
    parser.add_argument("--neck", choices=["none", "image", "capacity"], default="none")
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    seed_all(7)
    path = next(Path("artifacts/profile_train").glob("*.jpg"))
    d = json.loads((ROOT / "metadata/coco_annotations_v1.1/kenai-train.json").read_text())
    info = next(i for i in d["images"] if i["file_name"] == path.name)
    annotations = [a for a in d["annotations"] if a["image_id"] == info["id"]]
    with Image.open(path) as im:
        assert im.mode == "L" and im.size == (info["width"], info["height"])
        x, scales = letterbox(im, args.size)
        annotated = im.convert("RGB")
        draw = ImageDraw.Draw(annotated)
        for a in annotations:
            bx, by, bw, bh = a["bbox"]
            draw.rectangle((bx, by, bx + bw, by + bh), outline="red", width=3)
        geometry = (
            Path("artifacts/profile_train_geometry.png")
            if args.size == DEFAULT_SIZE
            else Path(f"artifacts/profile_train_geometry-{args.size}.png")
        )
        annotated.save(geometry)
    raw = torch.tensor([a["bbox"] for a in annotations], dtype=torch.float32).reshape(-1, 4)
    target = {
        "boxes": scale_boxes(xywh_to_xyxy(raw), scales),
        "labels": torch.ones(len(raw), dtype=torch.int64),
    }
    resolution = "" if args.size == DEFAULT_SIZE else f"-r{args.size}"
    out = args.out or Path(
        f"artifacts/profile-b{args.batch}-{'finetune' if args.finetune else 'frozen'}{resolution}"
    )
    if (out / "profile.json").exists() or (out / "optimizer_resume_check.pt").exists():
        raise ValueError("Preserve an existing profile; use a new output directory")
    resource_options = {"allocation": args.allocation} if args.allocation is not None else {}
    with Resources(out.name, out, **resource_options) as resources:
        target = {key: value.cuda() for key, value in target.items()}
        model = detector(frozen=not args.finetune, size=args.size, neck=args.neck).cuda().train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
        dataset = DetectionData(ROOT / "manifests/train-010.json", ROOT, size=args.size)
        sampler = torch.Generator().manual_seed(7)
        seen = []
        times = []
        for step in range(4):
            start = time.monotonic()
            opt.zero_grad()
            for _ in range(args.accumulation):
                indices = torch.randint(len(dataset), (args.batch,), generator=sampler).tolist()
                items = [dataset.get(index) for index in indices]
                images = [item[0].cuda() for item in items]
                targets = [{key: value.cuda() for key, value in item[1].items()} for item in items]
                seen.extend(item[3]["file_name"] for item in items)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    losses = model(images, targets)
                    loss = sum(losses.values()) / args.accumulation
                loss.backward()
                if args.accumulation > 1:
                    resources.check()
            opt.step()
            torch.cuda.synchronize()
            times.append(time.monotonic() - start)
            print(
                json.dumps(
                    {"step": step, "loss": float(loss.detach()), "seconds": times[-1], **resources.check()}
                ),
                flush=True,
            )
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            before = model([x.cuda()])[0]
        checkpoint(out / "reload_check.pt", {"model": model.state_dict()})
        model.load_state_dict(
            torch.load(out / "reload_check.pt", map_location="cpu", weights_only=True)["model"]
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            after = model([x.cuda()])[0]
        assert all(torch.equal(before[k], after[k]) for k in before)
        # Resume an actual detector optimizer/RNG transition and compare the next update.
        model.train()
        checkpoint(
            out / "optimizer_resume_check.pt",
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            },
        )

        def next_update():
            opt.zero_grad(set_to_none=True)
            total = 0
            for _ in range(args.accumulation):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    objective = (
                        sum(model([x.cuda()] * args.batch, [target] * args.batch).values())
                        / args.accumulation
                    )
                objective.backward()
                total += float(objective.detach())
                if args.accumulation > 1:
                    resources.check()
            opt.step()
            return total

        reference_loss = next_update()
        expected = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        replays = []
        for _ in range(2):
            # AdamW may alias CPU step tensors from load_state_dict. Reload the
            # immutable disk snapshot for each replay, never its mutated object.
            state = torch.load(out / "optimizer_resume_check.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(state["model"])
            opt.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["cpu_rng"])
            torch.cuda.set_rng_state_all(state["cuda_rng"])
            assert_state_equal(model.state_dict(), state["model"])
            assert_state_equal(opt.state_dict(), state["optimizer"])
            assert_state_equal(torch.get_rng_state(), state["cpu_rng"])
            assert_state_equal(torch.cuda.get_rng_state_all(), state["cuda_rng"])
            loss = next_update()
            groups = {}
            for name, value in model.state_dict().items():
                key = "encoder" if name.startswith("backbone.encoder.") else "head"
                row = groups.setdefault(key, {"squared_delta": 0.0, "squared_reference": 0.0, "max_abs": 0.0})
                delta = value.detach().cpu().double() - expected[name].double()
                row["squared_delta"] += float(delta.square().sum())
                row["squared_reference"] += float(expected[name].double().square().sum())
                row["max_abs"] = max(row["max_abs"], float(delta.abs().max()))
            save_json(
                out / f"replay-{len(replays) + 1}.json",
                {"loss": loss, "reference_loss": reference_loss, "groups": groups},
            )
            for row in groups.values():
                row["relative_l2"] = (row["squared_delta"] / max(row["squared_reference"], 1e-30)) ** 0.5
                assert row["relative_l2"] <= 1e-4 and row["max_abs"] <= 1e-3, row
            assert abs(loss - reference_loss) <= 1e-3 * max(abs(reference_loss), 1e-8)
            replays.append({"loss": loss, "groups": groups})
        for group in ("encoder", "head"):
            assert (
                replays[0]["groups"][group]["relative_l2"]
                <= 3 * replays[1]["groups"][group]["relative_l2"] + 1e-8
            )
        resources.check()
        save_json(
            out / "profile.json",
            {
                "frame": path.name,
                "split": "official Kenai TRAIN",
                "annotations": len(raw),
                "microbatch": args.batch,
                "accumulation": args.accumulation,
                "effective_batch": args.batch * args.accumulation,
                "detector_size": args.size,
                "microbatch_seconds": times,
                "profile_training_frames": seen,
                "timing_includes_input_decode_and_transfer": True,
                "checkpoint_reload_identical": True,
                "real_detector_optimizer_resume_matches_next_update": True,
                "loaded_state_exact": True,
                "bitwise_training_trajectory_claimed": False,
                "reference_loss": reference_loss,
                "numerical_replays": replays,
                "numerical_bounds": "relative L2 <=1e-4, max abs <=1e-3, loss relative <=1e-3; first replay relative error <=3*second replay+1e-8",
                "purpose": "resource/geometry/reload verification only; not a detection research result",
            },
        )


if __name__ == "__main__":
    main()
