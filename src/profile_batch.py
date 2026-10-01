"""Real TRAIN batch profiling and checkpoint reload verification (no stdlib name collision)."""

import argparse
import json
import time
from pathlib import Path

import platform_compat  # noqa: F401
import torch
from PIL import Image, ImageDraw

from acquire import ROOT, save_json
from data import letterbox, scale_boxes, xywh_to_xyxy
from models import detector
from runtime import Resources, checkpoint, seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--finetune", action="store_true")
    args = parser.parse_args()
    seed_all(7)
    path = next(Path("artifacts/profile_train").glob("*.jpg"))
    d = json.loads((ROOT / "metadata/coco_annotations_v1.1/kenai-train.json").read_text())
    info = next(i for i in d["images"] if i["file_name"] == path.name)
    annotations = [a for a in d["annotations"] if a["image_id"] == info["id"]]
    with Image.open(path) as im:
        assert im.mode == "L" and im.size == (info["width"], info["height"])
        x, scales = letterbox(im)
        annotated = im.convert("RGB")
        draw = ImageDraw.Draw(annotated)
        for a in annotations:
            bx, by, bw, bh = a["bbox"]
            draw.rectangle((bx, by, bx + bw, by + bh), outline="red", width=3)
        annotated.save("artifacts/profile_train_geometry.png")
    raw = torch.tensor([a["bbox"] for a in annotations], dtype=torch.float32).reshape(-1, 4)
    target = {
        "boxes": scale_boxes(xywh_to_xyxy(raw), scales).cuda(),
        "labels": torch.ones(len(raw), dtype=torch.int64).cuda(),
    }
    out = Path(f"artifacts/profile-b{args.batch}-{'finetune' if args.finetune else 'frozen'}")
    with Resources(out.name, out) as resources:
        model = detector(frozen=not args.finetune).cuda().train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
        times = []
        for step in range(4):
            start = time.monotonic()
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = model([x.cuda()] * args.batch, [target] * args.batch)
                loss = sum(losses.values())
            loss.backward()
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
            with torch.autocast("cuda", dtype=torch.bfloat16):
                objective = sum(model([x.cuda()] * args.batch, [target] * args.batch).values())
            objective.backward()
            opt.step()

        next_update()
        expected = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        state = torch.load(out / "optimizer_resume_check.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        next_update()
        assert all(
            torch.allclose(v.cpu(), expected[k], rtol=1e-5, atol=1e-7) for k, v in model.state_dict().items()
        )
        resources.check()
        save_json(
            out / "profile.json",
            {
                "frame": path.name,
                "split": "official Kenai TRAIN",
                "annotations": len(raw),
                "microbatch": args.batch,
                "microbatch_seconds": times,
                "checkpoint_reload_identical": True,
                "real_detector_optimizer_resume_matches_next_update": True,
                "purpose": "resource/geometry/reload verification only; not a detection research result",
            },
        )


if __name__ == "__main__":
    main()
