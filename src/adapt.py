"""DINO-style Kenai TRAIN-only self-supervised adaptation, configuration A."""

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import platform_compat  # noqa: F401
import torch
from PIL import Image
from torchvision import transforms as T

from acquire import ROOT, save_json
from models import (
    PRETRAINED,
    PRETRAINED_SHA256,
    SOURCE_REVISION,
    DinoObjective,
    ema_update,
    encoder,
    projection,
)
from runtime import (
    Resources,
    append_curve,
    checkpoint,
    code_identity,
    file_sha,
    restore_rng,
    rng_state,
    seed_all,
)


class Views:
    def __init__(self):
        post = [
            T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.8),
            T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.5),
            T.Grayscale(3),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
        self.global_crop = T.Compose(
            [T.RandomResizedCrop(224, scale=(0.4, 1.0), interpolation=T.InterpolationMode.BILINEAR)] + post
        )
        self.local_crop = T.Compose(
            [T.RandomResizedCrop(112, scale=(0.1, 0.4), interpolation=T.InterpolationMode.BILINEAR)] + post
        )

    def __call__(self, image):
        assert image.mode == "L"
        return [
            self.global_crop(image),
            self.global_crop(image),
            self.local_crop(image),
            self.local_crop(image),
        ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--accumulation", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--stop-after", type=int)
    p.add_argument(
        "--profile",
        action="store_true",
        help="Four-update resource probe on available TRAIN files; never export encoder",
    )
    args = p.parse_args()
    manifest = args.root / "manifests/ssl-train.json"
    image_list = json.loads(manifest.read_text())
    assert set(image_list) == {"split", "filenames"} and image_list["split"] == "official-kenai-train"
    names = image_list["filenames"]
    if args.profile:
        names = [name for name in names if (args.root / "images/kenai" / name).exists()]
        if not names:
            raise ValueError("No real Kenai TRAIN image is available for profiling")
        args.stop_after = 4
    config = {
        "adaptation_configuration": "A",
        "steps": args.steps,
        "batch": args.batch,
        "accumulation": args.accumulation,
        "seed": args.seed,
        "manifest_sha256": file_sha(manifest),
        "global_size": 224,
        "local_size": 112,
        "out_dim": 4096,
        "peak_lr": 1e-5,
        "teacher_temperature_warmup": [0.04, 0.07, 200],
        "resource_profile_only": args.profile,
        "source_revision": SOURCE_REVISION,
        "published_sha256": PRETRAINED_SHA256,
        "code_identity": code_identity(),
        "protocol_sha256": file_sha("PROTOCOL.md"),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    save_json(args.out / "config.json", config)
    seed_all(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    views = Views()
    with Resources(args.out.name, args.out) as resources:
        student = encoder().cuda().train()
        teacher = encoder().cuda().eval().requires_grad_(False)
        sh, th = projection().cuda(), projection().cuda()
        th.load_state_dict(sh.state_dict(), strict=True)
        th.requires_grad_(False).eval()
        # Do not optimize the weight-normalization scale; freeze output weights initially.
        sh.last_layer.weight_g.requires_grad_(False)
        objective = DinoObjective().cuda()
        optimizer = torch.optim.AdamW(
            [p for p in list(student.parameters()) + list(sh.parameters()) if p.requires_grad],
            lr=1e-5,
            weight_decay=0.04,
        )
        first, exposures = 0, Counter()
        if args.resume:
            state = torch.load(args.resume, map_location="cpu", weights_only=True)
            assert state["config"] == config
            for key, obj in [
                ("student", student),
                ("teacher", teacher),
                ("student_head", sh),
                ("teacher_head", th),
                ("objective", objective),
            ]:
                obj.load_state_dict(state[key], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            first = state["step"]
            exposures.update(state["exposures"])
            restore_rng(state["rng"], gen)

        def save(step):
            checkpoint(
                args.out / "checkpoint.pt",
                {
                    "student": student.state_dict(),
                    "teacher": teacher.state_dict(),
                    "student_head": sh.state_dict(),
                    "teacher_head": th.state_dict(),
                    "objective": objective.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": config,
                    "exposures": dict(exposures),
                    "rng": rng_state(gen),
                },
            )

        last_step = first
        try:
            for step in range(first, args.steps):
                started = time.monotonic()
                phase = step / max(1, args.steps - 1)
                lr = (
                    1e-5 * (step + 1) / 100
                    if step < 100
                    else 1e-6 + 9e-6 * (1 + math.cos(math.pi * (step - 100) / max(1, args.steps - 100))) / 2
                )
                for g in optimizer.param_groups:
                    g["lr"] = lr
                temp = 0.04 + 0.03 * min(1, step / 200)
                momentum = 1 - (1 - 0.996) * (1 + math.cos(math.pi * phase)) / 2
                optimizer.zero_grad(set_to_none=True)
                raw_teacher, mean_loss, entropy, diversity = [], 0.0, 0.0, 0.0
                feature_std, prototype_count = 0.0, 0.0
                for _ in range(args.accumulation):
                    indices = torch.randint(len(names), (args.batch,), generator=gen).tolist()
                    items = []
                    for i in indices:
                        with Image.open(args.root / "images/kenai" / names[i]) as im:
                            items.append(views(im))
                        exposures[names[i]] += 1
                    xx = [torch.stack([item[c] for item in items]).cuda() for c in range(4)]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        with torch.no_grad():
                            teacher_features = [
                                teacher.forward_features(x)["x_norm_clstoken"] for x in xx[:2]
                            ]
                            tt = [th(f) for f in teacher_features]
                        ss = [sh(student.forward_features(x)["x_norm_clstoken"]) for x in xx]
                        loss, probs = objective(ss, tt, temp)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite DINO objective")
                    (loss / args.accumulation).backward()
                    raw_teacher.extend([t.detach().float() for t in tt])
                    mean_loss += float(loss.detach()) / args.accumulation
                    pp = torch.cat(probs)
                    feature_std += (
                        float(torch.cat(teacher_features).float().std(0).mean()) / args.accumulation
                    )
                    prototype_count += len(torch.unique(pp.argmax(-1))) / args.accumulation
                    entropy += float(-(pp * pp.clamp_min(1e-12).log()).sum(-1).mean()) / args.accumulation
                    mp = pp.mean(0)
                    diversity += float(-(mp * mp.clamp_min(1e-12).log()).sum()) / args.accumulation
                if step < 100:
                    for parameter in sh.last_layer.parameters():
                        parameter.grad = None
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(student.parameters()) + list(sh.parameters()), 3
                )
                if not torch.isfinite(grad_norm):
                    raise ValueError("Nonfinite DINO gradients")
                optimizer.step()
                ema_update(teacher, student, momentum)
                ema_update(th, sh, momentum)
                objective.update_center(raw_teacher)
                last_step = step + 1
                if step % 25 == 0 or last_step == args.steps:
                    torch.cuda.synchronize()
                    row = {
                        "step": last_step,
                        "loss": mean_loss,
                        "teacher_entropy": entropy,
                        "batch_marginal_entropy": diversity,
                        "teacher_feature_std": feature_std,
                        "teacher_distinct_argmax_prototypes": prototype_count,
                        "grad_norm": float(grad_norm),
                        "lr": lr,
                        "teacher_temp": temp,
                        "ema_momentum": momentum,
                        "step_seconds": time.monotonic() - started,
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
            if last_step == args.steps and not args.profile:
                original = torch.load(PRETRAINED, map_location="cpu", weights_only=True)
                squared_change = sum(
                    float((v.detach().cpu().float() - original[k].float()).square().sum())
                    for k, v in teacher.state_dict().items()
                )
                squared_original = sum(float(v.float().square().sum()) for v in original.values())
                save_json(
                    args.out / "weight_change.json",
                    {
                        "teacher_l2_change": squared_change**0.5,
                        "teacher_relative_l2_change": (squared_change / squared_original) ** 0.5,
                        "published_sha256": PRETRAINED_SHA256,
                        "successful_updates": last_step,
                    },
                )
                checkpoint(
                    args.out / "encoder.pt",
                    {
                        "encoder": teacher.state_dict(),
                        "config": config,
                        "successful_updates": last_step,
                        "export": "EMA teacher encoder",
                    },
                )
            save_json(
                args.out / "exposure_summary.json",
                {
                    "available_frames": len(names),
                    "unique_images_seen": len(exposures),
                    "frame_presentations": sum(exposures.values()),
                    "successful_updates": last_step,
                },
            )
        except BaseException:
            save(last_step)
            raise


if __name__ == "__main__":
    main()
