"""DINO-style image-only Kenai TRAIN adaptation: historical A and spatial B."""

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
from spatial import PatchObjective, masked_patch_loss, patch_masks
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
    assert_state_equal,
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
    p.add_argument("--configuration", choices=["A", "B"], default="A")
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
    if (args.out / "checkpoint.pt").exists() and args.resume is None:
        raise ValueError("Checkpoint already exists; explicit resume required")
    manifest = args.root / "manifests/ssl-train.json"
    image_list = json.loads(manifest.read_text())
    assert set(image_list) == {"split", "filenames"} and image_list["split"] == "official-kenai-train"
    names = image_list["filenames"]
    if args.profile:
        names = [name for name in names if (args.root / "images/kenai" / name).exists()]
        if not names:
            raise ValueError("No real Kenai TRAIN image is available for profiling")
        args.stop_after = args.stop_after or 4
    config = {
        "adaptation_configuration": args.configuration,
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
    spatial = args.configuration == "B"
    if spatial:
        config["patch_objective"] = {
            "mask_ratio": 0.4,
            "masked_patches_per_image": 102,
            "patches_per_crop": 256,
            "mask_seed": args.seed + 20000,
            "head_seed": args.seed + 30000,
            "mask_token_initialization": "published checkpoint mask_token",
            "out_dim": 4096,
            "coefficient": 1.0,
            "student_temperature": 0.1,
            "teacher_temperature": "same scheduled temperature as CLS; distinct center",
            "center_momentum": 0.9,
            "reduction": "masked-patch mean per image, then mean over two global crops and effective batch",
            "padding": "RandomResizedCrop has no padded region; every 224/14 patch is valid",
            "correspondence": "same crop tensor, same row-major patch mask on student and unmasked teacher",
            "global_path": "unchanged unmasked CLS; extra masked student forward",
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
        modules = [
            ("student", student),
            ("teacher", teacher),
            ("student_head", sh),
            ("teacher_head", th),
            ("objective", objective),
        ]
        parameters = list(student.parameters()) + list(sh.parameters())
        mask_gen = torch.Generator().manual_seed(args.seed + 20000)
        if spatial:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(args.seed + 30000)
                sph, tph = projection().cuda(), projection().cuda()
            tph.load_state_dict(sph.state_dict(), strict=True)
            tph.requires_grad_(False).eval()
            sph.last_layer.weight_g.requires_grad_(False)
            patch_objective = PatchObjective().cuda()
            modules.extend(
                [
                    ("student_patch_head", sph),
                    ("teacher_patch_head", tph),
                    ("patch_objective", patch_objective),
                ]
            )
            parameters += list(sph.parameters())
        optimizer = torch.optim.AdamW(
            [p for p in parameters if p.requires_grad],
            lr=1e-5,
            weight_decay=0.04,
        )
        first, exposures = 0, Counter()
        if args.resume:
            state = torch.load(args.resume, map_location="cpu", weights_only=True)
            assert state["config"] == config
            for key, obj in modules:
                obj.load_state_dict(state[key], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            first = state["step"]
            exposures.update(state["exposures"])
            restore_rng(state["rng"], gen)
            if spatial:
                mask_gen.set_state(state["mask_rng"].cpu())
            if args.profile:
                for key, obj in modules:
                    assert_state_equal(obj.state_dict(), state[key], key)
                assert_state_equal(optimizer.state_dict(), state["optimizer"], "optimizer")
                assert_state_equal(rng_state(gen), state["rng"], "rng")
                if spatial:
                    assert_state_equal(mask_gen.get_state(), state["mask_rng"], "mask_rng")
                save_json(
                    args.out / f"exact_reload_step{first}.json",
                    {"step": first, "all_loaded_states_exact": True},
                )
            del state

        def save(step):
            checkpoint(
                args.out / "checkpoint.pt",
                {
                    **{key: obj.state_dict() for key, obj in modules},
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": config,
                    "exposures": dict(exposures),
                    "rng": rng_state(gen),
                    **({"mask_rng": mask_gen.get_state()} if spatial else {}),
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
                patch_means, mean_patch_loss = [], 0.0
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
                            teacher_outputs = [teacher.forward_features(x) for x in xx[:2]]
                            teacher_features = [f["x_norm_clstoken"] for f in teacher_outputs]
                            tt = [th(f) for f in teacher_features]
                        ss = [sh(student.forward_features(x)["x_norm_clstoken"]) for x in xx]
                        loss, probs = objective(ss, tt, temp)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite DINO objective")
                    (loss / args.accumulation).backward()
                    if spatial:
                        for image, teacher_output in zip(xx[:2], teacher_outputs, strict=True):
                            mask = patch_masks(args.batch, 256, mask_gen).cuda()
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                patch_loss, patch_mean = masked_patch_loss(
                                    student,
                                    teacher_output["x_norm_patchtokens"],
                                    sph,
                                    tph,
                                    patch_objective,
                                    image,
                                    mask,
                                    temp,
                                )
                            if not torch.isfinite(patch_loss):
                                raise ValueError("Nonfinite masked-patch loss")
                            (patch_loss / (2 * args.accumulation)).backward()
                            mean_patch_loss += float(patch_loss.detach()) / (2 * args.accumulation)
                            patch_means.append(patch_mean)
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
                    if spatial:
                        for parameter in sph.last_layer.parameters():
                            parameter.grad = None
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 3)
                if not torch.isfinite(grad_norm):
                    raise ValueError("Nonfinite DINO gradients")
                optimizer.step()
                ema_update(teacher, student, momentum)
                ema_update(th, sh, momentum)
                objective.update_center(raw_teacher)
                if spatial:
                    ema_update(tph, sph, momentum)
                    patch_objective.update_center(patch_means)
                last_step = step + 1
                if args.profile or step % 25 == 0 or last_step == args.steps:
                    torch.cuda.synchronize()
                    row = {
                        "step": last_step,
                        "loss": mean_loss,
                        **(
                            {
                                "global_loss": mean_loss,
                                "patch_loss": mean_patch_loss,
                                "total_loss": mean_loss + mean_patch_loss,
                            }
                            if spatial
                            else {}
                        ),
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
