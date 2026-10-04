"""Conditional paired spatial continuation: B-CONTROL versus C-MOTION."""

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import platform_compat  # noqa: F401
import torch
from PIL import Image
from adapt import Views

from acquire import ROOT, save_json
from spatial import PatchObjective, masked_patch_loss, patch_masks
from motion import DynamicAuxiliary, SharedPairViews, consecutive_pairs, signed_difference
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--dynamic", action="store_true")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=8, help="paired microbatch; two frames per pair")
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
    if args.steps != 1000 or args.batch != 8 or args.accumulation != 1:
        raise ValueError("Declared pilot uses 1000 updates, eight pairs, no accumulation")
    if (args.out / "checkpoint.pt").exists() and args.resume is None:
        raise ValueError("Checkpoint already exists; explicit resume required")
    manifest = args.root / "manifests/ssl-train.json"
    image_list = json.loads(manifest.read_text())
    assert set(image_list) == {"split", "filenames"} and image_list["split"] == "official-kenai-train"
    names = image_list["filenames"]
    metadata_path = args.root / "metadata/metadata/kenai-train.json"
    metadata = json.loads(metadata_path.read_text())
    pairs = consecutive_pairs(names, metadata)
    calibration = json.loads(args.calibration.read_text())
    parent_sha = file_sha(args.parent)
    if calibration["status"] != "complete TRAIN-only numerical calibration" or calibration["seed"] != 7:
        raise ValueError("The pilot uses one fixed seed-7 TRAIN calibration")
    if args.seed not in (7, 13, 23) or (
        args.seed == 7 and calibration["parent_checkpoint_sha256"] != parent_sha
    ):
        raise ValueError("Calibration does not match the seed-7 B parent or allowed replication seeds")
    coefficient = calibration["selected_coefficient"] if args.dynamic else 0.0
    if args.profile:
        names = [name for name in names if (args.root / "images/kenai" / name).exists()]
        if not names:
            raise ValueError("No real Kenai TRAIN image is available for profiling")
        args.stop_after = args.stop_after or 4
    config = {
        "adaptation_configuration": "C-MOTION" if args.dynamic else "B-CONTROL",
        "parent_checkpoint": str(args.parent),
        "parent_checkpoint_sha256": parent_sha,
        "parent_ssl_updates": 2000,
        "calibration_sha256": file_sha(args.calibration),
        "calibration_seed": 7,
        "motion_coefficient": coefficient,
        "difference_scale": calibration["difference_scale"],
        "auxiliary_seed": args.seed + 40000,
        "auxiliary_lr": 1e-4 if args.dynamic else None,
        "paired_frames": True,
        "pairs": len(pairs),
        "train_clip_metadata_sha256": file_sha(metadata_path),
        "steps": args.steps,
        "batch": args.batch,
        "accumulation": args.accumulation,
        "seed": args.seed,
        "manifest_sha256": file_sha(manifest),
        "global_size": 224,
        "local_size": 112,
        "out_dim": 4096,
        "peak_lr": 1e-6,
        "teacher_temperature_warmup": "constant 0.07 from the completed B parent",
        "resource_profile_only": args.profile,
        "source_revision": SOURCE_REVISION,
        "published_sha256": PRETRAINED_SHA256,
        "code_identity": code_identity(),
        "protocol_sha256": file_sha("PROTOCOL.md"),
    }
    spatial = True
    if spatial:
        config["patch_objective"] = {
            "mask_ratio": 0.4,
            "masked_patches_per_image": 102,
            "patches_per_crop": 256,
            "original_B_mask_seed": args.seed + 20000,
            "mask_rng_initialization": "restored from complete B parent",
            "original_B_head_seed": args.seed + 30000,
            "head_initialization": "trained student/teacher heads restored from complete B parent",
            "mask_token_initialization": "trained B student/teacher mask tokens restored from parent",
            "out_dim": 4096,
            "coefficient": 1.0,
            "student_temperature": 0.1,
            "teacher_temperature": "constant 0.07 as CLS; distinct restored patch center",
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
    views = SharedPairViews(Views())
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
        parent = torch.load(args.parent, map_location="cpu", weights_only=True)
        if parent["step"] != 2000 or parent["config"]["adaptation_configuration"] != "B":
            raise ValueError("Pilot requires a completed B training checkpoint")
        if (
            parent["config"]["manifest_sha256"] != config["manifest_sha256"]
            or parent["config"]["seed"] != args.seed
        ):
            raise ValueError("Parent data membership/seed mismatch")
        for key, module in modules:
            module.load_state_dict(parent[key], strict=True)
        optimizer.load_state_dict(parent["optimizer"])
        restore_rng(parent["rng"], gen)
        mask_gen.set_state(parent["mask_rng"].cpu())
        del parent
        # Construction in an isolated CPU RNG context cannot perturb paired views.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed + 40000)
            dynamic = DynamicAuxiliary(difference_scale=calibration["difference_scale"]).cuda()
        dynamic.requires_grad_(args.dynamic)
        modules.append(("dynamic_auxiliary", dynamic))
        if args.dynamic:
            parameters += list(dynamic.parameters())
            optimizer.add_param_group({"params": list(dynamic.parameters()), "lr": 1e-4, "role": "dynamic"})
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
                lr = 1e-6
                for g in optimizer.param_groups:
                    g["lr"] = 1e-4 if g.get("role") == "dynamic" else lr
                temp = 0.07
                momentum = 1 - (1 - 0.996) * (1 + math.cos(math.pi * phase)) / 2
                optimizer.zero_grad(set_to_none=True)
                raw_teacher, mean_loss, entropy, diversity = [], 0.0, 0.0, 0.0
                feature_std, prototype_count = 0.0, 0.0
                patch_means, mean_patch_loss = [], 0.0
                dynamic_values = Counter()
                for _ in range(args.accumulation):
                    indices = torch.randint(len(pairs), (args.batch,), generator=gen).tolist()
                    items = []
                    for i in indices:
                        first_name, second_name = pairs[i]
                        with (
                            Image.open(args.root / "images/kenai" / first_name) as first_image,
                            Image.open(args.root / "images/kenai" / second_name) as second_image,
                        ):
                            first_views, second_views = views(first_image, second_image)
                        items.extend([first_views, second_views])
                        exposures[first_name] += 1
                        exposures[second_name] += 1
                    xx = [torch.stack([item[c] for item in items]).cuda() for c in range(4)]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        with torch.no_grad():
                            teacher_outputs = [teacher.forward_features(x) for x in xx[:2]]
                            teacher_features = [f["x_norm_clstoken"] for f in teacher_outputs]
                            tt = [th(f) for f in teacher_features]
                        student_outputs = [student.forward_features(x) for x in xx]
                        ss = [sh(output["x_norm_clstoken"]) for output in student_outputs]
                        global_loss, probs = objective(ss, tt, temp)
                        loss = global_loss
                        if args.dynamic:
                            features = student_outputs[0]["x_norm_patchtokens"]
                            difference = signed_difference(xx[0][0::2], xx[0][1::2])
                            auxiliary_loss, terms = dynamic(features[0::2], features[1::2], difference)
                            loss = loss + coefficient * auxiliary_loss
                            dynamic_values.update(
                                {key: float(value.detach()) for key, value in terms.items()}
                            )
                            dynamic_values["raw_dynamic_loss"] += float(auxiliary_loss.detach())
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite DINO objective")
                    (loss / args.accumulation).backward()
                    if spatial:
                        for image, teacher_output in zip(xx[:2], teacher_outputs, strict=True):
                            mask = patch_masks(len(image), 256, mask_gen).cuda()
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
                    mean_loss += float(global_loss.detach()) / args.accumulation
                    pp = torch.cat(probs)
                    feature_std += (
                        float(torch.cat(teacher_features).float().std(0).mean()) / args.accumulation
                    )
                    prototype_count += len(torch.unique(pp.argmax(-1))) / args.accumulation
                    entropy += float(-(pp * pp.clamp_min(1e-12).log()).sum(-1).mean()) / args.accumulation
                    mp = pp.mean(0)
                    diversity += float(-(mp * mp.clamp_min(1e-12).log()).sum()) / args.accumulation
                # B's output layers are already trained; do not refreeze them here.
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
                                "total_loss": mean_loss
                                + mean_patch_loss
                                + coefficient * dynamic_values["raw_dynamic_loss"],
                                "dynamic_coefficient": coefficient,
                                **dict(dynamic_values),
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
                        "parent_successful_updates": 2000,
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
                    "available_pairs": len(pairs),
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
