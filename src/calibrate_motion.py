"""One TRAIN-only numerical calibration; no optimizer updates or validation input."""

import argparse
import json
import math
import statistics
from pathlib import Path

import platform_compat  # noqa: F401
import torch
from PIL import Image

from acquire import ROOT, save_json
from adapt import Views
from models import DinoObjective, encoder, projection
from motion import DynamicAuxiliary, SharedPairViews, consecutive_pairs, signed_difference
from runtime import Resources, code_identity, file_sha, restore_rng, seed_all
from spatial import PatchObjective, masked_patch_loss, patch_masks


def gradient_norm(loss, parameters, retain_graph=False):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    result = math.sqrt(
        sum(float(value.detach().float().square().sum()) for value in gradients if value is not None)
    )
    if not math.isfinite(result) or result <= 1e-12:
        raise ValueError("Nonfinite or degenerate encoder-gradient calibration")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/motion-calibration"))
    args = parser.parse_args()
    if (args.out / "calibration.json").exists():
        raise ValueError("The single numerical calibration already exists; do not tune another")
    manifest = ROOT / "manifests/ssl-train.json"
    data = json.loads(manifest.read_text())
    assert set(data) == {"split", "filenames"} and data["split"] == "official-kenai-train"
    pairs = consecutive_pairs(
        data["filenames"], json.loads((ROOT / "metadata/metadata/kenai-train.json").read_text())
    )
    parent = torch.load(args.parent, map_location="cpu", weights_only=True)
    assert parent["step"] == 2000 and parent["config"]["adaptation_configuration"] == "B"
    assert parent["config"]["seed"] == 7 and parent["config"]["manifest_sha256"] == file_sha(manifest)
    seed_all(7)
    with Resources(args.out.name, args.out) as resources:
        student, teacher = (
            encoder("random").cuda().train(),
            encoder("random").cuda().eval().requires_grad_(False),
        )
        sh, th, sph, tph = [projection().cuda() for _ in range(4)]
        th.requires_grad_(False).eval()
        tph.requires_grad_(False).eval()
        sh.last_layer.weight_g.requires_grad_(False)
        sph.last_layer.weight_g.requires_grad_(False)
        global_objective, patch_objective = DinoObjective().cuda(), PatchObjective().cuda()
        for key, module in [
            ("student", student),
            ("teacher", teacher),
            ("student_head", sh),
            ("teacher_head", th),
            ("student_patch_head", sph),
            ("teacher_patch_head", tph),
            ("objective", global_objective),
            ("patch_objective", patch_objective),
        ]:
            module.load_state_dict(parent[key], strict=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(40007)
            dynamic = DynamicAuxiliary().cuda()
        generator = torch.Generator()
        mask_generator = torch.Generator()
        restore_rng(parent["rng"], generator)
        mask_generator.set_state(parent["mask_rng"].cpu())
        del parent
        views = SharedPairViews(Views())
        batches, names = [], []
        squared, pixels = 0.0, 0
        for _ in range(4):
            items, batch_names = [], []
            for index in torch.randint(len(pairs), (8,), generator=generator).tolist():
                a, b = pairs[index]
                with (
                    Image.open(ROOT / "images/kenai" / a) as first,
                    Image.open(ROOT / "images/kenai" / b) as second,
                ):
                    av, bv = views(first, second)
                items.extend([av, bv])
                batch_names.append([a, b])
            batch = [torch.stack([item[crop] for item in items]) for crop in range(4)]
            delta = signed_difference(batch[0][0::2], batch[0][1::2])
            squared += float(delta.square().sum())
            pixels += delta.numel()
            batches.append(batch)
            names.append(batch_names)
        rms = math.sqrt(squared / pixels)
        if not math.isfinite(rms) or rms <= 1e-6:
            raise ValueError("Real paired TRAIN stream has insufficient numerical change")
        scale = 1 / max(rms, 0.001)
        dynamic.difference_scale.fill_(scale)
        rows = []
        for index, batch in enumerate(batches):
            images = [image.cuda() for image in batch]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    teacher_output = [teacher.forward_features(image) for image in images[:2]]
                    targets = [th(output["x_norm_clstoken"]) for output in teacher_output]
                output = [student.forward_features(image) for image in images]
                predictions = [sh(features["x_norm_clstoken"]) for features in output]
                global_loss, _ = global_objective(predictions, targets, 0.07)
                patch_terms = []
                for image, target in zip(images[:2], teacher_output, strict=True):
                    mask = patch_masks(16, 256, mask_generator).cuda()
                    term, _ = masked_patch_loss(
                        student, target["x_norm_patchtokens"], sph, tph, patch_objective, image, mask, 0.07
                    )
                    patch_terms.append(term)
                spatial_loss = global_loss + torch.stack(patch_terms).mean()
                features = output[0]["x_norm_patchtokens"]
                raw, terms = dynamic(
                    features[0::2], features[1::2], signed_difference(images[0][0::2], images[0][1::2])
                )
            parameters = list(student.parameters())
            spatial_norm = gradient_norm(spatial_loss, parameters, retain_graph=True)
            difference_norm = gradient_norm(
                raw, list(dynamic.difference_encoder.parameters()), retain_graph=True
            )
            frame_projection_norm = gradient_norm(
                raw, list(dynamic.frame_projector.parameters()), retain_graph=True
            )
            dynamic_norm = gradient_norm(raw, parameters)
            rows.append(
                {
                    "batch": index,
                    "spatial_encoder_gradient_norm": spatial_norm,
                    "dynamic_encoder_gradient_norm": dynamic_norm,
                    "difference_branch_gradient_norm": difference_norm,
                    "frame_projector_gradient_norm": frame_projection_norm,
                    "unclamped_coefficient": 0.1 * spatial_norm / dynamic_norm,
                    "spatial_loss": float(spatial_loss.detach()),
                    "raw_dynamic_loss": float(raw.detach()),
                    **{key: float(value.detach()) for key, value in terms.items()},
                }
            )
            print(json.dumps({**rows[-1], **resources.check()}), flush=True)
            del spatial_loss, raw, terms, patch_terms, output, predictions, features, global_loss
        coefficient = min(1.0, max(0.001, statistics.median(row["unclamped_coefficient"] for row in rows)))
        ratios = [
            coefficient * row["dynamic_encoder_gradient_norm"] / row["spatial_encoder_gradient_norm"]
            for row in rows
        ]
        if not 0.01 <= statistics.median(ratios) <= 1:
            raise ValueError(
                "Bounded coefficient cannot produce admissible TRAIN gradient scale; pilot not admitted"
            )
        record = {
            "status": "complete TRAIN-only numerical calibration",
            "source_identity": code_identity(),
            "protocol_sha256": file_sha("PROTOCOL.md"),
            "seed": 7,
            "parent_checkpoint_sha256": file_sha(args.parent),
            "manifest_sha256": file_sha(manifest),
            "difference_rms": rms,
            "difference_scale": scale,
            "selected_coefficient": coefficient,
            "rule": "scale=1/max(TRAIN signed-delta RMS,0.001); coefficient=clamp(median(0.1*spatial encoder-gradient norm/raw dynamic encoder-gradient norm),0.001,1)",
            "numerical_admission": "median weighted encoder-gradient ratio must lie in [0.01,1]; otherwise omit pilot without a coefficient sweep",
            "weighted_gradient_ratios": ratios,
            "batches": rows,
            "paired_frames": names,
            "optimizer_updates": 0,
            "frame_presentations": 64,
            "dimension": 128,
            "reduction": "mean MSE + frame/difference variance hinges + 0.01 times both off-diagonal covariance penalties",
            "variance_covariance_dtype": "float32; autocast disabled for covariance",
            "static_and_difference_branches_receive_gradients": True,
            "inference": "auxiliary is training-only; one frame remains sufficient",
        }
        save_json(args.out / "calibration.json", record)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
