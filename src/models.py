"""Inspected, pinned official DINOv2 and a common spatial detector."""

import os
import hashlib
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import platform_compat  # noqa: F401
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

PROJECT = Path(__file__).resolve().parents[1]
os.environ["XFORMERS_DISABLED"] = "1"
sys.path.insert(0, str(PROJECT / "vendor/dinov2"))
from dinov2.hub.backbones import dinov2_vits14_reg  # noqa: E402
from dinov2.layers import DINOHead  # noqa: E402

PRETRAINED = PROJECT / "artifacts/pretrained/dinov2_vits14_reg4_pretrain.pth"
SOURCE_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
PRETRAINED_SHA256 = "f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb"


def encoder(kind="published", adapted=None):
    revision = subprocess.check_output(
        ["git", "-C", str(PROJECT / "vendor/dinov2"), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise ValueError("DINOv2 source revision changed")
    if kind != "random" and adapted is None:
        if hashlib.sha256(PRETRAINED.read_bytes()).hexdigest() != PRETRAINED_SHA256:
            raise ValueError("Published initialization checksum changed")
    model = dinov2_vits14_reg(pretrained=False)
    if kind != "random":
        state = torch.load(adapted or PRETRAINED, map_location="cpu", weights_only=True)
        if adapted is not None and (state.get("successful_updates", 0) < 1 or "encoder" not in state):
            raise ValueError("Adapted encoder must have completed genuine SSL updates")
        model.load_state_dict(state.get("encoder", state), strict=True)
    return model


class SpatialPyramid(nn.Module):
    out_channels = 96

    def __init__(self, backbone, frozen=True):
        super().__init__()
        self.encoder = backbone
        self.frozen = frozen
        self.encoder.requires_grad_(not frozen)
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(384, 96, 1), nn.GroupNorm(8, 96), nn.GELU(), nn.Conv2d(96, 96, 3, padding=1)
                )
                for _ in range(4)
            ]
        )

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, x):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            f = self.encoder.forward_features(x)["x_norm_patchtokens"]
            f = f.transpose(1, 2).reshape(x.shape[0], 384, x.shape[2] // 14, x.shape[3] // 14)
        return OrderedDict(
            (
                str(i),
                p(
                    F.interpolate(
                        f, size=(x.shape[2] // s, x.shape[3] // s), mode="bilinear", align_corners=False
                    )
                ),
            )
            for i, (s, p) in enumerate(zip((8, 16, 32, 64), self.projections))
        )


def detector(kind="published", adapted=None, frozen=True, seed=7):
    # Independent encoder RNG cannot alter the matched head initialization.
    torch.manual_seed(seed + 10000)
    enc = encoder(kind, adapted)
    torch.manual_seed(seed)
    backbone = SpatialPyramid(enc, frozen)
    anchors = AnchorGenerator(((16,), (32,), (64,), (128,)), ((0.5, 1.0, 2.0),) * 4)
    return FasterRCNN(
        backbone,
        num_classes=2,
        min_size=448,
        max_size=448,
        size_divisible=112,
        rpn_anchor_generator=anchors,
        box_roi_pool=MultiScaleRoIAlign(["0", "1", "2", "3"], 7, 2),
        rpn_pre_nms_top_n_train=1000,
        rpn_post_nms_top_n_train=500,
        rpn_pre_nms_top_n_test=500,
        rpn_post_nms_top_n_test=200,
        box_score_thresh=0.001,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
    )


def projection():
    return DINOHead(384, 4096, hidden_dim=1024, bottleneck_dim=256)


class DinoObjective(nn.Module):
    """Single-device DINO cross-view CE, previous center, detached teacher."""

    def __init__(self, dim=4096):
        super().__init__()
        self.register_buffer("center", torch.zeros(1, dim))

    def forward(self, students, teachers, temp):
        targets = [F.softmax((t.detach().float() - self.center) / temp, dim=-1) for t in teachers]
        terms = [
            -(t * F.log_softmax(s.float() / 0.1, dim=-1)).sum(-1).mean()
            for j, t in enumerate(targets)
            for i, s in enumerate(students)
            if i != j
        ]
        return torch.stack(terms).mean(), targets

    @torch.no_grad()
    def update_center(self, logits):
        self.center.mul_(0.9).add_(torch.cat(logits).float().mean(0, keepdim=True), alpha=0.1)


@torch.no_grad()
def ema_update(teacher, student, momentum):
    for t, s in zip(teacher.parameters(), student.parameters(), strict=True):
        t.mul_(momentum).add_(s.detach(), alpha=1 - momentum)
