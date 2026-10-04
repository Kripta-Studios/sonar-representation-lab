"""Conditional pilot components; no Motion training is admitted by this module.

Independent implementation of a signed-difference auxiliary with VICReg-style
variance/covariance regularization, motivated by MotionJEPA's DISReg equations.
The upstream implementation has been inspected, not incorporated. The full
action-conditioned world model and upstream SIGReg implementation are not used.
"""

from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def consecutive_pairs(filenames, metadata):
    """Only consecutive numeric indices present in the official TRAIN list."""
    if len(set(filenames)) != len(filenames):
        raise ValueError("Duplicate TRAIN filename")
    clips = defaultdict(dict)
    meta = {item["clip_name"]: item for item in metadata}
    for filename in filenames:
        clip, index = Path(filename).stem.rsplit("_", 1)
        index = int(index)
        if clip not in meta or not 0 <= index < meta[clip]["num_frames"]:
            raise ValueError("Filename is outside official clip metadata")
        if index in clips[clip]:
            raise ValueError("Duplicate numeric frame index")
        clips[clip][index] = filename
    return [
        (frames[index], frames[index + 1])
        for clip, frames in sorted(clips.items())
        for index in sorted(frames)
        if index + 1 in frames
    ]


class SharedPairViews:
    """Replay all crop/jitter/blur randomness on the adjacent image of a pair."""

    def __init__(self, views):
        self.views = views

    def __call__(self, first, second):
        if first.mode != "L" or second.mode != "L" or first.size != second.size:
            raise ValueError("Pair must share official grayscale geometry")
        before = torch.get_rng_state()
        a = self.views(first)
        after = torch.get_rng_state()
        try:
            torch.set_rng_state(before)
            b = self.views(second)
        finally:
            torch.set_rng_state(after)
        return a, b


def signed_difference(first, second, valid=None):
    """ImageNet-normalized repeated-gray tensors -> signed gray intensity delta.

    Mean cancels exactly; channel zero's .229 std restores the [0,1] image scale.
    Crop views have no padding. An explicit validity mask can zero invalid pixels.
    """
    if first.shape != second.shape or first.ndim != 4 or first.shape[1] != 3:
        raise ValueError("Expected corresponding N,3,H,W grayscale-adapter tensors")
    difference = (second[:, :1].float() - first[:, :1].float()) * 0.229
    if valid is not None:
        if valid.shape != difference.shape or valid.dtype != torch.bool:
            raise ValueError("Validity mask differs from difference-image grid")
        difference = difference * valid
    return difference


def variance_covariance(embedding):
    """Positive variance penalty at constants, plus off-diagonal covariance cost."""
    embedding = embedding.float()
    if embedding.ndim != 2 or len(embedding) < 2:
        raise ValueError("Anti-degeneracy statistics require at least two examples")
    with torch.autocast(embedding.device.type, enabled=False):
        centered = embedding - embedding.mean(0)
        covariance = centered.T @ centered / (len(embedding) - 1)
        variance = F.relu(1 - torch.sqrt(covariance.diag() + 1e-4)).mean()
        off_diagonal = covariance - torch.diag_embed(covariance.diag())
        return variance, off_diagonal.square().sum() / embedding.shape[1]


class DynamicAuxiliary(nn.Module):
    """Training-only latent branches; detector spatial outputs remain unchanged."""

    def __init__(self, dim=128, difference_scale=1.0):
        super().__init__()
        if not 0 < difference_scale <= 1000:
            raise ValueError("Difference scale must be positive and bounded")
        self.register_buffer("difference_scale", torch.tensor(float(difference_scale)))
        self.frame_projector = nn.Linear(384, dim)
        self.difference_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(2 * dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, dim),
        )

    def forward(self, first_patches, second_patches, difference):
        first = self.frame_projector(first_patches.float().mean(1))
        second = self.frame_projector(second_patches.float().mean(1))
        target = self.difference_encoder(difference * self.difference_scale)
        prediction = self.predictor(torch.cat([first, second], dim=-1))
        # Both learned branches retain gradients; neither is an EMA/stop-grad target.
        mse = F.mse_loss(prediction.float(), target.float())
        zv, zc = variance_covariance(torch.cat([first, second]))
        dv, dc = variance_covariance(target)
        raw_loss = mse + zv + dv + 0.01 * (zc + dc)
        return raw_loss, {
            "motion_mse": mse,
            "frame_variance_penalty": zv,
            "difference_variance_penalty": dv,
            "frame_covariance": zc,
            "difference_covariance": dc,
        }
