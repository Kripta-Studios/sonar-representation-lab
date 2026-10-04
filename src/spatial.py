"""Single-device masked-patch distillation grounded in pinned DINOv2/iBOT.

Original implementation, following Apache-2.0 DINOv2's indexed masked CE,
per-image mask normalization and previous-center teacher targets. Unlike its
combined masked global path, B retains A's separate unmasked CLS forwards.
"""

import torch
from torch import nn
from torch.nn import functional as F


def patch_masks(batch, patches, generator, ratio=0.4, valid=None):
    """Sample patch-only indices with a dedicated CPU RNG, never CLS/registers."""
    if not 0 < ratio < 1:
        raise ValueError("Mask ratio must be strictly between zero and one")
    if valid is None:
        valid = torch.ones(batch, patches, dtype=torch.bool)
    if valid.shape != (batch, patches) or valid.dtype != torch.bool:
        raise ValueError("Invalid patch-validity shape/type")
    masks = torch.zeros_like(valid, device="cpu")
    for row in range(batch):
        indices = torch.where(valid[row].cpu())[0]
        if len(indices) < 2:
            raise ValueError("At least two valid patches required")
        count = max(1, min(len(indices) - 1, int(len(indices) * ratio)))
        selected = indices[torch.randperm(len(indices), generator=generator)[:count]]
        masks[row, selected] = True
    return masks


class PatchObjective(nn.Module):
    def __init__(self, dim=4096):
        super().__init__()
        self.register_buffer("center", torch.zeros(1, dim))

    def forward(self, student_logits, teacher_logits, masks, temperature):
        if student_logits.shape != teacher_logits.shape or int(masks.sum()) != len(student_logits):
            raise ValueError("Patch indices/logits do not correspond")
        targets = F.softmax((teacher_logits.detach().float() - self.center) / temperature, dim=-1)
        ce = -(targets * F.log_softmax(student_logits.float() / 0.1, dim=-1)).sum(-1)
        weights = (1 / masks.sum(-1).clamp_min(1)).unsqueeze(-1).expand_as(masks)[masks]
        return (ce * weights).sum() / len(masks)

    @torch.no_grad()
    def update_center(self, per_crop_means):
        # Every production crop has 256 valid patches and exactly 102 masks/image.
        self.center.mul_(0.9).add_(torch.stack(per_crop_means).mean(0), alpha=0.1)


def masked_patch_loss(student, teacher_features, student_head, teacher_head, objective, image, mask, temp):
    """The same boolean mask selects both sides of the same, unaltered global crop."""
    with torch.no_grad():
        teacher_logits = teacher_head(teacher_features[mask])
    masked_features = student.forward_features(image, masks=mask)["x_norm_patchtokens"]
    student_logits = student_head(masked_features[mask])
    loss = objective(student_logits, teacher_logits, mask, temp)
    return loss, teacher_logits.detach().float().mean(0, keepdim=True)
