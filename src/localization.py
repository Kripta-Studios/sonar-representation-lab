"""One bounded fine-scale image-feature intervention and its capacity control."""

import torch
from torch import nn
from torch.nn import functional as F


class LocalizationBranch(nn.Module):
    """Comparable three-convolution branches; shared detector geometry stays intact."""

    def __init__(self, kind):
        super().__init__()
        if kind not in ("image", "capacity"):
            raise ValueError(f"Unknown localization branch: {kind}")
        self.kind = kind
        channels = (1, 32, 64, 96) if kind == "image" else (96, 32, 64, 96)
        blocks = []
        for index in range(3):
            kernel = 1 if kind == "capacity" and index == 0 else 3
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels[index],
                        channels[index + 1],
                        kernel,
                        stride=2 if kind == "image" else 1,
                        padding=kernel // 2,
                    ),
                    nn.GroupNorm(8, channels[index + 1]),
                    nn.GELU(),
                )
            )
        self.layers = nn.Sequential(*blocks)

    def forward(self, x, pyramid):
        # x is the detector's ImageNet-normalized, repeated grayscale input.
        # Undo channel-zero normalization; top-left letterbox padding remains raw zero.
        source = 2 * (x[:, :1] * 0.229 + 0.485) - 1 if self.kind == "image" else pyramid["0"]
        extra = self.layers(source)
        assert extra.shape[-2:] == pyramid["0"].shape[-2:]
        result = pyramid.copy()
        result["0"] = pyramid["0"] + 0.1 * extra
        result["1"] = pyramid["1"] + 0.1 * F.avg_pool2d(extra, 2)
        return result


def install_branch(model, kind, seed):
    if kind == "none":
        return model
    # Construct after the common backbone/head, without consuming their RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed + 50000).get_state())
        model.backbone.localization_branch = LocalizationBranch(kind)
    return model
