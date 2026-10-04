import numpy as np
import pytest
import torch
from PIL import Image

from adapt import Views
from motion import (
    DynamicAuxiliary,
    SharedPairViews,
    consecutive_pairs,
    signed_difference,
    variance_covariance,
)


def test_pair_stream_rejects_missing_adjacency_and_clip_boundaries():
    meta = [{"clip_name": clip, "num_frames": 6} for clip in ["a", "b"]]
    names = ["a_0.jpg", "a_1.jpg", "a_3.jpg", "b_4.jpg", "b_5.jpg"]
    assert consecutive_pairs(names, meta) == [("a_0.jpg", "a_1.jpg"), ("b_4.jpg", "b_5.jpg")]
    assert consecutive_pairs(list(reversed(names)), meta) == consecutive_pairs(names, meta)
    with pytest.raises(ValueError):
        consecutive_pairs(["b_6.jpg"], meta)
    with pytest.raises(ValueError):
        consecutive_pairs(names + names[:1], meta)


def normalize(gray):
    return (
        gray.repeat(1, 3, 1, 1) - torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    ) / torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]


def test_signed_translation_zero_difference_and_invalid_padding():
    a = torch.zeros(1, 1, 16, 16)
    b = a.clone()
    a[:, :, 5:8, 4:7] = 1
    b[:, :, 5:8, 5:8] = 1
    actual = signed_difference(normalize(a), normalize(b))
    assert torch.allclose(actual, b - a)
    assert actual.min() < 0 and actual.max() > 0 and actual.sum() == 0
    assert not signed_difference(normalize(a), normalize(a)).any()
    valid = torch.ones_like(a, dtype=torch.bool)
    valid[:, :, :, 4] = False
    assert not signed_difference(normalize(a), normalize(b), valid)[:, :, :, 4].any()


def test_shared_geometry_photometry_and_rng_stream():
    pixels = np.arange(96 * 160, dtype=np.uint8).reshape(96, 160)
    image = Image.fromarray(pixels)
    views = Views()
    paired = SharedPairViews(views)
    torch.manual_seed(813)
    expected = views(image)
    after = torch.get_rng_state()
    torch.manual_seed(813)
    a, b = paired(image, image)
    assert torch.equal(after, torch.get_rng_state())
    for first, second, reference in zip(a, b, expected, strict=True):
        assert torch.equal(first, second) and torch.equal(first, reference)
    assert not signed_difference(a[0][None], b[0][None]).any()


def test_both_latent_branches_receive_gradients_and_constants_are_penalized():
    torch.set_num_threads(4)
    model = DynamicAuxiliary()
    a = torch.randn(4, 16, 384, requires_grad=True)
    b = torch.randn(4, 16, 384, requires_grad=True)
    difference = torch.randn(4, 1, 32, 32) * 0.1
    loss, terms = model(a, b, difference)
    loss.backward()
    assert torch.isfinite(loss) and a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0
    assert sum(p.grad.abs().sum() for p in model.difference_encoder.parameters()) > 0
    assert sum(p.grad.abs().sum() for p in model.frame_projector.parameters()) > 0
    assert all(torch.isfinite(value) for value in terms.values())
    penalty, covariance = variance_covariance(torch.zeros(4, 128))
    assert penalty > 0.9 and covariance == 0
