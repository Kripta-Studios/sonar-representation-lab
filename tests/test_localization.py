"""Focused spatial-interface, shared-initialization and historical regressions."""

import types
import zipfile
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from localization import LocalizationBranch, install_branch
from models import detector


@pytest.mark.parametrize("kind", ["image", "capacity"])
def test_branch_shapes_gradients_and_single_frame(kind):
    torch.set_num_threads(4)
    branch = LocalizationBranch(kind)
    x = torch.zeros(1, 3, 672, 672, requires_grad=True)
    pyramid = OrderedDict(
        (str(i), torch.randn(1, 96, size, size, requires_grad=True))
        for i, size in enumerate((84, 42, 21, 10))
    )
    actual = branch(x, pyramid)
    assert [value.shape for value in actual.values()] == [value.shape for value in pyramid.values()]
    assert actual["2"] is pyramid["2"] and actual["3"] is pyramid["3"]
    assert torch.allclose(
        actual["1"] - pyramid["1"], torch.nn.functional.avg_pool2d(actual["0"] - pyramid["0"], 2), atol=1e-6
    )
    sum(value.square().mean() for value in actual.values()).backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in branch.parameters()
    )
    assert (x.grad is not None) == (kind == "image")


def test_shared_initialization_rng_and_geometry():
    torch.set_num_threads(4)
    reference = detector("random", frozen=False, seed=7, size=672)
    expected_rng = torch.get_rng_state().clone()
    expected = reference.state_dict()
    for kind in ("image", "capacity"):
        candidate = detector("random", frozen=False, seed=7, size=672, neck=kind)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        state = candidate.state_dict()
        assert all(torch.equal(value, state[name]) for name, value in expected.items())
        assert candidate.rpn.anchor_generator.sizes == reference.rpn.anchor_generator.sizes
        assert candidate.roi_heads.box_roi_pool.output_size == reference.roi_heads.box_roi_pool.output_size
        assert candidate.transform.size_divisible == reference.transform.size_divisible


def test_legacy_source_checkpoint_and_predictions_regression():
    torch.set_num_threads(4)
    archive = Path("artifacts/localization-block/historical_source.zip")
    with zipfile.ZipFile(archive) as snapshot:
        source = snapshot.read("src/models.py").decode("utf8")
    legacy = types.ModuleType("historical_models")
    legacy.__file__ = str(Path("src/models.py").resolve())
    exec(compile(source, legacy.__file__, "exec"), legacy.__dict__)
    old = legacy.detector("random", seed=7, size=448).eval()
    new = detector("random", seed=7, size=448).eval()
    assert old.state_dict().keys() == new.state_dict().keys()
    assert all(torch.equal(value, new.state_dict()[name]) for name, value in old.state_dict().items())
    torch.manual_seed(709)
    image = torch.rand(3, 448, 448)
    with torch.no_grad():
        before, after = old([image])[0], new([image])[0]
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_raw_grayscale_padding_and_branch_identity():
    branch = LocalizationBranch("image")
    captured = []
    hook = branch.layers[0].register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    x = torch.full((1, 3, 448, 448), -0.485 / 0.229)
    x[:, 0, :100, :100] = (0.75 - 0.485) / 0.229
    pyramid = OrderedDict((str(i), torch.zeros(1, 96, size, size)) for i, size in enumerate((56, 28, 14, 7)))
    branch(x, pyramid)
    hook.remove()
    assert torch.allclose(captured[0][:, :, :100, :100], torch.full((1, 1, 100, 100), 0.5))
    assert torch.allclose(captured[0][:, :, 100:, :], torch.full((1, 1, 348, 448), -1.0))
    counts = {
        kind: sum(p.numel() for p in LocalizationBranch(kind).parameters()) for kind in ("image", "capacity")
    }
    assert abs(counts["image"] - counts["capacity"]) / counts["image"] < 0.05


def test_invalid_branch_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        LocalizationBranch("other")


def test_accumulated_sampler_preserves_exact_image_stream():
    full = torch.Generator().manual_seed(7)
    small = torch.Generator().manual_seed(7)
    expected = torch.cat([torch.randint(16859, (8,), generator=full) for _ in range(2000)])
    actual = torch.cat([torch.randint(16859, (1,), generator=small) for _ in range(16000)])
    assert torch.equal(expected, actual)
    assert torch.equal(full.get_state(), small.get_state())


def test_optional_branch_does_not_seed_global_cuda_rng(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: calls.append(seed))
    model = torch.nn.Module()
    model.backbone = torch.nn.Module()
    torch.manual_seed(7)
    before = torch.get_rng_state().clone()
    calls.clear()
    install_branch(model, "image", 7)
    assert calls == []
    assert torch.equal(before, torch.get_rng_state())
