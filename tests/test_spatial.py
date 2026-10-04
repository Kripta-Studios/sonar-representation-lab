import torch
import pytest
from torch import nn
from torch.nn import functional as F

from models import DinoObjective, ema_update
from spatial import PatchObjective, masked_patch_loss, patch_masks


def test_mask_rng_is_separate_resumable_and_excludes_invalid_regions():
    torch.manual_seed(11)
    original = torch.get_rng_state().clone()
    gen = torch.Generator().manual_seed(20007)
    valid = torch.tensor([[True] * 10 + [False] * 6] * 2)
    mask = patch_masks(2, 16, gen, valid=valid)
    assert mask.sum(1).tolist() == [4, 4]
    assert not mask[:, 10:].any()
    assert torch.equal(original, torch.get_rng_state())
    state = gen.get_state()
    expected = patch_masks(2, 256, gen)
    gen.set_state(state)
    assert torch.equal(expected, patch_masks(2, 256, gen))
    assert expected.sum(1).tolist() == [102, 102]
    with pytest.raises(ValueError):
        patch_masks(1, 2, gen, valid=torch.zeros(1, 2, dtype=torch.bool))


def test_patch_loss_known_answer_teacher_stop_gradient_and_distinct_centers():
    objective, global_objective = PatchObjective(3), DinoObjective(3)
    masks = torch.tensor([[True, False, False], [False, True, True]])
    student = torch.tensor([[0.1, 0.2, 0.3], [0.3, 0.1, 0.2], [0.0, 0.0, 0.0]], requires_grad=True)
    teacher = torch.tensor([[0.3, 0.1, 0.2], [0.1, 0.1, 0.1], [0.2, 0.3, 0.1]], requires_grad=True)
    loss = objective(student, teacher, masks, 0.07)
    ce = -(F.softmax(teacher.detach() / 0.07, -1) * F.log_softmax(student / 0.1, -1)).sum(-1)
    assert torch.allclose(loss, (ce[0] + (ce[1] + ce[2]) / 2) / 2)
    loss.backward()
    assert teacher.grad is None and student.grad.abs().sum() > 0
    objective.update_center([teacher.detach().mean(0, keepdim=True)])
    assert torch.allclose(objective.center, teacher.detach().mean(0, keepdim=True) * 0.1)
    assert not global_objective.center.any()
    assert not objective.center.requires_grad


def test_corresponding_patch_selection_masked_student_and_ema():
    class FakeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.5))

        def forward_features(self, image, masks):
            self.seen = (image.clone(), masks.clone())
            return {"x_norm_patchtokens": image * self.weight}

    student = FakeEncoder()
    teacher_features = torch.arange(24).float().reshape(2, 4, 3) / 20
    crop = teacher_features + 0.1
    mask = torch.tensor([[True, False, True, False], [False, True, False, True]])
    sh, th = nn.Linear(3, 3), nn.Linear(3, 3)
    th.requires_grad_(False)
    objective = PatchObjective(3)
    loss, center = masked_patch_loss(student, teacher_features, sh, th, objective, crop, mask, 0.07)
    direct = objective(sh((crop * student.weight)[mask]), th(teacher_features[mask]), mask, 0.07)
    assert torch.equal(loss, direct)
    loss.backward()
    assert student.weight.grad.abs() > 0
    assert all(p.grad is None for p in th.parameters())
    assert torch.equal(student.seen[0], crop) and torch.equal(student.seen[1], mask)
    assert not center.requires_grad
    before = th.weight.clone()
    ema_update(th, sh, 0.9)
    assert torch.allclose(th.weight, before * 0.9 + sh.weight.detach() * 0.1)
