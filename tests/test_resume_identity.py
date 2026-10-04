import pytest
import torch

from runtime import assert_state_equal


def test_exact_reload_identity_rejects_corrupt_optimizer_and_rng_values():
    state = {"optimizer": {"moment": torch.tensor([0.1, 0.2])}, "rng": [torch.tensor([4, 7])], "step": 2}
    copied = {"optimizer": {"moment": torch.tensor([0.1, 0.2])}, "rng": [torch.tensor([4, 7])], "step": 2}
    assert_state_equal(state, copied)
    copied["optimizer"]["moment"][0] += 1e-6
    with pytest.raises(AssertionError, match="moment"):
        assert_state_equal(state, copied)
    copied["optimizer"]["moment"] = state["optimizer"]["moment"].clone()
    copied["rng"][0][0] += 1
    with pytest.raises(AssertionError, match="rng"):
        assert_state_equal(state, copied)
