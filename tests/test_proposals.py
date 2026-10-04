import torch

from analysis.proposals import proposal_coverage, select_frames


def test_subset_is_filename_based_and_order_independent():
    images = [{"id": i, "file_name": f"clip_a_{i}.jpg"} for i in range(20)]
    images += [{"id": 100 + i, "file_name": f"clip_b_{i}.jpg"} for i in range(20)]
    chosen = select_frames(images)
    assert len(chosen) == 16
    assert chosen == select_frames(list(reversed(images)))
    assert len({i["id"] for i in chosen}) == 16


def test_proposal_limits_iou_and_empty_negatives():
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    proposals = torch.tensor([[2.0, 0.0, 12.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    assert proposal_coverage(proposals, gt, 1, 0.5).tolist() == [True, False]
    assert proposal_coverage(proposals, gt, 2, 0.75).tolist() == [False, True]
    assert proposal_coverage(proposals, gt[:0], 2, 0.5).numel() == 0
    assert not proposal_coverage(proposals[:0], gt, 2, 0.5).any()
