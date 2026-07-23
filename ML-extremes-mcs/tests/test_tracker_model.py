import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tracker_model as tm
import tracking_targets as tt

"""

Tests for tracker_model.py. All synthetic: small grids sized for the
U-Net's 16x downsampling (32 x 48). No training is performed; tests
cover shapes, probability constraints, degenerate storm counts, loss
behavior, and a full forward+backward smoke test wiring the model to
tracking_targets-format targets.

"""

H, W = 32, 48


def make_mask(ids_boxes):
    m = np.zeros((H, W), dtype=int)
    for i, (y0, y1, x0, x1) in ids_boxes.items():
        m[y0:y1, x0:x1] = i
    return m


def test_mask_pool_values():
    feats = torch.zeros((3, H, W))
    feats[0, 2:6, 4:10] = 2.0
    mask = make_mask({7: (2, 6, 4, 10)})
    ids, emb, cent, area = tm.mask_pool(feats, mask)
    assert ids == [7]
    assert torch.isclose(emb[0, 0], torch.tensor(2.0))
    assert torch.isclose(emb[0, 1], torch.tensor(0.0))
    assert area[0] == 24
    # centroid of rows 2..5 is 3.5/32, cols 4..9 is 6.5/48
    assert torch.isclose(cent[0, 0], torch.tensor(3.5 / H))
    assert torch.isclose(cent[0, 1], torch.tensor(6.5 / W))


def test_mask_pool_empty():
    ids, emb, cent, area = tm.mask_pool(torch.zeros((3, H, W)),
                                        np.zeros((H, W), dtype=int))
    assert ids == [] and emb.shape == (0, 3)


def test_association_head_rows_sum_to_one():
    head = tm.AssociationHead(emb_dim=8, hidden=16)
    emb0, emb1 = torch.randn(3, 8), torch.randn(2, 8)
    cent0, cent1 = torch.rand(3, 2), torch.rand(2, 2)
    area0, area1 = torch.ones(3) * 50, torch.ones(2) * 40
    fwd, bwd = head(emb0, cent0, area0, emb1, cent1, area1)
    assert fwd.shape == (3, 3)   # 2 storms + dustbin
    assert bwd.shape == (2, 4)   # 3 storms + dustbin
    assert torch.allclose(fwd.sum(dim=1), torch.ones(3), atol=1e-5)
    assert torch.allclose(bwd.sum(dim=1), torch.ones(2), atol=1e-5)


def test_association_head_no_storms_next_frame():
    # all forward mass must land in the dustbin when t+1 is empty
    head = tm.AssociationHead(emb_dim=8, hidden=16)
    emb0 = torch.randn(2, 8)
    fwd, bwd = head(emb0, torch.rand(2, 2), torch.ones(2) * 30,
                    torch.zeros(0, 8), torch.zeros(0, 2), torch.zeros(0))
    assert fwd.shape == (2, 1)
    assert torch.allclose(fwd, torch.ones(2, 1))
    assert bwd.shape == (0, 3)


def test_trackernet_forward_shapes():
    net = tm.TrackerNet(n_channels=1, n_classes=2)
    x = torch.randn(2, 3, 1, H, W)
    feats, logits = net(x)
    assert feats.shape == (2, 3, 64, H, W)
    assert logits.shape == (2, 3, 2, H, W)


def test_trackernet_associate_shapes():
    net = tm.TrackerNet(n_channels=1, n_classes=2)
    x = torch.randn(1, 2, 1, H, W)
    feats, _ = net(x)
    m0 = make_mask({5: (2, 8, 4, 12), 9: (20, 28, 30, 44)})
    m1 = make_mask({5: (3, 9, 5, 13)})
    ids0, ids1, fwd, bwd = net.associate(feats[0, 0], feats[0, 1], m0, m1)
    assert ids0 == [5, 9] and ids1 == [5]
    assert fwd.shape == (2, 2) and bwd.shape == (1, 3)


def test_dice_loss_perfect_and_worst():
    target = torch.zeros(1, H, W)
    target[0, 4:12, 4:12] = 1
    perfect = torch.full((1, 2, H, W), -20.0)
    perfect[0, 1, 4:12, 4:12] = 20.0
    perfect[0, 0, 4:12, 4:12] = -20.0
    perfect[0, 0][target[0] == 0] = 20.0
    assert tm.dice_loss(perfect, target) < 0.01
    inverted = perfect.flip(dims=(1,))
    assert tm.dice_loss(inverted, target) > 0.9


def test_association_kl_zero_when_equal():
    t = torch.tensor([[0.6, 0.3, 0.1], [0.0, 0.2, 0.8]])
    assert tm.association_kl(t.clone(), t) < 1e-6
    off = torch.tensor([[0.1, 0.3, 0.6], [0.8, 0.2, 0.0]])
    assert tm.association_kl(off, t) > 0.1


def test_association_kl_empty():
    assert tm.association_kl(torch.zeros(0, 3), torch.zeros(0, 3)) == 0


def test_end_to_end_loss_backward():
    # full wiring: masks -> targets (tracking_targets) -> model ->
    # combined loss -> gradients reach both heads and the backbone
    net = tm.TrackerNet(n_channels=1, n_classes=2)
    m0 = make_mask({1: (2, 10, 4, 14), 2: (18, 28, 28, 44)})
    m1 = make_mask({1: (3, 11, 6, 16), 2: (18, 26, 30, 44)})

    ids0, ids1, f, b = tt.overlap_fractions(m0, m1)
    target = {'forward': torch.from_numpy(f).float(),
              'backward': torch.from_numpy(b).float()}

    x = torch.randn(1, 2, 1, H, W)
    feats, logits = net(x)
    preds = [net.associate(feats[0, 0], feats[0, 1], m0, m1)[2:]]

    binary = torch.from_numpy(
        np.stack([(m0 > 0), (m1 > 0)]).astype(np.int64)
    )
    loss = tm.tracking_loss(logits[0], binary, preds, [target])
    assert torch.isfinite(loss)
    loss.backward()

    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert any(g.abs().sum() > 0 for g in grads)
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in net.assoc.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in net.backbone.parameters())
