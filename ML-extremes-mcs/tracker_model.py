import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import unet

"""

v1 probabilistic storm-tracking model: tracking-by-detection with the
repo U-Net as a shared per-frame backbone and two heads.

The segmentation head is the U-Net's existing output convolution. The
association head turns per-storm embeddings (mask-pooled backbone
features) into row-stochastic assignment matrices with a dustbin column
-- the same format as the targets from tracking_targets.py, so splits
appear as mass on two columns, dissipation/genesis as dustbin mass.

v1 has no temporal mixing: frames are encoded independently and all
temporal reasoning happens in the association head. Temporal attention
at the U-Net bottleneck is the planned v1.5 ablation.

"""


def mask_pool(features, mask):
    """
    Pool backbone features over each storm's pixel footprint.
    Args:
        features (tensor): (d, lat, lon) feature map for one frame.
        mask (tensor or array): (lat, lon) integer track mask, 0 =
                                background.
    Returns:
        ids (list): Sorted track IDs present in the mask.
        emb (tensor): (n, d) per-storm mean-pooled embeddings.
        cent (tensor): (n, 2) storm centroids in normalized [0, 1]
                       (lat, lon) coordinates.
        area (tensor): (n,) storm areas as pixel counts.
    """
    if not torch.is_tensor(mask):
        mask = torch.from_numpy(np.asarray(mask))
    mask = mask.long()

    ids = sorted(int(i) for i in torch.unique(mask).tolist() if i != 0)
    d = features.shape[0]
    H, W = mask.shape

    emb = features.new_zeros((len(ids), d))
    cent = features.new_zeros((len(ids), 2))
    area = features.new_zeros((len(ids),))
    for k, i in enumerate(ids):
        where = (mask == i)
        yy, xx = torch.nonzero(where, as_tuple=True)
        emb[k] = features[:, where].mean(dim=1)
        cent[k, 0] = yy.float().mean() / H
        cent[k, 1] = xx.float().mean() / W
        area[k] = where.sum()
    return ids, emb, cent, area


class AssociationHead(nn.Module):
    """
    Scores storm pairs across consecutive frames and produces
    row-stochastic forward/backward assignment matrices with dustbins.

    Pair features are the two embeddings plus explicit geometry
    (centroid offset, log area ratio), injecting the position
    information that overlap-based targets alone do not carry.
    """

    def __init__(self, emb_dim=64, hidden=128):
        """
        Initialization.
        Args:
            emb_dim (int): Backbone feature channels per storm.
            hidden (int): Hidden width of the scoring MLPs.
        """
        super().__init__()
        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * emb_dim + 3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.dustbin_mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def pair_scores(self, emb0, cent0, area0, emb1, cent1, area1):
        """
        Affinity logits for every (storm at t, storm at t+1) pair.
        Args:
            emb0, cent0, area0: (n0, d), (n0, 2), (n0,) from mask_pool.
            emb1, cent1, area1: (n1, d), (n1, 2), (n1,) from mask_pool.
        Returns:
            (n0, n1) tensor of unnormalized affinity scores.
        """
        n0, n1 = emb0.shape[0], emb1.shape[0]
        if n0 == 0 or n1 == 0:
            return emb0.new_zeros((n0, n1))
        e0 = emb0[:, None, :].expand(n0, n1, -1)
        e1 = emb1[None, :, :].expand(n0, n1, -1)
        offset = cent1[None, :, :] - cent0[:, None, :]
        logratio = (torch.log(area1[None, :] + 1.0)
                    - torch.log(area0[:, None] + 1.0))[..., None]
        feats = torch.cat([e0, e1, offset, logratio], dim=-1)
        return self.pair_mlp(feats).squeeze(-1)

    def forward(self, emb0, cent0, area0, emb1, cent1, area1):
        """
        Predict assignment matrices for one consecutive-frame pair.
        Returns:
            forward_mat (tensor): (n0, n1 + 1) rows sum to 1; last
                                  column is the dissipation dustbin.
            backward_mat (tensor): (n1, n0 + 1) rows sum to 1; last
                                   column is the genesis dustbin.
        """
        scores = self.pair_scores(emb0, cent0, area0, emb1, cent1, area1)
        bin0 = self.dustbin_mlp(emb0) if emb0.shape[0] else emb0.new_zeros((0, 1))
        bin1 = self.dustbin_mlp(emb1) if emb1.shape[0] else emb1.new_zeros((0, 1))
        forward_mat = torch.softmax(torch.cat([scores, bin0], dim=1), dim=1)
        backward_mat = torch.softmax(torch.cat([scores.t(), bin1], dim=1), dim=1)
        return forward_mat, backward_mat


class TrackerNet(nn.Module):
    """
    Shared U-Net backbone with segmentation and association heads.

    Reuses the layers of an unet.UNet instance directly (rather than
    modifying unet.py) so backbone weights can be initialized from an
    existing trained segmentation checkpoint.
    """

    def __init__(self, n_channels=1, n_classes=2, bilinear=True,
                 assoc_hidden=128):
        """
        Initialization.
        Args:
            n_channels (int): Input fields per frame.
            n_classes (int): Segmentation classes. Defaults to 2.
            bilinear (bool): U-Net upsampling mode.
            assoc_hidden (int): Association MLP hidden width.
        """
        super().__init__()
        self.backbone = unet.UNet(n_channels=n_channels,
                                  n_classes=n_classes, bilinear=bilinear)
        self.feat_dim = 64  # channels out of the U-Net's last Up block
        self.assoc = AssociationHead(emb_dim=self.feat_dim,
                                     hidden=assoc_hidden)

    def encode(self, x):
        """
        Run the U-Net, exposing penultimate features and seg logits.
        Args:
            x (tensor): (batch, channels, lat, lon) single-frame input.
        Returns:
            feats (tensor): (batch, 64, lat, lon) penultimate features.
            logits (tensor): (batch, n_classes, lat, lon) seg logits.
        """
        b = self.backbone
        x1 = b.inc(x)
        x2 = b.down1(x1)
        x3 = b.down2(x2)
        x4 = b.down3(x3)
        x5 = b.down4(x4)
        y = b.up1(x5, x4)
        y = b.up2(y, x3)
        y = b.up3(y, x2)
        feats = b.up4(y, x1)
        return feats, b.outc(feats)

    def forward(self, x):
        """
        Encode a temporal window, frames independent (v1).
        Args:
            x (tensor): (batch, window, channels, lat, lon).
        Returns:
            feats (tensor): (batch, window, 64, lat, lon).
            logits (tensor): (batch, window, n_classes, lat, lon).
        """
        B, T, C, H, W = x.shape
        feats, logits = self.encode(x.reshape(B * T, C, H, W))
        return (feats.reshape(B, T, -1, H, W),
                logits.reshape(B, T, -1, H, W))

    def associate(self, feats_t0, feats_t1, mask_t0, mask_t1):
        """
        Predict assignment matrices for one frame pair using known
        storm footprints (ground-truth masks during teacher-forced
        training; predicted connected components at inference).
        Args:
            feats_t0, feats_t1 (tensor): (64, lat, lon) frame features.
            mask_t0, mask_t1: (lat, lon) integer track masks.
        Returns:
            ids_t0, ids_t1 (list): Storm IDs, row/column order of the
                                   matrices.
            forward_mat, backward_mat: See AssociationHead.forward.
        """
        ids0, emb0, cent0, area0 = mask_pool(feats_t0, mask_t0)
        ids1, emb1, cent1, area1 = mask_pool(feats_t1, mask_t1)
        fwd, bwd = self.assoc(emb0, cent0, area0, emb1, cent1, area1)
        return ids0, ids1, fwd, bwd


def dice_loss(logits, target):
    """
    Soft Dice loss for the storm class, robust to class imbalance.
    Args:
        logits (tensor): (batch, 2, lat, lon) segmentation logits.
        target (tensor): (batch, lat, lon) binary storm mask.
    Returns:
        Scalar loss in [0, 1]; 0 for a perfect prediction.
    """
    prob = torch.softmax(logits, dim=1)[:, 1]
    target = target.float()
    inter = (prob * target).sum(dim=(1, 2))
    denom = prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    return (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()


def association_kl(pred, target):
    """
    Row-wise KL divergence KL(target || pred) between assignment
    matrices (both row-stochastic, dustbin included).
    Args:
        pred (tensor): (n, m) predicted matrix from AssociationHead.
        target (tensor): (n, m) soft target from tracking_targets.
    Returns:
        Scalar mean KL over rows; 0 when pred equals target.
    """
    if pred.shape[0] == 0:
        return pred.new_zeros(())
    target = target.clamp_min(0)
    logp = torch.log(pred.clamp_min(1e-8))
    logt = torch.log(target.clamp_min(1e-8))
    kl = (target * (logt - logp)).sum(dim=1)
    return kl.mean()


def tracking_loss(logits, binary_masks, assoc_preds, assoc_targets,
                  lam=1.0):
    """
    Combined detection + association loss for one window.
    Args:
        logits (tensor): (window, 2, lat, lon) seg logits.
        binary_masks (tensor): (window, lat, lon) binary storm masks.
        assoc_preds (list): Per-pair (forward_mat, backward_mat) tuples.
        assoc_targets (list): Per-pair target dicts from
                              dataset_temporal (forward/backward keys).
        lam (float): Weight of the association term.
    Returns:
        Scalar total loss.
    """
    seg = (F.cross_entropy(logits, binary_masks.long())
           + dice_loss(logits, binary_masks))
    assoc = logits.new_zeros(())
    for (fwd, bwd), tgt in zip(assoc_preds, assoc_targets):
        assoc = assoc + association_kl(fwd, tgt['forward'].to(fwd))
        assoc = assoc + association_kl(bwd, tgt['backward'].to(bwd))
    if assoc_preds:
        assoc = assoc / len(assoc_preds)
    return seg + lam * assoc
