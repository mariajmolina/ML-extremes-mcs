import datetime
import os
import sys

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import train_tracker as tr
from test_dataset_inputs import build_archive

"""

Tests for train_tracker.py: the combined windows+inputs dataset and one
real optimization step on synthetic data (tiny grids, CPU). Reuses the
synthetic ERA5 mini-archive builder from test_dataset_inputs; the mask
grid here is sized to the synthetic archive's coarse 2.5-degree domain
slice (13 x 33 for lat 20-50 / lon 220-300).

"""


def write_masks(dirpath, n_hours=4):
    paths = []
    for h in range(n_hours):
        m = np.zeros((13, 33), dtype=float)
        m[3:7, 4 + h: 10 + h] = 5        # storm 5, advecting east
        m[8:11, 20:26] = 9               # storm 9, stationary
        ds = xr.Dataset(
            {'cloudtracknumber': (('time', 'lat', 'lon'), m[None])}
        )
        path = os.path.join(dirpath, f'mcstrack_20050605_{7 + h:02d}00.nc')
        ds.to_netcdf(path)
        paths.append(path)
    return paths


def test_windows_with_inputs(tmp_path):
    archive = build_archive(str(tmp_path / 'era5'))
    mask_dir = tmp_path / 'masks'
    mask_dir.mkdir()
    files = write_masks(str(mask_dir))

    ds = tr.WindowsWithInputs(files, archive, window=3)
    assert len(ds) == 2
    item = ds[0]
    assert item['inputs'].shape == (3, 1, 13, 33)
    assert item['track_masks'].shape == (3, 13, 33)
    # inputs and masks correspond to the same valid times
    assert item['times'][0] == '2005-06-05T07:00:00'
    # synthetic archive encodes init/hour into values: 07Z -> 506.01 (negated)
    assert np.allclose(item['inputs'][0].numpy(), -506.01)


def test_parse_years():
    assert tr.parse_years('2004-2007') == [2004, 2005, 2006, 2007]
    assert tr.parse_years('2016,2018') == [2016, 2018]


def test_pad_lat_shapes():
    t = torch.zeros(3, 121, 321)
    assert tr.pad_lat(t).shape == (3, 128, 321)


def test_training_step_reduces_loss(tmp_path):
    # one real optimization loop on synthetic data: loss must be finite
    # and decrease over a handful of steps on this tiny fixed sample
    archive = build_archive(str(tmp_path / 'era5'))
    mask_dir = tmp_path / 'masks'
    mask_dir.mkdir()
    files = write_masks(str(mask_dir))

    # pad synthetic grids 13x33 -> 16x48 won't divide by 16; use the
    # model on an upsampled copy instead: repeat to 52x66 -> pad later.
    ds = tr.WindowsWithInputs(files, archive, window=2,
                              mean=-506.0, std=1.0)
    item = ds[0]

    net = tr.tracker_model.TrackerNet(n_channels=1, n_classes=2)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    x = item['inputs'].unsqueeze(0).repeat_interleave(4, dim=3)
    x = x.repeat_interleave(2, dim=4)      # (1, 2, 1, 52, 66)
    x = torch.nn.functional.pad(x, (0, 14, 0, 12))   # -> (64, 80)
    track = item['track_masks'].repeat_interleave(4, dim=1)
    track = track.repeat_interleave(2, dim=2)
    track = torch.nn.functional.pad(track, (0, 14, 0, 12))

    losses = []
    for _ in range(5):
        feats, logits = net(x)
        _, _, pf, pb = net.associate(feats[0, 0], feats[0, 1],
                                     track[0], track[1])
        loss = tr.tracker_model.tracking_loss(
            logits[0], (track > 0).long(),
            [(pf, pb)], [item['targets'][0]],
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))

    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]
