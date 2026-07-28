import glob
import os
import sys

import numpy as np
import pytest
import torch
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import dataset_temporal as dt

"""

Tests for dataset_temporal.py. Synthetic mcstrack-format files are
written to a pytest tmp_path; integration tests run against real files
in a local sample_data/ directory when present and are skipped
otherwise.

"""

SAMPLE_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'sample_data'
)


def write_mask(dirpath, stamp, mask):
    """
    Write a minimal mcstrack-format file (mcstrack_YYYYMMDD_HHMM.nc).
    """
    path = os.path.join(dirpath, f'mcstrack_{stamp}.nc')
    ds = xr.Dataset(
        {'cloudtracknumber': (('time', 'lat', 'lon'),
                              mask[None].astype(float))}
    )
    ds.to_netcdf(path)
    return path


def simple_mask(track_id=0):
    m = np.zeros((10, 12), dtype=int)
    if track_id:
        m[2:5, 3:7] = track_id
    return m


def test_consecutive_runs_respects_gaps(tmp_path):
    # four hourly files with a gap after the second: 00,01, [gap], 04,05
    paths = [write_mask(tmp_path, s, simple_mask(1))
             for s in ['20050601_0000', '20050601_0100',
                       '20050601_0400', '20050601_0500']]
    windows = dt.consecutive_runs(paths, window=2)
    stamps = [[os.path.basename(f)[9:22] for f in w] for w in windows]
    assert stamps == [['20050601_0000', '20050601_0100'],
                      ['20050601_0400', '20050601_0500']]


def test_consecutive_runs_stride_one(tmp_path):
    paths = [write_mask(tmp_path, f'20050601_{h:02d}00', simple_mask(1))
             for h in range(4)]
    assert len(dt.consecutive_runs(paths, window=3)) == 2


def test_consecutive_runs_stride(tmp_path):
    paths = [write_mask(tmp_path, f'20050601_{h:02d}00', simple_mask(1))
             for h in range(9)]
    # stride = window - 1 = 2: windows tile so each hour-pair transition
    # appears exactly once: [0,1,2], [2,3,4], [4,5,6], [6,7,8]
    windows = dt.consecutive_runs(paths, window=3, stride=2)
    starts = [os.path.basename(w[0])[9:22] for w in windows]
    assert starts == ['20050601_0000', '20050601_0200',
                      '20050601_0400', '20050601_0600']
    # stride 3: non-overlapping tiling [0,1,2], [3,4,5], [6,7,8]
    assert len(dt.consecutive_runs(paths, window=3, stride=3)) == 3
    # stride survives a gap: reset happens regardless of stride phase
    gappy = paths[:4] + paths[6:]
    windows = dt.consecutive_runs(gappy, window=3, stride=2)
    assert all(
        dt.timestamp_of(w[2]) - dt.timestamp_of(w[0])
        == __import__('datetime').timedelta(hours=2)
        for w in windows
    )


def test_dataset_item_shapes(tmp_path):
    paths = [write_mask(tmp_path, f'20050601_{h:02d}00', simple_mask(7))
             for h in range(3)]
    ds = dt.TemporalMaskDataset(paths, window=3)
    assert len(ds) == 1
    item = ds[0]
    assert item['track_masks'].shape == (3, 10, 12)
    assert item['track_masks'].dtype == torch.long
    assert item['binary_masks'].max() == 1.0
    assert len(item['targets']) == 2
    assert len(item['times']) == 3
    assert item['times'][1] == '2005-06-01T01:00:00'


def test_dataset_targets_track_continuation(tmp_path):
    m0 = simple_mask(7)
    m1 = np.roll(m0, 1, axis=1)  # storm advects one pixel east
    paths = [write_mask(tmp_path, '20050601_0000', m0),
             write_mask(tmp_path, '20050601_0100', m1)]
    item = dt.TemporalMaskDataset(paths, window=2)[0]
    tgt = item['targets'][0]
    assert tgt['ids_t0'] == [7] and tgt['ids_t1'] == [7]
    assert tgt['forward'].shape == (1, 2)  # one storm + dustbin column
    assert [e['kind'] for e in tgt['events']] == ['continuation']


def test_window_below_two_rejected(tmp_path):
    with pytest.raises(ValueError):
        dt.TemporalMaskDataset([], window=1)


def test_collate_windows(tmp_path):
    paths = [write_mask(tmp_path, f'20050601_{h:02d}00', simple_mask(3))
             for h in range(4)]
    ds = dt.TemporalMaskDataset(paths, window=3)
    batch = dt.collate_windows([ds[0], ds[1]])
    assert batch['track_masks'].shape == (2, 3, 10, 12)
    assert len(batch['targets']) == 2
    assert len(batch['targets'][0]) == 2


@pytest.mark.skipif(
    not glob.glob(os.path.join(SAMPLE_DIR, 'mcstrack_20050601_*.nc')),
    reason='local sample_data/ with June 1 2005 mcstrack files not found'
)
def test_real_june1_window_count_and_split():
    files = sorted(glob.glob(
        os.path.join(SAMPLE_DIR, 'mcstrack_20050601_*.nc')
    ))
    ds = dt.TemporalMaskDataset(files, window=3)
    assert len(ds) == len(files) - 2
    # window starting 07z covers pairs (07z,08z) and (08z,09z);
    # the 08z->09z pair contains the id 76 -> (76, 78) split
    item = ds[7]
    kinds = [e['kind'] for e in item['targets'][1]['events']]
    assert 'split' in kinds
    split = next(e for e in item['targets'][1]['events']
                 if e['kind'] == 'split')
    assert split['ids_t0'] == [76]
    assert sorted(split['ids_t1']) == [76, 78]
