import datetime
import os
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import dataset_inputs as di
from test_dataset_inputs import build_archive

"""

Tests for ERA5AnalysisLoader and MultiChannelLoader against synthetic
mini-archives mimicking the analysis-table layout: monthly surface
files (CAPE-style) and daily pressure-level files (u-wind-style) with
END-INCLUSIVE filename date ranges and descending latitude.

"""


def build_analysis_archive(root):
    lat = np.arange(90.0, -90.1, -2.5)
    lon = np.arange(0.0, 360.0, 2.5)

    # monthly surface file (CAPE-style): value = day + hour/100
    times = [datetime.datetime(2005, 6, 1) + datetime.timedelta(hours=h)
             for h in range(30 * 24)]
    data = np.zeros((len(times), len(lat), len(lon)), dtype=np.float32)
    for k, t in enumerate(times):
        data[k] = t.day + t.hour / 100.0
    month_dir = os.path.join(root, 'sfc', '200506')
    os.makedirs(month_dir, exist_ok=True)
    xr.Dataset(
        {'CAPE': (('time', 'latitude', 'longitude'), data)},
        coords={'time': times, 'latitude': lat, 'longitude': lon},
    ).to_netcdf(os.path.join(
        month_dir,
        'e5.oper.an.sfc.128_059_cape.ll025sc.2005060100_2005063023.nc'))

    # daily pressure-level files (U-style): value = level + hour/100
    levels = np.array([500.0, 850.0])
    pl_dir = os.path.join(root, 'pl', '200506')
    os.makedirs(pl_dir, exist_ok=True)
    for day in (5, 6):
        times = [datetime.datetime(2005, 6, day) + datetime.timedelta(hours=h)
                 for h in range(24)]
        data = np.zeros((len(times), len(levels), len(lat), len(lon)),
                        dtype=np.float32)
        for k, t in enumerate(times):
            for li, lv in enumerate(levels):
                data[k, li] = lv + t.hour / 100.0
        xr.Dataset(
            {'U': (('time', 'level', 'latitude', 'longitude'), data)},
            coords={'time': times, 'level': levels,
                    'latitude': lat, 'longitude': lon},
        ).to_netcdf(os.path.join(
            pl_dir,
            f'e5.oper.an.pl.128_131_u.ll025uv.200506{day:02d}00_'
            f'200506{day:02d}23.nc'))
    return os.path.join(root, 'sfc'), os.path.join(root, 'pl')


@pytest.fixture()
def archives(tmp_path):
    return build_analysis_archive(str(tmp_path))


def test_surface_loader_time_selection(archives):
    sfc, _ = archives
    ld = di.ERA5AnalysisLoader(sfc, var='CAPE', file_glob='*_cape.*.nc')
    f = ld.frame(datetime.datetime(2005, 6, 12, 7))
    assert np.allclose(f, 12.07)
    # end-inclusive boundary: hour 23 of the last day is in this file
    f = ld.frame(datetime.datetime(2005, 6, 30, 23))
    assert np.allclose(f, 30.23)


def test_pressure_level_selection(archives):
    _, pl = archives
    ld = di.ERA5AnalysisLoader(pl, var='U', file_glob='*128_131_u.*.nc',
                               level=850)
    f = ld.frame(datetime.datetime(2005, 6, 5, 9))
    assert np.allclose(f, 850.09)
    ld500 = di.ERA5AnalysisLoader(pl, var='U', file_glob='*128_131_u.*.nc',
                                  level=500)
    assert np.allclose(ld500.frame(datetime.datetime(2005, 6, 5, 9)),
                       500.09)


def test_analysis_bounds_flip_and_zscore(archives):
    sfc, _ = archives
    ld = di.ERA5AnalysisLoader(sfc, var='CAPE', file_glob='*_cape.*.nc',
                               mean=12.07, std=2.0)
    f = ld.frame(datetime.datetime(2005, 6, 12, 7))
    assert f.shape == (13, 33)   # 2.5-degree grid, 20-50N x 220-300E
    assert np.allclose(f, 0.0)   # z-scored


def test_missing_analysis_file_raises(archives):
    sfc, _ = archives
    ld = di.ERA5AnalysisLoader(sfc, var='CAPE', file_glob='*_cape.*.nc')
    with pytest.raises(FileNotFoundError):
        ld.frame(datetime.datetime(2007, 1, 5, 9))


def test_multichannel_stacks_mixed_loaders(archives, tmp_path):
    sfc, pl = archives
    fc = build_archive(str(tmp_path / 'fc'))   # forecast-format TTR

    multi = di.MultiChannelLoader([
        di.ERA5ForecastLoader(fc, negate=False),
        di.ERA5AnalysisLoader(sfc, var='CAPE', file_glob='*_cape.*.nc'),
        di.ERA5AnalysisLoader(pl, var='U', file_glob='*128_131_u.*.nc',
                              level=850),
    ])
    w = multi.window(['2005-06-05T09:00:00', '2005-06-05T10:00:00'])
    assert w.shape == (2, 3, 13, 33)
    # channel identities: forecast encode, day.hour, level.hour
    assert np.allclose(w[0, 0], 506.03)   # TTR chunk encode(init,h)
    assert np.allclose(w[0, 1], 5.09)     # CAPE day+hour/100
    assert np.allclose(w[0, 2], 850.09)   # u850
    # second frame advances the hour in every channel
    assert np.allclose(w[1, 1], 5.10)
    assert np.allclose(w[1, 2], 850.10)


def test_multichannel_rejects_empty():
    with pytest.raises(ValueError):
        di.MultiChannelLoader([])


def test_channels_factory_and_dataset(archives, tmp_path):
    import train_tracker as tr
    from test_train_tracker import write_masks
    sfc, pl = archives
    fc = build_archive(str(tmp_path / 'fc2'))

    specs = [
        {'kind': 'forecast', 'dir': fc, 'negate': False},
        {'kind': 'analysis', 'dir': sfc, 'var': 'CAPE',
         'glob': '*_cape.*.nc'},
        {'kind': 'analysis', 'dir': pl, 'var': 'U',
         'glob': '*128_131_u.*.nc', 'level': 850},
    ]
    mask_dir = tmp_path / 'masks'
    mask_dir.mkdir()
    files = write_masks(str(mask_dir))

    ds = tr.WindowsWithInputs(files, era5_dir='unused', window=2,
                              loader_factory=tr.channels_factory(specs))
    item = ds[0]
    assert item['inputs'].shape == (2, 3, 13, 33)
    # channel identities at 2005-06-05 07:00
    assert np.allclose(item['inputs'][0, 1].numpy(), 5.07)   # CAPE
    assert np.allclose(item['inputs'][0, 2].numpy(), 850.07)  # u850


def test_build_channel_loader_kinds(archives):
    import train_tracker as tr
    sfc, _ = archives
    ld = tr.build_channel_loader(
        {'kind': 'analysis', 'dir': sfc, 'var': 'CAPE',
         'glob': '*_cape.*.nc', 'mean': 1.0, 'std': 2.0})
    assert isinstance(ld, di.ERA5AnalysisLoader)
    assert ld.mean == 1.0
    with pytest.raises(ValueError):
        tr.build_channel_loader({'kind': 'quantum', 'dir': sfc})
