import datetime
import os
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import dataset_inputs as di

"""

Tests for dataset_inputs.py against a synthetic mini-archive that
mimics the RDA d633000 forecast-format layout: YYYYMM directories,
half-month chunk filenames with init-range suffixes, and
(forecast_initial_time, forecast_hour, latitude, longitude) variables
on a descending-latitude global-style grid.

"""


def build_archive(root):
    """
    Two chunk files for June 2005 on a coarse 2.5-degree grid, with
    TTR(init, hour, lat, lon) = encode(init, hour) constant per field
    so tests can verify exactly which forecast slice was read.
    """
    lat = np.arange(90.0, -90.1, -2.5)     # descending, like ERA5
    lon = np.arange(0.0, 360.0, 2.5)
    hours = np.arange(1, 13)

    def encode(init, hour):
        return init.day * 100.0 + init.hour + hour / 100.0

    for start, end in [
        (datetime.datetime(2005, 6, 1, 6), datetime.datetime(2005, 6, 16, 6)),
        (datetime.datetime(2005, 6, 16, 6), datetime.datetime(2005, 7, 1, 6)),
    ]:
        inits = []
        t = start
        while t < end:   # end label is exclusive, matching d633000
            inits.append(t)
            t += datetime.timedelta(hours=12)
        data = np.zeros((len(inits), len(hours), len(lat), len(lon)),
                        dtype=np.float32)
        for a, init in enumerate(inits):
            for b, h in enumerate(hours):
                data[a, b] = encode(init, int(h))
        ds = xr.Dataset(
            {'TTR': (('forecast_initial_time', 'forecast_hour',
                      'latitude', 'longitude'), data)},
            coords={'forecast_initial_time': inits,
                    'forecast_hour': hours.astype('int32'),
                    'latitude': lat, 'longitude': lon},
        )
        month_dir = os.path.join(root, start.strftime('%Y%m'))
        os.makedirs(month_dir, exist_ok=True)
        name = (f"e5.oper.fc.sfc.accumu.128_179_ttr.ll025sc."
                f"{start.strftime('%Y%m%d%H')}_{end.strftime('%Y%m%d%H')}.nc")
        ds.to_netcdf(os.path.join(month_dir, name))
    return root


@pytest.fixture()
def archive(tmp_path):
    return build_archive(str(tmp_path))


def loader(archive, **kw):
    kw.setdefault('negate', False)
    return di.ERA5ForecastLoader(archive, **kw)


@pytest.mark.parametrize("valid,expect_init_h", [
    (datetime.datetime(2005, 6, 5, 7),  (datetime.datetime(2005, 6, 5, 6), 1)),
    (datetime.datetime(2005, 6, 5, 18), (datetime.datetime(2005, 6, 5, 6), 12)),
    (datetime.datetime(2005, 6, 5, 19), (datetime.datetime(2005, 6, 5, 18), 1)),
    (datetime.datetime(2005, 6, 6, 0),  (datetime.datetime(2005, 6, 5, 18), 6)),
    (datetime.datetime(2005, 6, 6, 6),  (datetime.datetime(2005, 6, 5, 18), 12)),
])
def test_init_and_hour_mapping(valid, expect_init_h):
    assert di.ERA5ForecastLoader.init_and_hour(valid) == expect_init_h


def test_frame_reads_correct_forecast_slice(archive):
    ld = loader(archive)
    # valid 2005-06-05 09:00 -> init 06-05 06Z, hour 3 -> encode = 506.03
    f = ld.frame(datetime.datetime(2005, 6, 5, 9))
    assert np.allclose(f, 506.03)
    # valid 2005-06-20 00:00 -> init 06-19 18Z, hour 6 (second chunk)
    f = ld.frame(datetime.datetime(2005, 6, 20, 0))
    assert np.allclose(f, 1918.06)


def test_chunk_boundary_init(archive):
    # regression: the init labeled as a chunk's END belongs to the NEXT
    # chunk (2004-05-16 06Z KeyError found on real d633000 data).
    ld = loader(archive)
    # valid 06-16 07:00 -> init 06-16 06Z -> must load from chunk 2,
    # whose encode value is day*100 + inithour + hour/100
    f = ld.frame(datetime.datetime(2005, 6, 16, 7))
    assert np.allclose(f, 1606.01)
    # and the hour before the boundary still comes from chunk 1
    f = ld.frame(datetime.datetime(2005, 6, 16, 6))   # init 06-15 18Z
    assert np.allclose(f, 1518.12)


def test_frame_orientation_and_bounds(archive):
    ld = loader(archive, lat_bounds=(20.0, 50.0), lon_bounds=(220.0, 300.0))
    f = ld.frame(datetime.datetime(2005, 6, 5, 9))
    # 2.5-degree synthetic grid: 20..50 lat -> 13 rows, 220..300 -> 33 cols
    assert f.shape == (13, 33)


def test_latitude_flip(archive):
    # overwrite one file with latitude-dependent values to check flip
    ld = loader(archive)
    path = ld._file_for_init(datetime.datetime(2005, 6, 5, 6))
    ds = xr.open_dataset(path)
    data = ds['TTR'].values
    lat = ds['latitude'].values  # descending
    data[:] = lat[None, None, :, None]  # value == latitude
    ds['TTR'].values = data
    ds.to_netcdf(path + '.tmp'); ds.close()
    os.replace(path + '.tmp', path)

    f = ld.frame(datetime.datetime(2005, 6, 5, 9))
    # returned rows must be ascending in latitude, matching the masks
    assert f[0, 0] < f[-1, 0]
    assert f[0, 0] == pytest.approx(-90.0) or f[0, 0] <= f[-1, 0]


def test_negation_and_zscore(archive):
    ld = loader(archive, negate=True, mean=-506.03, std=2.0)
    f = ld.frame(datetime.datetime(2005, 6, 5, 9))
    # raw 506.03 -> negated -506.03 -> zscore (x - mean)/std = 0
    assert np.allclose(f, 0.0)


def test_difference_mode(archive):
    ld = loader(archive, difference=True)
    # encode(init, h) - encode(init, h-1) = 0.01 for h > 1
    f = ld.frame(datetime.datetime(2005, 6, 5, 9))
    assert np.allclose(f, 0.01, atol=1e-4)
    # hour 1 has no predecessor; returns the raw hour-1 field
    f1 = ld.frame(datetime.datetime(2005, 6, 5, 7))
    assert np.allclose(f1, 506.01)


def test_window_stacks_iso_strings(archive):
    ld = loader(archive)
    times = ['2005-06-05T07:00:00', '2005-06-05T08:00:00',
             '2005-06-05T09:00:00']
    w = ld.window(times)
    assert w.shape[0] == 3 and w.shape[1] == 1
    assert np.allclose(w[0], 506.01) and np.allclose(w[2], 506.03)


def test_missing_chunk_raises(archive):
    ld = loader(archive)
    with pytest.raises(FileNotFoundError):
        ld.frame(datetime.datetime(2007, 1, 5, 9))


def test_compute_stats(archive):
    ld = loader(archive)
    mean, std = ld.compute_stats([datetime.datetime(2005, 6, 5, 7),
                                  datetime.datetime(2005, 6, 5, 8)])
    assert mean == pytest.approx((506.01 + 506.02) / 2, abs=1e-3)
    assert std == pytest.approx(0.005, abs=1e-3)
