import datetime
import glob
import os
import re

import numpy as np
import xarray as xr

"""

Loader aligning raw ERA5 input fields with temporal mask windows.

Reads directly from NCAR's permanent RDA ERA5 archive (d633000) rather
than preprocessed per-ID files, so the training pipeline has no
dependency on purgeable scratch space. Handles the forecast-format
tables (e.g., TTR/OLR in e5.oper.fc.sfc.accumu): half-month chunk files
with (forecast_initial_time, forecast_hour) dimensions are mapped to
plain valid times via valid = init + hour, with inits at 06Z and 18Z
daily and hours 1..12 giving continuous hourly coverage.

Spatial handling matches the FLEXTRKR mask files used as labels: the
requested lat/lon box is sliced from the global grid and latitude is
flipped from ERA5's native descending order to ascending, so returned
arrays align with mask arrays row-for-row without further flips.

"""


class ERA5ForecastLoader:
    """
    Maps valid-time timestamps to fields from ERA5 forecast-format
    files and returns mask-aligned (time, lat, lon) arrays.
    """

    def __init__(self, archive_dir, var='TTR', file_glob='*_ttr.*.nc',
                 lat_bounds=(20.0, 50.0), lon_bounds=(220.0, 300.0),
                 negate=True, difference=False, mean=None, std=None):
        """
        Initialization.
        Args:
            archive_dir (str): Table directory containing YYYYMM
                               subdirectories (e.g., .../e5.oper.fc.sfc.accumu/).
            var (str): Variable name inside the files. Defaults to TTR.
            file_glob (str): Filename pattern selecting the variable's
                             files within a month directory.
            lat_bounds (tuple): (south, north) of the mask domain.
            lon_bounds (tuple): (west, east) in 0-360 convention.
            negate (bool): Negate values (ERA5 TTR is negative-upward;
                           legacy pipeline trains on -TTR). Default True.
            difference (bool): Difference successive forecast hours if
                               the archive stores cumulative-since-init
                               accumulations. Default False.
            mean, std (float): Optional z-score constants; if both are
                               given, windows are normalized.
        """
        self.archive_dir = archive_dir
        self.var = var
        self.file_glob = file_glob
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds
        self.negate = negate
        self.difference = difference
        self.mean = mean
        self.std = std
        self._cache_path = None
        self._cache_ds = None

    @staticmethod
    def init_and_hour(valid):
        """
        Decompose a valid time into (forecast_initial_time, forecast_hour).
        Inits are 06Z and 18Z daily with hours 1..12:
        07..18Z come from the same day's 06Z init; 19..23Z from the same
        day's 18Z init; 00..06Z from the previous day's 18Z init.
        Args:
            valid (datetime.datetime): Valid time (whole hours).
        Returns:
            (datetime.datetime, int): Forecast init time and hour.
        """
        h = valid.hour
        day = datetime.datetime(valid.year, valid.month, valid.day)
        if 7 <= h <= 18:
            return day + datetime.timedelta(hours=6), h - 6
        if h >= 19:
            return day + datetime.timedelta(hours=18), h - 18
        return day - datetime.timedelta(hours=6), h + 6  # prev day 18Z

    def _file_for_init(self, init):
        """
        Locate the half-month chunk file whose init range contains the
        given forecast_initial_time, by parsing filename date ranges
        (e.g., ...2005060106_2005061606.nc).
        """
        for month_dir in {init.strftime('%Y%m'),
                          (init + datetime.timedelta(days=16)).strftime('%Y%m')}:
            pattern = os.path.join(self.archive_dir, month_dir, self.file_glob)
            for path in sorted(glob.glob(pattern)):
                m = re.search(r'\.(\d{10})_(\d{10})\.nc$', path)
                if not m:
                    continue
                start = datetime.datetime.strptime(m.group(1), '%Y%m%d%H')
                end = datetime.datetime.strptime(m.group(2), '%Y%m%d%H')
                # filename end is EXCLUSIVE: the end-labeled init lives
                # in the next chunk (verified against d633000 files)
                if start <= init < end:
                    return path
        raise FileNotFoundError(
            f"No {self.var} chunk file covering init {init} under "
            f"{self.archive_dir}"
        )

    def _open(self, path):
        """
        Open a chunk file with a one-file cache (windows usually draw
        several hours from the same chunk).
        """
        if path != self._cache_path:
            if self._cache_ds is not None:
                self._cache_ds.close()
            self._cache_ds = xr.open_dataset(path)
            self._cache_path = path
        return self._cache_ds

    def frame(self, valid):
        """
        Load one mask-aligned field at a valid time.
        Args:
            valid (datetime.datetime): Valid time (whole hours).
        Returns:
            2d numpy array (lat ascending, lon), aligned to the mask grid.
        """
        init, hour = self.init_and_hour(valid)
        ds = self._open(self._file_for_init(init))
        da = ds[self.var].sel(forecast_initial_time=init)

        if self.difference and hour > 1:
            arr = (da.sel(forecast_hour=hour)
                   - da.sel(forecast_hour=hour - 1))
        else:
            arr = da.sel(forecast_hour=hour)

        south, north = self.lat_bounds
        west, east = self.lon_bounds
        # ERA5 latitude is descending; slice north->south then flip.
        arr = arr.sel(latitude=slice(north, south),
                      longitude=slice(west, east))
        out = arr.values[::-1, :].astype(np.float32)

        if self.negate:
            out = -out
        if self.mean is not None and self.std is not None:
            out = (out - self.mean) / self.std
        return out

    def window(self, times):
        """
        Load a temporal window of mask-aligned fields.
        Args:
            times (list): Valid times — datetimes or ISO strings, as
                          produced by dataset_temporal (item['times']).
        Returns:
            numpy array (window, 1, lat, lon) ready to stack as model
            input channels.
        """
        frames = []
        for t in times:
            if isinstance(t, str):
                t = datetime.datetime.fromisoformat(t)
            frames.append(self.frame(t))
        return np.stack(frames)[:, None, :, :]

    def compute_stats(self, times):
        """
        Compute z-score constants over a set of valid times (e.g., the
        training years), before normalization is enabled.
        Args:
            times (list): Valid times to aggregate over.
        Returns:
            (mean, std) floats over all pixels and times.
        """
        acc, acc2, n = 0.0, 0.0, 0
        for t in times:
            if isinstance(t, str):
                t = datetime.datetime.fromisoformat(t)
            f = self.frame(t).astype(np.float64)
            acc += f.sum()
            acc2 += (f ** 2).sum()
            n += f.size
        mean = acc / n
        var = acc2 / n - mean ** 2
        return float(mean), float(np.sqrt(max(var, 0.0)))


class ERA5AnalysisLoader(ERA5ForecastLoader):
    """
    Loader for ERA5 *analysis* tables (e5.oper.an.sfc, e5.oper.an.pl):
    instantaneous snapshots on a plain hourly time axis, stored as
    monthly (surface) or daily (pressure-level) files whose filename
    date ranges are END-INCLUSIVE (...YYYYMM0100_YYYYMMDD23.nc covers
    hour 23). Pressure-level variables additionally select a level.

    Simpler than the forecast loader (no init/hour decomposition);
    inherits the spatial slicing/flip, normalization, and window/
    compute_stats interfaces so channels from both loader types can be
    stacked interchangeably.
    """

    def __init__(self, archive_dir, var, file_glob, level=None,
                 lat_bounds=(20.0, 50.0), lon_bounds=(220.0, 300.0),
                 negate=False, mean=None, std=None):
        """
        Initialization.
        Args:
            archive_dir (str): Table directory containing YYYYMM
                               subdirectories (e.g., .../e5.oper.an.sfc/).
            var (str): Variable name inside the files (e.g., 'CAPE', 'U').
            file_glob (str): Filename pattern (e.g., '*_cape.*.nc',
                             '*128_131_u.*.nc').
            level (float): Pressure level in hPa for an.pl tables
                           (e.g., 850); None for surface tables.
            negate (bool): Defaults False (unlike TTR, analysis fields
                           are used as stored).
            Remaining args as in ERA5ForecastLoader.
        """
        super().__init__(archive_dir, var=var, file_glob=file_glob,
                         lat_bounds=lat_bounds, lon_bounds=lon_bounds,
                         negate=negate, difference=False,
                         mean=mean, std=std)
        self.level = level

    def _file_for_time(self, valid):
        """
        Locate the analysis file whose (end-inclusive) filename range
        contains the valid time.
        """
        month_dir = valid.strftime('%Y%m')
        pattern = os.path.join(self.archive_dir, month_dir, self.file_glob)
        for path in sorted(glob.glob(pattern)):
            m = re.search(r'\.(\d{10})_(\d{10})\.nc$', path)
            if not m:
                continue
            start = datetime.datetime.strptime(m.group(1), '%Y%m%d%H')
            end = datetime.datetime.strptime(m.group(2), '%Y%m%d%H')
            if start <= valid <= end:   # analysis ranges are inclusive
                return path
        raise FileNotFoundError(
            f"No {self.var} analysis file covering {valid} under "
            f"{self.archive_dir}"
        )

    def frame(self, valid):
        """
        Load one mask-aligned field at a valid time.
        Args:
            valid (datetime.datetime): Valid time (whole hours).
        Returns:
            2d numpy array (lat ascending, lon), aligned to the mask grid.
        """
        ds = self._open(self._file_for_time(valid))
        da = ds[self.var].sel(time=valid)
        if self.level is not None:
            da = da.sel(level=self.level)

        south, north = self.lat_bounds
        west, east = self.lon_bounds
        da = da.sel(latitude=slice(north, south),
                    longitude=slice(west, east))
        out = da.values[::-1, :].astype(np.float32)

        if self.negate:
            out = -out
        if self.mean is not None and self.std is not None:
            out = (out - self.mean) / self.std
        return out


class MultiChannelLoader:
    """
    Stacks frames from several loaders (forecast and/or analysis) into
    multi-channel windows, so a TrackerNet with n_channels > 1 can take
    e.g. [OLR, CAPE, u850, v850] per frame. Presents the same window()
    interface as a single loader.
    """

    def __init__(self, loaders):
        """
        Initialization.
        Args:
            loaders (list): Loader instances in channel order; each must
                provide frame(valid) returning a (lat, lon) array on the
                same grid.
        """
        if not loaders:
            raise ValueError("MultiChannelLoader needs at least one loader.")
        self.loaders = loaders

    def window(self, times):
        """
        Load a temporal window with one channel per loader.
        Args:
            times (list): Valid times, datetimes or ISO strings.
        Returns:
            numpy array (window, n_channels, lat, lon).
        """
        frames = []
        for t in times:
            if isinstance(t, str):
                t = datetime.datetime.fromisoformat(t)
            frames.append(np.stack([ld.frame(t) for ld in self.loaders]))
        return np.stack(frames)
