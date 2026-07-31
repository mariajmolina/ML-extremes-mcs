import datetime
import os

import numpy as np
import torch
from torch.utils.data import Dataset

import tracking_targets

"""

PyTorch dataset yielding temporal windows for spatiotemporal storm
tracking. Each sample is a run of T consecutive hourly FLEXTRKR mask
files: the per-frame track masks plus, for each of the T-1 hour pairs,
the soft association matrices from tracking_targets.py that serve as
tracking targets.

Windows are only formed from files exactly one hour apart, so gaps in
the archive (e.g., the November-February off-season) never produce a
window that silently spans missing hours.

Input atmospheric fields (e.g., ERA5 OLR) are intentionally not loaded
here; frame timestamps are returned so a wrapper or collate step can
align input tensors from the existing per-ID file pipeline. This keeps
the module testable against mask files alone.

Association matrices vary in size with the number of storms per frame,
so default tensor batching does not apply; use ``collate_windows`` (or
batch_size=1) with a DataLoader.

"""


def timestamp_of(filepath):
    """
    Parse the timestamp from an mcstrack filename.
    Args:
        filepath (str): Path to mcstrack_YYYYMMDD_HHMM.nc file.
    Returns:
        datetime.datetime of the file's timestep.
    """
    name = os.path.basename(filepath)
    return datetime.datetime.strptime(name[9:22], '%Y%m%d_%H%M')


def consecutive_runs(filepaths, window, stride=1):
    """
    Group time-sorted mask files into windows of consecutive hours.
    Args:
        filepaths (list): mcstrack file paths (any order).
        window (int): Number of consecutive hourly frames per window.
        stride (int): Hours the window advances between samples within
                      a consecutive run. 1 (default) yields maximally
                      overlapping windows; window - 1 covers every
                      hour-pair transition exactly once.
    Returns:
        List of window lists, each holding ``window`` file paths whose
        timestamps are exactly one hour apart. A run of N consecutive
        files yields 1 + floor((N - window) / stride) windows.
    """
    files = sorted(filepaths, key=timestamp_of)
    out = []
    run = []
    since_last = None
    for f in files:
        if run and (timestamp_of(f) - timestamp_of(run[-1])
                    != datetime.timedelta(hours=1)):
            run = []
            since_last = None
        run.append(f)
        if len(run) >= window:
            if since_last is None or since_last >= stride:
                out.append(run[-window:])
                since_last = 1
            else:
                since_last += 1
    return out


class TemporalMaskDataset(Dataset):
    """
    Dataset of temporal windows over FLEXTRKR pixel masks.

    Each item is a dict:
        ``track_masks``  (window, lat, lon) long tensor of track IDs.
        ``binary_masks`` (window, lat, lon) float tensor, storm=1.
        ``targets``      list of window-1 dicts from
                         tracking_targets (ids_t0, ids_t1, forward,
                         backward, events).
        ``times``        list of window ISO timestamp strings, for
                         aligning input fields loaded elsewhere.
    """

    def __init__(self, filepaths, window=3, msk_var='cloudtracknumber',
                 min_frac=0.15, stride=1):
        """
        Initialization.
        Args:
            filepaths (list): mcstrack file paths; windows are formed
                              only across exactly-consecutive hours.
            window (int): Frames per sample. Defaults to 3.
            msk_var (str): Mask variable. Defaults to
                           ``cloudtracknumber``.
            min_frac (float): Event-classification threshold passed to
                              tracking_targets.
            stride (int): Window advance in hours; see
                          consecutive_runs().
        """
        if window < 2:
            raise ValueError("window must be >= 2 to form target pairs.")
        self.windows = consecutive_runs(filepaths, window, stride=stride)
        self.window = window
        self.msk_var = msk_var
        self.min_frac = min_frac

    def __len__(self):
        """
        Number of temporal windows.
        """
        return len(self.windows)

    def __getitem__(self, index):
        """
        Build one temporal window sample.
        Args:
            index (int): Window index.
        """
        files = self.windows[index]
        masks = [tracking_targets.load_mask(f, msk_var=self.msk_var)
                 for f in files]

        targets = []
        for k in range(len(masks) - 1):
            ids_t0, ids_t1, forward, backward = (
                tracking_targets.overlap_fractions(masks[k], masks[k + 1])
            )
            events = tracking_targets.classify_events(
                ids_t0, ids_t1, forward, backward, min_frac=self.min_frac
            )
            targets.append({'ids_t0': ids_t0, 'ids_t1': ids_t1,
                            'forward': torch.from_numpy(forward).float(),
                            'backward': torch.from_numpy(backward).float(),
                            'events': events})

        track = torch.from_numpy(np.stack(masks)).long()
        return {'track_masks': track,
                'binary_masks': (track > 0).float(),
                'targets': targets,
                'times': [timestamp_of(f).isoformat() for f in files]}


def collate_windows(samples):
    """
    Collate function for DataLoader batching of temporal windows.

    Mask tensors are stacked along a new batch dimension; association
    targets and times, whose shapes vary with storm count, are kept as
    per-sample lists.

    Args:
        samples (list): Items from TemporalMaskDataset.
    Returns:
        Dict with batched ``track_masks``/``binary_masks`` of shape
        (batch, window, lat, lon), plus ``targets`` and ``times`` lists.
    """
    return {
        'track_masks': torch.stack([s['track_masks'] for s in samples]),
        'binary_masks': torch.stack([s['binary_masks'] for s in samples]),
        'targets': [s['targets'] for s in samples],
        'times': [s['times'] for s in samples],
    }
