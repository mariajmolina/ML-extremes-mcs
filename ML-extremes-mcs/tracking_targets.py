import numpy as np
import xarray as xr

"""

Module for building storm-association training targets from FLEXTRKR
pixel-level mask files (mcstrack_*.nc).

Given two consecutive hourly masks, this module produces soft assignment
matrices describing where each storm object's area "goes" between t and
t+1. These serve as training targets for a probabilistic tracking model:
splits, merges, genesis, and lysis appear naturally as one-to-many,
many-to-one, and dustbin assignments rather than as special cases.

Design notes (verified against 2005 FLEXTRKR data):
- Track IDs cannot be used directly as continuity targets. FLEXTRKR
  relays IDs across physically continuous storms: a parent splits off a
  child with a new ID and later dies by merging into that same child.
- The surviving ID does not reliably mark the larger piece of a split
  at the MCS-mask level, so no largest-object assumption is made here.
- Coexisting neighbor storms exchange small boundary overlaps (~IoU 0.01)
  hour to hour that do not correspond to physical events, while real
  split/merge events show overlap fractions of roughly 0.2-0.5. Event
  classification therefore takes a min_frac threshold, but the soft
  matrices themselves are returned unthresholded so a model can learn
  from the full overlap geometry.

"""


def load_mask(filepath, msk_var='cloudtracknumber'):
    """
    Load a pixel-level track mask from a FLEXTRKR mcstrack file.
    Args:
        filepath (str): Path to mcstrack_YYYYMMDD_HHMM.nc file.
        msk_var (str): Mask variable. Defaults to ``cloudtracknumber``.
    Returns:
        2d array (lat, lon) of integer track IDs, 0 where no storm.
    """
    with xr.open_dataset(filepath) as ds:
        mask = ds[msk_var].isel(time=0).values
    return np.nan_to_num(mask).astype(int)


def extract_objects(mask):
    """
    Extract per-storm boolean footprints from an integer track mask.
    Args:
        mask (array): 2d integer track mask (0 = background).
    Returns:
        Dict of track ID to 2d boolean footprint array.
    """
    return {int(i): mask == i for i in np.unique(mask) if i != 0}


def overlap_fractions(mask_t0, mask_t1):
    """
    Compute soft assignment matrices between storms at two times.

    The forward matrix answers "what fraction of storm i's area at t0
    lies under storm j at t1"; rows sum to <= 1 and the residual is the
    dustbin column (area not covered by any t1 storm, i.e. dissipation).
    The backward matrix is the same from t1's perspective; its dustbin
    holds area not explained by any t0 storm (i.e. new growth/genesis).

    Args:
        mask_t0 (array): 2d integer track mask at time t.
        mask_t1 (array): 2d integer track mask at time t+1.
    Returns:
        ids_t0 (list): Sorted track IDs present at t0.
        ids_t1 (list): Sorted track IDs present at t1.
        forward (array): (n0, n1+1) matrix; last column is the dustbin.
        backward (array): (n1, n0+1) matrix; last column is the dustbin.
    """
    objs_t0 = extract_objects(mask_t0)
    objs_t1 = extract_objects(mask_t1)
    ids_t0 = sorted(objs_t0)
    ids_t1 = sorted(objs_t1)

    overlap = np.zeros((len(ids_t0), len(ids_t1)))
    for a, i in enumerate(ids_t0):
        for b, j in enumerate(ids_t1):
            overlap[a, b] = np.count_nonzero(objs_t0[i] & objs_t1[j])

    area_t0 = np.array([np.count_nonzero(objs_t0[i]) for i in ids_t0])
    area_t1 = np.array([np.count_nonzero(objs_t1[j]) for j in ids_t1])

    forward = np.zeros((len(ids_t0), len(ids_t1) + 1))
    if len(ids_t0):
        forward[:, :-1] = overlap / area_t0[:, None]
        forward[:, -1] = 1.0 - forward[:, :-1].sum(axis=1)

    backward = np.zeros((len(ids_t1), len(ids_t0) + 1))
    if len(ids_t1):
        backward[:, :-1] = overlap.T / area_t1[:, None]
        backward[:, -1] = 1.0 - backward[:, :-1].sum(axis=1)

    return ids_t0, ids_t1, forward, backward


def classify_events(ids_t0, ids_t1, forward, backward, min_frac=0.15):
    """
    Classify t -> t+1 storm transitions from soft assignment matrices.

    A link is counted only where the overlap fraction exceeds min_frac,
    which suppresses the hour-to-hour boundary jitter between coexisting
    neighbor storms. Note a single transition can produce several events
    (e.g., a parent whose area goes mostly to a new child is part of that
    child's split and may itself continue or dissipate).

    Args:
        ids_t0, ids_t1, forward, backward: Output of overlap_fractions().
        min_frac (float): Minimum overlap fraction for a physical link.
    Returns:
        List of event dicts with keys ``kind`` (continuation, split,
        merge, genesis, lysis), ``ids_t0``, ``ids_t1``.
    """
    events = []
    fwd_links = {
        i: [j for b, j in enumerate(ids_t1) if forward[a, b] > min_frac]
        for a, i in enumerate(ids_t0)
    }
    bwd_links = {
        j: [i for a, i in enumerate(ids_t0) if backward[b, a] > min_frac]
        for b, j in enumerate(ids_t1)
    }

    for i, links in fwd_links.items():
        if not links:
            events.append({'kind': 'lysis', 'ids_t0': [i], 'ids_t1': []})
        elif len(links) > 1:
            events.append({'kind': 'split', 'ids_t0': [i], 'ids_t1': links})

    for j, links in bwd_links.items():
        if not links:
            events.append({'kind': 'genesis', 'ids_t0': [], 'ids_t1': [j]})
        elif len(links) > 1:
            events.append({'kind': 'merge', 'ids_t0': links, 'ids_t1': [j]})

    for i, links in fwd_links.items():
        if len(links) == 1 and bwd_links.get(links[0]) == [i]:
            events.append(
                {'kind': 'continuation', 'ids_t0': [i], 'ids_t1': links}
            )

    return events


def build_targets(filepath_t0, filepath_t1, msk_var='cloudtracknumber',
                  min_frac=0.15):
    """
    Build association targets for one consecutive pair of mask files.
    Args:
        filepath_t0 (str): mcstrack file at time t.
        filepath_t1 (str): mcstrack file at time t+1.
        msk_var (str): Mask variable. Defaults to ``cloudtracknumber``.
        min_frac (float): Threshold for event classification only; the
                          returned matrices are not thresholded.
    Returns:
        Dict with ids_t0, ids_t1, forward, backward, events.
    """
    mask_t0 = load_mask(filepath_t0, msk_var=msk_var)
    mask_t1 = load_mask(filepath_t1, msk_var=msk_var)
    ids_t0, ids_t1, forward, backward = overlap_fractions(mask_t0, mask_t1)
    events = classify_events(ids_t0, ids_t1, forward, backward,
                             min_frac=min_frac)
    return {'ids_t0': ids_t0, 'ids_t1': ids_t1,
            'forward': forward, 'backward': backward, 'events': events}


def build_target_sequence(filepaths, msk_var='cloudtracknumber',
                          min_frac=0.15):
    """
    Build association targets for a time-ordered sequence of mask files.
    Args:
        filepaths (list): Time-ordered list of mcstrack file paths.
        msk_var (str): Mask variable. Defaults to ``cloudtracknumber``.
        min_frac (float): Threshold for event classification.
    Returns:
        List of build_targets() dicts, one per consecutive pair.
    """
    masks = [load_mask(f, msk_var=msk_var) for f in filepaths]
    out = []
    for k in range(len(masks) - 1):
        ids_t0, ids_t1, forward, backward = overlap_fractions(
            masks[k], masks[k + 1]
        )
        events = classify_events(ids_t0, ids_t1, forward, backward,
                                 min_frac=min_frac)
        out.append({'ids_t0': ids_t0, 'ids_t1': ids_t1,
                    'forward': forward, 'backward': backward,
                    'events': events})
    return out
