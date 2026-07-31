import os
import sys
import glob

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tracking_targets as tt

"""

Tests for tracking_targets.py.

The synthetic cases reproduce storm-transition geometries observed in
the 2005 FLEXTRKR ERA5 masks (mcstracking_3pctl), including the ID-relay
and boundary-jitter behaviors that motivated the module's design. The
integration tests run against real mcstrack files when a local
sample_data/ directory is present (see repo README) and are skipped
otherwise.

"""

SAMPLE_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'sample_data'
)


def blank():
    return np.zeros((40, 60), dtype=int)


def test_continuation_simple():
    m0, m1 = blank(), blank()
    m0[10:20, 10:20] = 5
    m1[11:21, 12:22] = 5  # same storm, advected slightly
    ids0, ids1, fwd, bwd = tt.overlap_fractions(m0, m1)
    events = tt.classify_events(ids0, ids1, fwd, bwd)
    assert [e['kind'] for e in events] == ['continuation']
    # most of the area maps across, remainder goes to the dustbin
    assert fwd[0, 0] > 0.5
    assert np.isclose(fwd[0].sum(), 1.0)


def test_split_one_to_many():
    # one storm at t covers the footprints of two storms at t+1
    # (June 1 2005: id 76 -> ids 76 + 78, near-even split)
    m0, m1 = blank(), blank()
    m0[10:30, 10:30] = 76
    m1[10:30, 10:19] = 76
    m1[10:30, 21:30] = 78
    ids0, ids1, fwd, bwd = tt.overlap_fractions(m0, m1)
    events = tt.classify_events(ids0, ids1, fwd, bwd)
    kinds = sorted(e['kind'] for e in events)
    assert 'split' in kinds
    split = next(e for e in events if e['kind'] == 'split')
    assert split['ids_t0'] == [76]
    assert sorted(split['ids_t1']) == [76, 78]


def test_split_new_id_gets_larger_piece():
    # June 11 2005: parent 92 kept a 20px rump, new id 97 got the
    # dominant piece. No largest-object assumption should be made.
    m0, m1 = blank(), blank()
    m0[10:26, 10:26] = 92
    m1[10:15, 10:26] = 92   # smaller piece keeps the parent ID (~31%)
    m1[16:26, 10:26] = 97   # larger piece gets the new ID (~62%)
    ids0, ids1, fwd, bwd = tt.overlap_fractions(m0, m1)
    events = tt.classify_events(ids0, ids1, fwd, bwd)
    split = next(e for e in events if e['kind'] == 'split')
    assert sorted(split['ids_t1']) == [92, 97]
    # forward mass toward the NEW id exceeds mass toward the parent id
    assert fwd[0, ids1.index(97)] > fwd[0, ids1.index(92)]


def test_merge_many_to_one():
    # June 3 2005: id 84 dies into id 83
    m0, m1 = blank(), blank()
    m0[10:20, 10:20] = 83
    m0[10:20, 22:32] = 84
    m1[10:20, 10:30] = 83
    ids0, ids1, fwd, bwd = tt.overlap_fractions(m0, m1)
    events = tt.classify_events(ids0, ids1, fwd, bwd)
    merge = next(e for e in events if e['kind'] == 'merge')
    assert sorted(merge['ids_t0']) == [83, 84]
    assert merge['ids_t1'] == [83]


def test_genesis_and_lysis():
    m0, m1 = blank(), blank()
    m0[5:10, 5:10] = 1     # dies with no successor
    m1[25:32, 40:50] = 2   # appears with no predecessor
    ids0, ids1, fwd, bwd = tt.overlap_fractions(m0, m1)
    events = tt.classify_events(ids0, ids1, fwd, bwd)
    kinds = sorted(e['kind'] for e in events)
    assert kinds == ['genesis', 'lysis']
    # lysis storm's forward mass is all dustbin
    assert np.isclose(fwd[0, -1], 1.0)
    assert np.isclose(bwd[0, -1], 1.0)


def test_boundary_jitter_not_an_event():
    # coexisting neighbors exchanging a sliver of pixels (~IoU 0.01)
    # must classify as two continuations, not a split or merge
    m0, m1 = blank(), blank()
    m0[10:30, 10:20] = 1
    m0[10:30, 20:30] = 2
    m1[10:30, 10:21] = 1   # id 1 nudges one column into id 2's area
    m1[10:30, 21:30] = 2
    ids0, ids1, fwd, bwd = tt.overlap_fractions(m0, m1)
    events = tt.classify_events(ids0, ids1, fwd, bwd)
    kinds = sorted(e['kind'] for e in events)
    assert kinds == ['continuation', 'continuation']


def test_soft_matrices_unthresholded():
    # thresholding applies to event classification only: jitter overlap
    # must still be present in the soft matrix for the model to learn from
    m0, m1 = blank(), blank()
    m0[10:30, 10:20] = 1
    m0[10:30, 20:30] = 2
    m1[10:30, 10:21] = 1
    m1[10:30, 21:30] = 2
    ids0, ids1, fwd, bwd = tt.overlap_fractions(m0, m1)
    assert fwd[ids0.index(2), ids1.index(1)] > 0


def test_empty_masks():
    ids0, ids1, fwd, bwd = tt.overlap_fractions(blank(), blank())
    assert ids0 == [] and ids1 == []
    assert tt.classify_events(ids0, ids1, fwd, bwd) == []


@pytest.mark.skipif(
    not glob.glob(os.path.join(SAMPLE_DIR, 'mcstrack_20050601_*.nc')),
    reason='local sample_data/ with June 1 2005 mcstrack files not found'
)
def test_real_june1_relay():
    # June 1 2005: 76 splits -> (76, 78) at 09z, 76 merges into 77 at
    # 13z, 77 merges into 78 at 16z (the "ID relay" pattern)
    files = sorted(glob.glob(
        os.path.join(SAMPLE_DIR, 'mcstrack_20050601_*.nc')
    ))
    seq = tt.build_target_sequence(files)

    def kinds_at(hour):
        return {(e['kind'], tuple(e['ids_t0']), tuple(e['ids_t1']))
                for e in seq[hour]['events']}

    assert ('split', (76,), (76, 78)) in kinds_at(8)    # 08z -> 09z
    assert ('merge', (76, 77), (77,)) in kinds_at(12)   # 12z -> 13z
    assert ('merge', (77, 78), (78,)) in kinds_at(15)   # 15z -> 16z


@pytest.mark.skipif(
    not glob.glob(os.path.join(SAMPLE_DIR, 'mcstrack_20050611_*.nc')),
    reason='local sample_data/ with June 11 2005 mcstrack files not found'
)
def test_real_june11_new_id_dominant():
    # June 11 2005 06z -> 07z: parent 92 keeps 20px, new id 97 gets 382px
    f0 = os.path.join(SAMPLE_DIR, 'mcstrack_20050611_0600.nc')
    f1 = os.path.join(SAMPLE_DIR, 'mcstrack_20050611_0700.nc')
    out = tt.build_targets(f0, f1)
    fwd = out['forward']
    row = out['ids_t0'].index(92)
    assert fwd[row, out['ids_t1'].index(97)] > fwd[row, out['ids_t1'].index(92)]
