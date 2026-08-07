import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import evaluate_tracker as ev

"""

Tests for evaluate_tracker.py metric functions (pure computation, no
model or data files needed).

"""


def rows(*r):
    return torch.tensor(r, dtype=torch.float32)


def test_association_metrics_perfect():
    t = {'forward': rows([0.7, 0.2, 0.1]), 'backward': rows([1.0, 0.0])}
    preds = [(t['forward'].clone(), t['backward'].clone())]
    m = ev.association_metrics(preds, [t])
    assert m['top1'] == 1.0
    assert m['mean_kl'] < 1e-6
    assert m['rows'] == 2


def test_association_metrics_wrong_argmax():
    t = {'forward': rows([0.9, 0.1]), 'backward': rows([0.9, 0.1])}
    preds = [(rows([0.2, 0.8]), rows([0.2, 0.8]))]
    m = ev.association_metrics(preds, [t])
    assert m['top1'] == 0.0
    assert m['mean_kl'] > 0.5


def test_event_scores_matching():
    true = [{'kind': 'split', 'ids_t0': [76], 'ids_t1': [76, 78], 'pair': 0},
            {'kind': 'continuation', 'ids_t0': [5], 'ids_t1': [5], 'pair': 0}]
    pred = [{'kind': 'split', 'ids_t0': [76], 'ids_t1': [78, 76], 'pair': 0},
            {'kind': 'lysis', 'ids_t0': [5], 'ids_t1': [], 'pair': 0}]
    s = ev.event_scores(pred, true)
    # split matched despite child order; continuation missed (called lysis)
    assert s['split']['precision'] == 1.0 and s['split']['recall'] == 1.0
    assert s['continuation']['recall'] == 0.0
    assert np.isnan(s['merge']['recall'])  # no true merges -> nan
    assert s['lysis']['precision'] == 0.0  # predicted lysis was wrong


def test_calibration_perfect_and_ece():
    # construct predictions equal to outcomes' frequencies per bin:
    # 100 cells at p=0.8 with 80 links, 100 at p=0.2 with 20 links
    probs = np.array([0.8] * 100 + [0.2] * 100)
    outcomes = np.array([1.0] * 80 + [0.0] * 20 + [1.0] * 20 + [0.0] * 80)
    ece, brier, centers, conf, acc, count = \
        ev.expected_calibration_error(probs, outcomes)
    assert ece < 1e-6
    # miscalibrated: same predictions, all outcomes zero
    ece2, *_ = ev.expected_calibration_error(probs,
                                             np.zeros_like(outcomes))
    assert ece2 > 0.4


def test_calibration_data_soft_outcomes():
    t = {'forward': rows([0.7, 0.2, 0.1]), 'backward': rows([1.0, 0.0])}
    preds = [(t['forward'].clone(), t['backward'].clone())]
    probs, outs = ev.calibration_data(preds, [t])
    assert probs.shape == outs.shape == (5,)
    # outcomes are the soft target masses themselves, NOT thresholded
    assert np.allclose(sorted(outs), [0.0, 0.1, 0.2, 0.7, 1.0])
    # a model predicting exactly the targets is perfectly calibrated
    ece, *_ = ev.expected_calibration_error(probs, outs)
    assert ece < 1e-6


def test_reliability_figure_writes(tmp_path):
    probs = np.random.RandomState(0).rand(500)
    outcomes = (np.random.RandomState(1).rand(500) < probs).astype(float)
    path = str(tmp_path / 'rel.png')
    ece, brier = ev.reliability_figure(probs, outcomes, path)
    assert os.path.exists(path)
    assert 0 <= ece < 0.2  # outcomes drawn from probs -> near-calibrated
