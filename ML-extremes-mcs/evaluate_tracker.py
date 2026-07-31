import argparse
import json
import os

import numpy as np
import torch

import tracking_targets
import tracker_model
import train_tracker

"""

Evaluation for a trained tracker checkpoint: association accuracy,
event-level precision/recall (split/merge/genesis/lysis/continuation),
and probability calibration (reliability diagram, ECE, Brier score) on
held-out years. Metrics are computed teacher-forced (ground-truth storm
footprints), isolating association skill from segmentation skill; the
same protocol as training validation.

Calibration is the load-bearing check for the paper: Lemma 3 predicts
the KL-trained model's probabilities should match outcome frequencies,
and the reliability diagram tests exactly that. A link "outcome" is
defined by thresholding the target mass at the same min_frac used for
event classification, and the model's predicted mass for that cell is
its forecast probability.

"""


def association_metrics(preds, targets):
    """
    Row-level association skill over a set of frame pairs.
    Args:
        preds (list): (forward, backward) torch matrices per pair.
        targets (list): Target dicts with 'forward'/'backward' tensors.
    Returns:
        Dict with top1 agreement and mean row KL (both directions).
    """
    hits, rows, kls = 0, 0, []
    for (pf, pb), tgt in zip(preds, targets):
        for pred, key in [(pf, 'forward'), (pb, 'backward')]:
            t = tgt[key].to(pred.device)
            if pred.shape[0] == 0:
                continue
            hits += int((pred.argmax(1) == t.argmax(1)).sum())
            rows += pred.shape[0]
            kls.append(float(tracker_model.association_kl(pred, t)))
    return {'top1': hits / rows if rows else float('nan'),
            'mean_kl': float(np.mean(kls)) if kls else float('nan'),
            'rows': rows}


def event_key(e):
    return (e['kind'], tuple(sorted(e['ids_t0'])),
            tuple(sorted(e['ids_t1'])))


def event_scores(pred_events, true_events):
    """
    Per-kind precision/recall/F1 comparing predicted vs target events.
    Args:
        pred_events, true_events (list): classify_events() outputs,
            already accumulated over all evaluated pairs.
    Returns:
        Dict kind -> {precision, recall, f1, n_true}.
    """
    kinds = ['continuation', 'split', 'merge', 'genesis', 'lysis']
    pred = {k: set() for k in kinds}
    true = {k: set() for k in kinds}
    for j, e in enumerate(pred_events):
        pred[e['kind']].add(event_key(e) + (e.get('pair', -1),))
    for j, e in enumerate(true_events):
        true[e['kind']].add(event_key(e) + (e.get('pair', -1),))

    out = {}
    for k in kinds:
        tp = len(pred[k] & true[k])
        p = tp / len(pred[k]) if pred[k] else float('nan')
        r = tp / len(true[k]) if true[k] else float('nan')
        f1 = (2 * p * r / (p + r)) if (pred[k] and true[k] and p + r > 0) \
            else float('nan')
        out[k] = {'precision': p, 'recall': r, 'f1': f1,
                  'n_true': len(true[k])}
    return out


def calibration_data(preds, targets, min_frac=0.15):
    """
    Collect (predicted probability, binary outcome) pairs for every
    matrix cell, dustbins included.
    Args:
        preds, targets: As in association_metrics().
        min_frac (float): Threshold defining a true link outcome.
    Returns:
        (probs, outcomes) float arrays of equal length.
    """
    probs, outs = [], []
    for (pf, pb), tgt in zip(preds, targets):
        for pred, key in [(pf, 'forward'), (pb, 'backward')]:
            t = tgt[key].cpu().numpy()
            p = pred.detach().cpu().numpy()
            if p.size == 0:
                continue
            probs.append(p.ravel())
            outs.append((t > min_frac).ravel().astype(float))
    if not probs:
        return np.array([]), np.array([])
    return np.concatenate(probs), np.concatenate(outs)


def expected_calibration_error(probs, outcomes, n_bins=10):
    """
    ECE over equal-width probability bins, plus the diagram data.
    Returns:
        (ece, brier, bin_centers, bin_conf, bin_acc, bin_count)
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
    conf = np.full(n_bins, np.nan)
    acc = np.full(n_bins, np.nan)
    count = np.zeros(n_bins)
    for b in range(n_bins):
        sel = idx == b
        count[b] = sel.sum()
        if count[b]:
            conf[b] = probs[sel].mean()
            acc[b] = outcomes[sel].mean()
    valid = count > 0
    ece = float(np.sum(count[valid] / len(probs)
                       * np.abs(acc[valid] - conf[valid])))
    brier = float(np.mean((probs - outcomes) ** 2))
    centers = (edges[:-1] + edges[1:]) / 2
    return ece, brier, centers, conf, acc, count


def reliability_figure(probs, outcomes, path, n_bins=10):
    """
    Save a reliability diagram (predicted probability vs observed
    frequency) with a histogram of prediction counts.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ece, brier, centers, conf, acc, count = expected_calibration_error(
        probs, outcomes, n_bins
    )
    fig, (ax, hx) = plt.subplots(
        2, 1, figsize=(5, 6), height_ratios=[3, 1], sharex=True
    )
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='perfect calibration')
    ax.plot(conf, acc, 'o-', color='#1a6a9a', label='model')
    ax.set_ylabel('observed link frequency')
    ax.set_title(f'Reliability — ECE {ece:.3f}, Brier {brier:.3f}')
    ax.legend(loc='upper left')
    hx.bar(centers, count, width=0.08, color='#1a6a9a')
    hx.set_yscale('log')
    hx.set_xlabel('predicted link probability')
    hx.set_ylabel('count')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return ece, brier


def evaluate(net, dataset, device='cpu', max_windows=None, min_frac=0.15):
    """
    Run teacher-forced evaluation over a dataset of windows.
    Returns:
        Dict of association metrics, event scores, and calibration
        arrays (probs/outcomes kept for figure generation).
    """
    net.eval()
    preds, targets = [], []
    pred_events, true_events = [], []

    n = len(dataset) if max_windows is None else min(max_windows,
                                                     len(dataset))
    with torch.no_grad():
        for w in range(n):
            item = dataset[w]
            x = train_tracker.pad_lat(item['inputs']).unsqueeze(0).to(device)
            track = train_tracker.pad_lat(item['track_masks']).to(device)
            feats, _ = net(x)
            for k, tgt in enumerate(item['targets']):
                if not tgt['ids_t0'] and not tgt['ids_t1']:
                    continue
                ids0, ids1, pf, pb = net.associate(
                    feats[0, k], feats[0, k + 1], track[k], track[k + 1]
                )
                preds.append((pf, pb))
                targets.append(tgt)
                pe = tracking_targets.classify_events(
                    ids0, ids1, pf.cpu().numpy(), pb.cpu().numpy(),
                    min_frac=min_frac,
                )
                for e in pe:
                    e['pair'] = (w, k)
                for e in tgt['events']:
                    e = dict(e)
                    e['pair'] = (w, k)
                    true_events.append(e)
                pred_events.extend(pe)

    assoc = association_metrics(preds, targets)
    events = event_scores(pred_events, true_events)
    probs, outcomes = calibration_data(preds, targets, min_frac)
    return {'association': assoc, 'events': events,
            'probs': probs, 'outcomes': outcomes,
            'n_windows': n, 'n_pairs': len(preds)}


def main():
    p = argparse.ArgumentParser(description='Evaluate a tracker checkpoint.')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--mask-root', required=True)
    p.add_argument('--era5', required=True)
    p.add_argument('--years', default='2018-2019')
    p.add_argument('--window', type=int, default=3)
    p.add_argument('--stride', type=int, default=2)
    p.add_argument('--mean', type=float, default=None)
    p.add_argument('--std', type=float, default=None)
    p.add_argument('--difference', action='store_true')
    p.add_argument('--max-windows', type=int, default=None)
    p.add_argument('--temporal-mixing', default='none',
                   choices=['none', 'bottleneck'],
                   help='must match the checkpoint being evaluated')
    p.add_argument('--out', default='eval_out')
    args = p.parse_args()

    files = train_tracker.mask_files_for_years(
        args.mask_root, train_tracker.parse_years(args.years)
    )
    dataset = train_tracker.WindowsWithInputs(
        files, args.era5, args.window, args.mean, args.std,
        args.difference, stride=args.stride,
    )
    print(f"evaluating on {len(files)} files -> {len(dataset)} windows",
          flush=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = tracker_model.TrackerNet(
        n_channels=1, n_classes=2, temporal_mixing=args.temporal_mixing
    ).to(device)
    ck = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(ck['model'] if 'model' in ck else ck)
    print(f"loaded {args.checkpoint} (epoch {ck.get('epoch', '?')})",
          flush=True)

    res = evaluate(net, dataset, device, args.max_windows)

    os.makedirs(args.out, exist_ok=True)
    ece, brier = reliability_figure(
        res['probs'], res['outcomes'],
        os.path.join(args.out, 'reliability.png'),
    )

    summary = {'association': res['association'],
               'events': res['events'],
               'ece': ece, 'brier': brier,
               'n_windows': res['n_windows'],
               'n_pairs': res['n_pairs'],
               'checkpoint': args.checkpoint, 'years': args.years}
    with open(os.path.join(args.out, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}/summary.json and reliability.png")


if __name__ == '__main__':
    main()
