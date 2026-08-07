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


def calibration_data(preds, targets, min_frac=None):
    """
    Collect (predicted probability, target mass) pairs for every matrix
    cell, dustbins included.

    Outcomes are the SOFT target masses, not thresholded booleans: the
    targets are overlap fractions, and per the proper-scoring-rule
    argument the model estimates their conditional mean -- so
    calibration means E[target | prediction = p] = p. (An earlier
    version binarized targets at 0.15, which manufactured apparent
    under-confidence -- a correct prediction of a 0.2-mass overlap was
    scored against an "outcome" of 1.0 -- and made ECE insensitive to
    both the model and the softmax temperature.)

    Args:
        preds, targets: As in association_metrics().
        min_frac: Unused; retained for call-site compatibility.
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
            outs.append(np.clip(t, 0.0, 1.0).ravel())
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
    ax.set_ylabel('mean observed target mass')
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


def evaluate(net, dataset, device='cpu', max_windows=None, min_frac=0.15,
             temperature=1.0):
    """
    Run teacher-forced evaluation over a dataset of windows.
    Args:
        temperature (float): Softmax temperature applied to association
            logits (fit on validation via --fit-temperature; 1.0 = raw).
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
                    feats[0, k], feats[0, k + 1], track[k], track[k + 1],
                    temperature=temperature,
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


def fit_temperature(net, dataset, device='cpu', max_windows=300,
                    min_frac=0.15,
                    grid=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                          1.0, 1.2, 1.5)):
    """
    Grid-search the softmax temperature minimizing ECE on a (validation)
    dataset. The expensive U-Net encoding runs once per window; the
    cheap association head is re-run per candidate temperature.
    Returns:
        (best_T, {T: ece}) with diagnostics printed per grid point.
    """
    net.eval()
    per_t = {t: ([], []) for t in grid}
    n = min(max_windows, len(dataset))
    with torch.no_grad():
        for w in range(n):
            item = dataset[w]
            x = train_tracker.pad_lat(item['inputs']).unsqueeze(0).to(device)
            track = train_tracker.pad_lat(item['track_masks']).to(device)
            feats, _ = net(x)
            for k, tgt in enumerate(item['targets']):
                if not tgt['ids_t0'] and not tgt['ids_t1']:
                    continue
                for t in grid:
                    _, _, pf, pb = net.associate(
                        feats[0, k], feats[0, k + 1],
                        track[k], track[k + 1], temperature=t,
                    )
                    p, o = calibration_data([(pf, pb)], [tgt], min_frac)
                    if p.size:
                        per_t[t][0].append(p)
                        per_t[t][1].append(o)

    eces = {}
    for t in grid:
        probs = np.concatenate(per_t[t][0])
        outs = np.concatenate(per_t[t][1])
        ece, brier, *_ = expected_calibration_error(probs, outs)
        eces[t] = ece
        print(f"  T={t:.2f}  ECE {ece:.4f}  Brier {brier:.4f}", flush=True)
    best = min(eces, key=eces.get)
    print(f"best temperature: {best} (ECE {eces[best]:.4f})", flush=True)
    return best, eces


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
    p.add_argument('--temperature', type=float, default=1.0,
                   help='softmax temperature for association logits '
                        '(fit on validation via --fit-temperature)')
    p.add_argument('--channels', default=None,
                   help='JSON channel-spec file; must match the one the '
                        'checkpoint was trained with')
    p.add_argument('--fit-temperature', action='store_true',
                   help='grid-search T minimizing ECE on --years '
                        '(use validation years!), print best, exit')
    p.add_argument('--out', default='eval_out')
    args = p.parse_args()

    channel_specs = None
    if args.channels:
        with open(args.channels) as fh:
            channel_specs = json.load(fh)
    factory = (train_tracker.channels_factory(channel_specs)
               if channel_specs is not None else None)
    n_channels = len(channel_specs) if channel_specs is not None else 1

    files = train_tracker.mask_files_for_years(
        args.mask_root, train_tracker.parse_years(args.years)
    )
    dataset = train_tracker.WindowsWithInputs(
        files, args.era5, args.window, args.mean, args.std,
        args.difference, stride=args.stride, loader_factory=factory,
    )
    print(f"evaluating on {len(files)} files -> {len(dataset)} windows "
          f"({n_channels} channel(s))", flush=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = tracker_model.TrackerNet(
        n_channels=n_channels, n_classes=2,
        temporal_mixing=args.temporal_mixing
    ).to(device)
    ck = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(ck['model'] if 'model' in ck else ck)
    print(f"loaded {args.checkpoint} (epoch {ck.get('epoch', '?')})",
          flush=True)

    if args.fit_temperature:
        fit_temperature(net, dataset, device,
                        max_windows=args.max_windows or 300)
        return

    res = evaluate(net, dataset, device, args.max_windows,
                   temperature=args.temperature)

    os.makedirs(args.out, exist_ok=True)
    ece, brier = reliability_figure(
        res['probs'], res['outcomes'],
        os.path.join(args.out, 'reliability.png'),
    )

    summary = {'association': res['association'],
               'events': res['events'],
               'ece': ece, 'brier': brier,
               'temperature': args.temperature,
               'n_windows': res['n_windows'],
               'n_pairs': res['n_pairs'],
               'checkpoint': args.checkpoint, 'years': args.years}
    with open(os.path.join(args.out, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}/summary.json and reliability.png")


if __name__ == '__main__':
    main()
