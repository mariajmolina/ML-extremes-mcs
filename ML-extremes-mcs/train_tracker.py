import argparse
import datetime
import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import dataset_inputs
import dataset_temporal
import tracker_model

"""

Training script for the v1 probabilistic tracker (TrackerNet).

Combines the temporal mask windows (dataset_temporal), raw-ERA5 input
loading (dataset_inputs), and the two-head model/loss (tracker_model)
into a teacher-forced training loop with year-based train/validation
splits, gradient accumulation, checkpointing, and a --smoke mode for
cheap end-to-end verification before spending GPU hours.

Example (compute normalization constants first, then train):

    python train_tracker.py --compute-stats --mask-root <masks> --era5 <archive>
    python train_tracker.py --mask-root <masks> --era5 <archive> \
        --mean <m> --std <s> --out runs/v1

"""


def mask_files_for_years(mask_root, years):
    """
    Collect mcstrack files for a list of years.
    Args:
        mask_root (str): Directory containing per-year subdirectories.
        years (list): Year integers.
    Returns:
        Sorted list of file paths.
    """
    files = []
    for yr in years:
        files += glob.glob(os.path.join(mask_root, str(yr), 'mcstrack_*.nc'))
    return sorted(files)


def pad_lat(t, total=7):
    """
    Pad the latitude axis 121 -> 128 so the U-Net's four 2x
    downsamplings divide evenly. Works for (..., lat, lon) tensors.
    Args:
        t (tensor): Input with lat as the second-to-last dim.
        total (int): Rows to add. Defaults to 7 (3 south, 4 north).
    """
    return torch.nn.functional.pad(t, (0, 0, total // 2, total - total // 2))


class WindowsWithInputs(Dataset):
    """
    Wraps TemporalMaskDataset, attaching aligned ERA5 input tensors to
    each window. The ERA5 loader is created lazily per process so its
    cached netCDF handle is never shared across DataLoader workers.
    """

    def __init__(self, mask_files, era5_dir, window=3, mean=None,
                 std=None, difference=False, stride=1):
        """
        Initialization.
        Args:
            mask_files (list): mcstrack files; windows form only across
                               exactly-consecutive hours.
            era5_dir (str): ERA5 forecast-table directory (accumu).
            window (int): Frames per sample.
            mean, std (float): z-score constants for the input field.
            difference (bool): Passed to ERA5ForecastLoader.
            stride (int): Window advance in hours (window - 1 covers
                          each transition exactly once).
        """
        self.masks = dataset_temporal.TemporalMaskDataset(
            mask_files, window=window, stride=stride
        )
        self.era5_dir = era5_dir
        self.mean = mean
        self.std = std
        self.difference = difference
        self._loader = None

    def __len__(self):
        return len(self.masks)

    @property
    def loader(self):
        if self._loader is None:
            self._loader = dataset_inputs.ERA5ForecastLoader(
                self.era5_dir, mean=self.mean, std=self.std,
                difference=self.difference,
            )
        return self._loader

    def __getitem__(self, index):
        item = self.masks[index]
        item['inputs'] = torch.from_numpy(
            self.loader.window(item['times'])
        ).float()
        return item


def run_epoch(net, data, optimizer=None, lam=1.0, device='cpu',
              accum=1, max_steps=None, log_every=50):
    """
    One pass over the data; trains when an optimizer is given.
    Returns:
        (mean_loss, association_top1) over processed windows.
    """
    training = optimizer is not None
    net.train(training)
    losses, hits, pairs = [], 0, 0

    for step, item in enumerate(data):
        if max_steps is not None and step >= max_steps:
            break

        x = pad_lat(item['inputs']).unsqueeze(0).to(device)
        track = pad_lat(item['track_masks']).to(device)
        binary = (track > 0).long()

        with torch.set_grad_enabled(training):
            feats, logits = net(x)
            preds, targets = [], []
            for k in range(track.shape[0] - 1):
                tgt = item['targets'][k]
                if not tgt['ids_t0']:
                    continue
                _, _, pf, pb = net.associate(
                    feats[0, k], feats[0, k + 1], track[k], track[k + 1]
                )
                preds.append((pf, pb))
                targets.append(tgt)
                agree = (pf.argmax(dim=1)
                         == tgt['forward'].to(pf.device).argmax(dim=1))
                hits += int(agree.sum())
                pairs += agree.numel()

            loss = tracker_model.tracking_loss(
                logits[0], binary, preds, targets, lam=lam
            )

        if training:
            (loss / accum).backward()
            if (step + 1) % accum == 0:
                optimizer.step()
                optimizer.zero_grad()

        losses.append(float(loss))
        if training and step % log_every == 0:
            print(f"  step {step:6d}  loss {np.mean(losses[-log_every:]):.4f}",
                  flush=True)

    top1 = hits / pairs if pairs else float('nan')
    return float(np.mean(losses)) if losses else float('nan'), top1


def parse_years(spec):
    """Parse '2004-2015' or '2016,2017' into a list of ints."""
    if '-' in spec:
        a, b = spec.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in spec.split(',')]


def main():
    p = argparse.ArgumentParser(description='Train the v1 tracker.')
    p.add_argument('--mask-root', required=True,
                   help='dir with per-year mcstrack subdirectories')
    p.add_argument('--era5', required=True,
                   help='ERA5 e5.oper.fc.sfc.accumu directory')
    p.add_argument('--out', default='runs/v1', help='output directory')
    p.add_argument('--train-years', default='2004-2015')
    p.add_argument('--valid-years', default='2016-2017')
    p.add_argument('--window', type=int, default=3)
    p.add_argument('--stride', type=int, default=1,
                   help='window advance in hours; window-1 covers each '
                        'transition once (cheaper epochs)')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lam', type=float, default=1.0,
                   help='association loss weight')
    p.add_argument('--accum', type=int, default=8,
                   help='gradient accumulation steps')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--mean', type=float, default=None)
    p.add_argument('--std', type=float, default=None)
    p.add_argument('--difference', action='store_true',
                   help='difference cumulative ERA5 accumulations')
    p.add_argument('--init-backbone', default=None,
                   help='optional unet state_dict to initialize from')
    p.add_argument('--resume', default=None, help='checkpoint to resume')
    p.add_argument('--smoke', action='store_true',
                   help='tiny end-to-end run: few steps, no epochs loop')
    p.add_argument('--compute-stats', action='store_true',
                   help='sample training times, print mean/std, exit')
    args = p.parse_args()

    train_files = mask_files_for_years(args.mask_root,
                                       parse_years(args.train_years))
    valid_files = mask_files_for_years(args.mask_root,
                                       parse_years(args.valid_years))
    print(f"train mask files: {len(train_files)} | "
          f"valid mask files: {len(valid_files)}", flush=True)

    if args.compute_stats:
        loader = dataset_inputs.ERA5ForecastLoader(
            args.era5, difference=args.difference
        )
        times = sorted(random.Random(0).sample(
            [dataset_temporal.timestamp_of(f) for f in train_files],
            min(500, len(train_files)),
        ))
        mean, std = loader.compute_stats(times)
        print(f"--mean {mean:.6g} --std {std:.6g}")
        return

    train_ds = WindowsWithInputs(train_files, args.era5, args.window,
                                 args.mean, args.std, args.difference,
                                 stride=args.stride)
    valid_ds = WindowsWithInputs(valid_files, args.era5, args.window,
                                 args.mean, args.std, args.difference,
                                 stride=args.stride)
    print(f"train windows: {len(train_ds)} | valid windows: {len(valid_ds)}",
          flush=True)

    def one(batch):
        return batch[0]

    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True,
                          num_workers=args.workers, collate_fn=one)
    valid_dl = DataLoader(valid_ds, batch_size=1, shuffle=False,
                          num_workers=args.workers, collate_fn=one)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device, flush=True)
    net = tracker_model.TrackerNet(n_channels=1, n_classes=2).to(device)

    if args.init_backbone:
        net.backbone.load_state_dict(
            torch.load(args.init_backbone, map_location=device)
        )
        print('backbone initialized from', args.init_backbone, flush=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    start_epoch, best = 0, float('inf')
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        net.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        start_epoch, best = ck['epoch'] + 1, ck['best']
        print(f"resumed from {args.resume} at epoch {start_epoch}", flush=True)

    os.makedirs(args.out, exist_ok=True)

    if args.smoke:
        loss, top1 = run_epoch(net, train_dl, optimizer, args.lam, device,
                               accum=2, max_steps=10, log_every=1)
        print(f"SMOKE OK  loss {loss:.4f}  assoc-top1 {top1:.3f}", flush=True)
        return

    for epoch in range(start_epoch, args.epochs):
        print(f"epoch {epoch}", flush=True)
        tr_loss, tr_top1 = run_epoch(net, train_dl, optimizer, args.lam,
                                     device, accum=args.accum)
        with torch.no_grad():
            va_loss, va_top1 = run_epoch(net, valid_dl, None, args.lam,
                                         device)
        print(f"epoch {epoch}  train {tr_loss:.4f}/{tr_top1:.3f}  "
              f"valid {va_loss:.4f}/{va_top1:.3f}", flush=True)

        state = {'model': net.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch, 'best': best,
                 'valid_loss': va_loss, 'valid_top1': va_top1}
        torch.save(state, os.path.join(args.out, 'last.pt'))
        if va_loss < best:
            best = va_loss
            state['best'] = best
            torch.save(state, os.path.join(args.out, 'best.pt'))
            print(f"  new best ({best:.4f}) saved", flush=True)


if __name__ == '__main__':
    main()
