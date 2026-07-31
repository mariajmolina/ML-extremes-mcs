# Probabilistic MCS Tracking — Project Handoff

State of the work as of 2026-07-31, written so anyone (including future
contributors with no context) can pick it up without re-deriving decisions.

## 1. What this project is

A neural spatiotemporal storm-tracking framework producing **probabilistic**
storm tracks — uncertainty over track continuity, genesis, lysis, splits, and
merges — trained on FLEXTRKR-labeled ERA5 data, intended for later transfer to
km-scale model output (DYAMOND/CESM). The existing per-frame U-Net detection
work in this repo is the baseline; the tracking framework is the paper.

Confirmed novelty (lit search, 2026-07): no published learned probabilistic
storm/MCS association. Closest prior art a reviewer will cite:
- **Makris & Prieur, IEEE TGRS 2014** — Bayesian MHT tracker with explicit
  genesis/lysis/split/merge for convective clouds. Hand-designed, not learned.
  Frame novelty as "first *learned* probabilistic MCS association," never
  "first probabilistic MCS tracking." Read in full before writing.
- **GNN-DOL, Scientific Reports 2023** — learned assignment with division
  constraints, biological cell tracking (mechanism analogue).
- **SuperGlue, CVPR 2020** — origin of the dustbin mechanism (cite as source).
- **Prein et al., JGR 2024 (10.1029/2023JD040254)** — deterministic MCS
  trackers disagree with each other; the paper's best motivation citation.

## 2. Architecture of the code (PR stack)

All on the fork `shaheen-bhattacharya/ML-extremes-mcs`, draft PRs against
`mariajmolina/ML-extremes-mcs`:

| PR | branch | contents |
|----|--------|----------|
| #1 | `tracking-association-targets` | `tracking_targets.py` — soft assignment matrices (forward/backward overlap fractions w/ dustbins) + event classifier |
| #2 | `temporal-window-dataset` | `dataset_temporal.py` — gap-aware consecutive-hour windows, stride support, collate fn |
| #3 | `tracker-model-v1` | `tracker_model.py` — TrackerNet (repo U-Net backbone reused unmodified + mask pooling + AssociationHead), Dice + row-KL losses |
| #4 | `era5-input-loader` | `dataset_inputs.py` — reads raw ERA5 from permanent RDA archive (standalone PR) |
| #5 | `training-v1` (merge of #3+#4) | `train_tracker.py`, `jobs/train_derecho_gpu.sh` |
| #6 | `evaluation-v1` (stacked on #5) | `evaluate_tracker.py` — association skill, event P/R, calibration/ECE |

Tests: `tests/` — 52 passing (pytest). Run from `ML-extremes-mcs/` module dir.
Demo: `pipeline_testdrive.ipynb` walks the whole pipeline on real files.

## 3. Key design decisions and WHY (do not re-litigate without data)

1. **Track-ID equality is NOT the supervision signal.** Verified on 2005
   masks: FLEXTRKR relays IDs across physically continuous storms (parent
   splits off child, later merges into it — e.g. tracks 76→78 on 2005-06-01),
   and the surviving ID does not reliably keep the larger split piece
   (2005-06-11: parent kept 20px, new ID got 382px). Targets are built from
   **overlap geometry**; IDs only delineate objects within a frame.
2. **Soft matrices, thresholded events.** Neighboring storms exchange tiny
   boundary overlaps (~IoU 0.01) hourly that are noise; real events show
   fractions ≳0.2. Event *labels* use min_frac=0.15 (inside the empirically
   empty gap); the *training targets* are unthresholded.
3. **No advection correction (v1).** 16-year self-overlap scan (155,179
   storm-hours, 2004–2019): median hourly self-overlap 0.824, stable every
   year; only 0.85% of storm-hours fall below 0.15 — half genuine small
   fast-movers, half split/merge handoffs. Documented refinement, not a
   prerequisite. (Scan script: `~/selfoverlap_scan.py` on glade; records CSV
   in `sample_data/` locally and `~/selfoverlap_records.csv` on glade.)
4. **Raw RDA ERA5, not preprocessed files.** The old per-ID training files
   (`dl_files/1H`) were purged from scratch. `dataset_inputs.py` reads the
   permanent archive directly. Two facts learned the hard way:
   - chunk filename end-dates are **exclusive** (the end-labeled init lives
     in the next file);
   - RDA's TTR values are **already hourly** (flat across forecast hours) —
     never pass `--difference`.
5. **KL loss on row-stochastic targets** is a proper scoring rule → the
   optimal prediction is the outcome frequency → calibration is a testable
   prediction, not a hope. See `sample_data/lemmas.tex` (laptop) for the
   formal writeup; `architecture.tex` and `tracking_targets_doc.tex` sit
   beside it (methods-section material).

## 4. Environments and data

- **Laptop venv**: `~/mcs_venv` (python 3.12; xarray, netCDF4, torch,
  matplotlib, pytest, ipykernel). Kernel path for VS Code:
  `~/mcs_venv/bin/python`.
- **NCAR env**: `/glade/work/sbhatta/conda-envs/mcs` (python 3.11, torch
  2.13+cu130 via pip). `module load conda && conda activate <path>`.
- **NCAR repo clone**: `~/repo-mcs` (glade home).
- **Masks (labels)**: `/glade/derecho/scratch/molina/cesm_mcs/mcs_flextrkr_era5/mcstracking_3pctl/<year>/mcstrack_YYYYMMDD_HHMM.nc`,
  years 2004–2019, hourly Mar–Oct, 121×321 grid (20–50N, 220–300E, lat
  ascending). **Scratch purges** — if missing, ask about the canonical copy.
  No FLEXTRKR trackstats files exist (confirmed) — split/merge history must
  be derived from masks, which is what the target builder does.
- **ERA5 inputs**: `/glade/campaign/collections/rda/data/d633000/e5.oper.fc.sfc.accumu/`
  (permanent). TTR = OLR-like; loader negates it.
- **Normalization constants** (train years 2004–2015 sample):
  `--mean 916080 --std 139413` (≈254 W/m² mean OLR — physically sane).
- **Local sample data**: `~/ML-extremes-mcs/sample_data/` — 5 days of 2005
  masks (June 1/3/11/24, July 25) with verified split/merge case studies.
- **Allocation**: project UMCP0056, ends 2027-01-31 (hours expire then).
  Derecho-GPU 1,000 h; first training run used only a few.

## 5. Runbook

Train (first real run used exactly this):
```
qsub -v MEAN=916080,STD=139413,TRAIN_YEARS=2004-2007,VALID_YEARS=2016,STRIDE=2,EPOCHS=5 \
     -l walltime=12:00:00 jobs/train_derecho_gpu.sh
```
Resume after walltime: add `RESUME=/glade/work/sbhatta/mcs_runs/v1/last.pt`.
Smoke test any change first: `qsub -v SMOKE=1,MEAN=...,STD=... -l walltime=00:20:00 ...`.

Evaluate a checkpoint (see PR #6 body for the inline-qsub variant):
```
python evaluate_tracker.py --checkpoint /glade/work/sbhatta/mcs_runs/v1/best.pt \
  --mask-root <masks> --era5 <archive> --years 2017 \
  --mean 916080 --std 139413 --out eval_v1
```
Outputs `summary.json` + `reliability.png`.

**Year hygiene**: train 2004–2015, validate 2016–2017, TEST 2018–2019.
Never touch 2018–2019 until final paper numbers.

Monitoring: `qstat -u sbhatta`; job logs stream live (`-k oed` is set);
`qhist -u sbhatta --days N` for charges; sam.ucar.edu for balances.

## 6. Results so far

- **v1 training run** (job 6926975, 2026-07-27): 4 train years, stride 2,
  5 epochs, 12 h walltime, single A100. Final: train loss 0.542 / top-1
  0.955; **valid (2016) loss 0.513 / top-1 0.948**. Still improving at the
  last epoch — more epochs/years is easy upside. Checkpoints:
  `/glade/work/sbhatta/mcs_runs/v1/{best,last}.pt`.
- **Evaluation of scout v1 on 2017** (eval_v1/, 2026-07-31): association
  top-1 0.952, mean KL 0.056 (21,619 rows). Event F1 — continuation 0.96,
  lysis 0.82, genesis 0.81, **split 0.19, merge 0.18** (recall ~0.5,
  precision ~0.11 — over-predicts rare events ~4:1). **ECE 0.168**,
  Brier 0.128 — not yet calibrated. Diagnosis: class imbalance (123 true
  splits vs 10,050 continuations in 2017) + no post-hoc calibration.
- **v1-full** (12 train years, 10 epochs, job 6965179, 5.5 GPU-h):
  valid (2016) loss 0.502 / top-1 0.962 — better than scout. 2017 eval
  pending (`eval_v1full/`). This is the paper-model candidate.
- **v1.5 ablation** (job 6965180, matched to scout settings): valid
  0.535 / 0.949 vs scout v1's 0.513 / 0.948 — **bottleneck temporal
  attention gives no benefit** at these settings. Useful negative
  result: justifies the simpler v1 architecture; deprioritizes v2.
- Total GPU spend through all of the above: ~7.5 of 1,000 hours.

### Priority remedies for the two weaknesses (in order)
1. Evaluate v1-full on 2017 (likely free improvement on all metrics).
2. **Temperature scaling** for calibration: single scalar T dividing the
   association logits before softmax, fitted on validation (2016) to
   minimize NLL/ECE, applied at eval. Standard, cheap, often halves ECE.
3. **Rare-event reweighting**: upweight association-loss rows whose
   target is a split/merge (e.g. weight rows with >=2 above-threshold
   entries, or weight by 1/kind-frequency); alternatively raise --lam.
   Retrain, re-evaluate split/merge F1.
4. If split/merge F1 stays low after 2-3: paper narrows to calibrated
   continuity/genesis/lysis with split/merge as characterized future
   work — narrowing, not scrapping; the core result (learned
   probabilistic association works) already stands.

## 7. What's next, in order, with specs

1. **Read the 2017 evaluation.** If ECE is small and split/merge recall is
   reasonable, the core claim stands; the reliability diagram is the paper's
   central figure. If split/merge skill is weak: try lam > 1 (association
   loss weight), longer training, more years — in that order.
2. **v1.5 — bottleneck temporal attention.** Spec: at the U-Net bottleneck
   (after `down4`, ~(B*T, 512, 8, 20) for padded input), reshape to put T on
   a sequence axis per spatial location, apply 1–2 layers of multi-head
   self-attention across T only (not space), add residually, continue the
   decoder unchanged. Gate behind `--temporal-mixing {none,bottleneck}` in
   TrackerNet/train_tracker so v1 remains reproducible. Ablation = same run
   config, one flag changed.
3. **The stricter test — inference-mode evaluation.** Spec: segmentation
   softmax → storm mask (argmax or p>0.5) → `scipy.ndimage.label` for
   connected components (consider dropping objects < ~20px) → mask-pool on
   *predicted* footprints → associate. Score by first matching predicted to
   true objects per frame (greedy IoU matching, threshold ~0.3), then event
   P/R as in evaluate_tracker. New module (e.g. `inference_tracker.py`), keep
   teacher-forced eval untouched for comparison. This measures the detection→
   association error cascade — a paper section of its own.
4. **DYAMOND/CESM transfer** (the abstract's promise): run inference-mode
   tracking on the MCSMIP files the repo's DYAMOND notebooks target. No
   labels there — evaluation is qualitative + statistics comparisons
   (life-cycle distributions vs FLEXTRKR-on-OBS).
5. **Paper assembly.** LaTeX building blocks live in laptop
   `sample_data/`: `architecture.tex`, `lemmas.tex`, `tracking_targets_doc.tex`.
   Venue candidates discussed: AMS AIES / JAMES (teacher's community), IEEE
   TGRS (Makris & Prieur lineage), NeurIPS Climate Change AI workshop
   (fast, prestigious-enough). Teacher decides.

## 8. Gotchas

- PBS logs normally appear only at job end; `-k oed` (already in the job
  script) streams them live. Derecho has no qpeek/qcat.
- `.gitignore` excludes `*.pbs` — job scripts are named `.sh` on purpose.
- `config.py` is gitignored AND tracked-legacy; its `/glade/scratch` paths
  are dead (filesystem renamed `/glade/derecho/scratch`).
- Everything under `molina/` directories is read-only by convention: pull
  data from there, never write.
- The repo has substantial legacy code (TF-era `previous_tf_and_003/`,
  notebooks with broken imports). It is intentionally left untouched;
  additive changes only.
- Old per-ID pipeline (`file_creation.py`, `dataset.py`) flips X but not y:
  that is CORRECT (ERA5 native lat is descending, masks ascending) — do not
  "fix" it. `dataset_inputs.py` handles the same flip explicitly.
