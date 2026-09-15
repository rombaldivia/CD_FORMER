# Official NTU120 training and temporal reframing

This workflow updates the supplied center-window trainer. It does not attach
new accuracy claims to the historical repository results.

## Data separation

| Protocol | Official training membership | Official test membership |
| --- | --- | --- |
| XSUB | PySKL `xsub_train`, checked against the NTU120 training subject list | PySKL `xsub_val` |
| XSET | PySKL `xset_train`, checked for even-numbered setups | PySKL `xset_val`, odd-numbered setups |

Definitions follow [PySKL's NTU preprocessing source](https://github.com/kennymckormick/pyskl/blob/main/tools/data/ntu_preproc.py).
The script uses the retained clips in the supplied preprocessed file. It rejects
duplicate IDs, missing annotations, overlapping partitions, and metadata that
disagrees with the official subject/setup rule. It never falls back to a random
split of the full dataset.

Only official training is divided: 90% for optimization and 10% for internal
validation, stratified by action with split seed 42. This internal split is an
experimental choice, not an additional official NTU protocol. It is not claimed
to hold out subjects or camera recordings internally. The complete official test
partition remains the final evaluation set. Checkpoint selection and early stopping
use internal validation. Class weights use optimization samples only.

The saved `splits.json` contains all sample IDs, actual counts, and a fingerprint.
Keep the split seed and validation fraction fixed across frame lengths and model
initialization seeds. A separate model is trained for each protocol. Checkpoint
metadata rejects transfer between protocols or different internal splits. Existing
weights without provenance need an independent audit; a fresh T16 run establishes
the starting point for this workflow.

## Model and training settings

| Setting | Trainer | Kaggle launcher |
| --- | --- | --- |
| Frame lengths | 16, 24 or 32 per run | 16, 24 and 32 in order |
| Embedding dimension / heads / layers | 192 / 6 / 8 | 192 / 6 / 8 |
| Input | First retained person; center crop or last-frame padding | Same |
| Normalization / augmentation | Original per-frame normalization, temporal roll jitter ±2, horizontal flip | Same |
| Optimizer | AdamW, learning rate 5e-5, weight decay 1e-4 | Same |
| Schedule | Cosine, stepped once per epoch | Same |
| Batch size | 90 (original default) | 32 per GPU, configurable |
| Epoch limit / patience | 120 / 10 | Same |
| Seed / internal split seed | 42 / 42 | Same |
| Dropout | 0.25 | Same |

The backbone and preprocessing preserve the supplied script. In particular,
T=16/24/32 refers to contiguous center windows, not uniform sampling over the
full action. The existing loss is a batch-level focal-like transform of weighted
cross-entropy; it has not been converted to per-example focal loss. Progressive
unfreezing, classification-head reset, label smoothing and temporal token dropout
are not added by this protocol update. The FFN width remains PyTorch's original
default of 2048.

Two training-loop defects are corrected: AMP gradients are unscaled before
clipping, and the cosine schedule advances per epoch instead of per batch.
OneCycle, if selected, advances per successful optimizer update.

## Run on Kaggle

Enable Internet, select a GPU accelerator (two T4 GPUs permit concurrent
protocols), and attach the dataset containing `ntu120_3danno.pkl`.

```bash
python scripts/kaggle_train_official.py \
  --protocols xsub xset --frames 16 24 32 \
  --initialization reframe --batch 32 --epochs 120 --stop 10 \
  --seed 42 --split-seed 42 --val-fraction 0.1 \
  --d_model 192 --heads 6 --layers 8
```

At startup the launcher prints PyTorch's CUDA inventory. A normal dual-T4 run
should report two visible devices. If it reports `visible devices: 0`, the
notebook session has no CUDA accelerator exposed to PyTorch; select the Kaggle
GPU accelerator (T4 x2 for parallel XSUB/XSET) and restart the session before
running the command again.

With two visible GPUs, XSUB and XSET run concurrently. Their tqdm rows are pinned
independently, so the notebook can show both live at once, for example:

```text
XSUB GPU0 | TRAIN T16 E001/120 ... loss=... acc=... lr=...
XSET GPU1 | TRAIN T16 E001/120 ... loss=... acc=... lr=...
```

The training `acc` value is cumulative Top-1 accuracy over the batches processed
so far in the current epoch. During internal validation the same protocol row is
reused and reports cumulative validation Top-1 accuracy. These live values are
monitoring only; checkpoint selection still uses the completed internal-validation
accuracy, and the official test partition remains untouched until training and
checkpoint selection are finished.

With one visible GPU, XSUB and XSET run sequentially. Every protocol first trains
T16 from scratch. T24 and T32 each load that same protocol's selected T16
checkpoint; only the resized temporal embedding is reinitialized before further
training. The architecture and internal split must match. `--initialization
scratch` instead trains each T independently. Set `--pkl
/exact/path/ntu120_3danno.pkl` if discovery finds zero or multiple annotation
files.

Each run evaluates official test after its own training completes. No official
test metric controls the launcher or selects another frame length. Use
`--skip-test` to withhold final evaluations during development, and rerun with
the same configuration without that flag when experiments are fixed; completed
training resumes directly to final evaluation.

To inspect the outer split before training:

```bash
python cd_former_official.py --pkl /path/ntu120_3danno.pkl \
  --protocol xsub --frames 16 --check-splits
```

For a single run:

```bash
python cd_former_official.py --pkl /path/ntu120_3danno.pkl \
  --protocol xsub --frames 16 --batch 32 --workers 2 --device cuda \
  --outdir /kaggle/working/cdformer_official_runs/xsub/T16_seed42_split42 \
  --resume auto
```

For a same-protocol temporal transfer:

```bash
python cd_former_official.py --pkl /path/ntu120_3danno.pkl \
  --protocol xsub --frames 24 --batch 32 --workers 2 --device cuda \
  --pretrained /kaggle/working/cdformer_official_runs/xsub/T16_seed42_split42/best_graphormer.pth \
  --outdir /kaggle/working/cdformer_official_runs/xsub/T24_seed42_split42 \
  --resume auto
```

## Checkpoints and recovery

Outputs default to `/kaggle/working/cdformer_official_runs/PROTOCOL/Tn_seedN_splitN`.

- `splits.json`: membership, counts and fingerprint.
- `config.json`: run arguments.
- `best_graphormer.pth`: weights selected using internal validation.
- `last.pth`: latest completed epoch, model, optimizer, scheduler, AMP scaler,
  random states, history, patience and an embedded copy of the best checkpoint.
- `metrics/curve.csv`: internal training/validation history.
- `metrics/official_test.json`: final accuracy, sample count and checkpoint epoch.
- `metrics/official_test_predictions.csv`: per-clip predictions for the official test.
- `metrics/report.csv` and confusion matrices: final class-level results.

Checkpoint replacement is atomic. If a session stops mid-epoch, resume replays
that incomplete epoch from the preceding completed boundary. Resume requires
the original configuration; it does not silently change the batch size, epoch
budget or optimizer schedule. A completed run reuses its final result rather
than training or evaluating it again.

Automatic resume requires checkpoint files to remain available. To continue
in a new Kaggle session, save the output directory and attach those saved outputs
as an input dataset. Point the launcher at the directory containing `xsub/` and
`xset/`:

```bash
python scripts/kaggle_train_official.py \
  --restore-root /kaggle/input/my-saved-run/cdformer_official_runs \
  --protocols xsub xset --frames 16 24 32 --batch 32
```

The launcher copies missing run directories to the writable output location and
then resumes. It does not automatically publish notebook outputs or upload data.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests cover official membership, validation isolation, deterministic holdouts,
malformed split rejection, checkpoint provenance, T16/T24/T32 transfers,
forward passes, full-state resume, final-test routing and launcher paths. They
use small synthetic inputs. GPU execution, memory use, runtime and NTU120 accuracy
must be measured on the actual Kaggle run.
