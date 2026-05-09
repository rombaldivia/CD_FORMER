# CD-Former

**CD-Former** is a Graphormer-based implementation for Human Action Recognition (HAR) using 3D skeleton data from NTU RGB+D 120. The repository includes a complete training and evaluation pipeline based on PyTorch, with support for spatial skeleton modeling, reset-based training, frozen-backbone fine-tuning, and evaluation across NTU120 validation splits.

<p align="center">
  <img src="assets/figures/architecture.png" alt="CD-Former architecture" width="780">
</p>

<p align="center"><em>Suggested figure: CD-Former architecture or training pipeline.</em></p>

---

## Overview

CD-Former is designed for skeleton-based human action recognition. The current implementation focuses on NTU RGB+D 120 and exposes the main experimental settings through command-line arguments, making it easier to reproduce the training and evaluation setup reported in the manuscript.

The model is implemented in a single main script:

```text
graphormer_spatial_reset_full.py
```

---

## Main Features

- Graphormer-based spatial modeling for 3D skeleton sequences.
- Support for NTU RGB+D 120 training and validation splits.
- Evaluation-only mode for trained checkpoints.
- Configurable number of frames, embedding dimension, attention heads, and Transformer layers.
- Reset-head training strategy.
- Frozen-backbone fine-tuning with scheduled unfreezing.
- CUDA support for GPU acceleration.

---

## Repository Structure

```text
CD_FORMER/
├── graphormer_spatial_reset_full.py
├── best_graphormer.pth
├── ntu120_3danno.pkl
├── README.md
├── requirements.txt
├── scripts/
│   ├── eval_cdformer.sh
│   └── train_cdformer.sh
└── assets/
    ├── figures/
    │   └── architecture.png
    └── videos/
        └── demo.mp4
```

Large files such as datasets, checkpoints, and videos may be omitted from the repository depending on GitHub size limits. If they are not included, place them locally using the same names shown above.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/rombaldivia/CD_FORMER.git
cd CD_FORMER
```

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

This implementation expects the NTU RGB+D 120 annotation file in PKL format:

```text
ntu120_3danno.pkl
```

The expected split names used in the current training and evaluation commands are:

```text
xsub_train
xsub_val
xset_val
```

Place the dataset file in the root directory of the repository or update the `--pkl` argument with the correct path.

---

## Evaluation

Use this command to evaluate a trained CD-Former checkpoint:

```bash
python graphormer_spatial_reset_full.py \
  --eval_only \
  --weights best_graphormer.pth \
  --val_xsub xsub_val \
  --val_xset xset_val \
  --frames 32 \
  --d_model 192 \
  --heads 8 \
  --layers 12 \
  --device cuda
```

For Google Colab, use:

```python
!python graphormer_spatial_reset_full.py \
  --eval_only \
  --weights best_graphormer.pth \
  --val_xsub xsub_val \
  --val_xset xset_val \
  --frames 32 --d_model 192 --heads 8 --layers 12 \
  --device cuda
```

---

## Training

Use this command to train CD-Former on the NTU120 cross-subject split:

```bash
python graphormer_spatial_reset_full.py \
  --pkl ntu120_3danno.pkl \
  --train_split xsub_train \
  --val_split xsub_val \
  --val_xsub xsub_val \
  --val_xset xset_val \
  --frames 32 \
  --d_model 192 \
  --heads 8 \
  --layers 12 \
  --reset_head \
  --freeze_n 12 \
  --unfreeze_epoch 15 \
  --device cuda
```

For Google Colab, use:

```python
!python graphormer_spatial_reset_full.py \
  --pkl ntu120_3danno.pkl \
  --train_split xsub_train \
  --val_split xsub_val \
  --val_xsub xsub_val \
  --val_xset xset_val \
  --frames 32 --d_model 192 --heads 8 --layers 12 \
  --reset_head --freeze_n 12 --unfreeze_epoch 15 \
  --device cuda
```

---

## Demo Video

A demonstration video can be added under:

```text
assets/videos/demo.mp4
```

When the file is uploaded to the repository, it can be referenced from this section.

<p align="center">
  <video src="assets/videos/demo.mp4" controls width="780"></video>
</p>

If GitHub does not render the video inline, open it directly from the `assets/videos/` folder.

---

## Suggested Media Files

To improve the presentation of the repository, add the following files:

```text
assets/figures/architecture.png      Main CD-Former architecture diagram
assets/figures/pipeline.png          Dataset and training pipeline
assets/figures/confusion_matrix.png  Evaluation result visualization
assets/videos/demo.mp4               Short demonstration video
```

---

## Reproducibility Notes

For reproducible experiments, keep the following settings consistent:

```text
frames   = 32
d_model  = 192
heads    = 8
layers   = 12
```

Recommended training configuration:

```text
reset_head      = enabled
freeze_n        = 12
unfreeze_epoch  = 15
device          = cuda
```

---

## Citation

If this repository is useful for your research, please cite the accompanying manuscript:

```bibtex
@article{baldivia2026cdformer,
  title={CD-Former},
  author={Baldivia Calderon de la Barca, Romel Antonio and others},
  year={2026},
  note={Manuscript under review}
}
```

---

## License

This repository is intended for academic and research use.
