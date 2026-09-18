# CD-Former

<p align="center">
  <strong>A Contextual Dynamic Transformer for Skeleton-Based Human Action Recognition</strong>
</p>

<p align="center">
  Pure-Transformer skeleton action recognition with contextual joint-frame tokenization and standard multi-head self-attention.
</p>

---

## Overview

CD-Former is a pure-Transformer framework for skeleton-based Human Action Recognition (HAR). It represents 3D skeleton sequences as contextual joint-frame tokens and models spatiotemporal relationships using standard multi-head self-attention.

The public repository is intentionally limited to the **manuscript-aligned source code, training/evaluation utilities, documentation required to run the code, and figures**. Trained model parameters are not included.

---

## Architecture

The manuscript reference configuration uses:

- embedding dimension: **192**
- attention heads: **8**
- Transformer encoder layers: **12**
- feed-forward dimension: **2048**
- standard Post-LN Transformer encoder
- temporal and joint-identity embeddings
- first indexed skeleton stream for multi-body recordings
- frame-wise z-score normalization
- center cropping for long clips and last-frame padding for short clips

The term **dynamic** refers to the input-dependent attention relations recomputed from each contextualized sequence. It does not denote a new attention operator, adaptive frame count, routing mechanism, or dynamic network topology.

![CD-Former architecture](assets/figures/cdformer_architecture.png)

---

## Repository structure

```text
CD_FORMER/
├── train_cdformer.py
├── graphormer_frames_reset_eval.py
├── README.md
├── requirements.txt
├── scripts/
│   ├── kaggle_train.sh
│   └── eval_cdformer.sh
└── assets/
    └── figures/
```

---

## Installation

```bash
git clone https://github.com/rombaldivia/CD_FORMER.git
cd CD_FORMER

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Dataset

The evaluation code uses the PySKL NTU RGB+D 120 annotation file:

```text
ntu120_3danno.pkl
```

The dataset itself is not distributed in this repository.

---

## Training

The public training entry point is `train_cdformer.py`. It keeps XSUB and XSET as **separate protocol runs** and reads the corresponding PySKL split keys directly:

- XSUB: `xsub_train` for optimization and `xsub_val` for protocol evaluation.
- XSET: `xset_train` for optimization and `xset_val` for protocol evaluation.

This matches the checkpoint-selection procedure disclosed in the manuscript. The `*_val` names are PySKL naming conventions for the official protocol evaluation partitions.

The default training configuration follows the manuscript reference setting: `d_model=192`, 8 attention heads, 12 encoder layers, FFN width 2048, dropout 0.15, AdamW with learning rate `8e-3` and weight decay 0.1, label smoothing 0.1, temporal token dropout 0.2, temporal jitter 6, cosine annealing, a maximum of 200 epochs, patience 30, and an effective batch size of 400. On Kaggle, the effective batch is realized with gradient accumulation by default (`20 x 20 = 400`) to fit T4 memory.

A single protocol run is:

```bash
python train_cdformer.py \
  --pkl /path/to/ntu120_3danno.pkl \
  --protocol xsub \
  --frames 32 \
  --outdir /kaggle/working/cdformer_runs/xsub/T32_seed42
```

The two-stage classification-head reset described in the manuscript is available by initializing a second run from a locally generated first-stage checkpoint:

```bash
python train_cdformer.py \
  --pkl /path/to/ntu120_3danno.pkl \
  --protocol xsub \
  --frames 32 \
  --init-checkpoint /path/to/first_stage/best_model.pth \
  --reset-head \
  --freeze-layers 12 \
  --unfreeze-epoch 15 \
  --outdir /kaggle/working/cdformer_runs/xsub/T32_reset_seed42
```

### Kaggle dual-T4 run

For Kaggle with two visible T4 GPUs, `scripts/kaggle_train.sh` launches XSUB on GPU 0 and XSET on GPU 1 without combining their data or checkpoints:

```bash
PKL_PATH=/kaggle/input/<dataset>/ntu120_3danno.pkl \
FRAMES=32 \
SEED=42 \
bash scripts/kaggle_train.sh
```

For the second head-reset stage, provide the two locally stored first-stage checkpoints:

```bash
INIT_XSUB=/kaggle/input/<private-checkpoints>/xsub_best_model.pth \
INIT_XSET=/kaggle/input/<private-checkpoints>/xset_best_model.pth \
PKL_PATH=/kaggle/input/<dataset>/ntu120_3danno.pkl \
FRAMES=32 \
SEED=42 \
bash scripts/kaggle_train.sh
```

All generated `.pth` files, logs, and run outputs remain outside the public repository and are blocked by `.gitignore`.

---

## Evaluation

The public release contains the evaluation code only. To evaluate a local CD-Former model file, provide its path explicitly.

For a single temporal configuration:

```bash
python graphormer_frames_reset_eval.py \
  --pkl /path/to/ntu120_3danno.pkl \
  --weights /path/to/local_model.pth \
  --val_xsub xsub_val \
  --val_xset xset_val \
  --frames 32 \
  --d_model 192 \
  --heads 8 \
  --layers 12 \
  --batch 32 \
  --device cuda \
  --outdir results/metrics_eval_32f
```

The evaluation script reports classification metrics, parameter count, analytical GFLOPs, throughput in samples/s, and latency in ms/sample.

---

## Manuscript-aligned analytical complexity

The repository follows the same principal matrix-operation convention used in the manuscript:

```text
MACs_total =
    T*J*C*d
    + L*(4*M*d^2 + 2*M^2*d + 2*M*d*d_ff)
    + d*K

FLOPs_total = 2 * MACs_total
```

with (M=TJ+1).

For the manuscript architecture:

| Frames | Analytical GFLOPs |
|---:|---:|
| 16 | 10.47 |
| 24 | 16.80 |
| 32 | 23.87 |

---

## Qualitative analysis

Manuscript-oriented attention and confusion-matrix figures are available under `assets/figures/`.

![CD-Former qualitative attention visualization](assets/figures/cdformer_attention_examples.png)

These visualizations are descriptive and are not interpreted as causal feature-importance evidence.

---

## Citation

```bibtex
@article{baldivia2026cdformer,
  title  = {CD-Former: A Contextual Dynamic Transformer for Skeleton-Based Human Action Recognition},
  author = {Baldivia Calderon de la Barca, Romel Antonio and others},
  year   = {2026},
  note   = {Manuscript under peer review}
}
```

---

## License

The source code is provided for noncommercial academic and research use under the repository license.
