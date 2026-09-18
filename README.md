# CD-Former

Code accompanying **CD-Former: A Contextual Dynamic Transformer for Skeleton-Based Human Action Recognition**.

CD-Former uses joint-frame tokens with learnable temporal and joint embeddings and a standard Transformer encoder for skeleton action recognition.

## Model

The configuration used in the paper is:

- embedding dimension: 192
- attention heads: 8
- encoder layers: 12
- FFN dimension: 2048
- Post-LN Transformer encoder
- 16, 24, and 32 frame inputs
- first indexed skeleton stream for multi-person clips
- frame-wise z-score normalization
- center crop for long clips and last-frame padding for short clips

In the paper, *dynamic* refers to the attention matrix being recomputed from the current input sequence.

![CD-Former architecture](assets/figures/cdformer_architecture.png)

## Files

```text
CD_FORMER/
├── train_cdformer.py
├── graphormer_frames_reset_eval.py
├── requirements.txt
├── scripts/
│   ├── kaggle_train.sh
│   └── eval_cdformer.sh
└── assets/
    └── figures/
```

Model checkpoints and datasets are not included in the repository.

## Installation

```bash
git clone https://github.com/rombaldivia/CD_FORMER.git
cd CD_FORMER

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The NTU RGB+D 120 experiments use the PySKL annotation file:

```text
ntu120_3danno.pkl
```

The protocol keys used by the scripts are:

```text
XSUB: xsub_train / xsub_val
XSET: xset_train / xset_val
```

## Training

A single run can be started with:

```bash
python train_cdformer.py \
  --pkl /path/to/ntu120_3danno.pkl \
  --protocol xsub \
  --frames 32 \
  --seed 42 \
  --outdir /kaggle/working/cdformer_runs/xsub/T32_seed42
```

Default settings follow the configuration reported in the paper: AdamW, learning rate `8e-3`, weight decay `0.1`, dropout `0.15`, label smoothing `0.1`, temporal dropout `0.2`, temporal jitter `6`, cosine annealing, 200 epochs, and patience 30.

The training script also supports the head-reset stage:

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

### Kaggle

`scripts/kaggle_train.sh` runs XSUB on GPU 0 and XSET on GPU 1.

```bash
PKL_PATH=/kaggle/input/<dataset>/ntu120_3danno.pkl \
FRAMES=32 \
SEED=42 \
bash scripts/kaggle_train.sh
```

For the head-reset stage:

```bash
INIT_XSUB=/kaggle/input/<checkpoints>/xsub_best_model.pth \
INIT_XSET=/kaggle/input/<checkpoints>/xset_best_model.pth \
PKL_PATH=/kaggle/input/<dataset>/ntu120_3danno.pkl \
FRAMES=32 \
SEED=42 \
bash scripts/kaggle_train.sh
```

## Evaluation

Evaluate one checkpoint with:

```bash
python graphormer_frames_reset_eval.py \
  --pkl /path/to/ntu120_3danno.pkl \
  --weights /path/to/xsub_T32_model.pth \
  --protocol xsub \
  --frames 32 \
  --device cuda \
  --outdir results/xsub_T32
```

The evaluator uses the same `CDFormer` class as the training script and checks that the checkpoint matches the requested protocol, frame length, and model structure.

To evaluate the six NTU120 protocol/frame checkpoints, set:

```text
WEIGHTS_XSUB_16
WEIGHTS_XSUB_24
WEIGHTS_XSUB_32
WEIGHTS_XSET_16
WEIGHTS_XSET_24
WEIGHTS_XSET_32
```

then run:

```bash
PKL_PATH=/path/to/ntu120_3danno.pkl \
DEVICE=cuda \
bash scripts/eval_cdformer.sh
```

## Complexity

The FLOP count follows the convention used in the paper:

```text
MACs =
    T*J*C*d
    + L*(4*M*d^2 + 2*M^2*d + 2*M*d*d_ff)
    + d*K

FLOPs = 2 * MACs
M = T*J + 1
```

| Frames | GFLOPs |
|---:|---:|
| 16 | 10.47 |
| 24 | 16.80 |
| 32 | 23.87 |

## Figures

Additional figures used in the paper are available in `assets/figures/`.

![Attention examples](assets/figures/cdformer_attention_examples.png)

