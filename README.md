# CD-Former

<p align="center">
  <strong>A Contextual Dynamic Transformer for Skeleton-Based Human Action Recognition</strong>
</p>

<p align="center">
  Pure-Transformer skeleton action recognition with contextual joint-frame tokenization and standard multi-head self-attention.
</p>

---

## Reviewer-facing repository note

This public repository accompanies the CD-Former manuscript and exposes the **source code, training/evaluation utilities, documentation, and figures only**. **Trained model weights/checkpoints are intentionally not distributed publicly.**

The manuscript evaluates CD-Former with temporal windows of **16, 24, and 32 frames**. Its reference architecture uses:

- embedding dimension: **192**
- attention heads: **8**
- Transformer encoder layers: **12**
- feed-forward dimension: **2048**
- standard Post-LN Transformer encoder
- first indexed skeleton stream for multi-body recordings
- frame-wise z-score normalization
- center cropping for long clips and last-frame padding for short clips

The term **dynamic** in CD-Former refers to the input-dependent attention relations recomputed from each contextualized sequence. It does **not** denote a new attention operator, dynamic network topology, adaptive frame count, or routing/gating mechanism.

---

## Important note on the reported experimental protocol

The manuscript explicitly discloses that the historical reported checkpoints were selected by monitoring the PySKL protocol-specific partitions named `xsub_val` and `xset_val`, which correspond to the official NTU RGB+D 120 evaluation partitions. Therefore, the manuscript results are described as **checkpoint-selected evaluation values**, not strictly untouched-test estimates.

The repository also contains `cd_former_official.py` and `scripts/kaggle_train_official.py`, which implement a **protocol-clean retraining workflow** that creates an internal validation split only from the official training partition and reserves the official evaluation partition for final testing. This newer workflow is provided for future reproducible retraining and **is not the procedure that generated the historical manuscript numbers**.

See `docs/official_protocol_training.md` for that separate workflow.

---

## Architecture

CD-Former represents each skeleton joint at each frame as a joint-frame token. A linear projection maps 3D coordinates to the latent dimension, and learnable temporal and joint-identity embeddings provide contextual information before standard multi-head self-attention.

The final representation combines:

1. the learned CLS token, and
2. the mean of all non-CLS token outputs,

using element-wise addition before the linear classification head.

![CD-Former architecture](assets/figures/cdformer_architecture.png)

---

## Repository structure

```text
CD_FORMER/
├── graphormer_frames_reset_eval.py
├── cd_former_official.py
├── README.md
├── requirements.txt
├── scripts/
│   ├── eval_cdformer.sh
│   ├── train_cdformer.sh
│   └── kaggle_train_official.py
├── docs/
│   └── official_protocol_training.md
├── tests/
└── assets/
    └── figures/
```

Datasets, trained checkpoints, and experiment output directories are excluded from the public repository.

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

The code uses the PySKL NTU RGB+D 120 annotation file:

```text
ntu120_3danno.pkl
```

The PySKL split names `xsub_val` and `xset_val` are the protocol-specific **official evaluation partitions**, despite the historical `val` naming.

Large dataset files are not stored in this repository.

---

## Evaluation with local checkpoints

Public weights are not included. To evaluate the three manuscript temporal configurations, provide one local checkpoint for each setting:

```bash
WEIGHTS_16=/path/to/CDFormer_16f.pth \
WEIGHTS_24=/path/to/CDFormer_24f.pth \
WEIGHTS_32=/path/to/CDFormer_32f.pth \
DEVICE=cuda \
bash scripts/eval_cdformer.sh
```

For a single configuration:

```bash
python graphormer_frames_reset_eval.py \
  --pkl /path/to/ntu120_3danno.pkl \
  --weights /path/to/local_checkpoint.pth \
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

The evaluation script reports classification metrics, parameter count, and **analytical GFLOPs using the same principal matrix-operation convention described in the manuscript (1 MAC = 2 FLOPs)**. Runtime throughput is reported in **samples/s**, with latency in **ms/sample**.

---

## Manuscript-aligned analytical complexity

For sequence length (M=TJ+1), embedding dimension (d), feed-forward width (d_{ff}), (L) encoder layers, input channels (C), joints (J), and classes (K), the repository follows:

```text
MACs_total =
    T*J*C*d
    + L*(4*M*d^2 + 2*M^2*d + 2*M*d*d_ff)
    + d*K

FLOPs_total = 2 * MACs_total
```

For the manuscript architecture ((J=25), (C=3), (d=192), (d_{ff}=2048), (L=12), (K=120)):

| Frames | Analytical GFLOPs |
|---:|---:|
| 16 | 10.47 |
| 24 | 16.80 |
| 32 | 23.87 |

These values intentionally exclude element-wise activation, normalization, Softmax, and bias-addition operations, matching the manuscript definition.

---

## Qualitative analysis

The repository contains the manuscript-oriented attention and confusion-matrix visualizations under `assets/figures/`.

![CD-Former qualitative attention visualization](assets/figures/cdformer_attention_examples.png)

These visualizations are descriptive and are not presented as causal feature-importance evidence.

---

## Code-only public release

The following are intentionally excluded from the public repository:

- trained `.pth`, `.pt`, and `.ckpt` files
- datasets and annotation binaries
- experiment outputs and logs
- private training artifacts

The `.gitignore` enforces these exclusions for future commits.

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

The source code is provided for noncommercial academic and research use under the repository license. Trained model weights are not part of the public release.
