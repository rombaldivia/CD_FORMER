# CD-Former

<p align="center">
  <strong>Contextual Dynamic Transformer for Efficient Skeleton-Based Human Action Recognition</strong>
</p>

<p align="center">
  A pure-Transformer research architecture for compact spatiotemporal modeling of 3D human motion on NTU RGB+D 120.
</p>

---

## Official-protocol training on Kaggle

The current training entrypoint is [`cd_former_official.py`](cd_former_official.py).
It selects the official NTU120 **XSUB or XSET split first**, then holds out 10%
of that protocol's training partition for internal validation. PySKL's
`xsub_val` and `xset_val` are the **official test partitions**; they are used
only after checkpoint selection on internal validation.

```bash
python scripts/kaggle_train_official.py \
  --protocols xsub xset --frames 16 24 32 \
  --initialization reframe --batch 32 --epochs 120 --stop 10
```

The Kaggle launcher detects an attached `ntu120_3danno.pkl`, uses separate GPUs
for XSUB and XSET when available, and resumes each run from its last completed
epoch. Within each protocol, T24 and T32 independently transfer the corresponding
T16 weights and reinitialize only the temporal embedding. Model defaults follow
the supplied trainer: **dimension 192, 6 heads, 8 layers**. These differ from the
historical evaluation defaults documented below. The Kaggle batch defaults to
32 per GPU; the supplied trainer's original batch default was 90.

See [the training guide](docs/official_protocol_training.md) for individual runs,
split checks, checkpoint recovery, exact settings and verification limits.
The legacy fine-tuning and evaluation scripts are retained for historical use;
the new Kaggle launcher uses the official-protocol entrypoint above.

---

## Research Overview

**CD-Former** is a pure-Transformer framework for skeleton-based Human Action Recognition (HAR). It represents 3D joint sequences as compact spatiotemporal tokens and models motion through **Contextual Dynamic Self-Attention**, allowing temporal progression and joint-level dependencies to be learned within a unified attention architecture.

The project is designed around three engineering priorities:

- **Architectural clarity** — a Transformer-only formulation without graph-convolutional, recurrent, convolutional, hybrid, or multi-branch modules.
- **Computational efficiency** — configurable temporal resolution and deployment-aware measurements for both conventional and edge-oriented environments.
- **Reproducibility** — explicit evaluation scripts, checkpoint handling, validation reports, and documented experimental settings.

This repository provides the implementation and evaluation code associated with the accompanying CD-Former manuscript. Model checkpoints and trained weights are not distributed in this public repository.

---

## Research Contribution

CD-Former investigates whether a compact pure-Transformer model can capture the structural and temporal properties of skeleton sequences without depending on hand-designed graph operators or additional architectural branches.

Its principal contribution is the integration of contextual temporal and joint-aware representations directly into the attention process. This creates a single modeling pathway in which motion evolution, joint relationships, and sequence-level classification are optimized together.

The resulting architecture is intended to balance recognition capability, model complexity, implementation simplicity, and practical deployability.

---

## Model Architecture

![CD-Former architecture](assets/figures/cdformer_architecture.png)

CD-Former receives a skeleton sequence, projects the input features into a latent representation, adds temporal and joint embeddings, prepends a CLS token, and processes the resulting sequence through stacked CD-Former blocks.

The final prediction combines the global CLS representation with average-pooled sequence features before the classification head. This fusion preserves both explicitly aggregated sequence information and the learned global token representation.

---

## Contextual Dynamic Self-Attention

![Contextual Dynamic Attention block](assets/figures/contextual_dynamic_attention.png)

The attention module integrates temporal and joint-aware embeddings before computing multi-head self-attention. This enables the network to represent:

- Frame-level motion progression.
- Joint-level spatial dependencies.
- Long-range temporal interactions.
- Sequence-wide contextual relationships.

The design avoids a separate graph-processing stage and instead learns contextual structure directly through attention.

---

## Qualitative Analysis

![CD-Former qualitative attention visualization](assets/figures/cdformer_attention_examples.png)

The qualitative visualization presents representative skeleton sequences, temporal-attention behavior, joint-attention behavior, and the corresponding top-3 action predictions. These examples are intended to support interpretation of how the model distributes attention across time and anatomical structure.

---

## Validation Results

| 16 Frames | 24 Frames | 32 Frames |
|:---:|:---:|:---:|
| ![16-frame validation confusion matrix](assets/figures/confusion_matrix_16f.png) | ![24-frame validation confusion matrix](assets/figures/confusion_matrix_24f.png) | ![32-frame validation confusion matrix](assets/figures/confusion_matrix_32f.png) |

The repository supports evaluation at multiple temporal resolutions to examine the relationship between sequence length, recognition behavior, and computational cost.

---

## Technical Profile

| Category | Implementation |
| --- | --- |
| Task | Skeleton-based Human Action Recognition |
| Dataset | NTU RGB+D 120 |
| Evaluation protocols | Cross-Subject (`xsub`) and Cross-Setup (`xset`) |
| Core architecture | Pure Transformer |
| Attention mechanism | Contextual Dynamic Self-Attention |
| Framework | PyTorch |
| Supported frame settings | 16, 24, and 32 frames |
| Deployment analysis | GFLOPs, FPS, latency, RAM, and VRAM measurements |
| Outputs | Metrics, CSV reports, and confusion matrices |

---

## Main Capabilities

- Pure-Transformer architecture for 3D skeleton-based HAR.
- Contextual temporal and joint-aware representations.
- Compact spatiotemporal tokenization.
- Configurable embedding dimension, attention heads, Transformer depth, and frame count.
- Support for NTU RGB+D 120 `xsub` and `xset` validation splits.
- Local-checkpoint loading and temporal-embedding adaptation.
- Top-1, Top-5, recall, F1, balanced accuracy, Cohen's kappa, and Matthews correlation coefficient.
- GFLOPs, FPS, latency, RAM, and VRAM reporting.
- Automatic generation of confusion matrices and structured CSV results.
- Evaluation workflows for conventional and edge-oriented hardware.

---

## Repository Structure

```text
CD_FORMER/
├── graphormer_frames_reset_eval.py
├── README.md
├── requirements.txt
├── scripts/
│   ├── eval_cdformer.sh
│   └── train_cdformer.sh
└── assets/
    ├── figures/
    │   ├── cdformer_architecture.png
    │   ├── contextual_dynamic_attention.png
    │   ├── cdformer_attention_examples.png
    │   ├── confusion_matrix_16f.png
    │   ├── confusion_matrix_24f.png
    │   └── confusion_matrix_32f.png
    └── videos/
        └── demo.mp4
```

Large datasets and trained model weights are not stored in the repository. The NTU RGB+D 120 annotation resource can be downloaded automatically by the evaluation workflow.

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

The evaluation pipeline uses:

```text
ntu120_3danno.pkl
```

When the file is not already available, the script can retrieve it from the OpenMMLab/MMAction public resource.

Expected validation splits:

```text
xsub_val
xset_val
```

---

## Model Weights

Trained CD-Former checkpoints are not distributed in this public repository. To evaluate a locally available checkpoint, provide its path explicitly through `--weights` or the `WEIGHTS_PATH` environment variable.

---

## Evaluation

Run the complete evaluation workflow with a local checkpoint:

```bash
WEIGHTS_PATH=/path/to/local/CD_former.pth bash scripts/eval_cdformer.sh
```

The workflow evaluates the supported frame configurations and stores the results under:

```text
results/metrics_eval_16f
results/metrics_eval_24f
results/metrics_eval_32f
```

To evaluate a specific configuration manually:

```bash
python graphormer_frames_reset_eval.py \
  --pkl /data/nturgbd/ntu120_3danno.pkl \
  --weights /path/to/local/CD_former.pth \
  --val_xsub xsub_val \
  --val_xset xset_val \
  --frames 32 \
  --d_model 192 \
  --heads 8 \
  --layers 12 \
  --batch 32 \
  --device cpu \
  --num_workers 2 \
  --outdir results/metrics_eval_32f
```

For CUDA-enabled environments:

```bash
DEVICE=cuda bash scripts/eval_cdformer.sh
```

---

## Reproducibility Profile

Default evaluation settings:

```text
d_model  = 192
heads    = 8
layers   = 12
frames   = 16, 24, 32
device   = cpu by default, cuda optional
```

The evaluation pipeline records predictive metrics and computational measurements in a structured format to support comparison across frame settings and hardware environments.

---

## Code Ocean

For a reproducible execution, mount or provide a checkpoint privately and pass its local path at runtime:

```bash
WEIGHTS_PATH=/path/to/local/CD_former.pth bash /code/scripts/eval_cdformer.sh
```

Model weights are intentionally excluded from the public source repository.

---

## Demonstration

The repository includes a project demonstration:

[View CD-Former demonstration](assets/videos/demo.mp4)

---

## Research Status

CD-Former is an active research project. The repository currently focuses on architecture documentation, evaluation code, multi-frame validation, computational reporting, qualitative analysis, and reproducible execution.

Experimental results, manuscript terminology, and supporting resources may continue to evolve during the review and publication process.

---

## Citation

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

This repository is licensed for noncommercial academic and research use. Commercial use requires a separate written license agreement with the author. See `LICENSE` and `COMMERCIAL.md` for details.

---

<p align="center">
  <strong>Context-aware motion modeling through a compact pure-Transformer architecture.</strong>
</p>
