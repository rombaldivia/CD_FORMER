# CD-Former

<p align="center">
  <strong>Contextual Dynamic Transformer for Efficient Skeleton-Based Human Action Recognition</strong>
</p>

<p align="center">
  A pure-Transformer research architecture for compact spatiotemporal modeling of 3D human motion on NTU RGB+D 120.
</p>

---

## Research Overview

**CD-Former** is a pure-Transformer framework for skeleton-based Human Action Recognition (HAR). It represents 3D joint sequences as compact spatiotemporal tokens and models motion through **Contextual Dynamic Self-Attention**, allowing temporal progression and joint-level dependencies to be learned within a unified attention architecture.

The project is designed around three engineering priorities:

- **Architectural clarity** — a Transformer-only formulation without graph-convolutional, recurrent, convolutional, hybrid, or multi-branch modules.
- **Computational efficiency** — configurable temporal resolution and deployment-aware measurements for both conventional and edge-oriented environments.
- **Reproducibility** — explicit evaluation scripts, checkpoint handling, validation reports, and documented experimental settings.

This repository provides the implementation and evaluation resources associated with the accompanying CD-Former manuscript.

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
- Pretrained-checkpoint loading and temporal-embedding adaptation.
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
├── checkpoints/
│   └── CD_former.pth
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

Large datasets are not stored in the repository. The NTU RGB+D 120 annotation resource can be downloaded automatically by the evaluation workflow.

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

## Pretrained Checkpoint

Place the pretrained checkpoint at:

```text
checkpoints/CD_former.pth
```

---

## Evaluation

Run the complete evaluation workflow with:

```bash
bash scripts/eval_cdformer.sh
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
  --weights checkpoints/CD_former.pth \
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

A reproducible Code Ocean execution can use:

```bash
bash /code/scripts/eval_cdformer.sh
```

When the checkpoint is stored directly as `/code/CD_former.pth`:

```bash
WEIGHTS_PATH=/code/CD_former.pth CODE_DIR=/code RESULTS_DIR=/results bash /code/scripts/eval_cdformer.sh
```

---

## Demonstration

The repository includes a project demonstration:

[View CD-Former demonstration](assets/videos/demo.mp4)

---

## Research Status

CD-Former is an active research project. The repository currently focuses on architecture documentation, pretrained evaluation, multi-frame validation, efficiency reporting, qualitative analysis, and reproducible execution.

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
