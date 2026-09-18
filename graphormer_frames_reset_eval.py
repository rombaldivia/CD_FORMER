#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Protocol-specific CD-Former evaluation for NTU RGB+D 120.

This file imports the public CDFormer implementation from train_cdformer.py so
training and evaluation use exactly the same architecture. It does not redefine
or alter the model.

A checkpoint is evaluated only on its matching protocol partition:
  XSUB -> xsub_val
  XSET -> xset_val

Accepted checkpoint formats:
  1) checkpoints produced by train_cdformer.py (key: "model")
  2) dictionaries containing "state_dict"
  3) raw PyTorch state dictionaries
"""

import argparse
import pickle
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
)

from train_cdformer import CDFormer, NTU120Dataset


def cpu_mem():
    return f"{psutil.virtual_memory().used / (1024 ** 3):.2f}"


def gpu_mem():
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return f"{used:.2f}/{total:.2f}"
    return "0/0"


def top5_acc(y_true, probas):
    top5 = np.argsort(probas, axis=1)[:, -5:]
    correct = sum(y_true[i] in top5[i] for i in range(len(y_true)))
    return 100.0 * correct / max(len(y_true), 1)


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ids_to_idx(data, split_key, bad_ids):
    if split_key not in data["split"]:
        raise ValueError(f"Split {split_key!r} was not found in the annotation file.")

    index = {}
    for i, sample in enumerate(data["annotations"]):
        frame_dir = sample.get("frame_dir")
        if frame_dir is None:
            raise ValueError(f"Annotation {i} has no frame_dir")
        if frame_dir in index:
            raise ValueError(f"Duplicate frame_dir: {frame_dir}")
        index[frame_dir] = i

    wanted = data["split"][split_key]
    missing = [name for name in wanted if name not in index]
    if missing:
        raise ValueError(
            f"{split_key} contains missing annotations; first IDs: {missing[:3]}"
        )

    return [index[name] for name in wanted if name not in bad_ids]


def extract_state_dict(raw):
    if isinstance(raw, dict) and "model" in raw:
        state = raw["model"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    else:
        state = raw

    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a valid model state dictionary.")

    return {k.removeprefix("module."): v for k, v in state.items()}


def load_checkpoint(model, path, device, protocol, frames):
    raw = torch.load(path, map_location=device)

    if isinstance(raw, dict):
        config = raw.get("config")
        if isinstance(config, dict):
            ckpt_protocol = config.get("protocol")
            ckpt_frames = config.get("frames")

            if ckpt_protocol is not None and ckpt_protocol != protocol:
                raise ValueError(
                    f"Protocol mismatch: checkpoint={ckpt_protocol}, "
                    f"requested={protocol}"
                )
            if ckpt_frames is not None and int(ckpt_frames) != int(frames):
                raise ValueError(
                    f"Frame mismatch: checkpoint={ckpt_frames}, requested={frames}"
                )

    state = extract_state_dict(raw)
    base = model.state_dict()

    missing = [k for k in base if k not in state]
    unexpected = [k for k in state if k not in base]
    shape_mismatch = [
        k for k in base
        if k in state and tuple(base[k].shape) != tuple(state[k].shape)
    ]

    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "Checkpoint does not exactly match the current CD-Former architecture. "
            f"Missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"shape_mismatch={shape_mismatch[:5]}"
        )

    model.load_state_dict(state, strict=True)
    print(f"Loaded checkpoint: {path}")
    print(f"Verified {len(base)}/{len(base)} model tensors.")


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(x)

        prob = F.softmax(logits, dim=1)
        pred = prob.argmax(1)

        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
        y_prob.extend(prob.cpu().numpy())

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    throughput = len(y_true) / max(elapsed, 1e-8)
    latency = elapsed / max(len(y_true), 1) * 1000.0

    return (
        np.asarray(y_true),
        np.asarray(y_pred),
        np.asarray(y_prob),
        throughput,
        latency,
    )


def main(args):
    set_seed(args.seed)

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    if args.device == "cuda" and not use_cuda:
        raise RuntimeError("CUDA requested but no CUDA device is available.")
    device = torch.device("cuda" if use_cuda else "cpu")

    with open(args.pkl, "rb") as handle:
        data = pickle.load(handle)

    bad_ids = set()
    if args.missing_txt:
        with open(args.missing_txt, encoding="utf-8") as handle:
            bad_ids = {line.strip() for line in handle if line.strip()}

    # Preserve the exact public architecture from train_cdformer.py.
    model = CDFormer(
        seq_len=args.frames,
        num_joints=25,
        num_classes=120,
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
        d_ff=args.d_ff,
    ).to(device)

    load_checkpoint(
        model,
        args.weights,
        device,
        protocol=args.protocol,
        frames=args.frames,
    )

    # Same analytical convention used in the manuscript.
    J, C, K = 25, 3, 120
    d, d_ff, L = args.d_model, args.d_ff, args.layers
    M = args.frames * J + 1
    macs = (
        args.frames * J * C * d
        + L * (4 * M * d * d + 2 * M * M * d + 2 * M * d * d_ff)
        + d * K
    )
    gflops = 2 * macs / 1e9
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    split_key = f"{args.protocol}_val"
    indices = ids_to_idx(data, split_key, bad_ids)
    dataset = NTU120Dataset(
        data["annotations"],
        indices,
        frames=args.frames,
        training=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=args.num_workers > 0,
    )

    print(f"Device: {device}")
    print(f"Protocol: {args.protocol.upper()}")
    print(f"Split: {split_key} ({len(dataset)} samples)")
    print(f"Frames: {args.frames}")
    print(f"Analytical GFLOPs: {gflops:.2f} | Params: {params_m:.2f}M")

    y_true, y_pred, y_prob, throughput, latency = validate(
        model, loader, device
    )

    recall = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100
    top1 = (y_true == y_pred).mean() * 100
    top5 = top5_acc(y_true, y_prob)
    balanced_acc = balanced_accuracy_score(y_true, y_pred) * 100
    kappa = cohen_kappa_score(y_true, y_pred) * 100
    matthews = matthews_corrcoef(y_true, y_pred) * 100
    ram, vram = cpu_mem(), gpu_mem()

    print(
        f"Recall {recall:.2f}% | F1 {f1:.2f}% | "
        f"Top1/5 {top1:.2f}/{top5:.2f}%"
    )
    print(
        f"Balanced Acc {balanced_acc:.2f}% | "
        f"Kappa/Matthews {kappa:.2f}/{matthews:.2f}%"
    )
    print(
        f"Throughput {throughput:.2f} samples/s | "
        f"Latency {latency:.1f} ms/sample | RAM/VRAM {ram}/{vram}"
    )

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    report = pd.DataFrame(
        classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
    ).T
    report.to_csv(outdir / f"report_{args.protocol}.csv", index=True)

    summary = {
        "Protocol": args.protocol,
        "Split": split_key,
        "Frames": args.frames,
        "Samples": len(dataset),
        "Recall": f"{recall:.2f}%",
        "F1": f"{f1:.2f}%",
        "Top-1": f"{top1:.2f}%",
        "Top-5": f"{top5:.2f}%",
        "Balanced Accuracy": f"{balanced_acc:.2f}%",
        "Kappa": f"{kappa:.2f}%",
        "Matthews": f"{matthews:.2f}%",
        "GFLOPs": f"{gflops:.2f}",
        "Params(M)": f"{params_m:.2f}",
        "Throughput(samples/s)": round(throughput, 2),
        "Latency(ms/sample)": round(latency, 1),
        "RAM/VRAM(GB)": f"{ram}/{vram}",
    }
    pd.DataFrame([summary]).to_csv(
        outdir / f"metrics_{args.protocol}.csv", index=False
    )

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar()
    plt.title(f"Confusion ({args.protocol.upper()}, T={args.frames})")
    plt.tight_layout()
    plt.savefig(
        outdir / f"confmat_{args.protocol}.png",
        dpi=200,
    )
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate one protocol-specific CD-Former checkpoint."
    )
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--protocol", choices=["xsub", "xset"], required=True)
    parser.add_argument("--frames", type=int, choices=[16, 24, 32], required=True)
    parser.add_argument("--missing-txt", dest="missing_txt", default="")
    parser.add_argument("--d-model", dest="d_model", type=int, default=192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--d-ff", dest="d_ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="./metrics_eval")
    main(parser.parse_args())
