#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CD-Former evaluation script for NTU RGB+D 120 skeleton-based HAR.

This script evaluates a trained CD-Former checkpoint on xsub and xset validation splits.
It reports Top-1/Top-5 accuracy, recall, F1-score, balanced accuracy, Cohen's kappa,
Matthews correlation coefficient, GFLOPs, FPS, latency, RAM/VRAM usage, classification
reports, and normalized confusion matrices.
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
import torch.nn as nn
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
    return 100 * correct / len(y_true)


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MMAction2KeypointDataset(torch.utils.data.Dataset):
    def __init__(self, data, idx_list, num_frames=16):
        self.samples = [data["annotations"][i] for i in idx_list]
        self.num_frames = num_frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        kp = sample["keypoint"]
        if kp.ndim == 4:
            kp = kp[0]
        label = sample["label"]
        T = kp.shape[0]
        if T > self.num_frames:
            start = (T - self.num_frames) // 2
            kp = kp[start:start + self.num_frames]
        elif T < self.num_frames:
            pad = np.repeat(kp[-1][None], self.num_frames - T, axis=0)
            kp = np.concatenate([kp, pad], axis=0)
        kp = (kp - kp.mean(axis=1, keepdims=True)) / (kp.std(axis=1, keepdims=True) + 1e-5)
        return torch.from_numpy(kp).float(), label


class GraphormerForHAR(nn.Module):
    def __init__(self, num_joints, seq_len, num_classes=120,
                 d_model=192, num_heads=8, num_layers=12, dropout_p=0.0):
        super().__init__()
        self.proj = nn.Linear(3, d_model)
        self.temb = nn.Embedding(seq_len, d_model)
        self.jemb = nn.Embedding(num_joints, d_model)
        self.drop = nn.Dropout(dropout_p)
        self.norm = nn.LayerNorm(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout_p,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B, T, J, C = x.shape
        x = self.proj(x)
        x = x + self.temb(torch.arange(T, device=x.device)).unsqueeze(1)
        x = x + self.jemb(torch.arange(J, device=x.device))
        x = self.norm(self.drop(x.reshape(B, T * J, -1)))
        x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1)
        x = self.enc(x)
        cls = x[:, 0]
        avg = x[:, 1:].mean(1)
        return self.head(cls + avg)


def ids_to_idx(data, split_key, bad_ids):
    if split_key not in data["split"]:
        raise ValueError(f"Split '{split_key}' was not found in the annotation file.")
    wanted = set(data["split"][split_key])
    return [
        i for i, sample in enumerate(data["annotations"])
        if sample["frame_dir"] in wanted and sample["frame_dir"] not in bad_ids
    ]


def load_checkpoint(model, path, device, frames):
    raw = torch.load(path, map_location=device)
    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    if "temb.weight" in state_dict and state_dict["temb.weight"].shape[0] != frames:
        print(f"Reset temporal embedding: {state_dict['temb.weight'].shape[0]} -> {frames}")
        del state_dict["temb.weight"]
    base = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in base and v.shape == base[k].shape}
    print(f"Loading {len(filtered)}/{len(base)} matching layers ({len(filtered) / len(base) * 100:.1f}%)")
    model.load_state_dict(filtered, strict=False)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    start = time.time()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
        prob = F.softmax(logits, dim=1)
        pred = prob.argmax(1)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
        y_prob.extend(prob.cpu().numpy())
    elapsed = time.time() - start
    fps = len(y_true) * loader.dataset.num_frames / max(elapsed, 1e-8)
    latency = elapsed / max(len(y_true), 1) * 1000
    return np.array(y_true), np.array(y_pred), np.array(y_prob), fps, latency


def main(args):
    set_seed(42)
    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)

    bad_ids = set()
    if args.missing_txt:
        with open(args.missing_txt) as f:
            bad_ids = {line.strip() for line in f if line.strip()}

    model = GraphormerForHAR(
        num_joints=25,
        seq_len=args.frames,
        num_classes=args.num_classes,
        d_model=args.d_model,
        num_heads=args.heads,
        num_layers=args.layers,
        dropout_p=args.dropout,
    ).to(device)
    load_checkpoint(model, args.weights, device, args.frames)

    try:
        from thop import profile
        dummy_input = torch.randn(1, args.frames, 25, 3).to(device)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        gflops = flops / 1e9
        params_m = params / 1e6
        print(f"GFLOPs: {gflops:.2f} | Params: {params_m:.2f}M")
    except Exception as exc:
        print(f"[Warning] GFLOPs could not be estimated: {exc}")
        gflops, params_m = 0.0, 0.0

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    splits = {"xsub": args.val_xsub, "xset": args.val_xset}
    summaries = []

    for name, split_key in splits.items():
        print(f"\n===== Evaluating {name.upper()} ({split_key}) =====")
        idxs = ids_to_idx(data, split_key, bad_ids)
        dataset = MMAction2KeypointDataset(data, idxs, args.frames)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch,
            num_workers=args.num_workers,
            pin_memory=use_cuda,
        )
        y_true, y_pred, y_prob, fps, latency = validate(model, loader, device)

        recall = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100
        top1 = (y_true == y_pred).mean() * 100
        top5 = top5_acc(y_true, y_prob)
        balanced_acc = balanced_accuracy_score(y_true, y_pred) * 100
        kappa = cohen_kappa_score(y_true, y_pred) * 100
        matthews = matthews_corrcoef(y_true, y_pred) * 100
        ram, vram = cpu_mem(), gpu_mem()

        print(f"Frames {args.frames} | Recall {recall:.2f}% | F1 {f1:.2f}% | Top1/5 {top1:.2f}/{top5:.2f}%")
        print(f"Balanced Acc {balanced_acc:.2f}% | Kappa/Matthews {kappa:.2f}/{matthews:.2f}%")
        print(f"FPS {fps:.2f} | Latency {latency:.1f} ms | RAM/VRAM {ram}/{vram}")

        report = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True, zero_division=0)).T
        report.to_csv(outdir / f"report_{name}.csv", index=True)

        summary = {
            "Subset": name,
            "Split": split_key,
            "Frames": args.frames,
            "Recall": f"{recall:.2f}%",
            "F1": f"{f1:.2f}%",
            "Top-1": f"{top1:.2f}%",
            "Top-5": f"{top5:.2f}%",
            "Balanced Accuracy": f"{balanced_acc:.2f}%",
            "Kappa": f"{kappa:.2f}%",
            "Matthews": f"{matthews:.2f}%",
            "GFLOPs": f"{gflops:.2f}",
            "Params(M)": f"{params_m:.2f}",
            "FPS": round(fps, 2),
            "Latency(ms)": round(latency, 1),
            "RAM/VRAM(GB)": f"{ram}/{vram}",
        }
        summaries.append(summary)
        pd.DataFrame([summary]).to_csv(outdir / f"metrics_{name}.csv", index=False)

        cm = confusion_matrix(y_true, y_pred, normalize="true")
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, cmap="viridis", vmin=0, vmax=1)
        plt.colorbar()
        plt.title(f"Confusion ({name})")
        plt.tight_layout()
        plt.savefig(outdir / f"confmat_{name}.png", dpi=200)
        plt.close()

    pd.DataFrame(summaries).to_csv(outdir / "metrics_summary.csv", index=False)
    print(f"\nResults saved in {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--missing_txt", default="")
    parser.add_argument("--val_xsub", default="xsub_val")
    parser.add_argument("--val_xset", default="xset_val")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_classes", type=int, default=120)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--outdir", default="./metrics_eval")
    main(parser.parse_args())
