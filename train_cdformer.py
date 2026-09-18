#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CD-Former training on NTU RGB+D 120.

Protocol split names in the PySKL ntu120_3danno.pkl file:
  - XSUB: xsub_train -> optimization, xsub_val -> protocol evaluation
  - XSET: xset_train -> optimization, xset_val -> protocol evaluation

Defaults match the manuscript reference configuration:
d_model=192, heads=8, layers=12, FFN=2048, dropout=0.15,
AdamW(lr=8e-3, weight_decay=0.1), label smoothing=0.1,
temporal token dropout=0.2, temporal jitter=6, effective batch=400,
200 epochs, patience=30, cosine annealing.

The two-stage head-reset procedure is supported by first training a checkpoint,
then launching a second run with --init-checkpoint and --reset-head. When
--reset-head is used, the first --freeze-layers encoder layers are frozen for
--unfreeze-epoch completed epochs and then unfrozen.

Trained checkpoints are written only to the requested output directory and are
not part of the public repository.
"""

import argparse
import json
import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NTU120Dataset(Dataset):
    def __init__(
        self,
        annotations,
        indices,
        frames=32,
        training=False,
        temporal_dropout=0.0,
        temporal_jitter=0,
    ):
        self.annotations = annotations
        self.indices = list(indices)
        self.frames = int(frames)
        self.training = bool(training)
        self.temporal_dropout = float(temporal_dropout)
        self.temporal_jitter = int(temporal_jitter)

    def __len__(self):
        return len(self.indices)

    def _fixed_window(self, kp):
        length = kp.shape[0]
        if length >= self.frames:
            start = (length - self.frames) // 2
            kp = kp[start:start + self.frames]
        else:
            pad = np.repeat(kp[-1][None], self.frames - length, axis=0)
            kp = np.concatenate([kp, pad], axis=0)
        return kp

    def __getitem__(self, item):
        sample = self.annotations[self.indices[item]]
        kp = np.asarray(sample["keypoint"], dtype=np.float32)

        # PySKL NTU keypoints may contain a body dimension (M,T,V,C).
        # The manuscript input representation uses the first indexed stream.
        if kp.ndim == 4:
            kp = kp[0]
        if kp.ndim != 3 or kp.shape[-1] != 3:
            raise ValueError(f"Unexpected keypoint shape: {kp.shape}")

        kp = self._fixed_window(kp)

        # Temporal jitter: one random circular shift for the clip.
        if self.training and self.temporal_jitter > 0:
            delta = random.randint(-self.temporal_jitter, self.temporal_jitter)
            kp = np.roll(kp, shift=delta, axis=0)

        # Frame-wise z-score over joints, independently for x/y/z.
        mean = kp.mean(axis=1, keepdims=True)
        std = kp.std(axis=1, keepdims=True)
        kp = (kp - mean) / (std + 1e-5)

        # Temporal token dropout: mask complete temporal positions.
        if self.training and self.temporal_dropout > 0:
            n_drop = int(math.floor(self.temporal_dropout * self.frames))
            if n_drop > 0:
                dropped = np.random.choice(self.frames, size=n_drop, replace=False)
                kp[dropped] = 0.0

        return torch.from_numpy(kp).float(), int(sample["label"])


class CDFormer(nn.Module):
    def __init__(
        self,
        seq_len,
        num_joints=25,
        num_classes=120,
        d_model=192,
        heads=8,
        layers=12,
        dropout=0.15,
        d_ff=2048,
    ):
        super().__init__()
        self.proj = nn.Linear(3, d_model)
        self.temb = nn.Embedding(seq_len, d_model)
        self.jemb = nn.Embedding(num_joints, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        b, t, j, _ = x.shape
        x = self.proj(x)
        t_ids = torch.arange(t, device=x.device)
        j_ids = torch.arange(j, device=x.device)
        x = x + self.temb(t_ids).unsqueeze(1)
        x = x + self.jemb(j_ids)
        x = self.norm(self.drop(x.reshape(b, t * j, -1)))
        x = torch.cat([self.cls.expand(b, -1, -1), x], dim=1)
        x = self.enc(x)
        h_cls = x[:, 0]
        h_avg = x[:, 1:].mean(dim=1)
        return self.head(h_cls + h_avg)


def resolve_protocol_indices(data, protocol):
    train_key = f"{protocol}_train"
    eval_key = f"{protocol}_val"
    split = data.get("split", {})
    annotations = data.get("annotations", [])

    if train_key not in split or eval_key not in split:
        raise KeyError(f"Missing {train_key!r} or {eval_key!r} in annotation file")

    id_to_index = {}
    for idx, sample in enumerate(annotations):
        frame_dir = sample.get("frame_dir")
        if frame_dir is None:
            raise ValueError(f"Annotation {idx} has no frame_dir")
        if frame_dir in id_to_index:
            raise ValueError(f"Duplicate frame_dir: {frame_dir}")
        id_to_index[frame_dir] = idx

    def map_ids(ids, key):
        missing = [name for name in ids if name not in id_to_index]
        if missing:
            raise ValueError(f"{key} contains missing IDs, first: {missing[:3]}")
        return [id_to_index[name] for name in ids]

    train_idx = map_ids(split[train_key], train_key)
    eval_idx = map_ids(split[eval_key], eval_key)

    if set(train_idx) & set(eval_idx):
        raise ValueError(f"{protocol.upper()} train/evaluation partitions overlap")

    return train_idx, eval_idx, train_key, eval_key


def load_model_checkpoint(model, path, device):
    raw = torch.load(path, map_location=device)
    if isinstance(raw, dict) and "model" in raw:
        state = raw["model"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    else:
        state = raw

    state = {k.removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match the requested architecture. "
            f"Missing={missing[:5]}, unexpected={unexpected[:5]}"
        )


def reset_classification_head(model):
    nn.init.xavier_uniform_(model.head.weight)
    if model.head.bias is not None:
        nn.init.zeros_(model.head.bias)


def set_encoder_frozen(model, n_layers, frozen):
    n_layers = min(max(int(n_layers), 0), len(model.enc.layers))
    for layer in model.enc.layers[:n_layers]:
        for parameter in layer.parameters():
            parameter.requires_grad = not frozen


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in tqdm(loader, desc="EVAL", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(x)
            loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / max(total, 1), 100.0 * total_correct / max(total, 1)


def train_one_epoch(
    model,
    loader,
    device,
    criterion,
    optimizer,
    scaler,
    accumulation_steps,
    epoch,
    epochs,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    running_correct = 0
    seen = 0

    bar = tqdm(loader, desc=f"TRAIN {epoch:03d}/{epochs:03d}", leave=False)
    for step, (x, y) in enumerate(bar, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(x)
            raw_loss = criterion(logits, y)
            loss = raw_loss / accumulation_steps

        scaler.scale(loss).backward()

        do_step = step % accumulation_steps == 0 or step == len(loader)
        if do_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch = y.size(0)
        running_loss += raw_loss.item() * batch
        running_correct += (logits.argmax(dim=1) == y).sum().item()
        seen += batch
        bar.set_postfix(
            loss=f"{running_loss / max(seen,1):.4f}",
            acc=f"{100.0 * running_correct / max(seen,1):.2f}%",
        )

    return running_loss / max(seen, 1), 100.0 * running_correct / max(seen, 1)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_acc, args):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "best_accuracy": float(best_acc),
        "config": vars(args),
    }
    torch.save(payload, path)


def main(args):
    seed_everything(args.seed)

    if args.reset_head and not args.init_checkpoint:
        raise ValueError("--reset-head requires --init-checkpoint")
    if args.micro_batch < 1 or args.accumulation_steps < 1:
        raise ValueError("batch and accumulation values must be positive")

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA requested but no CUDA device is available")

    with open(args.pkl, "rb") as handle:
        data = pickle.load(handle)

    train_idx, eval_idx, train_key, eval_key = resolve_protocol_indices(
        data, args.protocol
    )

    print(f"Protocol: {args.protocol.upper()}")
    print(f"Optimization split: {train_key} ({len(train_idx)} samples)")
    print(f"Evaluation split:   {eval_key} ({len(eval_idx)} samples)")
    print(f"Frames: {args.frames}")
    print(
        f"Micro-batch: {args.micro_batch} | accumulation: {args.accumulation_steps} "
        f"| effective batch: {args.micro_batch * args.accumulation_steps}"
    )

    train_ds = NTU120Dataset(
        data["annotations"],
        train_idx,
        frames=args.frames,
        training=True,
        temporal_dropout=args.temporal_dropout,
        temporal_jitter=args.temporal_jitter,
    )
    eval_ds = NTU120Dataset(
        data["annotations"],
        eval_idx,
        frames=args.frames,
        training=False,
    )

    common_loader = dict(
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.micro_batch,
        shuffle=True,
        drop_last=False,
        **common_loader,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.eval_batch,
        shuffle=False,
        drop_last=False,
        **common_loader,
    )

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

    if args.init_checkpoint:
        load_model_checkpoint(model, args.init_checkpoint, device)
        print(f"Loaded initialization checkpoint: {args.init_checkpoint}")

    if args.reset_head:
        reset_classification_head(model)
        set_encoder_frozen(model, args.freeze_layers, frozen=True)
        print(
            f"Classification head reset; first {args.freeze_layers} encoder layers "
            f"frozen for {args.unfreeze_epoch} completed epochs."
        )

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )

    best_acc = -1.0
    stale_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        if (
            args.reset_head
            and args.freeze_layers > 0
            and epoch == args.unfreeze_epoch + 1
        ):
            set_encoder_frozen(model, args.freeze_layers, frozen=False)
            print(f"Encoder unfrozen before epoch {epoch}.")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer,
            scaler,
            args.accumulation_steps,
            epoch,
            args.epochs,
        )
        eval_loss, eval_acc = evaluate(model, eval_loader, device, criterion)
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "eval_loss": eval_loss,
            "eval_accuracy": eval_acc,
        }
        history.append(row)
        (outdir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

        print(
            f"[{args.protocol.upper()} T{args.frames}] "
            f"E{epoch:03d} train={train_acc:.2f}% eval={eval_acc:.2f}% "
            f"lr={optimizer.param_groups[0]['lr']:.6g}"
        )

        save_checkpoint(
            outdir / "last_model.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            best_acc,
            args,
        )

        if eval_acc > best_acc:
            best_acc = eval_acc
            stale_epochs = 0
            save_checkpoint(
                outdir / "best_model.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_acc,
                args,
            )
            print(f"New best: {best_acc:.2f}%")
        else:
            stale_epochs += 1

        if stale_epochs >= args.patience:
            print(
                f"Early stopping after {stale_epochs} epochs without improvement. "
                f"Best evaluation accuracy: {best_acc:.2f}%"
            )
            break

    print(f"Finished. Best evaluation accuracy: {best_acc:.2f}%")
    print(f"Local outputs: {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CD-Former on NTU RGB+D 120 XSUB or XSET."
    )
    parser.add_argument("--pkl", required=True, help="Path to ntu120_3danno.pkl")
    parser.add_argument("--protocol", choices=["xsub", "xset"], required=True)
    parser.add_argument("--frames", type=int, choices=[16, 24, 32], default=32)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--d-model", dest="d_model", type=int, default=192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--d-ff", dest="d_ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.15)

    parser.add_argument("--lr", type=float, default=8e-3)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--temporal-dropout", type=float, default=0.2)
    parser.add_argument("--temporal-jitter", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)

    # Kaggle-friendly gradient accumulation: 20 x 20 = effective batch 400.
    parser.add_argument("--micro-batch", type=int, default=20)
    parser.add_argument("--accumulation-steps", type=int, default=20)
    parser.add_argument("--eval-batch", type=int, default=32)

    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--reset-head", action="store_true")
    parser.add_argument("--freeze-layers", type=int, default=12)
    parser.add_argument("--unfreeze-epoch", type=int, default=15)

    main(parser.parse_args())
