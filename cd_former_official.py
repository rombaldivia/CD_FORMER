#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CD-Former / Graphormer training with the official NTU RGB+D 120 protocols.

Select XSUB or XSET from the PySKL annotation file BEFORE creating an
internal validation holdout. PySKL's xsub_val / xset_val are OFFICIAL TEST
partitions, used only for the final report. Default internal validation:
10% of the selected official training partition, stratified by action.

Examples (replace the annotation path):
  python cd_former_official.py --pkl ntu120_3danno.pkl --protocol xsub --frames 16
  python cd_former_official.py --pkl ntu120_3danno.pkl --protocol xset --frames 16
  python cd_former_official.py --pkl ntu120_3danno.pkl --protocol xsub --frames 24

--check-splits validates and records the partition without training.
--skip-test trains/selects a checkpoint without evaluating official test.
--resume auto restores the last completed epoch, including optimizer, scheduler,
AMP scaler and random-number-generator states. Reuse the same configuration.
--pretrained transfers a checkpoint produced by this script with the SAME
protocol and internal split; a different temporal embedding is reinitialized.
Legacy checkpoints without split provenance cannot establish a clean run.

The original backbone and center-window preprocessing are preserved.
The existing loss remains the original batch-level focal-like objective;
this protocol update does not turn it into per-example focal loss.
AMP clipping and cosine scheduler stepping are corrected.

Split definitions:
https://github.com/kennymckormick/pyskl/blob/main/tools/data/ntu_preproc.py
"""

import os, random, argparse, pickle, hashlib, json, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm

NTU120_TRAIN_SUBJECTS = frozenset({
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
    38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
    80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103,
})


def ntu_identity(name):
    """Parse the standard PySKL NTU video identifier, without guessing a split."""
    match = re.fullmatch(r'S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})', name)
    if match is None:
        raise ValueError(f'Invalid NTU frame_dir: {name!r}')
    setup, camera, subject, repetition, action = map(int, match.groups())
    if not (1 <= setup <= 32 and 1 <= camera <= 3 and 1 <= subject <= 106
            and repetition in (1, 2) and 1 <= action <= 120):
        raise ValueError(f'Out-of-range NTU120 identifier: {name!r}')
    return setup, subject, action


def make_protocol_split(data, protocol, val_fraction=0.1, split_seed=42):
    """Return train/internal-val/official-test indices and an auditable manifest.

    Fail on malformed or incomplete split metadata; never fall back to a
    random split of all annotations. Counts follow the actual preprocessed
    file, rather than hard-coded raw-dataset sample counts.
    """
    if protocol not in ('xsub', 'xset'):
        raise ValueError('protocol must be xsub or xset')
    if not 0 < val_fraction < 1:
        raise ValueError('val_fraction must be between 0 and 1')
    annotations = data.get('annotations')
    if not isinstance(annotations, list) or not annotations:
        raise ValueError('Expected a nonempty PySKL annotations list')
    split = data.get('split')
    train_key, test_key = f'{protocol}_train', f'{protocol}_val'
    if not isinstance(split, dict) or train_key not in split or test_key not in split:
        raise ValueError(f'PKL must contain split[{train_key!r}] and split[{test_key!r}]. '
                         'No random-split fallback is allowed.')

    index = {}
    expected_train, expected_test = set(), set()
    labels = []
    for i, item in enumerate(annotations):
        if not isinstance(item, dict):
            raise ValueError(f'Annotation {i} is not a dictionary')
        name = item.get('frame_dir')
        if not isinstance(name, str):
            raise ValueError(f'Annotation {i} is missing frame_dir')
        if name in index:
            raise ValueError(f'Duplicate annotation: {name}')
        setup, subject, action = ntu_identity(name)
        label = item.get('label')
        if not isinstance(label, (int, np.integer)) or int(label) != action - 1:
            raise ValueError(f'Label must be zero-based and match frame_dir: {name}')
        labels.append(int(label))
        index[name] = i
        is_train = subject in NTU120_TRAIN_SUBJECTS if protocol == 'xsub' else setup % 2 == 0
        (expected_train if is_train else expected_test).add(name)

    groups = []
    for key in (train_key, test_key):
        entries = split[key]
        if not isinstance(entries, (list, tuple)) or not entries:
            raise ValueError(f'{key} must be a nonempty list of video identifiers')
        if any(not isinstance(name, str) for name in entries):
            raise ValueError(f'{key} contains a non-string identifier')
        names = set(entries)
        if len(names) != len(entries):
            raise ValueError(f'Duplicate video identifiers in {key}')
        missing = names - set(index)
        if missing:
            raise ValueError(f'{key} references missing annotations: {sorted(missing)[:3]}')
        groups.append(names)
    official_train, official_test = groups
    if official_train & official_test:
        raise ValueError('Official training and test partitions overlap')
    if official_train | official_test != set(index):
        raise ValueError('Official partitions do not cover all supplied annotations')
    if official_train != expected_train or official_test != expected_test:
        raise ValueError(f'{protocol.upper()} metadata disagrees with the official '
                         'NTU120 subject/setup rule')

    # Sorted IDs make the holdout independent of annotation-list ordering and T.
    official_idx = np.array([index[name] for name in sorted(official_train)], dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    idx_tr, idx_va = train_test_split(
        official_idx, test_size=val_fraction, stratify=labels[official_idx],
        random_state=split_seed,
    )
    idx_tr = np.asarray(sorted(idx_tr, key=lambda i: annotations[i]['frame_dir']), dtype=np.int64)
    idx_va = np.asarray(sorted(idx_va, key=lambda i: annotations[i]['frame_dir']), dtype=np.int64)
    idx_te = np.array([index[name] for name in sorted(official_test)], dtype=np.int64)
    train_ids = [annotations[i]['frame_dir'] for i in idx_tr]
    val_ids = [annotations[i]['frame_dir'] for i in idx_va]
    test_ids = [annotations[i]['frame_dir'] for i in idx_te]
    if set(train_ids) & set(val_ids) or (set(train_ids) | set(val_ids)) & official_test:
        raise ValueError('Train / internal validation / official test overlap')
    if set(train_ids) | set(val_ids) != official_train:
        raise ValueError('Internal split changed official training membership')

    manifest = {
        'schema_version': 1, 'dataset': 'NTU RGB+D 120', 'protocol': protocol,
        'official_train_key': train_key, 'official_test_key': test_key,
        'val_fraction': float(val_fraction), 'split_seed': int(split_seed),
        'internal_split_method': 'stratified_by_action_within_official_train',
        'counts': {'official_train': len(official_idx), 'train': len(idx_tr),
                   'internal_val': len(idx_va), 'official_test': len(idx_te)},
        'train_ids': train_ids, 'internal_val_ids': val_ids, 'official_test_ids': test_ids,
    }
    fingerprint = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode('utf-8')
    manifest['split_sha256'] = hashlib.sha256(fingerprint).hexdigest()
    return idx_tr, idx_va, idx_te, manifest


def check_checkpoint_protocol(checkpoint, manifest):
    """Only transfer weights whose recorded training/validation split matches."""
    meta = checkpoint.get('protocol_metadata') if isinstance(checkpoint, dict) else None
    if not isinstance(meta, dict):
        raise ValueError('Checkpoint has no protocol metadata. Start the official T16 '
                         'run from scratch; legacy weights need a separate provenance audit.')
    if meta.get('protocol') != manifest['protocol']:
        raise ValueError('Cannot transfer weights between XSUB and XSET runs')
    if meta.get('split_sha256') != manifest['split_sha256']:
        raise ValueError('Checkpoint internal train/validation split does not match this run')
    if not isinstance(checkpoint.get('state_dict'), dict):
        raise ValueError('Checkpoint is missing state_dict')


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def random_state():
    numpy_state = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_random_state(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def check_resume_config(checkpoint, args):
    keys = ('frames', 'seed', 'd_model', 'heads', 'layers', 'dropout', 'batch',
            'epochs', 'lr', 'scheduler', 'warmup_epochs', 'stop', 'jitter', 'workers')
    saved = checkpoint.get('config', {})
    mismatch = [key for key in keys if saved.get(key) != getattr(args, key)]
    if bool(saved.get('pretrained')) != bool(args.pretrained):
        mismatch.append('pretrained initialization mode')
    if mismatch:
        raise ValueError('Resume configuration changed: ' + ', '.join(mismatch))
    needed = ('optimizer', 'scheduler_state', 'scaler', 'rng', 'logs', 'patience', 'best_checkpoint')
    if any(key not in checkpoint for key in needed):
        raise ValueError('Exact resume requires last.pth; use --pretrained for weights-only transfer')


def set_seed(seed=42):
    random.seed(seed);  np.random.seed(seed)
    torch.manual_seed(seed);  torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def remap_keys(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        nk = (k.replace('input_proj', 'proj')
                .replace('ln.',        'norm.')
                .replace('cls_token',  'cls'))
        if nk.startswith('encoder.'):
            nk = 'enc' + nk[len('encoder'):]
        out[nk] = v
    return out


def reframe_state_dict(state_dict, target_state, allow_reframe=True):
    """Keep all compatible weights; reset only a resized temporal table."""
    sd = remap_keys(state_dict)
    if set(sd) != set(target_state):
        raise ValueError('Checkpoint parameter names do not match this backbone: '
                         f'missing={sorted(set(target_state) - set(sd))}, '
                         f'unexpected={sorted(set(sd) - set(target_state))}')
    reset_temporal = False
    for key in list(sd):
        source_shape, target_shape = tuple(sd[key].shape), tuple(target_state[key].shape)
        if source_shape == target_shape:
            continue
        if (allow_reframe and key == 'temb.weight' and len(source_shape) == 2
                and len(target_shape) == 2 and source_shape[1] == target_shape[1]):
            del sd[key]
            reset_temporal = True
        else:
            raise ValueError(f'Incompatible checkpoint tensor {key}: '
                             f'{source_shape} versus {target_shape}')
    return sd, reset_temporal


class MMAction2KeypointDataset(Dataset):
    def __init__(self, pkl_path=None, num_frames=16, jitter=2, is_train=True, samples=None):
        if samples is None:
            pkl_path = Path(pkl_path);  assert pkl_path.is_file()
            with open(pkl_path, 'rb') as f: data = pickle.load(f)
            samples = data['annotations']
        self.samples = samples
        self.num_frames, self.jitter = num_frames, jitter
        self.is_train = is_train
        first = self.samples[0]['keypoint']
        if first.ndim == 4:  first = first[0]
        self.joints, self.channels = first.shape[1:]

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        item  = self.samples[idx]
        kp    = item['keypoint'];  kp = kp[0] if kp.ndim == 4 else kp
        label = item['label']

        if self.is_train:
            shift = np.random.randint(-self.jitter, self.jitter + 1)
            kp    = np.roll(kp, shift, axis=0)
        # pad / crop
        T = kp.shape[0]
        if T > self.num_frames:
            s  = (T - self.num_frames) // 2
            kp = kp[s:s+self.num_frames]
        elif T < self.num_frames:
            pad = np.repeat(kp[-1][None, ...], self.num_frames-T, axis=0)
            kp  = np.concatenate([kp, pad], axis=0)
        # Normalización por frame
        kp = (kp - kp.mean(axis=1, keepdims=True)) / (kp.std(axis=1, keepdims=True) + 1e-5)
        # Flip horizontal aleatorio
        if self.is_train and np.random.rand() < 0.5:
            kp[..., 0] *= -1
        return torch.from_numpy(kp).float(), label

class GraphormerForHAR(nn.Module):
    def __init__(self, num_joints, seq_len, num_classes=120,
                 d_model=256, num_heads=8, num_layers=6, dropout_p=0.2, in_channels=3):
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)
        self.temb = nn.Embedding(seq_len,   d_model)
        self.jemb = nn.Embedding(num_joints, d_model)
        self.drop = nn.Dropout(dropout_p)
        self.norm = nn.LayerNorm(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, num_heads, dropout=dropout_p, batch_first=True)
        self.enc  = nn.TransformerEncoder(enc_layer, num_layers)
        self.cls  = nn.Parameter(torch.randn(1, 1, d_model))
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):  # x: (B, T, J, C)
        B, T, J, C = x.shape
        x = self.proj(x)
        x = x + self.temb(torch.arange(T, device=x.device)).unsqueeze(1) \
                + self.jemb(torch.arange(J, device=x.device))
        x = self.norm(self.drop(x.reshape(B, T*J, -1)))
        x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1)
        x = self.enc(x)
        cls = x[:, 0]
        avg = x[:, 1:].mean(dim=1)
        return self.head(cls + avg)

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, weight=None):
        super().__init__()
        self.g = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.g * ce).mean()

@torch.no_grad()
def validate(model, loader, crit, device, desc='Internal validation', progress=True):
    model.eval()
    loss = 0.
    cor = 0
    tot = 0
    preds = []
    gts = []
    for x, y in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, disable=not progress):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type,
                                dtype=torch.float16,
                                enabled=device.type == 'cuda'):
            out = model(x)
            l = crit(out, y)
        loss += l.item()
        p = out.argmax(1)
        cor += (p == y).sum().item()
        tot += y.size(0)
        preds += p.cpu().tolist()
        gts += y.cpu().tolist()
    return loss / len(loader), 100 * cor / tot, preds, gts

def main(a):
    if a.epochs < 1 or a.batch < 1 or a.stop < 1 or a.workers < 0 or a.jitter < 0:
        raise ValueError('epochs, batch and stop must be positive; workers and jitter nonnegative')
    if a.scheduler == 'onecycle' and not 0 < a.warmup_epochs < a.epochs:
        raise ValueError('OneCycle requires 0 < warmup_epochs < epochs')
    set_seed(a.seed)
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable. Enable a GPU in Kaggle settings.')
    device = torch.device('cuda' if a.device != 'cpu' and torch.cuda.is_available() else 'cpu')
    print(f"▶ device: {device}")

    # Official outer split FIRST. The held-out official *_val is the test set.
    with open(a.pkl, 'rb') as f:
        data = pickle.load(f)
    idx_tr, idx_va, idx_te, manifest = make_protocol_split(
        data, a.protocol, a.val_fraction, a.split_seed,
    )
    ds_full = MMAction2KeypointDataset(num_frames=a.frames, jitter=a.jitter,
                                     is_train=False, samples=data['annotations'])
    labels = [s['label'] for s in ds_full.samples]
    outdir = Path(a.outdir) if a.outdir else (
        Path('runs') / a.protocol / f'T{a.frames}_seed{a.seed}_split{a.split_seed}'
    )
    metrics_dir = outdir / 'metrics'
    best_path = outdir / 'best_graphormer.pth'
    last_path = outdir / 'last.pth'
    resume_path = last_path if a.resume == 'auto' else (Path(a.resume) if a.resume else None)
    if a.resume == 'auto' and not last_path.is_file():
        resume_path = None
    if a.resume and a.resume != 'auto' and not resume_path.is_file():
        raise FileNotFoundError(resume_path)
    manifest_path = outdir / 'splits.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous.get('split_sha256') != manifest['split_sha256']:
            raise ValueError('Output directory belongs to a different split. Use a new --outdir.')
    if not a.check_splits and resume_path is None and (best_path.exists() or (metrics_dir / 'curve.csv').exists()):
        raise FileExistsError('This output directory already contains a training run. '
                              'Use a new --outdir to avoid overwriting its results.')
    outdir.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f"▶ protocol: {a.protocol.upper()} | frames: {a.frames}")
    print(f"▶ official training: {manifest['counts']['official_train']:,}")
    print(f"▶ official test source: {manifest['official_test_key']}")
    print(f"▶ output: {outdir.resolve()}")

    # Verificación de clases por split
    for name, idx in zip(['Train (internal)', 'Validation (internal)', 'Test (official)'],
                         [idx_tr, idx_va, idx_te]):
        classes_here = set([labels[i] for i in idx])
        print(f"{name}: {len(idx):,} samples, {len(classes_here)} classes.")
        if len(classes_here) < 120:
            print(f"⚠️  Advertencia: {name} tiene solo {len(classes_here)} clases de 120 posibles.")
    if a.check_splits:
        print('✓ Official rules, membership and split isolation validated. No training or test evaluation.')
        return

    ds_train = MMAction2KeypointDataset(num_frames=a.frames, jitter=a.jitter,
                                      is_train=True, samples=ds_full.samples)
    loader_options = dict(batch_size=a.batch, num_workers=a.workers,
                          pin_memory=device.type == 'cuda')
    dl_tr = DataLoader(Subset(ds_train, idx_tr), shuffle=True, **loader_options)
    dl_va = DataLoader(Subset(ds_full, idx_va), shuffle=False, **loader_options)
    # Construct and iterate the official test loader only after checkpoint selection.

    # Detectar número de canales
    in_channels = ds_full.channels

    class_w = torch.tensor(1. / (np.bincount([labels[i] for i in idx_tr], minlength=120) + 1e-6), dtype=torch.float32, device=device)
    model = GraphormerForHAR(ds_full.joints, a.frames, 120, a.d_model, a.heads, a.layers, a.dropout, in_channels).to(device)

    # Every transferred checkpoint must share this protocol and internal holdout.
    source_path = resume_path or a.pretrained
    checkpoint = None
    if source_path:
        checkpoint = torch.load(source_path, map_location='cpu', weights_only=True)
        check_checkpoint_protocol(checkpoint, manifest)
        for key in ('d_model', 'heads', 'layers'):
            if checkpoint.get('config', {}).get(key) != getattr(a, key):
                raise ValueError(f'Checkpoint architecture differs: {key}')
        if resume_path is not None:
            check_resume_config(checkpoint, a)
        sd, reset_temporal = reframe_state_dict(
            checkpoint['state_dict'], model.state_dict(), allow_reframe=resume_path is None,
        )
        incompatible = model.load_state_dict(sd, strict=not reset_temporal)
        if reset_temporal and (set(incompatible.missing_keys) != {'temb.weight'}
                               or incompatible.unexpected_keys):
            raise ValueError('Unexpected missing parameters while reframing')
        print(f"🔄 Loaded protocol-matched weights: {source_path}")
        if reset_temporal:
            print(f'▶ Reinitialized temporal embedding for T={a.frames}; other weights transferred.')
        if resume_path is not None:
            best_checkpoint = checkpoint['best_checkpoint']
            check_checkpoint_protocol(best_checkpoint, manifest)
            if best_checkpoint.get('epoch') != checkpoint.get('best_epoch'):
                raise ValueError('Embedded best checkpoint is inconsistent with resume state')
            if resume_path.resolve() != last_path.resolve() and best_path.exists():
                raise FileExistsError('Destination already contains a run; choose a new --outdir')
            # The embedded best state also recovers an interruption between
            # best.pth and last.pth writes; last.pth is the epoch boundary.
            atomic_torch_save(best_checkpoint, best_path)

    (outdir / 'config.json').write_text(json.dumps(vars(a), indent=2) + '\n', encoding='utf-8')
    crit = FocalLoss(1.5, class_w)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

    if a.scheduler == "onecycle":
        steps_per_epoch = len(dl_tr)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch,
            pct_start=a.warmup_epochs/max(a.epochs,1), div_factor=25.0, final_div_factor=1e4)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')

    # live-LR control
    lr_file = outdir / 'hyper_lr.txt'
    lr_file.write_text(str(a.lr))
    last_mtime = lr_file.stat().st_mtime
    best = float('-inf')
    best_epoch = 0
    patience = 0
    logs = []
    best_checkpoint = None if resume_path is None else checkpoint['best_checkpoint']
    start_epoch = 1
    metrics_dir.mkdir(exist_ok=True)
    if resume_path is not None:
        opt.load_state_dict(checkpoint['optimizer'])
        sched.load_state_dict(checkpoint['scheduler_state'])
        scaler.load_state_dict(checkpoint['scaler'])
        best = checkpoint['best_internal_val_accuracy']
        best_epoch = checkpoint['best_epoch']
        patience = checkpoint['patience']
        logs = checkpoint['logs']
        start_epoch = checkpoint['epoch'] + 1
        restore_random_state(checkpoint['rng'])
        print(f'▶ Resuming after completed epoch {checkpoint["epoch"]}; best epoch {best_epoch}.')
        if checkpoint.get('training_finished'):
            start_epoch = a.epochs + 1

    metadata = {'protocol': a.protocol, 'split_sha256': manifest['split_sha256'],
                'frames': a.frames, 'split_seed': a.split_seed, 'val_fraction': a.val_fraction}

    for ep in range(start_epoch, a.epochs + 1):
        # Hot-reload LR
        if lr_file.exists() and lr_file.stat().st_mtime != last_mtime:
            try:
                new_lr = float(lr_file.read_text().strip())
                for g in opt.param_groups: g['lr'] = new_lr
                print(f"⚡ LR actualizado → {new_lr:.3e}")
                last_mtime = lr_file.stat().st_mtime
            except ValueError:
                print("❌ hyper_lr.txt no contiene número válido")
        # Train
        model.train()
        tot = 0.
        pbar = tqdm(dl_tr, desc=f'{a.protocol.upper()} T{a.frames} E{ep}/{a.epochs}',
                    dynamic_ncols=True, disable=a.no_progress)
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast(device_type=device.type,
                                    dtype=torch.float16,
                                    enabled=device.type == 'cuda'):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{opt.param_groups[0]["lr"]:.2e}')
            if a.scheduler == 'onecycle' and scaler.get_scale() >= scale_before:
                sched.step()
        if a.scheduler == 'cosine':
            sched.step()
        tl = tot / len(dl_tr)
        # Val
        vl, va, preds, gts = validate(model, dl_va, crit, device, progress=not a.no_progress)
        # Log & ckpt
        print(f"E{ep:03}/{a.epochs} | Tr {tl:.3f} | Internal val {vl:.3f} | Acc {va:.2f}%")
        logs.append([ep, tl, vl, va])
        pd.DataFrame(logs, columns=['epoch', 'train', 'val', 'acc']).to_csv(metrics_dir / 'curve.csv', index=False)
        if va > best:
            best = va
            best_epoch = ep
            patience = 0
            best_checkpoint = {
                'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 'epoch': ep,
                'best_internal_val_accuracy': best, 'config': vars(a),
                'protocol_metadata': metadata,
            }
            atomic_torch_save(best_checkpoint, best_path)
        else:
            patience += 1
            print(f"   patience {patience}/{a.stop}")
        if a.plot_every > 0 and (ep % a.plot_every == 0 or ep == best_epoch):
            cm = confusion_matrix(gts, preds, labels=list(range(120)), normalize='true')
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, cmap='YlGnBu', vmin=0, vmax=1, square=True, cbar=True, annot=False)
            plt.xlabel("Predicción"); plt.ylabel("Etiqueta real")
            plt.tight_layout()
            plt.savefig(metrics_dir / 'internal_val_confmat.png', dpi=250)
            plt.close()
        atomic_torch_save({
            'state_dict': model.state_dict(), 'epoch': ep, 'best_epoch': best_epoch,
            'best_internal_val_accuracy': best, 'config': vars(a), 'protocol_metadata': metadata,
            'optimizer': opt.state_dict(), 'scheduler_state': sched.state_dict(),
            'scaler': scaler.state_dict(), 'rng': random_state(), 'logs': logs,
            'best_checkpoint': best_checkpoint,
            'patience': patience, 'training_finished': ep == a.epochs or patience >= a.stop,
        }, last_path)
        if patience >= a.stop:
            print('⏹ Early stop (internal validation)')
            break
    if a.skip_test:
        print(f'✓ Checkpoint selected on internal validation: {best_path}')
        print('▶ Official test was not evaluated (--skip-test).')
        return
    result_path = metrics_dir / 'official_test.json'
    if result_path.exists() and start_epoch > a.epochs:
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if (result.get('split_sha256') == manifest['split_sha256']
                and result.get('checkpoint_epoch') == best_epoch):
            print(f'✓ Run already completed: official test accuracy {result["official_test_accuracy"]:.4f}%')
            return
    # Only now evaluate the selected checkpoint on this protocol's official test.
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    check_checkpoint_protocol(checkpoint, manifest)
    model.load_state_dict(checkpoint['state_dict'])
    dl_te = DataLoader(Subset(ds_full, idx_te), shuffle=False, **loader_options)
    _, acc, preds, gts = validate(model, dl_te, crit, device,
                                 desc=f'{a.protocol.upper()} official test', progress=not a.no_progress)
    print(f"🔬 {a.protocol.upper()} official test accuracy: {acc:.4f}% "
          f"({len(gts):,} samples; checkpoint epoch {checkpoint['epoch']})")
    pd.DataFrame(classification_report(gts, preds, labels=list(range(120)),
                                       output_dict=True, zero_division=0)).T.to_csv(metrics_dir / 'report.csv')
    pd.DataFrame({'frame_dir': manifest['official_test_ids'], 'label': gts,
                  'prediction': preds}).to_csv(metrics_dir / 'official_test_predictions.csv', index=False)
    result = {
        'protocol': a.protocol, 'frames': a.frames, 'seed': a.seed,
        'split_sha256': manifest['split_sha256'], 'checkpoint_epoch': checkpoint['epoch'],
        'selection_metric': 'internal_val_accuracy', 'official_test_n': len(gts),
        'official_test_correct': sum(p == y for p, y in zip(preds, gts)),
        'official_test_accuracy': acc,
    }
    cm = confusion_matrix(gts, preds, labels=list(range(120)), normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap='YlGnBu', vmin=0, vmax=1, square=True, cbar=True, annot=False)
    plt.xlabel("Predicción"); plt.ylabel("Etiqueta real")
    plt.tight_layout()
    plt.savefig(metrics_dir / 'official_test_confmat.png', dpi=250)
    plt.close()
    result_tmp = result_path.with_suffix('.json.tmp')
    result_tmp.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    os.replace(result_tmp, result_path)
    print(f"📊 saved in {metrics_dir}")

if __name__ == '__main__':
    pa = argparse.ArgumentParser()
    pa.add_argument('--pkl', required=True)
    pa.add_argument('--protocol', required=True, choices=['xsub', 'xset'])
    pa.add_argument('--val-fraction', type=float, default=0.1,
                    help='Fraction of OFFICIAL TRAIN reserved for internal validation')
    pa.add_argument('--split-seed', type=int, default=42,
                    help='Keep fixed across T16/T24/T32 and transferred checkpoints')
    pa.add_argument('--seed', type=int, default=42)
    pa.add_argument('--outdir', default='', help='Default: runs/PROTOCOL/Tn_seedN_splitN')
    pa.add_argument('--workers', type=int, default=4)
    pa.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    pa.add_argument('--no-progress', action='store_true')
    pa.add_argument('--plot-every', type=int, default=5, help='Validation confusion-matrix interval; 0 disables')
    pa.add_argument('--check-splits', action='store_true', help='Validate split and exit before training')
    pa.add_argument('--skip-test', action='store_true', help='Do not evaluate official test after training')
    pa.add_argument('--pretrained', default='', help='Same-protocol weights; supports temporal reframing')
    pa.add_argument('--resume', nargs='?', const='auto', default='',
                    help='Full-state last.pth path, or auto to resume this output directory when available')
    pa.add_argument('--frames',  type=int, default=16, choices=[16, 24, 32])
    pa.add_argument('--jitter',  type=int, default=2)
    pa.add_argument('--batch',   type=int, default=90)
    pa.add_argument('--epochs',  type=int, default=120)
    pa.add_argument('--stop',    type=int, default=10)
    pa.add_argument('--lr',      type=float, default=5e-5)
    pa.add_argument('--d_model', type=int, default=192)
    pa.add_argument('--heads',   type=int, default=6)
    pa.add_argument('--layers',  type=int, default=8)
    pa.add_argument('--dropout', type=float, default=0.25)
    pa.add_argument('--scheduler', default='cosine', choices=['cosine', 'onecycle'])
    pa.add_argument('--warmup_epochs', type=int, default=3)
    main(pa.parse_args())
