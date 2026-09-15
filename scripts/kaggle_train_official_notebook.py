#!/usr/bin/env python3
"""Kaggle notebook launcher for the official NTU120 CD-Former workflow.

This file intentionally follows the display pattern used by the latest NestSAR
Kaggle dual-T4 launcher:

* the notebook parent owns exactly two tqdm widgets (XSUB and XSET);
* child workers never stream carriage-return progress into the notebook;
* each persistent bar resets for the current epoch/phase and fills by batches;
* BEST is shown first and is updated only after a completed validation;
* T16 -> T24 -> T32 reuse the same two widget rows.

Run this file inside the Kaggle notebook kernel with runpy.run_path(...). The
actual training implementation remains cd_former_official.py and the official
launcher helpers remain scripts/kaggle_train_official.py.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys
import time


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "kaggle_train_official.py"

spec = importlib.util.spec_from_file_location("cdformer_kaggle_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def notebook_tqdm():
    """Select the same explicit notebook/terminal split used by NestSAR."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        notebook = shell is not None and getattr(shell, "kernel", None) is not None
    except ImportError:
        notebook = False

    if notebook:
        try:
            import ipywidgets  # noqa: F401
            from tqdm.notebook import tqdm
        except ImportError as exc:
            raise RuntimeError(
                "Notebook progress needs tqdm and ipywidgets in the Kaggle kernel."
            ) from exc
        return tqdm

    from tqdm import tqdm
    return tqdm


def make_bars(protocols, protocol_gpu):
    tqdm = notebook_tqdm()
    bars = {}
    for position, protocol in enumerate(protocols):
        gpu = protocol_gpu[protocol]
        bar = tqdm(
            total=1,
            desc=f"{protocol.upper()} G{gpu} setup",
            position=position,
            leave=True,
            mininterval=0.5,
            dynamic_ncols=True,
        )
        bar._cdformer_tag = None
        bar.set_postfix({"BEST": "--"}, refresh=True)
        bars[protocol] = bar
    return bars


def best_text(best_by_stage, frames):
    record = best_by_stage.get(frames)
    if not record:
        return "--"
    return f"{record['acc']:.4f}%@E{record['epoch']:02d}"


def update_completed_best(status, best_by_stage):
    """Only a completed EPOCH summary may promote BEST."""
    if status.get("phase") != "EPOCH":
        return
    frames = int(status.get("frames", 0) or 0)
    acc = base.parse_float(status.get("val_acc"))
    epoch = int(status.get("epoch", 0) or 0)
    if not frames or acc is None or not epoch:
        return
    old = best_by_stage.get(frames)
    if old is None or acc > old["acc"]:
        best_by_stage[frames] = {"acc": acc, "epoch": epoch}


def phase_label(phase):
    return {
        "TRAIN": "Train",
        "VAL": "Validate",
        "TEST": "Test",
        "EPOCH": "Epoch done",
        "START": "Starting",
        "STAGE_DONE": "Stage done",
        "PROTOCOL_DONE": "Done",
        "FAILED": "Failed",
        "TEST_RESULT": "Test done",
    }.get(phase, phase.title() if phase else "Starting")


def update_bar(bar, protocol, gpu, status, best_by_stage):
    """NestSAR-style: one persistent row; reset on epoch/phase change."""
    phase = status.get("phase", "START")
    frames = int(status.get("frames", 0) or 0)
    epoch = int(status.get("epoch", 0) or 0)

    if phase in ("TRAIN", "VAL", "TEST"):
        total = max(int(status.get("total", 0) or 0), 1)
        current = min(max(int(status.get("done", 0) or 0), 0), total)
        # A new epoch or a Train->Val/Test phase change resets the SAME widget.
        tag = (frames, phase, epoch, total)
        if getattr(bar, "_cdformer_tag", None) != tag:
            bar.reset(total=total)
            bar._cdformer_tag = tag

        ep_text = f" E{epoch:03d}" if epoch else ""
        bar.set_description_str(
            f"{protocol.upper()} G{gpu} T{frames}{ep_text} {phase_label(phase)}",
            refresh=False,
        )
        bar.n = current

        stats = {"BEST": best_text(best_by_stage, frames)}
        if status.get("acc") is not None:
            key = "val" if phase == "VAL" else ("test" if phase == "TEST" else "tr")
            stats[key] = f"{status['acc']}%"
        if status.get("loss") is not None:
            stats["loss"] = status["loss"]
        if status.get("lr") is not None and phase == "TRAIN":
            stats["lr"] = status["lr"]
        bar.set_postfix(stats, refresh=False)
        bar.refresh()
        return

    if phase == "EPOCH":
        # Keep the just-completed validation visually complete until the next
        # Train status resets the widget for E(epoch+1).
        total = max(int(getattr(bar, "total", 1) or 1), 1)
        bar.n = total
        bar.set_description_str(
            f"{protocol.upper()} G{gpu} T{frames} E{epoch:03d} Complete",
            refresh=False,
        )
        stats = {"BEST": best_text(best_by_stage, frames)}
        if status.get("val_acc") is not None:
            stats["val"] = f"{status['val_acc']}%"
        if status.get("train_acc") is not None:
            stats["tr"] = f"{status['train_acc']}%"
        bar.set_postfix(stats, refresh=False)
        bar.refresh()
        return

    # Setup/stage transitions use the same bar with a tiny 1-step placeholder.
    tag = (frames, phase, epoch, 1)
    if getattr(bar, "_cdformer_tag", None) != tag:
        bar.reset(total=1)
        bar._cdformer_tag = tag

    if phase in ("STAGE_DONE", "PROTOCOL_DONE", "TEST_RESULT"):
        bar.n = 1
    else:
        bar.n = 0

    if frames:
        desc = f"{protocol.upper()} G{gpu} T{frames} {phase_label(phase)}"
    else:
        desc = f"{protocol.upper()} G{gpu} {phase_label(phase)}"
    bar.set_description_str(desc, refresh=False)

    stats = {"BEST": best_text(best_by_stage, frames)}
    if phase == "TEST_RESULT" and status.get("acc") is not None:
        stats["test"] = f"{status['acc']}%"
    elif phase == "STAGE_DONE" and status.get("test_acc"):
        stats["test"] = f"{status['test_acc']}%"
    elif phase == "FAILED":
        stats["rc"] = status.get("rc", "?")
    bar.set_postfix(stats, refresh=False)
    bar.refresh()


def run(args):
    import torch

    pkl = base.find_annotation(args.pkl)
    count, inventory = base.cuda_inventory(torch)
    gpu_ids = args.gpus if args.gpus is not None else list(range(count))

    if not gpu_ids or any(g < 0 or g >= count for g in gpu_ids):
        raise RuntimeError(
            "No valid visible GPU. Select Kaggle GPU T4 x2 and restart the session."
        )
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU IDs must not contain duplicates")
    if len(set(args.protocols)) != len(args.protocols):
        raise ValueError("Protocols must not contain duplicates")

    stages = base.frame_order(args.frames, args.initialization)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    logdir = output / "logs"
    logdir.mkdir(parents=True, exist_ok=True)

    logs = {p: logdir / f"{p}_worker.log" for p in args.protocols}
    for path in logs.values():
        path.write_text("", encoding="utf-8")

    jobs = min(len(gpu_ids), len(args.protocols))
    queues = [args.protocols[i::jobs] for i in range(jobs)]
    protocol_gpu = {}
    for worker_index, queue in enumerate(queues):
        for protocol in queue:
            protocol_gpu[protocol] = gpu_ids[worker_index]

    print("=" * 118)
    print("CD-FORMER | OFFICIAL NTU120 | NESTSAR-STYLE NOTEBOOK PROGRESS")
    print("Notebook owns exactly two persistent bars; workers write only logs")
    print("=" * 118)
    print(f"CUDA available: {torch.cuda.is_available()} | visible devices: {count}")
    for index, name in inventory:
        print(f"  GPU {index}: {name}")
    print(f"Annotation: {pkl}")
    print(f"Stages: {stages}")
    print(f"Output: {output}")
    print(
        "GPU assignment: "
        + ", ".join(
            f"{p.upper()}→GPU{protocol_gpu[p]}" for p in args.protocols
        )
    )
    print(f"Worker logs: {logdir}")
    print("=" * 118, flush=True)

    if args.dry_run:
        for protocol in args.protocols:
            gpu = protocol_gpu[protocol]
            for frames in stages:
                print(
                    f"[{protocol.upper()} T{frames} | GPU {gpu}] "
                    + shlex.join(base.command_for(args, protocol, frames, pkl))
                )
        return []

    bars = {} if args.no_progress else make_bars(args.protocols, protocol_gpu)
    best = {p: {} for p in args.protocols}
    fingerprints = {p: None for p in args.protocols}

    def worker(worker_index):
        records = []
        for protocol in queues[worker_index]:
            records.extend(
                base.run_protocol(
                    args,
                    protocol,
                    gpu_ids[worker_index],
                    pkl,
                    logs[protocol],
                )
            )
        return records

    executor = ThreadPoolExecutor(max_workers=jobs)
    futures = [executor.submit(worker, i) for i in range(jobs)]
    records = []

    try:
        while not all(f.done() for f in futures):
            for protocol in args.protocols:
                status = base.read_latest_status(logs[protocol])
                if status is None:
                    continue

                fingerprint = tuple(
                    sorted((str(k), str(v)) for k, v in status.items())
                )
                if fingerprint == fingerprints[protocol]:
                    continue
                fingerprints[protocol] = fingerprint

                update_completed_best(status, best[protocol])
                if protocol in bars:
                    update_bar(
                        bars[protocol],
                        protocol,
                        protocol_gpu[protocol],
                        status,
                        best[protocol],
                    )

            for future in futures:
                if future.done() and future.exception() is not None:
                    raise future.exception()
            time.sleep(0.35)

        for future in futures:
            records.extend(future.result())

        # One final status refresh catches a worker that completed between polls.
        for protocol in args.protocols:
            status = base.read_latest_status(logs[protocol])
            if status is not None:
                update_completed_best(status, best[protocol])
                if protocol in bars:
                    update_bar(
                        bars[protocol],
                        protocol,
                        protocol_gpu[protocol],
                        status,
                        best[protocol],
                    )

    except BaseException:
        base.terminate_children()
        for future in futures:
            future.cancel()
        for protocol, bar in bars.items():
            bar.set_postfix({"FAILED": "see worker log"}, refresh=True)
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        for bar in bars.values():
            bar.close()

    records.sort(key=lambda r: (r["protocol"], r["frames"]))
    (output / "summary.json").write_text(
        json.dumps(records, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 118)
    print("CD-FORMER OFFICIAL RESULTS")
    print("=" * 118)
    if records:
        for result in records:
            print(
                f"{result['protocol'].upper()} T{result['frames']}: "
                f"{result['official_test_accuracy']:.4f}% on "
                f"{result['official_test_n']:,} official test clips"
            )
    else:
        print("Official test skipped; checkpoints selected on internal validation.")
    print("=" * 118, flush=True)
    return records


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pkl", default="")
    p.add_argument(
        "--protocols", nargs="+", choices=["xsub", "xset"], default=["xsub", "xset"]
    )
    p.add_argument(
        "--frames", nargs="+", type=int, choices=[16, 24, 32], default=[16, 24, 32]
    )
    p.add_argument("--initialization", choices=["reframe", "scratch"], default="reframe")
    p.add_argument("--gpus", nargs="+", type=int, default=None)
    p.add_argument("--output-root", default="/kaggle/working/cdformer_official_runs")
    p.add_argument("--restore-root", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--stop", type=int, default=10)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--d_model", type=int, default=192)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--jitter", type=int, default=2)
    p.add_argument("--scheduler", choices=["cosine", "onecycle"], default="cosine")
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--plot-every", type=int, default=5)
    p.add_argument("--skip-test", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
