#!/usr/bin/env python3
"""Train official NTU120 protocols on Kaggle, with epoch-level auto-resume.

With two visible GPUs, XSUB uses one GPU and XSET uses the other. With one,
the protocols run sequentially. Each protocol trains T16 first, then transfers
its own T16 weights independently to T24 and T32. Use --initialization scratch
for independent random initialization at every frame length.

Each child receives a fixed tqdm row and label, so XSUB/GPU0 and XSET/GPU1 can
remain visible together while reporting live loss and Top-1 accuracy.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = set()
LOCK = threading.Lock()
STOP = threading.Event()


def find_annotation(explicit='', input_root='/kaggle/input'):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    matches = sorted(Path(input_root).rglob('ntu120_3danno.pkl'))
    if len(matches) != 1:
        raise ValueError(
            f'Found {len(matches)} ntu120_3danno.pkl files under {input_root}. '
            'Attach the NTU120 annotation dataset and set --pkl to its exact path. '
            f'Candidates: {[str(p) for p in matches]}'
        )
    return matches[0].resolve()


def frame_order(frames, initialization):
    stages = sorted(set(frames))
    if initialization == 'reframe' and 16 not in stages:
        stages.insert(0, 16)
    return stages


def run_directory(args, protocol, frames):
    return Path(args.output_root) / protocol / f'T{frames}_seed{args.seed}_split{args.split_seed}'


def command_for(args, protocol, frames, pkl):
    cmd = [
        sys.executable, '-u', str(ROOT / 'cd_former_official.py'),
        '--pkl', str(pkl), '--protocol', protocol, '--frames', str(frames),
        '--seed', str(args.seed), '--split-seed', str(args.split_seed),
        '--val-fraction', str(args.val_fraction),
        '--outdir', str(run_directory(args, protocol, frames)),
        '--device', 'cuda', '--workers', str(args.workers), '--batch', str(args.batch),
        '--epochs', str(args.epochs), '--stop', str(args.stop), '--lr', str(args.lr),
        '--d_model', str(args.d_model), '--heads', str(args.heads), '--layers', str(args.layers),
        '--dropout', str(args.dropout), '--jitter', str(args.jitter),
        '--scheduler', args.scheduler, '--warmup_epochs', str(args.warmup_epochs),
        '--plot-every', str(args.plot_every), '--resume', 'auto',
    ]
    if args.initialization == 'reframe' and frames != 16:
        cmd += [
            '--pretrained',
            str(run_directory(args, protocol, 16) / 'best_graphormer.pth'),
        ]
    if args.skip_test:
        cmd.append('--skip-test')
    if args.no_progress:
        cmd.append('--no-progress')
    return cmd


def restore_run(args, protocol, frames):
    if not args.restore_root:
        return
    relative = Path(protocol) / run_directory(args, protocol, frames).name
    source = Path(args.restore_root) / relative
    target = run_directory(args, protocol, frames)
    if target.exists():
        return
    if source.is_dir():
        if not (source / 'last.pth').is_file():
            raise ValueError(f'Restored run has no complete-epoch last.pth: {source}')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
        print(f'[{protocol.upper()} T{frames}] Restored {source}', flush=True)


def run_protocol(args, protocol, gpu, pkl):
    env = os.environ.copy()
    inherited_visible = [
        part.strip() for part in env.get('CUDA_VISIBLE_DEVICES', '').split(',') if part.strip()
    ]
    env['CUDA_VISIBLE_DEVICES'] = inherited_visible[gpu] if inherited_visible else str(gpu)
    env['PYTHONUNBUFFERED'] = '1'
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    env.setdefault('OMP_NUM_THREADS', '2')
    env.setdefault('MKL_NUM_THREADS', '2')
    env.setdefault('TQDM_MININTERVAL', '0.2')
    # The child sees its selected card as cuda:0, but the display keeps the
    # launcher's logical assignment so the notebook shows XSUB GPU0 / XSET GPU1.
    env['CDFORMER_PROGRESS_POSITION'] = str(gpu)
    env['CDFORMER_PROGRESS_LABEL'] = f'{protocol.upper()} GPU{gpu}'

    results = []
    for frames in frame_order(args.frames, args.initialization):
        if STOP.is_set():
            raise RuntimeError('Launcher interrupted')
        cmd = command_for(args, protocol, frames, pkl)
        print(f'\n[{protocol.upper()} T{frames} | GPU {gpu}] {shlex.join(cmd)}', flush=True)
        if args.dry_run:
            continue
        restore_run(args, protocol, frames)
        with LOCK:
            if STOP.is_set():
                raise RuntimeError('Launcher interrupted')
            proc = subprocess.Popen(cmd, env=env, cwd=ROOT, start_new_session=True)
            ACTIVE.add(proc)
        try:
            code = proc.wait()
            if code:
                raise subprocess.CalledProcessError(code, cmd)
        finally:
            with LOCK:
                ACTIVE.discard(proc)
        result_path = run_directory(args, protocol, frames) / 'metrics' / 'official_test.json'
        if result_path.is_file():
            results.append(json.loads(result_path.read_text(encoding='utf-8')))
    return results


def terminate_children():
    STOP.set()
    with LOCK:
        children = list(ACTIVE)
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for child in children:
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()


def cuda_inventory(torch):
    count = torch.cuda.device_count()
    rows = []
    for index in range(count):
        try:
            name = torch.cuda.get_device_name(index)
        except Exception as exc:  # diagnostics must not hide the real launcher error
            name = f'<name unavailable: {exc}>'
        rows.append((index, name))
    return count, rows


def main(args):
    import torch

    pkl = find_annotation(args.pkl)
    count, inventory = cuda_inventory(torch)
    gpu_ids = args.gpus if args.gpus is not None else list(range(count))
    if args.dry_run and not gpu_ids:
        gpu_ids = [0]

    print(f'CUDA available: {torch.cuda.is_available()} | visible devices: {count}', flush=True)
    if inventory:
        for index, name in inventory:
            print(f'  GPU {index}: {name}', flush=True)
    else:
        print(
            '  No CUDA device is visible to PyTorch. In Kaggle select a GPU accelerator '
            '(T4 x2 for concurrent XSUB/XSET) and restart the session.',
            flush=True,
        )

    if not gpu_ids or (not args.dry_run and any(g < 0 or g >= count for g in gpu_ids)):
        raise RuntimeError(
            'No valid visible GPU was selected. Kaggle must report at least one CUDA device; '
            '--gpus uses the visible PyTorch indices shown above.'
        )
    if len(set(gpu_ids)) != len(gpu_ids) or len(set(args.protocols)) != len(args.protocols):
        raise ValueError('GPU IDs and protocols must not contain duplicates')

    print(
        f'Annotation: {pkl}\n'
        f'Stages: {frame_order(args.frames, args.initialization)}\n'
        f'Outputs: {args.output_root}\n'
        f'Initialization: {args.initialization}\n'
        f'GPU assignment: '
        + ', '.join(
            f'{protocol.upper()}→GPU{gpu_ids[i % len(gpu_ids)]}'
            for i, protocol in enumerate(args.protocols)
        ),
        flush=True,
    )

    jobs = min(len(gpu_ids), len(args.protocols))
    # Assign each GPU its own sequential protocol queue; never share a GPU
    # between two concurrently running training processes.
    queues = [args.protocols[i::jobs] for i in range(jobs)]

    def worker(i):
        records = []
        for protocol in queues[i]:
            records.extend(run_protocol(args, protocol, gpu_ids[i], pkl))
        return records

    executor = ThreadPoolExecutor(max_workers=jobs)
    futures = [executor.submit(worker, i) for i in range(jobs)]
    records = []
    try:
        for future in as_completed(futures):
            records.extend(future.result())
    except BaseException:
        terminate_children()
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    if not args.dry_run:
        output = Path(args.output_root)
        output.mkdir(parents=True, exist_ok=True)
        records.sort(key=lambda r: (r['protocol'], r['frames']))
        (output / 'summary.json').write_text(
            json.dumps(records, indent=2) + '\n', encoding='utf-8'
        )
        for r in records:
            print(
                f"{r['protocol'].upper()} T{r['frames']}: "
                f"{r['official_test_accuracy']:.4f}% on "
                f"{r['official_test_n']:,} official test clips",
                flush=True,
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pkl', default='')
    parser.add_argument(
        '--protocols', nargs='+', choices=['xsub', 'xset'], default=['xsub', 'xset']
    )
    parser.add_argument(
        '--frames', nargs='+', type=int, choices=[16, 24, 32], default=[16, 24, 32]
    )
    parser.add_argument('--initialization', choices=['reframe', 'scratch'], default='reframe')
    parser.add_argument('--gpus', nargs='+', type=int, default=None)
    parser.add_argument('--output-root', default='/kaggle/working/cdformer_official_runs')
    parser.add_argument(
        '--restore-root', default='',
        help='Previous saved output root containing xsub/ and xset/',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument(
        '--batch', type=int, default=32,
        help='Per-GPU training batch; fixed across all stages',
    )
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--stop', type=int, default=10)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--d_model', type=int, default=192)
    parser.add_argument('--heads', type=int, default=6)
    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--jitter', type=int, default=2)
    parser.add_argument('--scheduler', choices=['cosine', 'onecycle'], default='cosine')
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--plot-every', type=int, default=5)
    parser.add_argument('--skip-test', action='store_true')
    parser.add_argument('--no-progress', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    main(parser.parse_args())
