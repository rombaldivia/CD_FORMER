#!/usr/bin/env python3
"""Train official NTU120 protocols on Kaggle, with epoch-level auto-resume.

With two visible GPUs, XSUB uses one GPU and XSET uses the other. With one,
the protocols run sequentially. Each protocol trains T16 first, then transfers
its own T16 weights independently to T24 and T32. Use --initialization scratch
for independent random initialization at every frame length.

Notebook progress follows the proven NestSAR dual-T4 pattern: child training
processes write all output (including their local tqdm updates) to per-protocol
log files, while this parent process owns the only notebook tqdm rows. This
prevents competing carriage returns from XSUB and XSET from corrupting Kaggle
output and keeps exactly one persistent row per protocol.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = set()
LOCK = threading.Lock()
STOP = threading.Event()
ANSI_RE = re.compile(r'\x1b\[[0-9;?]*[ -/]*[@-~]')


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
        return None
    relative = Path(protocol) / run_directory(args, protocol, frames).name
    source = Path(args.restore_root) / relative
    target = run_directory(args, protocol, frames)
    if target.exists():
        return None
    if source.is_dir():
        if not (source / 'last.pth').is_file():
            raise ValueError(f'Restored run has no complete-epoch last.pth: {source}')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
        return source
    return None


def child_environment(protocol, gpu):
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
    # Child tqdm is redirected to a log file, so it always uses row zero.
    # The parent owns notebook row positioning.
    env['CDFORMER_PROGRESS_POSITION'] = '0'
    env['CDFORMER_PROGRESS_LABEL'] = protocol.upper()
    return env


def run_protocol(args, protocol, gpu, pkl, log_path):
    env = child_environment(protocol, gpu)
    results = []
    stages = frame_order(args.frames, args.initialization)

    with log_path.open('a', encoding='utf-8', buffering=1) as log_handle:
        for stage_index, frames in enumerate(stages):
            if STOP.is_set():
                raise RuntimeError('Launcher interrupted')

            cmd = command_for(args, protocol, frames, pkl)
            restored = restore_run(args, protocol, frames)
            log_handle.write(
                f'LAUNCH|protocol={protocol}|gpu={gpu}|frames={frames}|'
                f'stage={stage_index + 1}|stages={len(stages)}\n'
            )
            if restored is not None:
                log_handle.write(f'RESTORED|source={restored}\n')
            log_handle.write('COMMAND|' + shlex.join(cmd) + '\n')
            log_handle.flush()

            if args.dry_run:
                log_handle.write(
                    f'STAGE_DONE|protocol={protocol}|frames={frames}|dry_run=1\n'
                )
                continue

            with LOCK:
                if STOP.is_set():
                    raise RuntimeError('Launcher interrupted')
                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    cwd=ROOT,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                ACTIVE.add(proc)
            try:
                code = proc.wait()
                if code:
                    log_handle.write(
                        f'FAILED|protocol={protocol}|frames={frames}|rc={code}\n'
                    )
                    log_handle.flush()
                    raise subprocess.CalledProcessError(code, cmd)
            finally:
                with LOCK:
                    ACTIVE.discard(proc)

            result_path = run_directory(args, protocol, frames) / 'metrics' / 'official_test.json'
            if result_path.is_file():
                result = json.loads(result_path.read_text(encoding='utf-8'))
                results.append(result)
                test_acc = result.get('official_test_accuracy', '')
            else:
                test_acc = ''
            log_handle.write(
                f'STAGE_DONE|protocol={protocol}|frames={frames}|test_acc={test_acc}\n'
            )
            log_handle.flush()

        log_handle.write(f'PROTOCOL_DONE|protocol={protocol}\n')
        log_handle.flush()
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


def clean_line(raw):
    return ANSI_RE.sub('', raw).strip()


def parse_float(text, default=None):
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def parse_progress_line(line, current_frames):
    match = re.search(
        r'\|\s*(TRAIN|VAL|TEST)\s+T(\d+)(?:\s+E(\d+)/(\d+))?',
        line,
    )
    if match is None:
        return None
    phase, frames, epoch, epochs = match.groups()
    pairs = re.findall(r'(?<![A-Za-z])(\d+)/(\d+)', line)
    done, total = (0, 0)
    if pairs:
        # The final x/y pair is tqdm's batch counter rather than E001/120.
        done, total = map(int, pairs[-1])
    loss_match = re.search(r'\bloss=([0-9.eE+\-]+)', line)
    acc_match = re.search(r'\bacc=([0-9.]+)%', line)
    lr_match = re.search(r'\blr=([0-9.eE+\-]+)', line)
    return {
        'phase': phase,
        'frames': int(frames or current_frames or 0),
        'epoch': int(epoch or 0),
        'epochs': int(epochs or 0),
        'done': done,
        'total': total,
        'loss': loss_match.group(1) if loss_match else None,
        'acc': acc_match.group(1) if acc_match else None,
        'lr': lr_match.group(1) if lr_match else None,
    }


def read_latest_status(log_path):
    if not log_path.is_file():
        return None
    try:
        with log_path.open('rb') as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 262144), os.SEEK_SET)
            text = handle.read().decode('utf-8', errors='ignore')
    except OSError:
        return None

    current_frames = None
    latest = None
    for raw in text.splitlines():
        line = clean_line(raw)
        if not line:
            continue

        launch = re.match(
            r'LAUNCH\|protocol=([^|]+)\|gpu=(\d+)\|frames=(\d+)\|stage=(\d+)\|stages=(\d+)',
            line,
        )
        if launch:
            protocol, gpu, frames, stage, stages = launch.groups()
            current_frames = int(frames)
            latest = {
                'phase': 'START', 'protocol': protocol, 'gpu': int(gpu),
                'frames': current_frames, 'stage': int(stage), 'stages': int(stages),
            }
            continue

        parsed = parse_progress_line(line, current_frames)
        if parsed is not None:
            latest = parsed
            continue

        epoch = re.match(
            r'E(\d+)/(\d+) \| Tr ([0-9.eE+\-]+) \| Train Acc ([0-9.]+)% '
            r'\| Internal val ([0-9.eE+\-]+) \| Acc ([0-9.]+)%',
            line,
        )
        if epoch:
            ep, epochs, train_loss, train_acc, val_loss, val_acc = epoch.groups()
            latest = {
                'phase': 'EPOCH', 'frames': current_frames or 0,
                'epoch': int(ep), 'epochs': int(epochs),
                'train_loss': train_loss, 'train_acc': train_acc,
                'val_loss': val_loss, 'val_acc': val_acc,
            }
            continue

        final_test = re.search(
            r'official test accuracy:\s*([0-9.]+)%\s*\(([0-9,]+) samples; checkpoint epoch (\d+)\)',
            line,
        )
        if final_test:
            acc, samples, checkpoint_epoch = final_test.groups()
            latest = {
                'phase': 'TEST_RESULT', 'frames': current_frames or 0,
                'acc': acc, 'samples': samples,
                'checkpoint_epoch': int(checkpoint_epoch),
            }
            continue

        already_done = re.search(
            r'Run already completed: official test accuracy\s+([0-9.]+)%',
            line,
        )
        if already_done:
            latest = {
                'phase': 'TEST_RESULT', 'frames': current_frames or 0,
                'acc': already_done.group(1),
            }
            continue

        stage_done = re.match(
            r'STAGE_DONE\|protocol=([^|]+)\|frames=(\d+)\|test_acc=(.*)',
            line,
        )
        if stage_done:
            protocol, frames, test_acc = stage_done.groups()
            latest = {
                'phase': 'STAGE_DONE', 'protocol': protocol,
                'frames': int(frames), 'test_acc': test_acc,
            }
            continue

        failed = re.match(
            r'FAILED\|protocol=([^|]+)\|frames=(\d+)\|rc=(\d+)', line
        )
        if failed:
            protocol, frames, rc = failed.groups()
            latest = {
                'phase': 'FAILED', 'protocol': protocol,
                'frames': int(frames), 'rc': int(rc),
            }
            continue

        protocol_done = re.match(r'PROTOCOL_DONE\|protocol=([^|]+)', line)
        if protocol_done:
            latest = {'phase': 'PROTOCOL_DONE', 'protocol': protocol_done.group(1)}

    return latest


def stage_fraction(status, stages, epochs):
    frames = int(status.get('frames', stages[0] if stages else 0) or 0)
    try:
        stage_index = stages.index(frames)
    except ValueError:
        stage_index = 0
    phase = status.get('phase', '')

    if phase == 'START':
        return float(stage_index)
    if phase in ('TRAIN', 'VAL'):
        epoch = max(int(status.get('epoch', 1)), 1)
        local_epochs = max(int(status.get('epochs', epochs) or epochs), 1)
        done = max(int(status.get('done', 0)), 0)
        total = max(int(status.get('total', 0)), 1)
        frac = min(max(done / total, 0.0), 1.0)
        base = epoch - 1
        if phase == 'TRAIN':
            within = base + 0.78 * frac
        else:
            within = base + 0.78 + 0.21 * frac
        return stage_index + min(within / local_epochs, 0.999)
    if phase == 'EPOCH':
        epoch = max(int(status.get('epoch', 0)), 0)
        local_epochs = max(int(status.get('epochs', epochs) or epochs), 1)
        return stage_index + min(epoch / local_epochs, 0.999)
    if phase in ('TEST', 'TEST_RESULT'):
        return stage_index + 0.995
    if phase == 'STAGE_DONE':
        return float(stage_index + 1)
    if phase == 'PROTOCOL_DONE':
        return float(len(stages))
    return float(stage_index)


def compact_status(status, best_by_stage):
    phase = status.get('phase', 'WAIT')
    frames = int(status.get('frames', 0) or 0)
    best = best_by_stage.get(frames)
    best_text = '?' if best is None else f'{best:.2f}%'

    if phase == 'START':
        return f'T{frames} STARTING GPU{status.get("gpu", "?")}'
    if phase == 'TRAIN':
        extras = []
        if status.get('loss') is not None:
            extras.append(f'loss={status["loss"]}')
        if status.get('acc') is not None:
            extras.append(f'acc={status["acc"]}%')
        extras.append(f'best={best_text}')
        return (
            f'T{frames} E{status.get("epoch", 0):03} TRAIN '
            f'{status.get("done", 0)}/{status.get("total", 0)} ' + ' '.join(extras)
        )
    if phase == 'VAL':
        extras = []
        if status.get('loss') is not None:
            extras.append(f'loss={status["loss"]}')
        if status.get('acc') is not None:
            extras.append(f'acc={status["acc"]}%')
        extras.append(f'best={best_text}')
        return (
            f'T{frames} E{status.get("epoch", 0):03} VAL '
            f'{status.get("done", 0)}/{status.get("total", 0)} ' + ' '.join(extras)
        )
    if phase == 'EPOCH':
        return (
            f'T{frames} E{status.get("epoch", 0):03} '
            f'train={status.get("train_acc", "?")}% '
            f'val={status.get("val_acc", "?")}% best={best_text}'
        )
    if phase == 'TEST':
        return (
            f'T{frames} TEST {status.get("done", 0)}/{status.get("total", 0)} '
            f'acc={status.get("acc", "?")}%' 
        )
    if phase == 'TEST_RESULT':
        return f'T{frames} TEST={status.get("acc", "?")}%'
    if phase == 'STAGE_DONE':
        value = status.get('test_acc')
        return f'T{frames} DONE' + (f' test={value}%' if value else '')
    if phase == 'FAILED':
        return f'T{frames} FAILED rc={status.get("rc", "?")}'
    if phase == 'PROTOCOL_DONE':
        return 'DONE'
    return phase


def tail_for_error(log_path, lines=60):
    if not log_path.is_file():
        return '<log missing>'
    text = log_path.read_text(encoding='utf-8', errors='ignore')
    clean = [clean_line(line) for line in text.splitlines() if clean_line(line)]
    return '\n'.join(clean[-lines:])


def main(args):
    import torch

    pkl = find_annotation(args.pkl)
    count, inventory = cuda_inventory(torch)
    gpu_ids = args.gpus if args.gpus is not None else list(range(count))
    if args.dry_run and not gpu_ids:
        gpu_ids = [0]

    print('=' * 118)
    print('CD-FORMER | OFFICIAL NTU120 | CLEAN INTERNAL VALIDATION | KAGGLE')
    print('Notebook progress is parent-owned: one persistent row per protocol')
    print('=' * 118)
    print(f'CUDA available: {torch.cuda.is_available()} | visible devices: {count}')
    if inventory:
        for index, name in inventory:
            print(f'  GPU {index}: {name}')
    else:
        print(
            '  No CUDA device is visible to PyTorch. In Kaggle select a GPU accelerator '
            '(T4 x2 for concurrent XSUB/XSET) and restart the session.'
        )

    if not gpu_ids or (not args.dry_run and any(g < 0 or g >= count for g in gpu_ids)):
        raise RuntimeError(
            'No valid visible GPU was selected. Kaggle must report at least one CUDA device; '
            '--gpus uses the visible PyTorch indices shown above.'
        )
    if len(set(gpu_ids)) != len(gpu_ids) or len(set(args.protocols)) != len(args.protocols):
        raise ValueError('GPU IDs and protocols must not contain duplicates')

    stages = frame_order(args.frames, args.initialization)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    logdir = output / 'logs'
    logdir.mkdir(parents=True, exist_ok=True)
    logs = {protocol: logdir / f'{protocol}_worker.log' for protocol in args.protocols}
    for path in logs.values():
        path.write_text('', encoding='utf-8')

    jobs = min(len(gpu_ids), len(args.protocols))
    queues = [args.protocols[i::jobs] for i in range(jobs)]
    protocol_gpu = {}
    for worker_index, queue in enumerate(queues):
        for protocol in queue:
            protocol_gpu[protocol] = gpu_ids[worker_index]

    print(f'Annotation: {pkl}')
    print(f'Stages: {stages}')
    print(f'Outputs: {output}')
    print(f'Initialization: {args.initialization}')
    print(
        'GPU assignment: ' + ', '.join(
            f'{protocol.upper()}→GPU{protocol_gpu[protocol]}' for protocol in args.protocols
        )
    )
    print(f'Worker logs: {logdir}')
    print('=' * 118, flush=True)

    if args.dry_run:
        for protocol in args.protocols:
            gpu = protocol_gpu[protocol]
            for frames in stages:
                print(
                    f'[{protocol.upper()} T{frames} | GPU {gpu}] '
                    + shlex.join(command_for(args, protocol, frames, pkl))
                )
        return

    bars = {}
    best_by_protocol = {protocol: {} for protocol in args.protocols}
    last_status_fingerprint = {protocol: None for protocol in args.protocols}
    if not args.no_progress:
        for position, protocol in enumerate(args.protocols):
            bars[protocol] = tqdm(
                total=len(stages),
                desc=f'{protocol.upper()} CD-Former',
                position=position,
                leave=True,
                dynamic_ncols=True,
                mininterval=0.25,
                smoothing=0.05,
            )
            bars[protocol].set_postfix_str(
                f'WAITING GPU{protocol_gpu[protocol]}', refresh=True
            )

    def worker(i):
        records = []
        for protocol in queues[i]:
            records.extend(
                run_protocol(args, protocol, gpu_ids[i], pkl, logs[protocol])
            )
        return records

    executor = ThreadPoolExecutor(max_workers=jobs)
    futures = [executor.submit(worker, i) for i in range(jobs)]
    records = []

    try:
        while not all(future.done() for future in futures):
            for protocol in args.protocols:
                status = read_latest_status(logs[protocol])
                if status is None:
                    continue
                fingerprint = tuple(sorted((str(k), str(v)) for k, v in status.items()))
                if fingerprint == last_status_fingerprint[protocol]:
                    continue
                last_status_fingerprint[protocol] = fingerprint

                frames = int(status.get('frames', 0) or 0)
                if status.get('phase') == 'EPOCH':
                    val_acc = parse_float(status.get('val_acc'))
                    if val_acc is not None:
                        old = best_by_protocol[protocol].get(frames)
                        if old is None or val_acc > old:
                            best_by_protocol[protocol][frames] = val_acc

                if protocol in bars:
                    bar = bars[protocol]
                    bar.n = min(
                        float(len(stages)),
                        stage_fraction(status, stages, args.epochs),
                    )
                    bar.set_postfix_str(
                        compact_status(status, best_by_protocol[protocol]),
                        refresh=True,
                    )

            for future in futures:
                if future.done() and future.exception() is not None:
                    raise future.exception()
            time.sleep(0.35)

        for future in futures:
            records.extend(future.result())

    except BaseException:
        terminate_children()
        for future in futures:
            future.cancel()
        if bars:
            for protocol, bar in bars.items():
                bar.set_postfix_str('FAILED — see worker log', refresh=True)
                bar.close()
            print()
            for protocol in args.protocols:
                print(f'\n{protocol.upper()} worker log tail:\n{tail_for_error(logs[protocol])}')
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    if bars:
        for protocol, bar in bars.items():
            bar.total = len(stages)
            bar.n = len(stages)
            protocol_records = [r for r in records if r.get('protocol') == protocol]
            if protocol_records:
                last = sorted(protocol_records, key=lambda r: r.get('frames', 0))[-1]
                bar.set_postfix_str(
                    f'DONE T{last.get("frames", "?")} '
                    f'test={last.get("official_test_accuracy", float("nan")):.4f}%',
                    refresh=True,
                )
            else:
                bar.set_postfix_str('DONE — official test skipped', refresh=True)
            bar.close()

    records.sort(key=lambda r: (r['protocol'], r['frames']))
    (output / 'summary.json').write_text(
        json.dumps(records, indent=2) + '\n', encoding='utf-8'
    )

    print()
    print('=' * 118)
    print('CD-FORMER OFFICIAL RESULTS')
    print('=' * 118)
    if records:
        for r in records:
            print(
                f"{r['protocol'].upper()} T{r['frames']}: "
                f"{r['official_test_accuracy']:.4f}% on "
                f"{r['official_test_n']:,} official test clips"
            )
    else:
        print('Official test skipped; checkpoints were selected only on internal validation.')
    print('=' * 118, flush=True)


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
