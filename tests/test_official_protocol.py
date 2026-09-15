"""Focused CPU tests; synthetic fixtures do not represent benchmark results."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cd_former_official as trainer

spec = importlib.util.spec_from_file_location('kaggle_launcher', ROOT / 'scripts/kaggle_train_official.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def fixture():
    rng = np.random.default_rng(18)
    data = {'annotations': [], 'split': {k: [] for k in
            ('xsub_train', 'xsub_val', 'xset_train', 'xset_val')}}
    for setup in (1, 2):
        for subject in (1, 3):
            for camera in (1, 2, 3):
                for repetition in (1, 2):
                    for action in (1, 2, 3):
                        name = f'S{setup:03}C{camera:03}P{subject:03}R{repetition:03}A{action:03}'
                        data['annotations'].append({
                            'frame_dir': name, 'label': action - 1,
                            # Small joint dimension keeps execution tests fast on CPU.
                            'keypoint': rng.normal(size=(1, 35, 3, 3)).astype(np.float32),
                        })
                        data['split']['xsub_train' if subject == 1 else 'xsub_val'].append(name)
                        data['split']['xset_train' if setup == 2 else 'xset_val'].append(name)
    return data


def arguments(pkl, output):
    return SimpleNamespace(
        pkl=str(pkl), outdir=str(output), protocol='xsub', frames=16,
        seed=42, split_seed=42, val_fraction=0.1, batch=16, epochs=2,
        stop=10, workers=0, device='cpu', d_model=8, heads=2, layers=1,
        dropout=0.25, jitter=2, lr=0.001, scheduler='cosine', warmup_epochs=1,
        no_progress=True, plot_every=0, skip_test=True, check_splits=False,
        pretrained='', resume='',
    )


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_official_membership_and_stable_internal_holdout(self):
        data = fixture()
        for protocol in ('xsub', 'xset'):
            tr, va, te, manifest = trainer.make_protocol_split(data, protocol)
            ids = lambda idx: {data['annotations'][i]['frame_dir'] for i in idx}
            self.assertFalse(ids(tr) & ids(va))
            self.assertFalse((ids(tr) | ids(va)) & ids(te))
            self.assertEqual(ids(tr) | ids(va), set(data['split'][protocol + '_train']))
            self.assertEqual(ids(te), set(data['split'][protocol + '_val']))
            reverse = {**data, 'annotations': list(reversed(data['annotations']))}
            self.assertEqual(manifest, trainer.make_protocol_split(reverse, protocol)[3])
            different = trainer.make_protocol_split(data, protocol, split_seed=128)[3]
            self.assertEqual(manifest['official_test_ids'], different['official_test_ids'])
            self.assertNotEqual(manifest['internal_val_ids'], different['internal_val_ids'])

    def test_rejects_malformed_official_splits(self):
        data = fixture()
        cases = []
        bad = copy.deepcopy(data); del bad['split']; cases.append(bad)
        bad = copy.deepcopy(data); bad['annotations'].append(bad['annotations'][0]); cases.append(bad)
        bad = copy.deepcopy(data); bad['split']['xsub_train'].append(bad['split']['xsub_val'][0]); cases.append(bad)
        bad = copy.deepcopy(data); bad['split']['xsub_train'].pop(); cases.append(bad)
        bad = copy.deepcopy(data)
        bad['split']['xsub_train'], bad['split']['xsub_val'] = bad['split']['xsub_val'], bad['split']['xsub_train']
        cases.append(bad)
        for bad in cases:
            with self.assertRaises(ValueError):
                trainer.make_protocol_split(bad, 'xsub')

    def test_pretrained_protocol_guards(self):
        data = fixture()
        xsub = trainer.make_protocol_split(data, 'xsub')[3]
        xset = trainer.make_protocol_split(data, 'xset')[3]
        checkpoint = {'state_dict': {}, 'protocol_metadata': {
            'protocol': 'xsub', 'split_sha256': xsub['split_sha256']}}
        trainer.check_checkpoint_protocol(checkpoint, xsub)
        with self.assertRaises(ValueError):
            trainer.check_checkpoint_protocol(checkpoint, xset)
        with self.assertRaises(ValueError):
            trainer.check_checkpoint_protocol({}, xsub)

    def test_reframe_transfers_only_compatible_parameters(self):
        source = trainer.GraphormerForHAR(3, 16, d_model=8, num_heads=2, num_layers=1)
        for frames in (16, 24, 32):
            target = trainer.GraphormerForHAR(3, frames, d_model=8, num_heads=2, num_layers=1)
            sd, reset = trainer.reframe_state_dict(source.state_dict(), target.state_dict())
            self.assertEqual(reset, frames != 16)
            target.load_state_dict(sd, strict=not reset)
            self.assertTrue(torch.equal(target.proj.weight, source.proj.weight))
            target.eval()
            with torch.no_grad():
                output = target(torch.zeros(2, frames, 3, 3))
            self.assertEqual(tuple(output.shape), (2, 120))
            self.assertTrue(torch.isfinite(output).all())
        incompatible = trainer.GraphormerForHAR(3, 24, d_model=16, num_heads=2, num_layers=1)
        with self.assertRaises(ValueError):
            trainer.reframe_state_dict(source.state_dict(), incompatible.state_dict())

    def test_epoch_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pkl = root / 'ntu120_3danno.pkl'
            pkl.write_bytes(pickle.dumps(fixture()))
            full_args = arguments(pkl, root / 'full')
            with redirect_stdout(io.StringIO()):
                trainer.main(full_args)
            interrupted_args = arguments(pkl, root / 'interrupted')
            original_save = trainer.atomic_torch_save
            def save_then_interrupt(value, path):
                original_save(value, path)
                if Path(path).name == 'last.pth' and value['epoch'] == 1:
                    raise InterruptedError('Simulated session interruption after a complete epoch')
            with patch.object(trainer, 'atomic_torch_save', side_effect=save_then_interrupt):
                with redirect_stdout(io.StringIO()), self.assertRaises(InterruptedError):
                    trainer.main(interrupted_args)
            # Recovery uses the best state embedded in last.pth even if the
            # separately exported best checkpoint was lost or half-updated.
            (root / 'interrupted/best_graphormer.pth').unlink()
            interrupted_args.resume = 'auto'
            with redirect_stdout(io.StringIO()):
                trainer.main(interrupted_args)
            full = torch.load(root / 'full/last.pth', weights_only=True)
            resumed = torch.load(root / 'interrupted/last.pth', weights_only=True)
            self.assertEqual(full['epoch'], 2)
            self.assertEqual(full['logs'], resumed['logs'])
            for name, value in full['state_dict'].items():
                self.assertTrue(torch.equal(value, resumed['state_dict'][name]), name)
            self.assertEqual(full['scheduler_state'], resumed['scheduler_state'])
            for parameter, state in full['optimizer']['state'].items():
                for name, value in state.items():
                    self.assertTrue(torch.equal(value, resumed['optimizer']['state'][parameter][name]))
            changed = copy.copy(interrupted_args); changed.batch = 8
            with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                trainer.main(changed)
            # Completed training can evaluate official test without training again.
            interrupted_args.skip_test = False
            with redirect_stdout(io.StringIO()):
                trainer.main(interrupted_args)
            result = json.loads((root / 'interrupted/metrics/official_test.json').read_text())
            self.assertEqual(result['official_test_n'], 36)
            self.assertEqual(result['selection_metric'], 'internal_val_accuracy')
            self.assertEqual(result['checkpoint_epoch'], resumed['best_epoch'])
            result_path = root / 'interrupted/metrics/official_test.json'
            modified = result_path.stat().st_mtime_ns
            with redirect_stdout(io.StringIO()):
                trainer.main(interrupted_args)
            self.assertEqual(modified, result_path.stat().st_mtime_ns)

    def test_kaggle_paths_and_transfer_plan(self):
        args = arguments('/fake/ntu120_3danno.pkl', '/fake/output')
        args.output_root = '/fake/output'
        args.initialization = 'reframe'
        self.assertEqual(launcher.frame_order([32, 24], 'reframe'), [16, 24, 32])
        self.assertEqual(launcher.frame_order([32], 'scratch'), [32])
        for protocol in ('xsub', 'xset'):
            cmd = launcher.command_for(args, protocol, 24, args.pkl)
            source = cmd[cmd.index('--pretrained') + 1]
            self.assertIn('/' + protocol + '/T16_', source)
            self.assertEqual(cmd[cmd.index('--protocol') + 1], protocol)
            self.assertEqual(cmd[cmd.index('--resume') + 1], 'auto')

    def test_reframe_training_with_onecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pkl = root / 'ntu120_3danno.pkl'
            pkl.write_bytes(pickle.dumps(fixture()))
            source = arguments(pkl, root / 't16')
            source.scheduler = 'onecycle'
            with redirect_stdout(io.StringIO()):
                trainer.main(source)
            target = copy.copy(source)
            target.frames = 24
            target.outdir = str(root / 't24')
            target.pretrained = str(root / 't16/best_graphormer.pth')
            with redirect_stdout(io.StringIO()):
                trainer.main(target)
            before = torch.load(root / 't16/last.pth', weights_only=True)
            after = torch.load(root / 't24/last.pth', weights_only=True)
            self.assertEqual(after['state_dict']['temb.weight'].shape[0], 24)
            self.assertEqual(before['protocol_metadata']['split_sha256'],
                             after['protocol_metadata']['split_sha256'])
            self.assertEqual(after['scheduler_state']['last_epoch'], 4)
            target.resume = 'auto'
            target.pretrained = ''
            with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                trainer.main(target)


if __name__ == '__main__':
    unittest.main()
