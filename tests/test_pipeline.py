"""Offline regression test of the fusion training/resume/inference lifecycle."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]


class FusionPipelineTest(unittest.TestCase):
    def run_cli(self, *args):
        completed = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT,
                                   capture_output=True, text=True, timeout=180)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_train_resume_fuse_evaluate(self):
        with tempfile.TemporaryDirectory(prefix='c2rf-pipeline-') as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(42)
            for split in ['train', 'test']:
                for folder in ['vi', 'ir']:
                    target = root / 'data' / split / 'OSDataset' / folder
                    target.mkdir(parents=True)
                    for index in range(2):
                        image = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
                        Image.fromarray(image).save(target / ('%d.png' % index))
            common = ['--dataset', 'OSDataset', '--device', 'cpu', '--nThreads', '0',
                      '--train_dataroot', root / 'data/train', '--test_dataroot', root / 'data/test',
                      '--output_dir', root / 'outputs']
            training = ['--batch_size', '2', '--crop_size', '32', '--n_ep_Fu', '2',
                        '--warmup_epochs', '1', '--max_steps', '1', '--model_save_freq', '1']
            self.run_cli('train_Fu.py', *common, *training)
            base = root / 'outputs/OSDataset/baseline'
            before = torch.load(base / 'epoch_0001.pt', weights_only=True)
            after = torch.load(base / 'latest.pt', weights_only=True)
            self.assertEqual(after['stage'], 'fusion')
            self.assertTrue(all(torch.equal(before['model'][key], value)
                                for key, value in after['model'].items() if key.startswith(('encoder.', 'decoder.'))))
            self.assertTrue(any(not torch.equal(before['model'][key], value)
                                for key, value in after['model'].items() if key.startswith('fusion.')))
            self.run_cli('train_Fu.py', *common, *training, '--experiment', 'resumed',
                         '--resume', base / 'epoch_0001.pt')
            resumed = torch.load(root / 'outputs/OSDataset/resumed/latest.pt', weights_only=True)
            self.assertTrue(all(torch.equal(value, resumed['model'][key])
                                for key, value in after['model'].items()))
            self.run_cli('fuse.py', *common, '--fusion_checkpoint', base / 'latest.pt')
            self.assertEqual(len(list((base / 'fused').glob('*.png'))), 2)
            with Image.open(base / 'fused/0.png') as image:
                self.assertEqual(image.size, (32, 32))
            self.run_cli('evaluate_fusion.py', *common)
            report = json.loads((base / 'metrics.json').read_text())
            self.assertEqual(report['pairs'], 2)
            self.assertTrue(all(np.isfinite(value) for value in report['mean'].values()))


if __name__ == '__main__':
    unittest.main()
