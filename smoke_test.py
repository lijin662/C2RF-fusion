"""Check real fusion data and a random-weight forward pass without training."""
import json
from pathlib import Path
import time

import torch

from dataset import Fusion_Data, FusionTestData
from options import TrainOptions
from utils.device import resolve_device
from utils.fusion import FusionModel, seed_everything
from utils.utils import RGB2YCrCb


if __name__ == '__main__':
    opts = TrainOptions().parse()
    seed_everything(opts.seed)
    device = resolve_device(opts.device, opts.gpu)
    report = {'dataset': opts.dataset, 'device': str(device), 'phase': opts.phase,
              'weights': 'random initialization; execution check only'}
    if opts.phase == 'train':
        data = Fusion_Data(opts)
        report['pairs'] = len(data)
        report['fusion_shapes'] = [list(x.shape) for x in data[0][:-1]]
    else:
        data = FusionTestData(opts)
        ir, vi, name = data[0]
        model = FusionModel().eval().to(device)
        start = time.perf_counter()
        with torch.inference_mode():
            fused = model(RGB2YCrCb(ir.to(device))[0], RGB2YCrCb(vi.to(device))[0])
        if not torch.isfinite(fused).all():
            raise RuntimeError('Non-finite fusion output')
        report.update({'pairs': len(data), 'image': name, 'input_shape': list(ir.shape),
                       'output_shape': list(fused.shape),
                       'elapsed_seconds': round(time.perf_counter() - start, 3)})
    destination = Path(opts.output_dir) / opts.dataset / opts.experiment / 'smoke'
    destination.mkdir(parents=True, exist_ok=True)
    (destination / ('%s-report.json' % opts.phase)).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print('PASS')
